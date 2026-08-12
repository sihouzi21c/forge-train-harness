"""Pure PyTorch training of MiniCPM4 8B (muP, no MTP) with 2-D DP × TP.

DP+TP sibling of ``train_qwen3_dense.py`` / ``train_pure_mup_mtp.py``. It
reuses that family's dataloader / LR schedule / prefetch / determinism /
ordered-teardown machinery verbatim (single SSOT) and swaps in:

* the self-written tensor-parallel model :class:`MiniCPM4_8B_TP` (``tp_size==1``
  degenerates to the dense single-card proxy, so one entry covers both shapes);
* the muP optimizer groups (matrix weights at ``lr/width_mult``) crossed with
  the Megatron wd-by-dim policy;
* a **vocab-parallel** masked cross-entropy over the model's vocab-sharded
  logits (no full-logit gather);
* :mod:`evals.harness_dptp` for the DP loss/grad collective and the
  TP-namespaced bitwise capture (each TP rank hashes its own shard under a
  ``tp<rank>.`` key; ``finalize`` merges them into the one comparator dump).

Topology comes from ``--tensor-parallel-size`` (env default
``TENSOR_PARALLEL_SIZE``, 1 when absent); ``dp_size = world_size //
tp_size`` and ``global_rank = dp_rank * tp_size + tp_rank`` (Megatron
convention, TP inner). The ``[LOSS] step=…`` wire line is byte-identical to the
other refs so ``evals/_common.parse_loss_lines`` consumes it unchanged.
"""
from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

# Must be set before CUDA initializes anywhere in the process.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.distributed as dist

# Shared, architecture-agnostic machinery (single SSOT with the other refs):
# dataloader dispatch, LR schedule, prefetch teardown, determinism stack.
from train_pure_mup_mtp import (
    PrefetchedBatcher,
    H100_BF16_PEAK_FLOPS,
    build_dataloader,
    data_loader_from_config,
    compute_lr,
    count_trainable_params,
    enable_determinism,
)

from model_minicpm4_8b_tp import (
    FFN_HIDDEN_SIZE,
    HEAD_DIM,
    HIDDEN_SIZE,
    MAX_SEQ_LEN,
    NUM_HEADS,
    NUM_KV_HEADS,
    NUM_LAYERS,
    VOCAB_SIZE,
    MiniCPM4_8B_TP,
    precompute_rope_freqs,
    vocab_parallel_cross_entropy,
)

# Bootstrap the harness root onto sys.path so ``evals`` is importable when the
# launcher runs this entry directly (torchrun only puts ref/reference on path).
import sys as _sys

_HARNESS_ROOT = str(Path(__file__).resolve().parents[2])
if _HARNESS_ROOT not in _sys.path:
    _sys.path.insert(0, _HARNESS_ROOT)
from evals import harness_dptp


def masked_vocab_parallel_ce(logits_shard, labels, mask, output_layer):
    """(sum_loss [fp32], num_tokens [fp32], per_token_nll [B,S]) for
    vocab-sharded logits, mirroring :func:`train_pure_mup_mtp.masked_ce` but
    using the vocab-parallel CE (TP all-reduces fold in internally; the
    per-token NLL is identical on every TP rank)."""
    B, S, Vshard = logits_shard.shape
    nll = vocab_parallel_cross_entropy(
        logits_shard.reshape(-1, Vshard), labels.reshape(-1), output_layer
    )
    m = mask.reshape(-1).float()
    return (nll * m).sum(), m.sum(), nll.reshape(B, S)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path-file", required=True)
    p.add_argument("--data-config", default="",
                   help="Path to the data-value SSOT (config/data.toml, pointed "
                        "at by FORGE_DATA_TOML). [data].data_loader selects the "
                        "loader kind; no DATA_LOADER env value.")
    p.add_argument("--train-iters", type=int, default=1000)
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--global-batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--lr-warmup-iters", type=int, default=2000)
    p.add_argument("--lr-decay-iters", type=int, default=150000)
    p.add_argument("--lr-wsd-decay-iters", type=int, default=0)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.95)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--init-method-std", type=float, default=0.1)
    p.add_argument("--tensor-parallel-size", type=int,
                   default=int(os.environ.get("TENSOR_PARALLEL_SIZE", "1")),
                   help="TP degree; dp_size = world_size // tp_size.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--recompute-num-layers", type=int, default=8,
                   help="Layers to recompute (-1 = all when --recompute set).")
    p.add_argument("--deterministic", dest="deterministic",
                   action="store_true", default=True,
                   help="Enable full bit-wise determinism stack (default).")
    p.add_argument("--no-deterministic", dest="deterministic",
                   action="store_false",
                   help="Disable determinism for throughput-oriented runs.")
    harness_dptp.add_capture_cli_args(p)
    args = p.parse_args()

    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    tp_size = max(1, int(args.tensor_parallel_size))
    if world_size % tp_size != 0:
        raise ValueError(f"world_size={world_size} not divisible by tp_size={tp_size}")
    dp_size = world_size // tp_size
    layout = harness_dptp.init_groups(dp_size, tp_size)
    rank = layout.global_rank
    is_log_rank = rank == 0

    if args.deterministic:
        enable_determinism(args.seed)
        if is_log_rank:
            print("  determinism: ON (use --no-deterministic to disable)")
    elif is_log_rank:
        print("  determinism: OFF")

    # Data is replicated within a TP group and sharded across DP replicas, so
    # the effective data-parallel world is dp_size.
    grad_accum_steps = args.global_batch_size // (args.micro_batch_size * dp_size)
    if is_log_rank:
        os.makedirs(args.out_dir, exist_ok=True)
        print("=== Pure PyTorch (no TE) MiniCPM4 8B muP + self-written DP×TP ===")
        print(f"  world={world_size}, dp={dp_size}, tp={tp_size}, "
              f"grad_accum={grad_accum_steps}")
        print(f"  lr={args.lr}, init_std={args.init_method_std}, layers={NUM_LAYERS}, "
              f"hidden={HIDDEN_SIZE}, kv_heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}, "
              f"vocab={VOCAB_SIZE}")

    # ── Model (muP knobs from the generic product projection env; a missing
    # key is a render/projection bug — fail fast, no baked-in fallback) ──────
    mup_base = int(os.environ["MUP_BASE_HIDDEN_SIZE"])
    mup_emb = float(os.environ["MUP_EMB_SCALE"])
    mup_depth = float(os.environ["MUP_DEPTH_SCALE"])

    torch.manual_seed(args.seed)
    model = MiniCPM4_8B_TP(
        vocab_size=VOCAB_SIZE,
        mup_base_hidden_size=mup_base,
        mup_emb_scale=mup_emb,
        mup_depth_scale=mup_depth,
        tp_group=layout.tp_group,
    )
    # init_weights draws the full logical tensor from one seeded generator on
    # every rank (identical stream) and slices this rank's shard — so DP
    # replicas of a given TP shard are already bit-identical. The within-DP
    # broadcast below is belt-and-suspenders (src = lowest global rank of this
    # rank's DP group = its tp_rank).
    model.init_weights(init_std=args.init_method_std, seed=args.seed)
    model = model.to(device=device, dtype=torch.bfloat16)
    if dp_size > 1:
        for p_ in model.parameters():
            dist.broadcast(p_.data, src=layout.tp_rank, group=layout.dp_group)

    rope_freqs = precompute_rope_freqs(MAX_SEQ_LEN, device=device)
    num_params = count_trainable_params(model)

    # ── FP32 master + AdamW (muP lr groups × wd-by-dim) ─────────────────────
    bf16_params = [p_ for p_ in model.parameters() if p_.requires_grad]
    fp32_master = [p_.detach().float().clone().requires_grad_(True) for p_ in bf16_params]
    bf16_to_master = {id(b): m for b, m in zip(bf16_params, fp32_master)}
    lr_groups = model.mup_lr_groups(args.lr)
    optim_groups = []
    for g in lr_groups:
        wd_params, no_wd_params = [], []
        for p_ in g["params"]:
            master = bf16_to_master[id(p_)]
            (wd_params if master.dim() >= 2 else no_wd_params).append(master)
        if wd_params:
            optim_groups.append(
                {"params": wd_params, "lr": g["lr"], "weight_decay": args.weight_decay}
            )
        if no_wd_params:
            optim_groups.append(
                {"params": no_wd_params, "lr": g["lr"], "weight_decay": 0.0}
            )

    optim = torch.optim.AdamW(
        optim_groups,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=1e-8, weight_decay=args.weight_decay,
        fused=True,
    )

    # Install capture (strict no-op unless --hash-output is set). Single-step
    # alignment mode hijacks optim.step and dumps on the first step; persistent
    # mode returns a session driven by the begin_step / reduce_* / capture calls
    # and merged by harness_dptp.finalize() at end of training.
    harness_dptp.install_from_args(model, optim, args)

    base_lr_per_group = [g["lr"] for g in optim_groups]
    lr_mult_per_group = [lr / args.lr for lr in base_lr_per_group]
    wd_per_group = [g["weight_decay"] for g in optim_groups]
    if is_log_rank:
        print(f"  optim has {len(optim_groups)} groups, "
              f"lr_mults={lr_mult_per_group}, wd={wd_per_group}")
        print(f"  trainable params (this rank's shards) = {num_params / 1e6:.2f}M")

    # ── Data (DP-sharded, TP-replicated) ─────────────────────────────────────
    with open(args.data_path_file) as f:
        raw = f.read().strip().split()
    dl = build_dataloader(
        data_path_args=raw,
        dp_rank=layout.dp_rank, world_size=dp_size,
        micro_batch_size=args.micro_batch_size,
        seq_length=MAX_SEQ_LEN, seed=args.seed,
        loader=data_loader_from_config(args.data_config),
    )

    # ── Harness-facing loss dump (rank 0 only) ───────────────────────────────
    loss_dump_path = os.environ.get("LOSS_DUMP_FILE", "")
    loss_dump_fh = None
    if is_log_rank and loss_dump_path:
        path_obj = Path(loss_dump_path)
        if not path_obj.is_absolute():
            base = Path(os.environ.get("DUMP_DIR", args.out_dir))
            path_obj = base / loss_dump_path
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        loss_dump_fh = path_obj.open("w", encoding="utf-8", buffering=1)

    # ── Training loop ────────────────────────────────────────────────────────
    fp32_grad_bufs = [torch.zeros_like(p_) for p_ in fp32_master]
    tokens_per_step = args.global_batch_size * MAX_SEQ_LEN
    # MFU(standard) — physical-GEMM enumeration for the 8B muP geometry (no MTP
    # term). Each GEMM [m,k]·[k,n] = 2mkn fwd FLOPs; training adds dgrad + wgrad
    # → 3× fwd per GEMM. The TP shards sum to exactly these aggregate FLOPs.
    H_q = float(NUM_HEADS * HEAD_DIM)
    H_kv = float(NUM_KV_HEADS * HEAD_DIM)
    H = float(HIDDEN_SIZE)
    S = float(MAX_SEQ_LEN)
    ffn = float(FFN_HIDDEN_SIZE)
    V = float(VOCAB_SIZE)
    fwd_attn_proj_per_layer = 2.0 * H * (H_q + 2.0 * H_kv) + 2.0 * H_q * H
    fwd_attn_score_per_layer = 2.0 * S * H_q
    fwd_mlp_per_layer = 6.0 * H * ffn
    fwd_per_layer = fwd_attn_proj_per_layer + fwd_attn_score_per_layer + fwd_mlp_per_layer
    fwd_lm_head = 2.0 * H * V
    fwd_per_token = float(NUM_LAYERS) * fwd_per_layer + fwd_lm_head
    train_per_token = 3.0 * fwd_per_token
    flops_per_step = train_per_token * tokens_per_step
    peak_total = H100_BF16_PEAK_FLOPS * max(world_size, 1)

    next_batch_fn = PrefetchedBatcher(dl, device)

    for step in range(args.train_iters):
        harness_dptp.begin_step(step)
        torch.cuda.synchronize()
        t0 = time.time()

        for buf in fp32_grad_bufs:
            buf.zero_()
        local_lm_sum = torch.zeros(1, device=device, dtype=torch.float64)
        local_lm_n = torch.zeros(1, device=device, dtype=torch.float64)

        for _mb in range(grad_accum_steps):
            input_ids, labels, loss_mask = next_batch_fn()
            for p_ in bf16_params:
                p_.grad = None
                if getattr(p_, "main_grad", None) is not None:
                    p_.main_grad = None

            logits = model(input_ids, rope_freqs, recompute=args.recompute,
                           recompute_num_layers=args.recompute_num_layers)
            lm_sum, lm_n, nll = masked_vocab_parallel_ce(
                logits, labels, loss_mask, model.output)
            if _mb == 0 and harness_dptp.capturing():
                harness_dptp.capture('loss.per_token.preallreduce', nll)
            obj = lm_sum
            obj.backward()
            local_lm_sum += lm_sum.detach().double()
            local_lm_n += lm_n.detach().double()
            with torch.no_grad():
                for p_bf16, buf in zip(bf16_params, fp32_grad_bufs):
                    g = getattr(p_bf16, "main_grad", None)
                    if g is None:
                        g = p_bf16.grad
                    if g is not None:
                        buf.add_(g)

        reported_lm, g_lm_n = harness_dptp.reduce_loss_scalar(local_lm_sum, local_lm_n)
        local_lm = (local_lm_sum / local_lm_n.clamp(min=1.0)).item()

        norm_factor = 1.0 / max(g_lm_n, 1.0)
        harness_dptp.reduce_grads(bf16_params, fp32_grad_bufs, norm_factor=norm_factor)

        for p_fp32, buf in zip(fp32_master, fp32_grad_bufs):
            p_fp32.grad = buf

        lr = compute_lr(step + 1, args.lr, args.min_lr,
                        args.lr_warmup_iters, args.lr_decay_iters, args.lr_wsd_decay_iters)
        for pg, mult in zip(optim.param_groups, lr_mult_per_group):
            pg["lr"] = lr * mult

        grad_norm = torch.nn.utils.clip_grad_norm_(fp32_master, args.clip_grad).item()
        optim.step()

        with torch.no_grad():
            for p_bf16, p_fp32 in zip(bf16_params, fp32_master):
                p_bf16.data.copy_(p_fp32.data)

        torch.cuda.synchronize()
        step_time = time.time() - t0
        total = reported_lm
        mfu = flops_per_step / (step_time * peak_total) * 100.0 if step_time > 0 else 0.0

        if is_log_rank and (step + 1) % args.log_interval == 0:
            print(
                f" iteration {step + 1:6d} | total_loss: {total:.4e} | "
                f"lm_loss: {reported_lm:.4e} | "
                f"local_lm: {local_lm:.4e} | grad norm: {grad_norm:.3f} | "
                f"lr: {lr:.6e} | time(ms): {step_time * 1000:.1f}",
                flush=True,
            )
            loss_line = (
                f"[LOSS] step={step + 1} global_loss={total:.9e} "
                f"grad_norm={grad_norm:.9e} time_s={step_time:.6f} "
                f"mfu_e2e_standard={mfu:.6f}"
            )
            print(loss_line, flush=True)
            if loss_dump_fh is not None:
                loss_dump_fh.write(loss_line + "\n")
                loss_dump_fh.flush()

    if is_log_rank:
        print("=== Training complete ===")
        if loss_dump_fh is not None:
            loss_dump_fh.close()

    # Merge the per-TP-rank capture shards into the single comparator dump —
    # must run before destroy_process_group (it is a TP collective).
    harness_dptp.finalize()

    # ── Ordered teardown (exit-0 contract; see train_pure_mup_mtp) ──────────
    torch.cuda.synchronize()
    next_batch_fn.close()
    try:
        _dl_close = getattr(dl, "close", None)
        if callable(_dl_close):
            _dl_close()
    except Exception:
        pass
    try:
        del next_batch_fn, dl
    except Exception:
        pass
    gc.collect()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
