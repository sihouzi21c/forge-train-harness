"""Pure PyTorch training of Qwen3 0.6B (dense GQA + QK-Norm, no muP/MTP).

DP-only training entry, sibling of ``train_pure_mup_mtp.py``. It reuses
that module's dataloader / LR schedule / loss / prefetch / determinism /
ordered-teardown machinery verbatim (single SSOT) and only swaps in the
Qwen3 model, a single-lr-group optimizer (standard parameterization, no
muP matrix split), and an MFU FLOP count without the MTP/Eagle term.

Per-step progress is printed in the same two forms as the MiniCPM entry: a
human-readable ``iteration …`` line and the machine-readable
``[LOSS] step=N global_loss=… grad_norm=… time_s=… mfu_e2e_standard=…``
line (mirrored to ``$LOSS_DUMP_FILE`` when set). The wire format is
identical so ``evals/_common.parse_loss_lines`` consumes both unchanged.

Determinism contract is inherited from ``train_pure_mup_mtp`` (the same
``enable_determinism`` stack; ``--no-deterministic`` to disable).
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

# Shared, architecture-agnostic machinery (single SSOT with the MiniCPM
# entry). Importing these keeps the dataloader dispatch, LR schedule, loss
# reduction, prefetch teardown, and determinism stack identical across both
# reference models.
from train_pure_mup_mtp import (
    PrefetchedBatcher,
    H100_BF16_PEAK_FLOPS,
    build_dataloader,
    data_loader_from_config,
    compute_lr,
    count_trainable_params,
    enable_determinism,
    masked_ce,
)

from model_qwen3 import (
    FFN_HIDDEN_SIZE,
    HEAD_DIM,
    HIDDEN_SIZE,
    MAX_SEQ_LEN,
    NUM_HEADS,
    NUM_KV_HEADS,
    NUM_LAYERS,
    Qwen3Dense,
    VOCAB_SIZE,
    precompute_rope_freqs,
)

# ── Capture wiring (harness_dp) ──────────────────────────────────────
# The reference carries its own M1–M5 capture call sites via harness_dp
# (install + begin_step + reduce_loss_scalar + reduce_grads + capture),
# replacing the runtime source-patching that ref/bridges/interposer.py used to
# do. Bootstrap the harness root onto sys.path so ``evals`` is importable when
# the launcher runs this entry directly (torchrun only puts ref/reference on
# the path).
import sys as _sys

_HARNESS_ROOT = str(Path(__file__).resolve().parents[2])
if _HARNESS_ROOT not in _sys.path:
    _sys.path.insert(0, _HARNESS_ROOT)
from evals import harness_dp


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
    p.add_argument("--init-method-std", type=float, default=0.02)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--recompute-num-layers", type=int, default=8,
                   help="Number of layers to recompute (-1 = all when --recompute is set; 8 = last 8, default)")
    p.add_argument("--deterministic", dest="deterministic",
                   action="store_true", default=True,
                   help="Enable full bit-wise determinism stack (default).")
    p.add_argument("--no-deterministic", dest="deterministic",
                   action="store_false",
                   help="Disable determinism for throughput-oriented runs.")
    harness_dp.add_capture_cli_args(p)
    args = p.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    if args.deterministic:
        enable_determinism(args.seed)
        if rank == 0:
            print("  determinism: ON (use --no-deterministic to disable)")
    elif rank == 0:
        print("  determinism: OFF")

    grad_accum_steps = args.global_batch_size // (args.micro_batch_size * world_size)
    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        print("=== Pure PyTorch (no TE) Qwen3 0.6B dense + QK-Norm ===")
        print(f"  world_size={world_size}, grad_accum={grad_accum_steps}")
        print(f"  lr={args.lr}, init_std={args.init_method_std}, "
              f"layers={NUM_LAYERS}, hidden={HIDDEN_SIZE}, kv_heads={NUM_KV_HEADS}, "
              f"head_dim={HEAD_DIM}, vocab={VOCAB_SIZE}")

    # ── Model ────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    model = Qwen3Dense()
    model.init_weights(init_std=args.init_method_std, seed=args.seed)

    model = model.to(device=device, dtype=torch.bfloat16)
    if world_size > 1:
        for p_ in model.parameters():
            dist.broadcast(p_.data, src=0)

    rope_freqs = precompute_rope_freqs(MAX_SEQ_LEN, device=device)
    num_params = count_trainable_params(model)

    # ── FP32 master + AdamW (single lr group × wd-by-dim) ──────────
    # Standard parameterization: one lr for all params (no muP matrix
    # split). Megatron-aligned wd policy still applies: dim >= 2 -> wd,
    # dim < 2 (RMSNorm.weight) -> 0.
    bf16_params = [p_ for p_ in model.parameters() if p_.requires_grad]
    fp32_master = [p_.detach().float().clone().requires_grad_(True) for p_ in bf16_params]
    bf16_to_master = {id(b): m for b, m in zip(bf16_params, fp32_master)}
    lr_groups = model.lr_groups(args.lr)
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
    # M1 mode (persistent=False) hijacks optim.step and dumps on first step;
    # persistent mode returns a live session driven by the begin_step /
    # reduce_* / capture calls below and dumped at process exit.
    harness_dp.install_from_args(model, optim, args)

    base_lr_per_group = [g["lr"] for g in optim_groups]
    lr_mult_per_group = [lr / args.lr for lr in base_lr_per_group]
    wd_per_group = [g["weight_decay"] for g in optim_groups]
    if rank == 0:
        print(f"  optim has {len(optim_groups)} groups, "
              f"lr_mults={lr_mult_per_group}, wd={wd_per_group}")
        print(f"  trainable params = {num_params / 1e6:.2f}M")

    # ── Data ─────────────────────────────────────────────────────────
    with open(args.data_path_file) as f:
        raw = f.read().strip().split()
    dl = build_dataloader(
        data_path_args=raw,
        dp_rank=rank, world_size=world_size,
        micro_batch_size=args.micro_batch_size,
        seq_length=MAX_SEQ_LEN, seed=args.seed,
        loader=data_loader_from_config(args.data_config),
    )

    # ── Harness-facing artefacts ─────────────────────────────────────
    loss_dump_path = os.environ.get("LOSS_DUMP_FILE", "")
    loss_dump_fh = None
    if rank == 0 and loss_dump_path:
        path_obj = Path(loss_dump_path)
        if not path_obj.is_absolute():
            base = Path(os.environ.get("DUMP_DIR", args.out_dir))
            path_obj = base / loss_dump_path
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        loss_dump_fh = path_obj.open("w", encoding="utf-8", buffering=1)

    # ── Training loop ────────────────────────────────────────────────
    fp32_grad_bufs = [torch.zeros_like(p_) for p_ in fp32_master]
    tokens_per_step = args.global_batch_size * MAX_SEQ_LEN
    # MFU(standard) — physical-GEMM enumeration for the Qwen3 dense
    # geometry (no MTP term). Each GEMM [m,k]·[k,n] = 2mkn fwd FLOPs;
    # training adds dgrad + wgrad of equal cost -> 3x fwd per GEMM.
    #   QKV proj : Q(H->H_q) + K(H->H_kv) + V(H->H_kv) = 2H(H_q + 2 H_kv)
    #   Wo proj  : H_q -> H                            = 2 H_q H
    #   attn     : causal Q·Kᵀ + causal attn·V         = 2 S H_q (½ causal)
    #   MLP      : SwiGLU (gate/up/down) = 3 GEMMs [H,ffn]= 6 H·ffn
    #   LM head  : H -> V                              = 2 H V
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
        harness_dp.begin_step(step)
        torch.cuda.synchronize()
        t0 = time.time()

        for buf in fp32_grad_bufs:
            buf.zero_()
        local_lm_sum = torch.zeros(1, device=device, dtype=torch.float64)
        local_lm_n = torch.zeros(1, device=device, dtype=torch.float64)

        for _mb in range(grad_accum_steps):
            harness_dp.begin_microbatch(_mb)
            input_ids, labels, loss_mask = next_batch_fn()
            for p_ in bf16_params:
                p_.grad = None
                if getattr(p_, "main_grad", None) is not None:
                    p_.main_grad = None

            logits, _ = model(input_ids, rope_freqs, recompute=args.recompute,
                              recompute_num_layers=args.recompute_num_layers)
            if harness_dp.capturing():
                _B, _S, _V = logits.shape
                _nll = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, _V).float(), labels.reshape(-1), reduction='none'
                ).reshape(_B, _S)
                harness_dp.capture('loss.per_token.preallreduce', _nll)
            lm_sum, lm_n = masked_ce(logits, labels, loss_mask)
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
        harness_dp.end_microbatch()

        reported_lm, g_lm_n = harness_dp.reduce_loss_scalar(local_lm_sum, local_lm_n)
        local_lm = (local_lm_sum / local_lm_n.clamp(min=1.0)).item()

        norm_factor = 1.0 / max(g_lm_n, 1.0)
        harness_dp.reduce_grads(bf16_params, fp32_grad_bufs, norm_factor=norm_factor)

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

        if rank == 0 and (step + 1) % args.log_interval == 0:
            print(
                f" iteration {step + 1:6d} | total_loss: {total:.4e} | "
                f"lm_loss: {reported_lm:.4e} | "
                f"local_lm: {local_lm:.4e} | grad norm: {grad_norm:.3f} | "
                f"lr: {lr:.6e} | time(ms): {step_time * 1000:.1f}",
                flush=True,
            )
            # Harness wire-format line — MUST stay in sync with
            # ``evals/_common.parse_loss_lines`` (float precision .9e).
            loss_line = (
                f"[LOSS] step={step + 1} global_loss={total:.9e} "
                f"grad_norm={grad_norm:.9e} time_s={step_time:.6f} "
                f"mfu_e2e_standard={mfu:.6f}"
            )
            print(loss_line, flush=True)
            if loss_dump_fh is not None:
                loss_dump_fh.write(loss_line + "\n")
                loss_dump_fh.flush()

    if rank == 0:
        print("=== Training complete ===")
        if loss_dump_fh is not None:
            loss_dump_fh.close()

    # ── Ordered teardown (exit-0 contract; see train_pure_mup_mtp) ──
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
