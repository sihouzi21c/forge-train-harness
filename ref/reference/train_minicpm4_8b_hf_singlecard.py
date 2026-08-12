"""Single-card mixed-precision training of MiniCPM4-8B straight off the
HuggingFace ``transformers`` modeling code.

This is an INDEPENDENT oracle, deliberately separate from the self-written
``train_minicpm4_8b_tp.py`` / ``model_minicpm4_8b_tp.py`` pair (which is the
bitwise gate truth). Here we let ``transformers`` build and run the model —
muP (``scale_emb`` / ``scale_depth`` / ``dim_model_base``) is applied inside
the official modeling code, not by us — so this serves as a framework-native
cross-check of the loss trend, not a bitwise reference.

Design (per the agreed spec):

* **Weights**: random init from ``AutoConfig`` (no ``from_pretrained`` download).
  The official openbmb/MiniCPM4-8B config supplies every architecture + muP
  knob; we only override ``num_hidden_layers`` so the model fits one card.
* **Single card**: plain ``cuda:0``, no ``torch.distributed``.
* **Mixed precision**: weights stay fp32 (the AdamW master copy); the forward +
  loss run under ``torch.autocast(bf16)``. bf16 has fp32 dynamic range, so no
  ``GradScaler`` — this is the canonical bf16 AMP recipe (same as HF Trainer's
  ``bf16=True``).
* **Fit**: full 32 layers + fp32 master + Adam(m/v) ≈ 112 GB > 80 GB, so we
  reduce layers (``--num-layers``, default 4). ``--gradient-checkpointing``
  trades compute for activation memory when pushing the layer count up.
* **Data**: the same Ultra-FineWeb streaming pipeline the main refs use
  (``hf_stream_dataloader`` via ``train_pure_mup_mtp.build_dataloader``), so the
  token stream matches the rest of the harness.
* **Output**: plain stdout logging (``step | loss | lr | grad_norm | tok/s``);
  it does NOT emit the harness ``[LOSS]`` wire line or install any capture.

muP optimizer-side scaling: a muP model is trained with matrix-like weights at
``lr / width_mult`` (``width_mult = hidden_size / dim_model_base = 16``). The HF
modeling code does the *forward* muP scaling but not this *optimizer* scaling,
so we apply it here (toggle with ``--no-mup-lr``) to train the model as intended.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

# Must be set before CUDA initializes (cuBLAS deterministic workspace).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F

# Reuse the harness's Ultra-FineWeb streaming dataloader (single SSOT for the
# token stream). build_dataloader dispatches to hf_stream_dataloader when
# [data].data_loader = hf, read from config/data.toml via --data-config exactly
# as the DP×TP ref does — no DATA_LOADER env value.
from train_pure_mup_mtp import build_dataloader, data_loader_from_config


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path-file", required=True,
                   help="File containing the DATA_PATH token string (weight hf://… pairs).")
    p.add_argument("--data-config", default="",
                   help="Path to the data-value SSOT (config/data.toml, pointed "
                        "at by FORGE_DATA_TOML). [data].data_loader selects the "
                        "loader kind; no DATA_LOADER env value.")
    p.add_argument("--out-dir", required=True)
    # Model source: a local snapshot dir (offline) or a Hub id (needs network).
    p.add_argument("--model-id", default="openbmb/MiniCPM4-8B",
                   help="HF repo id or local dir with config.json + modeling code.")
    p.add_argument("--num-layers", type=int, default=4,
                   help="Override num_hidden_layers so the model fits one card.")
    p.add_argument("--attn-impl", default="sdpa",
                   choices=["eager", "sdpa", "flash_attention_2"],
                   help="attn_implementation passed to from_config.")
    p.add_argument("--gradient-checkpointing", action="store_true")
    # Training shape.
    p.add_argument("--train-iters", type=int, default=50)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--global-batch-size", type=int, default=8)
    p.add_argument("--seq-length", type=int, default=4096)
    # Optim / schedule.
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--lr-warmup-iters", type=int, default=10)
    p.add_argument("--lr-decay-iters", type=int, default=0,
                   help="Cosine decay horizon; 0 = constant after warmup.")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.95)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--mup-lr", dest="mup_lr", action="store_true", default=True,
                   help="Scale matrix-like weights' lr by 1/width_mult (muP, default).")
    p.add_argument("--no-mup-lr", dest="mup_lr", action="store_false")
    p.add_argument("--log-interval", type=int, default=1)
    return p.parse_args()


def compute_lr(step, base_lr, min_lr, warmup, decay_iters):
    """Linear warmup → cosine decay to min_lr (constant if decay_iters<=warmup)."""
    if warmup > 0 and step <= warmup:
        return base_lr * step / warmup
    if decay_iters <= warmup:
        return base_lr
    progress = min(1.0, (step - warmup) / max(1, decay_iters - warmup))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * coeff


def _install_transformers_compat_shim():
    """Make the official MiniCPM4 remote modeling (written for transformers
    ~4.56) import under the newer transformers shipped on the box (5.9).

    The remote ``modeling_minicpm.py`` pulls two symbols that newer transformers
    relocated/removed. Re-providing them is enough — the rest of the modeling
    (cache, attention, masking) already uses APIs present in 5.9. Must run
    BEFORE the remote module is first imported (i.e. before from_config).
    """
    import transformers.utils.import_utils as iu
    import transformers.utils as U
    import transformers.pytorch_utils as pu

    if not hasattr(iu, "is_torch_fx_available"):
        iu.is_torch_fx_available = lambda: False
    if not hasattr(U, "is_torch_fx_available"):
        U.is_torch_fx_available = iu.is_torch_fx_available
    if not hasattr(pu, "is_torch_greater_or_equal_than_1_13"):
        pu.is_torch_greater_or_equal_than_1_13 = True


def build_model(args, device):
    from transformers import AutoConfig, AutoModelForCausalLM

    _install_transformers_compat_shim()

    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    # Reduce depth so the optimizer state fits one 80 GB card.
    config.num_hidden_layers = args.num_layers
    # Train at the harness 4096 window with plain RoPE: drop the longrope
    # rope_scaling (an inference-time long-context path, not exercised at the
    # 4096 training window) so _init_rope uses the default rotary embedding.
    config.rope_scaling = None
    config.max_position_embeddings = args.seq_length
    config.use_cache = False

    width_mult = getattr(config, "hidden_size", 4096) / getattr(config, "dim_model_base", 256)

    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_config(
        config, trust_remote_code=True, attn_implementation=args.attn_impl,
    )
    model = model.to(device=device, dtype=torch.float32)  # fp32 master weights
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    return model, config, width_mult


def build_optimizer(model, args, width_mult):
    """AdamW with weight-decay-by-dim and (optional) muP per-parameter lr.

    * decay: only tensors with ndim>=2 get weight_decay (norms/biases excluded).
    * muP lr: matrix-like weights inside the transformer blocks are scaled by
      1/width_mult; embeddings / lm_head / norms keep the base lr. This is the
      standard muP optimizer recipe for a model whose forward already applies
      the muP activation scaling.
    """
    groups = {}  # (lr_scale, wd) -> [params]
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_matrix = p.dim() >= 2
        wd = args.weight_decay if is_matrix else 0.0
        # muP: scale only the hidden matrix weights (the GQA/MLP projections),
        # not the vocab embedding or the tied lm_head.
        mup_matrix = (
            is_matrix
            and "embed" not in name.lower()
            and "lm_head" not in name.lower()
            and "norm" not in name.lower()
        )
        lr_scale = (1.0 / width_mult) if (args.mup_lr and mup_matrix) else 1.0
        groups.setdefault((lr_scale, wd), []).append(p)

    optim_groups = [
        {"params": ps, "lr": args.lr * lr_scale, "weight_decay": wd,
         "_lr_scale": lr_scale}
        for (lr_scale, wd), ps in groups.items()
    ]
    optim = torch.optim.AdamW(
        optim_groups, betas=(args.adam_beta1, args.adam_beta2), eps=1e-8,
        fused=True,
    )
    return optim


def masked_lm_loss(logits, labels, loss_mask):
    """Token-mean CE over the pre-shifted (tokens, labels) from the loader."""
    B, S, V = logits.shape
    nll = F.cross_entropy(
        logits.reshape(-1, V).float(), labels.reshape(-1),
        reduction="none",
    )
    m = loss_mask.reshape(-1).float()
    return (nll * m).sum() / m.sum().clamp(min=1.0)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required (single-card GPU training script).")
    device = "cuda:0"
    torch.cuda.set_device(0)
    os.makedirs(args.out_dir, exist_ok=True)

    dp_size = args.global_batch_size // args.micro_batch_size
    if dp_size < 1 or args.global_batch_size % args.micro_batch_size != 0:
        raise ValueError(
            f"global_batch_size={args.global_batch_size} must be a positive "
            f"multiple of micro_batch_size={args.micro_batch_size}")
    grad_accum_steps = dp_size  # single card → all accumulation is local

    model, config, width_mult = build_model(args, device)
    optim = build_optimizer(model, args, width_mult)
    lr_scale_per_group = [g["_lr_scale"] for g in optim.param_groups]

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=== HF transformers MiniCPM4-8B single-card bf16 AMP ===", flush=True)
    print(f"  model_id={args.model_id} layers={config.num_hidden_layers} "
          f"hidden={getattr(config, 'hidden_size', '?')} "
          f"vocab={getattr(config, 'vocab_size', '?')} "
          f"width_mult={width_mult:g} attn={args.attn_impl}", flush=True)
    print(f"  mbs={args.micro_batch_size} gbs={args.global_batch_size} "
          f"grad_accum={grad_accum_steps} seq={args.seq_length} "
          f"mup_lr={args.mup_lr} grad_ckpt={args.gradient_checkpointing}", flush=True)
    print(f"  trainable params = {n_params / 1e6:.2f}M  lr={args.lr}", flush=True)

    # ── Ultra-FineWeb stream (single card → dp_rank=0, world_size=1) ─────────
    with open(args.data_path_file) as f:
        raw = f.read().strip().split()
    dl = build_dataloader(
        data_path_args=raw, dp_rank=0, world_size=1,
        micro_batch_size=args.micro_batch_size, seq_length=args.seq_length,
        seed=args.seed, loader=data_loader_from_config(args.data_config),
    )
    data_iter = iter(dl)

    tokens_per_step = args.global_batch_size * args.seq_length
    loss_log = (Path(args.out_dir) / "hf_singlecard_loss.txt").open("w", buffering=1)

    for step in range(1, args.train_iters + 1):
        torch.cuda.synchronize()
        t0 = time.time()

        lr = compute_lr(step, args.lr, args.min_lr,
                        args.lr_warmup_iters, args.lr_decay_iters)
        for pg, scale in zip(optim.param_groups, lr_scale_per_group):
            pg["lr"] = lr * scale

        optim.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _mb in range(grad_accum_steps):
            batch = next(data_iter)
            tokens = batch["tokens"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            loss_mask = batch["loss_mask"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(input_ids=tokens)
                logits = out.logits if hasattr(out, "logits") else out[0]
            loss = masked_lm_loss(logits, labels, loss_mask)
            (loss / grad_accum_steps).backward()
            step_loss += loss.item() / grad_accum_steps

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.clip_grad).item()
        optim.step()

        torch.cuda.synchronize()
        dt = time.time() - t0
        tok_s = tokens_per_step / dt if dt > 0 else 0.0

        if step % args.log_interval == 0:
            mem = torch.cuda.max_memory_allocated() / 1e9
            line = (f"step {step:5d} | loss {step_loss:.6f} | lr {lr:.6e} | "
                    f"grad_norm {grad_norm:.4f} | tok/s {tok_s:8.1f} | "
                    f"peak_mem {mem:.1f}GB | time {dt*1000:.0f}ms")
            print(line, flush=True)
            loss_log.write(line + "\n")

    print("=== Training complete ===", flush=True)
    loss_log.close()
    close = getattr(dl, "close", None)
    if callable(close):
        close()


if __name__ == "__main__":
    main()
