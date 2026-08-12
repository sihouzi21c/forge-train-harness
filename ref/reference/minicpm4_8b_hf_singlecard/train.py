"""Single-card bf16 mixed-precision pretraining of a transformers-library model
(MiniCPM4-8B), laid out by the four blocks a single-card HF pretrainer needs:

  block 1 — model (model.py + : structure + random init + muP width_mult. The
            hf_model/)           MiniCPM4-8B definition (config.json + official
                                 modeling_minicpm.py) is bundled in hf_model/.
  block 2 — data  (data.py)   : tokenizer/dataloader → {tokens, labels, loss_mask}
  block 3 — train (optim.py + : AdamW (wd-by-dim, muP lr) + warmup→cosine, then
            this file)           forward → bf16 loss → backward (grad-accum)
                                 → clip → step
  block 4 — env   (run.sh)    : device/env/network launcher (HF_ENDPOINT, data conf)

Mixed precision is the bf16-master recipe (pure torch, no framework imported):
bf16 params + fp32 master + fp32 grad accumulation + copy-back — the same storage
scheme the torch ref (and Megatron's Float16Optimizer) use.

This is an INDEPENDENT oracle (framework-native loss-trend cross-check), NOT the
bitwise gate truth — that is the self-written train_minicpm4_8b_tp.py pair.
"""
from __future__ import annotations

import os

# Must be set before CUDA initializes (cuBLAS deterministic workspace), hence
# before any module that imports torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import gc
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model import build_model
from optim import build_optimizer, compute_lr
from data import build_stream

# The MiniCPM4-8B definition (config.json + the official modeling_minicpm.py it
# points at via auto_map) is bundled in hf_model/ next to this script, so the
# model architecture is right here — no Hub lookup needed to see what we run.
_BUNDLED_MODEL_DIR = str(Path(__file__).resolve().parent / "hf_model")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path-file", required=True,
                   help="File with DATA_PATH: '<weight> <glob> …' parquet globs "
                        "under FORGE_DATA_DIR.")
    p.add_argument("--out-dir", required=True)
    # Model.
    p.add_argument("--model-id", default=_BUNDLED_MODEL_DIR,
                   help="Defaults to the bundled hf_model/ (MiniCPM4-8B config + "
                        "modeling, offline). Pass a Hub id (openbmb/MiniCPM4-8B) "
                        "or another local dir to override.")
    p.add_argument("--num-layers", type=int, default=4,
                   help="Override num_hidden_layers so the model fits one card.")
    p.add_argument("--attn-impl", default="sdpa",
                   choices=["eager", "sdpa", "flash_attention_2"])
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


def masked_lm_loss(logits, labels, loss_mask):
    """Token-mean CE over the pre-shifted (tokens, labels) from the loader."""
    B, S, V = logits.shape
    nll = F.cross_entropy(
        logits.reshape(-1, V).float(), labels.reshape(-1), reduction="none",
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

    if args.global_batch_size % args.micro_batch_size != 0:
        raise ValueError(
            f"global_batch_size={args.global_batch_size} must be a positive "
            f"multiple of micro_batch_size={args.micro_batch_size}")
    grad_accum_steps = args.global_batch_size // args.micro_batch_size

    # ── block 1: model (bf16 params; the optimizer keeps the fp32 master) ────
    model, config, width_mult = build_model(
        model_id=args.model_id, num_layers=args.num_layers,
        seq_length=args.seq_length, attn_impl=args.attn_impl,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed, device=device,
    )

    # ── block 3a: optimizer + schedule ─────────────────────────────────────
    optim, lr_scales, master_pairs = build_optimizer(
        model, lr=args.lr, weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
        width_mult=width_mult, mup_lr=args.mup_lr,
    )
    # fp32 grad buffers the bf16 grads accumulate into each step.
    fp32_grad_bufs = [torch.zeros_like(m) for _, m in master_pairs]

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=== HF transformers MiniCPM4-8B single-card (bf16-master) ===", flush=True)
    print(f"  model_id={args.model_id} layers={config.num_hidden_layers} "
          f"hidden={getattr(config, 'hidden_size', '?')} "
          f"vocab={getattr(config, 'vocab_size', '?')} "
          f"width_mult={width_mult:g} attn={args.attn_impl}", flush=True)
    print(f"  mbs={args.micro_batch_size} gbs={args.global_batch_size} "
          f"grad_accum={grad_accum_steps} seq={args.seq_length} "
          f"mup_lr={args.mup_lr} grad_ckpt={args.gradient_checkpointing}", flush=True)
    print(f"  trainable params = {n_params / 1e6:.2f}M  lr={args.lr}", flush=True)

    # ── block 2: data ───────────────────────────────────────────────────────
    data_loader = build_stream(
        args.data_path_file, args.micro_batch_size, args.seq_length, args.seed)
    data_iter = iter(data_loader)

    tokens_per_step = args.global_batch_size * args.seq_length
    loss_log = (Path(args.out_dir) / "hf_singlecard_loss.txt").open("w", buffering=1)

    # ── block 3b: training loop ─────────────────────────────────────────────
    for step in range(1, args.train_iters + 1):
        torch.cuda.synchronize()
        t0 = time.time()

        lr = compute_lr(step, args.lr, args.min_lr,
                        args.lr_warmup_iters, args.lr_decay_iters)
        for pg, scale in zip(optim.param_groups, lr_scales):
            pg["lr"] = lr * scale

        for buf in fp32_grad_bufs:
            buf.zero_()
        for p_bf16, _ in master_pairs:
            p_bf16.grad = None

        step_loss = 0.0
        for _mb in range(grad_accum_steps):
            batch = next(data_iter)
            tokens = batch["tokens"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            loss_mask = batch["loss_mask"].to(device, non_blocking=True)

            # Params are bf16 → forward runs in bf16 natively, no autocast.
            out = model(input_ids=tokens)
            logits = out.logits if hasattr(out, "logits") else out[0]
            loss = masked_lm_loss(logits, labels, loss_mask)
            (loss / grad_accum_steps).backward()
            step_loss += loss.item() / grad_accum_steps

            # Drain the bf16 .grad into the fp32 buffer so the next micro-batch
            # accumulates fresh (fp32 grad accumulation).
            with torch.no_grad():
                for (p_bf16, _), buf in zip(master_pairs, fp32_grad_bufs):
                    if p_bf16.grad is not None:
                        buf.add_(p_bf16.grad)
                        p_bf16.grad = None

        for (_, m), buf in zip(master_pairs, fp32_grad_bufs):
            m.grad = buf
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [m for _, m in master_pairs], args.clip_grad).item()
        optim.step()
        with torch.no_grad():  # fp32 master → bf16 model param
            for p_bf16, m in master_pairs:
                p_bf16.data.copy_(m.data)

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

    # Ordered teardown (mirrors train_pure_mup_mtp): drain GPU work, then
    # release the streaming loader's pyarrow chain and the CUDA allocator cache
    # deterministically here rather than leaving it to interpreter-finalize GC.
    torch.cuda.synchronize()
    data_loader.close()
    del data_loader, data_iter
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
