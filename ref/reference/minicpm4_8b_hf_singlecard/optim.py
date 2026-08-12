"""Block 3a — optimizer + LR schedule.

AdamW with two muP/regularization rules a muP model needs:
  * weight_decay only on matrix-like weights (ndim>=2); norms/biases excluded.
  * muP lr: matrix weights inside the blocks scaled by 1/width_mult; embeddings /
    lm_head / norms keep base lr (the forward already applies muP activation scaling).

bf16-master recipe: the model's params are bf16; the optimizer updates fp32
*master* copies. The loop routes bf16 grads → fp32 and copies master → bf16 param
each step (same storage scheme as Megatron's Float16Optimizer — but this is pure
torch, no framework imported). build_optimizer returns the (bf16 param, fp32
master) pairs the loop needs.
"""
from __future__ import annotations

import math

import torch


def build_optimizer(model, lr, weight_decay, betas, width_mult, mup_lr):
    groups = {}  # (lr_scale, wd) -> [fp32 master params the optimizer steps]
    master_pairs = []  # (bf16 model param, fp32 master)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_matrix = p.dim() >= 2
        wd = weight_decay if is_matrix else 0.0
        mup_matrix = (
            is_matrix
            and "embed" not in name.lower()
            and "lm_head" not in name.lower()
            and "norm" not in name.lower()
        )
        lr_scale = (1.0 / width_mult) if (mup_lr and mup_matrix) else 1.0
        master = p.detach().float().clone().requires_grad_(True)
        master_pairs.append((p, master))
        groups.setdefault((lr_scale, wd), []).append(master)

    optim_groups = [
        {"params": ps, "lr": lr * s, "weight_decay": wd, "_lr_scale": s}
        for (s, wd), ps in groups.items()
    ]
    optim = torch.optim.AdamW(optim_groups, betas=betas, eps=1e-8, fused=True)
    lr_scales = [g["_lr_scale"] for g in optim.param_groups]
    return optim, lr_scales, master_pairs


def compute_lr(step, base_lr, min_lr, warmup, decay_iters):
    """Linear warmup → cosine decay to min_lr (constant if decay_iters<=warmup)."""
    if warmup > 0 and step <= warmup:
        return base_lr * step / warmup
    if decay_iters <= warmup:
        return base_lr
    progress = min(1.0, (step - warmup) / max(1, decay_iters - warmup))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * coeff
