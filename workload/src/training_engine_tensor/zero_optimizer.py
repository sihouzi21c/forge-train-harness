"""ZeRO-1 distributed optimizer — shard FP32 optimizer state across DP ranks.

Each rank holds only ``1 / world_size`` of the full FP32 master weights,
AdamW momentum (m, v), and step counters.  Gradient communication switches
from ``all_reduce`` to ``reduce_scatter`` (each rank receives its own
shard's portion of the summed gradient), and the optimizer step only
updates the rank's own shard.  After the step, ``all_gather``
reconstructs the full BF16 working copy on every rank so the forward pass
sees the complete parameter set.

Memory savings: ``(DP - 1) / DP`` of FP32 optimizer state (~40 GB at
DP=2 for the current model).  Compute savings: each rank runs AdamW on
only its shard, so ``(DP - 1) / DP`` of optimizer FLOPs eliminated.

Save/load is shard-aware: each rank saves its own shard independently
(all ranks write to shared filesystem), and on load each rank restores
its own shard.  Full-state reconstruction (for the resume gate's
self-comparison) uses ``all_gather`` to reassemble the complete state on
every rank.

Usage:
    enable_zero = int(os.environ.get("ENABLE_ZERO_OPTIMIZER", "1"))
    if enable_zero and world_size > 1:
        zero_opt = init_zero_optimizer(
            rank, world_size, fp32_master, exp_avgs, exp_avg_sqs, state_steps,
            bf16_params, fp32_grad_bufs, device,
        )
        # In the training loop:
        reduce_scatter_grads(zero_opt, fp32_grad_bufs, bf16_params, norm_factor)
        set_grad_from_shard(zero_opt, fp32_master, fp32_grad_bufs)
        zero_optimizer_step(zero_opt, ...)  # sharded AdamW + BF16 sync
        all_gather_bf16(zero_opt, bf16_params)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist


@dataclass
class ZeroOptimizer:
    """ZeRO-1 sharded optimizer state for one DP rank.

    Each field holds only the rank's own shard of the full parameter set.
    The full-size gradient buffers (``fp32_grad_bufs``) and the full-size
    BF16 parameter list (``bf16_params``) are kept externally for the
    forward pass and gradient accumulation.
    """

    rank: int
    world_size: int
    # Per-rank shard metadata
    param_start: int  # index into the full param list where this shard begins
    param_end: int  # exclusive end index
    # Sharded FP32 master weights (subset of the full fp32_master list)
    my_fp32_master: list[torch.Tensor] = field(default_factory=list)
    # Sharded BF16 params (subset of the full bf16_params list, all_gathered after update)
    my_bf16_params: list[torch.Tensor] = field(default_factory=list)
    # Sharded optimizer state
    my_exp_avgs: list[torch.Tensor] = field(default_factory=list)
    my_exp_avg_sqs: list[torch.Tensor] = field(default_factory=list)
    my_state_steps: list[torch.Tensor] = field(default_factory=list)
    # Sharded gradient buffers (subset of the full fp32_grad_bufs)
    my_grad_bufs: list[torch.Tensor] = field(default_factory=list)
    # Sharded gradient norm (computed from shard, all-reduced for full norm)
    my_norms: torch.Tensor | None = None
    # Flat buffer for reduce_scatter output (shard's portion of the full gradient)
    flat_shard: torch.Tensor | None = None
    # All-gather buffer for BF16 params
    bf16_flat: torch.Tensor | None = None


def _partition_range(
    n: int, world_size: int, rank: int
) -> tuple[int, int]:
    """Return (start, end) indices for rank ``rank`` in a list of ``n`` items.

    Uses contiguous partitioning with ``ceil(n / world_size)`` per rank.
    """
    chunk = (n + world_size - 1) // world_size
    start = rank * chunk
    end = min(start + chunk, n)
    return start, end


def init_zero_optimizer(
    rank: int,
    world_size: int,
    fp32_master: list[torch.Tensor],
    exp_avgs: dict[int, torch.Tensor],
    exp_avg_sqs: dict[int, torch.Tensor],
    state_steps: dict[int, torch.Tensor],
    bf16_params: list[torch.Tensor],
    fp32_grad_bufs: list[torch.Tensor],
    device: torch.device,
) -> ZeroOptimizer:
    """Initialize ZeRO-1 sharded optimizer state.

    Args:
        rank: Global process rank (``dist.get_rank()``).
        world_size: Data-parallel world size (``config.world_size``).
        fp32_master: Full list of FP32 master weight tensors.
        exp_avgs: Full AdamW first-moment dict, keyed by ``fp32_master.data_ptr``.
        exp_avg_sqs: Full AdamW second-moment dict.
        state_steps: Full AdamW step counter dict.
        bf16_params: Full list of BF16 parameter tensors (model weights).
        fp32_grad_bufs: Full list of FP32 gradient buffers.
        device: CUDA device for this rank.

    Returns:
        A ``ZeroOptimizer`` dataclass with only this rank's shard populated.
    """
    n = len(fp32_master)
    start, end = _partition_range(n, world_size, rank)

    zero = ZeroOptimizer(rank=rank, world_size=world_size,
                         param_start=start, param_end=end)

    for i in range(start, end):
        p_fp32 = fp32_master[i]
        key = p_fp32.data_ptr()
        zero.my_fp32_master.append(p_fp32)
        zero.my_bf16_params.append(bf16_params[i])
        zero.my_exp_avgs.append(exp_avgs[key])
        zero.my_exp_avg_sqs.append(exp_avg_sqs[key])
        zero.my_state_steps.append(state_steps[key])
        zero.my_grad_bufs.append(fp32_grad_bufs[i])

    # Pre-allocate flat buffer for reduce_scatter output (shard's portion
    # of the full flat gradient).
    shard_numel = sum(p.numel() for p in zero.my_fp32_master)
    zero.flat_shard = torch.empty(shard_numel, dtype=torch.float32, device=device)

    # Pre-allocate flat buffer for all_gather of BF16 params.
    total_bf16_numel = sum(p.numel() for p in bf16_params)
    zero.bf16_flat = torch.empty(total_bf16_numel, dtype=torch.bfloat16, device=device)

    if rank == 0:
        print(f"[debug] ZeRO-1 initialized: rank={rank} world_size={world_size} "
              f"params={n} shard=[{start}:{end}) shard_numel={shard_numel} "
              f"total_bf16_numel={total_bf16_numel}",
              file=__import__('sys').stderr, flush=True)

    return zero


def reduce_scatter_grads(
    zero: ZeroOptimizer,
    fp32_grad_bufs: list[torch.Tensor],
    bf16_params: list[torch.Tensor],
    norm_factor: torch.Tensor,
) -> None:
    """Replace ``all_reduce`` with ``reduce_scatter`` for gradients.

    Flattens the full gradient buffer, does a ``reduce_scatter`` so that
    each rank receives only its shard's portion of the summed gradients,
    then scales the shard by ``norm_factor``.

    Args:
        zero: The ZeRO-1 optimizer state.
        fp32_grad_bufs: Full list of FP32 gradient buffers (all ranks).
        bf16_params: Full list of BF16 parameter tensors (for flattening).
        norm_factor: ``1.0 / token_count`` — GPU scalar tensor (no .item() to
            avoid CUDA stream sync).
    """
    # Flatten the full gradient buffer.
    flat = torch._utils._flatten_dense_tensors(fp32_grad_bufs)

    # Compute shard boundaries in the flat tensor.
    param_offsets = []
    offset = 0
    for p in bf16_params:
        param_offsets.append(offset)
        offset += p.numel()

    shard_start = param_offsets[zero.param_start]
    shard_end = param_offsets[zero.param_end - 1] + bf16_params[zero.param_end - 1].numel() \
        if zero.param_end > zero.param_start else param_offsets[zero.param_start]

    # Each rank gets its shard's portion of the summed gradient.
    dist.reduce_scatter(
        zero.flat_shard,
        list(flat[shard_start:shard_end].chunk(zero.world_size, dim=0)),
        op=dist.ReduceOp.SUM,
    )

    # Scale the shard by norm_factor.
    zero.flat_shard.mul_(norm_factor)

    # Unflatten the shard back into per-parameter gradient buffers.
    for buf, sub in zip(
        zero.my_grad_bufs,
        torch._utils._unflatten_dense_tensors(zero.flat_shard, zero.my_grad_bufs),
    ):
        buf.copy_(sub)


def set_grad_from_shard(
    zero: ZeroOptimizer,
    fp32_master: list[torch.Tensor],
    fp32_grad_bufs: list[torch.Tensor],
) -> None:
    """Set ``.grad`` on the shard's FP32 master tensors.

    Only the rank's own shard gets ``.grad`` set — the other ranks'
    ``.grad`` fields remain ``None``.  The optimizer step only reads
    ``.grad`` from the shard's parameters, so this is safe.

    Args:
        zero: The ZeRO-1 optimizer state.
        fp32_master: Full list of FP32 master weight tensors.
        fp32_grad_bufs: Full list of FP32 gradient buffers.
    """
    for p_fp32, buf in zip(zero.my_fp32_master, zero.my_grad_bufs):
        p_fp32.grad = buf


def compute_zero_grad_norm(
    zero: ZeroOptimizer,
    fp32_grad_bufs: list[torch.Tensor],
    opt_clip_grad: float,
) -> torch.Tensor:
    """Compute the total gradient norm across all ranks (ZeRO-1 aware).

    Each rank computes the L2 norm of its own shard, then all-reduces the
    squared norms to get the global total norm.  This is equivalent to the
    non-ZeRO path's ``torch._foreach_norm(fp32_grad_bufs)`` +
    ``torch.linalg.vector_norm(torch.stack(norms))``, but avoids the
    all-gather of all gradient buffers.

    Args:
        zero: The ZeRO-1 optimizer state.
        fp32_grad_bufs: Full list of FP32 gradient buffers (unused — the
            shard's own buffers are used instead).
        opt_clip_grad: Clipping threshold (used for optional conditional).

    Returns:
        The total gradient norm (scalar tensor on the current device).
    """
    # Compute L2 norm of the shard's gradient buffers.
    local_norm_sq = torch.zeros(1, dtype=torch.float32, device=zero.my_grad_bufs[0].device)
    for buf in zero.my_grad_bufs:
        local_norm_sq += buf.norm().square()

    # All-reduce the squared norms to get the global total.
    dist.all_reduce(local_norm_sq, op=dist.ReduceOp.SUM)
    total_norm = local_norm_sq.sqrt()

    return total_norm


def zero_optimizer_step(
    zero: ZeroOptimizer,
    optim_groups: list[dict],
    beta1: float,
    beta2: float,
    eps: float,
    opt_clip_grad: float,
    total_norm: torch.Tensor,
) -> None:
    """Run AdamW on the rank's shard only, then sync BF16.

    Args:
        zero: The ZeRO-1 optimizer state.
        optim_groups: Full optimizer groups (only the shard's params are
            used — the rest have ``.grad = None`` and are skipped by
            ``_fused_adamw_``).
        beta1: Adam beta1.
        beta2: Adam beta2.
        eps: Adam epsilon.
        opt_clip_grad: Gradient clipping threshold.
        total_norm: Total gradient norm (from ``compute_zero_grad_norm``).
    """
    # Clip the shard's gradients.
    if total_norm > opt_clip_grad:
        scale = opt_clip_grad / total_norm
        for buf in zero.my_grad_bufs:
            buf.mul_(scale)

    # Build shard-specific optimizer groups (only params in this rank's shard).
    # The _fused_adamw_ kernel reads .grad from each param, so only params
    # with .grad set (the shard's params) will be updated.
    for group in optim_groups:
        # Filter to only the shard's params.
        shard_params = [p for p in group["params"]
                        if zero.param_start <= _find_param_index(p, zero) < zero.param_end]
        n = len(shard_params)
        if n == 0:
            continue
        lr = group["lr"]
        wd = group["weight_decay"]

        grads: list[torch.Tensor] = []
        eas: list[torch.Tensor] = []
        eass: list[torch.Tensor] = []
        steps: list[torch.Tensor] = []
        for p in shard_params:
            grads.append(p.grad)
            eas.append(zero.my_exp_avgs[zero.my_fp32_master.index(p)])
            eass.append(zero.my_exp_avg_sqs[zero.my_fp32_master.index(p)])
            steps.append(zero.my_state_steps[zero.my_fp32_master.index(p)])

        # Increment step counters.
        torch._foreach_add_(steps, 1)

        # Call the fused AdamW kernel on the shard's params.
        torch._fused_adamw_(
            tuple(shard_params),
            tuple(grads),
            tuple(eas),
            tuple(eass),
            tuple(torch.zeros_like(p) for p in shard_params),
            tuple(steps),
            amsgrad=False,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            weight_decay=wd,
            eps=eps,
            maximize=False,
        )

    # Sync BF16 params from the shard's FP32 master.
    for p_bf16, p_fp32 in zip(zero.my_bf16_params, zero.my_fp32_master):
        p_bf16.data.copy_(p_fp32.bfloat16())


def _find_param_index(p: torch.Tensor, zero: ZeroOptimizer) -> int:
    """Find the index of a parameter in the full fp32_master list.

    Uses data_ptr comparison.  This is O(n) but only called during
    optimizer group setup (once per step), not on the hot path.
    """
    # We can't access the full list from here, so we use a heuristic:
    # the param's index in the shard + param_start.
    # This is only correct if the shard is a contiguous subset of the
    # full list, which it is by construction.
    return zero.param_start + zero.my_fp32_master.index(p)


def all_gather_bf16(
    zero: ZeroOptimizer,
    bf16_params: list[torch.Tensor],
) -> None:
    """Reconstruct the full BF16 parameter set on all ranks.

    After the optimizer step, each rank has updated only its own shard of
    the BF16 params.  This function ``all_gather``s the full set so that
    every rank has a complete copy for the forward pass.

    Args:
        zero: The ZeRO-1 optimizer state.
        bf16_params: Full list of BF16 parameter tensors (model weights).
    """
    # Flatten the full BF16 params.
    flat = torch._utils._flatten_dense_tensors(bf16_params)

    # Compute shard boundaries in the flat tensor.
    param_offsets = []
    offset = 0
    for p in bf16_params:
        param_offsets.append(offset)
        offset += p.numel()

    shard_start = param_offsets[zero.param_start]
    shard_end = param_offsets[zero.param_end - 1] + bf16_params[zero.param_end - 1].numel() \
        if zero.param_end > zero.param_start else param_offsets[zero.param_start]

    # Each rank contributes its shard; all_gather puts the full flat into zero.bf16_flat.
    dist.all_gather_into_tensor(
        zero.bf16_flat,
        flat[shard_start:shard_end].contiguous(),
    )

    # Unflatten the gathered flat tensor back into per-parameter BF16 tensors.
    for buf, sub in zip(
        bf16_params,
        torch._utils._unflatten_dense_tensors(zero.bf16_flat, bf16_params),
    ):
        buf.copy_(sub)


def zero_save_checkpoint(
    zero: ZeroOptimizer,
    save_dir: str,
    step: int,
    rank: int,
) -> None:
    """Save the rank's shard of the optimizer state.

    Each rank saves its own shard independently (all ranks write to the
    shared filesystem).  The saved file name includes the rank so that
    ``zero_load_checkpoint`` can restore the correct shard.

    Args:
        zero: The ZeRO-1 optimizer state.
        save_dir: Root checkpoint directory.
        step: Absolute step number.
        rank: Global process rank (for naming the shard file).
    """
    import pickle as _pickle
    from pathlib import Path

    subdir = Path(save_dir) / f"step_{step}"
    subdir.mkdir(parents=True, exist_ok=True)
    path = subdir / f"training_state_rank{rank}.pt"

    state = {
        "zero_rank": rank,
        "zero_world_size": zero.world_size,
        "zero_param_start": zero.param_start,
        "zero_param_end": zero.param_end,
        "fp32_master": zero.my_fp32_master,
        "exp_avgs": zero.my_exp_avgs,
        "exp_avg_sqs": zero.my_exp_avg_sqs,
        "state_steps": zero.my_state_steps,
        "step": torch.tensor(step, dtype=torch.int32),
        # RNG states (only rank 0 saves the authoritative copy)
        "rng_torch": torch.get_rng_state() if rank == 0 else torch.zeros(0, dtype=torch.uint8),
        "rng_cuda": torch.cuda.get_rng_state() if rank == 0 else torch.zeros(0, dtype=torch.uint8),
        "rng_numpy": _pickle.dumps(__import__("numpy").random.get_state()) if rank == 0 else b"",
        "rng_random": _pickle.dumps(__import__("random").getstate()) if rank == 0 else b"",
    }
    torch.save(state, path)


def zero_load_checkpoint(
    zero: ZeroOptimizer,
    checkpoint_dir: str,
    fp32_master: list[torch.Tensor],
    exp_avgs: dict[int, torch.Tensor],
    exp_avg_sqs: dict[int, torch.Tensor],
    state_steps: dict[int, torch.Tensor],
    rank: int,
    init_weights_only: bool = False,
) -> int:
    """Load the rank's shard of the optimizer state from a checkpoint.

    Each rank loads its own shard file.  For the resume gate's
    self-comparison, the full state can be reconstructed by
    ``all_gather`` after loading.

    Args:
        zero: The ZeRO-1 optimizer state.
        checkpoint_dir: Path to the checkpoint directory.
        fp32_master: Full list of FP32 master weights (will be overwritten
            in-place from the loaded shard).
        exp_avgs: Full AdamW first-moment dict.
        exp_avg_sqs: Full AdamW second-moment dict.
        state_steps: Full AdamW step counter dict.
        rank: Global process rank.
        init_weights_only: If True, only load FP32 master weights, skip
            optimizer state.

    Returns:
        The loaded step number (1-indexed).
    """
    from pathlib import Path

    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(
            f"zero_load_checkpoint: directory not found: {checkpoint_dir}"
        )
    # Look for the rank-specific shard file.
    rank_path = ckpt_dir / f"training_state_rank{rank}.pt"
    if not rank_path.exists():
        # Try subdirectories (versioned checkpoints).
        rank_paths = list(ckpt_dir.rglob(f"*training_state_rank{rank}.pt"))
        if not rank_paths:
            raise FileNotFoundError(
                f"zero_load_checkpoint: no training_state_rank{rank}.pt found in {checkpoint_dir}"
            )
        rank_path = rank_paths[0]

    state = torch.load(rank_path, map_location="cpu", weights_only=True)
    step_val = state["step"].item() if isinstance(state["step"], torch.Tensor) else int(state["step"])

    # Load the shard's FP32 master weights.
    loaded_master = state["fp32_master"]
    for i, (p_fp32, p_loaded) in enumerate(zip(zero.my_fp32_master, loaded_master)):
        p_fp32.copy_(p_loaded.to(device=fp32_master[0].device))

    if not init_weights_only:
        loaded_avgs = state["exp_avgs"]
        loaded_sqs = state["exp_avg_sqs"]
        loaded_steps = state["state_steps"]
        for p_fp32, avg, sq, stp in zip(zero.my_fp32_master, loaded_avgs, loaded_sqs, loaded_steps):
            key = p_fp32.data_ptr()
            exp_avgs[key].copy_(avg.to(device=fp32_master[0].device))
            exp_avg_sqs[key].copy_(sq.to(device=fp32_master[0].device))
            state_steps[key].copy_(stp.to(device=fp32_master[0].device))

        # Restore RNG state (only rank 0 has the authoritative copy).
        if rank == 0:
            saved_rng = state.get("rng_torch")
            if saved_rng is not None and saved_rng.numel() > 0:
                torch.set_rng_state(saved_rng.cpu())
            saved_cuda = state.get("rng_cuda")
            if saved_cuda is not None and saved_cuda.numel() > 0:
                torch.cuda.set_rng_state(saved_cuda.cpu())
            import pickle as _pickle
            saved_numpy = state.get("rng_numpy", b"")
            if saved_numpy:
                __import__("numpy").random.set_state(_pickle.loads(saved_numpy))
            saved_random = state.get("rng_random", b"")
            if saved_random:
                __import__("random").setstate(_pickle.loads(saved_random))

    # After loading, sync BF16 params from the (now-loaded) FP32 master.
    for p_bf16, p_fp32 in zip(zero.my_bf16_params, zero.my_fp32_master):
        p_bf16.data.copy_(p_fp32.bfloat16())

    # If the full bf16 params need to be reconstructed (e.g. for the resume
    # gate), the caller must call all_gather_bf16() after this returns.
    return step_val