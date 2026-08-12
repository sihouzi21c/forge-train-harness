"""harness_dptp — model-agnostic data-parallel + tensor-parallel reduction
primitives with built-in bitwise capture.

What this is
------------
The 2-D (DP × TP) sibling of :mod:`evals.harness_dp`. The MiniCPM4-8B DP+TP
reference imports ``harness_dptp``, calls :func:`init_groups` once to build
the DP and TP subgroups, :func:`install` once after the model + optimizer are
built, then drives :func:`begin_step` / :func:`reduce_loss_scalar` /
:func:`reduce_grads` (and :func:`capture` for model-specific extras) through
the loop, and :func:`finalize` once at the end. The DP collective and the
hash capture both live here; the ref never mentions ``dist.all_reduce``.

Rank layout (Megatron convention, TP inner)
-------------------------------------------
``global_rank = dp_rank * tp_size + tp_rank``. TP groups are the contiguous
blocks ``[d*tp_size, (d+1)*tp_size)``; DP groups are the strided sets
``range(t, world, tp_size)`` (one per TP shard index ``t``). So a DP group
holds the *same* TP shard across the data-parallel replicas, and a TP group
holds all shards of one replica.

Relationship to ``harness_hook`` / ``harness_dp``
-------------------------------------------------
Like ``harness_dp``, this is a thin wrapper over
:func:`harness_hook.install` → :class:`harness_hook.CaptureSession`. Key
namespacing needs nothing TP-specific here: :func:`harness_hook.install`
prefixes every record with the global ``rank<r>.`` automatically, so each TP
rank captures its *different* tensors (its own vocab / column / row shard)
under disjoint keys (``step_<n>.rank<r>.<...>``). :func:`finalize` just has
every rank write its own ``<output_file>.rank<r>`` — no all-gather, no merge
collective — and the dispatcher merges the per-rank files by their disjoint
``rank<r>.`` prefixes at diff time.

Why no separate TP grad all-reduce
----------------------------------
For *sharded* params (column / row / vocab parallel) each TP rank owns a
disjoint slice; its grad is DP-reduced over the DP group (across the replicas
that hold the *same* slice) and that is the whole story. For *replicated*
params (RMSNorm weights) the incoming activation gradient is already summed
across TP by the ``_CopyToTP`` backward all-reduce in the model (there is no
sequence parallelism), so every TP rank computes an identical full grad. A
plain DP all-reduce then keeps the replicas in sync. Hence :func:`reduce_grads`
is a DP-group all-reduce only — exactly :mod:`harness_dp`'s collective, just
scoped to ``_DP_GROUP``.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
import torch.distributed as dist

from evals import harness_hook

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

__all__ = [
    "add_capture_cli_args",
    "begin_microbatch",
    "begin_step",
    "capture",
    "capturing",
    "end_microbatch",
    "finalize",
    "init_groups",
    "install",
    "install_from_args",
    "reduce_grads",
    "reduce_loss_scalar",
]


class Layout(NamedTuple):
    """Resolved 2-D parallel topology for the current process."""

    dp_group: Any
    tp_group: Any
    dp_size: int
    tp_size: int
    dp_rank: int
    tp_rank: int
    global_rank: int


# Process-global parallel + capture state. ``_SESSION`` is ``None`` whenever
# capture is inactive (level 0 or no ``output_file``), which makes every
# primitive below a no-op. ``_LAYOUT`` is set by :func:`init_groups`.
_SESSION: harness_hook.CaptureSession | None = None
_FQN_BY_ID: dict[int, str] = {}
_LAYOUT: Layout | None = None


def init_groups(dp_size: int, tp_size: int) -> Layout:
    """Build the DP and TP subgroups for a ``dp_size × tp_size`` topology.

    Must be called after ``dist.init_process_group`` on every rank.
    ``world_size`` must equal ``dp_size * tp_size``. Returns (and stashes in
    the module global) a :class:`Layout` with both subgroups and this rank's
    coordinates under the ``global_rank = dp_rank * tp_size + tp_rank`` map.

    Every rank participates in *all* ``new_group`` calls (the collective
    contract), but keeps only the two groups it belongs to.
    """
    global _LAYOUT

    world = dist.get_world_size()
    if world != dp_size * tp_size:
        raise ValueError(f"world_size={world} != dp_size*tp_size={dp_size}*{tp_size}")
    rank = dist.get_rank()
    dp_rank = rank // tp_size
    tp_rank = rank % tp_size

    tp_group = None
    for d in range(dp_size):
        ranks = list(range(d * tp_size, (d + 1) * tp_size))
        g = dist.new_group(ranks)
        if rank in ranks:
            tp_group = g

    dp_group = None
    for t in range(tp_size):
        ranks = list(range(t, world, tp_size))
        g = dist.new_group(ranks)
        if rank in ranks:
            dp_group = g

    _LAYOUT = Layout(
        dp_group=dp_group,
        tp_group=tp_group,
        dp_size=dp_size,
        tp_size=tp_size,
        dp_rank=dp_rank,
        tp_rank=tp_rank,
        global_rank=rank,
    )
    return _LAYOUT


def add_capture_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register ``--hash-capture-level`` / ``--hash-output`` / ``--persistent``
    on the ref's argument parser (identical to :mod:`harness_dp`)."""
    parser.add_argument("--hash-capture-level", type=int, default=0)
    parser.add_argument("--hash-output", type=str, default="")
    parser.add_argument("--persistent", action="store_true")


def _default_writer_rank_predicate() -> bool:
    """Only global rank 0 (``dp_rank==0 and tp_rank==0``) writes the merged
    dump. Used by :func:`finalize`; the per-rank session itself never writes."""
    lay = _LAYOUT
    if lay is None:
        return int(os.environ.get("RANK", "0")) == 0
    return lay.dp_rank == 0 and lay.tp_rank == 0


def install_from_args(
    model: Any,
    optimizer: Any,
    args: argparse.Namespace,
    *,
    grad_attrs: tuple[str, ...] = ("main_grad", "grad"),
) -> harness_hook.CaptureSession | None:
    """Convenience: pull the capture knobs off ``args`` and call :func:`install`.

    ``--hash-output`` takes precedence over the legacy ``HOOK_OUTPUT_FILE`` env
    var; an empty result means a strict no-op install.

    Canonical-state bootstrap: when no capture output is requested but
    ``CANONICAL_STATE_OUTPUT_FILE`` is set, dump the FP32 master state and
    exit — same contract as :mod:`harness_dp`, with two TP-specific twists:

    - the launcher collapses the canonical run to ``TP=1`` (init draws the
      FULL logical tensor and slices per rank, so the TP=1 "shard" IS the
      complete state any TP world would slice from — a TP>1 rank-0 dump
      would only hold rank 0's shards);
    - the dump is ``immediate`` (at install, not at the hijacked first
      ``optimizer.step``): the full 32-layer 8B at TP=1 only fits on an
      80 GB card before the fp32 grad buffers and AdamW state materialize.
    """
    output_file = (getattr(args, "hash_output", "") or "").strip()
    if not output_file:
        output_file = (os.environ.get("HOOK_OUTPUT_FILE", "") or "").strip()
    if not output_file:
        canonical = (os.environ.get("CANONICAL_STATE_OUTPUT_FILE", "") or "").strip()
        if canonical:
            harness_hook.install_canonical_state_dump(
                model,
                optimizer,
                output_file=canonical,
                writer_rank_predicate=_default_writer_rank_predicate,
                immediate=True,
            )
            return None
    return install(
        model,
        optimizer,
        output_file=output_file or None,
        hash_capture_level=int(getattr(args, "hash_capture_level", 0)),
        persistent=bool(getattr(args, "persistent", False)),
        grad_attrs=grad_attrs,
    )


def install(
    model: Any,
    optimizer: Any,
    *,
    output_file: Path | str | None,
    hash_capture_level: int = 2,
    persistent: bool = True,
    grad_attrs: tuple[str, ...] = ("main_grad", "grad"),
) -> harness_hook.CaptureSession | None:
    """Install the capture session; keys carry the global ``rank<r>.`` prefix.

    Requires :func:`init_groups` to have run. Every record the session emits is
    keyed ``step_<n>.rank<r>.<...>`` (``r`` = global ``RANK``, added
    automatically by :func:`harness_hook.install`), so each rank's records are
    disjoint by construction — no TP-specific key namespace and no gather:
    :func:`finalize` has every rank write its own ``<file>.rank<r>`` dump and
    the dispatcher merges them.

    In single-step alignment mode (``persistent=False``) the hook owns its own
    dump-and-exit, returns ``None``, and the module global stays ``None`` so
    every primitive below is inert.
    """
    global _SESSION, _FQN_BY_ID

    if _LAYOUT is None:
        raise RuntimeError("harness_dptp.install called before init_groups")

    session = harness_hook.install(
        model,
        optimizer,
        output_file=output_file,
        hash_capture_level=hash_capture_level,
        persistent=persistent,
        writer_rank_predicate=_default_writer_rank_predicate,
        grad_attrs=grad_attrs,
    )
    _SESSION = session
    if session is None:
        return None

    inner = model
    while hasattr(inner, "module"):
        inner = inner.module
    _FQN_BY_ID = {id(p): name for name, p in inner.named_parameters()}
    return session


def capturing() -> bool:
    """True iff a capture session is active."""
    return _SESSION is not None


def begin_step(n: int) -> None:
    """Start step ``n``'s capture namespace (no-op when capture is inactive)."""
    if _SESSION is not None:
        _SESSION.begin_step(n)


def begin_microbatch(i: int) -> None:
    """Enter microbatch ``i``'s namespace within the current step (no-op when
    capture is inactive). Pair with :func:`end_microbatch` after the
    grad-accumulation loop so per-step captures are not mb-prefixed."""
    if _SESSION is not None:
        _SESSION.begin_microbatch(i)


def end_microbatch() -> None:
    """Exit the microbatch namespace, back to ``step_<n>.`` (no-op when
    capture is inactive)."""
    if _SESSION is not None:
        _SESSION.end_microbatch()


def capture(key: str, tensor: Any) -> None:
    """Hash ``tensor`` under ``key`` for the current step (model-specific
    extras such as ``loss.per_token.preallreduce``)."""
    if _SESSION is not None:
        _SESSION.capture(key, tensor)


def _world_size(group: Any) -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(group)
    return 1


def _dp_group() -> Any:
    return _LAYOUT.dp_group if _LAYOUT is not None else None


def reduce_loss_scalar(
    local_sum: torch.Tensor,
    local_n: torch.Tensor,
    *,
    aux: torch.Tensor | None = None,
) -> tuple[float, float] | tuple[float, float, torch.Tensor]:
    """DP-reduce the loss statistics and capture the loss scalar either side.

    Identical semantics to :func:`harness_dp.reduce_loss_scalar`, scoped to the
    DP group. TP ranks process replicated data, so ``local_sum`` / ``local_n``
    are the same across a TP group; the single DP all-reduce sums them over the
    data-parallel replicas only (the right ``global_n`` for ``norm_factor``).
    The captured scalar is keyed under this rank's ``rank<r>.`` prefix.
    """
    group = _dp_group()
    if _SESSION is not None:
        _SESSION.capture(
            "loss.scalar.preallreduce",
            local_sum.float() / local_n.float().clamp(min=1.0),
        )

    parts = [local_sum, local_n]
    if aux is not None:
        parts.append(aux)
    stats = torch.cat(parts)
    if _world_size(group) > 1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=group)

    g_sum = stats[0].item()
    g_n = stats[1].item()
    reported = g_sum / max(g_n, 1.0)

    if _SESSION is not None:
        _SESSION.capture(
            "loss.scalar.postallreduce",
            torch.tensor([reported], dtype=torch.float32),
        )

    if aux is not None:
        return reported, g_n, stats[2:]
    return reported, g_n


def reduce_grads(
    params: list[Any],
    grad_bufs: list[torch.Tensor],
    *,
    norm_factor: float,
) -> None:
    """DP-reduce gradient buffers in place and capture them either side.

    A DP-group all-reduce only — see the module docstring for why replicated
    params need no extra TP reduce. Per-FQN grads are captured under this rank's
    ``rank<r>.`` prefix, so a column-parallel weight's two shards land as
    ``grad.<fqn>.preallreduce`` under distinct global ranks (e.g. ``rank0.`` and
    ``rank1.``) and the dispatcher-side merge keeps them distinct.
    """
    group = _dp_group()
    if _SESSION is not None:
        _SESSION.capture_batch(
            [
                (f"grad.{_FQN_BY_ID[id(p)]}.preallreduce", b)
                for p, b in zip(params, grad_bufs, strict=False)
            ]
        )

    if _world_size(group) > 1:
        flat = torch._utils._flatten_dense_tensors(grad_bufs)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
        flat.mul_(norm_factor)
        for buf, sub in zip(
            grad_bufs, torch._utils._unflatten_dense_tensors(flat, grad_bufs), strict=False
        ):
            buf.copy_(sub)
    else:
        for buf in grad_bufs:
            buf.mul_(norm_factor)

    if _SESSION is not None:
        _SESSION.capture_batch(
            [
                (f"grad.{_FQN_BY_ID[id(p)]}.postallreduce", b)
                for p, b in zip(params, grad_bufs, strict=False)
            ]
        )


def finalize() -> None:
    """Have every rank write its own ``<file>.rank<r>`` dump.

    Called once on every rank after the training loop. Each rank's records are
    already namespaced by the global ``rank<r>.`` key prefix, so there is
    nothing to gather or merge here (and no collective — this can run at any
    point after the loop): each rank persists its own records, and the
    dispatcher merges the per-rank files at diff time. Global rank 0 also
    writes the unsuffixed ``output_file`` (single-rank / back-compat). No-op
    when capture is inactive.
    """
    if _SESSION is None:
        return

    from evals.harness_hook import dump_capture_files

    try:
        dump_capture_files(
            _SESSION._output_file,
            _SESSION._records,
            _SESSION._graph,
            is_writer=_default_writer_rank_predicate(),
        )
    except Exception as exc:  # never mask the training-loop exit
        sys.stderr.write(f"[harness_dptp] finalize dump failed: {exc!r}\n")
