"""Per-parameter gradient hash, keyed by FQN.

Reference implementation. The collector and its ``grad_attrs``
fallback chain are stable; ref stacks whose populated gradient lives
on a different attribute (FSDP ``_local_shard`` slot, DeepSpeed grad
buckets, …) pass an explicit ``grad_attrs`` tuple to
:func:`evals.harness_hook.install`. See that function's docstring and
``recipes/README.md`` for the bridge pattern.

Same wrapper-unwrap policy as :mod:`._module_hook`. Called twice
per step in M2–M5 — once after ``loss.backward()`` and *before* the
DP all-reduce (``allreduce="pre"``), once after the all-reduce and
*before* ``optimizer.step()`` (``allreduce="post"``). The two flavours
land under distinct keys so the diff can pinpoint whether divergence
appeared during local backward or during the all-reduce reduction.

M1 single-step (no DP, no allreduce in scope) calls only once with
``allreduce="post"`` since pre/post are identical when there is no
reduction; the single sweep is what optim.step would actually consume.

The gradient is hashed in its native device dtype; no fp32-CPU copy
is held in memory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["collect_param_gradients"]


def collect_param_gradients(
    model: Any,
    *,
    captured_records: dict[str, dict[str, Any]],
    allreduce: str,
    grad_attrs: tuple[str, ...] = ("main_grad", "grad"),
    step_prefix_getter: Callable[[], str] = lambda: "",
) -> None:
    """Store populated-gradient hash records.

    Parameters
    ----------
    model:
        nn.Module (or wrapped). We walk ``named_parameters()``.
    captured_records:
        Mutated in-place: each populated gradient produces a
        ``{hash, shape, dtype}`` record (see
        :func:`._dump.hash_tensor`) keyed as
        ``f"{prefix}grad.{name}.{allreduce}allreduce"``.
    allreduce:
        ``"pre"`` (post-backward, pre DP all-reduce) or ``"post"``
        (post all-reduce, pre optim.step). Determines the key
        suffix; the bridge calls this collector at the right hook
        point in its training loop.
    grad_attrs:
        Fallback chain of attribute names. Megatron's distributed
        optimizer accumulates into ``main_grad`` (and may zero
        ``grad``); plain PyTorch uses ``grad``. We try them in order
        and keep the first non-``None`` tensor — single source of
        truth for what the optimizer would actually consume.
    step_prefix_getter:
        Returns the per-step namespace prefix (``""`` for M1
        single-step, ``"step_<n>."`` for M2–M5 persistent).

    Implementation note: the per-FQN hashes are dispatched to the
    cross-tensor thread pool in :mod:`._dump` so blake2b's single-
    core ~1 GB/s ceiling doesn't serialise 100+ params (mostly fp32
    grad bufs, headed by ~800 MB embedding tables for 1B-scale
    models) onto one core per step. The call still blocks until
    every hash completes — no async / callback layer.
    """
    from ._dump import hash_tensors_parallel

    if allreduce not in ("pre", "post"):
        raise ValueError(f"allreduce must be 'pre' or 'post', got {allreduce!r}")

    inner = model
    while hasattr(inner, "module"):
        inner = inner.module

    prefix = step_prefix_getter()
    suffix = f"{allreduce}allreduce"

    items: list[tuple[str, Any]] = []
    for name, p in inner.named_parameters():
        g = None
        for attr in grad_attrs:
            g = getattr(p, attr, None)
            if g is not None:
                break
        if g is None:
            continue
        items.append((f"{prefix}grad.{name}.{suffix}", g))

    for key, record in hash_tensors_parallel(items):
        captured_records[key] = record
