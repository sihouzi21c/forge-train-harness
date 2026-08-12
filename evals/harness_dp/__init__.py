"""harness_dp — model-agnostic data-parallel reduction primitives with
built-in M1–M5 capture.

What this is
------------
The reference training loop used to hand-roll the DP all-reduce inline and
have its capture points *injected at runtime* by ``ref/bridges/interposer.py``
(source string-replacement). This package moves those call sites **into the
ref itself**: the ref imports ``harness_dp``, calls :func:`install` once after
the model + optimizer are built, then drives :func:`begin_step` /
:func:`reduce_loss_scalar` / :func:`reduce_grads` (and :func:`capture` for
model-specific extras) through the loop. The DP collective and the M1–M5
hash capture both live here; the ref no longer mentions ``dist.all_reduce``.

Relationship to ``harness_hook``
--------------------------------
``harness_dp`` is a thin wrapper over :mod:`evals.harness_hook`. The hook
package holds the one true "given a model + optimizer, produce the hash dump"
implementation (:func:`harness_hook.install` → :class:`CaptureSession`, plus
the blake2b ``hash_tensor`` / ``hash_tensors_parallel`` machinery). We reuse
it verbatim — same session, same hashing — so the dump produced through these
primitives is byte-for-byte what the interposer produced before.

Capture families emitted (unchanged from the interposer wiring)
---------------------------------------------------------------
* ``fwd.<fqn>#<call>`` / ``bwd.<fqn>#<call>`` — module forward / full-backward
  hooks (registered by :func:`harness_hook.install` at ``hash_capture_level
  >= 2``); ``#<call>`` is the per-forward call index.
* ``grad.<fqn>.{pre,post}allreduce`` — per-parameter grad buffers, captured
  by :func:`reduce_grads` either side of the DP all-reduce.
* ``loss.scalar.{pre,post}allreduce`` — captured by :func:`reduce_loss_scalar`.
* ``loss.per_token.preallreduce`` — model-specific NLL recompute, captured by
  the ref via :func:`capture` (not a DP primitive).

The forward-looking ``tokens.{pre,post}allreduce`` family is **not** emitted
here; :func:`reduce_loss_scalar` computes the global token count internally
(for ``norm_factor``) but does not hash it, so the dump key set stays identical
to the pre-refactor reference.

Two lifecycles, one entry
-------------------------
:func:`install` forwards ``persistent`` to :func:`harness_hook.install`:

* ``persistent=False`` (M1.fwd / M1.bwd, single step): the hook hijacks
  ``optimizer.step``, sweeps grads, dumps, and ``raise SystemExit(0)``.
  :func:`install` returns ``None`` and the module-global session stays
  ``None``, so every :func:`begin_step` / :func:`reduce_*` / :func:`capture`
  call in the ref is a zero-cost no-op — the dump comes purely from the
  module hooks + the step-time grad sweep.
* ``persistent=True`` (M2–M5, multi step): returns a live
  :class:`CaptureSession`. The ref drives it through the loop; an ``atexit``
  handler dumps at process end (covers normal exit and exceptions alike).
"""

from __future__ import annotations

import atexit
import os
import sys
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from evals import harness_hook

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable
    from pathlib import Path

__all__ = [
    "add_capture_cli_args",
    "begin_microbatch",
    "begin_step",
    "capture",
    "capturing",
    "end_microbatch",
    "install",
    "install_from_args",
    "reduce_grads",
    "reduce_loss_scalar",
]

# Process-global capture state, mirroring the interposer's
# ``_SESSION`` / ``_HARNESS_GRAD_NAMES`` globals. ``_SESSION`` is ``None``
# whenever capture is inactive (single-step M1 mode, level 0, or no
# ``output_file``), which makes every primitive below a no-op.
_SESSION: harness_hook.CaptureSession | None = None
_FQN_BY_ID: dict[int, str] = {}


def add_capture_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register the capture CLI flags on the ref's argument parser.

    The dispatcher (``evals._common.run_ref_capture``) forwards
    ``--hash-capture-level N --hash-output PATH [--persistent]`` to the
    launcher, which passes them straight through to the ref's python entry.
    These mirror the flags the interposer used to parse-and-strip; now the
    ref owns them. Defaults match the old interposer (``level=0``,
    ``output=""``) so a non-capture trajectory run installs a strict no-op.
    """
    parser.add_argument("--hash-capture-level", type=int, default=0)
    parser.add_argument("--hash-output", type=str, default="")
    parser.add_argument("--persistent", action="store_true")


def install_from_args(
    model: Any,
    optimizer: Any,
    args: argparse.Namespace,
    *,
    writer_rank_predicate: Callable[[], bool] | None = None,
    grad_attrs: tuple[str, ...] = ("main_grad", "grad"),
) -> harness_hook.CaptureSession | None:
    """Convenience: pull the capture knobs off ``args`` and call :func:`install`.

    ``--hash-output`` takes precedence over the legacy ``HOOK_OUTPUT_FILE``
    env var (same precedence the interposer used); an empty result means a
    strict no-op install.

    Canonical-state bootstrap: when no capture output is requested but
    ``CANONICAL_STATE_OUTPUT_FILE`` is set, install the one-shot FP32 master
    dump instead. The torch bridge stopped routing through the interposer
    (whose ``elif CANONICAL_OUTPUT`` branch owned this), so the ref-side
    capture wiring owns it now. The hook dumps on the first ``optimizer.step``
    and exits, so the surrounding gate's ``num_steps`` is irrelevant.
    """
    output_file = (getattr(args, "hash_output", "") or "").strip()
    if not output_file:
        output_file = (os.environ.get("HOOK_OUTPUT_FILE", "") or "").strip()
    if not output_file:
        canonical = (os.environ.get("CANONICAL_STATE_OUTPUT_FILE", "") or "").strip()
        if canonical:
            canon_kwargs: dict[str, Any] = {"output_file": canonical}
            if writer_rank_predicate is not None:
                canon_kwargs["writer_rank_predicate"] = writer_rank_predicate
            harness_hook.install_canonical_state_dump(model, optimizer, **canon_kwargs)
            return None
    return install(
        model,
        optimizer,
        output_file=output_file or None,
        hash_capture_level=int(getattr(args, "hash_capture_level", 0)),
        persistent=bool(getattr(args, "persistent", False)),
        writer_rank_predicate=writer_rank_predicate,
        grad_attrs=grad_attrs,
    )


def install(
    model: Any,
    optimizer: Any,
    *,
    output_file: Path | str | None,
    hash_capture_level: int = 2,
    persistent: bool = True,
    writer_rank_predicate: Callable[[], bool] | None = None,
    grad_attrs: tuple[str, ...] = ("main_grad", "grad"),
) -> harness_hook.CaptureSession | None:
    """Install the capture session and bind the per-parameter FQN map.

    Delegates to :func:`harness_hook.install`. In persistent mode the returned
    session is stashed in the module global and an ``atexit`` dump is
    registered (mirroring the interposer's ``_install_torch_ref_loop_globals``
    + ``_wire_hook``). In single-step mode the hook returns ``None`` (it owns
    its own dump-and-exit), so the global stays ``None`` and every primitive
    below is inert.

    ``model`` is walked via ``named_parameters()`` (unwrapping ``.module``
    chains) to build ``id(param) -> fqn``; :func:`reduce_grads` resolves grad
    keys through this map, so the ref hands it the real model parameters (whose
    ids live in the map), not the fp32 master clones.
    """
    global _SESSION, _FQN_BY_ID

    kwargs: dict[str, Any] = {
        "output_file": output_file,
        "hash_capture_level": hash_capture_level,
        "persistent": persistent,
        "grad_attrs": grad_attrs,
    }
    if writer_rank_predicate is not None:
        kwargs["writer_rank_predicate"] = writer_rank_predicate

    session = harness_hook.install(model, optimizer, **kwargs)
    _SESSION = session
    if session is None:
        return None

    inner = model
    while hasattr(inner, "module"):
        inner = inner.module
    _FQN_BY_ID = {id(p): name for name, p in inner.named_parameters()}

    def _atexit_dump() -> None:
        try:
            if _SESSION is not None:
                _SESSION.dump()
        except Exception as exc:  # never mask the underlying training-loop exit
            sys.stderr.write(f"[harness_dp] atexit dump failed: {exc!r}\n")

    atexit.register(_atexit_dump)
    return session


def capturing() -> bool:
    """True iff a capture session is active.

    Lets the ref guard model-specific recompute (e.g. the per-token NLL) so it
    costs nothing on non-capture trajectory runs — mirrors the interposer's
    ``if _HARNESS_SESSION is not None`` guard around the same recompute.
    """
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
    """Hash ``tensor`` under ``key`` for the current step.

    For model-specific captures that are not DP primitives — today only
    ``loss.per_token.preallreduce`` (the per-token NLL the ref recomputes on
    its first microbatch).
    """
    if _SESSION is not None:
        _SESSION.capture(key, tensor)


def _world_size(group: Any) -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(group)
    return 1


def reduce_loss_scalar(
    local_sum: torch.Tensor,
    local_n: torch.Tensor,
    *,
    aux: torch.Tensor | None = None,
    group: Any = None,
) -> tuple[float, float] | tuple[float, float, torch.Tensor]:
    """DP-reduce the loss statistics and capture the loss scalar either side.

    Captures ``loss.scalar.preallreduce`` = ``local_sum/local_n.clamp(min=1)``,
    sum-reduces ``cat([local_sum, local_n, *(aux,)])`` in a **single**
    all-reduce, then captures ``loss.scalar.postallreduce`` = the reported
    global mean.

    Returns ``(reported_mean, global_n)``; when ``aux`` is given, returns
    ``(reported_mean, global_n, aux_reduced)`` where ``aux_reduced`` is the
    summed ``aux`` slice (used by the MiniCPM ref for the MTP loss, which the
    reference fuses into the same collective — passing it as ``aux`` keeps the
    buffer and the single all-reduce byte-identical to the pre-refactor ref).

    ``global_n`` is returned so the caller can form ``norm_factor =
    1/max(global_n, 1)``; the token count itself is not hashed (no
    ``tokens.*`` family in this step).
    """
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
    group: Any = None,
) -> None:
    """DP-reduce gradient buffers in place and capture them either side.

    Captures ``grad.<fqn>.preallreduce`` for each buffer, sum-reduces the
    flattened buffers across the DP group, scales by ``norm_factor``, copies
    back, then captures ``grad.<fqn>.postallreduce``. With a single process the
    all-reduce is skipped and the buffers are scaled in place (identical
    numerics to the reference's ``world_size == 1`` branch).

    ``params`` supplies the FQN keys: each is resolved through the
    ``id(param) -> fqn`` map built at :func:`install`, so the ref passes its
    real model parameters here (positionally aligned with ``grad_bufs``).
    """
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
