"""Standard alignment–resume capture hook — the only thing the harness ships for these gates.

What this package is
--------------------
The alignment–resume alignment gates compare hashed records (forward activations,
backward signals, per-parameter gradients, per-token loss tensor,
loss scalars) between a candidate engine and a reference run. The
dispatcher only cares about a file contract:

  <dump_dir>/<ref_capture_basename>            (JSON dict; values are
                                                ``{hash, shape, dtype}``;
                                                keys follow
                                                ``{step_<n>.}rank<r>.{mb<m>.}<family>``
                                                where ``<family>`` is
                                                ``fwd.<fqn>#<call>`` /
                                                ``bwd.<fqn>#<call>`` /
                                                ``grad.<fqn>.{pre,post}allreduce`` /
                                                ``loss.*``. ``rank<r>.`` (global
                                                RANK) is on every key; ``mb<m>.``
                                                on fwd/bwd/loss.per_token only;
                                                ``#<call>`` the per-forward call
                                                index; ``step_<n>.`` persistent
                                                mode only)
  <dump_dir>/<ref_capture_basename>.graph.json (per-module execution order)
  <dump_dir>/<candidate_capture_basename>      (same schema, ours side)
  <dump_dir>/<candidate_capture_basename>.graph.json

This package holds the **one true implementation** of "given a model
+ optimizer, produce that file pair" — :func:`install`. Pure PyTorch,
zero framework assumptions. Whatever interposes :func:`install` into
the customer training stack is the **bridge author's job**: see
``evals/harness_hook/recipes/README.md`` for the contract and for the
common interposition patterns (customer entry calls ``install``
inline / agent writes an outside-of-the-customer-entry bridge that
monkey-patches the framework call returning ``(model, optimizer)``).

Two modes
---------
:func:`install` covers two lifecycles under one entry point:

* ``persistent=False`` (alignment): on the first ``optimizer.step``, sweep
  gradients, write the dump, run ordered teardown, and ``raise
  SystemExit(0)``. Returns ``None``. This is the existing alignment lifecycle.

* ``persistent=True`` (bitwise-singlecard / bitwise-multicard / bitwise-perf / resume): returns a
  :class:`CaptureSession`. The bridge drives the session explicitly
  through the training loop — ``begin_step`` at the start of each
  step, ``capture`` and ``capture_grads`` at the appropriate hook
  points, ``dump`` at the end. No SystemExit; the training loop
  runs to completion.

Capture levels
--------------
``hash_capture_level`` is the single per-suite TOML knob threaded
from ``[evals.<suite>].hash_capture_level`` all the way to
``install`` as a typed int (no env var on the way; see the SSOT
section of the project plan).

* ``0`` — install is a no-op; the bridge can call install/session
  methods unconditionally and pay zero cost
* ``1`` — capture loss family (manual via session.capture) + grad
  family (via session.capture_grads, both pre/post allreduce)
* ``2`` — level 1 + module forward hooks (``fwd.<fqn>#<call>``) + module
  full-backward hooks (``bwd.<fqn>#<call>``)

Layering
--------

::

    evals/harness_hook/
        __init__.py         ← install + CaptureSession + install_canonical_state_dump
        _module_hook.py     ← forward / full-backward hooks
        _grad_collector.py  ← gradient hash sweep (pre/post allreduce)
        _canonical_state.py ← FP32 master-weight harvest (bootstrap)
        _dump.py            ← hash_tensor + atomic JSON + sibling .graph.json
        recipes/
            README.md       ← bridge patterns (docs only, no code)

The ``_*.py`` modules are pure-PyTorch and import nothing from any
training framework. There is no per-framework plugin layer — the
harness deliberately does not preempt what shape the customer's
launcher / Python entry / monkey-patch surface takes.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any

from ._canonical_state import collect_canonical_state, default_master_weight_fn
from ._dump import dump_capture_files, hash_tensor, resolve_dump_path
from ._grad_collector import collect_param_gradients
from ._module_hook import (
    register_module_forward_hooks,
    register_module_full_backward_hooks,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

__all__ = [
    "CaptureSession",
    "collect_canonical_state",
    "collect_param_gradients",
    "default_master_weight_fn",
    "dump_capture_files",
    "hash_tensor",
    "install",
    "install_canonical_state_dump",
    "register_module_forward_hooks",
    "register_module_full_backward_hooks",
    "resolve_dump_path",
]


def _default_writer_rank_predicate() -> bool:
    """Default writer-rank: ``RANK == 0`` (or single-process).

    Bridges that use a non-trivial parallel layout (Megatron's
    parallel_state, FSDP, ...) pass their own ``writer_rank_predicate``
    to :func:`install`. The default works for plain ``torchrun`` and
    single-process invocations.
    """
    return int(os.environ.get("RANK", "0")) == 0


def _rank_prefix() -> str:
    """``rank<r>.`` namespace where ``r`` is the global ``RANK`` (default 0).

    Prepended to every capture key so each process's records are distinct
    without any tensor/data-parallel-specific scheme (replaces the old
    ``tp<rank>.`` key prefix). Single-process runs get ``rank0.``. The
    dispatcher merges the per-rank ``<file>.rank<r>`` dumps by these
    disjoint prefixes before diffing.
    """
    return f"rank{os.environ.get('RANK', '0')}."


# Per-process state for alignment single-step mode. Persistent (bitwise-singlecard–resume) mode
# uses CaptureSession's own per-instance state and never touches these
# module-level globals.
_CAPTURED_RECORDS: dict[str, dict[str, Any]] = {}
_CAPTURED_GRAPH: list[dict] = []
_INSTALLED = [False]


def _ordered_teardown_then_exit() -> None:
    """Release GPU / NCCL state cleanly, then ``raise SystemExit(0)``.

    The alignment capture hook hijacks ``optimizer.step`` and never returns
    to the training loop, so this is the single teardown site for the
    single-step hook path. A gate PASSes iff the process exits 0 AND
    its artifact is valid, so we must NOT ``os._exit`` (which masks a
    crashing finalizer): instead drain in-flight GPU work, coordinate
    the NCCL release across ranks, free the allocator cache, and exit
    via ``SystemExit(0)``. ``SystemExit`` is a ``BaseException`` — a
    framework ``except Exception`` cannot swallow it — and it lets
    normal interpreter shutdown (atexit hooks, buffer flushes) run.

    All ranks reach this from the same first ``optimizer.step``, so the
    ``barrier`` is balanced. Teardown is best-effort: a failure here
    must not mask the artifact already on disk, so each step is guarded
    and we still exit 0.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception as exc:  # teardown is best-effort, never mask the artifact
        sys.stderr.write(f"[harness_hook] teardown synchronize failed: {exc!r}\n")
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    except Exception as exc:  # teardown is best-effort, never mask the artifact
        sys.stderr.write(f"[harness_hook] teardown NCCL release failed: {exc!r}\n")
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # teardown is best-effort, never mask the artifact
        sys.stderr.write(f"[harness_hook] teardown empty_cache failed: {exc!r}\n")
    sys.stdout.flush()
    sys.stderr.flush()
    raise SystemExit(0)


class CaptureSession:
    """Per-step capture handle for bitwise-singlecard–resume persistent mode.

    The bridge instantiates exactly one session via
    ``install(..., persistent=True)`` and drives it through the
    training loop. All methods are level-aware no-ops when the
    underlying ``hash_capture_level == 0``, so the bridge code is
    uniform regardless of the configured level.
    """

    def __init__(
        self,
        model: Any,
        *,
        output_file: Path,
        hash_capture_level: int,
        grad_attrs: tuple[str, ...],
        writer_rank_predicate: Callable[[], bool],
    ) -> None:
        self._model = model
        self._output_file = output_file
        self._level = int(hash_capture_level)
        self._grad_attrs = grad_attrs
        self._writer_rank_predicate = writer_rank_predicate
        self._records: dict[str, dict[str, Any]] = {}
        self._graph: list[dict] = []
        self._step: int = 0
        self._mb: int | None = None

    def _prefix(self) -> str:
        # Namespace axes, in key order:
        #   step_<n>.  — the training step (begin_step)
        #   rank<r>.   — the global RANK (always present; disambiguates every
        #                rank's records so the dispatcher can merge them)
        #   mb<m>.     — the grad-accum microbatch WITHIN a step
        #                (begin_microbatch / end_microbatch), empty outside
        #                the microbatch loop so per-step captures
        #                (loss.scalar / capture_grads) are NOT mb-prefixed.
        # The per-forward call index (``#<call>``) lives in the per-tensor
        # key and resets naturally per microbatch because the mb namespace is
        # part of it.
        mb = "" if self._mb is None else f"mb{self._mb}."
        return f"step_{self._step}.{_rank_prefix()}{mb}"

    def begin_step(self, n: int) -> None:
        """Set the per-step namespace; subsequent hook fires and
        ``capture*`` calls key under ``step_<n>.<...>``. A new step also
        clears any lingering microbatch namespace.
        """
        if self._level <= 0:
            return
        self._step = int(n)
        self._mb = None

    def begin_microbatch(self, i: int) -> None:
        """Set the per-microbatch namespace WITHIN the current step; subsequent
        hook fires and ``capture*`` calls key under ``step_<n>.rank<r>.mb<m>.<...>``.

        The driver calls this at the top of each grad-accumulation microbatch
        (``for i in range(grad_accum): begin_microbatch(i); ...``) so
        per-microbatch tensors — module activations/dgrads and the manual
        ``loss.per_token`` capture — stay distinct instead of the later
        microbatch overwriting the earlier one. MUST be paired with
        :func:`end_microbatch` after the loop so per-step captures
        (``loss.scalar`` / :func:`capture_grads`) are not mb-prefixed.
        """
        if self._level <= 0:
            return
        self._mb = int(i)

    def end_microbatch(self) -> None:
        """Clear the microbatch namespace (back to ``step_<n>.``).

        Call after the grad-accumulation loop, before the per-step
        ``loss.scalar`` captures and :func:`capture_grads` — those are
        post-accumulation and must NOT carry an ``mb<i>.`` prefix.
        """
        if self._level <= 0:
            return
        self._mb = None

    def capture(self, key: str, tensor: Any) -> None:
        """Hash ``tensor`` and record under ``step_<n>.<key>``.

        Used by the bridge for tensors not auto-captured by module /
        grad hooks: ``loss.per_token.preallreduce``,
        ``loss.scalar.preallreduce``, ``loss.scalar.postallreduce``.
        """
        if self._level <= 0:
            return
        self._records[f"{self._prefix()}{key}"] = hash_tensor(tensor)

    def capture_batch(self, items: list[tuple[str, Any]]) -> None:
        """Hash a list of ``(suffix, tensor)`` pairs across worker threads.

        Used by bridges whose grad lives outside ``model.named_parameters()``
        (e.g. the torch ref's external ``fp32_grad_bufs`` list) — the
        per-FQN hashes run on the cross-tensor thread pool so blake2b's
        single-core ~1 GB/s ceiling doesn't serialise hundreds of param
        tensors onto one core. Each key gets prefixed with the current
        ``step_<n>.``; sync — the call blocks until every hash finishes.
        """
        if self._level <= 0:
            return
        from ._dump import hash_tensors_parallel

        prefix = self._prefix()
        prefixed = [(f"{prefix}{key}", t) for key, t in items]
        for key, record in hash_tensors_parallel(prefixed):
            self._records[key] = record

    def capture_grads(self, *, allreduce: str) -> None:
        """Sweep ``model.named_parameters()`` and hash each populated
        gradient under ``step_<n>.grad.<fqn>.{allreduce}allreduce``.

        The bridge calls this twice per step: ``allreduce="pre"``
        after backward and *before* DP all-reduce; ``allreduce="post"``
        after the all-reduce and *before* ``optimizer.step``.
        """
        if self._level <= 0:
            return
        collect_param_gradients(
            self._model,
            captured_records=self._records,
            allreduce=allreduce,
            grad_attrs=self._grad_attrs,
            step_prefix_getter=self._prefix,
        )

    def dump(self) -> None:
        """Atomic JSON write to ``output_file`` on the writer rank.

        Called by the bridge at end of training. Level 0 is a no-op.
        """
        if self._level <= 0:
            return
        is_writer = self._writer_rank_predicate()
        dump_capture_files(
            self._output_file,
            self._records,
            self._graph,
            is_writer=is_writer,
        )


def install(
    model: Any,
    optimizer: Any,
    *,
    output_file: Path | str | None,
    hash_capture_level: int = 2,
    persistent: bool = False,
    writer_rank_predicate: Callable[[], bool] = _default_writer_rank_predicate,
    grad_attrs: tuple[str, ...] = ("main_grad", "grad"),
) -> CaptureSession | None:
    """Install the capture hook on ``model`` + ``optimizer``.

    Two modes, both reached through this same entry — the bridge
    picks which one by the ``persistent`` flag, threaded from the
    per-suite TOML knob.

    Parameters
    ----------
    model:
        Any ``nn.Module`` (or wrapped one — we unwrap ``.module``
        attributes). We walk ``named_modules()`` / ``named_parameters()``
        and key tensors by the resulting FQN. No rename table.
    optimizer:
        The optimizer whose ``.step`` we hijack in single-step mode
        (``persistent=False``). In persistent mode the optimizer is
        not modified; the bridge runs the real training loop and
        calls session methods explicitly.
    output_file:
        Absolute path (or ``None`` for a strict no-op) of the JSON
        hash dump. When ``None``, install is a no-op regardless of
        ``hash_capture_level`` — bridges can call
        ``install(..., output_file=None)`` in production without harm.
    hash_capture_level:
        ``0`` / ``1`` / ``2`` (see package docstring). ``0`` makes
        install a no-op; the bridge can still call session methods
        and pay no cost.
    persistent:
        ``False`` (alignment.forward / alignment.backward): single-step. On first
        ``optimizer.step``, sweep grads (under
        ``grad.<fqn>.postallreduce`` key — pre/post collapse in alignment
        because there is no DP allreduce in the single-step shape),
        dump JSON, raise ``SystemExit(0)``. Returns ``None``.

        ``True`` (bitwise-singlecard / bitwise-multicard / bitwise-perf / resume): multi-step. Returns a
        :class:`CaptureSession`; the bridge MUST drive it
        (``begin_step`` / ``capture`` / ``capture_grads`` /
        ``dump``) and call ``session.dump()`` at end of training.
    writer_rank_predicate:
        Zero-arg callable, returns ``True`` iff this process should
        write the artifact. Default: ``RANK == 0``.
    grad_attrs:
        Fallback chain of attribute names to harvest gradients from.
        Default covers Megatron's distributed optimizer slot first,
        plain PyTorch ``.grad`` second.

    Every key carries a ``rank<r>.`` prefix (global ``RANK``) automatically,
    so each process's records are disjoint; ``harness_dptp`` no longer needs
    a per-rank key namespace, and the dispatcher merges the per-rank
    ``<file>.rank<r>`` dumps at diff time.

    Returns
    -------
    ``CaptureSession`` when ``persistent=True``, ``None`` otherwise.
    """
    from pathlib import Path

    if output_file is None:
        return None  # strict no-op

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if persistent:
        # Persistent mode: instantiate a session, register module
        # hooks (level 2 only) that key under the session's current
        # step prefix, return the session for the bridge to drive.
        session = CaptureSession(
            model,
            output_file=output_file,
            hash_capture_level=hash_capture_level,
            grad_attrs=grad_attrs,
            writer_rank_predicate=writer_rank_predicate,
        )
        if hash_capture_level >= 2:
            register_module_forward_hooks(
                model,
                captured_records=session._records,
                captured_graph=session._graph,
                step_prefix_getter=session._prefix,
            )
            register_module_full_backward_hooks(
                model,
                captured_records=session._records,
                step_prefix_getter=session._prefix,
            )
        return session

    # Single-step alignment mode — existing behavior.
    if _INSTALLED[0]:
        return None
    _INSTALLED[0] = True

    # Single-step alignment is one rank, one step, one microbatch: module
    # activations/dgrads key under ``rank<r>.mb0.`` (uniform with persistent
    # mode); post-step grads under ``rank<r>.`` (no mb — post-accumulation).
    _fwd_bwd_prefix = f"{_rank_prefix()}mb0."
    if hash_capture_level >= 2:
        register_module_forward_hooks(
            model,
            captured_records=_CAPTURED_RECORDS,
            captured_graph=_CAPTURED_GRAPH,
            step_prefix_getter=lambda: _fwd_bwd_prefix,
        )
        register_module_full_backward_hooks(
            model,
            captured_records=_CAPTURED_RECORDS,
            step_prefix_getter=lambda: _fwd_bwd_prefix,
        )

    def _capture_and_exit(*_args, **_kwargs):
        try:
            is_writer = writer_rank_predicate()
            if hash_capture_level >= 1:
                collect_param_gradients(
                    model,
                    captured_records=_CAPTURED_RECORDS,
                    allreduce="post",
                    grad_attrs=grad_attrs,
                    step_prefix_getter=_rank_prefix,
                )
            dump_capture_files(
                output_file,
                _CAPTURED_RECORDS,
                _CAPTURED_GRAPH,
                is_writer=is_writer,
            )
        except Exception as exc:
            sys.stderr.write(f"[harness_hook] capture dump failed: {exc!r}\n")
        finally:
            _ordered_teardown_then_exit()

    optimizer.step = _capture_and_exit
    return None


def install_canonical_state_dump(
    model: Any,
    optimizer: Any,
    *,
    output_file: Path | str | None,
    writer_rank_predicate: Callable[[], bool] = _default_writer_rank_predicate,
    master_weight_fn: Callable[[str, Any], Any] | None = None,
    immediate: bool = False,
) -> None:
    """Install a one-shot FP32 canonical-state dump on ``model + optimizer``.

    Sibling of :func:`install`. Same lifecycle (``output_file=None``
    no-op, idempotent, hijacks ``optimizer.step``, ordered teardown +
    ``raise SystemExit(0)`` afterwards), but the captured artifact is
    the FP32 master state every downstream training stack should
    bootstrap from — the ``canonical_state_fp32.pt`` file referenced
    by ``$FORGE_CHECKPOINT_ROOT`` in
    ``ref/reference/train_minicpm4_0.5b_gsm8k.sh``, harness suites
    (``op-long``, ``eval_train_steps``, ``eval_resume_train``,
    ``eval_long_train``, ``eval_capture_align``), and the ours-side
    loader at ``train_engine/src/training_engine_tensor/parameters.py``.

    The bootstrap artifact stays a plain ``torch.save`` tensor file
    (not a JSON hash dump) — the comparator never touches it, the
    ours-side loader needs the real fp32 bytes to initialize from.

    Parameters
    ----------
    model:
        Any ``nn.Module`` (or wrapped one — we unwrap ``.module``
        chains, same policy as :func:`install`). We walk
        ``named_parameters()`` and key tensors by the resulting FQN.
    optimizer:
        The optimizer whose ``.step`` we hijack.
    output_file:
        Absolute path (or ``None`` for a strict no-op) of the
        ``.pt`` file to write.
    writer_rank_predicate:
        Same semantics as :func:`install`. Default: ``RANK == 0``.
    master_weight_fn:
        Per-parameter master-weight resolver. Default tries
        ``param.main_param`` then ``param.data``.
    immediate:
        Dump right now instead of hijacking ``optimizer.step``. The
        master weights are final at install time (nothing mutates them
        before the first step), so the artifact is byte-identical either
        way — but the immediate path skips the forward/backward and,
        crucially, exits before post-install allocations (fp32 grad
        buffers, AdamW ``exp_avg``/``exp_avg_sq``) materialize. That is
        what lets a full 32-layer 8B canonical fit on a single 80 GB
        card (~49 GB bf16 weights + fp32 master vs ~115 GB with the full
        training footprint).

    Mutual-exclusion note
    ---------------------
    :func:`install` (single-step mode) and
    :func:`install_canonical_state_dump` share the process-wide
    ``_INSTALLED`` flag, because both hijack ``optimizer.step`` and
    only one hijack can take effect per process. Bridges keep them in
    separate Python entries — one bridge for alignment capture, one bridge
    for canonical-state bootstrap. Persistent mode does not touch
    ``_INSTALLED``, so it composes freely with neither.
    """
    from pathlib import Path

    if output_file is None:
        return  # strict no-op
    if _INSTALLED[0]:
        return
    _INSTALLED[0] = True

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    def _dump_and_exit(*_args, **_kwargs):
        try:
            is_writer = writer_rank_predicate()
            state = collect_canonical_state(model, master_weight_fn=master_weight_fn)
            if is_writer:
                import torch

                output_file.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = output_file.with_suffix(output_file.suffix + ".tmp")
                torch.save(state, tmp_path)
                tmp_path.replace(output_file)
                sys.stderr.write(
                    f"[harness_hook] canonical-state wrote {output_file} "
                    f"({len(state)} tensors, fp32/cpu)\n"
                )
        except Exception as exc:
            sys.stderr.write(f"[harness_hook] canonical-state dump failed: {exc!r}\n")
        finally:
            _ordered_teardown_then_exit()

    if immediate:
        _dump_and_exit()
    else:
        optimizer.step = _dump_and_exit
