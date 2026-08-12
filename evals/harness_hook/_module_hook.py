"""Forward / backward hooks + execution-order graph capture.

Reference implementation. The hook + graph-trace shape is stable;
ref stacks where the per-rank ``named_modules()`` view does not match
the canonical layout (TP>1 splits, FSDP sharded modules, or a
captured slice that the bridge wants to filter to a subset of
submodules) interpose by registering on a wrapper of the model
before calling :func:`evals.harness_hook.install`. See that
function's docstring and ``recipes/README.md`` for the bridge
pattern.

Pure PyTorch: only depends on ``torch.nn``. Imported lazily so this
module loads in environments that haven't loaded torch yet (it doesn't,
in practice, but staying lazy is free).

FQN policy
----------
Records are keyed by ``named_modules()`` FQN verbatim — no rename.
The candidate engine is responsible for emitting matching FQNs if it
wants its records compared; the dispatcher's diff loop silently skips
keys present on only one side (see
:func:`evals._capture_diff.diff_capture_dicts`). The sibling
``.graph.json`` makes one-sided mismatches interpretable.

Hash policy
-----------
Each hook fire reduces the captured tensor to a fixed-size
``{hash, shape, dtype}`` record via :func:`._dump.hash_tensor` and
drops the tensor immediately. The capture process never accumulates
tensors in CPU memory — uniform policy for every alignment–resume dump
(forward / backward module activations, manual loss captures, and
parameter gradients).

Key axes
--------
A module record key is ``{namespace}fwd|bwd.<fqn>#<call>`` with two axes:

* **namespace prefix** (from ``step_prefix_getter()``) — the position in
  the run: ``{step_<n>.}rank<r>.{mb<m>.}``, i.e. the training step
  (persistent only), the global ``rank<r>.`` (always), and the
  grad-accumulation microbatch. Single-step alignment
  (``persistent=False``) is driven with ``rank<r>.mb0.``; persistent
  bridges pass ``step_<n>.rank<r>.`` and, per microbatch,
  ``step_<n>.rank<r>.mb<m>.``.
* **call index** (``#<call>`` suffix) — how many times THIS module has
  already fired *under the current prefix*. It counts re-use of a shared
  module within one forward (e.g. an ``output`` head used by both the
  main + MTP paths → ``#0`` / ``#1``) and, because the prefix carries
  ``mb<m>.``, it restarts at ``#0`` each microbatch. The two axes never
  conflate: microbatch is the prefix, call is the suffix.

So alignment keys land as ``rank<r>.mb0.fwd.<fqn>#<call>`` and persistent
keys as ``step_<n>.rank<r>.mb<m>.fwd.<fqn>#<call>``.

Wrapper unwrap
--------------
We unwrap any chain of single-level ``.module`` attributes (covers
``DistributedDataParallel``, ``FSDP``, Megatron's ``Float16Module``,
and similar containers). Deeper or non-``.module`` wrapping is visible
as a truncated FQN that the candidate cannot match — surfacing the
mismatch is preferable to silently inventing an extra prefix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "register_module_forward_hooks",
    "register_module_full_backward_hooks",
]


def _first_tensor(obj: Any) -> Any:
    import torch

    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (tuple, list)):
        for x in obj:
            t = _first_tensor(x)
            if t is not None:
                return t
    return None


def _flat_tensors(obj: Any, out: list | None = None) -> list:
    import torch

    if out is None:
        out = []
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, (tuple, list)):
        for x in obj:
            _flat_tensors(x, out)
    elif isinstance(obj, dict):
        for x in obj.values():
            _flat_tensors(x, out)
    return out


def _hash_or_submit(t: Any, async_hash: bool):
    """Return a ``{hash,shape,dtype}`` record, or a Future of one.

    When ``async_hash`` (default), the snapshot + blake2b are dispatched off
    the training critical path via ``_dump.submit_tensor_hash``: a CUDA
    tensor is copied into a pooled pinned host buffer on a side stream (so
    the copy overlaps the next layers' compute) and hashed once the copy's
    event fires. The caller may immediately drop/overwrite ``t`` after this
    returns. The Future is resolved to the record at dump time (see
    ``_dump.dump_capture_files`` → ``_resolve_future_records``). When
    ``async_hash`` is False, hashing is synchronous (original behaviour).
    """
    from ._dump import hash_tensor

    if not async_hash:
        return hash_tensor(t)
    from ._dump import submit_tensor_hash

    return submit_tensor_hash(t)


def register_module_forward_hooks(
    model: Any,
    *,
    captured_records: dict[str, dict[str, Any]],
    captured_graph: list[dict],
    step_prefix_getter: Callable[[], str] = lambda: "",
    async_hash: bool = True,
) -> None:
    """Attach one forward hook per non-root ``named_modules()`` entry.

    On each forward, the hook appends an entry to ``captured_graph``
    (``{order, fqn, class, input_shapes, input_dtypes, output_shape,
    output_dtype}``) and stores the first-tensor output's hash record
    at ``captured_records[f"{prefix}fwd.{fqn}#{call}"]`` (see
    :func:`._dump.hash_tensor` for the record shape). The tensor
    itself is hashed in place and dropped — nothing is held in CPU
    memory beyond the 32-hex digest + shape + dtype.

    ``step_prefix_getter()`` is invoked on every fire to derive the
    namespace prefix (``{step_<n>.}rank<r>.{mb<m>.}``): alignment
    single-step is driven with ``rank<r>.mb0.``, persistent bridges with
    ``step_<n>.rank<r>.`` and, per microbatch, ``step_<n>.rank<r>.mb<m>.``.
    """
    inner = model
    while hasattr(inner, "module"):
        inner = inner.module

    order_counter = [0]
    # Per-(prefix, fqn) call counter. Every forward fire of a module gets a
    # monotonic ``#<call>`` suffix keyed by the CURRENT prefix, so every
    # re-use of a shared module within one forward is captured distinctly
    # (e.g. an ``output`` head on both the main + MTP paths → ``#0`` / ``#1``)
    # — replacing the old first-fire-wins dedup that dropped the 2nd path.
    # Because ``prefix`` carries ``step_<n>.`` and (in a grad-accum loop)
    # ``mb<i>.``, the counter restarts at ``#0`` each step and each
    # microbatch: the microbatch axis is the prefix, the call axis is the
    # suffix — the two never conflate into one flat number.
    fire_counts: dict[str, int] = {}

    def _make_hook(fqn: str, cls_name: str):
        def fn(_module, inputs, output):
            out_t = _first_tensor(output)
            if out_t is None:
                return
            prefix = step_prefix_getter()
            base = f"{prefix}fwd.{fqn}"
            fire_idx = fire_counts.get(base, 0)
            fire_counts[base] = fire_idx + 1
            key = f"{base}#{fire_idx}"
            in_ts = _flat_tensors(inputs)
            entry: dict = {
                "order": order_counter[0],
                "step_prefix": prefix,
                "fqn": fqn,
                "fire_index": fire_idx,
                "class": cls_name,
                "input_shapes": [tuple(t.shape) for t in in_ts],
                "input_dtypes": [str(t.dtype) for t in in_ts],
                "output_shape": tuple(out_t.shape),
                "output_dtype": str(out_t.dtype),
            }
            captured_graph.append(entry)
            order_counter[0] += 1
            captured_records[key] = _hash_or_submit(out_t, async_hash)

        return fn

    for fqn, module in inner.named_modules():
        if not fqn:
            continue
        module.register_forward_hook(_make_hook(fqn, type(module).__name__))


def register_module_full_backward_hooks(
    model: Any,
    *,
    captured_records: dict[str, dict[str, Any]],
    step_prefix_getter: Callable[[], str] = lambda: "",
    async_hash: bool = True,
) -> None:
    """Attach one full-backward hook per non-root ``named_modules()`` entry.

    On each backward fire, the hook stores the hash record of the
    first non-``None`` entry in ``grad_input`` at
    ``captured_records[f"{prefix}bwd.{fqn}"]``. ``grad_input`` is the
    gradient w.r.t. the module's inputs — the upstream signal flowing
    into the module during backprop, distinct from the per-parameter
    gradients harvested by :mod:`._grad_collector`.

    Used by Level 2 only (alignment / bitwise-singlecard / bitwise-multicard forward-align + per-module
    backward sanity). Module-level backward signals catch divergence
    inside a layer (e.g. a custom kernel) before it propagates into
    the parameter gradients.
    """
    inner = model
    while hasattr(inner, "module"):
        inner = inner.module

    # Per-(prefix, fqn) call counter — same scheme as the forward hook.
    # Every backward fire gets a ``#<call>`` suffix keyed by the current
    # prefix, so a shared module used on multiple paths in one backward
    # (e.g. ``output`` on both the main and MTP heads) records BOTH dgrad
    # paths — replacing the first-fire-wins dedup that dropped all but the
    # first. The microbatch axis rides in the prefix (``mb<i>.``), not this
    # counter, so the call index restarts per microbatch.
    fire_counts: dict[str, int] = {}

    def _make_hook(fqn: str):
        def fn(_module, grad_input, _grad_output):
            if grad_input is None:
                return
            for g in grad_input:
                if g is None:
                    continue
                prefix = step_prefix_getter()
                base = f"{prefix}bwd.{fqn}"
                fire_idx = fire_counts.get(base, 0)
                fire_counts[base] = fire_idx + 1
                key = f"{base}#{fire_idx}"
                captured_records[key] = _hash_or_submit(g, async_hash)
                return  # first non-None grad-input only

        return fn

    for fqn, module in inner.named_modules():
        if not fqn:
            continue
        module.register_full_backward_hook(_make_hook(fqn))
