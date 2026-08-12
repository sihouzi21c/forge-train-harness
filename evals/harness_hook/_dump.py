"""Atomic JSON hash-dump + sibling ``.graph.json`` writer.

Each captured tensor is reduced to a fixed-size record at hook fire
time::

    {"hash": "<32-hex>", "shape": [...], "dtype": "<name>"}

— never accumulated in CPU memory. The producer writes a single JSON
file mapping FQN-keyed names to records; the comparator reads it back
as plain Python dicts. Only the writer rank writes; other ranks return
cleanly. The caller is expected to run ordered teardown and
``raise SystemExit(0)`` afterwards on every rank so the NCCL barrier is
balanced and collectives release.

A module-level :class:`ThreadPoolExecutor` is exposed via
:func:`hash_tensors_parallel` so multi-FQN batch sites (the per-step
grad sweep in :mod:`._grad_collector`) can parallelise across cores
without changing the per-tensor sync semantics of :func:`hash_tensor`
itself. ``hashlib.blake2b`` releases the GIL during ``update``, so the
threads make real progress; each call still blocks until its own hash
completes (no async / callback layer), matching the bridge-author
contract.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    import torch

__all__ = [
    "dump_capture_files",
    "hash_tensor",
    "hash_tensors_parallel",
    "resolve_dump_path",
]


# Single source of truth for the hash algorithm. blake2b-128 is stdlib,
# ~1 GB/s/core, and 128-bit collision space is overkill for byte-equality
# discrimination across the few hundred FQN records a single capture
# produces.
_HASH_DIGEST_BYTES = 16

# Per-chunk element count for streaming hashing — bounds peak CPU
# memory to (chunk_elems × itemsize) regardless of full tensor size.
# 8M elements ≈ 16 MB at bf16, 32 MB at fp32.
_HASH_CHUNK_ELEMS = 8 * 1024 * 1024


def _pool_workers() -> int:
    """Worker count for the cross-tensor hash thread pool.

    blake2b's bottleneck is single-core CPU throughput (~1 GB/s),
    so the practical ceiling is min(physical cores - 2, 16) — leave
    a couple of cores for the main training thread + the launcher.
    """
    env = os.environ.get("FORGE_HASH_WORKERS", "").strip()
    if env.isdigit():
        return max(1, int(env))
    cpu = os.cpu_count() or 4
    return max(1, min(16, cpu - 2))


# Module-level singleton pool. Lazy-init on first parallel batch so
# pure-M1 single-step bridges (which never call the batch entry) don't
# pay the thread-pool startup cost.
_POOL: ThreadPoolExecutor | None = None


def _pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=_pool_workers(), thread_name_prefix="harness-hash")
    return _POOL


def hash_tensor(t: torch.Tensor) -> dict[str, Any]:
    """Stream a tensor's raw bytes into blake2b-128; return the record.

    The returned dict carries the hex digest plus shape and dtype so
    a hash mismatch failure can still surface a useful diagnostic
    (shape or dtype drift). The input tensor's storage is touched
    chunk-by-chunk on its native device, copied to CPU per chunk,
    re-viewed as ``uint8`` and fed to the hash; the full tensor never
    materializes on CPU at once. After this call the producer is
    expected to drop its reference so the device tensor can be freed.

    Sync by design — no side stream, no thread pool **inside** the
    per-tensor hash (the bytes order matters for blake2b, so chunks
    cannot be parallelised within a single tensor without changing
    the wire format). Cross-tensor parallelism lives in
    :func:`hash_tensors_parallel`.
    """
    import torch

    t = t.detach().contiguous()
    shape = list(t.shape)
    dtype = str(t.dtype).removeprefix("torch.")

    h = hashlib.blake2b(digest_size=_HASH_DIGEST_BYTES)
    if t.numel() > 0:
        flat = t.reshape(-1)
        for chunk in flat.split(_HASH_CHUNK_ELEMS):
            chunk_cpu = chunk.cpu()
            # uint8 view works for any element-size dtype (bf16, fp16,
            # int*, uint*) and gives numpy() a buffer it actually
            # supports (numpy has no native bf16).
            byte_view = chunk_cpu if chunk_cpu.dtype == torch.uint8 else chunk_cpu.view(torch.uint8)
            h.update(byte_view.numpy())

    return {"hash": h.hexdigest(), "shape": shape, "dtype": dtype}


def hash_tensors_parallel(
    items: Iterable[tuple[str, torch.Tensor]],
) -> list[tuple[str, dict[str, Any]]]:
    """Hash a batch of ``(key, tensor)`` pairs across worker threads.

    Each tensor is hashed by :func:`hash_tensor` on a worker thread;
    ``blake2b.update`` releases the GIL so the threads make real
    progress. The caller blocks until every hash completes (no
    async / callback layer); the resulting list preserves the input
    order so caller code can iterate ``items`` in lockstep.

    Used by :func:`._grad_collector.collect_param_gradients` for the
    per-step grad sweep where 100+ parameter tensors (some hundreds
    of MB each at fp32 grad-buf shapes) would otherwise serialise on
    a single CPU core and blow the step budget. The thread pool is a
    module-level singleton (see :func:`_pool`) so repeat calls reuse
    workers.
    """
    items_list = list(items)
    if not items_list:
        return []
    if len(items_list) == 1:
        # Skip the pool dispatch overhead for trivial batches.
        key, tensor = items_list[0]
        return [(key, hash_tensor(tensor))]

    pool = _pool()
    futures = [pool.submit(hash_tensor, t) for _, t in items_list]
    return [(items_list[i][0], futures[i].result()) for i in range(len(items_list))]


def _resolve_future_records(captured_records: dict[str, Any]) -> None:
    """Resolve any async-hash Futures in ``captured_records`` in place.

    The module forward/backward hooks submit blake2b to the shared pool
    and store a ``Future`` instead of the ``{hash,shape,dtype}`` record
    (see ``_module_hook._hash_or_submit``). Call this once before
    serialising so every value is a plain record. ``.result()`` re-raises
    any hashing error on the caller thread (fail-fast at dump). Records
    already resolved (sync captures: loss / grads) pass through untouched.
    """
    from concurrent.futures import Future

    for key, value in list(captured_records.items()):
        if isinstance(value, Future):
            captured_records[key] = value.result()


# ── Overlapped async capture (pinned copy on a side stream) ──────────
# The module hooks fire on the hot forward/backward path. To keep the
# device→host snapshot off the critical path, a CUDA tensor is copied
# into a PINNED host buffer on a dedicated copy stream (so the copy
# overlaps the next layers' compute) and the blake2b runs on the hash
# pool once a CUDA event says the copy is done. Pinned buffers are
# pooled + reused (a per-fire ``cudaHostAlloc`` otherwise dominates), and
# the pool caps in-flight buffers to bound pinned host memory + the GPU
# tensors held alive until their copy completes (backpressure: the hook
# blocks when the cap is hit).
_COPY_STREAM = None
_PINNED_POOL = None


def _copy_stream():
    import torch

    global _COPY_STREAM
    if _COPY_STREAM is None:
        _COPY_STREAM = torch.cuda.Stream()
    return _COPY_STREAM


class _PinnedBufferPool:
    """Shape/dtype-keyed pool of pinned host buffers with an in-flight cap."""

    def __init__(self, max_inflight: int) -> None:
        self._free: dict[tuple, list] = collections.defaultdict(list)
        self._cond = threading.Condition()
        self._inflight = 0
        self._max = max_inflight

    def acquire(self, shape, dtype):
        import torch

        key = (tuple(shape), dtype)
        with self._cond:
            while self._inflight >= self._max:
                self._cond.wait()  # backpressure: bounds pinned mem + GPU held
            self._inflight += 1
            buf = self._free[key].pop() if self._free[key] else None
        if buf is None:
            buf = torch.empty(tuple(shape), dtype=dtype, device="cpu", pin_memory=True)
        return buf

    def release(self, buf) -> None:
        key = (tuple(buf.shape), buf.dtype)
        with self._cond:
            self._free[key].append(buf)
            self._inflight -= 1
            self._cond.notify()


def _pinned_pool() -> _PinnedBufferPool:
    global _PINNED_POOL
    if _PINNED_POOL is None:
        cap = int(os.environ.get("FORGE_HASH_PINNED_MAX", "64") or "64")
        _PINNED_POOL = _PinnedBufferPool(max(1, cap))
    return _PINNED_POOL


def submit_tensor_hash(t):
    """Snapshot ``t`` and submit its blake2b to the hash pool; return a Future.

    CUDA tensors take the overlapped path (pinned copy on the copy stream +
    event); CPU tensors are cloned and hashed off-thread. Either way the
    caller may immediately drop / overwrite ``t`` after this returns — the
    snapshot has captured the bytes (the CUDA source is kept alive until
    its copy completes).
    """
    import torch

    if not (isinstance(t, torch.Tensor) and t.is_cuda):
        return _pool().submit(hash_tensor, t.detach().clone())

    t = t.detach()
    if not t.is_contiguous():
        t = t.contiguous()
    pool = _pinned_pool()
    buf = pool.acquire(t.shape, t.dtype)
    stream = _copy_stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        buf.copy_(t, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(stream)

    def _task(_keep=t, buf=buf, ev=ev):
        ev.synchronize()  # copy done → buf holds a faithful snapshot
        try:
            return hash_tensor(buf)
        finally:
            pool.release(buf)

    return _pool().submit(_task)


def resolve_dump_path(env_value: str, default_basename: str, dump_dir_env: str) -> Path | None:
    """Resolve a dump-path env var, optionally relative to ``DUMP_DIR``.

    Used by M1 bridges that read the conventional
    ``HOOK_OUTPUT_FILE`` / ``DUMP_DIR`` env contract. Returns ``None``
    only when neither the env var nor ``dump_dir_env`` is set.
    """
    from pathlib import Path

    if env_value:
        p = Path(env_value)
        if p.is_absolute():
            return p
        if dump_dir_env:
            return Path(dump_dir_env) / p
        return p
    if dump_dir_env:
        return Path(dump_dir_env) / default_basename
    return None


def dump_capture_files(
    output_file: Path,
    captured_records: dict[str, dict[str, Any]],
    captured_graph: list[dict],
    *,
    is_writer: bool,
) -> None:
    """Atomic-write the JSON hash dict + sibling ``.graph.json``.

    Non-writer ranks are a no-op. Writer rank writes to a ``.tmp``
    file then atomically renames, so a partial dump never appears
    under ``output_file``. ``captured_records`` maps FQN-keyed names
    (``rank<r>.mb<m>.fwd.<fqn>#<call>`` / ``rank<r>.grad.<fqn>`` /
    ``step_<n>.rank<r>.mb<m>.<...>``) to the
    ``{hash, shape, dtype}`` records produced by :func:`hash_tensor`.
    """
    # Every rank persists its OWN capture to ``<file>.rank<N>`` so all-rank
    # alignment is possible — TP shards, and per-rank activations/dgrads on
    # a DP rank's own data shard, differ across ranks and were previously
    # discarded (only rank0 wrote). rank0 additionally writes the canonical
    # unsuffixed ``<file>`` for existing single-rank consumers (back-compat).
    _resolve_future_records(captured_records)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    rank = int(os.environ.get("RANK", "0"))
    targets = [output_file.with_name(f"{output_file.name}.rank{rank}")]
    if is_writer:
        targets.append(output_file)

    records_json = json.dumps(captured_records, indent=2)
    graph_json = json.dumps(captured_graph, indent=2)
    for tgt in targets:
        tmp_path = tgt.with_suffix(tgt.suffix + ".tmp")
        tmp_path.write_text(records_json, encoding="utf-8")
        tmp_path.replace(tgt)
        graph_path = tgt.with_name(tgt.name + ".graph.json")
        graph_path.write_text(graph_json, encoding="utf-8")

    sys.stderr.write(
        f"[harness_hook] capture wrote {len(targets)} file(s) "
        f"(rank {rank}, {len(captured_records)} hash records, "
        f"{len(captured_graph)} graph entries); primary={targets[0]}\n"
    )
