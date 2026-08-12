"""Harness-provided capture-offload utility — candidate-importable.

Overlapped hash capture for the bitwise gates: **async** D2H on a dedicated
copy stream into a fixed pinned staging ring, blake2b-128 on a shared thread
pool. Digests are byte-identical to the reference producer
(``evals.harness_hook._dump``): detach → contiguous → flatten → 8M-element
chunks → uint8 view → blake2b (digest_size=16).

Unlike ``evals.harness_hook`` (a reference-side helper the anti-proxy guard
forbids inside the candidate), THIS module is explicitly candidate-facing:
the engine under ``workload/src`` MAY ``from evals.capture_offload import
OffloadHasher, hash_tensor, ...``. Using the harness's measurement utility
is not proxying — the engine being measured is still the candidate's own.

The design encodes constraints learned from real crashes during loop
``c0879843c0c7``'s bitwise-multicard round (07-06/07):

* **Fixed pinned ring, allocated once** — the original SIGKILL came from
  per-tensor pinned buffers going through torch's caching host allocator,
  which never returns pages to the OS and grew past a 302 GiB container
  cgroup cap. A single fixed-capacity ring (``FORGE_HASH_STAGING_MB``,
  default 4096) cannot grow; backpressure recycles segments instead.
* **No CUDA on worker threads** — a worker touching the single CUDA queue
  (``CUDA_DEVICE_MAX_CONNECTIONS=1``, the determinism requirement) while
  the training thread was mid-collective deadlocked DP ranks. Workers here
  only blake2b host bytes; copy-completion events are queried/synced on the
  training thread exclusively.
* **Bounded backlog via a per-microbatch wait** — ``flush(wait=True)`` at
  each microbatch boundary joins the outstanding jobs and drains the ring;
  both ranks wait identically after backward and before the next
  collective, so it cannot deadlock a peer.

Why async D2H is legal under ``CUDA_DEVICE_MAX_CONNECTIONS=1``: that knob
serialises *kernel-launch* work queues; D2H transfers ride the GPU's DMA
copy engines, a separate hardware channel, so a side-stream copy into
pinned memory overlaps compute. The copy stream ``wait_stream``s the
producing stream, so bytes are snapshotted in stream order. The one
semantic requirement inherited from that ordering: a captured tensor must
not be mutated in place afterwards — the capture points the bitwise gates
use (activations, loss) satisfy this, and grad buffers mutated between
pre/post-allreduce sweeps must use the synchronous :func:`hash_batch_sync`
barrier, never the async path (same rule as before).

Set ``FORGE_HASH_STAGING_MB=0`` to fall back to the legacy blocking-copy
path (pageable per-tensor snapshots on the training thread).
"""

from __future__ import annotations

import hashlib
import os
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

import torch

_HASH_DIGEST_BYTES = 16
_HASH_CHUNK_ELEMS = 8 * 1024 * 1024
# Pinned staging-ring capacity (MiB). Fits one ~2.4 GB [B,S,V] logits tensor
# plus a microbatch of small activations with room to recycle; 0 disables the
# ring (legacy blocking D2H). Fixed allocation — never grows.
_STAGING_MB_ENV = "FORGE_HASH_STAGING_MB"
_STAGING_DEFAULT_MB = 4096
_RING_ALIGN = 64  # byte alignment for segment offsets (any dtype view is happy)


def hash_tensor(t: torch.Tensor) -> dict:
    """Stream a tensor's raw bytes into blake2b-128; return the record.

    Byte-identical to the reference ``harness_hook._dump.hash_tensor``: detach,
    contiguous, flatten, split into 8M-element chunks, copy each to CPU, view as
    uint8, feed to blake2b. Shape + dtype accompany the digest.
    """
    t = t.detach().contiguous()
    shape = list(t.shape)
    dtype = str(t.dtype).removeprefix("torch.")
    h = hashlib.blake2b(digest_size=_HASH_DIGEST_BYTES)
    if t.numel() > 0:
        flat = t.reshape(-1)
        for chunk in flat.split(_HASH_CHUNK_ELEMS):
            chunk_cpu = chunk.cpu()
            byte_view = chunk_cpu if chunk_cpu.dtype == torch.uint8 else chunk_cpu.view(torch.uint8)
            h.update(byte_view.numpy())
    return {"hash": h.hexdigest(), "shape": shape, "dtype": dtype}


# ── Capture offload: blocking D2H on the train thread, blake2b on a pool ──────
#
# Inline ``hash_tensor(t)`` on every live CUDA activation does two serial things
# on the training thread: a device→host copy AND a single-core blake2b (~1 GB/s).
# For the multi-card bitwise gate (grad_accum=10 → ~35k records/rank incl. the
# ~2.4 GB ``[B,S,V]`` logits captured twice per microbatch — ~48 GB of D2H+blake2b
# per step) the single-core blake2b dominates and the committed serial code does
# not finish even one step in >230 s (GPU idle ~2 %).
#
# This offloads the blake2b (only) to a thread pool while keeping the D2H on the
# training thread. Design constraints, each learned from a concrete failure on
# this box:
#   * **Pageable host snapshot, freed after hashing.** ``t.to("cpu")`` returns a
#     regular (non-pinned) tensor; it is refcount-freed the moment its digest is
#     done. Pinned buffers went through torch's caching host allocator, which
#     never returns pages to the OS and blew past the 302 GiB CGROUP limit
#     (``memory.limit_in_bytes``, shared by both ranks) → SIGKILL at step 4.
#   * **No CUDA on worker threads.** Workers only run ``blake2b`` on host bytes;
#     they never call ``.cpu()`` / sync an event. A worker touching the single
#     CUDA queue (``CUDA_DEVICE_MAX_CONNECTIONS=1``) while the training thread was
#     mid-collective deadlocked rank-0-spins / rank-1-idle.
#   * **Bounded backlog via a per-microbatch wait.** :meth:`flush` (``wait=True``
#     at the microbatch boundary) blocks until the dispatched digests finish, so
#     at most one microbatch's host snapshots are alive at once. Both ranks wait
#     identically after backward and before the next collective, so it cannot
#     deadlock a peer.
# ``blake2b.update`` releases the GIL, so the ~3.3k small activations hash across
# cores while the training thread issues the next D2H. Every digest is
# byte-identical to a serial ``hash_tensor``. The engine may not import
# ``evals.harness_hook`` (anti-proxy), so this is an independent reimplementation.

_POOL: ThreadPoolExecutor | None = None


def _pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        workers = int(os.environ.get("FORGE_HASH_WORKERS", "0") or "0")
        if workers <= 0:
            workers = min(16, max(4, (os.cpu_count() or 4)))
        _POOL = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="engine-hash")
    return _POOL


def _hash_cpu(cpu_t: torch.Tensor, shape: list, dtype: str) -> dict:
    """blake2b a host-resident (already-copied, pageable) snapshot.

    Runs on a hash worker — host memory only, never CUDA. Streams the bytes in
    the same 8M-element chunks as :func:`hash_tensor`, so the digest is
    byte-identical; ``cpu_t`` is dropped when this returns (freed by refcount).
    """
    h = hashlib.blake2b(digest_size=_HASH_DIGEST_BYTES)
    if cpu_t.numel() > 0:
        flat = cpu_t.reshape(-1)
        for chunk in flat.split(_HASH_CHUNK_ELEMS):
            byte_view = chunk if chunk.dtype == torch.uint8 else chunk.view(torch.uint8)
            h.update(byte_view.numpy())
    return {"hash": h.hexdigest(), "shape": shape, "dtype": dtype}


class _StagingRing:
    """Fixed-capacity pinned host buffer carved into transient segments.

    One contiguous ``torch.empty(cap, dtype=uint8, pin_memory=True)`` allocated
    at first use — a single allocation that can never grow, so torch's caching
    host allocator cannot hoard pages past the cgroup cap (the c087 SIGKILL).
    ``alloc(nbytes)`` hands out a uint8 view at the bump pointer; segments are
    retired in FIFO order once their blake2b finishes (copy-then-hash pipeline
    order is FIFO by construction). If the ring is full, the caller reclaims
    completed segments — waiting on hash *futures*, never on CUDA — until the
    request fits; requests larger than the whole ring fall back to the legacy
    blocking path. Training-thread only; no locking needed.
    """

    def __init__(self, capacity: int) -> None:
        self.cap = capacity
        self.buf = torch.empty(capacity, dtype=torch.uint8, pin_memory=True)
        self.head = 0  # next free offset
        self.tail = 0  # oldest live offset
        self.live = 0  # bytes currently reserved
        self.segments: deque = deque()  # (offset, size, Future|None) in FIFO order

    def _fits(self, n: int) -> bool:
        if self.live + n > self.cap:
            return False
        if self.head >= self.tail and (self.live or self.head != self.tail):
            # live region [tail, head): free is [head, cap) then [0, tail)
            return self.head + n <= self.cap or n <= self.tail
        if self.live == 0:
            return n <= self.cap
        # live region wraps ([tail, cap)+[0, head)): free is [head, tail) only
        return self.head + n <= self.tail

    def alloc(self, nbytes: int, *, blocking: bool = True):
        """uint8 view of ``nbytes`` (aligned), or None if it cannot fit.

        ``blocking=False`` never waits: returns None when reclaiming would
        require joining an unfinished hash. Callers must only use
        ``blocking=True`` after every copy feeding the ring has been
        DISPATCHED to the pool (else the join could wait on a Future nothing
        will ever set — the training thread owns dispatch).
        """
        n = ((nbytes + _RING_ALIGN - 1) // _RING_ALIGN) * _RING_ALIGN
        if n > self.cap:
            return None
        while not self._fits(n):
            if not self.segments:
                return None  # all reclaimed yet still no room (fragmentation edge)
            off, size, fut = self.segments[0]
            if fut is not None:
                if not blocking and not fut.done():
                    return None
                fut.result()  # host-side blake2b join — never a CUDA wait
            self.segments.popleft()
            self.live -= size
            self.tail = (off + size) % self.cap
            if not self.segments:
                self.head = self.tail = 0  # empty ring: reset to avoid fragmentation
        if self.head + n > self.cap:  # wrap: skip the tail gap
            self.live += self.cap - self.head
            self.segments.append((self.head, self.cap - self.head, None))
            self.head = 0
        off = self.head
        self.head = (self.head + n) % self.cap
        self.live += n
        seg = self.buf.narrow(0, off, nbytes)
        self._pending_meta = (off, n)
        return seg

    def commit(self, fut: Future) -> None:
        """Attach the hash future to the most recent alloc (FIFO retire key)."""
        off, n = self._pending_meta
        self.segments.append((off, n, fut))

    def drain(self) -> None:
        """Join every outstanding segment's hash and empty the ring."""
        while self.segments:
            _off, _size, fut = self.segments.popleft()
            if fut is not None:
                fut.result()
        self.live = 0
        self.head = self.tail = 0


_RING: _StagingRing | None = None
_COPY_STREAM: torch.cuda.Stream | None = None


def _staging_ring() -> _StagingRing | None:
    """The process-wide pinned ring, or None when disabled/CPU-only."""
    global _RING
    if _RING is None:
        mb = int(os.environ.get(_STAGING_MB_ENV, str(_STAGING_DEFAULT_MB)) or "0")
        if mb <= 0 or not torch.cuda.is_available():
            return None
        _RING = _StagingRing(mb * 1024 * 1024)
    return _RING


def _copy_stream() -> torch.cuda.Stream:
    global _COPY_STREAM
    if _COPY_STREAM is None:
        _COPY_STREAM = torch.cuda.Stream()
    return _COPY_STREAM


def _hash_pinned_segment(seg: torch.Tensor, shape: list, dtype: str) -> dict:
    """blake2b a pinned uint8 segment (bytes of one tensor) on a hash worker.

    The copy event was synchronized on the TRAINING thread before submit, so
    the bytes are stable; the worker touches host memory only. Chunking is on
    the raw byte stream — identical digest to element-chunked hashing because
    blake2b is a running stream (8M elems of any dtype is a whole number of
    bytes, so chunk boundaries never split the stream semantics).
    """
    h = hashlib.blake2b(digest_size=_HASH_DIGEST_BYTES)
    n = seg.numel()
    step = _HASH_CHUNK_ELEMS  # bytes per update; stream-equivalent to any chunking
    for i in range(0, n, step):
        h.update(seg.narrow(0, i, min(step, n - i)).numpy())
    return {"hash": h.hexdigest(), "shape": shape, "dtype": dtype}


class _OffloadHasher:
    """Per-recorder queue of dispatched blake2b jobs, joined by :meth:`flush`.

    ``enqueue(tensor)`` snapshots a CUDA tensor to host and submits its blake2b
    to the pool, returning the ``Future``. Fast path: an **async** D2H on the
    dedicated copy stream into the pinned staging ring — the copy engine
    overlaps the training stream's compute, and the training thread only pays
    an event-sync just before handing the bytes to a worker. Tensors larger
    than the ring (or when the ring is disabled) use the legacy blocking
    pageable copy. ``flush(wait=True)`` at each microbatch boundary joins the
    outstanding jobs and drains ring segments. CPU / non-tensor inputs are
    hashed off-thread directly.
    """

    def __init__(self) -> None:
        self._jobs: list[Future] = []
        # (event, seg, shape, dtype) copies issued but not yet handed to the pool
        self._in_flight: deque = deque()

    def enqueue(self, tensor) -> Future:
        if not (isinstance(tensor, torch.Tensor) and tensor.is_cuda):
            snap = (
                tensor.detach().contiguous().clone() if isinstance(tensor, torch.Tensor) else tensor
            )
            fut = _pool().submit(hash_tensor, snap)
            self._jobs.append(fut)
            return fut
        t = tensor.detach()
        if not t.is_contiguous():
            t = t.contiguous()
        shape = list(t.shape)
        dtype = str(t.dtype).removeprefix("torch.")

        ring = _staging_ring()
        nbytes = t.numel() * t.element_size()
        if ring is not None and t.numel() > 0:
            # Dispatch older finished copies first so ring space recycles and
            # hash workers start while this copy is still in the DMA queue.
            self._dispatch_ready()
            seg = ring.alloc(nbytes, blocking=False)
            if seg is None and nbytes <= ring.cap:
                # Ring full of not-yet-hashed segments. Sync + dispatch every
                # in-flight copy FIRST (training thread owns dispatch), so a
                # blocking reclaim can only ever wait on Futures the pool will
                # complete — never on one nothing has been told to run.
                self._dispatch_ready(wait=True)
                seg = ring.alloc(nbytes, blocking=True)
            if seg is not None:
                stream = _copy_stream()
                stream.wait_stream(torch.cuda.current_stream())  # bytes in stream order
                with torch.cuda.stream(stream):
                    seg.copy_(t.reshape(-1).view(torch.uint8), non_blocking=True)
                # Tell the caching allocator the source is in use on the copy
                # stream: without this, the training thread freeing `t` right
                # after enqueue could reuse its memory before the DMA runs.
                t.record_stream(stream)
                ev = torch.cuda.Event()
                ev.record(stream)
                fut: Future = Future()
                ring.commit(fut)
                self._in_flight.append((ev, seg, shape, dtype, fut))
                self._jobs.append(fut)
                return fut

        # Legacy blocking path: pageable snapshot on the training thread.
        cpu_t = t.to("cpu")
        fut = _pool().submit(_hash_cpu, cpu_t, shape, dtype)
        self._jobs.append(fut)
        return fut

    def _dispatch_ready(self, *, wait: bool = False) -> None:
        """Hand completed copies to the hash pool (training thread only).

        Event query/sync happens HERE — never on a worker — honoring the
        no-CUDA-on-workers rule. With ``wait=True`` every in-flight copy is
        synced and dispatched (microbatch boundary).
        """
        while self._in_flight:
            ev, seg, shape, dtype, fut = self._in_flight[0]
            if not wait and not ev.query():
                break
            ev.synchronize()
            self._in_flight.popleft()

            def _run(seg=seg, shape=shape, dtype=dtype, fut=fut):
                try:
                    fut.set_result(_hash_pinned_segment(seg, shape, dtype))
                except BaseException as exc:
                    fut.set_exception(exc)

            _pool().submit(_run)

    def flush(self, *, wait: bool = False) -> None:
        self._dispatch_ready(wait=wait)
        if not self._jobs:
            return
        jobs, self._jobs = self._jobs, []
        if wait:
            for f in jobs:
                f.result()  # re-raises worker errors on the training thread


def _resolve_futures(records: dict) -> None:
    """Replace any pending hash Futures in ``records`` with their result.

    ``.result()`` blocks until the async blake2b completes and re-raises any
    worker error on the caller thread (fail-fast at dump time). Records already
    resolved to plain dicts pass through untouched.
    """
    for key, value in list(records.items()):
        if isinstance(value, Future):
            records[key] = value.result()


def _hash_batch_sync(items) -> list:
    """Hash ``(key, tensor)`` pairs across the pool and BLOCK for all results.

    Returns finished ``{hash,shape,dtype}`` records — every source tensor has
    been read before this returns. Used for the grad captures, whose buffers are
    mutated in place by the DP all-reduce between the ``pre`` and ``post``
    sweeps: an async snapshot could race that mutation, so the pre-sweep must
    fully materialise the pre-reduce bytes here. A full barrier on the training
    thread — both ranks run it identically after backward and BEFORE
    ``reduce_grads``'s collective. blake2b releases the GIL, so the per-param
    hashes still parallelise across cores.
    """
    items = list(items)
    if not items:
        return []
    if len(items) == 1:
        key, t = items[0]
        return [(key, hash_tensor(t))]
    pool = _pool()
    futures = [pool.submit(hash_tensor, t) for _, t in items]
    return [(items[i][0], futures[i].result()) for i in range(len(items))]


def _shutdown_hash_pool() -> None:
    """Join the hash workers so no thread is live during CUDA teardown.

    Called after the dump has resolved every Future (all work already done), so
    this is an instant join. Also releases the pinned staging ring and copy
    stream. Clearing the globals lets a subsequent capture in the same process
    (only tests do this) re-init cleanly.
    """
    global _POOL, _RING, _COPY_STREAM
    if _POOL is not None:
        _POOL.shutdown(wait=True)
        _POOL = None
    if _RING is not None:
        _RING.drain()
        _RING = None
    _COPY_STREAM = None


# ── Public aliases (the candidate-facing API surface) ────────────────────────
OffloadHasher = _OffloadHasher
resolve_futures = _resolve_futures
hash_batch_sync = _hash_batch_sync
shutdown_hash_pool = _shutdown_hash_pool
