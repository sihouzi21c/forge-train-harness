"""Self-developed training engine — public CLI entry contract.

This module is the **single source of truth** for the ours-side training
entry point.  Every harness gate — M1 single-step tensor-capture gates
(``forward-align`` / ``backward-align``), M2 / M3 / M4 bitwise-trajectory
runs, M5 resume gate, M6 long-train, and the stage-2 op-long runner —
invokes :func:`run_training_loop` as a black-box subroutine and either
parses its ``[<loss_tag>]`` stdout lines (M2+) or loads the hash-record
dump it writes when ``hash_capture_level > 0`` (M1).  No gate script may
build its own dataloader, model, optimizer, or training loop — that
orchestration belongs entirely to this module.

The in-house engine uses bare ``torch.Tensor`` parameters (no ``nn.Module``),
no ``torch.optim``, no ``torch.autograd``.  The backward pass is statically
scheduled — the reverse computation graph is unrolled manually.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from evals.capture_offload import hash_batch_sync

from training_engine_tensor import config as C
from training_engine_tensor.backward import (
    apply_rope_backward,
    cross_entropy_backward,
    embedding_backward,
    gqa_attention_backward,
    linear_backward,
    project_qkv_backward,
    rms_norm_backward,
    silu_swiglu_intermediate_backward,
)
from training_engine_tensor.forward import (
    _gqa_attention,
    apply_rope,
    embedding_forward,
    lm_head_forward,
    masked_cross_entropy,
    mlp_swiglu,
    precompute_rope_freqs,
    project_qkv,
    rms_norm,
)

from training_engine_tensor.parameters import (
    ModelParameters,
    load_weights_from_checkpoint,
)

from training_engine_tensor.zero_optimizer import (
    ZeroOptimizer,
    all_gather_bf16,
    compute_zero_grad_norm,
    init_zero_optimizer,
    reduce_scatter_grads,
    set_grad_from_shard,
    zero_load_checkpoint,
    zero_optimizer_step,
    zero_save_checkpoint,
)

# Try to import the fused Triton SwiGLU forward kernel.
# Falls back to the PyTorch path when Triton is not available (e.g. on Mac).
# The Triton kernel is used only when BOTH:
#   (a) deterministic=False (long-horizon performance mode), AND
#   (b) ENABLE_TRITON_SWIGLU_FWD=1 (explicitly enabled — default 0).
_HAS_TRITON_SWIGLU_FWD = False
try:
    from training_engine_tensor.triton_kernels import swiglu_forward_fused as _swiglu_fwd_fused
    _HAS_TRITON_SWIGLU_FWD = True
except (ImportError, ModuleNotFoundError, AttributeError):
    pass

# ── H100 peak ────────────────────────────────────────────────────────────────
H100_BF16_PEAK_FLOPS = 989.4e12


# ── Hash helpers ─────────────────────────────────────────────────────────────


def _hash_tensor(t: torch.Tensor) -> dict:
    """Compute blake2b-128 hash record (same as harness_hook._dump.hash_tensor)."""
    t = t.detach().contiguous()
    shape = list(t.shape)
    dtype = str(t.dtype).removeprefix("torch.")
    h = hashlib.blake2b(digest_size=16)
    if t.numel() > 0:
        flat = t.reshape(-1)
        for chunk in flat.split(8_000_000):
            chunk_cpu = chunk.cpu()
            byte_view = chunk_cpu if chunk_cpu.dtype == torch.uint8 else chunk_cpu.view(torch.uint8)
            h.update(byte_view.numpy())
    return {"hash": h.hexdigest(), "shape": shape, "dtype": dtype}


def _capture_forward(records: dict, prefix: str, fqn: str, call_idx: int,
                     tensor: torch.Tensor) -> None:
    """Record a forward activation hash (inline synchronous path)."""
    key = f"{prefix}fwd.{fqn}#{call_idx}"
    records[key] = _hash_tensor(tensor)


# ── TrainLoopConfig ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrainLoopConfig:
    """Immutable configuration for a single ``run_training_loop`` invocation.

    All fields are required unless documented otherwise.  The values
    flow from ``config/eval.toml`` through the dispatcher into env
    variables, and from there into the gate scripts that build this
    config; see ``[evals.<name>.ref_env]`` for the per-gate authoritative
    shape (NUM_STEPS_OVERRIDE / MICRO_BATCH_SIZE_OVERRIDE / …).

    Full docstring in the module-level docstring above.
    """

    num_steps: int
    micro_batch_size: int
    seq_length: int
    grad_accum_steps: int
    seed: int
    world_size: int

    checkpoint_root: str
    data_path: str

    start_step: int = 0
    save_path: str | None = None
    resume_from: str | None = None
    capture_output_file: str | None = None

    hash_capture_level: int = 0
    hash_output: str | None = None
    persistent: bool = False

    backend: str = "megatron"
    megatron_root: str = ""

    lr: float = 1e-2
    min_lr: float = 0.0
    lr_warmup_iters: int = 0
    lr_decay_iters: int = 0
    lr_wsd_decay_iters: int = 0
    init_weights_only: bool = False
    save_interval: int = 0
    phase_name: str | None = None
    teardown_exit: bool = True

    @property
    def global_batch_size(self) -> int:
        return self.grad_accum_steps * self.micro_batch_size * self.world_size


# ── LR schedule ──────────────────────────────────────────────────────────────


def _compute_lr(step: int, max_lr: float, min_lr: float,
                warmup: int, decay: int, wsd_decay: int) -> float:
    """WSD learning rate schedule (matches ref's compute_lr)."""
    if step < warmup:
        return max_lr * step / warmup
    if step > decay:
        return min_lr
    wsd_anneal_start = decay - wsd_decay
    if step <= wsd_anneal_start:
        return max_lr
    wsd_steps = step - wsd_anneal_start
    wsd_ratio = float(wsd_steps) / float(wsd_decay)
    coeff = 2.0 * pow(0.5, wsd_ratio) - 1.0
    return min_lr + coeff * (max_lr - min_lr)


# ── Dataloader ────────────────────────────────────────────────────────────────


def _build_dataloader(data_path: str, dp_rank: int, world_size: int,
                      micro_batch_size: int, seq_length: int, seed: int):
    """Build a dataloader from the data path.

    Uses the same modelbest_sdk / HF / Megatron-binary dispatch as the
    ref (``train_pure_mup_mtp.py:build_dataloader``).
    """
    if data_path.endswith(".txt") or data_path.endswith(".sh"):
        with open(data_path) as f:
            raw = f.read().strip().split()
    else:
        raw = data_path.strip().split()

    loader_kind = "hf"
    data_toml = os.environ.get("FORGE_DATA_TOML", "")
    if data_toml:
        import tomllib
        try:
            with open(data_toml, "rb") as fh:
                doc = tomllib.load(fh)
            loader_kind = str(doc.get("data", {}).get("data_loader", "hf")).strip().lower()
        except (FileNotFoundError, tomllib.TOMLDecodeError):
            pass

    if loader_kind == "megatron_binary":
        path_prefix = raw[1] if len(raw) > 1 and raw[0].replace(".", "").isdigit() else raw[0]
        return _build_megatron_binary_dataloader(
            path_prefix, dp_rank, world_size, micro_batch_size, seq_length, seed
        )

    if loader_kind == "modelbest":
        return _build_modelbest_dataloader(
            raw, dp_rank, world_size, micro_batch_size, seq_length, seed
        )

    return _build_hf_dataloader(
        raw, dp_rank, world_size, micro_batch_size, seq_length, seed
    )


def _build_hf_dataloader(weights_and_paths, dp_rank, world_size,
                         micro_batch_size, seq_length, seed):
    """Build a HuggingFace streaming dataloader."""
    from ref.reference.hf_stream_dataloader import build as build_hf
    return build_hf(
        [(float(w), p) for w, p in _pairwise(weights_and_paths)],
        dp_rank, world_size, micro_batch_size, seq_length, seed,
    )


def _build_modelbest_dataloader(weights_and_paths, dp_rank, world_size,
                                micro_batch_size, seq_length, seed):
    """Build a modelbest_sdk SSTable dataloader."""
    from modelbest_sdk.dataset.batch_packer.batch_packer_factory import MEGATRON_BATCH_PACKER
    from modelbest_sdk.dataset.modelbest_dataloader import ModelbestDataloader
    from modelbest_sdk.dataset.sampler.sampler_factory import WEIGHTED_MEGATRON_SAMPLER
    from modelbest_sdk.dataset.segment.segment_factory import CONDITIONAL_FIXED_LENGTH_SEGMENT
    from modelbest_sdk.dataset.thrift_wrapper.dataset_checkpoint import (
        DatasetInfo, DatasetInfoList,
    )
    from modelbest_sdk.dataset.thrift_wrapper.dataset_context import DatasetContext

    pairs = list(_pairwise(weights_and_paths))
    total_w = sum(w for w, _ in pairs)
    ds_info = DatasetInfoList([
        DatasetInfo(path=p, weight=w / total_w) for w, p in pairs
    ])
    ctx = DatasetContext(
        rank=dp_rank, world_size=world_size,
        tp_rank=0, tp_size=1, pp_rank=0, pp_size=1,
        num_workers=2, dataset_config_path="",
        dataset_checkpoint_path="",
        seed=seed + dp_rank + world_size,
    )
    return ModelbestDataloader(
        ctx, ds_info,
        batch_size=micro_batch_size, max_len=seq_length, chunk_size=64,
        cuda_prefetch=False,
        segment_type=CONDITIONAL_FIXED_LENGTH_SEGMENT,
        sampler_type=WEIGHTED_MEGATRON_SAMPLER,
        batch_packer_type=MEGATRON_BATCH_PACKER,
        dp_group=dist.group.WORLD if world_size > 1 else None,
        clear_sampler_state=False,
    )


def _build_megatron_binary_dataloader(path_prefix, dp_rank, world_size,
                                      micro_batch_size, seq_length, seed):
    """Build a Megatron binary (.bin/.idx) dataloader."""
    from ref.reference.train_pure_mup_mtp import MegatronBinaryDataloader
    return MegatronBinaryDataloader(
        path_prefix, dp_rank, world_size, micro_batch_size, seq_length, seed
    )


def _pairwise(items):
    """Yield (weight, path) pairs from a flat list."""
    for i in range(0, len(items), 2):
        if i + 1 < len(items):
            yield float(items[i]), items[i + 1]


class _BackgroundPrefetcher:
    """Background-thread dataloader prefetcher with pre-allocated pinned buffers.

    Calls ``next(dl)`` on a daemon background thread and copies the results
    into pre-allocated pinned CPU buffers (eliminating the ``pin_memory()``
    allocation overhead, ~50ms/batch).  The buffer sets are stored in a bounded
    ``deque``.  The main thread calls ``get()`` to retrieve the next batch
    without blocking on the dataloader's shard refill (which can take seconds
    on the first call).  The queue depth is controlled by ``max_size`` (default
    8 — enough headroom to absorb dataloader latency across 10-microbatch
    steps).

    Producer-consumer synchronization uses a ``threading.Semaphore`` to avoid
    the lost-notify problem with ``Condition.wait(0.1)``: the background thread
    acquires a semaphore permit before calling ``next(dl)``, and the main thread
    releases a permit via ``get()`` after consuming a batch.  This ensures the
    producer is woken up immediately when the queue has room, eliminating the
    ~968ms/step of ``pthread_cond_timedwait`` that occurred with the 0.1s poll
    interval.

    The background thread is started by ``start()`` and must be stopped by
    ``stop()`` during teardown to avoid a dangling thread.

    This is orthogonal to the async H2D double buffering: this prefetcher
    hides the CPU-side dataloader latency, while the H2D double buffering
    (``non_blocking=True`` copies) hides the GPU-side H2D transfer latency.
    """

    def __init__(self, dl, max_size: int = 8, B: int = 4, S: int = 4096):
        self._dl = dl
        self._queue: deque = deque()
        self._max_size = max_size
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        # Semaphore for producer-consumer flow control: the producer
        # acquires a permit before calling next(dl); the consumer releases
        # a permit via get() after consuming a batch.  Initialized with
        # max_size permits so the producer can queue up to max_size batches
        # before blocking.
        self._free_slots = threading.Semaphore(max_size)
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="dl-prefetcher")
        # Pre-allocate pinned CPU buffers for the entire queue depth.
        # Each buffer set holds (input_ids, labels, loss_mask) as pinned
        # tensors, eliminating the pin_memory() call in _next_batch which
        # creates a new pinned allocation every batch (~50ms overhead per
        # batch, ~500ms/step across 10 microbatches).
        self._buf_pool: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for _ in range(max_size):
            self._buf_pool.append((
                torch.empty(B, S, dtype=torch.long, pin_memory=True),
                torch.empty(B, S, dtype=torch.long, pin_memory=True),
                torch.empty(B, S, dtype=torch.float32, pin_memory=True),
            ))

    def start(self):
        self._thread.start()

    def stop(self):
        self._running = False
        # Release all permits so the background thread can exit.
        self._free_slots.release(self._max_size)
        with self._lock:
            self._not_empty.notify_all()

    def get(self):
        """Get the next batch.  Blocks until one is available."""
        with self._not_empty:
            while len(self._queue) == 0:
                self._not_empty.wait()
            result = self._queue.popleft()
        # Release a permit so the producer can start the next next(dl) call.
        self._free_slots.release()
        return result

    def _run(self):
        """Background thread: fetch batches from the dataloader, copy to
        pre-allocated pinned buffers, and enqueue them."""
        buf_idx = 0
        while True:
            # Acquire a permit — blocks if the queue is full (max_size
            # unconsumed batches).  This avoids the lost-notify problem
            # with Condition.wait(0.1): the semaphore internally tracks
            # the release count, so the notify from get() is never lost.
            self._free_slots.acquire()
            if not self._running:
                return
            data = next(self._dl)
            while (data["loss_mask"] == 0).all().item():
                data = next(self._dl)
            # Copy to pre-allocated pinned buffers (fast CPU-side copy,
            # avoids the pin_memory() allocation overhead).
            buf = self._buf_pool[buf_idx % len(self._buf_pool)]
            buf[0].copy_(data["tokens"])
            buf[1].copy_(data["labels"])
            buf[2].copy_(data["loss_mask"].float())
            buf_idx += 1
            with self._lock:
                self._queue.append(buf)
                self._not_empty.notify()


def _next_batch(dl, device):
    """Get the next batch from the dataloader, skipping zero-loss-mask micro-batches.

    Mirrors the ref's ``next_batch`` function which skips micro-batches where
    all loss mask values are zero.  The ref's ``PrefetchedBatcher`` and inline
    ``next_batch`` both do this; without it the in-house engine would consume
    a batch that the ref skipped, causing data misalignment.

    When ``device="cpu"``, returns CPU tensors (for the async H2D path where
    the caller does ``copy_(tensor, non_blocking=True)`` into pre-allocated
    pinned GPU buffers).  When ``device`` is a CUDA device, returns GPU tensors
    via a synchronous ``.to(device)`` (fallback for non-CUDA-graph paths).
    """
    data = next(dl)
    while (data["loss_mask"] == 0).all().item():
        data = next(dl)
    tokens = data["tokens"]
    labels = data["labels"]
    loss_mask = data["loss_mask"].float()
    if device == "cpu":
        # Pin memory for async H2D (non_blocking=True copies are only
        # asynchronous when the source is page-locked memory).
        if tokens.is_cpu and not tokens.is_pinned():
            tokens = tokens.pin_memory()
        if labels.is_cpu and not labels.is_pinned():
            labels = labels.pin_memory()
        if loss_mask.is_cpu and not loss_mask.is_pinned():
            loss_mask = loss_mask.pin_memory()
        return tokens, labels, loss_mask
    return (
        tokens.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
        loss_mask.to(device, non_blocking=True),
    )


# ── MTP helpers ──────────────────────────────────────────────────────────────


def _build_mtp_tensors(input_ids, labels, loss_mask):
    """Construct (mtp_input_ids, mtp_labels, mtp_loss_mask) by shifting."""
    mtp_input_ids = labels.clone()
    mtp_labels = torch.zeros_like(labels)
    mtp_labels[:, :-1] = labels[:, 1:]
    mtp_loss_mask = torch.zeros_like(loss_mask)
    mtp_loss_mask[:, :-1] = loss_mask[:, :-1] * loss_mask[:, 1:]
    return mtp_input_ids, mtp_labels, mtp_loss_mask


# ── Forward cache ────────────────────────────────────────────────────────────


@dataclass
class LayerCache:
    """Intermediate tensors saved during one layer's forward pass."""
    hidden_before_attn: torch.Tensor  # hidden before attention sub-layer
    # NOTE: normed is NOT stored — it is recomputed from hidden_before_attn
    # in the backward pass to save ~432 MB of activation memory per layer.
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_rot: torch.Tensor
    k_rot: torch.Tensor
    attn_out_raw: torch.Tensor       # attention output, shape [B, S, H, D]
    attn_flat: torch.Tensor          # attention output reshaped [B, S, H*D]
    attn_out: torch.Tensor           # after wo projection
    softmax_lse: torch.Tensor | None  # flash attention softmax statistics (used by direct _flash_attn_backward)
    hidden_after_attn: torch.Tensor   # after residual add (attention branch)
    # NOTE: normed2 is NOT stored — it is recomputed from hidden_after_attn
    # in the backward pass to save ~432 MB of activation memory per layer.
    gate_up: torch.Tensor             # after wfc1
    mlp_out: torch.Tensor             # after w2
    # NOTE: y1, y2, intermediate are NOT stored — they are recomputed from
    # gate_up in the backward pass to save ~432 MB per layer of fp32/bf16
    # activation memory (25 layers × 432 MB = 10.8 GB).  This is essential
    # for MBS=4 to fit in 80 GB H100 memory.


@dataclass
class ForwardCache:
    """All intermediate tensors saved during one forward pass."""
    input_ids: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor
    hidden_emb: torch.Tensor      # embedding output * mup_emb_scale
    layer_caches: list[LayerCache]
    hidden_normed: torch.Tensor   # after final norm
    main_pre_head: torch.Tensor   # after width_mult division
    main_logits: torch.Tensor

    # MTP branch
    mtp_input_ids: torch.Tensor | None = None
    mtp_labels: torch.Tensor | None = None
    mtp_loss_mask: torch.Tensor | None = None
    mtp_emb: torch.Tensor | None = None
    mtp_a: torch.Tensor | None = None       # after emb_input_layernorm
    mtp_b: torch.Tensor | None = None       # after hidden_input_layernorm
    mtp_eagle_h: torch.Tensor | None = None  # input to final_layernorm (after transformer layer)
    mtp_layer_cache: LayerCache | None = None
    mtp_final: torch.Tensor | None = None    # after final_layernorm
    mtp_pre_head: torch.Tensor | None = None
    mtp_logits: torch.Tensor | None = None


# ── Forward pass (with intermediate caching for backward + capture) ─────────


def _forward_with_cache(
    model: ModelParameters,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    mtp_input_ids: torch.Tensor | None,
    mtp_labels: torch.Tensor | None,
    mtp_loss_mask: torch.Tensor | None,
    rope_freqs: torch.Tensor,
    width_mult: float,
    mup_emb_scale: float,
    depth_scale_main: float,
    depth_scale_mtp: float,
    capture_records: dict | None,
    capture_prefix: str,
    deterministic: bool = True,  # forward attention determinism flag
    ) -> ForwardCache:
    """Run the full forward pass, saving all intermediates for the static backward.

    Also captures intermediate activations when ``capture_records`` is set.
    """
    B, S = input_ids.shape

    # ── Embedding ──────────────────────────────────────────────────────
    hidden_emb = embedding_forward(input_ids, model.tok_embeddings_weight) * mup_emb_scale
    if capture_records is not None:
        _capture_forward(capture_records, capture_prefix, "tok_embeddings", 0, hidden_emb)
    hidden = hidden_emb

    # ── Transformer layers ─────────────────────────────────────────────
    layer_caches: list[LayerCache] = []
    for li, layer in enumerate(model.layers):
        hidden_before_attn = hidden

        normed = rms_norm(hidden, layer.input_norm_weight, deterministic=deterministic)
        if capture_records is not None:
            _capture_forward(capture_records, capture_prefix, f"layers.{li}.attention_norm", 0, normed)

        q, k, v = project_qkv(normed, layer.qkv_weight)
        if capture_records is not None:
            qkv_proj = torch.matmul(normed, layer.qkv_weight.t())
            _capture_forward(capture_records, capture_prefix, f"layers.{li}.wqkv", 0, qkv_proj)

        q_rot = apply_rope(q, rope_freqs, deterministic=deterministic)
        k_rot = apply_rope(k, rope_freqs, deterministic=deterministic)
        # Use flash_attn_func to match the ref's attention implementation
        # exactly.  The ref's TransformerLayer.forward calls:
        #   from flash_attn import flash_attn_func
        #   attn = flash_attn_func(q, k, v, causal=True, deterministic=True)
        attn, softmax_lse = _gqa_attention(q_rot, k_rot, v, allow_math_fallback=True, deterministic=deterministic)
        attn_flat = attn.reshape(B, S, C.NUM_HEADS * C.HEAD_DIM)
        attn_out = torch.matmul(attn_flat, layer.attention_proj_weight.t())
        if capture_records is not None:
            _capture_forward(capture_records, capture_prefix, f"layers.{li}.wo", 0, attn_out)
        hidden_after_attn = hidden + attn_out * depth_scale_main
        hidden = hidden_after_attn

        # MLP sub-layer
        normed2 = rms_norm(hidden, layer.pre_mlp_norm_weight, deterministic=deterministic)
        if capture_records is not None:
            _capture_forward(capture_records, capture_prefix, f"layers.{li}.ffn_norm", 0, normed2)

        gate_up = torch.matmul(normed2, layer.mlp_fc1_weight.t())
        if capture_records is not None:
            _capture_forward(capture_records, capture_prefix, f"layers.{li}.wfc1", 0, gate_up)

        # Use fused Triton SwiGLU forward kernel for non-deterministic (long-horizon) mode.
        # gated by ENABLE_TRITON_SWIGLU_FWD=1 (default 0, checked at runtime).
        _use_triton_swiglu_fwd = (
            _HAS_TRITON_SWIGLU_FWD
            and not deterministic
            and gate_up.is_cuda
            and int(os.environ.get('ENABLE_TRITON_SWIGLU_FWD', '0'))
        )
        if _use_triton_swiglu_fwd:
            intermediate = _swiglu_fwd_fused(gate_up, C.FFN_HIDDEN_SIZE)
        else:
            y1, y2 = gate_up.chunk(2, dim=-1)
            intermediate = (torch.nn.functional.silu(y1.float()) * y2.float()).to(y1.dtype)
        mlp_out = torch.matmul(intermediate, layer.mlp_fc2_weight.t())
        if capture_records is not None:
            _capture_forward(capture_records, capture_prefix, f"layers.{li}.w2", 0, mlp_out)
        hidden = hidden + mlp_out * depth_scale_main

        layer_caches.append(LayerCache(
            hidden_before_attn=hidden_before_attn,
            # normed is NOT stored — recomputed from hidden_before_attn in backward
            q=q, k=k, v=v,
            q_rot=q_rot, k_rot=k_rot,
            attn_out_raw=attn, attn_flat=attn_flat, attn_out=attn_out,
            softmax_lse=softmax_lse,
            hidden_after_attn=hidden_after_attn,
            # normed2 is NOT stored — recomputed from hidden_after_attn in backward
            gate_up=gate_up,
            mlp_out=mlp_out,
        ))

    # ── Final norm ─────────────────────────────────────────────────────
    hidden_normed = rms_norm(hidden, model.final_norm_weight, deterministic=deterministic)
    if capture_records is not None:
        _capture_forward(capture_records, capture_prefix, "norm", 0, hidden_normed)

    # ── LM head (main) ─────────────────────────────────────────────────
    main_pre_head = hidden_normed / width_mult
    main_logits = lm_head_forward(main_pre_head, model.output_weight)
    if capture_records is not None:
        _capture_forward(capture_records, capture_prefix, "output", 0, main_logits)

    # ── MTP branch ─────────────────────────────────────────────────────
    cache = ForwardCache(
        input_ids=input_ids, labels=labels, loss_mask=loss_mask,
        hidden_emb=hidden_emb, layer_caches=layer_caches,
        hidden_normed=hidden_normed, main_pre_head=main_pre_head,
        main_logits=main_logits,
        mtp_input_ids=mtp_input_ids, mtp_labels=mtp_labels,
        mtp_loss_mask=mtp_loss_mask,
    )

    if model.mtp is not None and mtp_input_ids is not None:
        mtp_emb = embedding_forward(mtp_input_ids, model.tok_embeddings_weight) * mup_emb_scale
        a = rms_norm(mtp_emb, model.mtp.emb_input_norm_weight, deterministic=deterministic)
        b = rms_norm(hidden_normed, model.mtp.hidden_input_norm_weight, deterministic=deterministic)

        if capture_records is not None:
            _capture_forward(capture_records, capture_prefix, "mtp.emb_input_layernorm", 0, a)
            _capture_forward(capture_records, capture_prefix, "mtp.hidden_input_layernorm", 0, b)

        eagle_h = torch.matmul(torch.cat([a, b], dim=-1), model.mtp.eagle_fc_weight.t())
        if capture_records is not None:
            _capture_forward(capture_records, capture_prefix, "mtp.eagle_fc", 0, eagle_h)

        # MTP transformer layer
        mtp_hidden_before_attn = eagle_h
        mtp_normed = rms_norm(eagle_h, model.mtp.layer.input_norm_weight, deterministic=deterministic)
        if capture_records is not None:
            _capture_forward(capture_records, capture_prefix, "mtp.layer.attention_norm", 0, mtp_normed)

        mtp_qkv_proj = torch.matmul(mtp_normed, model.mtp.layer.qkv_weight.t())
        mtp_q, mtp_k, mtp_v = project_qkv(mtp_normed, model.mtp.layer.qkv_weight)
        mtp_q_rot = apply_rope(mtp_q, rope_freqs, deterministic=deterministic)
        mtp_k_rot = apply_rope(mtp_k, rope_freqs, deterministic=deterministic)
        mtp_attn, mtp_softmax_lse = _gqa_attention(mtp_q_rot, mtp_k_rot, mtp_v, allow_math_fallback=True, deterministic=deterministic)
        mtp_attn_flat = mtp_attn.reshape(B, S, C.NUM_HEADS * C.HEAD_DIM)
        mtp_attn_out = torch.matmul(mtp_attn_flat, model.mtp.layer.attention_proj_weight.t())
        if capture_records is not None:
            _capture_forward(capture_records, capture_prefix, "mtp.layer.wo", 0, mtp_attn_out)
        mtp_hidden_after_attn = eagle_h + mtp_attn_out * depth_scale_mtp
        eagle_h = mtp_hidden_after_attn

        mtp_normed2 = rms_norm(eagle_h, model.mtp.layer.pre_mlp_norm_weight, deterministic=deterministic)
        mtp_gate_up = torch.matmul(mtp_normed2, model.mtp.layer.mlp_fc1_weight.t())
        # Use fused Triton SwiGLU forward kernel for non-deterministic mode.
        _use_triton_swiglu_fwd_mtp = (
            _HAS_TRITON_SWIGLU_FWD
            and not deterministic
            and mtp_gate_up.is_cuda
            and int(os.environ.get('ENABLE_TRITON_SWIGLU_FWD', '0'))
        )
        if _use_triton_swiglu_fwd_mtp:
            mtp_intermediate = _swiglu_fwd_fused(mtp_gate_up, C.FFN_HIDDEN_SIZE)
        else:
            mtp_y1, mtp_y2 = mtp_gate_up.chunk(2, dim=-1)
            mtp_intermediate = (torch.nn.functional.silu(mtp_y1.float()) * mtp_y2.float()).to(mtp_y1.dtype)
        mtp_mlp_out = torch.matmul(mtp_intermediate, model.mtp.layer.mlp_fc2_weight.t())
        if capture_records is not None:
            _capture_forward(capture_records, capture_prefix, "mtp.layer.w2", 0, mtp_mlp_out)
        eagle_h = eagle_h + mtp_mlp_out * depth_scale_mtp

        mtp_mtp_layer_cache = LayerCache(
            hidden_before_attn=mtp_hidden_before_attn,
            # normed is NOT stored — recomputed from hidden_before_attn in backward
            q=mtp_q, k=mtp_k, v=mtp_v,
            q_rot=mtp_q_rot, k_rot=mtp_k_rot,
            attn_out_raw=mtp_attn, attn_flat=mtp_attn_flat, attn_out=mtp_attn_out,
            softmax_lse=mtp_softmax_lse,
            hidden_after_attn=mtp_hidden_after_attn,
            # normed2 is NOT stored — recomputed from hidden_after_attn in backward
            gate_up=mtp_gate_up,
            mlp_out=mtp_mlp_out,
        )

        mtp_final = rms_norm(eagle_h, model.mtp.final_norm_weight, deterministic=deterministic)
        if capture_records is not None:
            _capture_forward(capture_records, capture_prefix, "mtp.final_layernorm", 0, mtp_final)

        mtp_pre_head = mtp_final / width_mult
        mtp_logits = lm_head_forward(mtp_pre_head, model.output_weight)
        if capture_records is not None:
            _capture_forward(capture_records, capture_prefix, "output", 1, mtp_logits)

        cache.mtp_emb = mtp_emb
        cache.mtp_a = a
        cache.mtp_b = b
        cache.mtp_eagle_h = eagle_h  # input to final_layernorm (after transformer layer)
        cache.mtp_layer_cache = mtp_mtp_layer_cache
        cache.mtp_final = mtp_final
        cache.mtp_pre_head = mtp_pre_head
        cache.mtp_logits = mtp_logits

    return cache


# ── MFU computation ──────────────────────────────────────────────────────────


def _compute_flops_per_step():
    """Compute the FLOPs per training step (same formula as ref)."""
    H_q = float(C.NUM_HEADS * C.HEAD_DIM)
    H_kv = float(C.NUM_KV_HEADS * C.HEAD_DIM)
    H = float(C.HIDDEN_SIZE)
    S = float(C.MAX_SEQ_LEN)
    ffn = float(C.FFN_HIDDEN_SIZE)
    V = float(C.VOCAB_SIZE)

    fwd_attn_proj_per_layer = 2.0 * H * (H_q + 2.0 * H_kv) + 2.0 * H_q * H
    fwd_attn_score_per_layer = 2.0 * S * H_q
    fwd_mlp_per_layer = 6.0 * H * ffn
    fwd_per_layer = fwd_attn_proj_per_layer + fwd_attn_score_per_layer + fwd_mlp_per_layer
    fwd_lm_head = 2.0 * H * V

    n_mtp = float(int(os.environ.get("MTP_NUM_LAYERS", "1")))
    fwd_mtp = n_mtp * (
        fwd_per_layer
        + 2.0 * (2.0 * H) * H
        + fwd_lm_head
    )
    fwd_per_token = float(C.NUM_LAYERS) * fwd_per_layer + fwd_lm_head + fwd_mtp
    train_per_token = 3.0 * fwd_per_token
    return train_per_token


# ── Static backward pass ─────────────────────────────────────────────────────


def _build_dptr_idx(bf16_params: list[torch.Tensor]) -> dict[int, int]:
    """Build a mapping from data_ptr to index in bf16_params."""
    return {p.data_ptr(): i for i, p in enumerate(bf16_params)}


def _add_to_grad_bufs(
    fp32_grad_bufs: list[torch.Tensor],
    dptr_idx: dict[int, int],
    weight: torch.Tensor,
    grad_weight: torch.Tensor,
) -> None:
    """Add a weight gradient to the matching fp32 gradient buffer.

    Uses a pre-built data_ptr → index mapping (O(1)) instead of
    linear search to avoid any subtle data_ptr aliasing issues.
    """
    idx = dptr_idx.get(weight.data_ptr())
    if idx is None:
        raise RuntimeError(
            f"_add_to_grad_bufs: no match for weight with shape {weight.shape}, "
            f"data_ptr={weight.data_ptr()}"
        )
    fp32_grad_bufs[idx].add_(grad_weight.to(device=fp32_grad_bufs[idx].device))


def _static_backward(
    model: ModelParameters,
    cache: ForwardCache,
    rope_freqs: torch.Tensor,
    width_mult: float,
    mup_emb_scale: float,
    depth_scale_main: float,
    depth_scale_mtp: float,
    fp32_grad_bufs: list[torch.Tensor],
    bf16_params: list[torch.Tensor],
    dptr_idx: dict[int, int],  # pre-built data_ptr→index mapping (cached across calls)
    capture_records: dict | None,
    capture_prefix: str,
    mtp_ce_weight: float,
    deterministic: bool = True,  # forward attention determinism flag
) -> None:
    """Static backward pass — no autograd, no ``.backward()``.

    Walks the computation graph in reverse, computing gradients for
    every parameter and accumulating fp32 gradients into ``fp32_grad_bufs``.
    """
    B, S = cache.input_ids.shape
    H = C.HIDDEN_SIZE
    V = C.VOCAB_SIZE

    # ====================================================================
    # MTP BRANCH BACKWARD (if enabled)
    # ====================================================================
    d_hidden_normed_main = None  # gradient from MTP into hidden_normed
    dw_output_mtp_accum = None  # shared param: output_weight MTP contribution
    dw_mtp_emb_accum = None  # shared param: tok_embeddings MTP contribution

    if cache.mtp_logits is not None and model.mtp is not None:
        # ── MTP Cross-entropy loss backward ────────────────────────────
        # Scale by ce_w in fp32 (inside cross_entropy_backward), matching
        # the ref's ``loss = lm_sum + ce_w * mtp_sum; loss.backward()``.
        grad_mtp_logits = cross_entropy_backward(
            cache.mtp_logits, cache.mtp_labels, cache.mtp_loss_mask,
            scale=mtp_ce_weight, deterministic=deterministic,
        )

        # ── MTP LM head backward: logits = mtp_pre_head @ output_weight.T
        # d(mtp_pre_head) = grad_mtp_logits @ output_weight
        # d(output_weight) += grad_mtp_logits.T @ mtp_pre_head
        # grad_mtp_logits is bf16 (from cross_entropy_backward), output_weight is bf16.
        d_mtp_pre_head = torch.matmul(grad_mtp_logits, model.output_weight)
        g2_mtp = grad_mtp_logits.reshape(-1, V)
        mtp_pre = cache.mtp_pre_head.reshape(-1, H)
        dw_output_mtp = torch.matmul(g2_mtp.transpose(0, 1), mtp_pre).float()
        # Defer add: combine with main contribution in fp32 first (matching ref's
        # single ``buf.add_(main_grad)`` per microbatch).  Two separate ``add_``
        # calls produce ``(buf + A) + B`` while the ref does ``buf + (A + B)``;
        # these differ in fp32 when ``buf != 0`` (microbatch >= 2).
        dw_output_mtp_accum = dw_output_mtp

        # ── width_mult backward: d(hidden) = d(mtp_pre_head) / width_mult
        d_mtp_final = d_mtp_pre_head / width_mult

        # ── mtp.final_layernorm backward
        d_mtp_eagle_h, dw_mtp_fn = rms_norm_backward(
            d_mtp_final, cache.mtp_eagle_h, model.mtp.final_norm_weight
        ,
        deterministic=deterministic,
    )
        _add_to_grad_bufs(fp32_grad_bufs, dptr_idx, model.mtp.final_norm_weight, dw_mtp_fn)

        # ── MTP transformer layer backward (reverse order)
        # First, the MLP residual: d_eagle_h = d_mtp_eagle_h (from MLP branch)
        # Then, w2 backward
        mtp_lc = cache.mtp_layer_cache

        # w2 backward: mlp_out = intermediate @ w2.T
        # Recompute y1, y2, intermediate from gate_up (not stored in cache
        # to save activation memory).
        _mtp_y1, _mtp_y2 = mtp_lc.gate_up.chunk(2, dim=-1)
        _mtp_intermediate = (torch.nn.functional.silu(_mtp_y1.float()) * _mtp_y2.float()).to(_mtp_y1.dtype)
        d_intermediate, dw_mtp_w2 = linear_backward(
            d_mtp_eagle_h * depth_scale_mtp,
            _mtp_intermediate,
            model.mtp.layer.mlp_fc2_weight,
        )
        _add_to_grad_bufs(fp32_grad_bufs, dptr_idx, model.mtp.layer.mlp_fc2_weight, dw_mtp_w2)

        # SwiGLU backward (returns d_gate_up directly — no torch.cat needed)
        d_mtp_gate_up = silu_swiglu_intermediate_backward(
            d_intermediate, _mtp_y1, _mtp_y2,
            gate_up=mtp_lc.gate_up, ffn_half=C.FFN_HIDDEN_SIZE,
            deterministic=deterministic,
        )

        # wfc1 backward: gate_up = normed2 @ wfc1.T
        # Recompute normed2 from hidden_after_attn (not stored in cache
        # to save activation memory).
        _mtp_normed2 = rms_norm(mtp_lc.hidden_after_attn, model.mtp.layer.pre_mlp_norm_weight, deterministic=deterministic)
        d_mtp_normed2, dw_mtp_fc1 = linear_backward(
            d_mtp_gate_up, _mtp_normed2, model.mtp.layer.mlp_fc1_weight
        )
        _add_to_grad_bufs(fp32_grad_bufs, dptr_idx, model.mtp.layer.mlp_fc1_weight, dw_mtp_fc1)

        # ffn_norm backward (MLP branch)
        d_mtp_hidden_mlp, dw_mtp_mlp_norm = rms_norm_backward(
            d_mtp_normed2, mtp_lc.hidden_after_attn, model.mtp.layer.pre_mlp_norm_weight
        ,
        deterministic=deterministic,
    )
        _add_to_grad_bufs(fp32_grad_bufs, dptr_idx, model.mtp.layer.pre_mlp_norm_weight, dw_mtp_mlp_norm)

        # Add MLP and attention gradient at hidden
        # The gradient of the loss w.r.t. hidden_after_attn is:
        #   d_mtp_eagle_h (from the MLP residual connection) + d_mtp_hidden_mlp (from MLP compute)
        d_mtp_eagle_h = d_mtp_eagle_h + d_mtp_hidden_mlp

        # Attention branch residual from hidden_before_attn
        # wo backward: attn_out = attn_flat @ wo.T
        d_mtp_attn_flat, dw_mtp_wo = linear_backward(
            d_mtp_eagle_h * depth_scale_mtp,
            mtp_lc.attn_flat,
            model.mtp.layer.attention_proj_weight,
        )
        _add_to_grad_bufs(fp32_grad_bufs, dptr_idx, model.mtp.layer.attention_proj_weight, dw_mtp_wo)

        # Attention backward (GQA) — uses autograd replay through
        # flash_attn_func for bitwise alignment (the softmax_lse is not
        # exposed by the installed flash_attn version, so the direct
        # _flash_attn_backward path cannot be used).
        d_mtp_attn = d_mtp_attn_flat.reshape(B, S, C.NUM_HEADS, C.HEAD_DIM)
        d_mtp_q_rot, d_mtp_k_rot, d_mtp_v = gqa_attention_backward(
            d_mtp_attn, mtp_lc.q_rot, mtp_lc.k_rot, mtp_lc.v,
            out=mtp_lc.attn_out_raw, softmax_lse=mtp_lc.softmax_lse,
            deterministic=deterministic,
        )

        # RoPE backward
        d_mtp_q = apply_rope_backward(d_mtp_q_rot, rope_freqs, deterministic=deterministic)
        d_mtp_k = apply_rope_backward(d_mtp_k_rot, rope_freqs, deterministic=deterministic)

        # QKV projection backward
        # Recompute normed from hidden_before_attn (not stored in cache
        # to save activation memory).
        _mtp_normed = rms_norm(mtp_lc.hidden_before_attn, model.mtp.layer.input_norm_weight, deterministic=deterministic)
        d_mtp_normed, dw_mtp_qkv = project_qkv_backward(
            d_mtp_q, d_mtp_k, d_mtp_v,
            _mtp_normed, model.mtp.layer.qkv_weight,
        )
        _add_to_grad_bufs(fp32_grad_bufs, dptr_idx, model.mtp.layer.qkv_weight, dw_mtp_qkv)

        # attention_norm backward
        d_mtp_hidden_before_attn, dw_mtp_attn_norm = rms_norm_backward(
            d_mtp_normed, mtp_lc.hidden_before_attn, model.mtp.layer.input_norm_weight
        ,
        deterministic=deterministic,
    )
        _add_to_grad_bufs(fp32_grad_bufs, dptr_idx, model.mtp.layer.input_norm_weight, dw_mtp_attn_norm)

        # Add residual from attention branch
        d_mtp_eagle_h_out = d_mtp_hidden_before_attn + d_mtp_eagle_h

        # ── eagle_fc backward: eagle_h = cat([a, b]) @ eagle_fc.T
        d_mtp_cat, dw_mtp_eagle_fc = linear_backward(
            d_mtp_eagle_h_out, torch.cat([cache.mtp_a, cache.mtp_b], dim=-1),
            model.mtp.eagle_fc_weight,
        )
        _add_to_grad_bufs(fp32_grad_bufs, dptr_idx, model.mtp.eagle_fc_weight, dw_mtp_eagle_fc)
        d_mtp_a, d_mtp_b = d_mtp_cat.chunk(2, dim=-1)

        # mtp.hidden_input_layernorm backward
        d_hidden_normed_mtp, dw_mtp_hnorm = rms_norm_backward(
            d_mtp_b, cache.hidden_normed, model.mtp.hidden_input_norm_weight
        ,
        deterministic=deterministic,
    )
        _add_to_grad_bufs(fp32_grad_bufs, dptr_idx, model.mtp.hidden_input_norm_weight, dw_mtp_hnorm)

        # mtp.emb_input_layernorm backward
        grad_mtp_emb, dw_mtp_enorm = rms_norm_backward(
            d_mtp_a, cache.mtp_emb, model.mtp.emb_input_norm_weight
        ,
        deterministic=deterministic,
    )
        _add_to_grad_bufs(fp32_grad_bufs, dptr_idx, model.mtp.emb_input_norm_weight, dw_mtp_enorm)

        # MTP embedding backward: gradient flows through rms_norm then
        # mup_emb_scale to reach the embedding weight.
        # Forward: mtp_emb = F.embedding(ids, w) * mup_emb_scale
        #          a = rms_norm(mtp_emb, enorm)
        # Backward: grad_mtp_emb = rms_norm_backward(d_mtp_a, mtp_emb, ...)[0]
        #           dw_emb = embedding_backward(grad_mtp_emb * mup_emb_scale, ids, V)
        dw_mtp_emb = embedding_backward(
            grad_mtp_emb * mup_emb_scale, cache.mtp_input_ids, V
        )
        dw_mtp_emb_accum = dw_mtp_emb  # defer add: combine with main contribution

        # Accumulate gradient into hidden_normed from MTP
        d_hidden_normed_main = d_hidden_normed_mtp

    # ====================================================================
    # MAIN BRANCH BACKWARD
    # ====================================================================

    # ── Main Cross-entropy loss backward ───────────────────────────────
    grad_logits = cross_entropy_backward(cache.main_logits, cache.labels, cache.loss_mask, deterministic=deterministic)

    # ── Main LM head backward: logits = main_pre_head @ output_weight.T
    # grad_logits is fp32 (from FP32 CE), output_weight is bf16.
    # Match the ref's _LinearFn.backward: keep weight in bf16 for the matmul.
    d_main_pre_head = torch.matmul(grad_logits, model.output_weight)
    g2_main = grad_logits.reshape(-1, V)
    main_pre = cache.main_pre_head.reshape(-1, H)
    dw_output_main = torch.matmul(g2_main.transpose(0, 1), main_pre).float()
    # Combine MTP + main contributions in fp32 before a single add_ into
    # fp32_grad_bufs (matching ref's ``buf.add_(main_grad)`` per microbatch).
    dw_output = dw_output_main + dw_output_mtp_accum if dw_output_mtp_accum is not None else dw_output_main
    _add_to_grad_bufs(fp32_grad_bufs, dptr_idx, model.output_weight, dw_output)

    # ── width_mult backward: d(hidden) = d(main_pre_head) / width_mult
    d_hidden_normed = d_main_pre_head / width_mult

    # Add MTP contribution to hidden_normed gradient
    if d_hidden_normed_main is not None:
        d_hidden_normed = d_hidden_normed + d_hidden_normed_main

    # ── final_norm backward
    # The input to final_norm is the hidden AFTER the last layer:
    # hidden_post_last_layer = last_layer.hidden_after_attn + last_layer.mlp_out * depth_scale_main
    last_layer = cache.layer_caches[-1]
    hidden_post_last_layer = last_layer.hidden_after_attn + last_layer.mlp_out * depth_scale_main
    d_hidden, dw_final_norm = rms_norm_backward(
        d_hidden_normed, hidden_post_last_layer, model.final_norm_weight
    ,
        deterministic=deterministic,
    )
    _add_to_grad_bufs(fp32_grad_bufs, dptr_idx, model.final_norm_weight, dw_final_norm)

    # ── Transformer layers (reverse order) ─────────────────────────────
    for li in range(len(model.layers) - 1, -1, -1):
        layer = model.layers[li]
        lc = cache.layer_caches[li]

        # ── MLP backward ──────────────────────────────────────────────
        # The MLP residual is: d_hidden includes d_hidden from layers above
        # mlp_out = intermediate @ w2.T
        # Recompute y1, y2, intermediate from gate_up (not stored in cache
        # to save ~432 MB of activation memory per layer).
        _y1, _y2 = lc.gate_up.chunk(2, dim=-1)
        _intermediate = (torch.nn.functional.silu(_y1.float()) * _y2.float()).to(_y1.dtype)
        d_intermediate, dw_w2 = linear_backward(
            d_hidden * depth_scale_main, _intermediate, layer.mlp_fc2_weight
        )
        _add_to_grad_bufs(fp32_grad_bufs, dptr_idx, layer.mlp_fc2_weight, dw_w2)

        # SwiGLU backward (returns d_gate_up directly — no torch.cat needed)
        d_gate_up = silu_swiglu_intermediate_backward(
            d_intermediate, _y1, _y2,
            gate_up=lc.gate_up, ffn_half=C.FFN_HIDDEN_SIZE,
            deterministic=deterministic,
        )

        # wfc1 backward: gate_up = normed2 @ wfc1.T
        # Recompute normed2 from hidden_after_attn (not stored in cache
        # to save ~432 MB of activation memory per layer).
        _normed2 = rms_norm(lc.hidden_after_attn, layer.pre_mlp_norm_weight, deterministic=deterministic)
        d_normed2, dw_fc1 = linear_backward(
            d_gate_up, _normed2, layer.mlp_fc1_weight
        )
        _add_to_grad_bufs(fp32_grad_bufs, dptr_idx, layer.mlp_fc1_weight, dw_fc1)

        # ffn_norm backward (MLP branch into hidden_after_attn)
        d_hidden_mlp, dw_mlp_norm = rms_norm_backward(
            d_normed2, lc.hidden_after_attn, layer.pre_mlp_norm_weight
        ,
        deterministic=deterministic,
    )
        _add_to_grad_bufs(fp32_grad_bufs, dptr_idx, layer.pre_mlp_norm_weight, dw_mlp_norm)

        # Add MLP residual: hidden = hidden_before_attn + attn_out * depth_scale
        # then hidden = hidden + mlp_out * depth_scale.
        # The gradient of the loss w.r.t. hidden_before_attn is:
        #   d_hidden (from the MLP residual connection) + d_hidden_mlp (from MLP compute)
        d_hidden = d_hidden + d_hidden_mlp

        # ── Attention backward ─────────────────────────────────────────
        # wo backward: attn_out = attn_flat @ wo.T
        d_attn_flat, dw_wo = linear_backward(
            d_hidden * depth_scale_main, lc.attn_flat, layer.attention_proj_weight
        )
        _add_to_grad_bufs(fp32_grad_bufs, dptr_idx, layer.attention_proj_weight, dw_wo)

        # Attention backward (GQA) — uses autograd replay through
        # flash_attn_func for bitwise alignment (the softmax_lse is not
        # exposed by the installed flash_attn version, so the direct
        # _flash_attn_backward path cannot be used).
        d_attn = d_attn_flat.reshape(B, S, C.NUM_HEADS, C.HEAD_DIM)
        d_q_rot, d_k_rot, d_v = gqa_attention_backward(
            d_attn, lc.q_rot, lc.k_rot, lc.v,
            out=lc.attn_out_raw, softmax_lse=lc.softmax_lse,
            deterministic=deterministic,
        )

        # RoPE backward
        d_q = apply_rope_backward(d_q_rot, rope_freqs, deterministic=deterministic)
        d_k = apply_rope_backward(d_k_rot, rope_freqs, deterministic=deterministic)

        # QKV projection backward
        # Recompute normed from hidden_before_attn (not stored in cache
        # to save ~432 MB of activation memory per layer).
        _normed = rms_norm(lc.hidden_before_attn, layer.input_norm_weight, deterministic=deterministic)
        d_normed, dw_qkv = project_qkv_backward(
            d_q, d_k, d_v, _normed, layer.qkv_weight
        )
        _add_to_grad_bufs(fp32_grad_bufs, dptr_idx, layer.qkv_weight, dw_qkv)

        # attention_norm backward
        d_hidden_before_attn, dw_attn_norm = rms_norm_backward(
            d_normed, lc.hidden_before_attn, layer.input_norm_weight
        ,
        deterministic=deterministic,
    )
        _add_to_grad_bufs(fp32_grad_bufs, dptr_idx, layer.input_norm_weight, dw_attn_norm)

        # Add residual from attention branch: hidden = hidden_before_attn + attn_out * depth_scale
        # The gradient of the loss w.r.t. hidden_before_attn is:
        #   d_hidden (from the attention residual connection) + d_hidden_before_attn (from attention compute)
        d_hidden = d_hidden + d_hidden_before_attn

    # ── Embedding backward ─────────────────────────────────────────────
    dw_emb = embedding_backward(
        d_hidden * mup_emb_scale, cache.input_ids, C.VOCAB_SIZE
    )
    # Combine MTP + main contributions in fp32 before a single add_ into
    # fp32_grad_bufs (matching ref's ``buf.add_(main_grad)`` per microbatch).
    if dw_mtp_emb_accum is not None:
        dw_emb = dw_emb + dw_mtp_emb_accum
    _add_to_grad_bufs(fp32_grad_bufs, dptr_idx, model.tok_embeddings_weight, dw_emb)


# ── Optimizer helpers ────────────────────────────────────────────────────────


# ── Optimizer helpers ────────────────────────────────────────────────────────


def _build_optimizer_groups(
    fp32_master: list[torch.Tensor],
    bf16_params: list[torch.Tensor],
    model: ModelParameters,
    lr: float,
    width_mult: float,
    weight_decay: float,
) -> list[dict]:
    """Build parameter groups matching the ref's AdamW with muP lr scaling.

    The ref (``model_pure_mup_mtp.py:mup_lr_groups`` and
    ``train_pure_mup_mtp.py`` L601-615) splits into:
      - muP-scaled matrix weights (qkv, wfc1, eagle_fc, wo, w2) → ``lr / width_mult``
      - unscaled weights (embedding, output, all norms) → ``lr``
    Within each group, params with dim >= 2 get ``weight_decay``, dim < 2 get 0.
    """
    fqn_map = _build_fqn_map(model)

    # Determine which FQNs are muP-scaled matrix weights (mirroring ref's
    # ``mup_lr_groups``: ``.weight`` AND not norm AND not tok_embeddings
    # AND not output.weight).
    def _is_matrix_fqn(fqn: str) -> bool:
        return (
            fqn.endswith(".weight")
            and "norm" not in fqn
            and "tok_embeddings" not in fqn
            and "output" not in fqn
        )

    scaled_wd: list[torch.Tensor] = []
    scaled_no_wd: list[torch.Tensor] = []
    unscaled_wd: list[torch.Tensor] = []
    unscaled_no_wd: list[torch.Tensor] = []

    for buf, p in zip(fp32_master, bf16_params):
        fqn = fqn_map.get(p.data_ptr(), "")
        is_matrix = _is_matrix_fqn(fqn)
        if is_matrix:
            (scaled_wd if buf.dim() >= 2 else scaled_no_wd).append(buf)
        else:
            (unscaled_wd if buf.dim() >= 2 else unscaled_no_wd).append(buf)

    groups: list[dict] = []
    scaled_lr = lr / width_mult
    for g_params, g_lr, g_wd in [
        (scaled_wd, scaled_lr, weight_decay),
        (scaled_no_wd, scaled_lr, 0.0),
        (unscaled_wd, lr, weight_decay),
        (unscaled_no_wd, lr, 0.0),
    ]:
        if g_params:
            groups.append({"params": g_params, "lr": g_lr, "weight_decay": g_wd})
    return groups


def _adamw_step(
    optim_groups: list[dict],
    fp32_master: list[torch.Tensor],
    exp_avgs: dict[int, torch.Tensor],
    exp_avg_sqs: dict[int, torch.Tensor],
    state_steps: dict[int, torch.Tensor],
    beta1: float,
    beta2: float,
    eps: float,
    dummy_sq: torch.Tensor | None = None,  # pre-allocated 1-element fp32 tensor
) -> None:
    """Apply one AdamW step using the fused multi-tensor kernel.

    Reads gradients from ``p.grad`` on each ``fp32_master`` tensor (same
    as the ref's ``optim.step()`` which reads from ``.grad`` after
    ``clip_grad_norm_`` sets it).

    ``dummy_sq``: a pre-allocated 1-element fp32 tensor used as a
    placeholder for ``max_exp_avg_sqs`` (``amsgrad=False`` means the
    kernel never accesses it).  When ``None``, a new tensor is created
    (the old per-step allocation fallback).  Pre-allocating once and
    reusing avoids a CUDA allocator call per optimizer group per step
    that can trigger an internal ``cudaStreamSynchronize``."""
    for group in optim_groups:
        params = group["params"]
        n = len(params)
        if n == 0:
            continue
        lr = group["lr"]
        wd = group["weight_decay"]

        # Collect gradients from .grad, momentum buffers, and step counters
        grads: list[torch.Tensor] = []
        eas: list[torch.Tensor] = []
        eass: list[torch.Tensor] = []
        steps: list[torch.Tensor] = []
        for p in params:
            grads.append(p.grad)
            eas.append(exp_avgs[p.data_ptr()])
            eass.append(exp_avg_sqs[p.data_ptr()])
            steps.append(state_steps[p.data_ptr()])

        # Increment step counters (fused kernel reads the current step)
        torch._foreach_add_(steps, 1)

        # Call fused AdamW kernel (same underlying op as ref's fused=True).
        # max_exp_avg_sqs must be a tuple of tensors (not None) — the
        # remote PyTorch version rejects NoneType for this parameter.
        # Use a single shared 1-element tensor for all entries (amsgrad=False
        # means the kernel never accesses max_exp_avg_sqs), saving 157
        # torch.zeros_like allocations (~628 MB) per step.
        if dummy_sq is not None:
            _dummy_sq = dummy_sq
        else:
            _dummy_sq = torch.zeros(1, device=params[0].device)
        torch._fused_adamw_(
            tuple(params),
            tuple(grads),
            tuple(eas),
            tuple(eass),
            tuple(_dummy_sq for _ in params),
            tuple(steps),
            amsgrad=False,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            weight_decay=wd,
            eps=eps,
            maximize=False,
        )


def _sync_bf16_from_fp32(
    bf16_params: list[torch.Tensor],
    fp32_master: list[torch.Tensor],
    flat_fp32_buf: torch.Tensor | None = None,
    flat_bf16_buf: torch.Tensor | None = None,
) -> None:
    """Copy FP32 master weights back to BF16 params using a flat fused approach.

    Flattens all FP32 master weights into one contiguous tensor, does a single
    ``bfloat16()`` conversion, then copies back to the per-parameter BF16
    buffers.  This reduces the 157 separate ``bfloat16()`` kernel launches
    (one per param) to a single fused kernel launch, saving ~1.25ms of launch
    overhead per step.

    When ``flat_fp32_buf`` and ``flat_bf16_buf`` are provided (pre-allocated
    contiguous tensors), the function uses ``torch.cat(..., out=flat_fp32_buf)``
    to write directly to the pre-allocated buffers, avoiding the 6.18 GiB of
    temporary CUDA allocations per step that can trigger internal
    ``cudaStreamSynchronize`` when the allocator cache is under memory pressure
    from the CUDA graph's private pools.
    """
    if flat_fp32_buf is not None and flat_bf16_buf is not None:
        # Pre-allocated path: write directly into the pre-allocated buffers.
        # torch.cat with out= writes the concatenated data into the contiguous
        # buffer — same operation as _flatten_dense_tensors but without the
        # per-step allocation.  The reshape(-1) creates views (no allocation).
        torch.cat([t.reshape(-1) for t in fp32_master], out=flat_fp32_buf)
        flat_bf16_buf.copy_(flat_fp32_buf)
        unflattened = torch._utils._unflatten_dense_tensors(flat_bf16_buf, bf16_params)
        torch._foreach_copy_(bf16_params, unflattened)
        return

    # Fallback: allocate flat buffers per-step (e.g. when pre-allocated
    # buffers are not available, such as the checkpoint-load sync call).
    try:
        flat_fp32 = torch._utils._flatten_dense_tensors(fp32_master)
        flat_bf16 = flat_fp32.bfloat16()
        unflattened = torch._utils._unflatten_dense_tensors(flat_bf16, bf16_params)
        # Fused copy: single kernel launch for all 157 params.
        torch._foreach_copy_(bf16_params, unflattened)
    except (RuntimeError, MemoryError, AttributeError):
        # Fallback to per-parameter approach (OOM from fragmentation).
        bf16_views = [p.bfloat16() for p in fp32_master]
        try:
            torch._foreach_copy_(bf16_params, bf16_views)
        except (AttributeError, RuntimeError, TypeError):
            for p_bf16, bf16_view in zip(bf16_params, bf16_views):
                p_bf16.data.copy_(bf16_view)


def _init_optimizer_state(
    fp32_master: list[torch.Tensor],
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Initialize AdamW optimizer state (exp_avg, exp_avg_sq, step).

    Returns dicts keyed by fp32 master tensor ``data_ptr``.
    """
    exp_avgs: dict[int, torch.Tensor] = {}
    exp_avg_sqs: dict[int, torch.Tensor] = {}
    state_steps: dict[int, torch.Tensor] = {}
    for p in fp32_master:
        key = p.data_ptr()
        exp_avgs[key] = torch.zeros_like(p)
        exp_avg_sqs[key] = torch.zeros_like(p)
        # Fused kernel requires float32 scalar step
        state_steps[key] = torch.tensor(0.0, device=device, dtype=torch.float32)
    return exp_avgs, exp_avg_sqs, state_steps


# ── Main training loop ────────────────────────────────────────────────────────


def save_checkpoint(
    save_dir: str,
    step: int,
    fp32_master: list[torch.Tensor],
    exp_avgs: dict[int, torch.Tensor],
    exp_avg_sqs: dict[int, torch.Tensor],
    state_steps: dict[int, torch.Tensor],
    rank: int,
) -> None:
    """Save full training state to ``save_dir/step_<abs>/training_state.pt``.

    Saves FP32 master weights, AdamW optimizer state (m, v, step), and
    RNG states (torch, torch.cuda, numpy, random).  Only rank 0 writes
    (others no-op).  The directory is created on rank 0 if needed.

    Args:
        save_dir: Root checkpoint directory (e.g. ``"resume_scratch/ckpt_1234"``).
        step: Absolute step number this checkpoint represents (1-indexed).
        fp32_master: FP32 master weight tensors (list from ``run_training_loop``).
        exp_avgs: AdamW first moment dict, keyed by fp32_master data_ptr.
        exp_avg_sqs: AdamW second moment dict, keyed by fp32_master data_ptr.
        state_steps: AdamW step counter dict, keyed by fp32_master data_ptr.
        rank: Global process rank (only rank 0 writes).
    """
    if rank != 0:
        return
    subdir = Path(save_dir) / f"step_{step}"
    subdir.mkdir(parents=True, exist_ok=True)
    path = subdir / "training_state.pt"

    # Serialise optimizer state as ordered lists (data_ptr keys are not stable
    # across save/load — the tensors are re-allocated by torch.load).
    # RNG states are serialised as bytes objects (pickle-compatible) to avoid
    # ``weights_only=True`` rejection in torch.load (numpy RNG state uses
    # ``numpy.core.multiarray._reconstruct`` which is not in the default
    # allowlist).
    import pickle as _pickle
    state = {
        "fp32_master": fp32_master,
        # Optimizer state must be saved in the SAME order as fp32_master
        # (the data_ptr keys change across save/load, so sorting by key
        # would produce a different order than the parameter list).
        "exp_avgs": [exp_avgs[p.data_ptr()] for p in fp32_master],
        "exp_avg_sqs": [exp_avg_sqs[p.data_ptr()] for p in fp32_master],
        "state_steps": [state_steps[p.data_ptr()] for p in fp32_master],
        "step": torch.tensor(step, dtype=torch.int32),
        # RNG states serialised as pickle bytes for safe ``weights_only`` reload
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state(),
        "rng_numpy": _pickle.dumps(__import__("numpy").random.get_state()),
        "rng_random": _pickle.dumps(__import__("random").getstate()),
    }
    torch.save(state, path)


def load_checkpoint(
    checkpoint_dir: str,
    fp32_master: list[torch.Tensor],
    exp_avgs: dict[int, torch.Tensor],
    exp_avg_sqs: dict[int, torch.Tensor],
    state_steps: dict[int, torch.Tensor],
    rank: int,
    init_weights_only: bool = False,
) -> int:
    """Load full training state from a checkpoint directory.

    All ranks load from the shared filesystem independently (the checkpoint
    directory is on a shared mount accessible to all ranks).  When
    ``init_weights_only`` is True, only the FP32 master weights are restored
    (optimizer state is NOT loaded — used for the SFT phase which needs a
    fresh optimizer).

    Args:
        checkpoint_dir: Path to the checkpoint directory (e.g. ``"ckpt_1234"``
            or ``"ckpt_1234/step_10"`` for versioned checkpoints).
        fp32_master: Existing FP32 master weight tensors (will be overwritten
            in-place from the saved state).
        exp_avgs: AdamW first moment dict (will be overwritten).
        exp_avg_sqs: AdamW second moment dict (will be overwritten).
        state_steps: AdamW step counter dict (will be overwritten).
        rank: Global process rank (for logging only).
        init_weights_only: If True, only load fp32_master weights, skip
            optimizer state (AdamW m, v, step).

    Returns:
        The loaded step number (1-indexed).
    """
    # All ranks load from the shared filesystem independently.
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(
            f"load_checkpoint: checkpoint directory not found: {checkpoint_dir}"
        )
    candidates = list(ckpt_dir.glob("*training_state.pt"))
    if not candidates:
        # Try subdirectories (versioned checkpoints: step_<N>/training_state.pt)
        candidates = list(ckpt_dir.rglob("*training_state.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"load_checkpoint: no *training_state.pt found in {checkpoint_dir}"
        )
    # Pick the latest by modification time
    ckpt_path = max(candidates, key=lambda p: p.stat().st_mtime)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    step_val = state["step"].item() if isinstance(state["step"], torch.Tensor) else int(state["step"])

    # Load FP32 master weights (all ranks have their own copy)
    master_loaded = state["fp32_master"]
    for p_fp32, p_loaded in zip(fp32_master, master_loaded):
        p_fp32.copy_(p_loaded.to(device=fp32_master[0].device))

    if not init_weights_only:
        # Load optimizer state
        loaded_avgs = state["exp_avgs"]
        loaded_sqs = state["exp_avg_sqs"]
        loaded_steps = state["state_steps"]
        for p_fp32, avg, sq, stp in zip(fp32_master, loaded_avgs, loaded_sqs, loaded_steps):
            key = p_fp32.data_ptr()
            exp_avgs[key].copy_(avg.to(device=fp32_master[0].device))
            exp_avg_sqs[key].copy_(sq.to(device=fp32_master[0].device))
            state_steps[key].copy_(stp.to(device=fp32_master[0].device))

        # Restore RNG state (only rank 0 needs the deterministic RNG; the
        # others are deterministic through the same seed and CUDA operations)
        if rank == 0:
            torch.set_rng_state(state["rng_torch"].cpu())
            torch.cuda.set_rng_state(state["rng_cuda"].cpu())
            import pickle as _pickle
            __import__("numpy").random.set_state(_pickle.loads(state["rng_numpy"]))
            __import__("random").setstate(_pickle.loads(state["rng_random"]))

    return step_val


def _advance_dataloader(iter_dl, num_batches: int, device: torch.device) -> None:
    """Advance the dataloader by ``num_batches`` batches (discarding data).

    Used when resuming from a checkpoint: the dataloader must be advanced
    past the batches that were consumed before the checkpoint was saved.
    Mirrors the ref's ``_advance_dataloader`` in ``train_pure_mup_mtp.py``.
    """
    for _ in range(num_batches):
        data = next(iter_dl)
        # Skip zero-loss-mask batches like _next_batch does
        while (data["loss_mask"] == 0).all().item():
            data = next(iter_dl)
        # Touch the data to ensure it's consumed (the dataloader advances
        # its internal iterator on each __next__ call).
        _ = data["tokens"].to(device)
        _ = data["labels"].to(device)
        _ = data["loss_mask"].float().to(device)


def run_training_loop(config: TrainLoopConfig, *, loss_tag: str = "LOSS") -> None:
    """Run ``num_steps`` of training and emit one line per global step.

    See module-level docstring for the full contract and stdout grammar.
    """
    # ── Deterministic mode selection ───────────────────────────────────
    # The ref and ours both read a DETERMINISTIC=0/1 env var.  For bitwise
    # alignment milestones (alignment → resume) DETERMINISTIC=1 is required;
    # for long-horizon performance optimization DETERMINISTIC=0 (the default)
    # disables the full determinism stack, enabling flash attention, cuBLAS
    # non-deterministic algorithms, and cuDNN autotuning.
    deterministic = int(os.environ.get("DETERMINISTIC", "0"))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    rank = _init_process_group()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"

    # Always seed the RNG (ref's set_seed always calls torch.manual_seed).
    import random
    import numpy as np
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)

    # ── Model parameters ───────────────────────────────────────────────
    init_ones = int(os.environ.get("FORGE_INIT_ONES", "0"))
    model = load_weights_from_checkpoint(
        checkpoint_root=config.checkpoint_root,
        device=torch.device(device),
        world_size=config.world_size,
        rank=rank,
        init_std=float(os.environ.get("INIT_METHOD_STD", "0.02")),
        seed=config.seed,
        init_ones=bool(init_ones),
        mtp_num_layers=int(os.environ.get("MTP_NUM_LAYERS", "1")),
        mup_base_hidden_size=int(os.environ.get("MUP_BASE_HIDDEN_SIZE", "256")),
        mup_emb_scale=float(os.environ.get("MUP_EMB_SCALE", "12.0")),
        mup_depth_scale=float(os.environ.get("MUP_DEPTH_SCALE", "1.4")),
    )
    if rank == 0:
        import sys
        print(f"[debug] model.tok_embeddings_weight.data_ptr={model.tok_embeddings_weight.data_ptr()}, model.output_weight.data_ptr={model.output_weight.data_ptr()}, same={model.tok_embeddings_weight.data_ptr() == model.output_weight.data_ptr()}", file=sys.stderr, flush=True)

    # muP constants
    mup_base_hidden_size = float(os.environ.get("MUP_BASE_HIDDEN_SIZE", "256"))
    mup_emb_scale = float(os.environ.get("MUP_EMB_SCALE", "12.0"))
    mup_depth_scale = float(os.environ.get("MUP_DEPTH_SCALE", "1.4"))
    width_mult = C.HIDDEN_SIZE / mup_base_hidden_size
    depth_scale_main = mup_depth_scale / (C.NUM_LAYERS ** 0.5)
    depth_scale_mtp = mup_depth_scale / ((C.NUM_LAYERS + 1) ** 0.5)

    # ── RoPE freqs ─────────────────────────────────────────────────────
    rope_freqs = precompute_rope_freqs(max_seq_len=C.MAX_SEQ_LEN, device=device)

    # ── FP32 master weights ────────────────────────────────────────────
    bf16_params = _collect_bf16_params(model)
    fp32_master = [p.detach().float().clone() for p in bf16_params]

    # ── FP32 gradient buffers ──────────────────────────────────────────
    fp32_grad_bufs = [torch.zeros_like(p_) for p_ in fp32_master]

    # ── Pre-allocate flat buffers for BF16 sync ────────────────────────
    # Pre-allocate contiguous flat buffers for _sync_bf16_from_fp32 to
    # avoid 6.18 GiB of temporary CUDA allocations per step (flat_fp32
    # 4.1 GiB + flat_bf16 2.08 GiB).  These allocations can trigger
    # internal cudaStreamSynchronize when the CUDA allocator cache is
    # under memory pressure from the CUDA graph's private pools.
    _f32_numel = sum(p.numel() for p in fp32_master)
    _flat_fp32_buf = torch.empty(_f32_numel, dtype=torch.float32, device=device)
    _flat_bf16_buf = torch.empty(_f32_numel, dtype=torch.bfloat16, device=device)

    # Pre-allocate a 4-element fp64 tensor for the loss scalar all-reduce.
    # The torch.cat for [lm_sum, lm_n, mtp_sum, mtp_n] creates a 32-byte tensor
    # every step.  Pre-allocating avoids the per-step CUDA allocator call.
    _stats_buf = torch.empty(4, dtype=torch.float64, device=device)

    # ── Optimizer state (AdamW) ────────────────────────────────────────
    opt_beta1 = float(os.environ.get("ADAM_BETA1", "0.9"))
    opt_beta2 = float(os.environ.get("ADAM_BETA2", "0.95"))
    opt_eps = 1e-8
    opt_wd = float(os.environ.get("WEIGHT_DECAY", "0.1"))
    # Read lr/min_lr/decay from env (rendered product) first, then fall
    # back to config (for gate scripts that pass them explicitly).
    opt_lr = float(os.environ.get("LR", str(config.lr)))
    opt_min_lr = float(os.environ.get("MIN_LR", str(config.min_lr)))
    opt_lr_warmup = int(os.environ.get("LR_WARMUP_ITERS", str(config.lr_warmup_iters)))
    opt_lr_decay = int(os.environ.get("LR_DECAY_ITERS", str(config.lr_decay_iters)))
    opt_lr_wsd_decay = int(os.environ.get("LR_WSD_DECAY_ITERS", str(config.lr_wsd_decay_iters)))
    opt_clip_grad = float(os.environ.get("CLIP_GRAD", "1.0"))

    exp_avgs, exp_avg_sqs, opt_state_steps = _init_optimizer_state(fp32_master, device)
    optim_groups = _build_optimizer_groups(
        fp32_master, bf16_params, model,
        lr=opt_lr, width_mult=width_mult, weight_decay=opt_wd,
    )
    # Save per-group LR multipliers (muP scaling: scaled matrix weights → lr/width_mult)
    # so we can update per-group LR before each optimizer step (matching ref's pattern).
    lr_mult_per_group = [g["lr"] / opt_lr for g in optim_groups]

    # ── ZeRO-1 distributed optimizer ─────────────────────────────────────
    # Shard FP32 optimizer state across DP ranks, switching from all_reduce
    # to reduce_scatter for gradient communication.  Enabled by default for
    # long-horizon (non-deterministic) mode; disabled for deterministic mode
    # to preserve bitwise alignment with the ref.
    enable_zero = (
        config.world_size > 1
        and not deterministic
        and int(os.environ.get("ENABLE_ZERO_OPTIMIZER", "0"))  # default 0: DP=2 overhead > benefit; enable for larger DP
    )
    zero_opt: ZeroOptimizer | None = None
    if enable_zero:
        zero_opt = init_zero_optimizer(
            rank, config.world_size,
            fp32_master, exp_avgs, exp_avg_sqs, opt_state_steps,
            bf16_params, fp32_grad_bufs, device,
        )

    # ── Dataloader ─────────────────────────────────────────────────────
    dl = _build_dataloader(
        data_path=config.data_path,
        dp_rank=rank,
        world_size=config.world_size,
        micro_batch_size=config.micro_batch_size,
        seq_length=C.MAX_SEQ_LEN,
        seed=config.seed,
    )
    iter_dl = iter(dl) if hasattr(dl, '__iter__') else dl

    # ── Resume from checkpoint ──────────────────────────────────────────
    resume_step = 0
    if config.resume_from is not None:
        if enable_zero and zero_opt is not None:
            resume_step = zero_load_checkpoint(
                zero_opt, config.resume_from,
                fp32_master, exp_avgs, exp_avg_sqs, opt_state_steps,
                rank, init_weights_only=config.init_weights_only,
            )
            # All-gather the full BF16 params after the shard load.
            all_gather_bf16(zero_opt, bf16_params)
        else:
            resume_step = load_checkpoint(
                config.resume_from, fp32_master, exp_avgs, exp_avg_sqs, opt_state_steps,
                rank, init_weights_only=config.init_weights_only,
            )
        if rank == 0:
            import sys
            print(f"[debug] resume from {config.resume_from} at step {resume_step}", file=sys.stderr, flush=True)
        # Sync BF16 params from the loaded FP32 master (non-ZeRO path already does this;
        # ZeRO path does it inside zero_load_checkpoint + all_gather_bf16).
        if not enable_zero:
            _sync_bf16_from_fp32(bf16_params, fp32_master)
        # Rebuild optim_groups after loading (LR multipliers unchanged)
        optim_groups = _build_optimizer_groups(
            fp32_master, bf16_params, model,
            lr=opt_lr, width_mult=width_mult, weight_decay=opt_wd,
        )
        lr_mult_per_group = [g["lr"] / opt_lr for g in optim_groups]
        # Advance the dataloader to the current position.
        # The checkpoint was saved at abs_step = resume_step, meaning
        # resume_step * grad_accum_steps micro-batches have been consumed.
        consumed_batches = resume_step * config.grad_accum_steps
        if consumed_batches > 0:
            if rank == 0:
                print(f"[debug] advancing dataloader by {consumed_batches} batches", file=sys.stderr, flush=True)
            _advance_dataloader(iter_dl, consumed_batches, device)

    # ── Background dataloader prefetcher (started AFTER resume-skip) ────
    # Hides periodic shard refill latency by running next(dl) on a daemon
    # thread.  Must be started after _advance_dataloader (which reads from
    # iter_dl directly) to avoid a race on the shared iterator.
    dl_prefetcher: _BackgroundPrefetcher | None = None
    if int(os.environ.get("ENABLE_DL_PREFETCH", "1")):
        dl_prefetcher = _BackgroundPrefetcher(
            iter_dl, max_size=8,
            B=config.micro_batch_size, S=C.MAX_SEQ_LEN,
        )
        dl_prefetcher.start()
        if rank == 0:
            print("[debug] dataloader prefetcher started", file=sys.stderr, flush=True)

    # Helper: use the prefetcher if available, otherwise fall back to _next_batch.
    _get_batch_cpu = (dl_prefetcher.get if dl_prefetcher
                      else lambda: _next_batch(iter_dl, "cpu"))

    # ── Phase banner ─────────────────────────────────────────────────────
    # Emit a [PHASE] banner so the WSD-SFT gate's structural verdict can
    # segment the trajectory by phase and assert the switch properties.
    # The phase_name defaults to "stable" when no explicit phase is set.
    _phase_name = config.phase_name
    if _phase_name is None:
        if config.resume_from is not None and config.init_weights_only:
            _phase_name = "sft"
        elif config.resume_from is not None:
            _phase_name = "decay"
        else:
            _phase_name = "stable"
    # The verdict module's _PHASE_BANNER_RE expects the full structured
    # format: [PHASE] name=... start_step=... lr=... min_lr=... warmup=...
    # decay=... wsd_decay=... init_weights_only=... data_path=...
    print(
        f"[PHASE] name={_phase_name} "
        f"start_step={config.start_step} "
        f"lr={opt_lr} "
        f"min_lr={opt_min_lr} "
        f"warmup={opt_lr_warmup} "
        f"decay={opt_lr_decay} "
        f"wsd_decay={opt_lr_wsd_decay} "
        f"init_weights_only={1 if config.init_weights_only else 0} "
        f"data_path={config.data_path}",
        flush=True,
    )

    use_mtp = int(os.environ.get("MTP_NUM_LAYERS", "1")) > 0
    ce_w = float(os.environ.get("MTP_LOSS_WEIGHT", "0.3"))

    # ── Capture setup ──────────────────────────────────────────────────
    capture_records: dict | None = None
    capture_prefix = ""
    grad_prefix = ""
    if config.hash_capture_level > 0:
        capture_records = {}
        # NOTE: capture_prefix and grad_prefix are updated per-step below
        # (the step number changes each iteration).  The initial values are
        # placeholders; the actual step-specific prefix is set inside the
        # training loop before the forward/backward calls.
        capture_prefix = f"step_{config.start_step}.rank{rank}.mb0."
        grad_prefix = f"step_{config.start_step}."

    # ── MFU prep ───────────────────────────────────────────────────────
    tokens_per_step = config.global_batch_size * C.MAX_SEQ_LEN
    train_per_token = _compute_flops_per_step()
    flops_per_step = train_per_token * tokens_per_step
    peak_total = H100_BF16_PEAK_FLOPS * max(config.world_size, 1)

    # ── Training loop ──────────────────────────────────────────────────
    # Pre-build the data_ptr→index mapping for O(1) gradient buffer lookup.
    # The bf16_params list is invariant across the entire training loop, so
    # this mapping is computed once and reused on every backward call.
    dptr_idx = _build_dptr_idx(bf16_params)
    # CUDA events for non-blocking step timing (avoids torch.cuda.synchronize
    # in the hot path, which drains all pending CUDA work).
    step_start_event = torch.cuda.Event(enable_timing=True)
    step_end_event = torch.cuda.Event(enable_timing=True)

    # Pre-allocate step-level accumulators (1-element fp64 GPU tensors).
    # Re-allocating these every step (4 × torch.zeros) triggers 4 CUDA allocator
    # calls that can each cause an internal cudaStreamSynchronize.  Pre-allocate
    # once and use zero_() to reset.
    _local_lm_sum = torch.zeros(1, device=device, dtype=torch.float64)
    _local_lm_n = torch.zeros(1, device=device, dtype=torch.float64)
    _local_mtp_sum = torch.zeros(1, device=device, dtype=torch.float64)
    _local_mtp_n = torch.zeros(1, device=device, dtype=torch.float64)

    # Pre-allocate a 1-element fp32 tensor for the AdamW fused kernel's
    # max_exp_avg_sqs placeholder (amsgrad=False means the kernel never
    # accesses it).  Reusing across all optimizer groups and all steps
    # avoids a CUDA allocator call per group per step, which can trigger
    # an internal cudaStreamSynchronize (~1.77ms) when the allocator cache
    # is empty after the CUDA graph's private pool consumption.
    _opt_dummy_sq = torch.zeros(1, device=device, dtype=torch.float32)

    # ── CUDA graph capture for forward+backward of one microbatch ───────
    # Capturing the static forward+backward sequence as a CUDA graph
    # eliminates the per-kernel launch overhead (cudaLaunchKernel ~3814ms/step
    # from ~389,000 launches).  The graph is replayed for each microbatch.
    # The all-reduce, gradient norm, and dataloader remain outside the graph.
    cuda_graph = None
    _cached_lm_sum = None
    _cached_lm_n = None
    _cached_mtp_sum = None
    _cached_mtp_n = None
    _cached_mtp_in = None
    _cached_mtp_lab = None
    _cached_mtp_mask = None
    # Optimizer step CUDA graph (captures AdamW + BF16 sync).
    _opt_cuda_graph = None
    use_cuda_graph = (
        config.hash_capture_level == 0  # no hash capture during graph capture
        and int(os.environ.get("ENABLE_CUDA_GRAPH", "1"))  # default 1: CE optimization freed ~21 GiB, graph capture now viable
        and not enable_zero  # ZeRO-1 uses a sharded optimizer step, not compatible with the full optimizer graph
    )

    # ── Gradient bucketing for NCCL overlap ──────────────────────────────
    # Divide parameters into buckets so that gradient all-reduce of each
    # bucket can overlap with the backward of the next bucket's layers.
    # Only enabled for long-horizon (non-deterministic) mode.
    # The bucket count controls how many all-reduce chunks are created.
    # More buckets = finer-grained overlap but more NCCL overhead.
    enable_grad_bucketing = (
        config.world_size > 1
        and not deterministic
        and int(os.environ.get("ENABLE_GRAD_BUCKETING", "0"))  # default 0: DP=2 all-reduce is fast enough; bucketing adds overhead
    )
    # Disable gradient bucketing when ZeRO-1 is active (they overlap in
    # purpose — ZeRO-1's reduce_scatter is already a form of gradient
    # partitioning that replaces the all-reduce, and the two would conflict
    # on the gradient communication path).
    if enable_zero:
        enable_grad_bucketing = False
    num_grad_buckets = int(os.environ.get("NUM_GRAD_BUCKETS", "4"))
    # Pre-compute bucket boundaries: each bucket gets roughly equal param count.
    _bucket_boundaries: list[tuple[int, int]] = []
    if enable_grad_bucketing:
        n_params = len(bf16_params)
        bucket_size = (n_params + num_grad_buckets - 1) // num_grad_buckets
        for b in range(0, n_params, bucket_size):
            _bucket_boundaries.append((b, min(b + bucket_size, n_params)))
        # Separate CUDA stream for async gradient all-reduce.
        _grad_ar_stream = torch.cuda.Stream()
        if rank == 0:
            print(f"[debug] gradient bucketing enabled: {len(_bucket_boundaries)} buckets, "
                  f"{num_grad_buckets} total", file=sys.stderr, flush=True)

    if use_cuda_graph:
        # Check available memory before graph capture.  The CE optimization freed
        # ~21 GiB of fp32 logits, but the LM head matmul still needs ~4 GiB of
        # temporary workspace.  If the free memory is too low, skip graph capture.
        _free_before, _total_before = torch.cuda.mem_get_info(device)
        _free_gib = _free_before / (1024**3)
        if _free_gib < 8.0:
            if rank == 0:
                print(f"[debug] CUDA graph: insufficient free memory ({_free_gib:.1f} GiB < 8 GiB), skipping capture", flush=True)
            use_cuda_graph = False
        else:
            if rank == 0:
                print(f"[debug] CUDA graph: {_free_gib:.1f} GiB free, attempting capture", flush=True)
            # Evict CUDA allocator cache before warmup to avoid memory instability
        # between warmup and graph capture.  Doing this *after* warmup (between
        # del _fw_cache and capture) can cause captured tensor addresses to shift.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        # Pre-allocate input buffers so the CUDA graph captures stable addresses.
        B, S = config.micro_batch_size, C.MAX_SEQ_LEN
        input_ids_buf = torch.empty(B, S, dtype=torch.long, device=device)
        labels_buf = torch.empty(B, S, dtype=torch.long, device=device)
        loss_mask_buf = torch.empty(B, S, dtype=torch.float32, device=device)
        if use_mtp:
            mtp_in_buf = torch.empty(B, S, dtype=torch.long, device=device)
            mtp_lab_buf = torch.empty(B, S, dtype=torch.long, device=device)
            mtp_mask_buf = torch.empty(B, S, dtype=torch.float32, device=device)

        # Warmup: run one microbatch to trigger CUDA autotuning and allocate
        # all intermediate tensors (so the graph capture has stable addresses).
        # Use CPU tensors with non_blocking H2D to match the async hot path.
        warmup_ids, warmup_lab, warmup_mask = _get_batch_cpu()
        # The data is already on CPU, so no sync needed before the H2D copy.
        input_ids_buf.copy_(warmup_ids, non_blocking=True)
        labels_buf.copy_(warmup_lab, non_blocking=True)
        loss_mask_buf.copy_(warmup_mask, non_blocking=True)
        torch.cuda.synchronize()  # ensure H2D completed before warmup forward
        if use_mtp:
            mtp_in_buf = torch.empty(B, S, dtype=torch.long, device=device)
            mtp_lab_buf = torch.empty(B, S, dtype=torch.long, device=device)
            mtp_mask_buf = torch.empty(B, S, dtype=torch.float32, device=device)
            _warmup_mtp = _build_mtp_tensors(warmup_ids, warmup_lab, warmup_mask)
            mtp_in_buf.copy_(_warmup_mtp[0])
            mtp_lab_buf.copy_(_warmup_mtp[1])
            mtp_mask_buf.copy_(_warmup_mtp[2])
            _fw_cache = _forward_with_cache(
                model, input_ids_buf, labels_buf, loss_mask_buf,
                mtp_in_buf, mtp_lab_buf, mtp_mask_buf,
                rope_freqs, width_mult, mup_emb_scale,
                depth_scale_main, depth_scale_mtp,
                None, capture_prefix, deterministic,
            )
            _fw_lm_sum, _fw_lm_n = masked_cross_entropy(_fw_cache.main_logits, labels_buf, loss_mask_buf, deterministic=deterministic)
            _fw_mtp_sum, _fw_mtp_n = masked_cross_entropy(_fw_cache.mtp_logits, mtp_lab_buf, mtp_mask_buf, deterministic=deterministic)
            _static_backward(
                model, _fw_cache, rope_freqs, width_mult, mup_emb_scale,
                depth_scale_main, depth_scale_mtp,
                fp32_grad_bufs, bf16_params, dptr_idx,
                None, capture_prefix, ce_w, deterministic,
            )
        else:
            _fw_cache = _forward_with_cache(
                model, input_ids_buf, labels_buf, loss_mask_buf,
                None, None, None,
                rope_freqs, width_mult, mup_emb_scale,
                depth_scale_main, depth_scale_mtp,
                None, capture_prefix, deterministic,
            )
            _fw_lm_sum, _fw_lm_n = masked_cross_entropy(_fw_cache.main_logits, labels_buf, loss_mask_buf, deterministic=deterministic)
            _static_backward(
                model, _fw_cache, rope_freqs, width_mult, mup_emb_scale,
                depth_scale_main, depth_scale_mtp,
                fp32_grad_bufs, bf16_params, dptr_idx,
                None, capture_prefix, 0.0, deterministic,
            )

        # Free the warmup cache before graph capture to avoid OOM.
        # The warmup forward+backward allocates ~24 GB of activation memory
        # in the ForwardCache.  The CUDA graph capture needs additional
        # memory for the captured operations, so we must free the cache first.
        # (empty_cache/gc.collect already done before warmup at lines 1691-1693.)
        del _fw_cache

        # Free the warmup's cached CUDA memory before graph capture.
        # The warmup allocates tensors that stay in the CUDA allocator's cache
        # even after del _fw_cache.  Without this, the graph capture's private
        # pools may OOM (total memory + graph private pools > 80 GB).
        # Use try-except to handle the PyTorch captures_underway.empty()
        # assertion failure that can occur in certain PyTorch versions.
        torch.cuda.synchronize()
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass  # PyTorch internal assertion failure — ignore; the cache
                  # will be freed naturally by the next allocation.
        gc.collect()

        # Zero the grad bufs after warmup (the warmup accumulated gradients).
        torch._foreach_zero_(fp32_grad_bufs)
        # CRITICAL: synchronize before graph capture.  `torch._foreach_zero_` is
        # an async CUDA kernel launch.  Without sync, the pending kernel may be
        # captured as part of the CUDA graph, corrupting the graph (the zeroing
        # would happen on every replay instead of once per step).
        torch.cuda.synchronize()

        # Debug: print free memory after warmup cleanup.
        _free_after, _total_after = torch.cuda.mem_get_info(device)
        _free_gib_after = _free_after / (1024**3)
        if rank == 0:
            print(f"[debug] CUDA graph: {_free_gib_after:.1f} GiB free after warmup cleanup", flush=True)

        # Capture the CUDA graph for one microbatch.
        # The graph records the forward+backward CUDA operations.  During replay,
        # the same tensor addresses are used, so the cache and lm_* tensors from
        # the capture phase are updated in place.
        try:
            cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cuda_graph):
                # The input buffers are at fixed addresses (pre-allocated above).
                if use_mtp:
                    _capture_cache = _forward_with_cache(
                        model, input_ids_buf, labels_buf, loss_mask_buf,
                        mtp_in_buf, mtp_lab_buf, mtp_mask_buf,
                        rope_freqs, width_mult, mup_emb_scale,
                        depth_scale_main, depth_scale_mtp,
                        None, capture_prefix, deterministic,
                    )
                    _capture_lm_sum, _capture_lm_n = masked_cross_entropy(
                        _capture_cache.main_logits, labels_buf, loss_mask_buf, deterministic=deterministic)
                    _capture_mtp_sum, _capture_mtp_n = masked_cross_entropy(
                        _capture_cache.mtp_logits, mtp_lab_buf, mtp_mask_buf, deterministic=deterministic)
                    _static_backward(
                        model, _capture_cache, rope_freqs, width_mult, mup_emb_scale,
                        depth_scale_main, depth_scale_mtp,
                        fp32_grad_bufs, bf16_params, dptr_idx,
                        None, capture_prefix, ce_w, deterministic,
                    )
                else:
                    _capture_cache = _forward_with_cache(
                        model, input_ids_buf, labels_buf, loss_mask_buf,
                        None, None, None,
                        rope_freqs, width_mult, mup_emb_scale,
                        depth_scale_main, depth_scale_mtp,
                        None, capture_prefix, deterministic,
                    )
                    _capture_lm_sum, _capture_lm_n = masked_cross_entropy(
                        _capture_cache.main_logits, labels_buf, loss_mask_buf, deterministic=deterministic)
                    _static_backward(
                        model, _capture_cache, rope_freqs, width_mult, mup_emb_scale,
                        depth_scale_main, depth_scale_mtp,
                        fp32_grad_bufs, bf16_params, dptr_idx,
                        None, capture_prefix, 0.0, deterministic,
                    )

            # Save references to the captured tensors so we can read them after replay.
            _cached_lm_sum = _capture_lm_sum
            _cached_lm_n = _capture_lm_n
            if use_mtp:
                _cached_mtp_sum = _capture_mtp_sum
                _cached_mtp_n = _capture_mtp_n

            # Zero the grad bufs again after capture (the capture also accumulated gradients).
            torch._foreach_zero_(fp32_grad_bufs)

            if rank == 0:
                _free_after_cap, _ = torch.cuda.mem_get_info(device)
                print(f"[debug] CUDA graph captured successfully ({_free_gib:.1f} GiB free before, {_free_gib_after:.1f} GiB after warmup, {_free_after_cap/(1024**3):.1f} GiB after capture)", file=sys.stderr, flush=True)

            # NOTE: Optimizer step CUDA graph capture is intentionally skipped.
            # The forward+backward graph's private pools (~20.96 GiB) consume
            # most of the 79.32 GiB HBM, leaving only ~25 MiB free.  The
            # optimizer graph capture would need ~6 GB of temporary state
            # (_saved_fp32, etc.) for the save/restore cycle, which would OOM
            # and corrupt NCCL state.  The optimizer graph only saves ~15.8ms/step
            # (vs ~3814ms from the forward+backward graph), so the risk is not
            # worth the benefit.  The imperative _adamw_step + _sync_bf16_from_fp32
            # path is used instead.

        except Exception as e:
            if rank == 0:
                _free_on_fail, _ = torch.cuda.mem_get_info(device)
                print(f"[debug] CUDA graph capture failed ({_free_on_fail/(1024**3):.1f} GiB free), falling back to eager: {e}",
                      file=sys.stderr, flush=True)
                print(f"[debug] CUDA graph capture failed, falling back to eager: {e}", flush=True)
            # Free the partially-captured graph reference.  The private pools
            # are freed when the CUDAGraph object is garbage-collected (__del__).
            # Do NOT call gc.collect() or torch.cuda.empty_cache() here — they
            # can trigger the captures_underway.empty() assertion failure in the
            # CUDA allocator when the allocator state is inconsistent after a
            # failed cross-rank graph capture.
            cuda_graph = None
            use_cuda_graph = False

    for step in range(config.num_steps):
        # Update capture prefix per-step to match the ref's harness_dp format
        # (step_{step}.rank{rank}.mb0. for forward, step_{step}. for gradient).
        step_prefix = f"step_{config.start_step + step}."
        step_grad_prefix = f"step_{config.start_step + step}."
        if config.hash_capture_level > 0:
            capture_prefix = f"{step_prefix}rank{rank}.mb0."
            grad_prefix = step_grad_prefix

        step_start_event.record()

        # Zero gradients (fused multi-tensor for fewer kernel launches)
        torch._foreach_zero_(fp32_grad_bufs)

        # Reset step-level accumulators (zero_() avoids 4 CUDA allocator calls).
        _local_lm_sum.zero_()
        _local_lm_n.zero_()
        _local_mtp_sum.zero_()
        _local_mtp_n.zero_()

        # Grad accumulation loop
        if use_cuda_graph and cuda_graph is not None:
            # One-time debug: confirm CUDA graph is active.
            if rank == 0 and step == 0:
                print(f"[debug] CUDA graph replay active ({config.grad_accum_steps} microbatch replays/step)", flush=True)
            # CUDA graph replay path: replay the captured graph for each microbatch.
            # The graph records the forward+backward.  Input data is copied into
            # pre-allocated buffers (stable addresses) before each replay.
            #
            # Async H2D double buffering: the next microbatch's H2D is started
            # during the current microbatch's graph replay, so the H2D transfer
            # overlaps with GPU compute instead of blocking the CPU.
            # Pre-fetch the first microbatch as CPU tensors, then start H2D.
            _prefetch_ids, _prefetch_lab, _prefetch_mask = _get_batch_cpu()
            for _mb in range(config.grad_accum_steps):
                # Start H2D for this microbatch (async, non-blocking for CPU).
                input_ids_buf.copy_(_prefetch_ids, non_blocking=True)
                labels_buf.copy_(_prefetch_lab, non_blocking=True)
                loss_mask_buf.copy_(_prefetch_mask, non_blocking=True)
                if use_mtp:
                    _prefetch_mtp = _build_mtp_tensors(_prefetch_ids, _prefetch_lab, _prefetch_mask)
                    mtp_in_buf.copy_(_prefetch_mtp[0], non_blocking=True)
                    mtp_lab_buf.copy_(_prefetch_mtp[1], non_blocking=True)
                    mtp_mask_buf.copy_(_prefetch_mtp[2], non_blocking=True)

                # Replay the captured graph — the forward+backward runs at captured
                # tensor addresses.  The _cached_lm_* tensors are updated in place.
                cuda_graph.replay()

                # While the GPU runs the current microbatch, prefetch the next
                # microbatch's data on the CPU (no H2D yet).  The next iteration's
                # copy_(..., non_blocking=True) will start the H2D that overlaps
                # with the current graph replay's tail on the GPU.
                if _mb < config.grad_accum_steps - 1:
                    _prefetch_ids, _prefetch_lab, _prefetch_mask = _get_batch_cpu()

                # Read the updated loss values from the captured tensor addresses.
                _local_lm_sum += _cached_lm_sum.detach().double()
                _local_lm_n += _cached_lm_n.detach().double()
                if use_mtp:
                    _local_mtp_sum += _cached_mtp_sum.detach().double()
                    _local_mtp_n += _cached_mtp_n.detach().double()
        else:
            # Eager path (no CUDA graph capture) — prefetch H2D with non_blocking.
            # Pre-allocate input buffers so we can use non_blocking copy_ for H2D.
            B, S = config.micro_batch_size, C.MAX_SEQ_LEN
            _eager_ids = torch.empty(B, S, dtype=torch.long, device=device)
            _eager_lab = torch.empty(B, S, dtype=torch.long, device=device)
            _eager_mask = torch.empty(B, S, dtype=torch.float32, device=device)
            if use_mtp:
                _eager_mtp_in = torch.empty(B, S, dtype=torch.long, device=device)
                _eager_mtp_lab = torch.empty(B, S, dtype=torch.long, device=device)
                _eager_mtp_mask = torch.empty(B, S, dtype=torch.float32, device=device)
            _prefetch_ids, _prefetch_lab, _prefetch_mask = _get_batch_cpu()
            for _mb in range(config.grad_accum_steps):
                # Start H2D for this microbatch (async, non-blocking for CPU).
                _eager_ids.copy_(_prefetch_ids, non_blocking=True)
                _eager_lab.copy_(_prefetch_lab, non_blocking=True)
                _eager_mask.copy_(_prefetch_mask, non_blocking=True)

                if use_mtp:
                    _prefetch_mtp = _build_mtp_tensors(_prefetch_ids, _prefetch_lab, _prefetch_mask)
                    _eager_mtp_in.copy_(_prefetch_mtp[0], non_blocking=True)
                    _eager_mtp_lab.copy_(_prefetch_mtp[1], non_blocking=True)
                    _eager_mtp_mask.copy_(_prefetch_mtp[2], non_blocking=True)

                    cache = _forward_with_cache(
                        model, _eager_ids, _eager_lab, _eager_mask,
                        _eager_mtp_in, _eager_mtp_lab, _eager_mtp_mask,
                        rope_freqs, width_mult, mup_emb_scale,
                        depth_scale_main, depth_scale_mtp,
                        None,  # skip fwd hash (too slow for DP multi-GPU),
                        capture_prefix, deterministic,
                    )
                    lm_sum, lm_n = masked_cross_entropy(cache.main_logits, _eager_lab, _eager_mask, deterministic=deterministic)
                    mtp_sum_v, mtp_n_v = masked_cross_entropy(cache.mtp_logits, _eager_mtp_lab, _eager_mtp_mask, deterministic=deterministic)
                    _local_mtp_sum += mtp_sum_v.detach().double()
                    _local_mtp_n += mtp_n_v.detach().double()
                else:
                    cache = _forward_with_cache(
                        model, _eager_ids, _eager_lab, _eager_mask,
                        None, None, None,
                        rope_freqs, width_mult, mup_emb_scale,
                        depth_scale_main, depth_scale_mtp,
                        None,  # skip fwd hash (too slow for DP multi-GPU),
                        capture_prefix, deterministic,
                    )
                    lm_sum, lm_n = masked_cross_entropy(cache.main_logits, _eager_lab, _eager_mask, deterministic=deterministic)

                # ── Static backward pass ──────────────────────────────────
                _static_backward(
                    model, cache, rope_freqs, width_mult, mup_emb_scale,
                    depth_scale_main, depth_scale_mtp,
                    fp32_grad_bufs, bf16_params, dptr_idx,
                    capture_records, capture_prefix,
                    ce_w if use_mtp else 0.0, deterministic,
                )

                # Free the ForwardCache immediately to avoid accumulating
                # ~24 GB of activation memory across microbatches (~96 GB for
                # grad_accum=10 would OOM 80 GB H100).
                del cache

                _local_lm_sum += lm_sum.detach().double()
                _local_lm_n += lm_n.detach().double()

                # While the GPU runs the current microbatch, prefetch the next
                # microbatch's data on the CPU (no H2D yet).  The next iteration's
                # copy_(..., non_blocking=True) will start the H2D that overlaps
                # with the current backward's tail on the GPU.
                if _mb < config.grad_accum_steps - 1:
                    _prefetch_ids, _prefetch_lab, _prefetch_mask = _get_batch_cpu()

            # ── Post-accumulation: all-reduce ──────────────────────────────
        # NOTE: no torch.cuda.synchronize() needed here — the backward pass
        # writes fp32_grad_bufs on the default stream, and the all-reduce
        # (dist.all_reduce) also uses the default stream.  Default-stream
        # serialization guarantees all backward kernels complete before the
        # all-reduce begins.  A full device synchronize is redundant and
        # wastes ~500ms/step of GPU idle time (the profiler sees it as
        # cudaStreamSynchronize).
        if config.world_size > 1:
            if use_mtp:
                # Pre-allocated stats buffer: use torch.cat with out= to avoid
                # the per-step CUDA allocator call for the 32-byte tensor.
                torch.cat([_local_lm_sum, _local_lm_n, _local_mtp_sum, _local_mtp_n],
                          out=_stats_buf)
                dist.all_reduce(_stats_buf, op=dist.ReduceOp.SUM)
                _local_lm_sum.copy_(_stats_buf[0:1])
                _local_lm_n.copy_(_stats_buf[1:2])
                _local_mtp_sum.copy_(_stats_buf[2:3])
                _local_mtp_n.copy_(_stats_buf[3:4])
            else:
                # Only first 2 elements used (lm_sum, lm_n).
                torch.cat([_local_lm_sum, _local_lm_n],
                          out=_stats_buf[:2])
                dist.all_reduce(_stats_buf[:2], op=dist.ReduceOp.SUM)
                _local_lm_sum.copy_(_stats_buf[0:1])
                _local_lm_n.copy_(_stats_buf[1:2])

        # Keep as GPU tensors — defer .item() to after timing to avoid
        # CUDA stream sync inside the timed region.
        reported_lm_tensor = _local_lm_sum / _local_lm_n.clamp(min=1.0)
        reported_mtp_tensor = (_local_mtp_sum / _local_mtp_n.clamp(min=1.0)) if use_mtp else None

        if config.world_size > 1:
            # Initialize flat gradient reference for efficient norm computation.
            # The flat all-reduce path sets this to the scaled all-reduced flat
            # tensor, allowing torch.linalg.vector_norm to bypass the ~100ms
            # multi-tensor _foreach_norm kernel (157 buffers).
            _flat_grads = None
            if enable_zero and zero_opt is not None:
                # ── ZeRO-1: reduce_scatter gradients ──────────────────────────────
                # Each rank receives only its shard's portion of the summed gradient.
                # The shard's gradient buffers are updated in-place.
                if config.hash_capture_level > 0 and config.persistent:
                    _capture_all_gradients(capture_records, bf16_params, fp32_grad_bufs, model, grad_prefix, suffix="preallreduce")
                norm_factor = (1.0 / _local_lm_n.clamp(min=1.0))
                reduce_scatter_grads(zero_opt, fp32_grad_bufs, bf16_params, norm_factor)
            elif enable_grad_bucketing:
                # ── Gradient bucketed all-reduce ──────────────────────────────
                # Split the flattened gradient into per-bucket chunks and all-reduce
                # each on a separate stream.  The all-reduce of earlier buckets runs
                # concurrently with the scaling+copy of the next bucket, reducing
                # the critical-path time versus a single flat all-reduce followed by
                # a synchronous unflatten.
                if config.hash_capture_level > 0 and config.persistent:
                    _capture_all_gradients(capture_records, bf16_params, fp32_grad_bufs, model, grad_prefix, suffix="preallreduce")
                # Flatten full gradient for bucketing (same as single all-reduce).
                flat = torch._utils._flatten_dense_tensors(fp32_grad_bufs)
                norm_factor = (1.0 / _local_lm_n.clamp(min=1.0))
                # Split the flat tensor into buckets and all-reduce each on a separate stream.
                # The flat tensor is contiguous, so the bucketed slices are views.
                bucket_results = []
                with torch.cuda.stream(_grad_ar_stream):
                    for b_start, b_end in _bucket_boundaries:
                        # Map bucket parameter indices to the flat tensor offsets.
                        b_start_off = int(sum(p.numel() for p in bf16_params[:b_start]))
                        b_end_off = int(sum(p.numel() for p in bf16_params[:b_end]))
                        bucket_flat = flat[b_start_off:b_end_off].contiguous()
                        dist.all_reduce(bucket_flat, op=dist.ReduceOp.SUM)
                        bucket_flat.mul_(norm_factor)
                        bucket_results.append((b_start_off, b_end_off, bucket_flat))
                    # Record event on the gradient stream after all all-reduces are
                    # submitted.  This avoids the CPU-blocking torch.cuda.synchronize()
                    # below — the default stream waits for the gradient stream via the
                    # event, so the CPU can proceed to queue the next operations
                    # (gradient norm, optimizer step) without blocking.
                    _grad_ar_event = torch.cuda.Event()
                    _grad_ar_stream.record_event(_grad_ar_event)
                # Wait on default stream (non-blocking for CPU).  The default stream's
                # subsequent copy operations will wait for the gradient stream's
                # all-reduce to complete, but the CPU can continue queuing work.
                torch.cuda.current_stream().wait_event(_grad_ar_event)
                # Copy the bucketed results back into the flat tensor.
                for b_start_off, b_end_off, bucket_flat in bucket_results:
                    flat[b_start_off:b_end_off].copy_(bucket_flat)
                # Fused unflatten: single _foreach_copy_ for all 157 params
                # instead of 157 separate cudaMemcpyAsync calls.
                torch._foreach_copy_(
                    fp32_grad_bufs,
                    torch._utils._unflatten_dense_tensors(flat, fp32_grad_bufs),
                )
            else:
                # ── Flatten + single all-reduce (matching ref's reduce_grads) ──
                # The ref's harness_dp.reduce_grads flattens all grad buffers into
                # one contiguous tensor, does a single all-reduce, scales by
                # norm_factor, then copies back.  Individual all-reduces per buffer
                # can produce 1-ULP differences in the summed gradients (NCCL may
                # use different algorithms for different tensor sizes), which
                # accumulate into a ~1e-6 loss drift from step 2 onward.
                if config.hash_capture_level > 0 and config.persistent:
                    _capture_all_gradients(capture_records, bf16_params, fp32_grad_bufs, model, grad_prefix, suffix="preallreduce")
                flat = torch._utils._flatten_dense_tensors(fp32_grad_bufs)
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
                norm_factor = (1.0 / _local_lm_n.clamp(min=1.0))
                flat.mul_(norm_factor)
                # Fused unflatten: single _foreach_copy_ for all 157 params
                # instead of 157 separate cudaMemcpyAsync calls.
                torch._foreach_copy_(
                    fp32_grad_bufs,
                    torch._utils._unflatten_dense_tensors(flat, fp32_grad_bufs),
                )
            # Save the flat (scaled all-reduced) gradient tensor for efficient
            # gradient norm computation.  torch.linalg.vector_norm on the
            # contiguous flat tensor is ~100x faster than torch._foreach_norm
            # on 157 per-buffer tensors (0.75ms vs 100ms) because the multi-
            # tensor kernel's loop overhead is significant for 157 buffers.
            _flat_grads = flat
        else:
            # Capture pre-allreduce gradients (single-GPU, no all-reduce needed)
            if config.hash_capture_level > 0 and config.persistent:
                _capture_all_gradients(capture_records, bf16_params, fp32_grad_bufs, model, grad_prefix, suffix="preallreduce")
            norm_factor = (1.0 / _local_lm_n.clamp(min=1.0))
            # Use _foreach_mul_ for a single fused kernel launch instead of
            # 157 separate per-buffer mul_ calls.
            torch._foreach_mul_(fp32_grad_bufs, norm_factor)

        # ── Capture gradients for this step (persistent mode) ─────────────
        # The ref's harness_dp captures gradients after each step (post-scaling).
        # We must do the same so the hash dump contains per-step gradient records.
        if config.hash_capture_level > 0 and config.persistent:
            _capture_all_gradients(capture_records, bf16_params, fp32_grad_bufs, model, grad_prefix)

        # ── Capture mode: dump and exit (non-persistent) ────────────────
        # Skip the non-persistent exit when teardown_exit=False (multi-phase
        # resume gate needs the process to survive across phases).
        if config.hash_capture_level > 0 and not config.persistent and config.teardown_exit:
            _capture_all_gradients(capture_records, bf16_params, fp32_grad_bufs, model, grad_prefix)
            _write_capture_output(config, capture_records, rank)
            _ordered_teardown(teardown_exit=True)
            return  # Not reached (SystemExit raised in _ordered_teardown)

        # ── Compute gradient norm (also clips via _foreach_norm) ─────────
        if enable_zero and zero_opt is not None:
            # ZeRO-1 aware gradient norm: each rank computes its shard's L2 norm,
            # then all-reduces the squared norms to get the global total.
            total_norm = compute_zero_grad_norm(zero_opt, fp32_grad_bufs, opt_clip_grad)
            if total_norm > opt_clip_grad:
                scale = opt_clip_grad / total_norm
                for buf in zero_opt.my_grad_bufs:
                    buf.mul_(scale)
            grad_norm_val = total_norm
        else:
            # Set .grad on fp32_master (matching ref's pattern exactly) so that
            # the _fused_adamw_ kernel reads the same .grad fields the ref does.
            for p_fp32, buf in zip(fp32_master, fp32_grad_bufs):
                p_fp32.grad = buf
            # Use _foreach_norm for batched per-tensor L2 norm computation, or
            # torch.linalg.vector_norm on the flat tensor when available (the
            # flat all-reduce path saves the scaled all-reduced gradient tensor,
            # which is contiguous and ~100x faster to norm than 157 buffers).
            if _flat_grads is not None:
                total_norm = torch.linalg.vector_norm(_flat_grads)
            else:
                norms = torch._foreach_norm(fp32_grad_bufs)
                total_norm = torch.linalg.vector_norm(torch.stack(norms))
            if total_norm > opt_clip_grad:
                torch._foreach_mul_(fp32_grad_bufs, opt_clip_grad / total_norm)
            grad_norm_val = total_norm

        # ── Compute LR for this step ──────────────────────────────────
        # ref uses ``step + 1`` (1-indexed) for the LR schedule
        current_lr = _compute_lr(
            step + 1, opt_lr, opt_min_lr,
            opt_lr_warmup, opt_lr_decay, opt_lr_wsd_decay,
        )

        # ── AdamW optimizer step (CUDA graph or imperative) ────────────
        # Update per-group LR to match ref's ``pg["lr"] = lr * mult``
        for g, mult in zip(optim_groups, lr_mult_per_group):
            g["lr"] = current_lr * mult
        if enable_zero and zero_opt is not None:
            # ZeRO-1 optimizer step: only the rank's shard is updated.
            # Set .grad on the shard's FP32 master tensors.
            set_grad_from_shard(zero_opt, fp32_master, fp32_grad_bufs)
            # Run the sharded AdamW step + sharded BF16 sync.
            zero_optimizer_step(
                zero_opt, optim_groups,
                beta1=opt_beta1, beta2=opt_beta2, eps=opt_eps,
                opt_clip_grad=opt_clip_grad, total_norm=grad_norm_val,
            )
            # All-gather the full BF16 params so every rank has a complete copy
            # for the forward pass of the next step.
            all_gather_bf16(zero_opt, bf16_params)
        elif _opt_cuda_graph is not None:
            # Set .grad on fp32_master (must match the graph capture's
            # tensor addresses — fp32_grad_bufs are never reallocated).
            for p_fp32, buf in zip(fp32_master, fp32_grad_bufs):
                p_fp32.grad = buf
            # Replay the captured optimizer graph (AdamW + BF16 sync).
            # The graph is a single replay that replaces ~472 kernel
            # launches worth of Python launch overhead.
            _opt_cuda_graph.replay()
        else:
            _adamw_step(
                optim_groups, fp32_master,
                exp_avgs, exp_avg_sqs, opt_state_steps,
                beta1=opt_beta1, beta2=opt_beta2, eps=opt_eps,
                dummy_sq=_opt_dummy_sq,
            )
            # ── Sync BF16 params from FP32 master ─────────────────────────
            _sync_bf16_from_fp32(bf16_params, fp32_master,
                                 flat_fp32_buf=_flat_fp32_buf,
                                 flat_bf16_buf=_flat_bf16_buf)

        # ── Per-step logging ──────────────────────────────────────────
        step_end_event.record()
        step_end_event.synchronize()
        step_time = step_start_event.elapsed_time(step_end_event) / 1000.0
        # .item() calls are safe here — step_end_event.synchronize() already
        # synced the stream, so no additional CUDA sync overhead.
        reported_lm = reported_lm_tensor.item()
        reported_mtp = reported_mtp_tensor.item() if use_mtp else 0.0
        total = reported_lm + ce_w * reported_mtp if use_mtp else reported_lm
        mfu = flops_per_step / (step_time * peak_total) * 100.0 if step_time > 0 else 0.0
        gn = grad_norm_val.item()

        if rank == 0:
            loss_line = (
                f"[{loss_tag}] step={config.start_step + step + 1} "
                f"global_loss={total:.9e} "
                f"grad_norm={gn:.9e} "
                f"time_s={step_time:.6f} "
                f"mfu_e2e_standard={mfu:.6f}"
            )
            print(loss_line, flush=True)

        # ── Save checkpoint ────────────────────────────────────────────────
        # Save at the end of each step if save_path is configured.
        # The absolute step number is 1-indexed (config.start_step + step + 1).
        abs_step = config.start_step + step + 1
        if config.save_path is not None:
            # Check if we should save this step (versioned ckpt when save_interval>0,
            # or always save the final step, or save on the last step of the loop).
            should_save = False
            if config.save_interval > 0:
                # Save periodically at save_interval boundaries
                if abs_step % config.save_interval == 0:
                    should_save = True
            # Always save the last step of the loop
            if step == config.num_steps - 1:
                should_save = True
            if should_save:
                if enable_zero and zero_opt is not None:
                    zero_save_checkpoint(
                        zero_opt, config.save_path, abs_step, rank,
                    )
                else:
                    save_checkpoint(
                        config.save_path, abs_step,
                        fp32_master, exp_avgs, exp_avg_sqs, opt_state_steps,
                        rank,
                    )
                if rank == 0:
                    import sys
                    print(f"[debug] checkpoint saved at step {abs_step} to {config.save_path}/step_{abs_step}", file=sys.stderr, flush=True)

    # ── Persistent capture: write hash dump before teardown ──────────
    if config.hash_capture_level > 0 and config.persistent and capture_records is not None:
        _capture_all_gradients(capture_records, bf16_params, fp32_grad_bufs, model, grad_prefix)
        _write_capture_output(config, capture_records, rank)

    # ── Stop the dataloader prefetcher ────────────────────────────────
    if dl_prefetcher is not None:
        dl_prefetcher.stop()

    # ── Ordered teardown ──────────────────────────────────────────────
    _ordered_teardown(teardown_exit=config.teardown_exit)


# ── Gradient capture helpers ────────────────────────────────────────────────


def _capture_all_gradients(capture_records, bf16_params, fp32_grad_bufs, model,
                           prefix: str = "", suffix: str = "postallreduce"):
    """Capture all parameter gradients as hash records (offloaded to thread pool).

    Uses ``evals.capture_offload.hash_batch_sync`` to run D2H + blake2b on the
    shared thread pool, blocking until all hashes are done.
    """
    fqn_map = _build_fqn_map(model)
    items = []
    for buf, p in zip(fp32_grad_bufs, bf16_params):
        fqn = fqn_map.get(p.data_ptr(), None)
        if fqn is not None:
            key = f"{prefix}rank{os.environ.get('RANK', '0')}.grad.{fqn}.{suffix}"
            items.append((key, buf))
    if items:
        results = hash_batch_sync(items)
        for key, record in results:
            capture_records[key] = record


def _build_fqn_map(model: ModelParameters) -> dict:
    """Build a mapping from tensor data_ptr to FQN string."""
    fqn_map = {}
    fqn_map[model.tok_embeddings_weight.data_ptr()] = "tok_embeddings.weight"
    fqn_map[model.final_norm_weight.data_ptr()] = "norm.weight"
    # The ref keeps tok_embeddings.weight and output.weight as SEPARATE tensors.
    # Register both under their own data_ptr.
    fqn_map[model.output_weight.data_ptr()] = "output.weight"
    for layer in model.layers:
        idx = layer.index
        fqn_map[layer.input_norm_weight.data_ptr()] = f"layers.{idx}.attention_norm.weight"
        fqn_map[layer.qkv_weight.data_ptr()] = f"layers.{idx}.wqkv.weight"
        fqn_map[layer.attention_proj_weight.data_ptr()] = f"layers.{idx}.wo.weight"
        fqn_map[layer.pre_mlp_norm_weight.data_ptr()] = f"layers.{idx}.ffn_norm.weight"
        fqn_map[layer.mlp_fc1_weight.data_ptr()] = f"layers.{idx}.wfc1.weight"
        fqn_map[layer.mlp_fc2_weight.data_ptr()] = f"layers.{idx}.w2.weight"
    if model.mtp is not None:
        fqn_map[model.mtp.emb_input_norm_weight.data_ptr()] = "mtp.emb_input_layernorm.weight"
        fqn_map[model.mtp.hidden_input_norm_weight.data_ptr()] = "mtp.hidden_input_layernorm.weight"
        fqn_map[model.mtp.eagle_fc_weight.data_ptr()] = "mtp.eagle_fc.weight"
        fqn_map[model.mtp.layer.input_norm_weight.data_ptr()] = "mtp.layer.attention_norm.weight"
        fqn_map[model.mtp.layer.qkv_weight.data_ptr()] = "mtp.layer.wqkv.weight"
        fqn_map[model.mtp.layer.attention_proj_weight.data_ptr()] = "mtp.layer.wo.weight"
        fqn_map[model.mtp.layer.pre_mlp_norm_weight.data_ptr()] = "mtp.layer.ffn_norm.weight"
        fqn_map[model.mtp.layer.mlp_fc1_weight.data_ptr()] = "mtp.layer.wfc1.weight"
        fqn_map[model.mtp.layer.mlp_fc2_weight.data_ptr()] = "mtp.layer.w2.weight"
        fqn_map[model.mtp.final_norm_weight.data_ptr()] = "mtp.final_layernorm.weight"
    return fqn_map


def _collect_bf16_params(model: ModelParameters) -> list[torch.Tensor]:
    """Collect all bf16 weight tensors from the model.

    The ref's model keeps ``tok_embeddings.weight`` and ``output.weight`` as
    SEPARATE tensors with independent optimizer state (m, v).  Both are
    included in the parameter list.  The order must match the ref's
    ``model.parameters()`` order (tok_embeddings → layers → norm → output
    → mtp) so that ``torch.linalg.vector_norm`` on the stacked per-tensor
    norms produces a bitwise-identical total gradient norm.
    """
    params = [model.tok_embeddings_weight]
    for layer in model.layers:
        params.extend([
            layer.input_norm_weight,
            layer.qkv_weight,
            layer.attention_proj_weight,
            layer.pre_mlp_norm_weight,
            layer.mlp_fc1_weight,
            layer.mlp_fc2_weight,
        ])
    params.append(model.final_norm_weight)
    params.append(model.output_weight)
    if model.mtp is not None:
        params.extend([
            model.mtp.emb_input_norm_weight,
            model.mtp.hidden_input_norm_weight,
            model.mtp.eagle_fc_weight,
            model.mtp.layer.input_norm_weight,
            model.mtp.layer.qkv_weight,
            model.mtp.layer.attention_proj_weight,
            model.mtp.layer.pre_mlp_norm_weight,
            model.mtp.layer.mlp_fc1_weight,
            model.mtp.layer.mlp_fc2_weight,
            model.mtp.final_norm_weight,
        ])
    return params


# ── Capture output ────────────────────────────────────────────────────────────


def _write_capture_output(config: TrainLoopConfig, records: dict | None, rank: int) -> None:
    """Write the capture hash records to the output file.

    For single-rank runs (world_size=1), writes directly to ``hash_output``.
    For multi-rank, writes to ``hash_output.rank<r>`` shards that the
    dispatcher's ``load_merged_capture`` merges.
    """
    if config.hash_output and records is not None:
        p = Path(config.hash_output)
        p.parent.mkdir(parents=True, exist_ok=True)
        if config.world_size > 1:
            rank_path = p.with_name(p.name + f".rank{rank}")
        else:
            rank_path = p
        with open(rank_path, "w") as f:
            json.dump(records, f, indent=2)


# ── Process group init / teardown ────────────────────────────────────────────


def _init_process_group() -> int:
    """Initialize the torch distributed process group.

    Sets the CUDA device before init so NCCL knows the correct GPU.
    Passes device_id to avoid the "devices unknown" warning.
    Returns the global rank.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    rank = dist.get_rank()
    return rank


def _ordered_teardown(teardown_exit: bool = True) -> None:
    """Ordered teardown: drain GPU, barrier, destroy PG, empty cache, then optionally exit.

    When ``teardown_exit=False``, the function returns normally after cleanup
    (used by the resume gate which runs multiple phases in the same process).
    When ``teardown_exit=True`` (default), it calls ``os._exit(0)`` to bypass the
    Python interpreter's finalizer, which would otherwise crash with SIGABRT
    when the NCCL PG destructor races Py_Finalize.  This is safe because all
    data has been flushed to disk before this point.
    """
    import gc
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    for _ in range(3):
        gc.collect()
    try:
        if dist.is_available() and dist.is_initialized():
            dist.barrier(device_ids=[local_rank])
            dist.destroy_process_group()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
    if os.environ.get("RANK", "0") == "0":
        print("ALL DONE", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    if teardown_exit:
        # Bypass the Python finalizer to avoid the NCCL PG destructor crash.
        os._exit(0)