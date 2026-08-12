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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

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
    """Record a forward activation hash."""
    key = f"{prefix}fwd.{fqn}#{call_idx}"
    records[key] = _hash_tensor(tensor)


def _capture_gradient(records: dict, prefix: str, fqn: str,
                      tensor: torch.Tensor) -> None:
    """Record a parameter gradient hash."""
    key = f"rank{os.environ.get('RANK', '0')}.grad.{fqn}.postallreduce"
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


def _next_batch(dl, device):
    """Get the next batch from the dataloader."""
    data = next(dl)
    return (
        data["tokens"].to(device),
        data["labels"].to(device),
        data["loss_mask"].float().to(device),
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
    normed: torch.Tensor              # after attention_norm
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_rot: torch.Tensor
    k_rot: torch.Tensor
    attn_flat: torch.Tensor           # attention output reshaped [B, S, H*D]
    attn_out: torch.Tensor            # after wo projection
    hidden_after_attn: torch.Tensor   # after residual add (attention branch)
    normed2: torch.Tensor             # after ffn_norm
    gate_up: torch.Tensor             # after wfc1
    y1: torch.Tensor
    y2: torch.Tensor
    intermediate: torch.Tensor        # after silu(gate) * up
    mlp_out: torch.Tensor             # after w2


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

        normed = rms_norm(hidden, layer.input_norm_weight)
        if capture_records is not None:
            _capture_forward(capture_records, capture_prefix, f"layers.{li}.attention_norm", 0, normed)

        q, k, v = project_qkv(normed, layer.qkv_weight)
        if capture_records is not None:
            _capture_forward(capture_records, capture_prefix, f"layers.{li}.wqkv", 0, q)

        q_rot = apply_rope(q, rope_freqs)
        k_rot = apply_rope(k, rope_freqs)
        allow_math = not torch.cuda.is_available()
        attn = _gqa_attention(q_rot, k_rot, v, allow_math_fallback=allow_math)
        attn_flat = attn.reshape(B, S, C.NUM_HEADS * C.HEAD_DIM)
        attn_out = torch.matmul(attn_flat, layer.attention_proj_weight.t())
        if capture_records is not None:
            _capture_forward(capture_records, capture_prefix, f"layers.{li}.wo", 0, attn_out)
        hidden_after_attn = hidden + attn_out * depth_scale_main
        hidden = hidden_after_attn

        # MLP sub-layer
        normed2 = rms_norm(hidden, layer.pre_mlp_norm_weight)
        if capture_records is not None:
            _capture_forward(capture_records, capture_prefix, f"layers.{li}.ffn_norm", 0, normed2)

        gate_up = torch.matmul(normed2, layer.mlp_fc1_weight.t())
        if capture_records is not None:
            _capture_forward(capture_records, capture_prefix, f"layers.{li}.wfc1", 0, gate_up)

        y1, y2 = gate_up.chunk(2, dim=-1)
        intermediate = (torch.sigmoid(y1.float()) * y1 * y2.float()).to(y1.dtype)
        mlp_out = torch.matmul(intermediate, layer.mlp_fc2_weight.t())
        if capture_records is not None:
            _capture_forward(capture_records, capture_prefix, f"layers.{li}.w2", 0, mlp_out)
        hidden = hidden + mlp_out * depth_scale_main

        layer_caches.append(LayerCache(
            hidden_before_attn=hidden_before_attn,
            normed=normed, q=q, k=k, v=v,
            q_rot=q_rot, k_rot=k_rot,
            attn_flat=attn_flat, attn_out=attn_out,
            hidden_after_attn=hidden_after_attn,
            normed2=normed2, gate_up=gate_up, y1=y1, y2=y2,
            intermediate=intermediate, mlp_out=mlp_out,
        ))

    # ── Final norm ─────────────────────────────────────────────────────
    hidden_normed = rms_norm(hidden, model.final_norm_weight)
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
        a = rms_norm(mtp_emb, model.mtp.emb_input_norm_weight)
        b = rms_norm(hidden_normed, model.mtp.hidden_input_norm_weight)

        if capture_records is not None:
            _capture_forward(capture_records, capture_prefix, "mtp.emb_input_layernorm", 0, a)
            _capture_forward(capture_records, capture_prefix, "mtp.hidden_input_layernorm", 0, b)

        eagle_h = torch.matmul(torch.cat([a, b], dim=-1), model.mtp.eagle_fc_weight.t())
        if capture_records is not None:
            _capture_forward(capture_records, capture_prefix, "mtp.eagle_fc", 0, eagle_h)

        # MTP transformer layer
        mtp_hidden_before_attn = eagle_h
        mtp_normed = rms_norm(eagle_h, model.mtp.layer.input_norm_weight)
        if capture_records is not None:
            _capture_forward(capture_records, capture_prefix, "mtp.layer.attention_norm", 0, mtp_normed)

        mtp_q, mtp_k, mtp_v = project_qkv(mtp_normed, model.mtp.layer.qkv_weight)
        mtp_q_rot = apply_rope(mtp_q, rope_freqs)
        mtp_k_rot = apply_rope(mtp_k, rope_freqs)
        mtp_attn = _gqa_attention(mtp_q_rot, mtp_k_rot, mtp_v, allow_math_fallback=allow_math)
        mtp_attn_flat = mtp_attn.reshape(B, S, C.NUM_HEADS * C.HEAD_DIM)
        mtp_attn_out = torch.matmul(mtp_attn_flat, model.mtp.layer.attention_proj_weight.t())
        if capture_records is not None:
            _capture_forward(capture_records, capture_prefix, "mtp.layer.wo", 0, mtp_attn_out)
        mtp_hidden_after_attn = eagle_h + mtp_attn_out * depth_scale_mtp
        eagle_h = mtp_hidden_after_attn

        mtp_normed2 = rms_norm(eagle_h, model.mtp.layer.pre_mlp_norm_weight)
        mtp_gate_up = torch.matmul(mtp_normed2, model.mtp.layer.mlp_fc1_weight.t())
        mtp_y1, mtp_y2 = mtp_gate_up.chunk(2, dim=-1)
        mtp_intermediate = (torch.sigmoid(mtp_y1.float()) * mtp_y1 * mtp_y2.float()).to(mtp_y1.dtype)
        mtp_mlp_out = torch.matmul(mtp_intermediate, model.mtp.layer.mlp_fc2_weight.t())
        if capture_records is not None:
            _capture_forward(capture_records, capture_prefix, "mtp.layer.w2", 0, mtp_mlp_out)
        eagle_h = eagle_h + mtp_mlp_out * depth_scale_mtp

        mtp_mtp_layer_cache = LayerCache(
            hidden_before_attn=mtp_hidden_before_attn,
            normed=mtp_normed, q=mtp_q, k=mtp_k, v=mtp_v,
            q_rot=mtp_q_rot, k_rot=mtp_k_rot,
            attn_flat=mtp_attn_flat, attn_out=mtp_attn_out,
            hidden_after_attn=mtp_hidden_after_attn,
            normed2=mtp_normed2, gate_up=mtp_gate_up,
            y1=mtp_y1, y2=mtp_y2,
            intermediate=mtp_intermediate, mlp_out=mtp_mlp_out,
        )

        mtp_final = rms_norm(eagle_h, model.mtp.final_norm_weight)
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


def _add_to_grad_bufs(fp32_grad_bufs: list[torch.Tensor],
                      bf16_params: list[torch.Tensor],
                      weight: torch.Tensor,
                      grad_weight: torch.Tensor) -> None:
    """Add a weight gradient to the matching fp32 gradient buffer."""
    for buf, p in zip(fp32_grad_bufs, bf16_params):
        if p.data_ptr() == weight.data_ptr():
            buf.add_(grad_weight.to(device=buf.device))
            return


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
    capture_records: dict | None,
    capture_prefix: str,
    mtp_ce_weight: float,
) -> None:
    """Static backward pass — no autograd, no ``.backward()``.

    Walks the computation graph in reverse, computing gradients for
    every parameter and accumulating fp32 gradients into ``fp32_grad_bufs``.
    """
    B, S = cache.input_ids.shape
    H = C.HIDDEN_SIZE
    V = C.VOCAB_SIZE
    allow_math = True  # always use math fallback for backward (flash_attn backward not accessible as standalone)

    # ====================================================================
    # MTP BRANCH BACKWARD (if enabled)
    # ====================================================================
    d_hidden_normed_main = None  # gradient from MTP into hidden_normed

    if cache.mtp_logits is not None and model.mtp is not None:
        # ── MTP Cross-entropy loss backward ────────────────────────────
        grad_mtp_logits = cross_entropy_backward(
            cache.mtp_logits, cache.mtp_labels, cache.mtp_loss_mask
        )

        # ── MTP LM head backward: logits = mtp_pre_head @ output_weight.T
        # d(mtp_pre_head) = grad_mtp_logits @ output_weight
        # d(output_weight) += grad_mtp_logits.T @ mtp_pre_head
        # grad_mtp_logits is fp32 (from FP32 CE), output_weight is bf16
        d_mtp_pre_head = torch.matmul(grad_mtp_logits, model.output_weight.to(dtype=grad_mtp_logits.dtype))
        g2_mtp = grad_mtp_logits.reshape(-1, V)
        mtp_pre = cache.mtp_pre_head.reshape(-1, H)
        dw_output_mtp = torch.matmul(g2_mtp.transpose(0, 1), mtp_pre.to(dtype=grad_mtp_logits.dtype)).float()
        _add_to_grad_bufs(fp32_grad_bufs, bf16_params, model.output_weight, dw_output_mtp)

        # ── width_mult backward: d(hidden) = d(mtp_pre_head) / width_mult
        d_mtp_final = d_mtp_pre_head / width_mult

        # ── mtp.final_layernorm backward
        d_mtp_eagle_h, dw_mtp_fn = rms_norm_backward(
            d_mtp_final, cache.mtp_eagle_h, model.mtp.final_norm_weight
        )
        _add_to_grad_bufs(fp32_grad_bufs, bf16_params, model.mtp.final_norm_weight, dw_mtp_fn)

        # ── MTP transformer layer backward (reverse order)
        # First, the MLP residual: d_eagle_h = d_mtp_eagle_h (from MLP branch)
        # Then, w2 backward
        mtp_lc = cache.mtp_layer_cache

        # w2 backward: mlp_out = intermediate @ w2.T
        d_intermediate, dw_mtp_w2 = linear_backward(
            d_mtp_eagle_h * depth_scale_mtp,
            mtp_lc.intermediate,
            model.mtp.layer.mlp_fc2_weight,
        )
        _add_to_grad_bufs(fp32_grad_bufs, bf16_params, model.mtp.layer.mlp_fc2_weight, dw_mtp_w2)

        # SwiGLU backward
        d_mtp_y1, d_mtp_y2 = silu_swiglu_intermediate_backward(
            d_intermediate, mtp_lc.y1, mtp_lc.y2
        )
        d_mtp_gate_up = torch.cat([d_mtp_y1, d_mtp_y2], dim=-1)

        # wfc1 backward: gate_up = normed2 @ wfc1.T
        d_mtp_normed2, dw_mtp_fc1 = linear_backward(
            d_mtp_gate_up, mtp_lc.normed2, model.mtp.layer.mlp_fc1_weight
        )
        _add_to_grad_bufs(fp32_grad_bufs, bf16_params, model.mtp.layer.mlp_fc1_weight, dw_mtp_fc1)

        # ffn_norm backward (MLP branch)
        d_mtp_hidden_mlp, dw_mtp_mlp_norm = rms_norm_backward(
            d_mtp_normed2, mtp_lc.hidden_after_attn, model.mtp.layer.pre_mlp_norm_weight
        )
        _add_to_grad_bufs(fp32_grad_bufs, bf16_params, model.mtp.layer.pre_mlp_norm_weight, dw_mtp_mlp_norm)

        # Add MLP and attention gradient at hidden
        d_mtp_eagle_h = d_mtp_hidden_mlp

        # Attention branch residual from hidden_before_attn
        # wo backward: attn_out = attn_flat @ wo.T
        d_mtp_attn_flat, dw_mtp_wo = linear_backward(
            d_mtp_eagle_h * depth_scale_mtp,
            mtp_lc.attn_flat,
            model.mtp.layer.attention_proj_weight,
        )
        _add_to_grad_bufs(fp32_grad_bufs, bf16_params, model.mtp.layer.attention_proj_weight, dw_mtp_wo)

        # Attention backward (GQA)
        d_mtp_attn = d_mtp_attn_flat.reshape(B, S, C.NUM_HEADS, C.HEAD_DIM)
        d_mtp_q_rot, d_mtp_k_rot, d_mtp_v = gqa_attention_backward(
            d_mtp_attn, mtp_lc.q_rot, mtp_lc.k_rot, mtp_lc.v,
            allow_math_fallback=allow_math,
        )

        # RoPE backward
        d_mtp_q = apply_rope_backward(d_mtp_q_rot, rope_freqs)
        d_mtp_k = apply_rope_backward(d_mtp_k_rot, rope_freqs)

        # QKV projection backward
        d_mtp_normed, dw_mtp_qkv = project_qkv_backward(
            d_mtp_q, d_mtp_k, d_mtp_v,
            mtp_lc.normed, model.mtp.layer.qkv_weight,
        )
        _add_to_grad_bufs(fp32_grad_bufs, bf16_params, model.mtp.layer.qkv_weight, dw_mtp_qkv)

        # attention_norm backward
        d_mtp_hidden_before_attn, dw_mtp_attn_norm = rms_norm_backward(
            d_mtp_normed, mtp_lc.hidden_before_attn, model.mtp.layer.input_norm_weight
        )
        _add_to_grad_bufs(fp32_grad_bufs, bf16_params, model.mtp.layer.input_norm_weight, dw_mtp_attn_norm)

        # Add residual from attention branch
        d_mtp_eagle_h_out = d_mtp_hidden_before_attn + d_mtp_eagle_h

        # ── eagle_fc backward: eagle_h = cat([a, b]) @ eagle_fc.T
        d_mtp_cat, dw_mtp_eagle_fc = linear_backward(
            d_mtp_eagle_h_out, torch.cat([cache.mtp_a, cache.mtp_b], dim=-1),
            model.mtp.eagle_fc_weight,
        )
        _add_to_grad_bufs(fp32_grad_bufs, bf16_params, model.mtp.eagle_fc_weight, dw_mtp_eagle_fc)
        d_mtp_a, d_mtp_b = d_mtp_cat.chunk(2, dim=-1)

        # mtp.hidden_input_layernorm backward
        d_hidden_normed_mtp, dw_mtp_hnorm = rms_norm_backward(
            d_mtp_b, cache.hidden_normed, model.mtp.hidden_input_norm_weight
        )
        _add_to_grad_bufs(fp32_grad_bufs, bf16_params, model.mtp.hidden_input_norm_weight, dw_mtp_hnorm)

        # mtp.emb_input_layernorm backward
        _, dw_mtp_enorm = rms_norm_backward(
            d_mtp_a, cache.mtp_emb, model.mtp.emb_input_norm_weight
        )
        _add_to_grad_bufs(fp32_grad_bufs, bf16_params, model.mtp.emb_input_norm_weight, dw_mtp_enorm)

        # MTP embedding backward
        dw_mtp_emb = embedding_backward(
            d_mtp_a * mup_emb_scale, cache.mtp_input_ids, V
        )
        _add_to_grad_bufs(fp32_grad_bufs, bf16_params, model.tok_embeddings_weight, dw_mtp_emb)

        # Accumulate gradient into hidden_normed from MTP
        d_hidden_normed_main = d_hidden_normed_mtp

    # ====================================================================
    # MAIN BRANCH BACKWARD
    # ====================================================================

    # ── Main Cross-entropy loss backward ───────────────────────────────
    grad_logits = cross_entropy_backward(cache.main_logits, cache.labels, cache.loss_mask)

    # ── Main LM head backward: logits = main_pre_head @ output_weight.T
    # grad_logits is fp32 (from FP32 CE), output_weight is bf16
    d_main_pre_head = torch.matmul(grad_logits, model.output_weight.to(dtype=grad_logits.dtype))
    g2_main = grad_logits.reshape(-1, V)
    main_pre = cache.main_pre_head.reshape(-1, H)
    dw_output_main = torch.matmul(g2_main.transpose(0, 1), main_pre.to(dtype=grad_logits.dtype)).float()
    _add_to_grad_bufs(fp32_grad_bufs, bf16_params, model.output_weight, dw_output_main)

    # ── width_mult backward: d(hidden) = d(main_pre_head) / width_mult
    d_hidden_normed = d_main_pre_head / width_mult

    # Add MTP contribution to hidden_normed gradient
    if d_hidden_normed_main is not None:
        d_hidden_normed = d_hidden_normed + d_hidden_normed_main

    # ── final_norm backward
    d_hidden, dw_final_norm = rms_norm_backward(
        d_hidden_normed, cache.layer_caches[-1].mlp_out, model.final_norm_weight
    )
    # Actually, the input to final_norm is the hidden AFTER the last layer, not mlp_out.
    # Let me fix this: we need the hidden BEFORE final_norm as the rms_norm input.
    # The hidden before final_norm is the output of the last layer, which is cached
    # in the last layer cache's hidden state after the MLP residual add.
    # But we don't save that directly. Let me reconstruct it.
    # hidden_post_last_layer = cache.layer_caches[-1].hidden_after_attn + cache.layer_caches[-1].mlp_out * depth_scale_main
    last_layer = cache.layer_caches[-1]
    hidden_post_last_layer = last_layer.hidden_after_attn + last_layer.mlp_out * depth_scale_main
    d_hidden, dw_final_norm = rms_norm_backward(
        d_hidden_normed, hidden_post_last_layer, model.final_norm_weight
    )
    _add_to_grad_bufs(fp32_grad_bufs, bf16_params, model.final_norm_weight, dw_final_norm)

    # ── Transformer layers (reverse order) ─────────────────────────────
    for li in range(len(model.layers) - 1, -1, -1):
        layer = model.layers[li]
        lc = cache.layer_caches[li]

        # ── MLP backward ──────────────────────────────────────────────
        # The MLP residual is: d_hidden includes d_hidden from layers above
        # mlp_out = intermediate @ w2.T
        d_intermediate, dw_w2 = linear_backward(
            d_hidden * depth_scale_main, lc.intermediate, layer.mlp_fc2_weight
        )
        _add_to_grad_bufs(fp32_grad_bufs, bf16_params, layer.mlp_fc2_weight, dw_w2)

        # SwiGLU backward: intermediate = silu(y1) * y2
        d_y1, d_y2 = silu_swiglu_intermediate_backward(d_intermediate, lc.y1, lc.y2)
        d_gate_up = torch.cat([d_y1, d_y2], dim=-1)

        # wfc1 backward: gate_up = normed2 @ wfc1.T
        d_normed2, dw_fc1 = linear_backward(
            d_gate_up, lc.normed2, layer.mlp_fc1_weight
        )
        _add_to_grad_bufs(fp32_grad_bufs, bf16_params, layer.mlp_fc1_weight, dw_fc1)

        # ffn_norm backward (MLP branch into hidden_after_attn)
        d_hidden_mlp, dw_mlp_norm = rms_norm_backward(
            d_normed2, lc.hidden_after_attn, layer.pre_mlp_norm_weight
        )
        _add_to_grad_bufs(fp32_grad_bufs, bf16_params, layer.pre_mlp_norm_weight, dw_mlp_norm)

        # Attention branch residual
        d_hidden = d_hidden_mlp

        # ── Attention backward ─────────────────────────────────────────
        # wo backward: attn_out = attn_flat @ wo.T
        d_attn_flat, dw_wo = linear_backward(
            d_hidden * depth_scale_main, lc.attn_flat, layer.attention_proj_weight
        )
        _add_to_grad_bufs(fp32_grad_bufs, bf16_params, layer.attention_proj_weight, dw_wo)

        # Attention backward (GQA)
        d_attn = d_attn_flat.reshape(B, S, C.NUM_HEADS, C.HEAD_DIM)
        d_q_rot, d_k_rot, d_v = gqa_attention_backward(
            d_attn, lc.q_rot, lc.k_rot, lc.v,
            allow_math_fallback=allow_math,
        )

        # RoPE backward
        d_q = apply_rope_backward(d_q_rot, rope_freqs)
        d_k = apply_rope_backward(d_k_rot, rope_freqs)

        # QKV projection backward
        d_normed, dw_qkv = project_qkv_backward(
            d_q, d_k, d_v, lc.normed, layer.qkv_weight
        )
        _add_to_grad_bufs(fp32_grad_bufs, bf16_params, layer.qkv_weight, dw_qkv)

        # attention_norm backward
        d_hidden_before_attn, dw_attn_norm = rms_norm_backward(
            d_normed, lc.hidden_before_attn, layer.input_norm_weight
        )
        _add_to_grad_bufs(fp32_grad_bufs, bf16_params, layer.input_norm_weight, dw_attn_norm)

        # Add residual from attention branch: hidden = hidden_before_attn + attn_out * depth_scale
        # So d_hidden_before_attn includes d_hidden from attn_out * depth_scale
        # and from the MLP residual (which we already added)
        d_hidden = d_hidden_before_attn

    # ── Embedding backward ─────────────────────────────────────────────
    dw_emb = embedding_backward(
        d_hidden * mup_emb_scale, cache.input_ids, C.VOCAB_SIZE
    )
    _add_to_grad_bufs(fp32_grad_bufs, bf16_params, model.tok_embeddings_weight, dw_emb)


# ── Main training loop ────────────────────────────────────────────────────────


def run_training_loop(config: TrainLoopConfig, *, loss_tag: str = "LOSS") -> None:
    """Run ``num_steps`` of training and emit one line per global step.

    See module-level docstring for the full contract and stdout grammar.
    """
    rank = _init_process_group()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"

    # ── Seed ───────────────────────────────────────────────────────────
    torch.manual_seed(config.seed)

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

    use_mtp = int(os.environ.get("MTP_NUM_LAYERS", "1")) > 0
    ce_w = float(os.environ.get("MTP_LOSS_WEIGHT", "0.3"))

    # ── Capture setup ──────────────────────────────────────────────────
    capture_records: dict | None = None
    capture_prefix = ""
    if config.hash_capture_level > 0:
        capture_records = {}
        capture_prefix = f"rank{rank}.mb0."

    # ── MFU prep ───────────────────────────────────────────────────────
    tokens_per_step = config.global_batch_size * C.MAX_SEQ_LEN
    train_per_token = _compute_flops_per_step()
    flops_per_step = train_per_token * tokens_per_step
    peak_total = H100_BF16_PEAK_FLOPS * max(config.world_size, 1)

    # ── Learning rate schedule params ──────────────────────────────────
    lr = float(os.environ.get("LR", str(config.lr)))
    min_lr = float(os.environ.get("MIN_LR", str(config.min_lr)))
    lr_warmup_iters = int(os.environ.get("LR_WARMUP_ITERS", str(config.lr_warmup_iters)))
    lr_decay_iters = int(os.environ.get("LR_DECAY_ITERS", str(config.lr_decay_iters)))
    lr_wsd_decay_iters = int(os.environ.get("LR_WSD_DECAY_ITERS", str(config.lr_wsd_decay_iters)))

    # ── Training loop ──────────────────────────────────────────────────
    for step in range(config.num_steps):
        torch.cuda.synchronize()
        t0 = time.time()

        # Zero gradients
        for buf in fp32_grad_bufs:
            buf.zero_()

        local_lm_sum = torch.zeros(1, device=device, dtype=torch.float64)
        local_lm_n = torch.zeros(1, device=device, dtype=torch.float64)
        local_mtp_sum = torch.zeros(1, device=device, dtype=torch.float64)
        local_mtp_n = torch.zeros(1, device=device, dtype=torch.float64)

        # Grad accumulation loop
        for _mb in range(config.grad_accum_steps):
            input_ids, labels, loss_mask = _next_batch(iter_dl, device)

            if use_mtp:
                mtp_in, mtp_lab, mtp_mask = _build_mtp_tensors(input_ids, labels, loss_mask)
                cache = _forward_with_cache(
                    model, input_ids, labels, loss_mask,
                    mtp_in, mtp_lab, mtp_mask,
                    rope_freqs, width_mult, mup_emb_scale,
                    depth_scale_main, depth_scale_mtp,
                    capture_records if config.hash_capture_level >= 2 else None,
                    capture_prefix,
                )
                lm_sum, lm_n = masked_cross_entropy(cache.main_logits, labels, loss_mask)
                mtp_sum_v, mtp_n_v = masked_cross_entropy(cache.mtp_logits, mtp_lab, mtp_mask)
                local_mtp_sum += mtp_sum_v.detach().double()
                local_mtp_n += mtp_n_v.detach().double()
            else:
                cache = _forward_with_cache(
                    model, input_ids, labels, loss_mask,
                    None, None, None,
                    rope_freqs, width_mult, mup_emb_scale,
                    depth_scale_main, depth_scale_mtp,
                    capture_records if config.hash_capture_level >= 2 else None,
                    capture_prefix,
                )
                lm_sum, lm_n = masked_cross_entropy(cache.main_logits, labels, loss_mask)

            # ── Static backward pass ──────────────────────────────────
            _static_backward(
                model, cache, rope_freqs, width_mult, mup_emb_scale,
                depth_scale_main, depth_scale_mtp,
                fp32_grad_bufs, bf16_params,
                capture_records, capture_prefix,
                ce_w if use_mtp else 0.0,
            )

            local_lm_sum += lm_sum.detach().double()
            local_lm_n += lm_n.detach().double()

        # ── Post-accumulation: all-reduce ──────────────────────────────
        if config.world_size > 1:
            dist.all_reduce(local_lm_sum)
            dist.all_reduce(local_lm_n)
            if use_mtp:
                dist.all_reduce(local_mtp_sum)
                dist.all_reduce(local_mtp_n)

        reported_lm = (local_lm_sum / local_lm_n.clamp(min=1.0)).item()
        reported_mtp = (local_mtp_sum / local_mtp_n.clamp(min=1.0)).item() if use_mtp else 0.0

        if config.world_size > 1:
            for buf in fp32_grad_bufs:
                dist.all_reduce(buf)

        # ── Capture mode: dump and exit ───────────────────────────────
        if config.hash_capture_level > 0:
            _capture_all_gradients(capture_records, bf16_params, fp32_grad_bufs, model)
            _write_capture_output(config, capture_records, rank)
            _ordered_teardown()
            return  # Not reached (SystemExit raised in _ordered_teardown)

        # ── Per-step logging ──────────────────────────────────────────
        torch.cuda.synchronize()
        step_time = time.time() - t0
        total = reported_lm + ce_w * reported_mtp if use_mtp else reported_lm
        mfu = flops_per_step / (step_time * peak_total) * 100.0 if step_time > 0 else 0.0

        if rank == 0:
            loss_line = (
                f"[{loss_tag}] step={config.start_step + step + 1} "
                f"global_loss={total:.9e} "
                f"grad_norm={0.0:.9e} "
                f"time_s={step_time:.6f} "
                f"mfu_e2e_standard={mfu:.6f}"
            )
            print(loss_line, flush=True)

    # ── Ordered teardown ──────────────────────────────────────────────
    _ordered_teardown()


# ── Gradient capture helpers ────────────────────────────────────────────────


def _capture_all_gradients(capture_records, bf16_params, fp32_grad_bufs, model):
    """Capture all parameter gradients as hash records."""
    fqn_map = _build_fqn_map(model)
    for buf, p in zip(fp32_grad_bufs, bf16_params):
        fqn = fqn_map.get(p.data_ptr(), None)
        if fqn is not None:
            _capture_gradient(capture_records, "", fqn, buf)


def _build_fqn_map(model: ModelParameters) -> dict:
    """Build a mapping from tensor data_ptr to FQN string."""
    fqn_map = {}
    fqn_map[model.tok_embeddings_weight.data_ptr()] = "tok_embeddings.weight"
    fqn_map[model.final_norm_weight.data_ptr()] = "norm.weight"
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
    """Collect all bf16 weight tensors from the model."""
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
    """Write the capture hash records to the output file."""
    if config.hash_output and records is not None:
        p = Path(config.hash_output)
        p.parent.mkdir(parents=True, exist_ok=True)
        rank_path = p.with_name(p.name + f".rank{rank}")
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


def _ordered_teardown() -> None:
    """Ordered teardown: drain GPU, barrier, destroy PG, empty cache, then exit.

    Uses ``os._exit(0)`` to bypass the Python interpreter's finalizer,
    which would otherwise crash with SIGABRT when the NCCL PG destructor
    races Py_Finalize.  This is safe because all data has been flushed
    to disk before this point.
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
    # Bypass the Python finalizer to avoid the NCCL PG destructor crash.
    os._exit(0)