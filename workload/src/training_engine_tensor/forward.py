"""Forward primitives for the self-developed training engine.

Every function is a pure-torch primitive operating on bare ``torch.Tensor``
arguments — no ``nn.Module``, no ``nn.functional``, no autograd.  The
caller is responsible for threading the compute in the correct order.

The primitives are designed to be bitwise-identical to the corresponding
section of ``ref/reference/model_pure_mup_mtp.py`` when given the same
inputs.
"""

from __future__ import annotations

import torch

from training_engine_tensor.config import (
    FFN_HIDDEN_SIZE,
    HIDDEN_SIZE,
    NUM_HEADS,
    NUM_KV_HEADS,
    HEAD_DIM,
    NORM_EPS,
    ROPE_THETA,
)


# ── RoPE ──────────────────────────────────────────────────────────────────


def precompute_rope_freqs(max_seq_len: int = 4096,
                           head_dim: int | None = None,
                           device: str = "cpu") -> torch.Tensor:
    """Precompute cosine/sine frequencies for RoPE.

    Returns ``[max_seq_len, head_dim]`` fp32 tensor.
    """
    d = head_dim if head_dim is not None else HEAD_DIM
    inv_freq = 1.0 / (
        ROPE_THETA ** (torch.arange(0, d, 2, dtype=torch.float32, device=device) / d)
    )
    seq = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(seq, inv_freq)
    return torch.cat((freqs, freqs), dim=-1)  # [S, D]


def apply_rope(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding.

    ``t`` shape ``[B, S, H, D]``, ``freqs`` shape ``[S, D]``.
    Returns rotated tensor of the same shape and dtype as ``t``.
    """
    cos_ = torch.cos(freqs).to(t.dtype)[None, :, None, :]
    sin_ = torch.sin(freqs).to(t.dtype)[None, :, None, :]
    half = t.shape[-1] // 2
    x1 = t[..., :half]
    x2 = t[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return t * cos_ + rotated * sin_


# ── QKV projection (GQA) ──────────────────────────────────────────────────


def project_qkv(hidden: torch.Tensor, weight: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project ``hidden`` through the packed QKV weight with GQA interleaving.

    Returns ``(q, k, v)`` where:
        q: ``[B, S, NUM_HEADS, HEAD_DIM]``
        k: ``[B, S, NUM_KV_HEADS, HEAD_DIM]``
        v: ``[B, S, NUM_KV_HEADS, HEAD_DIM]``
    """
    projected = torch.matmul(hidden, weight.t())  # [B, S, qkv_width]
    leading = hidden.shape[:-1]
    nkv = NUM_KV_HEADS
    nq_per_kv = NUM_HEADS // nkv
    d = HEAD_DIM
    group_width = (nq_per_kv + 2) * d
    grouped = projected.view(*leading, nkv, group_width)
    q = grouped[..., : nq_per_kv * d].reshape(*leading, NUM_HEADS, d)
    k = grouped[..., nq_per_kv * d: nq_per_kv * d + d]
    v = grouped[..., nq_per_kv * d + d:]
    return q, k, v


# ── RMSNorm ────────────────────────────────────────────────────────────────


def rms_norm(hidden: torch.Tensor, weight: torch.Tensor,
             eps: float | None = None) -> torch.Tensor:
    """RMSNorm.  Follows the ``(x32 * r).to(bf16) * weight`` chain.

    This is the primitive decomposition that ``nn.RMSNorm`` uses on CUDA
    and is the ONLY chain that is bitwise-identical to the ref's RMSNorm
    at bf16.  See ``test_training_engine_forward.py::TestRmsNormAndSwiglu``.

    For bf16 inputs, computation is in fp32. For fp32/fp64 inputs, the
    computation stays in the input dtype.
    """
    eps_val = eps if eps is not None else NORM_EPS
    if hidden.dtype == torch.bfloat16:
        x32 = hidden.float()
        r = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps_val)
        return (x32 * r).to(hidden.dtype) * weight
    # For higher precision types, stay in the input dtype
    r = torch.rsqrt(hidden.pow(2).mean(dim=-1, keepdim=True) + eps_val)
    return (hidden * r) * weight


# ── SwiGLU MLP ─────────────────────────────────────────────────────────────


def mlp_swiglu(hidden: torch.Tensor, fc1_weight: torch.Tensor,
               fc2_weight: torch.Tensor) -> torch.Tensor:
    """SwiGLU MLP: ``silu(gate) * up`` projected through ``fc2``.

    SwiGLU uses fp32 for the SiLU activation multiplication (``silu(y1.float()) * y2.float()``),
    then casts back to the input dtype before the fc2 projection.
    """
    gate_up = torch.matmul(hidden, fc1_weight.t())  # [B, S, 2*ffn]
    y_1, y_2 = gate_up.chunk(2, dim=-1)
    intermediate = (torch.sigmoid(y_1.float()) * y_1.float() * y_2.float()).to(y_1.dtype)
    return torch.matmul(intermediate, fc2_weight.t())


# ── GQA Attention (flash attention path with softmax stats for backward) ────


def _gqa_attention_flash(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         causal: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Flash attention forward that also returns softmax statistics.

    Returns ``(output, softmax_lse)`` where:
        output: ``[B, S, NUM_HEADS, HEAD_DIM]``
        softmax_lse: ``[B, NUM_HEADS, S]``

    The flash attention CUDA kernel always computes ``softmax_lse`` internally
    (needed for its own backward).  We pass ``return_softmax=False`` to avoid
    a kernel-version check that rejects ``return_softmax=True`` with
    ``dropout_p=0.0``; the returned ``softmax_lse`` is still valid.
    """
    from flash_attn.flash_attn_interface import _flash_attn_forward

    d = q.shape[-1]
    softmax_scale = d ** -0.5
    out, _, _, _, _, softmax_lse, _, _ = _flash_attn_forward(
        q, k, v, dropout_p=0.0, softmax_scale=softmax_scale,
        causal=causal, window_size=(-1, -1), alibi_slopes=None,
        return_softmax=False,
    )
    return out, softmax_lse


# ── GQA Attention (math fallback for CPU / non-flash paths) ────────────────


def _gqa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   allow_math_fallback: bool = False,
                   ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """GQA scaled dot-product attention.

    On CUDA, dispatches to ``flash_attn`` (via ``_flash_attn_forward``) when
    ``allow_math_fallback`` is ``False``.  Returns ``(output, softmax_lse)``
    where ``softmax_lse`` is ``None`` for the math fallback path.

    When ``allow_math_fallback`` is ``True`` (or on CPU), uses the explicit
    math backend (softmax + matmul) and returns ``(output, None)``.

    ``q`` shape ``[B, S, NUM_HEADS, HEAD_DIM]``.
    ``k``, ``v`` shape ``[B, S, NUM_KV_HEADS, HEAD_DIM]``.
    Returns ``(output, softmax_lse)`` where ``output`` is ``[B, S, NUM_HEADS, HEAD_DIM]``.
    """
    if q.device.type == "cuda" and not allow_math_fallback:
        return _gqa_attention_flash(q, k, v, causal=True)

    if not allow_math_fallback:
        raise NotImplementedError(
            "_gqa_attention: flash_attn not available on this device. "
            "Pass allow_math_fallback=True for the CPU math path."
        )

    # Math fallback: causal softmax attention (returns None for softmax_lse)
    # flash_attn uses [B, S, H, D] layout; for math fallback we permute to [B, H, S, D]
    B, S, H_q, D = q.shape
    _, _, H_kv, _ = k.shape

    # Repeat KV heads to match Q heads
    rep = H_q // H_kv
    if rep > 1:
        k = k.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, S, H_q, D)
        v = v.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, S, H_q, D)

    # Permute to [B, H, S, D] for standard attention computation
    q_attn = q.permute(0, 2, 1, 3)  # [B, H_q, S, D]
    k_attn = k.permute(0, 2, 1, 3)  # [B, H_q, S, D]
    v_attn = v.permute(0, 2, 1, 3)  # [B, H_q, S, D]

    scale = D ** 0.5
    scores = torch.matmul(q_attn.to(dtype=q.dtype), k_attn.to(dtype=q.dtype).transpose(-2, -1)) / scale  # [B, H_q, S, S]

    # Causal mask: upper triangle
    causal_mask = torch.triu(torch.ones(S, S, device=q.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)  # [B, H_q, S, S]
    out = torch.matmul(attn_weights.to(v_attn.dtype), v_attn)  # [B, H_q, S, D]
    return out.permute(0, 2, 1, 3), None  # [B, S, H_q, D]


# ── Decoder layer forward ──────────────────────────────────────────────────


def decoder_layer_forward(
    hidden: torch.Tensor,
    layer: "LayerParameterView",  # type: ignore[name-defined]  # noqa: F821
    rope_freqs: torch.Tensor,
    depth_scale: float = 1.0,
    allow_math_fallback: bool = False,
) -> torch.Tensor:
    """One transformer decoder layer forward pass.

    ``hidden`` shape ``[B, S, HIDDEN_SIZE]``.
    ``layer`` is a :class:`training_engine_tensor.parameters.LayerParameterView`.
    ``rope_freqs`` shape ``[S, HEAD_DIM]``.
    Returns updated ``hidden`` of the same shape.
    """
    from training_engine_tensor.parameters import LayerParameterView  # noqa: F811

    B, S, _ = hidden.shape

    # Attention sub-layer
    normed = rms_norm(hidden, layer.input_norm_weight)
    q, k, v = project_qkv(normed, layer.qkv_weight)
    q = apply_rope(q, rope_freqs)
    k = apply_rope(k, rope_freqs)
    attn, _ = _gqa_attention(q, k, v, allow_math_fallback=allow_math_fallback)
    attn_flat = attn.reshape(B, S, NUM_HEADS * HEAD_DIM)
    hidden = hidden + torch.matmul(attn_flat, layer.attention_proj_weight.t()) * depth_scale

    # MLP sub-layer
    normed2 = rms_norm(hidden, layer.pre_mlp_norm_weight)
    mlp_out = mlp_swiglu(normed2, layer.mlp_fc1_weight, layer.mlp_fc2_weight)
    hidden = hidden + mlp_out * depth_scale

    return hidden


# ── Embedding lookup ────────────────────────────────────────────────────────


def embedding_forward(idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Embedding lookup: ``weight[idx] * 1.0`` (no scale applied here).

    The caller (train_loop) multiplies by ``mup_emb_scale`` after lookup.
    """
    return weight[idx.long()]


# ── LM head ─────────────────────────────────────────────────────────────────


def lm_head_forward(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """LM head: ``hidden @ weight.T``.

    ``hidden`` shape ``[B, S, HIDDEN_SIZE]``.
    ``weight`` shape ``[VOCAB_SIZE, HIDDEN_SIZE]``.
    Returns ``[B, S, VOCAB_SIZE]``.
    """
    return torch.matmul(hidden, weight.t())


# ── Cross-entropy loss (FP32) ───────────────────────────────────────────────


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor,
                         loss_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 cross-entropy with loss masking.

    ``logits`` shape ``[B, S, V]``, ``labels`` shape ``[B, S]``,
    ``loss_mask`` shape ``[B, S]``.

    Returns ``(sum_loss [fp32 scalar], num_tokens [fp32 scalar])``.
    Mirrors the ref's ``masked_ce``.
    """
    B, S, V = logits.shape
    logits_f32 = logits.float().reshape(-1, V)
    labels_flat = labels.reshape(-1)

    # Log-sum-exp trick for numerical stability
    logits_max = logits_f32.max(dim=-1, keepdim=True).values
    logits_stable = logits_f32 - logits_max
    log_softmax = logits_stable - logits_stable.exp().sum(dim=-1, keepdim=True).log()
    nll = -log_softmax[torch.arange(logits_stable.shape[0], device=logits.device), labels_flat]

    mask = loss_mask.reshape(-1).float()
    return (nll * mask).sum(), mask.sum()