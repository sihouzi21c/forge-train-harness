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


# ── GQA Attention (uses F.scaled_dot_product_attention for bitwise alignment with ref) ────


def _gqa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   allow_math_fallback: bool = False,
                   ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """GQA scaled dot-product attention.

    Always uses ``F.scaled_dot_product_attention`` for bitwise alignment
    with the ref.  The ref's determinism stack disables flash SDP and
    mem-efficient SDP, falling back to the math backend when supported.
    For GQA (different head counts), the math backend may not be
    available and the flash attention backend is used instead.

    Returns ``(output, None)`` — the ``softmax_lse`` is always ``None``
    because ``F.scaled_dot_product_attention`` does not expose it.
    Gradients are computed via ``torch.autograd.grad`` in the matching
    backward function.

    ``q`` shape ``[B, S, NUM_HEADS, HEAD_DIM]``.
    ``k``, ``v`` shape ``[B, S, NUM_KV_HEADS, HEAD_DIM]``.
    """
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True, attn_mask=None, dropout_p=0.0, scale=None
    )
    return out, None


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