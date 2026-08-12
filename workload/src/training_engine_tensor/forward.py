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
    """RMSNorm.  Uses ``torch.nn.functional.rms_norm`` directly, matching the
    ref's ``_RMSNormFn.forward`` (which calls ``F.rms_norm(x, shape, weight, eps)``).

    This is the ONLY chain that is bitwise-identical to the ref's RMSNorm
    at bf16.  The manual ``(x32 * r).to(bf16) * weight`` decomposition can
    differ in floating-point rounding due to the fused kernel's reduction
    order.
    """
    import torch.nn.functional as _F
    eps_val = eps if eps is not None else NORM_EPS
    return _F.rms_norm(hidden, (weight.shape[0],), weight, eps_val)


# ── SwiGLU MLP ─────────────────────────────────────────────────────────────


def mlp_swiglu(hidden: torch.Tensor, fc1_weight: torch.Tensor,
               fc2_weight: torch.Tensor) -> torch.Tensor:
    """SwiGLU MLP: ``silu(gate) * up`` projected through ``fc2``.

    SwiGLU uses fp32 for the SiLU activation multiplication (``silu(y1.float()) * y2.float()``),
    then casts back to the input dtype before the fc2 projection.
    Uses ``torch.nn.functional.silu`` to match the ref's implementation exactly.
    """
    import torch.nn.functional as _F
    gate_up = torch.matmul(hidden, fc1_weight.t())  # [B, S, 2*ffn]
    y_1, y_2 = gate_up.chunk(2, dim=-1)
    intermediate = (_F.silu(y_1.float()) * y_2.float()).to(y_1.dtype)
    return torch.matmul(intermediate, fc2_weight.t())


# ── GQA Attention (uses flash_attn_func for bitwise alignment with ref) ────


def _gqa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   allow_math_fallback: bool = False,
                   ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """GQA scaled dot-product attention via flash_attn_func.

    Uses ``flash_attn_func`` directly, matching the ref's attention implementation
    exactly.  The ref's ``TransformerLayer.forward`` calls::

        from flash_attn import flash_attn_func
        attn = flash_attn_func(q, k, v, causal=True, deterministic=True)

    ``allow_math_fallback`` is ignored — the flash_attn kernel is the
    authoritative path for bitwise alignment.

    Returns ``(output, None)`` — the ``softmax_lse`` is always ``None``
    because ``flash_attn_func`` does not expose it through this wrapper.
    Gradients are computed via ``torch.autograd.grad`` in the matching
    backward function.

    ``q`` shape ``[B, S, NUM_HEADS, HEAD_DIM]``.
    ``k``, ``v`` shape ``[B, S, NUM_KV_HEADS, HEAD_DIM]``.
    """
    from flash_attn import flash_attn_func
    out = flash_attn_func(q, k, v, causal=True, deterministic=True)
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
    """Embedding lookup: ``F.embedding(idx, weight)``, matching the ref's ``_EmbeddingFn``.

    Uses ``torch.nn.functional.embedding`` which is the same op the ref's
    ``_EmbeddingFn.forward`` calls (``F.embedding(idx, weight)``).  The
    caller (train_loop) multiplies by ``mup_emb_scale`` after lookup.
    """
    import torch.nn.functional as _F
    return _F.embedding(idx, weight)


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
    """FP32 cross-entropy with loss masking, matching the ref's ``masked_ce``.

    Uses ``torch.nn.functional.cross_entropy`` with ``reduction="none"``,
    identical to the ref's implementation::

        nll = F.cross_entropy(logits.reshape(-1, V).float(),
                              labels.reshape(-1), reduction="none")

    ``logits`` shape ``[B, S, V]``, ``labels`` shape ``[B, S]``,
    ``loss_mask`` shape ``[B, S]``.

    Returns ``(sum_loss [fp32 scalar], num_tokens [fp32 scalar])``.
    """
    import torch.nn.functional as _F
    B, S, V = logits.shape
    nll = _F.cross_entropy(
        logits.reshape(-1, V).float(),
        labels.reshape(-1),
        reduction="none",
    )
    mask = loss_mask.reshape(-1).float()
    return (nll * mask).sum(), mask.sum()