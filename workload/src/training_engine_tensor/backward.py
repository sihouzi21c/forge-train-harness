"""Backward primitives for the self-developed training engine.

Every function implements the **static backward** of a forward primitive
from :mod:`training_engine_tensor.forward`.  The caller (train_loop) is
responsible for threading these in reverse computation order —
no autograd engine, no ``.backward()`` calls.

Each function returns ``(grad_inputs..., grad_weights...)`` where
``grad_inputs`` are the gradients w.r.t. the forward inputs (in the same
order as the forward signature) and ``grad_weights`` are the gradients
w.r.t. the weight tensors (always fp32, for accumulation into the
``main_grad`` buffer — see ``constraint.md`` §FP32 precision specification).

The math in each function is designed to be bitwise-identical to the
corresponding backward of the ref's custom autograd Functions
(``_LinearFn``, ``_RMSNormFn``, ``_EmbeddingFn``) when given the same
intermediate tensors.
"""

from __future__ import annotations

import torch

from training_engine_tensor.config import (
    HEAD_DIM,
    NUM_HEADS,
    NUM_KV_HEADS,
    NORM_EPS,
)


# ── Linear backward (y = x @ W.T) ──────────────────────────────────────────


def linear_backward(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward of ``y = x @ weight.T``.

    Args:
        grad_out: Gradient w.r.t. output, shape ``[*, out_dim]``.
        x: Forward input, shape ``[*, in_dim]``.
        weight: Shape ``[out_dim, in_dim]``.

    Returns:
        ``(grad_in, grad_weight)`` where ``grad_in`` has the same shape
        as ``x`` (same dtype) and ``grad_weight`` is fp32 (same shape as
        ``weight``).
    """
    if grad_out.shape[-1] != weight.shape[0]:
        raise ValueError(
            f"linear_backward: grad_out last dim {grad_out.shape[-1]} != "
            f"weight out_dim {weight.shape[0]}"
        )

    # dX = grad_out @ weight
    grad_in = torch.matmul(grad_out, weight)

    # dW = grad_out.T @ x (reshaped to 2D)
    n = weight.shape[0]
    k = weight.shape[1]
    g2 = grad_out.reshape(-1, n)
    x2 = x.reshape(-1, k)
    grad_weight = torch.matmul(g2.transpose(0, 1), x2)

    return grad_in, grad_weight


# ── RMSNorm backward ────────────────────────────────────────────────────────


def rms_norm_backward(
    grad_out: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    eps: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward of :func:`training_engine_tensor.forward.rms_norm`.

    The dgrad path replays the forward through autograd (same as the
    ref's ``_RMSNormFn.backward``) to stay bitwise-identical.  The
    wgrad is computed in fp32 from the closed form.

    Returns ``(grad_in, grad_weight)``.
    """
    eps_val = eps if eps is not None else NORM_EPS

    # dgrad: replay through autograd for bitwise match with ref.
    # Use the hidden dtype (float64 for tests, float32 for bf16 production).
    dtype = hidden.dtype
    with torch.enable_grad():
        xd = hidden.detach().requires_grad_(True)
        wd = weight.detach().requires_grad_(True)

        xd_f = xd.to(dtype=dtype)
        r = torch.rsqrt(xd_f.pow(2).mean(dim=-1, keepdim=True) + eps_val)
        out = (xd_f * r).to(dtype=xd.dtype) * wd

        grad_in, grad_weight_ref = torch.autograd.grad(out, (xd, wd), grad_out)

    # wgrad: closed-form (same dtype as grad_out)
    hidden_f = hidden.to(dtype=dtype)
    normed = hidden_f * torch.rsqrt(
        hidden_f.pow(2).mean(dim=-1, keepdim=True) + eps_val
    )
    grad_weight_fp32 = (grad_out * normed).reshape(-1, weight.shape[0]).sum(0)

    return grad_in, grad_weight_fp32


# ── SwiGLU intermediate backward (silu(gate) * up) ─────────────────────────


def silu_swiglu_intermediate_backward(
    grad_out: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward of ``silu(gate) * up``.

    ``silu(x) = sigmoid(x) * x``.
    ``d(silu(x))/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))``.

    Uses the input dtype for the activation gradient computation.
    """
    dtype = grad_out.dtype
    gate_f = gate.to(dtype=dtype)
    up_f = up.to(dtype=dtype)
    grad_out_f = grad_out.to(dtype=dtype)

    sig = torch.sigmoid(gate_f)
    silu = sig * gate_f
    d_silu = sig * (1.0 + gate_f * (1.0 - sig))

    grad_gate = grad_out_f * d_silu * up_f
    grad_up = grad_out_f * silu

    return grad_gate.to(gate.dtype), grad_up.to(up.dtype)


# ── Embedding backward ──────────────────────────────────────────────────────


def embedding_backward(
    grad_out: torch.Tensor,
    token_ids: torch.Tensor,
    vocab_size: int | None = None,
) -> torch.Tensor:
    """Backward of embedding lookup.

    ``grad_out`` shape ``[B, S, H]``.
    ``token_ids`` shape ``[B, S]``.

    Returns ``[vocab_size, H]`` fp32 gradient.
    """
    max_id = token_ids.max().item()
    if vocab_size is not None and max_id >= vocab_size:
        raise ValueError(
            f"embedding_backward: token id {max_id} out of vocab {vocab_size}"
        )
    V = vocab_size if vocab_size is not None else int(max_id) + 1
    H = grad_out.shape[-1]

    flat_ids = token_ids.reshape(-1)
    flat_grad = grad_out.reshape(-1, H)
    grad_embedding = torch.zeros(V, H, dtype=flat_grad.dtype, device=grad_out.device)
    grad_embedding.index_add_(0, flat_ids, flat_grad)
    return grad_embedding


# ── RoPE backward ───────────────────────────────────────────────────────────


def apply_rope_backward(grad_out: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Backward of :func:`training_engine_tensor.forward.apply_rope`.

    ``grad_out`` shape ``[B, S, H, D]``.
    ``freqs`` shape ``[S, D]``.

    RoPE is linear in the input tensor, so the backward is the same as
    the forward applied to ``grad_out``.
    """
    S_freqs = freqs.shape[0]
    S_grad = grad_out.shape[1]
    if S_freqs < S_grad:
        raise ValueError(
            f"apply_rope_backward: freqs length {S_freqs} < grad seq {S_grad}"
        )
    if freqs.shape[-1] != grad_out.shape[-1]:
        raise ValueError(
            f"apply_rope_backward: freqs head_dim {freqs.shape[-1]} != "
            f"grad_out head_dim {grad_out.shape[-1]}"
        )

    # RoPE backward: dL/dt = dL/dout * cos + rotated * sin
    # where rotated = cat([dL/dout[..., half:], -dL/dout[..., :half]], dim=-1)
    # (the inverse of the forward rotate)
    cos_ = torch.cos(freqs).to(grad_out.dtype)[None, :, None, :]
    sin_ = torch.sin(freqs).to(grad_out.dtype)[None, :, None, :]
    half = grad_out.shape[-1] // 2
    x1 = grad_out[..., :half]
    x2 = grad_out[..., half:]
    rotated = torch.cat((x2, -x1), dim=-1)
    return grad_out * cos_ + rotated * sin_


# ── QKV projection backward ─────────────────────────────────────────────────


def project_qkv_backward(
    grad_q: torch.Tensor,
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward of :func:`training_engine_tensor.forward.project_qkv`.

    The forward uses GQA interleaving: ``[Q_heads | K | V]`` per KV group.
    The backward must reverse this layout.

    Returns ``(grad_hidden, grad_weight)``.
    """
    # Derive head dimensions from tensor shapes instead of config constants
    # so the function works with arbitrary test inputs.
    _, _, H_q_raw, d = grad_q.shape
    _, _, nkv, _ = grad_k.shape
    nq_per_kv = H_q_raw // nkv

    if nq_per_kv * nkv != H_q_raw:
        raise ValueError(
            f"project_qkv_backward: Q heads {H_q_raw} not divisible by KV heads {nkv}"
        )
    if grad_k.shape[-1] != d:
        raise ValueError(
            f"project_qkv_backward: K head_dim {grad_k.shape[-1]} != {d}"
        )
    if grad_v.shape[-1] != d:
        raise ValueError(
            f"project_qkv_backward: V head_dim {grad_v.shape[-1]} != {d}"
        )

    leading = hidden.shape[:-1]
    H = hidden.shape[-1]

    # Reconstruct the full projected gradient in the interleaved layout
    grad_q_flat = grad_q.reshape(*leading, H_q_raw, d)
    grad_k_flat = grad_k.reshape(*leading, nkv, d)
    grad_v_flat = grad_v.reshape(*leading, nkv, d)

    # Build the per-group gradient
    group_width = (nq_per_kv + 2) * d
    grad_grouped = torch.zeros(*leading, nkv, group_width, dtype=grad_q.dtype, device=grad_q.device)
    grad_grouped[..., : nq_per_kv * d] = grad_q_flat.reshape(*leading, nkv, nq_per_kv * d)
    grad_grouped[..., nq_per_kv * d: nq_per_kv * d + d] = grad_k_flat
    grad_grouped[..., nq_per_kv * d + d:] = grad_v_flat

    qkv_out_dim = grad_q.shape[-2] * d + 2 * nkv * d
    grad_projected = grad_grouped.reshape(*leading, qkv_out_dim)

    # d hidden = grad_projected @ weight
    grad_hidden = torch.matmul(grad_projected, weight)

    # d weight = grad_projected.T @ hidden
    g2 = grad_projected.reshape(-1, grad_projected.shape[-1])
    h2 = hidden.reshape(-1, H)
    grad_weight = torch.matmul(g2.transpose(0, 1), h2)

    return grad_hidden, grad_weight


def _check_head_dims(grad_q, grad_k, grad_v, nkv, nq_per_kv, d):
    msg_parts = []
    if grad_q.shape[-1] != d:
        msg_parts.append(f"grad_q head_dim {grad_q.shape[-1]} != {d}")
    if grad_k.shape[-1] != d:
        msg_parts.append(f"grad_k head_dim {grad_k.shape[-1]} != {d}")
    if grad_v.shape[-1] != d:
        msg_parts.append(f"grad_v head_dim {grad_v.shape[-1]} != {d}")
    if grad_q.shape[-2] != NUM_HEADS:
        msg_parts.append(f"grad_q num_heads {grad_q.shape[-2]} != {NUM_HEADS}")
    if grad_k.shape[-2] != nkv:
        msg_parts.append(f"grad_k kv_heads {grad_k.shape[-2]} != {nkv}")
    if grad_v.shape[-2] != nkv:
        msg_parts.append(f"grad_v kv_heads {grad_v.shape[-2]} != {nkv}")
    if nq_per_kv * nkv != NUM_HEADS:
        msg_parts.append(
            f"nq_per_kv {nq_per_kv} * nkv {nkv} != NUM_HEADS {NUM_HEADS}"
        )
    if msg_parts:
        raise ValueError("project_qkv_backward: " + "; ".join(msg_parts))


# ── GQA attention backward (math fallback) ──────────────────────────────────


def gqa_attention_backward(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    allow_math_fallback: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward of :func:`training_engine_tensor.forward._gqa_attention`.

    When ``allow_math_fallback`` is ``True``, uses the explicit softmax
    backward (same as the math fallback forward).  On CUDA without the
    fallback flag, raises ``NotImplementedError`` (the flash_attn
    backward is handled by the flash_attn library's own autograd Function
    and is not accessible as a standalone primitive).

    Returns ``(grad_q, grad_k, grad_v)``.
    """
    # Validate shapes
    if grad_out.shape != q.shape:
        raise ValueError(
            f"gqa_attention_backward: grad_out shape {grad_out.shape} != "
            f"q shape {q.shape}"
        )

    if not allow_math_fallback:
        raise NotImplementedError(
            "gqa_attention_backward: flash_attn backward is not available as a "
            "standalone primitive.  Pass allow_math_fallback=True for the CPU "
            "math fallback path."
        )

    B, S, H_q, D = q.shape
    _, _, H_kv, _ = k.shape
    dtype = grad_out.dtype

    # Repeat KV heads to match Q heads
    rep = H_q // H_kv
    if rep > 1:
        k_exp = k.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, S, H_q, D)
        v_exp = v.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, S, H_q, D)
    else:
        k_exp = k
        v_exp = v

    # Permute to [B, H, S, D] for standard attention backward
    q_attn = q.permute(0, 2, 1, 3)  # [B, H_q, S, D]
    k_attn = k_exp.permute(0, 2, 1, 3)  # [B, H_q, S, D]
    v_attn = v_exp.permute(0, 2, 1, 3)  # [B, H_q, S, D]
    grad_out_attn = grad_out.permute(0, 2, 1, 3)  # [B, H_q, S, D]

    scale = D ** 0.5
    causal_mask = torch.triu(
        torch.ones(S, S, device=q.device, dtype=torch.bool), diagonal=1
    )

    # Forward re-compute (no grad needed)
    with torch.no_grad():
        scores = torch.matmul(q_attn.to(dtype=dtype), k_attn.to(dtype=dtype).transpose(-2, -1)) / scale
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)  # [B, H_q, S, S]

    # d(loss) / d(attn_weights) = grad_out @ v^T
    d_attn = torch.matmul(grad_out_attn.to(dtype=dtype), v_attn.to(dtype=dtype).transpose(-2, -1))  # [B, H_q, S, S]

    # Softmax backward
    d_scores = attn_weights * (d_attn - (d_attn * attn_weights).sum(dim=-1, keepdim=True))
    d_scores = d_scores.masked_fill(causal_mask, 0.0)
    d_scores = d_scores / scale

    # dQ = d_scores @ k
    grad_q_attn = torch.matmul(d_scores.to(q_attn.dtype), k_attn)

    # dK = d_scores^T @ q
    dk_full = torch.matmul(d_scores.transpose(-2, -1).to(k_attn.dtype), q_attn)
    # dV = attn_weights^T @ grad_out
    dv_full = torch.matmul(attn_weights.transpose(-2, -1).to(v_attn.dtype), grad_out_attn)

    # Permute back to [B, S, H, D]
    grad_q_out = grad_q_attn.permute(0, 2, 1, 3)
    dk_full = dk_full.permute(0, 2, 1, 3)
    dv_full = dv_full.permute(0, 2, 1, 3)

    # Reduce KV gradients if repeated
    if rep > 1:
        grad_k_out = dk_full.reshape(B, S, H_kv, rep, D).sum(dim=-2)
        grad_v_out = dv_full.reshape(B, S, H_kv, rep, D).sum(dim=-2)
    else:
        grad_k_out = dk_full
        grad_v_out = dv_full

    return grad_q_out, grad_k_out, grad_v_out


# ── Cross-entropy loss backward (FP32) ──────────────────────────────────────


def cross_entropy_backward(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Backward of cross-entropy loss w.r.t. logits.

    Returns the gradient w.r.t. logits in fp32 (``[B, S, V]``).
    This is ``softmax(logits) - one_hot(labels)``, masked by ``loss_mask``.
    """
    B, S, V = logits.shape
    logits_f32 = logits.float().reshape(-1, V)
    labels_flat = labels.reshape(-1)

    logits_max = logits_f32.max(dim=-1, keepdim=True).values
    logits_stable = logits_f32 - logits_max
    softmax = logits_stable.exp() / logits_stable.exp().sum(dim=-1, keepdim=True)

    grad_logits = softmax.clone()
    grad_logits[torch.arange(grad_logits.shape[0], device=logits.device), labels_flat] -= 1.0

    mask = loss_mask.reshape(-1, 1).float()
    grad_logits = grad_logits * mask
    return grad_logits.reshape(B, S, V)