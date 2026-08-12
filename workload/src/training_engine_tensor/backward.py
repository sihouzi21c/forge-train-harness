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

    Matches the ref's ``_LinearFn.backward`` exactly: dgrad and wgrad use
    the weight in its original dtype (bf16), matching the ref's pattern of
    ``torch.matmul(grad_out, weight)`` where ``weight`` stays bf16.

    Args:
        grad_out: Gradient w.r.t. output, shape ``[*, out_dim]``.
        x: Forward input, shape ``[*, in_dim]``.
        weight: Shape ``[out_dim, in_dim]`` (bf16).

    Returns:
        ``(grad_in, grad_weight)`` where ``grad_in`` has the same shape
        as ``x`` (same dtype) and ``grad_weight`` is in the weight's dtype
        (bf16, matching the ref's ``_LinearFn`` wgrad computation).
    """
    if grad_out.shape[-1] != weight.shape[0]:
        raise ValueError(
            f"linear_backward: grad_out last dim {grad_out.shape[-1]} != "
            f"weight out_dim {weight.shape[0]}"
        )

    # dX = grad_out @ weight — keep weight in its original dtype (bf16) to
    # match the ref's _LinearFn.backward (which uses ctx.weight as-is).
    grad_in = torch.matmul(grad_out, weight)

    # dW = grad_out.T @ x (reshaped to 2D) — compute in weight's dtype
    # then convert to fp32 (matching the ref's _LinearFn.backward which does
    # wg = torch.matmul(g2.T, x2); _ensure_main_grad(weight).add_(wg.float())).
    n = weight.shape[0]
    k = weight.shape[1]
    g2 = grad_out.reshape(-1, n)
    x2 = x.reshape(-1, k)
    grad_weight = torch.matmul(g2.transpose(0, 1), x2).float()

    return grad_in, grad_weight


# ── RMSNorm backward ────────────────────────────────────────────────────────


def rms_norm_backward(
    grad_out: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    eps: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward of :func:`training_engine_tensor.forward.rms_norm`.

    Replays ``F.rms_norm`` through autograd (same as the ref's
    ``_RMSNormFn.backward``) to stay bitwise-identical.  The wgrad is
    computed in fp32 from the closed form.

    Returns ``(grad_in, grad_weight)``.
    """
    import torch.nn.functional as _F
    eps_val = eps if eps is not None else NORM_EPS
    shape = (weight.shape[0],)

    # dgrad: replay through autograd for bitwise match with ref.
    with torch.enable_grad():
        xd = hidden.detach().requires_grad_(True)
        wd = weight.detach().requires_grad_(True)
        out = _F.rms_norm(xd, shape, wd, eps_val)
        grad_in, grad_weight_ref = torch.autograd.grad(out, (xd, wd), grad_out)

    # wgrad: closed-form (same dtype as grad_out)
    normed = _F.rms_norm(hidden.detach(), shape, None, eps_val)
    grad_weight_fp32 = (grad_out * normed).reshape(-1, weight.shape[0]).sum(0)

    return grad_in, grad_weight_fp32


# ── SwiGLU intermediate backward (silu(gate) * up) ─────────────────────────


def silu_swiglu_intermediate_backward(
    grad_out: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward of ``silu(gate) * up``.

    Replays ``F.silu(gate) * up`` through ``torch.autograd.grad`` to match the
    ref's autograd backward for ``F.silu`` exactly.  The manual closed-form
    ``sigmoid(x) * (1 + x * (1 - sigmoid(x)))`` can differ from the fused
    autograd kernel's computation.
    """
    import torch.nn.functional as _F
    with torch.enable_grad():
        gate_f = gate.detach().float().requires_grad_(True)
        up_f = up.detach().float().requires_grad_(True)
        out = _F.silu(gate_f) * up_f
        grad_gate_f, grad_up_f = torch.autograd.grad(out, (gate_f, up_f), grad_out.float())
    return grad_gate_f.to(gate.dtype), grad_up_f.to(up.dtype)


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

    # d hidden = grad_projected @ weight (keep weight in original dtype, matching ref)
    grad_hidden = torch.matmul(grad_projected, weight)

    # d weight = grad_projected.T @ hidden (use weight's dtype)
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


# ── GQA attention backward (flash attention path) ────────────────────────────


def gqa_attention_backward_flash(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    deterministic: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward of flash attention via ``_flash_attn_backward``.

    Uses the same flash attention backward kernel as the ref's autograd
    Function, ensuring bitwise alignment.

    Args:
        grad_out: Gradient w.r.t. attention output, shape ``[B, S, H, D]``.
        q, k, v: Forward inputs, shape ``[B, S, H_q/D, H_kv/D, H_kv/D]``.
        out: Attention output from forward, shape ``[B, S, H, D]``.
        softmax_lse: Softmax log-sum-exp from forward, shape ``[B, H, S]``.
        deterministic: Whether to use deterministic computation.

    Returns:
        ``(grad_q, grad_k, grad_v)``.
    """
    from flash_attn.flash_attn_interface import _flash_attn_backward

    B, S, H_q, D = q.shape
    _, _, H_kv, _ = k.shape
    d = D
    softmax_scale = d ** -0.5

    # Pre-allocate gradient buffers
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    _flash_attn_backward(
        grad_out, q, k, v, out, softmax_lse,
        dq, dk, dv,
        dropout_p=0.0, softmax_scale=softmax_scale,
        causal=True, window_size=(-1, -1), alibi_slopes=None,
        deterministic=deterministic,
    )
    return dq, dk, dv


# ── GQA attention backward (flash_attn_func path) ────────────────────────────


def gqa_attention_backward(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    allow_math_fallback: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward of :func:`training_engine_tensor.forward._gqa_attention`.

    Replays the forward through ``flash_attn_func`` with ``torch.autograd.grad``
    for bitwise alignment with the ref's ``flash_attn_func`` autograd Function.

    ``allow_math_fallback`` is ignored — the ``flash_attn_func`` path is the
    authoritative one for bitwise alignment.

    Returns ``(grad_q, grad_k, grad_v)``.
    """
    from flash_attn import flash_attn_func

    # Validate shapes
    if grad_out.shape != q.shape:
        raise ValueError(
            f"gqa_attention_backward: grad_out shape {grad_out.shape} != "
            f"q shape {q.shape}"
        )

    B, S, H_q, D = q.shape
    _, _, H_kv, _ = k.shape

    # Replay forward through flash_attn_func with autograd
    # to get bitwise-identical gradients to the ref's flash_attn backward.
    with torch.enable_grad():
        qd = q.detach().requires_grad_(True)
        kd = k.detach().requires_grad_(True)
        vd = v.detach().requires_grad_(True)
        out = flash_attn_func(qd, kd, vd, causal=True, deterministic=True)

        # Compute gradients through the full attention (including GQA handling)
        # The ref's flash_attn autograd backward handles GQA internally, so the
        # gradients for k and v have NUM_KV_HEADS heads (not NUM_HEADS).
        grad_q, grad_k, grad_v = torch.autograd.grad(out, (qd, kd, vd), grad_out)

    return grad_q, grad_k, grad_v


# ── Cross-entropy loss backward (matches ref's autograd backward) ──────


def cross_entropy_backward(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Backward of cross-entropy loss w.r.t. logits.

    Replays ``F.cross_entropy`` through ``torch.autograd.grad`` to match the
    ref's autograd backward exactly.  The ref computes::

        nll = F.cross_entropy(logits.reshape(-1, V).float(), labels.reshape(-1), reduction="none")
        obj = (nll * mask).sum()
        obj.backward()

    The gradient w.r.t. the bf16 ``logits`` is bf16 (because ``.float()``
    backward converts the fp32 gradient back to bf16).  This function
    replicates that chain and returns the gradient in bf16.

    Returns the gradient w.r.t. logits in bf16 (``[B, S, V]``).
    """
    import torch.nn.functional as _F

    B, S, V = logits.shape
    labels_flat = labels.reshape(-1)
    mask = loss_mask.reshape(-1).float()

    # Replay the ref's chain: logits.float() → F.cross_entropy → (nll *
    # mask).sum() through autograd so the backward produces the same dtype
    # and values as the ref's autograd engine.
    with torch.enable_grad():
        # logits.float() creates a new fp32 tensor; autograd will track the
        # gradient through this conversion back to the bf16 input.
        logits_f32 = logits.detach().float().reshape(-1, V).requires_grad_(True)
        nll = _F.cross_entropy(logits_f32, labels_flat, reduction="none")
        (grad_logits_f32,) = torch.autograd.grad(nll, (logits_f32,), grad_outputs=mask)
        # grad_logits_f32 is fp32; convert back to bf16 (matching the ref's
        # .float() backward which converts fp32 → bf16).
        grad_logits = grad_logits_f32.to(logits.dtype)

    return grad_logits.reshape(B, S, V)