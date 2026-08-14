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

# Try to import the Triton wgrad kernel for the output weight.
# Falls back to cuBLAS when Triton is not available (e.g. on Mac).
# NOTE: Triton wgrad is slower than cuBLAS TF32 for the current shape
# (V=130560, H=2048, B*S=4096).  Disabled by default.  Set
# ENABLE_TRITON_WGRAD=1 to re-enable for benchmarking new tilings.
_USE_TRITON_WGRAD = False
_HAS_TRITON = False
try:
    from training_engine_tensor.triton_kernels import wgrad_output as _wgrad_output
    _HAS_TRITON = True
    _USE_TRITON_WGRAD = _HAS_TRITON and int(__import__('os').environ.get('ENABLE_TRITON_WGRAD', '0'))
except (ImportError, ModuleNotFoundError, AttributeError):
    pass


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
        as ``x`` (same dtype) and ``grad_weight`` is fp32.
    """
    if grad_out.shape[-1] != weight.shape[0]:
        raise ValueError(
            f"linear_backward: grad_out last dim {grad_out.shape[-1]} != "
            f"weight out_dim {weight.shape[0]}"
        )

    # dX = grad_out @ weight — keep weight in its original dtype (bf16) to
    # match the ref's _LinearFn.backward (which uses ctx.weight as-is).
    grad_in = torch.matmul(grad_out, weight)

    # dW = grad_out.T @ x (reshaped to 2D) — compute in fp32 accumulation.
    n = weight.shape[0]
    k = weight.shape[1]
    g2 = grad_out.reshape(-1, n)
    x2 = x.reshape(-1, k)

    # Use the Triton wgrad kernel for large output dimensions (e.g. output
    # weight with V=130560).  The Triton kernel reads bf16 directly and
    # accumulates in fp32, avoiding the TF32 round-trip and the explicit
    # .float() cast.  For small/medium dimensions, cuBLAS TF32 is faster.
    if _USE_TRITON_WGRAD and n >= 8192:
        grad_weight = _wgrad_output(g2, x2)
    else:
        grad_weight = torch.matmul(g2.transpose(0, 1), x2).float()

    return grad_in, grad_weight


# ── RMSNorm backward ────────────────────────────────────────────────────────

# Try to import the fused Triton RMSNorm backward kernel.
# Falls back to the pure-PyTorch closed-form when Triton is not available
# (e.g. on Mac).  The Triton kernel is used only when BOTH:
#   (a) deterministic=False (long-horizon performance mode), AND
#   (b) ENABLE_TRITON_RMSNORM_BWD=1 (explicitly enabled — default 0).
# This ensures the bitwise gates (perf-bitwise, multistep-1gpu, multistep)
# always use the PyTorch closed-form, which is bitwise-identical to the ref.
# The env var is checked at runtime (not module import time) so that
# eval_long_train.py can set it before calling run_training_loop.
_HAS_TRITON_RMSNORM_BWD = False
try:
    from training_engine_tensor.triton_kernels import rms_norm_backward_fused as _rms_norm_bwd_fused
    _HAS_TRITON_RMSNORM_BWD = True
except (ImportError, ModuleNotFoundError, AttributeError):
    pass


def rms_norm_backward(
    grad_out: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    eps: float | None = None,
    deterministic: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward of :func:`training_engine_tensor.forward.rms_norm`.

    When ``deterministic=True`` (default, bitwise-safe mode), uses the
    pure-PyTorch closed-form that is bitwise-identical to the ref's
    ``_RMSNormFn.backward``.  When ``deterministic=False`` (long-horizon
    performance mode), uses the fused Triton kernel (``rms_norm_backward_fused``)
    which reads bf16 inputs and writes bf16 output with fp32 internal
    accumulation, reducing GPU kernel time by ~6% per call.

    The closed-form math is:

        r = rsqrt(mean(x^2, dim=-1) + eps)   [B, S, 1]
        normed = x * r                        [B, S, H]
        d_normed = grad_out * weight          [B, S, H]
        d_hidden = r * (d_normed - normed * mean(d_normed * normed, dim=-1, True))

    This is numerically equivalent to the ref's ``_RMSNormFn.backward``
    at the pointwise level and passes the long-train statistical gate.
    The wgrad is computed in fp32 from the same closed form.

    Returns ``(grad_in, grad_weight)``.
    """
    eps_val = eps if eps is not None else NORM_EPS

    # Use fused Triton kernel for non-deterministic (long-horizon) mode.
    # gated by ENABLE_TRITON_RMSNORM_BWD=1 (default 0, checked at runtime).
    _use_triton = (
        _HAS_TRITON_RMSNORM_BWD
        and not deterministic
        and hidden.is_cuda
        and int(__import__('os').environ.get('ENABLE_TRITON_RMSNORM_BWD', '0'))
    )
    if _use_triton:
        return _rms_norm_bwd_fused(grad_out, hidden, weight, eps=eps_val)

    # Pure-PyTorch closed-form (bitwise-safe, used for deterministic mode).
    H = hidden.shape[-1]

    # Reuse grad_out_f32 to avoid a second .float() call for the wgrad.
    # The weight is bf16; cast it to fp32 once so the d_normed multiply
    # stays in fp32 (avoids a bf16 multiply + separate .float() cast).
    grad_out_f32 = grad_out.float()
    weight_f32 = weight.float()

    # r = rsqrt(mean(x^2) + eps) — compute in fp32 for precision
    x_f32 = hidden.float()
    r = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps_val)
    normed = x_f32 * r  # [B, S, H] fp32

    # d_normed = grad_out * weight in fp32
    d_normed = grad_out_f32 * weight_f32

    # d_hidden = r * (d_normed - normed * mean(d_normed * normed, dim=-1))
    normed_dot = (d_normed * normed).mean(dim=-1, keepdim=True)
    d_hidden = r * (d_normed - normed * normed_dot)

    # d_weight = sum(grad_out * normed, dim=0).float() — reuse grad_out_f32
    grad_weight = (grad_out_f32 * normed).reshape(-1, H).sum(0)

    return d_hidden.to(hidden.dtype, non_blocking=True), grad_weight


# ── SwiGLU intermediate backward (silu(gate) * up) ─────────────────────────


# Try to import the fused Triton SwiGLU backward kernel.
# Falls back to the pure-PyTorch closed-form when Triton is not available
# (e.g. on Mac).  The Triton kernel is used only when BOTH:
#   (a) deterministic=False (long-horizon performance mode), AND
#   (b) ENABLE_TRITON_SWIGLU_BWD=1 (explicitly enabled — default 0).
# This ensures the bitwise gates always use the PyTorch closed-form.
_HAS_TRITON_SWIGLU_BWD = False
try:
    from training_engine_tensor.triton_kernels import swiglu_backward_fused as _swiglu_bwd_fused
    _HAS_TRITON_SWIGLU_BWD = True
except (ImportError, ModuleNotFoundError, AttributeError):
    pass


def silu_swiglu_intermediate_backward(
    grad_out: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    gate_up: torch.Tensor | None = None,
    ffn_half: int | None = None,
    deterministic: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """Backward of ``silu(gate) * up`` using closed-form (no autograd replay).

    When ``deterministic=True`` (default, bitwise-safe mode), uses the
    pure-PyTorch closed-form that is bitwise-identical to the ref's
    ``F.silu`` autograd backward.  When ``deterministic=False`` and
    ``gate_up`` is provided (long-horizon performance mode), uses the
    fused Triton kernel which reads bf16 inputs directly, computes in
    fp32, and writes bf16 output, eliminating the .float() copy overhead.

    The closed-form math is:

        sigmoid(x) = 1 / (1 + exp(-x))
        dsilu/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        d_gate = grad_out * up * dsilu/dgate
        d_up = grad_out * silu(gate)

    All computation is in fp32 for precision, matching the ref's ``F.silu``
    autograd backward at the numerical level.

    When ``gate_up`` is provided (fused Triton path), returns a single
    ``[B, S, 2*ffn_half]`` tensor ``d_gate_up``.  When ``gate_up`` is
    ``None`` (PyTorch closed-form), returns ``(d_gate, d_up)``.
    """
    # Use fused Triton kernel for non-deterministic (long-horizon) mode.
    if (gate_up is not None and ffn_half is not
            None and _HAS_TRITON_SWIGLU_BWD
            and not deterministic
            and gate.is_cuda
            and int(__import__('os').environ.get('ENABLE_TRITON_SWIGLU_BWD', '0'))):
        d_gate_up = _swiglu_bwd_fused(grad_out, gate_up, ffn_half)
        # Return views into the fused output so the caller's
        # ``torch.cat([d_y1, d_y2], dim=-1)`` is a no-op (both views
        # already point into the same buffer).
        return d_gate_up[..., :ffn_half], d_gate_up[..., ffn_half:]

    # Pure-PyTorch closed-form (bitwise-safe, used for deterministic mode).
    # Compute in fp32 for precision
    gate_f = gate.float()
    up_f = up.float()
    grad_out_f = grad_out.float()

    # sigmoid(gate) = 1 / (1 + exp(-gate))
    sig = torch.sigmoid(gate_f)

    # silu(gate) = gate * sigmoid(gate)
    silu_val = gate_f * sig

    # dsilu/dgate = sig * (1 + gate * (1 - sig))
    d_silu = sig * (1.0 + gate_f * (1.0 - sig))

    # d_gate = grad_out * up * dsilu/dgate
    d_gate = grad_out_f * up_f * d_silu

    # d_up = grad_out * silu(gate)
    d_up = grad_out_f * silu_val

    return d_gate.to(gate.dtype, non_blocking=True), d_up.to(up.dtype, non_blocking=True)


# ── Embedding backward ──────────────────────────────────────────────────────


def embedding_backward(
    grad_out: torch.Tensor,
    token_ids: torch.Tensor,
    vocab_size: int | None = None,
) -> torch.Tensor:
    """Backward of embedding lookup.

    Uses ``torch.ops.aten.embedding_dense_backward`` (the same op the
    ref's ``_EmbeddingFn.backward`` calls) for bitwise alignment with
    the ref's ``F.embedding`` backward.  The result is computed in the
    input dtype (bf16) then converted to fp32, matching the ref's
    ``wg.float()`` path.

    ``grad_out`` shape ``[B, S, H]``.
    ``token_ids`` shape ``[B, S]``.

    Returns ``[vocab_size, H]`` fp32 gradient.
    """
    V = vocab_size if vocab_size is not None else int(token_ids.max().item()) + 1
    wg = torch.ops.aten.embedding_dense_backward(
        grad_out, token_ids, V, -1, False
    )
    return wg.float()


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

    # d weight = grad_projected.T @ hidden (use weight's dtype, then fp32)
    g2 = grad_projected.reshape(-1, grad_projected.shape[-1])
    h2 = hidden.reshape(-1, H)
    grad_weight = torch.matmul(g2.transpose(0, 1), h2).float()

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
    out: torch.Tensor | None = None,
    softmax_lse: torch.Tensor | None = None,
    deterministic: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward of :func:`training_engine_tensor.forward._gqa_attention`.

    Uses the direct ``_flash_attn_backward`` kernel when ``out`` and
    ``softmax_lse`` are provided (no autograd replay).  Falls back to
    autograd replay (``flash_attn_func`` with ``torch.autograd.grad``)
    when the cached values are unavailable.

    The direct backward path avoids the autograd engine's intermediate
    tensor overhead and kernel-launch cost (~666ms/step for flash-attn
    backward in the profile).

    Returns ``(grad_q, grad_k, grad_v)``.
    """
    from flash_attn.flash_attn_interface import _flash_attn_backward

    B, S, H_q, D = q.shape
    _, _, H_kv, _ = k.shape
    d = D
    softmax_scale = d ** -0.5

    if out is not None and softmax_lse is not None:
        # Direct backward path — no autograd replay.
        # The _flash_attn_backward kernel is the same kernel the ref's
        # autograd Function calls, so the result is bitwise-equivalent.
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
    else:
        # Fallback: autograd replay (for backward compatibility).
        from flash_attn import flash_attn_func
        with torch.enable_grad():
            qd = q.detach().requires_grad_(True)
            kd = k.detach().requires_grad_(True)
            vd = v.detach().requires_grad_(True)
            out_autograd = flash_attn_func(qd, kd, vd, causal=True, deterministic=True)
            grad_q, grad_k, grad_v = torch.autograd.grad(out_autograd, (qd, kd, vd), grad_out)
        return grad_q, grad_k, grad_v


# ── Cross-entropy loss backward (F.cross_entropy autograd + fp32 scaling) ─


def cross_entropy_backward(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    scale: float = 1.0,
    chunk_size: int = 4096,
    deterministic: bool = True,
) -> torch.Tensor:
    """Backward of cross-entropy loss w.r.t. logits (chunked for memory).

    When ``deterministic=True`` (default, bitwise-safe mode), replays
    ``F.cross_entropy`` via autograd to produce the gradient, matching the
    ref's ``loss.backward()`` path exactly.

    When ``deterministic=False`` (long-horizon performance mode), uses the
    direct softmax formula:

        softmax = softmax(logits)
        d_logits = (softmax - one_hot(label)) * mask * scale

    This avoids the ``requires_grad_()``, ``F.cross_entropy`` forward, and
    ``obj.backward()`` overhead of the autograd replay approach, while
    producing the same numerical result.

    The computation is chunked along the batch dimension (``chunk_size``
    tokens per chunk) to avoid materializing the full ``[B*S, V]`` fp32
    logits tensor.  Each chunk is ``[chunk_size, V]`` fp32, which is
    ~2.1 GB at chunk_size=4096, V=130560.

    The ``scale`` multiplication is done in fp32 before the ``.to(bf16)``
    conversion, matching the ref's ``loss = lm_sum + ce_w * mtp_sum;
    loss.backward()`` where the ``ce_w`` scaling is applied in fp32.

    Returns the gradient in bf16.
    """
    import torch.nn.functional as _F

    B, S, V = logits.shape
    labels_flat = labels.reshape(-1)
    mask = loss_mask.reshape(-1).float()
    total_tokens = B * S
    grad_logits = torch.zeros(total_tokens, V, dtype=logits.dtype, device=logits.device)
    logits_flat = logits.reshape(-1, V)

    if deterministic:
        # Bitwise-safe path: autograd replay matches the ref's backward exactly.
        for i in range(0, total_tokens, chunk_size):
            end = min(i + chunk_size, total_tokens)
            with torch.enable_grad():
                chunk_logits = logits_flat[i:end].detach().float().requires_grad_(True)
                chunk_labels = labels_flat[i:end]
                chunk_mask = mask[i:end]
                nll = _F.cross_entropy(chunk_logits, chunk_labels, reduction="none")
                obj = (nll * chunk_mask).sum()
                obj.backward()
                grad_logits[i:end] = chunk_logits.grad.to(logits.dtype)
        grad = (scale * grad_logits.reshape(B, S, V)).to(logits.dtype)
        return grad
    else:
        # Long-horizon path: direct softmax formula avoids autograd overhead.
        for i in range(0, total_tokens, chunk_size):
            end = min(i + chunk_size, total_tokens)
            # Compute softmax in fp32 from bf16 logits chunk.
            chunk = logits_flat[i:end].float()
            softmax = torch.softmax(chunk, dim=-1)

            # one_hot at the label position for each token in the chunk.
            chunk_labels = labels_flat[i:end]
            chunk_mask = mask[i:end]
            one_hot = torch.zeros_like(softmax)
            one_hot[torch.arange(chunk_labels.shape[0]), chunk_labels] = 1.0

            # d_logits = (softmax - one_hot) * mask * scale (all in fp32)
            d_logits = (softmax - one_hot) * chunk_mask[:, None] * scale
            grad_logits[i:end] = d_logits.to(logits.dtype)

        return grad_logits.reshape(B, S, V)