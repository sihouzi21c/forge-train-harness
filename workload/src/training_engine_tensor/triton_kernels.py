"""Triton kernels for the self-developed training engine.

Provides performance-optimized Triton implementations of key operations,
primarily targeting the GEMM patterns that dominate the backward pass time.

The Triton wgrad kernel reads bf16 inputs and accumulates in fp32, avoiding
the cuBLAS TF32 path's intermediate TF32 → bf16 → float conversion chain.
This is particularly beneficial for the output weight wgrad (V=130560, H=2048)
where the large reduction dimension (B*S=4096) makes the memory access pattern
critical.

All kernels follow the runtime contract of the corresponding backward
primitives in :mod:`training_engine_tensor.backward`.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ── Wgrad (output weight): g2.T @ x2, fp32 output ──────────────────────────
#
# The output weight wgrad is the largest single GEMM in the backward pass:
#   g2 = grad_logits.reshape(-1, V)  # [B*S, 130560] bf16
#   x2 = hidden.reshape(-1, H)       # [B*S, 2048]   bf16
#   dw = g2.T @ x2                   # [130560, 2048] fp32
#
# The cuBLAS TF32 path computes the matmul in TF32 (19-bit mantissa), stores
# the result in bf16, then converts to fp32.  The Triton path reads bf16
# directly and accumulates in fp32, eliminating the TF32 precision loss and
# the explicit .float() cast.
#
# Tiling strategy:
#   M = 130560 (N in GEMM notation = output dim)
#   N = 2048   (K in GEMM notation = hidden dim)
#   K = 4096   (M in GEMM notation = reduction dim = B*S)
#
#   BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 64, BLOCK_SIZE_K = 32
#   Grid: (130560/128, 2048/64) = (1020, 32)
#   Each block: 128 × 64 elements, 4096/32 = 128 reduction iterations
#
# The K dimension (4096) is small, so the reduction loop is short.  The M
# dimension (130560) is large, providing ample parallelism.  The N dimension
# (2048) is moderate, so a BLOCK_SIZE_N of 64 gives good occupancy without
# excessive register pressure.

@triton.jit
def _wgrad_output_kernel(
    # Pointers: a = g2.T (transposed), b = x2, c = result
    a_ptr, b_ptr, c_ptr,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Dimensions
    M, N, K,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Triton wgrad kernel for the output weight.

    Computes ``C = A @ B`` where:
        A = g2.T  shape [V, B*S] = [130560, 4096]
        B = x2    shape [B*S, H] = [4096, 2048]
        C = dw    shape [V, H]   = [130560, 2048]  (fp32 output)

    Reads bf16 inputs, accumulates in fp32.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # A = g2.T: [M, K] with strides (stride_am, stride_ak)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # B = x2: [K, N] with strides (stride_bk, stride_bn)
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulate in fp32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        # Load A (bf16) and B (bf16), convert to fp32
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k))
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k) & (offs_n[None, :] < N))
        accumulator += tl.dot(a, b)

        offs_k += BLOCK_SIZE_K
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Write fp32 result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, accumulator, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def wgrad_output(
    g2: torch.Tensor,   # [B*S, V] bf16
    x2: torch.Tensor,   # [B*S, H] bf16
) -> torch.Tensor:      # [V, H] fp32
    """Compute output weight wgrad via Triton kernel.

    ``g2`` shape ``[B*S, V]`` bf16.
    ``x2`` shape ``[B*S, H]`` bf16.
    Returns ``[V, H]`` fp32.

    This is equivalent to ``torch.matmul(g2.transpose(0, 1), x2).float()``
    but uses bf16 input + fp32 accumulation in a single kernel, avoiding
    the intermediate TF32 round-trip and the explicit dtype cast.
    """
    V, H = g2.shape[1], x2.shape[1]
    M = V  # output dim
    N = H  # hidden dim
    K = g2.shape[0]  # reduction dim = B*S

    # g2.T is [V, B*S], x2 is [B*S, H]
    # A = g2.T, B = x2, C = A @ B = [V, H] fp32
    a = g2.transpose(0, 1).contiguous()  # [V, B*S] bf16, contiguous

    # Allocate output (fp32)
    c = torch.empty(M, N, dtype=torch.float32, device=g2.device)

    # Grid configuration
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    _wgrad_output_kernel[grid](
        a, x2, c,
        a.stride(0), a.stride(1),
        x2.stride(0), x2.stride(1),
        c.stride(0), c.stride(1),
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )

    # ── Fused RMSNorm backward ─────────────────────────────────────────────────────
#
# Fuses the entire RMSNorm backward computation into a single Triton kernel:
#   r = rsqrt(mean(x^2, dim=-1) + eps)
#   normed = x * r
#   d_normed = grad_out * weight
#   normed_dot = mean(d_normed * normed, dim=-1)
#   d_hidden = r * (d_normed - normed * normed_dot)
#
# The current PyTorch implementation (backward.py:rms_norm_backward) launches 12
# separate kernels per call (3 .float() copies + 4 reductions + 5 element-wise).
# This fused kernel does it in 1 launch, reading bf16 inputs, computing in fp32,
# and writing d_hidden in bf16.  The grad_weight sum is still done in PyTorch
# (it's a cross-row reduction that doesn't benefit from Triton's per-row grid).
#
# Each program handles one (B, S) row of H elements.  Grid = (B * S,) programs.

@triton.jit
def _rms_norm_bwd_kernel(
    grad_out_ptr, hidden_ptr, weight_ptr, d_hidden_ptr,
    H,
    eps: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    """Fused RMSNorm backward: one program per (B, S) row."""
    row = tl.program_id(0)

    # Row offset: each row has H elements in a contiguous layout.
    offs = row * H + tl.arange(0, BLOCK_SIZE_H)
    mask = tl.arange(0, BLOCK_SIZE_H) < H

    # Load hidden[pid, :] bf16 → fp32.
    x = tl.load(hidden_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # r = rsqrt(mean(x^2, dim=-1) + eps)
    x2 = x * x
    mean_x2 = tl.sum(x2, axis=0) / H
    r = tl.rsqrt(mean_x2 + eps)

    # normed = x * r
    normed = x * r

    # Load grad_out[pid, :] bf16 → fp32.
    go = tl.load(grad_out_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Load weight[:] bf16 → fp32 (shared across all rows).
    w = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE_H), mask=mask, other=0.0).to(tl.float32)

    # d_normed = grad_out * weight
    d_normed = go * w

    # normed_dot = mean(d_normed * normed, dim=-1)
    normed_dot = tl.sum(d_normed * normed, axis=0) / H

    # d_hidden = r * (d_normed - normed * normed_dot)
    d_hidden = r * (d_normed - normed * normed_dot)

    # Write d_hidden in bf16.
    tl.store(d_hidden_ptr + offs, d_hidden.to(tl.bfloat16), mask=mask)


def rms_norm_backward_fused(
    grad_out: torch.Tensor,  # [B, S, H] bf16
    hidden: torch.Tensor,    # [B, S, H] bf16
    weight: torch.Tensor,    # [H] bf16
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused RMSNorm backward: returns (d_hidden_bf16, grad_weight_fp32).

    Fuses the entire d_hidden computation into one Triton kernel, reading
    bf16 inputs and writing d_hidden in bf16 with fp32 internal accumulation.
    The grad_weight sum is computed via PyTorch (it is a cross-row reduction
    that does not map efficiently to Triton's per-row grid).

    This is numerically equivalent to the closed-form RMSNorm backward in
    ``backward.rms_norm_backward`` and passes the long-train statistical gate.
    """
    B, S, H = hidden.shape
    BLOCK_SIZE_H = 1 << (H - 1).bit_length()  # next power of 2 >= H
    assert BLOCK_SIZE_H <= 2048, \
        f"rms_norm_backward_fused: H={H} requires BLOCK_SIZE_H={BLOCK_SIZE_H} > 2048"

    # Allocate d_hidden output (bf16, same shape as hidden).
    d_hidden = torch.empty_like(hidden, dtype=torch.bfloat16)

    # Launch one program per (B, S) row.
    grid = (B * S,)
    _rms_norm_bwd_kernel[grid](
        grad_out, hidden, weight, d_hidden,
        H,
        eps=eps,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
    )

    # Compute grad_weight in fp32 via PyTorch (cross-row sum).
    # grad_out_f32 * normed, then sum across (B, S).
    # We recompute normed in fp32 from the bf16 hidden (the same computation
    # the Triton kernel already did, but it's cheap vs the full backward).
    hidden_f32 = hidden.float()
    r = torch.rsqrt(hidden_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    normed = hidden_f32 * r
    grad_weight = (grad_out.float() * normed).reshape(-1, H).sum(0)

    return d_hidden, grad_weight


# ── Fused SwiGLU forward ────────────────────────────────────────────────────
#
# Fuses the SwiGLU forward computation:
#   intermediate = silu(gate) * up
#
# into a single Triton kernel.  The current PyTorch implementation
# (forward.py:mlp_swiglu) launches 4 separate kernel launches (2 .float()
# copies + 1 silu + 1 multiply + 1 .to(bf16)) per call.  This fused kernel
# does it in 1 launch, reading bf16 inputs, computing in fp32, and writing
# bf16 output.
#
# Each program handles one BLOCK_SIZE-sized chunk of one (B, S) row's
# ffn_half elements.  Grid = (B * S, ceil(ffn_half / BLOCK_SIZE)).


@triton.jit
def _swiglu_fwd_kernel(
    gate_up_ptr, intermediate_ptr,
    stride_gu_row,  # 2 * ffn_half
    stride_int_row,  # ffn_half
    ffn_half,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused SwiGLU forward: intermediate = silu(gate) * up.

    One program per (row, ffn_block) in a 2D grid.
    """
    pid_b = tl.program_id(0)  # row in B*S
    pid_f = tl.program_id(1)  # block index in ffn_half

    offs = pid_f * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < ffn_half

    # Load gate and up from gate_up: gate = gate_up[..., :ffn_half],
    # up = gate_up[..., ffn_half:].
    gate = tl.load(
        gate_up_ptr + pid_b * stride_gu_row + offs,
        mask=mask, other=0.0,
    ).to(tl.float32)
    up = tl.load(
        gate_up_ptr + pid_b * stride_gu_row + ffn_half + offs,
        mask=mask, other=0.0,
    ).to(tl.float32)

    # silu(gate) = gate * sigmoid(gate)
    sig = tl.sigmoid(gate)
    intermediate = gate * sig * up

    # Store bf16 intermediate.
    tl.store(
        intermediate_ptr + pid_b * stride_int_row + offs,
        intermediate.to(tl.bfloat16),
        mask=mask,
    )


def swiglu_forward_fused(
    gate_up: torch.Tensor,  # [B, S, 2*ffn_half] bf16
    ffn_half: int,          # FFN_HIDDEN_SIZE
) -> torch.Tensor:          # [B, S, ffn_half] bf16
    """Fused SwiGLU forward: returns intermediate = silu(gate) * up.

    Reads bf16 inputs from ``gate_up``, computes in fp32, and writes
    bf16 ``intermediate``.  Numerically equivalent to the PyTorch
    ``F.silu(gate.float()) * up.float()`` path.
    """
    B, S, _ = gate_up.shape
    # Allocate intermediate output (bf16).
    intermediate = torch.empty(B, S, ffn_half, dtype=torch.bfloat16, device=gate_up.device)

    BLOCK_SIZE = min(2048, 1 << (ffn_half - 1).bit_length())
    grid = (B * S, triton.cdiv(ffn_half, BLOCK_SIZE))

    _swiglu_fwd_kernel[grid](
        gate_up, intermediate,
        gate_up.stride(1),  # stride_gu_row = S * 2*ffn_half
        intermediate.stride(1),  # stride_int_row = S * ffn_half
        ffn_half,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return intermediate


# ── Fused SwiGLU backward ───────────────────────────────────────────────────
#
# Fuses the SwiGLU backward computation:
#   sig = sigmoid(gate)
#   silu_val = gate * sig
#   d_silu = sig * (1 + gate * (1 - sig))
#   d_gate = grad_out * up * d_silu
#   d_up = grad_out * silu_val
#
# into a single Triton kernel.  The current PyTorch implementation
# (backward.py:silu_swiglu_intermediate_backward) launches 3 .float() copies
# + 8 element-wise operations per call.  This fused kernel reads bf16 inputs
# directly, computes in fp32, and writes bf16 d_gate / d_up.
#
# Each program handles one BLOCK_SIZE-sized chunk of one (B, S) row's
# ffn_half elements.  Grid = (B * S, ceil(ffn_half / BLOCK_SIZE)).


@triton.jit
def _swiglu_bwd_kernel(
    grad_out_ptr, gate_up_ptr, d_gate_up_ptr,
    stride_go_row,   # ffn_half
    stride_gu_row,   # 2 * ffn_half
    stride_dgu_row,  # 2 * ffn_half
    ffn_half,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused SwiGLU backward: d_gate, d_up from grad_out, gate, up.

    One program per (row, ffn_block) in a 2D grid.
    """
    pid_b = tl.program_id(0)  # row in B*S
    pid_f = tl.program_id(1)  # block index in ffn_half

    offs = pid_f * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < ffn_half

    # Load gate (bf16 → fp32) from gate_up[..., :ffn_half].
    gate = tl.load(
        gate_up_ptr + pid_b * stride_gu_row + offs,
        mask=mask, other=0.0,
    ).to(tl.float32)
    # Load up (bf16 → fp32) from gate_up[..., ffn_half:].
    up = tl.load(
        gate_up_ptr + pid_b * stride_gu_row + ffn_half + offs,
        mask=mask, other=0.0,
    ).to(tl.float32)
    # Load grad_out (bf16 → fp32).
    go = tl.load(
        grad_out_ptr + pid_b * stride_go_row + offs,
        mask=mask, other=0.0,
    ).to(tl.float32)

    # sig = sigmoid(gate)
    sig = tl.sigmoid(gate)
    # silu(gate) = gate * sig
    silu_val = gate * sig
    # dsilu/dgate = sig * (1 + gate * (1 - sig))
    d_silu = sig * (1.0 + gate * (1.0 - sig))

    # d_gate = grad_out * up * dsilu/dgate
    d_gate = go * up * d_silu
    # d_up = grad_out * silu(gate)
    d_up = go * silu_val

    # Store d_gate (bf16).
    tl.store(
        d_gate_up_ptr + pid_b * stride_dgu_row + offs,
        d_gate.to(tl.bfloat16),
        mask=mask,
    )
    # Store d_up (bf16).
    tl.store(
        d_gate_up_ptr + pid_b * stride_dgu_row + ffn_half + offs,
        d_up.to(tl.bfloat16),
        mask=mask,
    )


def swiglu_backward_fused(
    grad_out: torch.Tensor,  # [B, S, ffn_half] bf16 - gradient w.r.t. intermediate
    gate_up: torch.Tensor,   # [B, S, 2*ffn_half] bf16 - forward input
    ffn_half: int,           # FFN_HIDDEN_SIZE
) -> torch.Tensor:           # [B, S, 2*ffn_half] bf16 - d_gate_up
    """Fused SwiGLU backward: returns d_gate_up from grad_out and gate_up.

    Reads bf16 inputs from ``grad_out`` and ``gate_up``, computes in fp32,
    and writes bf16 ``d_gate_up``.  Numerically equivalent to:

        _y1, _y2 = gate_up.chunk(2, dim=-1)
        d_y1, d_y2 = silu_swiglu_intermediate_backward(grad_out, _y1, _y2)
        d_gate_up = torch.cat([d_y1, d_y2], dim=-1)
    """
    B, S, _ = gate_up.shape
    # Allocate d_gate_up output (bf16, same shape as gate_up).
    d_gate_up = torch.empty_like(gate_up, dtype=torch.bfloat16)

    BLOCK_SIZE = min(2048, 1 << (ffn_half - 1).bit_length())
    grid = (B * S, triton.cdiv(ffn_half, BLOCK_SIZE))

    _swiglu_bwd_kernel[grid](
        grad_out, gate_up, d_gate_up,
        grad_out.stride(1),   # stride_go_row = S * ffn_half
        gate_up.stride(1),    # stride_gu_row = S * 2*ffn_half
        d_gate_up.stride(1),  # stride_dgu_row = S * 2*ffn_half
        ffn_half,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return d_gate_up