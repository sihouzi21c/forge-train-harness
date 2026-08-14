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

    return c