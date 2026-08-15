# profile snapshot — profile-snapshot

step_time_ms=4503.736  mfu_e2e_standard=29.1955  profiled_steps=12  gpu_kernel_per_step_ms=2883.076  gpu_idle_per_step_ms=1620.660  gpu_memop_per_step_ms=44.257  cuda_api_per_step_ms=3148.486  os_runtime_per_step_ms=58202.478
# gpu_idle = step - gpu_kernel (valid only when compute kernels share one stream); cuda_api / os_runtime are CPU time, do NOT subtract from step_time.
world_size=2  micro_batch_size=4  seq_length=4096  grad_accum_steps=10
nsys_rep=/user/sunhaojun/.forge_train/77459cccb4da/.artifacts/runs/profile-snapshot-20260815-053709-d8b31d/ours/profile.nsys-rep

## top 15 GPU kernels by total time

| kernel | per_step_ms | pct | instances |
|---|---:|---:|---:|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | 566.589 | 19.70% | 2050 |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | 281.966 | 9.80% | 4157 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | 264.561 | 9.20% | 6391 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 223.580 | 7.80% | 2248 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | 213.476 | 7.40% | 2324 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | 111.808 | 3.90% | 17670 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 93.427 | 3.20% | 4316 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | 88.554 | 3.10% | 6557 |
| _rope_kernel | 83.726 | 2.90% | 8314 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | 74.813 | 2.60% | 4247 |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | 74.311 | 2.60% | 166 |
| _ce_bwd_kernel | 63.362 | 2.20% | 664 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | 56.482 | 2.00% | 2075 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | 55.988 | 1.90% | 2158 |
| _swiglu_bwd_kernel | 44.577 | 1.50% | 2075 |

## GPU memory operations

| op | per_step_ms | pct | count |
|---|---:|---:|---:|
| [CUDA memcpy Host-to-Device] | 42.056 | 95.00% | 818 |
| [CUDA memset] | 1.899 | 4.30% | 25625 |
| [CUDA memcpy Device-to-Device] | 0.295 | 0.70% | 189 |
| [CUDA memcpy Device-to-Host] | 0.007 | 0.00% | 32 |

## top 10 CUDA API calls by total time

| api | per_step_ms | pct | calls |
|---|---:|---:|---:|
| cudaMemcpyAsync | 1571.551 | 49.90% | 1039 |
| cudaGraphLaunch_v10000 | 758.418 | 24.10% | 82 |
| cudaStreamSynchronize | 573.951 | 18.20% | 349 |
| cudaMalloc | 102.738 | 3.30% | 807 |
| cudaLaunchKernel | 78.398 | 2.50% | 6442 |
| cudaFree | 24.406 | 0.80% | 203 |
| cudaDeviceSynchronize | 12.511 | 0.40% | 5 |
| cudaEventSynchronize | 12.366 | 0.40% | 8 |
| cudaMemsetAsync | 2.879 | 0.10% | 648 |
| cuModuleLoadData | 2.263 | 0.10% | 11 |

## top 10 OS runtime calls by total time

| os call | per_step_ms | pct | calls |
|---|---:|---:|---:|
| poll | 22609.862 | 38.80% | 1431 |
| pthread_cond_timedwait | 11572.200 | 19.90% | 13602 |
| pthread_cond_wait | 7509.889 | 12.90% | 236 |
| epoll_wait | 5841.487 | 10.00% | 264 |
| sem_clockwait | 5000.052 | 8.60% | 6 |
| sem_wait | 4492.762 | 7.70% | 119 |
| read | 302.828 | 0.50% | 3856 |
| ioctl | 222.751 | 0.40% | 7550 |
| usleep | 158.268 | 0.30% | 33844 |
| nanosleep | 111.574 | 0.20% | 154 |

## Δ from previous snapshot

Δ from long-horizon_round75

Δ step_time_ms=+13.592  Δ mfu_e2e_standard=-0.0869  Δ profiled_steps=+0  Δ gpu_kernel_per_step_ms=+12.516  Δ gpu_idle_per_step_ms=+1.076  Δ gpu_memop_per_step_ms=+2.676  Δ cuda_api_per_step_ms=-84.606  Δ os_runtime_per_step_ms=-747.008

| kernel | Δ per_step_ms | Δ pct |  state |
|---|---:|---:|---|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | +3.802 | +0.10% | tracked |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | +0.597 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | +0.148 | +0.00% | tracked |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | +0.156 | +0.00% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | +0.592 | +0.00% | tracked |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | +0.174 | +0.00% | tracked |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | +0.565 | +0.00% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | -0.004 | +0.00% | tracked |
| _rope_kernel | +0.177 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | +0.032 | +0.00% | tracked |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | -0.009 | +0.00% | tracked |
| _ce_bwd_kernel | +0.142 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | +0.134 | +0.00% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | +0.122 | +0.00% | tracked |
| _swiglu_bwd_kernel | +0.019 | -0.10% | tracked |
