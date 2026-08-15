# profile snapshot — profile-snapshot

step_time_ms=4223.458  mfu_e2e_standard=31.1308  profiled_steps=12  gpu_kernel_per_step_ms=2863.453  gpu_idle_per_step_ms=1360.005  gpu_memop_per_step_ms=65.785  cuda_api_per_step_ms=3025.443  os_runtime_per_step_ms=56269.870
# gpu_idle = step - gpu_kernel (valid only when compute kernels share one stream); cuda_api / os_runtime are CPU time, do NOT subtract from step_time.
world_size=2  micro_batch_size=4  seq_length=4096  grad_accum_steps=10
nsys_rep=/user/sunhaojun/.forge_train/77459cccb4da/.artifacts/runs/profile-snapshot-20260815-085243-732588/ours/profile.nsys-rep

## top 15 GPU kernels by total time

| kernel | per_step_ms | pct | instances |
|---|---:|---:|---:|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | 562.655 | 19.60% | 2050 |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | 280.096 | 9.80% | 4163 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | 261.714 | 9.10% | 6391 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 220.914 | 7.70% | 2254 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | 209.442 | 7.30% | 2324 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | 111.248 | 3.90% | 17670 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 92.974 | 3.20% | 4316 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | 88.531 | 3.10% | 6557 |
| _rope_kernel | 83.221 | 2.90% | 8328 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | 74.345 | 2.60% | 4259 |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | 74.301 | 2.60% | 166 |
| _ce_bwd_kernel | 63.045 | 2.20% | 664 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | 55.621 | 1.90% | 2075 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | 55.338 | 1.90% | 2158 |
| _swiglu_bwd_kernel | 44.535 | 1.60% | 2075 |

## GPU memory operations

| op | per_step_ms | pct | count |
|---|---:|---:|---:|
| [CUDA memcpy Host-to-Device] | 63.627 | 96.70% | 818 |
| [CUDA memset] | 1.854 | 2.80% | 25649 |
| [CUDA memcpy Device-to-Device] | 0.297 | 0.50% | 197 |
| [CUDA memcpy Device-to-Host] | 0.007 | 0.00% | 32 |

## top 10 CUDA API calls by total time

| api | per_step_ms | pct | calls |
|---|---:|---:|---:|
| cudaMemcpyAsync | 1583.458 | 52.30% | 1047 |
| cudaGraphLaunch_v10000 | 752.824 | 24.90% | 82 |
| cudaStreamSynchronize | 408.342 | 13.50% | 349 |
| cudaMalloc | 111.176 | 3.70% | 806 |
| cudaLaunchKernel | 78.213 | 2.60% | 6443 |
| cudaFree | 56.578 | 1.90% | 203 |
| cudaEventSynchronize | 12.528 | 0.40% | 8 |
| cudaDeviceSynchronize | 12.416 | 0.40% | 5 |
| cudaStreamCreateWithPriority | 2.134 | 0.10% | 128 |
| cuLibraryLoadData | 2.120 | 0.10% | 19 |

## top 10 OS runtime calls by total time

| os call | per_step_ms | pct | calls |
|---|---:|---:|---:|
| poll | 21892.812 | 38.90% | 1369 |
| pthread_cond_timedwait | 11393.329 | 20.20% | 26564 |
| pthread_cond_wait | 7603.351 | 13.50% | 261 |
| epoll_wait | 5718.787 | 10.20% | 290 |
| sem_clockwait | 4166.709 | 7.40% | 5 |
| sem_wait | 4157.364 | 7.40% | 119 |
| read | 289.414 | 0.50% | 4098 |
| ioctl | 283.034 | 0.50% | 7724 |
| usleep | 177.800 | 0.30% | 38599 |
| pthread_mutex_lock | 168.620 | 0.30% | 3709 |

## Δ from previous snapshot

Δ from long-horizon_round80

Δ step_time_ms=+2.374  Δ mfu_e2e_standard=-0.0176  Δ profiled_steps=+0  Δ gpu_kernel_per_step_ms=+2.738  Δ gpu_idle_per_step_ms=-0.364  Δ gpu_memop_per_step_ms=+4.884  Δ cuda_api_per_step_ms=+117.831  Δ os_runtime_per_step_ms=+682.856

| kernel | Δ per_step_ms | Δ pct |  state |
|---|---:|---:|---|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | +0.639 | +0.00% | tracked |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | +0.544 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | +0.319 | +0.00% | tracked |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | +0.277 | +0.00% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | +0.282 | +0.00% | tracked |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | +0.014 | +0.00% | tracked |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | +0.028 | +0.00% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | -0.002 | +0.00% | tracked |
| _rope_kernel | +0.117 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | +0.123 | +0.00% | tracked |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | +0.015 | +0.00% | tracked |
| _ce_bwd_kernel | -0.016 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | +0.054 | +0.00% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | +0.027 | +0.00% | tracked |
| _swiglu_bwd_kernel | +0.001 | +0.00% | tracked |
