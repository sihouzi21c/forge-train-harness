# profile snapshot — profile-snapshot

step_time_ms=4686.844  mfu_e2e_standard=28.0534  profiled_steps=12  gpu_kernel_per_step_ms=2868.260  gpu_idle_per_step_ms=1818.584  gpu_memop_per_step_ms=34.081  cuda_api_per_step_ms=3195.870  os_runtime_per_step_ms=59231.651
# gpu_idle = step - gpu_kernel (valid only when compute kernels share one stream); cuda_api / os_runtime are CPU time, do NOT subtract from step_time.
world_size=2  micro_batch_size=4  seq_length=4096  grad_accum_steps=10
nsys_rep=/user/sunhaojun/.forge_train/77459cccb4da/.artifacts/runs/profile-snapshot-20260814-191014-fdc7a3/ours/profile.nsys-rep

## top 15 GPU kernels by total time

| kernel | per_step_ms | pct | instances |
|---|---:|---:|---:|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | 547.292 | 19.10% | 1998 |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | 272.956 | 9.50% | 4048 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | 257.504 | 9.00% | 6231 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 217.076 | 7.60% | 2187 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | 207.446 | 7.20% | 2266 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | 108.734 | 3.80% | 17232 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 90.429 | 3.20% | 4210 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | 86.379 | 3.00% | 6394 |
| _rope_kernel | 82.384 | 2.90% | 8096 |
| void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::Opaque... | 78.256 | 2.70% | 2023 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | 72.515 | 2.50% | 4131 |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | 72.514 | 2.50% | 162 |
| _ce_bwd_kernel | 61.669 | 2.20% | 648 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | 54.799 | 1.90% | 2024 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | 54.358 | 1.90% | 2105 |

## GPU memory operations

| op | per_step_ms | pct | count |
|---|---:|---:|---:|
| [CUDA memcpy Host-to-Device] | 29.738 | 87.30% | 800 |
| [CUDA memcpy Device-to-Device] | 2.481 | 7.30% | 1256 |
| [CUDA memset] | 1.856 | 5.40% | 24953 |
| [CUDA memcpy Device-to-Host] | 0.006 | 0.00% | 28 |

## top 10 CUDA API calls by total time

| api | per_step_ms | pct | calls |
|---|---:|---:|---:|
| cudaMemcpyAsync | 1671.873 | 52.30% | 2084 |
| cudaGraphLaunch_v10000 | 736.117 | 23.00% | 79 |
| cudaStreamSynchronize | 499.861 | 15.60% | 348 |
| cudaLaunchKernel | 61.209 | 1.90% | 5850 |
| cudaStreamCreateWithPriority | 49.087 | 1.50% | 128 |
| cuMemSetAccess | 46.545 | 1.50% | 707 |
| cudaFree | 46.032 | 1.40% | 2 |
| cuMemCreate | 29.258 | 0.90% | 5784 |
| cudaDeviceSynchronize | 12.914 | 0.40% | 5 |
| cuMemRelease | 12.559 | 0.40% | 2145 |

## top 10 OS runtime calls by total time

| os call | per_step_ms | pct | calls |
|---|---:|---:|---:|
| poll | 22763.193 | 38.40% | 1339 |
| pthread_cond_timedwait | 11531.892 | 19.50% | 2333 |
| sem_clockwait | 9304.071 | 15.70% | 523 |
| pthread_cond_wait | 7981.625 | 13.50% | 206 |
| epoll_wait | 5853.374 | 9.90% | 124 |
| read | 343.957 | 0.60% | 3840 |
| sem_wait | 325.689 | 0.50% | 10 |
| ioctl | 259.127 | 0.40% | 26060 |
| pthread_rwlock_wrlock | 220.208 | 0.40% | 55 |
| usleep | 173.176 | 0.30% | 37944 |

## Δ from previous snapshot

Δ from long-horizon_round59

Δ step_time_ms=-2869.534  Δ mfu_e2e_standard=+10.6523  Δ profiled_steps=+0  Δ gpu_kernel_per_step_ms=-2389.971  Δ gpu_idle_per_step_ms=-479.563  Δ gpu_memop_per_step_ms=-106.188  Δ cuda_api_per_step_ms=-2037.816  Δ os_runtime_per_step_ms=-34160.876

| kernel | Δ per_step_ms | Δ pct |  state |
|---|---:|---:|---|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | (new) | (new) | new |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | -72.665 | +2.90% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | -68.246 | +2.80% | tracked |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | -62.594 | +2.30% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | -58.992 | +2.10% | tracked |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | -327.693 | -4.50% | tracked |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | -254.404 | -3.40% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | -521.392 | -8.60% | tracked |
| _rope_kernel | (new) | (new) | new |
| void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::Opaque... | (new) | (new) | new |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | (new) | (new) | new |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | -21.442 | +0.70% | tracked |
| _ce_bwd_kernel | (new) | (new) | new |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | (new) | (new) | new |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | (new) | (new) | new |
| void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_ke... | -369.690 | -7.00% | dropped |
| void at::native::vectorized_elementwise_kernel<(int)8, at::native::bfloat16_c... | -274.262 | -5.20% | dropped |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFuncto... | -243.327 | -4.60% | dropped |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)4, float, float, float, ... | -197.323 | -3.80% | dropped |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | -171.489 | -3.30% | dropped |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunc... | -98.278 | -1.90% | dropped |
| void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unna... | -94.491 | -1.80% | dropped |
