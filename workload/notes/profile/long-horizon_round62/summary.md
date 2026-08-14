# profile snapshot — profile-snapshot

step_time_ms=4683.103  mfu_e2e_standard=28.0755  profiled_steps=12  gpu_kernel_per_step_ms=2875.026  gpu_idle_per_step_ms=1808.077  gpu_memop_per_step_ms=59.841  cuda_api_per_step_ms=3324.085  os_runtime_per_step_ms=59777.390
# gpu_idle = step - gpu_kernel (valid only when compute kernels share one stream); cuda_api / os_runtime are CPU time, do NOT subtract from step_time.
world_size=2  micro_batch_size=4  seq_length=4096  grad_accum_steps=10
nsys_rep=/user/sunhaojun/.forge_train/77459cccb4da/.artifacts/runs/profile-snapshot-20260814-193458-ca45ec/ours/profile.nsys-rep

## top 15 GPU kernels by total time

| kernel | per_step_ms | pct | instances |
|---|---:|---:|---:|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | 547.630 | 19.00% | 2000 |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | 273.648 | 9.50% | 4050 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | 257.607 | 9.00% | 6237 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 217.214 | 7.60% | 2187 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | 207.457 | 7.20% | 2268 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | 108.788 | 3.80% | 17248 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 90.452 | 3.10% | 4212 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | 86.440 | 3.00% | 6399 |
| _rope_kernel | 82.592 | 2.90% | 8100 |
| void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::Opaque... | 78.297 | 2.70% | 2025 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | 72.686 | 2.50% | 4131 |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | 72.509 | 2.50% | 162 |
| _ce_bwd_kernel | 61.683 | 2.10% | 648 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | 54.820 | 1.90% | 2025 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | 54.371 | 1.90% | 2106 |

## GPU memory operations

| op | per_step_ms | pct | count |
|---|---:|---:|---:|
| [CUDA memcpy Host-to-Device] | 57.757 | 96.50% | 800 |
| [CUDA memset] | 1.787 | 3.00% | 24980 |
| [CUDA memcpy Device-to-Device] | 0.291 | 0.50% | 157 |
| [CUDA memcpy Device-to-Host] | 0.006 | 0.00% | 29 |

## top 10 CUDA API calls by total time

| api | per_step_ms | pct | calls |
|---|---:|---:|---:|
| cudaMemcpyAsync | 1637.811 | 49.30% | 986 |
| cudaGraphLaunch_v10000 | 750.430 | 22.60% | 80 |
| cudaStreamSynchronize | 618.241 | 18.60% | 349 |
| cudaLaunchKernel | 102.404 | 3.10% | 6367 |
| cuMemSetAccess | 77.467 | 2.30% | 707 |
| cuMemCreate | 59.220 | 1.80% | 5784 |
| cudaFree | 18.768 | 0.60% | 2 |
| cudaDeviceSynchronize | 12.852 | 0.40% | 5 |
| cuMemRelease | 12.110 | 0.40% | 2145 |
| cudaEventSynchronize | 9.582 | 0.30% | 7 |

## top 10 OS runtime calls by total time

| os call | per_step_ms | pct | calls |
|---|---:|---:|---:|
| poll | 23366.861 | 39.10% | 1339 |
| pthread_cond_timedwait | 11749.020 | 19.70% | 2256 |
| sem_clockwait | 9396.080 | 15.70% | 534 |
| pthread_cond_wait | 7791.178 | 13.00% | 205 |
| epoll_wait | 5935.318 | 9.90% | 84 |
| sem_wait | 333.935 | 0.60% | 10 |
| read | 323.116 | 0.50% | 3856 |
| ioctl | 266.624 | 0.40% | 26071 |
| usleep | 152.351 | 0.30% | 32836 |
| pthread_rwlock_wrlock | 149.225 | 0.20% | 30 |

## Δ from previous snapshot

Δ from long-horizon_round61

Δ step_time_ms=-3.741  Δ mfu_e2e_standard=+0.0221  Δ profiled_steps=+0  Δ gpu_kernel_per_step_ms=+6.766  Δ gpu_idle_per_step_ms=-10.507  Δ gpu_memop_per_step_ms=+25.760  Δ cuda_api_per_step_ms=+128.215  Δ os_runtime_per_step_ms=+545.739

| kernel | Δ per_step_ms | Δ pct |  state |
|---|---:|---:|---|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | +0.338 | -0.10% | tracked |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | +0.692 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | +0.103 | +0.00% | tracked |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | +0.138 | +0.00% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | +0.011 | +0.00% | tracked |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | +0.054 | +0.00% | tracked |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | +0.023 | -0.10% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | +0.061 | +0.00% | tracked |
| _rope_kernel | +0.208 | +0.00% | tracked |
| void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::Opaque... | +0.041 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | +0.171 | +0.00% | tracked |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | -0.005 | +0.00% | tracked |
| _ce_bwd_kernel | +0.014 | -0.10% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | +0.021 | +0.00% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | +0.013 | +0.00% | tracked |
