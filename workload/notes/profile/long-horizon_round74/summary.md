# profile snapshot — profile-snapshot

step_time_ms=4501.150  mfu_e2e_standard=29.2109  profiled_steps=12  gpu_kernel_per_step_ms=2830.837  gpu_idle_per_step_ms=1670.313  gpu_memop_per_step_ms=75.707  cuda_api_per_step_ms=3181.513  os_runtime_per_step_ms=58756.888
# gpu_idle = step - gpu_kernel (valid only when compute kernels share one stream); cuda_api / os_runtime are CPU time, do NOT subtract from step_time.
world_size=2  micro_batch_size=4  seq_length=4096  grad_accum_steps=10
nsys_rep=/user/sunhaojun/.forge_train/77459cccb4da/.artifacts/runs/profile-snapshot-20260815-020005-7fc94a/ours/profile.nsys-rep

## top 15 GPU kernels by total time

| kernel | per_step_ms | pct | instances |
|---|---:|---:|---:|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | 555.738 | 19.60% | 2020 |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | 277.021 | 9.80% | 4095 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | 260.364 | 9.20% | 6298 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 220.054 | 7.80% | 2214 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | 209.901 | 7.40% | 2291 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | 109.993 | 3.90% | 17415 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 91.512 | 3.20% | 4254 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | 87.261 | 3.10% | 6462 |
| _rope_kernel | 83.615 | 3.00% | 8190 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | 73.547 | 2.60% | 4182 |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | 73.413 | 2.60% | 164 |
| _ce_bwd_kernel | 62.434 | 2.20% | 656 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | 55.475 | 2.00% | 2045 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | 55.003 | 1.90% | 2127 |
| _swiglu_bwd_kernel | 43.919 | 1.60% | 2045 |

## GPU memory operations

| op | per_step_ms | pct | count |
|---|---:|---:|---:|
| [CUDA memcpy Host-to-Device] | 73.607 | 97.20% | 806 |
| [CUDA memset] | 1.798 | 2.40% | 25241 |
| [CUDA memcpy Device-to-Device] | 0.295 | 0.40% | 189 |
| [CUDA memcpy Device-to-Host] | 0.007 | 0.00% | 32 |

## top 10 CUDA API calls by total time

| api | per_step_ms | pct | calls |
|---|---:|---:|---:|
| cudaMemcpyAsync | 1587.851 | 49.90% | 1027 |
| cudaGraphLaunch_v10000 | 742.986 | 23.40% | 80 |
| cudaStreamSynchronize | 575.551 | 18.10% | 352 |
| cudaLaunchKernel | 86.472 | 2.70% | 6422 |
| cuMemSetAccess | 58.632 | 1.80% | 705 |
| cuMemCreate | 35.228 | 1.10% | 5996 |
| cudaFree | 31.829 | 1.00% | 2 |
| cudaDeviceSynchronize | 12.501 | 0.40% | 5 |
| cudaEventSynchronize | 12.410 | 0.40% | 8 |
| cuMemRelease | 11.575 | 0.40% | 2143 |

## top 10 OS runtime calls by total time

| os call | per_step_ms | pct | calls |
|---|---:|---:|---:|
| poll | 22645.998 | 38.50% | 1487 |
| pthread_cond_timedwait | 11626.414 | 19.80% | 11324 |
| pthread_cond_wait | 7672.793 | 13.10% | 238 |
| epoll_wait | 5933.705 | 10.10% | 418 |
| sem_clockwait | 5000.054 | 8.50% | 6 |
| sem_wait | 4545.888 | 7.70% | 119 |
| read | 293.619 | 0.50% | 4023 |
| ioctl | 262.837 | 0.40% | 26489 |
| usleep | 186.956 | 0.30% | 39869 |
| nanosleep | 176.266 | 0.30% | 231 |

## Δ from previous snapshot

Δ from long-horizon_round72

Δ step_time_ms=+3.255  Δ mfu_e2e_standard=-0.0214  Δ profiled_steps=+0  Δ gpu_kernel_per_step_ms=-2.073  Δ gpu_idle_per_step_ms=+5.328  Δ gpu_memop_per_step_ms=+2.054  Δ cuda_api_per_step_ms=-7.504  Δ os_runtime_per_step_ms=+816.684

| kernel | Δ per_step_ms | Δ pct |  state |
|---|---:|---:|---|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | -0.253 | +0.00% | tracked |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | -0.221 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | -0.373 | +0.00% | tracked |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | -0.215 | +0.00% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | -0.073 | +0.00% | tracked |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | -0.048 | +0.00% | tracked |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | -0.063 | +0.00% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | -0.056 | +0.00% | tracked |
| _rope_kernel | -0.057 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | -0.067 | +0.00% | tracked |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | -0.006 | +0.00% | tracked |
| _ce_bwd_kernel | -0.049 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | -0.047 | +0.00% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | -0.045 | +0.00% | tracked |
| _swiglu_bwd_kernel | -0.023 | +0.00% | tracked |
