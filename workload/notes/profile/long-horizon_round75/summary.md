# profile snapshot — profile-snapshot

step_time_ms=4490.144  mfu_e2e_standard=29.2824  profiled_steps=12  gpu_kernel_per_step_ms=2870.560  gpu_idle_per_step_ms=1619.584  gpu_memop_per_step_ms=41.581  cuda_api_per_step_ms=3233.092  os_runtime_per_step_ms=58949.486
# gpu_idle = step - gpu_kernel (valid only when compute kernels share one stream); cuda_api / os_runtime are CPU time, do NOT subtract from step_time.
world_size=2  micro_batch_size=4  seq_length=4096  grad_accum_steps=10
nsys_rep=/user/sunhaojun/.forge_train/77459cccb4da/.artifacts/runs/profile-snapshot-20260815-044532-96d04c/ours/profile.nsys-rep

## top 15 GPU kernels by total time

| kernel | per_step_ms | pct | instances |
|---|---:|---:|---:|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | 562.787 | 19.60% | 2050 |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | 281.369 | 9.80% | 4157 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | 264.413 | 9.20% | 6391 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 223.424 | 7.80% | 2248 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | 212.884 | 7.40% | 2324 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | 111.634 | 3.90% | 17670 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 92.862 | 3.20% | 4316 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | 88.558 | 3.10% | 6557 |
| _rope_kernel | 83.549 | 2.90% | 8314 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | 74.781 | 2.60% | 4247 |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | 74.320 | 2.60% | 166 |
| _ce_bwd_kernel | 63.220 | 2.20% | 664 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | 56.348 | 2.00% | 2075 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | 55.866 | 1.90% | 2158 |
| _swiglu_bwd_kernel | 44.558 | 1.60% | 2075 |

## GPU memory operations

| op | per_step_ms | pct | count |
|---|---:|---:|---:|
| [CUDA memcpy Host-to-Device] | 39.388 | 94.70% | 818 |
| [CUDA memset] | 1.891 | 4.50% | 25625 |
| [CUDA memcpy Device-to-Device] | 0.295 | 0.70% | 189 |
| [CUDA memcpy Device-to-Host] | 0.007 | 0.00% | 32 |

## top 10 CUDA API calls by total time

| api | per_step_ms | pct | calls |
|---|---:|---:|---:|
| cudaMemcpyAsync | 1573.072 | 48.70% | 1039 |
| cudaGraphLaunch_v10000 | 752.808 | 23.30% | 82 |
| cudaStreamSynchronize | 570.043 | 17.60% | 349 |
| cudaMalloc | 147.536 | 4.60% | 766 |
| cudaLaunchKernel | 80.896 | 2.50% | 6442 |
| cudaFree | 66.229 | 2.00% | 183 |
| cudaEventSynchronize | 12.475 | 0.40% | 8 |
| cudaDeviceSynchronize | 12.368 | 0.40% | 5 |
| cuLaunchKernelEx | 8.014 | 0.20% | 620 |
| cudaMemsetAsync | 3.478 | 0.10% | 648 |

## top 10 OS runtime calls by total time

| os call | per_step_ms | pct | calls |
|---|---:|---:|---:|
| poll | 22626.142 | 38.40% | 1460 |
| pthread_cond_timedwait | 11531.901 | 19.60% | 10997 |
| pthread_cond_wait | 7983.993 | 13.50% | 231 |
| epoll_wait | 5905.526 | 10.00% | 396 |
| sem_clockwait | 5000.051 | 8.50% | 6 |
| sem_wait | 4606.864 | 7.80% | 119 |
| read | 343.696 | 0.60% | 3992 |
| ioctl | 281.414 | 0.50% | 7321 |
| nanosleep | 167.719 | 0.30% | 221 |
| usleep | 124.539 | 0.20% | 26847 |

## Δ from previous snapshot

Δ from long-horizon_round74

Δ step_time_ms=-11.006  Δ mfu_e2e_standard=+0.0715  Δ profiled_steps=+0  Δ gpu_kernel_per_step_ms=+39.723  Δ gpu_idle_per_step_ms=-50.729  Δ gpu_memop_per_step_ms=-34.126  Δ cuda_api_per_step_ms=+51.579  Δ os_runtime_per_step_ms=+192.598

| kernel | Δ per_step_ms | Δ pct |  state |
|---|---:|---:|---|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | +7.049 | +0.00% | tracked |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | +4.348 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | +4.049 | +0.00% | tracked |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | +3.370 | +0.00% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | +2.983 | +0.00% | tracked |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | +1.641 | +0.00% | tracked |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | +1.350 | +0.00% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | +1.297 | +0.00% | tracked |
| _rope_kernel | -0.066 | -0.10% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | +1.234 | +0.00% | tracked |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | +0.907 | +0.00% | tracked |
| _ce_bwd_kernel | +0.786 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | +0.873 | +0.00% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | +0.863 | +0.00% | tracked |
| _swiglu_bwd_kernel | +0.639 | +0.00% | tracked |
