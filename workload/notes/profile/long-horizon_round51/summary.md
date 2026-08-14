# profile snapshot — profile-snapshot

step_time_ms=7598.896  mfu_e2e_standard=17.3025  profiled_steps=12  gpu_kernel_per_step_ms=5991.317  gpu_idle_per_step_ms=1607.579  gpu_memop_per_step_ms=117.972  cuda_api_per_step_ms=6515.280  os_runtime_per_step_ms=88265.489
# gpu_idle = step - gpu_kernel (valid only when compute kernels share one stream); cuda_api / os_runtime are CPU time, do NOT subtract from step_time.
world_size=2  micro_batch_size=4  seq_length=4096  grad_accum_steps=10
nsys_rep=/user/sunhaojun/.forge_train/77459cccb4da/.artifacts/runs/profile-snapshot-20260814-104251-a86fbc/ours/profile.nsys-rep

## top 15 GPU kernels by total time

| kernel | per_step_ms | pct | instances |
|---|---:|---:|---:|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | 680.500 | 11.40% | 2525 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | 591.773 | 9.90% | 32434 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | 562.524 | 9.40% | 41971 |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)4, float, float, float, ... | 378.398 | 6.30% | 1020 |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | 336.326 | 5.60% | 5100 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 334.805 | 5.60% | 15504 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | 317.141 | 5.30% | 7853 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 272.490 | 4.50% | 2754 |
| void at::native::vectorized_elementwise_kernel<(int)8, at::native::bfloat16_c... | 265.934 | 4.40% | 49100 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | 259.592 | 4.30% | 2856 |
| void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_ke... | 237.423 | 4.00% | 33444 |
| void at::native::<unnamed>::cunn_SoftMaxBackward<(int)4, float, float, float,... | 192.292 | 3.20% | 808 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 165.204 | 2.80% | 31005 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFuncto... | 95.274 | 1.60% | 21721 |
| void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unna... | 91.652 | 1.50% | 16114 |

## GPU memory operations

| op | per_step_ms | pct | count |
|---|---:|---:|---:|
| [CUDA memcpy Device-to-Device] | 58.666 | 49.70% | 2543 |
| [CUDA memcpy Host-to-Device] | 56.918 | 48.20% | 926 |
| [CUDA memset] | 2.377 | 2.00% | 31404 |
| [CUDA memcpy Device-to-Host] | 0.011 | 0.00% | 50 |

## top 10 CUDA API calls by total time

| api | per_step_ms | pct | calls |
|---|---:|---:|---:|
| cudaMemcpyAsync | 3102.203 | 47.60% | 2719 |
| cudaGraphLaunch_v10000 | 1855.304 | 28.50% | 100 |
| cudaStreamSynchronize | 1209.625 | 18.60% | 369 |
| cudaLaunchKernel | 140.197 | 2.20% | 15623 |
| cuMemSetAccess | 63.242 | 1.00% | 639 |
| cuMemCreate | 53.795 | 0.80% | 5720 |
| cuMemRelease | 25.370 | 0.40% | 2369 |
| cuMemUnmap | 24.377 | 0.40% | 2369 |
| cudaFree | 13.376 | 0.20% | 2 |
| cudaEventSynchronize | 11.868 | 0.20% | 10 |

## top 10 OS runtime calls by total time

| os call | per_step_ms | pct | calls |
|---|---:|---:|---:|
| poll | 34850.279 | 39.50% | 1890 |
| pthread_cond_timedwait | 17366.606 | 19.70% | 1698 |
| sem_clockwait | 14775.557 | 16.70% | 882 |
| pthread_cond_wait | 10755.516 | 12.20% | 233 |
| epoll_wait | 8806.513 | 10.00% | 78 |
| pthread_rwlock_wrlock | 375.107 | 0.40% | 44 |
| sem_wait | 328.211 | 0.40% | 10 |
| read | 292.901 | 0.30% | 3234 |
| ioctl | 252.261 | 0.30% | 27244 |
| usleep | 131.859 | 0.10% | 28929 |

## Δ from previous snapshot

Δ from long-horizon_round50

Δ step_time_ms=-5.523  Δ mfu_e2e_standard=+0.0126  Δ profiled_steps=+0  Δ gpu_kernel_per_step_ms=+0.209  Δ gpu_idle_per_step_ms=-5.732  Δ gpu_memop_per_step_ms=+12.879  Δ cuda_api_per_step_ms=-37.654  Δ os_runtime_per_step_ms=-689.792

| kernel | Δ per_step_ms | Δ pct |  state |
|---|---:|---:|---|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | +0.391 | +0.00% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | +0.198 | +0.00% | tracked |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | +0.001 | +0.00% | tracked |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)4, float, float, float, ... | +0.004 | +0.00% | tracked |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | -0.202 | +0.00% | tracked |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | -0.020 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | -0.043 | +0.00% | tracked |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | -0.245 | -0.10% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)8, at::native::bfloat16_c... | +0.062 | +0.00% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | -0.096 | +0.00% | tracked |
| void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_ke... | +0.044 | +0.00% | tracked |
| void at::native::<unnamed>::cunn_SoftMaxBackward<(int)4, float, float, float,... | +0.017 | +0.00% | tracked |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | -0.052 | +0.00% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFuncto... | +0.030 | +0.00% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unna... | +0.015 | +0.00% | tracked |
