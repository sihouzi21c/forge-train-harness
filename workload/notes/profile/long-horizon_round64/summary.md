# profile snapshot — profile-snapshot

step_time_ms=4655.799  mfu_e2e_standard=28.2428  profiled_steps=12  gpu_kernel_per_step_ms=2873.962  gpu_idle_per_step_ms=1781.837  gpu_memop_per_step_ms=47.728  cuda_api_per_step_ms=3317.475  os_runtime_per_step_ms=60714.249
# gpu_idle = step - gpu_kernel (valid only when compute kernels share one stream); cuda_api / os_runtime are CPU time, do NOT subtract from step_time.
world_size=2  micro_batch_size=4  seq_length=4096  grad_accum_steps=10
nsys_rep=/user/sunhaojun/.forge_train/77459cccb4da/.artifacts/runs/profile-snapshot-20260814-205035-6081b5/ours/profile.nsys-rep

## top 15 GPU kernels by total time

| kernel | per_step_ms | pct | instances |
|---|---:|---:|---:|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | 547.712 | 19.10% | 2000 |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | 273.229 | 9.50% | 4050 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | 257.179 | 8.90% | 6237 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 217.233 | 7.60% | 2187 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | 207.482 | 7.20% | 2268 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | 108.807 | 3.80% | 17248 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 90.483 | 3.10% | 4212 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | 86.443 | 3.00% | 6399 |
| _rope_kernel | 82.482 | 2.90% | 8100 |
| void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::Opaque... | 78.328 | 2.70% | 2025 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | 72.554 | 2.50% | 4131 |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | 72.511 | 2.50% | 162 |
| _ce_bwd_kernel | 61.655 | 2.10% | 648 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | 54.817 | 1.90% | 2025 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | 54.378 | 1.90% | 2106 |

## GPU memory operations

| op | per_step_ms | pct | count |
|---|---:|---:|---:|
| [CUDA memcpy Host-to-Device] | 45.637 | 95.60% | 800 |
| [CUDA memset] | 1.790 | 3.80% | 24980 |
| [CUDA memcpy Device-to-Device] | 0.295 | 0.60% | 189 |
| [CUDA memcpy Device-to-Host] | 0.006 | 0.00% | 29 |

## top 10 CUDA API calls by total time

| api | per_step_ms | pct | calls |
|---|---:|---:|---:|
| cudaMemcpyAsync | 1623.955 | 49.00% | 1018 |
| cudaGraphLaunch_v10000 | 748.581 | 22.60% | 80 |
| cudaStreamSynchronize | 613.496 | 18.50% | 349 |
| cudaLaunchKernel | 119.060 | 3.60% | 6344 |
| cuMemSetAccess | 76.032 | 2.30% | 707 |
| cuMemCreate | 40.864 | 1.20% | 5784 |
| cudaFree | 21.363 | 0.60% | 2 |
| cuMemRelease | 18.491 | 0.60% | 2145 |
| cuMemUnmap | 15.997 | 0.50% | 2145 |
| cudaDeviceSynchronize | 12.922 | 0.40% | 5 |

## top 10 OS runtime calls by total time

| os call | per_step_ms | pct | calls |
|---|---:|---:|---:|
| poll | 23505.752 | 38.70% | 1458 |
| pthread_cond_timedwait | 11899.801 | 19.60% | 2158 |
| sem_clockwait | 9380.305 | 15.40% | 532 |
| pthread_cond_wait | 7946.429 | 13.10% | 206 |
| epoll_wait | 6081.114 | 10.00% | 296 |
| sem_wait | 351.591 | 0.60% | 10 |
| ioctl | 329.153 | 0.50% | 26067 |
| read | 318.841 | 0.50% | 4121 |
| pthread_mutex_lock | 177.469 | 0.30% | 330 |
| usleep | 175.626 | 0.30% | 38549 |

## Δ from previous snapshot

Δ from long-horizon_round62

Δ step_time_ms=-27.304  Δ mfu_e2e_standard=+0.1673  Δ profiled_steps=+0  Δ gpu_kernel_per_step_ms=-1.064  Δ gpu_idle_per_step_ms=-26.240  Δ gpu_memop_per_step_ms=-12.113  Δ cuda_api_per_step_ms=-6.610  Δ os_runtime_per_step_ms=+936.859

| kernel | Δ per_step_ms | Δ pct |  state |
|---|---:|---:|---|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | +0.082 | +0.10% | tracked |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | -0.419 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | -0.428 | -0.10% | tracked |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | +0.019 | +0.00% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | +0.025 | +0.00% | tracked |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | +0.019 | +0.00% | tracked |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | +0.031 | +0.00% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | +0.003 | +0.00% | tracked |
| _rope_kernel | -0.110 | +0.00% | tracked |
| void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::Opaque... | +0.031 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | -0.132 | +0.00% | tracked |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | +0.002 | +0.00% | tracked |
| _ce_bwd_kernel | -0.028 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | -0.003 | +0.00% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | +0.007 | +0.00% | tracked |
