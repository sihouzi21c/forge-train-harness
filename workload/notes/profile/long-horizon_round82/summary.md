# profile snapshot — profile-snapshot

step_time_ms=4214.442  mfu_e2e_standard=31.1975  profiled_steps=12  gpu_kernel_per_step_ms=2872.328  gpu_idle_per_step_ms=1342.114  gpu_memop_per_step_ms=58.334  cuda_api_per_step_ms=2967.331  os_runtime_per_step_ms=55649.369
# gpu_idle = step - gpu_kernel (valid only when compute kernels share one stream); cuda_api / os_runtime are CPU time, do NOT subtract from step_time.
world_size=2  micro_batch_size=4  seq_length=4096  grad_accum_steps=10
nsys_rep=/user/sunhaojun/.forge_train/77459cccb4da/.artifacts/runs/profile-snapshot-20260815-095843-a12ed6/ours/profile.nsys-rep

## top 15 GPU kernels by total time

| kernel | per_step_ms | pct | instances |
|---|---:|---:|---:|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | 563.256 | 19.60% | 2054 |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | 280.919 | 9.80% | 4179 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | 263.852 | 9.20% | 6404 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 223.033 | 7.80% | 2268 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | 211.210 | 7.40% | 2331 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | 111.651 | 3.90% | 17709 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 93.187 | 3.20% | 4326 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | 88.726 | 3.10% | 6572 |
| _rope_kernel | 83.483 | 2.90% | 8358 |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | 75.197 | 2.60% | 168 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | 74.793 | 2.60% | 4284 |
| _ce_bwd_kernel | 63.818 | 2.20% | 672 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | 55.759 | 1.90% | 2079 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | 55.484 | 1.90% | 2163 |
| _swiglu_bwd_kernel | 44.632 | 1.60% | 2079 |

## GPU memory operations

| op | per_step_ms | pct | count |
|---|---:|---:|---:|
| [CUDA memcpy Host-to-Device] | 56.109 | 96.20% | 818 |
| [CUDA memset] | 1.922 | 3.30% | 25745 |
| [CUDA memcpy Device-to-Device] | 0.296 | 0.50% | 197 |
| [CUDA memcpy Device-to-Host] | 0.007 | 0.00% | 32 |

## top 10 CUDA API calls by total time

| api | per_step_ms | pct | calls |
|---|---:|---:|---:|
| cudaMemcpyAsync | 1574.832 | 53.10% | 1047 |
| cudaGraphLaunch_v10000 | 753.877 | 25.40% | 82 |
| cudaStreamSynchronize | 437.374 | 14.70% | 349 |
| cudaMalloc | 63.010 | 2.10% | 702 |
| cudaFree | 62.122 | 2.10% | 203 |
| cudaLaunchKernel | 43.860 | 1.50% | 5999 |
| cudaEventSynchronize | 10.951 | 0.40% | 8 |
| cudaDeviceSynchronize | 10.002 | 0.30% | 5 |
| cudaStreamCreateWithPriority | 4.565 | 0.20% | 128 |
| cuLibraryLoadData | 2.144 | 0.10% | 19 |

## top 10 OS runtime calls by total time

| os call | per_step_ms | pct | calls |
|---|---:|---:|---:|
| poll | 21599.300 | 38.80% | 1561 |
| pthread_cond_timedwait | 11201.445 | 20.10% | 24385 |
| pthread_cond_wait | 7398.380 | 13.30% | 267 |
| epoll_wait | 5794.836 | 10.40% | 706 |
| sem_wait | 4169.092 | 7.50% | 119 |
| sem_clockwait | 4166.710 | 7.50% | 5 |
| nanosleep | 296.790 | 0.50% | 375 |
| read | 284.253 | 0.50% | 3956 |
| ioctl | 231.908 | 0.40% | 7249 |
| usleep | 120.415 | 0.20% | 25553 |

## Δ from previous snapshot

Δ from long-horizon_round81

Δ step_time_ms=-9.016  Δ mfu_e2e_standard=+0.0667  Δ profiled_steps=+0  Δ gpu_kernel_per_step_ms=+8.875  Δ gpu_idle_per_step_ms=-17.891  Δ gpu_memop_per_step_ms=-7.451  Δ cuda_api_per_step_ms=-58.112  Δ os_runtime_per_step_ms=-620.501

| kernel | Δ per_step_ms | Δ pct |  state |
|---|---:|---:|---|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | +0.601 | +0.00% | tracked |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | +0.823 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | +2.138 | +0.10% | tracked |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | +2.119 | +0.10% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | +1.768 | +0.10% | tracked |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | +0.403 | +0.00% | tracked |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | +0.213 | +0.00% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | +0.195 | +0.00% | tracked |
| _rope_kernel | +0.262 | +0.00% | tracked |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | +0.896 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | +0.448 | +0.00% | tracked |
| _ce_bwd_kernel | +0.773 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | +0.138 | +0.00% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | +0.146 | +0.00% | tracked |
| _swiglu_bwd_kernel | +0.097 | +0.00% | tracked |
