# profile snapshot — profile-snapshot

step_time_ms=7556.378  mfu_e2e_standard=17.4011  profiled_steps=12  gpu_kernel_per_step_ms=5258.231  gpu_idle_per_step_ms=2298.147  gpu_memop_per_step_ms=140.269  cuda_api_per_step_ms=5233.686  os_runtime_per_step_ms=93392.527
# gpu_idle = step - gpu_kernel (valid only when compute kernels share one stream); cuda_api / os_runtime are CPU time, do NOT subtract from step_time.
world_size=2  micro_batch_size=4  seq_length=4096  grad_accum_steps=10
nsys_rep=/user/sunhaojun/.forge_train/77459cccb4da/.artifacts/runs/profile-snapshot-20260814-170853-bd3dff/ours/profile.nsys-rep

## top 15 GPU kernels by total time

| kernel | per_step_ms | pct | instances |
|---|---:|---:|---:|
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | 607.771 | 11.60% | 32239 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | 436.427 | 8.30% | 42243 |
| void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_ke... | 369.690 | 7.00% | 34346 |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | 345.621 | 6.60% | 5240 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 344.833 | 6.60% | 15924 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | 325.750 | 6.20% | 8056 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 279.670 | 5.30% | 2835 |
| void at::native::vectorized_elementwise_kernel<(int)8, at::native::bfloat16_c... | 274.262 | 5.20% | 50395 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | 266.438 | 5.10% | 2931 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFuncto... | 243.327 | 4.60% | 23127 |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)4, float, float, float, ... | 197.323 | 3.80% | 840 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 171.489 | 3.30% | 31861 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunc... | 98.278 | 1.90% | 841 |
| void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unna... | 94.491 | 1.80% | 16552 |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | 93.956 | 1.80% | 210 |

## GPU memory operations

| op | per_step_ms | pct | count |
|---|---:|---:|---:|
| [CUDA memcpy Host-to-Device] | 77.207 | 55.00% | 2624 |
| [CUDA memcpy Device-to-Device] | 60.362 | 43.00% | 2567 |
| [CUDA memset] | 2.692 | 1.90% | 32264 |
| [CUDA memcpy Device-to-Host] | 0.008 | 0.00% | 40 |

## top 10 CUDA API calls by total time

| api | per_step_ms | pct | calls |
|---|---:|---:|---:|
| cudaStreamSynchronize | 2513.519 | 48.00% | 2034 |
| cudaLaunchKernel | 2237.465 | 42.80% | 435143 |
| cudaMemsetAsync | 165.475 | 3.20% | 32264 |
| cuLaunchKernelEx | 143.857 | 2.70% | 32474 |
| cudaMemcpyAsync | 93.919 | 1.80% | 5231 |
| cuMemSetAccess | 36.066 | 0.70% | 459 |
| cudaFree | 15.991 | 0.30% | 2 |
| cudaEventSynchronize | 11.456 | 0.20% | 10 |
| cuMemCreate | 10.585 | 0.20% | 3718 |
| cuLibraryLoadData | 2.385 | 0.00% | 19 |

## top 10 OS runtime calls by total time

| os call | per_step_ms | pct | calls |
|---|---:|---:|---:|
| poll | 33714.683 | 36.10% | 1867 |
| pthread_cond_wait | 18082.883 | 19.40% | 6216 |
| pthread_cond_timedwait | 17464.298 | 18.70% | 48479 |
| sem_clockwait | 14525.025 | 15.60% | 852 |
| epoll_wait | 8479.933 | 9.10% | 52 |
| sem_wait | 336.275 | 0.40% | 10 |
| read | 295.503 | 0.30% | 3246 |
| ioctl | 152.015 | 0.20% | 19853 |
| pthread_mutex_lock | 99.812 | 0.10% | 5709 |
| usleep | 81.641 | 0.10% | 16697 |

## Δ from previous snapshot

Δ from long-horizon_round57

Δ step_time_ms=-9.180  Δ mfu_e2e_standard=+0.0210  Δ profiled_steps=+0  Δ gpu_kernel_per_step_ms=+4.791  Δ gpu_idle_per_step_ms=-13.971  Δ gpu_memop_per_step_ms=+17.108  Δ cuda_api_per_step_ms=-24.678  Δ os_runtime_per_step_ms=-111.108

| kernel | Δ per_step_ms | Δ pct |  state |
|---|---:|---:|---|
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | +1.015 | +0.10% | tracked |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | +0.316 | +0.00% | tracked |
| void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_ke... | +0.341 | +0.00% | tracked |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | +0.416 | +0.00% | tracked |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | +0.458 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | +0.298 | +0.00% | tracked |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | -0.095 | +0.00% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)8, at::native::bfloat16_c... | +0.247 | +0.00% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | +0.264 | +0.00% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFuncto... | +0.130 | +0.00% | tracked |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)4, float, float, float, ... | -0.019 | +0.00% | tracked |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | +0.147 | +0.00% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunc... | -0.002 | +0.00% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unna... | +0.095 | +0.00% | tracked |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | +0.001 | +0.00% | tracked |
