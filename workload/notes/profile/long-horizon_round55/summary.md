# profile snapshot — profile-snapshot

step_time_ms=7557.160  mfu_e2e_standard=17.3991  profiled_steps=12  gpu_kernel_per_step_ms=5237.532  gpu_idle_per_step_ms=2319.628  gpu_memop_per_step_ms=110.513  cuda_api_per_step_ms=5258.899  os_runtime_per_step_ms=93571.971
# gpu_idle = step - gpu_kernel (valid only when compute kernels share one stream); cuda_api / os_runtime are CPU time, do NOT subtract from step_time.
world_size=2  micro_batch_size=4  seq_length=4096  grad_accum_steps=10
nsys_rep=/user/sunhaojun/.forge_train/77459cccb4da/.artifacts/runs/profile-snapshot-20260814-142422-aef80a/ours/profile.nsys-rep

## top 15 GPU kernels by total time

| kernel | per_step_ms | pct | instances |
|---|---:|---:|---:|
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | 604.649 | 11.50% | 32068 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | 434.221 | 8.30% | 42040 |
| void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_ke... | 367.866 | 7.00% | 34181 |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | 344.408 | 6.60% | 5226 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 343.437 | 6.60% | 15862 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | 323.943 | 6.20% | 8012 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 279.728 | 5.30% | 2835 |
| void at::native::vectorized_elementwise_kernel<(int)8, at::native::bfloat16_c... | 273.124 | 5.20% | 50228 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | 264.866 | 5.10% | 2914 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFuncto... | 242.104 | 4.60% | 22996 |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)4, float, float, float, ... | 196.381 | 3.70% | 836 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 171.013 | 3.30% | 31776 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunc... | 97.812 | 1.90% | 837 |
| void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unna... | 94.141 | 1.80% | 16490 |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | 93.953 | 1.80% | 210 |

## GPU memory operations

| op | per_step_ms | pct | count |
|---|---:|---:|---:|
| [CUDA memcpy Device-to-Device] | 60.088 | 54.40% | 2563 |
| [CUDA memcpy Host-to-Device] | 47.738 | 43.20% | 2616 |
| [CUDA memset] | 2.679 | 2.40% | 32145 |
| [CUDA memcpy Device-to-Host] | 0.008 | 0.00% | 40 |

## top 10 CUDA API calls by total time

| api | per_step_ms | pct | calls |
|---|---:|---:|---:|
| cudaStreamSynchronize | 2525.874 | 48.00% | 2026 |
| cudaLaunchKernel | 2241.746 | 42.60% | 434995 |
| cudaMemsetAsync | 165.359 | 3.10% | 32145 |
| cuLaunchKernelEx | 142.410 | 2.70% | 32354 |
| cudaMemcpyAsync | 64.717 | 1.20% | 5219 |
| cuMemSetAccess | 47.613 | 0.90% | 459 |
| cudaFree | 25.868 | 0.50% | 2 |
| cuMemCreate | 25.577 | 0.50% | 3718 |
| cudaEventSynchronize | 11.492 | 0.20% | 10 |
| cudaGetDeviceProperties_v2_v12000 | 3.996 | 0.10% | 6 |

## top 10 OS runtime calls by total time

| os call | per_step_ms | pct | calls |
|---|---:|---:|---:|
| poll | 33743.453 | 36.10% | 1911 |
| pthread_cond_wait | 17853.800 | 19.10% | 6234 |
| pthread_cond_timedwait | 17520.815 | 18.70% | 47856 |
| sem_clockwait | 14510.624 | 15.50% | 850 |
| epoll_wait | 8573.709 | 9.20% | 213 |
| sem_wait | 341.746 | 0.40% | 10 |
| read | 324.245 | 0.30% | 3439 |
| ioctl | 188.267 | 0.20% | 19817 |
| usleep | 110.047 | 0.10% | 23916 |
| pthread_mutex_lock | 105.090 | 0.10% | 6559 |

## Δ from previous snapshot

Δ from long-horizon_round54

Δ step_time_ms=-10.792  Δ mfu_e2e_standard=+0.0249  Δ profiled_steps=+0  Δ gpu_kernel_per_step_ms=-611.693  Δ gpu_idle_per_step_ms=+600.901  Δ gpu_memop_per_step_ms=-48.317  Δ cuda_api_per_step_ms=-717.889  Δ os_runtime_per_step_ms=-1644.377

| kernel | Δ per_step_ms | Δ pct |  state |
|---|---:|---:|---|
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | -70.338 | +0.00% | tracked |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | -51.348 | +0.00% | tracked |
| void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_ke... | -43.383 | +0.00% | tracked |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | -39.750 | +0.00% | tracked |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | -39.816 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | -38.020 | +0.00% | tracked |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | -32.269 | +0.00% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)8, at::native::bfloat16_c... | -31.875 | +0.00% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | -31.332 | +0.00% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFuncto... | -28.777 | +0.00% | tracked |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)4, float, float, float, ... | -23.510 | -0.10% | tracked |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | -19.740 | +0.00% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunc... | -11.698 | +0.00% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unna... | -10.911 | +0.00% | tracked |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | -10.747 | +0.00% | tracked |
