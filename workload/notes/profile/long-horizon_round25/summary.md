# profile snapshot — profile-snapshot

step_time_ms=7167.399  mfu_e2e_standard=18.3472  profiled_steps=12  gpu_kernel_per_step_ms=4908.787  gpu_idle_per_step_ms=2258.611  gpu_memop_per_step_ms=114.434  cuda_api_per_step_ms=5807.139  os_runtime_per_step_ms=80834.836
# gpu_idle = step - gpu_kernel (valid only when compute kernels share one stream); cuda_api / os_runtime are CPU time, do NOT subtract from step_time.
world_size=2  micro_batch_size=4  seq_length=4096  grad_accum_steps=10
nsys_rep=/user/sunhaojun/.forge_train/77459cccb4da/.artifacts/runs/profile-snapshot-20260813-115246-28c470/ours/profile.nsys-rep

## top 15 GPU kernels by total time

| kernel | per_step_ms | pct | instances |
|---|---:|---:|---:|
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | 600.433 | 12.20% | 32779 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | 568.859 | 11.60% | 37291 |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)4, float, float, float, ... | 381.689 | 7.80% | 1030 |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | 342.606 | 7.00% | 5175 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 340.030 | 6.90% | 15706 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | 316.833 | 6.50% | 7931 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 274.942 | 5.60% | 2807 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | 258.398 | 5.30% | 2884 |
| void at::native::vectorized_elementwise_kernel<(int)8, at::native::bfloat16_c... | 248.083 | 5.10% | 43066 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 171.787 | 3.50% | 31877 |
| void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_ke... | 164.277 | 3.30% | 22407 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFuncto... | 96.801 | 2.00% | 21939 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | 91.770 | 1.90% | 5304 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>:... | 86.328 | 1.80% | 5175 |
| void at::native::vectorized_elementwise_kernel<(int)8, at::native::AUnaryFunc... | 78.284 | 1.60% | 11279 |

## GPU memory operations

| op | per_step_ms | pct | count |
|---|---:|---:|---:|
| [CUDA memcpy Device-to-Device] | 60.918 | 53.20% | 4225 |
| [CUDA memcpy Host-to-Device] | 50.646 | 44.30% | 626 |
| [CUDA memset] | 2.862 | 2.50% | 31827 |
| [CUDA memcpy Device-to-Host] | 0.008 | 0.00% | 40 |

## top 10 CUDA API calls by total time

| api | per_step_ms | pct | calls |
|---|---:|---:|---:|
| cudaLaunchKernel | 3868.066 | 66.60% | 389799 |
| cudaStreamSynchronize | 809.959 | 13.90% | 666 |
| cuMemCreate | 357.860 | 6.20% | 3263 |
| cudaMemsetAsync | 273.563 | 4.70% | 31827 |
| cuLaunchKernelEx | 226.045 | 3.90% | 32033 |
| cuMemSetAccess | 106.235 | 1.80% | 448 |
| cudaMemcpyAsync | 59.449 | 1.00% | 4891 |
| cudaFree | 51.747 | 0.90% | 2 |
| cudaGetDeviceProperties_v2_v12000 | 28.985 | 0.50% | 6 |
| cudaEventSynchronize | 21.439 | 0.40% | 10 |

## top 10 OS runtime calls by total time

| os call | per_step_ms | pct | calls |
|---|---:|---:|---:|
| poll | 34646.969 | 42.90% | 2176 |
| pthread_cond_timedwait | 17458.985 | 21.60% | 1084 |
| pthread_cond_wait | 9614.737 | 11.90% | 8015 |
| epoll_wait | 9057.212 | 11.20% | 642 |
| sem_clockwait | 7500.076 | 9.30% | 9 |
| ioctl | 1183.987 | 1.50% | 19432 |
| usleep | 287.075 | 0.40% | 62610 |
| read | 282.942 | 0.40% | 3567 |
| nanosleep | 269.197 | 0.30% | 342 |
| accept | 169.198 | 0.20% | 85 |

## Δ from previous snapshot

Δ from long-horizon_round24

Δ step_time_ms=-86.708  Δ mfu_e2e_standard=+0.2213  Δ profiled_steps=+0  Δ gpu_kernel_per_step_ms=-1197.124  Δ gpu_idle_per_step_ms=+1110.415  Δ gpu_memop_per_step_ms=-47.871  Δ cuda_api_per_step_ms=-136.939  Δ os_runtime_per_step_ms=+785.801

| kernel | Δ per_step_ms | Δ pct |  state |
|---|---:|---:|---|
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | +189.879 | +5.50% | tracked |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | -26.865 | +1.80% | tracked |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)4, float, float, float, ... | -18.518 | +1.20% | tracked |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | -14.402 | +1.20% | tracked |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | -13.272 | +1.10% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | -14.909 | +1.10% | tracked |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | -11.375 | +0.90% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | -12.075 | +0.90% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)8, at::native::bfloat16_c... | -61.704 | +0.00% | tracked |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | -41.100 | +0.00% | tracked |
| void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_ke... | -3.380 | +0.60% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFuncto... | (new) | (new) | new |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | (new) | (new) | new |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>:... | -48.626 | -0.40% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)8, at::native::AUnaryFunc... | (new) | (new) | new |
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | -666.431 | -10.90% | dropped |
| void at::native::<unnamed>::cunn_SoftMaxBackward<(int)4, float, float, float,... | -188.352 | -3.10% | dropped |
| void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unna... | -99.744 | -1.60% | dropped |
