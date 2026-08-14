# profile snapshot — profile-snapshot

step_time_ms=7221.451  mfu_e2e_standard=18.2070  profiled_steps=12  gpu_kernel_per_step_ms=6027.979  gpu_idle_per_step_ms=1193.472  gpu_memop_per_step_ms=132.039  cuda_api_per_step_ms=6911.351  os_runtime_per_step_ms=92096.924
# gpu_idle = step - gpu_kernel (valid only when compute kernels share one stream); cuda_api / os_runtime are CPU time, do NOT subtract from step_time.
world_size=2  micro_batch_size=4  seq_length=4096  grad_accum_steps=10
nsys_rep=/user/sunhaojun/.forge_train/77459cccb4da/.artifacts/runs/profile-snapshot-20260814-055852-5c2c80/ours/profile.nsys-rep

## top 15 GPU kernels by total time

| kernel | per_step_ms | pct | instances |
|---|---:|---:|---:|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | 680.755 | 11.30% | 2525 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | 604.394 | 10.00% | 41984 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | 593.410 | 9.80% | 32439 |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)4, float, float, float, ... | 378.052 | 6.30% | 1020 |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | 336.205 | 5.60% | 5103 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 334.835 | 5.60% | 15510 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | 317.210 | 5.30% | 7854 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 272.535 | 4.50% | 2757 |
| void at::native::vectorized_elementwise_kernel<(int)8, at::native::bfloat16_c... | 266.044 | 4.40% | 49126 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | 259.170 | 4.30% | 2856 |
| void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_ke... | 198.404 | 3.30% | 27948 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 196.668 | 3.30% | 36537 |
| void at::native::<unnamed>::cunn_SoftMaxBackward<(int)4, float, float, float,... | 192.159 | 3.20% | 808 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFuncto... | 95.470 | 1.60% | 21726 |
| void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unna... | 91.692 | 1.50% | 16123 |

## GPU memory operations

| op | per_step_ms | pct | count |
|---|---:|---:|---:|
| [CUDA memcpy Host-to-Device] | 70.858 | 53.70% | 932 |
| [CUDA memcpy Device-to-Device] | 58.666 | 44.40% | 2543 |
| [CUDA memset] | 2.505 | 1.90% | 31429 |
| [CUDA memcpy Device-to-Host] | 0.010 | 0.00% | 50 |

## top 10 CUDA API calls by total time

| api | per_step_ms | pct | calls |
|---|---:|---:|---:|
| cudaMemcpyAsync | 3139.115 | 45.40% | 2725 |
| cudaGraphLaunch_v10000 | 1856.060 | 26.90% | 101 |
| cudaStreamSynchronize | 921.608 | 13.30% | 369 |
| cuMemCreate | 458.004 | 6.60% | 5720 |
| cuMemSetAccess | 189.398 | 2.70% | 639 |
| cudaLaunchKernel | 118.469 | 1.70% | 15633 |
| cuMemUnmap | 75.907 | 1.10% | 2369 |
| cudaGetDeviceProperties_v2_v12000 | 35.844 | 0.50% | 12 |
| cudaStreamCreateWithPriority | 33.984 | 0.50% | 128 |
| cuMemRelease | 28.251 | 0.40% | 2369 |

## top 10 OS runtime calls by total time

| os call | per_step_ms | pct | calls |
|---|---:|---:|---:|
| poll | 35841.536 | 38.90% | 1935 |
| pthread_cond_timedwait | 18188.923 | 19.70% | 2103 |
| sem_clockwait | 14631.257 | 15.90% | 864 |
| pthread_cond_wait | 10781.847 | 11.70% | 237 |
| epoll_wait | 9154.897 | 9.90% | 54 |
| ioctl | 1719.364 | 1.90% | 27212 |
| pthread_rwlock_wrlock | 381.792 | 0.40% | 46 |
| usleep | 346.707 | 0.40% | 71737 |
| sem_wait | 315.094 | 0.30% | 10 |
| read | 275.757 | 0.30% | 3249 |

## Δ from previous snapshot

Δ from long-horizon_round46

Δ step_time_ms=+21.407  Δ mfu_e2e_standard=-0.0543  Δ profiled_steps=+0  Δ gpu_kernel_per_step_ms=+6027.979  Δ gpu_idle_per_step_ms=-6006.572  Δ gpu_memop_per_step_ms=+132.039  Δ cuda_api_per_step_ms=+6911.312  Δ os_runtime_per_step_ms=+4800.756

| kernel | Δ per_step_ms | Δ pct |  state |
|---|---:|---:|---|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | (new) | (new) | new |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | (new) | (new) | new |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | (new) | (new) | new |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)4, float, float, float, ... | (new) | (new) | new |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | (new) | (new) | new |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | (new) | (new) | new |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | (new) | (new) | new |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | (new) | (new) | new |
| void at::native::vectorized_elementwise_kernel<(int)8, at::native::bfloat16_c... | (new) | (new) | new |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | (new) | (new) | new |
| void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_ke... | (new) | (new) | new |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | (new) | (new) | new |
| void at::native::<unnamed>::cunn_SoftMaxBackward<(int)4, float, float, float,... | (new) | (new) | new |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFuncto... | (new) | (new) | new |
| void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unna... | (new) | (new) | new |
