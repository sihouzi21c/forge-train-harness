# profile snapshot — profile-snapshot

step_time_ms=3252.101  mfu_e2e_standard=16.1717  profiled_steps=12  gpu_kernel_per_step_ms=2665.248  gpu_idle_per_step_ms=586.852  gpu_memop_per_step_ms=78.073  cuda_api_per_step_ms=2161.158  os_runtime_per_step_ms=40326.671
# gpu_idle = step - gpu_kernel (valid only when compute kernels share one stream); cuda_api / os_runtime are CPU time, do NOT subtract from step_time.
world_size=2  micro_batch_size=2  seq_length=4096  grad_accum_steps=8
nsys_rep=/user/sunhaojun/.forge_train/77459cccb4da/.artifacts/runs/profile-snapshot-20260813-053321-5299ac/ours/profile.nsys-rep

## top 15 GPU kernels by total time

| kernel | per_step_ms | pct | instances |
|---|---:|---:|---:|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | 326.147 | 12.20% | 2059 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | 251.592 | 9.40% | 33847 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFuncto... | 172.026 | 6.50% | 61564 |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)4, float, float, float, ... | 153.470 | 5.80% | 332 |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | 152.130 | 5.70% | 4137 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | 142.096 | 5.30% | 21937 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | 128.146 | 4.80% | 6436 |
| void at::native::vectorized_elementwise_kernel<(int)8, at::native::bfloat16_c... | 112.982 | 4.20% | 42494 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 109.663 | 4.10% | 2241 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | 104.562 | 3.90% | 2311 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 91.536 | 3.40% | 8276 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 85.301 | 3.20% | 30268 |
| void at::native::<unnamed>::cunn_SoftMaxBackward<(int)4, float, float, float,... | 78.204 | 2.90% | 166 |
| void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_ke... | 67.389 | 2.50% | 18094 |
| void at::native::vectorized_elementwise_kernel<(int)8, at::native::FillFuncto... | 59.372 | 2.20% | 66358 |

## GPU memory operations

| op | per_step_ms | pct | count |
|---|---:|---:|---:|
| [CUDA memcpy Host-to-Device] | 56.122 | 71.90% | 563 |
| [CUDA memcpy Device-to-Device] | 19.340 | 24.80% | 6259 |
| [CUDA memset] | 2.603 | 3.30% | 29915 |
| [CUDA memcpy Device-to-Host] | 0.008 | 0.00% | 40 |

## top 10 CUDA API calls by total time

| api | per_step_ms | pct | calls |
|---|---:|---:|---:|
| cudaLaunchKernel | 1644.128 | 76.10% | 502446 |
| cudaStreamSynchronize | 181.943 | 8.40% | 603 |
| cuLaunchKernelEx | 78.729 | 3.60% | 25629 |
| cudaMemcpyAsync | 64.282 | 3.00% | 6862 |
| cudaDeviceSynchronize | 60.922 | 2.80% | 31 |
| cudaMemsetAsync | 55.729 | 2.60% | 29915 |
| cuMemSetAccess | 35.484 | 1.60% | 481 |
| cuMemCreate | 22.249 | 1.00% | 3855 |
| cudaFree | 10.805 | 0.50% | 2 |
| cudaGetDeviceProperties_v2_v12000 | 2.916 | 0.10% | 6 |

## top 10 OS runtime calls by total time

| os call | per_step_ms | pct | calls |
|---|---:|---:|---:|
| poll | 17068.826 | 42.30% | 1218 |
| pthread_cond_timedwait | 8558.106 | 21.20% | 564 |
| pthread_cond_wait | 5731.590 | 14.20% | 20446 |
| epoll_wait | 4498.955 | 11.20% | 444 |
| sem_clockwait | 3333.367 | 8.30% | 4 |
| read | 288.940 | 0.70% | 3280 |
| ioctl | 187.133 | 0.50% | 18213 |
| nanosleep | 187.027 | 0.50% | 244 |
| usleep | 141.055 | 0.30% | 30241 |
| pthread_mutex_lock | 87.302 | 0.20% | 52 |

## Δ from previous snapshot

Δ from previous: unavailable (no prior snapshot; prev_dir=n/a)
