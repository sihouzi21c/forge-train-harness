# profile snapshot — profile-snapshot

step_time_ms=7255.992  mfu_e2e_standard=18.1212  profiled_steps=12  gpu_kernel_per_step_ms=6109.738  gpu_idle_per_step_ms=1146.254  gpu_memop_per_step_ms=133.975  cuda_api_per_step_ms=5412.841  os_runtime_per_step_ms=76868.418
# gpu_idle = step - gpu_kernel (valid only when compute kernels share one stream); cuda_api / os_runtime are CPU time, do NOT subtract from step_time.
world_size=2  micro_batch_size=4  seq_length=4096  grad_accum_steps=10
nsys_rep=/user/sunhaojun/.forge_train/77459cccb4da/.artifacts/runs/profile-snapshot-20260813-113214-3e3f4c/ours/profile.nsys-rep

## top 15 GPU kernels by total time

| kernel | per_step_ms | pct | instances |
|---|---:|---:|---:|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | 670.476 | 11.00% | 2480 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | 595.339 | 9.70% | 44469 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | 411.455 | 6.70% | 30198 |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)4, float, float, float, ... | 400.171 | 6.50% | 1080 |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | 356.621 | 5.80% | 5387 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 352.682 | 5.80% | 16148 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | 331.049 | 5.40% | 8277 |
| void at::native::vectorized_elementwise_kernel<(int)8, at::native::bfloat16_c... | 309.508 | 5.10% | 56016 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 285.978 | 4.70% | 2916 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | 270.038 | 4.40% | 3011 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 212.891 | 3.50% | 38963 |
| void at::native::<unnamed>::cunn_SoftMaxBackward<(int)4, float, float, float,... | 190.305 | 3.10% | 800 |
| void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_ke... | 167.675 | 2.70% | 23713 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>:... | 134.694 | 2.20% | 8074 |
| void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unna... | 99.569 | 1.60% | 17444 |

## GPU memory operations

| op | per_step_ms | pct | count |
|---|---:|---:|---:|
| [CUDA memcpy Device-to-Device] | 99.748 | 74.50% | 9629 |
| [CUDA memcpy Host-to-Device] | 30.778 | 23.00% | 638 |
| [CUDA memset] | 3.441 | 2.60% | 38523 |
| [CUDA memcpy Device-to-Host] | 0.008 | 0.00% | 40 |

## top 10 CUDA API calls by total time

| api | per_step_ms | pct | calls |
|---|---:|---:|---:|
| cudaLaunchKernel | 4234.986 | 78.20% | 500055 |
| cudaStreamSynchronize | 496.408 | 9.20% | 678 |
| cudaMemsetAsync | 224.865 | 4.20% | 38523 |
| cuLaunchKernelEx | 160.768 | 3.00% | 33376 |
| cudaDeviceSynchronize | 89.861 | 1.70% | 10 |
| cudaMemcpyAsync | 72.636 | 1.30% | 10307 |
| cudaFree | 37.151 | 0.70% | 2 |
| cuMemCreate | 35.395 | 0.70% | 3672 |
| cuMemSetAccess | 33.919 | 0.60% | 453 |
| cudaEventSynchronize | 21.698 | 0.40% | 10 |

## top 10 OS runtime calls by total time

| os call | per_step_ms | pct | calls |
|---|---:|---:|---:|
| poll | 32984.858 | 42.90% | 1900 |
| pthread_cond_timedwait | 16686.650 | 21.70% | 1036 |
| pthread_cond_wait | 9676.899 | 12.60% | 26975 |
| epoll_wait | 8506.787 | 11.10% | 274 |
| sem_clockwait | 7500.077 | 9.80% | 9 |
| ioctl | 370.433 | 0.50% | 19433 |
| usleep | 287.954 | 0.40% | 62306 |
| read | 282.152 | 0.40% | 3283 |
| accept | 127.233 | 0.20% | 10054 |
| nanosleep | 115.782 | 0.20% | 159 |

## Δ from previous snapshot

Δ from previous: unavailable (shape mismatch on ['micro_batch_size', 'grad_accum_steps']; prev_dir=bitwise-perf_round0)
