# profile snapshot — profile-snapshot

step_time_ms=7254.107  mfu_e2e_standard=18.1259  profiled_steps=12  gpu_kernel_per_step_ms=6105.911  gpu_idle_per_step_ms=1148.196  gpu_memop_per_step_ms=162.305  cuda_api_per_step_ms=5944.078  os_runtime_per_step_ms=80049.035
# gpu_idle = step - gpu_kernel (valid only when compute kernels share one stream); cuda_api / os_runtime are CPU time, do NOT subtract from step_time.
world_size=2  micro_batch_size=4  seq_length=4096  grad_accum_steps=10
nsys_rep=/user/sunhaojun/.forge_train/77459cccb4da/.artifacts/runs/profile-snapshot-20260813-114309-9f6803/ours/profile.nsys-rep

## top 15 GPU kernels by total time

| kernel | per_step_ms | pct | instances |
|---|---:|---:|---:|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | 666.431 | 10.90% | 2465 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | 595.724 | 9.80% | 44522 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | 410.554 | 6.70% | 30080 |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)4, float, float, float, ... | 400.207 | 6.60% | 1080 |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | 357.008 | 5.80% | 5395 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 353.302 | 5.80% | 16180 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | 331.742 | 5.40% | 8301 |
| void at::native::vectorized_elementwise_kernel<(int)8, at::native::bfloat16_c... | 309.787 | 5.10% | 56070 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 286.317 | 4.70% | 2916 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | 270.473 | 4.40% | 3019 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 212.887 | 3.50% | 38975 |
| void at::native::<unnamed>::cunn_SoftMaxBackward<(int)4, float, float, float,... | 188.352 | 3.10% | 792 |
| void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_ke... | 167.657 | 2.70% | 23702 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>:... | 134.954 | 2.20% | 8090 |
| void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unna... | 99.744 | 1.60% | 17476 |

## GPU memory operations

| op | per_step_ms | pct | count |
|---|---:|---:|---:|
| [CUDA memcpy Device-to-Device] | 99.502 | 61.30% | 9595 |
| [CUDA memcpy Host-to-Device] | 59.353 | 36.60% | 638 |
| [CUDA memset] | 3.442 | 2.10% | 38545 |
| [CUDA memcpy Device-to-Host] | 0.008 | 0.00% | 40 |

## top 10 CUDA API calls by total time

| api | per_step_ms | pct | calls |
|---|---:|---:|---:|
| cudaLaunchKernel | 4287.288 | 72.10% | 498926 |
| cudaStreamSynchronize | 576.789 | 9.70% | 678 |
| cudaMemsetAsync | 240.234 | 4.00% | 38545 |
| cuMemCreate | 224.620 | 3.80% | 3672 |
| cuLaunchKernelEx | 161.046 | 2.70% | 33440 |
| cuMemSetAccess | 129.767 | 2.20% | 453 |
| cudaMemcpyAsync | 101.796 | 1.70% | 10273 |
| cudaStreamCreateWithPriority | 91.734 | 1.50% | 128 |
| cudaFree | 88.786 | 1.50% | 2 |
| cudaEventSynchronize | 21.463 | 0.40% | 10 |

## top 10 OS runtime calls by total time

| os call | per_step_ms | pct | calls |
|---|---:|---:|---:|
| poll | 34052.352 | 42.50% | 2218 |
| pthread_cond_timedwait | 17324.697 | 21.60% | 1089 |
| pthread_cond_wait | 9690.737 | 12.10% | 26975 |
| epoll_wait | 9021.086 | 11.30% | 787 |
| sem_clockwait | 7500.076 | 9.40% | 9 |
| ioctl | 1012.917 | 1.30% | 19446 |
| nanosleep | 331.212 | 0.40% | 416 |
| read | 309.298 | 0.40% | 3517 |
| usleep | 278.878 | 0.30% | 59968 |
| accept | 194.735 | 0.20% | 2093 |

## Δ from previous snapshot

Δ from long-horizon_round23

Δ step_time_ms=-1.885  Δ mfu_e2e_standard=+0.0047  Δ profiled_steps=+0  Δ gpu_kernel_per_step_ms=-3.827  Δ gpu_idle_per_step_ms=+1.942  Δ gpu_memop_per_step_ms=+28.330  Δ cuda_api_per_step_ms=+531.237  Δ os_runtime_per_step_ms=+3180.617

| kernel | Δ per_step_ms | Δ pct |  state |
|---|---:|---:|---|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | -4.045 | -0.10% | tracked |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | +0.385 | +0.10% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | -0.901 | +0.00% | tracked |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)4, float, float, float, ... | +0.036 | +0.10% | tracked |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | +0.387 | +0.00% | tracked |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | +0.620 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | +0.693 | +0.00% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)8, at::native::bfloat16_c... | +0.279 | +0.00% | tracked |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | +0.339 | +0.00% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | +0.435 | +0.00% | tracked |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | -0.004 | +0.00% | tracked |
| void at::native::<unnamed>::cunn_SoftMaxBackward<(int)4, float, float, float,... | -1.953 | +0.00% | tracked |
| void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_ke... | -0.018 | +0.00% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>:... | +0.260 | +0.00% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unna... | +0.175 | +0.00% | tracked |
