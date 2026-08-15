# profile snapshot — profile-snapshot

step_time_ms=4495.211  mfu_e2e_standard=29.2498  profiled_steps=12  gpu_kernel_per_step_ms=2834.367  gpu_idle_per_step_ms=1660.844  gpu_memop_per_step_ms=39.543  cuda_api_per_step_ms=3277.186  os_runtime_per_step_ms=59332.537
# gpu_idle = step - gpu_kernel (valid only when compute kernels share one stream); cuda_api / os_runtime are CPU time, do NOT subtract from step_time.
world_size=2  micro_batch_size=4  seq_length=4096  grad_accum_steps=10
nsys_rep=/user/sunhaojun/.forge_train/77459cccb4da/.artifacts/runs/profile-snapshot-20260815-025633-54e079/ours/profile.nsys-rep

## top 15 GPU kernels by total time

| kernel | per_step_ms | pct | instances |
|---|---:|---:|---:|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | 556.387 | 19.60% | 2020 |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | 277.446 | 9.80% | 4095 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | 261.041 | 9.20% | 6299 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 220.447 | 7.80% | 2214 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | 210.369 | 7.40% | 2291 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | 110.082 | 3.90% | 17417 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 91.580 | 3.20% | 4254 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | 87.281 | 3.10% | 6462 |
| _rope_kernel | 83.714 | 3.00% | 8190 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | 73.634 | 2.60% | 4182 |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | 73.414 | 2.60% | 164 |
| _ce_bwd_kernel | 62.481 | 2.20% | 656 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | 55.594 | 2.00% | 2045 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | 55.105 | 1.90% | 2127 |
| _swiglu_bwd_kernel | 43.931 | 1.50% | 2045 |

## GPU memory operations

| op | per_step_ms | pct | count |
|---|---:|---:|---:|
| [CUDA memcpy Host-to-Device] | 37.443 | 94.70% | 806 |
| [CUDA memset] | 1.800 | 4.60% | 25242 |
| [CUDA memcpy Device-to-Device] | 0.293 | 0.70% | 189 |
| [CUDA memcpy Device-to-Host] | 0.007 | 0.00% | 32 |

## top 10 CUDA API calls by total time

| api | per_step_ms | pct | calls |
|---|---:|---:|---:|
| cudaMemcpyAsync | 1538.023 | 46.90% | 1027 |
| cudaGraphLaunch_v10000 | 744.178 | 22.70% | 80 |
| cudaStreamSynchronize | 566.168 | 17.30% | 352 |
| cuMemCreate | 89.780 | 2.70% | 5996 |
| cuMemSetAccess | 83.435 | 2.50% | 705 |
| cudaLaunchKernel | 79.040 | 2.40% | 6422 |
| cudaStreamCreateWithPriority | 71.447 | 2.20% | 128 |
| cudaFree | 23.222 | 0.70% | 2 |
| cuMemRelease | 15.579 | 0.50% | 2143 |
| cuMemUnmap | 13.346 | 0.40% | 2143 |

## top 10 OS runtime calls by total time

| os call | per_step_ms | pct | calls |
|---|---:|---:|---:|
| poll | 22900.135 | 38.60% | 1345 |
| pthread_cond_timedwait | 11803.560 | 19.90% | 11810 |
| pthread_cond_wait | 7688.674 | 13.00% | 242 |
| epoll_wait | 5867.558 | 9.90% | 54 |
| sem_clockwait | 5000.050 | 8.40% | 6 |
| sem_wait | 4609.950 | 7.80% | 119 |
| ioctl | 391.533 | 0.70% | 26594 |
| read | 342.049 | 0.60% | 3723 |
| usleep | 191.387 | 0.30% | 42044 |
| pthread_rwlock_wrlock | 126.292 | 0.20% | 33 |

## Δ from previous snapshot

Δ from long-horizon_round70

Δ step_time_ms=+1.648  Δ mfu_e2e_standard=-0.0116  Δ profiled_steps=+0  Δ gpu_kernel_per_step_ms=-0.348  Δ gpu_idle_per_step_ms=+1.996  Δ gpu_memop_per_step_ms=-16.424  Δ cuda_api_per_step_ms=+8.984  Δ os_runtime_per_step_ms=+837.710

| kernel | Δ per_step_ms | Δ pct |  state |
|---|---:|---:|---|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | -0.154 | +0.00% | tracked |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | +0.229 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | -0.057 | +0.00% | tracked |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | +0.125 | +0.00% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | -0.092 | +0.00% | tracked |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | -0.028 | +0.00% | tracked |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | -0.039 | +0.00% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | -0.032 | +0.00% | tracked |
| _rope_kernel | +0.025 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | +0.022 | +0.00% | tracked |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | -0.004 | +0.00% | tracked |
| _ce_bwd_kernel | +0.004 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | -0.022 | +0.00% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | -0.019 | +0.00% | tracked |
| _swiglu_bwd_kernel | -0.018 | -0.10% | tracked |
