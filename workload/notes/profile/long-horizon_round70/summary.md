# profile snapshot — profile-snapshot

step_time_ms=4493.563  mfu_e2e_standard=29.2614  profiled_steps=12  gpu_kernel_per_step_ms=2834.715  gpu_idle_per_step_ms=1658.848  gpu_memop_per_step_ms=55.967  cuda_api_per_step_ms=3268.202  os_runtime_per_step_ms=58494.827
# gpu_idle = step - gpu_kernel (valid only when compute kernels share one stream); cuda_api / os_runtime are CPU time, do NOT subtract from step_time.
world_size=2  micro_batch_size=4  seq_length=4096  grad_accum_steps=10
nsys_rep=/user/sunhaojun/.forge_train/77459cccb4da/.artifacts/runs/profile-snapshot-20260814-234940-d89522/ours/profile.nsys-rep

## top 15 GPU kernels by total time

| kernel | per_step_ms | pct | instances |
|---|---:|---:|---:|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | 556.541 | 19.60% | 2021 |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | 277.217 | 9.80% | 4096 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | 261.098 | 9.20% | 6301 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 220.322 | 7.80% | 2214 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | 210.461 | 7.40% | 2292 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | 110.110 | 3.90% | 17423 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 91.619 | 3.20% | 4256 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | 87.313 | 3.10% | 6465 |
| _rope_kernel | 83.689 | 3.00% | 8191 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | 73.612 | 2.60% | 4182 |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | 73.418 | 2.60% | 164 |
| _ce_bwd_kernel | 62.477 | 2.20% | 656 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | 55.616 | 2.00% | 2046 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | 55.124 | 1.90% | 2128 |
| _swiglu_bwd_kernel | 43.949 | 1.60% | 2046 |

## GPU memory operations

| op | per_step_ms | pct | count |
|---|---:|---:|---:|
| [CUDA memcpy Host-to-Device] | 53.826 | 96.20% | 806 |
| [CUDA memset] | 1.839 | 3.30% | 25248 |
| [CUDA memcpy Device-to-Device] | 0.295 | 0.50% | 189 |
| [CUDA memcpy Device-to-Host] | 0.007 | 0.00% | 32 |

## top 10 CUDA API calls by total time

| api | per_step_ms | pct | calls |
|---|---:|---:|---:|
| cudaMemcpyAsync | 1569.822 | 48.00% | 1027 |
| cudaGraphLaunch_v10000 | 743.057 | 22.70% | 80 |
| cudaStreamSynchronize | 575.362 | 17.60% | 352 |
| cuMemCreate | 99.460 | 3.00% | 5996 |
| cudaLaunchKernel | 86.695 | 2.70% | 6422 |
| cuMemSetAccess | 80.547 | 2.50% | 705 |
| cudaGetDeviceProperties_v2_v12000 | 33.660 | 1.00% | 12 |
| cudaFree | 15.759 | 0.50% | 2 |
| cuMemRelease | 13.953 | 0.40% | 2143 |
| cuMemUnmap | 13.850 | 0.40% | 2143 |

## top 10 OS runtime calls by total time

| os call | per_step_ms | pct | calls |
|---|---:|---:|---:|
| poll | 22746.310 | 38.90% | 1435 |
| pthread_cond_timedwait | 11496.328 | 19.70% | 7543 |
| pthread_cond_wait | 7584.556 | 13.00% | 219 |
| epoll_wait | 5870.856 | 10.00% | 296 |
| sem_clockwait | 5000.052 | 8.50% | 6 |
| sem_wait | 4589.286 | 7.80% | 119 |
| ioctl | 313.229 | 0.50% | 26137 |
| read | 295.120 | 0.50% | 3757 |
| nanosleep | 124.974 | 0.20% | 170 |
| usleep | 115.729 | 0.20% | 25126 |

## Δ from previous snapshot

Δ from long-horizon_round69

Δ step_time_ms=+7.842  Δ mfu_e2e_standard=-0.0505  Δ profiled_steps=+0  Δ gpu_kernel_per_step_ms=+1.242  Δ gpu_idle_per_step_ms=+6.600  Δ gpu_memop_per_step_ms=+4.481  Δ cuda_api_per_step_ms=+3.262  Δ os_runtime_per_step_ms=-426.568

| kernel | Δ per_step_ms | Δ pct |  state |
|---|---:|---:|---|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | +0.313 | +0.00% | tracked |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | -0.098 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | +0.136 | +0.00% | tracked |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | +0.051 | +0.00% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | +0.146 | +0.00% | tracked |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | +0.030 | +0.00% | tracked |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | +0.061 | +0.00% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | +0.036 | +0.00% | tracked |
| _rope_kernel | -0.001 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | +0.035 | +0.00% | tracked |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | +0.005 | +0.00% | tracked |
| _ce_bwd_kernel | -0.044 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | +0.081 | +0.00% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | +0.080 | +0.00% | tracked |
| _swiglu_bwd_kernel | +0.032 | +0.10% | tracked |
