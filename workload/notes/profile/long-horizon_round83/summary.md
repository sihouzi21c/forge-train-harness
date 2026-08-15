# profile snapshot — profile-snapshot

step_time_ms=4211.436  mfu_e2e_standard=31.2199  profiled_steps=12  gpu_kernel_per_step_ms=2871.078  gpu_idle_per_step_ms=1340.358  gpu_memop_per_step_ms=33.211  cuda_api_per_step_ms=3038.402  os_runtime_per_step_ms=55878.602
# gpu_idle = step - gpu_kernel (valid only when compute kernels share one stream); cuda_api / os_runtime are CPU time, do NOT subtract from step_time.
world_size=2  micro_batch_size=4  seq_length=4096  grad_accum_steps=10
nsys_rep=/user/sunhaojun/.forge_train/77459cccb4da/.artifacts/runs/profile-snapshot-20260815-104221-b06676/ours/profile.nsys-rep

## top 15 GPU kernels by total time

| kernel | per_step_ms | pct | instances |
|---|---:|---:|---:|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | 563.070 | 19.60% | 2053 |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | 281.012 | 9.80% | 4178 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | 263.369 | 9.20% | 6402 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 223.045 | 7.80% | 2268 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | 211.015 | 7.30% | 2330 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | 111.595 | 3.90% | 17704 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 93.148 | 3.20% | 4326 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | 88.711 | 3.10% | 6571 |
| _rope_kernel | 83.484 | 2.90% | 8356 |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | 75.196 | 2.60% | 168 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | 74.809 | 2.60% | 4284 |
| _ce_bwd_kernel | 63.815 | 2.20% | 672 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | 55.692 | 1.90% | 2078 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | 55.459 | 1.90% | 2163 |
| _swiglu_bwd_kernel | 44.605 | 1.60% | 2078 |

## GPU memory operations

| op | per_step_ms | pct | count |
|---|---:|---:|---:|
| [CUDA memcpy Host-to-Device] | 31.023 | 93.40% | 818 |
| [CUDA memset] | 1.886 | 5.70% | 25741 |
| [CUDA memcpy Device-to-Device] | 0.295 | 0.90% | 189 |
| [CUDA memcpy Device-to-Host] | 0.007 | 0.00% | 32 |

## top 10 CUDA API calls by total time

| api | per_step_ms | pct | calls |
|---|---:|---:|---:|
| cudaMemcpyAsync | 1552.173 | 51.10% | 1039 |
| cudaGraphLaunch_v10000 | 750.749 | 24.70% | 82 |
| cudaStreamSynchronize | 442.991 | 14.60% | 349 |
| cudaStreamCreateWithPriority | 82.097 | 2.70% | 128 |
| cudaMalloc | 80.940 | 2.70% | 702 |
| cudaFree | 56.797 | 1.90% | 203 |
| cudaLaunchKernel | 50.764 | 1.70% | 5999 |
| cudaEventSynchronize | 11.235 | 0.40% | 8 |
| cudaHostAlloc | 3.358 | 0.10% | 388 |
| cudaGetDeviceProperties_v2_v12000 | 2.234 | 0.10% | 2 |

## top 10 OS runtime calls by total time

| os call | per_step_ms | pct | calls |
|---|---:|---:|---:|
| poll | 21882.266 | 39.20% | 1254 |
| pthread_cond_timedwait | 11480.217 | 20.50% | 35110 |
| pthread_cond_wait | 7495.066 | 13.40% | 288 |
| epoll_wait | 5606.054 | 10.00% | 51 |
| sem_clockwait | 4166.709 | 7.50% | 5 |
| sem_wait | 3963.834 | 7.10% | 114 |
| read | 333.027 | 0.60% | 3766 |
| ioctl | 256.989 | 0.50% | 7315 |
| usleep | 168.496 | 0.30% | 36796 |
| accept | 136.165 | 0.20% | 7236 |

## Δ from previous snapshot

Δ from long-horizon_round82

Δ step_time_ms=-3.006  Δ mfu_e2e_standard=+0.0224  Δ profiled_steps=+0  Δ gpu_kernel_per_step_ms=-1.250  Δ gpu_idle_per_step_ms=-1.756  Δ gpu_memop_per_step_ms=-25.123  Δ cuda_api_per_step_ms=+71.071  Δ os_runtime_per_step_ms=+229.233

| kernel | Δ per_step_ms | Δ pct |  state |
|---|---:|---:|---|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | -0.186 | +0.00% | tracked |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | +0.093 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | -0.483 | +0.00% | tracked |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | +0.012 | +0.00% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | -0.195 | -0.10% | tracked |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | -0.056 | +0.00% | tracked |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | -0.039 | +0.00% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | -0.015 | +0.00% | tracked |
| _rope_kernel | +0.001 | +0.00% | tracked |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | -0.001 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | +0.016 | +0.00% | tracked |
| _ce_bwd_kernel | -0.003 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | -0.067 | +0.00% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | -0.025 | +0.00% | tracked |
| _swiglu_bwd_kernel | -0.027 | +0.00% | tracked |
