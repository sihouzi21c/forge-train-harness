# profile snapshot — profile-snapshot

step_time_ms=4212.056  mfu_e2e_standard=31.2151  profiled_steps=12  gpu_kernel_per_step_ms=2869.369  gpu_idle_per_step_ms=1342.687  gpu_memop_per_step_ms=43.471  cuda_api_per_step_ms=2896.036  os_runtime_per_step_ms=56279.547
# gpu_idle = step - gpu_kernel (valid only when compute kernels share one stream); cuda_api / os_runtime are CPU time, do NOT subtract from step_time.
world_size=2  micro_batch_size=4  seq_length=4096  grad_accum_steps=10
nsys_rep=/user/sunhaojun/.forge_train/77459cccb4da/.artifacts/runs/profile-snapshot-20260815-115349-d70372/ours/profile.nsys-rep

## top 15 GPU kernels by total time

| kernel | per_step_ms | pct | instances |
|---|---:|---:|---:|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | 562.562 | 19.60% | 2054 |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | 280.830 | 9.80% | 4179 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | 262.894 | 9.20% | 6405 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 222.646 | 7.80% | 2268 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | 210.656 | 7.30% | 2331 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | 111.620 | 3.90% | 17711 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 93.157 | 3.20% | 4326 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | 88.727 | 3.10% | 6572 |
| _rope_kernel | 83.465 | 2.90% | 8358 |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | 75.218 | 2.60% | 168 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | 74.748 | 2.60% | 4284 |
| _ce_bwd_kernel | 63.751 | 2.20% | 672 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | 55.674 | 1.90% | 2079 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | 55.442 | 1.90% | 2163 |
| _swiglu_bwd_kernel | 44.628 | 1.60% | 2079 |

## GPU memory operations

| op | per_step_ms | pct | count |
|---|---:|---:|---:|
| [CUDA memcpy Host-to-Device] | 41.238 | 94.90% | 818 |
| [CUDA memset] | 1.931 | 4.40% | 25746 |
| [CUDA memcpy Device-to-Device] | 0.295 | 0.70% | 189 |
| [CUDA memcpy Device-to-Host] | 0.007 | 0.00% | 32 |

## top 10 CUDA API calls by total time

| api | per_step_ms | pct | calls |
|---|---:|---:|---:|
| cudaMemcpyAsync | 1559.227 | 53.80% | 1039 |
| cudaGraphLaunch_v10000 | 747.667 | 25.80% | 82 |
| cudaStreamSynchronize | 437.948 | 15.10% | 349 |
| cudaMalloc | 64.705 | 2.20% | 702 |
| cudaLaunchKernel | 42.420 | 1.50% | 5999 |
| cudaFree | 25.393 | 0.90% | 203 |
| cudaEventSynchronize | 11.203 | 0.40% | 8 |
| cuLibraryLoadData | 1.954 | 0.10% | 19 |
| cudaHostAlloc | 1.649 | 0.10% | 388 |
| cudaGraphInstantiateWithFlags_v11040 | 1.492 | 0.10% | 1 |

## top 10 OS runtime calls by total time

| os call | per_step_ms | pct | calls |
|---|---:|---:|---:|
| poll | 22179.195 | 39.40% | 1459 |
| pthread_cond_timedwait | 11544.350 | 20.50% | 41501 |
| pthread_cond_wait | 7479.480 | 13.30% | 294 |
| epoll_wait | 5798.127 | 10.30% | 449 |
| sem_clockwait | 4166.709 | 7.40% | 5 |
| sem_wait | 3995.492 | 7.10% | 114 |
| read | 280.534 | 0.50% | 4091 |
| nanosleep | 189.536 | 0.30% | 247 |
| pthread_mutex_lock | 172.259 | 0.30% | 5545 |
| ioctl | 124.204 | 0.20% | 7282 |

## Δ from previous snapshot

Δ from long-horizon_round83

Δ step_time_ms=+0.620  Δ mfu_e2e_standard=-0.0048  Δ profiled_steps=+0  Δ gpu_kernel_per_step_ms=-1.709  Δ gpu_idle_per_step_ms=+2.329  Δ gpu_memop_per_step_ms=+10.260  Δ cuda_api_per_step_ms=-142.366  Δ os_runtime_per_step_ms=+400.945

| kernel | Δ per_step_ms | Δ pct |  state |
|---|---:|---:|---|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | -0.508 | +0.00% | tracked |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | -0.182 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | -0.475 | +0.00% | tracked |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | -0.399 | +0.00% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | -0.359 | +0.00% | tracked |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | +0.025 | +0.00% | tracked |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | +0.009 | +0.00% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | +0.016 | +0.00% | tracked |
| _rope_kernel | -0.019 | +0.00% | tracked |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | +0.022 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | -0.061 | +0.00% | tracked |
| _ce_bwd_kernel | -0.064 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | -0.018 | +0.00% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | -0.017 | +0.00% | tracked |
| _swiglu_bwd_kernel | +0.023 | +0.00% | tracked |
