# profile snapshot — profile-snapshot

step_time_ms=4221.084  mfu_e2e_standard=31.1484  profiled_steps=12  gpu_kernel_per_step_ms=2860.715  gpu_idle_per_step_ms=1360.369  gpu_memop_per_step_ms=60.901  cuda_api_per_step_ms=2907.612  os_runtime_per_step_ms=55587.014
# gpu_idle = step - gpu_kernel (valid only when compute kernels share one stream); cuda_api / os_runtime are CPU time, do NOT subtract from step_time.
world_size=2  micro_batch_size=4  seq_length=4096  grad_accum_steps=10
nsys_rep=/user/sunhaojun/.forge_train/77459cccb4da/.artifacts/runs/profile-snapshot-20260815-074820-ba2670/ours/profile.nsys-rep

## top 15 GPU kernels by total time

| kernel | per_step_ms | pct | instances |
|---|---:|---:|---:|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | 562.016 | 19.60% | 2050 |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | 279.552 | 9.80% | 4162 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | 261.395 | 9.10% | 6391 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 220.637 | 7.70% | 2253 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | 209.160 | 7.30% | 2324 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | 111.234 | 3.90% | 17670 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 92.946 | 3.20% | 4316 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | 88.533 | 3.10% | 6557 |
| _rope_kernel | 83.104 | 2.90% | 8324 |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | 74.286 | 2.60% | 166 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | 74.222 | 2.60% | 4256 |
| _ce_bwd_kernel | 63.061 | 2.20% | 664 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | 55.567 | 1.90% | 2075 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | 55.311 | 1.90% | 2158 |
| _swiglu_bwd_kernel | 44.534 | 1.60% | 2075 |

## GPU memory operations

| op | per_step_ms | pct | count |
|---|---:|---:|---:|
| [CUDA memcpy Host-to-Device] | 58.702 | 96.40% | 818 |
| [CUDA memset] | 1.897 | 3.10% | 25644 |
| [CUDA memcpy Device-to-Device] | 0.295 | 0.50% | 189 |
| [CUDA memcpy Device-to-Host] | 0.007 | 0.00% | 32 |

## top 10 CUDA API calls by total time

| api | per_step_ms | pct | calls |
|---|---:|---:|---:|
| cudaMemcpyAsync | 1572.041 | 54.10% | 1039 |
| cudaGraphLaunch_v10000 | 751.439 | 25.80% | 82 |
| cudaStreamSynchronize | 409.364 | 14.10% | 349 |
| cudaLaunchKernel | 72.149 | 2.50% | 6442 |
| cudaMalloc | 53.185 | 1.80% | 807 |
| cudaFree | 28.356 | 1.00% | 203 |
| cudaEventSynchronize | 12.661 | 0.40% | 8 |
| cuLibraryLoadData | 2.231 | 0.10% | 19 |
| cudaGraphInstantiateWithFlags_v11040 | 1.433 | 0.00% | 1 |
| cudaMemsetAsync | 0.917 | 0.00% | 648 |

## top 10 OS runtime calls by total time

| os call | per_step_ms | pct | calls |
|---|---:|---:|---:|
| poll | 21494.584 | 38.70% | 1431 |
| pthread_cond_timedwait | 11109.387 | 20.00% | 25498 |
| pthread_cond_wait | 8034.129 | 14.50% | 272 |
| epoll_wait | 5617.103 | 10.10% | 444 |
| sem_clockwait | 4166.708 | 7.50% | 5 |
| sem_wait | 4084.532 | 7.30% | 119 |
| read | 307.515 | 0.60% | 4082 |
| nanosleep | 187.154 | 0.30% | 244 |
| pthread_mutex_lock | 171.419 | 0.30% | 3967 |
| ioctl | 141.783 | 0.30% | 7678 |

## Δ from previous snapshot

Δ from long-horizon_round79

Δ step_time_ms=+7.211  Δ mfu_e2e_standard=-0.0532  Δ profiled_steps=+0  Δ gpu_kernel_per_step_ms=+1.807  Δ gpu_idle_per_step_ms=+5.404  Δ gpu_memop_per_step_ms=+16.062  Δ cuda_api_per_step_ms=-48.651  Δ os_runtime_per_step_ms=+220.064

| kernel | Δ per_step_ms | Δ pct |  state |
|---|---:|---:|---|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | +0.387 | +0.00% | tracked |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | +0.129 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | +0.093 | +0.00% | tracked |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | +0.474 | +0.00% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | +0.128 | +0.00% | tracked |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | +0.054 | +0.00% | tracked |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | +0.054 | +0.00% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | -0.008 | +0.00% | tracked |
| _rope_kernel | +0.115 | +0.00% | tracked |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | -0.001 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | +0.101 | +0.00% | tracked |
| _ce_bwd_kernel | +0.047 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | +0.083 | +0.00% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | +0.067 | +0.00% | tracked |
| _swiglu_bwd_kernel | +0.006 | +0.00% | tracked |
