# profile snapshot — profile-snapshot

step_time_ms=4213.873  mfu_e2e_standard=31.2016  profiled_steps=12  gpu_kernel_per_step_ms=2858.908  gpu_idle_per_step_ms=1354.965  gpu_memop_per_step_ms=44.839  cuda_api_per_step_ms=2956.263  os_runtime_per_step_ms=55366.950
# gpu_idle = step - gpu_kernel (valid only when compute kernels share one stream); cuda_api / os_runtime are CPU time, do NOT subtract from step_time.
world_size=2  micro_batch_size=4  seq_length=4096  grad_accum_steps=10
nsys_rep=/user/sunhaojun/.forge_train/77459cccb4da/.artifacts/runs/profile-snapshot-20260815-070842-b0d548/ours/profile.nsys-rep

## top 15 GPU kernels by total time

| kernel | per_step_ms | pct | instances |
|---|---:|---:|---:|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | 561.629 | 19.60% | 2050 |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | 279.423 | 9.80% | 4159 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | 261.302 | 9.10% | 6391 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 220.163 | 7.70% | 2250 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | 209.032 | 7.30% | 2324 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | 111.180 | 3.90% | 17670 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 92.892 | 3.20% | 4316 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | 88.541 | 3.10% | 6557 |
| _rope_kernel | 82.989 | 2.90% | 8318 |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | 74.287 | 2.60% | 166 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | 74.121 | 2.60% | 4251 |
| _ce_bwd_kernel | 63.014 | 2.20% | 664 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | 55.484 | 1.90% | 2075 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | 55.244 | 1.90% | 2158 |
| _swiglu_bwd_kernel | 44.528 | 1.60% | 2075 |

## GPU memory operations

| op | per_step_ms | pct | count |
|---|---:|---:|---:|
| [CUDA memcpy Host-to-Device] | 42.606 | 95.00% | 818 |
| [CUDA memset] | 1.931 | 4.30% | 25633 |
| [CUDA memcpy Device-to-Device] | 0.295 | 0.70% | 189 |
| [CUDA memcpy Device-to-Host] | 0.007 | 0.00% | 32 |

## top 10 CUDA API calls by total time

| api | per_step_ms | pct | calls |
|---|---:|---:|---:|
| cudaMemcpyAsync | 1560.875 | 52.80% | 1039 |
| cudaGraphLaunch_v10000 | 751.009 | 25.40% | 82 |
| cudaStreamSynchronize | 404.291 | 13.70% | 349 |
| cudaMalloc | 105.328 | 3.60% | 807 |
| cudaLaunchKernel | 77.517 | 2.60% | 6442 |
| cudaFree | 21.469 | 0.70% | 203 |
| cudaEventSynchronize | 12.557 | 0.40% | 8 |
| cudaDeviceSynchronize | 12.413 | 0.40% | 5 |
| cudaMemsetAsync | 2.395 | 0.10% | 648 |
| cudaStreamCreateWithPriority | 2.381 | 0.10% | 128 |

## top 10 OS runtime calls by total time

| os call | per_step_ms | pct | calls |
|---|---:|---:|---:|
| poll | 21900.408 | 39.60% | 1308 |
| pthread_cond_timedwait | 11281.984 | 20.40% | 26087 |
| pthread_cond_wait | 7259.013 | 13.10% | 259 |
| epoll_wait | 5605.724 | 10.10% | 173 |
| sem_clockwait | 4166.709 | 7.50% | 5 |
| sem_wait | 4141.242 | 7.50% | 119 |
| read | 309.586 | 0.60% | 3940 |
| ioctl | 183.227 | 0.30% | 7566 |
| pthread_mutex_lock | 166.853 | 0.30% | 3299 |
| usleep | 94.312 | 0.20% | 20588 |

## Δ from previous snapshot

Δ from long-horizon_round78

Δ step_time_ms=-280.167  Δ mfu_e2e_standard=+1.9430  Δ profiled_steps=+0  Δ gpu_kernel_per_step_ms=-13.828  Δ gpu_idle_per_step_ms=-266.339  Δ gpu_memop_per_step_ms=-30.107  Δ cuda_api_per_step_ms=-154.041  Δ os_runtime_per_step_ms=-3431.241

| kernel | Δ per_step_ms | Δ pct |  state |
|---|---:|---:|---|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | -0.194 | +0.00% | tracked |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | -1.523 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | -2.555 | -0.10% | tracked |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | -2.594 | -0.10% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | -3.493 | -0.10% | tracked |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | -0.382 | +0.00% | tracked |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | -0.324 | +0.00% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | -0.019 | +0.00% | tracked |
| _rope_kernel | -0.396 | +0.00% | tracked |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | -0.021 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | -0.501 | +0.00% | tracked |
| _ce_bwd_kernel | -0.135 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | -0.739 | -0.10% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | -0.540 | +0.00% | tracked |
| _swiglu_bwd_kernel | -0.029 | +0.00% | tracked |
