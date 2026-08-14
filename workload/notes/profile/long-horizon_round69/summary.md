# profile snapshot — profile-snapshot

step_time_ms=4485.721  mfu_e2e_standard=29.3119  profiled_steps=12  gpu_kernel_per_step_ms=2833.473  gpu_idle_per_step_ms=1652.248  gpu_memop_per_step_ms=51.486  cuda_api_per_step_ms=3264.940  os_runtime_per_step_ms=58921.395
# gpu_idle = step - gpu_kernel (valid only when compute kernels share one stream); cuda_api / os_runtime are CPU time, do NOT subtract from step_time.
world_size=2  micro_batch_size=4  seq_length=4096  grad_accum_steps=10
nsys_rep=/user/sunhaojun/.forge_train/77459cccb4da/.artifacts/runs/profile-snapshot-20260814-234408-87ed78/ours/profile.nsys-rep

## top 15 GPU kernels by total time

| kernel | per_step_ms | pct | instances |
|---|---:|---:|---:|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | 556.228 | 19.60% | 2020 |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | 277.315 | 9.80% | 4095 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | 260.962 | 9.20% | 6299 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 220.271 | 7.80% | 2214 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | 210.315 | 7.40% | 2291 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | 110.080 | 3.90% | 17418 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 91.558 | 3.20% | 4254 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | 87.277 | 3.10% | 6463 |
| _rope_kernel | 83.690 | 3.00% | 8190 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | 73.577 | 2.60% | 4182 |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | 73.413 | 2.60% | 164 |
| _ce_bwd_kernel | 62.521 | 2.20% | 656 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | 55.535 | 2.00% | 2045 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | 55.044 | 1.90% | 2127 |
| _swiglu_bwd_kernel | 43.917 | 1.50% | 2045 |

## GPU memory operations

| op | per_step_ms | pct | count |
|---|---:|---:|---:|
| [CUDA memcpy Host-to-Device] | 49.385 | 95.90% | 806 |
| [CUDA memset] | 1.799 | 3.50% | 25243 |
| [CUDA memcpy Device-to-Device] | 0.295 | 0.60% | 189 |
| [CUDA memcpy Device-to-Host] | 0.007 | 0.00% | 32 |

## top 10 CUDA API calls by total time

| api | per_step_ms | pct | calls |
|---|---:|---:|---:|
| cudaMemcpyAsync | 1559.817 | 47.80% | 1027 |
| cudaGraphLaunch_v10000 | 742.291 | 22.70% | 80 |
| cudaStreamSynchronize | 567.342 | 17.40% | 352 |
| cuMemSetAccess | 102.519 | 3.10% | 705 |
| cuMemCreate | 94.449 | 2.90% | 5996 |
| cudaLaunchKernel | 91.869 | 2.80% | 6422 |
| cudaFree | 25.472 | 0.80% | 2 |
| cuMemUnmap | 22.863 | 0.70% | 2143 |
| cuMemRelease | 20.938 | 0.60% | 2143 |
| cudaEventSynchronize | 12.473 | 0.40% | 8 |

## top 10 OS runtime calls by total time

| os call | per_step_ms | pct | calls |
|---|---:|---:|---:|
| poll | 22937.508 | 38.90% | 1474 |
| pthread_cond_timedwait | 11615.623 | 19.70% | 7564 |
| sem_clockwait | 9259.813 | 15.70% | 559 |
| pthread_cond_wait | 7562.448 | 12.80% | 220 |
| epoll_wait | 5961.110 | 10.10% | 380 |
| sem_wait | 327.467 | 0.60% | 10 |
| ioctl | 320.774 | 0.50% | 26478 |
| read | 294.020 | 0.50% | 3802 |
| nanosleep | 160.174 | 0.30% | 212 |
| usleep | 127.951 | 0.20% | 26909 |

## Δ from previous snapshot

Δ from long-horizon_round68

Δ step_time_ms=-10.603  Δ mfu_e2e_standard=+0.0688  Δ profiled_steps=+0  Δ gpu_kernel_per_step_ms=-0.139  Δ gpu_idle_per_step_ms=-10.464  Δ gpu_memop_per_step_ms=+6.329  Δ cuda_api_per_step_ms=+29.582  Δ os_runtime_per_step_ms=+445.876

| kernel | Δ per_step_ms | Δ pct |  state |
|---|---:|---:|---|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | +0.047 | +0.00% | tracked |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | +0.071 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | -0.270 | +0.00% | tracked |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | -0.238 | +0.00% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | +0.128 | +0.00% | tracked |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | +0.026 | +0.00% | tracked |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | +0.012 | +0.00% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | +0.053 | +0.00% | tracked |
| _rope_kernel | +0.010 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | -0.045 | +0.00% | tracked |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | -0.008 | +0.00% | tracked |
| _ce_bwd_kernel | -0.051 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | -0.011 | +0.00% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | -0.039 | +0.00% | tracked |
| _swiglu_bwd_kernel | +0.013 | +0.00% | tracked |
