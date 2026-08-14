# profile snapshot — profile-snapshot

step_time_ms=4492.025  mfu_e2e_standard=29.2716  profiled_steps=12  gpu_kernel_per_step_ms=2828.236  gpu_idle_per_step_ms=1663.789  gpu_memop_per_step_ms=33.678  cuda_api_per_step_ms=3149.074  os_runtime_per_step_ms=57141.085
# gpu_idle = step - gpu_kernel (valid only when compute kernels share one stream); cuda_api / os_runtime are CPU time, do NOT subtract from step_time.
world_size=2  micro_batch_size=4  seq_length=4096  grad_accum_steps=10
nsys_rep=/user/sunhaojun/.forge_train/77459cccb4da/.artifacts/runs/profile-snapshot-20260814-230145-552dfa/ours/profile.nsys-rep

## top 15 GPU kernels by total time

| kernel | per_step_ms | pct | instances |
|---|---:|---:|---:|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | 555.133 | 19.60% | 2017 |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | 276.677 | 9.80% | 4092 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | 260.039 | 9.20% | 6290 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 219.885 | 7.80% | 2214 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | 209.631 | 7.40% | 2288 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | 109.881 | 3.90% | 17394 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 91.430 | 3.20% | 4250 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | 87.181 | 3.10% | 6455 |
| _rope_kernel | 83.505 | 3.00% | 8184 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | 73.502 | 2.60% | 4182 |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | 73.411 | 2.60% | 164 |
| _ce_bwd_kernel | 62.416 | 2.20% | 656 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | 55.386 | 2.00% | 2042 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | 54.975 | 1.90% | 2125 |
| _swiglu_bwd_kernel | 43.857 | 1.60% | 2042 |

## GPU memory operations

| op | per_step_ms | pct | count |
|---|---:|---:|---:|
| [CUDA memcpy Host-to-Device] | 31.544 | 93.70% | 806 |
| [CUDA memset] | 1.832 | 5.40% | 25223 |
| [CUDA memcpy Device-to-Device] | 0.295 | 0.90% | 189 |
| [CUDA memcpy Device-to-Host] | 0.007 | 0.00% | 32 |

## top 10 CUDA API calls by total time

| api | per_step_ms | pct | calls |
|---|---:|---:|---:|
| cudaMemcpyAsync | 1558.972 | 49.50% | 1027 |
| cudaGraphLaunch_v10000 | 748.976 | 23.80% | 80 |
| cudaStreamSynchronize | 522.447 | 16.60% | 352 |
| cudaLaunchKernel | 87.057 | 2.80% | 6422 |
| cuMemSetAccess | 68.092 | 2.20% | 705 |
| cuMemCreate | 67.878 | 2.20% | 5996 |
| cudaFree | 37.270 | 1.20% | 2 |
| cudaDeviceSynchronize | 12.464 | 0.40% | 5 |
| cudaEventSynchronize | 12.454 | 0.40% | 8 |
| cuMemUnmap | 10.189 | 0.30% | 2143 |

## top 10 OS runtime calls by total time

| os call | per_step_ms | pct | calls |
|---|---:|---:|---:|
| poll | 21817.233 | 38.20% | 1359 |
| pthread_cond_timedwait | 11099.343 | 19.40% | 1956 |
| sem_clockwait | 9132.439 | 16.00% | 502 |
| pthread_cond_wait | 7556.819 | 13.20% | 207 |
| epoll_wait | 5671.751 | 9.90% | 218 |
| ioctl | 351.576 | 0.60% | 26472 |
| read | 347.477 | 0.60% | 3780 |
| sem_wait | 332.846 | 0.60% | 10 |
| usleep | 211.716 | 0.40% | 46319 |
| pthread_rwlock_wrlock | 165.832 | 0.30% | 33 |

## Δ from previous snapshot

Δ from long-horizon_round64

Δ step_time_ms=-163.774  Δ mfu_e2e_standard=+1.0288  Δ profiled_steps=+0  Δ gpu_kernel_per_step_ms=-45.726  Δ gpu_idle_per_step_ms=-118.048  Δ gpu_memop_per_step_ms=-14.050  Δ cuda_api_per_step_ms=-168.401  Δ os_runtime_per_step_ms=-3573.164

| kernel | Δ per_step_ms | Δ pct |  state |
|---|---:|---:|---|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | +7.421 | +0.50% | tracked |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | +3.448 | +0.30% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | +2.860 | +0.30% | tracked |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | +2.652 | +0.20% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | +2.149 | +0.20% | tracked |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | +1.074 | +0.10% | tracked |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | +0.947 | +0.10% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | +0.738 | +0.10% | tracked |
| _rope_kernel | +1.023 | +0.10% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | +0.948 | +0.10% | tracked |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | +0.900 | +0.10% | tracked |
| _ce_bwd_kernel | +0.761 | +0.10% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | +0.569 | +0.10% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | +0.597 | +0.00% | tracked |
| _swiglu_bwd_kernel | (new) | (new) | new |
| void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::Opaque... | -78.328 | -2.70% | dropped |
