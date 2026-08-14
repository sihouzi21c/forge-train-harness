# profile snapshot — profile-snapshot

step_time_ms=4496.324  mfu_e2e_standard=29.2431  profiled_steps=12  gpu_kernel_per_step_ms=2833.612  gpu_idle_per_step_ms=1662.712  gpu_memop_per_step_ms=45.157  cuda_api_per_step_ms=3235.358  os_runtime_per_step_ms=58475.519
# gpu_idle = step - gpu_kernel (valid only when compute kernels share one stream); cuda_api / os_runtime are CPU time, do NOT subtract from step_time.
world_size=2  micro_batch_size=4  seq_length=4096  grad_accum_steps=10
nsys_rep=/user/sunhaojun/.forge_train/77459cccb4da/.artifacts/runs/profile-snapshot-20260814-231006-f896de/ours/profile.nsys-rep

## top 15 GPU kernels by total time

| kernel | per_step_ms | pct | instances |
|---|---:|---:|---:|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | 556.181 | 19.60% | 2019 |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | 277.244 | 9.80% | 4094 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | 261.232 | 9.20% | 6295 |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | 220.509 | 7.80% | 2214 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | 210.187 | 7.40% | 2290 |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | 110.054 | 3.90% | 17407 |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | 91.546 | 3.20% | 4252 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | 87.224 | 3.10% | 6459 |
| _rope_kernel | 83.680 | 3.00% | 8186 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | 73.622 | 2.60% | 4182 |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | 73.421 | 2.60% | 164 |
| _ce_bwd_kernel | 62.572 | 2.20% | 656 |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | 55.546 | 2.00% | 2044 |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | 55.083 | 1.90% | 2126 |
| _swiglu_bwd_kernel | 43.904 | 1.50% | 2044 |

## GPU memory operations

| op | per_step_ms | pct | count |
|---|---:|---:|---:|
| [CUDA memcpy Host-to-Device] | 43.046 | 95.30% | 806 |
| [CUDA memset] | 1.810 | 4.00% | 25234 |
| [CUDA memcpy Device-to-Device] | 0.294 | 0.70% | 189 |
| [CUDA memcpy Device-to-Host] | 0.007 | 0.00% | 32 |

## top 10 CUDA API calls by total time

| api | per_step_ms | pct | calls |
|---|---:|---:|---:|
| cudaMemcpyAsync | 1561.877 | 48.30% | 1027 |
| cudaGraphLaunch_v10000 | 743.915 | 23.00% | 80 |
| cudaStreamSynchronize | 571.449 | 17.70% | 352 |
| cuMemSetAccess | 124.728 | 3.90% | 705 |
| cuMemCreate | 82.790 | 2.60% | 5996 |
| cudaLaunchKernel | 76.733 | 2.40% | 6422 |
| cudaFree | 15.682 | 0.50% | 2 |
| cudaDeviceSynchronize | 12.492 | 0.40% | 5 |
| cudaEventSynchronize | 12.344 | 0.40% | 8 |
| cuMemUnmap | 11.733 | 0.40% | 2143 |

## top 10 OS runtime calls by total time

| os call | per_step_ms | pct | calls |
|---|---:|---:|---:|
| poll | 22661.605 | 38.80% | 1304 |
| pthread_cond_timedwait | 11452.472 | 19.60% | 9216 |
| sem_clockwait | 9262.549 | 15.80% | 558 |
| pthread_cond_wait | 7687.791 | 13.10% | 224 |
| epoll_wait | 5769.793 | 9.90% | 105 |
| ioctl | 340.103 | 0.60% | 26473 |
| sem_wait | 333.380 | 0.60% | 10 |
| read | 317.913 | 0.50% | 3815 |
| pthread_rwlock_wrlock | 176.832 | 0.30% | 33 |
| usleep | 120.516 | 0.20% | 26002 |

## Δ from previous snapshot

Δ from long-horizon_round67

Δ step_time_ms=+4.299  Δ mfu_e2e_standard=-0.0285  Δ profiled_steps=+0  Δ gpu_kernel_per_step_ms=+5.376  Δ gpu_idle_per_step_ms=-1.077  Δ gpu_memop_per_step_ms=+11.479  Δ cuda_api_per_step_ms=+86.284  Δ os_runtime_per_step_ms=+1334.434

| kernel | Δ per_step_ms | Δ pct |  state |
|---|---:|---:|---|
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(in... | +1.048 | +0.00% | tracked |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (i... | +0.567 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT | +1.193 | +0.00% | tracked |
| nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN | +0.624 | +0.00% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | +0.556 | +0.00% | tracked |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_c... | +0.173 | +0.00% | tracked |
| void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_ke... | +0.116 | +0.00% | tracked |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunc... | +0.043 | +0.00% | tracked |
| _rope_kernel | +0.175 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT | +0.120 | +0.00% | tracked |
| void at::native::<unnamed>::cunn_SoftMaxForward<(int)8, c10::BFloat16, float,... | +0.010 | +0.00% | tracked |
| _ce_bwd_kernel | +0.156 | +0.00% | tracked |
| nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT | +0.160 | +0.00% | tracked |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN | +0.108 | +0.00% | tracked |
| _swiglu_bwd_kernel | +0.047 | -0.10% | tracked |
