# profile snapshot — profile-snapshot

step_time_ms=7200.044  mfu_e2e_standard=18.2613  profiled_steps=12  gpu_kernel_per_step_ms=0.000  gpu_idle_per_step_ms=7200.044  gpu_memop_per_step_ms=0.000  cuda_api_per_step_ms=0.039  os_runtime_per_step_ms=87296.168
# gpu_idle = step - gpu_kernel (valid only when compute kernels share one stream); cuda_api / os_runtime are CPU time, do NOT subtract from step_time.
world_size=2  micro_batch_size=4  seq_length=4096  grad_accum_steps=10
nsys_rep=/user/sunhaojun/.forge_train/77459cccb4da/.artifacts/runs/profile-snapshot-20260814-055257-cf91eb/ours/profile.nsys-rep

## top 15 GPU kernels by total time

| kernel | per_step_ms | pct | instances |
|---|---:|---:|---:|

## GPU memory operations

| op | per_step_ms | pct | count |
|---|---:|---:|---:|

## top 10 CUDA API calls by total time

| api | per_step_ms | pct | calls |
|---|---:|---:|---:|
| cudaGetDeviceProperties_v2_v12000 | 0.031 | 79.40% | 2 |
| cuGetProcAddress_v2 | 0.007 | 18.30% | 419 |
| cudaGetDriverEntryPoint_v11030 | 0.001 | 1.50% | 32 |
| cuModuleGetLoadingMode | 0.000 | 0.40% | 2 |
| cuInit | 0.000 | 0.40% | 1 |

## top 10 OS runtime calls by total time

| os call | per_step_ms | pct | calls |
|---|---:|---:|---:|
| poll | 33930.093 | 38.90% | 2064 |
| pthread_cond_timedwait | 17072.681 | 19.60% | 1724 |
| sem_clockwait | 14365.758 | 16.50% | 833 |
| pthread_cond_wait | 10913.237 | 12.50% | 228 |
| epoll_wait | 8802.874 | 10.10% | 526 |
| ioctl | 530.855 | 0.60% | 27187 |
| sem_wait | 338.405 | 0.40% | 10 |
| read | 278.621 | 0.30% | 3713 |
| pthread_rwlock_wrlock | 223.692 | 0.30% | 30 |
| nanosleep | 221.527 | 0.30% | 285 |

## Δ from previous snapshot

Δ from long-horizon_round26

Δ step_time_ms=+188.640  Δ mfu_e2e_standard=-0.4910  Δ profiled_steps=+0  Δ gpu_kernel_per_step_ms=+0.000  Δ gpu_idle_per_step_ms=+188.640  Δ gpu_memop_per_step_ms=+0.000  Δ cuda_api_per_step_ms=-2.030  Δ os_runtime_per_step_ms=+16475.357

| kernel | Δ per_step_ms | Δ pct |  state |
|---|---:|---:|---|
