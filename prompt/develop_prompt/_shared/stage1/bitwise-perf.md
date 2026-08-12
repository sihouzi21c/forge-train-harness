# bitwise-perf — Bitwise performance optimization (MFU ramp-up milestone)

> **Prereq**: bitwise-multicard PASS. **Postreq**: bitwise-perf PASS → proceed to resume.

## Goal

Without breaking bitwise-multicard bitwise alignment, raise end-to-end `MFU(standard)` above the `mfu_e2e_target` declared in `@@FORGE_CONFIG_DIR@@/eval.toml [evals.perf-bitwise]`. Place all engineering optimizations that **do not change the operator path and have identical numerical results** in this milestone, to avoid mixing these bugs into long-horizon operator fusion / system-level long-running optimization, which would pollute the judgment of fusion benefit.

## Alignment basis

The L0 ref script's `HARNESS_GATE=perf-bitwise` preset. The full run shape (WORLD_SIZE / MICRO_BATCH_SIZE_OVERRIDE / GLOBAL_BATCH_SIZE_OVERRIDE / NUM_STEPS_OVERRIDE / GATE_WINDOW_*) is the SSOT under `@@FORGE_CONFIG_DIR@@/eval.toml [evals.perf-bitwise].ref_env` (`grad_accum` is derived as `GBS / (MBS × WORLD_SIZE)`, not a ref_env key); the warmup steps before the gate window (see `warmup_steps` and `GATE_WINDOW_START` in that same table) cover the warmup of dataloader / CUDA graph / fused kernels etc., not counted in MFU averaging.

## Execution form

Same `_run_bitwise_trajectory` driver as bitwise-multicard, only extended relative to bitwise-multicard (see `[evals.perf-bitwise].ref_env` for `NUM_STEPS_OVERRIDE` / `GATE_WINDOW_*`, and `mfu_e2e_target` raised from 0 to its declared value).

## Allowed optimization techniques (**must keep `loss` / `grad_norm` bitwise consistent with the ref baseline at the same time**)

1. Replace "forward replay + autograd" with native backward kernels (e.g., fused attention backward, RoPE backward, etc.). On the megatron backend, this can additionally leverage TransformerEngine kernels (TE fused attention backward, TE RoPE backward, etc.); without TE the bitwise-aligned backward replacement set is much smaller.
2. Resident FP32 `main_grad` buffer, accumulated in-place directly in wgrad / embedding backward, replacing per-microbatch `torch.zeros` + Python dict accumulation.
3. Remove unnecessary `torch.cuda.synchronize()` / blocking CPU-GPU calls from the hot path.
4. Buffer pool reuse, shape-stable static tensor allocation, remove redundant `.contiguous()` / temporary `.clone()`.
5. Adjust scheduling order so that communication / computation do not wait for each other (provided numerical results remain exactly the same; non-deterministic reduce is not allowed).

## Disallowed optimization techniques (all reserved for long-horizon)

1. Any operator fusion that changes the mathematical computation path (even if equivalent under FP64; as long as bf16 rounding order changes, it counts as long-horizon).
2. Any optimization that breaks `loss` / `grad_norm` bitwise consistency with the ref baseline.
3. Changing precision (e.g., changing a certain FP32 segment to bf16) — violates the FP32 precision specification (see `constraint.md`); absolutely forbidden in the bitwise-perf milestone; also not allowed in the long-horizon milestone (long-horizon only allows the "internal computation path" of fused operators to change; the input/output dtype level must remain the same as the baseline operator being replaced, see `long-horizon.md` §long-horizon operator fusion precision path constraints).

## Required output

Every round of performance optimization must record in `workload/notes/perf_log.md`: the change point, the expected benefit, the bitwise regression test result (`max_abs_diff = 0`), and the measured MFU change.

## Gate (both must pass)

1. `loss` / `grad_norm` bitwise exactly the same as the ref baseline (`max_abs_diff == 0`, reusing bitwise-multicard's alignment logic), no fallback tolerance.
2. `avg MFU(standard)` ≥ the `mfu_e2e_target` declared in `[evals.perf-bitwise]` (counted from `warmup_steps` onward, using the same standard MFU formula as long-horizon, see `constraint.md` §MFU convention).

harness suite: `bin/harness run perf-bitwise`

## Methodology (agent self-stop condition)

1. **At most 3 rounds** of performance optimization iteration within this milestone. In each round, do as many bitwise-constrained optimizations as you believe can be done — **whether or not the gate has already passed, complete all the bitwise-preserving optimizations you can think of before stopping**. Each round must first run `bin/harness run perf-bitwise`; if bitwise verification fails, prioritize repairing bitwise before continuing performance optimization — "first abandon bitwise then come back to fix it" is forbidden.
2. If after 3 rounds you still cannot reach the target, do not keep grinding inside bitwise-perf: clearly record in `perf_log.md` the reason for hitting the ceiling (CPU-bound, communication bottleneck, kernel's own efficiency, etc.), then enter resume.
3. Before starting bitwise-perf, you must run `bin/harness run profile-snapshot M4_round0` at least once to capture an nsys trace and identify the top-N hotspots; purely guessing the bottleneck from step-time logs is not allowed.

## Profiling cadence (hard rule)

Every round whose commit touches the perf hot path
(`workload/src/training_engine_tensor/{forward,backward,kernels,triton_kernels,optimizer,nccl,train_loop,dataloader,parameters,config}.py`)
MUST produce a fresh profile snapshot **in the same commit**, after the gate has PASSed:

    bin/harness run profile-snapshot M4_round<N>

The round's `perf_log.md` entry must (a) cite
`workload/notes/profile/M4_round<N>/summary.md` by path and (b) quote at
least one `Δ from M4_round<N-1>` line that justifies the change. Bitwise-
repair-only / docs-only / refactor-only rounds (no diff in the paths listed
above) are exempt.

Reading order each round: summary.md top section (step_time + MFU +
profiled_steps + Δ) → top-15 GPU kernels (which CUDA kernel moved) →
memory-op + CUDA-API + OS-runtime blocks (whether the move was a real op
or a launch-overhead artifact, and whether host stalls are comm or
GPU-wait). All time columns are **per-step** (window total divided by
`profiled_steps`); the `instances` / `calls` counts stay
window-cumulative. Hints at the bottom are mechanical suggestions, not
decisions.

`profile-snapshot` is the only suite that wraps rank 0 with `nsys profile`.
The verdict gates (`perf-bitwise`, `long-train`, `long-train-smoke`,
`resume-gate-20`, `loss-gate-200`) never carry profiler overhead — the
verdict gates' MFU / loss_rel decision stays clean.

## Constraints

1. `micro_batch_size` must match `@@FORGE_CONFIG_DIR@@/eval.toml [evals.perf-bitwise].ref_env.MICRO_BATCH_SIZE_OVERRIDE` (same value as bitwise-multicard constraint #7).
2. bitwise-multicard constraint #5 (no injecting baseline runtime state, see `constraint.md`) also applies.
3. Any bitwise-perf change must pass all alignment–bitwise-multicard bitwise regression tests before entering resume.
4. `DP` (= `@@FORGE_CONFIG_DIR@@/eval.toml [evals.perf-bitwise].ref_env.WORLD_SIZE`) + real dataloader; world_size and data config cannot be reduced.
5. **The MFU is measured WITH the level-1 loss+grad capture running inline.**
   Naive blocking capture alone can eat 40–65% of the step and sink you below
   the floor while your compute is fine (measure the level-0 baseline first to
   see the gap). Before optimizing kernels, take the capture off the training
   thread: `from evals.capture_offload import OffloadHasher, hash_batch_sync`
   is candidate-legal (see bitwise-multicard § Capture wall-clock for the
   three crash constraints if you roll your own).

> **State review (as of bitwise-perf completion)**:
> - The in-house framework is bitwise aligned with the ref baseline on the multi-GPU + real dataloader path (DP from `@@FORGE_CONFIG_DIR@@/eval.toml [evals.perf-bitwise].ref_env.WORLD_SIZE`).
> - `avg MFU(standard)` ≥ the `mfu_e2e_target`.
> - Before entering resume, ensure all alignment–bitwise-perf harness suites still PASS (any bitwise-perf change must not break alignment–bitwise-multicard bitwise).
