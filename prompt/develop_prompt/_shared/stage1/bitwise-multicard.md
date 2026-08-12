# bitwise-multicard — Multi-GPU multi-step training full alignment (real dataloader, bitwise)

> **Prereq**: bitwise-singlecard PASS. **Postreq**: bitwise-multicard PASS → proceed to bitwise-perf.

## Goal

Directly extend bitwise-singlecard's single-GPU bitwise to the multi-GPU DP defined in `@@FORGE_CONFIG_DIR@@/eval.toml [evals.multistep].ref_env.WORLD_SIZE` (consistent with the bitwise-perf–long-horizon topology, avoiding any intermediate DP step). Apart from the parallel topology, completely isomorphic to bitwise-singlecard: same `evals/scripts/eval_train_steps.py`, same dispatcher driver, same step count, same real external dataloader.

## Alignment basis

The L0 ref script's `HARNESS_GATE=multistep` preset. The full run shape (WORLD_SIZE / MICRO_BATCH_SIZE_OVERRIDE / GLOBAL_BATCH_SIZE_OVERRIDE / GRAD_ACCUM_STEPS / NUM_STEPS_OVERRIDE / GATE_WINDOW_*) is the SSOT under `@@FORGE_CONFIG_DIR@@/eval.toml [evals.multistep].ref_env` (`grad_accum` is derived as `GBS / (MBS × WORLD_SIZE)`, not a ref_env key).

## Execution form

dispatcher on one side `shell-exec`s the ref script (torchrun with `NPROC_PER_NODE = WORLD_SIZE`), and on the other side launches the same number of ranks on the ours side via `launch_dp.py NUM_PROCS=WORLD_SIZE` (WORLD_SIZE from `@@FORGE_CONFIG_DIR@@/eval.toml [evals.multistep].ref_env`).

## Acceptance criteria

With the same input and same seed, each step's `loss` / `grad_norm` is **bitwise exactly the same as the ref baseline (`max_abs_diff == 0`)**; if multi-card multi-step drift occurs, the specific step and event must be located (first single-GPU bitwise-singlecard PASS then multi-GPU; multi-card drift can be directly attributed to the communication layer).

## Gate

**`loss` bitwise match (`max_abs_diff == 0`), `grad_norm` bitwise match (`max_abs_diff == 0`), no fallback tolerance.**

harness suite: `bin/harness run multistep`

## Constraints

1. Read the ref script's attention implementation and use the **same attention backend the ref uses** (same call, same backend toggles); do not assume a specific stack or maintain a backend-specific divergence — mirror whatever the ref does.
2. "mask / dropout makes it impossible to fully align" can no longer be used as an exemption reason for incompleteness.
3. Relying on large-scale replay of extra intermediate tensors to fake the real training entry point is not allowed.
4. Exactly the same computation path and precision strategy as the ref must be used to ensure bitwise match.
5. **Injecting baseline runtime state in the bitwise gate is forbidden** — see `constraint.md` §Forbidden injection of baseline runtime state.
6. The precision alignment process may ignore MFU.
7. **`micro_batch_size` must match `@@FORGE_CONFIG_DIR@@/eval.toml [evals.multistep].ref_env.MICRO_BATCH_SIZE_OVERRIDE`**.
8. **`global_batch_size` is pinned by the harness (GBS = 40) and is NOT yours
   to change — do not modify it anywhere.** The shape SSOT lives in the
   launch-frozen config dir (read-only) and the dispatcher reads
   GBS / WORLD_SIZE / step count / gate window ONLY from the frozen ref-side
   product (`ref/config/multistep.toml`, `chmod a-w`). Editing the shape keys
   of your writable ours product (`workload/src/config/multistep.toml`) cannot
   change the gate's shape — it only makes your run diverge from the ref
   trajectory and fail bitwise, and shape edits there are treated as gate
   tampering by review. Do not "tune" GBS/grad_accum for speed or memory; the
   shipped shape and timeout are calibrated together.

## Capture wall-clock (read BEFORE your first gate run — do not rediscover this)

The `multistep` gate runs at `hash_capture_level=2`: every module's fwd/bwd
tensors are blake2b-hashed per microbatch (~tens of GB per step at
grad_accum=10). Facts that previously cost each loop 2–5 hours to rediscover:

1. **The ref runs LIVE on every gate attempt** (the ref-cache is disabled at
   `hash_capture_level > 0` by policy — do not add caching yourself). Every
   attempt therefore costs ref (~520s on 0.5B / ~760–830s on 1B) + ours,
   sequentially, inside one `timeout_s + 60` window. The shipped budget is
   calibrated for exactly that (ref + optimized ours + overhead, ×1.5
   margin) — it fits; do not re-derive a budget audit, and make each
   attempt count because retries are not discounted.
2. **Do not hand-roll async hash capture.** The harness ships a
   candidate-legal utility — `from evals.capture_offload import
   OffloadHasher, hash_tensor, hash_batch_sync, resolve_futures` — whose
   digests are byte-identical to the ref producer. Using it is NOT an
   anti-proxy violation (the guard bans `evals.harness_hook`, not this
   module). Naive inline `.cpu()` + single-core blake2b serialises the
   training thread (~5–10× the budget).
3. If you still build your own capture, three hard constraints, each learned
   from a real crash: **(a)** worker threads must never touch CUDA
   (`CUDA_DEVICE_MAX_CONNECTIONS=1` → DP-collective deadlock); **(b)** never
   block on capture backpressure before a collective (rank livelock);
   **(c)** use pageable — not pinned — host snapshots and drain per
   microbatch: the pinned caching allocator hoards pages and the container
   cgroup cap (`/sys/fs/cgroup/memory.max`, shared by all ranks, often far
   below physical RAM) delivers a bare SIGKILL. Grad buffers mutated in
   place between pre/post-allreduce sweeps must be hashed with a
   synchronous barrier, never async.
