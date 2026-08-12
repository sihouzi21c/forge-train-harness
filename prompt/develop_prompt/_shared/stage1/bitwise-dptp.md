# bitwise-dptp — Multi-GPU DP×TP multi-step training full alignment (real dataloader, bitwise)

> **Prereq**: bitwise-singlecard PASS. **Postreq**: bitwise-dptp PASS → proceed to bitwise-perf.

## Goal

Directly extend bitwise-singlecard's single-GPU bitwise to the multi-GPU **DP×TP** (data-parallel × tensor-parallel) topology defined in `@@FORGE_CONFIG_DIR@@/eval.toml [evals.dptp].ref_env` (consistent with the bitwise-perf–long-horizon topology, avoiding any intermediate DP-only step). Apart from the parallel topology this is isomorphic to bitwise-singlecard: same `evals/scripts/eval_train_steps.py`, same dispatcher driver, same step count, same real external dataloader. The difference from `bitwise-multicard` (DP-only, used by the smaller models) is that the model is sharded across a **tensor-parallel group** as well as replicated across a data-parallel group — so the in-house engine must build its own TP group and the comparator must reassemble the TP shards before diffing.

## Alignment basis

The L0 ref script's `HARNESS_GATE=dptp` preset. The full run shape (`WORLD_SIZE` / `TENSOR_PARALLEL_SIZE` / `MICRO_BATCH_SIZE_OVERRIDE` / `GLOBAL_BATCH_SIZE_OVERRIDE` / `GRAD_ACCUM_STEPS` / `NUM_STEPS_OVERRIDE` / `GATE_WINDOW_*` / `NUM_LAYERS_OVERRIDE`) is the SSOT under `@@FORGE_CONFIG_DIR@@/eval.toml [evals.dptp].ref_env`. The data-parallel size is **derived**: `DP = WORLD_SIZE / TENSOR_PARALLEL_SIZE`, and `GBS = MBS × DP` (NOT `MBS × WORLD_SIZE` — the TP ranks share one data shard). Megatron rank layout is TP-inner: `global_rank = dp_rank * TP_SIZE + tp_rank`.

## Execution form

The dispatcher on one side `shell-exec`s the ref script (torchrun with `NPROC_PER_NODE = WORLD_SIZE`, the ref launcher consuming `TENSOR_PARALLEL_SIZE` as its `TP_SIZE`), and on the other side launches the same number of ranks on the ours side via `launch_dp.py NUM_PROCS=WORLD_SIZE`. The in-house engine reads `tensor_parallel_size` from its rendered product and **builds its own TP group** (the DP group is the complement); `launch_dp` flattens all `WORLD_SIZE` ranks. Before comparison, `harness_dptp.finalize` **all-gathers the per-TP-rank shards into a single dump** so the comparator reads one reassembled tensor set per logical step.

> Proxy note: `[evals.dptp].ref_env.NUM_LAYERS_OVERRIDE` compresses the official layer count to a small proxy (e.g. 4 layers) for a fast capture — ref and ours both read `FORGE_NUM_LAYERS`, so the bitwise comparison is unaffected by the reduced depth. This milestone validates the DP×TP communication + sharding path, not full-scale throughput (that is `bitwise-perf` / `long-horizon`).

## Acceptance criteria

With the same input and same seed, each step's `loss` / `grad_norm` is **bitwise exactly the same as the ref baseline (`max_abs_diff == 0`)** after the TP shards are reassembled; if DP×TP multi-step drift occurs, the specific step and event must be located. Order of attribution: single-GPU bitwise-singlecard PASS first, then DP×TP — drift here is directly attributable to the **tensor-parallel split / all-reduce / TP-shard reassembly** path (column/row-parallel GEMM split, TP all-reduce placement, or the all-gather merge), not to single-card compute.

## Gate

**`loss` bitwise match (`max_abs_diff == 0`), `grad_norm` bitwise match (`max_abs_diff == 0`), no fallback tolerance.**

harness suite: `bin/harness run dptp`

## Constraints

1. Read the ref script's attention implementation and use the **same attention backend the ref uses** (same call, same backend toggles); do not assume a specific stack or maintain a backend-specific divergence — mirror whatever the ref does.
2. "mask / dropout makes it impossible to fully align" can no longer be used as an exemption reason for incompleteness.
3. Relying on large-scale replay of extra intermediate tensors to fake the real training entry point is not allowed.
4. Exactly the same computation path and precision strategy as the ref must be used to ensure bitwise match — including the **tensor-parallel GEMM split** (column/row-parallel layout) and the **TP all-reduce reduction order**, which must match the ref's TP operator calling mode (a different split or reduction order produces 1-ULP diffs).
5. **Injecting baseline runtime state in the bitwise gate is forbidden** — see `constraint.md` §Forbidden injection of baseline runtime state.
6. The precision alignment process may ignore MFU.
7. **`micro_batch_size` must match `@@FORGE_CONFIG_DIR@@/eval.toml [evals.dptp].ref_env.MICRO_BATCH_SIZE_OVERRIDE`**, and the in-house TP group size must match `[evals.dptp].ref_env.TENSOR_PARALLEL_SIZE`.
