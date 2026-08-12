# bitwise-singlecard — Single-GPU multi-step training full alignment (real dataloader, bitwise)

> **Prereq**: alignment.forward + alignment.backward PASS. **Postreq**: bitwise-singlecard PASS → proceed to bitwise-multicard.

## Goal

On top of alignment's forward+backward bitwise, add `optimizer.step` / LR scheduler, **directly integrate the real external dataloader**, and complete an 8-step ref-vs-ours bitwise alignment in the `WORLD_SIZE=1` single-GPU form. bitwise-singlecard simultaneously validates two things:

1. The multi-step optimizer correctness of the single-GPU pure computation path (isolating NCCL communication noise).
2. The correct integration of the real dataloader path: both ref and ours sides are pointed at the **same dataset** by `@@FORGE_CONFIG_DIR@@/data.toml` (`[data].conf_path`, whose data-conf script exports `DATA_PATH` + `DATA_LOADER`), so the same per-step batch is produced on both sides.

## Alignment basis

The L0 ref script's `HARNESS_GATE=multistep-1gpu` preset. The full run shape (WORLD_SIZE / MICRO_BATCH_SIZE_OVERRIDE / GLOBAL_BATCH_SIZE_OVERRIDE / GRAD_ACCUM_STEPS / NUM_STEPS_OVERRIDE / GATE_WINDOW_*) is the SSOT under `@@FORGE_CONFIG_DIR@@/eval.toml [evals.multistep-1gpu].ref_env`; read it for the concrete values.

## Execution form

dispatcher (`evals.dispatcher._run_bitwise_trajectory`) on one side `shell-exec`s the ref script to obtain the ref single-GPU trajectory, and on the other side launches `evals/scripts/eval_train_steps.py` → `run_training_loop` via `launch_dp.py NUM_PROCS=1` to obtain the ours single-GPU trajectory. Both sides load the same data, selected by `@@FORGE_CONFIG_DIR@@/data.toml` → `DATA_PATH` + `DATA_LOADER`.

## Acceptance criteria

On the gate window declared in `[evals.multistep-1gpu].ref_env.GATE_WINDOW_*`, each step's `loss` / `grad_norm` is **bitwise exactly the same (`max_abs_diff == 0`)**; if multi-step drift occurs, the specific step and event must be located.

## Gate

**`loss` bitwise match (`max_abs_diff == 0`), `grad_norm` bitwise match (`max_abs_diff == 0`), no fallback tolerance.**

harness suite: `bin/harness run multistep-1gpu`

## Constraints

1. Read the ref script's attention implementation and use the **same attention backend the ref uses** (same call, same backend toggles); do not assume a specific stack or maintain a backend-specific divergence — mirror whatever the ref does.
2. "mask / dropout makes it impossible to fully align" can no longer be used as an exemption reason for incompleteness.
3. Relying on large-scale replay of extra intermediate tensors to fake the real training entry point is not allowed.
4. Exactly the same computation path and precision strategy as the ref must be used to ensure bitwise match (see `constraint.md` §FP32 precision specification + §Precision alignment requirement).
5. **The external dataloader selected by `@@FORGE_CONFIG_DIR@@/data.toml` (`[data].conf_path` → `DATA_LOADER`) is an external dependency, not within the in-house scope**, but its call entry point must live inside the workload's `dataloader.py` (driven by `train_loop`); the harness layer / gate script must not construct an iterator by itself.
6. `micro_batch_size` must match `@@FORGE_CONFIG_DIR@@/eval.toml [evals.multistep-1gpu].ref_env.MICRO_BATCH_SIZE_OVERRIDE`.
7. **Engine reads optimizer / model hyperparameters directly from `@@FORGE_CONFIG_DIR@@/optim.toml` and `@@FORGE_CONFIG_DIR@@/model.toml`.** The `multistep-1gpu` suite's `env_inputs` only injects shape knobs (`MICRO_BATCH_SIZE_OVERRIDE`, `GLOBAL_BATCH_SIZE_OVERRIDE`, `SEQ_LEN_OVERRIDE`, etc.) — it does **not** inject `FORGE_LR` / `FORGE_WD` / `FORGE_BETAS` / `FORGE_EPS` / `FORGE_*` for optimizer hp. Do not assume environment-variable-driven optimizer config; load the toml inside `train_loop`.
