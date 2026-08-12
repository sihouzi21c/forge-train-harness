# resume — Resume bitwise (save/load lossless round-trip)

> **Prereq**: bitwise-perf PASS. **Postreq**: resume PASS → proceed to long-horizon.

## Goal

Implement checkpoint save / load after bitwise-perf completes — train for a while, save the checkpoint, restore from the checkpoint and continue training; the `loss` / `grad_norm` after restoration must be **bitwise exactly the same (`max_abs_diff == 0`)** as the corresponding step of "uninterrupted training by the same engine in the same process" on `[10, 20)`. In other words, `save_checkpoint` + `load_resume_checkpoint` must be a lossless round-trip.

## Acceptance criteria (both required)

1. checkpoint save/load round-trip preserves **all numerical state**: model weights (including master FP32 weights), optimizer state (`m` / `v` / `step_count` / `num_samples`), dataloader position, RNG state.
2. The corresponding step's `loss` / `grad_norm` after restoration is `max_abs_diff == 0` vs uninterrupted training (no 1e-7-level tolerance is allowed).

## Gate

harness suite: `bin/harness run resume-gate-20` (script self-comparison — in the same process, first run reference uninterrupted for 20 steps, then run the resume path for 10 steps → save → load → 10 steps; dispatcher compares `[LOSS_REF]` vs `[LOSS_RES]` at every step in 10–19 for `max_abs_diff == 0`).

Gate formula (self-comparison):

- **Phase R (reference)**: from init checkpoint, train uninterrupted for 20 steps, emit `[LOSS_REF] step=N ...`
- **Phase A+B (resume)**: from the same init checkpoint, train 10 steps → `save_checkpoint` → `load_resume_checkpoint` → continue training to 20 steps, emit `[LOSS_RES] step=N ...`
- **Decision**: `max(|loss_ref[i] - loss_res[i]|) == 0` and `max(|grad_norm_ref[i] - grad_norm_res[i]|) == 0` for `i ∈ [10, 20)`.

## Forbidden escape hatches

1. Relaxing the gate standard from `max_abs_diff == 0` to any relative / absolute tolerance is forbidden; the implementation goal of `save_checkpoint`/`load_resume_checkpoint` is lossless round-trip.
2. **Forbidden** to change the reference-vs-resume comparison to "run reference alone, run resume alone, then align with some external baseline" — only same-process self-comparison can strictly prove that save/load does not lose precision.
3. Leaking the intermediate state of the reference phase (weight snapshot, optimizer state, RNG, etc.) to the resume phase to "game scores" is forbidden; the real save → restart training → load path must be strictly followed.

## Constraints

1. `micro_batch_size` must match `@@FORGE_CONFIG_DIR@@/eval.toml [evals.resume-gate-20].ref_env.MICRO_BATCH_SIZE_OVERRIDE` (same value as bitwise-multicard constraint #7).
2. All resume changes must keep all alignment–bitwise-perf bitwise regression passing.

## Second gate: `wsd-sft-70` (WSD-SFT 3-phase switch correctness)

The resume milestone ALSO requires `bin/harness run wsd-sft-70` PASS — the WSD-SFT
3-phase gate (stable→decay→sft is "resume + phase switching", so it lives here).
It runs the ours engine through the 3-phase bash driver
(`evals/scripts/train_ours_al.sh` → `train_ours_phase.py`) **twice** (a clean
baseline + the main run, both never crashing) and requires:

1. **Run-to-run bitwise determinism**: the two runs' per-phase per-step `[LOSS]`
   trajectories are bitwise equal (`gate_atol = 0`).
2. **Structural `[PHASE]` switch assertions** (parsed from the engine's
   `[PHASE] name=…` banner — see the stdout grammar on `TrainLoopConfig`):
   - phase order is `stable → decay → sft`;
   - `decay` is a FULL resume: `start_step == stable_steps` (counter carried),
     `init_weights_only == 0`, corpus swapped from stable;
   - `sft` is WEIGHTS-ONLY: `start_step == 0` (counter reset),
     `init_weights_only == 1` (fresh optimizer + warmup), lr drops to the new
     peak, corpus swapped from decay.

Engine work this needs (contract on `TrainLoopConfig` — see train_loop.py):
per-phase WSD schedule fields (`lr / min_lr / lr_warmup_iters / lr_decay_iters /
lr_wsd_decay_iters`, evaluated at the ABSOLUTE step), `init_weights_only`,
`save_interval` (periodic versioned ckpts to `<save_path>/step_<abs>/`),
`phase_name` + the `[PHASE]` banner. The same machinery is later exercised
under crash injection by the production milestone's `production-resume-70`.

## Remote job scripts (resume milestone deliverable, not a gate)

The resume milestone also ships two **cctl PyTorchJob** training scripts — a short
(default 20-step) real remote-GPU job per side, submitted with the tracked
launcher `evals/scripts/launch_resume.sh` (runs on the mac, where `cctl` lives,
after `bin/harness sync push`; preview with `DRY_RUN=1`):

- `launch_resume.sh ref` → per-pod entry `evals/scripts/resume_train_ref.sh` —
  the L0 reference stack trained for `[train].iters` steps.
- `launch_resume.sh ours` → per-pod entry `evals/scripts/resume_train_ours.sh` —
  the ours engine trained for the same recipe with periodic versioned
  `step_<abs>/` checkpoints; a re-submit with the same `SAVE_ROOT` resumes from
  the latest checkpoint instead of step 0 (`CHECKPOINT_ROOT` required).

Step count, card count, shape and LR live in `config/train/resume_20.toml`
(`[train].iters` / `[launch].gpus_per_node`; env overrides win). Geometry /
optimizer constants come from the rendered `resume-gate-20` products — render
and sync push before submitting. Neither job carries a verdict: verify by
`returncode 0`, the `[LOSS]` lines in `cctl logs`, and (ours) the `step_<abs>/`
dirs on the shared FS.

> **State review (as of resume completion)**:
> - checkpoint save / load is a lossless round-trip.
> - `resume-gate-20` continues to PASS (`max_abs_diff == 0`).
> - `wsd-sft-70` PASSes (3-phase switch correctness, bitwise self-comp).
> - All alignment–bitwise-perf bitwise regression still passes.
