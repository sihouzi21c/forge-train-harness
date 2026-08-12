# production — Ours-only WSD-SFT 3-phase production long-train on a cctl PyTorchJob

> **Prereq**: long-horizon complete (review-side throughput check passed — the loop advances this milestone itself; engine optimized, `long-train` + `resume-gate-20` green) AND `wsd-sft-70` PASS (3-phase stable→decay→sft switch correctness, resume milestone). **This milestone has two steps**: (1) `production-resume-70` crash-resume self-comp gate PASS on the devspace, then (2) `production-train` — the real cluster PyTorchJob. **Postreq**: `production-train` PASS → stage1 complete (production is the terminal stage1 milestone; it co-emits `STAGE_STATUS: finished`).

## Goal

With the engine aligned (alignment–resume bitwise) and optimized (long-horizon MFU), run the **real production WSD-SFT long-train of the ours engine only** — the full `stable → decay → sft` recipe from the frozen phase table `config/train/wsdsft_05b_prod.toml` — as a **cctl PyTorchJob**, and persist resumable checkpoints periodically so the run survives a crash/preemption and resumes where it stopped. production is the payoff milestone: it produces the trained SFT checkpoint, not another alignment gate.

**Card cap**: this line's cluster policy caps each person at **2–4 concurrent GPUs**. The job is therefore a **single node × 4 GPUs** (2 if the pool is tight — the card count is `[launch].gpus_per_node` in the recipe toml, which both the launcher and the pod read; `GPU_PER_NODE` env still overrides) and the phase table is the DOWNSCALED recipe (`stable 1800 / decay 360 / sft 180` — same phase ratios, same warmup ratios, same per-phase LR corners as the full-scale reference recipe recorded in the toml's header; GBS/MBS/seq are NOT downscaled — `grad_accum` absorbs the small DP world). Do not request more nodes; the launcher refuses `WORKERS > 0` without an explicit quota acknowledgement.

This is NOT a local dispatcher run. You **submit a distributed job to the cluster** and shepherd it to completion. The per-pod command is `evals/scripts/production_train.sh`, a thin wrapper over the proven `evals/scripts/train_ours_al.sh` 3-phase driver: it sets the per-phase periodic-save cadence, the engine config pointers, and the persistent save root, then runs the same stable→decay→sft pipeline with directory-level checkpoint handoff and built-in crash-restart resume.

## What production is NOT

- **No ref subprocess.** The production job never launches the L0 ref script. bitwise-singlecard–long-horizon and `wsd-sft-70` already cover alignment; the production job only proves the ours engine trains long, checkpoints, and resumes. (The pre-flight `production-resume-70` gate is also ref-free — ours-vs-ours self-comparison.)
- **No loss / MFU threshold gate on the cluster job.** The job is checkpoint-only (the engine still emits `[LOSS]`, but no verdict parses it). The pre-flight self-comp gate *does* have a bitwise verdict.
- **The cluster job is not a `bin/harness run`.** It is a real job you submit with the tracked launcher, not a local suite run. **But the pre-flight `production-resume-70` gate IS a `bin/harness run`** (step 1 below) — you must pass it before submitting the cluster job.

## Step 1 — pre-flight: `production-resume-70` crash-resume self-comp gate (required)

Before spending the cluster job, prove crash-resume is lossless **on the fused production engine** with a fast devspace gate: `bin/harness run production-resume-70` (2 cards, 20/20/20 steps, minutes). It runs the ours 3-phase line **twice** — once never crashing (baseline), once SIGKILLed mid-`stable`/`decay`/`sft` (crash_steps) and resumed from the latest `step_<abs>/` checkpoint — and requires the two `[LOSS]` trajectories to be **bitwise equal per phase per step** (`gate_atol = 0`), plus the `[PHASE]` structural switch assertions.

Why this shape (three lines of defense, one gate each):
- **① transition structure is correct** is proved at the resume milestone by `wsd-sft-70` (two independent never-crash ours runs must match bitwise + the `[PHASE]` assertions).
- **② transitions actually happen** is proved by the `[PHASE]` structural assertions (corpus swaps `stable≠decay≠sft`, `decay start_step = stable_steps`, `sft init_weights_only = 1` + `start_step = 0`) — it is what stops a degenerate "trained stable the whole time" run from passing a self-comparison.
- **③ crash-resume is self-consistent on the fused engine** is proved here by ours-crashed vs ours-not-crashed. **Self-comparison, not vs-ref**, because after long-horizon operator fusion the ours engine can no longer match the ref bitwise — but ours-vs-ours cancels the fusion, so `gate_atol = 0` is still reachable. Run it det-ON with fusion ON so it exercises the real production kernel path.

This gate needs the engine to support periodic versioned save (`save_interval` → `<save_path>/step_<abs>/training_state.pt` every N absolute steps + a phase-final ckpt under `step_<phase_end>/`) and resume-from-`step_<abs>/`-dir; if that is not yet in the tracked engine, port it (from the resume machinery) before running. The per-phase WSD schedule fields and the `[PHASE]` banner grammar are specified on `TrainLoopConfig` (train_loop.py) — implement to that contract.

## Step 2 — measure, then submit the job

**Measure step time first.** Before submitting the long job, run a short probe at the REAL production shape (GBS 1280, MBS 10, seq 4096, 4 cards — e.g. `STABLE_ITERS=20 DECAY_ITERS=0 SFT_ITERS=0` through `train_ours_al.sh` on the devspace) and record seconds/step. Project the full 2340-step recipe: if it lands materially past ~30 h, REPORT the projection in your round summary and escalate rather than silently shrinking the phase table — the toml is frozen; changing the recipe is a human decision.

Submit the PyTorchJob with the tracked launcher **`evals/scripts/launch_production.sh`** — do not hand-assemble the `cctl` flags. **`cctl` lives on the MAC, not the devspace, so the launcher runs LOCALLY on the mac** — but every path it bakes into the job spec (`SAVE_ROOT`, the `--entry` script) must be a **devspace** path (the shared FS the pod mounts), never a mac-local path. That is why `REMOTE_WORKDIR` is **required and explicit**. First `bin/harness sync push` so the worktree (launcher + entry + rendered products) lands on the devspace, then run the launcher on the mac:

```bash
# after: bin/harness sync push  (lands the worktree on the devspace shared FS)
# run ON THE MAC (where cctl is); REMOTE_WORKDIR is the DEVSPACE checkout so
# every path baked into the pod spec is a shared-FS path the pod can see.
REMOTE_WORKDIR=@@REMOTE_WORKDIR@@ \
  CHECKPOINT_ROOT=<production from-scratch init dir on the shared FS> \
  bash evals/scripts/launch_production.sh
# inspect the plan first without spending GPUs: prefix DRY_RUN=1
```

The launcher runs `cctl pytorchjob create` with `--workers 0` (single node) × `GPU_PER_NODE=4` and injects via `--env`:

1. **`CHECKPOINT_ROOT`** — the production from-scratch init dir (required; launcher and entry both fail loudly if unset). Must match the active `[model]` geometry.
2. **`SAVE_ROOT`** — **the launcher composes this per-loop; do not hardcode a shared literal.** It is `<REMOTE_WORKDIR>/.artifacts/production/wsdsft` — under **this loop's** remote checkout, which is (a) persistent shared FS, (b) per-loop isolated so two agents' jobs never trample the same dir, (c) under `.artifacts`, which `bin/harness sync push --delete` EXCLUDES — so a later sync never wipes the checkpoints.
3. The shape (GBS 1280 / MBS 10 / seq 4096), the recipe (1800/360/180 + per-phase LR/warmup), the per-phase save cadence (stable 200 / decay 50 / sft 25) and the card count all come from the recipe toml `config/train/wsdsft_05b_prod.toml` — `production_train.sh` machine-reads it (env-wins) via `tools/train_recipe_to_env.py`, so there is nothing to edit in any script. Leave the toml frozen. A non-0.5B variant selects its own recipe file on the launcher (e.g. `RECIPE_REL=config/train/wsdsft_1b_prod.toml` for the 1B line — GBS 1024 / MBS 4, same phase table). `MICRO_BATCH_SIZE` is the only memory-bound knob — changing it does not change the global batch (`grad_accum` absorbs it).
4. The entry projects the engine's hyperparameters from the **rendered ours product** (`workload/src/config/production-train.toml` via `FORGE_GATE`/`FORGE_OURS_CONFIG_DIR`) — render products and sync push BEFORE submitting, or the entry fails loudly.

**Record the launched PyTorchJob id IMMEDIATELY after submit.** The moment `cctl` returns the id, persist it to a durable manifest `$SAVE_ROOT/pytorchjob.json` (fields: `job_id`, `submit_time`, `nodes`, `git_sha`, `save_root`, `status`) **and** state the id in your round summary. The manifest names the **currently-active** job and is the *only* durable record that one exists: `$SAVE_ROOT` is persistent, so it survives across rounds/crashes exactly like the checkpoints. Without it a later round has no way to know a job is already running — and will either double-submit (wasting the quota) or lose track of the deliverable. Two rules:
- **Before submitting any NEW job, read this manifest first.** If it names a job that `cctl` still reports Running/Pending, do **not** submit another — attach to and monitor the existing one. (At most one live production job per loop; two would race on the same `$SAVE_ROOT`.)
- **On every (re-)submit, OVERWRITE the manifest with the new id.** A crash-resume is a brand-new PyTorchJob with a **new** id (see Crash-resume), so the recorded id is only meaningful for the job currently alive.

## Step 3 — supervise (mandatory crash drill included)

Monitor with **both** signals — they answer different questions:

- **`cctl pytorchjob get` / `cctl logs` — job HEALTH.** Is the job running, did it start training, did it crash? Scan for NCCL init failures, OOM, Python tracebacks, pod restarts; confirm `[LOSS]` lines are being emitted; use the status for the pod phase (Running / Pending / Failed / Succeeded). This is the primary way you notice a job died and needs a re-submit.
- **Shared FS — PROGRESS ground truth.** The newest `step_<abs>/` dir under `$SAVE_ROOT/{stable,decay,sft}/` tells you how far the run actually got. Cross-check: if the logs look alive but no new `step_<abs>/` appears across a full save interval, treat it as stalled (hung dataloader / deadlock), not healthy.

Poll the job id recorded in `$SAVE_ROOT/pytorchjob.json`. **Do not exit after submitting.** Stay in this round and use `ScheduleWakeup` to poll both signals every ~30–60 min; only finish when all three phase-final checkpoints exist (below) or the job fails unrecoverably.

**Mandatory crash drill (once, mid-stable).** At this scale the job may simply run to completion, leaving the resume machinery untested in production conditions — so testing it is part of the PASS contract. After the stable phase has passed at least its second periodic save (`step_400`+ on the shared FS), stop the job (`cctl pytorchjob stop <id> --reason "resume drill"`), re-run the launcher with the SAME `SAVE_ROOT` (new job id — overwrite the manifest), and verify from the new job's logs that the stable phase crash-restarts from the latest `step_<abs>/` (the driver logs `[PHASE stable] crash-restart from step_<N>`), not step 0. Record the drill (old id → new id, resumed step) in the round summary. Cost: at most one save interval of recompute.

## Crash-resume

Resume is **checkpoint-driven, not job-id-driven.** Checkpoints persist to `$SAVE_ROOT` (per-loop, persistent), so any job pointed at the same `SAVE_ROOT` rediscovers them. To resume a crashed / preempted / OOM'd run, **submit a NEW PyTorchJob** (re-run `launch_production.sh` — it gets a **new** id) with the **same** `SAVE_ROOT`; then overwrite `$SAVE_ROOT/pytorchjob.json` (per Step 2). `production_train.sh` → `train_ours_al.sh`'s `latest_ckpt_step()` scans each phase's save dir, finds the highest completed `step_<abs>/`, full-resumes that phase from it (weights + optim m/v/step + data cursor), and skips any phase that already reached its final step. The job id changing across the re-submit is irrelevant — only `SAVE_ROOT` staying fixed matters. No manual step bookkeeping.

## PASS contract & required output

**PASS iff** every phase exits `returncode == 0` **AND** the crash drill was performed and resumed correctly **AND** all three phase-final checkpoints exist and are non-empty:
- `$SAVE_ROOT/stable/step_1800/training_state.pt`
- `$SAVE_ROOT/decay/step_2160/training_state.pt` (abs = stable + decay)
- `$SAVE_ROOT/sft/step_180/training_state.pt` (the deliverable)

FAIL on any missing/empty final ckpt or any non-zero phase. Report the deliverable SFT ckpt path, the measured step time, and the crash-drill record (`resumed_from_step`). No `perf_log.md` entry required (production is not a perf milestone).

## Constraints

1. **Do not change the recipe for speed.** GBS / MBS / seq and the phase step counts are the frozen contract (`config/train/wsdsft_05b_prod.toml`); the phase table was already downscaled for the card cap — shrinking it further defeats the milestone. `MICRO_BATCH_SIZE` is the only memory-bound knob.
2. **Stay inside the card cap.** Single node, ≤4 GPUs. The launcher enforces this; do not bypass it with `I_HAVE_QUOTA=1` unless a human has confirmed the quota.
3. **Persistent save root only.** `SAVE_ROOT` must be the launcher-composed per-loop path on the shared FS, never the pod-local disk — that is what lets a re-submit resume instead of restarting at step 0.
4. **Data volume.** At GBS 1280 × 2340 steps the run consumes ~12B tokens — far more than the alignment gates. The dataset under the active `[data]` axis / the per-phase data confs must supply enough unique data, or the streaming loader wraps and repeats. Ensure the corpus mirror / prefetch is staged before submitting.
5. **Checkpoint matches geometry.** `CHECKPOINT_ROOT` init state and the active `[model]` axis must be the same model.

## Finish signal

When `production-train` PASSes in this round's commit (3/3 phase-final checkpoints present, crash drill recorded, all phases `returncode == 0`), append both lines on their own to the commit message:

```
MILESTONE_STATUS: production PASS
STAGE_STATUS: finished
```

`STAGE_STATUS: finished` is the sole signal that ends stage1 (`review_stage1.md` owns that contract). Do not emit it unless the `production-train` milestone is PASS in evidence this commit produced.
