#!/bin/bash
# Ours-side WSD-SFT 3-phase PRODUCTION wrapper — the per-pod command of the
# capped cctl PyTorchJob (1 node × 2-4 GPUs; see launch_production.sh). This is
# a THIN production wrapper over train_ours_al.sh: it only sets the production
# delta (per-phase periodic-save cadence, real init, engine config pointers,
# the persistent save root) and then exec's train_ours_al.sh, which owns the
# actual stable->decay->sft pipeline, the directory-level ckpt handoff, AND the
# crash-restart resume logic.
#
# EVERY pod runs this same script. A PyTorchJob operator injects the distributed
# contract (WORLD_SIZE = node count, RANK = node rank, MASTER_ADDR/MASTER_PORT);
# train_ours_al.sh reads them straight through to torchrun. Under this line's
# per-person card cap the job is single-node, so those default sanely too. To
# change card count, change [launch].gpus_per_node in the recipe TOML (or the
# launcher's GPU_PER_NODE env) — nothing here.
#
# Crash-resume is automatic and CHECKPOINT-driven, not job-id-driven: on a job
# crash you submit a NEW PyTorchJob (new id) with the SAME SAVE_ROOT; this script
# re-runs and train_ours_al.sh's latest_ckpt_step() rediscovers each phase's
# newest step_<abs>/ under $SAVE_ROOT (persistent shared storage) and
# full-resumes from it (weights + optim m/v/step + data cursor). Completed
# phases are skipped. So a preempted run picks up where it stopped, not step 0 —
# the id changing across the re-submit is irrelevant; only SAVE_ROOT is fixed.
#
# NOT a gate: no ref subprocess, no bitwise/loss/MFU verdict. The deliverable is
# the trained SFT checkpoint. Verify success by the presence of the phase-final
# step dirs on the shared FS (see production.md PASS contract), not by [LOSS].

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

set -euo pipefail

# ── Recipe TOML — the run-shape SSOT, MACHINE-READ here. Projects [launch] +
#    [common] + [phases.*] into env with env-wins semantics (a launcher --env
#    or an operator export still beats the TOML), then train_ours_al.sh picks
#    every value up through its ${VAR:-default} surface. Card count and phase
#    step counts are therefore TOML-configurable without touching any script.
#    The launcher passes the pod-visible path via --env TRAIN_RECIPE_TOML. ──
TRAIN_RECIPE_TOML="${TRAIN_RECIPE_TOML:-$REPO_ROOT/config/train/wsdsft_05b_prod.toml}"
[[ -f "$TRAIN_RECIPE_TOML" ]] || {
    echo "FATAL: production recipe TOML not found: $TRAIN_RECIPE_TOML" >&2
    exit 1
}
# eval-form, not `source <(...)` — old bash (mac 3.2) silently sources zero
# bytes from a process substitution; eval is version-robust.
eval "$(python3 "$REPO_ROOT/tools/train_recipe_to_env.py" "$TRAIN_RECIPE_TOML")"

# ── Topology. GPUS_PER_NODE: launcher --env > recipe [launch] > 4;
#    WORLD_SIZE/RANK/MASTER_* come from the operator. ──
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

# ── Init selection is NOT env-driven on this line: the rendered ours product
#    (workload/src/config/production-train.toml [cli].forge_init_ones) carries
#    it, and the engine derives the init subdir via runtime_config — the old
#    FORGE_INIT_ONES env redirect is retired harness-wide. Nothing to set here;
#    the product the FORGE_GATE pointer below names is the single source. ──

# ── Periodic versioned checkpoints, PER PHASE (train_ours_al.sh threads each
#    into the engine's save_interval for that phase). 200/50/25 over the
#    downscaled 1800/360/180 recipe => ~9/7/7 intermediate ckpts per phase,
#    each a full resumable state (a crash loses at most one interval, ~1-2 h).
#    Values come from the recipe TOML [common] sourced above; the :-fallbacks
#    only fire if the TOML drops the keys. ──
export STABLE_SAVE_INTERVAL="${STABLE_SAVE_INTERVAL:-200}"
export DECAY_SAVE_INTERVAL="${DECAY_SAVE_INTERVAL:-50}"
export SFT_SAVE_INTERVAL="${SFT_SAVE_INTERVAL:-25}"

# ── Persistent checkpoint root. REQUIRED — the launcher composes a per-loop
#    path and passes it via --env, so a bare-entry run without it fails loud.
#    It MUST be a persistent shared-FS path that is per-loop unique: a shared
#    literal means two agents' jobs write the same dir and trample each other's
#    checkpoints. launch_production.sh sets it to
#    <remote_workdir>/.artifacts/production/wsdsft — persistent, per-loop
#    isolated, and under .artifacts (excluded from `sync push --delete` so a
#    later sync never wipes the checkpoints). train_ours_al.sh lays out
#    <SAVE_ROOT>/{stable,decay,sft}/step_<abs>/ under here. ──
export SAVE_ROOT="${SAVE_ROOT:?SAVE_ROOT must be a per-loop persistent shared-FS path (set by launch_production.sh); a shared literal collides across agents}"

# ── Canonical from-scratch init state for the stable phase's first launch.
#    Must match the active [model] geometry (0.5B here). Required — fail loudly
#    if the submitter did not point it at the production init dir. ──
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:?CHECKPOINT_ROOT must be the production from-scratch init dir (matching the active [model] geometry)}"

# ── Engine hyperparameter pointers (single source = the rendered product).
#    The ours engine's runtime_config.load() reads its model/optim/muP/MTP
#    shape from FORGE_OURS_CONFIG_DIR/<FORGE_GATE>.toml. A GATE run gets these
#    injected by the dispatcher (suite_process_env); this dispatcher-LESS
#    production path must set the same pointers itself, else the engine
#    FATAL-crashes at runtime_config.load(). The product must have been
#    rendered (tools/render_gate_configs.py) before `sync push` landed the
#    worktree — fail loud here if it is missing rather than 3 phases later. ──
FORGE_SUITE="${FORGE_PRODUCTION_SUITE:-production-train}"
export FORGE_GATE="$FORGE_SUITE"
export FORGE_OURS_CONFIG_DIR="${FORGE_OURS_CONFIG_DIR:-$REPO_ROOT/workload/src/config}"
[[ -f "$FORGE_OURS_CONFIG_DIR/$FORGE_GATE.toml" ]] || {
    echo "FATAL: rendered ours product not found: $FORGE_OURS_CONFIG_DIR/$FORGE_GATE.toml" >&2
    echo "       Render it (python -m tools.render_gate_configs) and sync push before submitting." >&2
    exit 1
}
# Data-axis pointer (runtime_config.data_loader() reads [data].data_loader).
export FORGE_DATA_TOML="${FORGE_DATA_TOML:-$REPO_ROOT/config/data.toml}"
[[ -f "$FORGE_DATA_TOML" ]] || {
    echo "FATAL: FORGE_DATA_TOML not found: $FORGE_DATA_TOML (the per-loop config dir's data.toml)" >&2
    exit 1
}

echo "=== production_train (node_rank=${RANK:-0}/${WORLD_SIZE:-1} nodes, ${GPUS_PER_NODE} GPU/node) ===" >&2
echo "  recipe = $TRAIN_RECIPE_TOML (stable/decay/sft = ${STABLE_ITERS:-?}/${DECAY_ITERS:-?}/${SFT_ITERS:-?})" >&2
echo "  SAVE_ROOT=$SAVE_ROOT" >&2
echo "  save_interval stable/decay/sft = $STABLE_SAVE_INTERVAL/$DECAY_SAVE_INTERVAL/$SFT_SAVE_INTERVAL" >&2
echo "  engine product = $FORGE_OURS_CONFIG_DIR/$FORGE_GATE.toml" >&2

exec bash "$SCRIPT_DIR/train_ours_al.sh"
