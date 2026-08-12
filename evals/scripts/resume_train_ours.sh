#!/bin/bash
# Ours-side RESUME-milestone short-train job entry — the per-pod command of the
# cctl PyTorchJob submitted by launch_resume.sh ours. ONE phase of the
# self-developed engine (train_ours_phase.py -> run_training_loop) for a SHORT
# window (default 20 steps; config/train/resume_20.toml [train].iters), with
# periodic versioned checkpoints and CHECKPOINT-driven resume:
#
#   * first submit  -> fresh start from CHECKPOINT_ROOT, trains iters steps,
#                      writes <SAVE_ROOT>/step_<abs>/ every save_interval + a
#                      final ckpt at step_<iters>.
#   * re-submit with the SAME SAVE_ROOT -> rediscovers the latest step_<abs>/
#                      and FULL-resumes the remaining span (weights + optim
#                      m/v/step + data cursor). Already-complete -> no-op.
#
# This is the resume milestone's proof that the ours engine trains and
# checkpoint-resumes as a real remote-GPU job (the devspace gates prove it
# in-process; this proves it under the PyTorchJob operator). NOT a gate: no
# ref subprocess, no verdict — verify by the step_<abs>/ dirs on the shared FS
# and the [LOSS] lines in `cctl logs`. Sibling: resume_train_ref.sh (the
# ref-side truth run of the same recipe).
#
# Env in: SAVE_ROOT (required, per-loop persistent), CHECKPOINT_ROOT (required),
# RESUME_RECIPE_TOML / FORGE_RESUME_SUITE / DATA_CONF / GPUS_PER_NODE optional;
# WORLD_SIZE/RANK/MASTER_* from the PyTorchJob operator. Every recipe knob is
# env-overridable (env > recipe TOML > fallback).

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
OURS_ENTRY="$SCRIPT_DIR/train_ours_phase.py"

set -euo pipefail

export PYTHONPATH="$REPO_ROOT/workload/src:$SCRIPT_DIR:${PYTHONPATH:-}"

# ── Recipe TOML (env-wins projection): ITERS / SAVE_INTERVAL / shape / LR. ──
RESUME_RECIPE_TOML="${RESUME_RECIPE_TOML:-$REPO_ROOT/config/train/resume_20.toml}"
[[ -f "$RESUME_RECIPE_TOML" ]] || {
    echo "FATAL: resume recipe TOML not found: $RESUME_RECIPE_TOML" >&2
    exit 1
}
# eval-form, not `source <(...)` — old bash (mac 3.2) silently sources zero
# bytes from a process substitution; eval is version-robust.
eval "$(python3 "$REPO_ROOT/tools/train_recipe_to_env.py" "$RESUME_RECIPE_TOML")"

# ── Distributed topology (PyTorchJob operator: WORLD_SIZE = node count). ──
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
NNODES="${WORLD_SIZE:-1}"
NODE_RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-23456}"
DP_WORLD_SIZE=$(( NNODES * GPUS_PER_NODE ))

# ── Determinism env (must precede CUDA init; mirrors train_ours_al.sh). ──
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

# ── Required roots (the launcher composes SAVE_ROOT per loop). ──
SAVE_ROOT="${SAVE_ROOT:?SAVE_ROOT must be a per-loop persistent shared-FS path (set by launch_resume.sh)}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:?CHECKPOINT_ROOT must be the canonical init dir (matching the active [model] geometry)}"

# ── Engine hyperparameter pointers — same contract as production_train.sh:
#    runtime_config.load() reads model/optim/muP/MTP from the rendered ours
#    product FORGE_OURS_CONFIG_DIR/<FORGE_GATE>.toml. Default product =
#    resume-gate-20 (the resume milestone's own rendered shape). ──
FORGE_SUITE="${FORGE_RESUME_SUITE:-resume-gate-20}"
export FORGE_GATE="$FORGE_SUITE"
export FORGE_OURS_CONFIG_DIR="${FORGE_OURS_CONFIG_DIR:-$REPO_ROOT/workload/src/config}"
[[ -f "$FORGE_OURS_CONFIG_DIR/$FORGE_GATE.toml" ]] || {
    echo "FATAL: rendered ours product not found: $FORGE_OURS_CONFIG_DIR/$FORGE_GATE.toml" >&2
    echo "       Render it (python -m tools.render_gate_configs) and sync push before submitting." >&2
    exit 1
}
export FORGE_DATA_TOML="${FORGE_DATA_TOML:-$REPO_ROOT/config/data.toml}"
[[ -f "$FORGE_DATA_TOML" ]] || {
    echo "FATAL: FORGE_DATA_TOML not found: $FORGE_DATA_TOML" >&2
    exit 1
}

# ── Corpus: DATA_CONF exports DATA_PATH; default = the [data] axis conf. ──
if [[ -z "${DATA_CONF:-}" ]]; then
    DATA_CONF="$REPO_ROOT/$(python3 -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["data"]["conf_path"])' "$FORGE_DATA_TOML")"
fi
[[ -f "$DATA_CONF" ]] || { echo "FATAL: DATA_CONF not found: $DATA_CONF" >&2; exit 1; }
# shellcheck disable=SC1090  # DATA_CONF is resolved at runtime by design
source "$DATA_CONF"
[[ -n "${DATA_PATH:-}" ]] || { echo "FATAL: DATA_CONF exported no DATA_PATH" >&2; exit 1; }
export DATA_PATH

# ── Shape (recipe [train], env-overridable). ──
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-80}"
export SEQ_LENGTH="${SEQ_LENGTH:-4096}"
export SEED="${SEED:-1234}"
export BACKEND="${BACKEND:-torch}"
export MEGATRON_ROOT="${MEGATRON_ROOT:-}"
ITERS="${ITERS:-20}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10}"

if (( GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * DP_WORLD_SIZE) != 0 )); then
    echo "ERROR: GBS=$GLOBAL_BATCH_SIZE not divisible by MBS=$MICRO_BATCH_SIZE * DP=$DP_WORLD_SIZE" >&2
    exit 1
fi
export GRAD_ACCUM_STEPS=$(( GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * DP_WORLD_SIZE) ))

# ── LR schedule (recipe [train]; evaluated at the absolute step, so a resumed
#    run continues the same curve). ──
export LR="${LR:-1e-2}" MIN_LR="${MIN_LR:-0}"
export LR_WARMUP_ITERS="${WARMUP:-0}"
export LR_DECAY_ITERS="${LR_DECAY_ITERS:-$ITERS}"
export LR_WSD_DECAY_ITERS="${WSD_DECAY_ITERS:-0}"

# latest_ckpt_step <dir> — max absolute N over <dir>/step_<N>/ that contain a
# *training_state.pt (same discovery train_ours_al.sh uses; guards against a
# half-written dir).
latest_ckpt_step() {
    local dir="$1" d n best=""
    [[ -d "$dir" ]] || return 0
    for d in "$dir"/step_*/; do
        [[ -d "$d" ]] || continue
        compgen -G "${d}*training_state.pt" >/dev/null || continue
        n="${d%/}"; n="${n##*/step_}"
        [[ "$n" =~ ^[0-9]+$ ]] || continue
        n=$(( 10#$n ))
        if [[ -z "$best" || "$n" -gt "$best" ]]; then best="$n"; fi
    done
    [[ -n "$best" ]] && echo "$best"
    return 0
}

# ── Checkpoint-driven entry split (fresh / resume-remaining / already-done). ──
export SAVE_PATH="$SAVE_ROOT"
export INIT_WEIGHTS_ONLY=0
export PHASE_NAME=resume
latest=$(latest_ckpt_step "$SAVE_ROOT")
if [[ -n "$latest" && "$latest" -ge "$ITERS" ]]; then
    echo "[resume_train_ours] already complete (latest=$latest >= $ITERS) — nothing to do." >&2
    exit 0
elif [[ -n "$latest" ]]; then
    echo "[resume_train_ours] resuming from step_$latest (remaining=$(( ITERS - latest )))." >&2
    export RESUME_FROM="$SAVE_ROOT/step_$latest" START_STEP="$latest" NUM_STEPS=$(( ITERS - latest ))
else
    echo "[resume_train_ours] fresh start (0..$ITERS)." >&2
    export RESUME_FROM="" START_STEP=0 NUM_STEPS="$ITERS"
fi
mkdir -p "$SAVE_PATH"

echo "=== resume_train_ours (node_rank=$NODE_RANK/$NNODES nodes, $GPUS_PER_NODE GPU/node, DP=$DP_WORLD_SIZE) ===" >&2
echo "  recipe = $RESUME_RECIPE_TOML  iters=$ITERS save_interval=$SAVE_INTERVAL" >&2
echo "  GBS=$GLOBAL_BATCH_SIZE MBS=$MICRO_BATCH_SIZE accum=$GRAD_ACCUM_STEPS seq=$SEQ_LENGTH seed=$SEED" >&2
echo "  SAVE_ROOT=$SAVE_ROOT  product=$FORGE_OURS_CONFIG_DIR/$FORGE_GATE.toml" >&2

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN: torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK $OURS_ENTRY" >&2
    exit 0
fi

exec torchrun \
    --nproc_per_node="$GPUS_PER_NODE" \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "$OURS_ENTRY"
