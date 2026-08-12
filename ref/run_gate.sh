#!/usr/bin/env bash
# Single ref-side gate runner (thin-dispatcher refactor, step 1).
#
# Contract:  bash ref/run_gate.sh <gate> <run_dir>     (cwd = workspace root)
#
# One script for every gate — NO branching on the gate name. All behavior
# differences come from the values read out of the gate's rendered ref
# product (ref/config/<gate>.toml) and the axis-resolved runtime env:
#
#   * hash_capture_level == 0  → exec the bare L0 ref launcher
#     (FORGE_REF_SCRIPT_PATH); the product's shape / model / optim values
#     are already process env (full generic projection below), consumed
#     by the launcher under the upper-cased [cli] key names.
#   * hash_capture_level  > 0  → exec the capture bridge
#     (FORGE_REF_CAPTURE_SCRIPT_PATH) with the --hash-* CLI wire, dumping
#     persistent hash records to <run_dir>/ref_hash_dump.json.
#
# Shared runner logic (product lookup, env projection, hash-capture wire)
# lives in evals/scripts/gate_runner_common.sh — one library for both sides.
#
# Artifacts land under <run_dir> by fixed conventions the harness-side
# verdict script reads back:
#   <run_dir>/dump/ref_loss.txt      per-step [LOSS] trajectory dump
#   <run_dir>/dump/{save,tensorboard}
#   <run_dir>/ref_hash_dump.json     (hash_capture_level > 0 only)
set -euo pipefail

GATE=${1:?usage: run_gate.sh <gate> <run_dir>}
RUN_DIR=${2:?usage: run_gate.sh <gate> <run_dir>}
mkdir -p "$RUN_DIR"

source evals/scripts/gate_runner_common.sh

REF_PRODUCT=$(resolve_product ref "$GATE")

# Full generic projection (mirror of the ours side): product export first
# (every [cli] key under its upper-cased name + [env] verbatim), then the
# axis-resolved runtime env — deployment paths, asset dirs, SSOT pointers,
# resolved launcher/bridge paths, MASTER_ADDR/PORT — eval'd last so it wins.
project_env ref "$GATE" "$REF_PRODUCT"

# Routing values, already process env from the full product export above:
# the capture level decides bare-launcher vs. bridge below; num_steps
# decides persistent (multi-step) vs. single-step capture mode.
HASH_CAPTURE_LEVEL=${HASH_CAPTURE_LEVEL:-0}
NUM_STEPS=${NUM_STEPS:-1}

# Per-run artifact paths, derived from run_dir by fixed convention.
export LOCAL_MODE=1
export FORGE_GATE="$GATE"
export DUMP_DIR="$RUN_DIR/dump"
mkdir -p "$DUMP_DIR"
export LOSS_DUMP_FILE="$DUMP_DIR/ref_loss.txt"
export SAVE_PATH="$DUMP_DIR/save"
export TENSORBOARD_DIR="$DUMP_DIR/tensorboard"

# Suite-level extra CLI tokens (pre-quoted list from runtime_env).
EXTRA_ARGS=()
if [ -n "${FORGE_REF_EXTRA_ARGS:-}" ]; then
    eval "EXTRA_ARGS=($FORGE_REF_EXTRA_ARGS)"
fi

setup_hash_capture ref "$RUN_DIR" "$HASH_CAPTURE_LEVEL" "$NUM_STEPS"

if [ "$HASH_CAPTURE_LEVEL" -gt 0 ]; then
    : "${FORGE_REF_CAPTURE_SCRIPT_PATH:?hash_capture_level > 0 but no ref_capture_script configured}"
    exec bash "$FORGE_REF_CAPTURE_SCRIPT_PATH" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
        ${HASH_ARGS[@]+"${HASH_ARGS[@]}"}
else
    exec bash "$FORGE_REF_SCRIPT_PATH" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
fi
