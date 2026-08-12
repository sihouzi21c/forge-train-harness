#!/usr/bin/env bash
# WSD-SFT ours-side gate runner (wsd-sft-70 / production-resume-70).
#
# Contract:  bash evals/scripts/run_ours_wsd_sft.sh <gate> <run_dir>
#            (cwd = workspace root; <run_dir> = <artifact_dir>/ours)
#
# Self-comparison shape: TWO ours runs of the same 3-phase stable→decay→sft
# line — ① a clean baseline through the plain driver (train_ours_al.sh,
# never crashes) and ② the main run through the registry-routed driver
# (FORGE_OURS_ENTRY: the plain driver again for wsd-sft-70, the SIGKILL
# crash-resume wrapper train_ours_al_resume.sh for production-resume-70).
# NO branching on the gate name: phase table / crash schedule / save
# interval all come from the rendered ours product; the driver difference
# comes from the registry `script` value via runtime_env's FORGE_OURS_ENTRY.
#
# Artifacts under <run_dir> the wsd_sft verdict reads back:
#   wsdsft_clean_baseline_output.log   baseline stdout ([PHASE]/[LOSS] wire)
#   wsdsft_trajectory_output.log       main-run stdout
#   clean_returncode.txt / main_returncode.txt
# Scratches live under $PWD/tmp/ (NOT .artifacts and NOT /tmp — on a
# devspace both sit on the 50 GiB docker overlay), unique per run via the
# artifact dir name.
set -euo pipefail

GATE=${1:?usage: run_ours_wsd_sft.sh <gate> <run_dir>}
RUN_DIR=${2:?usage: run_ours_wsd_sft.sh <gate> <run_dir>}
mkdir -p "$RUN_DIR"

source evals/scripts/gate_runner_common.sh

OURS_PRODUCT=$(resolve_product ours "$GATE" "")
project_env ours "$GATE" "$OURS_PRODUCT"

: "${FORGE_OURS_ENTRY:?suite registry declares no script for $GATE}"
: "${WORLD_SIZE:?product carries no world_size}"
: "${STABLE_STEPS:?product carries no stable_steps}"
: "${DECAY_STEPS:?product carries no decay_steps}"
: "${SFT_STEPS:?product carries no sft_steps}"

# comparison_basis: "self" only on this line (the pure-torch L0 ref has no
# 3-phase wrapper). Fail fast before burning two runs.
if [ "${COMPARISON_BASIS:-self}" != "self" ]; then
    echo "ERROR: comparison_basis=${COMPARISON_BASIS} is not available on this line (no 3-phase ref wrapper); use \"self\"" >&2
    exit 1
fi

# ── Product → driver env contract (the legacy handler's _ours_env, in sh).
# Single-node topology fold: the bash driver reads WORLD_SIZE as NODE count
# (torchrun semantics), which collides with the product's DP-proc
# world_size. The gate runs on ONE node, so fold the DP world into
# GPUS_PER_NODE and pin the driver to a single node; torchrun then spawns
# world_size local ranks and sets each child's WORLD_SIZE itself. ──
export GPUS_PER_NODE="$WORLD_SIZE"
export WORLD_SIZE=1
export RANK=0
export SEED="${SEED:-1234}"
export SEQ_LENGTH="${SEQ_LENGTH:-4096}"
export STABLE_ITERS="$STABLE_STEPS"
export STABLE_WARMUP="${STABLE_WARMUP:-2}"
export DECAY_ITERS="$DECAY_STEPS"
export DECAY_WARMUP="${DECAY_WARMUP:-0}"
export SFT_ITERS="$SFT_STEPS"
export SFT_WARMUP="${SFT_WARMUP:-3}"
DEFAULT_SFT_WSD=$(( SFT_STEPS - SFT_WARMUP ))
[ "$DEFAULT_SFT_WSD" -lt 0 ] && DEFAULT_SFT_WSD=0
export SFT_WSD_DECAY_ITERS="${SFT_WSD:-$DEFAULT_SFT_WSD}"
# MICRO/GLOBAL_BATCH_SIZE and the per-phase *_LR / *_MIN_LR keys pass
# through under their product-export names (already process env from the
# full product export above).
# Per-phase data confs, pinned absolute so the gate fixes the exact corpora
# regardless of the driver's defaults.
export STABLE_DATA_CONF="$PWD/ref/reference/wsdsft_stable_data_conf.sh"
export DECAY_DATA_CONF="$PWD/ref/reference/wsdsft_decay_data_conf.sh"
export SFT_DATA_CONF="$PWD/ref/reference/wsdsft_sft_data_conf.sh"
export SAVE_INTERVAL="${SAVE_INTERVAL:-0}"
# product_env space-joins list values; the resume driver parses commas.
MAIN_CRASH_STEPS="${CRASH_STEPS:-}"
MAIN_CRASH_STEPS="${MAIN_CRASH_STEPS// /,}"

SCRATCH_TAG="$(basename "$(dirname "$RUN_DIR")")"

# ── ① Clean baseline: the plain never-crash driver. Identical env shape to
#    the main run except SAVE_ROOT and the crash schedule (SAVE_INTERVAL is
#    the same — writing a versioned ckpt never changes the loss), so any
#    per-step loss diff is attributable to crash-resume alone. ──
set +e
SAVE_ROOT="$PWD/tmp/wsdsft_ours_clean_scratch_$SCRATCH_TAG" CRASH_STEPS="" \
    bash evals/scripts/train_ours_al.sh \
    > "$RUN_DIR/wsdsft_clean_baseline_output.log" 2>&1
CLEAN_RC=$?
set -e
echo "$CLEAN_RC" > "$RUN_DIR/clean_returncode.txt"
if [ "$CLEAN_RC" -ne 0 ]; then
    echo "ERROR: clean baseline failed (returncode=$CLEAN_RC)" >&2
    exit 1
fi

# ── ② Main run through the registry driver. A non-zero rc is recorded, NOT
#    fatal here — the verdict folds it into its judgment with full context. ──
set +e
SAVE_ROOT="$PWD/tmp/wsdsft_ours_scratch_$SCRATCH_TAG" CRASH_STEPS="$MAIN_CRASH_STEPS" \
    bash "$FORGE_OURS_ENTRY" \
    > "$RUN_DIR/wsdsft_trajectory_output.log" 2>&1
MAIN_RC=$?
set -e
echo "$MAIN_RC" > "$RUN_DIR/main_returncode.txt"
