#!/usr/bin/env bash
# Production ours-side runner (thin-dispatcher refactor, step 5).
#
# Contract:  bash evals/scripts/run_ours_production.sh <gate> <run_dir>
#            (cwd = workspace root; <run_dir> = <artifact_dir>/ours)
#
# The ONLY multi-launch gate runner: walks the product's total NUM_STEPS in
# PRODUCTION_SAVE_SEGMENTS equal segments; each segment is a fresh engine
# launch that saves a full resume checkpoint, and a restarted run (after a
# crash) skips segments whose checkpoint already exists — crash-resume.
# NO branching on the gate name: segments / checkpoint root / per-segment
# timeout all come from VALUES (product keys with convention defaults).
#
# Per-segment values cross into the engine via the per-run env whitelist
# (NUM_STEPS / START_STEP / FORGE_SAVE_PATH / FORGE_RESUME_FROM — env-first
# in _gate_entry.GateInputs); every static key stays product-first from the
# single frozen --config. No per-segment TOML copies are composed.
#
# Artifacts under <run_dir> the harness-side verdict reads back:
#   training_output_step_<N>.log   one per segment (stdout+stderr)
#   segments.jsonl                 one record per attempted segment:
#                                  {"start","end","returncode","timed_out",
#                                   "timeout_s","elapsed_s"}
#   resumed_from.txt               abs step this invocation resumed at (0 = fresh)
# and <artifact_dir>/checkpoint_root.txt — the STABLE checkpoint root, which
# lives OUTSIDE the per-run dir (and outside <runs> pruning) so a brand-new
# run discovers the same root:  <.artifacts>/production_train/<gate>/checkpoints
set -euo pipefail

GATE=${1:?usage: run_ours_production.sh <gate> <run_dir>}
RUN_DIR=${2:?usage: run_ours_production.sh <gate> <run_dir>}
mkdir -p "$RUN_DIR"

OURS_PRODUCT="workload/src/config/$GATE.toml"
[ -f "$OURS_PRODUCT" ] || { echo "ERROR: ours product not found: $OURS_PRODUCT" >&2; exit 1; }

eval "$(python3 tools/product_env.py "$OURS_PRODUCT")"
eval "$(python3 evals/scripts/runtime_env.py ours "$GATE")"

: "${FORGE_OURS_ENTRY:?suite registry declares no script for $GATE}"
: "${FORGE_OURS_LAUNCHER:?suite registry declares no launcher for $GATE}"
: "${NUM_STEPS:?product carries no num_steps}"

# Segmentation values: product keys with the legacy convention defaults
# (dispatcher constant _PRODUCTION_SAVE_SEGMENTS=4 + the stable-root path
# formula). PRODUCTION_SEGMENT_TIMEOUT_S is optional (0/unset = no per-
# segment ceiling; the executor's suite timeout still bounds the whole run).
: "${PRODUCTION_SAVE_SEGMENTS:=4}"
: "${PRODUCTION_SEGMENT_TIMEOUT_S:=0}"
TOTAL_STEPS=$NUM_STEPS
if [ "$((TOTAL_STEPS % PRODUCTION_SAVE_SEGMENTS))" -ne 0 ]; then
    echo "ERROR: num_steps=$TOTAL_STEPS must be divisible by $PRODUCTION_SAVE_SEGMENTS so each segment ends on a checkpoint" >&2
    exit 1
fi
SEG_STEPS=$((TOTAL_STEPS / PRODUCTION_SAVE_SEGMENTS))

# Stable checkpoint root: <run_dir> = <artifact_dir>/ours, <artifact_dir> =
# <runs>/<run-name>, so three dirnames up is the artifact root.
ART_DIR="$(dirname "$RUN_DIR")"
ART_ROOT="$(dirname "$(dirname "$ART_DIR")")"
: "${PRODUCTION_CKPT_ROOT:=$ART_ROOT/production_train/$GATE/checkpoints}"
# Record the stable root in the per-run dir so each run is traceable to
# where its checkpoints actually landed (run dir != checkpoint dir).
echo "$PRODUCTION_CKPT_ROOT" > "$ART_DIR/checkpoint_root.txt"

# Data-readiness gate: block on the corpus prefetch sentinel when
# [data].prefetch_target_gb is configured (fail-fast on error/timeout).
python3 evals/scripts/production_ckpt.py prefetch-wait

# Crash-resume: discover the last completed checkpoint; skip segments at or
# below it and resume the next one from its state.
IFS=$'\t' read -r RESUMED_FROM PREV_CKPT < <(python3 evals/scripts/production_ckpt.py latest "$PRODUCTION_CKPT_ROOT")
echo "$RESUMED_FROM" > "$RUN_DIR/resumed_from.txt"

SEGMENTS_JSONL="$RUN_DIR/segments.jsonl"
: > "$SEGMENTS_JSONL"

k=1
while [ "$k" -le "$PRODUCTION_SAVE_SEGMENTS" ]; do
    ABS_END=$((SEG_STEPS * k))
    START_STEP=$((ABS_END - SEG_STEPS))
    k=$((k + 1))
    if [ "$ABS_END" -le "$RESUMED_FROM" ]; then
        # This segment's checkpoint already exists from a prior run.
        continue
    fi
    STEP_DIR="$PRODUCTION_CKPT_ROOT/step_$ABS_END"

    SEG_CMD=(python3 "$FORGE_OURS_LAUNCHER" "$FORGE_OURS_ENTRY" --config "$OURS_PRODUCT")
    if [ "$PRODUCTION_SEGMENT_TIMEOUT_S" -gt 0 ]; then
        # TERM then KILL; rc=124 marks a per-segment timeout. (A TERM'd
        # launch_dp may orphan rank children; the next segment's
        # abort_if_gpu_dirty fails fast on a still-dirty GPU.)
        SEG_CMD=(timeout -k 30 "$PRODUCTION_SEGMENT_TIMEOUT_S" "${SEG_CMD[@]}")
    fi

    SEG_T0=$SECONDS
    set +e
    # env(1), not bash prefix assignments: an expansion like
    # ${PREV_CKPT:+FORGE_RESUME_FROM=...} is a command WORD after expansion,
    # never a prefix assignment — env accepts assignments as arguments.
    env NUM_STEPS="$SEG_STEPS" START_STEP="$START_STEP" FORGE_SAVE_PATH="$STEP_DIR" \
        ${PREV_CKPT:+"FORGE_RESUME_FROM=$PREV_CKPT"} \
        "${SEG_CMD[@]}" > "$RUN_DIR/training_output_step_$ABS_END.log" 2>&1
    RC=$?
    set -e
    ELAPSED=$((SECONDS - SEG_T0))
    TIMED_OUT=false
    if [ "$PRODUCTION_SEGMENT_TIMEOUT_S" -gt 0 ] && [ "$RC" -eq 124 ]; then
        TIMED_OUT=true
    fi
    printf '{"start": %d, "end": %d, "returncode": %d, "timed_out": %s, "timeout_s": %s, "elapsed_s": %d}\n' \
        "$START_STEP" "$ABS_END" "$RC" "$TIMED_OUT" "$PRODUCTION_SEGMENT_TIMEOUT_S" "$ELAPSED" \
        >> "$SEGMENTS_JSONL"

    if [ "$RC" -ne 0 ]; then
        echo "ERROR: segment [$START_STEP,$ABS_END) failed (returncode=$RC)" >&2
        exit 1
    fi
    if ! python3 evals/scripts/production_ckpt.py check "$STEP_DIR"; then
        echo "ERROR: segment [$START_STEP,$ABS_END) completed but checkpoint $STEP_DIR is incomplete" >&2
        exit 1
    fi
    PREV_CKPT=$STEP_DIR
done
