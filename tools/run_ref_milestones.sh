#!/bin/bash
# ============================================================================
# Run the pure-torch reference under every Stage 1 FORGE_GATE preset, twice,
# with the streaming HF dataloader (gsm8k vehicle by default), and verify:
#   1. the ref trains through each milestone shape without crashing, and
#   2. the per-batch data hashes ([DATAHASH] lines emitted by
#      hf_stream_dataloader when FORGE_HF_DATA_DUMP=1) are bitwise-identical
#      across the two runs (same seed → same data).
#
# Run from the harness/ dir on a GPU box (2 GPUs cover all presets):
#   GSM8K_DIR=/opt/forge-data/gsm8k \
#   TOKENIZER_MODEL=/opt/forge-data/tokenizer/tokenizer.model \
#   bash tools/run_ref_milestones.sh
#
# Env knobs:
#   PRESETS="forward-align multistep ..."   override the preset list
#   LONG_TRAIN_STEPS=200                     cap long-train steps (default 200)
#   OUT_ROOT=/tmp/ref_milestones             artifact root
# ============================================================================
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HARNESS_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
REF_SCRIPT="$HARNESS_DIR/ref/reference/run_16gpu_1000step_pure_mup_mtp.sh"
DATA_CONF="${DATA_CONF:-$HARNESS_DIR/ref/reference/gsm8k_hf_data_conf.sh}"
OUT_ROOT="${OUT_ROOT:-/tmp/ref_milestones}"
LONG_TRAIN_STEPS="${LONG_TRAIN_STEPS:-200}"

PRESETS="${PRESETS:-forward-align backward-align multistep-1gpu multistep perf-bitwise resume-gate-20 long-train}"

mkdir -p "$OUT_ROOT"
overall_ok=1

extract_hashes() {  # $1 = run log -> stdout: sorted DATAHASH tuples
    # Robust to multi-rank stdout interleaving: torchrun runs N ranks that
    # all print() to the same fd, so two ranks' lines occasionally land
    # concatenated without a newline. Extract each rank/batch/sha tuple by
    # pattern (not by line) so the comparison reflects data, not framing.
    grep -oE 'rank=[0-9]+ batch=[0-9]+ sha256=[0-9a-f]+' "$1" | sort
}

run_once() {  # $1 preset  $2 run-tag -> writes $OUT_ROOT/<preset>.<tag>.log
    local preset="$1" tag="$2"
    local log="$OUT_ROOT/${preset}.${tag}.log"
    # The launcher consumes the rendered ref product as environment (generic
    # projection). Project it here, THEN export this script's own overrides
    # (data conf / capped long-train steps) so they win over product values.
    local product="${FORGE_REF_CONFIG_DIR:-$HARNESS_DIR/ref/config}/${preset}.toml"
    if [[ ! -f "$product" ]]; then
        {
            echo "error: gate product not found: $product"
            echo "       (render it first with tools/render_gate_configs.py)"
        } >"$log"
        return 3
    fi
    (
        eval "$(python3 "$HARNESS_DIR/tools/product_env.py" "$product")" || exit 3
        export LOCAL_MODE=1 \
            FORGE_GATE="$preset" \
            DATA_CONF="$DATA_CONF" \
            FORGE_HF_DATA_DUMP=1 \
            OUT_DIR="$OUT_ROOT/${preset}.${tag}.out" \
            GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
        if [[ "$preset" == "long-train" ]]; then
            export NUM_STEPS="$LONG_TRAIN_STEPS"
        fi
        exec bash "$REF_SCRIPT"
    ) >"$log" 2>&1
    return $?
}

for preset in $PRESETS; do
    echo "=== preset: $preset ==="
    run_once "$preset" r1; rc1=$?
    run_once "$preset" r2; rc2=$?
    if [[ $rc1 -ne 0 || $rc2 -ne 0 ]]; then
        echo "  [FAIL] ref run crashed (rc1=$rc1 rc2=$rc2) — see $OUT_ROOT/${preset}.r*.log"
        tail -n 25 "$OUT_ROOT/${preset}.r1.log" | sed 's/^/    /'
        overall_ok=0
        continue
    fi
    h1="$OUT_ROOT/${preset}.r1.hashes"; h2="$OUT_ROOT/${preset}.r2.hashes"
    extract_hashes "$OUT_ROOT/${preset}.r1.log" >"$h1"
    extract_hashes "$OUT_ROOT/${preset}.r2.log" >"$h2"
    n=$(wc -l <"$h1")
    if [[ "$n" -eq 0 ]]; then
        echo "  [WARN] ran OK but no [DATAHASH] lines (FORGE_HF_DATA_DUMP not honored?)"
        overall_ok=0
    elif diff -q "$h1" "$h2" >/dev/null; then
        echo "  [PASS] ran OK, $n data-hash lines bitwise-identical across 2 runs"
    else
        echo "  [FAIL] data hashes differ between runs:"
        diff "$h1" "$h2" | head -n 6 | sed 's/^/    /'
        overall_ok=0
    fi
done

echo
if [[ $overall_ok -eq 1 ]]; then
    echo "RESULT: ALL PRESETS OK + DATA BITWISE-CONSISTENT"
    exit 0
fi
echo "RESULT: FAILURES ABOVE"
exit 1
