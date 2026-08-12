#!/usr/bin/env bash
# ============================================================================
# gate_sweep_torch.sh — post-Phase-2 ref-side execution smoke across the Stage 1 ref gates
# (torch ref only). Proves the freshly-installed release-image environment can
# DRIVE every gate shape end-to-end, NOT that ours == ref (the "ours" training
# engine does not exist until Stage 1, so there is nothing to compare against).
#
# What it runs (each capped to --steps, default 5, so the whole sweep finishes
# under --budget-s, default 300s):
#   alignment           forward-align / backward-align  (DP=1, 1 step)   — same shape via this
#                                                          script; true tensor
#                                                          capture is Stage 1's
#                                                          bridge, not here.
#   bitwise-singlecard  multistep-1gpu                  (DP=1, 5 steps)
#   bitwise-multicard   multistep                       (DP=min(2,nGPU), 5 steps)
#   long-horizon        long-train                      (DP=min(2,nGPU), 5 steps)
# Skipped: bitwise-perf perf-bitwise (5 steps measures no meaningful MFU + eats budget);
#          resume resume-gate-20 (an ours-side save/load self-compare — no ref-only
#          meaning) → reported as N/A.
#
# Each gate is a real torchrun of the L0 torch ref script
# (run_16gpu_1000step_pure_mup_mtp.sh) under LOCAL_MODE=1. The launcher takes
# ALL gate parameters from the environment (generic projection contract), so
# the sweep projects the rendered ref product itself (tools/product_env.py)
# and then overrides NUM_STEPS / WORLD_SIZE for the capped smoke shape. We do
# NOT call `harness run` — that is a ref-vs-ours comparison and would fail on
# the not-yet-built engine.
#
# Pass/“can-run” = ref script exits 0 AND prints the expected `[LOSS] step=N`
# marker. Budget overflow marks the remaining gates skipped_budget (NOT fail).
#
# Exit codes: 0 = all runnable gates passed, 1 = a gate failed, 2 = usage error.
# ============================================================================

set -o pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# No pin sourcing here: the sweep runs ref gates and consumes no version pins.
# Pin authority (when needed) is the per-image Dockerfile via _pins.py; --image
# only labels the run + picks baked-data defaults.

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
IMAGE=""
WORKDIR="$PWD"
DATA_AXIS="gsm8k"
JSON=0
BUDGET_S=300
STEPS=5

usage() {
    cat >&2 <<EOF
Usage: $(basename "$0") [--image ngc2501] [--workdir DIR] [--data gsm8k] [--budget-s 300] [--steps 5] [--json]

Ref-side execution smoke across the Stage 1 gates for the torch ref. Runs each gate's
reference pipeline (capped to --steps), asserting it can run end-to-end and
prints the expected [LOSS] step marker. This is NOT a ref-vs-ours correctness
gate (the ours engine does not exist until Stage 1).

  --image     ngc2501 — the single forge_train CUDA-12.8 release image; labels
              the run. Default: ngc2501.
  --workdir   directory containing harness/ (default: \$PWD)
  --data      gsm8k (default; modelbest not yet wired for the sweep)
  --budget-s  total wall-clock budget in seconds (default 300 = ~5 min)
  --steps     per-gate step cap for the multi-step gates (default 5)
  --json      machine-readable JSON output

Exit codes: 0=all runnable gates passed, 1=a gate failed, 2=usage error.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)    IMAGE="$2"; shift 2 ;;
        --workdir)  WORKDIR="$2"; shift 2 ;;
        --data)     DATA_AXIS="$2"; shift 2 ;;
        --budget-s) BUDGET_S="$2"; shift 2 ;;
        --steps)    STEPS="$2"; shift 2 ;;
        --json)     JSON=1; shift ;;
        -h|--help)  usage; exit 0 ;;
        *) echo "error: unknown argument '$1'" >&2; usage; exit 2 ;;
    esac
done

case "$DATA_AXIS" in
    gsm8k) ;;
    *) echo "error: --data gsm8k is the only sweep axis for now" >&2; exit 2 ;;
esac

WORKDIR="$(cd "$WORKDIR" 2>/dev/null && pwd || echo "$WORKDIR")"

# Single release image: forge_train (ngc2501). No manifest sourcing: the sweep
# consumes no version pins.
if [[ -z "$IMAGE" ]]; then
    IMAGE="ngc2501"
fi
case "$IMAGE" in
    ngc2501) ;;
    *) echo "error: --image must be ngc2501" >&2; exit 2 ;;
esac

# Resolve a ref-side script by basename. The env scripts live in harness/env/
# and the ref scripts in the sibling harness/ref/reference/, so the script-dir
# anchor (_SCRIPT_DIR/../ref/reference) is layout-independent — it resolves
# regardless of how WORKDIR was synced. Historical ${WORKDIR}/harness/ref/...
# paths are kept as fallbacks; all-miss returns the historical path so the
# existing "not found" error below still fires.
_resolve_ref() {
    local n="$1" c
    for c in "${_SCRIPT_DIR}/../ref/reference/${n}" \
             "${WORKDIR}/harness/ref/reference/${n}" \
             "${WORKDIR}/ref/reference/${n}"; do
        if [[ -f "$c" ]]; then
            ( cd "$(dirname "$c")" && printf '%s/%s' "$PWD" "$n" )
            return 0
        fi
    done
    printf '%s/harness/ref/reference/%s' "$WORKDIR" "$n"
}
REF_SCRIPT="$(_resolve_ref run_16gpu_1000step_pure_mup_mtp.sh)"
GSM8K_PREP="$(_resolve_ref gsm8k_prepare_torch.py)"
FDD="${FORGE_DATA_DIR:-/opt/forge-data}"
SWEEP_OUT="${WORKDIR}/.artifacts/gate_sweep_torch"
mkdir -p "$SWEEP_OUT"

if [[ ! -f "$REF_SCRIPT" ]]; then
    echo "error: torch ref script not found: $REF_SCRIPT" >&2; exit 2
fi

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
if [[ $JSON -eq 0 && -t 1 ]]; then
    C_OK=$'\033[32m'; C_FAIL=$'\033[31m'; C_WARN=$'\033[33m'
    C_DIM=$'\033[2m'; C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'
else
    C_OK=""; C_FAIL=""; C_WARN=""; C_DIM=""; C_RESET=""; C_BOLD=""
fi
json_escape() {
    local s="$1"; s=${s//\\/\\\\}; s=${s//\"/\\\"}; s=${s//$'\n'/\\n}
    s=${s//$'\r'/\\r}; s=${s//$'\t'/\\t}; printf '%s' "$s"
}

# timeout(1) is coreutils; present on the Linux release images (verify_env_torch
# uses it too). On a host without it we still run, just without the hard cap.
HAVE_TIMEOUT=0
command -v timeout >/dev/null 2>&1 && HAVE_TIMEOUT=1
run_capped() {  # run_capped <secs> <cmd...>
    local t="$1"; shift
    if (( HAVE_TIMEOUT )); then timeout "$t" "$@"; else "$@"; fi
}

# ---------------------------------------------------------------------------
# The ~5-minute wait banner — ALWAYS printed (to stderr) before the first run
# so the user does not mistake a live torchrun for a hang.
# ---------------------------------------------------------------------------
cat >&2 <<EOF
${C_BOLD}Gate sweep (ref-side execution smoke, Stage 1 gates, ≤${STEPS} steps each)${C_RESET}
  This will take up to ~$(( BUDGET_S / 60 )) minutes. Each gate launches a REAL
  torchrun of the torch reference on H100 — first-step CUDA graph / kernel
  autotune is slow. ${C_BOLD}This is NOT stuck${C_RESET}; progress prints per gate below.
  Semantics: proves the environment can RUN every gate shape — NOT ref-vs-ours
  correctness (the training engine you build in Stage 1 does not exist yet).
EOF

# ---------------------------------------------------------------------------
# Resolve gsm8k data → a .bin/.idx prefix the torch ref can read.
#  1. honor an explicit FORGE_DATA_PATH / DATA_PATH the caller exported;
#  2. else reuse a previously-prepared prefix under .artifacts;
#  3. else prepare from the baked parquet (FORGE_DATA_DIR/gsm8k) one time.
# ---------------------------------------------------------------------------
TOKENIZER_MODEL="${TOKENIZER_MODEL:-${FDD}/tokenizer/tokenizer.model}"
DATA_PREFIX="${FORGE_DATA_PATH:-${DATA_PATH:-}}"
PREP_OUT="${WORKDIR}/.artifacts/data/gsm8k_megatron"
if [[ -z "$DATA_PREFIX" ]]; then
    if [[ -f "${PREP_OUT}/gsm8k_train_text_document.bin" && -f "${PREP_OUT}/gsm8k_train_text_document.idx" ]]; then
        DATA_PREFIX="${PREP_OUT}/gsm8k_train_text_document"
    elif [[ -d "${FDD}/gsm8k" && -f "$GSM8K_PREP" && -f "$TOKENIZER_MODEL" ]]; then
        echo "${C_DIM}  preparing gsm8k .bin/.idx from baked parquet (${FDD}/gsm8k)…${C_RESET}" >&2
        mkdir -p "$PREP_OUT"
        if GSM8K_DIR="${FDD}/gsm8k" TOKENIZER_MODEL="$TOKENIZER_MODEL" OUTPUT_DIR="$PREP_OUT" \
               python3 "$GSM8K_PREP" >"${SWEEP_OUT}/gsm8k_prep.log" 2>&1; then
            DATA_PREFIX="${PREP_OUT}/gsm8k_train_text_document"
        else
            echo "error: gsm8k prep failed — see ${SWEEP_OUT}/gsm8k_prep.log" >&2
            exit 1
        fi
    fi
fi
if [[ -z "$DATA_PREFIX" || ! -f "${DATA_PREFIX}.bin" ]]; then
    echo "error: no gsm8k .bin/.idx data available (set FORGE_DATA_PATH=<prefix> or" >&2
    echo "       provide baked data under ${FDD}/gsm8k + tokenizer at $TOKENIZER_MODEL)" >&2
    exit 1
fi
# The torch ref reads --data-path-file as "weight path [weight path ...]".
DATA_PATH_FILE="${SWEEP_OUT}/data_path.txt"
printf '1.0 %s\n' "$DATA_PREFIX" > "$DATA_PATH_FILE"

# Visible GPU count → cap DP world size for the multi-GPU gates.
N_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -c . || echo 0)"
(( N_GPU < 1 )) && N_GPU=1

# ---------------------------------------------------------------------------
# Gate table (parallel indexed arrays; cheapest-first so the budget is spent
# on the informative big gates last). ref_ws is the preset DP world size; the
# sweep caps it at N_GPU. exp_step is the [LOSS] step marker we assert.
# ---------------------------------------------------------------------------
G_MILE=(alignment     bitwise-singlecard  bitwise-multicard  long-horizon)
G_GATE=(forward-align multistep-1gpu      multistep          long-train)
G_REFWS=(1            1               2          2)
G_STEPS=(1            "$STEPS"        "$STEPS"   "$STEPS")   # align is a 1-step gate
# shellcheck disable=SC2034  # parallel note array kept for readability/future use
G_NOTE=("forward/backward-align shape (DP=1, 1 step)" \
        "single-GPU multi-step" \
        "multi-GPU DP multi-step" \
        "production-shape trajectory (capped)")

# Result arrays (by index).
R_STATUS=(); R_DETAIL=(); R_ELAPSED=(); R_WS=()

START="$SECONDS"
any_fail=0
for k in "${!G_GATE[@]}"; do
    gate="${G_GATE[$k]}"
    steps="${G_STEPS[$k]}"
    ws="${G_REFWS[$k]}"; (( ws > N_GPU )) && ws="$N_GPU"
    exp_step="$steps"
    R_WS[$k]="$ws"

    remaining=$(( BUDGET_S - (SECONDS - START) ))
    if (( remaining <= 10 )); then
        R_STATUS[$k]="skipped_budget"; R_ELAPSED[$k]=0
        R_DETAIL[$k]="skipped — ${BUDGET_S}s budget exhausted before this gate"
        printf '%s[SKIP]%s %-18s %-15s budget exhausted\n' "$C_WARN" "$C_RESET" "${G_MILE[$k]}" "$gate" >&2
        continue
    fi
    # Never let one gate exceed the remaining budget.
    gate_timeout="$remaining"; (( gate_timeout > 180 )) && gate_timeout=180

    log="${SWEEP_OUT}/${gate}.log"
    printf '%s[RUN]%s  %-18s %-15s DP=%s steps=%s (timeout %ss)…\n' \
        "$C_DIM" "$C_RESET" "${G_MILE[$k]}" "$gate" "$ws" "$steps" "$gate_timeout" >&2

    g0="$SECONDS"
    # The launcher consumes the rendered ref product as environment (generic
    # projection). Project it here, THEN export the sweep's own capped shape
    # (NUM_STEPS / WORLD_SIZE) so the smoke caps win over the product values.
    (
        _product="${FORGE_REF_CONFIG_DIR:-$(dirname "$REF_SCRIPT")/../config}/${gate}.toml"
        if [[ ! -f "$_product" ]]; then
            echo "error: gate product not found: $_product" >&2
            echo "       (render it first with tools/render_gate_configs.py)" >&2
            exit 3
        fi
        eval "$(python3 "$(dirname "$REF_SCRIPT")/../../tools/product_env.py" "$_product")" || exit 3
        export FORGE_GATE="$gate" \
            LOCAL_MODE=1 \
            NUM_STEPS="$steps" \
            WORLD_SIZE="$ws" \
            GPUS_PER_NODE="$ws" \
            OUT_DIR="${SWEEP_OUT}/${gate}" \
            TOKENIZER_MODEL="$TOKENIZER_MODEL" \
            DATA_PATH_FILE="$DATA_PATH_FILE"
        run_capped "$gate_timeout" bash "$REF_SCRIPT"
    ) >"$log" 2>&1
    rc=$?
    elapsed=$(( SECONDS - g0 ))
    R_ELAPSED[$k]="$elapsed"

    if (( rc == 124 )); then
        R_STATUS[$k]="fail"; any_fail=1
        R_DETAIL[$k]="timed out after ${gate_timeout}s (see ${log})"
        printf '%s[FAIL]%s %-18s %-15s timeout %ss\n' "$C_FAIL" "$C_RESET" "${G_MILE[$k]}" "$gate" "$gate_timeout" >&2
    elif (( rc != 0 )); then
        R_STATUS[$k]="fail"; any_fail=1
        tailmsg="$(tail -c 240 "$log" 2>/dev/null | tr '\n' ' ')"
        R_DETAIL[$k]="ref script exited ${rc}: ${tailmsg}"
        printf '%s[FAIL]%s %-18s %-15s rc=%s (see %s)\n' "$C_FAIL" "$C_RESET" "${G_MILE[$k]}" "$gate" "$rc" "$log" >&2
    elif grep -qE "^\[LOSS\] step=${exp_step}[[:space:]]" "$log"; then
        R_STATUS[$k]="pass"
        R_DETAIL[$k]="ran ${steps} step(s) at DP=${ws} in ${elapsed}s; [LOSS] step=${exp_step} seen"
        printf '%s[OK]%s   %-18s %-15s %ss\n' "$C_OK" "$C_RESET" "${G_MILE[$k]}" "$gate" "$elapsed" >&2
    else
        R_STATUS[$k]="fail"; any_fail=1
        R_DETAIL[$k]="exited 0 but no '[LOSS] step=${exp_step}' marker (see ${log}; dataloader likely produced 0 batches)"
        printf '%s[FAIL]%s %-18s %-15s no step=%s marker\n' "$C_FAIL" "$C_RESET" "${G_MILE[$k]}" "$gate" "$exp_step" >&2
    fi
done
TOTAL_ELAPSED=$(( SECONDS - START ))

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
n_pass=0; n_run=0
for k in "${!G_GATE[@]}"; do
    [[ "${R_STATUS[$k]}" == "pass" ]] && n_pass=$(( n_pass + 1 ))
    [[ "${R_STATUS[$k]}" == "pass" || "${R_STATUS[$k]}" == "fail" ]] && n_run=$(( n_run + 1 ))
done

if (( JSON )); then
    printf '{\n'
    printf '  "kind": "ref-side-execution-smoke",\n'
    printf '  "backend": "torch",\n'
    printf '  "image": "%s",\n' "${IMAGE:-unknown}"
    printf '  "data": "%s",\n' "$DATA_AXIS"
    printf '  "budget_s": %s,\n' "$BUDGET_S"
    printf '  "steps_cap": %s,\n' "$STEPS"
    printf '  "visible_gpus": %s,\n' "$N_GPU"
    printf '  "total_elapsed_s": %s,\n' "$TOTAL_ELAPSED"
    printf '  "verdict": "%s",\n' "$([[ $any_fail -eq 0 ]] && echo ready || echo failed)"
    printf '  "gates": [\n'
    n=${#G_GATE[@]}
    for k in "${!G_GATE[@]}"; do
        sep=","; (( k == n - 1 )) && sep=""
        printf '    {"milestone": "%s", "gate": "%s", "world_size": %s, "steps": %s, "expected_marker": "[LOSS] step=%s", "status": "%s", "elapsed_s": %s, "detail": "%s"}%s\n' \
            "${G_MILE[$k]}" "${G_GATE[$k]}" "${R_WS[$k]:-0}" "${G_STEPS[$k]}" "${G_STEPS[$k]}" \
            "${R_STATUS[$k]}" "${R_ELAPSED[$k]:-0}" "$(json_escape "${R_DETAIL[$k]}")" "$sep"
    done
    printf '  ],\n'
    printf '  "skipped": [\n'
    printf '    {"milestone": "bitwise-perf", "gate": "perf-bitwise", "status": "skipped", "reason": "5-step run measures no meaningful MFU; protects the 5-min budget"},\n'
    printf '    {"milestone": "resume", "gate": "resume-gate-20", "status": "na", "reason": "ours-side save/load self-compare — no ref-only meaning"}\n'
    printf '  ],\n'
    printf '  "summary": "%s/%s runnable gates passed in %ss"\n' "$n_pass" "$n_run" "$TOTAL_ELAPSED"
    printf '}\n'
else
    printf '\n%sGate sweep summary%s  (image=%s, data=%s, GPUs=%s, budget=%ss)\n' \
        "$C_BOLD" "$C_RESET" "${IMAGE:-unknown}" "$DATA_AXIS" "$N_GPU" "$BUDGET_S"
    printf '%s----------------------------------------------------------------%s\n' "$C_DIM" "$C_RESET"
    for k in "${!G_GATE[@]}"; do
        case "${R_STATUS[$k]}" in
            pass)           mark="${C_OK}[OK]${C_RESET}  " ;;
            fail)           mark="${C_FAIL}[FAIL]${C_RESET}" ;;
            skipped_budget) mark="${C_WARN}[SKIP]${C_RESET}" ;;
            *)              mark="[??]  " ;;
        esac
        printf '%s %-18s %-15s %s\n' "$mark" "${G_MILE[$k]}" "${G_GATE[$k]}" "${R_DETAIL[$k]}"
    done
    printf '%s[N/A]%s  %-18s %-15s skipped (no meaningful MFU at %s steps)\n' "$C_DIM" "$C_RESET" "bitwise-perf" "perf-bitwise" "$STEPS"
    printf '%s[N/A]%s  %-18s %-15s ours-side self-compare — no ref-only meaning\n' "$C_DIM" "$C_RESET" "resume" "resume-gate"
    printf '%s----------------------------------------------------------------%s\n' "$C_DIM" "$C_RESET"
    if (( any_fail == 0 )); then
        printf '%sSWEEP: ready%s — %s/%s runnable gates ran ref-side OK in %ss.\n' \
            "$C_OK$C_BOLD" "$C_RESET" "$n_pass" "$n_run" "$TOTAL_ELAPSED"
    else
        printf '%sSWEEP: failed%s — %s/%s runnable gates passed (%ss). Inspect logs under %s.\n' \
            "$C_FAIL$C_BOLD" "$C_RESET" "$n_pass" "$n_run" "$TOTAL_ELAPSED" "$SWEEP_OUT"
    fi
fi

exit $(( any_fail == 0 ? 0 : 1 ))
