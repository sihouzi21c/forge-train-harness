#!/usr/bin/env bash
# ============================================================================
# verify_env_torch.sh — Phase 2 Script 2 for the **torch** ref backend.
#
# Returns exit 0 iff the torch ref (run_16gpu_1000step_pure_mup_mtp.sh +
# train_pure_mup_mtp.py) can run end-to-end on this host with the active
# `[data]` axis.
#
# Pin authority: the release Dockerfile (the SSOT), parsed by
# harness/env/_pins.py from --image:
#   ngc2501  -> repo-root Dockerfile   (forge_train, CUDA 12.8)
# There are no versions.*.env manifests. A package the Dockerfile leaves to its
# base image (flash_attn/triton, transformer_engine) is unpinned -> the version
# check degrades to "present is enough".
#
# Modes (--mode):
#   imports       ~10s   Python imports + version-pin checks.
#                        Used by Step 2 (install loop) every iteration.
#   parse         ~15s   imports + invoke ref script with PRINT_GATE_METADATA=1
#                        for every gate; assert gate_metadata.json emits.
#                        Validates ref-script syntax + preset table.
#   milestones    ~30s   parse + per-gate metadata-content asserts (gate name,
#                        world size, num_steps) + data-axis env-var contract.
#                        DEFAULT. Run once at end of install loop.
#   install-smoke ~3m    imports + env-var contract + pack-checks + a real
#                        ref-side run of forward-align (1 GPU, no NCCL) and
#                        multistep (DP=2, NCCL) via harness.cli on the ACTIVE
#                        model/data. The Phase 2 install-loop terminator: proves
#                        the env can run both the single-card and collective
#                        paths. Skips the smoke if any prior check failed;
#                        multistep needs ≥2 GPUs.
#
# Exit codes: 0 = ready, 1 = failed, 2 = usage error
# ============================================================================

set -o pipefail

# ---------------------------------------------------------------------------
# Resolve this script's dir. Pin authority is the per-image Dockerfile, parsed
# by _pins.py once --image is known (below) — there are no versions.*.env files.
# ---------------------------------------------------------------------------
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
MODE="milestones"
WORKDIR="$PWD"
DATA_AXIS=""        # gsm8k | modelbest | ultra_fineweb | (auto-detect from harness/config/data.toml)
IMAGE=""            # ngc2501 (default; the single forge_train CUDA-12.8 image)
JSON=0

usage() {
    cat >&2 <<EOF
Usage: $(basename "$0") [--mode imports|parse|milestones|install-smoke] [--image ngc2501] [--workdir DIR] [--data gsm8k|modelbest|ultra_fineweb] [--json]

  --mode      imports        python imports + version-pin checks
              parse          + invoke each gate with PRINT_GATE_METADATA
              milestones     + assert per-gate metadata + data env-var contract  (default)
              install-smoke  + pack-checks + ref-side forward-align (1 GPU) and
                             multistep (DP=2, NCCL) via harness.cli (≥2 GPUs)
  --image     ngc2501 — the single forge_train CUDA-12.8 release image whose
              Dockerfile pins to enforce (parsed by _pins.py). Default: ngc2501.
  --workdir   directory containing harness/ (default: \$PWD)
  --data      gsm8k | modelbest | ultra_fineweb (default: read harness/config/data.toml)
  --json      machine-readable JSON output

Exit codes: 0=ready, 1=failed, 2=usage error.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)    MODE="$2"; shift 2 ;;
        --image)   IMAGE="$2"; shift 2 ;;
        --workdir) WORKDIR="$2"; shift 2 ;;
        --data)    DATA_AXIS="$2"; shift 2 ;;
        --json)    JSON=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown argument '$1'" >&2; usage; exit 2 ;;
    esac
done

case "$MODE" in
    imports|parse|milestones|install-smoke) ;;
    *) echo "error: --mode must be imports|parse|milestones|install-smoke" >&2; exit 2 ;;
esac

WORKDIR="$(cd "$WORKDIR" 2>/dev/null && pwd || echo "$WORKDIR")"

# Resolve where the ACTIVE per-loop axis configs (eval/data/ref/...) live.
# Candidates, in priority order:
#   1. $FORGE_CONFIG_DIR        — the per-loop SSOT the wrapper/web export
#                                 (<loop>/config/); set when verify runs under
#                                 the loop env.
#   2. <WORKDIR>/config         — the lease mirror of (1), read the same way
#                                 the dispatcher reads config/eval.toml; what
#                                 an agent reaches via cwd-relative config/.
#   3. <WORKDIR>/harness/config — the DEPRECATED shared foot-gun. provision now
#                                 skips those axis basenames so it no longer
#                                 holds active values; kept only as a last
#                                 resort for older layouts.
# Pick the first candidate that actually carries the active eval.toml (the 7
# axes are always staged together, so eval.toml presence marks the real dir).
# A bare local FORGE_CONFIG_DIR on a remote verify simply fails the -f test and
# falls through, so listing it first is safe across local/ssh.
ACTIVE_CFG_DIR=""
for _cfg_cand in "${FORGE_CONFIG_DIR:-}" "${WORKDIR}/config" "${WORKDIR}/harness/config"; do
    if [[ -n "$_cfg_cand" && -f "${_cfg_cand}/eval.toml" ]]; then
        ACTIVE_CFG_DIR="$_cfg_cand"
        break
    fi
done
# Nothing carried an active eval.toml: fall back to the workspace mirror so the
# data/ref auto-detect guards below behave exactly as before (read-if-present).
[[ -n "$ACTIVE_CFG_DIR" ]] || ACTIVE_CFG_DIR="${WORKDIR}/config"
unset _cfg_cand

# Auto-detect data axis if not given. We grep the active data.toml's
# conf_path for the corpus marker (ultra_fineweb / modelbest); default = gsm8k.
if [[ -z "$DATA_AXIS" ]]; then
    if [[ -r "${ACTIVE_CFG_DIR}/data.toml" ]]; then
        if grep -q 'ultra_fineweb' "${ACTIVE_CFG_DIR}/data.toml" 2>/dev/null; then
            DATA_AXIS="ultra_fineweb"
        elif grep -q 'modelbest' "${ACTIVE_CFG_DIR}/data.toml" 2>/dev/null; then
            DATA_AXIS="modelbest"
        else
            DATA_AXIS="gsm8k"
        fi
    else
        DATA_AXIS="gsm8k"
    fi
fi
case "$DATA_AXIS" in
    gsm8k|modelbest|ultra_fineweb) ;;
    *) echo "error: --data must be gsm8k|modelbest|ultra_fineweb" >&2; exit 2 ;;
esac

# ---------------------------------------------------------------------------
# Single release image: forge_train (CUDA 12.8). Its version pins are parsed
# from the repo-root Dockerfile (the SSOT) by _pins.py. --image defaults to
# ngc2501 when absent.
# ---------------------------------------------------------------------------
if [[ -z "$IMAGE" ]]; then
    IMAGE="ngc2501"
fi
case "$IMAGE" in
    ngc2501)
        _pins="$(python3 "${_SCRIPT_DIR}/_pins.py" --image "$IMAGE" --env-dir "${_SCRIPT_DIR}" 2>/dev/null)"
        if [[ -n "$_pins" ]]; then
            # shellcheck disable=SC1090
            eval "$_pins"
        else
            echo "warning: could not parse pins from the ${IMAGE} Dockerfile; running present-is-enough" >&2
        fi
        ;;
    *) echo "error: --image must be ngc2501" >&2; exit 2 ;;
esac

# Resolve a ref-side script by basename. The env scripts (this file) live in
# harness/env/ and the ref scripts in the sibling harness/ref/reference/, so the
# script-dir anchor (_SCRIPT_DIR/../ref/reference) is layout-independent — it
# resolves regardless of how WORKDIR was synced onto a remote. We keep the
# historical ${WORKDIR}/harness/ref/reference path (and the WORKDIR==harness/
# form) as fallbacks for backward compatibility; if all candidates miss we
# return the historical path so the existing "not found" error still fires.
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
# shellcheck disable=SC2034  # resolved for parity/validation; not directly consumed
PY_ENTRY="$(_resolve_ref train_pure_mup_mtp.py)"

# Workspace for gate metadata dumps — under <workspace>/tmp (the loop's
# mounted volume), NOT /tmp (docker overlay / 50 GiB ephemeral-storage cap).
mkdir -p "${WORKDIR}/tmp"
TMP_DUMP_ROOT="$(mktemp -d "${WORKDIR}/tmp/fttv2.XXXXXX")"
cleanup() { rm -rf "$TMP_DUMP_ROOT" 2>/dev/null; }
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Pin values. The *_PIN names below were set by _pins.py (eval'd above) from the
# image Dockerfile; an EMPTY value means "the Dockerfile does not pin this (it
# is base-image-owned) — present is enough", and the checks skip the version
# compare. Host/base facts the Dockerfile cannot pin (python floor, compute cap,
# cuDNN/NCCL bundled by the base) keep inline values and are host gates or
# warn-only, never image pins. NB `${VAR-}` (no colon) preserves an explicit
# empty from _pins.py instead of substituting a default.
# ---------------------------------------------------------------------------
PYTHON_MIN="3.11"                 # Dockerfile base-contract floor (>= 3.11)
PYTHON_PIN_VAL="3.12"             # informational (NGC 25.01 / deadsnakes 3.12)
COMPUTE_CAP="${GPU_COMPUTE_CAP_PIN:-9.0}"   # host gate: H100 sm_90
CUDNN_PIN_VAL=""                  # base-image-owned (report-only)
NCCL_PIN_VAL=""                   # base-image-owned (report-only)

CUDA_PIN_VAL="${CUDA_PIN-}"                       # torch CUDA prefix (12.4 / 12)
TORCH_PIN_VAL="${TORCH_PIN-}"
TORCH_PIN_BASE="${TORCH_PIN_VAL%%+*}"
TORCH_PIN_MODE_VAL="${TORCH_PIN_MODE-}"          # base | floor | (empty)
TE_PIN_VAL="${TRANSFORMER_ENGINE_PIN-}"
TRITON_PIN_VAL="${TRITON_PIN-}"
FLASH_ATTN_PIN_VAL="${FLASH_ATTN_PIN-}"
NUMPY_PIN_VAL="${NUMPY_PIN-}"
PANDAS_PIN_VAL="${PANDAS_PIN-}"
PYARROW_PIN_VAL="${PYARROW_PIN-}"
SENTENCEPIECE_PIN_VAL="${SENTENCEPIECE_PIN-}"
CUTLASS_DSL_PIN_VAL="${NVIDIA_CUTLASS_DSL_PIN-}"
CUDA_BINDINGS_PIN_VAL="${CUDA_BINDINGS_PIN-}"
MODELBEST_SDK_PIN_VAL="${MODELBEST_SDK_PIN-}"
MODELBEST_SDK_PYPI="https://pypi.org/project/modelbest-sdk/"
TOKENIZER_ID="openbmb/MiniCPM4-0.5B"

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
if [[ $JSON -eq 0 && -t 1 ]]; then
    C_OK=$'\033[32m'; C_FAIL=$'\033[31m'; C_WARN=$'\033[33m'
    C_DIM=$'\033[2m'; C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'
else
    C_OK=""; C_FAIL=""; C_WARN=""; C_DIM=""; C_RESET=""; C_BOLD=""
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
json_escape() {
    local s="$1"
    s=${s//\\/\\\\}
    s=${s//\"/\\\"}
    s=${s//$'\n'/\\n}
    s=${s//$'\r'/\\r}
    s=${s//$'\t'/\\t}
    printf '%s' "$s"
}

CHK_ID=(); CHK_NAME=(); CHK_STATUS=(); CHK_DETAIL=(); CHK_HINT=()
record() {
    CHK_ID+=("$1"); CHK_NAME+=("$2"); CHK_STATUS+=("$3"); CHK_DETAIL+=("$4"); CHK_HINT+=("${5:-}")
}

# ===========================================================================
# Layer 1 — imports + version pins
# ===========================================================================

# 1. python version
py_ver="$(python3 -c 'import sys; print("%d.%d.%d"%sys.version_info[:3])' 2>/dev/null)"
if [[ -z "$py_ver" ]]; then
    record 1 python_version fail "python3 not found" \
        "Install Python ${PYTHON_PIN_VAL} (floor ${PYTHON_MIN})"
else
    py_major="${py_ver%%.*}"
    py_minor="${py_ver#*.}"; py_minor="${py_minor%%.*}"
    min_major="${PYTHON_MIN%%.*}"
    min_minor="${PYTHON_MIN#*.}"
    if (( py_major > min_major )) || ( (( py_major == min_major )) && (( py_minor >= min_minor )) ); then
        record 1 python_version pass "python ${py_ver} (>= ${PYTHON_MIN})" ""
    else
        record 1 python_version fail "python ${py_ver} < ${PYTHON_MIN}" \
            "Provision python ${PYTHON_PIN_VAL} (conda/micromamba)"
    fi
fi

# 2-5. torch + cuda + compute_cap
torch_out="$(python3 - <<'PYEOF' 2>&1
import json, sys
try:
    import torch
except Exception as e:
    print(json.dumps({"ok": False, "step": "import", "err": str(e)}))
    sys.exit(0)
res = {"ok": True, "version": torch.__version__,
       "cuda": torch.version.cuda,
       "cuda_available": torch.cuda.is_available(),
       "compute_cap": None}
if torch.cuda.is_available():
    try:
        cc = torch.cuda.get_device_capability(0)
        res["compute_cap"] = f"{cc[0]}.{cc[1]}"
    except Exception as e:
        res["compute_cap_err"] = str(e)
print(json.dumps(res))
PYEOF
)"
if echo "$torch_out" | grep -q '"ok": false'; then
    record 2 torch_version fail "import torch failed: $(echo "$torch_out" | sed -n 's/.*"err": "\([^"]*\)".*/\1/p')" \
        "Install torch ${TORCH_PIN_VAL} (CUDA ${CUDA_PIN_VAL} build)"
    record 3 torch_cuda fail "skipped — torch not importable" ""
    record 4 cuda_version fail "skipped — torch not importable" ""
    record 5 gpu_compute_cap fail "skipped — torch not importable" ""
else
    tv="$(echo "$torch_out" | sed -n 's/.*"version": "\([^"]*\)".*/\1/p')"
    tc="$(echo "$torch_out" | sed -n 's/.*"cuda": "\([^"]*\)".*/\1/p')"
    ta="$(echo "$torch_out" | sed -n 's/.*"cuda_available": \([a-z]*\).*/\1/p')"
    cc="$(echo "$torch_out" | sed -n 's/.*"compute_cap": "\([^"]*\)".*/\1/p')"
    # Pin comparison honors TORCH_PIN_MODE (set by _pins.py from the Dockerfile):
    #   empty pin — Dockerfile parse failed / no image → present is enough (warn).
    #   base      — base form (strip "+<label>") must equal the pin base.
    #   floor     — base major.minor must be >= the pin's. Used by ngc2501, whose
    #               digest-pinned NGC base owns the exact heavy-stack labels and
    #               the Dockerfile only floors torch >= 2.3.
    tv_base="${tv%%+*}"
    if [[ -z "$TORCH_PIN_VAL" ]]; then
        torch_ver_status="warn"; torch_ver_detail="torch ${tv} (no Dockerfile pin parsed — present-is-enough)"
    elif [[ "$tv" == "$TORCH_PIN_VAL" || "$tv_base" == "$TORCH_PIN_BASE" ]]; then
        torch_ver_status="pass"; torch_ver_detail="torch ${tv} matches pin (${TORCH_PIN_VAL})"
    elif [[ "$TORCH_PIN_MODE_VAL" == "floor" ]]; then
        _tvmm="${tv_base%%[!0-9.]*}"; _tvmaj="${_tvmm%%.*}"; _tvmin="${_tvmm#*.}"; _tvmin="${_tvmin%%.*}"
        _pnmaj="${TORCH_PIN_BASE%%.*}"; _pnmin="${TORCH_PIN_BASE#*.}"; _pnmin="${_pnmin%%.*}"
        if [[ "$_tvmaj" =~ ^[0-9]+$ && "$_tvmin" =~ ^[0-9]+$ ]] \
           && { (( _tvmaj > _pnmaj )) || { (( _tvmaj == _pnmaj )) && (( _tvmin >= _pnmin )); }; }; then
            torch_ver_status="pass"; torch_ver_detail="torch ${tv} (>= floor ${TORCH_PIN_BASE})"
        else
            torch_ver_status="fail"; torch_ver_detail="torch ${tv} < floor ${TORCH_PIN_BASE}"
        fi
    else
        torch_ver_status="fail"; torch_ver_detail="torch ${tv} != pin ${TORCH_PIN_VAL}"
    fi
    record 2 torch_version "$torch_ver_status" "$torch_ver_detail" \
        "Pin is torch ${TORCH_PIN_VAL}; a different torch changes numerics the gates compare against"

    if [[ "$ta" == "true" ]]; then
        record 3 torch_cuda pass "torch.cuda.is_available() = True (cuda=${tc})" ""
        if [[ "$tc" == "${CUDA_PIN_VAL}"* ]]; then
            record 4 cuda_version pass "torch built against CUDA ${tc} (pin: ${CUDA_PIN_VAL})" ""
        else
            record 4 cuda_version fail "torch built against CUDA ${tc}, pin is ${CUDA_PIN_VAL}" \
                "Reinstall torch built against CUDA ${CUDA_PIN_VAL} (mismatch breaks TE/flash_attn ABI)"
        fi
        if [[ "$cc" == "$COMPUTE_CAP" ]]; then
            record 5 gpu_compute_cap pass "GPU compute_cap=${cc} (H100)" ""
        else
            record 5 gpu_compute_cap fail "GPU compute_cap=${cc}, need ${COMPUTE_CAP}" \
                "Reproduction requires H100 (sm_90); FA4 + AOT GEMM kernels need WGMMA/TMA"
        fi
    else
        record 3 torch_cuda fail "torch.cuda.is_available() = False" \
            "Wrong torch wheel (CPU-only) or no GPU visible; install the CUDA ${CUDA_PIN_VAL} build"
        record 4 cuda_version fail "skipped — torch.cuda unavailable" ""
        record 5 gpu_compute_cap fail "skipped — torch.cuda unavailable" ""
    fi
fi

# 6. cuDNN + NCCL (reported by torch; ABI-bound to the CUDA build)
cudnn_nccl="$(python3 - <<'PYEOF' 2>/dev/null
import torch
try:
    cudnn = torch.backends.cudnn.version()
except Exception:
    cudnn = None
try:
    nccl = ".".join(map(str, torch.cuda.nccl.version()))
except Exception:
    nccl = None
print(f"{cudnn}|{nccl}")
PYEOF
)"
cudnn_raw="${cudnn_nccl%%|*}"; nccl_have="${cudnn_nccl##*|}"
# torch reports cuDNN as an int like 91301 → 9.13.1
if [[ "$cudnn_raw" =~ ^[0-9]+$ ]]; then
    cudnn_have="$(( cudnn_raw / 10000 )).$(( (cudnn_raw / 100) % 100 )).$(( cudnn_raw % 100 ))"
else
    cudnn_have=""
fi
if [[ -z "$cudnn_have" || -z "$nccl_have" || "$nccl_have" == "None" ]]; then
    record 6 cudnn_nccl warn "could not read cuDNN/NCCL from torch" \
        "cuDNN/NCCL ship bundled with the torch build; absence usually means a CPU-only wheel"
elif [[ -z "$CUDNN_PIN_VAL" && -z "$NCCL_PIN_VAL" ]]; then
    record 6 cudnn_nccl pass "cuDNN ${cudnn_have} + NCCL ${nccl_have} (base-owned, no pin)" ""
elif [[ "$cudnn_have" == "$CUDNN_PIN_VAL" && "$nccl_have" == "$NCCL_PIN_VAL" ]]; then
    record 6 cudnn_nccl pass "cuDNN ${cudnn_have} + NCCL ${nccl_have}" ""
else
    record 6 cudnn_nccl warn "cuDNN ${cudnn_have} / NCCL ${nccl_have} (pins: ${CUDNN_PIN_VAL} / ${NCCL_PIN_VAL})" \
        "Drift here usually means a different torch build than the pin"
fi

# 7. transformer_engine — WARN-only for the torch ref. The pure-torch L0 ref
#    (train_pure_mup_mtp.py / model_pure_mup_mtp.py) imports flash_attn, NOT
#    transformer_engine, so TE is not load-bearing here. It is only needed by
#    the Megatron/HF pretrain entries. Absence is therefore informational, not a
#    verdict failure.
te_v="$(python3 -c 'import transformer_engine as te; print(te.__version__)' 2>/dev/null)"
if [[ -z "$te_v" ]]; then
    record 7 transformer_engine warn "transformer_engine not importable (not required by the torch ref)" \
        "Only needed by Megatron/HF pretrain entries + some Stage-2 ops; the torch ref uses flash_attn instead."
elif [[ -z "$TE_PIN_VAL" ]]; then
    record 7 transformer_engine pass "TE ${te_v} present (no pin for this image)" ""
elif [[ "$te_v" == "$TE_PIN_VAL" ]]; then
    record 7 transformer_engine pass "TE ${te_v} matches pin" ""
else
    record 7 transformer_engine warn "TE ${te_v}, pin is ${TE_PIN_VAL}" \
        "Mismatch may cause RMSNorm/fused-attn ABI drift; align to ${TE_PIN_VAL}"
fi

# 8. triton (bundled with the torch wheel)
triton_v="$(python3 -c 'import triton; print(triton.__version__)' 2>/dev/null)"
if [[ -z "$triton_v" ]]; then
    record 8 triton fail "import triton failed" \
        "Triton ships with the torch wheel; reinstall torch (pin: ${TORCH_PIN_VAL:-base-owned})"
elif [[ -z "$TRITON_PIN_VAL" ]]; then
    record 8 triton pass "triton ${triton_v} present (base-owned, no pin)" ""
elif [[ "$triton_v" == "$TRITON_PIN_VAL" ]]; then
    record 8 triton pass "triton ${triton_v} matches pin" ""
else
    record 8 triton warn "triton ${triton_v}, pin is ${TRITON_PIN_VAL}" \
        "Triton is bundled with torch; drift means a different torch build"
fi

# 9. flash_attn (HARD for torch ref: train_pure_mup_mtp.py imports flash_attn_func)
fa_v="$(python3 -c 'import flash_attn; print(flash_attn.__version__)' 2>/dev/null)"
if [[ -z "$fa_v" ]]; then
    record 9 flash_attn fail "import flash_attn failed" \
        "flash_attn is HARD for the torch ref (train_pure_mup_mtp.py imports flash_attn_func). It ships in the forge_train (ngc2501) base image."
elif [[ -z "$FLASH_ATTN_PIN_VAL" ]]; then
    record 9 flash_attn pass "flash_attn ${fa_v} present (base-owned, no pin)" ""
elif [[ "$fa_v" == "$FLASH_ATTN_PIN_VAL" ]]; then
    record 9 flash_attn pass "flash_attn ${fa_v} matches pin" ""
else
    record 9 flash_attn warn "flash_attn ${fa_v}, pin is ${FLASH_ATTN_PIN_VAL}" \
        "torch ref was validated against ${FLASH_ATTN_PIN_VAL}; ABI mismatch with this torch build possible"
fi

# 10. numpy<2 ABI pin
np_v="$(python3 -c 'import numpy; print(numpy.__version__)' 2>/dev/null)"
if [[ -z "$np_v" ]]; then
    record 10 numpy fail "import numpy failed" "Install: pip install 'numpy==${NUMPY_PIN_VAL:-1.26.4}'"
elif [[ -n "$NUMPY_PIN_VAL" && "$np_v" == "$NUMPY_PIN_VAL" ]]; then
    record 10 numpy pass "numpy ${np_v} matches pin" ""
elif [[ "$np_v" == 1.* ]]; then
    if [[ -z "$NUMPY_PIN_VAL" ]]; then
        record 10 numpy pass "numpy ${np_v} (no pin; numpy<2 ABI satisfied)" ""
    else
        record 10 numpy warn "numpy ${np_v} (pin ${NUMPY_PIN_VAL}; numpy<2 ABI still satisfied)" \
            "Align to ${NUMPY_PIN_VAL} for bit-exact reproducibility"
    fi
else
    record 10 numpy fail "numpy ${np_v} (numpy 2.x breaks the bundled-extension ABI)" \
        "Reinstall: pip install --force-reinstall 'numpy==${NUMPY_PIN_VAL:-1.26.4}'"
fi

# 11. Core data-path packages. sentencepiece is load-bearing (tokenizer).
#     pandas/pyarrow are only used by the one-shot gsm8k parquet → bin/idx
#     prep; they arrive transitively via `datasets` and carry no pin
#     (PANDAS_PIN/PYARROW_PIN empty) — an empty pin means "present is enough",
#     so we skip the version comparison for those.
declare -A DATA_PINS=( [pandas]="$PANDAS_PIN_VAL" [pyarrow]="$PYARROW_PIN_VAL" [sentencepiece]="$SENTENCEPIECE_PIN_VAL" )
data_fail=(); data_warn=()
for mod in pandas pyarrow sentencepiece; do
    mv="$(python3 -c "import ${mod} as m; print(getattr(m,'__version__','?'))" 2>/dev/null)"
    if [[ -z "$mv" ]]; then
        data_fail+=("${mod} (missing)")
    elif [[ -n "${DATA_PINS[$mod]}" && "$mv" != "${DATA_PINS[$mod]}" ]]; then
        data_warn+=("${mod} ${mv}!=${DATA_PINS[$mod]}")
    fi
done
if (( ${#data_fail[@]} > 0 )); then
    record 11 data_core fail "missing: ${data_fail[*]}" \
        "Install: pip install 'pandas==${PANDAS_PIN_VAL:-2.2.2}' 'pyarrow==${PYARROW_PIN_VAL:-24.0.0}' 'sentencepiece==${SENTENCEPIECE_PIN_VAL}'"
elif (( ${#data_warn[@]} > 0 )); then
    record 11 data_core warn "version drift: ${data_warn[*]}" \
        "Pins: pandas ${PANDAS_PIN_VAL:-(transitive)}, pyarrow ${PYARROW_PIN_VAL:-(transitive)}, sentencepiece ${SENTENCEPIECE_PIN_VAL}"
else
    record 11 data_core pass "sentencepiece ${SENTENCEPIECE_PIN_VAL}; pandas/pyarrow present" ""
fi

# 12. CuTeDSL stack (FA4 + custom-GEMM candidate engine dep; torch ref itself
# does not need it, so absence is WARN, not fail).
cutlass_v="$(python3 -c 'import importlib.metadata as m; print(m.version("nvidia-cutlass-dsl"))' 2>/dev/null)"
if [[ -z "$cutlass_v" ]]; then
    record 12 cutlass_dsl warn "nvidia-cutlass-dsl not installed" \
        "Optional for torch ref; required by Stage 2 ops. Install: pip install --no-deps 'nvidia-cutlass-dsl==${CUTLASS_DSL_PIN_VAL:-4.4.2}' 'cuda-bindings==${CUDA_BINDINGS_PIN_VAL:-12.9.6}'"
elif [[ -z "$CUTLASS_DSL_PIN_VAL" ]]; then
    record 12 cutlass_dsl pass "nvidia-cutlass-dsl ${cutlass_v} present (no pin)" ""
elif [[ "$cutlass_v" == "$CUTLASS_DSL_PIN_VAL" ]]; then
    record 12 cutlass_dsl pass "nvidia-cutlass-dsl ${cutlass_v}" ""
else
    record 12 cutlass_dsl warn "nvidia-cutlass-dsl ${cutlass_v}, pin is ${CUTLASS_DSL_PIN_VAL}" \
        "FA4/CuTeDSL kernels validated only at ${CUTLASS_DSL_PIN_VAL}; bump risks AOT-shape regression"
fi

# 13. torchrun binary
if command -v torchrun >/dev/null 2>&1; then
    record 13 torchrun pass "torchrun: $(command -v torchrun)" ""
else
    record 13 torchrun fail "torchrun not in PATH" \
        "torchrun ships with torch; reinstall torch ${TORCH_PIN_VAL} or add its bin dir to PATH"
fi

# 14. system tools required by ref scripts (gcc/ninja/git/rsync/tmux)
sys_missing=()
for t in gcc ninja git rsync tmux; do
    command -v "$t" >/dev/null 2>&1 || sys_missing+=("$t")
done
if (( ${#sys_missing[@]} == 0 )); then
    record 14 system_tools pass "gcc ninja git rsync tmux all in PATH" ""
else
    record 14 system_tools fail "missing: ${sys_missing[*]}" \
        "Install via apt (apt-get install -y ${sys_missing[*]}) or conda"
fi

# 15. modelbest_sdk — required ONLY when data=modelbest
if [[ "$DATA_AXIS" == "modelbest" ]]; then
    # train_pure_mup_mtp.py imports several submodules on the modelbest path.
    ms_out="$(python3 - <<'PYEOF' 2>&1
try:
    import modelbest_sdk
    from modelbest_sdk.dataset.modelbest_dataloader import ModelbestDataloader  # noqa
    from modelbest_sdk.dataset.thrift_wrapper.dataset_context import DatasetContext  # noqa
    import importlib.metadata as m
    print("OK " + m.version("modelbest-sdk"))
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
PYEOF
)"
    if [[ "$ms_out" == OK* ]]; then
        ms_v="${ms_out#OK }"
        if [[ -z "$MODELBEST_SDK_PIN_VAL" ]]; then
            record 15 modelbest_sdk pass "modelbest_sdk ${ms_v} + submodules importable (no pin)" ""
        elif [[ "$ms_v" == "$MODELBEST_SDK_PIN_VAL" ]]; then
            record 15 modelbest_sdk pass "modelbest_sdk ${ms_v} + submodules importable" ""
        else
            record 15 modelbest_sdk warn "modelbest_sdk ${ms_v}, pin is ${MODELBEST_SDK_PIN_VAL}" \
                "Align: pip install 'modelbest-sdk==${MODELBEST_SDK_PIN_VAL}' (${MODELBEST_SDK_PYPI})"
        fi
    else
        record 15 modelbest_sdk fail "$ms_out" \
            "data=modelbest needs modelbest_sdk. Install: pip install 'modelbest-sdk==${MODELBEST_SDK_PIN_VAL:-0.3.1}' (${MODELBEST_SDK_PYPI})"
    fi
fi

# ===========================================================================
# Layer 2 — ref script parse (PRINT_GATE_METADATA per gate)
# ===========================================================================

# Bail early if Layer 1 hard-failed — running ref scripts would just produce
# confusing import errors.
LAYER1_HARDFAIL=0
for i in "${!CHK_ID[@]}"; do
    [[ "${CHK_STATUS[$i]}" == "fail" ]] && LAYER1_HARDFAIL=1 && break
done

# Multi-card bitwise gate is variant-dependent: DP×TP models (8B) use `dptp`,
# DP-only models use `multistep`. Detect from the active eval.toml so 8B does
# not try to run the `multistep` suite it has no [evals.multistep] section for.
MC_GATE=multistep
if [[ -f "${ACTIVE_CFG_DIR}/eval.toml" ]] && grep -qE '^\[evals\.dptp\]' "${ACTIVE_CFG_DIR}/eval.toml"; then
    MC_GATE=dptp
fi

# Gates we walk (one per milestone). Names match the ref scripts' case table.
GATES=(forward-align backward-align multistep-1gpu "$MC_GATE" perf-bitwise resume-gate-20 long-train)

if [[ "$MODE" == "parse" || "$MODE" == "milestones" ]] && (( LAYER1_HARDFAIL == 0 )); then
    if [[ ! -f "$REF_SCRIPT" ]]; then
        record 20 ref_script_exists fail "ref script not found: $REF_SCRIPT" \
            "Run from a workdir containing harness/ref/reference/"
    elif ! bash -n "$REF_SCRIPT" 2>/dev/null; then
        record 20 ref_script_exists fail "ref script has bash syntax errors" \
            "Run 'bash -n $REF_SCRIPT' for details"
    else
        record 20 ref_script_exists pass "ref script syntax OK: $REF_SCRIPT" ""

        for gate in "${GATES[@]}"; do
            dump="${TMP_DUMP_ROOT}/${gate}"
            mkdir -p "$dump"
            chk_id=$((100 + ${#CHK_ID[@]}))
            chk_name="gate_${gate//-/_}"
            err_file="${dump}/stderr.log"
            if FORGE_GATE="$gate" PRINT_GATE_METADATA=1 LOCAL_MODE=1 DUMP_DIR="$dump" \
                WORLD_SIZE=1 GPUS_PER_NODE=1 \
                bash "$REF_SCRIPT" >/dev/null 2>"$err_file"; then
                meta="${dump}/gate_metadata.json"
                if [[ -f "$meta" ]]; then
                    if [[ "$MODE" == "milestones" ]]; then
                        gate_in_json="$(python3 -c "import json,sys; print(json.load(open('$meta')).get('gate',''))" 2>/dev/null)"
                        if [[ "$gate_in_json" == "$gate" ]]; then
                            record "$chk_id" "$chk_name" pass \
                                "gate=${gate}: gate_metadata.json written, gate field matches" ""
                        else
                            record "$chk_id" "$chk_name" fail \
                                "gate=${gate}: gate_metadata.json present but gate field='${gate_in_json}'" \
                                "Ref script preset table likely misrouted; check $REF_SCRIPT case '${gate})'"
                        fi
                    else
                        record "$chk_id" "$chk_name" pass \
                            "gate=${gate}: gate_metadata.json written" ""
                    fi
                else
                    record "$chk_id" "$chk_name" fail \
                        "gate=${gate}: ref script exited 0 but did not write gate_metadata.json" \
                        "Check DUMP_DIR forwarding in $REF_SCRIPT (look for write_gate_metadata call)"
                fi
            else
                err_msg="$(tail -c 400 "$err_file" 2>/dev/null | tr '\n' ' ' | head -c 400)"
                record "$chk_id" "$chk_name" fail \
                    "gate=${gate}: ref script exited nonzero (last 400B: ${err_msg})" \
                    "FORGE_GATE='${gate}' may be unknown to this ref version; see case table in $REF_SCRIPT"
            fi
            rm -f "$err_file"
        done
    fi
fi

# ===========================================================================
# Layer 3 — milestones mode: data-axis env-var contract
# ===========================================================================
#
# The PRINT_GATE_METADATA path EXITS BEFORE the ref script's strict
# `: "${X:?...}"` env-var checks fire. So parse mode passes even when the
# env contract is incomplete. milestones mode does the missing check
# ourselves so Phase 2 ↔ Phase 3 handoff is gate-tight.

if [[ "$MODE" == "milestones" || "$MODE" == "install-smoke" ]] && (( LAYER1_HARDFAIL == 0 )); then

    _fdd="${FORGE_DATA_DIR:-/opt/forge-data}"

    # gsm8k baked-prefix convenience (unchanged): point FORGE_DATA_PATH at a
    # prepared bin/idx if one already exists (prep itself is the playbook's job).
    if [[ "$DATA_AXIS" == "gsm8k" && -z "${DATA_PATH:-}" && -z "${FORGE_DATA_PATH:-}" ]]; then
        for _pref in \
            "${WORKDIR}/harness/.artifacts/data/gsm8k_megatron/gsm8k_train_text_document" \
            "${WORKDIR}/.artifacts/data/gsm8k_megatron/gsm8k_train_text_document"; do
            if [[ -f "${_pref}.bin" && -f "${_pref}.idx" ]]; then
                export FORGE_DATA_PATH="$_pref"; break
            fi
        done
    fi

    # ----- Model-aware tokenizer routing + ultra_fineweb packaging checks -----
    # Read the ACTIVE model's tokenizer name from the seeded [ref] config and
    # drive everything through the harness SSOT — config_runtime._resolve_tokenizer
    # (0.5B -> SentencePiece tokenizer.model, 1B -> HF tokenizer.json),
    # ensure_forge_data_env_defaults, and the real hf_stream_dataloader. No
    # hardcoded 0.5B default and no hand-built DATA_PATH, so the gate can't pass
    # a wiring bug the real run would hit. One python pass emits CHECK lines.
    _refcfg="${ACTIVE_CFG_DIR}/ref.toml"
    TOKENIZER_NAME="$TOKENIZER_ID"
    if [[ -r "$_refcfg" ]]; then
        _t="$(sed -n 's/^[[:space:]]*tokenizer[[:space:]]*=[[:space:]]*"\(.*\)".*/\1/p' "$_refcfg" | head -1)"
        [[ -n "$_t" ]] && TOKENIZER_NAME="$_t"
    fi
    _pack_py="${TMP_DUMP_ROOT}/pack_checks.py"
    cat > "$_pack_py" <<'PY'
import os, glob
from pathlib import Path
fdd = os.environ.get("FORGE_DATA_DIR", "/opt/forge-data")
axis = os.environ.get("PACK_DATA_AXIS", "")
name = os.environ.get("PACK_TOK_NAME", "openbmb/MiniCPM4-0.5B")
TAB = "\t"
def emit(i, n, st, msg, fix=""):
    msg = str(msg).replace("\t", " ").replace("\n", " ")
    fix = str(fix).replace("\t", " ").replace("\n", " ")
    print(TAB.join(["CHECK", str(i), n, st, msg, fix]))
from harness.config_runtime import _resolve_tokenizer, ensure_forge_data_env_defaults
NR = Path("/tmp/__verify_no_resources__")
# (30) active model's tokenizer routes + loads with the correct impl
path = _resolve_tokenizer(NR, {"tokenizer": name}, data_root=fdd)
print(TAB.join(["RESOLVEDTOK", path or ""]))
if not path or not os.path.isfile(path):
    emit(30, "tokenizer_resolve", "fail",
         f"tokenizer '{name}' did not resolve to a baked file (got {path!r})",
         "image must bake the tokenizer for this model (Dockerfile BAKE_DATA)")
else:
    from gsm8k_prepare_torch import _load_tokenizer
    want = ("SentencePieceProcessor" if path.endswith(".model")
            else "_HFTokenizerAdapter" if path.endswith(".json") else "")
    try:
        tok = _load_tokenizer(path); impl = type(tok).__name__
        eos = tok.eos_id(); ids = list(tok.encode("你好 hello 123"))
        if (not want or impl == want) and eos is not None and eos >= 0 and ids:
            emit(30, "tokenizer_resolve", "pass",
                 f"{name} -> {os.path.basename(path)} ({impl}, eos={eos}, encode->{len(ids)} ids)")
        else:
            emit(30, "tokenizer_resolve", "fail",
                 f"{name} -> {path}: impl={impl}(want {want}) eos={eos} ids={len(ids)}",
                 "check _load_tokenizer dispatch / tokenizer sidecar files")
    except Exception as e:
        emit(30, "tokenizer_resolve", "fail",
             f"{name} -> {path}: _load_tokenizer raised {type(e).__name__}: {e}", "")
# (32) model-blind guard: 0.5B and 1B must route to DISTINCT files when both baked
a = _resolve_tokenizer(NR, {"tokenizer": "openbmb/MiniCPM4-0.5B"}, data_root=fdd)
b = _resolve_tokenizer(NR, {"tokenizer": "openbmb/MiniCPM5-1B"}, data_root=fdd)
if a and b and os.path.isfile(a) and os.path.isfile(b):
    if a != b:
        emit(32, "tokenizer_route_distinct", "pass",
             f"0.5B->{os.path.basename(a)} / 1B->{os.path.basename(b)} (distinct, model-correct)")
    else:
        emit(32, "tokenizer_route_distinct", "fail",
             f"0.5B and 1B both resolved to {a} (model-blind!)",
             "config_runtime._DOCKER_BAKED_TOKENIZER must map each model to its own tokenizer")
# (31/33) ultra_fineweb: baked data present + hf streaming tokenizes correctly
if axis == "ultra_fineweb":
    ensure_forge_data_env_defaults(fdd)
    ufd = os.environ.get("ULTRA_FINEWEB_DIR", "")
    en = len(glob.glob(os.path.join(ufd, "en", "*.parquet"))) if ufd else 0
    zh = len(glob.glob(os.path.join(ufd, "zh", "*.parquet"))) if ufd else 0
    if ufd and os.path.isdir(ufd) and en > 0 and zh > 0:
        emit(31, "data_axis_ultra_fineweb", "pass",
             f"ULTRA_FINEWEB_DIR={ufd} (en={en} zh={zh} parquet, auto-resolved)")
    else:
        emit(31, "data_axis_ultra_fineweb", "fail",
             f"ufd={ufd!r} en={en} zh={zh}: baked ultra_fineweb missing",
             "image must bake /opt/forge-data/ultra_fineweb/{en,zh}/*.parquet (BAKE_ULTRA_FINEWEB=1)")
    if path and os.path.isfile(path) and en > 0 and zh > 0:
        os.environ["TOKENIZER_MODEL"] = path
        import subprocess
        conf = os.environ.get("PACK_DATA_CONF", "")
        try:
            dp = subprocess.run(["bash", "-c", f"source {conf}; printf '%s' \"$DATA_PATH\""],
                                capture_output=True, text=True,
                                env=dict(os.environ, ULTRA_FINEWEB_DIR=ufd)).stdout.strip()
            import hf_stream_dataloader as H
            parts = dp.split()
            wp = [(float(parts[i]), parts[i + 1]) for i in range(0, len(parts), 2)]
            dl = H.build(wp, dp_rank=0, world_size=1, micro_batch_size=2,
                         seq_length=16, seed=1234, buffer_size=64)
            t = next(iter(dl))["tokens"]
            from gsm8k_prepare_torch import _load_tokenizer as LT
            tk = LT(path); dec = getattr(tk, "_tok", None) or tk
            text = dec.decode(t[0].tolist())
            import torch
            shp = tuple(t.shape)
            if shp == (2, 16) and t.dtype == torch.int64 and text.strip():
                emit(33, "ultrafineweb_hf_tokenize", "pass",
                     f"batch {shp} {t.dtype}; decoded: {text[:36]!r}")
            else:
                emit(33, "ultrafineweb_hf_tokenize", "fail",
                     f"batch {shp} {t.dtype} text={text[:36]!r}",
                     "hf dataloader produced no/garbage tokens")
        except Exception as e:
            emit(33, "ultrafineweb_hf_tokenize", "fail",
                 f"hf dataloader/tokenize raised {type(e).__name__}: {e}",
                 "check DATA_CONF source + ultra_fineweb parquet + tokenizer.json")
PY
    _pack_out="$(PYTHONPATH="${WORKDIR}/harness:${WORKDIR}/harness/ref/reference" \
        FORGE_DATA_DIR="$_fdd" PACK_DATA_AXIS="$DATA_AXIS" PACK_TOK_NAME="$TOKENIZER_NAME" \
        PACK_DATA_CONF="${DATA_CONF:-${WORKDIR}/harness/ref/reference/ultra_fineweb_data_conf.sh}" \
        python3 "$_pack_py" 2>&1)"
    while IFS=$'\t' read -r _tag _a _b _c _d _e; do
        case "$_tag" in
            CHECK) record "$_a" "$_b" "$_c" "$_d" "$_e" ;;
            # RESOLVEDTOK is informational only — the smoke keeps the matched
            # 0.5B tokenizer set above; we do not override it here.
        esac
    done <<< "$_pack_out"

    # ultra_fineweb: export the resolved data env so the Layer-S smoke's ref
    # subprocess inherits it (the ref sources DATA_CONF -> DATA_PATH whose
    # hf:// tokens read local_env=ULTRA_FINEWEB_DIR).
    if [[ "$DATA_AXIS" == "ultra_fineweb" ]]; then
        _ufd="${ULTRA_FINEWEB_DIR:-${_fdd}/ultra_fineweb}"
        [[ -d "$_ufd" ]] && export ULTRA_FINEWEB_DIR="$_ufd"
        [[ -z "${DATA_CONF:-}" ]] && export DATA_CONF="${WORKDIR}/harness/ref/reference/ultra_fineweb_data_conf.sh"
    fi

    # 31 — gsm8k / modelbest data-axis path checks (ultra_fineweb handled above)
    if [[ "$DATA_AXIS" == "gsm8k" ]]; then
        # DATA_PATH must exist and have <prefix>.bin + .idx
        dp="${DATA_PATH:-${FORGE_DATA_PATH:-}}"
        if [[ -z "$dp" ]]; then
            record 31 data_axis_gsm8k fail \
                "data=gsm8k but DATA_PATH/FORGE_DATA_PATH not set" \
                "Run: GSM8K_DIR=... TOKENIZER_MODEL=... python3 ref/reference/gsm8k_prepare_torch.py, then export FORGE_DATA_PATH=<output>/gsm8k_train_text_document"
        elif [[ -f "${dp}.bin" && -f "${dp}.idx" ]]; then
            record 31 data_axis_gsm8k pass \
                "data=gsm8k: ${dp}.bin + ${dp}.idx exist" ""
        else
            record 31 data_axis_gsm8k fail \
                "data=gsm8k: ${dp}.bin or ${dp}.idx missing" \
                "Run: GSM8K_DIR=... TOKENIZER_MODEL=... python3 ref/reference/gsm8k_prepare_torch.py"
        fi
    elif [[ "$DATA_AXIS" == "modelbest" ]]; then
        dpf="${DATA_PATH_FILE:-}"
        dc="${DATA_CONF:-}"
        if [[ -n "$dpf" && -f "$dpf" ]]; then
            record 31 data_axis_modelbest pass \
                "data=modelbest: DATA_PATH_FILE=$dpf exists ($(wc -l <"$dpf") shards)" ""
        elif [[ -n "$dc" && -f "$dc" ]]; then
            record 31 data_axis_modelbest pass \
                "data=modelbest: DATA_CONF=$dc exists (will populate DATA_PATH on source)" ""
        else
            record 31 data_axis_modelbest fail \
                "data=modelbest: neither DATA_PATH_FILE nor DATA_CONF resolves to an existing file" \
                "Set DATA_PATH_FILE=<shard-list>.txt OR DATA_CONF=<conf>.sh per [data].conf_path"
        fi
    fi
fi

# ===========================================================================
# Layer S — install-smoke: ref-side smoke of forward-align (1 GPU, no NCCL)
# and multistep (DP=2, NCCL) via harness.cli on the active config.
# ===========================================================================
#
# Goal: prove the freshly installed env can actually run the torch ref
# end-to-end, without paying the cost of a full milestones sweep. We run
# two preset gates from the ref script's case table:
#   - forward-align (WORLD_SIZE=1): single-card, no NCCL
#   - multistep     (WORLD_SIZE=2): DP=2 NCCL trajectory
# multistep is skipped (with an explicit fail) when fewer than 2 GPUs are visible.
#
# Skipped unless mode == install-smoke AND every prior check is clean —
# a real torchrun on a broken env wastes ~60s and obscures the real failure.

if [[ "$MODE" == "install-smoke" ]]; then
    ANY_FAIL=0
    for i in "${!CHK_ID[@]}"; do
        [[ "${CHK_STATUS[$i]}" == "fail" ]] && ANY_FAIL=1 && break
    done

    if (( ANY_FAIL == 0 )); then
        n_gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -c . || echo 0)"
        # Multi-card gate topology is variant-dependent: multistep is DP=2
        # (2 ranks); dptp is DP×TP (world_size from its gate_config, e.g.
        # 4 = dp2×tp2). Derive the rank count so the smoke launches the right
        # number of GPUs instead of a fixed 2 (which would under-provision dptp).
        MC_NGPU=2
        if [[ "$MC_GATE" == "dptp" ]]; then
            MC_NGPU="$(grep -oE '^world_size_override[[:space:]]*=[[:space:]]*[0-9]+' "${ACTIVE_CFG_DIR}/gate_config/dptp.toml" 2>/dev/null | grep -oE '[0-9]+' | tail -1)"
            MC_NGPU="${MC_NGPU:-4}"
        fi
        MC_GPUSPEC="0"; for ((_g = 1; _g < MC_NGPU; _g++)); do MC_GPUSPEC="${MC_GPUSPEC},${_g}"; done
        record 40 smoke_gate_select pass "visible GPUs=${n_gpu}; running forward-align (1 GPU) + ${MC_GATE} (${MC_NGPU} ranks)" ""
        _SMK="${WORKDIR}/.artifacts/verify_torch_install_smoke"; mkdir -p "$_SMK"
        export PYTHONPATH="${WORKDIR}/harness${PYTHONPATH:+:$PYTHONPATH}"

        # forward-align: single-card single-step fwd+bwd capture, no
        # NCCL. multistep: DP=2 multi-step ref trajectory — the FIRST gate
        # that exercises NCCL communication. Both run the ACTIVE [model]/[ref]
        # via harness.cli, so the model-correct geometry + tokenizer are used
        # (a 1B run trains 1B).
        #
        # Verdict comes from the run's own artifacts (ground truth), not a
        # stdout substring — a prior version matched "ref_steps_observed" which
        # ALSO appears in the failure message, so a stale eval.toml
        # (NUM_STEPS_OVERRIDE=1 vs GATE_WINDOW_END=9) false-passed. smoke_eval.py:
        #   forward-align: ours engine absent here, so the ENV signal is
        #        the REF producing its single-card (no-NCCL) fwd+bwd capture
        #        dump; assert ref_dump__*/<basename>.pt exists and is non-empty.
        #   multistep — the ours engine is agent-authored and absent here, so the gate
        #        status is "failed" by design; the ENV signal is the REF
        #        completing the full window, so assert ref_dump__*/ref_loss.txt
        #        carries [LOSS] step lines covering [GATE_WINDOW_START,_END) read
        #        from the ACTIVE eval.toml (SSOT).
        _smoke_eval="${TMP_DUMP_ROOT}/smoke_eval.py"
        cat > "$_smoke_eval" <<'PY'
import json, sys, glob, os, re
run_dir = sys.argv[1] if len(sys.argv) > 1 else ""
suite = sys.argv[2] if len(sys.argv) > 2 else ""
eval_toml = sys.argv[3] if len(sys.argv) > 3 else ""
def out(st, msg):
    print(st + " " + " ".join(str(msg).split())); raise SystemExit
if not run_dir or not os.path.isfile(os.path.join(run_dir, "result.json")):
    out("FAIL", f"no result.json under run dir {run_dir!r} (gate did not produce a run)")
d = json.load(open(os.path.join(run_dir, "result.json")))
status, summary = d.get("status"), d.get("summary", "")
if suite == "forward-align":
    # forward-align: single-card single-step fwd+bwd CAPTURE, no NCCL. The
    # ref emits a tensor dump (ref_dump__*/<basename>.pt + .graph.json), NOT
    # a [LOSS] trajectory like the multistep gates. The ours engine is absent
    # here (status "failed" by design), so the ENV signal is the REF producing
    # its single-card capture dump.
    dumps = [p for p in sorted(glob.glob(os.path.join(run_dir, "ref_dump__*", "*.pt")))
             if os.path.getsize(p) > 0]
    if dumps:
        out("PASS", "ref produced single-card fwd+bwd capture dump (no NCCL); ours-side absent as expected")
    out("FAIL", f"ref produced no capture dump; status={status}: {summary}")
if suite in ("multistep", "dptp"):
    want = 8
    try:
        seg = re.search(r'\[evals\.%s\.ref_env\]' % re.escape(suite) + r'(.*?)(\n\[|\Z)', open(eval_toml).read(), re.S)
        seg = seg.group(1) if seg else ""
        def g(k):
            m = re.search(r'%s\s*=\s*"(\d+)"' % k, seg); return int(m.group(1)) if m else None
        s, e = g("GATE_WINDOW_START"), g("GATE_WINDOW_END")
        if s is not None and e is not None: want = e - s
    except Exception:
        pass
    losses = (sorted(glob.glob(os.path.join(run_dir, "ref_dump__*", "ref_loss.txt")))
              or sorted(glob.glob(os.path.join(run_dir, "ref_dump__*", "ref_stdout.log"))))
    got = len(re.findall(r'\[LOSS\] step=\d+', open(losses[0]).read())) if losses else 0
    if got >= want:
        out("PASS", f"ref ran full DP=2 trajectory {got}/{want} steps (NCCL OK); ours-side absent as expected")
    if "ref-script trajectory missing steps" in summary:
        out("FAIL", f"ref trajectory incomplete {got}/{want} — check [evals.{suite}.ref_env] NUM_STEPS_OVERRIDE vs GATE_WINDOW_END and NCCL/DP=2 init")
    out("FAIL", f"ref produced {got}/{want} trajectory steps; status={status}: {summary}")
out("FAIL", f"unknown smoke gate {suite!r}")
PY

        # Wall-clock cap for a smoke ref run = the gate's OWN declared REF-side
        # budget. The smoke runs a fresh ref only (ours engine absent → ~1s), so
        # the SSOT is config_runtime.suite_ref_timeout_s ([evals.<gate>].ref_
        # timeout_s → timeout_s → default), NOT a fixed constant and NOT the
        # candidate-side timeout_s. This distinction is load-bearing for the 8B
        # multi-card smoke: its gate is `dptp`, whose ref (Megatron TP under
        # hash capture) declares ref_timeout_s=1200 while timeout_s (candidate
        # side) is only 900 — a 900s cap would SIGTERM the ref early. For the
        # DP-only variants ref_timeout_s is unset and equals timeout_s (0.5B/
        # qwen3=600, 1B=1100). The old hardcoded 400s undercut every variant,
        # SIGTERMing the ref mid-trajectory ("no result.json" → false FAIL).
        # Fallback 1200 = conservative catch-all (>= every known variant) used
        # only if the active eval.toml can't be read.
        _gate_budget_s() {  # $1=gate name -> ref-side wall-clock seconds
            python3 - "$1" "${ACTIVE_CFG_DIR}/eval.toml" <<'PY'
import sys, tomllib
try:
    from harness.config_runtime import suite_ref_timeout_s
    wc = tomllib.load(open(sys.argv[2], "rb"))
    print(int(suite_ref_timeout_s(wc, sys.argv[1])))
except Exception:
    print(1200)
PY
        }

        # $1=suite $2=gpu-spec $3=record-id $4=record-name
        _smoke_one() {
            local suite="$1" gpu="$2" rid="$3" rname="$4"
            local log="${_SMK}/${suite}.log"
            local budget; budget="$(_gate_budget_s "$suite")"
            # FORGE_REF_CACHE=0 forces a FRESH ref run every time: the smoke's
            # purpose is to prove the env can actually run the path, and a cached
            # ref trajectory would skip the (single-card / DP=2 NCCL) ref launch
            # entirely — leaving only the absent ours engine to "run" in ~1s.
            FORGE_REF_CACHE=0 timeout "$budget" python3 -m harness.cli run "$suite" --gpu "$gpu" > "$log" 2>&1 || true
            local run_dir; run_dir="$(grep -oE 'Artifacts: [^[:space:]]+' "$log" | tail -1 | sed 's/^Artifacts: //')"
            local v st msg
            v="$(python3 "$_smoke_eval" "$run_dir" "$suite" "${ACTIVE_CFG_DIR}/eval.toml" 2>&1)"
            st="${v%% *}"; msg="${v#* }"
            if [[ "$st" == "PASS" ]]; then
                record "$rid" "$rname" pass "${suite}: ${msg}" ""
            else
                record "$rid" "$rname" fail "${suite}: ${msg}" "Inspect ${log}${run_dir:+ and }${run_dir}"
            fi
        }

        _smoke_one forward-align 0 41 smoke_fwd_align
        if (( n_gpu >= MC_NGPU )); then
            _smoke_one "$MC_GATE" "$MC_GPUSPEC" 42 smoke_multistep
        else
            record 42 smoke_multistep fail \
                "${MC_GATE} needs >= ${MC_NGPU} GPUs for the multi-card NCCL path (visible=${n_gpu})" \
                "Run install-smoke on a host with >= ${MC_NGPU} GPUs"
        fi
    else
        record 40 smoke_gate_select fail "prior layer failed — smoke skipped to surface the real cause" \
            "Fix the failing checks above, then rerun --mode install-smoke"
    fi
fi

# ===========================================================================
# Verdict
# ===========================================================================
fails=()
for i in "${!CHK_ID[@]}"; do
    if [[ "${CHK_STATUS[$i]}" == "fail" ]]; then
        fails+=("${CHK_NAME[$i]}: ${CHK_DETAIL[$i]}")
    fi
done

if (( ${#fails[@]} == 0 )); then
    VERDICT="ready"; EXIT=0
else
    VERDICT="failed"; EXIT=1
fi

# ===========================================================================
# Output
# ===========================================================================
if (( JSON )); then
    printf '{\n'
    printf '  "verdict": "%s",\n' "$VERDICT"
    printf '  "backend": "torch",\n'
    printf '  "image": "%s",\n' "${IMAGE:-unknown}"
    printf '  "mode": "%s",\n' "$MODE"
    printf '  "data": "%s",\n' "$DATA_AXIS"
    printf '  "workdir": "%s",\n' "$(json_escape "$WORKDIR")"
    printf '  "checks": [\n'
    n=${#CHK_ID[@]}
    for i in "${!CHK_ID[@]}"; do
        hint_field="null"
        [[ -n "${CHK_HINT[$i]}" ]] && hint_field="\"$(json_escape "${CHK_HINT[$i]}")\""
        sep=","; (( i == n - 1 )) && sep=""
        printf '    {"id": "%s", "name": "%s", "status": "%s", "detail": "%s", "fix_hint": %s}%s\n' \
            "$(json_escape "${CHK_ID[$i]}")" \
            "$(json_escape "${CHK_NAME[$i]}")" \
            "${CHK_STATUS[$i]}" \
            "$(json_escape "${CHK_DETAIL[$i]}")" \
            "$hint_field" \
            "$sep"
    done
    printf '  ],\n'
    printf '  "blocking_reasons": ['
    nb=${#fails[@]}
    if (( nb == 0 )); then
        printf ']\n'
    else
        printf '\n'
        for i in "${!fails[@]}"; do
            sep=","; (( i == nb - 1 )) && sep=""
            printf '    "%s"%s\n' "$(json_escape "${fails[$i]}")" "$sep"
        done
        printf '  ]\n'
    fi
    printf '}\n'
else
    printf '%sTorch ref env verification%s  (mode=%s, image=%s, data=%s, workdir=%s)\n' \
        "$C_BOLD" "$C_RESET" "$MODE" "${IMAGE:-unknown}" "$DATA_AXIS" "$WORKDIR"
    printf '%s----------------------------------------------------------------%s\n' \
        "$C_DIM" "$C_RESET"
    for i in "${!CHK_ID[@]}"; do
        case "${CHK_STATUS[$i]}" in
            pass) mark="${C_OK}[OK]${C_RESET}  " ;;
            fail) mark="${C_FAIL}[FAIL]${C_RESET}" ;;
            warn) mark="${C_WARN}[WARN]${C_RESET}" ;;
            *)    mark="[??]  " ;;
        esac
        printf '%s %-22s %s\n' "$mark" "${CHK_NAME[$i]}" "${CHK_DETAIL[$i]}"
        if [[ -n "${CHK_HINT[$i]}" && "${CHK_STATUS[$i]}" != "pass" ]]; then
            printf '       %s-> %s%s\n' "$C_DIM" "${CHK_HINT[$i]}" "$C_RESET"
        fi
    done
    printf '%s----------------------------------------------------------------%s\n' \
        "$C_DIM" "$C_RESET"
    if [[ "$VERDICT" == "ready" ]]; then
        printf '%sVERDICT: ready%s — torch ref can run on this host.\n' \
            "$C_OK$C_BOLD" "$C_RESET"
    else
        printf '%sVERDICT: failed%s\n' "$C_FAIL$C_BOLD" "$C_RESET"
        for r in "${fails[@]}"; do
            printf '  %s* %s%s\n' "$C_FAIL" "$r" "$C_RESET"
        done
    fi
fi

exit $EXIT
