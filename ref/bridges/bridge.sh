#!/bin/bash
# ============================================================================
# Unified M1 capture bridge — dispatches to the interposer for both the
# Megatron and Torch backends.
#
# Replaces the per-backend sitecustomize.py / pure_torch_bridge.sh pair
# with a single entry point.  The interposer (interposer.py) is loaded
# as the torchrun Python entry via a sed-patched copy of the ref
# launcher.  No sitecustomize.py, no PYTHONPATH pollution, no TE
# recursive-subprocess bomb.
#
# Env contract (set by the dispatcher via _common.py:run_ref_capture):
#   FORGE_BACKEND           "megatron" | "torch"
#   HOOK_OUTPUT_FILE        absolute path for the M1 tensor dump
#   DUMP_DIR / FORGE_GATE   gate metadata env
#   MEGATRON_ROOT           (megatron only) path to on-remote Megatron checkout
#   ref_env keys            generic product projection (WORLD_SIZE /
#                           NUM_STEPS / ... via tools/product_env.py)
#
# The bridge:
#   1. Determines the ref launcher and the sed needle per backend.
#   2. Creates a temp copy of the launcher with the Python entry
#      replaced by interposer.py.
#   3. Exports BRIDGE_* env vars for the interposer.
#   4. Execs the patched launcher, forwarding all args.
# ============================================================================
set -euo pipefail

BRIDGE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$BRIDGE_DIR/../.." && pwd)
INTERPOSER_PY="$BRIDGE_DIR/interposer.py"

if [[ ! -f "$INTERPOSER_PY" ]]; then
    echo "ERROR: interposer missing: $INTERPOSER_PY" >&2
    exit 2
fi

FORGE_BACKEND="${FORGE_BACKEND:?FORGE_BACKEND must be set (megatron|torch)}"

# ── Shared: create ephemeral patched launcher ────────────────────────
# Under <workspace>/tmp (mounted volume), not /tmp (docker overlay).
mkdir -p "$REPO_ROOT/tmp"
PATCH_DIR=$(mktemp -d "$REPO_ROOT/tmp/patch.XXXXXX")
trap 'rm -rf "$PATCH_DIR"' EXIT

case "$FORGE_BACKEND" in
# ── Megatron ─────────────────────────────────────────────────────────
megatron)
    FORGE_BRIDGE_REF_SCRIPT="${FORGE_BRIDGE_REF_SCRIPT:-train_minicpm4_0.5b_gsm8k.sh}"
    REF_LAUNCHER="$REPO_ROOT/ref/reference/$FORGE_BRIDGE_REF_SCRIPT"
    if [[ ! -f "$REF_LAUNCHER" ]]; then
        echo "ERROR: ref launcher not found: $REF_LAUNCHER" >&2
        exit 2
    fi

    PATCHED_LAUNCHER="$PATCH_DIR/patched_launcher.sh"

    # The Megatron launcher does `cd "$MEGATRON_ROOT"` then runs
    # `torchrun ... pretrain_gpt.py \`.  Replace the inline entry
    # point with the absolute path to the interposer.
    python3 - "$REF_LAUNCHER" "$INTERPOSER_PY" "$PATCHED_LAUNCHER" <<'HEREDOC_PY'
import sys
src, interposer, dst = sys.argv[1:4]
with open(src) as fh:
    body = fh.read()
needle = "pretrain_gpt.py"
count = body.count(needle)
if count < 1:
    raise SystemExit(
        f"refusing to patch: '{needle}' not found in {src}; "
        "ref launcher schema changed?"
    )
patched = body.replace(needle, interposer, 1)
with open(dst, "w") as fh:
    fh.write(patched)
HEREDOC_PY
    chmod +x "$PATCHED_LAUNCHER"

    # pretrain_gpt.py is relative to MEGATRON_ROOT (the launcher cd's there)
    export BRIDGE_ORIGINAL_ENTRY="pretrain_gpt.py"
    export BRIDGE_REF_DIR="${MEGATRON_ROOT:?MEGATRON_ROOT must be set for megatron backend}"
    export BRIDGE_REPO_ROOT="$REPO_ROOT"
    export BRIDGE_BACKEND="megatron"

    # The on-remote Megatron clone asserts NCCL_ALGO is set.
    export NCCL_ALGO="${NCCL_ALGO:-Ring}"
    # Capture needs a single step when invoked standalone; under the gate
    # path the caller-projected product value is inherited and wins.
    export NUM_STEPS="${NUM_STEPS:-1}"

    # DATA_PATH from gsm8k_data_conf.sh is in weighted-shard format
    # ("1.0 /path/to/prefix") consumed by the torch ref's --data-path-file
    # pipeline.  Megatron's launcher expects a plain prefix.  Strip the
    # leading weight token when the value matches "<number> <path>".
    if [[ "${DATA_PATH:-}" =~ ^[0-9]+(\.[0-9]+)?[[:space:]]+(.+)$ ]]; then
        export DATA_PATH="${BASH_REMATCH[2]}"
    fi

    exec bash "$PATCHED_LAUNCHER" "$@"
    ;;

# ── Torch ────────────────────────────────────────────────────────────
torch)
    # Self-insert path (harness_dp): the torch refs (train_pure_mup_mtp.py /
    # train_qwen3_dense.py) now import ``evals.harness_dp`` and own their
    # capture call sites, so the bridge no longer sed-patches PY_ENTRY into the
    # interposer. It just execs the ACTIVE model's L0 launcher, forwarding the
    # dispatcher's ``--hash-capture-level/--hash-output/--persistent`` args (and
    # the HOOK_OUTPUT_FILE env) straight through. The launcher's trailing
    # ``"$PY_ENTRY" "${PY_ARGS[@]}" "$@"`` carries those flags into the ref,
    # which self-parses them via ``harness_dp.add_capture_cli_args`` and installs
    # the same CaptureSession the interposer used to inject. FORGE_BRIDGE_REF_SCRIPT
    # is set by the dispatcher from [ref].ref_script (qwen3 → run_qwen3_dense.sh,
    # minicpm → pure_mup_mtp); the default keeps standalone invocations on 0.5B/1B.
    # The interposer is retained ONLY for the megatron branch above.
    FORGE_BRIDGE_REF_SCRIPT="${FORGE_BRIDGE_REF_SCRIPT:-run_16gpu_1000step_pure_mup_mtp.sh}"
    REF_LAUNCHER="$REPO_ROOT/ref/reference/$FORGE_BRIDGE_REF_SCRIPT"
    if [[ ! -f "$REF_LAUNCHER" ]]; then
        echo "ERROR: ref launcher not found: $REF_LAUNCHER" >&2
        exit 2
    fi

    export LOCAL_MODE="${LOCAL_MODE:-1}"

    exec bash "$REF_LAUNCHER" "$@"
    ;;

*)
    echo "ERROR: unknown FORGE_BACKEND=$FORGE_BACKEND (expected megatron|torch)" >&2
    exit 1
    ;;
esac
