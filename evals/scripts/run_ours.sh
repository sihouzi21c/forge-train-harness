#!/usr/bin/env bash
# Generic ours-side gate runner (thin-dispatcher refactor, step 1).
#
# Contract:  bash evals/scripts/run_ours.sh <gate> <run_dir> [<label> ...]
#            (cwd = workspace root)
#
# One script for every non-production gate — NO branching on the gate name.
# All behavior differences come from the VALUES in the gate's rendered ours
# product (workload/src/config/<gate>.toml) plus the axis-resolved runtime
# env. Lives under evals/scripts/ (harness-owned), NOT workload/ — the
# agent-writable side must not be able to rewrite the gate runner.
#
# Shared runner logic (product lookup + @milestone variant selection, env
# projection, hash-capture wire) lives in
# evals/scripts/gate_runner_common.sh — one library for both sides.
#
# Env layering (later wins, matching the legacy dispatcher order):
#   1. full ours product export (determinism [env] flags must be process
#      env before torch imports; NUM_PROCS for launch_dp.py; etc.)
#   2. runtime env (deployment fallbacks, assets, MASTER_ADDR/PORT with
#      "auto" resolution, FORGE_OURS_ENTRY / FORGE_OURS_LAUNCHER routing)
#
# The engine's sole gate input remains the product itself, passed as the
# single --config arg (read by _gate_entry.GateInputs; env is fallback only).
#
# Artifacts land under <run_dir> by fixed conventions the harness-side
# verdict script reads back:
#   <run_dir>/ours_hash_dump.json    (hash_capture_level > 0 only; the
#     resume engine appends .ref.json/.res.json to this base — two dumps)
# stdout (with the [LOSS] wire lines) is captured by the dispatcher to
#   <run_dir>/ours.log
set -euo pipefail

GATE=${1:?usage: run_ours.sh <gate> <run_dir>}
RUN_DIR=${2:?usage: run_ours.sh <gate> <run_dir>}
LABEL=${3:-}
mkdir -p "$RUN_DIR"

source evals/scripts/gate_runner_common.sh

OURS_PRODUCT=$(resolve_product ours "$GATE" "$LABEL")

project_env ours "$GATE" "$OURS_PRODUCT"

# The engine re-reads its product as $FORGE_OURS_CONFIG_DIR/$FORGE_GATE.toml
# (runtime_config._product_cli), but runtime_env.py above exported the PLAIN
# gate name — wrong when the label selected a @<milestone> variant (the engine
# would silently load the plain product while --config carries the variant).
# Re-derive FORGE_GATE from the SELECTED product so it is the single source
# of truth for both channels.
FORGE_GATE="$(basename "$OURS_PRODUCT" .toml)"
export FORGE_GATE

: "${FORGE_OURS_ENTRY:?suite registry declares no script for $GATE}"
: "${FORGE_OURS_LAUNCHER:?suite registry declares no launcher for $GATE}"

# Value-driven nsys wire: a product carrying nsys_profile=true (the profile
# variants) has rank0 wrapped by nsys in launch_dp.py; the report lands at
# the run-dir convention <run_dir>/profile.nsys-rep the verdict reads back.
if [ "${NSYS_PROFILE:-0}" = "1" ]; then
    export FORGE_NSYS_RANK0_OUTPUT="$RUN_DIR/profile"
fi

# Value-driven hash-capture wire (mirror of the ref side): the product's
# HASH_CAPTURE_LEVEL / NUM_STEPS are already process env here (full product
# export above). Entries that don't consume the wire (eval_train_steps)
# simply ignore it.
setup_hash_capture ours "$RUN_DIR" "${HASH_CAPTURE_LEVEL:-0}" "${NUM_STEPS:-1}"

# Resume scratch convention: Phase-A checkpoint root for the resume engine.
# ABSOLUTE under the workspace's tmp/ (mounted volume — NOT .artifacts and
# NOT /tmp: on a devspace both sit on the docker overlay whose 50 GiB
# ephemeral-storage cap a ~14 GiB checkpoint would blow). Unique per run via
# the run dir's name (<run_dir> = <artifact_dir>/ours, so take the parent).
# Consumed through the GateInputs env fallback; non-resume entries ignore it.
RESUME_SCRATCH_DIR="$PWD/tmp/resume_scratch_$(basename "$(dirname "$RUN_DIR")")"
export RESUME_SCRATCH_DIR

exec python3 "$FORGE_OURS_LAUNCHER" "$FORGE_OURS_ENTRY" \
    --config "$OURS_PRODUCT" \
    ${HASH_ARGS[@]+"${HASH_ARGS[@]}"}
