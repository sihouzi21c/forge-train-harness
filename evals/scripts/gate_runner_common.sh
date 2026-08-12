#!/usr/bin/env bash
# Shared gate-runner library (ref-generic-projection plan, Step 1).
#
# Sourced by BOTH side runners — ``ref/run_gate.sh`` and
# ``evals/scripts/run_ours.sh`` (cwd = workspace root) — so the logic the
# two sides genuinely share lives exactly once. Lives under evals/scripts/
# (harness-owned): the agent-writable side must not be able to rewrite the
# gate runners.
#
# Every file here is part of the ref trajectory-cache key
# (dispatcher._ref_execution_dep_files): editing this library correctly
# invalidates cached ref runs.

# resolve_product SIDE GATE [LABEL]
#
# Prints the side's product path after an existence check. ref reads the
# frozen ``ref/config/<gate>.toml``; ours reads the agent-writable
# ``workload/src/config/<gate>.toml``, where a labeled run (e.g.
# profile-snapshot "long-horizon_round3") strips the ``_roundN`` suffix to a
# milestone and prefers the renderer's per-milestone variant product
# ``<gate>@<milestone>.toml`` when it exists on disk.
resolve_product() {
    local side=$1 gate=$2 label=${3:-}
    local product
    if [ "$side" = "ref" ]; then
        product="ref/config/$gate.toml"
    else
        product="workload/src/config/$gate.toml"
        if [ -n "$label" ]; then
            local milestone="${label%_round*}"
            local variant="workload/src/config/$gate@$milestone.toml"
            [ -f "$variant" ] && product="$variant"
        fi
    fi
    if [ ! -f "$product" ]; then
        echo "ERROR: $side product not found: $product" >&2
        return 1
    fi
    printf '%s\n' "$product"
}

# project_env SIDE GATE PRODUCT
#
# Injects the gate environment into the calling shell, in the fixed order
# "product export first, runtime overrides second" — the runtime env is
# eval'd last so it WINS on collision (deployment keys like MASTER_PORT
# stay runtime-authoritative over freeze-filled product values).
#
# BOTH sides use the same fully generic projection (product_env.py, zero
# key knowledge): every [cli] key arrives as its upper-cased name, [env]
# verbatim. Ref launchers consume those names directly — no per-key
# registry / translation layer exists anywhere on the path.
project_env() {
    local side=$1 gate=$2 product=$3
    eval "$(python3 tools/product_env.py "$product")"
    eval "$(python3 evals/scripts/runtime_env.py "$side" "$gate")"
}

# setup_hash_capture SIDE RUN_DIR [LEVEL] [NUM_STEPS]
#
# Fills the global HASH_ARGS array with the capture CLI wire and exports the
# side's dump-file env var. persistent is value-driven from num_steps:
# multi-step trajectory gates capture per-step records (step_<n>. key
# namespace); the single-step alignment capture must stay persistent=False
# (rank<r>.mb0. namespace) or its keys can never intersect the other side's.
#
#   ref : dump = <run_dir>/ref_hash_dump.json, exported as HOOK_OUTPUT_FILE
#         (back-compat env wire for legacy bridges; the CLI args override)
#   ours: dump = <run_dir>/ours_hash_dump.json, exported as
#         FORGE_CAPTURE_OUTPUT_FILE (the alignment engine's tensor-dump path
#         is the SAME file as the hash dump — one artifact)
setup_hash_capture() {
    local side=$1 run_dir=$2 level=${3:-0} num_steps=${4:-1}
    HASH_ARGS=()
    if [ "$level" -le 0 ]; then
        return 0
    fi
    local dump="$run_dir/${side}_hash_dump.json"
    HASH_ARGS=(--hash-capture-level "$level" --hash-output "$dump")
    if [ "$num_steps" -gt 1 ]; then
        HASH_ARGS+=(--persistent)
    fi
    if [ "$side" = "ref" ]; then
        export HOOK_OUTPUT_FILE="$dump"
    else
        export FORGE_CAPTURE_OUTPUT_FILE="$dump"
    fi
}
