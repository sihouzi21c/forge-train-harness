#!/usr/bin/env bash
# resume-milestone LAUNCHER — submit the short (default 20-step) cctl
# PyTorchJob for the REF or the OURS side. This is the SUBMITTER (runs once, on
# the box that has `cctl` — the MAC, not the devspace), NOT the per-pod
# command. The per-pod command is picked by the side argument:
#
#   bash evals/scripts/launch_resume.sh ref    ->  resume_train_ref.sh
#   bash evals/scripts/launch_resume.sh ours   ->  resume_train_ours.sh
#
# One launcher, two sides: both jobs share the topology/quota plumbing and the
# recipe TOML (config/train/resume_20.toml — [launch].gpus_per_node and
# [train].iters are the configurable card count / step count; env GPU_PER_NODE
# / ITERS still win). The two per-pod entries are the "ref side model
# submitting training job script" / "ours side model submitting training job
# script" of the resume milestone.
#
# Run it AFTER `bin/harness sync push` has landed this loop's worktree (with
# rendered products) on the devspace, from the mac:
#   REMOTE_WORKDIR=<loop devspace checkout, == @@REMOTE_WORKDIR@@> \
#     bash evals/scripts/launch_resume.sh ref
#   REMOTE_WORKDIR=... CHECKPOINT_ROOT=<canonical init dir> \
#     bash evals/scripts/launch_resume.sh ours
# Preview with DRY_RUN=1 (prints the cctl args, submits nothing).
# To stop a running job:  cctl pytorchjob stop <id> --reason "manual"
#
# Re-submitting the OURS side with the same REMOTE_WORKDIR resumes from the
# latest step_<abs>/ under its SAVE_ROOT (checkpoint-driven, like production).

set -euo pipefail

SIDE="${1:-${SIDE:-}}"
case "$SIDE" in
    ref|ours) : ;;
    *)
        echo "usage: launch_resume.sh <ref|ours>   (or SIDE=ref|ours)" >&2
        exit 1 ;;
esac

# ── Recipe TOML — [launch] is the topology SSOT; read from the LOCAL checkout
#    (this runs on the mac), pod-visible path passed via --env. env-wins. ──
LOCAL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RECIPE_REL="${RECIPE_REL:-config/train/resume_20.toml}"
[[ -f "$LOCAL_ROOT/$RECIPE_REL" ]] || {
    echo "FATAL: recipe TOML not found in the local checkout: $LOCAL_ROOT/$RECIPE_REL" >&2
    exit 1
}
# eval-form, NOT `source <(...)`: the mac's bash 3.2 sources ZERO bytes from a
# process substitution (silently — the recipe would never load).
eval "$(python3 "$LOCAL_ROOT/tools/train_recipe_to_env.py" --section launch "$LOCAL_ROOT/$RECIPE_REL")"

# ── Per-loop devspace checkout root (same contract + footgun guards as
#    launch_production.sh: pod-visible shared-FS path, never mac-local). ──
REMOTE_WORKDIR="${REMOTE_WORKDIR:?REMOTE_WORKDIR must be the loop DEVSPACE checkout root (== @@REMOTE_WORKDIR@@); this runs on the mac so it cannot be auto-derived}"
case "$REMOTE_WORKDIR" in
    /Users/*)
        echo "FATAL: REMOTE_WORKDIR=$REMOTE_WORKDIR is a mac-local path; the pod mounts the shared FS and cannot see it." >&2
        exit 1 ;;
    /*) : ;;
    *)
        echo "FATAL: REMOTE_WORKDIR=$REMOTE_WORKDIR must be an absolute devspace path." >&2
        exit 1 ;;
esac

# Per-side SAVE_ROOT under .artifacts (excluded from `sync push --delete`).
SAVE_ROOT="${SAVE_ROOT:-${REMOTE_WORKDIR}/.artifacts/resume/${SIDE}}"
ENTRY_SCRIPT="${ENTRY_SCRIPT:-${REMOTE_WORKDIR}/evals/scripts/resume_train_${SIDE}.sh}"

# Canonical init dir — required for the OURS engine only (the ref entry builds
# its own deterministic init from the product's forge_init_ones).
if [[ "$SIDE" == "ours" ]]; then
    CHECKPOINT_ROOT="${CHECKPOINT_ROOT:?CHECKPOINT_ROOT must be the canonical init dir on the shared FS (ours side)}"
fi

# ── Cluster / image / topology (mirror launch_production.sh; env overridable).
PROJECT="${PROJECT:-loopharness}"
CLUSTER="${CLUSTER:-paratera_shandong}"
RESOURCE_POOL="${RESOURCE_POOL:-faxin}"
BILLING_ACCOUNT_ID="${BILLING_ACCOUNT_ID:-N00007}"
IMAGE="${IMAGE:-infra/forge_train:0.9}"
WORKERS="${WORKERS:-0}"                              # env > recipe [launch].workers > 0
GPU_PER_NODE="${GPU_PER_NODE:-${GPUS_PER_NODE:-2}}"  # env > recipe [launch].gpus_per_node > 2
GPU_MODEL="${GPU_MODEL:-h100}"
CPU_PER_NODE="${CPU_PER_NODE:-32}"
MEM_PER_NODE="${MEM_PER_NODE:-256}"
PRIORITY="${PRIORITY:-NORMAL}"

# ── Quota guards (per-person cap: single node, ≤4 cards). ──
if (( WORKERS > 0 )) && [[ "${I_HAVE_QUOTA:-0}" != "1" ]]; then
    echo "FATAL: WORKERS=$WORKERS requests $((WORKERS + 1)) nodes, but this line's per-person cap is 2-4 cards (single node)." >&2
    echo "       If the quota has genuinely been raised for this run, re-run with I_HAVE_QUOTA=1." >&2
    exit 1
fi
if (( GPU_PER_NODE > 4 )) && [[ "${I_HAVE_QUOTA:-0}" != "1" ]]; then
    echo "FATAL: GPU_PER_NODE=$GPU_PER_NODE exceeds the per-person 4-card cap. Re-run with I_HAVE_QUOTA=1 only if the quota is cleared." >&2
    exit 1
fi

CCTL="${CCTL:-cctl}"

echo "── Submitting resume-${SIDE} short train (card-capped) ──" >&2
echo "  nodes=$((WORKERS + 1)) x $GPU_PER_NODE GPU = $(( (WORKERS + 1) * GPU_PER_NODE )) cards" >&2
echo "  recipe=$RECIPE_REL${ITERS:+  ITERS=$ITERS (env override)}" >&2
echo "  SAVE_ROOT=$SAVE_ROOT" >&2
echo "  ENTRY=$ENTRY_SCRIPT" >&2
echo "  cluster=$CLUSTER pool=$RESOURCE_POOL image=$IMAGE billing=$BILLING_ACCOUNT_ID" >&2

CCTL_ARGS=(
    pytorchjob create
    --project           "$PROJECT"
    --cluster           "$CLUSTER"
    --resource-pool     "$RESOURCE_POOL"
    --billing-account-id "$BILLING_ACCOUNT_ID"
    --image             "$IMAGE"
    --workers           "$WORKERS"
    --gpu               "$GPU_PER_NODE"
    --gpu-model         "$GPU_MODEL"
    --cpu               "$CPU_PER_NODE"
    --memory            "$MEM_PER_NODE"
    --priority          "$PRIORITY"
    --env               SAVE_ROOT="$SAVE_ROOT"
    --env               GPUS_PER_NODE="$GPU_PER_NODE"
    --env               RESUME_RECIPE_TOML="${REMOTE_WORKDIR}/${RECIPE_REL}"
    --entry             "bash ${ENTRY_SCRIPT}"
)
if [[ "$SIDE" == "ours" ]]; then
    CCTL_ARGS+=( --env CHECKPOINT_ROOT="$CHECKPOINT_ROOT" )
fi
# Optional passthroughs (only when explicitly set — the pod otherwise reads
# the recipe TOML / its own defaults).
[[ -n "${ITERS:-}" ]]              && CCTL_ARGS+=( --env ITERS="$ITERS" )
[[ -n "${FORGE_RESUME_SUITE:-}" ]] && CCTL_ARGS+=( --env FORGE_RESUME_SUITE="$FORGE_RESUME_SUITE" )
[[ -n "${DATA_CONF:-}" ]]          && CCTL_ARGS+=( --env DATA_CONF="$DATA_CONF" )

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN: $CCTL ${CCTL_ARGS[*]}" >&2
    exit 0
fi

exec "$CCTL" "${CCTL_ARGS[@]}"
