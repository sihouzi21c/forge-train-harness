#!/usr/bin/env bash
# production-train LAUNCHER — submit the CAPPED cctl PyTorchJob for the ours
# WSD-SFT 3-phase long-train. This is the SUBMITTER (runs once, on the box that
# has `cctl`), NOT the per-pod command. The per-pod command is
# evals/scripts/production_train.sh (passed as --entry below); the pod runs
# that, torchrun handles the local ranks.
#
# ── CARD CAP (cluster policy) ─────────────────────────────────────────────
# This line's per-person concurrent GPU quota is 2-4 cards. The launcher is
# therefore PINNED to a single node (--workers 0 = 1 master, no workers) and
# the card count comes from the recipe TOML [launch].gpus_per_node (default 4;
# set 2 there — or GPU_PER_NODE env — for a 2-card run). It REFUSES a
# multi-node request unless I_HAVE_QUOTA=1 explicitly acknowledges the quota
# is cleared — so an agent can never accidentally scale out past the cap.
# train_ours_al.sh itself is node-count agnostic; if the quota ever lifts,
# raising WORKERS with the override is the only change needed.
#
# WHERE IT RUNS: on the box that has `cctl` — for this project that is the MAC,
# NOT the devspace. So every path this bakes into the PyTorchJob spec
# (SAVE_ROOT, ENTRY_SCRIPT) must be a path the pod mounts (the shared /user
# NAS the devspace checkout lives on), never a mac-local path. That is why
# REMOTE_WORKDIR is REQUIRED and explicit — it cannot be auto-derived from the
# launcher's own (mac) location.
#
# Run it AFTER `bin/harness sync push` has landed this loop's worktree on the
# devspace, from the mac:
#   REMOTE_WORKDIR=<loop devspace checkout, == @@REMOTE_WORKDIR@@> \
#   CHECKPOINT_ROOT=<production init dir on the shared FS> \
#     bash evals/scripts/launch_production.sh
# Preview the plan first with DRY_RUN=1 (prints the cctl args, submits nothing).
#
# To stop a running job:  cctl pytorchjob stop <id> --reason "manual"

set -euo pipefail

# ── Recipe TOML — [launch] is the topology SSOT (gpus_per_node / workers).
#    Read from the LOCAL checkout (this runs on the mac); the pod-visible copy
#    of the SAME file is passed to the entry via --env TRAIN_RECIPE_TOML so
#    both sides read one recipe. env-wins: GPU_PER_NODE / WORKERS env
#    overrides still beat the TOML. ──
LOCAL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RECIPE_REL="${RECIPE_REL:-config/train/wsdsft_05b_prod.toml}"
[[ -f "$LOCAL_ROOT/$RECIPE_REL" ]] || {
    echo "FATAL: recipe TOML not found in the local checkout: $LOCAL_ROOT/$RECIPE_REL" >&2
    exit 1
}
# eval-form, NOT `source <(...)`: the mac's bash 3.2 sources ZERO bytes from a
# process substitution (silently — the recipe would never load).
eval "$(python3 "$LOCAL_ROOT/tools/train_recipe_to_env.py" --section launch "$LOCAL_ROOT/$RECIPE_REL")"

# ── Per-loop persistent SAVE_ROOT (the whole point of this launcher) ─────────
# REMOTE_WORKDIR is this loop's DEVSPACE checkout root == @@REMOTE_WORKDIR@@ ==
# <[remote].workspace>/.forge_train/<loop_id>. REQUIRED and explicit: this
# launcher runs on the mac (where cctl lives), so a BASH_SOURCE-derived
# fallback would bake the mac checkout path into the pod spec and the pod
# would fail to find production_train.sh. There is deliberately NO fallback.
# It is: (1) persistent shared FS, (2) per-loop isolated (no cross-agent
# trample), (3) under .artifacts, which `bin/harness sync push --delete`
# EXCLUDES — so a later sync never wipes the checkpoints.
REMOTE_WORKDIR="${REMOTE_WORKDIR:?REMOTE_WORKDIR must be the loop DEVSPACE checkout root (== @@REMOTE_WORKDIR@@, e.g. /user/<user>/.../.forge_train/<loop_id>); this runs on the mac so it cannot be auto-derived}"

# Guard the footgun: a mac-local path (or a relative one) baked into the job
# spec is invisible to the pod. Require an absolute shared-FS path; reject
# /Users/... outright.
case "$REMOTE_WORKDIR" in
    /Users/*)
        echo "FATAL: REMOTE_WORKDIR=$REMOTE_WORKDIR is a mac-local path; the pod mounts the shared FS and cannot see it. Pass the devspace checkout path (@@REMOTE_WORKDIR@@)." >&2
        exit 1 ;;
    /*) : ;;
    *)
        echo "FATAL: REMOTE_WORKDIR=$REMOTE_WORKDIR must be an absolute devspace path." >&2
        exit 1 ;;
esac
SAVE_ROOT="${SAVE_ROOT:-${REMOTE_WORKDIR}/.artifacts/production/wsdsft}"

# ── Per-pod command (absolute so it is cwd-independent inside the pod) ────────
ENTRY_SCRIPT="${ENTRY_SCRIPT:-${REMOTE_WORKDIR}/evals/scripts/production_train.sh}"

# ── Canonical from-scratch init state — REQUIRED (must match [model]). ──
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:?CHECKPOINT_ROOT must be the production from-scratch init dir on the shared FS}"

# ── Cluster / image / topology. Values mirror config/remote/devspace.toml
#    (the same H100 pool the loop's devspace comes from); overridable via env. ─
PROJECT="${PROJECT:-loopharness}"
CLUSTER="${CLUSTER:-paratera_shandong}"
RESOURCE_POOL="${RESOURCE_POOL:-faxin}"
BILLING_ACCOUNT_ID="${BILLING_ACCOUNT_ID:-N00007}"
IMAGE="${IMAGE:-infra/forge_train:0.9}"
WORKERS="${WORKERS:-0}"            # env > recipe [launch].workers > 0 (card cap)
GPU_PER_NODE="${GPU_PER_NODE:-${GPUS_PER_NODE:-4}}"  # env > recipe [launch].gpus_per_node > 4
GPU_MODEL="${GPU_MODEL:-h100}"
CPU_PER_NODE="${CPU_PER_NODE:-32}"
MEM_PER_NODE="${MEM_PER_NODE:-256}"
PRIORITY="${PRIORITY:-NORMAL}"

# ── Quota guards ─────────────────────────────────────────────────────────────
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

echo "── Submitting production-train (WSD-SFT 3-phase, ours, card-capped) ──" >&2
echo "  nodes=$((WORKERS + 1)) x $GPU_PER_NODE GPU = $(( (WORKERS + 1) * GPU_PER_NODE )) cards" >&2
echo "  CHECKPOINT_ROOT=$CHECKPOINT_ROOT" >&2
echo "  SAVE_ROOT=$SAVE_ROOT" >&2
echo "  ENTRY=$ENTRY_SCRIPT" >&2
echo "  cluster=$CLUSTER pool=$RESOURCE_POOL image=$IMAGE billing=$BILLING_ACCOUNT_ID" >&2

# DRY_RUN=1: print the plan and exit (no submit) — inspect before spending GPUs.
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
    --env               CHECKPOINT_ROOT="$CHECKPOINT_ROOT"
    --env               SAVE_ROOT="$SAVE_ROOT"
    --env               GPUS_PER_NODE="$GPU_PER_NODE"
    --env               TRAIN_RECIPE_TOML="${REMOTE_WORKDIR}/${RECIPE_REL}"
    --entry             "bash ${ENTRY_SCRIPT}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN: $CCTL ${CCTL_ARGS[*]}" >&2
    exit 0
fi

exec "$CCTL" "${CCTL_ARGS[@]}"
