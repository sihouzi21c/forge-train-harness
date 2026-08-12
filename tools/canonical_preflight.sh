#!/usr/bin/env bash
# Dispatch the canonical-state preflight to the host where the ref gates
# actually run. Extracted from agent-loop.sh so the local-vs-remote command
# construction is unit-testable (see tests/test_canonical_preflight_dispatch.py).
#
# WHY this split exists: the tokenizer + data mounts the preflight fetches
# into (e.g. [ref].forge_tokenizer_dir = /opt/forge-data/...) and the GPUs
# the canonical dump needs exist ONLY where the ref gates run. For
# [remote].kind = local that host is this machine; for ssh / devspace it is
# the SSH host. Running `evals.canonical_preflight` locally for such a loop
# crashes with `PermissionError: /opt/forge-data` on the launcher (no such
# mount, no GPU), which is exactly the boot failure this dispatcher removes.
#
# For [remote].kind = job the "exec host" is a 0-GPU filesystem gateway, so
# the 2-GPU canonical bridge CANNOT run on it via ssh (that path died with
# `bridge.sh rc=6`). Instead the preflight is submitted as an ephemeral
# `cctl` GPU job via tools/gpu_job.py — the exact same dispatch every other
# GPU suite uses in job mode.
#
# Env contract (all exported by agent-loop.sh's shell_exports / bootstrap):
#   REMOTE_ENABLED    true|false
#   REMOTE_KIND       local|ssh|devspace|job
#   REMOTE_SSH_HOST   ssh alias        (required for ssh/devspace exec)
#   REMOTE_WORKDIR    remote workdir holding the synced harness (ssh/devspace)
#   REMOTE_CONFIG_DIR remote per-loop config dir                (ssh/devspace)
#   FORGE_CONFIG_DIR  local per-loop config dir  (job: read by tools.gpu_job)
#   LOOP_ID           per-loop id                (job: read by tools.gpu_job)
#   PYTHON            local interpreter carrying the harness deps
#
# CANONICAL_PREFLIGHT_PRINT=1 prints the resolved command(s) — one per line —
# instead of executing them, so the dispatch logic is testable with no ssh
# round-trip and no subprocess side effects.
set -euo pipefail

python_bin="${PYTHON:-python3}"

# Seed the shared remote workdir with the current workspace, retrying past
# Teleport reverse-tunnel unavailability. Two cases share this window:
#   * a transient tunnel flap between the lease's SSH probe and here;
#   * a devspace provisioned BY this launch, whose node agent takes
#     minutes to register its reverse tunnel — the old 5x8s (~40 s)
#     budget always lost that race ("no node reverse tunnel found").
# This rsync is the loop's FIRST use of the tunnel, so the default budget
# must cover fresh-provision agent registration (12x30s ≈ 6 min). Both the
# ssh-exec and job paths read the code from this shared volume, so both
# seed it here.
seed_remote_workspace() {
  local attempts="${CANONICAL_PREFLIGHT_SEED_ATTEMPTS:-12}"
  local delay="${CANONICAL_PREFLIGHT_SEED_DELAY:-30}"
  local i
  for i in $(seq 1 "$attempts"); do
    if "${seed_cmd[@]}"; then
      return 0
    fi
    if (( i == attempts )); then
      echo "ERROR: canonical-preflight sync push failed after ${attempts} attempts (remote tunnel unreachable)" >&2
      return 1
    fi
    echo "canonical-preflight sync push attempt ${i}/${attempts} failed; retrying in ${delay}s" >&2
    sleep "$delay"
  done
}

if [[ "${REMOTE_ENABLED:-false}" == "true" ]]; then
  # Idempotent rsync; the dev agent re-syncs each round anyway, but the
  # preflight runs BEFORE the first round so the wrapper must seed it here.
  seed_cmd=("$python_bin" -m harness.cli sync push)

  if [[ "${REMOTE_KIND:-ssh}" == "job" ]]; then
    # kind=job: the devspace is a 0-GPU filesystem gateway — the 2-GPU
    # canonical bridge cannot run on it. Submit it as an ephemeral cctl GPU
    # job, run LOCALLY on the launcher (where cctl/tsh live), exactly like
    # remote_run.sh dispatches suites in job mode. gpu_job.py resolves the
    # shared workdir from remote.toml and reads FORGE_CONFIG_DIR / LOOP_ID
    # from the env, so no REMOTE_SSH_HOST / REMOTE_WORKDIR is needed here.
    ws_root="$(cd "$(dirname "$0")/.." && pwd)"
    job_outer="${CANONICAL_PREFLIGHT_JOB_OUTER:-1800}"
    gpu_job_cmd=("$python_bin" -m tools.gpu_job --canonical --outer "$job_outer")
    if [[ "${CANONICAL_PREFLIGHT_PRINT:-0}" == "1" ]]; then
      printf '%s\n' "${seed_cmd[*]}"
      printf '%s\n' "${gpu_job_cmd[*]}"
      exit 0
    fi
    seed_remote_workspace || exit 1
    export PYTHONPATH="${ws_root}${PYTHONPATH:+:$PYTHONPATH}"
    exec "${gpu_job_cmd[@]}"
  fi

  # kind=ssh/devspace: the exec host carries the mounts + GPUs — run the
  # preflight there over ssh. FORGE_CONFIG_DIR points at the synced per-loop
  # config; PYTHONPATH is the flattened remote workdir (sync push lands
  # harness/ contents at the workdir root).
  : "${REMOTE_SSH_HOST:?REMOTE_SSH_HOST required for remote canonical preflight}"
  : "${REMOTE_WORKDIR:?REMOTE_WORKDIR required for remote canonical preflight}"
  : "${REMOTE_CONFIG_DIR:?REMOTE_CONFIG_DIR required for remote canonical preflight}"
  remote_inner="cd ${REMOTE_WORKDIR} && FORGE_CONFIG_DIR=${REMOTE_CONFIG_DIR} PYTHONPATH=${REMOTE_WORKDIR} python3 -m evals.canonical_preflight"
  preflight_cmd=(ssh "$REMOTE_SSH_HOST" "$remote_inner")

  if [[ "${CANONICAL_PREFLIGHT_PRINT:-0}" == "1" ]]; then
    printf '%s\n' "${seed_cmd[*]}"
    printf '%s\n' "${preflight_cmd[*]}"
    exit 0
  fi

  seed_remote_workspace || exit 1
  exec "${preflight_cmd[@]}"
fi

# Local loop: run the preflight in this process (this machine IS the exec host).
local_cmd=("$python_bin" -m evals.canonical_preflight)
if [[ "${CANONICAL_PREFLIGHT_PRINT:-0}" == "1" ]]; then
  printf '%s\n' "${local_cmd[*]}"
  exit 0
fi
exec "${local_cmd[@]}"
