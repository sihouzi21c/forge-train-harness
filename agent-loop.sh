#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"

# A corp egress proxy may not allow the agent CLI's API host; the agent
# CLI honors http(s)_proxy and would then get a 502. The API host is
# typically reachable directly, so drop the proxy vars before any agent
# call. Harness suites are local-only and need no outbound HTTP.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# Skip network-bound pre-commit hooks (ruff fetches from github.com which
# is reachable but pre-commit's full-clone path hits long timeouts). The
# local framework_guard / path-isolation hook still runs.
export SKIP="${SKIP:+$SKIP,}ruff,ruff-format"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pre-scan args for --loop-id so the bootstrap block below sees the
# user-supplied id BEFORE it decides the workspace directory. The
# flag is also consumed by the regular CLI parser further down (so
# the rest of the script ignores it). LOOP_WEB_ID is the SSOT both
# halves read; web/routers/loop.py sets the same env var when it
# launches a managed loop, so CLI and web flows converge here.
#
# At the same time, capture an optional `--config-dir <dir>` pointing
# at a directory of per-axis TOMLs so the bootstrap block can copy the
# seven known axes into the per-loop config dir before the workspace is
# provisioned. The flag is also recognized (and skipped) by the main
# CLI parser.
CFG_DIR_OVERRIDE="${CFG_DIR_OVERRIDE:-}"
for ((_i=1; _i<=$#; _i++)); do
  if [[ "${!_i}" == "--loop-id" ]]; then
    _j=$((_i+1))
    if [[ $_j -le $# ]]; then
      LOOP_WEB_ID="${!_j}"
      export LOOP_WEB_ID
    fi
  elif [[ "${!_i}" == "--config-dir" ]]; then
    _j=$((_i+1))
    if [[ $_j -le $# ]]; then
      CFG_DIR_OVERRIDE="${!_j}"
    fi
  fi
done
unset _i _j
export CFG_DIR_OVERRIDE

# Require Python ≥3.10 (dataclass slots=True, PEP 604 unions)
PYTHON=""
for _candidate in python3.12 python3.11 python3.10 python3; do
  _path="$(command -v "$_candidate" 2>/dev/null)" || continue
  _ver="$("$_path" -c 'import sys; print(sys.version_info >= (3,10))' 2>/dev/null)" || continue
  [[ "$_ver" == "True" ]] && { PYTHON="$_path"; break; }
done
[[ -z "$PYTHON" ]] && { echo "ERROR: Python ≥3.10 not found in PATH" >&2; exit 1; }
export PYTHON

# ── CLI auto-provision (bootstrap) ───────────────────────────────────
# Every loop — CLI or web — runs inside an isolated copy of harness/ at
# $FORGE_TRAIN_DIR/<loop_id>/workspace/. The web side provisions the
# workspace eagerly (web/routers/loop.py:_provision_workspace) and
# exports FORGE_TRAIN_PROVISIONED=1 to skip this block. When invoked
# from a source checkout, we copy harness/ into the registry directory
# and re-exec the workspace's copy. This keeps the source worktree
# pristine and unifies CLI/web flows.
#
# Unified-agent-log layering: agent-loop.sh shells out to
# harness/tools/spawn_managed_agent.py (and friends), which import
# web.agents.* to write each spawned cursor-cli / claude session into
# .artifacts/web-agents/<id>/. The web/ package only lives in the
# original source repo root, while harness runtime config must resolve
# against the provisioned workspace root. Keep those roots separate.
if [[ "${FORGE_TRAIN_PROVISIONED:-0}" != "1" ]]; then
  _bootstrap_repo_root="$(cd "$SCRIPT_DIR/.." && pwd)"
  LOOP_ID="${LOOP_WEB_ID:-$("$PYTHON" -c 'import uuid; print(uuid.uuid4().hex[:12])')}"
  : "${FORGE_TRAIN_DIR:=$_bootstrap_repo_root/.artifacts/forge_train}"
  : "${FORGE_SOURCE_ROOT:=$_bootstrap_repo_root}"
  : "${FORGE_AGENTS_DIR:=$_bootstrap_repo_root/.artifacts/web-agents}"

  # Fire-and-forget trajectory backup: push all terminal agent-log trees
  # to OBS on every loop launch (append-only, idempotent). Default-on;
  # FORGE_OBS_PUSH_ON_START=0 disables. The push script soft-skips when
  # obsutil/credentials are absent, the background subshell keeps loop
  # startup latency at zero, and the script-presence guard covers
  # workspace copies that ship without the repo-root scripts/ dir.
  if [[ "${FORGE_OBS_PUSH_ON_START:-1}" == "1" \
        && -f "$_bootstrap_repo_root/scripts/obs_traj_sync.py" ]]; then
    mkdir -p "$_bootstrap_repo_root/.artifacts"
    (FORGE_AGENTS_DIR="$FORGE_AGENTS_DIR" \
       bash "$_bootstrap_repo_root/scripts/obs_traj_push.sh" --quiet \
       >>"$_bootstrap_repo_root/.artifacts/obs_push.log" 2>&1 || true) &
  fi

  # Record which commit/branch of the source checkout this loop launches
  # from (incl. the branch a git worktree was created from) BEFORE the
  # workspace copy strips .git. First capture wins on resume. Best-effort:
  # a non-git source root records nothing and never blocks the launch.
  "$PYTHON" - "$_bootstrap_repo_root" "$FORGE_TRAIN_DIR/$LOOP_ID" <<'PY' 2>/dev/null || true
import sys
sys.path.insert(0, sys.argv[1])
from web.agents import provenance
provenance.write(sys.argv[1], sys.argv[2])
PY

  _bootstrap_workspace="$FORGE_TRAIN_DIR/$LOOP_ID/workspace"
  if [[ -f "$_bootstrap_workspace/agent-loop.sh" ]]; then
    # Resume path: workspace already provisioned (CLI --loop-id <id>
    # pointing at a prior loop, or a re-entrant run). Re-rsyncing
    # harness/ here would clobber the agent's accumulated edits
    # under workload/src/training_engine_tensor/ and the previous
    # stage-status markers, defeating the resume contract.
    echo "[$(date '+%F %T')] resuming loop $LOOP_ID — workspace exists, skipping bootstrap copy" >&2
  else
    mkdir -p "$_bootstrap_workspace"
    "$PYTHON" - "$SCRIPT_DIR" "$_bootstrap_workspace" <<'PY'
import shutil
import stat
import sys
import textwrap
from pathlib import Path

src, dst = sys.argv[1:3]
shutil.copytree(
    src,
    dst,
    dirs_exist_ok=True,
    ignore=shutil.ignore_patterns(
        ".git", ".artifacts", "__pycache__", ".pytest_cache",
        # Per-loop axis configs live (frozen) at <loop>/config/<axis>.toml
        # and are pointed at via FORGE_CONFIG_DIR. The matching basenames
        # under harness/config/ in the source tree are gitignored user-local
        # copies; if any happens to exist, copytree would seed a stale
        # alternative into <workspace>/config/ that diverges from the
        # frozen truth the harness CLI actually reads. Drop them so the
        # workspace ships only committed templates (harness/config/<axis>/*.toml)
        # and the active values come from the lease-side symlink (see
        # tools/agent_loop_lease.sh _devspace_claim_and_freeze).
        "ref.toml", "data.toml", "remote.toml",
        "agent.toml", "eval.toml", "model.toml", "optim.toml",
    ),
)
# Per-workspace harness shim. Baked with the workspace's absolute
# path so PATH lookups always reach this workspace's harness, even
# when a sibling worktree's `pip install -e .` left a global
# /usr/local/bin/harness behind (root cause of the
# bitwise-singlecard/bitwise-perf cliff).
bin_dir = Path(dst) / "bin"
bin_dir.mkdir(parents=True, exist_ok=True)
shim = bin_dir / "harness"
shim.write_text(
    textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        # Auto-generated per-workspace harness shim. DO NOT COMMIT.
        # Workspace: {dst}
        export PYTHONPATH="{dst}${{PYTHONPATH:+:$PYTHONPATH}}"
        exec python3 -m harness.cli "$@"
        """
    ),
    encoding="utf-8",
)
shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
PY
    # The workspace sits inside the source repo's worktree. Without an
    # isolated .git here, any `git` call by the dev/review agent walks
    # up to the source `.git` (web_harness branch + every sibling
    # branch), letting it `git show <other-branch>:...` complete
    # engine implementations from the history. Seed a fresh per-loop
    # repo with a baseline snapshot so review's `git diff HEAD~1 HEAD`
    # works after the agent's first commit. Failures here are
    # non-fatal — GIT_CEILING_DIRECTORIES (exported below) still
    # fail-closes the up-walk to a clean "not a git repository" error.
    if [[ ! -d "$_bootstrap_workspace/.git" ]]; then
      (
        set +e
        cd "$_bootstrap_workspace" || exit 0
        if [[ ! -f .gitignore ]]; then
          cat > .gitignore <<'GI'
.artifacts/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
GI
        fi
        GIT_CEILING_DIRECTORIES="$FORGE_TRAIN_DIR" git init -q -b harness 2>/dev/null \
          || { GIT_CEILING_DIRECTORIES="$FORGE_TRAIN_DIR" git init -q; \
               GIT_CEILING_DIRECTORIES="$FORGE_TRAIN_DIR" git checkout -q -b harness 2>/dev/null; }
        GIT_CEILING_DIRECTORIES="$FORGE_TRAIN_DIR" git config user.name harness
        GIT_CEILING_DIRECTORIES="$FORGE_TRAIN_DIR" git config user.email harness@local
        GIT_CEILING_DIRECTORIES="$FORGE_TRAIN_DIR" git config commit.gpgsign false
        GIT_CEILING_DIRECTORIES="$FORGE_TRAIN_DIR" git add -A
        GIT_CEILING_DIRECTORIES="$FORGE_TRAIN_DIR" git commit -q --allow-empty \
          -m "Initial workspace snapshot (harness bootstrap)"
      ) || echo "[bootstrap] WARN: failed to seed workspace git repo at $_bootstrap_workspace" >&2
    fi
  fi
  export FORGE_TRAIN_PROVISIONED=1
  export FORGE_REPO_ROOT="$_bootstrap_workspace"
  export FORGE_TRAIN_DIR FORGE_REPO_ROOT FORGE_SOURCE_ROOT FORGE_AGENTS_DIR
  # ── Per-loop config dir (sibling of workspace) ───────────────────
  # Replaces the legacy shared `harness/config/*.toml` foot-gun: each
  # loop owns an immutable copy under
  # `.artifacts/forge_train/<id>/config/`. Skill flow pre-stages the
  # files there; legacy/in-flight callers may point at a directory of
  # axis TOMLs via `--config-dir <dir>`. Missing axes fail-fast so a
  # half-staged config never reaches the agent.
  # Mirrors harness.loop_layout.loop_config_dir (the topology SSOT). Bash
  # can't import it without paying interpreter startup on every launch, so
  # the literal is kept native and pinned to the SSOT by a value-equivalence
  # test (test_agent_loop_config_flag) — same pattern as web/paths.py.
  _cfg_dir="$FORGE_TRAIN_DIR/$LOOP_ID/config"
  # Resume path: the dir already exists, is frozen, and the lease (if
  # any) was claimed by the original launch. Skip the staging/lease/
  # freeze pipeline entirely — repeating it would either fail
  # (chmod -w'd files reject `cp`) or double-claim a devspace.
  _cfg_already_present=0
  if [[ -d "$_cfg_dir" ]] && [[ -f "$_cfg_dir/remote.toml" ]]; then
    _cfg_already_present=1
  fi
  # The LOOP_REGISTER_ONLY=1 fast-path exits after writing session.json
  # without spawning the agent — no config_runtime call, no lease, no
  # freeze. Demanding a per-loop config dir here would block every
  # test/CI smoke that exercises the registration path.
  if (( _cfg_already_present == 0 )) && [[ "${LOOP_REGISTER_ONLY:-0}" != "1" ]]; then
    mkdir -p "$_cfg_dir"
    if [[ -n "${CFG_DIR_OVERRIDE:-}" ]]; then
      if [[ ! -d "$CFG_DIR_OVERRIDE" ]]; then
        echo "ERROR: --config-dir not a directory: $CFG_DIR_OVERRIDE" >&2
        exit 2
      fi
      for _axis in ref data remote agent eval model optim; do
        if [[ -f "$CFG_DIR_OVERRIDE/$_axis.toml" ]]; then
          cp "$CFG_DIR_OVERRIDE/$_axis.toml" "$_cfg_dir/$_axis.toml"
        fi
      done
      unset _axis
    fi
    for _axis in ref data remote agent eval model optim; do
      if [[ ! -f "$_cfg_dir/$_axis.toml" ]]; then
        echo "ERROR: per-loop config missing: $_cfg_dir/$_axis.toml" >&2
        echo "       Either pre-stage via the new-looptask skill or pass --config-dir /path/to/configdir" >&2
        exit 2
      fi
    done
    unset _axis
    # NOTE: the devspace lease claim, hostname rewrite, and config-dir
    # freeze used to live HERE — but this block is CLI-only (the web flow
    # sets FORGE_TRAIN_PROVISIONED=1 and skips it). The lifecycle now runs
    # in the shared post-bootstrap path via tools/agent_loop_lease.sh so
    # both CLI and web claim a lease + freeze exactly once. This block now
    # only stages the per-loop config (CLI --config-dir copy + fail-fast
    # presence check); it deliberately leaves the dir WRITABLE so the
    # shared helper can claim/rewrite/freeze it.
  fi
  unset _cfg_already_present
  export FORGE_CONFIG_DIR="$_cfg_dir"
  unset _cfg_dir
  # Per-workspace shim wins over any global /usr/local/bin/harness
  # left by a sibling worktree's `pip install -e .`. The shim was
  # written above with this workspace's absolute path baked in.
  export PATH="$_bootstrap_workspace/bin:$PATH"
  # Belt-and-braces: even if a subagent runs `git` from a workspace
  # subdirectory before any local .git is found, the ceiling stops the
  # walk at the registry root and prevents leaking the source repo.
  export GIT_CEILING_DIRECTORIES="$FORGE_TRAIN_DIR"
  export LOOP_WEB_ID="$LOOP_ID"
  exec bash "$_bootstrap_workspace/agent-loop.sh" "$@"
fi

WORKSPACE="$SCRIPT_DIR"
# Pin cwd to the workspace root before anything spawns `python -m harness`.
# `python -m` inserts cwd at sys.path[0] *before* PYTHONPATH, so a launcher
# that cd'd into the inner `harness/` directory would silently shadow the
# workspace's package with the inner package of the same name. The
# workspace_contract harness_cli_resolves_locally invariant fires when this
# happens; this cd makes that invariant a backstop rather than the first
# line of defence.
cd "$WORKSPACE"
export PYTHONPATH="$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
# Run from the workspace so Python's implicit sys.path[0] (the process
# cwd, prepended by `python -m`) points at THIS workspace's harness
# package rather than the source checkout we were launched from. Without
# this, `import harness` resolves to the source tree (cwd wins over
# PYTHONPATH) and the workspace-contract env-probe fails with a
# misleading "global editable install is shadowing" violation. Every
# path-sensitive call below already uses an absolute $WORKSPACE /
# $SCRIPT_DIR anchor or an explicit --workspace / -C, so changing the
# process cwd here is safe.
cd "$WORKSPACE"

# Resolve roots for the already-provisioned path (web server set them;
# or this script was invoked directly from a source checkout without the
# bootstrap block running, e.g. tests with FORGE_TRAIN_PROVISIONED=1 pre-set).
if [[ -z "${FORGE_REPO_ROOT:-}" ]]; then
  FORGE_REPO_ROOT="$SCRIPT_DIR"
  export FORGE_REPO_ROOT
fi
if [[ -z "${FORGE_SOURCE_ROOT:-}" ]]; then
  FORGE_SOURCE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  export FORGE_SOURCE_ROOT
fi
: "${FORGE_AGENTS_DIR:=$FORGE_SOURCE_ROOT/.artifacts/web-agents}"
export FORGE_AGENTS_DIR FORGE_SOURCE_ROOT
mkdir -p "$FORGE_AGENTS_DIR"

# Re-export the git ceiling in the already-provisioned branch too, so
# the protection is consistent whether we got here via the CLI
# bootstrap above (which already set it) or via the web router (whose
# `_build_env` sets it, but only when the inherited environment makes
# it visible to us — manual test harnesses may pre-set
# FORGE_TRAIN_PROVISIONED=1 without it). Without this, a subagent run
# from any subdirectory of `$WORKSPACE` could still walk up past the
# workspace's own `.git` and reach the source repo's `.git`.
if [[ -z "${GIT_CEILING_DIRECTORIES:-}" ]]; then
  : "${FORGE_TRAIN_DIR:=$(cd "$FORGE_REPO_ROOT/../.." 2>/dev/null && pwd)}"
  if [[ -n "$FORGE_TRAIN_DIR" ]]; then
    export FORGE_TRAIN_DIR
    export GIT_CEILING_DIRECTORIES="$FORGE_TRAIN_DIR"
  fi
fi

# ── Workspace contract (fail-fast infra invariants) ──────────────────
# Run after PYTHONPATH/ceiling are set so `import harness` resolves to
# the provisioned workspace's copy. A non-zero exit aborts the loop
# with a machine-readable fix instruction in stderr; see
# harness/harness/workspace_contract.py for the invariant list.
#
# Skipped under LOOP_REGISTER_ONLY=1 (test fast-path that bypasses any
# heavy config eval) and when SKIP_ENV_PROBE=1 is set explicitly. The
# escape hatch is intentionally narrow — agent loops MUST run the
# probe so bitwise-singlecard/bitwise-perf-class infra bugs cannot
# silently waste agent time.
if [[ "${LOOP_REGISTER_ONLY:-0}" != "1" && "${SKIP_ENV_PROBE:-0}" != "1" ]]; then
  if ! "$PYTHON" -m harness.cli env-probe; then
    echo "[$(date '+%F %T')] env-probe FAILED — refusing to start loop." >&2
    echo "Fix the violation reported above, then re-run agent-loop.sh." >&2
    exit 2
  fi
fi

# ── CLI flags ────────────────────────────────────────────────────────
# A leading positional token is treated as a subcommand; flags follow.
# Currently the only non-default subcommand is `status`, a no-side-
# effects backend-auth probe. Anything else (or omission) runs the loop.
SUBCOMMAND=""
if [[ $# -gt 0 && "$1" != -* ]]; then
  SUBCOMMAND="$1"
  shift
fi

RESET_STATE=0
CLI_MODEL=""
CLI_REVIEW_MODEL=""
CLI_RUNS_PER_STAGE=""
CLI_MAX_FAILS=""
CLI_STAGES=""
CLI_MAX_MODE=""
CLI_API_KEY=""
CLI_BACKEND=""
MANUAL_APPROVE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reset-state)
      RESET_STATE=1
      shift
      ;;
    --loop-id)
      # Consumed by the pre-scan above the bootstrap block (which
      # exported LOOP_WEB_ID before the workspace path was chosen).
      # Accept it here so the parser does not reject it as unknown.
      shift 2
      ;;
    --config-dir)
      # Consumed by the pre-scan into CFG_DIR_OVERRIDE (which was
      # applied during bootstrap workspace provisioning). Accept it
      # here so the parser does not reject the flag.
      shift 2
      ;;
    --model)
      CLI_MODEL="$2"
      shift 2
      ;;
    --review-model)
      CLI_REVIEW_MODEL="$2"
      shift 2
      ;;
    --runs-per-stage)
      CLI_RUNS_PER_STAGE="$2"
      shift 2
      ;;
    --max-fails)
      CLI_MAX_FAILS="$2"
      shift 2
      ;;
    --stages)
      CLI_STAGES="$2"
      shift 2
      ;;
    --max-mode)
      CLI_MAX_MODE="true"
      shift
      ;;
    --manual-approve)
      MANUAL_APPROVE="true"
      shift
      ;;
    --api-key)
      CLI_API_KEY="$2"
      shift 2
      ;;
    --backend)
      CLI_BACKEND="$2"
      shift 2
      ;;
    --help|-h)
      cat <<'USAGE'
Usage: bash agent-loop.sh [SUBCOMMAND] [OPTIONS]

Subcommands:
  (default)               Run the goal-driven agent loop.
  status                  Probe backend auth and exit (no side effects).
                          Runs the same check-backend helper the loop
                          uses at startup.

Options:
  --api-key <key>         Backend API key (optional; cursor-cli falls back to Cursor CLI login).
  --backend <name>        Agent backend: cursor-cli, claude-code, or codex (default from [agent].backend).
  --reset-state           Wipe per-stage status markers and re-run all stages.
  --loop-id <id>          Resume an existing loop: reuse the workspace at
                          .artifacts/forge_train/<id>/workspace/ (skipping
                          the bootstrap copy that would overwrite the
                          agent's accumulated edits) and pin the session
                          id. Default: fresh UUID per invocation.
  --config-dir <dir>      Copy the per-loop axis TOMLs (ref/data/remote/
                          agent/eval/model/optim) from <dir> into the
                          loop's frozen config dir (.artifacts/forge_train/
                          <id>/config/). Only the seven known axes are
                          copied. Axes missing from both <dir> and the
                          skill-pre-staged dir fail the launch.
  --model <slug>          Agent model (overrides [agent].model).
  --review-model <slug>   Review agent model (defaults to --model value).
  --runs-per-stage <n>    Per-stage round cap; 0 = unlimited (default from dense_training.toml).
  --max-fails <n>         Consecutive REVIEW_VERDICT: FAIL before abort.
  --stages <s1,s2,...>    Comma-separated stage list (default: all stages).
  --max-mode              Enable Max Mode for the agent CLI.
  --manual-approve        Pause between stages; wait for approval via gate file.
  -h, --help              Show this help.
USAGE
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

case "$SUBCOMMAND" in
  ""|run) ;;
  status) ;;
  *)
    echo "Unknown subcommand: $SUBCOMMAND" >&2
    exit 1
    ;;
esac

# `status` subcommand: pure auth probe, no state mutation.
if [[ "$SUBCOMMAND" == "status" ]]; then
  status_cli=$($PYTHON "$WORKSPACE/tools/agent_loop_config.py" active-cli)
  exec $PYTHON "$WORKSPACE/tools/agent_loop_config.py" check-backend "$status_cli"
fi

# Test-only fast-path: integration tests for the disk-registry / wrapper
# event contract set LOOP_REGISTER_ONLY=1 to verify session.json is
# published without triggering the heavy config eval (which resolves
# [ref].megatron and would clone Megatron-LM on a fresh workspace).
# Build session.json directly from CLI args and exit before any heavy
# config / EXIT-trap machinery (including finalize_wrapper_session,
# which assumes init_wrapper_session has already run).
if [[ "${LOOP_REGISTER_ONLY:-0}" == "1" ]]; then
  _REGONLY_LOOP_ID="${LOOP_WEB_ID:-$($PYTHON -c 'import uuid; print(uuid.uuid4().hex[:12])')}"
  _REGONLY_BACKEND="${CLI_BACKEND:-cursor-cli}"
  _REGONLY_MODEL="${CLI_MODEL:-}"
  _REGONLY_STAGES="${CLI_STAGES:-stage1}"
  _REGONLY_MAX_MODE="${CLI_MAX_MODE:-false}"
  _REGONLY_DIR="${FORGE_TRAIN_DIR:-$FORGE_REPO_ROOT/.artifacts/forge_train}/$_REGONLY_LOOP_ID"
  mkdir -p "$_REGONLY_DIR"
  _REGONLY_STARTED="$($PYTHON -c 'import time; print(time.time())')"
  _REGONLY_LABEL="$(date '+%Y-%m-%d %H:%M:%S')"
  $PYTHON - "$_REGONLY_DIR/session.json" "$_REGONLY_LOOP_ID" "$_REGONLY_LABEL" \
    "$$" "$_REGONLY_STARTED" "$WORKSPACE" "${LOOP_OUTPUT_LOG:-}" \
    "$_REGONLY_BACKEND" "$_REGONLY_MODEL" "$_REGONLY_STAGES" "$_REGONLY_MAX_MODE" <<'PY'
import json, os, sys, time
(target, loop_id, label, pid, started_at, workspace, output_file,
 backend, model, stages, max_mode) = sys.argv[1:12]
stages_list = stages.replace(",", " ").split()
data = {
    "loop_id": loop_id,
    "label": label,
    "mode": "external",
    "status": "completed",
    "pid": int(pid),
    "started_at": float(started_at),
    "ended_at": time.time(),
    "exit_code": 0,
    "args": {
        "backend": backend,
        "model": model,
        "stages": " ".join(stages_list),
        "max_mode": max_mode,
    },
    "workspace_dir": workspace,
    "log_dir": None,
    "output_file": output_file or None,
}
tmp = target + ".tmp." + str(os.getpid())
with open(tmp, "w", encoding="utf-8") as fh:
    json.dump(data, fh, ensure_ascii=False, indent=2)
os.replace(tmp, target)
PY
  # Also init + finalize the loop-wrapper Session so the unified
  # transcript contract (system.init + loop_exit lines, session.json
  # with state=completed) holds for the LOOP_REGISTER_ONLY=1 fast-path.
  LOOP_ID="$_REGONLY_LOOP_ID"
  export LOOP_ID
  "$PYTHON" "$FORGE_SOURCE_ROOT/harness/tools/loop_wrapper_init.py" \
    --loop-id "$_REGONLY_LOOP_ID" \
    --workspace "$WORKSPACE" \
    --backend "$_REGONLY_BACKEND" \
    --model "$_REGONLY_MODEL" \
    --stages "$_REGONLY_STAGES" \
    --pid "$$" || true
  "$PYTHON" -c "
import sys
sys.path.insert(0, '$FORGE_SOURCE_ROOT')
from web.agents import spawn
spawn.append_loop_event('$_REGONLY_LOOP_ID', 'loop_exit', {'state': 'completed', 'exit_code': 0})
spawn.finalize_wrapper_session(loop_id='$_REGONLY_LOOP_ID', exit_code=0, state='completed')
" || true
  echo "[loop-register] LOOP_REGISTER_ONLY=1 set; session.json published at $_REGONLY_DIR/session.json"
  exit 0
fi

# ── Devspace lease + config freeze ───────────────────────────────────
# Single place BOTH the CLI bootstrap and the web launch claim the
# exclusive devspace, rewrite [remote].hostname, and freeze the per-loop
# config dir. Must precede the `agent_loop_config.py shell` eval below,
# which reads the (now-rewritten) hostname.
#
# Resolve LOOP_ID up-front (the claim is keyed on it). LOOP_WEB_ID is the
# SSOT — set by the CLI bootstrap re-exec and by the web router — so pin
# it here too, keeping the identity that line ~590's LOOP_ID resolution
# reuses verbatim.
LOOP_ID="${LOOP_WEB_ID:-$($PYTHON -c 'import uuid; print(uuid.uuid4().hex[:12])')}"
export LOOP_ID
LOOP_WEB_ID="$LOOP_ID"
export LOOP_WEB_ID
# shellcheck source=tools/agent_loop_lease.sh
source "$WORKSPACE/tools/agent_loop_lease.sh"
_devspace_claim_and_freeze "$FORGE_CONFIG_DIR"

# ── Provision-only fast-path ─────────────────────────────────────────
# The meta loop's harness_configs gate provisions this forge workspace +
# renders its gate config (done by the lines above), then pre-runs each
# gate's ref side via `bin/harness run` to prove the handed-off DP×TP ref
# bundle is runnable — WITHOUT entering the dev loop. Exit right here, after
# claim+freeze+render: everything below (config eval, backend auth, the
# stage loop) is dev-loop-only and would need agent CLI auth the gate has no
# use for. Mirrors the LOOP_REGISTER_ONLY fast-path, but one step later (it
# needs the rendered gate config the freeze produces).
if [[ "${LOOP_PROVISION_ONLY:-0}" == "1" ]]; then
  echo "[provision-only] workspace provisioned + gate config rendered: $WORKSPACE"
  exit 0
fi

# ── Config ───────────────────────────────────────────────────────────
eval "$($PYTHON "$WORKSPACE/tools/stage2_config.py" shell)"
eval "$($PYTHON "$WORKSPACE/tools/agent_loop_config.py" shell)"

# CLI overrides (take precedence over dense_training.toml defaults)
[[ -n "$CLI_MODEL" ]] && MODEL="$CLI_MODEL"
[[ -n "$CLI_REVIEW_MODEL" ]] && REVIEW_MODEL="$CLI_REVIEW_MODEL"
[[ -z "$REVIEW_MODEL" ]] && REVIEW_MODEL="$MODEL"
[[ -n "$CLI_RUNS_PER_STAGE" ]] && RUNS_PER_STAGE="$CLI_RUNS_PER_STAGE"
[[ -n "$CLI_MAX_FAILS" ]] && MAX_CONSECUTIVE_REVIEW_FAILS="$CLI_MAX_FAILS"
[[ -n "$CLI_MAX_MODE" ]] && MAX_MODE="$CLI_MAX_MODE"
[[ -n "$CLI_BACKEND" ]] && AGENT_BACKEND="$CLI_BACKEND"

case "$AGENT_BACKEND" in
  cursor-cli|claude-code|codex) ;;
  *)
    echo "ERROR: unsupported --backend '$AGENT_BACKEND' (expected cursor-cli, claude-code, or codex)" >&2
    exit 1
    ;;
esac

# Config-supplied API key ([agent].api_key -> AGENT_API_KEY) is a fallback
# for the --api-key flag: an explicit CLI key always wins. This keeps the
# key in the per-loop frozen config (gitignored), exported only inside this
# process below — never written to a shell profile or ~/.claude.json.
[[ -z "$CLI_API_KEY" && -n "$AGENT_API_KEY" ]] && CLI_API_KEY="$AGENT_API_KEY"

if [[ -n "$CLI_API_KEY" ]]; then
  case "$AGENT_BACKEND" in
    cursor-cli) export CURSOR_API_KEY="$CLI_API_KEY" ;;
    claude-code) export ANTHROPIC_API_KEY="$CLI_API_KEY" ;;
    codex) export "${CODEX_API_KEY_ENV:-OPENAI_API_KEY}=$CLI_API_KEY" ;;
  esac
fi

# Process-local proxy override for claude-code. [agent].base_url is empty
# by default, so the Claude Code CLI keeps using its logged-in claude.ai
# OAuth subscription and the global default is untouched. When set, this
# export lives only in agent-loop.sh's process tree (this tmux pane and the
# dev/review/stage2 children it spawns); other shells, tmux sessions, and
# future `claude` invocations see no env and stay on the official account.
[[ "$AGENT_BACKEND" == claude-code && -n "$AGENT_BASE_URL" ]] && export ANTHROPIC_BASE_URL="$AGENT_BASE_URL"

# Pre-loop backend auth check (maps AGENT_BACKEND → CLI short name)
case "$AGENT_BACKEND" in
  claude-code) _check_cli="claude" ;;
  codex) _check_cli="codex" ;;
  *) _check_cli="cursor" ;;
esac
$PYTHON "$WORKSPACE/tools/agent_loop_config.py" check-backend "$_check_cli" || {
  echo "ERROR: backend availability check failed — see diagnostic above." >&2
  exit 1
}

if [[ -n "$CLI_STAGES" ]]; then
  IFS=',' read -ra STAGES <<< "$CLI_STAGES"
else
  # macOS ships bash 3.2 which lacks `mapfile`; use `while read` for portability.
  STAGES=()
  while IFS= read -r _stage_line; do
    [[ -n "$_stage_line" ]] && STAGES+=("$_stage_line")
  done < <($PYTHON "$WORKSPACE/tools/agent_loop_config.py" stages)
fi

LOG_DIR="$WORKSPACE/.artifacts/agent-logs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"

# ── Loop identity + wrapper-side event channel ───────────────────────
# LOOP_ID is the SSOT for the loop's identity. LOOP_WEB_ID is set by
# the bootstrap block above (or by the web router before exec), so
# the same id flows through the workspace bootstrap, the wrapper
# session.json, and every spawned child's parent_agent_id.
LOOP_ID="${LOOP_WEB_ID:-$($PYTHON -c 'import uuid; print(uuid.uuid4().hex[:12])')}"
export LOOP_ID

# Create the synthetic loop-wrapper Session entry up-front so any
# subsequent _loop_event call has a stdout.log to append to. The
# helper writes session.json + a system.init seed line under
# $FORGE_AGENTS_DIR/loop-$LOOP_ID/.
"$PYTHON" "$FORGE_SOURCE_ROOT/harness/tools/loop_wrapper_init.py" \
  --loop-id "$LOOP_ID" \
  --workspace "$WORKSPACE" \
  --backend "$AGENT_BACKEND" \
  --model "$MODEL" \
  --stages "${STAGES[*]}" \
  --pid "$$"

# _loop_event <subtype> [k1 v1 k2 v2 ...]
# Single writer for wrapper-side orchestration text. Each call appends
# one typed NDJSON record to the wrapper's stdout.log so the frontend
# chat renderer can show stage/round headers, spawn_child cards, and
# verdict pills in the same view as cursor-cli/claude assistant text.
#
# Key-value pairs are passed directly to the Python helper which builds
# JSON internally.  One Python invocation per event, no $() command
# substitution at the call site.  Internally infallible: errors are
# swallowed so a diagnostic call can NEVER kill the script regardless
# of set -e / pipefail state.
_loop_event() {
  local subtype="$1"; shift
  "$PYTHON" "$FORGE_SOURCE_ROOT/harness/tools/loop_wrapper_event.py" \
    --loop-id "$LOOP_ID" --subtype "$subtype" -- "$@" \
    2>/dev/null || true
}

# ── Per-stage state machine ──────────────────────────────────────────
# Status file per stage: <state_dir>/<stage>.status with content
# `pending` / `in-progress` / `finished`.  Written by parse_review_status
# below after each round; read at startup so a restart of agent-loop.sh
# resumes from the first non-finished stage.
mkdir -p "$AGENT_LOOP_STATE_DIR"
if (( RESET_STATE == 1 )); then
  rm -f "$AGENT_LOOP_STATE_DIR"/*.status \
        "$AGENT_LOOP_STATE_DIR"/*.milestone \
        "$AGENT_LOOP_STATE_DIR"/*.round
  echo "[$(date '+%F %T')] --reset-state: cleared $AGENT_LOOP_STATE_DIR" >&2
fi

# Stage 1 milestone state I/O lives in agent_loop_milestone.sh so the
# pure parser/validator helpers can be unit-tested without sourcing
# the whole loop. agent-loop.sh adds the stage-aware wrappers below
# and emits the milestone_advanced loop event.
# shellcheck source=tools/agent_loop_milestone.sh
source "$WORKSPACE/tools/agent_loop_milestone.sh"

# Resolve the Stage 1 milestone progression order once (SSOT: active
# eval.toml [stage1].milestone_order, surfaced by agent_loop_config.py) and
# export it as a space-separated list. The milestone helpers sourced above
# stay python-free and read it from this env var (see that file's header).
FORGE_MILESTONE_ORDER="$("$PYTHON" "$WORKSPACE/tools/agent_loop_config.py" \
  stage-milestones stage1 2>/dev/null | tr '\n' ' ')"
export FORGE_MILESTONE_ORDER

# Which milestone the run_agent_stage slow-poll throttle fires on: the
# multi-day production long-train (the eval.toml suite whose runner_kind is
# `production-train`). Resolved from the active config so the throttle keys
# on the SSOT, never a hardcoded name. Empty when the config ships no such
# suite (e.g. the 8b suite, which ends at bitwise-dptp) -> never throttles.
FORGE_SLOW_POLL_MILESTONE="$("$PYTHON" "$WORKSPACE/tools/agent_loop_config.py" \
  slow-poll-milestone stage1 2>/dev/null | tr -d '\n')"
export FORGE_SLOW_POLL_MILESTONE

# Review-gated milestones (space-separated). For these, the dev agent's
# MILESTONE_STATUS self-declaration alone does NOT advance the milestone:
# the acceptance bar lives only in the review prompt (e.g. the long-horizon
# review-side MFU floor, deliberately withheld from the dev agent — see
# prompt/review_prompt/review_stage1.md), so advance_stage_milestone
# additionally requires this round's review verdict to be PASS and the
# declaration to be freshly committed within the round. Set to "" to
# restore the fully decoupled historical behavior for every milestone.
FORGE_REVIEW_GATED_MILESTONES="${FORGE_REVIEW_GATED_MILESTONES-long-horizon}"
export FORGE_REVIEW_GATED_MILESTONES

stage_state_file() {
  printf '%s/%s.status\n' "$AGENT_LOOP_STATE_DIR" "$1"
}

stage_milestone_file() {
  printf '%s/%s.milestone\n' "$AGENT_LOOP_STATE_DIR" "$1"
}

# Per-stage round counter file: integer count of the highest round that
# has started. Persisted at the top of each round so a crashed/killed
# loop resumes from N+1 (where N is the dead round) instead of 1.
stage_round_file() {
  printf '%s/%s.round\n' "$AGENT_LOOP_STATE_DIR" "$1"
}

read_stage_round() {
  local f
  f="$(stage_round_file "$1")"
  if [[ -s "$f" ]]; then
    local raw
    raw=$(head -n1 "$f" | tr -d '[:space:]')
    if [[ "$raw" =~ ^[0-9]+$ ]]; then
      printf '%s' "$raw"
      return 0
    fi
    _loop_event info text "WARN: ignoring invalid round value '$raw' for $1; resetting to 0" stage "$1"
  fi
  printf '0'
}

write_stage_round() {
  local stage="$1" value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    _loop_event info text "WARN: refused invalid round value '$value' for $stage" stage "$stage"
    return 0
  fi
  printf '%s\n' "$value" > "$(stage_round_file "$stage")"
}

read_stage_status() {
  local f
  f="$(stage_state_file "$1")"
  if [[ -s "$f" ]]; then
    head -n1 "$f" | tr -d '[:space:]'
  else
    printf 'pending'
  fi
}

write_stage_status() {
  local stage="$1" status="$2"
  case "$status" in
    pending|in-progress|finished) ;;
    *)
      _loop_event info text "WARN: ignoring invalid stage status '$status' for $stage" stage "$stage"
      return 0
      ;;
  esac
  printf '%s\n' "$status" > "$(stage_state_file "$stage")"
}

# Absolute path to the per-round dedicated review status file. The review
# agent writes its two-line machine contract here (REVIEW_VERDICT /
# STAGE_STATUS); the loop reads these structured fields from this file
# instead of grepping the agent's free-form stdout. The stdout is
# stream-json with the agent's whole final message (prose + the two
# contract lines) packed into one JSON `text` field, so a substring grep
# used to mis-parse a contract string the agent merely *quoted* in its
# reasoning. A dedicated plain-text file restores real line anchoring.
review_status_file() {
  local stage="$1" round="$2"
  printf '%s/%s_round_%s_review_status.txt' "$LOG_DIR" "$stage" "$round"
}

# Echo the value of a single contract key (REVIEW_VERDICT | STAGE_STATUS)
# from the dedicated review status file. Anchored full-line match only,
# so a quoted occurrence inside any stray prose line is ignored; the last
# well-formed line wins. Empty output if the file is absent or the key
# has no well-formed line.
read_review_contract() {
  local file="$1" key="$2"
  [[ -f "$file" ]] || return 0
  grep -oE "^${key}:[[:space:]]*[A-Za-z_-]+[[:space:]]*\$" "$file" 2>/dev/null \
    | tail -1 \
    | sed -E "s/^${key}:[[:space:]]*//; s/[[:space:]]+\$//"
}

# Parse the STAGE_STATUS contract line from the dedicated review status
# file; emit one of pending / in-progress / finished. Any other shape is
# treated as in-progress to avoid silently flipping to finished on a
# malformed or missing review status file.
parse_review_stage_status() {
  local file="$1"
  local raw
  raw=$(read_review_contract "$file" STAGE_STATUS)
  case "$raw" in
    finished|in-progress|pending) printf '%s' "$raw" ;;
    *) printf 'in-progress' ;;
  esac
}

# ── Per-stage milestone state machine (Stage 1 only) ─────────────────
# Stage 1 progresses through the linear named milestones injected via
# FORGE_MILESTONE_ORDER. The dev agent declares a
# milestone passed by adding `MILESTONE_STATUS: <name> PASS` to
# its commit message; advance_stage_milestone scans recent commits
# after every round and writes the highest declared milestone into
# <state_dir>/<stage>.milestone. Stages without a milestone manifest
# (currently stage2) skip the advance path entirely.

# True iff the given stage has a milestone manifest declared in
# tools/agent_loop_config.py::_STAGE_DIR_MANIFEST_MILESTONES. Stages
# without one (currently stage2) must NOT have any milestone argument
# threaded into build_prompt / get_stage_rule_files — the Python
# guardrail at stage_rule_files() raises on a milestone for a
# manifest-less stage. Use this helper to guard every milestone-aware
# code path so stage1 → stage2 transition is unattended.
stage_has_milestones() {
  local stage="$1"
  local mls
  mls=$("$PYTHON" "$WORKSPACE/tools/agent_loop_config.py" \
          stage-milestones "$stage" 2>/dev/null || true)
  [[ -n "$mls" ]]
}

read_stage_milestone() {
  # Echo the active milestone, defaulting to the first milestone in the
  # order if no file exists. Garbage contents are reset to that first
  # milestone with a warning routed through the loop event stream (info
  # subtype) so it's visible in the chat.
  local stage="$1"
  local f
  f="$(stage_milestone_file "$stage")"
  local warn
  local value
  value=$(read_stage_milestone_file "$f" 2>/tmp/_milestone_warn_$$ || true)
  if [[ -s /tmp/_milestone_warn_$$ ]]; then
    warn=$(cat /tmp/_milestone_warn_$$)
    _loop_event info text "$warn" stage "$stage"
  fi
  rm -f /tmp/_milestone_warn_$$
  printf '%s' "$value"
}

write_stage_milestone() {
  local stage="$1" value="$2"
  local f
  f="$(stage_milestone_file "$stage")"
  if ! write_stage_milestone_file "$f" "$value" 2>/dev/null; then
    _loop_event info \
      text "WARN: refused invalid milestone '$value' for $stage" \
      stage "$stage"
    return 1
  fi
}

# ── long-horizon auto-check (smoke cadence → harness-run full gate) ──
# The long-horizon advance depends entirely on review-side evidence, so
# the wrapper guarantees that evidence flows: after every long-horizon dev
# round, the cadence screen (tools/mfu_elastic_check.py --screen) fires
# once mfu_full_every_smokes smoke runs have accumulated since the newest
# full run, and the wrapper then runs the full long-train gate itself.
# The review agent evaluates the fresh evidence this same round and may
# issue MILESTONE_OVERRIDE (see review_stage1.md). The cadence is
# unconditional — its timing carries no information about the blinded
# policy. Disable with FORGE_LH_AUTOCHECK=0.
maybe_autorun_long_train() {
  local stage="$1" round="$2" milestone="$3"
  [[ "${FORGE_LH_AUTOCHECK:-1}" == "1" ]] || return 0
  [[ "$milestone" == "long-horizon" ]] || return 0
  local screen_ec=0
  ( cd "$WORKSPACE" && "$PYTHON" tools/mfu_elastic_check.py --screen ) \
    >/dev/null 2>&1 || screen_ec=$?
  # 1 = below screen threshold / full evidence already fresh; 2 = policy or
  # layout unavailable. Both mean: nothing to do this round.
  (( screen_ec == 0 )) || return 0
  _loop_event lh_autocheck \
    stage "$stage" round "$round" action full_long_train \
    text "long-horizon auto-check: screen tripped — running the full long-train gate harness-side."
  # GPU suites must follow the topology: local runs use `bin/harness run`;
  # any remote kind (ssh/devspace/job) goes through sync push +
  # tools/remote_run.sh (the harness never auto-SSHes from `run`). A bare
  # `bin/harness run` on a GPU-less controller fails identically every
  # round (loop 9f03ffd324af rounds 7–13: same 372-byte "requires a GPU"
  # error), so the evidence this check exists to produce never appears.
  local run_ec=0 _remote_kind=""
  _remote_kind=$("$PYTHON" - "${FORGE_CONFIG_DIR:-}/remote.toml" <<'PY' 2>/dev/null || true
import sys, tomllib
try:
    print(tomllib.load(open(sys.argv[1], "rb")).get("remote", {}).get("kind", "local"))
except Exception:
    print("local")
PY
)
  if [[ -z "$_remote_kind" || "$_remote_kind" == "local" ]]; then
    ( cd "$WORKSPACE" && bin/harness run long-train ) \
      >> "$LOG_DIR/${stage}_round_${round}_lh_autocheck.log" 2>&1 || run_ec=$?
  else
    ( cd "$WORKSPACE" && bin/harness sync push \
        && bash tools/remote_run.sh long-train ) \
      >> "$LOG_DIR/${stage}_round_${round}_lh_autocheck.log" 2>&1 || run_ec=$?
    # The remote runner appends its telemetry to the REMOTE copy of
    # mfu_history.jsonl; sync_push's built-in pull-back is what brings it
    # home. Run one more push NOW so this same round's review (whose
    # mfu_elastic_check reads the LOCAL per-loop history) sees the fresh
    # full-run row instead of waiting for the next push a round later.
    ( cd "$WORKSPACE" && bin/harness sync push ) \
      >> "$LOG_DIR/${stage}_round_${round}_lh_autocheck.log" 2>&1 || true
  fi
  _loop_event lh_autocheck_result \
    stage "$stage" round "$round" exit_code "$run_ec" \
    text "long-horizon auto-check: full long-train finished (exit ${run_ec}); review evaluates the fresh evidence this round."
  return 0
}

advance_stage_milestone() {
  # Scan recent commits in $WORKSPACE for MILESTONE_STATUS: <name> PASS
  # declarations and advance the persisted milestone to the successor (in
  # FORGE_MILESTONE_ORDER), capped at the last entry -- "<name> PASS" means
  # the gate for <name> has passed, so the next active milestone is the one
  # after it. Emits a milestone_advanced event only when the value actually
  # changes. No-op for stages without a milestone manifest.
  #
  # Milestones listed in FORGE_REVIEW_GATED_MILESTONES (default:
  # long-horizon) advance ONLY via the review-issued MILESTONE_OVERRIDE
  # status line (the per-round throughput check on harness-written
  # telemetry — see review_stage1.md). Dev commit declarations are retired
  # for gated milestones: their acceptance policy lives only on the review
  # side (the dev agent is deliberately blind to it), so a declaration —
  # fresh or historical — is never honored for them.
  local stage="$1" round="$2" review_passed="${3:-1}" round_base_sha="${4:-}"
  stage_has_milestones "$stage" || return 0

  local current
  current=$(read_stage_milestone "$stage")

  # Review-issued override: at a review-gated milestone the review agent
  # may advance the CURRENT milestone without a dev declaration by writing
  # `MILESTONE_OVERRIDE: <milestone>` (single token, the active milestone
  # name) into its status file — the per-round throughput check verified a
  # harness-written crossing (see review_stage1.md). Honored only for the
  # active milestone, only when it is review-gated, and only when this
  # round's review PASSed.
  if (( review_passed == 1 )) && [[ -n "$current" ]]; then
    local _override
    # `|| true`: the override line is absent on normal rounds, and the
    # grep inside read_review_contract exits 1 then — without the guard,
    # `set -euo pipefail` kills the whole loop right here.
    _override=$(read_review_contract "$(review_status_file "$stage" "$round")" MILESTONE_OVERRIDE || true)
    if [[ -n "$_override" && "$_override" == "$current" ]]; then
      local _gated_o
      # shellcheck disable=SC2086
      for _gated_o in ${FORGE_REVIEW_GATED_MILESTONES:-}; do
        [[ "$current" == "$_gated_o" ]] || continue
        local next_o
        next_o=$(next_milestone_after "$current" "$current")
        [[ -n "$next_o" ]] || break
        write_stage_milestone "$stage" "$next_o" || return 0
        local max_milestone_o
        # shellcheck disable=SC2086
        max_milestone_o=$(printf '%s\n' $FORGE_MILESTONE_ORDER | tail -n1)
        _loop_event milestone_auto_advanced \
          stage "$stage" round "$round" \
          from "$current" to "$next_o" max "$max_milestone_o" \
          text "review-issued MILESTONE_OVERRIDE advanced '${current}' (per-round throughput check; no dev declaration)."
        return 0
      done
    fi
  fi

  local highest
  highest=$(git -C "$WORKSPACE" log -n 50 --format=%B 2>/dev/null \
             | parse_commit_milestone_pass)
  [[ -z "$highest" ]] && return 0

  local next
  next=$(next_milestone_after "$highest" "$current")
  [[ -z "$next" ]] && return 0

  local _gated_m is_gated=0
  # Intentional word-splitting: space-separated milestone list.
  # shellcheck disable=SC2086
  for _gated_m in ${FORGE_REVIEW_GATED_MILESTONES:-}; do
    [[ "$highest" == "$_gated_m" ]] && { is_gated=1; break; }
  done
  if (( is_gated == 1 )); then
    # Dev declarations are RETIRED for review-gated milestones: the only
    # advance path is the review-issued MILESTONE_OVERRIDE handled above
    # (per-round throughput check on harness-written evidence). A commit
    # declaring a gated milestone carries no gating meaning — withhold
    # unconditionally so a stale or optimistic declaration can never
    # bypass the review's evidence-based decision.
    _loop_event milestone_advance_withheld \
      stage "$stage" round "$round" milestone "$highest" \
      review_passed "$review_passed" \
      text "milestone '${highest}' is review-gated: dev declarations are retired; it advances only via the review-issued MILESTONE_OVERRIDE."
    return 0
  fi

  # The commit whose body actually carries the just-passed declaration;
  # surfaced via the loop event for downstream attribution.
  local commit_sha
  commit_sha=$(git -C "$WORKSPACE" log -n 50 -E \
                 --grep="^MILESTONE_STATUS:[[:space:]]+${highest}[[:space:]]+PASS[[:space:]]*$" \
                 --format='%H' 2>/dev/null | head -n1)

  write_stage_milestone "$stage" "$next" || return 0
  # `max` reflects the actual ceiling enforced by
  # tools/agent_loop_milestone.sh::next_milestone_after (caps at the last
  # entry of FORGE_MILESTONE_ORDER) -- i.e. the terminal milestone. Derived
  # from the order SSOT so the dashboard's progress chip never reports a
  # stale ceiling when the milestone_order list changes.
  local max_milestone
  # shellcheck disable=SC2086
  max_milestone=$(printf '%s\n' $FORGE_MILESTONE_ORDER | tail -n1)
  _loop_event milestone_advanced \
    stage "$stage" round "$round" \
    from "$current" to "$next" max "$max_milestone" commit "$commit_sha"
}

# ── Per-stage rule file mapping ──────────────────────────────────────
# Python helper owns stage/rule/review/coding-guidelines.md mapping so shell stays a thin runner.
get_stage_rule_files() {
  local stage="$1" milestone="${2:-}"
  if [[ -n "$milestone" ]]; then
    $PYTHON "$WORKSPACE/tools/agent_loop_config.py" stage-rule-files "$stage" "$milestone"
  else
    $PYTHON "$WORKSPACE/tools/agent_loop_config.py" stage-rule-files "$stage"
  fi
}

get_stage_review_template() {
  local stage="$1"
  $PYTHON "$WORKSPACE/tools/agent_loop_config.py" review-template "$stage"
}

get_common_review_template() {
  $PYTHON "$WORKSPACE/tools/agent_loop_config.py" common-review-template
}

get_coding_guidelines_file() {
  $PYTHON "$WORKSPACE/tools/agent_loop_config.py" coding-guidelines
}

# ── Build prompt from stage-specific rule files ──────────────────────
build_prompt() {
  local stage="$1" milestone="${2:-}"
  cat <<HEADER
You are an autonomous developer agent operating inside a workspace that
exposes a \`harness\` CLI as the canonical command surface. Your workspace
is mounted read/write at:

  ${WORKSPACE}

Your active per-loop config dir (read-only; the seven frozen TOML axes
\`[ref]/[data]/[remote]/[agent]/[eval]/[model]/[optim]\` that are this
loop's configuration SSOT) is at:

  ${FORGE_CONFIG_DIR}

Whenever a standing rule file refers to \`@@FORGE_CONFIG_DIR@@/<axis>.toml\`,
that placeholder has already been substituted (at prompt-injection time)
to the absolute path above — read those files directly with your usual
file-read tool.

Current stage: **${stage}**
HEADER

  if [[ -n "$milestone" ]]; then
    cat <<MILESTONE_HEADER
Current milestone: **${milestone}** (only this milestone's MD is
included in the standing rule files below; prior and future milestones
are intentionally omitted so you focus on the active one).

### How to advance the milestone

When you have evidence in workload/notes/perf_log.md that ${milestone}'s
gate has passed, add the literal line below to your next commit's
message (mirrors the existing \`STAGE_STATUS: finished\` self-declaration
pattern):

    MILESTONE_STATUS: ${milestone} PASS

The loop will detect this on the next round, advance the active
milestone, and inject the next milestone's MD instead. Until you write
that line, the loop will keep pinning the prompt to ${milestone}.

MILESTONE_HEADER
  fi

  cat <<HEADER
Use the harness CLI to discover what this workload exposes — the harness
itself is workload-agnostic, so do NOT assume any specific suite, model,
baseline, or milestone naming from prior context:

  bin/harness info           # workload metadata + the full list of supported suites
  bin/harness doctor         # environment / GPU check
  bin/harness run <suite>    # run a specific suite (names come from \`bin/harness info\`)
  bin/harness run guard      # framework-import guard (local)
  bin/harness run anti-proxy # ref/harness-proxy lint (local; dev-side hard gate)

The dispatcher derives suite groups from \`config/eval.toml\`. Use
\`bin/harness info --json\` to inspect the current suite list for each stage.

You are working on **${stage}**. Focus ONLY on the suites and milestones
defined for this stage in the standing rule files below. Do not work on
suites belonging to other stages.

All workload-specific knowledge lives in the standing rule files below
and in \`config/eval.toml\`. Treat those, together with \`harness info\`,
as the single source of truth. If the rule files contradict any habit
or example you remember from elsewhere, the rule files win.

Work autonomously; do not ask for confirmation. Commit your changes when you finish.

HEADER

  # Optional [remote] plugin overlay: when `[remote].kind` is `ssh` or
  # `devspace` (REMOTE_ENABLED=true), inject the SSH remote-execution
  # prompt overlay between the HEADER and the coding-guidelines / stage
  # rules. The overlay's @@KEY@@ placeholders are substituted with
  # REMOTE_* values exported by tools/agent_loop_config.py. For
  # `kind = "devspace"` the devspace-only lifecycle overlay (drop
  # recovery + lease rebind) is appended too; ssh hosts have no
  # lifecycle so it is omitted. When `kind = "local"` (or missing) both
  # blocks are no-ops and the agent treats every command as local.
  local remote_overlay="$WORKSPACE/prompt/develop_prompt/remote-execution.md"
  local devspace_overlay="$WORKSPACE/prompt/develop_prompt/remote-execution-devspace.md"
  local job_overlay="$WORKSPACE/prompt/develop_prompt/remote-execution-job.md"
  if [[ "${REMOTE_ENABLED:-false}" == "true" && -f "$remote_overlay" ]]; then
    printf '# Source: %s (injected because [remote].kind = %s)\n\n' \
      "$(basename "$remote_overlay")" "${REMOTE_KIND:-ssh}"
    sed \
      -e "s|@@REMOTE_SSH_HOST@@|${REMOTE_SSH_HOST}|g" \
      -e "s|@@REMOTE_WORKDIR@@|${REMOTE_WORKDIR}|g" \
      -e "s|@@REMOTE_LOOP_ID@@|${REMOTE_LOOP_ID}|g" \
      -e "s|@@REMOTE_CONFIG_DIR@@|${REMOTE_CONFIG_DIR}|g" \
      "$remote_overlay"
    printf '\n\n'
    # Devspace lifecycle overlay: both kind=devspace (held GPU box) and
    # kind=job (0-GPU gateway) provision a cctl devspace with the same
    # create/rebind/release lifecycle, so the recovery contract applies to
    # both.
    if [[ ( "${REMOTE_KIND:-}" == "devspace" || "${REMOTE_KIND:-}" == "job" ) \
          && -f "$devspace_overlay" ]]; then
      printf '# Source: %s (injected because [remote].kind = %s)\n\n' \
        "$(basename "$devspace_overlay")" "${REMOTE_KIND}"
      sed \
        -e "s|@@REMOTE_SSH_HOST@@|${REMOTE_SSH_HOST}|g" \
        -e "s|@@REMOTE_WORKDIR@@|${REMOTE_WORKDIR}|g" \
        -e "s|@@REMOTE_LOOP_ID@@|${REMOTE_LOOP_ID}|g" \
        -e "s|@@REMOTE_CONFIG_DIR@@|${REMOTE_CONFIG_DIR}|g" \
        "$devspace_overlay"
      printf '\n\n'
    fi
    # Job overlay: appended for kind=job, where the devspace is a 0-GPU
    # filesystem gateway and GPU suites run as ephemeral cctl jobs. It
    # tells the agent to route GPU suites through tools/remote_run.sh
    # (never a bare ssh-run on the GPU-less gateway).
    if [[ "${REMOTE_KIND:-}" == "job" && -f "$job_overlay" ]]; then
      printf '# Source: %s (injected because [remote].kind = job)\n\n' \
        "$(basename "$job_overlay")"
      sed \
        -e "s|@@REMOTE_SSH_HOST@@|${REMOTE_SSH_HOST}|g" \
        -e "s|@@REMOTE_WORKDIR@@|${REMOTE_WORKDIR}|g" \
        -e "s|@@REMOTE_LOOP_ID@@|${REMOTE_LOOP_ID}|g" \
        -e "s|@@REMOTE_CONFIG_DIR@@|${REMOTE_CONFIG_DIR}|g" \
        "$job_overlay"
      printf '\n\n'
    fi
  fi

  # @@FORGE_CONFIG_DIR@@ in coding-guidelines / stage rule files is
  # substituted to the absolute per-loop config dir path so the agent
  # can read those frozen TOML files directly via its file-read tool.
  # Mirrors the @@REMOTE_*@@ substitution mechanism for the remote
  # overlay above; both rely on FORGE_CONFIG_DIR being exported by the
  # bootstrap block (line ~231) before build_prompt is ever called.
  local guidelines_file
  guidelines_file=$(get_coding_guidelines_file)
  if [[ -f "$guidelines_file" ]]; then
    printf '# Source: %s\n\n' "$(basename "$guidelines_file")"
    sed -e "s|@@FORGE_CONFIG_DIR@@|${FORGE_CONFIG_DIR}|g" "$guidelines_file"
    printf '\n\n'
  fi

  local rule_files
  rule_files=$(get_stage_rule_files "$stage" "$milestone")
  while IFS= read -r f; do
    if [[ -f "$f" ]]; then
      printf '# Source: %s\n\n' "$(basename "$f")"
      sed -e "s|@@FORGE_CONFIG_DIR@@|${FORGE_CONFIG_DIR}|g" "$f"
      printf '\n\n'
    fi
  done <<< "$rule_files"
}

# ── Unified spawn helper ─────────────────────────────────────────────
# Every cursor-cli / claude subprocess this loop starts goes through
# harness/tools/spawn_managed_agent.py. The helper registers a Session
# under $FORGE_AGENTS_DIR/<id>/, pumps stdout/stderr into that
# directory's stdout.log (real stream-json the frontend chat renderer
# understands), and exits with the backend CLI's exit code so this
# script's transient-failure detection / retry logic keeps working
# verbatim. The helper prints AGENT_ID=<id> on its own stdout once
# system.init lands; we capture that into _LAST_AGENT_ID so the caller
# can drive --resume on subsequent attempts within the same round and
# emit ``spawn_child`` wrapper events.
_LAST_AGENT_ID=""
_spawn_child_agent() {
  local kind="$1" prompt_file="$2" resume_agent_id="${3:-}" timeout_s="${4:-0}"
  local stage="${5:-}" round_no="${6:-}" seq="${7:-}"
  local model
  case "$kind" in
    loop_review) model="$REVIEW_MODEL" ;;
    *) model="$MODEL" ;;
  esac

  local spawn_args=(
    --backend "$AGENT_BACKEND"
    --model "$model"
    --workspace "$WORKSPACE"
    --prompt-file "$prompt_file"
    --kind "$kind"
    --loop-id "$LOOP_ID"
    --parent-agent "loop-$LOOP_ID"
  )
  # Forest nesting + ordering: forward stage/round/seq only when the caller
  # supplied all three (dev/review rounds do). spawn_managed_agent.py then
  # mints an ordered composite id that nests under the loop wrapper at
  # agents/<stage>/r<RRR>/; absent (plain chat) -> flat web-<uuid>. See
  # docs/forest-agent-log-layout.md.
  if [[ -n "$stage" && -n "$round_no" && -n "$seq" ]]; then
    spawn_args+=(--stage "$stage" --round "$round_no" --seq "$seq")
  fi
  if [[ -n "$resume_agent_id" ]]; then
    spawn_args+=(--resume "$resume_agent_id")
  fi
  # Per-round wall-clock cap: when >0, spawn_managed_agent.py kills the
  # agent's process group on expiry and exits 124 so the dev round can
  # roll itself back. Review spawns pass 0 (uncapped — they're bounded by
  # their own per-suite timeouts and finish in seconds).
  if [[ "$timeout_s" =~ ^[0-9]+$ ]] && (( timeout_s > 0 )); then
    spawn_args+=(--timeout-s "$timeout_s")
  fi
  if [[ "$MAX_MODE" == "true" ]]; then
    spawn_args+=(--max-mode)
  fi
  if [[ -n "$EFFORT" ]]; then
    spawn_args+=(--effort "$EFFORT")
  fi
  # Opus 4.8 logs no chain-of-thought unless the request asks for
  # `display: summarized`; SSOT is [agent].thinking_display (empty for
  # models that emit thinking by default, e.g. Fable 5).
  if [[ -n "${THINKING_DISPLAY:-}" ]]; then
    spawn_args+=(--thinking-display "$THINKING_DISPLAY")
  fi
  case "$AGENT_BACKEND" in
    cursor-cli) spawn_args+=(--api-key-env CURSOR_API_KEY) ;;
    claude-code) spawn_args+=(--api-key-env ANTHROPIC_API_KEY) ;;
    codex) spawn_args+=(--api-key-env "${CODEX_API_KEY_ENV:-OPENAI_API_KEY}") ;;
  esac

  local stdout_tmp; stdout_tmp="$(mktemp)"
  set +e
  "$PYTHON" "$FORGE_SOURCE_ROOT/harness/tools/spawn_managed_agent.py" \
    "${spawn_args[@]}" > "$stdout_tmp"
  local rc=$?
  # Leave errexit exactly as the caller set it. Dev/review callers wrap
  # this function in `set +e` so they can inspect rc and run retry logic;
  # re-enabling `set -e` here would make `return $rc` abort the wrapper
  # before the outer retry branch sees non-zero exits.

  _LAST_AGENT_ID=""
  if [[ -s "$stdout_tmp" ]]; then
    _LAST_AGENT_ID="$(awk -F= '/^AGENT_ID=/ {print $2; exit}' "$stdout_tmp" | tr -d '[:space:]')"
  fi
  rm -f "$stdout_tmp"

  # spawn_child is now emitted by spawn_managed_agent.py immediately
  # after spawn_session returns (before monitor_exit blocks). This makes
  # the event visible in the wrapper's stdout.log while the child is
  # still running, so /active and the Follow Live SSE scheduler can
  # discover it.
  return $rc
}

# Retry-decision helper. See harness/tools/agent_loop_retry.sh for the
# full rationale; in short: retry by default, only refuse for explicit
# user-intent signals (130/143/137) and our spawn-helper's own config
# errors (64/78). Sourced rather than inlined so the policy lives in
# one place and can be unit-tested standalone.
# shellcheck source=tools/agent_loop_retry.sh
source "$WORKSPACE/tools/agent_loop_retry.sh"

# Build a one-line human summary of why the dev agent exited. Used by
# the diagnostic info events around the retry decision so a post-mortem
# can see what the wrapper saw without needing to dig through the dev
# agent's stdout.log. Best-effort: never fails (returns the empty
# string if no log file is readable).
_classify_agent_failure() {
  local agent_id="$1"
  local rc="$2"
  case "$rc" in
    130) printf '%s' "interrupted (SIGINT, rc=130)"; return 0 ;;
    143) printf '%s' "terminated (SIGTERM, rc=143)"; return 0 ;;
    137) printf '%s' "killed (SIGKILL, rc=137)"; return 0 ;;
    64)  printf '%s' "spawn_managed_agent EX_USAGE (rc=64)"; return 0 ;;
    75)  printf '%s' "spawn_managed_agent EX_TEMPFAIL (rc=75; CLI failed to reach system.init - transient)"; return 0 ;;
    78)  printf '%s' "spawn_managed_agent EX_CONFIG (rc=78)"; return 0 ;;
  esac
  local log_file="$FORGE_AGENTS_DIR/$agent_id/stdout.log"
  if [[ -n "$agent_id" && -f "$log_file" ]]; then
    local last_line
    last_line="$(tail -n1 "$log_file" 2>/dev/null)"
    last_line="${last_line:0:200}"
    if [[ -n "$last_line" ]]; then
      printf 'rc=%s; last log line: %s' "$rc" "$last_line"
      return 0
    fi
  fi
  printf 'rc=%s; no diagnostic available' "$rc"
}

# ── Review agent (stage-aware) ───────────────────────────────────────
build_review_prompt() {
  local stage="$1"
  local round="$2"
  local milestone="${3:-}"
  local commit_msg
  commit_msg=$(cd "$WORKSPACE" && git log -1 --format='%H%n%an <%ae>%n%ad%n%n%s%n%n%b' 2>/dev/null || echo "(no commit)")

  local common_review_template review_template
  common_review_template=$(get_common_review_template)
  review_template=$(get_stage_review_template "$stage")
  if [[ -f "$common_review_template" && -f "$review_template" ]]; then
    cat "$common_review_template"
    printf '\n\n'
    cat "$review_template"
  else
    echo "ERROR: review prompt template missing at $common_review_template or $review_template" >&2
    return 1
  fi
  printf '\n'

  # Resolve the active backend so the review prompt points the agent
  # at the right per-backend stage rule file (the split into
  # ``{torch,megatron}/`` subfolders is selected by
  # ``[ref].backend`` in config/ref.toml).
  local active_backend
  active_backend=$($PYTHON -c '
import sys
sys.path.insert(0, "'"$WORKSPACE"'")
from harness import config_runtime
_, data = config_runtime.load_workload_config(None)
print((data.get("defaults", {}) or {}).get("backend", "megatron"))
' 2>/dev/null) || active_backend="megatron"

  printf '## Commit under review\n\n```\n%s\n```\n\n' "$commit_msg"
  printf '> Diff and project rules are not inlined. Use `git diff HEAD~1 HEAD`\n'
  printf '> (or the per-check commands above) to inspect changes, and read\n'
  printf '> rule files (`prompt/develop_prompt/%s/%s.md`, `prompt/project-guide.md`,\n' "$active_backend" "$stage"
  printf '> `prompt/review_prompt/coding-guidelines.md`) directly\n'
  printf '> when you need the standing rules.\n'

  # Loop-supplied context. The review agent needs `stage` and `round`
  # to fill the "## Persisted review note" section header in
  # `workload/notes/review.md` (review_common.md contract).
  # NB: format strings start with `-`, so use `%s` to avoid bash's
  # printf treating them as flags.
  printf '\n## Review context (loop-supplied)\n\n'
  printf '%s\n' "- stage: \`$stage\`"
  printf '%s\n' "- round: \`$round\`"
  if [[ -n "$milestone" ]]; then
    printf '%s\n' "- milestone: \`$milestone\`"
  fi

  # Carry-over GPU job(s): the previous dev round may have ended with a cctl
  # job still in flight (no wrapper-side poller collects it). Surface the
  # unconsumed suite name(s) so the review agent can nudge the NEXT dev to
  # collect the verdict via `remote_run.sh --poll <suite>`. NOT a FAIL — see
  # review_common.md "Carry-over GPU job".
  local _carry_over
  _carry_over="$(_unconsumed_gpu_suites)"
  if [[ -n "$_carry_over" ]]; then
    printf '%s\n' "- carry_over_jobs: \`$_carry_over\`"
  fi

  # Machine verdict sink. The loop reads ONLY this file for the verdict —
  # NOT your stdout — so the two contract lines must land here verbatim.
  local status_file
  status_file=$(review_status_file "$stage" "$round")
  printf '%s\n' "- review_status_file: \`$status_file\`"
  printf '\n%s\n' "Before you finish, use the file-write tool to write your machine verdict to the absolute \`review_status_file\` path above. That file must contain EXACTLY the two contract lines below — plus, ONLY when review_stage1.md's long-horizon per-round throughput check instructs it, one optional third line \`MILESTONE_OVERRIDE: <milestone>\` — and nothing else (no prose, no fences, no quoting):"
  printf '\n%s\n' '    REVIEW_VERDICT: PASS|FAIL'
  printf '%s\n' '    STAGE_STATUS: finished|in-progress|pending'
  printf '\n%s\n' "The loop parses these two structured fields from the file alone; if the file is missing or malformed it falls back to REVIEW_VERDICT: FAIL / STAGE_STATUS: in-progress. The verdict you print to stdout is for humans only and is no longer parsed."
}

run_review_agent() {
  local round="$1"
  local stage="$2"
  local milestone="${3:-}"
  local review_prompt_path="$LOG_DIR/${stage}_round_${round}_review_prompt.md"

  _loop_event info text "Review agent starting (${stage} round ${round})"

  build_review_prompt "$stage" "$round" "$milestone" > "$review_prompt_path"

  set +e
  # seq=2: review is the second spawn of the round (dev=1).
  _spawn_child_agent loop_review "$review_prompt_path" "" 0 "$stage" "$round" 2
  local review_ec=$?
  set -e
  local review_agent_id="$_LAST_AGENT_ID"

  if (( review_ec != 0 )) || [[ -z "$review_agent_id" ]]; then
    _loop_event review_verdict \
      stage "$stage" round "$round" verdict FAIL exit_code "$review_ec"
    _loop_event info text "Review agent crashed (exit ${review_ec}), treating as FAIL"
    return 1
  fi

  # Verdict comes from the dedicated review status file the agent wrote,
  # not its stdout. A missing/malformed file (agent never wrote it, or
  # wrote garbage) is treated as FAIL — the safe default.
  local status_file verdict
  status_file=$(review_status_file "$stage" "$round")
  verdict=$(read_review_contract "$status_file" REVIEW_VERDICT)
  case "$verdict" in
    PASS)
      _loop_event review_verdict \
        stage "$stage" round "$round" verdict PASS agent_id "$review_agent_id"
      return 0 ;;
    FAIL)
      _loop_event review_verdict \
        stage "$stage" round "$round" verdict FAIL agent_id "$review_agent_id"
      return 1 ;;
    *)
      _loop_event review_verdict \
        stage "$stage" round "$round" verdict FAIL agent_id "$review_agent_id" \
        text "no well-formed REVIEW_VERDICT line in review status file"
      return 1 ;;
  esac
}

# ── Main loop: goal-driven, status-machine across Stage 1 → Stage 2 ──
#
# Per-round flow:
#   1. dev agent runs (with the existing transient-failure retry loop)
#   2. review agent runs and emits two contract lines:
#        REVIEW_VERDICT: PASS | FAIL
#        STAGE_STATUS:   pending | in-progress | finished
#   3. STAGE_STATUS is persisted to <state_dir>/<stage>.status
#   4. Loop terminates this stage when:
#        * STAGE_STATUS == finished, OR
#        * round count reaches RUNS_PER_STAGE (when > 0), OR
#        * MAX_CONSECUTIVE_REVIEW_FAILS consecutive REVIEW_VERDICT=FAIL
# Failures are tracked with a consecutive counter so a single bad round
# doesn't abort a long-running goal-driven session, but persistent
# failures still surface promptly.
# Print the suite name(s) of any GPU-job handle under the per-loop gpu_jobs/
# dir that is still non-consumed — i.e. a cctl job the dev left in flight
# this round. Space-separated on one line; empty when nothing is pending.
# gpu_jobs/ lives at the loop root (parent of FORGE_CONFIG_DIR), beside
# mfu_history.jsonl. There is NO wrapper-side job poller / re-spawn: the dev
# round ends and review runs as usual; build_review_prompt reads this so the
# review agent can nudge the NEXT dev to collect the carry-over verdict.
_unconsumed_gpu_suites() {
  local dir; dir="$(dirname "$FORGE_CONFIG_DIR")/gpu_jobs"
  [[ -d "$dir" ]] || return 0
  "$PYTHON" - "$dir" <<'PY'
import glob, json, os, sys
suites = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    try:
        rec = json.load(open(f))
    except Exception:
        continue
    if isinstance(rec, dict) and rec.get("status") != "consumed":
        suites.append(str(rec.get("suite") or os.path.basename(f)[:-5]))
print(" ".join(suites))
PY
}

run_agent_stage() {
  local stage="$1"

  local cap_label
  if (( RUNS_PER_STAGE == 0 )); then
    cap_label="unlimited"
  else
    cap_label="cap=${RUNS_PER_STAGE}"
  fi

  local round
  round=$(read_stage_round "$stage")
  local consecutive_fails=0
  local stage_status
  stage_status=$(read_stage_status "$stage")
  if [[ "$stage_status" == "pending" ]]; then
    write_stage_status "$stage" "in-progress"
    stage_status="in-progress"
  fi

  while :; do
    round=$((round + 1))
    write_stage_round "$stage" "$round"

    if (( RUNS_PER_STAGE > 0 && round > RUNS_PER_STAGE )); then
      _loop_event info \
        text "${stage}: reached RUNS_PER_STAGE=${RUNS_PER_STAGE} cap without STAGE_STATUS=finished. Exiting stage." \
        stage "$stage"
      return 0
    fi

    # Rebuild the dev prompt every round so a milestone advance from
    # the previous round's commit takes effect immediately (the prompt
    # header pins the dev agent to the active milestone). PROMPT_PATH
    # is per-round so post-mortem can compare consecutive prompts.
    #
    # Stages without a milestone manifest (currently stage2) skip the
    # read entirely and feed an empty milestone to build_prompt /
    # get_stage_rule_files — both already short-circuit on the empty
    # string (build_prompt:833, get_stage_rule_files:800). Symmetric
    # with advance_stage_milestone's early-return.
    local current_milestone=""
    if stage_has_milestones "$stage"; then
      current_milestone=$(read_stage_milestone "$stage")
    fi
    PROMPT_PATH="$LOG_DIR/${stage}_round_${round}_prompt.md"
    build_prompt "$stage" "$current_milestone" > "$PROMPT_PATH"

    _loop_event round_start \
      stage "$stage" round "$round" cap "$cap_label" \
      milestone "$current_milestone"

    # Per-round wall-clock cap. Capture the pre-spawn HEAD as the rollback
    # baseline and derive the round deadline; both the spawn budget below
    # and the post-loop rollback key off these. ROUND_TIMEOUT_S=0 disables
    # the cap (round_deadline stays 0).
    local round_base_sha=""
    round_base_sha=$(git -C "$WORKSPACE" rev-parse HEAD 2>/dev/null || true)
    local round_deadline=0
    if (( ROUND_TIMEOUT_S > 0 )); then
      round_deadline=$(( $(date +%s) + ROUND_TIMEOUT_S ))
    fi
    local round_timed_out=0

    local dev_agent_id=""
    # Write the retry-continuation prompt to a file once per round; the
    # helper takes --prompt-file so we cannot pass it as an arg.
    local continue_prompt_file="$LOG_DIR/${stage}_round_${round}_continue_prompt.md"
    printf '%s' "$CURSOR_AGENT_RETRY_CONTINUE_PROMPT" > "$continue_prompt_file"

    attempt=1
    while (( attempt <= CURSOR_AGENT_ROUND_TRIES )); do
      # errexit disabled for the spawn (need to capture non-zero rc)
      # and should_retry_after_agent_failure (returns 1 for "give up").
      # _loop_event is internally infallible so it doesn't need set +e,
      # but the two exit-code captures do.
      set +e
      if (( attempt > 1 )); then
        _loop_event round_retry \
          stage "$stage" round "$round" attempt "$attempt"
      fi
      # Remaining round budget handed to this spawn. Retries share the
      # round's single deadline (the backoff sleeps below count against
      # it too), so the dev agent's cumulative wall time across attempts
      # never exceeds ROUND_TIMEOUT_S. A non-positive remainder means the
      # cap is already blown — treat it as a timeout without re-spawning.
      local round_spawn_timeout=0
      if (( round_deadline > 0 )); then
        round_spawn_timeout=$(( round_deadline - $(date +%s) ))
        if (( round_spawn_timeout <= 0 )); then
          round_timed_out=1
          break
        fi
      fi
      if (( attempt == 1 )); then
        _spawn_child_agent loop_dev_round "$PROMPT_PATH" "" "$round_spawn_timeout" "$stage" "$round" 1
      else
        _spawn_child_agent loop_dev_round "$continue_prompt_file" "$dev_agent_id" "$round_spawn_timeout" "$stage" "$round" 1
      fi
      agent_ec=$?
      [[ -n "$_LAST_AGENT_ID" ]] && dev_agent_id="$_LAST_AGENT_ID"

      if (( agent_ec == 124 )); then
        # EX_TIMEOUT: the round blew its wall-clock cap and the agent's
        # process group was already reaped by spawn_managed_agent.py. Not
        # a transient — stop retrying and fall through to the rollback.
        round_timed_out=1
        break
      fi

      if (( agent_ec == 0 )); then
        break
      fi

      failure_summary="$(_classify_agent_failure "$dev_agent_id" "$agent_ec")"
      retry_attempt_cap=$(( CURSOR_AGENT_ROUND_TRIES ))
      should_retry_after_agent_failure "$agent_ec"
      retry_verdict=$?

      if (( attempt < retry_attempt_cap )) && (( retry_verdict == 0 )); then
        sleep_s=$(( CURSOR_AGENT_RETRY_BASE_SLEEP * (1 << (attempt - 1)) ))
        if (( sleep_s > 120 )); then sleep_s=120; fi
        jitter=$((RANDOM % 3))
        _loop_event retry_decision \
          stage "$stage" round "$round" attempt "$attempt" \
          cap "$retry_attempt_cap" exit_code "$agent_ec" \
          action retry sleep_seconds "$((sleep_s + jitter))" \
          reason "$failure_summary"
        sleep $((sleep_s + jitter))
        attempt=$((attempt + 1))
        continue
      fi

      if (( retry_verdict != 0 )); then
        _loop_event retry_decision \
          stage "$stage" round "$round" attempt "$attempt" \
          cap "$retry_attempt_cap" exit_code "$agent_ec" \
          action give_up reason "$failure_summary"
      else
        _loop_event retry_decision \
          stage "$stage" round "$round" attempt "$attempt" \
          cap "$retry_attempt_cap" exit_code "$agent_ec" \
          action give_up \
          reason "attempt cap reached: $failure_summary"
      fi
      _loop_event info \
        text "agent exited with ${agent_ec} (${stage} round ${round}, ${failure_summary}); not retrying (attempt=${attempt}/${retry_attempt_cap})" \
        stage "$stage" round "$round"
      exit "$agent_ec"
    done
    set -e

    # Round wall-clock cap hit: the dev agent was killed mid-round. Roll
    # the workspace back to the pre-spawn baseline so a half-finished,
    # possibly broken commit never leaks into the next round (or into a
    # review). The next round then restarts cleanly from the last good
    # state. This is deliberately a hard reset of the per-loop throwaway
    # repo — preferred over salvaging partial work, per the round-cap
    # contract. Skip review/milestone/status for this round entirely.
    if (( round_timed_out == 1 )); then
      _loop_event round_timeout \
        stage "$stage" round "$round" cap_seconds "$ROUND_TIMEOUT_S" \
        text "round exceeded ${ROUND_TIMEOUT_S}s wall-clock cap; agent killed, rolling back."
      if [[ -n "$round_base_sha" ]]; then
        git -C "$WORKSPACE" reset --hard "$round_base_sha" >/dev/null 2>&1 || true
        git -C "$WORKSPACE" clean -fd >/dev/null 2>&1 || true
        _loop_event round_rollback \
          stage "$stage" round "$round" baseline "$round_base_sha" \
          text "workspace reset to pre-round HEAD; next round restarts from clean state."
      else
        _loop_event info \
          text "${stage} round ${round}: round timed out but no baseline SHA captured; skipping rollback." \
          stage "$stage" round "$round"
      fi
      sleep "$POLL_SECONDS"
      continue
    fi

    if (( agent_ec != 0 )); then
      _loop_event info \
        text "${stage} round ${round}: agent still failing after ${CURSOR_AGENT_ROUND_TRIES} attempt(s)." \
        stage "$stage" round "$round"
      exit "$agent_ec"
    fi

    # No wrapper-side GPU-job continuation guard: a cctl job the dev left in
    # flight is NOT polled or re-processed here — the round ends and review
    # runs as usual. build_review_prompt surfaces any carry-over job to the
    # review agent, which nudges the NEXT dev (via its perf_log one-liner) to
    # collect it with `tools/remote_run.sh --poll <suite>`. The persisted
    # handle + submit-refuses-while-in-flight (gpu_job.py) keep this safe from
    # duplicate submits; collecting the leftover verdict is the next dev's job.

    # long-horizon auto-check: maybe produce fresh full-gate evidence for
    # the review's per-round throughput check, so the milestone can advance
    # without waiting for a dev declaration (see maybe_autorun_long_train).
    maybe_autorun_long_train "$stage" "$round" "$current_milestone"

    review_passed=0
    if run_review_agent "$round" "$stage" "$current_milestone"; then
      consecutive_fails=0
      review_passed=1
    else
      consecutive_fails=$((consecutive_fails + 1))
      if (( MAX_CONSECUTIVE_REVIEW_FAILS > 0 && consecutive_fails >= MAX_CONSECUTIVE_REVIEW_FAILS )); then
        _loop_event info \
          text "${stage} round ${round}: REVIEW FAILED ${consecutive_fails}× in a row (cap ${MAX_CONSECUTIVE_REVIEW_FAILS}). Aborting." \
          stage "$stage" round "$round"
        exit 1
      fi
      _loop_event info \
        text "${stage} round ${round}: REVIEW FAILED (${consecutive_fails}/${MAX_CONSECUTIVE_REVIEW_FAILS}); continuing for self-heal." \
        stage "$stage" round "$round"
    fi

    # Read STAGE_STATUS from the dedicated review status file the review
    # agent wrote this round (same deterministic path build_review_prompt
    # handed it). Missing/malformed → safe default in-progress, handled
    # inside parse_review_stage_status.
    local review_status_path
    review_status_path=$(review_status_file "$stage" "$round")
    stage_status=$(parse_review_stage_status "$review_status_path")
    # Defence-in-depth: a FAIL'd review can never advance the stage to
    # `finished`, regardless of what STAGE_STATUS line the review agent
    # emitted. This enforces the review_common.md contract on the loop
    # side rather than trusting the review agent to self-police.
    if (( review_passed == 0 )) && [[ "$stage_status" == "finished" ]]; then
      _loop_event info \
        text "${stage} round ${round}: WARN — review FAILED but emitted STAGE_STATUS=finished; downgrading to in-progress (review_common.md contract)." \
        stage "$stage" round "$round"
      stage_status="in-progress"
    fi
    write_stage_status "$stage" "$stage_status"
    _loop_event stage_status \
      stage "$stage" round "$round" status "$stage_status" review_fails "$consecutive_fails"

    # Decoupled from stage_status: milestone advance is driven by the dev
    # agent's commit-message self-declaration, so a FAIL'd review can
    # still surface evidence the next round needs to splice in the next
    # milestone's MD — EXCEPT for review-gated milestones
    # (FORGE_REVIEW_GATED_MILESTONES, e.g. long-horizon), whose advance
    # requires this round's review to PASS a freshly declared commit
    # (see advance_stage_milestone).
    advance_stage_milestone "$stage" "$round" "$review_passed" "$round_base_sha"

    if [[ "$stage_status" == "finished" ]]; then
      _loop_event stage_done stage "$stage" round "$round"
      return 0
    fi

    # The production long-train is a multi-day run the dev agent only
    # babysits with a light status poll, so space its rounds out
    # (PRODUCTION_POLL_SECONDS, default 1800s) instead of spawning a fresh
    # dev+review every few minutes for days. Other milestones keep the
    # tight POLL_SECONDS cadence. FORGE_SLOW_POLL_MILESTONE is that
    # milestone, resolved from the production-train suite in eval.toml (empty
    # when the config ships no such suite, e.g. the 8b suite -> never
    # throttles), so the gate follows a milestone rename instead of a
    # hardcoded name. current_milestone is this round's starting milestone,
    # so the throttle engages the round after the stage enters it.
    if [[ -n "${FORGE_SLOW_POLL_MILESTONE:-}" \
          && "$current_milestone" == "$FORGE_SLOW_POLL_MILESTONE" ]]; then
      sleep "$PRODUCTION_POLL_SECONDS"
    else
      sleep "$POLL_SECONDS"
    fi
  done
}

# Wrapper-side bookkeeping echoes go to whatever stdout/stderr we were
# launched with (terminal for CLI, web router's loop debug fd for
# managed). The loop wrapper agent's stdout.log under
# $FORGE_AGENTS_DIR/loop-$LOOP_ID/ is the SSOT for typed events; these
# echoes are just an at-a-glance bash trace for ad hoc tail.
echo "=== Unified Harness Agent Loop (goal-driven, 2-stage pipeline) ==="
echo "workspace: $WORKSPACE"
echo "agent_backend: $AGENT_BACKEND"
echo "model: $MODEL"
echo "stages: ${STAGES[*]}"
if (( RUNS_PER_STAGE == 0 )); then
  echo "runs_per_stage: 0 (unlimited; loop stops on STAGE_STATUS=finished)"
else
  echo "runs_per_stage: $RUNS_PER_STAGE (hard cap)"
fi
echo "max_mode: $MAX_MODE"
echo "effort: ${EFFORT:-<default>}"
echo "thinking_display: ${THINKING_DISPLAY:-<default>}"
echo "max_consecutive_review_fails: $MAX_CONSECUTIVE_REVIEW_FAILS"
echo "state_dir: $AGENT_LOOP_STATE_DIR"
echo "stage2_max_concurrent: $STAGE2_MAX_CONCURRENT"
echo "stage2_max_op_long_failures: $STAGE2_MAX_OP_LONG_FAILURES"
echo "agents_dir: $FORGE_AGENTS_DIR"
echo ""

# ── Disk-based registry (SSOT) + best-effort web hint ────────────────
# The session.json under $FORGE_TRAIN_DIR/<loop_id>/ is the
# single source of truth: any agent-loop.sh process publishes its
# own status to disk, and the web server scans that directory.
# Neither side cares whether the other is currently online.
#
# The HTTP POST to /api/loop/register is a non-fatal "reload now"
# hint so the dashboard can react sub-second when it happens to be
# up. Its failure is invisible to the loop.
# LOOP_ID was already established right after LOG_DIR (so the wrapper
# init + _loop_event helpers could use it). Keep the re-export here for
# clarity and pick up the downstream label/started_at metadata.
export LOOP_ID
LOOP_STARTED_AT="$($PYTHON -c 'import time; print(time.time())')"
LOOP_LABEL="$(date '+%Y-%m-%d %H:%M:%S')"

_resolve_registry_dir() {
  if [[ -n "${FORGE_TRAIN_DIR:-}" ]]; then
    printf '%s' "$FORGE_TRAIN_DIR"
  else
    # When FORGE_TRAIN_PROVISIONED=1 was set by the bootstrap or by web,
    # SCRIPT_DIR resolves to <forge_train_dir>/<loop_id>/workspace, so
    # walking up two levels recovers the registry root.
    printf '%s' "$(cd "$SCRIPT_DIR/../.." && pwd)"
  fi
}

# _publish_session_json <status> [<exit_code>] [<ended_at>]
# Writes <registry>/<loop_id>/session.json atomically. Always uses
# tmp + rename so a partial write never appears to a concurrent
# scanner. Safe to call from a trap.
_publish_session_json() {
  local status="$1"
  local exit_code="${2:-}"
  local ended_at="${3:-}"
  local registry_dir
  registry_dir="$(_resolve_registry_dir)"
  local instance_dir="$registry_dir/$LOOP_ID"
  mkdir -p "$instance_dir" || return 0
  $PYTHON - "$instance_dir/session.json" "$LOOP_ID" "$LOOP_LABEL" \
           "$$" "$LOOP_STARTED_AT" "$status" "$exit_code" "$ended_at" \
           "$WORKSPACE" "$LOG_DIR" "${LOOP_OUTPUT_LOG:-}" \
           "$AGENT_BACKEND" "$MODEL" "${STAGES[*]}" "$MAX_MODE" \
           <<'PY' || return 0
import json, os, sys
(target, loop_id, label, pid, started_at, status, exit_code,
 ended_at, workspace, log_dir, output_file,
 backend, model, stages, max_mode) = sys.argv[1:16]
def _opt_num(v):
    if v == "":
        return None
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return None
data = {
    "loop_id": loop_id,
    "label": label,
    "mode": "external",
    "status": status,
    "pid": int(pid),
    "started_at": _opt_num(started_at),
    "ended_at": _opt_num(ended_at),
    "exit_code": _opt_num(exit_code),
    "args": {
        "backend": backend,
        "model": model,
        "stages": stages,
        "max_mode": max_mode,
    },
    "workspace_dir": workspace,
    "log_dir": log_dir or None,
    "output_file": output_file or None,
}
tmp = target + ".tmp." + str(os.getpid())
with open(tmp, "w", encoding="utf-8") as fh:
    json.dump(data, fh, ensure_ascii=False, indent=2)
os.replace(tmp, target)
PY
}

# Publish initial running record before doing anything else so a
# web server that starts up at any later moment can find this loop.
_publish_session_json running

# ── Trajectory-resume provenance (tools/fork_loop.py) ────────────────
# A forked loop carries forked_from.json next to session.json (NOT
# inside it — _publish_session_json rewrites that file wholesale on
# every start). Surface it once as a typed loop_forked event so the
# Loop tab can render "forked from <src> @ <milestone>"; the sentinel
# keeps later resumes of this loop from re-emitting on every restart.
_emit_loop_forked_once() {
  local instance_dir prov sentinel fields
  instance_dir="$(_resolve_registry_dir)/$LOOP_ID"
  prov="$instance_dir/forked_from.json"
  sentinel="$instance_dir/.loop_forked_emitted"
  [[ -f "$prov" && ! -f "$sentinel" ]] || return 0
  fields=$($PYTHON - "$prov" <<'PY' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(d.get("loop_id", ""), d.get("milestone", ""), d.get("anchor_sha", ""))
PY
  ) || return 0
  local src milestone sha
  read -r src milestone sha <<<"$fields"
  [[ -n "$src" ]] || return 0
  _loop_event loop_forked \
    src_loop_id "$src" milestone "$milestone" anchor_sha "$sha" \
    text "forked from ${src} @ milestone '${milestone}' (${sha:0:12})"
  : > "$sentinel"
}
_emit_loop_forked_once

# ── Wrapper-side remote MFU pull-back ────────────────────────────────
# When [remote].kind is ssh or devspace, the dev agent's `harness run` invocations
# execute on the SSH host and `harness/tools/mfu_record.py` writes
# `mfu_history.jsonl` on the *remote* filesystem. The web MFU badge
# reads the *local* copy under $FORGE_TRAIN_DIR/<loop_id>/, so without
# a wrapper-side pull the badge would never update on remote runs.
#
# We can't trust the dev agent to rsync this back -- in practice it
# forgets between rounds. Instead the wrapper runs a tiny background
# poller that rsyncs the telemetry artifact every 30 s. Failures are
# silent (-q): the loop's pass/fail signal must never depend on
# telemetry sync succeeding. ``--timeout=60`` is mandatory: the loop is
# serial, so a single rsync that hangs on a half-open ssh tunnel (e.g.
# after a devspace reclaim) would freeze every subsequent poll and the
# MFU badge would never update again -- the I/O watchdog lets the next
# iteration recover.
_REMOTE_SYNC_PID=""
if [[ "${REMOTE_ENABLED:-false}" == "true" && -n "${REMOTE_SSH_HOST:-}" \
      && -n "${REMOTE_WORKDIR:-}" && -n "${LOOP_ID:-}" ]]; then
  _local_loop_dir="$(_resolve_registry_dir)/$LOOP_ID"
  _remote_loop_dir="${REMOTE_WORKDIR}/.artifacts/forge_train/${LOOP_ID}"
  mkdir -p "$_local_loop_dir"
  (
    while true; do
      sleep 30
      rsync -aqz --timeout=60 \
        --include='mfu_history.jsonl' --exclude='*' \
        "${REMOTE_SSH_HOST}:${_remote_loop_dir}/" "${_local_loop_dir}/" \
        >/dev/null 2>&1 || true
      # The remote-emitted ``mfu_record`` loop_event landed in the REMOTE
      # wrapper stdout.log, not the local one the dashboard SSE reads. The
      # rsync above only brings back the history FILE, so re-emit a local
      # event per freshly-pulled entry (cursor-deduped) to feed the live
      # MFU badge. Best-effort: telemetry must never wedge the puller.
      "$PYTHON" "$FORGE_SOURCE_ROOT/harness/tools/mfu_backfill_events.py" \
        --loop-dir "$_local_loop_dir" --loop-id "$LOOP_ID" >/dev/null 2>&1 || true
    done
  ) &
  _REMOTE_SYNC_PID=$!
  echo "[remote-sync] background MFU puller pid=$_REMOTE_SYNC_PID (30s interval)"
fi

# Final-status writer: invoked from EXIT trap. Maps shell exit
# codes back to the loop status taxonomy the dashboard expects.
_loop_exit_handler() {
  local rc=$?
  local status
  local ended_at
  ended_at="$($PYTHON -c 'import time; print(time.time())')"
  if (( rc == 0 )); then
    status=completed
  elif (( rc == 130 || rc == 143 )); then
    # 130 = Ctrl-C (SIGINT), 143 = SIGTERM from /stop endpoint
    status=stopped
  else
    status=failed
  fi
  # Kill the wrapper-side remote MFU poller (if any) so it doesn't
  # outlive the loop and rsync against a dead remote forever.
  if [[ -n "${_REMOTE_SYNC_PID:-}" ]]; then
    kill "$_REMOTE_SYNC_PID" 2>/dev/null || true
    # One last opportunistic pull so the final mfu_history entry
    # from the just-finished run is captured before the wrapper
    # exits. ``--timeout=60`` keeps a dead/black-holed remote from
    # wedging the EXIT trap. Non-fatal on any error.
    if [[ "${REMOTE_ENABLED:-false}" == "true" && -n "${REMOTE_SSH_HOST:-}" \
          && -n "${REMOTE_WORKDIR:-}" ]]; then
      local _final_local
      _final_local="$(_resolve_registry_dir)/$LOOP_ID"
      local _final_remote="${REMOTE_WORKDIR}/.artifacts/forge_train/${LOOP_ID}"
      rsync -aqz --timeout=60 \
        --include='mfu_history.jsonl' --exclude='*' \
        "${REMOTE_SSH_HOST}:${_final_remote}/" "${_final_local}/" \
        >/dev/null 2>&1 || true
      # Emit the local ``mfu_record`` event for the final run's entry too,
      # so the badge reflects the last measurement before the wrapper exits.
      "$PYTHON" "$FORGE_SOURCE_ROOT/harness/tools/mfu_backfill_events.py" \
        --loop-dir "$_final_local" --loop-id "$LOOP_ID" >/dev/null 2>&1 || true
    fi
  fi
  _publish_session_json "$status" "$rc" "$ended_at"
  # Release the devspace lease this loop claimed during bootstrap. Only
  # `kind = "devspace"` claims one; ssh/local never do. Best-effort: a
  # missing reverse-lookup file is a no-op; cctl failures get swallowed
  # so they cannot block wrapper exit.
  # FORGE_SKIP_LEASE_RELEASE=1 (set by the canonical-preflight abort path)
  # keeps the leased box alive so a transient preflight failure does not
  # stop the machine the frozen hostname points at.
  if [[ -n "${LOOP_ID:-}" && "${REMOTE_KIND:-local}" == "devspace" \
        && "${FORGE_SKIP_LEASE_RELEASE:-0}" != "1" ]]; then
    "$PYTHON" -m tools.lease release devspace --loop-id "$LOOP_ID" \
      >/dev/null 2>&1 || true
  fi
  # Emit one final loop_event so the wrapper chat closes with a typed
  # row, then mark the wrapper Session terminal so the Agents tab and
  # the loop tab both stop showing it as running.
  if [[ -n "${LOOP_ID:-}" ]]; then
    _loop_event loop_exit state "$status" exit_code "$rc"
    "$PYTHON" -c "
import sys
sys.path.insert(0, '$FORGE_SOURCE_ROOT')
from web.agents import spawn
spawn.finalize_wrapper_session(
    loop_id='$LOOP_ID',
    exit_code=$rc,
    state='${status//\'/}',
)
" 2>/dev/null || true
  fi
  # Ship the freshly-terminal trajectory trees to OBS (append-only,
  # idempotent; scripts/obs_traj_sync.py). Runs after
  # finalize_wrapper_session so this loop's own tree qualifies as
  # terminal. Default-on; FORGE_OBS_PUSH_ON_EXIT=0 disables. The
  # background subshell never delays wrapper exit; the push script
  # soft-skips without obsutil/credentials and holds a run lock so
  # overlapping triggers (startup hook, parallel loop exits) collapse
  # into one worker.
  if [[ "${FORGE_OBS_PUSH_ON_EXIT:-1}" == "1" \
        && -n "${FORGE_SOURCE_ROOT:-}" \
        && -f "$FORGE_SOURCE_ROOT/scripts/obs_traj_sync.py" ]]; then
    mkdir -p "$FORGE_SOURCE_ROOT/.artifacts"
    (bash "$FORGE_SOURCE_ROOT/scripts/obs_traj_push.sh" --quiet \
       >>"$FORGE_SOURCE_ROOT/.artifacts/obs_push.log" 2>&1 || true) &
  fi
}
trap _loop_exit_handler EXIT

# Best-effort web hint. Configurable port. Failure is non-fatal.
_LOOP_WEB_PORT="${LOOP_WEB_PORT:-8421}"
_register_payload="$(printf '{"loop_id":"%s","pid":%d,"workspace":"%s","log_dir":"%s","output_file":"%s","label":"%s","args":{"backend":"%s","model":"%s","stages":"%s","max_mode":"%s"}}' \
  "$LOOP_ID" "$$" "$WORKSPACE" "$LOG_DIR" "${LOOP_OUTPUT_LOG:-}" \
  "$LOOP_LABEL" "$AGENT_BACKEND" "$MODEL" "${STAGES[*]}" "$MAX_MODE")"
curl -sS -X POST "http://127.0.0.1:${_LOOP_WEB_PORT}/api/loop/register" \
  -H 'Content-Type: application/json' \
  -H 'X-Loop-Source: agent-loop' \
  -d "$_register_payload" --max-time 2 >/dev/null 2>&1 && \
  echo "[loop-register] hinted dashboard on port ${_LOOP_WEB_PORT}" || \
  echo "[loop-register] dashboard not reachable on port ${_LOOP_WEB_PORT} (non-fatal; session.json on disk is SSOT)"

# Test-only short-circuit: integration tests for the registration
# block set LOOP_REGISTER_ONLY=1 to verify session.json was written
# without launching the agent CLI. The EXIT trap still fires so
# the published session.json reaches its terminal status.
if [[ "${LOOP_REGISTER_ONLY:-0}" == "1" ]]; then
  echo "[loop-register] LOOP_REGISTER_ONLY=1 set; exiting after registration."
  exit 0
fi

# Background corpus prefetch for the production milestone: when
# [data].prefetch_target_gb > 0, stage the production long-train corpus
# while the pre-production dev loop runs on the
# baked slice. Detached + logged; the production runner blocks on the
# sentinel (evals/dispatcher._await_prefetch_if_configured) before its
# first segment. No-op (fast exit 0) when the knob is unset, so loops
# without a production milestone are unaffected.
_prefetch_gb=$("$PYTHON" - <<'PYEOF' 2>/dev/null || echo 0
from harness import config_runtime
try:
    _, wc = config_runtime.load_workload_config(None)
    print(float((wc.get("data") or {}).get("prefetch_target_gb", 0) or 0))
except Exception:
    print(0)
PYEOF
)
if [[ -n "$_prefetch_gb" ]] && (( $(printf '%.0f' "$_prefetch_gb") > 0 )); then
  # Proxy-unset + HF_ENDPOINT pinning + --curl now live inside
  # `bin/harness prefetch` (harness/app.py::prefetch_command), so the
  # download always runs *in the workspace that executes it* and the
  # corpus bytes land local to the host that will actually train on them
  # — symmetric with how the agent invokes `bin/harness run`. Local mode
  # runs it directly; remote mode waits for the workspace to land on the
  # devspace, then runs it there (see each branch).
  if [[ "${REMOTE_ENABLED:-false}" == "true" && -n "${REMOTE_SSH_HOST:-}" \
        && -n "${REMOTE_WORKDIR:-}" ]]; then
    # Remote execution: the trainer runs on the devspace, so the corpus
    # must be staged THERE — not on this (local) wrapper host. Two startup
    # races make a wrapper-time launch impossible: (1) a freshly leased
    # devspace is not ssh-reachable for ~a minute after `cctl devspace
    # create` (Teleport reverse-tunnel registration lag), and (2) the
    # workspace + per-loop config only land once the dev agent runs its
    # first `bin/harness sync push`. So launch a DETACHED waiter that
    # polls until the remote shim exists — which proves the tree AND the
    # per-loop config/ are synced AND the tunnel is up — then ssh-launches
    # the prefetch there. `cd $WORKDIR && bin/harness` needs no remote
    # bootstrap (python -m puts cwd on sys.path[0]; the config push lands
    # the active *.toml under <workdir>/config) and prefetch needs only
    # curl + stdlib. The waiter does NOT rsync itself, to avoid racing the
    # agent's `--delete` sync. Bounded + non-fatal: the production runner
    # re-awaits the sentinel before its first segment regardless. production
    # is the terminal stage1 milestone, so the corpus has ample time to
    # stage after the first sync.
    _prefetch_log="$AGENT_LOOP_STATE_DIR/prefetch.log"
    mkdir -p "$AGENT_LOOP_STATE_DIR"
    echo "[prefetch] remote mode: corpus prefetch (${_prefetch_gb}GB) will launch on ${REMOTE_SSH_HOST} once the agent's first sync lands the workspace → $_prefetch_log"
    (
      # ~2h cap at 20s between probes. The first sync push can lag the
      # loop start by tens of minutes (agent CLI cold start + first round
      # of work before it touches the remote), and production is the
      # terminal stage1 milestone, so erring long is correct — a 15-min cap was observed
      # to expire just before the agent's first sync landed.
      for _att in $(seq 1 360); do
        if ssh -n -o ConnectTimeout=10 -o BatchMode=yes "${REMOTE_SSH_HOST}" \
             "test -x '${REMOTE_WORKDIR}/bin/harness'" 2>/dev/null; then
          if ssh -n "${REMOTE_SSH_HOST}" \
               "cd '${REMOTE_WORKDIR}' && mkdir -p .artifacts && (nohup bin/harness prefetch >.artifacts/prefetch.log 2>&1 &)"; then
            echo "[prefetch] launched on ${REMOTE_SSH_HOST}:${REMOTE_WORKDIR} after ${_att} probe(s)"
            exit 0
          fi
        fi
        sleep 20
      done
      echo "[prefetch] WARN: workspace not reachable on ${REMOTE_SSH_HOST} after ~2h; gave up (non-fatal; the production runner re-awaits the sentinel)"
    ) >>"$_prefetch_log" 2>&1 &
    disown || true
  else
    _prefetch_log="$AGENT_LOOP_STATE_DIR/prefetch.log"
    mkdir -p "$AGENT_LOOP_STATE_DIR"
    echo "[prefetch] launching background corpus prefetch (${_prefetch_gb}GB) → $_prefetch_log"
    nohup "$PYTHON" -m harness.cli prefetch >"$_prefetch_log" 2>&1 &
    disown || true
  fi
fi

# ── Canonical-state preflight ────────────────────────────────────────
# Produce both forge_init_ones canonicals (ones/ + no1/) once, before the
# first dev round, so no dev/review agent ever hand-runs
# tools/bootstrap_canonical.py (the missing-DATA_CONF footgun that cost
# alignment loops ~30 min of manual env setup). The Python entry is
# idempotent (existing canonicals are skipped), a no-op for non-torch ref
# backends, and self-assembles the ref-run env via the dispatcher's SSOT.
# A non-zero exit aborts the loop: the stage1 bitwise gates cannot pass
# without these canonicals, so failing here saves the agent from a doomed
# run. Set FORGE_SKIP_CANONICAL=1 to bypass (e.g. a GPU-less smoke run).
#
# tools/canonical_preflight.sh dispatches to the host where the ref gates
# actually run: locally for [remote].kind=local, on the SSH host (after
# seeding it) for ssh / devspace loops — the tokenizer/data mounts (e.g.
# /opt/forge-data) and the GPUs a canonical dump needs exist ONLY there, so
# running it on a remote-loop launcher would crash with PermissionError.
if [[ "${FORGE_SKIP_CANONICAL:-0}" != "1" ]]; then
  _loop_event info text "canonical preflight: ensuring forge_init_ones canonicals exist"
  mkdir -p "$LOG_DIR"
  _preflight_log="$LOG_DIR/canonical_preflight.log"
  if bash "$WORKSPACE/tools/canonical_preflight.sh" >"$_preflight_log" 2>&1; then
    _loop_event info text "canonical preflight: canonicals ready"
  else
    # Persist the captured output — the preflight's stderr was previously
    # written to the void, so the first-failure root cause (dead tunnel,
    # missing /opt/forge-data mount, DATA_CONF, …) was undiagnosable after
    # the fact. Now it lands in $_preflight_log and its tail is surfaced in
    # the loop chat.
    _preflight_tail="$(tail -n 40 "$_preflight_log" 2>/dev/null)"
    _loop_event info text "canonical preflight FAILED — log at ${_preflight_log}"$'\n'"${_preflight_tail}"
    echo "[$(date '+%F %T')] canonical preflight FAILED — refusing to start loop. Log: $_preflight_log" >&2
    cat "$_preflight_log" >&2 || true
    # Do NOT let the EXIT trap release (cctl devspace stop) the leased
    # devspace. A transient preflight failure would otherwise stop the very
    # machine, and because [remote].hostname is frozen for the loop, every
    # subsequent retry would ssh a now-dead host and fail forever — a
    # self-reinforcing dead loop. Keep the box up for debugging + retry.
    FORGE_SKIP_LEASE_RELEASE=1
    exit 2
  fi
fi

_first_stage="${STAGES[0]}"
for stage in "${STAGES[@]}"; do
  # Manual approval gate: pause before non-first stages
  if [[ "$MANUAL_APPROVE" == "true" && "$stage" != "$_first_stage" ]]; then
    _gate_file="$AGENT_LOOP_STATE_DIR/${stage}.gate"
    printf 'pending' > "$_gate_file"
    _loop_event info \
      text "${stage}: awaiting manual approval (approve via Web UI or write 'approved' to ${_gate_file})" \
      stage "$stage"
    while [[ "$(cat "$_gate_file" 2>/dev/null)" != "approved" ]]; do
      sleep 5
    done
    rm -f "$_gate_file"
    _loop_event info text "${stage}: approved, proceeding" stage "$stage"
  fi

  _loop_event stage_start stage "$stage"

  initial_status=$(read_stage_status "$stage")
  if [[ "$initial_status" == "finished" ]]; then
    _loop_event stage_skip \
      stage "$stage" text "status=finished — skipping (use --reset-state to rerun)."
    continue
  fi

  run_agent_stage "$stage"
done

_loop_event info text "All stages processed. State markers under $AGENT_LOOP_STATE_DIR."
