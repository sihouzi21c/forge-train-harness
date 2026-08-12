"""Remote workspace synchronization via rsync.

SSOT for the ``harness sync push`` command.  Encapsulates the rsync
exclude list, post-sync ``__pycache__`` cleanup, and remote validation
so the dev agent never needs to hand-craft rsync commands.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

__all__ = ["sync_push"]

_BASE_EXCLUDES = (
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    ".artifacts",
    ".ruff_cache",
    "workload/profile",
    # The profile verdict (evals/verdicts/profile.py) renders summary.md /
    # profile.json into the REMOTE workload/notes/profile/<label>/ — files
    # that exist only remotely. Without this exclude the mirror `--delete`
    # below wipes them on the next push (loop f6b9c438e05f round 8).
    "workload/notes",
)

_REQUIRED_FILES = (
    "pyproject.toml",
    "harness/config/defaults.toml",
)

_REQUIRED_DIRS = (
    "config/eval",
    "evals",
    "harness",
    "workload",
)


def _build_rsync_command(
    *,
    source: str,
    host: str,
    dest: str,
    stage: str,
    verbose: bool,
) -> list[str]:
    excludes = [e for e in _BASE_EXCLUDES if not (stage == "stage2" and e == ".git")]
    flags = "-avz" if verbose else "-az"
    cmd: list[str] = ["rsync", flags, "--delete"]
    for exc in excludes:
        cmd.append(f"--exclude={exc}")
    if not source.endswith("/"):
        source += "/"
    cmd.append(source)
    cmd.append(f"{host}:{dest}")
    return cmd


def _build_config_push_command(*, host: str, dest: str, config_dir: str) -> list[str]:
    """rsync the active per-loop config (top-level ``*.toml``) into the
    remote ``<dest>/config/``.

    The active config lives in the SIBLING ``FORGE_CONFIG_DIR``
    (``.artifacts/forge_train/<id>/config``), which the main workspace
    rsync excludes via ``.artifacts``. The remote ``harness run`` reads
    config from ``<workdir>/config`` (``FORGE_CONFIG_DIR`` is never
    exported across the ssh hop), so the active ``*.toml`` would never
    reach the remote on their own. This dedicated, ``--delete``-free push
    ships them every sync, so the remote always reads the SSOT config and
    the agent never has to hand-push files that the next ``--delete``
    rsync would orphan.
    """
    src = config_dir if config_dir.endswith("/") else config_dir + "/"
    return [
        "rsync",
        "-az",
        "--include=*.toml",
        "--exclude=*",
        src,
        f"{host}:{dest.rstrip('/')}/config/",
    ]


def _build_cleanup_command(*, host: str, dest: str) -> list[str]:
    find_cmd = f"find {dest} -name __pycache__ -type d -exec rm -rf {{}} + 2>/dev/null || true"
    return ["ssh", host, find_cmd]


def _build_validation_command(*, host: str, dest: str) -> list[str]:
    checks: list[str] = []
    for f in _REQUIRED_FILES:
        checks.append(f'test -f "{dest}/{f}" || echo "MISSING: {f}"')
    for d in _REQUIRED_DIRS:
        checks.append(f'test -d "{dest}/{d}" || echo "MISSING: {d}/"')
    body = "; ".join(checks)
    script = f'_out=$({body}); if [ -n "$_out" ]; then echo "$_out"; exit 1; fi'
    return ["ssh", host, script]


def _build_mfu_pull_command(*, host: str, dest: str, loop_id: str, local_dest: str) -> list[str]:
    """rsync the per-loop ``mfu_history.jsonl`` back from remote.

    ``tools/mfu_record.py`` is the SSOT writer and runs inside every
    gate subprocess. When ``[remote].kind`` is ``ssh`` / ``devspace``
    that subprocess runs on the remote host, so the file lives at
    ``<dest>/.artifacts/forge_train/<loop_id>/mfu_history.jsonl`` (the
    nested layout comes from the remote-side ``FORGE_TRAIN_DIR``
    export documented in ``prompt/develop_prompt/remote-execution.md``
    §Step 2). No other code path pulls it back, so the local copy that
    ``web/routers/artifacts.py:get_mfu`` reads silently lags.

    rsync without ``--inplace`` writes through a tempfile then renames
    atomically, so the web reader never observes a half-written file.
    """
    remote_path = f"{host}:{dest.rstrip('/')}/.artifacts/forge_train/{loop_id}/mfu_history.jsonl"
    return ["rsync", "-az", remote_path, local_dest]


def _pull_mfu_history(*, host: str, dest: str, source: str) -> None:
    """Best-effort: refresh local ``mfu_history.jsonl`` from remote.

    Failure here MUST NOT break ``sync_push`` — the gate's MFU
    measurement is telemetry, not a correctness signal. We log a
    one-line warning to stderr and return; ``web/routers/artifacts.py``
    already tolerates a missing or stale file.
    """
    loop_id = os.environ.get("LOOP_ID") or os.environ.get("LOOP_WEB_ID") or ""
    if not loop_id:
        return
    # ``Path(source).parent`` is the local loop dir
    # (``.artifacts/forge_train/<loop_id>``). The loop registration
    # logic creates it; we rely on its presence rather than ``mkdir``
    # so a fresh OSError here can't mask a deeper layout drift.
    local_dest = Path(source).parent / "mfu_history.jsonl"
    try:
        cmd = _build_mfu_pull_command(
            host=host, dest=dest, loop_id=loop_id, local_dest=str(local_dest)
        )
        subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except Exception as exc:  # best-effort — never affect sync_push outcome
        print(f"sync_push: mfu_history pull-back failed: {exc}", file=sys.stderr)


def _resolve_remote_params() -> tuple[str, str, str, str]:
    from tools import agent_loop_config

    remote = agent_loop_config.load_remote_config()
    if str(remote.get("kind", "local") or "local") == "local":
        raise RuntimeError(
            'Remote execution is disabled. Set [remote].kind = "ssh" or '
            '"devspace" in config/remote.toml.'
        )
    hostname = remote.get("hostname", "")
    if not hostname:
        raise RuntimeError("[remote].hostname is empty in config/remote.toml.")
    workspace_root = str(remote.get("workspace", "") or "$HOME")
    loop_id = os.environ.get("LOOP_ID") or os.environ.get("LOOP_WEB_ID") or ""
    if loop_id:
        remote_workdir = f"{workspace_root}/.forge_train/{loop_id}"
    else:
        remote_workdir = f"{workspace_root}/.forge_train"

    from harness import config_runtime

    local_workspace = str(config_runtime.repo_root())
    config_dir = str(config_runtime._user_config_dir())
    return local_workspace, hostname, remote_workdir, config_dir


def sync_push(*, stage: str = "stage1", verbose: bool = False) -> dict:
    source, host, dest, config_dir = _resolve_remote_params()

    mkdir_cmd = ["ssh", host, f"mkdir -p {dest}"]
    subprocess.run(mkdir_cmd, capture_output=True, text=True)

    rsync_cmd = _build_rsync_command(
        source=source,
        host=host,
        dest=dest,
        stage=stage,
        verbose=verbose,
    )
    result = subprocess.run(rsync_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"rsync failed (exit {result.returncode}): {result.stderr.strip()}")

    config_cmd = _build_config_push_command(host=host, dest=dest, config_dir=config_dir)
    cfg_result = subprocess.run(config_cmd, capture_output=True, text=True)
    if cfg_result.returncode != 0:
        raise RuntimeError(
            f"config rsync failed (exit {cfg_result.returncode}): {cfg_result.stderr.strip()}"
        )

    cleanup_cmd = _build_cleanup_command(host=host, dest=dest)
    subprocess.run(cleanup_cmd, capture_output=True, text=True)

    validation_cmd = _build_validation_command(host=host, dest=dest)
    val_result = subprocess.run(validation_cmd, capture_output=True, text=True)
    if val_result.returncode != 0:
        parts = [f"Post-sync validation failed on {host}:{dest} (exit {val_result.returncode})"]
        if val_result.stdout.strip():
            parts.append(val_result.stdout.strip())
        if val_result.stderr.strip():
            parts.append(f"stderr: {val_result.stderr.strip()}")
        raise RuntimeError("\n".join(parts))

    _pull_mfu_history(host=host, dest=dest, source=source)

    return {
        "command": "sync",
        "report": "text",
        "status": "ready",
        "payload": {
            "action": "push",
            "host": host,
            "dest": dest,
            "stage": stage,
        },
    }
