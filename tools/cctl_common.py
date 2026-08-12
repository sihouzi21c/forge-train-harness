"""Shared ``cctl`` shell-out primitives.

Single source of truth for the low-level facts every ``cctl``-driven
helper needs: which binary to invoke (``CCTL_BIN`` override), the
wall-clock cap on a single shell-out, the create-response → task-id
parse, and the status-phase vocabulary the server reports. Both
``tools/lease.py`` (devspace lifecycle) and ``tools/gpu_job.py``
(ephemeral GPU jobs) depend on this module; this module depends on
neither, so the layering stays a DAG.

Tests inject a fake ``cctl`` via the ``CCTL_BIN`` env-var override — no
real CLI, no network.
"""

from __future__ import annotations

import json
import os
import subprocess

__all__ = [
    "CCTL_CALL_TIMEOUT_S",
    "READY_PHASES",
    "TERMINAL_PHASES",
    "CctlError",
    "cctl_bin",
    "run_cctl",
    "task_id_from_create",
]

# Wall-clock cap on a single ``cctl`` shell-out. Bounds any caller (e.g.
# the lease EXIT-trap ``release``) so a hung Teleport tunnel behind a
# ``cctl`` call cannot block the process indefinitely.
CCTL_CALL_TIMEOUT_S = 120

# Server-reported task ``status`` strings (lower-cased), shared across
# devspace and job kinds since both bridge the same v2 tasks API.
READY_PHASES = frozenset({"running", "ready"})
TERMINAL_PHASES = frozenset({"failed", "killed", "stopped", "oomkilled", "invalid", "succeeded"})


class CctlError(RuntimeError):
    """``cctl`` returned non-zero, timed out, or emitted unparsable output."""


def cctl_bin() -> str:
    return os.environ.get("CCTL_BIN", "cctl")


def run_cctl(
    args: list[str], *, timeout: int = CCTL_CALL_TIMEOUT_S
) -> subprocess.CompletedProcess[str]:
    """Run ``cctl <args>`` with an explicit timeout and raise on failure."""
    try:
        proc = subprocess.run(
            [cctl_bin(), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise CctlError(f"cctl {' '.join(args)} timed out after {timeout}s") from exc
    if proc.returncode != 0:
        raise CctlError(
            f"cctl {' '.join(args)} failed (exit {proc.returncode}): "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
    return proc


def task_id_from_create(stdout: str) -> str:
    """Extract the server-assigned task id from a ``... create -o json`` reply."""
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise CctlError(f"cctl create returned non-JSON: {stdout!r}") from exc
    task_id = payload.get("id")
    if task_id is None:
        name = payload.get("name") or ""
        if name.startswith("tasks/"):
            task_id = name[len("tasks/") :]
    if not task_id:
        raise CctlError(f"cctl create response has no task id: {payload!r}")
    return str(task_id)
