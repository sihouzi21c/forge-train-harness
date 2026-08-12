"""Workspace contract: fail-fast invariants checked before the agent runs.

Every loop's workspace must satisfy a machine-checkable contract before
any LLM agent is allowed to start. Each invariant catches a specific
class of infrastructure bug that would otherwise burn agent runtime
through opaque mis-debugging (resource-hunt, editable-install
shadowing, future silent-config-drift).

Invariants are intentionally narrow: each one names exactly one failure
mode plus the executable fix. Adding an invariant is cheap; weakening
one requires evidence the bug class is gone.

Entry point is :func:`run_all`, wrapped by the ``harness env-probe``
CLI subcommand. ``agent-loop.sh`` (CLI) and ``web/routers/loop.py``
(web) both invoke the CLI subcommand after workspace provisioning and
refuse to start the loop on a non-zero exit.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from harness import config_runtime

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "WorkspaceContractError",
    "harness_cli_resolves_locally",
    "run_all",
    "subprocess_config_isomorphic",
]


class WorkspaceContractError(RuntimeError):
    """Raised when a workspace invariant fails.

    The ``args[0]`` message MUST include an executable fix instruction
    (a shell command, a config edit, or a documented escape hatch) so
    a human or agent can resolve the violation without further triage.
    """


def harness_cli_resolves_locally(workspace_root: Path) -> None:
    """The imported ``harness`` package MUST live under ``workspace_root``.

    Bug class: ``python -m harness`` (and any in-process ``import harness``)
    resolves to the wrong on-disk package. Two failure modes share this
    invariant and must be diagnosed *separately* because their fixes are
    mutually exclusive:

    * **cwd-shadow** — the launcher ran ``cd .../harness && bash
      agent-loop.sh`` (or any equivalent), so cwd is a directory whose
      own ``harness/`` Python package shadows the per-workspace shim.
      ``python -m`` inserts cwd at ``sys.path[0]`` *before* PYTHONPATH,
      so the cwd's package always wins. Fix: launch from the workspace
      root itself, not from inside it.
    * **global-install** — a sibling worktree's ``pip install -e .``
      left a globally-resolvable ``harness`` package on the default
      ``sys.path``. Fix: ``pip uninstall``.

    Hand-coding "pip uninstall" as the only fix once burned ~60 min of
    agent runtime on a cwd-shadow incident (2026-05-29) — the user kept
    running the suggested ``pip uninstall`` against a package that was
    never installed. The branching is now mandatory; see
    :func:`_shadow_diagnosis`.
    """
    # Re-import in case a prior `import harness` in this process bound
    # to a different path. importlib.reload is safe here because the
    # invariant is the first thing run from the CLI entrypoint.
    if "harness" in sys.modules:
        importlib.reload(sys.modules["harness"])
    import harness

    actual = Path(harness.__file__).resolve()
    expected_parent = (workspace_root / "harness").resolve()
    if not _is_under(actual, expected_parent):
        raise WorkspaceContractError(
            f"harness package resolved to {actual} but workspace root is "
            f"{workspace_root} — every edit you make in this workspace "
            f"will silently take no effect. "
            f"{_shadow_diagnosis(actual, workspace_root)}"
        )


def _shadow_diagnosis(actual: Path, workspace_root: Path) -> str:
    """Return a fix instruction tailored to the actual shadow source.

    Branches on whether the imported package lives under the launching
    cwd. The two branches name mutually-exclusive fixes — emitting both
    would be worse than emitting one wrong one, because the user would
    not know which to try first.
    """
    cwd = Path(os.getcwd()).resolve()
    if _is_under(actual, cwd):
        return (
            f"cwd-shadow: launching from {cwd} puts that directory at "
            f"sys.path[0] before any PYTHONPATH entry, so `python -m` "
            f"resolves `harness` to {actual} instead of the workspace's "
            f"copy. Fix: `cd {workspace_root}` and re-launch via the "
            f"per-workspace shim (`bin/harness ...` or "
            f"`bash agent-loop.sh ...`) — do NOT cd into the inner "
            f"`harness/` directory before launching."
        )
    return (
        f"global editable install: {actual} is outside both the workspace "
        f"({workspace_root}) and the current cwd ({cwd}). A sibling "
        f"worktree's `pip install -e .` left a global entrypoint that "
        f"shadows this workspace. "
        f"Fix: `pip uninstall -y training-engine-harness`, then re-run. "
        f"The per-workspace shim at <workspace>/bin/harness will continue "
        f"to work without any editable install."
    )


def _is_under(child: Path, parent: Path) -> bool:
    """True when *child* equals or sits beneath *parent* (post-resolve)."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


# Ordered: cheapest / most-likely-to-fire first so the agent gets the
# most actionable error early.
_INVARIANTS: list[Callable[[Path], None]] = [
    harness_cli_resolves_locally,
    # subprocess_config_isomorphic spawns a Python subprocess and is
    # therefore the most expensive invariant — keep it last so cheaper
    # checks fire first.
]


def subprocess_config_isomorphic(workspace_root: Path) -> None:
    """A child ``harness echo-config`` MUST see the same bytes the parent does.

    Bug class: a silent PYTHONPATH / cwd / env-override divergence
    between the loop's main Python process and any Python subprocess
    it spawns. When the divergence flips a single FORGE_* / TOML
    override, the child's workload-config diverges from the parent's,
    and the run silently uses different optim or model hyperparameters
    than the agent intended (this is the structural sibling of the
    cherry-pick drop fixed in the same commit).

    The invariant loads the config in-process, computes the canonical
    sha, then spawns ``python3 -m harness.cli echo-config`` and hashes
    its stdout. Any mismatch is a fatal contract violation. Skips
    silently when no user workload config (``config/eval.toml`` or
    required ``config/model.toml``) exists yet — fresh checkouts must
    be allowed to env-probe before they're fully configured; commit C
    of the hermetic-workspace plan only guards configured workspaces.
    """
    try:
        _, workload_config = config_runtime.load_workload_config(None)
    except FileNotFoundError:
        # Not yet configured (no eval.toml or model.toml). Loop launch
        # would have failed earlier on its own load path; nothing for
        # this invariant to compare.
        return
    parent_payload = config_runtime.canonical_workload_config_bytes(workload_config)
    parent_sha = hashlib.sha256(parent_payload).hexdigest()

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "harness.cli", "echo-config"],
            cwd=workspace_root,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WorkspaceContractError(
            f"`harness echo-config` subprocess failed to launch: {exc}. "
            f"Fix: ensure {sys.executable} can `import harness` from "
            f"{workspace_root} (PYTHONPATH={workspace_root}). The per-"
            f"workspace shim at <workspace>/bin/harness sets this; the "
            f"global editable install does not."
        ) from exc
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise WorkspaceContractError(
            f"`harness echo-config` exited {completed.returncode}: {stderr}. "
            f"The subprocess could not load the workload config the parent "
            f"just loaded successfully — almost always a PYTHONPATH / cwd "
            f"divergence. Fix: re-launch from the workspace root with "
            f"the per-workspace shim on PATH."
        )
    child_sha = hashlib.sha256(completed.stdout).hexdigest()
    if child_sha != parent_sha:
        raise WorkspaceContractError(
            f"workload_config sha drift between parent and child Python: "
            f"parent={parent_sha} child={child_sha}. The subprocess "
            f"resolved a different config than the in-process loader — "
            f"check FORGE_* env vars and `which python3` to find the "
            f"divergence."
        )


_INVARIANTS.append(subprocess_config_isomorphic)


def run_all(workspace_root: Path) -> None:
    """Run every invariant against *workspace_root*; raise on first failure.

    Resolves the path once up-front so each invariant sees a stable
    absolute root regardless of cwd or symlinks.
    """
    workspace_root = workspace_root.resolve()
    for invariant in _INVARIANTS:
        invariant(workspace_root)
