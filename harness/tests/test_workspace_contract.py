"""Tests for :mod:`harness.workspace_contract`.

Each invariant has a passing fixture (the live test environment, where
the contract MUST hold) plus a failing fixture (a synthesised root that
violates the invariant). The error message of every failure includes an
executable fix instruction; tests assert on the instruction string so
future weakening of error messages breaks here.

``harness_cli_resolves_locally`` diagnoses TWO distinct shadow modes and
emits a different fix per mode:

* **cwd-shadow** — the imported package lives under the launching cwd
  (e.g. someone ran ``cd .../harness && bash agent-loop.sh``; `python -m`
  inserts cwd at sys.path[0] *before* PYTHONPATH, so the cwd's own
  ``harness/`` wins over the workspace shim). Fix: launch from the
  workspace root, not from inside it.
* **global-install** — the imported package lives outside both the
  workspace and cwd (a sibling worktree's ``pip install -e .`` left a
  global entrypoint). Fix: ``pip uninstall``.

Misdiagnosing the cwd-shadow case as a global install was the proximate
cause of the 2026-05-29 loop-launch debugging spiral. These tests pin
the branching so future edits cannot regress.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import harness
from harness.workspace_contract import (
    WorkspaceContractError,
    harness_cli_resolves_locally,
    run_all,
)

# Real workspace root of this test process. harness/__init__.py lives at
# <workspace>/harness/__init__.py, so workspace == parents[1].
_REAL_WORKSPACE = Path(harness.__file__).resolve().parents[1]


class TestHarnessCliResolvesLocally(unittest.TestCase):
    def test_passes_when_workspace_matches_imported_package(self) -> None:
        # No exception when workspace_root really does own the imported
        # harness package — the contract must not false-positive on a
        # correctly provisioned workspace.
        harness_cli_resolves_locally(_REAL_WORKSPACE)

    def test_raises_when_workspace_is_unrelated(self) -> None:
        # Basic failure path: the error must name (a) the actual import
        # path and (b) the expected workspace, regardless of which
        # diagnosis branch fires. Specific fix-string assertions live in
        # the two branch-specific tests below.
        with self.assertRaises(WorkspaceContractError) as ctx:
            harness_cli_resolves_locally(Path("/nonexistent/sibling/worktree"))
        msg = str(ctx.exception)
        self.assertIn("harness package resolved to", msg)
        self.assertIn("/nonexistent/sibling/worktree", msg)

    def test_diagnoses_cwd_shadow_when_cwd_owns_harness(self) -> None:
        # When launched from a directory that itself contains a
        # ``harness/`` Python package, `python -m harness` resolves to
        # *that* package (sys.path[0] = cwd, inserted before PYTHONPATH).
        # The fix must tell the user to launch from the workspace root,
        # NOT to `pip uninstall` (the package is not globally installed).
        original_cwd = os.getcwd()
        try:
            os.chdir(_REAL_WORKSPACE)
            with self.assertRaises(WorkspaceContractError) as ctx:
                harness_cli_resolves_locally(Path("/nonexistent/sibling/worktree"))
            msg = str(ctx.exception)
            # Must surface the cwd-shadow diagnosis with an actionable cd hint.
            self.assertIn("cwd", msg.lower())
            self.assertIn(str(_REAL_WORKSPACE), msg)
            # MUST NOT recommend pip uninstall — there is no global install.
            self.assertNotIn("pip uninstall", msg)
        finally:
            os.chdir(original_cwd)

    def test_diagnoses_global_install_when_cwd_neutral(self) -> None:
        # When cwd has no ``harness/`` package of its own, any
        # ``import harness`` that lands outside the workspace can only
        # come from a globally-installed entrypoint. Fix: pip uninstall.
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as neutral:
            try:
                os.chdir(neutral)
                with self.assertRaises(WorkspaceContractError) as ctx:
                    harness_cli_resolves_locally(Path("/nonexistent/sibling/worktree"))
                msg = str(ctx.exception)
                self.assertIn("pip uninstall", msg)
            finally:
                os.chdir(original_cwd)


class TestRunAll(unittest.TestCase):
    def test_passes_on_real_workspace(self) -> None:
        run_all(_REAL_WORKSPACE)

    def test_raises_on_unrelated_workspace(self) -> None:
        with self.assertRaises(WorkspaceContractError):
            run_all(Path("/nonexistent/sibling/worktree"))


if __name__ == "__main__":
    unittest.main()
