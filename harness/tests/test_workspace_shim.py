"""Contract tests for the per-workspace ``bin/harness`` shim.

Commit B deleted the global ``[project.scripts]`` entrypoint so that
``pip install -e .`` from a sibling worktree can never again silently
shadow the current workspace's harness (the editable-install incident class).
Every workspace now ships its own ``<workspace>/bin/harness`` shim
generated at provision time with the workspace's absolute path baked
in. These tests pin both invariants the shim relies on:

1. ``pyproject.toml`` carries NO global ``harness`` entrypoint.
2. Both provision sites (CLI bootstrap in ``agent-loop.sh`` and the
   web ``_provision_workspace``) emit a shim that:
   - exists, is executable, contains the workspace's absolute path,
   - sets PYTHONPATH to the workspace, and
   - re-execs ``python3 -m harness.cli``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
HARNESS_DIR = REPO_ROOT / "harness"


class TestGlobalEntrypointRemoved(unittest.TestCase):
    """pyproject.toml MUST NOT re-introduce a global ``harness`` script."""

    def test_no_project_scripts_harness(self) -> None:
        pyproject = HARNESS_DIR / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        scripts = data.get("project", {}).get("scripts", {})
        self.assertNotIn(
            "harness",
            scripts,
            "A global [project.scripts] harness entrypoint was reintroduced. "
            "This is the bug class B was meant to eliminate — every "
            "workspace now ships its own bin/harness shim instead.",
        )


class TestAgentLoopBootstrapEmitsShim(unittest.TestCase):
    """The bash bootstrap in ``agent-loop.sh`` MUST write the shim.

    We grep for the textual signature rather than executing the bash
    script (which would require provisioning a full workspace). The
    actual byte-level shim contents are exercised by the web-side test
    below — both call sites use the same template.
    """

    def test_bootstrap_writes_bin_harness(self) -> None:
        script = (HARNESS_DIR / "agent-loop.sh").read_text(encoding="utf-8")
        # The bootstrap's inline Python writes the shim into <ws>/bin/harness.
        self.assertRegex(
            script,
            r'shim\s*=\s*bin_dir\s*/\s*"harness"',
            "agent-loop.sh bootstrap no longer writes bin/harness — the "
            "per-workspace shim invariant is broken.",
        )
        # And the bootstrap MUST prepend <ws>/bin to PATH before exec.
        self.assertIn(
            'export PATH="$_bootstrap_workspace/bin:$PATH"',
            script,
            "agent-loop.sh bootstrap no longer prepends the workspace's "
            "bin/ to PATH — the shim cannot win over a global harness.",
        )


class TestWebProvisionEmitsShim(unittest.TestCase):
    """``web/routers/loop.py`` MUST produce an identically-shaped shim.

    Imports the helper directly so the test does not require a running
    server, and validates the full shim contract against a tmp workspace.
    """

    def _load_helper(self):
        # Inject web/ root onto sys.path without polluting the global
        # interpreter state past this test method.
        web_root = REPO_ROOT
        sys.path.insert(0, str(web_root))
        try:
            from web.routers.loop import _write_workspace_shim  # type: ignore[import-not-found]
        finally:
            sys.path.pop(0)
        return _write_workspace_shim

    def test_shim_file_exists_executable_with_baked_path(self) -> None:
        write_workspace_shim = self._load_helper()

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            workspace.mkdir()
            shim_path = write_workspace_shim(workspace)

            self.assertTrue(shim_path.is_file())
            mode = shim_path.stat().st_mode
            self.assertTrue(mode & 0o111, f"shim not executable: mode={oct(mode)}")

            body = shim_path.read_text(encoding="utf-8")
            self.assertTrue(body.startswith("#!/usr/bin/env bash"))
            self.assertIn(str(workspace), body)
            # PYTHONPATH MUST be set to the workspace verbatim so the
            # subprocess's `import harness` resolves to this workspace's
            # copy, never the system or sibling-worktree copy.
            self.assertRegex(
                body,
                rf'export PYTHONPATH="{re.escape(str(workspace))}',
            )
            # Final exec must dispatch to the workspace-local harness module.
            self.assertIn("exec python3 -m harness.cli", body)

    def test_shim_runs_workspace_harness_when_path_prefixed(self) -> None:
        """``which harness`` resolves to ``<ws>/bin/harness`` when prepended."""
        write_workspace_shim = self._load_helper()

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            workspace.mkdir()
            write_workspace_shim(workspace)

            env = os.environ.copy()
            env["PATH"] = f"{workspace}/bin{os.pathsep}{env.get('PATH', '')}"

            which = subprocess.run(
                ["bash", "-c", "command -v harness"],
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            self.assertEqual(which.returncode, 0, which.stderr)
            self.assertEqual(
                Path(which.stdout.strip()).resolve(),
                (workspace / "bin" / "harness").resolve(),
            )


if __name__ == "__main__":
    unittest.main()
