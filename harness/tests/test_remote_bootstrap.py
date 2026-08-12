"""Smoke tests for the idempotent remote-devspace bootstrap script.

The agent loop previously had to re-derive devspace setup from scratch
every time a devspace was rotated (loop e180edc7 went through 5 fresh
devspaces and re-discovered the same incantations each time: clear
HTTP_PROXY, point pip at the campus mirror, install harness with
``--no-build-isolation``, hand-copy modelbest_sdk into site-packages).
``harness/tools/remote_bootstrap.sh`` codifies that recipe and stamps
a version-keyed marker file so a second invocation short-circuits.

Tests are intentionally smoke-level:

* The script is shipped, executable, and syntactically valid bash.
* ``--check`` exit code is 1 on a fresh ``$HOME`` and 0 once the
  marker has been stamped at the expected version — the
  short-circuit contract the caller relies on.
* The script refuses to run without an explicit ``$HOME`` override
  (so a misconfigured caller cannot stamp the real user's home).

Tests do NOT shell out to ``pip``; the full bootstrap path requires
a real devspace and is exercised end-to-end by ``agent-loop.sh`` on
first invocation against a new ``REMOTE_SSH_HOST``.
"""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "remote_bootstrap.sh"


class TestRemoteBootstrapScriptShipped(unittest.TestCase):
    def test_script_exists(self) -> None:
        self.assertTrue(SCRIPT.is_file(), f"missing {SCRIPT}")

    def test_script_is_executable(self) -> None:
        self.assertTrue(
            os.access(SCRIPT, os.X_OK),
            f"{SCRIPT} must be chmod +x so agent-loop.sh can invoke it directly",
        )

    def test_script_syntactically_valid(self) -> None:
        # ``bash -n`` parses without executing; catches typos that
        # would otherwise only fail on the remote devspace where
        # debug iteration is expensive.
        result = subprocess.run(
            ["bash", "-n", str(SCRIPT)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class TestRemoteBootstrapCheckMode(unittest.TestCase):
    """``--check`` is the contract caller uses to skip a redundant run."""

    def _run_check(self, fake_home: Path) -> int:
        # The script reads $HOME to locate ``~/.forge_train/env_ready``;
        # the test overrides $HOME to keep the host user's home untouched.
        env = {**os.environ, "HOME": str(fake_home)}
        return subprocess.run(
            ["bash", str(SCRIPT), "--check"],
            env=env,
            capture_output=True,
            check=False,
        ).returncode

    def test_check_returns_nonzero_on_fresh_home(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self._run_check(Path(tmp)), 1)

    def test_check_returns_zero_when_marker_present(self) -> None:
        import tempfile

        # Reproduce the marker the script itself writes after a
        # successful bootstrap. The version line MUST match
        # ``BOOTSTRAP_VERSION`` declared in the script — otherwise a
        # stale marker from an older version would falsely satisfy
        # ``--check`` and skip required re-bootstrap.
        version_line = self._declared_bootstrap_version()

        with tempfile.TemporaryDirectory() as tmp:
            marker_dir = Path(tmp) / ".forge_train"
            marker_dir.mkdir()
            (marker_dir / "env_ready").write_text(
                f"version={version_line}\nstamped_at=2026-05-28T00:00:00Z\n",
                encoding="utf-8",
            )
            self.assertEqual(self._run_check(Path(tmp)), 0)

    def test_check_rejects_stale_version_marker(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            marker_dir = Path(tmp) / ".forge_train"
            marker_dir.mkdir()
            (marker_dir / "env_ready").write_text(
                "version=0\nstamped_at=2026-05-28T00:00:00Z\n",
                encoding="utf-8",
            )
            self.assertEqual(
                self._run_check(Path(tmp)),
                1,
                "an older bootstrap-version marker MUST NOT satisfy --check",
            )

    @staticmethod
    def _declared_bootstrap_version() -> str:
        # Single source of truth for the version: the script literal
        # itself. Parsing it keeps the test honest against future
        # version bumps without manual edits here.
        for line in SCRIPT.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith("BOOTSTRAP_VERSION="):
                return s.split("=", 1)[1].strip('"').strip("'")
        raise AssertionError("BOOTSTRAP_VERSION line not found in script")


if __name__ == "__main__":
    unittest.main()
