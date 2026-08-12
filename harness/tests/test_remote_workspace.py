"""Unit tests for ``tools/remote_workspace.py`` — the devspace shared
remote-root validator/resolver (SSOT for both the lease claim gate and
the kind=job GPU-job workdir)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools import remote_workspace as rw  # noqa: E402


class ValidateSharedWorkspaceTest(unittest.TestCase):
    def test_accepts_user_gpfs_path(self) -> None:
        self.assertEqual(rw.validate_shared_workspace("/user/lishangzhan"), "/user/lishangzhan")
        self.assertEqual(rw.validate_shared_workspace("  /data/user/x  "), "/data/user/x")

    def test_rejects_empty(self) -> None:
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError):
                rw.validate_shared_workspace(bad)

    def test_rejects_home_and_tilde(self) -> None:
        for bad in ("$HOME", "~"):
            with self.assertRaises(ValueError):
                rw.validate_shared_workspace(bad)

    def test_rejects_pod_local_root(self) -> None:
        for bad in ("/root", "/root/.forge_train"):
            with self.assertRaises(ValueError):
                rw.validate_shared_workspace(bad)

    def test_rejects_relative(self) -> None:
        with self.assertRaises(ValueError):
            rw.validate_shared_workspace("user/lishangzhan")


class ResolveRemoteWorkdirTest(unittest.TestCase):
    def test_builds_forge_train_path(self) -> None:
        wd = rw.resolve_remote_workdir({"workspace": "/user/lishangzhan"}, "L1")
        self.assertEqual(wd, "/user/lishangzhan/.forge_train/L1")

    def test_empty_workspace_raises(self) -> None:
        with self.assertRaises(ValueError):
            rw.resolve_remote_workdir({"workspace": ""}, "L1")


class CliTest(unittest.TestCase):
    def test_validate_ok_returns_zero(self) -> None:
        self.assertEqual(rw.main(["--validate", "/user/lishangzhan"]), 0)

    def test_validate_bad_returns_nonzero(self) -> None:
        self.assertEqual(rw.main(["--validate", ""]), 3)
        self.assertEqual(rw.main(["--validate", "/root"]), 3)


if __name__ == "__main__":
    unittest.main()
