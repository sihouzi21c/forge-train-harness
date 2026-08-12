"""Tests for lock-guarded Cursor CLI max-mode toggling."""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from tools.cursor_max_mode import (
    _atomic_set_max_mode,
    cursor_max_mode_guard,
)


def _read_cfg(path: Path) -> dict:
    return json.loads(path.read_text())


def _make_fake_config(tmp_path: Path) -> Path:
    cfg_path = tmp_path / "cli-config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "maxMode": False,
                "model": {"modelId": "test-model", "maxMode": False},
            }
        )
    )
    return cfg_path


class TestAtomicSetMaxMode(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = Path(tempfile.mkdtemp())
        self._cfg_path = _make_fake_config(self._tmpdir)
        self._patcher = mock.patch(
            "tools.cursor_max_mode._config_path",
            return_value=self._cfg_path,
        )
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_enables_max_mode(self) -> None:
        _atomic_set_max_mode(True)
        cfg = _read_cfg(self._cfg_path)
        self.assertIs(cfg["maxMode"], True)
        self.assertIs(cfg["model"]["maxMode"], True)

    def test_disables_max_mode(self) -> None:
        _atomic_set_max_mode(True)
        _atomic_set_max_mode(False)
        cfg = _read_cfg(self._cfg_path)
        self.assertIs(cfg["maxMode"], False)
        self.assertIs(cfg["model"]["maxMode"], False)

    def test_creates_model_key_if_missing(self) -> None:
        self._cfg_path.write_text(json.dumps({"maxMode": False}))
        _atomic_set_max_mode(True)
        cfg = _read_cfg(self._cfg_path)
        self.assertIs(cfg["maxMode"], True)
        self.assertIs(cfg["model"]["maxMode"], True)


class TestCursorMaxModeGuard(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = Path(tempfile.mkdtemp())
        self._cfg_path = _make_fake_config(self._tmpdir)
        self._patcher = mock.patch(
            "tools.cursor_max_mode._config_path",
            return_value=self._cfg_path,
        )
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_sets_config_on_enter(self) -> None:
        with cursor_max_mode_guard(True):
            cfg = _read_cfg(self._cfg_path)
            self.assertIs(cfg["maxMode"], True)
        cfg_after = _read_cfg(self._cfg_path)
        self.assertIs(cfg_after["maxMode"], True)

    def test_false_resets_residual_true(self) -> None:
        _atomic_set_max_mode(True)
        self.assertIs(_read_cfg(self._cfg_path)["maxMode"], True)
        with cursor_max_mode_guard(False):
            cfg = _read_cfg(self._cfg_path)
            self.assertIs(cfg["maxMode"], False)
            self.assertIs(cfg["model"]["maxMode"], False)

    def test_releases_lock_on_exception(self) -> None:
        with self.assertRaises(RuntimeError), cursor_max_mode_guard(True):
            raise RuntimeError("boom")
        with cursor_max_mode_guard(False):
            cfg = _read_cfg(self._cfg_path)
            self.assertIs(cfg["maxMode"], False)

    def test_serializes_concurrent_access(self) -> None:
        results: list[bool] = []
        barrier = threading.Barrier(4)
        lock_order: list[int] = []

        def worker(val: bool, idx: int) -> None:
            barrier.wait(timeout=5)
            with cursor_max_mode_guard(val):
                lock_order.append(idx)
                cfg = _read_cfg(self._cfg_path)
                results.append(cfg["maxMode"] == val)

        threads = [
            threading.Thread(target=worker, args=(True, 0)),
            threading.Thread(target=worker, args=(False, 1)),
            threading.Thread(target=worker, args=(True, 2)),
            threading.Thread(target=worker, args=(False, 3)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertTrue(all(results), f"Some workers read wrong maxMode: {results}")
        self.assertEqual(len(lock_order), 4)


class TestAgentLoopMaxModeExport(unittest.TestCase):
    _created: bool = False

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        from tools import agent_loop_config

        cls._agent_toml = agent_loop_config.REPO_ROOT / "config" / "agent.toml"
        if not cls._agent_toml.exists():
            import shutil

            shutil.copy2(
                agent_loop_config.REPO_ROOT / "config" / "agent" / "cursor-gpt-5.5.toml",
                cls._agent_toml,
            )
            cls._created = True

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._created and cls._agent_toml.exists():
            cls._agent_toml.unlink()
        super().tearDownClass()

    def test_shell_exports_include_max_mode(self) -> None:
        from tools import agent_loop_config

        exports = agent_loop_config.shell_exports({})
        self.assertIn("export MAX_MODE=", exports)
        self.assertIn("export EFFORT=", exports)


if __name__ == "__main__":
    unittest.main()
