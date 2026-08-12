"""SSOT for the per-suite wall-clock budget: ``config_runtime.suite_timeout_s``.

These tests pin the contract that every runtime caller (``harness budget``
CLI, ``app._run_gpu_suite``, dispatcher ``_run_*`` functions, the
``tools/remote_run.sh`` wrapper) goes through this single helper instead
of each re-implementing ``cfg.get("timeout_s", DEFAULT)``.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["FORGE_REPO_ROOT"] = str(REPO_ROOT)

from harness import config_runtime  # noqa: E402


def _workload(**evals_extra: dict) -> dict:
    """Build a minimal valid workload_config dict for timeout tests."""
    base = {
        "evals": {
            "alpha": {"stage": "stage1", "runner_kind": "noop", "timeout_s": 42},
            "beta": {"stage": "stage1", "runner_kind": "noop"},  # no timeout_s
        },
        "local_suites": {
            "guard": {"stage": "local", "runner_kind": "guard"},
        },
    }
    base["evals"].update(evals_extra)
    return base


class TestSuiteTimeoutSSOT(unittest.TestCase):
    def test_per_suite_value_wins(self) -> None:
        wl = _workload()
        # explicit harness_config so the test doesn't depend on the
        # committed defaults.toml value.
        hc = {"defaults": {"default_timeout_s": 1234}}
        self.assertEqual(config_runtime.suite_timeout_s(wl, "alpha", hc), 42)

    def test_fallback_to_global_default(self) -> None:
        wl = _workload()
        hc = {"defaults": {"default_timeout_s": 1234}}
        # `beta` has no timeout_s; must fall back to the global default.
        self.assertEqual(config_runtime.suite_timeout_s(wl, "beta", hc), 1234)

    def test_local_suites_use_global_default(self) -> None:
        wl = _workload()
        hc = {"defaults": {"default_timeout_s": 1234}}
        # Local suites have no remote workload contract; they fall back
        # to the global default (this lets `harness budget unit` answer
        # without raising — the wrapper just won't enforce on local).
        self.assertEqual(config_runtime.suite_timeout_s(wl, "guard", hc), 1234)

    def test_unknown_suite_raises(self) -> None:
        wl = _workload()
        hc = {"defaults": {"default_timeout_s": 1234}}
        with self.assertRaises(ValueError):
            config_runtime.suite_timeout_s(wl, "does-not-exist", hc)

    def test_harness_config_defaults_loaded_when_omitted(self) -> None:
        # When harness_config is None, fall back to the committed
        # defaults.toml — must not raise and must return a positive int.
        wl = _workload()
        value = config_runtime.suite_timeout_s(wl, "beta")
        self.assertIsInstance(value, int)
        self.assertGreater(value, 0)

    def test_resolve_suite_config_inheritance_preserved(self) -> None:
        # ``timeout_s`` is a per-suite knob, NOT inheritable. A stray
        # ``timeout_s`` placed at ``[ref]`` (or any stage table) must NOT
        # leak into the budget via deep-merge — only the suite-level
        # value (or the global default) is allowed to win.
        wl = {
            "ref": {"timeout_s": 9999},
            "evals": {
                "gamma": {"stage": "stage1", "runner_kind": "noop", "timeout_s": 77},
                "delta": {"stage": "stage1", "runner_kind": "noop"},
            },
        }
        hc = {"defaults": {"default_timeout_s": 5}}
        self.assertEqual(config_runtime.suite_timeout_s(wl, "gamma", hc), 77)
        # `delta` has no per-suite timeout_s; must NOT pick up ref.timeout_s
        # (9999) — must fall back to defaults.default_timeout_s (5).
        self.assertEqual(config_runtime.suite_timeout_s(wl, "delta", hc), 5)

    def test_exposed_in_module_public_api(self) -> None:
        # Guard against accidental removal from __all__ — downstream
        # callers (app.py, dispatcher.py, cli.py) import it by name.
        self.assertIn("suite_timeout_s", config_runtime.__all__)
        self.assertTrue(callable(config_runtime.suite_timeout_s))


class TestSuiteRefTimeoutSSOT(unittest.TestCase):
    """``config_runtime.suite_ref_timeout_s`` is the single seam owning the
    ref-subprocess wall-clock budget. It lets a gate (e.g. perf-bitwise,
    whose Megatron ref capture under per-FQN hash records runs heavier than
    the ours-side replay) cap the ref side independently of the suite-wide
    ``timeout_s``, falling back to ``timeout_s`` and then the global default
    so existing gates that omit it keep their current behaviour.
    """

    def test_ref_timeout_wins_over_suite_timeout(self) -> None:
        wl = _workload(
            epsilon={
                "stage": "stage1",
                "runner_kind": "noop",
                "timeout_s": 720,
                "ref_timeout_s": 1500,
            }
        )
        hc = {"defaults": {"default_timeout_s": 1234}}
        self.assertEqual(config_runtime.suite_ref_timeout_s(wl, "epsilon", hc), 1500)

    def test_fallback_to_suite_timeout_when_ref_absent(self) -> None:
        # ``alpha`` has timeout_s=42 but no ref_timeout_s → ref side reuses
        # the suite budget (the historical behaviour before the key existed).
        wl = _workload()
        hc = {"defaults": {"default_timeout_s": 1234}}
        self.assertEqual(config_runtime.suite_ref_timeout_s(wl, "alpha", hc), 42)

    def test_fallback_to_global_default_when_both_absent(self) -> None:
        # ``beta`` has neither key → global default.
        wl = _workload()
        hc = {"defaults": {"default_timeout_s": 1234}}
        self.assertEqual(config_runtime.suite_ref_timeout_s(wl, "beta", hc), 1234)

    def test_unknown_suite_raises(self) -> None:
        wl = _workload()
        hc = {"defaults": {"default_timeout_s": 1234}}
        with self.assertRaises(ValueError):
            config_runtime.suite_ref_timeout_s(wl, "does-not-exist", hc)

    def test_ref_timeout_not_inheritable_from_ref_table(self) -> None:
        # A stray ``ref_timeout_s`` at ``[ref]`` must NOT leak into a suite
        # that declares neither ref_timeout_s nor timeout_s — only the
        # per-suite value (or the global default) may win.
        wl = {
            "ref": {"ref_timeout_s": 9999},
            "evals": {
                "delta": {"stage": "stage1", "runner_kind": "noop"},
            },
        }
        hc = {"defaults": {"default_timeout_s": 5}}
        self.assertEqual(config_runtime.suite_ref_timeout_s(wl, "delta", hc), 5)

    def test_exposed_in_module_public_api(self) -> None:
        self.assertIn("suite_ref_timeout_s", config_runtime.__all__)
        self.assertTrue(callable(config_runtime.suite_ref_timeout_s))


if __name__ == "__main__":
    unittest.main()
