"""Contract tests for ``harness run --timeout`` and ``HARNESS_RUN_TIMEOUT_S``.

The override path is the single, deliberately-friction-laden escape hatch
for the per-suite SSOT (``[evals.<suite>].timeout_s``). Plan §A / §E.3 pin
the following invariants — these tests guard them:

- CLI ``--timeout`` wins over the env var.
- Env var wins over the SSOT default.
- Invalid input (zero / negative / non-int) fails fast.
- ``harness budget <suite>`` always returns the SSOT, regardless of any
  override that may be active for a concurrent run.
- The result.json's ``details.classification`` records the source and the
  literal origin (``--timeout 12`` / ``HARNESS_RUN_TIMEOUT_S=8``).
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["FORGE_REPO_ROOT"] = str(REPO_ROOT)

from harness import app, cli, config_runtime  # noqa: E402


class TestResolveEffectiveTimeout(unittest.TestCase):
    def test_cli_override_wins_over_env_and_default(self) -> None:
        decision = app._resolve_effective_timeout(
            ssot_default=600,
            cli_override=12,
            env={"HARNESS_RUN_TIMEOUT_S": "8"},
        )
        self.assertEqual(decision["effective_timeout_s"], 12)
        self.assertEqual(decision["timeout_source"], "override")
        self.assertEqual(decision["timeout_origin_argv"], "--timeout 12")

    def test_env_wins_over_default(self) -> None:
        decision = app._resolve_effective_timeout(
            ssot_default=600,
            cli_override=None,
            env={"HARNESS_RUN_TIMEOUT_S": "8"},
        )
        self.assertEqual(decision["effective_timeout_s"], 8)
        self.assertEqual(decision["timeout_source"], "env")
        self.assertEqual(decision["timeout_origin_argv"], "HARNESS_RUN_TIMEOUT_S=8")

    def test_default_when_no_override(self) -> None:
        decision = app._resolve_effective_timeout(
            ssot_default=600,
            cli_override=None,
            env={},
        )
        self.assertEqual(decision["effective_timeout_s"], 600)
        self.assertEqual(decision["timeout_source"], "config")
        self.assertIsNone(decision["timeout_origin_argv"])

    def test_empty_env_treated_as_unset(self) -> None:
        decision = app._resolve_effective_timeout(
            ssot_default=600,
            cli_override=None,
            env={"HARNESS_RUN_TIMEOUT_S": ""},
        )
        self.assertEqual(decision["timeout_source"], "config")

    def test_cli_zero_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            app._resolve_effective_timeout(ssot_default=600, cli_override=0, env={})

    def test_cli_negative_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            app._resolve_effective_timeout(ssot_default=600, cli_override=-5, env={})

    def test_env_non_integer_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "HARNESS_RUN_TIMEOUT_S"):
            app._resolve_effective_timeout(
                ssot_default=600,
                cli_override=None,
                env={"HARNESS_RUN_TIMEOUT_S": "abc"},
            )

    def test_env_negative_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            app._resolve_effective_timeout(
                ssot_default=600,
                cli_override=None,
                env={"HARNESS_RUN_TIMEOUT_S": "-3"},
            )


class TestCliTimeoutFlag(unittest.TestCase):
    def test_run_subparser_exposes_timeout_flag(self) -> None:
        parser = cli._build_parser()
        parsed = parser.parse_args(["run", "op-status", "--timeout", "12"])
        self.assertEqual(parsed.timeout, 12)

    def test_run_timeout_defaults_to_none(self) -> None:
        parser = cli._build_parser()
        parsed = parser.parse_args(["run", "op-status"])
        self.assertIsNone(parsed.timeout)

    def test_run_timeout_rejects_non_integer(self) -> None:
        parser = cli._build_parser()
        # argparse raises SystemExit on type=int conversion failure.
        with self.assertRaises(SystemExit):
            parser.parse_args(["run", "op-status", "--timeout", "abc"])


class TestRequestSnapshotInjection(unittest.TestCase):
    """``_build_run_request`` is the single seam carrying effective_timeout
    into the runner; verify it overwrites the snapshot's per-suite value."""

    def test_effective_timeout_overrides_snapshot(self) -> None:
        harness_config = config_runtime.load_harness_config()
        _, workload_config = config_runtime.load_workload_config(
            None,
            include_user_config=False,
        )
        evals = workload_config.get("evals", {})
        suite = next(s for s, cfg in evals.items() if "timeout_s" in cfg)
        original = int(evals[suite]["timeout_s"])

        request = app._build_run_request(
            suite=suite,
            suite_args=[],
            report="text",
            harness_config=harness_config,
            workload_config=workload_config,
            effective_timeout_s=original + 1234,
        )

        snapshot_evals = request["workload_config"]["evals"]
        self.assertEqual(snapshot_evals[suite]["timeout_s"], original + 1234)

    def test_no_override_preserves_ssot(self) -> None:
        harness_config = config_runtime.load_harness_config()
        _, workload_config = config_runtime.load_workload_config(
            None,
            include_user_config=False,
        )
        suite = next(s for s, cfg in workload_config.get("evals", {}).items() if "timeout_s" in cfg)
        original = int(workload_config["evals"][suite]["timeout_s"])

        request = app._build_run_request(
            suite=suite,
            suite_args=[],
            report="text",
            harness_config=harness_config,
            workload_config=workload_config,
        )
        self.assertEqual(request["workload_config"]["evals"][suite]["timeout_s"], original)


class TestLocalSuiteRejectsTimeoutOverride(unittest.TestCase):
    def test_run_command_rejects_timeout_for_local_suite(self) -> None:
        with self.assertRaisesRegex(ValueError, "local suite"):
            app.run_command(suite="guard", override_timeout_s=60)


class TestBudgetIgnoresOverride(unittest.TestCase):
    """E.3(e): ``harness budget`` MUST always print the toml SSOT, even
    when ``HARNESS_RUN_TIMEOUT_S`` is set for a concurrent run.

    The override pipeline is a per-run decision; budget query is a config
    derivation. Conflating them would make ``budget`` lie about the SSOT.
    """

    def test_budget_unaffected_by_env_override(self) -> None:
        _, workload_config = config_runtime.load_workload_config(None)
        suite = next(s for s, cfg in workload_config.get("evals", {}).items() if "timeout_s" in cfg)
        ssot = config_runtime.suite_timeout_s(workload_config, suite)

        with mock.patch.dict(os.environ, {"HARNESS_RUN_TIMEOUT_S": "1"}):
            payload = app.budget_command(suite=suite)

        self.assertEqual(payload["payload"]["timeout_s"], ssot)


if __name__ == "__main__":
    unittest.main()
