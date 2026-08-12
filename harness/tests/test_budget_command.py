"""Contract tests for ``harness budget`` — the SSOT query CLI.

These tests pin three properties downstream consumers (tools/remote_run.sh,
agent prompts that quote `harness budget …` output) depend on:

1. ``app.budget_command`` returns a payload whose ``payload.timeout_s`` matches
   ``config_runtime.suite_timeout_s`` for the same inputs.
2. ``presentation.render`` for a budget payload emits a bare integer in text
   mode (so ``$(harness budget X)`` substitutes cleanly into arithmetic), and
   a structured JSON object in ``--json`` mode.
3. The CLI plumbing exposes a ``budget`` subcommand with positional ``suite``
   and ``--json`` flags.
"""

from __future__ import annotations

import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["FORGE_REPO_ROOT"] = str(REPO_ROOT)

from harness import app, cli, config_runtime, presentation  # noqa: E402


class TestBudgetCommandPayload(unittest.TestCase):
    def test_payload_matches_ssot_for_known_suite(self) -> None:
        # ``op-status`` is a stable, fast suite committed in eval.toml with
        # a small explicit timeout_s — picking it instead of a long-train
        # value keeps the test resilient to future budget retuning.
        _, workload_config = config_runtime.load_workload_config(None)
        suite = next(s for s, cfg in workload_config.get("evals", {}).items() if "timeout_s" in cfg)
        expected = config_runtime.suite_timeout_s(workload_config, suite)

        payload = app.budget_command(suite=suite)

        self.assertEqual(payload["command"], "budget")
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["payload"]["suite"], suite)
        self.assertEqual(payload["payload"]["timeout_s"], expected)
        self.assertEqual(payload["payload"]["source"], f"evals.{suite}.timeout_s")

    def test_payload_source_is_default_when_suite_omits_timeout(self) -> None:
        # Build a workload_config in-memory so the test does not depend
        # on whichever suite currently omits timeout_s in eval.toml.
        wl = {
            "evals": {
                "noop-suite": {"stage": "stage1", "runner_kind": "noop"},
            },
            "local_suites": {},
        }
        hc = {"defaults": {"default_timeout_s": 1234}}
        with (
            mock.patch.object(config_runtime, "load_harness_config", return_value=hc),
            mock.patch.object(
                config_runtime, "load_workload_config", return_value=(Path("/tmp/x"), wl)
            ),
        ):
            payload = app.budget_command(suite="noop-suite")
        self.assertEqual(payload["payload"]["timeout_s"], 1234)
        self.assertEqual(payload["payload"]["source"], "defaults.default_timeout_s")

    def test_unknown_suite_raises(self) -> None:
        with self.assertRaises(ValueError):
            app.budget_command(suite="this-suite-does-not-exist")


class TestBudgetRendering(unittest.TestCase):
    def test_text_render_is_bare_integer(self) -> None:
        # ``tools/remote_run.sh`` does ``budget=$(harness budget X)``
        # then ``$((budget + buffer))``. Anything other than a bare
        # integer on stdout would break that arithmetic — pin it.
        payload = {
            "command": "budget",
            "status": "ready",
            "report": "text",
            "payload": {"suite": "x", "timeout_s": 600, "source": "evals.x.timeout_s"},
        }
        self.assertEqual(presentation.render(payload, "text"), "600")

    def test_json_render_includes_suite_timeout_source(self) -> None:
        payload = {
            "command": "budget",
            "status": "ready",
            "report": "json",
            "payload": {"suite": "x", "timeout_s": 600, "source": "evals.x.timeout_s"},
        }
        rendered = json.loads(presentation.render(payload, "json"))
        self.assertEqual(rendered["payload"]["suite"], "x")
        self.assertEqual(rendered["payload"]["timeout_s"], 600)
        self.assertEqual(rendered["payload"]["source"], "evals.x.timeout_s")


class TestBudgetCli(unittest.TestCase):
    def test_parser_exposes_budget_subcommand(self) -> None:
        parser = cli._build_parser()
        sub = next(action for action in parser._actions if action.dest == "command")
        self.assertIn("budget", sub.choices)

    def test_parser_requires_suite_positional(self) -> None:
        parser = cli._build_parser()
        parsed = parser.parse_args(["budget", "op-status"])
        self.assertEqual(parsed.command, "budget")
        self.assertEqual(parsed.suite, "op-status")
        self.assertFalse(parsed.json_output)

    def test_parser_accepts_json_flag(self) -> None:
        parser = cli._build_parser()
        parsed = parser.parse_args(["budget", "op-status", "--json"])
        self.assertTrue(parsed.json_output)

    def test_cli_main_prints_bare_integer(self) -> None:
        # End-to-end: parse → app.budget_command → presentation.render →
        # stdout. Guards against the budget pipeline silently returning
        # something other than a bare integer.
        _, workload_config = config_runtime.load_workload_config(None)
        suite = next(s for s, cfg in workload_config.get("evals", {}).items() if "timeout_s" in cfg)
        expected = config_runtime.suite_timeout_s(workload_config, suite)

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli.main(["budget", suite])

        self.assertEqual(rc, 0)
        self.assertEqual(buf.getvalue().strip(), str(expected))


if __name__ == "__main__":
    unittest.main()
