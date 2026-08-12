"""Contract tests for ``evals.runner`` and the BEGIN/END wire protocol.

``evals.runner`` is the single suite-side process that the harness transport
spawns. The harness parses its result between fixed BEGIN/END markers — no
fuzzy JSON scanning. Because harness and evals cannot share imports across
that subprocess boundary, this test is the single enforcement point ensuring
the two marker definitions never drift apart.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evals import runner  # noqa: E402
from harness import run_schema, transport  # noqa: E402


class TestWireProtocolMarkerConsistency(unittest.TestCase):
    """The BEGIN/END markers are defined on both sides of the subprocess boundary."""

    def test_markers_match(self) -> None:
        self.assertEqual(transport._RESULT_BEGIN, run_schema.RESULT_BEGIN)
        self.assertEqual(transport._RESULT_END, run_schema.RESULT_END)
        self.assertEqual(runner._RESULT_BEGIN, run_schema.RESULT_BEGIN)
        self.assertEqual(runner._RESULT_END, run_schema.RESULT_END)


class TestModuleTopology(unittest.TestCase):
    def test_runner_module_exists(self) -> None:
        self.assertIsNotNone(
            importlib.util.find_spec("evals.runner"),
            "evals.runner must exist",
        )

    def test_legacy_workload_infra_is_gone(self) -> None:
        self.assertFalse(
            (REPO_ROOT / "workload" / "infra").exists(),
            "workload/infra/ should have been collapsed into evals/",
        )
        for legacy in ("entry.py", "runner.py", "suites.py", "remote_bridge.py"):
            self.assertFalse(
                (REPO_ROOT / "workload" / "infra" / legacy).exists(),
                f"workload/infra/{legacy} should be gone",
            )


class TestBracketedResultProtocol(unittest.TestCase):
    BEGIN = "---HARNESS-RESULT-BEGIN---"
    END = "---HARNESS-RESULT-END---"

    def test_extract_result_from_noisy_stdout(self) -> None:
        stdout = (
            "some torch warning\n"
            f"{self.BEGIN}\n"
            '{"status": "passed", "metrics": {"x": 1}}\n'
            f"{self.END}\n"
            "trailing noise\n"
        )
        result = transport._extract_bracketed_result(stdout)
        self.assertEqual(result, {"status": "passed", "metrics": {"x": 1}})

    def test_missing_markers_returns_none(self) -> None:
        self.assertIsNone(transport._extract_bracketed_result(""))
        self.assertIsNone(
            transport._extract_bracketed_result('{"status": "passed"}'),
        )

    def test_picks_last_block_if_multiple(self) -> None:
        stdout = (
            f"{self.BEGIN}\n"
            '{"status": "stale"}\n'
            f"{self.END}\n"
            f"{self.BEGIN}\n"
            '{"status": "fresh"}\n'
            f"{self.END}\n"
        )
        result = transport._extract_bracketed_result(stdout)
        self.assertEqual(result, {"status": "fresh"})

    def test_transport_rejects_malformed_result_payload(self) -> None:
        with self.assertRaises(ValueError):
            run_schema.validate_run_result({"status": "passed"})

    def test_transport_rejects_invalid_result_status(self) -> None:
        payload = {
            "status": "maybe",
            "suite": "unit",
            "summary": "bad",
            "metrics": {},
            "details": {},
        }
        with self.assertRaises(ValueError):
            run_schema.validate_run_result(payload)

    def test_validate_run_result_requires_schema_version(self) -> None:
        payload = {
            "status": "passed",
            "suite": "unit",
            "summary": "ok",
            "metrics": {},
            "details": {},
        }
        with self.assertRaisesRegex(ValueError, "schema_version"):
            run_schema.validate_run_result(payload)

    def test_validate_run_request_rejects_malformed_boundary_fields(self) -> None:
        request = {
            "schema_version": 1,
            "run_id": "test",
            "suite": "unit",
            "args": [],
            "report": "json",
            "target": "local",
            "gpu_count": 1,
            "artifact_relpath": ".artifacts/runs/test",
            "requested_at": "2026-05-11T00:00:00+00:00",
            "requested_from": "test-host",
            "workload_config": {},
            "workload_config_path": ".artifacts/runs/test/workload_config.json",
            "workload_config_sha": "0" * 64,
        }
        bad_cases = (
            {"args": ["ok", 1]},
            {"report": "yaml"},
            {"target": "remote"},
            {"gpu_count": 0},
            {"artifact_relpath": "/tmp/out"},
            {"artifact_relpath": "../out"},
            {"workload_config_path": "/abs/out.json"},
            {"workload_config_path": "../out.json"},
            {"workload_config_sha": "not-hex"},
            {"workload_config_sha": "0" * 63},
        )
        for override in bad_cases:
            with self.subTest(override=override):
                bad = dict(request)
                bad.update(override)
                with self.assertRaises(ValueError):
                    run_schema.validate_run_request(bad)


class TestRunnerCli(unittest.TestCase):
    def test_run_emits_result_between_markers(self) -> None:
        # Craft a minimal request that exercises the runner end-to-end with
        # a suite that fails fast (no torch / no remote) so we exit via the
        # ``_missing_script_result`` path rather than importing torch.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            (tmp_root / "evals" / "scripts").mkdir(parents=True)
            wc = {
                "workload": {"id": "x", "display_name": "x", "requires_cuda": False},
                "evals": {
                    "loss-gate-200": {
                        "checkpoint_root": "",
                        "num_steps": 1000,
                        "gate_start_step": 900,
                        "gate_end_step": 1000,
                        "max_avg_relative_loss_diff": 0.01,
                        "world_size": 8,
                    }
                },
            }
            request = {
                "schema_version": 1,
                "run_id": "test",
                "suite": "loss-gate-200",
                "args": [],
                "report": "json",
                "target": "local",
                "gpu_count": 8,
                "artifact_relpath": ".artifacts/runs/test",
                "requested_at": "2026-05-11T00:00:00+00:00",
                "requested_from": "test-host",
                "workload_config": wc,
                "workload_config_path": ".artifacts/runs/test/workload_config.json",
                "workload_config_sha": "0" * 64,
            }
            request_path = tmp_root / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")

            env = {"PATH": "/usr/bin:/bin"}
            # Run runner against the real repo root so it can import
            # evals.dispatcher; pass tmp_root as the working repo so it
            # writes artifacts into the tempdir.
            result = subprocess.run(
                [sys.executable, "-m", "evals.runner", "run", str(request_path), str(tmp_root)],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            parsed = transport._extract_bracketed_result(result.stdout)
            self.assertIsNotNone(parsed)
            assert parsed is not None  # for type checkers
            run_schema.validate_run_result(parsed)
            self.assertEqual(parsed["suite"], "loss-gate-200")

    def test_runner_writes_schema_version_to_result_artifact(self) -> None:
        request = {
            "schema_version": 1,
            "run_id": "test",
            "suite": "unit",
            "args": [],
            "report": "json",
            "target": "local",
            "gpu_count": 1,
            "artifact_relpath": ".artifacts/runs/test",
            "requested_at": "2026-05-11T00:00:00+00:00",
            "requested_from": "test-host",
            "workload_config": {"evals": {"unit": {"runner_kind": "unit"}}},
            "workload_config_path": ".artifacts/runs/test/workload_config.json",
            "workload_config_sha": "0" * 64,
        }
        raw_result = {
            "schema_version": 1,
            "status": "passed",
            "suite": "unit",
            "summary": "ok",
            "metrics": {},
            "details": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            with mock.patch("evals.runner.dispatcher.run_suite", return_value=raw_result):
                result = runner.run_request(request, repo_root)

            result_path = repo_root / ".artifacts" / "runs" / "test" / "result.json"
            artifact = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(artifact["schema_version"], 1)

    def test_runner_validates_request_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            request_path = tmp_root / "request.json"
            request_path.write_text(
                json.dumps({"suite": "loss-gate-200"}),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evals.runner",
                    "run",
                    str(request_path),
                    str(tmp_root),
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
                env={"PATH": "/usr/bin:/bin"},
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Malformed run request", result.stderr)


if __name__ == "__main__":
    unittest.main()
