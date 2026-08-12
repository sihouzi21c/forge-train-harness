"""End-to-end wiring of the SSOT timeout through ``app._run_gpu_suite``.

These tests cover the seam between layers (CLI/env decision → snapshot
injection → transport invocation → result.json stamping) using stub
transports so they remain hermetic — no real ``evals.runner`` subprocess
is launched.

The contract under test:

1. ``_run_gpu_suite`` resolves an ``effective_timeout_s`` once and
   passes ``effective + _TRANSPORT_BUDGET_BUFFER_S`` to ``transport.run``.
2. The same effective value is injected into the workload-config
   snapshot's ``evals[suite].timeout_s`` so the dispatcher reads the
   override naturally.
3. When the transport raises ``TransportTimeoutError``, ``_run_gpu_suite``
   synthesizes a structured failed result tagged
   ``classification.reason = "transport_killed_by_budget"`` instead of
   letting the exception escape with no provenance.
4. The transport budget buffer constant is exposed for the wrapper
   script and is a small positive integer (sanity guard).
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["FORGE_REPO_ROOT"] = str(REPO_ROOT)

from harness import app, config_runtime, run_schema, transport  # noqa: E402


def _pick_suite_with_timeout() -> tuple[str, int]:
    _, wl = config_runtime.load_workload_config(None)
    for s, cfg in wl.get("evals", {}).items():
        if "timeout_s" in cfg:
            return s, int(cfg["timeout_s"])
    raise RuntimeError("No suite with timeout_s in workload_config")


class _SuccessfulTransport:
    """Stub transport that records the ``timeout`` it was called with
    and returns a minimal valid run_result."""

    def __init__(self, suite: str) -> None:
        self.received_timeout: float | None = None
        self.received_request: dict[str, Any] | None = None
        self._suite = suite

    def doctor(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    def run(
        self,
        request: dict[str, Any],
        repo_root: Path,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        self.received_timeout = timeout
        self.received_request = request
        result = run_schema.make_run_result(
            status="passed",
            suite=self._suite,
            summary="stub success",
            metrics={},
            details={},
        )
        # Persist result.json so app's classification stamp can rewrite
        # it without surprise — matches the real runner contract.
        artifact_dir = repo_root / request["artifact_relpath"]
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
        return result


class _TimeoutTransport:
    """Stub transport that always raises ``TransportTimeoutError``."""

    def __init__(self, timeout_s: float) -> None:
        self._timeout_s = timeout_s

    def doctor(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    def run(
        self,
        request: dict[str, Any],
        repo_root: Path,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        raise transport.TransportTimeoutError(
            command=["fake"],
            timeout_s=self._timeout_s,
            returncode=-9,
            stdout="captured-stdout-tail",
            stderr="captured-stderr-tail",
        )


class TestBudgetBufferConstant(unittest.TestCase):
    def test_buffer_is_small_positive_int(self) -> None:
        # The constant is referenced by ``tools/remote_run.sh`` indirectly
        # (the wrapper adds its own larger outer buffer on top); a typo
        # here that promoted the buffer to "minutes" would silently let
        # runaway subprocesses live much longer than the budget intends.
        self.assertIsInstance(app._TRANSPORT_BUDGET_BUFFER_S, int)
        self.assertGreater(app._TRANSPORT_BUDGET_BUFFER_S, 0)
        self.assertLessEqual(app._TRANSPORT_BUDGET_BUFFER_S, 300)


class TestTransportReceivesEffectiveTimeout(unittest.TestCase):
    def test_ssot_default_path(self) -> None:
        suite, ssot = _pick_suite_with_timeout()
        stub = _SuccessfulTransport(suite)
        with mock.patch.object(transport, "create_transport", return_value=stub):
            app.run_command(suite=suite)
        self.assertEqual(
            stub.received_timeout,
            float(ssot + app._TRANSPORT_BUDGET_BUFFER_S),
        )
        # Snapshot carries the SSOT untouched.
        self.assertEqual(
            stub.received_request["workload_config"]["evals"][suite]["timeout_s"],
            ssot,
        )

    def test_cli_override_propagates(self) -> None:
        suite, _ = _pick_suite_with_timeout()
        stub = _SuccessfulTransport(suite)
        with mock.patch.object(transport, "create_transport", return_value=stub):
            app.run_command(suite=suite, override_timeout_s=11)
        self.assertEqual(
            stub.received_timeout,
            float(11 + app._TRANSPORT_BUDGET_BUFFER_S),
        )
        self.assertEqual(
            stub.received_request["workload_config"]["evals"][suite]["timeout_s"],
            11,
        )

    def test_env_override_propagates(self) -> None:
        suite, _ = _pick_suite_with_timeout()
        stub = _SuccessfulTransport(suite)
        with (
            mock.patch.dict(os.environ, {"HARNESS_RUN_TIMEOUT_S": "7"}),
            mock.patch.object(transport, "create_transport", return_value=stub),
        ):
            app.run_command(suite=suite)
        self.assertEqual(
            stub.received_timeout,
            float(7 + app._TRANSPORT_BUDGET_BUFFER_S),
        )
        self.assertEqual(
            stub.received_request["workload_config"]["evals"][suite]["timeout_s"],
            7,
        )


class TestTransportKillProducesStructuredResult(unittest.TestCase):
    def test_transport_timeout_synthesizes_failed_result(self) -> None:
        suite, _ = _pick_suite_with_timeout()
        stub = _TimeoutTransport(timeout_s=7.0)
        with mock.patch.object(transport, "create_transport", return_value=stub):
            payload = app.run_command(suite=suite, override_timeout_s=5)

        target = payload["targets"][0]["payload"]
        self.assertEqual(target["status"], "failed")
        classification = target["details"]["classification"]
        self.assertEqual(classification["reason"], "transport_killed_by_budget")
        self.assertEqual(classification["transport_timeout_s"], 7.0)
        self.assertEqual(classification["subprocess_returncode"], -9)
        # Stamp from `_stamp_timeout_classification` is additive — both
        # the kill metadata AND the override provenance survive.
        self.assertEqual(classification["timeout_source"], "override")
        self.assertEqual(classification["effective_timeout_s"], 5)
        self.assertIn("captured-stdout-tail", target["details"]["output_tail"])


class TestStampClassificationPreservesPreexisting(unittest.TestCase):
    """The runner may itself populate ``details.classification`` (e.g. a
    future ``runner_killed_by_budget`` marker). The stamp helper must
    merge in our fields without clobbering prior content."""

    def test_merge_preserves_existing_classification(self) -> None:
        result = run_schema.make_run_result(
            status="passed",
            suite="x",
            summary="ok",
            details={"classification": {"reason": "dispatcher_observed", "step": 7}},
        )
        artifact_dir = REPO_ROOT / ".artifacts" / "test-stamp-merge"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        # Pre-seed result.json so the stamp's rewrite has a target.
        (artifact_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
        decision = {
            "effective_timeout_s": 99,
            "timeout_source": "override",
            "timeout_origin_argv": "--timeout 99",
        }
        app._stamp_timeout_classification(result, decision, artifact_dir)
        cls = result["details"]["classification"]
        self.assertEqual(cls["reason"], "dispatcher_observed")
        self.assertEqual(cls["step"], 7)
        self.assertEqual(cls["timeout_source"], "override")
        self.assertEqual(cls["effective_timeout_s"], 99)


if __name__ == "__main__":
    unittest.main()
