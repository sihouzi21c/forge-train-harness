"""Verify the anti-proxy guard catches candidate-engine proxy / synthesis
patterns while leaving clean stubs and legitimate prose alone.

Three layers of coverage:

1. The committed ``workload/src/training_engine_tensor/train_loop.py``
   stub on this branch is clean; ``harness run anti-proxy`` must
   currently exit 0.
2. A tempdir fixture with the canonical hack patterns (subprocess
   shell-out to ``ref/bridges/bridge.sh``, ``ref_script_runner`` import,
   ``_run_ref_*`` / ``_synthetic_*`` function names, hardcoded
   ``mfu_e2e_*`` metric kwarg) must trip every category.
3. Docstring-only mentions of the same words MUST NOT trip the rule
   (otherwise the stub's architectural prose breaks the guard).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from harness import anti_proxy_guard  # noqa: E402

CLEAN_STUB = '''"""Candidate stub.

Architectural prose may legitimately mention ref/reference/<script>
and evals.harness_hook and subprocess as concepts; docstrings are
skipped so the prose does not trip the guard.
"""

from __future__ import annotations


def run_training_loop(config) -> int:
    return 0
'''


PROXY_FIXTURE = '''"""Engine docstring; the body below is the real engine code under audit."""

from __future__ import annotations

import subprocess

from tools import ref_script_runner


def _run_m1_capture(config):
    bridge = "ref/bridges/bridge.sh"
    return subprocess.run(["bash", bridge], check=True)


def _run_ref_script_loss_trajectory(config):
    return ref_script_runner.run_ref_script(config)


def _synthetic_resume_loss(step):
    return 0.5


def emit_metric(step):
    return f"[LOSS] mfu_e2e_standard=99.0 global_loss=2.5"
'''


class TestAntiProxyGuardRepoBaseline(unittest.TestCase):
    def test_committed_stub_passes(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "harness.cli", "run", "anti-proxy", "--json"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"baseline anti-proxy must pass on clean stub:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )


class TestAntiProxyGuardClean(unittest.TestCase):
    def test_clean_stub_under_workload_src_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp) / "workload" / "src" / "engine"
            engine.mkdir(parents=True)
            (engine / "stub.py").write_text(CLEAN_STUB, encoding="utf-8")
            violations = anti_proxy_guard.scan_paths([str(engine)])
            self.assertEqual([], violations, [v.reason for v in violations])


class TestAntiProxyGuardCatchesHacks(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.engine = Path(self._tmp.name) / "workload" / "src" / "engine"
        self.engine.mkdir(parents=True)
        (self.engine / "hacked.py").write_text(PROXY_FIXTURE, encoding="utf-8")
        self.violations = anti_proxy_guard.scan_paths([str(self.engine)])
        self.reasons = {v.reason for v in self.violations}

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_catches_ref_path_literal(self) -> None:
        self.assertIn("candidate references ref/ path literal", self.reasons)

    def test_catches_bridge_sh_literal(self) -> None:
        self.assertIn("candidate references ref bridge script", self.reasons)

    def test_catches_ref_script_runner_import(self) -> None:
        self.assertIn("candidate imports ref_script_runner helper", self.reasons)

    def test_catches_subprocess_run(self) -> None:
        self.assertIn(
            "candidate uses subprocess.run (proxy-to-ref smell)",
            self.reasons,
        )

    def test_catches_run_ref_name_prefix(self) -> None:
        self.assertIn("candidate name contains _run_ref_", self.reasons)

    def test_catches_synthetic_name_prefix(self) -> None:
        self.assertIn("candidate name contains _synthetic_", self.reasons)

    def test_catches_hardcoded_mfu_kwarg(self) -> None:
        self.assertTrue(
            any("hardcodes mfu_e2e_standard=99.0" in v.reason for v in self.violations),
            f"missed hardcoded mfu kwarg; reasons={sorted(self.reasons)}",
        )


class TestAntiProxyGuardSkipsDocstrings(unittest.TestCase):
    def test_docstring_only_mentions_do_not_trip(self) -> None:
        source = '''"""Module docstring talks about subprocess.run and ref/bridges/bridge.sh
and evals.harness_hook and _synthetic_resume_loss — all just words.
"""

from __future__ import annotations


def f():
    """Function docstring: subprocess.run is also fine in here."""
    return 0
'''
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp) / "workload" / "ops" / "stub"
            engine.mkdir(parents=True)
            (engine / "x.py").write_text(source, encoding="utf-8")
            violations = anti_proxy_guard.scan_paths([str(engine)])
            self.assertEqual(
                [],
                violations,
                f"docstring mentions tripped the guard: "
                f"{[(v.line_number, v.reason) for v in violations]}",
            )

    def test_assignment_string_not_docstring_is_flagged(self) -> None:
        source = """from __future__ import annotations

BRIDGE = "ref/bridges/bridge.sh"
"""
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp) / "workload" / "src" / "x"
            engine.mkdir(parents=True)
            (engine / "y.py").write_text(source, encoding="utf-8")
            violations = anti_proxy_guard.scan_paths([str(engine)])
            self.assertTrue(
                any(v.reason == "candidate references ref/ path literal" for v in violations),
                violations,
            )


if __name__ == "__main__":
    unittest.main()
