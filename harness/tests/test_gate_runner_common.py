"""Tests for the shared gate-runner library and the widened ref cache key.

``evals/scripts/gate_runner_common.sh`` is sourced by BOTH side runners
(``ref/run_gate.sh`` / ``evals/scripts/run_ours.sh``); its functions are
exercised here through real ``bash`` so the tests pin the shell semantics
the runners rely on (exit codes, stdout contract, HASH_ARGS shape, side
env-var names).

The cache-key tests pin the ref-generic-projection plan's Step-1 guarantee:
every file on the ref execution path is hashed by content, because the
environment fingerprint's git half is ``no-git`` inside loop workspaces and
cannot invalidate the cache there.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.dispatcher import (  # noqa: E402
    _REF_RUNNER_DEPENDENCY_FILES,
    _ref_execution_dep_files,
    _scripted_ref_cache_dir,
)

_LIB = REPO_ROOT / "evals" / "scripts" / "gate_runner_common.sh"
_RUNNERS = (
    REPO_ROOT / "ref" / "run_gate.sh",
    REPO_ROOT / "evals" / "scripts" / "run_ours.sh",
)


def _bash(script: str, *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f'set -euo pipefail; source "{_LIB}"; {script}'],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


@unittest.skipUnless(shutil.which("bash"), "bash required")
class TestRunnerSyntax(unittest.TestCase):
    def test_library_and_runners_parse(self) -> None:
        for path in (_LIB, *_RUNNERS):
            completed = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, f"{path}: {completed.stderr}")

    def test_both_runners_source_the_shared_library(self) -> None:
        for runner in _RUNNERS:
            self.assertIn(
                "source evals/scripts/gate_runner_common.sh",
                runner.read_text(),
                runner,
            )


@unittest.skipUnless(shutil.which("bash"), "bash required")
class TestResolveProduct(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "ref" / "config").mkdir(parents=True)
        (self.root / "workload" / "src" / "config").mkdir(parents=True)
        self.addCleanup(self._tmp.cleanup)

    def _write(self, rel: str) -> None:
        (self.root / rel).write_text("[cli]\n")

    def test_ref_product_path(self) -> None:
        self._write("ref/config/long-train.toml")
        completed = _bash("resolve_product ref long-train", cwd=self.root)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "ref/config/long-train.toml")

    def test_missing_product_fails_with_side_message(self) -> None:
        completed = _bash("resolve_product ref long-train", cwd=self.root)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "ERROR: ref product not found: ref/config/long-train.toml",
            completed.stderr,
        )

    def test_ours_plain_product(self) -> None:
        self._write("workload/src/config/long-train.toml")
        completed = _bash("resolve_product ours long-train", cwd=self.root)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "workload/src/config/long-train.toml")

    def test_label_selects_milestone_variant(self) -> None:
        self._write("workload/src/config/profile-snapshot.toml")
        self._write("workload/src/config/profile-snapshot@long-horizon.toml")
        completed = _bash(
            "resolve_product ours profile-snapshot long-horizon_round3",
            cwd=self.root,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            completed.stdout.strip(),
            "workload/src/config/profile-snapshot@long-horizon.toml",
        )

    def test_label_without_variant_falls_back_to_plain(self) -> None:
        self._write("workload/src/config/profile-snapshot.toml")
        completed = _bash(
            "resolve_product ours profile-snapshot long-horizon_round3",
            cwd=self.root,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "workload/src/config/profile-snapshot.toml")


@unittest.skipUnless(shutil.which("bash"), "bash required")
class TestSetupHashCapture(unittest.TestCase):
    def _run(self, call: str) -> list[str]:
        with tempfile.TemporaryDirectory() as tmp:
            completed = _bash(
                f'{call}; printf "%s\\n" "${{HASH_ARGS[@]+"${{HASH_ARGS[@]}}"}}"; '
                'printf "HOOK=%s\\n" "${HOOK_OUTPUT_FILE:-}"; '
                'printf "CAPTURE=%s\\n" "${FORGE_CAPTURE_OUTPUT_FILE:-}"',
                cwd=Path(tmp),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            return completed.stdout.splitlines()

    def test_level_zero_is_empty(self) -> None:
        lines = self._run("setup_hash_capture ref /run 0 1")
        self.assertEqual(lines, ["", "HOOK=", "CAPTURE="])

    def test_ref_single_step_wire(self) -> None:
        lines = self._run("setup_hash_capture ref /run 2 1")
        self.assertEqual(
            lines,
            [
                "--hash-capture-level",
                "2",
                "--hash-output",
                "/run/ref_hash_dump.json",
                "HOOK=/run/ref_hash_dump.json",
                "CAPTURE=",
            ],
        )

    def test_ours_multi_step_adds_persistent(self) -> None:
        lines = self._run("setup_hash_capture ours /run 1 12")
        self.assertEqual(
            lines,
            [
                "--hash-capture-level",
                "1",
                "--hash-output",
                "/run/ours_hash_dump.json",
                "--persistent",
                "HOOK=",
                "CAPTURE=/run/ours_hash_dump.json",
            ],
        )


class TestRefCacheKeyFileSet(unittest.TestCase):
    """The key must change when ANY file on the ref execution path changes."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        for rel in _REF_RUNNER_DEPENDENCY_FILES:
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {rel}\n")
        launcher = self.root / "ref" / "reference" / "run_stub.sh"
        launcher.parent.mkdir(parents=True, exist_ok=True)
        launcher.write_text("#!/bin/bash\n")
        (self.root / "ref" / "reference" / "model_stub.py").write_text("X = 1\n")
        self.product = self.root / "ref" / "config" / "long-train.toml"
        self.product.parent.mkdir(parents=True, exist_ok=True)
        self.product.write_text("[cli]\nnum_steps = 200\n")

    def _key(self) -> Path:
        deps = _ref_execution_dep_files(self.root, {})
        return _scripted_ref_cache_dir(self.root, "long-train", deps, self.product)

    def test_key_is_stable(self) -> None:
        self.assertEqual(self._key(), self._key())

    def test_every_dep_file_participates(self) -> None:
        base = self._key()
        mutated = [
            *(self.root / rel for rel in _REF_RUNNER_DEPENDENCY_FILES),
            self.root / "ref" / "reference" / "run_stub.sh",
            self.root / "ref" / "reference" / "model_stub.py",
        ]
        for path in mutated:
            original = path.read_text()
            path.write_text(original + "# mutated\n")
            self.assertNotEqual(base, self._key(), f"{path} not in cache key")
            path.write_text(original)
        self.assertEqual(base, self._key())

    def test_new_reference_file_changes_key(self) -> None:
        base = self._key()
        (self.root / "ref" / "reference" / "data_conf.sh").write_text("D=1\n")
        self.assertNotEqual(base, self._key())

    def test_product_bytes_participate(self) -> None:
        base = self._key()
        self.product.write_text("[cli]\nnum_steps = 201\n")
        self.assertNotEqual(base, self._key())

    def test_pycache_is_ignored(self) -> None:
        base = self._key()
        cache = self.root / "ref" / "reference" / "__pycache__"
        cache.mkdir()
        (cache / "model_stub.cpython-311.pyc").write_bytes(b"\x00")
        self.assertEqual(base, self._key())


if __name__ == "__main__":
    unittest.main(verbosity=2)
