"""Contract tests for the content-addressed workload-config protocol.

Commit C introduced a file-based config-passing protocol so the
``M2 silent-strip`` bug class becomes structurally impossible: the
parent serialises the full workload config to disk with a sha256
sidecar, and every consumer (subprocess loader, workspace contract,
runner) loads the bytes from disk and rejects any sha drift. This test
file pins the round-trip + drift-rejection invariants that make the
protocol safe.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from harness import config_runtime


class TestCanonicalBytes(unittest.TestCase):
    def test_canonicalisation_is_stable_under_key_order(self) -> None:
        a = {"workload": {"id": "x"}, "evals": {"unit": {"runner_kind": "unit"}}}
        b = {"evals": {"unit": {"runner_kind": "unit"}}, "workload": {"id": "x"}}
        self.assertEqual(
            config_runtime.canonical_workload_config_bytes(a),
            config_runtime.canonical_workload_config_bytes(b),
        )

    def test_canonical_bytes_end_with_newline(self) -> None:
        payload = config_runtime.canonical_workload_config_bytes({"x": 1})
        self.assertTrue(payload.endswith(b"\n"))


class TestDumpLoadRoundtrip(unittest.TestCase):
    def test_round_trip_preserves_every_field(self) -> None:
        cfg = {
            "workload": {"id": "x", "display_name": "x", "requires_cuda": False},
            "model": {"hidden_size": 4096, "num_layers": 28},
            "optim": {"lr": 0.001, "weight_decay": 0.1},
            "evals": {"unit": {"runner_kind": "unit"}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "workload_config.json"
            path, digest = config_runtime.dump_workload_config(cfg, dst)

            self.assertEqual(path, dst)
            self.assertTrue(dst.is_file())
            sidecar = dst.with_suffix(dst.suffix + ".sha256")
            self.assertTrue(sidecar.is_file())
            self.assertIn(digest, sidecar.read_text(encoding="utf-8"))

            loaded = config_runtime.load_workload_config_from_file(dst, expected_sha=digest)
            self.assertEqual(loaded, cfg)

    def test_load_rejects_sha_drift(self) -> None:
        cfg = {"workload": {"id": "x"}}
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "workload_config.json"
            path, digest = config_runtime.dump_workload_config(cfg, dst)

            # Mutate a single byte to simulate disk corruption / tampering.
            content = path.read_bytes()
            tampered = content.replace(b'"x"', b'"y"')
            self.assertNotEqual(content, tampered)
            path.write_bytes(tampered)

            with self.assertRaisesRegex(ValueError, "workload_config sha mismatch"):
                config_runtime.load_workload_config_from_file(path, expected_sha=digest)

    def test_sidecar_format_matches_sha256sum(self) -> None:
        """``<sha>  <filename>\\n`` is the format POSIX ``sha256sum`` emits."""
        cfg = {"x": 1}
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "config.json"
            _, digest = config_runtime.dump_workload_config(cfg, dst)
            sidecar = dst.with_suffix(dst.suffix + ".sha256")
            self.assertEqual(sidecar.read_text(encoding="utf-8"), f"{digest}  {dst.name}\n")


class TestSubprocessProtocolIsomorphism(unittest.TestCase):
    """``harness echo-config`` MUST emit the same canonical bytes the
    parent ``config_runtime`` loader produces. The workspace-contract
    invariant subprocess_config_isomorphic relies on this property to
    detect FORGE_* / cwd / PYTHONPATH divergences before the loop starts.
    """

    def test_parent_child_byte_equality(self) -> None:
        try:
            _, cfg = config_runtime.load_workload_config(None)
        except FileNotFoundError:
            self.skipTest("workload config not configured in this checkout")

        parent_bytes = config_runtime.canonical_workload_config_bytes(cfg)
        parent_sha = hashlib.sha256(parent_bytes).hexdigest()

        completed = subprocess.run(
            [sys.executable, "-m", "harness.cli", "echo-config"],
            cwd=str(config_runtime.repo_root()),
            capture_output=True,
            check=True,
            timeout=30,
        )
        child_sha = hashlib.sha256(completed.stdout).hexdigest()
        self.assertEqual(child_sha, parent_sha)
        # The stderr surface advertises the sha so operators can diff
        # against the parent without re-hashing stdout.
        self.assertIn(parent_sha, completed.stderr.decode("utf-8", errors="replace"))


class TestRequestFieldsCarryConfigPathAndSha(unittest.TestCase):
    """The ``_build_run_request`` seam MUST emit ``workload_config_path``
    and ``workload_config_sha`` so downstream consumers (runner subprocess,
    web orchestrator) can validate the on-disk config without trusting
    the in-memory dict the request was constructed from.
    """

    def test_request_schema_requires_path_and_sha(self) -> None:
        from harness import run_schema

        required = set(run_schema.schema_metadata()["request_required_keys"])
        self.assertIn("workload_config_path", required)
        self.assertIn("workload_config_sha", required)

    def test_request_rejects_bad_sha(self) -> None:
        from harness import run_schema

        base = {
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
        # Baseline passes.
        run_schema.validate_run_request(dict(base))
        # Non-hex sha -> reject.
        bad = dict(base, workload_config_sha="z" * 64)
        with self.assertRaisesRegex(ValueError, "workload_config_sha"):
            run_schema.validate_run_request(bad)
        # Wrong length -> reject.
        bad = dict(base, workload_config_sha="0" * 63)
        with self.assertRaisesRegex(ValueError, "workload_config_sha"):
            run_schema.validate_run_request(bad)
        # Unsafe relative path -> reject.
        bad = dict(base, workload_config_path="../escape.json")
        with self.assertRaisesRegex(ValueError, "workload_config_path"):
            run_schema.validate_run_request(bad)


class TestBuildRunRequestEmitsConfigFile(unittest.TestCase):
    """``app._build_run_request`` MUST write the workload config to disk
    inside the artifact directory and stamp the request with the file's
    sha. Without this stamp the runner subprocess cannot detect
    parent/child divergence even if it loads from the path.
    """

    def test_artifact_directory_receives_config_file(self) -> None:
        from unittest import mock

        from harness import app

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            runs_root = repo_root / ".artifacts" / "runs"
            runs_root.mkdir(parents=True)

            workload_config = {
                "workload": {"id": "x", "display_name": "x", "requires_cuda": False},
                "evals": {"unit": {"runner_kind": "unit", "timeout_s": 60}},
            }
            harness_cfg = {
                "artifacts": {
                    "subtrees": {"runs": "runs"},
                    "retention": {"per_suite": 0},
                }
            }

            with (
                mock.patch.object(config_runtime, "repo_root", return_value=repo_root),
                mock.patch.object(
                    config_runtime,
                    "artifact_subtree",
                    return_value=runs_root,
                ),
            ):
                request = app._build_run_request(
                    suite="unit",
                    suite_args=[],
                    report="json",
                    harness_config=harness_cfg,
                    workload_config=workload_config,
                )

            self.assertIn("workload_config_path", request)
            self.assertIn("workload_config_sha", request)

            on_disk = repo_root / request["workload_config_path"]
            self.assertTrue(on_disk.is_file(), f"expected {on_disk} to exist")
            actual = hashlib.sha256(on_disk.read_bytes()).hexdigest()
            self.assertEqual(actual, request["workload_config_sha"])

            loaded = json.loads(on_disk.read_bytes().decode("utf-8"))
            for top_level in ("workload", "evals"):
                self.assertIn(top_level, loaded)


if __name__ == "__main__":
    unittest.main()
