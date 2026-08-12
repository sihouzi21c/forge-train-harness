"""Pure-function and contract tests for the harness layer.

Covers only side-effect-free behavior so these tests run on any environment
(no torch, no remote target required).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from harness import app, config_runtime, transport  # noqa: E402


class TestTransportPublicSurface(unittest.TestCase):
    """Transport implementation classes must not leak into ``__all__``."""

    def test_transport_classes_are_private(self) -> None:
        self.assertFalse(
            hasattr(transport, "LocalTransport"),
            "LocalTransport should be private (_LocalTransport); "
            "use transport.create_transport() instead.",
        )
        self.assertTrue(callable(getattr(transport, "create_transport", None)))

    def test_config_dataclasses_not_in_all(self) -> None:
        exported = set(transport.__all__)
        self.assertNotIn("LocalTargetConfig", exported)
        self.assertIn("TargetConfig", exported)


class TestRepoRootDiscovery(unittest.TestCase):
    def test_repo_root_is_harness_repo(self):
        os.environ.pop("FORGE_REPO_ROOT", None)
        original = Path.cwd()
        try:
            os.chdir(REPO_ROOT)
            # Clear lru cache because it keyed on cwd/env
            config_runtime._resolve_repo_root.cache_clear()
            discovered = config_runtime.repo_root()
            self.assertEqual(discovered.resolve(), REPO_ROOT.resolve())
        finally:
            os.chdir(original)
            config_runtime._resolve_repo_root.cache_clear()


class TestUserConfigDirEnvOverride(unittest.TestCase):
    """``FORGE_CONFIG_DIR`` redirects every per-axis config path lookup.

    Per-loop config isolation depends on this single redirect point: the
    wrapper sets ``FORGE_CONFIG_DIR`` to the loop's
    ``.artifacts/forge_train/<id>/config/`` and every downstream axis
    helper (`_remote_config_path` etc.) inherits the new root without
    further plumbing.
    """

    def tearDown(self) -> None:
        os.environ.pop("FORGE_CONFIG_DIR", None)

    def test_user_config_dir_defaults_to_repo_config(self) -> None:
        os.environ.pop("FORGE_CONFIG_DIR", None)
        self.assertEqual(
            config_runtime._user_config_dir(),
            config_runtime.repo_root() / "config",
        )

    def test_user_config_dir_honors_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FORGE_CONFIG_DIR"] = tmp
            self.assertEqual(
                config_runtime._user_config_dir(),
                Path(tmp),
            )

    def test_axis_helpers_follow_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FORGE_CONFIG_DIR"] = tmp
            root = Path(tmp)
            self.assertEqual(config_runtime._remote_config_path(), root / "remote.toml")
            self.assertEqual(config_runtime._ref_config_path(), root / "ref.toml")
            self.assertEqual(config_runtime._data_config_path(), root / "data.toml")
            self.assertEqual(config_runtime._agent_config_path(), root / "agent.toml")
            self.assertEqual(config_runtime._model_config_path(), root / "model.toml")
            self.assertEqual(config_runtime._optim_config_path(), root / "optim.toml")

    def test_default_workload_config_path_follows_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FORGE_CONFIG_DIR"] = tmp
            user_copy = Path(tmp) / "eval.toml"
            user_copy.write_text("# placeholder\n", encoding="utf-8")
            self.assertEqual(
                config_runtime._default_workload_config_path(),
                user_copy,
            )


class TestInfoCommand(unittest.TestCase):
    def test_info_returns_expected_shape(self):
        os.environ["FORGE_REPO_ROOT"] = str(REPO_ROOT)
        config_runtime._resolve_repo_root.cache_clear()
        # Seed a per-loop config dir from the committed templates — the
        # top-level harness/config/*.toml files no longer exist (Method F).
        templates = REPO_ROOT / "config"
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp)
            for axis in ("ref", "data", "remote", "agent", "eval", "model", "optim"):
                src_dir = templates / axis
                if axis == "eval":
                    # eval templates live in variant directories:
                    # config/eval/<variant>/<variant>.toml (gate_config/ siblings
                    # are not axis templates).
                    variant = sorted(p for p in src_dir.iterdir() if p.is_dir())[0]
                    src = variant / f"{variant.name}.toml"
                else:
                    src = sorted(src_dir.glob("*.toml"))[0]
                (cfg / f"{axis}.toml").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            os.environ["FORGE_CONFIG_DIR"] = str(cfg)
            try:
                payload = app.info_command(json_output=True)
            finally:
                os.environ.pop("FORGE_CONFIG_DIR", None)
                os.environ.pop("FORGE_REPO_ROOT", None)
                config_runtime._resolve_repo_root.cache_clear()
        self.assertEqual(payload["command"], "info")
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["report"], "json")
        self.assertIn("workload", payload["payload"])
        self.assertIn("supported_suites", payload["payload"]["workload"])


class TestWriteJson(unittest.TestCase):
    def test_write_roundtrip(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "x.json"
            config_runtime.write_json(target, {"k": 1})
            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8")),
                {"k": 1},
            )


class TestBuildRunRequest(unittest.TestCase):
    """Requests do not encode gate execution shape; ref metadata owns it."""

    def setUp(self) -> None:
        os.environ["FORGE_REPO_ROOT"] = str(REPO_ROOT)
        config_runtime._resolve_repo_root.cache_clear()

    def tearDown(self) -> None:
        os.environ.pop("FORGE_REPO_ROOT", None)
        config_runtime._resolve_repo_root.cache_clear()

    def _harness_cfg(self) -> dict:
        return {
            "artifacts": {"root": ".artifacts", "subtrees": {"runs": "runs"}},
        }

    def test_request_gpu_count_is_none_pending_ref_metadata(self):
        wc = {
            "evals": {
                "loss-gate-200": {},
                "unrelated": {},
            }
        }
        req = app._build_run_request(
            suite="loss-gate-200",
            report="text",
            harness_config=self._harness_cfg(),
            workload_config=wc,
        )
        self.assertEqual(req["schema_version"], 1)
        self.assertIsNone(req["gpu_count"])
        self.assertEqual(req["suite"], "loss-gate-200")
        self.assertEqual(req["target"], "local")
        self.assertIn("run_id", req)
        self.assertEqual(set(req["workload_config"]["evals"]), {"loss-gate-200"})

    def test_runner_bound_snapshot_omits_harness_only_sections(self):
        # Audit follow-up: ``local_suites`` and ``automation`` are
        # harness-only surfaces — local suites bypass the runner
        # subprocess entirely (handled inside ``harness.app``) and
        # automation knobs are consumed by ``agent-loop.sh`` / Stage 2
        # fan-out helpers, never by a gate runner. The snapshot the
        # runner receives must therefore omit both.
        wc = {
            "workload": {
                "id": "x",
                "display_name": "x",
                "requires_cuda": False,
            },
            "env": {"FOO": "1"},
            "runtime": {"distributed": {"master_addr": "localhost", "master_port": "29500"}},
            "ref": {"seed": 1234},
            "evals": {"unit-suite": {"stage": "stage1", "runner_kind": "k"}},
            "local_suites": {
                "guard": {"stage": "local", "runner_kind": "guard"},
            },
            "automation": {
                "stage2": {"max_concurrent": 1, "max_op_long_failures": 0},
            },
        }
        snap = app._workload_config_snapshot(wc, "unit-suite")
        self.assertNotIn("local_suites", snap)
        self.assertNotIn("automation", snap)
        for key in ("workload", "env", "runtime", "ref"):
            self.assertIn(key, snap)
        self.assertEqual(set(snap["evals"]), {"unit-suite"})


class TestPublicSurface(unittest.TestCase):
    """Guard against leaking helpers that have no external consumer."""

    def test_config_runtime_exports(self):
        removed = (
            "default_report",
            "resolve_path",
            "load_local_harness_config",
            "REPORT_CHOICES",
            "read_json",
            "user_config_dir",
            "default_workload_config_path",
            "local_harness_config_path",
            "targets_config_path",
            "artifact_root",
        )
        for name in removed:
            self.assertFalse(
                hasattr(config_runtime, name),
                f"{name} must not be a public symbol of config_runtime",
            )

    def test_config_runtime_has_all(self):
        self.assertTrue(
            hasattr(config_runtime, "__all__"),
            "config_runtime must define __all__",
        )

    def test_cli_exports(self):
        from harness import cli

        self.assertFalse(
            hasattr(cli, "build_parser"),
            "build_parser should be private (_build_parser)",
        )
        self.assertTrue(hasattr(cli, "__all__"))

    def test_presentation_exports(self):
        from harness import presentation

        self.assertFalse(
            hasattr(presentation, "render_text"),
            "render_text should be private (_render_text)",
        )
        self.assertTrue(hasattr(presentation, "__all__"))

    def test_transport_has_all(self):
        self.assertTrue(
            hasattr(transport, "__all__"),
            "transport must define __all__",
        )

    def test_app_has_all(self):
        self.assertTrue(
            hasattr(app, "__all__"),
            "app must define __all__",
        )


class TestPruneOldRuns(unittest.TestCase):
    """Sliding-window retention for ``.artifacts/runs/<suite>-*``.

    Lives next to ``_build_run_request`` because both are part of the
    same run-allocation pipeline. The helper is pure (operates on the
    caller-supplied ``runs_root``) so the tests just stage a temp dir
    and assert on directory presence/absence.
    """

    @staticmethod
    def _mk(runs_root: Path, name: str) -> Path:
        path = runs_root / name
        path.mkdir(parents=True)
        (path / "result.json").write_text("{}", encoding="utf-8")
        return path

    def test_zero_keep_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = Path(tmp) / "runs"
            runs_root.mkdir()
            old = self._mk(runs_root, "forward-align-20250101-000000-aaaaaa")
            app._prune_old_runs(
                runs_root,
                "forward-align",
                {"artifacts": {"retention": {"per_suite": 0}}},
            )
            self.assertTrue(old.exists())

    def test_missing_retention_section_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = Path(tmp) / "runs"
            runs_root.mkdir()
            old = self._mk(runs_root, "forward-align-20250101-000000-aaaaaa")
            # Empty config (current minimal-test pattern); helper must
            # treat missing/non-int retention as "no retention".
            app._prune_old_runs(runs_root, "forward-align", {})
            self.assertTrue(old.exists())

    def test_keep_three_prunes_oldest_to_two(self) -> None:
        # Helper leaves ``keep - 1`` survivors so the caller's
        # about-to-be-created run lands as the keep-th entry. With
        # ``keep=3`` and 4 existing runs, 2 oldest go.
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = Path(tmp) / "runs"
            runs_root.mkdir()
            names = [
                "forward-align-20250101-000000-aaaaaa",
                "forward-align-20250102-000000-bbbbbb",
                "forward-align-20250103-000000-cccccc",
                "forward-align-20250104-000000-dddddd",
            ]
            for name in names:
                self._mk(runs_root, name)
            app._prune_old_runs(
                runs_root,
                "forward-align",
                {"artifacts": {"retention": {"per_suite": 3}}},
            )
            remaining = sorted(p.name for p in runs_root.iterdir())
            self.assertEqual(remaining, names[2:])

    def test_does_not_touch_other_suites(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = Path(tmp) / "runs"
            runs_root.mkdir()
            other = self._mk(runs_root, "backward-align-20250101-000000-eeeeee")
            for stamp in ("a", "b", "c", "d"):
                self._mk(runs_root, f"forward-align-2025010{stamp}-000000-xxxxxx")
            app._prune_old_runs(
                runs_root,
                "forward-align",
                {"artifacts": {"retention": {"per_suite": 2}}},
            )
            self.assertTrue(other.exists())
            forward_left = sorted(
                p.name for p in runs_root.iterdir() if p.name.startswith("forward-align-")
            )
            self.assertEqual(len(forward_left), 1)

    def test_missing_runs_root_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = Path(tmp) / "runs"  # never created
            app._prune_old_runs(
                runs_root,
                "forward-align",
                {"artifacts": {"retention": {"per_suite": 3}}},
            )
            self.assertFalse(runs_root.exists())


class TestLocalTargetConfig(unittest.TestCase):
    def test_default_values(self):
        cfg = transport.LocalTargetConfig()
        self.assertIsNone(cfg.gpu)
        self.assertEqual(cfg.env, {})

    def test_with_gpu(self):
        cfg = transport.LocalTargetConfig(gpu="0,1")
        self.assertEqual(cfg.gpu, "0,1")

    def test_create_transport_returns_transport(self):
        t = transport.create_transport({})
        self.assertTrue(hasattr(t, "doctor"))
        self.assertTrue(hasattr(t, "run"))


class TestDiffCaptureDicts(unittest.TestCase):
    """``evals._capture_diff.diff_capture_dicts`` comparison logic.

    Pure dict-in / list-out; no torch / GPU. Covers the ``M1.bwd``
    hardening: backward-align must compare ``bwd.*`` activation gradients
    (not only ``grad.*`` parameter gradients) and, under
    ``require_baseline_complete``, must FAIL a baseline key the candidate
    never emitted — closing the hole where a candidate hides a divergence
    by simply not dumping the diverging tensor.
    """

    def setUp(self) -> None:
        from evals._capture_diff import diff_capture_dicts

        self.diff = diff_capture_dicts
        # ref (baseline) captures grad.* + bwd.* + fwd.*
        self.ref = {
            "grad.layers.0.w.postallreduce": {"hash": "g0", "shape": [4, 4], "dtype": "float32"},
            "bwd.output": {"hash": "b0", "shape": [2, 4, 8], "dtype": "bfloat16"},
            "bwd.layers.0": {"hash": "b1", "shape": [2, 4, 8], "dtype": "bfloat16"},
            "fwd.output": {"hash": "f0", "shape": [2, 4, 8], "dtype": "bfloat16"},
        }
        self.bwd_prefix = ("grad.", "bwd.")

    def _by_name(self, entries):
        return {e.name: e for e in entries}

    def test_ref_authoritative_flags_missing_candidate_key(self):
        # candidate omits bwd.output (the real 1B M1 situation: ours never
        # dumped the diverging activation gradient).
        cand = {k: v for k, v in self.ref.items() if k != "bwd.output"}
        out = self._by_name(
            self.diff(cand, self.ref, key_prefix=self.bwd_prefix, require_baseline_complete=True)
        )
        self.assertIn("bwd.output", out, "missing baseline key must surface")
        self.assertFalse(out["bwd.output"].passed)
        self.assertIn("missing", (out["bwd.output"].reason or "").lower())

    def test_default_skips_missing_candidate_key(self):
        # Without the flag the old intersection behaviour holds (M2-M6 /
        # forward-align callers stay unchanged).
        cand = {k: v for k, v in self.ref.items() if k != "bwd.output"}
        out = self._by_name(self.diff(cand, self.ref, key_prefix=self.bwd_prefix))
        self.assertNotIn("bwd.output", out)

    def test_bwd_prefix_catches_divergence(self):
        cand = dict(self.ref)
        cand["bwd.output"] = {"hash": "DIFF", "shape": [2, 4, 8], "dtype": "bfloat16"}
        out = self._by_name(
            self.diff(cand, self.ref, key_prefix=self.bwd_prefix, require_baseline_complete=True)
        )
        self.assertIn("bwd.output", out)
        self.assertFalse(out["bwd.output"].passed)

    def test_bwd_prefix_passes_matching(self):
        out = self._by_name(
            self.diff(
                dict(self.ref), self.ref, key_prefix=self.bwd_prefix, require_baseline_complete=True
            )
        )
        self.assertTrue(out["bwd.output"].passed)
        self.assertTrue(out["bwd.layers.0"].passed)

    def test_prefix_excludes_fwd(self):
        out = self._by_name(
            self.diff(
                dict(self.ref), self.ref, key_prefix=self.bwd_prefix, require_baseline_complete=True
            )
        )
        self.assertNotIn("fwd.output", out)

    def test_grad_keys_still_compared(self):
        out = self._by_name(
            self.diff(
                dict(self.ref), self.ref, key_prefix=self.bwd_prefix, require_baseline_complete=True
            )
        )
        self.assertIn("grad.layers.0.w.postallreduce", out)
        self.assertTrue(out["grad.layers.0.w.postallreduce"].passed)


class TestBackwardAlignWiring(unittest.TestCase):
    """Static contract: the align verdict's comparison selector is keyed on
    the milestone VALUE (``evals/verdicts/align.py``, thin-dispatcher route):
    ``alignment.backward`` gates BOTH ``grad.*`` and ``bwd.*`` and is
    ref-authoritative; the forward branch stays ``fwd.*`` and lenient.
    """

    def setUp(self) -> None:
        src = (REPO_ROOT / "evals" / "verdicts" / "align.py").read_text(encoding="utf-8")
        marker = 'if milestone == "alignment.backward":'
        start = src.index(marker)
        end = src.index("\n\n", start)
        self.branch = src[start:end]
        bwd_end = self.branch.index("else:")
        self.bwd_branch = self.branch[:bwd_end]
        self.fwd_branch = self.branch[bwd_end:]

    def test_backward_align_gates_bwd_and_is_ref_authoritative(self):
        self.assertIn('("grad.", "bwd.")', self.bwd_branch)
        self.assertIn("require_baseline_complete = True", self.bwd_branch)

    def test_forward_align_unchanged(self):
        self.assertIn('key_prefix = "fwd."', self.fwd_branch)
        self.assertIn("require_baseline_complete = False", self.fwd_branch)
        self.assertNotIn("require_baseline_complete = True", self.fwd_branch)


if __name__ == "__main__":
    unittest.main()
