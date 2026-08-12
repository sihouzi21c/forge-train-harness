"""Shape / public-surface tests for the suites dispatcher.

These tests run without torch/CUDA because they only inspect module-level
names and function signatures — not runtime behavior.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from harness._compat import tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evals import dispatcher as suites  # noqa: E402
from harness.tests import _gate_render  # noqa: E402

_REGISTRY = REPO_ROOT / "config" / "eval" / "dense_training" / "dense_training.toml"


class TestPublicSurface(unittest.TestCase):
    """dispatcher should not export documentation-only symbols."""

    def test_documentation_symbols_removed(self) -> None:
        banned = (
            "StepResult",
            "TrainingReport",
            "TrainingEngineLoop",
            "TRAINING_ENGINE_CLI_MODULE",
            "TRAINING_ENGINE_CLI_SUBCOMMAND",
            "OUTPUT_PROTOCOL",
        )
        for name in banned:
            self.assertFalse(
                hasattr(suites, name),
                f"dispatcher.{name} is a documentation-only symbol; move to prompt/develop_prompt/*.md",
            )

    def test_internal_symbols_are_private(self) -> None:
        self.assertFalse(
            hasattr(suites, "WORKLOAD_SCRIPT_CONTRACT"),
            "WORKLOAD_SCRIPT_CONTRACT should be private (_WORKLOAD_SCRIPT_CONTRACT)",
        )
        self.assertFalse(
            hasattr(suites, "run_loss_gate"),
            "run_loss_gate should be private (_run_loss_gate)",
        )


class TestSuitesRegistry(unittest.TestCase):
    """Suite names come from workload config; dispatcher owns runner kinds only."""

    def test_runner_kinds_cover_all_configured_suites(self) -> None:
        # Stage1 suites route through the generic scripted executor
        # (ours_runner sh + evals.verdicts.<verdict>); runner_kind survives
        # only as the RUNNER_METRICS contract key. Stage2 (op-*) suites still
        # dispatch through dispatcher.RUNNER_KINDS.
        import importlib

        from harness.run_schema import RUNNER_METRICS

        self.assertTrue(
            hasattr(suites, "RUNNER_KINDS"),
            "dispatcher.RUNNER_KINDS must exist",
        )
        config = tomllib.loads(_REGISTRY.read_text(encoding="utf-8"))
        for suite, cfg in config["evals"].items():
            with self.subTest(suite=suite):
                self.assertIn("runner_kind", cfg)
                if "ours_runner" in cfg:
                    sh = REPO_ROOT / "evals" / "scripts" / f"{cfg['ours_runner']}.sh"
                    self.assertTrue(sh.exists(), f"missing ours runner sh: {sh}")
                    self.assertIn("verdict", cfg)
                    importlib.import_module(f"evals.verdicts.{cfg['verdict']}")
                    self.assertIn(cfg["runner_kind"], RUNNER_METRICS)
                else:
                    self.assertIn(cfg["runner_kind"], suites.RUNNER_KINDS)

    def test_resume_gate_default_shape_is_short_smoke(self) -> None:
        if _gate_render.missing_render_inputs():
            self.skipTest(f"render inputs absent: {_gate_render.missing_render_inputs()}")
        rendered = _gate_render.render_variant("dense_training")
        try:
            prod = rendered.product("resume-gate-20", "ref")
            self.assertEqual(int(prod.get("num_steps")), 20)
            self.assertEqual(int(prod.get("resume_save_step")), 10)
            self.assertEqual([int(x) for x in prod.get("gate_window")], [10, 20])
        finally:
            rendered.cleanup()

    def test_resume_gate_shape_sourced_from_product_not_shell_preset(self) -> None:
        # Collapse: the per-gate ``resume-gate-20)`` shell case preset (which
        # baked ``NUM_STEPS_OVERRIDE:-20`` / ``RESUME_SAVE_STEP:-10`` / window
        # defaults into the launcher) is gone. Every launcher sources its full
        # shape from the rendered gate product — no shell-baked fallback
        # survives. Every launcher consumes the caller-projected generic env
        # (bare NUM_STEPS, no in-launcher projection at all).
        generic_scripts = (
            "run_qwen3_dense.sh",
            "run_16gpu_1000step_pure_mup_mtp.sh",
            "run_minicpm4_8b_dptp.sh",
            "train_minicpm4_0.5b_fineweb_modelbestsdk.sh",
            "train_minicpm4_0.5b_gsm8k.sh",
        )

        for script_name in generic_scripts:
            with self.subTest(script_name=script_name):
                text = (REPO_ROOT / "ref" / "reference" / script_name).read_text(encoding="utf-8")
                self.assertNotIn("resume-gate-20)", text)
                # No in-launcher projection and no baked step default: the
                # generic env arrives from the caller, num_steps required.
                self.assertNotIn("gate_product_to_shell.py", text)
                self.assertNotRegex(text, r"NUM_STEPS(_OVERRIDE)?:-\d+")
                self.assertRegex(text, r"NUM_STEPS:\?")

    def test_stage_groups_are_not_a_hardcoded_public_surface(self) -> None:
        self.assertFalse(
            hasattr(suites, "STAGES"),
            "stage grouping must be derived from config/eval.toml via stage_groups()",
        )

    def test_suite_names_are_not_hardcoded_public_surface(self) -> None:
        self.assertFalse(
            hasattr(suites, "SUITES"),
            "suite names must come from config/eval.toml, not dispatcher.SUITES",
        )

    def test_dispatcher_doc_points_to_workload_config_for_suite_list(self) -> None:
        text = (REPO_ROOT / "evals" / "dispatcher.py").read_text(encoding="utf-8")
        self.assertIn("config/eval.toml", text)
        self.assertNotIn("forward-align, backward-align", text)

    def test_unknown_suite_raises(self) -> None:
        request = {"suite": "no-such-suite", "workload_config": {}}
        with self.assertRaises((KeyError, ValueError)):
            suites.run_suite(request, REPO_ROOT, REPO_ROOT / ".artifacts")


class TestRegistryConsistency(unittest.TestCase):
    """Runtime dispatcher must derive script metadata from workload config."""

    def test_dispatcher_does_not_copy_workload_script_contract(self) -> None:
        self.assertFalse(
            hasattr(suites, "_WORKLOAD_SCRIPT_CONTRACT"),
            "script/launcher/env_inputs metadata must come from config/eval.toml",
        )


class TestArtifactIsolation(unittest.TestCase):
    def test_dispatcher_does_not_use_global_dp_tmp_artifacts(self) -> None:
        source = (REPO_ROOT / "evals" / "dispatcher.py").read_text(encoding="utf-8")
        self.assertNotIn("/tmp/mg_results_dp.json", source)
        self.assertNotIn('"29501"', source)


class TestRefGateSharedHelper(unittest.TestCase):
    def test_ref_gate_helper_is_extracted_from_dispatcher(self) -> None:
        import evals.gate_common as gate_common

        self.assertTrue(hasattr(gate_common, "resolve_ref_trajectory"))
        # The remaining in-process consumer is the stage2 op-* handler module
        # (the thin dispatcher's verdicts read trajectory files directly).
        source = (REPO_ROOT / "evals" / "dispatcher_stage2.py").read_text(encoding="utf-8")
        self.assertIn("resolve_ref_trajectory", source)


@unittest.skipIf(
    _gate_render.missing_render_inputs(),
    f"render inputs absent: {_gate_render.missing_render_inputs()}",
)
class TestTunedM4M5Shape(unittest.TestCase):
    """perf-bitwise / resume returned for 1B & 0.6B: a fast 25-step gate with warmup 10, MFU
    window [11,26), and a smaller per-variant batch (1B MBS=2/GBS=32, 0.6B
    MBS=3/GBS=48 -> grad_accum=8 on DP=2). 0.5B keeps its original 50/20-step
    shape. Shapes now live in gate_config → rendered products.
    """

    # variant -> (micro_batch_size, global_batch_size)
    _TUNED = {
        "dense_training_1b": (2, 32),
        "dense_training_qwen3": (3, 48),
    }

    @classmethod
    def setUpClass(cls) -> None:
        cls.rendered = {v: _gate_render.render_variant(v) for v in (*cls._TUNED, "dense_training")}

    @classmethod
    def tearDownClass(cls) -> None:
        for r in cls.rendered.values():
            if r is not None:
                r.cleanup()

    def _assert_grad_accum_is_eight(self, prod) -> None:
        mbs = int(prod.get("micro_batch_size"))
        gbs = int(prod.get("global_batch_size"))
        ws = int(prod.get("world_size"))
        self.assertEqual(gbs % (mbs * ws), 0, "GBS must divide evenly by MBS*WS")
        self.assertEqual(gbs // (mbs * ws), 8)

    def test_perf_bitwise_tuned_shape(self) -> None:
        for variant, (mbs, gbs) in self._TUNED.items():
            with self.subTest(variant=variant):
                prod = self.rendered[variant].product("perf-bitwise", "ref")
                self.assertEqual(int(prod.get("warmup_steps")), 10)
                self.assertEqual(int(prod.get("num_steps")), 25)
                self.assertEqual(int(prod.get("micro_batch_size")), mbs)
                self.assertEqual(int(prod.get("global_batch_size")), gbs)
                self.assertEqual([int(x) for x in prod.get("gate_window")], [11, 26])
                self._assert_grad_accum_is_eight(prod)

    def test_resume_gate_tuned_shape(self) -> None:
        for variant, (mbs, gbs) in self._TUNED.items():
            with self.subTest(variant=variant):
                prod = self.rendered[variant].product("resume-gate-20", "ref")
                self.assertEqual(int(prod.get("num_steps")), 25)
                self.assertEqual(int(prod.get("micro_batch_size")), mbs)
                self.assertEqual(int(prod.get("global_batch_size")), gbs)
                self.assertEqual(int(prod.get("resume_save_step")), 10)
                self.assertEqual([int(x) for x in prod.get("gate_window")], [10, 25])
                self._assert_grad_accum_is_eight(prod)

    def test_half_b_shape_is_left_unchanged(self) -> None:
        half = self.rendered["dense_training"]
        self.assertEqual(int(half.product("perf-bitwise", "ref").get("num_steps")), 50)
        self.assertEqual(int(half.product("resume-gate-20", "ref").get("num_steps")), 20)


if __name__ == "__main__":
    unittest.main()
