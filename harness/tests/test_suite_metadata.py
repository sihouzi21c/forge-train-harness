"""Suite metadata contract exposed by config/eval.toml and harness info."""

from __future__ import annotations

import unittest
from pathlib import Path

from evals import dispatcher
from harness import app, config_runtime, run_schema
from harness._compat import tomllib
from harness.tests import _gate_render

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestSuiteMetadata(unittest.TestCase):
    def _workload_config(self) -> dict:
        return tomllib.loads(
            (REPO_ROOT / "config" / "eval" / "dense_training" / "dense_training.toml").read_text(
                encoding="utf-8"
            )
        )

    def test_every_suite_declares_stage(self) -> None:
        config = self._workload_config()
        stages = {"stage1", "stage2"}
        for suite, cfg in config["evals"].items():
            with self.subTest(suite=suite):
                self.assertIn("stage", cfg)
                self.assertIn(cfg["stage"], stages)

    def test_info_groups_suites_by_stage(self) -> None:
        payload = app.info_command(json_output=True)
        workload = payload["payload"]["workload"]
        self.assertIn("suites_by_stage", workload)
        self.assertEqual(
            set(workload["suites_by_stage"]),
            {"local", "stage1", "stage2"},
        )
        flattened = sorted(
            suite for suites in workload["suites_by_stage"].values() for suite in suites
        )
        self.assertEqual(flattened, workload["supported_suites"])
        self.assertIn("guard", workload["supported_suites"])
        self.assertIn("unit", workload["supported_suites"])
        self.assertTrue(workload["suite_metadata"]["guard"]["local"])
        self.assertFalse(workload["suite_metadata"]["guard"]["requires_cuda"])
        self.assertTrue(workload["suite_metadata"]["unit"]["local"])
        self.assertFalse(workload["suite_metadata"]["op-inventory"]["requires_cuda"])
        self.assertFalse(workload["suite_metadata"]["op-status"]["requires_cuda"])

    def test_config_runtime_owns_suite_metadata_helpers(self) -> None:
        config = self._workload_config()
        metadata = config_runtime.suite_metadata(config)
        grouped = config_runtime.suites_by_stage(config, include_local=True)
        payload = app.info_command(json_output=True)["payload"]["workload"]

        self.assertEqual(metadata, payload["suite_metadata"])
        self.assertEqual(grouped, payload["suites_by_stage"])
        self.assertFalse(hasattr(app, "_all_suite_metadata"))
        self.assertFalse(hasattr(app, "_suites_by_stage"))

    def test_workload_config_carries_runnable_script_contract(self) -> None:
        config = self._workload_config()
        self.assertFalse(hasattr(dispatcher, "_WORKLOAD_SCRIPT_CONTRACT"))
        for suite, suite_cfg in config["evals"].items():
            if "script" not in suite_cfg:
                continue
            with self.subTest(suite=suite):
                self.assertTrue((REPO_ROOT / suite_cfg["script"]).exists())
                if "launcher" in suite_cfg:
                    self.assertTrue((REPO_ROOT / suite_cfg["launcher"]).exists())
                self.assertIsInstance(suite_cfg.get("env_inputs", []), list)

    def test_workload_config_declares_valid_routing(self) -> None:
        """Every suite is runnable: scripted (ours_runner/verdict) or RUNNER_KINDS.

        Stage1 suites route through the generic scripted executor — their
        ``ours_runner`` sh must exist and ``verdict`` must name an importable
        ``evals.verdicts.<name>`` module. Their ``runner_kind`` survives only
        as the RUNNER_METRICS contract key. Stage2 (op-*) suites still
        dispatch through ``dispatcher.RUNNER_KINDS``.
        """
        import importlib

        from harness.run_schema import RUNNER_METRICS

        config = self._workload_config()
        for suite, suite_cfg in config["evals"].items():
            with self.subTest(suite=suite):
                self.assertIn("runner_kind", suite_cfg)
                if "ours_runner" in suite_cfg:
                    sh = REPO_ROOT / "evals" / "scripts" / f"{suite_cfg['ours_runner']}.sh"
                    self.assertTrue(sh.exists(), f"missing ours runner sh: {sh}")
                    self.assertIn("verdict", suite_cfg)
                    importlib.import_module(f"evals.verdicts.{suite_cfg['verdict']}")
                    self.assertIn(suite_cfg["runner_kind"], RUNNER_METRICS)
                else:
                    self.assertIn(suite_cfg["runner_kind"], dispatcher.RUNNER_KINDS)

    def test_info_exposes_run_contract_metadata(self) -> None:
        payload = app.info_command(json_output=True)
        contract = payload["payload"]["contracts"]["run"]
        self.assertEqual(contract, run_schema.schema_metadata())
        self.assertEqual(contract["schema_version"], 1)
        self.assertIn("request.json", contract["artifact_files"])
        self.assertIn("result.json", contract["artifact_files"])
        self.assertIn("schema_version", contract["request_required_keys"])
        self.assertIn("schema_version", contract["result_required_keys"])
        self.assertEqual(contract["result_markers"]["begin"], run_schema.RESULT_BEGIN)

    def test_info_exposes_suite_specific_contract_metadata(self) -> None:
        payload = app.info_command(json_output=True)
        contracts = payload["payload"]["contracts"]["suites"]

        self.assertEqual(contracts["guard"]["artifacts"], [])
        self.assertEqual(contracts["unit"]["artifacts"], [])
        self.assertIn("checks_total", contracts["forward-align"]["metrics"])
        self.assertIn("gate_pass", contracts["multistep"]["metrics"])
        self.assertIn("gate_pass", contracts["multistep-1gpu"]["metrics"])
        self.assertIn("avg_mfu_e2e_standard", contracts["perf-bitwise"]["metrics"])
        self.assertIn("pointwise_mean_rel", contracts["long-train"]["metrics"])
        # env_inputs is stage2-only now (suite_process_env lifting on op-*);
        # stage1 suites deliver values via rendered products + runtime_env.py.
        self.assertEqual(contracts["long-train"]["env_inputs"], [])
        self.assertIn("GLOBAL_BATCH_SIZE", contracts["op-long"]["env_inputs"])
        self.assertIn("MICRO_BATCH_SIZE", contracts["op-long"]["env_inputs"])
        self.assertIn("SEQ_LENGTH", contracts["op-long"]["env_inputs"])
        self.assertIn("max_abs_diff_loss", contracts["resume-gate-20"]["metrics"])
        self.assertIn("avg_relative_loss_diff", contracts["loss-gate-200"]["metrics"])
        self.assertNotIn(
            "op-unit",
            contracts,
            "op-unit was removed as a harness gate (subagent runs "
            "test_op.py directly in its worktree; see stage2.md and "
            "stage2-subagent-playbook.md)",
        )
        self.assertIn("OP_NAMES", contracts["op-long"]["env_inputs"])
        self.assertIn("mean_rel_diff", contracts["op-long"]["metrics"])
        self.assertIn("operators", contracts["op-status"]["metrics"])

    def test_stage2_positional_arg_contracts_are_metadata(self) -> None:
        payload = app.info_command(json_output=True)
        metadata = payload["payload"]["workload"]["suite_metadata"]
        self.assertNotIn("op-unit", metadata)
        self.assertEqual(metadata["op-long"]["args_usage"], "[names...]")
        self.assertEqual(metadata["op-long"]["args_min"], 0)
        self.assertTrue(metadata["op-long"]["args_unbounded"])

    def test_non_cuda_dispatcher_suites_keep_suite_runner_scope(self) -> None:
        metadata = app.info_command(json_output=True)["payload"]["workload"]["suite_metadata"]
        for suite in ("op-inventory", "op-status"):
            with self.subTest(suite=suite):
                self.assertFalse(metadata[suite]["requires_cuda"])
                self.assertFalse(metadata[suite]["local"])
                self.assertEqual(metadata[suite]["execution_scope"], "suite-runner")

    def test_dispatcher_stage_groups_are_derived_from_workload_config(self) -> None:
        config = self._workload_config()
        expected: dict[str, list[str]] = {}
        for suite, cfg in config["evals"].items():
            expected.setdefault(cfg["stage"], []).append(suite)
        expected = {stage: sorted(suites) for stage, suites in expected.items()}
        actual = {
            stage: sorted(suites) for stage, suites in dispatcher.stage_groups(config).items()
        }
        self.assertEqual(actual, expected)

    def test_runnable_suites_declare_milestone_in_workload_config(self) -> None:
        config = self._workload_config()
        for suite, suite_cfg in config["evals"].items():
            if "script" not in suite_cfg:
                continue
            # profile-snapshot owns no single milestone — it mirrors the gate
            # named by the label's milestone prefix (mirror_gate), so the
            # milestone is resolved per-run, not declared statically.
            if "mirror_gate" in suite_cfg:
                self.assertNotIn("milestone", suite_cfg)
                continue
            with self.subTest(suite=suite):
                self.assertIn("milestone", suite_cfg)

    def test_long_train_smoke_is_20_step_variant_of_long_train(self) -> None:
        """``long-train-smoke`` is the long-train iteration gate — short MFU run.

        Locks in the 20-step intent: same MBS/GBS/DP shape as
        ``long-train``, same MFU/loss thresholds, but only 20 steps with
        the [5, 20) measurement window. The 5-step warmup MUST be
        declared explicitly because dispatcher._run_long_train defaults
        ``warmup_steps`` to 50 — a default that would drop every sample
        from a 20-step run and leave ``mfu_values`` empty.
        """
        if _gate_render.missing_render_inputs():
            self.skipTest(f"render inputs absent: {_gate_render.missing_render_inputs()}")
        config = self._workload_config()
        full = config["evals"]["long-train"]
        smoke = config["evals"]["long-train-smoke"]

        # Identity / timeout live in the registry.
        self.assertEqual(smoke["runner_kind"], full["runner_kind"])
        self.assertEqual(smoke["script"], full["script"])
        self.assertEqual(smoke["launcher"], full["launcher"])
        self.assertLess(smoke["timeout_s"], full["timeout_s"])

        # Gate shape / thresholds live in gate_config → rendered products.
        rendered = _gate_render.render_variant("dense_training")
        try:
            full_p = rendered.product("long-train", "ref")
            smoke_p = rendered.product("long-train-smoke", "ref")
            self.assertEqual(int(smoke_p.get("num_steps")), 20)
            self.assertEqual([int(x) for x in smoke_p.get("gate_window")], [5, 20])
            self.assertEqual(int(smoke_p.get("warmup_steps")), 5)
            self.assertEqual(smoke_p.get("mfu_e2e_target"), full_p.get("mfu_e2e_target"))
            self.assertEqual(smoke_p.get("deterministic"), full_p.get("deterministic"))
            self.assertEqual(int(full_p.get("micro_batch_size")), 4)
            self.assertEqual(
                int(smoke_p.get("micro_batch_size")), int(full_p.get("micro_batch_size"))
            )
            self.assertEqual(
                int(smoke_p.get("global_batch_size")), int(full_p.get("global_batch_size"))
            )
            full_o = rendered.product("long-train", "ours")
            smoke_o = rendered.product("long-train-smoke", "ours")
            self.assertEqual(int(full_o.get("micro_batch_size")), 10)
            self.assertEqual(
                int(smoke_o.get("micro_batch_size")), int(full_o.get("micro_batch_size"))
            )
        finally:
            rendered.cleanup()

    def test_metrics_contract_covers_all_runner_kinds(self) -> None:
        """RUNNER_METRICS must stay in sync with the registry + local runners.

        Stage1 handlers left ``dispatcher.RUNNER_KINDS`` (scripted executor);
        the registry's ``runner_kind`` is now the contract key that maps each
        suite to its RUNNER_METRICS lower bound (app._suite_contracts).
        """
        from harness.run_schema import RUNNER_METRICS

        config = self._workload_config()
        registry_kinds = {cfg["runner_kind"] for cfg in config["evals"].values()}
        all_runner_kinds = (
            registry_kinds | set(dispatcher.RUNNER_KINDS) | set(app._LOCAL_SUITE_RUNNERS)
        )
        contract_kinds = set(RUNNER_METRICS)
        missing = sorted(all_runner_kinds - contract_kinds)
        extra = sorted(contract_kinds - all_runner_kinds)
        self.assertFalse(
            missing,
            f"RUNNER_METRICS is missing entries for: {missing}",
        )
        self.assertFalse(
            extra,
            f"RUNNER_METRICS has stale entries: {extra}",
        )


if __name__ == "__main__":
    unittest.main()
