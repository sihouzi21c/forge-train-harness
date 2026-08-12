"""Harness-core unit tests: config_runtime and local transport."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["FORGE_REPO_ROOT"] = str(REPO_ROOT)

from harness import app, cli, config_runtime, run_schema, transport  # noqa: E402


class TestRuntimeDefaults(unittest.TestCase):
    def test_defaults_from_committed_toml(self) -> None:
        harness_config = config_runtime.load_harness_config()
        defaults = config_runtime.runtime_defaults(harness_config)
        self.assertEqual(defaults.report, "text")

    def test_runtime_defaults_can_load_default_config(self) -> None:
        defaults = config_runtime.runtime_defaults()
        self.assertEqual(defaults.report, "text")

    def test_runs_artifact_subtree(self) -> None:
        harness_config = config_runtime.load_harness_config()
        subtree = config_runtime.artifact_subtree(harness_config, "runs")
        self.assertEqual(subtree, (REPO_ROOT / ".artifacts" / "runs").resolve())

    def test_harness_config_path(self) -> None:
        self.assertEqual(
            config_runtime.harness_config_path(),
            REPO_ROOT / "harness" / "config" / "defaults.toml",
        )

    def test_report_defaults_to_text(self) -> None:
        harness_config = config_runtime.load_harness_config()
        self.assertEqual(config_runtime.runtime_defaults(harness_config).report, "text")

    def test_default_report_helper_removed(self) -> None:
        self.assertFalse(hasattr(config_runtime, "default_report"))

    def test_pythonpath_helpers_are_public_runtime_contract(self) -> None:
        self.assertTrue(hasattr(config_runtime, "workload_src_path"))
        self.assertTrue(hasattr(config_runtime, "prepend_pythonpath"))
        self.assertTrue(hasattr(config_runtime, "remove_pythonpath_entry"))
        env = config_runtime.prepend_pythonpath({}, config_runtime.workload_src_path(REPO_ROOT))
        self.assertEqual(env["PYTHONPATH"], str(REPO_ROOT / "workload" / "src"))

    def test_remove_pythonpath_entry(self) -> None:
        workload_src = str(REPO_ROOT / "workload" / "src")
        env = {"PYTHONPATH": f"/tmp:{workload_src}:/opt"}
        cleaned = config_runtime.remove_pythonpath_entry(env, Path(workload_src))
        self.assertEqual(cleaned["PYTHONPATH"], "/tmp:/opt")
        self.assertEqual(env["PYTHONPATH"], f"/tmp:{workload_src}:/opt")

    def test_build_subprocess_env_centralizes_gpu_and_pythonpath(self) -> None:
        env = config_runtime.build_subprocess_env(
            repo_root=REPO_ROOT,
            source_env={"PYTHONPATH": "/tmp"},
            extra={"EXTRA_FLAG": "yes"},
            cuda_visible_devices="0",
            prepend_workload_src=True,
        )

        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(env["EXTRA_FLAG"], "yes")
        # repo_root is prepended on top of workload_src so evals/scripts/*.py
        # subprocesses can ``from evals._common import …`` without per-script
        # sys.path surgery; the original PYTHONPATH suffix is preserved.
        self.assertEqual(
            env["PYTHONPATH"],
            f"{REPO_ROOT}:{REPO_ROOT / 'workload' / 'src'}:/tmp",
        )

    def test_build_subprocess_env_opt_out_repo_root(self) -> None:
        env = config_runtime.build_subprocess_env(
            repo_root=REPO_ROOT,
            source_env={"PYTHONPATH": "/tmp"},
            prepend_workload_src=False,
            prepend_repo_root=False,
        )
        self.assertEqual(env["PYTHONPATH"], "/tmp")


class TestRepoRoot(unittest.TestCase):
    def test_repo_root_matches_env(self) -> None:
        self.assertEqual(config_runtime.repo_root(), REPO_ROOT.resolve())


class TestWorkloadConfig(unittest.TestCase):
    def test_load_default_workload_config(self) -> None:
        path, data = config_runtime.load_workload_config(None, include_user_config=False)
        self.assertEqual(
            path,
            (REPO_ROOT / "config" / "eval" / "dense_training" / "dense_training.toml").resolve(),
        )
        self.assertIn("workload", data)
        self.assertIn("evals", data)

    def test_load_workload_config_can_ignore_gitignored_user_profiles(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = "/user/" + "zhuzhui/local-only-data"
            ref_toml = Path(tmpdir) / "ref.toml"
            ref_toml.write_text(
                f'[ref]\ndata_path = "{local_path}"\nref_script = "train_minicpm4_0.5b_gsm8k.sh"\n',
                encoding="utf-8",
            )

            with mock.patch.object(config_runtime, "_ref_config_path", return_value=ref_toml):
                _, isolated = config_runtime.load_workload_config(
                    None,
                    include_user_config=False,
                )
                _, merged = config_runtime.load_workload_config(
                    None,
                    include_user_config=True,
                )

        self.assertNotIn("data_path", isolated.get("ref", {}))
        self.assertEqual(merged["ref"]["data_path"], local_path)

    def test_ref_profiles_declare_ref_script_as_basename(self) -> None:
        from harness._compat import tomllib

        ref_dir = REPO_ROOT / "config" / "ref"
        for path in sorted(ref_dir.glob("*.toml")):
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            ref_script = data.get("ref", {}).get("ref_script")
            self.assertIsInstance(ref_script, str, path)
            self.assertNotIn("/", ref_script, path)
            self.assertNotIn("\\", ref_script, path)
            self.assertTrue(
                (REPO_ROOT / "ref" / "reference" / ref_script).is_file(),
                f"{path}: ref_script {ref_script!r} does not exist under ref/reference/",
            )

    def test_resolve_suite_config_merges_ref_stage_and_suite(self) -> None:
        cfg = config_runtime.resolve_suite_config(
            {
                "ref": {"seed": 1234, "checkpoint_root": "/default"},
                "stage2": {
                    "checkpoint_root": "/stage2",
                    "data_path": "stage2-data",
                    "micro_batch_size": 10,
                },
                "evals": {
                    "op-long": {
                        "stage": "stage2",
                        "runner_kind": "op-long",
                        "checkpoint_root": "/suite",
                    },
                },
            },
            "op-long",
        )

        self.assertEqual(cfg["seed"], 1234)
        self.assertEqual(cfg["checkpoint_root"], "/suite")
        self.assertEqual(cfg["data_path"], "stage2-data")
        self.assertEqual(cfg["micro_batch_size"], 10)

    def test_resolve_suite_config_rejects_unknown_suite(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown suite"):
            config_runtime.resolve_suite_config({"evals": {}}, "missing")

    _RUNTIME: ClassVar[dict[str, dict[str, str]]] = {
        "distributed": {"master_addr": "localhost", "master_port": "29500"}
    }

    def test_validate_workload_config_rejects_missing_stage(self) -> None:
        with self.assertRaisesRegex(ValueError, "evals.bad.stage"):
            config_runtime.validate_workload_config(
                REPO_ROOT / "config" / "eval" / "dense_training" / "dense_training.toml",
                {
                    "workload": {
                        "id": "x",
                        "display_name": "x",
                        "requires_cuda": False,
                    },
                    "runtime": self._RUNTIME,
                    "evals": {"bad": {}},
                },
            )

    def test_validate_workload_config_script_without_env_inputs_is_valid(self) -> None:
        # env_inputs is a stage2-only key; the legacy "script implies
        # env_inputs" coupling is retired. A script suite without env_inputs
        # validates cleanly, while a non-list env_inputs still fails fast.
        base = {
            "workload": {"id": "x", "display_name": "x", "requires_cuda": False},
            "runtime": self._RUNTIME,
            "evals": {
                "bad": {
                    "stage": "stage1",
                    "runner_kind": "forward-align",
                    "script": "evals/scripts/eval_train_steps.py",
                },
            },
        }
        path = REPO_ROOT / "config" / "eval" / "dense_training" / "dense_training.toml"
        config_runtime.validate_workload_config(path, base)  # must not raise

        bad = {**base, "evals": {"bad": {**base["evals"]["bad"], "env_inputs": "DATA_PATH"}}}
        with self.assertRaisesRegex(ValueError, "evals.bad.env_inputs"):
            config_runtime.validate_workload_config(path, bad)

    def test_validate_workload_config_requires_runner_kind_for_all_suites(self) -> None:
        with self.assertRaisesRegex(ValueError, "evals.bad.runner_kind"):
            config_runtime.validate_workload_config(
                REPO_ROOT / "config" / "eval" / "dense_training" / "dense_training.toml",
                {
                    "workload": {"id": "x", "display_name": "x", "requires_cuda": False},
                    "runtime": self._RUNTIME,
                    "evals": {"bad": {"stage": "stage2"}},
                },
            )

    def test_validate_workload_config_rejects_lowercase_env_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "env_inputs"):
            config_runtime.validate_workload_config(
                REPO_ROOT / "config" / "eval" / "dense_training" / "dense_training.toml",
                {
                    "workload": {"id": "x", "display_name": "x", "requires_cuda": False},
                    "runtime": self._RUNTIME,
                    "evals": {
                        "bad": {
                            "stage": "stage1",
                            "runner_kind": "forward-align",
                            "script": "evals/scripts/eval_train_steps.py",
                            "env_inputs": ["checkpoint_root"],
                        },
                    },
                },
            )

    def test_validate_workload_config_requires_op_long_env_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "op-long.env_inputs"):
            config_runtime.validate_workload_config(
                REPO_ROOT / "config" / "eval" / "dense_training" / "dense_training.toml",
                {
                    "workload": {"id": "x", "display_name": "x", "requires_cuda": False},
                    "runtime": self._RUNTIME,
                    "evals": {
                        "op-long": {
                            "stage": "stage2",
                            "runner_kind": "op-long",
                            "env_inputs": ["OP_NAMES"],
                        },
                    },
                },
            )

    def test_validate_workload_config_rejects_global_env_duplicate_sources(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicates process env source"):
            config_runtime.validate_workload_config(
                REPO_ROOT / "config" / "eval" / "dense_training" / "dense_training.toml",
                {
                    "workload": {"id": "x", "display_name": "x", "requires_cuda": False},
                    "env": {"FOO": "global"},
                    "ref": {"foo": "default"},
                    "evals": {"unit": {"stage": "stage1", "runner_kind": "noop"}},
                },
            )

    def test_validate_workload_config_accepts_ref_env_string_overrides(self) -> None:
        config_runtime.validate_workload_config(
            REPO_ROOT / "config" / "eval" / "dense_training" / "dense_training.toml",
            {
                "workload": {"id": "x", "display_name": "x", "requires_cuda": False},
                "runtime": self._RUNTIME,
                "evals": {
                    "long-train": {
                        "stage": "stage1",
                        "runner_kind": "long-train",
                        "ref_env": {
                            "NUM_STEPS_OVERRIDE": "10",
                            "GATE_WINDOW_START": "5",
                            "GATE_WINDOW_END": "10",
                        },
                    },
                },
            },
        )

    def test_validate_workload_config_rejects_lowercase_ref_env_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, r"evals\.long-train\.ref_env keys"):
            config_runtime.validate_workload_config(
                REPO_ROOT / "config" / "eval" / "dense_training" / "dense_training.toml",
                {
                    "workload": {"id": "x", "display_name": "x", "requires_cuda": False},
                    "runtime": self._RUNTIME,
                    "evals": {
                        "long-train": {
                            "stage": "stage1",
                            "runner_kind": "long-train",
                            "ref_env": {"num_steps_override": "10"},
                        },
                    },
                },
            )

    def test_validate_workload_config_rejects_bool_ref_env_value(self) -> None:
        # bool subclasses int; reject explicitly so callers spell "1"/"0".
        with self.assertRaisesRegex(ValueError, r"ref_env\['ENABLE_FOO'\]"):
            config_runtime.validate_workload_config(
                REPO_ROOT / "config" / "eval" / "dense_training" / "dense_training.toml",
                {
                    "workload": {"id": "x", "display_name": "x", "requires_cuda": False},
                    "runtime": self._RUNTIME,
                    "evals": {
                        "long-train": {
                            "stage": "stage1",
                            "runner_kind": "long-train",
                            "ref_env": {"ENABLE_FOO": True},
                        },
                    },
                },
            )

    def test_validate_workload_config_rejects_non_table_ref_env(self) -> None:
        with self.assertRaisesRegex(ValueError, r"ref_env must be a table"):
            config_runtime.validate_workload_config(
                REPO_ROOT / "config" / "eval" / "dense_training" / "dense_training.toml",
                {
                    "workload": {"id": "x", "display_name": "x", "requires_cuda": False},
                    "runtime": self._RUNTIME,
                    "evals": {
                        "long-train": {
                            "stage": "stage1",
                            "runner_kind": "long-train",
                            "ref_env": ["NUM_STEPS_OVERRIDE=10"],
                        },
                    },
                },
            )

    def test_validate_workload_config_rejects_bad_requires_cuda_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires_cuda"):
            config_runtime.validate_workload_config(
                REPO_ROOT / "config" / "eval" / "dense_training" / "dense_training.toml",
                {
                    "workload": {"id": "x", "display_name": "x", "requires_cuda": False},
                    "runtime": self._RUNTIME,
                    "evals": {
                        "bad": {
                            "stage": "stage2",
                            "runner_kind": "op-status",
                            "requires_cuda": "no",
                        },
                    },
                },
            )

    def test_validate_workload_config_requires_stage2_automation_keys(self) -> None:
        # ``max_concurrent`` is a required positive int on ``automation.stage2``.
        # ``default_model`` moved to ``[agent].model``.
        with self.assertRaisesRegex(ValueError, "automation.stage2.max_concurrent"):
            config_runtime.validate_workload_config(
                REPO_ROOT / "config" / "eval" / "dense_training" / "dense_training.toml",
                {
                    "workload": {"id": "x", "display_name": "x", "requires_cuda": False},
                    "runtime": self._RUNTIME,
                    "evals": {},
                    "automation": {
                        "stage2": {"max_op_long_failures": 3},
                    },
                },
            )

    def test_validate_workload_config_rejects_bad_automation_count_type(self) -> None:
        # ``max_concurrent`` must be a positive int; a string violates the
        # automation.stage2 contract. ``max_op_long_failures`` (the per-op
        # safety net cap for the goal-driven subagent loop) is the only
        # other numeric field on ``automation.stage2``.
        with self.assertRaisesRegex(ValueError, "automation.stage2.max_concurrent"):
            config_runtime.validate_workload_config(
                REPO_ROOT / "config" / "eval" / "dense_training" / "dense_training.toml",
                {
                    "workload": {"id": "x", "display_name": "x", "requires_cuda": False},
                    "runtime": self._RUNTIME,
                    "evals": {},
                    "automation": {
                        "stage2": {
                            "max_concurrent": "3",
                            "max_op_long_failures": 3,
                        },
                    },
                },
            )

    def test_validate_workload_config_rejects_bad_agent_config(self) -> None:
        # ``[agent]`` must be a table if present; non-dict triggers an error.
        with self.assertRaisesRegex(ValueError, r"\[agent\] must be a table"):
            config_runtime.validate_workload_config(
                REPO_ROOT / "config" / "eval" / "dense_training" / "dense_training.toml",
                {
                    "workload": {"id": "x", "display_name": "x", "requires_cuda": False},
                    "runtime": self._RUNTIME,
                    "evals": {},
                    "agent": "not-a-table",
                },
            )


class TestCliSuiteArgs(unittest.TestCase):
    """Verify the `harness run <suite> <args...>` wiring is intact."""

    def test_run_help_points_to_info_instead_of_stale_suite_list(self) -> None:
        parser = cli._build_parser()
        run_action = next(action for action in parser._actions if action.dest == "command").choices[
            "run"
        ]
        suite_action = next(action for action in run_action._actions if action.dest == "suite")
        self.assertIn("harness info", suite_action.help)
        self.assertNotIn("resume-gate-20, loss-gate-200, guard, unit", suite_action.help)

    def test_parser_accepts_positional_suite_args(self) -> None:
        parser = cli._build_parser()
        parsed = parser.parse_args(["run", "op-long", "attention", "gemm_fc1"])
        self.assertEqual(parsed.suite, "op-long")
        self.assertEqual(parsed.suite_args, ["attention", "gemm_fc1"])

    def test_build_run_request_forwards_suite_args(self) -> None:
        harness_config = config_runtime.load_harness_config()
        _, workload_config = config_runtime.load_workload_config(
            None,
            include_user_config=False,
        )
        request = app._build_run_request(
            suite="op-long",
            suite_args=["attention"],
            report="text",
            harness_config=harness_config,
            workload_config=workload_config,
        )
        self.assertEqual(request["args"], ["attention"])
        self.assertEqual(request["suite"], "op-long")
        run_schema.validate_run_request(request)

    def test_app_validates_suite_arg_arity_from_metadata(self) -> None:
        # Self-contained: uses virtual suite names so the test does not
        # tie itself to whichever real suite currently happens to require
        # exactly one positional arg.
        suites = {
            "single-arg-suite": {"args_min": 1, "args_max": 1},
            "multi-arg-suite": {"args_min": 0, "args_unbounded": True},
        }
        app._check_suite_arg_arity("single-arg-suite", ["attention"], suites)
        app._check_suite_arg_arity("multi-arg-suite", [], suites)
        app._check_suite_arg_arity("multi-arg-suite", ["attention", "mlp"], suites)
        with self.assertRaisesRegex(ValueError, "expects at least 1"):
            app._check_suite_arg_arity("single-arg-suite", [], suites)
        with self.assertRaisesRegex(ValueError, "expects at most 1"):
            app._check_suite_arg_arity("single-arg-suite", ["attention", "mlp"], suites)

    def test_runtime_arity_helper_does_not_clash_with_schema_validator(self) -> None:
        # The runtime arity helper and the schema-side helper validate
        # different things (CLI arg count vs TOML field types). They
        # must have distinct names so a look-alike-validators trap
        # cannot resurface; pin the public symbols here.
        self.assertFalse(hasattr(app, "_validate_suite_args"))
        self.assertTrue(callable(getattr(app, "_check_suite_arg_arity", None)))
        self.assertTrue(callable(getattr(config_runtime, "_validate_suite_args_metadata", None)))


class TestLocalSuitesIsPrivate(unittest.TestCase):
    """`LOCAL_SUITES` is internal to harness.app — must not leak as public."""

    def test_no_public_local_suites_symbol(self) -> None:
        self.assertFalse(hasattr(app, "LOCAL_SUITES"))

    def test_local_suites_declare_runner_kind(self) -> None:
        _, config = config_runtime.load_workload_config(None, include_user_config=False)
        self.assertEqual(config["local_suites"]["guard"]["runner_kind"], "guard")
        self.assertEqual(config["local_suites"]["unit"]["runner_kind"], "unit")

    def test_local_runner_registry_is_explicit(self) -> None:
        self.assertEqual(set(app._LOCAL_SUITE_RUNNERS), {"guard", "unit", "anti-proxy"})
        source = (REPO_ROOT / "harness" / "app.py").read_text(encoding="utf-8")
        self.assertNotIn('if suite == "guard"', source)
        self.assertNotIn('if suite == "unit"', source)
        self.assertNotIn('if suite == "anti-proxy"', source)


class TestLocalSuiteResults(unittest.TestCase):
    def test_local_guard_result_uses_run_result_schema(self) -> None:
        result = app._run_local_suite("guard", REPO_ROOT)
        run_schema.validate_run_result(result)

    def test_result_builder_adds_schema_version(self) -> None:
        result = run_schema.make_run_result(
            status="passed",
            suite="unit",
            summary="ok",
        )
        self.assertEqual(result["schema_version"], 1)
        run_schema.validate_run_result(result)

    def test_unit_suite_does_not_prepend_workload_src_by_default(self) -> None:
        captured_env: dict[str, str] = {}

        def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
            captured_env.update(kwargs.get("env", {}))
            return SimpleNamespace(returncode=0, stderr="Ran 1 test in 0.001s\n", stdout="")

        with mock.patch("harness.app.subprocess.run", side_effect=fake_run):
            result = app._run_local_suite("unit", REPO_ROOT)

        run_schema.validate_run_result(result)
        self.assertNotIn(str(REPO_ROOT / "workload" / "src"), captured_env.get("PYTHONPATH", ""))


class TestLocalTransport(unittest.TestCase):
    """LocalTransport replaces the prior SSH-based transports."""

    def test_create_transport_returns_transport_protocol(self) -> None:
        harness_config = config_runtime.load_harness_config()
        t = transport.create_transport(harness_config)
        self.assertTrue(hasattr(t, "doctor"))
        self.assertTrue(hasattr(t, "run"))

    def test_check_local_python_passes_on_supported_interpreter(self) -> None:
        # ``_check_local_python`` enforces the pyproject
        # ``requires-python`` floor; pin both the ready path here and
        # the failure path in the sister test so the helper cannot
        # silently degrade back to an unconditional ``ready`` stub.
        check = transport._check_local_python()
        self.assertEqual(check["name"], "python")
        self.assertEqual(check["status"], "ready")
        self.assertIn("Python", check["detail"])

    def test_check_local_python_fails_on_too_old_interpreter(self) -> None:
        old_version = (3, 10, 0, "final", 0)
        with mock.patch.object(transport.sys, "version_info", old_version):
            check = transport._check_local_python()
        self.assertEqual(check["status"], "failed")
        self.assertIn(">= 3.11", check["detail"])

    def test_bridge_invocation_is_argv_and_env_not_shell(self) -> None:
        command, env = transport._build_bridge_invocation(
            repo_root=REPO_ROOT,
            action_args=["run", "request.json"],
            gpu="0,1",
            env={"EXTRA_FLAG": "yes"},
        )

        self.assertNotIn("bash", command)
        self.assertNotIn("-c", command)
        self.assertIn("-m", command)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(env["EXTRA_FLAG"], "yes")
        self.assertNotIn(str(REPO_ROOT / "workload" / "src"), env.get("PYTHONPATH", ""))

    def test_gpu_value_rejects_shell_metacharacters(self) -> None:
        with self.assertRaises(ValueError):
            transport._build_bridge_invocation(
                repo_root=REPO_ROOT,
                action_args=["doctor"],
                gpu="0;echo pwned",
                env=None,
            )


class TestDeploymentPathError(unittest.TestCase):
    """DeploymentPathError semantics and structured result generation."""

    def test_error_carries_missing_keys_sorted(self) -> None:
        exc = config_runtime.DeploymentPathError(["data_path", "checkpoint_root"])
        self.assertEqual(exc.missing, ["checkpoint_root", "data_path"])
        self.assertIn("checkpoint_root", str(exc))
        self.assertIn("config/ref", str(exc))

    def test_classification_in_result(self) -> None:
        exc = config_runtime.DeploymentPathError(["megatron_root"])
        result = {
            "status": "failed",
            "suite": "forward-align",
            "summary": str(exc),
            "metrics": {},
            "details": {
                "classification": "deployment_paths_missing",
                "missing_paths": exc.missing,
            },
        }
        self.assertEqual(result["details"]["classification"], "deployment_paths_missing")
        self.assertEqual(result["details"]["missing_paths"], ["megatron_root"])


class TestRefSourceValidation(unittest.TestCase):
    """Validate [ref] source fields and [data] table schemas."""

    def test_validate_ref_branch_without_megatron(self) -> None:
        with self.assertRaises(ValueError):
            config_runtime._validate_ref_source_fields(
                Path("test.toml"), {"megatron_branch": "main"}
            )

    def test_validate_ref_source_fields_ok(self) -> None:
        config_runtime._validate_ref_source_fields(
            Path("test.toml"),
            {"megatron": "git@x.git", "megatron_branch": "main", "tokenizer": "tok"},
        )

    def test_validate_data_unknown_keys_rejected(self) -> None:
        with self.assertRaises(ValueError):
            config_runtime._validate_data_config(Path("test.toml"), {"prep_script": "foo.sh"})

    def test_validate_data_conf_path_ok(self) -> None:
        config_runtime._validate_data_config(
            Path("test.toml"),
            {"conf_path": "ref/reference/minicpm4_0.5.stable.sh"},
        )

    def test_validate_data_conf_name_ok(self) -> None:
        config_runtime._validate_data_config(
            Path("test.toml"),
            {"conf_name": "minicpm4_0.5.stable.sh"},
        )

    def test_validate_data_mutual_exclusion(self) -> None:
        with self.assertRaises(ValueError):
            config_runtime._validate_data_config(
                Path("test.toml"),
                {"conf_path": "/a", "conf_name": "b"},
            )


class TestResolveSourcesMegatron(unittest.TestCase):
    """_resolve_megatron: local path vs vendored submodule fallback."""

    def test_local_path_used_as_is(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            result = config_runtime._resolve_megatron(REPO_ROOT, {"megatron": tmpdir})
            self.assertEqual(result, str(Path(tmpdir).resolve()))

    def test_empty_falls_back_to_submodule(self) -> None:
        submodule = REPO_ROOT / "harness" / "third_party" / "megatron" / "v15"
        result = config_runtime._resolve_megatron(REPO_ROOT, {})
        if submodule.is_dir() and any(submodule.iterdir()):
            self.assertEqual(result, str(submodule))
        else:
            self.assertEqual(result, "")


class TestResolveSourcesTokenizer(unittest.TestCase):
    """_resolve_tokenizer: local file detection."""

    def test_local_file_used_as_is(self) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".model") as f:
            result = config_runtime._resolve_tokenizer(REPO_ROOT, {"tokenizer": f.name})
            self.assertEqual(result, str(Path(f.name).resolve()))

    def test_empty_returns_empty(self) -> None:
        result = config_runtime._resolve_tokenizer(REPO_ROOT, {})
        self.assertEqual(result, "")


class TestResolveRef(unittest.TestCase):
    """_resolve_ref: data_conf resolution and source field resolution."""

    def test_data_conf_resolves_data_path(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            conf = Path(tmpdir) / "data_conf.sh"
            # The conf may still export DATA_LOADER, but it is deliberately
            # ignored: the loader kind is read from data.toml [data].data_loader,
            # never from a conf export. Only DATA_PATH is lifted.
            conf.write_text(
                'DATA_PATH="0.5 /data/a\n0.5 /data/b"\nDATA_LOADER="hf"\n',
                encoding="utf-8",
            )

            result = config_runtime._resolve_data_env_from_conf(str(conf))

        self.assertEqual(result, {"data_path": "0.5 /data/a\n0.5 /data/b"})

    def test_resolve_ref_sets_data_path_from_conf_when_unset(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            conf = Path(tmpdir) / "data_conf.sh"
            conf.write_text('DATA_PATH="/data/from-conf"\n', encoding="utf-8")
            data = {
                "ref": {
                    "ref_script": "train_minicpm4_0.5b_gsm8k.sh",
                    "checkpoint_root": "/ckpt",
                },
                "data": {"conf_path": str(conf)},
            }

            config_runtime._resolve_ref(Path("test.toml"), data)

        self.assertEqual(data["ref"]["data_conf"], str(conf))
        self.assertEqual(data["ref"]["data_path"], "/data/from-conf")


if __name__ == "__main__":
    unittest.main()
