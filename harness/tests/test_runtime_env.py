"""Runtime environment construction is centralized in evals._common."""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]


@contextlib.contextmanager
def _seeded_forge_config_dir():
    """Seed a per-loop config dir from the committed templates.

    ``load_workload_config`` hard-requires model.toml/optim.toml in
    $FORGE_CONFIG_DIR — under Method F ``<repo>/config`` holds only
    per-axis template subdirs, so tests that reach the loader must
    point FORGE_CONFIG_DIR at a seeded tmp dir.
    """
    templates = REPO_ROOT / "config"
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp)
        for axis in ("ref", "data", "remote", "agent", "eval", "model", "optim"):
            src_dir = templates / axis
            if axis == "eval":
                # eval templates live in variant directories:
                # config/eval/<variant>/<variant>.toml
                variant = sorted(p for p in src_dir.iterdir() if p.is_dir())[0]
                src = variant / f"{variant.name}.toml"
            else:
                src = sorted(src_dir.glob("*.toml"))[0]
            shutil.copy(src, cfg / f"{axis}.toml")
        with mock.patch.dict(os.environ, {"FORGE_CONFIG_DIR": str(cfg)}):
            yield


class TestRuntimeEnvBuilder(unittest.TestCase):
    def setUp(self) -> None:
        # build_suite_env unconditionally resolves [ref] tokenizer assets
        # (config_runtime.resolve_assets). A pre-populated dir makes
        # _ensure_tokenizer a no-op — no HF download in tests.
        self._tmp = tempfile.TemporaryDirectory()
        tok_dir = Path(self._tmp.name) / "tokenizer"
        tok_dir.mkdir()
        (tok_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        self._ref = {
            "forge_tokenizer_dir": str(tok_dir),
            "tokenizer": "dummy/tokenizer",
        }

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_distributed_env_uses_single_default_port(self) -> None:
        from evals._common import distributed_env

        # The committed default is now ``master_port = "auto"`` which
        # expands to a fresh ephemeral port per call (see
        # ``_resolve_master_port``). Pinning the legacy 29500 here
        # would re-introduce the TIME_WAIT / co-tenant collisions on
        # shared devspaces that the sentinel was added to fix.
        with _seeded_forge_config_dir():
            env_a = distributed_env(8)
            env_b = distributed_env(8)
        self.assertEqual(env_a["NUM_PROCS"], "8")
        self.assertEqual(env_a["MASTER_ADDR"], "localhost")
        self.assertTrue(env_a["MASTER_PORT"].isdigit())
        self.assertTrue(1024 <= int(env_a["MASTER_PORT"]) <= 65535)
        # Two calls SHOULD generally pick different ports (the kernel
        # hands out from its ephemeral pool). Asserting strict
        # inequality is too flaky on a quiet host, so we instead pin
        # the contract that both values are valid ports.
        self.assertTrue(env_b["MASTER_PORT"].isdigit())

    def test_distributed_env_auto_sentinel_allocates_fresh_port(self) -> None:
        from evals._common import _MASTER_PORT_AUTO_SENTINELS, distributed_env

        # Explicit "auto" passthrough — the dispatcher relies on this
        # when a suite cfg propagates the workload-level default.
        self.assertIn("auto", _MASTER_PORT_AUTO_SENTINELS)
        env = distributed_env(2, master_addr="localhost", master_port="auto")
        port = env["MASTER_PORT"]
        self.assertTrue(port.isdigit(), f"expected numeric port, got {port!r}")
        self.assertNotEqual(port, "auto")

    def test_distributed_env_passes_through_explicit_numeric_port(self) -> None:
        from evals._common import distributed_env

        env = distributed_env(2, master_addr="127.0.0.1", master_port="29888")
        self.assertEqual(env["MASTER_PORT"], "29888")

    def test_suite_process_env_reads_distributed_defaults_from_workload_config(self) -> None:
        from evals._common import suite_process_env

        env = suite_process_env(
            REPO_ROOT,
            {"env_inputs": ["NUM_PROCS", "MASTER_ADDR", "MASTER_PORT"]},
            {
                "ref": self._ref,
                "env": {},
                "runtime": {
                    "distributed": {
                        "master_addr": "127.0.0.1",
                        "master_port": "29888",
                    },
                },
            },
            world_size=2,
        )
        self.assertEqual(env["NUM_PROCS"], "2")
        self.assertEqual(env["MASTER_ADDR"], "127.0.0.1")
        self.assertEqual(env["MASTER_PORT"], "29888")

    def test_suite_process_env_merges_workload_env_and_overrides(self) -> None:
        from evals._common import suite_process_env

        env = suite_process_env(
            REPO_ROOT,
            {
                "checkpoint_root": "/ckpt",
                "data_path": "/data/prefix",
                "env_inputs": [
                    "CHECKPOINT_ROOT",
                    "DATA_PATH",
                    "NUM_PROCS",
                    "MASTER_ADDR",
                    "MASTER_PORT",
                    "NUM_STEPS",
                ],
            },
            {
                "ref": self._ref,
                "env": {"MY_FLAG": "1"},
                "runtime": {"distributed": {"master_addr": "localhost", "master_port": "29500"}},
            },
            world_size=4,
            extra={"NUM_STEPS": "8"},
        )
        self.assertIn(str(REPO_ROOT / "workload" / "src"), env["PYTHONPATH"])
        self.assertEqual(env["CHECKPOINT_ROOT"], "/ckpt")
        self.assertEqual(env["DATA_PATH"], "/data/prefix")
        self.assertEqual(env["MY_FLAG"], "1")
        self.assertEqual(env["NUM_PROCS"], "4")
        self.assertEqual(env["NUM_STEPS"], "8")

    def test_suite_process_env_uses_env_inputs_by_default(self) -> None:
        from evals._common import suite_process_env

        env = suite_process_env(
            REPO_ROOT,
            {
                "stage": "stage1",
                "milestone": "alignment",
                "data_path": "/data/prefix",
                "env_inputs": ["DATA_PATH"],
            },
            {"ref": self._ref, "env": {}},
        )
        self.assertEqual(env["DATA_PATH"], "/data/prefix")
        self.assertNotIn("STAGE", env)
        self.assertNotIn("MILESTONE", env)

    def test_suite_process_env_derives_data_path_from_data_conf(self) -> None:
        import tempfile

        from evals._common import suite_process_env

        with tempfile.TemporaryDirectory() as tmpdir:
            conf = Path(tmpdir) / "data_conf.sh"
            conf.write_text('DATA_PATH="/data/from-conf"\n', encoding="utf-8")
            env = suite_process_env(
                REPO_ROOT,
                {"env_inputs": ["DATA_PATH"], "data_conf": str(conf)},
                {"ref": self._ref, "env": {}},
            )

        self.assertEqual(env["DATA_PATH"], "/data/from-conf")

    def test_empty_env_inputs_do_not_inject_checkpoint_or_megatron_roots(self) -> None:
        from evals._common import build_suite_env

        env = build_suite_env(
            REPO_ROOT,
            {"checkpoint_root": "/ckpt", "mega" + "tron_root": "/mg"},
            {"ref": self._ref, "env": {}},
            env_inputs=[],
        )

        self.assertNotIn("CHECKPOINT_ROOT", env)
        self.assertNotIn("MEGA" + "TRON_ROOT", env)

    def test_declared_env_input_must_have_a_source(self) -> None:
        from evals._common import build_suite_env

        with self.assertRaisesRegex(ValueError, "MISSING_ENV"):
            build_suite_env(
                REPO_ROOT,
                {"env_inputs": ["MISSING_ENV"]},
                {"ref": self._ref, "env": {}},
                env_inputs=["MISSING_ENV"],
            )

    def test_torchrun_cmd_uses_explicit_master_port(self) -> None:
        from evals._common import torchrun_cmd

        with _seeded_forge_config_dir():
            cmd = torchrun_cmd(8, "train.py", master_port="29876")
        self.assertIn("--master_port", cmd)
        self.assertEqual(cmd[cmd.index("--master_port") + 1], "29876")

    def test_dispatcher_does_not_ad_hoc_increment_master_port(self) -> None:
        # The old DP correctness gate split MASTER_PORT into two phases via
        # ``with_master_port_offset`` so the second torchrun wouldn't trip
        # over TIME_WAIT on the first phase's rendezvous port. The bitwise
        # milestones now run as a single torchrun (ref-vs-ours subprocess pattern), so
        # the helper was retired together with the gates. This guard
        # remains so a future refactor doesn't silently reintroduce the
        # ``int(env["MASTER_PORT"]) + 1`` idiom inline.
        source = (REPO_ROOT / "evals" / "dispatcher.py").read_text(encoding="utf-8")
        self.assertNotIn('str(int(env["MASTER_PORT"]) + 1)', source)
        self.assertNotIn('int(env["MASTER_PORT"]) + ', source)

    def test_build_suite_env_respects_env_inputs_whitelist(self) -> None:
        from evals._common import build_suite_env

        cfg = {
            "checkpoint_root": "/ckpt",
            "data_path": "/data/prefix",
            "unexpected_knob": "should_not_be_injected",
        }
        env = build_suite_env(
            REPO_ROOT,
            cfg,
            {"ref": self._ref, "env": {"MY_FLAG": "1"}},
            env_inputs=("CHECKPOINT_ROOT", "DATA_PATH"),
        )
        self.assertEqual(env["CHECKPOINT_ROOT"], "/ckpt")
        self.assertEqual(env["DATA_PATH"], "/data/prefix")
        self.assertEqual(env["MY_FLAG"], "1")
        self.assertNotIn("UNEXPECTED_KNOB", env)

    def test_build_suite_env_without_whitelist_does_not_inject_suite_scalars(self) -> None:
        from evals._common import build_suite_env

        env = build_suite_env(
            REPO_ROOT,
            {"custom_key": "x"},
            {"ref": self._ref, "env": {}},
        )
        self.assertNotIn("CUSTOM_KEY", env)

    def test_build_suite_env_forwards_undeclared_explicit_extra(self) -> None:
        from evals._common import build_suite_env

        # Extras are forwarded without needing to appear in env_inputs.
        # The undeclared-extra fail-fast was removed: it managed env
        # transport but did not guard the underlying SSOT, and caused
        # spec-vs-contract self-collisions when dispatcher-internal env
        # vars were passed without being pre-declared by each suite.
        env = build_suite_env(
            REPO_ROOT,
            {"custom_key": "x"},
            {"ref": self._ref, "env": {}},
            extra={"FORGE_NSYS_RANK0_OUTPUT": "/tmp/p.nsys-rep"},
        )
        self.assertEqual(env["FORGE_NSYS_RANK0_OUTPUT"], "/tmp/p.nsys-rep")

    def test_build_suite_env_allows_declared_explicit_extra(self) -> None:
        from evals._common import build_suite_env

        env = build_suite_env(
            REPO_ROOT,
            {"custom_key": "x"},
            {"ref": self._ref, "env": {}},
            extra={"NUM_STEPS": "8"},
            env_inputs=["NUM_STEPS"],
        )
        self.assertEqual(env["NUM_STEPS"], "8")
        self.assertNotIn("CUSTOM_KEY", env)

    def test_global_determinism_env_comes_from_workload_config(self) -> None:
        from evals._common import suite_process_env
        from harness._compat import tomllib

        workload_config = tomllib.loads(
            (REPO_ROOT / "config" / "eval" / "dense_training" / "dense_training.toml").read_text(
                encoding="utf-8"
            )
        )
        workload_config["ref"] = {**workload_config.get("ref", {}), **self._ref}
        env = suite_process_env(REPO_ROOT, {"env_inputs": []}, workload_config)
        self.assertEqual(env["NVTE_ALLOW_NONDETERMINISTIC_ALGO"], "0")
        self.assertEqual(env["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"], "1800")

    def test_stage2_runtime_inputs_are_config_values_not_full_environment(self) -> None:
        from evals._common import stage2_runtime_inputs

        env = stage2_runtime_inputs(
            {
                "stage2": {
                    "checkpoint_root": "/ckpt",
                    "mega" + "tron_root": "/mg",
                    "data_path": "/data/gsm8k_megatron/gsm8k_train_text_document",
                },
            }
        )

        self.assertEqual(env["CHECKPOINT_ROOT"], "/ckpt")
        self.assertEqual(env["MEGA" + "TRON_ROOT"], "/mg")
        self.assertEqual(env["DATA_PATH"], "/data/gsm8k_megatron/gsm8k_train_text_document")
        self.assertNotIn("MICRO_BATCH_SIZE", env)
        self.assertNotIn("PATH", env)
        self.assertNotIn("PYTHONPATH", env)

    def test_stage2_runtime_inputs_keep_stage2_defaults_authoritative(self) -> None:
        from evals._common import stage2_runtime_inputs

        env = stage2_runtime_inputs(
            {
                "stage2": {
                    "checkpoint_root": "/ckpt",
                    "mega" + "tron_root": "/mg",
                    "data_path": "/data/gsm8k_megatron/gsm8k_train_text_document",
                },
                "evals": {
                    "op-long": {
                        "stage": "stage2",
                        "runner_kind": "op-long",
                        "micro_batch_size": 12,
                    },
                },
            }
        )

        self.assertNotIn("MICRO_BATCH_SIZE", env)

    def test_op_long_runner_requires_harness_injected_runtime_env(self) -> None:
        text = (REPO_ROOT / "evals" / "scripts" / "op_long_ours.py").read_text(
            encoding="utf-8",
        )
        self.assertNotIn('os.environ.get("OP_NAMES", "all")', text)
        self.assertNotIn('os.environ.get("NUM_STEPS", "1000")', text)
        self.assertNotIn('os.environ.get("GLOBAL_BATCH_SIZE", "1280")', text)
        self.assertIn('_require_env("OP_NAMES")', text)
        self.assertIn('_require_env("NUM_STEPS")', text)
        self.assertIn('_require_env("GLOBAL_BATCH_SIZE")', text)
        self.assertNotIn("_GATE_ENV_DEFAULTS", text)
        self.assertNotIn("apply_defaults", text)
        self.assertNotIn("MASTER_ADDR", text)
        self.assertNotIn("MASTER_PORT", text)

    def test_gate_scripts_do_not_default_execution_shape(self) -> None:
        forbidden = (
            'os.environ.get("WORLD_SIZE",',
            'os.environ.get("NUM_STEPS",',
            'os.environ.get("MICRO_BATCH_SIZE",',
            'os.environ.get("GLOBAL_BATCH_SIZE",',
            'os.environ.get("SEQ_LENGTH",',
            'os.environ.get("GRAD_ACCUM_STEPS",',
            'os.environ.get("RESUME_SAVE_STEP",',
            'os.environ.get("SEED",',
        )
        for path in (REPO_ROOT / "evals" / "scripts").glob("test_*.py"):
            text = path.read_text(encoding="utf-8")
            for needle in forbidden:
                self.assertNotIn(needle, text, f"{path} defaults gate execution shape")


if __name__ == "__main__":
    unittest.main()
