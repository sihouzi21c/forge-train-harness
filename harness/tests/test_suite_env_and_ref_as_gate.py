"""Tests for evals._common suite env helpers + Stage 2 ref-as-gate SSOT.

Two independent concerns share this file by convenience:

* ``TestSuiteEnv`` — pins the suite-env helpers in ``evals._common`` so a
  stage-2 (or any other) suite cannot silently drop the canonical PYTHONPATH
  / data-config inheritance / per-suite override semantics.
* ``TestRefAsGateSSOT`` — pins the "ref script as the only baseline" invariant
  for the ``[stage2]`` config block (no frozen JSON / SHA anchors).

Registry name lookup is owned by ``tools/stage2_config.operator_names``;
both contracts above must continue to hold against that SSOT.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class TestSuiteEnv(unittest.TestCase):
    """Lock the shared suite environment helpers in evals._common."""

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

    def test_suite_env_includes_pythonpath(self) -> None:
        from evals._common import base_env
        from harness import config_runtime

        env = base_env(REPO_ROOT)
        self.assertIn(str(config_runtime.workload_src_path(REPO_ROOT)), env["PYTHONPATH"])

    def test_suite_env_merges_workload_env(self) -> None:
        from evals._common import build_suite_env

        wc = {"ref": self._ref, "env": {"MY_VAR": "hello"}}
        env = build_suite_env(REPO_ROOT, {}, wc)
        self.assertEqual(env["MY_VAR"], "hello")

    def test_injected_pytorchjob_topology_env_reaches_suite(self) -> None:
        # Multi-node (kind=job, [remote].nodes > 1) contract: the PyTorchJob
        # operator injects the rendezvous env (RANK / WORLD_SIZE / MASTER_ADDR
        # / MASTER_PORT / GPUS_PER_NODE) into every pod, and the suite launcher
        # (train_ours_al.sh's torchrun --nnodes/--node_rank, the RL ray entry)
        # reads it straight from the process environment. build_suite_env must
        # therefore INHERIT ambient os.environ untouched for these keys — a
        # future refactor that builds a clean env would silently pin every pod
        # to single-node defaults. This locks the passthrough both generation
        # routes depend on.
        import os

        from evals._common import build_suite_env

        injected = {
            "RANK": "1",
            "WORLD_SIZE": "2",
            "MASTER_ADDR": "pytorchjob-master-0.pytorchjob-ns",
            "MASTER_PORT": "23456",
            "GPUS_PER_NODE": "8",
        }
        saved = {k: os.environ.get(k) for k in injected}
        try:
            os.environ.update(injected)
            env = build_suite_env(REPO_ROOT, {}, {"ref": self._ref})
            for key, value in injected.items():
                self.assertEqual(env[key], value, msg=f"{key} not passed through")
        finally:
            for key, prev in saved.items():
                if prev is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = prev

    def test_stage2_runtime_inputs_exposes_declared_values_only(self) -> None:
        from evals._common import stage2_runtime_inputs

        wc = {
            "stage2": {
                "checkpoint_root": "/ckpt",
                "mega" + "tron_root": "/mg",
                "data_path": "/data/gsm8k_megatron/gsm8k_train_text_document",
            }
        }
        env = stage2_runtime_inputs(wc)
        self.assertNotIn("MICRO_BATCH_SIZE", env)
        self.assertEqual(env["DATA_PATH"], "/data/gsm8k_megatron/gsm8k_train_text_document")
        self.assertEqual(env["CHECKPOINT_ROOT"], "/ckpt")

    def test_stage2_runtime_inputs_inherits_from_defaults(self) -> None:
        from evals._common import stage2_runtime_inputs

        wc = {
            "ref": {
                "checkpoint_root": "/default_ckpt",
                "mega" + "tron_root": "/default_mg",
            },
            "stage2": {
                "data_path": "/data/gsm8k_megatron/gsm8k_train_text_document",
            },
        }
        env = stage2_runtime_inputs(wc)
        self.assertEqual(env["CHECKPOINT_ROOT"], "/default_ckpt")
        self.assertEqual(env["DATA_PATH"], "/data/gsm8k_megatron/gsm8k_train_text_document")

    def test_suite_env_overrides_process_env_for_declared_contract(self) -> None:
        import os

        from evals._common import build_suite_env

        old = os.environ.get("MY_LOCKED_VAR")
        try:
            os.environ["MY_LOCKED_VAR"] = "original"
            wc = {"ref": self._ref, "env": {"my_locked_var": "override_attempt"}}
            env = build_suite_env(REPO_ROOT, {}, wc)
            self.assertEqual(env["MY_LOCKED_VAR"], "override_attempt")
        finally:
            if old is None:
                os.environ.pop("MY_LOCKED_VAR", None)
            else:
                os.environ["MY_LOCKED_VAR"] = old

    def test_suite_specific_config_overrides_global_env(self) -> None:
        from evals._common import build_suite_env

        wc = {"ref": self._ref, "env": {"micro_batch_size": "1"}}
        env = build_suite_env(
            REPO_ROOT,
            {"micro_batch_size": 10},
            wc,
            env_inputs=["MICRO_BATCH_SIZE"],
        )
        self.assertEqual(env["MICRO_BATCH_SIZE"], "10")


class TestRefAsGateSSOT(unittest.TestCase):
    """Lock the ref-as-gate SSOT: no frozen baseline keys in [stage2]."""

    def _load_stage2(self) -> dict:
        from harness._compat import tomllib

        config_path = REPO_ROOT / "config" / "eval" / "dense_training" / "dense_training.toml"
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
        return data.get("stage2", {})

    def test_stage2_has_required_runtime_fields(self) -> None:
        from harness._compat import tomllib

        config_path = REPO_ROOT / "config" / "eval" / "dense_training" / "dense_training.toml"
        tomllib.loads(config_path.read_text(encoding="utf-8"))
        stage2 = self._load_stage2()
        # ``micro_batch_size`` is owned by the L0 ref script's
        # FORGE_GATE preset (→ gate_metadata.json), not by [defaults]
        # or [stage2]. op-long ships its own MICRO_BATCH_SIZE_OVERRIDE
        # via [evals.op-long.ref_env]. Assert there is no [stage2]-local
        # ``micro_batch_size`` that would shadow the ref-script SSOT.
        self.assertNotIn("micro_batch_size", stage2)

    def test_no_frozen_baseline_keys(self) -> None:
        stage2 = self._load_stage2()
        for forbidden in (
            "frozen_baseline_job",
            "frozen_baseline_loss_file",
            "use_frozen_baseline",
            "baseline_ref",
            "baseline_loss_path",
        ):
            self.assertNotIn(
                forbidden,
                stage2,
                f"[stage2].{forbidden} reintroduces frozen-JSON fallback. "
                "SSOT model is ref-script-only — see "
                "README.md.",
            )

    def test_ref_script_exists(self) -> None:
        from harness._compat import tomllib

        for path in sorted((REPO_ROOT / "config" / "ref").glob("*.toml")):
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            ref_script = data["ref"]["ref_script"]
            ref = REPO_ROOT / "ref" / "reference" / ref_script
            self.assertTrue(
                ref.exists(),
                f"L0 ref script {ref} missing — ref-as-gate model requires "
                f"the basename declared in {path} [ref].ref_script "
                f"(got {ref_script!r}) to exist.",
            )

    def test_ref_script_helper_doc_does_not_mention_sha_anchor(self) -> None:
        text = (REPO_ROOT / "evals" / "_common.py").read_text(encoding="utf-8")
        self.assertNotIn("SHA mismatch", text)
        self.assertIn("no frozen SHA anchor", text)


if __name__ == "__main__":
    unittest.main()
