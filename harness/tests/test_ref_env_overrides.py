"""Tests for the ref-script invocation contract that survived the
gate-product collapse.

The dispatcher-side FORGE_* projection, the per-suite ``ref_env`` /
``optim_overrides`` env channels, and the ``ref → gate_metadata.json →
ours`` metadata round-trip are all gone: the rendered gate product is now
the sole source for both ref and ours. What remains here pins the parts of
the contract that are still live:

  - The ``[model]`` / ``[optim]`` axis schema validators still guard the
    axis templates (every required key present, scalar-typed).
  - ``run_ref_script`` keeps its training-shape-free signature so all
    overrides flow through ``extra_env`` (which the product emitter feeds).
  - ``cfg.ref_extra_args`` still appends bare CLI tokens to the L0 script.
  - ``DETERMINISTIC=0`` flips off the ours-side deterministic stack.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Engine-contract tests need ``training_engine_tensor`` on the path; the
# engine lives under ``workload/src/`` so harness unit tests can pull it
# in without copying the harness convention used elsewhere in this file.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENGINE_SRC = _REPO_ROOT / "workload" / "src"
if str(_ENGINE_SRC) not in sys.path:
    sys.path.insert(0, str(_ENGINE_SRC))


class _FakeRefRun:
    def __init__(self, *, succeeded: bool = True, metadata: dict | None = None) -> None:
        self.succeeded = succeeded
        self.metadata = metadata or {}
        self.loss_file = None
        self.stdout_path = Path("/dev/null")


def _seed_dummy_ref_script(repo_root: Path) -> Path:
    """Create a placeholder ref-script and minimal workload.toml.

    The dummy ref-script under ``ref/reference/`` satisfies the path-
    existence check inside ``run_via_ref_script``. The dummy eval
    registry at ``config/eval/dense_training/dense_training.toml`` (the
    Method F variant-directory layout ``_load_workload_config_for_ref``
    resolves) carries the required ``[workload]`` metadata plus
    ``[runtime.distributed]``, ``[ref].ref_script`` and an
    ``[evals.long-train]`` entry so ``ref_script`` resolves under the
    strict basename contract and ``suite_ref_timeout_s`` finds the
    suite. Both files are scaffold — the tests mock the actual
    ``run_ref_script`` call, so neither file's contents matter past
    these contracts.
    """
    ref_dir = repo_root / "ref" / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    script = ref_dir / "train_minicpm4_0.5b_gsm8k.sh"
    script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    # The user's gitignored ``config/ref.toml`` (loaded by
    # ``load_workload_config`` during test runs) may override
    # ``[ref].ref_script`` to a different basename. Seed every
    # ref-script basename this repo ships so the path-existence check
    # inside ``_resolve_customer_ref_script`` succeeds regardless of
    # which one wins the merge precedence.
    for sibling in ("run_16gpu_1000step_pure_mup_mtp.sh",):
        (ref_dir / sibling).write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")

    eval_dir = repo_root / "config" / "eval" / "dense_training"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "dense_training.toml").write_text(
        "[workload]\n"
        'id = "test-fixture"\n'
        'display_name = "Test fixture"\n'
        "requires_cuda = false\n"
        "[runtime.distributed]\n"
        'master_addr = "localhost"\n'
        'master_port = "29500"\n'
        "[ref]\n"
        'ref_script = "train_minicpm4_0.5b_gsm8k.sh"\n'
        "[evals.long-train]\n"
        'stage = "stage1"\n'
        'runner_kind = "long-train"\n'
        'script = "x.py"\n'
        'launcher = "y.py"\n'
        'env_inputs = ["A"]\n'
        "timeout_s = 600\n",
        encoding="utf-8",
    )
    return script


class TestRefScriptRunnerStillSSOTSafe(unittest.TestCase):
    """The runner keeps its SSOT contract: no training-shape parameters.

    ``tools/ref_script_runner.run_ref_script`` must still refuse to grow
    training-shape parameters in its signature; all such overrides flow
    through ``extra_env``, which is exactly the channel the gate-product
    emitter feeds.
    """

    def test_run_ref_script_signature_unchanged(self) -> None:
        import inspect

        from tools.ref_script_runner import run_ref_script

        params = inspect.signature(run_ref_script).parameters
        for forbidden in (
            "num_steps",
            "world_size",
            "micro_batch_size",
            "global_batch_size",
            "data_path",
            "seed",
            "ref_env",
        ):
            self.assertNotIn(forbidden, params)
        self.assertIn("extra_env", params)


class TestModelAxisSchemaValidation(unittest.TestCase):
    """The [model] axis (config/model.toml) schema validator guards the
    axis template: every geometry key present and scalar-typed. The
    renderer is now the sole projection of these into the gate products.
    """

    def test_validation_rejects_unknown_key(self) -> None:
        from harness.config_runtime import _validate_model_config

        bogus = {"name": "x", "num_layers": 24, "typo": 1}
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            _validate_model_config(Path("/tmp/x.toml"), bogus)

    def test_validation_rejects_missing_required(self) -> None:
        from harness.config_runtime import _validate_model_config

        partial = {"name": "x"}
        with self.assertRaisesRegex(ValueError, "missing required"):
            _validate_model_config(Path("/tmp/x.toml"), partial)

    def test_validation_rejects_bool(self) -> None:
        from harness.config_runtime import (
            _MODEL_GEOMETRY_KEYS,
            _validate_model_config,
        )

        full = {"name": "x"}
        for k in _MODEL_GEOMETRY_KEYS:
            full[k] = 1
        full["num_layers"] = True  # bool subclasses int — must be rejected
        with self.assertRaisesRegex(ValueError, r"\[model\]\.num_layers must be int or float"):
            _validate_model_config(Path("/tmp/x.toml"), full)


class TestOptimAxisSchemaValidation(unittest.TestCase):
    """The [optim] axis schema validator guards the axis template."""

    def test_validation_rejects_unknown_key(self) -> None:
        from harness.config_runtime import _validate_optim_config

        bogus = {"lr": 1e-4, "typo": 1}
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            _validate_optim_config(Path("/tmp/x.toml"), bogus)

    def test_validation_rejects_missing_required(self) -> None:
        from harness.config_runtime import _validate_optim_config

        partial = {"lr": 1e-4}
        with self.assertRaisesRegex(ValueError, "missing required"):
            _validate_optim_config(Path("/tmp/x.toml"), partial)


class TestRefExtraArgsForwarded(unittest.TestCase):
    """``cfg.ref_extra_args`` reaches the L0 ref-script command line.

    The ``long-train`` suite uses this to inject ``--no-deterministic``
    without editing the frozen ref script; the L0 script's trailing
    ``$@`` forwards it to its torchrun python entry. Caller-supplied
    ``extra_ref_args`` still wins over cfg so programmatic callers can
    override per-invocation.
    """

    def test_cfg_ref_extra_args_reaches_run_ref_script(self) -> None:
        from evals import _common

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            _seed_dummy_ref_script(repo_root)
            cfg = {
                "ref_extra_args": ["--no-deterministic"],
            }
            captured: dict[str, object] = {}

            def fake_run_ref_script(**kwargs: object) -> _FakeRefRun:
                captured["extra_ref_args"] = kwargs.get("extra_ref_args")
                return _FakeRefRun(metadata={"num_steps": 400})

            with (
                mock.patch.object(_common, "_ref_script_path_env", return_value={}),
                mock.patch(
                    "tools.ref_script_runner.run_ref_script",
                    side_effect=fake_run_ref_script,
                ),
                mock.patch("tools.ref_script_runner.parse_loss_dump", return_value={}),
                mock.patch("tools.ref_script_runner.parse_stdout_loss", return_value={}),
            ):
                _common.run_via_ref_script(
                    repo_root=repo_root,
                    suite_key="long-train",
                    cfg=cfg,
                    artifact_dir=repo_root / ".artifacts",
                    capture_loss_trace=False,
                )

        self.assertEqual(captured["extra_ref_args"], ["--no-deterministic"])

    def test_caller_extra_ref_args_wins_over_cfg(self) -> None:
        from evals import _common

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            _seed_dummy_ref_script(repo_root)
            cfg = {"ref_extra_args": ["--no-deterministic"]}
            captured: dict[str, object] = {}

            def fake_run_ref_script(**kwargs: object) -> _FakeRefRun:
                captured["extra_ref_args"] = kwargs.get("extra_ref_args")
                return _FakeRefRun(metadata={"num_steps": 400})

            with (
                mock.patch.object(_common, "_ref_script_path_env", return_value={}),
                mock.patch(
                    "tools.ref_script_runner.run_ref_script",
                    side_effect=fake_run_ref_script,
                ),
                mock.patch("tools.ref_script_runner.parse_loss_dump", return_value={}),
                mock.patch("tools.ref_script_runner.parse_stdout_loss", return_value={}),
            ):
                _common.run_via_ref_script(
                    repo_root=repo_root,
                    suite_key="long-train",
                    cfg=cfg,
                    artifact_dir=repo_root / ".artifacts",
                    capture_loss_trace=False,
                    extra_ref_args=["--override-from-caller"],
                )

        self.assertEqual(captured["extra_ref_args"], ["--override-from-caller"])

    def test_no_cfg_ref_extra_args_passes_none(self) -> None:
        from evals import _common

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            _seed_dummy_ref_script(repo_root)
            cfg: dict[str, object] = {}
            captured: dict[str, object] = {}

            def fake_run_ref_script(**kwargs: object) -> _FakeRefRun:
                captured["extra_ref_args"] = kwargs.get("extra_ref_args")
                return _FakeRefRun(metadata={"num_steps": 400})

            with (
                mock.patch.object(_common, "_ref_script_path_env", return_value={}),
                mock.patch(
                    "tools.ref_script_runner.run_ref_script",
                    side_effect=fake_run_ref_script,
                ),
                mock.patch("tools.ref_script_runner.parse_loss_dump", return_value={}),
                mock.patch("tools.ref_script_runner.parse_stdout_loss", return_value={}),
            ):
                _common.run_via_ref_script(
                    repo_root=repo_root,
                    suite_key="long-train",
                    cfg=cfg,
                    artifact_dir=repo_root / ".artifacts",
                    capture_loss_trace=False,
                )

        self.assertIsNone(captured["extra_ref_args"])

    def test_cfg_ref_extra_args_must_be_list(self) -> None:
        from evals import _common

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            _seed_dummy_ref_script(repo_root)
            cfg = {"ref_extra_args": "--no-deterministic"}  # str, not list

            with (
                mock.patch.object(_common, "_ref_script_path_env", return_value={}),
                mock.patch(
                    "tools.ref_script_runner.run_ref_script",
                    side_effect=lambda **_: _FakeRefRun(metadata={"num_steps": 1}),
                ),
                mock.patch("tools.ref_script_runner.parse_loss_dump", return_value={}),
                mock.patch("tools.ref_script_runner.parse_stdout_loss", return_value={}),
            ):
                _common.run_via_ref_script(
                    repo_root=repo_root,
                    suite_key="long-train",
                    cfg=cfg,
                    artifact_dir=repo_root / ".artifacts",
                    capture_loss_trace=False,
                )


class TestRefExtraArgsSchemaValidation(unittest.TestCase):
    """``config_runtime`` rejects malformed ``ref_extra_args`` at load time."""

    def test_non_list_raises(self) -> None:
        from harness import config_runtime

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dense_training.toml"
            path.write_text(
                "[workload]\n"
                'id = "x"\n'
                'display_name = "x"\n'
                "requires_cuda = false\n"
                "[runtime.distributed]\n"
                'master_addr = "localhost"\n'
                'master_port = "29500"\n'
                "[defaults]\n"
                'ref_script = "x.sh"\n'
                "[evals.long-train]\n"
                'stage = "stage1"\n'
                'runner_kind = "long-train"\n'
                'script = "x.py"\n'
                'launcher = "y.py"\n'
                'env_inputs = ["A"]\n'
                'ref_extra_args = "not-a-list"\n',
                encoding="utf-8",
            )
            # include_user_config=False keeps the synthetic fixture
            # hermetic (no $FORGE_CONFIG_DIR model/optim merge).
            with self.assertRaisesRegex(ValueError, "ref_extra_args.*list"):
                config_runtime.load_workload_config(str(path), include_user_config=False)

    def test_empty_token_raises(self) -> None:
        from harness import config_runtime

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dense_training.toml"
            path.write_text(
                "[workload]\n"
                'id = "x"\n'
                'display_name = "x"\n'
                "requires_cuda = false\n"
                "[runtime.distributed]\n"
                'master_addr = "localhost"\n'
                'master_port = "29500"\n'
                "[defaults]\n"
                'ref_script = "x.sh"\n'
                "[evals.long-train]\n"
                'stage = "stage1"\n'
                'runner_kind = "long-train"\n'
                'script = "x.py"\n'
                'launcher = "y.py"\n'
                'env_inputs = ["A"]\n'
                'ref_extra_args = ["--ok", ""]\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-empty"):
                config_runtime.load_workload_config(str(path), include_user_config=False)


class TestTrainLoopDeterministicEnv(unittest.TestCase):
    """``DETERMINISTIC=0`` flips off the deterministic stack ours-side."""

    def test_default_enables_full_determinism(self) -> None:
        """No env var → full deterministic stack ON (the bitwise-gate default).

        Module under test imports torch; skip when torch is unavailable
        in this CPU-only test env (matches the pattern used by other
        engine-contract tests).

        ``_enable_determinism`` is an agent-implemented helper inside
        ``training_engine_tensor.train_loop``; until the agent has
        filled in the stub body it does not exist as a symbol. Skip
        in that case so the harness check-in test suite stays green
        before the engine implementation lands.
        """
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available in this env")

        import torch

        try:
            from training_engine_tensor.train_loop import _enable_determinism
        except ImportError:
            self.skipTest(
                "_enable_determinism not yet implemented in "
                "training_engine_tensor.train_loop (agent stub)"
            )

        prev = os.environ.pop("DETERMINISTIC", None)
        # Cache flash-SDP flag we may flip — restore it after the test.
        prev_flash = torch.backends.cuda.flash_sdp_enabled()
        try:
            _enable_determinism(1234)
            self.assertFalse(torch.backends.cuda.flash_sdp_enabled())
            self.assertFalse(torch.backends.cuda.mem_efficient_sdp_enabled())
            self.assertTrue(torch.backends.cudnn.deterministic)
        finally:
            torch.backends.cuda.enable_flash_sdp(prev_flash)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.use_deterministic_algorithms(False, warn_only=False)
            if prev is not None:
                os.environ["DETERMINISTIC"] = prev

    def test_zero_disables_determinism(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available in this env")

        import torch

        try:
            from training_engine_tensor.train_loop import _enable_determinism
        except ImportError:
            self.skipTest(
                "_enable_determinism not yet implemented in "
                "training_engine_tensor.train_loop (agent stub)"
            )

        prev = os.environ.pop("DETERMINISTIC", None)
        prev_flash = torch.backends.cuda.flash_sdp_enabled()
        try:
            os.environ["DETERMINISTIC"] = "0"
            _enable_determinism(1234)
            self.assertTrue(torch.backends.cuda.flash_sdp_enabled())
            self.assertTrue(torch.backends.cuda.mem_efficient_sdp_enabled())
        finally:
            torch.backends.cuda.enable_flash_sdp(prev_flash)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.use_deterministic_algorithms(False, warn_only=False)
            if prev is not None:
                os.environ["DETERMINISTIC"] = prev
            elif "DETERMINISTIC" in os.environ:
                del os.environ["DETERMINISTIC"]


if __name__ == "__main__":
    unittest.main()
