"""Tests for ``evals/scripts/eval_resume_train.py`` — focuses on the
Phase-A/B checkpoint cleanup that prevents the ~14 GiB resume-checkpoint
leak from accumulating and triggering k8s ephemeral-storage eviction
(``临时存储超出限制`` — see helper docstring + 0bc1c5 perf_log
R8/R14/R17 devspace-recovery notes for the failure mode this guards).
The pre-fix path hardcoded ``/tmp/resume_ckpt_<ppid>`` (overlay disk
counted against the 50 GiB cap); the later ``.artifacts/runs/<...>/``
path hit the same cap on a devspace, so the current path takes a
dispatcher-injected ``RESUME_SCRATCH_DIR`` = workspace-relative
``tmp/resume_scratch_<run>`` on the loop's mounted volume.

The eval script imports ``training_engine_tensor`` at module top, which
lives outside the test sys.path. We inject a ``MagicMock`` for that
package into ``sys.modules`` BEFORE loading the script — same pattern
used by ``test_launch_dp.py`` for the launcher's runtime-only deps.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "evals" / "scripts" / "eval_resume_train.py"


def _load_eval_resume_module():
    # Inject a fake training_engine_tensor.train_loop so the module-top
    # ``from training_engine_tensor.train_loop import ...`` succeeds in
    # a test env that doesn't carry the workload engine. The mock's
    # ``run_training_loop`` / ``TrainLoopConfig`` are placeholders;
    # individual tests patch them on the loaded module as needed.
    fake_te = mock.MagicMock()
    fake_train_loop = mock.MagicMock()
    fake_train_loop.run_training_loop = mock.MagicMock()
    # TrainLoopConfig must support dataclasses.replace; make it a dataclass.
    import dataclasses

    @dataclasses.dataclass
    class _FakeConfig:
        num_steps: int = 0
        micro_batch_size: int = 0
        seq_length: int = 0
        grad_accum_steps: int = 0
        seed: int = 0
        world_size: int = 0
        checkpoint_root: str = ""
        data_path: str = ""
        backend: str = ""
        megatron_root: str = ""
        start_step: int = 0
        save_path: str = ""
        resume_from: str = ""
        # Mirror the hash-capture knobs added to
        # workload/src/training_engine_tensor/train_loop.TrainLoopConfig
        # so eval_resume_train can construct TrainLoopConfig with the
        # full kwarg set (`hash_capture_level` / `hash_output` /
        # `persistent` threaded from the dispatcher's --hash-* CLI args).
        hash_capture_level: int = 0
        hash_output: str = ""
        persistent: bool = False

    fake_train_loop.TrainLoopConfig = _FakeConfig
    sys.modules["training_engine_tensor"] = fake_te
    sys.modules["training_engine_tensor.train_loop"] = fake_train_loop

    # ``eval_resume_train.py`` lives in ``evals/scripts/`` and does
    # ``from _runner_utils import abort_if_gpu_dirty`` (a sibling module
    # only visible when the dir is on sys.path — torchrun adds it
    # implicitly via ``-m``, but the importlib spec_from_file_location
    # loader does not). Mock the helper so the module can import.
    sys.modules.setdefault("_runner_utils", mock.MagicMock())

    spec = importlib.util.spec_from_file_location("_eval_resume_train_under_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _required_env(tmp_root: str) -> dict[str, str]:
    return {
        "NUM_STEPS": "20",
        "RESUME_SAVE_STEP": "10",
        "BACKEND": "torch",
        "MICRO_BATCH_SIZE": "4",
        "SEQ_LENGTH": "4096",
        "GRAD_ACCUM_STEPS": "1",
        "SEED": "1234",
        "WORLD_SIZE": "2",
        "CHECKPOINT_ROOT": tmp_root,
        "DATA_PATH": tmp_root,
        "MEGATRON_ROOT": "",
        "RANK": "0",
        "RESUME_SCRATCH_DIR": tmp_root,
    }


class TestCleanupHelper(unittest.TestCase):
    """``_cleanup_resume_dir`` is the new helper that owns the
    rank-coordinated rmtree (only rank 0 cleans, others no-op)."""

    def setUp(self) -> None:
        self.mod = _load_eval_resume_module()

    def test_rank_0_removes_directory(self) -> None:
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "resume_ckpt_42"
            target.mkdir()
            (target / "rank_0.pt").write_bytes(b"fake checkpoint")
            with mock.patch.dict(os.environ, {"RANK": "0"}, clear=False):
                self.mod._cleanup_resume_dir(str(target))
            self.assertFalse(target.exists(), "rank 0 must remove the dir")

    def test_non_zero_rank_does_not_remove(self) -> None:
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "resume_ckpt_42"
            target.mkdir()
            (target / "rank_1.pt").write_bytes(b"fake checkpoint")
            with mock.patch.dict(os.environ, {"RANK": "1"}, clear=False):
                self.mod._cleanup_resume_dir(str(target))
            # Rank 1+ no-op so the directory survives the call. Only
            # rank 0 owns the rmtree (avoids N-rank rmtree race).
            self.assertTrue(target.exists(), "non-zero rank must NOT remove")

    def test_missing_directory_does_not_raise(self) -> None:
        # ``ignore_errors=True`` keeps the cleanup itself failure-safe:
        # if Phase A crashed before the dir existed, the finally block
        # should not shadow the original exception with an FS error.
        with mock.patch.dict(os.environ, {"RANK": "0"}, clear=False):
            self.mod._cleanup_resume_dir("/nonexistent/path/to/resume_ckpt_999999")


class TestMainCleansAfterSuccess(unittest.TestCase):
    """End-to-end: when both training phases succeed, the cleanup runs."""

    def setUp(self) -> None:
        self.mod = _load_eval_resume_module()

    def test_cleanup_invoked_after_phase_b_returns(self) -> None:
        with TemporaryDirectory() as tmp:
            env = _required_env(tmp)
            calls: list[str] = []

            def fake_rtl(*args, **kwargs):
                calls.append(kwargs.get("loss_tag", "?"))

            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch.object(self.mod, "run_training_loop", side_effect=fake_rtl),
                mock.patch.object(self.mod, "os") as mock_os,
                mock.patch.object(self.mod, "shutil") as mock_shutil,
            ):
                # Preserve env access on the patched os.
                mock_os.environ = os.environ
                mock_os.getppid.return_value = 4242
                self.mod.main()

            # Three phases were invoked (Phase R, A, B).
            self.assertEqual(calls, ["LOSS_REF", "LOSS_RES", "LOSS_RES"])
            # The cleanup path: only rank 0 rmtrees, ignore_errors=True.
            mock_shutil.rmtree.assert_called_once_with(f"{tmp}/ckpt_4242", ignore_errors=True)


class TestMainCleansAfterFailure(unittest.TestCase):
    """End-to-end: even when a phase raises, the finally block still
    cleans up. Without this guarantee, a failed Phase A would leak
    a partial checkpoint indefinitely (no later run owns it)."""

    def setUp(self) -> None:
        self.mod = _load_eval_resume_module()

    def test_cleanup_invoked_when_phase_b_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            env = _required_env(tmp)
            call_log: list[str] = []

            def fake_rtl(*args, **kwargs):
                tag = kwargs.get("loss_tag", "?")
                call_log.append(tag)
                if len(call_log) == 3:  # Phase B
                    raise RuntimeError("simulated phase B failure")

            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch.object(self.mod, "run_training_loop", side_effect=fake_rtl),
                mock.patch.object(self.mod, "os") as mock_os,
                mock.patch.object(self.mod, "shutil") as mock_shutil,
            ):
                mock_os.environ = os.environ
                mock_os.getppid.return_value = 7777
                with self.assertRaises(RuntimeError):
                    self.mod.main()

            # Cleanup still ran in finally; the original exception
            # propagated unchanged for the rank-0 traceback emit.
            mock_shutil.rmtree.assert_called_once_with(f"{tmp}/ckpt_7777", ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
