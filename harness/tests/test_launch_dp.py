"""Tests for the rank-0 nsys wrap in ``evals/scripts/launch_dp.py``.

The launcher itself is integration-level (spawns subprocess.Popen);
these tests exercise the pure helper that decides whether and how to
wrap rank 0's argv with ``nsys profile``.
"""

from __future__ import annotations

import contextlib
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_PATH = REPO_ROOT / "evals" / "scripts" / "launch_dp.py"


def _load_launcher_module():
    spec = importlib.util.spec_from_file_location("_launch_dp_under_test", LAUNCHER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestRank0NsysWrap(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _load_launcher_module()

    def test_returns_none_when_env_unset(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=False):
            for var in ("FORGE_NSYS_RANK0_OUTPUT",):
                if var in __import__("os").environ:
                    del __import__("os").environ[var]
            self.assertIsNone(self.mod._rank0_nsys_cmd(["script.py", "--flag"]))

    def test_returns_nsys_wrapped_cmd_when_env_set(self) -> None:
        out_path = "/tmp/nsys_run/rank0.nsys-rep"
        fake_nsys = "/usr/local/bin/nsys"
        with mock.patch.dict("os.environ", {"FORGE_NSYS_RANK0_OUTPUT": out_path}):
            with mock.patch("shutil.which", return_value=fake_nsys):
                cmd = self.mod._rank0_nsys_cmd(["evals/scripts/eval_x.py"])
        assert cmd is not None
        self.assertEqual(cmd[0], fake_nsys)
        self.assertEqual(cmd[1], "profile")
        self.assertIn("-t", cmd)
        self.assertIn("cuda,osrt", cmd)
        self.assertIn(f"--output={out_path}", cmd)
        self.assertIn("--force-overwrite=true", cmd)
        # python + child argv tail must appear after nsys flags
        self.assertEqual(cmd[-2], sys.executable)
        self.assertEqual(cmd[-1], "evals/scripts/eval_x.py")

    def test_raises_when_nsys_missing_from_path(self) -> None:
        with mock.patch.dict("os.environ", {"FORGE_NSYS_RANK0_OUTPUT": "/tmp/r0.nsys-rep"}):
            with mock.patch("shutil.which", return_value=None):
                with self.assertRaises(RuntimeError) as ctx:
                    self.mod._rank0_nsys_cmd(["script.py"])
        self.assertIn("nsys", str(ctx.exception))
        self.assertIn("PATH", str(ctx.exception))


class _FakeProc:
    """Minimal ``subprocess.Popen`` stand-in for ``_wait_with_fast_fail``.

    Tracks ``poll`` / ``terminate`` / ``wait`` / ``kill`` calls so the
    test can assert that the launcher reacts in seconds (not minutes)
    when a peer rank crashes.

    ``poll_results`` is an iterator of return values for ``poll()``
    consumed in order; once exhausted, ``poll()`` keeps returning the
    final value. ``terminate_on_call_exit_code`` is the returncode the
    fake assumes after ``terminate()`` is honoured.
    """

    def __init__(
        self,
        *,
        poll_results: list[int | None],
        terminate_on_call_exit_code: int = -15,
    ) -> None:
        self._poll_results = iter(poll_results)
        self._last_poll: int | None = None
        self._terminate_rc = terminate_on_call_exit_code
        self.returncode: int | None = None
        self.terminate_called = False
        self.kill_called = False
        self.wait_called_with_timeout: float | None = None

    def poll(self) -> int | None:
        with contextlib.suppress(StopIteration):
            self._last_poll = next(self._poll_results)
        if self._last_poll is not None:
            self.returncode = self._last_poll
        return self._last_poll

    def terminate(self) -> None:
        self.terminate_called = True
        # SIGTERM honoured: next poll will see the terminate exit code.
        self._last_poll = self._terminate_rc
        self.returncode = self._terminate_rc

    def wait(self, timeout: float | None = None) -> int:
        self.wait_called_with_timeout = timeout
        if self.returncode is None:
            # Unfinished + waited → behave as if the wait expired.
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout or 0.0)
        return self.returncode

    def kill(self) -> None:
        self.kill_called = True
        self.returncode = -9


class TestWaitWithFastFail(unittest.TestCase):
    """Recovery contract: a single rank crash must not block the launcher
    on the surviving rank's full torch.distributed rendezvous timeout
    (default 600 s — observed cumulative cost: ~3.7 h across 17 dev
    rounds in the 2026-05/06 sample).
    """

    def setUp(self) -> None:
        self.mod = _load_launcher_module()

    def test_all_success_returns_codes_and_terminates_nothing(self) -> None:
        # rank 0 finishes immediately with 0, rank 1 finishes one poll later.
        p0 = _FakeProc(poll_results=[0])
        p1 = _FakeProc(poll_results=[None, 0])
        with mock.patch("time.sleep") as sleep:
            codes = self.mod._wait_with_fast_fail([p0, p1], poll_interval_s=0.5)
        self.assertEqual(codes, [0, 0])
        self.assertFalse(p0.terminate_called)
        self.assertFalse(p1.terminate_called)
        # First poll showed not-all-done, so a single sleep happened
        # before the second poll cleared the loop.
        self.assertEqual(sleep.call_count, 1)
        self.assertEqual(sleep.call_args.args[0], 0.5)

    def test_rank0_crash_terminates_running_peer_immediately(self) -> None:
        # rank 0 crashes (exit 1) on the very first poll, rank 1 is still
        # running. The launcher MUST send SIGTERM to rank 1 instead of
        # waiting another ~600 s for its rendezvous timeout.
        p0 = _FakeProc(poll_results=[1])
        p1 = _FakeProc(poll_results=[None])  # still running when checked
        with mock.patch("time.sleep") as sleep:
            codes = self.mod._wait_with_fast_fail(
                [p0, p1], poll_interval_s=0.5, terminate_grace_s=5.0
            )
        # rank 0 keeps its 1; rank 1 is recorded as SIGTERM (-15).
        self.assertEqual(codes, [1, -15])
        self.assertFalse(p0.terminate_called)  # already dead
        self.assertTrue(p1.terminate_called)
        # Grace window honoured.
        self.assertEqual(p1.wait_called_with_timeout, 5.0)
        # No sleep needed — we fast-fail before the next poll cycle.
        self.assertEqual(sleep.call_count, 0)

    def test_default_terminate_grace_is_30s(self) -> None:
        # Locked-in: 30 s SIGTERM grace before SIGKILL. Short enough to
        # keep the fast-fail benefit (worst-case wait ~30 s vs 600 s
        # without the launcher hook) but generous enough for the child
        # to flush its traceback, tear down CUDA contexts, and release
        # NCCL handles cleanly.
        import inspect

        sig = inspect.signature(self.mod._wait_with_fast_fail)
        self.assertEqual(sig.parameters["terminate_grace_s"].default, 30.0)

    def test_terminate_grace_expired_promotes_to_kill(self) -> None:
        # A frozen peer ignores SIGTERM (wait() raises TimeoutExpired);
        # the launcher must escalate to SIGKILL rather than block.
        p0 = _FakeProc(poll_results=[1])
        # Stubborn rank 1: never sets returncode on terminate (force the
        # TimeoutExpired branch) — override terminate to be a no-op so
        # wait() then raises.
        p1 = _FakeProc(poll_results=[None])

        def _ignore_terminate() -> None:
            p1.terminate_called = True
            # Leave returncode as None so wait() raises TimeoutExpired.

        p1.terminate = _ignore_terminate  # type: ignore[method-assign]
        with mock.patch("time.sleep"):
            codes = self.mod._wait_with_fast_fail(
                [p0, p1], poll_interval_s=0.5, terminate_grace_s=0.1
            )
        self.assertEqual(codes, [1, -9])  # SIGKILL exit code
        self.assertTrue(p1.terminate_called)
        self.assertTrue(p1.kill_called)


if __name__ == "__main__":
    unittest.main()
