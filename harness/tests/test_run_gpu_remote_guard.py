"""Fail-fast guard for GPU suites launched on a CUDA-less remote-configured host.

When ``[remote].kind`` is ``ssh`` / ``devspace`` the GPU suites are meant
to run on the remote host via ``tools/remote_run.sh``. If an agent instead
runs ``bin/harness run <suite>`` locally on a Mac / CPU box, the local
transport would dispatch the suite and fail deep inside torch with an
opaque CUDA error. ``app._assert_runnable_locally`` turns that into an
actionable redirect — without auto-SSHing (local/remote separation is
intentional; see prompt/develop_prompt/remote-execution.md).
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["FORGE_REPO_ROOT"] = str(REPO_ROOT)

from harness import app, transport  # noqa: E402


class TestAssertRunnableLocally(unittest.TestCase):
    def test_devspace_without_local_cuda_raises_with_remote_run_redirect(self) -> None:
        with mock.patch.object(app, "_local_cuda_available", return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                app._assert_runnable_locally({"remote": {"kind": "devspace"}}, "forward-align")
        msg = str(ctx.exception)
        self.assertIn("tools/remote_run.sh", msg)
        self.assertIn("forward-align", msg)
        self.assertIn("devspace", msg)

    def test_ssh_without_local_cuda_raises(self) -> None:
        with mock.patch.object(app, "_local_cuda_available", return_value=False):
            with self.assertRaises(RuntimeError):
                app._assert_runnable_locally({"remote": {"kind": "ssh"}}, "multistep")

    def test_local_kind_never_guards(self) -> None:
        # kind = "local" means run everything on this host; the guard must
        # be a no-op even when no GPU is present.
        with mock.patch.object(app, "_local_cuda_available", return_value=False):
            app._assert_runnable_locally({"remote": {"kind": "local"}}, "forward-align")
            app._assert_runnable_locally({}, "forward-align")

    def test_remote_kind_with_local_cuda_present_passes(self) -> None:
        # On the remote GPU host the synced config still says kind=ssh, but
        # nvidia-smi is present — the guard must let the suite run there.
        with mock.patch.object(app, "_local_cuda_available", return_value=True):
            app._assert_runnable_locally({"remote": {"kind": "devspace"}}, "forward-align")


class TestRunGpuSuiteCallsGuardFirst(unittest.TestCase):
    def test_guard_runs_before_transport_creation(self) -> None:
        # The guard must fire before any transport / config work so the
        # caller gets the redirect instead of an opaque downstream failure.
        with (
            mock.patch.object(
                app,
                "_assert_runnable_locally",
                side_effect=RuntimeError("guard fired"),
            ),
            mock.patch.object(transport, "create_transport") as create,
        ):
            with self.assertRaisesRegex(RuntimeError, "guard fired"):
                app._run_gpu_suite(
                    suite="forward-align",
                    suite_args=[],
                    report="text",
                    harness_config={},
                    workload_config={"remote": {"kind": "devspace"}},
                    gpu=None,
                )
        create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
