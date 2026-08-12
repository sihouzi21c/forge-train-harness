"""Unit tests for the wall-clock timeout guarding the ``cctl`` shell-out.

``release()`` runs from the wrapper's EXIT trap and shells out to
``cctl devspace stop``. When the Teleport tunnel behind ``cctl`` hangs,
an un-bounded ``subprocess.run`` blocks the trap forever (observed in
loop ``9e2e20a5e638``: a single ``cctl devspace stop`` pinned the
wrapper for 5.5h). The shared ``cctl_common.run_cctl`` must therefore
pass an explicit ``timeout`` and translate ``TimeoutExpired`` into a
``CctlError``; ``lease._run_cctl`` re-raises that as ``LeaseError`` so
the caller's ``contextlib.suppress(LeaseError)`` can unwind cleanly.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools import cctl_common, lease  # noqa: E402


class RunCctlTimeoutTest(unittest.TestCase):
    def test_run_cctl_passes_timeout(self) -> None:
        with mock.patch.object(cctl_common.subprocess, "run") as run:
            run.return_value = subprocess.CompletedProcess(
                args=["cctl"], returncode=0, stdout="", stderr=""
            )
            cctl_common.run_cctl(["devspace", "stop", "tasks/1"])
        self.assertEqual(run.call_args.kwargs.get("timeout"), cctl_common.CCTL_CALL_TIMEOUT_S)

    def test_run_cctl_timeout_raises_cctl_error(self) -> None:
        with mock.patch.object(cctl_common.subprocess, "run") as run:
            run.side_effect = subprocess.TimeoutExpired(
                cmd=["cctl", "devspace", "stop"], timeout=cctl_common.CCTL_CALL_TIMEOUT_S
            )
            with self.assertRaisesRegex(cctl_common.CctlError, "timed out"):
                cctl_common.run_cctl(["devspace", "stop", "tasks/1"])

    def test_lease_run_cctl_translates_to_lease_error(self) -> None:
        # The lease wrapper preserves the original ``LeaseError`` contract
        # its EXIT-trap ``suppress(LeaseError)`` depends on.
        with mock.patch.object(cctl_common, "run_cctl", side_effect=cctl_common.CctlError("boom")):
            with self.assertRaises(lease.LeaseError):
                lease._run_cctl(["devspace", "stop", "tasks/1"])


if __name__ == "__main__":
    unittest.main()
