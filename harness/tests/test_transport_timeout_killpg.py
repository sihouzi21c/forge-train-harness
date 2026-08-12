"""Hard wall-clock contract for ``transport._run(timeout=…)``.

Plan §E.1 — without this test the killpg defense is paper. The fixture
spawns a child sleep that detaches from its parent's terminal session,
then sleeps itself. Without ``start_new_session=True`` + ``killpg``,
SIGTERM to the parent leaves the child alive — burning the post-timeout
budget the dispatcher's outer kill is supposed to bound.
"""

from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["FORGE_REPO_ROOT"] = str(REPO_ROOT)

from harness import transport  # noqa: E402


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class TestRunNoTimeoutBackwardsCompatible(unittest.TestCase):
    """``timeout=None`` is the only path used by ``doctor`` /
    ``nvidia-smi`` callers; it MUST remain a transparent ``subprocess.run``
    wrapper so behaviour for those paths does not regress."""

    def test_zero_returncode_passthrough(self) -> None:
        completed = transport._run(["true"], check=False)
        self.assertEqual(completed.returncode, 0)

    def test_nonzero_returncode_with_check_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Command failed"):
            transport._run(["false"], check=True)

    def test_stdout_captured(self) -> None:
        completed = transport._run(["echo", "hello"], check=False)
        self.assertEqual(completed.stdout.strip(), "hello")


class TestRunWithTimeoutSuccess(unittest.TestCase):
    def test_returns_when_command_completes_within_timeout(self) -> None:
        # ``true`` exits instantly; the 5 s budget is luxurious.
        completed = transport._run(["true"], check=False, timeout=5.0)
        self.assertEqual(completed.returncode, 0)


class TestRunWithTimeoutKillsProcessGroup(unittest.TestCase):
    """Defence against the original incident: GNU ``timeout`` only kills
    the direct child by default, leaving any backgrounded grandchild
    sleeping. ``_run(timeout=…)`` MUST kill the whole process group."""

    def test_grandchild_dies_after_timeout(self) -> None:
        # Parent shell forks a backgrounded sleep (the "grandchild") that
        # writes its PID to a tmp file, then itself sleeps long enough
        # to outlive the timeout. With killpg, the grandchild dies; with
        # only SIGTERM-to-direct-child, the grandchild survives.
        marker = REPO_ROOT / ".artifacts" / "test-transport-pgid.pid"
        marker.parent.mkdir(parents=True, exist_ok=True)
        if marker.exists():
            marker.unlink()
        script = f"sleep 60 & echo $! > {marker}; wait"

        start = time.monotonic()
        with self.assertRaises(transport.TransportTimeoutError) as ctx:
            transport._run(["bash", "-c", script], check=False, timeout=1.5)
        elapsed = time.monotonic() - start

        # SIGKILL fires within a generous bound (timeout + 30 s grace).
        self.assertLess(elapsed, 35.0)

        # Grandchild PID was recorded — wait a short window for the
        # killpg to propagate to it.
        self.assertTrue(marker.exists(), "marker file not written by fixture")
        try:
            grandchild_pid = int(marker.read_text().strip())
        finally:
            marker.unlink(missing_ok=True)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if not _pid_alive(grandchild_pid):
                break
            time.sleep(0.1)
        self.assertFalse(
            _pid_alive(grandchild_pid),
            f"grandchild {grandchild_pid} survived killpg — process group containment is broken",
        )

        # Returncode on the parent is the SIGKILL signal — confirms the
        # grace period elapsed and the second-stage SIGKILL fired (or
        # SIGTERM was honoured immediately; either way negative).
        self.assertLess(ctx.exception.returncode or 0, 0)


class TestTransportTimeoutErrorCarriesBuffers(unittest.TestCase):
    def test_buffers_attached(self) -> None:
        script = "echo first-line; sleep 60"
        with self.assertRaises(transport.TransportTimeoutError) as ctx:
            transport._run(["bash", "-c", script], check=False, timeout=1.0)
        # ``communicate`` returns whatever was buffered before the kill —
        # the early ``echo`` must be preserved so the upper layer can
        # surface it in result.json.
        self.assertIn("first-line", ctx.exception.stdout)
        self.assertEqual(ctx.exception.timeout_s, 1.0)


if __name__ == "__main__":
    unittest.main()
