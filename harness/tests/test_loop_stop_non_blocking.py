"""``POST /api/loop/{loop_id}/stop`` must NOT block the event loop.

The previous implementation walked the process tree by recursively
forking ``pgrep -P <pid>`` inside an ``async def`` handler. Each fork
took up to 3 s (``timeout=3``) of synchronous wall time, multiplied
by the depth of the agent-loop's descendant tree (bash → spawn helper
→ cursor agent → python workers …). While ``stop_loop`` was running,
every other tab's HTTP poll was frozen — so the user perceived a
multi-second site-wide freeze, not just a slow Stop button.

The managed-agent stop link (``web/agents/runner.stop``) does the
right thing already: one non-blocking ``os.killpg`` syscall against
the subprocess's process group. These tests pin the same contract on
the loop stop link:

1. ``_collect_descendant_pids`` runs at most one process snapshot
   (``ps`` / ``pgrep``) per invocation, not O(tree-depth) recursive
   forks.
2. ``stop_loop`` reaches the bash process group via ``os.killpg``
   first, so it terminates in one syscall instead of walking pids
   one by one.
3. While ``stop_loop`` is in flight, an independently-scheduled
   coroutine still makes progress — the asyncio event loop is not
   starved by blocking syscalls.
"""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


class TestLoopStopIsNonBlocking(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

        # Isolate the loop registry root so the test never touches
        # the developer's real .artifacts directory.
        import os

        self._prev_forge = os.environ.get("FORGE_TRAIN_DIR")
        os.environ["FORGE_TRAIN_DIR"] = self._tmp.name
        self.addCleanup(
            lambda: (
                os.environ.pop("FORGE_TRAIN_DIR", None)
                if self._prev_forge is None
                else os.environ.__setitem__("FORGE_TRAIN_DIR", self._prev_forge)
            )
        )

        from web.routers import loop as loop_router

        # Reset the in-memory registry to keep tests independent.
        loop_router._instances.clear()
        self.loop_router = loop_router

    def _seed_running_loop(self, loop_id: str, pid: int) -> None:
        inst = self.loop_router.LoopInstance(
            loop_id=loop_id,
            label="test",
            status="running",
            pid=pid,
            started_at=time.time(),
            workspace_dir=str(Path(self._tmp.name) / "ws"),
        )
        self.loop_router._instances[loop_id] = inst
        self.loop_router._instance_dir(loop_id).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # (1) Single process-snapshot per stop, no recursive fork-storm
    # ------------------------------------------------------------------
    def test_collect_descendants_uses_single_snapshot(self) -> None:
        """``_collect_descendant_pids`` MUST NOT fan out recursive ``pgrep``
        forks. A single ``ps``/``pgrep`` invocation is enough to know the
        whole parent→children map; recursion at the syscall layer is what
        blocks the event loop for seconds when the tree is deep."""

        # Fake a 4-level deep descendant tree: 100 → 101 → 102 → 103.
        # The legacy recursive impl would invoke pgrep four times (one
        # per level). The fixed impl must run exactly one snapshot.
        fake_tree = {
            "100": "101\n",
            "101": "102\n",
            "102": "103\n",
            "103": "",
        }
        call_count = {"n": 0}

        def fake_check_output(cmd, *args, **kwargs):
            call_count["n"] += 1
            # Accept both legacy ``pgrep -P <pid>`` and the single-shot
            # ``ps -A -o pid=,ppid=`` form. The contract is "one syscall",
            # not "use pgrep specifically".
            if cmd[:1] == ["ps"]:
                return (
                    "  100     1\n"
                    "  101   100\n"
                    "  102   101\n"
                    "  103   102\n"
                    "  999     1\n"  # unrelated process, must NOT be collected
                )
            if cmd[:2] == ["pgrep", "-P"]:
                return fake_tree.get(cmd[2], "")
            raise AssertionError(f"unexpected command: {cmd!r}")

        with mock.patch.object(subprocess, "check_output", side_effect=fake_check_output):
            descendants = self.loop_router._collect_descendant_pids(100)

        self.assertEqual(set(descendants), {101, 102, 103})
        self.assertEqual(
            call_count["n"],
            1,
            f"expected exactly one process snapshot, got {call_count['n']} "
            "— recursive subprocess fork blocks the event loop",
        )

    # ------------------------------------------------------------------
    # (2) Use killpg on the process group, not per-pid os.kill
    # ------------------------------------------------------------------
    def test_kill_tree_targets_process_group_first(self) -> None:
        """``_kill_tree`` MUST send the signal to the bash session's pgid
        in one ``os.killpg`` syscall before doing any per-pid sweep. The
        bash wrapper is launched with ``start_new_session=True`` so it
        IS the group leader; one syscall hits the whole group."""
        import signal as signal_mod

        killpg_calls: list[tuple[int, int]] = []

        def fake_getpgid(pid: int) -> int:
            return pid  # bash is its own group leader

        def fake_killpg(pgid: int, sig: int) -> None:
            killpg_calls.append((pgid, sig))

        # Empty descendant set keeps the test focused on the pgid path.
        with (
            mock.patch.object(self.loop_router, "_collect_descendant_pids", return_value=[]),
            mock.patch("os.getpgid", side_effect=fake_getpgid),
            mock.patch("os.killpg", side_effect=fake_killpg),
            mock.patch("os.kill"),
        ):
            self.loop_router._kill_tree(4242, signal_mod.SIGTERM)

        self.assertTrue(
            killpg_calls,
            "expected at least one os.killpg call — without it stop_loop "
            "falls back to per-pid os.kill and misses descendants",
        )
        self.assertEqual(killpg_calls[0], (4242, signal_mod.SIGTERM))

    # ------------------------------------------------------------------
    # (3) The stop handler does not starve the event loop
    # ------------------------------------------------------------------
    def test_stop_loop_does_not_starve_event_loop(self) -> None:
        """While ``stop_loop`` is running, a concurrent coroutine MUST
        still make progress. The legacy impl ran a recursive blocking
        ``subprocess.check_output`` directly inside the async handler,
        which froze every other in-flight request."""

        async def _run() -> tuple[float, bool]:
            self._seed_running_loop("aaaabbbb", pid=4242)

            # Simulate a tree walk that takes ~0.5 s of blocking work.
            # The fix moves this to ``asyncio.to_thread``; the legacy
            # impl runs it on the event loop and starves the heartbeat.
            sleep_chunk = 0.5

            def slow_check_output(cmd, *args, **kwargs):
                time.sleep(sleep_chunk)
                if cmd[:1] == ["ps"]:
                    return ""
                return ""

            heartbeat_ticks = {"n": 0}

            async def _heartbeat() -> None:
                # Ticks every 50 ms. If the event loop is starved for
                # 500 ms, we'd expect ~10 ticks; we tolerate >=3 to
                # avoid flakiness on slow CI but still fail hard if the
                # legacy "0 ticks" behavior comes back.
                while True:
                    heartbeat_ticks["n"] += 1
                    await asyncio.sleep(0.05)

            hb = asyncio.create_task(_heartbeat())

            with (
                mock.patch.object(subprocess, "check_output", side_effect=slow_check_output),
                mock.patch.object(self.loop_router, "_pid_alive", return_value=False),
                mock.patch("os.killpg"),
                mock.patch("os.getpgid", return_value=4242),
                mock.patch("os.kill"),
            ):
                t0 = time.monotonic()
                await self.loop_router.stop_loop("aaaabbbb")
                elapsed = time.monotonic() - t0

            hb.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await hb

            return elapsed, heartbeat_ticks["n"] >= 3

        elapsed, heartbeat_alive = asyncio.run(_run())
        self.assertTrue(
            heartbeat_alive,
            f"event loop was starved during stop_loop (elapsed={elapsed:.2f}s); "
            "heartbeat coroutine got fewer than 3 ticks. "
            "The blocking syscalls must be moved off the event loop "
            "via asyncio.to_thread.",
        )


class TestLoopStopVisibleStoppingState(unittest.TestCase):
    """The Stop button MUST give the same instant ``Stopping...`` visual
    feedback the managed-agent Stop button already has (see
    ``test_agent_stop_enters_visible_stopping_state`` in
    ``test_web_theme.py``). Without it the user clicks once, gets zero
    visual change, and concludes the dashboard hangs."""

    def setUp(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        self.index_html = (repo_root / "web" / "static" / "index.html").read_text(encoding="utf-8")
        self.app_js = (repo_root / "web" / "static" / "js" / "app.js").read_text(encoding="utf-8")

    def test_loop_stop_button_shows_stopping_label(self) -> None:
        # Button text flips to "Stopping..." while the request is in flight,
        # mirroring the managed-agent Stop button.
        self.assertIn(">Stopping...</span>", self.index_html)

    def test_loop_stop_button_disabled_while_stopping(self) -> None:
        # Frontend must disable the Stop button as soon as it's clicked so
        # the user can't double-fire and so the spinner state is observable.
        self.assertIn("isLoopStopping(selectedLoopId)", self.index_html)

    def test_app_js_tracks_loop_stopping_ids(self) -> None:
        # Per-loop "is stopping" map, marked/cleared by stopLoop(), so the
        # Loops list and the detail header agree on the same source of truth.
        self.assertIn("loopStoppingIds: {}", self.app_js)
        self.assertIn("isLoopStopping", self.app_js)


if __name__ == "__main__":
    unittest.main()
