"""Disk-based registry SSOT for the agent-loop ↔ web bridge.

The old design pushed loop registration over HTTP to a hard-coded port
(``LOOP_WEB_PORT:-8421``). That coupling broke in two ways:

1. When the web server was offline, an externally-started agent-loop
   was completely invisible — the registration HTTP call failed and
   nothing was persisted, so a later web-server restart had nothing
   to recover.
2. When the web server ran on a non-default port, the CLI silently
   targeted port 8421 and the loop went unregistered.

The fix is to flip the data-flow direction. Each agent-loop now owns
its own ``.artifacts/forge_train/<loop_id>/session.json`` as the SSOT
record of its existence; the web server reads that directory and is
free to come and go. The HTTP ``/register`` endpoint is downgraded
to a "please reload this loop_id from disk *now*" hint and is no
longer the write path.

These tests pin the contract on both sides of the bridge.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
AGENT_LOOP_SH = REPO_ROOT / "harness" / "agent-loop.sh"


# ---------------------------------------------------------------------------
# Side A — agent-loop.sh writes session.json itself, with NO HTTP dependency.
# ---------------------------------------------------------------------------


class TestAgentLoopWritesSessionJson(unittest.TestCase):
    """``agent-loop.sh`` MUST publish its own ``session.json`` on disk.

    We invoke a small fragment of the shell script with bash so we
    exercise the actual code, not a Python re-implementation of it.
    The fragment is the registration block — we source the script
    with a guard that exits right after the registration step.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def _run_registration(
        self,
        *,
        loops_dir: Path,
        loop_id: str = "",
        web_id: str = "",
        log_dir: str | None = None,
        output_file: str | None = None,
        label: str = "2026-05-23 13:48:06",
        backend: str = "cursor-cli",
        model: str = "gpt-5.5-high",
        stages: tuple[str, ...] = ("stage1",),
        max_mode: str = "false",
        workspace: str | None = None,
    ) -> subprocess.CompletedProcess:
        """Run just the registration fragment of agent-loop.sh."""
        # Helper script: invokes the same logic the published
        # session-writer in agent-loop.sh uses, then exits before
        # any agent CLI is launched. We dispatch into agent-loop.sh
        # via an env-guarded short-circuit so the registration
        # logic stays in one place (the script itself).
        script = AGENT_LOOP_SH.read_text(encoding="utf-8")
        # Use the documented sentinel env LOOP_REGISTER_ONLY=1 to
        # short-circuit the script right after the registration
        # block runs. If the sentinel is absent, the implementation
        # may not have shipped yet, so the test fails loudly here
        # rather than silently spawning a real agent loop.
        self.assertIn(
            "LOOP_REGISTER_ONLY",
            script,
            "agent-loop.sh must support LOOP_REGISTER_ONLY=1 short-circuit "
            "so this test can exercise the registration block in isolation.",
        )

        env = {
            **os.environ,
            "LOOP_REGISTER_ONLY": "1",
            "FORGE_TRAIN_DIR": str(loops_dir),
            "LOOP_WEB_ID": web_id,
            "LOOP_OUTPUT_LOG": output_file or "",
            # Block any accidental HTTP attempt: route curl to a nonexistent host
            # by setting LOOP_WEB_PORT to 1 (privileged, will refuse). The disk
            # writer must not depend on curl succeeding.
            "LOOP_WEB_PORT": "1",
        }
        # Use a writable temp dir as CWD; bypass real workspace so
        # WORKSPACE just resolves to wherever agent-loop.sh lives.
        cmd = ["bash", str(AGENT_LOOP_SH), "--backend", backend, "--model", model]
        if max_mode == "true":
            cmd.append("--max-mode")
        cmd.extend(["--stages", ",".join(stages), "--runs-per-stage", "1"])
        if loop_id:
            env["LOOP_WEB_ID"] = loop_id
        return subprocess.run(
            cmd,
            env=env,
            cwd=workspace or str(self.tmp_path),
            capture_output=True,
            text=True,
            timeout=20,
        )

    def test_session_json_is_written_directly_to_disk(self) -> None:
        """When LOOP_REGISTER_ONLY=1, session.json appears in FORGE_TRAIN_DIR.

        The web server is intentionally *not* running. The registration
        must succeed anyway, proving the disk-write path does not
        depend on HTTP.
        """
        loops_dir = self.tmp_path / "forge_train"
        loops_dir.mkdir()
        result = self._run_registration(loops_dir=loops_dir)
        self.assertEqual(
            result.returncode,
            0,
            f"agent-loop.sh registration block must exit 0 even when "
            f"web server is down; got rc={result.returncode}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}",
        )

        # Exactly one session.json should exist (we did not pin loop_id,
        # so the script must have generated one).
        sessions = list(loops_dir.glob("*/session.json"))
        self.assertEqual(
            len(sessions),
            1,
            f"expected exactly one session.json under {loops_dir}, got {len(sessions)}: {sessions}",
        )

        data = json.loads(sessions[0].read_text(encoding="utf-8"))
        # Contract: required keys present, mode='external', loop_id
        # matches its directory, pid is a positive integer. Status
        # transitions through running -> {completed|failed|stopped}
        # over the script's lifetime; the precise terminal value is
        # covered by ``test_exit_trap_writes_final_status``.
        for key in (
            "loop_id",
            "label",
            "mode",
            "status",
            "pid",
            "started_at",
            "args",
            "workspace_dir",
        ):
            self.assertIn(key, data, f"session.json missing key '{key}': {data!r}")
        self.assertEqual(data["mode"], "external")
        self.assertIn(data["status"], {"running", "completed", "failed", "stopped"})
        self.assertEqual(data["loop_id"], sessions[0].parent.name)
        self.assertIsInstance(data["pid"], int)
        self.assertGreater(data["pid"], 0)

    def test_session_json_honors_explicit_loop_web_id(self) -> None:
        """If LOOP_WEB_ID is set (managed mode), the script reuses it."""
        loops_dir = self.tmp_path / "forge_train"
        loops_dir.mkdir()
        explicit = "preassigned123"
        result = self._run_registration(loops_dir=loops_dir, web_id=explicit)
        self.assertEqual(result.returncode, 0, result.stderr)

        session = loops_dir / explicit / "session.json"
        self.assertTrue(
            session.is_file(),
            f"session.json must land at the explicit loop_id directory "
            f"({session}); siblings: {list(loops_dir.iterdir())}",
        )
        data = json.loads(session.read_text(encoding="utf-8"))
        self.assertEqual(data["loop_id"], explicit)

    def test_loop_id_cli_flag_pins_session_directory(self) -> None:
        """``--loop-id <id>`` is the CLI counterpart to ``LOOP_WEB_ID``.

        Without the flag every invocation generates a fresh UUID, so
        a "restart" silently spins up a new loop instead of reusing
        the previous workspace. The flag exposes the resume semantics
        as a first-class CLI surface: a user-supplied id flows into
        the bootstrap block (workspace directory) AND into the
        session.json writer (loop_id field), so both halves agree.
        """
        loops_dir = self.tmp_path / "forge_train"
        loops_dir.mkdir()
        explicit = "clipinned123"

        # Build env WITHOUT LOOP_WEB_ID — the flag must be the sole
        # source of truth for the loop id in this scenario.
        env = {
            **os.environ,
            "LOOP_REGISTER_ONLY": "1",
            "FORGE_TRAIN_DIR": str(loops_dir),
            "LOOP_OUTPUT_LOG": "",
            "LOOP_WEB_PORT": "1",
        }
        env.pop("LOOP_WEB_ID", None)
        cmd = [
            "bash",
            str(AGENT_LOOP_SH),
            "--loop-id",
            explicit,
            "--backend",
            "cursor-cli",
            "--model",
            "gpt-5.5-high",
            "--stages",
            "stage1",
            "--runs-per-stage",
            "1",
        ]
        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(self.tmp_path),
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"--loop-id flag must not be rejected; got rc={result.returncode}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}",
        )

        session = loops_dir / explicit / "session.json"
        self.assertTrue(
            session.is_file(),
            f"--loop-id must pin the session directory to {explicit}; "
            f"got siblings: {[p.name for p in loops_dir.iterdir()]}",
        )
        data = json.loads(session.read_text(encoding="utf-8"))
        self.assertEqual(data["loop_id"], explicit)

    def test_loop_id_flag_preserves_existing_workspace(self) -> None:
        """Resuming an existing loop_id must NOT clobber its workspace.

        The bootstrap block exists to provision a fresh workspace
        when one does not yet exist. On resume the workspace already
        carries the agent's accumulated edits (``workload/src/...``);
        a blind rsync overwrites that work with the pristine
        ``harness/`` source. The contract on resume: keep the
        existing workspace intact.
        """
        import shutil

        loops_dir = self.tmp_path / "forge_train"
        loops_dir.mkdir()
        explicit = "resumekeep01"
        workspace = loops_dir / explicit / "workspace"

        # Pre-provision the workspace by mirroring what the bootstrap
        # block would do, then drop a sentinel file the agent might
        # have written. If the resume path correctly skips the copy,
        # the sentinel survives; if it re-rsyncs harness/ over it,
        # the sentinel disappears (no such file in source harness/).
        src_harness = AGENT_LOOP_SH.parent
        shutil.copytree(
            src_harness,
            workspace,
            ignore=shutil.ignore_patterns(".git", ".artifacts", "__pycache__", ".pytest_cache"),
        )
        sentinel = workspace / "AGENT_SCRATCH_RESUME_MARKER"
        sentinel.write_text("preserved across resume\n", encoding="utf-8")

        env = {
            **os.environ,
            "LOOP_REGISTER_ONLY": "1",
            "FORGE_TRAIN_DIR": str(loops_dir),
            "LOOP_OUTPUT_LOG": "",
            "LOOP_WEB_PORT": "1",
        }
        env.pop("LOOP_WEB_ID", None)
        cmd = [
            "bash",
            str(AGENT_LOOP_SH),
            "--loop-id",
            explicit,
            "--backend",
            "cursor-cli",
            "--model",
            "gpt-5.5-high",
            "--stages",
            "stage1",
            "--runs-per-stage",
            "1",
        ]
        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(self.tmp_path),
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        self.assertTrue(
            sentinel.is_file(),
            "Resume path must NOT re-rsync harness/ over an existing "
            f"workspace ({workspace}); sentinel file was clobbered.",
        )

    def test_exit_trap_writes_final_status(self) -> None:
        """After agent-loop.sh exits, session.json must show a
        terminal status (not 'running'), so the dashboard does not
        keep showing a dead loop as alive.

        LOOP_REGISTER_ONLY=1 exits with rc=0, which maps to
        ``status=completed``.
        """
        loops_dir = self.tmp_path / "forge_train"
        loops_dir.mkdir()
        result = self._run_registration(loops_dir=loops_dir)
        self.assertEqual(result.returncode, 0, result.stderr)

        sessions = list(loops_dir.glob("*/session.json"))
        self.assertEqual(len(sessions), 1)
        data = json.loads(sessions[0].read_text(encoding="utf-8"))
        self.assertEqual(
            data["status"],
            "completed",
            f"EXIT trap must publish a terminal status on rc=0; got {data['status']!r}",
        )
        self.assertEqual(data["exit_code"], 0)
        self.assertIsNotNone(data["ended_at"])


# ---------------------------------------------------------------------------
# Side B — backend reads disk and surfaces new loops without /register.
# ---------------------------------------------------------------------------


class _BackendHarness:
    """Isolate ``web.routers.loop`` so disk scanner tests stay hermetic."""

    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.loops_dir = Path(self._tmp.name) / "forge_train"
        self.loops_dir.mkdir()
        from web.routers import loop as loop_router

        self.loop_router = loop_router
        self._patches = [
            mock.patch.object(loop_router, "FORGE_TRAIN_DIR", self.loops_dir),
            mock.patch.dict(loop_router._instances, {}, clear=True),
        ]
        for p in self._patches:
            p.start()

    def close(self) -> None:
        for p in reversed(self._patches):
            p.stop()
        self._tmp.cleanup()

    def write_session(
        self,
        loop_id: str,
        *,
        status: str = "running",
        mode: str = "external",
        pid: int | None = None,
        idle_seconds: float = 0.0,
        args: dict | None = None,
    ) -> Path:
        """Drop a session.json on disk *as if* agent-loop.sh wrote it."""
        instance_dir = self.loops_dir / loop_id
        instance_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "loop_id": loop_id,
            "label": "external test",
            "mode": mode,
            "status": status,
            "pid": pid if pid is not None else os.getpid(),
            "started_at": time.time() - idle_seconds,
            "ended_at": None,
            "exit_code": None,
            "args": args or {"backend": "cursor-cli", "model": "gpt-5.5-high"},
            "workspace_dir": str(self.loops_dir.parent / "workspace"),
            "log_dir": None,
            "output_file": None,
        }
        session = instance_dir / "session.json"
        session.write_text(json.dumps(data), encoding="utf-8")
        if idle_seconds:
            past = time.time() - idle_seconds
            os.utime(session, (past, past))
        return session


class TestDiskScannerPicksUpNewSessions(unittest.TestCase):
    """The backend MUST have a single function that scans
    ``FORGE_TRAIN_DIR`` and brings new ``session.json`` entries into
    ``_instances`` without any HTTP call from the CLI.

    Contract: ``_scan_disk_registry()`` is the public name.
    """

    def setUp(self) -> None:
        self.h = _BackendHarness()
        self.addCleanup(self.h.close)

    def test_scanner_imports_new_session_into_instances(self) -> None:
        # session.json on disk, but _instances is empty (mimics "I started
        # a CLI loop, the web server was offline, it just came back up").
        self.h.write_session("aaaaaaaaaaaa", pid=os.getpid())
        self.assertNotIn("aaaaaaaaaaaa", self.h.loop_router._instances)

        scan = getattr(self.h.loop_router, "_scan_disk_registry", None)
        self.assertIsNotNone(
            scan,
            "web.routers.loop must expose _scan_disk_registry() as the "
            "disk-based registry entry point.",
        )

        new_ids = scan()
        self.assertIn("aaaaaaaaaaaa", new_ids)
        inst = self.h.loop_router._instances.get("aaaaaaaaaaaa")
        self.assertIsNotNone(inst)
        self.assertEqual(inst.status, "running")
        self.assertEqual(inst.mode, "external")
        self.assertEqual(inst.pid, os.getpid())

    def test_scanner_marks_dead_pid_as_stopped(self) -> None:
        """If session.json claims status=running but the PID is dead,
        the scanner downgrades it. Otherwise stale records would
        forever look "running" in the UI."""
        # Pick a PID we know is dead: a child we spawn-and-reap.
        # macOS ships `true` only at /usr/bin/true; Linux ships it at
        # /bin/true. Use sh -c so we don't depend on the path.
        zombie = subprocess.Popen(["/bin/sh", "-c", "exit 0"])
        zombie.wait()
        dead_pid = zombie.pid

        self.h.write_session("bbbbbbbbbbbb", pid=dead_pid)
        self.h.loop_router._scan_disk_registry()

        inst = self.h.loop_router._instances.get("bbbbbbbbbbbb")
        self.assertIsNotNone(inst)
        self.assertEqual(
            inst.status,
            "stopped",
            f"scanner must reconcile dead PID {dead_pid} -> stopped",
        )

    def test_scanner_is_idempotent(self) -> None:
        """Repeated scans must not double-import or churn version."""
        self.h.write_session("cccccccccccc", pid=os.getpid())
        first = self.h.loop_router._scan_disk_registry()
        second = self.h.loop_router._scan_disk_registry()
        self.assertIn("cccccccccccc", first)
        self.assertEqual(
            second,
            [],
            "second scan with no new files must report nothing new",
        )

    def test_scanner_skips_drafts(self) -> None:
        """Drafts have their own reaper; the disk scanner must not
        revive a draft into the visible loop list."""
        self.h.write_session("dddddddddddd", status="draft", pid=os.getpid())
        new_ids = self.h.loop_router._scan_disk_registry()
        # We do allow the scanner to *record* the draft (so the
        # /drafts surface still sees it), but it MUST NOT report
        # the draft loop_id as "new" — that's a loop-list signal.
        self.assertNotIn("dddddddddddd", new_ids)

    def test_scanner_reconciles_restarted_terminal_instance(self) -> None:
        """If an in-memory instance is terminal but disk shows running
        with a live PID, the scanner must revive it.

        This handles the case where /register failed (server was
        reloading) and the disk scanner is the fallback recovery path.
        """
        loop_id = "restartedlp1"
        my_pid = os.getpid()

        # Simulate: instance is terminal in memory (old run exited).
        stale = self.h.loop_router.LoopInstance(
            loop_id=loop_id,
            label="prev run",
            mode="external",
            pid=None,
            started_at=time.time() - 600,
            ended_at=time.time() - 60,
            exit_code=1,
            status="failed",
            args={"backend": "cursor-cli", "model": "gpt-5.5-high"},
            workspace_dir=str(self.h.loops_dir.parent / "workspace"),
        )
        self.h.loop_router._instances[loop_id] = stale

        # Disk says "running" with a live PID (new run started).
        self.h.write_session(loop_id, status="running", pid=my_pid)

        new_ids = self.h.loop_router._scan_disk_registry()
        self.assertIn(loop_id, new_ids)

        inst = self.h.loop_router._instances[loop_id]
        self.assertEqual(inst.status, "running")
        self.assertEqual(inst.pid, my_pid)
        self.assertIsNone(inst.ended_at)
        self.assertIsNone(inst.exit_code)

    def test_scanner_does_not_revive_with_dead_pid(self) -> None:
        """Scanner must NOT revive a terminal instance if the disk's
        PID is dead — that means the restart already exited too."""
        zombie = subprocess.Popen(["/bin/sh", "-c", "exit 0"])
        zombie.wait()
        dead_pid = zombie.pid
        loop_id = "deadrevive01"

        stale = self.h.loop_router.LoopInstance(
            loop_id=loop_id,
            label="prev run",
            mode="external",
            pid=None,
            status="stopped",
            args={},
            workspace_dir=str(self.h.loops_dir.parent / "workspace"),
        )
        self.h.loop_router._instances[loop_id] = stale

        # Disk says "running" but PID is dead.
        self.h.write_session(loop_id, status="running", pid=dead_pid)

        new_ids = self.h.loop_router._scan_disk_registry()
        self.assertNotIn(loop_id, new_ids)
        self.assertEqual(
            self.h.loop_router._instances[loop_id].status,
            "stopped",
            "scanner must not revive an instance whose disk PID is dead",
        )


class TestStartupAttachesRunningSessions(unittest.TestCase):
    """When the web server starts up, every running session on disk
    should be reattached so the user sees their CLI-started loops
    immediately, no /register call required.
    """

    def setUp(self) -> None:
        self.h = _BackendHarness()
        self.addCleanup(self.h.close)

    def test_load_history_recovers_running_cli_loop(self) -> None:
        # A CLI-started loop that wrote session.json before web came up.
        my_pid = os.getpid()
        self.h.write_session("eeeeeeeeeeee", mode="external", pid=my_pid)

        # load_history is the existing recovery entry point. After
        # this call the running CLI loop must appear in _instances.
        self.h.loop_router._load_history()

        inst = self.h.loop_router._instances.get("eeeeeeeeeeee")
        self.assertIsNotNone(inst)
        self.assertEqual(inst.mode, "external")
        self.assertEqual(inst.status, "running")
        self.assertEqual(inst.pid, my_pid)


# ---------------------------------------------------------------------------
# Side C — /register endpoint is downgraded to a hint, not a write path.
# ---------------------------------------------------------------------------


class TestBackgroundScannerTask(unittest.TestCase):
    """A long-running task on the server picks up disk-side
    registrations even when no /register HTTP call ever arrives.

    Contract: ``run_disk_registry_scanner(stop_event, interval=...)``
    is an awaitable that scans on a schedule until the event is set.
    """

    def setUp(self) -> None:
        self.h = _BackendHarness()
        self.addCleanup(self.h.close)

    def test_scanner_task_picks_up_session_written_mid_run(self) -> None:
        runner = getattr(self.h.loop_router, "run_disk_registry_scanner", None)
        self.assertIsNotNone(
            runner,
            "web.routers.loop must expose run_disk_registry_scanner(stop_event, "
            "interval=) so server.py can kick off the periodic scan.",
        )

        async def _scenario():
            stop = asyncio.Event()
            attached: list[str] = []

            async def _record_attach(inst):
                attached.append(inst.loop_id)

            with mock.patch.object(self.h.loop_router, "_attach_to_loop", _record_attach):
                task = asyncio.create_task(runner(stop, interval=0.05))
                # Mid-run: a CLI loop writes its session.json.
                await asyncio.sleep(0.02)
                self.h.write_session("rrrrrrrrrrrr", pid=os.getpid())
                # Give the scanner a couple of ticks to notice.
                await asyncio.sleep(0.2)
                stop.set()
                await asyncio.wait_for(task, timeout=1.0)
            return attached

        attached = asyncio.run(_scenario())
        self.assertIn("rrrrrrrrrrrr", attached)
        self.assertIn("rrrrrrrrrrrr", self.h.loop_router._instances)


class TestSessionTimestampCoercion(unittest.TestCase):
    """``started_at`` / ``ended_at`` MUST be read back as float|None.

    External recovery tooling has historically backfilled these as
    ISO-8601 strings. When such a record sits in ``_instances`` next to
    a normally-written float record, ``list_loops`` crashes its numeric
    sort with ``TypeError: '<' not supported between 'float' and 'str'``
    and the whole loop list 500s — every loop becomes invisible in the
    UI. The single read boundary must normalize to epoch seconds.
    """

    def setUp(self) -> None:
        self.h = _BackendHarness()
        self.addCleanup(self.h.close)

    def _write_raw(self, loop_id: str, started_at, ended_at=None) -> Path:
        instance_dir = self.h.loops_dir / loop_id
        instance_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "loop_id": loop_id,
            "label": "iso test",
            "mode": "external",
            "status": "stopped",
            "pid": None,
            "started_at": started_at,
            "ended_at": ended_at,
            "exit_code": 0,
            "args": {},
            "workspace_dir": str(self.h.loops_dir.parent / "workspace"),
        }
        session = instance_dir / "session.json"
        session.write_text(json.dumps(data), encoding="utf-8")
        return session

    def test_iso_string_started_at_is_coerced_to_float(self) -> None:
        path = self._write_raw(
            "isots0000001",
            started_at="2026-05-31T14:10:00+00:00",
            ended_at="2026-05-31T15:10:00+00:00",
        )
        inst = self.h.loop_router._read_session_json("isots0000001", path)
        self.assertIsNotNone(inst)
        self.assertIsInstance(inst.started_at, float)
        self.assertIsInstance(inst.ended_at, float)

    def test_float_started_at_passes_through(self) -> None:
        now = time.time()
        path = self._write_raw("floatts00001", started_at=now)
        inst = self.h.loop_router._read_session_json("floatts00001", path)
        self.assertIsNotNone(inst)
        self.assertEqual(inst.started_at, now)
        self.assertIsNone(inst.ended_at)

    def test_list_loops_sort_survives_mixed_timestamp_types(self) -> None:
        """A float-typed and an ISO-string-typed record together must
        not crash the loop list sort once both are read in."""
        self._write_raw("mixedfloat01", started_at=time.time())
        self._write_raw("mixediso0001", started_at="2026-05-31T14:10:00+00:00")
        self.h.loop_router._load_history()

        snap = asyncio.run(self.h.loop_router.list_loops())
        ids = {item["loop_id"] for item in snap["loops"]}
        self.assertIn("mixedfloat01", ids)
        self.assertIn("mixediso0001", ids)


class TestRegisterEndpointIsReloadHint(unittest.TestCase):
    """``POST /api/loop/register`` becomes "please rescan now" rather
    than the only write path. Two consequences must hold:

    1. The endpoint succeeds even if ``session.json`` does not yet
       exist on disk (so we don't fail closed in races where the
       CLI's curl arrives before its disk write commits).

    2. After a successful register call, calling list/get returns
       the same loop that would have been picked up by the next
       scheduled disk scan.

    We treat the existing in-process write fallback inside
    ``register_loop`` as a transitional behavior; new code MUST
    prefer the disk-first path.
    """

    def setUp(self) -> None:
        self.h = _BackendHarness()
        self.addCleanup(self.h.close)

    def test_register_reloads_session_written_by_shell(self) -> None:
        """End-to-end: shell-side has already written session.json.
        The /register call comes in just to nudge the server; the
        server must surface what the shell put on disk."""
        my_pid = os.getpid()
        self.h.write_session("ffffffffffff", mode="external", pid=my_pid)

        req = self.h.loop_router.LoopRegisterRequest(
            loop_id="ffffffffffff",
            pid=my_pid,
            workspace=str(self.h.loops_dir.parent / "workspace"),
        )

        async def _bypass_attach(inst):
            # Disable the real tail/monitor wiring so this stays a unit
            # test. The shape is what we're testing here, not the
            # asyncio task plumbing.
            return None

        with mock.patch.object(self.h.loop_router, "_attach_to_loop", _bypass_attach):
            snap = asyncio.run(self.h.loop_router.register_loop(req, x_loop_source="agent-loop"))

        self.assertEqual(snap["loop_id"], "ffffffffffff")
        self.assertEqual(snap["mode"], "external")
        self.assertEqual(snap["status"], "running")
        self.assertIn("ffffffffffff", self.h.loop_router._instances)

    def test_register_revives_terminal_instance_on_external_restart(self) -> None:
        """External restart of the same loop_id MUST flip status back
        to ``running`` and re-attach the PID monitor.

        Scenario: a previous run exited (``stopped``/``failed``/
        ``completed``), so ``_instances`` holds a terminal row with
        a stale pid + exit_code + ended_at. The user re-runs
        ``agent-loop.sh`` with ``LOOP_WEB_ID=<same id>``. The shell:

        1. Writes a fresh ``session.json`` with ``status="running"``
           and the new live PID.
        2. POSTs ``/register`` as a "rescan now" hint.

        The register handler must NOT cling to the in-memory terminal
        snapshot. Required post-conditions:

        * In-memory ``inst.status == "running"``.
        * In-memory ``inst.pid`` is the new live PID.
        * Terminal residue (``ended_at``, ``exit_code``) is cleared.
        * ``_attach_to_loop`` is invoked so the new PID's exit will
          transition the row again (otherwise the row sticks at
          ``running`` forever once the second run dies).
        * The disk SSOT is NOT clobbered back to the stale terminal
          status (regression guard against the merge writing the
          old in-memory row over the fresh shell write).
        """
        loop_id = "revivedloop1"
        my_pid = os.getpid()

        # Step 1: simulate the previous run's terminal residue in
        # _instances. This mirrors what _monitor_exit would have
        # written after a SIGTERM (rc=-15 -> status=stopped).
        stale = self.h.loop_router.LoopInstance(
            loop_id=loop_id,
            label="prev run",
            mode="external",
            pid=None,
            started_at=time.time() - 600,
            ended_at=time.time() - 60,
            exit_code=-15,
            status="stopped",
            args={"backend": "cursor-cli", "model": "gpt-5.5-high"},
            workspace_dir=str(self.h.loops_dir.parent / "workspace"),
        )
        self.h.loop_router._instances[loop_id] = stale

        # Step 2: the shell of the restarted run writes its fresh
        # session.json with the new live PID.
        self.h.write_session(loop_id, mode="external", pid=my_pid)

        req = self.h.loop_router.LoopRegisterRequest(
            loop_id=loop_id,
            pid=my_pid,
            workspace=str(self.h.loops_dir.parent / "workspace"),
        )

        attached: list[str] = []

        async def _record_attach(inst):
            attached.append(inst.loop_id)

        with mock.patch.object(self.h.loop_router, "_attach_to_loop", _record_attach):
            snap = asyncio.run(self.h.loop_router.register_loop(req, x_loop_source="agent-loop"))

        self.assertEqual(snap["status"], "running")
        self.assertEqual(snap["pid"], my_pid)
        self.assertIsNone(snap["ended_at"])
        self.assertIsNone(snap["exit_code"])
        self.assertIn(
            loop_id,
            attached,
            "register_loop MUST re-attach the PID monitor when a "
            "terminal row is revived; otherwise the second run's "
            "exit never updates the status.",
        )

        # Disk SSOT regression guard: the fresh "running" record the
        # shell wrote must not be overwritten back to "stopped".
        on_disk = json.loads(
            (self.h.loops_dir / loop_id / "session.json").read_text(encoding="utf-8")
        )
        self.assertEqual(on_disk["status"], "running")
        self.assertEqual(on_disk["pid"], my_pid)
        self.assertIsNone(on_disk["ended_at"])
        self.assertIsNone(on_disk["exit_code"])

    def test_register_revives_when_disk_clobbered_by_old_monitor(self) -> None:
        """Register MUST promote to running even when disk was clobbered.

        Race scenario: old _monitor_exit writes "stopped" to disk
        AFTER the new run's session.json write but BEFORE /register
        arrives. The register handler must trust the live PID over
        the stale disk state.
        """
        loop_id = "clobbered001"
        my_pid = os.getpid()

        # In-memory state: old run is terminal.
        stale = self.h.loop_router.LoopInstance(
            loop_id=loop_id,
            label="prev run",
            mode="external",
            pid=None,
            started_at=time.time() - 600,
            ended_at=time.time() - 60,
            exit_code=-15,
            status="stopped",
            args={"backend": "cursor-cli", "model": "gpt-5.5-high"},
            workspace_dir=str(self.h.loops_dir.parent / "workspace"),
        )
        self.h.loop_router._instances[loop_id] = stale

        # Disk state: simulates old monitor clobbering the restart's
        # "running" write back to "stopped" (the race condition).
        self.h.write_session(loop_id, status="stopped", pid=my_pid)

        req = self.h.loop_router.LoopRegisterRequest(
            loop_id=loop_id,
            pid=my_pid,
            workspace=str(self.h.loops_dir.parent / "workspace"),
        )

        attached: list[str] = []

        async def _record_attach(inst):
            attached.append(inst.loop_id)

        with mock.patch.object(self.h.loop_router, "_attach_to_loop", _record_attach):
            snap = asyncio.run(self.h.loop_router.register_loop(req, x_loop_source="agent-loop"))

        self.assertEqual(
            snap["status"],
            "running",
            "register must trust the live PID over clobbered disk",
        )
        self.assertEqual(snap["pid"], my_pid)
        self.assertIn(loop_id, attached)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
