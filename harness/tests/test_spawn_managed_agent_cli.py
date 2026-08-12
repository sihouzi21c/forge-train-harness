"""Out-of-process CLI helpers for spawning frontend-displayable agents.

These tests pin the contract between ``agent-loop.sh`` and the two
Python CLIs it shells out to:

* ``harness/tools/spawn_managed_agent.py`` — synchronous wrapper that
  starts a backend CLI subprocess, registers it as a Session under
  ``.artifacts/web-agents/<id>/``, pumps stdout/stderr to disk, and
  exits with the CLI's exit code. Prints ``AGENT_ID=<id>`` on its own
  stdout so the shell can capture the assigned id.

* ``harness/tools/loop_wrapper_init.py`` — initializes the synthetic
  ``loop-<loop_id>`` Session entry for a loop's own orchestration
  events.

We exercise both with a fake ``agent`` binary that emits a single
``system.init`` event then exits 0 (or non-zero for failure paths),
so the test exercises real subprocess + asyncio + disk-write code
without any cursor-cli dependency.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SPAWN_SCRIPT = REPO_ROOT / "harness" / "tools" / "spawn_managed_agent.py"
WRAPPER_INIT_SCRIPT = REPO_ROOT / "harness" / "tools" / "loop_wrapper_init.py"


def _write_fake_agent_binary(
    tmpdir: Path,
    *,
    session_id: str = "fake-sess-1",
    exit_code: int = 0,
    extra_events: tuple[dict, ...] = (),
    linger_seconds: float = 0.0,
) -> Path:
    """Create a fake ``agent`` executable that emits one stream-json
    init line then optional extra events then exits.

    When ``linger_seconds > 0`` the binary sleeps that long after
    emitting its events before exiting, with the default (no custom
    handler) SIGTERM disposition — so a SIGTERM delivered to its
    process group terminates it, exactly like the real backend CLI.

    Returns the binary's directory (caller puts it on PATH).
    """
    bin_dir = tmpdir / "fake-bin"
    bin_dir.mkdir()
    binary = bin_dir / "agent"
    py_inline = textwrap.dedent(f"""\
        #!{sys.executable}
        import json, sys, time
        sys.stdout.write(json.dumps({{
            "type": "system",
            "subtype": "init",
            "session_id": "{session_id}",
            "model": "fake",
            "cwd": "/tmp",
            "permissionMode": "trust",
        }}) + "\\n")
        sys.stdout.flush()
        for event in {list(extra_events)!r}:
            sys.stdout.write(json.dumps(event) + "\\n")
            sys.stdout.flush()
        if {linger_seconds!r} > 0:
            time.sleep({linger_seconds!r})
        sys.exit({exit_code})
        """)
    binary.write_text(py_inline, encoding="utf-8")
    binary.chmod(0o755)
    return bin_dir


def _run_spawn_cli(
    *,
    tmpdir: Path,
    bin_dir: Path,
    args: list[str],
    agents_dir: Path,
) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "PYTHONPATH": f"{REPO_ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
        "FORGE_AGENTS_DIR": str(agents_dir),
    }
    return subprocess.run(
        [sys.executable, str(SPAWN_SCRIPT), *args],
        env=env,
        cwd=str(tmpdir),
        capture_output=True,
        text=True,
        timeout=30,
    )


class TestSpawnManagedAgentCli(unittest.TestCase):
    """``spawn_managed_agent.py`` is the SSOT spawn surface for agent-loop.sh."""

    def test_cli_writes_session_json_and_stdout_log_and_prints_agent_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bin_dir = _write_fake_agent_binary(tmp_path)
            agents_dir = tmp_path / "web-agents"

            prompt_file = tmp_path / "prompt.txt"
            prompt_file.write_text("hello there", encoding="utf-8")

            result = _run_spawn_cli(
                tmpdir=tmp_path,
                bin_dir=bin_dir,
                agents_dir=agents_dir,
                args=[
                    "--backend",
                    "cursor-cli",
                    "--model",
                    "fake",
                    "--workspace",
                    str(tmp_path),
                    "--prompt-file",
                    str(prompt_file),
                    "--kind",
                    "loop_dev_round",
                    "--loop-id",
                    "abc123",
                    "--parent-agent",
                    "loop-abc123",
                ],
            )
            self.assertEqual(
                result.returncode,
                0,
                f"helper must propagate CLI rc=0; got {result.returncode}\n"
                f"stdout={result.stdout!r}\nstderr={result.stderr!r}",
            )
            # AGENT_ID line is the contract; capture it.
            agent_id_line = next(
                (line for line in result.stdout.splitlines() if line.startswith("AGENT_ID=")),
                None,
            )
            self.assertIsNotNone(
                agent_id_line,
                f"helper must print AGENT_ID=<id> on its stdout; got {result.stdout!r}",
            )
            agent_id = agent_id_line.split("=", 1)[1].strip()
            self.assertTrue(agent_id.startswith("web-"), agent_id)

            sess_dir = agents_dir / agent_id
            self.assertTrue((sess_dir / "session.json").is_file())
            data = json.loads((sess_dir / "session.json").read_text(encoding="utf-8"))
            self.assertEqual(data["state"], "completed")
            self.assertEqual(data["kind"], "loop_dev_round")
            self.assertEqual(data["loop_id"], "abc123")
            self.assertEqual(data["parent_agent_id"], "loop-abc123")
            self.assertEqual(data["backend"], "cursor-cli")
            self.assertEqual(data["exit_code"], 0)

            stdout_log = sess_dir / "stdout.log"
            self.assertTrue(stdout_log.is_file())
            first_line = stdout_log.read_text(encoding="utf-8").splitlines()[0]
            init_event = json.loads(first_line)
            self.assertEqual(init_event["type"], "system")
            self.assertEqual(init_event["subtype"], "init")

    def test_cli_with_stage_round_seq_nests_under_wrapper(self) -> None:
        """With --stage/--round/--seq the child id is the ordered composite
        and its session.json lands at the nested forest path
        loop-<id>/agents/<stage>/r<RRR>/<full-id>/."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bin_dir = _write_fake_agent_binary(tmp_path)
            agents_dir = tmp_path / "web-agents"

            prompt_file = tmp_path / "prompt.txt"
            prompt_file.write_text("dev round", encoding="utf-8")

            result = _run_spawn_cli(
                tmpdir=tmp_path,
                bin_dir=bin_dir,
                agents_dir=agents_dir,
                args=[
                    "--backend",
                    "cursor-cli",
                    "--model",
                    "fake",
                    "--workspace",
                    str(tmp_path),
                    "--prompt-file",
                    str(prompt_file),
                    "--kind",
                    "loop_dev_round",
                    "--loop-id",
                    "abc123",
                    "--parent-agent",
                    "loop-abc123",
                    "--stage",
                    "stage1",
                    "--round",
                    "4",
                    "--seq",
                    "1",
                ],
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            agent_id_line = next(
                (line for line in result.stdout.splitlines() if line.startswith("AGENT_ID=")),
                None,
            )
            self.assertIsNotNone(agent_id_line)
            agent_id = agent_id_line.split("=", 1)[1].strip()
            self.assertTrue(agent_id.startswith("loop-abc123_stage1_r004_01_web-"), agent_id)

            sess_json = (
                agents_dir / "loop-abc123" / "agents" / "stage1" / "r004" / agent_id / "session.json"
            )
            self.assertTrue(sess_json.is_file(), f"expected nested session.json at {sess_json}")
            data = json.loads(sess_json.read_text(encoding="utf-8"))
            self.assertEqual(data["kind"], "loop_dev_round")
            self.assertEqual(data["loop_id"], "abc123")

    def test_loop_round_session_uses_kind_label_not_system_prompt(self) -> None:
        """``prompt_preview`` must NOT be the first 200 chars of the giant
        loop system prompt — otherwise every loop dev round in the
        sidebar / chat header shows the same identical boilerplate
        ("You are an autonomous developer agent operating inside a
        workspace that exposes a `harness` CLI ..."), which is what
        the user saw as "abnormal display" on the Agents tab.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bin_dir = _write_fake_agent_binary(tmp_path)
            agents_dir = tmp_path / "web-agents"

            giant_system_prompt = (
                "You are an autonomous developer agent operating inside a "
                "workspace that exposes a `harness` CLI as the canonical "
                "command surface. " * 50
            )
            prompt_file = tmp_path / "prompt.txt"
            prompt_file.write_text(giant_system_prompt, encoding="utf-8")

            result = _run_spawn_cli(
                tmpdir=tmp_path,
                bin_dir=bin_dir,
                agents_dir=agents_dir,
                args=[
                    "--backend",
                    "cursor-cli",
                    "--model",
                    "fake",
                    "--workspace",
                    str(tmp_path),
                    "--prompt-file",
                    str(prompt_file),
                    "--kind",
                    "loop_dev_round",
                    "--loop-id",
                    "453a9bf813e7",
                    "--parent-agent",
                    "loop-453a9bf813e7",
                ],
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            agent_id_line = next(
                (line for line in result.stdout.splitlines() if line.startswith("AGENT_ID=")),
                None,
            )
            self.assertIsNotNone(agent_id_line)
            agent_id = agent_id_line.split("=", 1)[1].strip()
            data = json.loads(
                (agents_dir / agent_id / "session.json").read_text(encoding="utf-8"),
            )
            preview = data["prompt_preview"]
            # Title must not start with the system-prompt boilerplate.
            self.assertNotIn("You are an autonomous developer agent", preview)
            # Title must identify the kind and pin a short loop-id tag
            # so the sidebar row is distinguishable across runs.
            self.assertIn("Loop dev round", preview)
            self.assertIn("453a9bf8", preview)
            # And it must fit on a single header line (we previously
            # capped at 200 chars; the new label is far shorter).
            self.assertLessEqual(len(preview), 200)

    def test_chat_session_still_uses_user_message_as_preview(self) -> None:
        """The user-typed ``chat`` flow MUST keep its existing behavior:
        the first 200 chars of the user's literal message are the title.
        Regressing this would erase every chat row's actual content."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bin_dir = _write_fake_agent_binary(tmp_path)
            agents_dir = tmp_path / "web-agents"

            prompt_file = tmp_path / "prompt.txt"
            prompt_file.write_text("写一篇 2000 字的鲸鱼龙文章", encoding="utf-8")

            result = _run_spawn_cli(
                tmpdir=tmp_path,
                bin_dir=bin_dir,
                agents_dir=agents_dir,
                args=[
                    "--backend",
                    "cursor-cli",
                    "--model",
                    "fake",
                    "--workspace",
                    str(tmp_path),
                    "--prompt-file",
                    str(prompt_file),
                    # default kind=chat
                ],
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            agent_id_line = next(
                (line for line in result.stdout.splitlines() if line.startswith("AGENT_ID=")),
                None,
            )
            self.assertIsNotNone(agent_id_line)
            agent_id = agent_id_line.split("=", 1)[1].strip()
            data = json.loads(
                (agents_dir / agent_id / "session.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(data["prompt_preview"], "写一篇 2000 字的鲸鱼龙文章")

    def test_spawn_child_event_written_to_wrapper_log(self) -> None:
        """When --loop-id is provided, the helper must write a spawn_child
        event to the wrapper's stdout.log so /active and Follow Live can
        discover the child while it is still running."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bin_dir = _write_fake_agent_binary(tmp_path)
            agents_dir = tmp_path / "web-agents"

            prompt_file = tmp_path / "prompt.txt"
            prompt_file.write_text("do something", encoding="utf-8")

            result = _run_spawn_cli(
                tmpdir=tmp_path,
                bin_dir=bin_dir,
                agents_dir=agents_dir,
                args=[
                    "--backend",
                    "cursor-cli",
                    "--model",
                    "fake",
                    "--workspace",
                    str(tmp_path),
                    "--prompt-file",
                    str(prompt_file),
                    "--kind",
                    "loop_dev_round",
                    "--loop-id",
                    "myloop42",
                    "--parent-agent",
                    "loop-myloop42",
                ],
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            agent_id_line = next(
                (line for line in result.stdout.splitlines() if line.startswith("AGENT_ID=")),
                None,
            )
            self.assertIsNotNone(agent_id_line)
            agent_id = agent_id_line.split("=", 1)[1].strip()

            wrapper_log = agents_dir / "loop-myloop42" / "stdout.log"
            self.assertTrue(wrapper_log.is_file(), "wrapper stdout.log must exist")
            records = [
                json.loads(line)
                for line in wrapper_log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            spawn_events = [
                r
                for r in records
                if r.get("type") == "loop_event" and r.get("subtype") == "spawn_child"
            ]
            self.assertEqual(len(spawn_events), 1, f"expected 1 spawn_child; got {spawn_events}")
            self.assertEqual(spawn_events[0]["agent_id"], agent_id)
            self.assertEqual(spawn_events[0]["kind"], "loop_dev_round")
            self.assertIn("ts", spawn_events[0])

    def test_no_spawn_child_event_without_loop_id(self) -> None:
        """Plain chat sessions (no --loop-id) must NOT write a spawn_child
        event; the wrapper stdout.log belongs to loop orchestration."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bin_dir = _write_fake_agent_binary(tmp_path)
            agents_dir = tmp_path / "web-agents"

            prompt_file = tmp_path / "prompt.txt"
            prompt_file.write_text("hello", encoding="utf-8")

            result = _run_spawn_cli(
                tmpdir=tmp_path,
                bin_dir=bin_dir,
                agents_dir=agents_dir,
                args=[
                    "--backend",
                    "cursor-cli",
                    "--model",
                    "fake",
                    "--workspace",
                    str(tmp_path),
                    "--prompt-file",
                    str(prompt_file),
                ],
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            wrapper_dir = agents_dir / "loop-None"
            self.assertFalse(
                wrapper_dir.exists(),
                "no wrapper dir should be created for a non-loop spawn",
            )

    def test_cli_propagates_nonzero_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bin_dir = _write_fake_agent_binary(tmp_path, exit_code=2)
            agents_dir = tmp_path / "web-agents"

            prompt_file = tmp_path / "prompt.txt"
            prompt_file.write_text("err", encoding="utf-8")
            result = _run_spawn_cli(
                tmpdir=tmp_path,
                bin_dir=bin_dir,
                agents_dir=agents_dir,
                args=[
                    "--backend",
                    "cursor-cli",
                    "--model",
                    "fake",
                    "--workspace",
                    str(tmp_path),
                    "--prompt-file",
                    str(prompt_file),
                ],
            )
            self.assertEqual(
                result.returncode,
                2,
                f"helper must mirror backend rc=2 so agent-loop.sh's "
                f"transient-failure detection still works; got rc={result.returncode}\n"
                f"stdout={result.stdout!r}\nstderr={result.stderr!r}",
            )
            agent_id_line = next(
                (line for line in result.stdout.splitlines() if line.startswith("AGENT_ID=")),
                None,
            )
            self.assertIsNotNone(agent_id_line)
            agent_id = agent_id_line.split("=", 1)[1].strip()
            data = json.loads(
                (agents_dir / agent_id / "session.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(data["state"], "failed")
            self.assertEqual(data["exit_code"], 2)


class TestSpawnManagedAgentSignalForwarding(unittest.TestCase):
    """When ``agent-loop.sh`` is stopped, the /stop endpoint SIGTERMs the
    bash process group. ``spawn_managed_agent.py`` is in that group; the
    backend CLI it spawned is NOT (it gets its own session via
    ``start_new_session``). The helper MUST forward the signal to the
    CLI's group so its own ``monitor_exit`` runs and writes the terminal
    state — otherwise the child session.json stays ``state=running`` with
    a leaked-but-alive CLI pid forever, showing "responding" in the UI.
    """

    def test_sigterm_kills_child_and_marks_interrupted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bin_dir = _write_fake_agent_binary(tmp_path, linger_seconds=60)
            agents_dir = tmp_path / "web-agents"

            prompt_file = tmp_path / "prompt.txt"
            prompt_file.write_text("long running round", encoding="utf-8")

            env = {
                **os.environ,
                "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
                "PYTHONPATH": f"{REPO_ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
                "FORGE_AGENTS_DIR": str(agents_dir),
            }
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(SPAWN_SCRIPT),
                    "--backend",
                    "cursor-cli",
                    "--model",
                    "fake",
                    "--workspace",
                    str(tmp_path),
                    "--prompt-file",
                    str(prompt_file),
                    "--kind",
                    "loop_dev_round",
                    "--loop-id",
                    "stoploop1",
                    "--parent-agent",
                    "loop-stoploop1",
                ],
                env=env,
                cwd=str(tmp_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Block until the helper prints AGENT_ID (init has landed and
            # session.json + the child CLI exist).
            agent_id = None
            assert proc.stdout is not None
            for _ in range(200):
                line = proc.stdout.readline()
                if not line:
                    break
                if line.startswith("AGENT_ID="):
                    agent_id = line.split("=", 1)[1].strip()
                    break
            self.assertIsNotNone(
                agent_id,
                "helper must print AGENT_ID before we can stop it",
            )
            assert agent_id is not None

            sess_file = agents_dir / agent_id / "session.json"
            self.assertTrue(sess_file.is_file())
            running = json.loads(sess_file.read_text(encoding="utf-8"))
            self.assertEqual(running["state"], "running")
            child_pid = running["pid"]
            self.assertIsInstance(child_pid, int)
            self.assertTrue(_pid_alive(child_pid), "child CLI must be alive pre-stop")

            # The actual stop: SIGTERM the helper, mirroring /stop's group
            # signal. The helper must forward to the child and exit cleanly.
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                self.fail("helper did not exit within 30s of SIGTERM")

            # Child CLI must be dead — not leaked as an orphan.
            self.assertFalse(
                _pid_alive(child_pid),
                f"child CLI pid {child_pid} leaked after helper SIGTERM",
            )

            # And its session must be reconciled to a terminal state by the
            # helper's own monitor_exit (NOT by a lazy read-time sweep).
            final = json.loads(sess_file.read_text(encoding="utf-8"))
            self.assertEqual(final["state"], "interrupted")
            self.assertIsNone(final["pid"])
            self.assertIsNotNone(final["ended_at"])


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class TestLoopWrapperInitCli(unittest.TestCase):
    """``loop_wrapper_init.py`` creates the loop-<id> Session entry on disk."""

    def test_wrapper_init_writes_session_json_and_initial_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            agents_dir = tmp_path / "web-agents"
            env = {
                **os.environ,
                "PYTHONPATH": f"{REPO_ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
                "FORGE_AGENTS_DIR": str(agents_dir),
            }
            result = subprocess.run(
                [
                    sys.executable,
                    str(WRAPPER_INIT_SCRIPT),
                    "--loop-id",
                    "abc123def456",
                    "--workspace",
                    str(tmp_path),
                    "--backend",
                    "cursor-cli",
                    "--model",
                    "opus",
                    "--stages",
                    "stage1 stage2",
                    "--pid",
                    str(os.getpid()),
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            wrapper_dir = agents_dir / "loop-abc123def456"
            self.assertTrue((wrapper_dir / "session.json").is_file())
            data = json.loads((wrapper_dir / "session.json").read_text(encoding="utf-8"))
            self.assertEqual(data["agent_id"], "loop-abc123def456")
            self.assertEqual(data["backend"], "loop-wrapper")
            self.assertEqual(data["kind"], "loop_wrapper")
            self.assertEqual(data["loop_id"], "abc123def456")
            self.assertEqual(data["state"], "running")
            self.assertEqual(data["pid"], os.getpid())

            stdout_log = wrapper_dir / "stdout.log"
            self.assertTrue(stdout_log.is_file())
            lines = stdout_log.read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(lines), 1)
            init_event = json.loads(lines[0])
            self.assertEqual(init_event["type"], "system")
            self.assertEqual(init_event["subtype"], "init")
            self.assertEqual(init_event["session_id"], "loop-abc123def456")


class TestLoopEventAppender(unittest.TestCase):
    """Wrapper-side orchestration text is one of:

      python -m harness.tools.loop_wrapper_event \\
          --loop-id <id> --subtype <subtype> [--payload '<json>']

    Each call appends one ``loop_event`` NDJSON line to the wrapper's
    stdout.log. This is the only writer agent-loop.sh uses to emit
    stage/round/spawn_child/verdict/info markers, so the frontend
    receives one well-typed stream and never has to grep wrapper echos
    out of mixed bash + LLM output.
    """

    def test_event_appender_writes_typed_line(self) -> None:
        EVENT_SCRIPT = REPO_ROOT / "harness" / "tools" / "loop_wrapper_event.py"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            agents_dir = tmp_path / "web-agents"
            env = {
                **os.environ,
                "PYTHONPATH": f"{REPO_ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
                "FORGE_AGENTS_DIR": str(agents_dir),
            }
            init = subprocess.run(
                [
                    sys.executable,
                    str(WRAPPER_INIT_SCRIPT),
                    "--loop-id",
                    "abcdef",
                    "--workspace",
                    str(tmp_path),
                    "--backend",
                    "cursor-cli",
                    "--model",
                    "opus",
                    "--stages",
                    "stage1",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(init.returncode, 0, init.stderr)

            evt = subprocess.run(
                [
                    sys.executable,
                    str(EVENT_SCRIPT),
                    "--loop-id",
                    "abcdef",
                    "--subtype",
                    "round_start",
                    "--payload",
                    '{"stage":"stage1","round":1}',
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(evt.returncode, 0, evt.stderr)

            stdout_log = agents_dir / "loop-abcdef" / "stdout.log"
            lines = stdout_log.read_text(encoding="utf-8").splitlines()
            # First line was init; second should be our round_start.
            self.assertGreaterEqual(len(lines), 2)
            record = json.loads(lines[1])
            self.assertEqual(record["type"], "loop_event")
            self.assertEqual(record["subtype"], "round_start")
            self.assertEqual(record["stage"], "stage1")
            self.assertEqual(record["round"], 1)
            self.assertIn("ts", record)


if __name__ == "__main__":
    unittest.main()
