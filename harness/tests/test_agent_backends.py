"""Tests for managed-agent backend separation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from web import auth
from web.agents import messages, store


class TestManagedAgentBackends(unittest.TestCase):
    def test_backend_registry_separates_cursor_and_claude_commands(self) -> None:
        from web.agents import backends

        names = {backend["name"] for backend in backends.list_backends()}
        self.assertIn("cursor-cli", names)
        self.assertIn("claude-code", names)
        self.assertIn("codex", names)

        cursor = backends.get_backend("cursor-cli")
        cursor_cmd = cursor.build_command(
            model="gpt-5.5-high",
            prompt="hello",
            workspace="/tmp/workspace",
            resume_session_id="cursor-session",
            max_mode=True,
        )
        self.assertIn("cursor_max_mode.py", " ".join(cursor_cmd))
        self.assertIn("agent", cursor_cmd)
        self.assertIn("--workspace", cursor_cmd)
        self.assertIn("--resume", cursor_cmd)
        self.assertIn(
            "--stream-partial-output",
            cursor_cmd,
            "cursor-cli web sessions must request partial streaming so the UI "
            "receives assistant text before the backend emits a full snapshot",
        )
        self.assertNotIn("claude", cursor_cmd)

        claude = backends.get_backend("claude-code")
        claude_cmd = claude.build_command(
            model="CLAUDE_aeqq93",
            prompt="hello",
            workspace="/tmp/workspace",
            resume_session_id="claude-session",
            max_mode=True,
            effort="max",
        )
        self.assertEqual(claude_cmd[0], "claude")
        self.assertIn("-p", claude_cmd)
        self.assertIn("--output-format", claude_cmd)
        self.assertIn("stream-json", claude_cmd)
        self.assertIn("--include-partial-messages", claude_cmd)
        self.assertIn("--resume", claude_cmd)
        self.assertIn("claude-session", claude_cmd)
        # Model selection is re-enabled for claude-code: a non-empty model
        # slug is forwarded as ``--model`` so profiles can pin a
        # proxy-routed id (e.g. ``claudecode-fable.toml``).
        self.assertIn("--model", claude_cmd)
        self.assertIn("CLAUDE_aeqq93", claude_cmd)
        # ``effort`` is the claude-only reasoning knob; ``max_mode`` is
        # cursor-only and must NOT leak into the claude invocation.
        self.assertIn("--effort", claude_cmd)
        self.assertIn("max", claude_cmd)
        self.assertNotIn("cursor_max_mode.py", " ".join(claude_cmd))
        # thinking_display defaults to empty => flag omitted entirely.
        self.assertNotIn("--thinking-display", " ".join(claude_cmd))

        claude_cmd_thinking = claude.build_command(
            model="claude-opus-4-8[1m]",
            prompt="hello",
            workspace="/tmp/workspace",
            resume_session_id=None,
            max_mode=False,
            effort="high",
            thinking_display="summarized",
        )
        # Opus 4.8 logs no chain-of-thought unless the request carries
        # `display: summarized`; the backend forwards the knob in the
        # single-token `=` form (variadic-safe, see --disallowedTools).
        self.assertIn("--thinking-display=summarized", claude_cmd_thinking)
        # The flag must precede the positional prompt argument.
        self.assertLess(
            claude_cmd_thinking.index("--thinking-display=summarized"),
            claude_cmd_thinking.index("hello"),
        )

        codex = backends.get_backend("codex")
        with mock.patch.dict("os.environ", {"CODEX_MODEL_PROVIDER": "llm-center"}, clear=True):
            codex_cmd = codex.build_command(
                model="o3",
                prompt="hello",
                workspace="/tmp/workspace",
                resume_session_id="codex-session",
                max_mode=True,
                effort="max",
            )
        self.assertEqual(codex_cmd[:3], ["codex", "exec", "resume"])
        self.assertIn("--json", codex_cmd)
        self.assertNotIn("--model", codex_cmd)
        self.assertNotIn("o3", codex_cmd)
        self.assertNotIn("-C", codex_cmd)
        self.assertNotIn("/tmp/workspace", codex_cmd)
        self.assertIn("model_reasoning_effort=high", codex_cmd)
        self.assertEqual(codex_cmd[-2], "codex-session")
        self.assertEqual(codex_cmd[-1], "hello")

    def test_codex_command_resolves_available_provider_model_when_config_default_missing(
        self,
    ) -> None:
        from web.agents import backends

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_exc) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "data": [
                            {"id": "GPT_5eqkpn", "modelName": "gpt-5.5"},
                            {"id": "GPT_ookipq", "modelName": "gpt-5.3-codex"},
                        ]
                    }
                ).encode("utf-8")

        def fake_urlopen(_request, timeout=0):
            return FakeResponse()

        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.toml").write_text(
                "\n".join(
                    [
                        'model = "GPT_missing"',
                        'model_provider = "lm_center"',
                        "",
                        "[model_providers.lm_center]",
                        'base_url = "https://llm.example/v1"',
                        'env_key = "LLM_CENTER_API_KEY"',
                    ]
                ),
                encoding="utf-8",
            )
            codex = backends.get_backend("codex")
            with (
                mock.patch.dict(
                    "os.environ",
                    {
                        "CODEX_HOME": tmp,
                        "LLM_CENTER_API_KEY": "sk-test",  # pragma: allowlist secret
                    },
                    clear=True,
                ),
                mock.patch("urllib.request.urlopen", fake_urlopen),
            ):
                codex_cmd = codex.build_command(
                    model="o3",
                    prompt="hello",
                    workspace="/tmp/workspace",
                    resume_session_id=None,
                    max_mode=True,
                )

        self.assertIn("-m", codex_cmd)
        self.assertIn("GPT_5eqkpn", codex_cmd)
        self.assertNotIn("o3", codex_cmd)
        self.assertNotIn("GPT_missing", codex_cmd)

    def test_claude_command_omits_model_and_effort_when_unset(self) -> None:
        """Empty model/effort must omit both flags so the CLI uses its own
        defaults (the default ``claudecode-default.toml`` profile)."""
        from web.agents import backends

        claude = backends.get_backend("claude-code")
        claude_cmd = claude.build_command(
            model="",
            prompt="hello",
            workspace="/tmp/workspace",
            resume_session_id=None,
            max_mode=True,
            effort="",
        )
        self.assertNotIn("--model", claude_cmd)
        self.assertNotIn("--effort", claude_cmd)
        # max_mode=True is a cursor concept; it must not add --effort here.
        self.assertNotIn("cursor_max_mode.py", " ".join(claude_cmd))

    def test_claude_command_disallows_async_wakeup_tools(self) -> None:
        """A one-shot ``claude -p`` round has no persistent session for an
        async "re-invoke me later" contract to fire into, so every self-wake /
        background-notification tool must be disabled at the CLI layer — a peer
        of the existing EnterPlanMode/ExitPlanMode block. Otherwise the agent
        yields mid-job expecting a wake-up that never arrives and orphans the
        in-flight GPU job: observed in loop ec285c274f96 M6 rounds 26/40/42,
        where the dev exited on "the monitor will notify me" / "scheduled a
        fallback; the background job will re-invoke me" — falling back to
        Monitor/ScheduleWakeup the moment ``run_in_background`` was hook-blocked.
        ``run_in_background`` itself is a Bash *parameter* (not a tool), so it
        stays on the PreToolUse hook; these are standalone tools and belong in
        ``--disallowedTools``.
        """
        from web.agents import backends

        claude = backends.get_backend("claude-code")
        cmd = claude.build_command(
            model="",
            prompt="hello",
            workspace="/tmp/workspace",
            resume_session_id=None,
            max_mode=False,
            effort="",
        )
        disallow = next(
            (t.split("=", 1)[1] for t in cmd if t.startswith("--disallowedTools=")),
            "",
        )
        blocked = set(disallow.split(","))
        for tool in ("Monitor", "ScheduleWakeup", "CronCreate"):
            self.assertIn(tool, blocked, f"{tool} (async self-wake) must be disallowed")
        # the pre-existing plan-mode block must be preserved
        self.assertIn("EnterPlanMode", blocked)
        self.assertIn("ExitPlanMode", blocked)

    def test_claude_code_build_env_does_not_force_proxy_base_url(self) -> None:
        """claude-code must NOT inject the llm-center proxy base URL.

        Forcing ``ANTHROPIC_BASE_URL`` onto the internal proxy bypassed the
        operator's claude.ai OAuth subscription and 401'd whenever no valid
        proxy key was provisioned. With the proxy default removed, an unset
        ``ANTHROPIC_BASE_URL`` must stay unset so the Claude Code CLI falls
        back to its logged-in OAuth credentials.
        """
        from web.agents import backends

        claude = backends.get_backend("claude-code")
        with mock.patch.dict(
            "os.environ", {}, clear=True
        ):  # no ANTHROPIC_BASE_URL in the environment
            env = claude.build_env(api_key=None)
        self.assertNotIn(
            "ANTHROPIC_BASE_URL",
            env,
            "claude-code must not force a proxy base URL; an unset value "
            "must remain unset so the CLI uses its OAuth login",
        )
        self.assertNotIn(
            "ANTHROPIC_API_KEY",
            env,
            "no api_key resolved -> ANTHROPIC_API_KEY must stay unset",
        )

    def test_claude_code_build_env_preserves_explicit_base_url(self) -> None:
        """An operator who explicitly exports ``ANTHROPIC_BASE_URL`` (e.g. to
        opt back into a proxy) must have it passed through untouched."""
        from web.agents import backends

        claude = backends.get_backend("claude-code")
        with mock.patch.dict(
            "os.environ",
            {"ANTHROPIC_BASE_URL": "https://example.test/llm"},
            clear=True,
        ):
            env = claude.build_env(api_key="sk-test")
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "https://example.test/llm")
        self.assertEqual(env["ANTHROPIC_API_KEY"], "sk-test")

    def test_codex_build_env_uses_openai_api_key(self) -> None:
        from web.agents import backends

        with tempfile.TemporaryDirectory() as tmp:
            codex = backends.get_backend("codex")
            with mock.patch.dict(
                "os.environ",
                {"CODEX_HOME": tmp, "OPENAI_BASE_URL": "https://example.test/v1"},
                clear=True,
            ):
                env = codex.build_env(api_key="sk-test")
        self.assertEqual(env["OPENAI_API_KEY"], "sk-test")
        self.assertEqual(env["OPENAI_BASE_URL"], "https://example.test/v1")

    def test_codex_build_env_uses_active_provider_env_key(self) -> None:
        from web.agents import backends

        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.toml").write_text(
                "\n".join(
                    [
                        'model_provider = "lm_center"',
                        "",
                        "[model_providers.lm_center]",
                        'env_key = "LLM_CENTER_API_KEY"',
                    ]
                ),
                encoding="utf-8",
            )
            codex = backends.get_backend("codex")
            with mock.patch.dict("os.environ", {"CODEX_HOME": tmp}, clear=True):
                env = codex.build_env(api_key="sk-test")
        self.assertEqual(env["LLM_CENTER_API_KEY"], "sk-test")
        self.assertNotIn("OPENAI_API_KEY", env)

    def test_claude_model_catalog_is_empty_model_selection_is_disabled(self) -> None:
        """Model selection for claude-code is intentionally disabled: the
        Claude Code CLI's own default model is the only supported choice,
        so ``list_models`` returns an empty catalogue (see
        ``backends._build_claude_command``)."""
        from web.agents import runner

        models = __import__("asyncio").run(runner.list_models(backend="claude-code"))
        self.assertEqual(models, [])

    def test_codex_model_catalog_is_empty_model_selection_is_disabled(self) -> None:
        from web.agents import runner

        models = __import__("asyncio").run(runner.list_models(backend="codex"))
        self.assertEqual(models, [])

    def test_codex_transcript_events_render_text_and_tools(self) -> None:
        rendered = messages.rebuild(
            [
                {
                    "type": "web_user",
                    "message": {"content": [{"type": "text", "text": "hello"}]},
                },
                {"type": "response.output_text.delta", "delta": "Hi"},
                {
                    "type": "item.started",
                    "item": {
                        "type": "command_execution",
                        "id": "tool-1",
                        "command": "pwd",
                    },
                },
                {
                    "type": "item.completed",
                    "item": {
                        "type": "command_execution",
                        "id": "tool-1",
                        "aggregated_output": "/tmp\n",
                    },
                },
                {"type": "response.completed"},
            ]
        )

        self.assertEqual(rendered[0]["role"], "user")
        self.assertEqual(rendered[1]["role"], "assistant")
        self.assertIn("Hi", rendered[1]["content"])
        self.assertEqual(rendered[1]["_toolCalls"][0]["name"], "shell")
        self.assertEqual(rendered[1]["_toolCalls"][0]["result"], "/tmp\n")

    def test_session_persists_backend_and_backend_session_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_dir = store.AGENTS_DIR
            store.AGENTS_DIR = Path(tmp)
            try:
                sess = store.Session(
                    agent_id="web-agent",
                    backend="claude-code",
                    backend_session_id="claude-session",
                    state=store.STATE_RUNNING,
                    model="sonnet",
                    workspace="/tmp/workspace",
                    created_at=store.now_iso(),
                )
                store.save(sess)

                loaded = store.load("web-agent")
                self.assertIsNotNone(loaded)
                assert loaded is not None
                self.assertEqual(loaded.agent_id, "web-agent")
                self.assertEqual(loaded.backend, "claude-code")
                self.assertEqual(loaded.backend_session_id, "claude-session")
            finally:
                store.AGENTS_DIR = old_dir

    def test_zombie_session_with_dead_pid_reconciles_to_interrupted_on_load(self) -> None:
        """Sessions spawned by an external owner (agent-loop.sh ->
        spawn_managed_agent.py) and synthetic loop wrappers persist
        ``state=running`` until the owner's monitor_exit fires. If the
        owner is SIGKILL'd / OOM-killed / loses its container before
        EXIT runs, ``session.json`` stays at ``state=running`` with a
        dead PID forever and the Agents/Loop list keeps a permanent
        "responding" ghost row.

        ``store.load()`` must reconcile this lazily on read — no
        periodic disk scanner — and persist the fix so subsequent
        reads don't re-probe the dead pid.
        """
        import os

        # Find a PID that is guaranteed not to be alive. Forking + waiting
        # gives a deterministic dead-pid value on this OS.
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        os.waitpid(pid, 0)
        # ``pid`` is now a reaped, definitely-dead PID we can stamp into
        # the test session.

        with tempfile.TemporaryDirectory() as tmp:
            old_dir = store.AGENTS_DIR
            store.AGENTS_DIR = Path(tmp)
            try:
                sess = store.Session(
                    agent_id="web-zombie",
                    state=store.STATE_RUNNING,
                    model="claude-opus-4-7-thinking-high",
                    workspace="/tmp/workspace",
                    created_at=store.now_iso(),
                    backend="cursor-cli",
                    pid=pid,
                    parent_agent_id="loop-deadbeef",
                    loop_id="deadbeef",
                    kind="loop_dev_round",
                )
                store.save(sess)

                # Sanity check: the on-disk row says running until load
                # touches it.
                import json

                raw = json.loads(
                    (Path(tmp) / "web-zombie" / "session.json").read_text(),
                )
                self.assertEqual(raw["state"], store.STATE_RUNNING)
                self.assertEqual(raw["pid"], pid)

                loaded = store.load("web-zombie")
                self.assertIsNotNone(loaded)
                assert loaded is not None
                self.assertEqual(loaded.state, store.STATE_INTERRUPTED)
                self.assertIsNone(loaded.pid)
                self.assertIsNotNone(
                    loaded.ended_at,
                    "reconciled zombies must record an ended_at timestamp",
                )
                self.assertEqual(loaded.exit_code, -1)

                # And the fix must persist so the next read sees the
                # reconciled state without re-probing the dead pid.
                persisted = json.loads(
                    (Path(tmp) / "web-zombie" / "session.json").read_text(),
                )
                self.assertEqual(persisted["state"], store.STATE_INTERRUPTED)
                self.assertIsNone(persisted["pid"])
                self.assertEqual(persisted["exit_code"], -1)
            finally:
                store.AGENTS_DIR = old_dir

    def test_load_does_not_reconcile_when_pid_is_alive(self) -> None:
        """A running session whose recorded PID is still alive must
        stay ``state=running`` — reconciling it would surface a
        false-positive zombie and falsely mark an active agent as
        interrupted."""
        import os

        with tempfile.TemporaryDirectory() as tmp:
            old_dir = store.AGENTS_DIR
            store.AGENTS_DIR = Path(tmp)
            try:
                sess = store.Session(
                    agent_id="web-alive",
                    state=store.STATE_RUNNING,
                    model="opus",
                    workspace="/tmp/workspace",
                    created_at=store.now_iso(),
                    backend="cursor-cli",
                    pid=os.getpid(),  # our own pid is definitely alive
                )
                store.save(sess)
                loaded = store.load("web-alive")
                self.assertIsNotNone(loaded)
                assert loaded is not None
                self.assertEqual(loaded.state, store.STATE_RUNNING)
                self.assertEqual(loaded.pid, os.getpid())
            finally:
                store.AGENTS_DIR = old_dir

    def test_load_does_not_reconcile_when_pid_is_missing(self) -> None:
        """Sessions persisted without a tracked pid (e.g. the initial
        Session row written before the subprocess actually launches)
        must not be reconciled — we have no signal to make any
        terminal decision from."""
        with tempfile.TemporaryDirectory() as tmp:
            old_dir = store.AGENTS_DIR
            store.AGENTS_DIR = Path(tmp)
            try:
                sess = store.Session(
                    agent_id="web-pending",
                    state=store.STATE_RUNNING,
                    model="opus",
                    workspace="/tmp/workspace",
                    created_at=store.now_iso(),
                    backend="cursor-cli",
                    pid=None,
                )
                store.save(sess)
                loaded = store.load("web-pending")
                self.assertIsNotNone(loaded)
                assert loaded is not None
                self.assertEqual(loaded.state, store.STATE_RUNNING)
                self.assertIsNone(loaded.pid)
            finally:
                store.AGENTS_DIR = old_dir

    def _make_alive_session(
        self,
        tmp: str,
        agent_id: str,
        *,
        kind: str = "chat",
        stdout_age_seconds: float = 0.0,
    ) -> store.Session:
        """Persist a running session whose stdout.log mtime is N seconds in the past.

        Used by silence-based reconciliation tests so we don't have to
        wait wall-clock time for the threshold to elapse.
        """
        import os
        import time

        sess = store.Session(
            agent_id=agent_id,
            state=store.STATE_RUNNING,
            model="opus",
            workspace="/tmp/workspace",
            created_at=store.now_iso(),
            backend="cursor-cli",
            pid=os.getpid(),
            kind=kind,
        )
        store.save(sess)
        stdout_path = store.stdout_file(agent_id)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_bytes(b'{"type":"system","subtype":"init"}\n')
        target = time.time() - stdout_age_seconds
        os.utime(stdout_path, (target, target))
        sess_path = store.session_file(agent_id)
        os.utime(sess_path, (target, target))
        return sess

    def test_last_output_ts_returns_newest_mtime_across_session_and_stdout(self) -> None:
        """`last_output_ts` must combine session.json + stdout.log signals
        so a session that has only ever rewritten its metadata still
        registers activity, and an empty session (no files yet) returns
        ``None`` instead of a sentinel epoch."""
        import os
        import time

        with tempfile.TemporaryDirectory() as tmp:
            old_dir = store.AGENTS_DIR
            store.AGENTS_DIR = Path(tmp)
            try:
                self.assertIsNone(store.last_output_ts("web-missing"))

                self._make_alive_session(tmp, "web-fresh", stdout_age_seconds=0)
                stdout_path = store.stdout_file("web-fresh")
                sess_path = store.session_file("web-fresh")
                target = time.time() - 100
                os.utime(stdout_path, (target, target))
                later = time.time() - 5
                os.utime(sess_path, (later, later))

                ts = store.last_output_ts("web-fresh")
                self.assertIsNotNone(ts)
                assert ts is not None
                self.assertAlmostEqual(ts, later, delta=1.0)
            finally:
                store.AGENTS_DIR = old_dir

    def test_alive_chat_session_with_recent_output_is_not_silent(self) -> None:
        """Within the silence threshold, a live chat must remain
        responding; over-eager silence flagging would interrupt a
        normal mid-turn stream where the model is just thinking."""
        with tempfile.TemporaryDirectory() as tmp:
            old_dir = store.AGENTS_DIR
            store.AGENTS_DIR = Path(tmp)
            try:
                sess = self._make_alive_session(tmp, "web-active", stdout_age_seconds=2.0)
                self.assertFalse(store.is_alive_but_silent(sess))
            finally:
                store.AGENTS_DIR = old_dir

    def test_alive_chat_session_with_no_output_for_threshold_is_silent(self) -> None:
        """The user-reported failure mode: PID alive but the wrapper has
        emitted no events for many minutes because the inner ``agent``
        CLI is blocked on a never-returning shell tool call (e.g. a
        chat agent that invoked ``bash agent-loop.sh``). The snapshot
        layer must observe this via ``is_alive_but_silent`` and stop
        reporting the chat as responding."""
        with tempfile.TemporaryDirectory() as tmp:
            old_dir = store.AGENTS_DIR
            store.AGENTS_DIR = Path(tmp)
            try:
                sess = self._make_alive_session(
                    tmp,
                    "web-stalled",
                    stdout_age_seconds=store.STALE_OUTPUT_THRESHOLD_SECONDS + 60,
                )
                self.assertTrue(store.is_alive_but_silent(sess))
            finally:
                store.AGENTS_DIR = old_dir

    def test_silent_loop_wrapper_is_not_flagged_as_alive_but_silent(self) -> None:
        """Loop wrappers (``kind="loop_wrapper"``) legitimately go quiet
        between stage/round boundaries while their training subprocess
        runs for hours. Flagging them would flood the loop view with
        false "stalled" pills on every long round. Restrict the
        silence detector to chat-kind sessions only."""
        with tempfile.TemporaryDirectory() as tmp:
            old_dir = store.AGENTS_DIR
            store.AGENTS_DIR = Path(tmp)
            try:
                sess = self._make_alive_session(
                    tmp,
                    "loop-quiet",
                    kind="loop_wrapper",
                    stdout_age_seconds=store.STALE_OUTPUT_THRESHOLD_SECONDS + 600,
                )
                self.assertFalse(store.is_alive_but_silent(sess))
            finally:
                store.AGENTS_DIR = old_dir

    def test_is_alive_but_silent_requires_pid_to_be_alive(self) -> None:
        """A dead PID is the zombie path, handled by ``_reconcile_zombie``;
        ``is_alive_but_silent`` must only return True when both the PID
        is alive AND the output is stale, so the two reconcilers stay
        orthogonal."""
        import os

        with tempfile.TemporaryDirectory() as tmp:
            old_dir = store.AGENTS_DIR
            store.AGENTS_DIR = Path(tmp)
            try:
                pid = os.fork()
                if pid == 0:
                    os._exit(0)
                os.waitpid(pid, 0)

                sess = store.Session(
                    agent_id="web-dead-and-silent",
                    state=store.STATE_RUNNING,
                    model="opus",
                    workspace="/tmp/workspace",
                    created_at=store.now_iso(),
                    backend="cursor-cli",
                    pid=pid,
                    kind="chat",
                )
                store.save(sess)
                stdout = store.stdout_file("web-dead-and-silent")
                stdout.parent.mkdir(parents=True, exist_ok=True)
                stdout.write_bytes(b"")
                target = 0.0
                os.utime(stdout, (target, target))
                self.assertFalse(store.is_alive_but_silent(sess))
            finally:
                store.AGENTS_DIR = old_dir

    def test_snapshot_flips_to_stalled_for_alive_silent_chat(self) -> None:
        """`_snapshot` must surface the alive-but-silent case as
        ``is_responding=False`` + ``is_stalled=True`` +
        ``conversation_status="stalled"`` so the frontend stops
        falsely showing the pill as "responding" indefinitely.

        Reconciliation must NOT mutate persisted state (the wrapper
        may legitimately resume output later when the long subprocess
        finally returns) and must NOT kill the PID (the user's
        descendant training subprocess often wants to keep running)."""
        from web.agents import runner
        from web.routers import agent as agent_router

        with tempfile.TemporaryDirectory() as tmp:
            old_dir = store.AGENTS_DIR
            store.AGENTS_DIR = Path(tmp)
            try:
                sess = self._make_alive_session(
                    tmp,
                    "web-ghost",
                    stdout_age_seconds=store.STALE_OUTPUT_THRESHOLD_SECONDS + 120,
                )
                with mock.patch.object(runner, "is_managed_running", return_value=False):
                    snap = agent_router._snapshot(sess)

                self.assertFalse(snap["is_responding"])
                self.assertFalse(snap["is_running"])
                self.assertTrue(snap["is_stalled"])
                self.assertEqual(snap["conversation_status"], "stalled")
                self.assertIsNotNone(snap["last_output_ts"])

                reloaded = store.load("web-ghost")
                assert reloaded is not None
                self.assertEqual(reloaded.state, store.STATE_RUNNING)
                self.assertEqual(reloaded.pid, sess.pid)
            finally:
                store.AGENTS_DIR = old_dir

    def test_snapshot_flips_to_stalled_even_when_in_memory_handle_is_active(self) -> None:
        """The user-reported failure is a chat agent the web server
        itself spawned: ``is_managed_running`` is True because the
        ``monitor_task`` is still awaiting ``proc.wait()`` on the
        alive-but-blocked wrapper. Silence detection MUST apply
        regardless of in-memory ownership — otherwise the original
        complaint (web-spawned chat stuck on a never-returning
        ``bash agent-loop.sh`` tool call) would not be fixed."""
        from web.agents import runner
        from web.routers import agent as agent_router

        with tempfile.TemporaryDirectory() as tmp:
            old_dir = store.AGENTS_DIR
            store.AGENTS_DIR = Path(tmp)
            try:
                sess = self._make_alive_session(
                    tmp,
                    "web-monitored-ghost",
                    stdout_age_seconds=store.STALE_OUTPUT_THRESHOLD_SECONDS + 120,
                )
                with mock.patch.object(runner, "is_managed_running", return_value=True):
                    snap = agent_router._snapshot(sess)

                self.assertFalse(snap["is_responding"])
                self.assertTrue(snap["is_stalled"])
                self.assertEqual(snap["conversation_status"], "stalled")
            finally:
                store.AGENTS_DIR = old_dir

    def test_snapshot_stays_responding_for_actively_streaming_chat(self) -> None:
        """An agent whose stdout was touched within the silence
        threshold (mid-turn streaming, recent tool call activity)
        must keep its "responding" label so the UI does not
        flicker between states as events arrive."""
        from web.agents import runner
        from web.routers import agent as agent_router

        with tempfile.TemporaryDirectory() as tmp:
            old_dir = store.AGENTS_DIR
            store.AGENTS_DIR = Path(tmp)
            try:
                sess = self._make_alive_session(tmp, "web-streaming", stdout_age_seconds=5.0)
                with mock.patch.object(runner, "is_managed_running", return_value=True):
                    snap = agent_router._snapshot(sess)

                self.assertTrue(snap["is_responding"])
                self.assertFalse(snap["is_stalled"])
                self.assertEqual(snap["conversation_status"], "responding")
            finally:
                store.AGENTS_DIR = old_dir

    def test_claude_stream_json_rebuilds_frontend_messages(self) -> None:
        events = [
            {
                "type": "web_user",
                "message": {"content": [{"type": "text", "text": "Say hi"}]},
            },
            {"type": "system", "subtype": "init", "session_id": "claude-session"},
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text"},
                },
            },
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Hello"},
                },
            },
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "Read",
                    },
                },
            },
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": '{"path": "README.md"}',
                    },
                },
            },
            {
                "type": "stream_event",
                "event": {"type": "content_block_stop", "index": 1},
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-1",
                            "content": "file contents",
                        }
                    ]
                },
            },
            {"type": "result", "subtype": "success"},
        ]

        rebuilt = messages.rebuild(events)

        self.assertEqual(rebuilt[0]["role"], "user")
        self.assertEqual(rebuilt[0]["content"], "Say hi")
        self.assertEqual(rebuilt[1]["role"], "assistant")
        self.assertIn("Hello", rebuilt[1]["content"])
        self.assertEqual(rebuilt[1]["status"], "completed")
        self.assertEqual(rebuilt[1]["_toolCalls"][0]["id"], "tool-1")
        self.assertEqual(rebuilt[1]["_toolCalls"][0]["name"], "Read")
        self.assertEqual(rebuilt[1]["_toolCalls"][0]["args"], {"path": "README.md"})
        self.assertEqual(rebuilt[1]["_toolCalls"][0]["result"], "file contents")

    def test_cursor_cli_writes_synthetic_user_event_upfront(self) -> None:
        """Cursor-cli must declare ``synthetic_user_event = True`` so the
        spawn path writes the prompt to ``stdout.log`` upfront instead of
        waiting for the CLI to echo it back.

        This mirrors AutoSkill's ``core/agent/run.py:_run_turn`` pattern:
        the user prompt is the authoritative record of "what task this
        agent is doing", and must be on disk as soon as ``spawn_session``
        returns. Without the synthetic event a loop_dev_round agent —
        whose 10K+ char system prompt routinely keeps cursor-cli busy
        for minutes before any stream-json event lands — appears in the
        UI as an empty pane with no visible task.
        """
        from web.agents import backends

        cursor = backends.get_backend("cursor-cli")
        self.assertTrue(
            cursor.synthetic_user_event,
            "cursor-cli must opt into the synthetic upfront-user-event "
            "write so the frontend renders the task immediately instead "
            "of staring at a blank pane while the CLI bootstraps",
        )

        event_bytes = backends.synthetic_user_event("hello world")
        decoded = event_bytes.decode("utf-8").rstrip("\n")
        payload = __import__("json").loads(decoded)
        self.assertEqual(payload["type"], "web_user")
        self.assertEqual(payload["message"]["content"][0]["text"], "hello world")

    def test_cursor_echo_user_event_is_collapsed_into_synthetic_web_user(self) -> None:
        """The frontend must not show two user rows when cursor-cli's
        own ``{"type":"user",...}`` echo of the prompt lands after the
        upfront ``web_user`` row that ``spawn_session`` wrote.

        Regression guard: before the dedupe lived in
        ``messages._handle_user`` flipping ``cursor-cli`` to
        ``synthetic_user_event=True`` would have produced a duplicate
        turn for every cursor-cli chat — the upfront web_user opens
        turn 1 and the CLI's later user event opened a redundant
        turn 2 with the same text.
        """
        events = [
            {
                "type": "web_user",
                "message": {"content": [{"type": "text", "text": "do the thing"}]},
            },
            {"type": "system", "subtype": "init", "session_id": "s1"},
            {
                "type": "user",
                "message": {"content": [{"type": "text", "text": "do the thing"}]},
            },
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "On it."}]},
            },
            {"type": "result", "subtype": "success"},
        ]
        rebuilt = messages.rebuild(events)
        user_rows = [m for m in rebuilt if m.get("role") == "user"]
        self.assertEqual(
            len(user_rows),
            1,
            "the cursor-cli echo of the prompt must collapse into the "
            "upfront web_user row instead of opening a second turn",
        )
        self.assertEqual(user_rows[0]["content"], "do the thing")
        assistants = [m for m in rebuilt if m.get("role") == "assistant"]
        self.assertEqual(len(assistants), 1)
        self.assertEqual(assistants[0]["content"], "On it.")

    def test_distinct_user_messages_still_open_new_turns(self) -> None:
        """The dedupe must only collapse echoes — a genuine follow-up
        user message with different text MUST still open a new turn.
        Without this guard the dedupe would swallow real conversation
        continuations and the chat would look frozen on turn 1.
        """
        events = [
            {
                "type": "web_user",
                "message": {"content": [{"type": "text", "text": "first prompt"}]},
            },
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "first reply"}]},
            },
            {"type": "result", "subtype": "success"},
            {
                "type": "web_user",
                "message": {"content": [{"type": "text", "text": "follow up"}]},
            },
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "second reply"}]},
            },
            {"type": "result", "subtype": "success"},
        ]
        rebuilt = messages.rebuild(events)
        user_contents = [m["content"] for m in rebuilt if m.get("role") == "user"]
        self.assertEqual(user_contents, ["first prompt", "follow up"])

    def test_router_exposes_backend_aware_contract(self) -> None:
        router_text = (
            Path(__file__).resolve().parents[3] / "web" / "routers" / "agent.py"
        ).read_text(encoding="utf-8")
        agents_js = (
            Path(__file__).resolve().parents[3] / "web" / "static" / "js" / "agents.js"
        ).read_text(encoding="utf-8")
        index_html = (
            Path(__file__).resolve().parents[3] / "web" / "static" / "index.html"
        ).read_text(encoding="utf-8")

        self.assertIn('backend: str = "cursor-cli"', router_text)
        self.assertIn('@router.get("/backends")', router_text)
        self.assertIn("backend=req.backend", router_text)
        self.assertIn("runner.list_models(auth.api_key(backend), backend=backend)", router_text)
        self.assertIn("fetch('/api/agent/backends')", agents_js)
        self.assertIn("newAgentBackend", agents_js)
        self.assertIn("backend: this.newAgentBackend", agents_js)
        self.assertIn('x-model="newAgentBackend"', index_html)

    def test_auth_env_only_returns_env_var_key(self) -> None:
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-good"}, clear=False):
            self.assertEqual(auth.api_key("claude-code"), "env-good")
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}, clear=False):
            self.assertIsNone(auth.api_key("claude-code"))

    def test_codex_auth_uses_active_provider_env_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.toml"
            cfg.write_text(
                "\n".join(
                    [
                        'model_provider = "lm_center"',
                        "",
                        "[model_providers.lm_center]",
                        'env_key = "LLM_CENTER_API_KEY"',
                    ]
                ),
                encoding="utf-8",
            )
            with mock.patch.dict(
                "os.environ",
                {
                    "CODEX_HOME": tmp,
                    "LLM_CENTER_API_KEY": "env-good",
                    "OPENAI_API_KEY": "",
                },
                clear=True,
            ):
                self.assertEqual(auth.api_key("codex"), "env-good")
                status = auth.all_status()
        self.assertTrue(status["codex"]["has_env_key"])
        self.assertEqual(status["codex"]["env_var"], "LLM_CENTER_API_KEY")

    def test_auth_all_status_reports_env_key_presence(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {"CURSOR_API_KEY": "ck", "ANTHROPIC_API_KEY": ""},
            clear=False,
        ):
            status = auth.all_status()
            self.assertTrue(status["cursor-cli"]["has_env_key"])
            self.assertFalse(status["claude-code"]["has_env_key"])

    def test_auth_status_delegates_to_check_backend_available(self) -> None:
        """runner.auth_status must delegate to agent_loop_config.check_backend_available."""
        import json

        from web.agents import runner

        ok_json = json.dumps({"ok": True, "email": "user@example.com"})
        with mock.patch(
            "tools.agent_loop_config.check_backend_available",
            return_value=(True, ok_json),
        ):
            status = __import__("asyncio").run(runner.auth_status("cursor-cli"))
        self.assertTrue(status["is_authenticated"])
        self.assertEqual(status["email"], "user@example.com")

    def test_auth_status_surfaces_failure_from_check_backend(self) -> None:
        from web.agents import runner

        with mock.patch(
            "tools.agent_loop_config.check_backend_available",
            return_value=(False, "cursor backend unauthenticated"),
        ):
            status = __import__("asyncio").run(runner.auth_status("cursor-cli"))
        self.assertFalse(status["is_authenticated"])
        self.assertIn("unauthenticated", status["auth_error"])

    def test_cursor_resume_append_preserves_full_history(self) -> None:
        """Cursor CLI does NOT replay prior conversation on ``--resume`` — it
        emits only a fresh ``system.init`` plus the new turn's events. The
        previous truncation strategy (``open_mode="wb"``) therefore destroyed
        every earlier turn the moment the user sent a second message.

        The corrected strategy is to always append: the new ``system.init``
        line interleaves cleanly because :func:`messages.rebuild` ignores
        ``type=system`` events, and every prior user/assistant pair is
        retained as its own turn.

        This test pins the empirical fact (verified against the on-disk
        ``stdout.log`` vs the IDE-side ``agent-transcripts/<id>.jsonl``)
        by exercising the rebuilder on the appended-log shape and asserting
        both turns survive without duplication.
        """
        run1_events = [
            {"type": "system", "subtype": "init", "session_id": "s1"},
            {"type": "user", "message": {"content": [{"type": "text", "text": "hello"}]}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Hi there"}]}},
            {"type": "result", "subtype": "success"},
        ]
        # Run 2 on --resume: cursor CLI emits a fresh init and ONLY the new
        # turn (no replay of run 1).
        run2_events = [
            {"type": "system", "subtype": "init", "session_id": "s1"},
            {"type": "user", "message": {"content": [{"type": "text", "text": "continue"}]}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Sure"}]}},
            {"type": "result", "subtype": "success"},
        ]

        # FIX scenario: appending keeps both turns intact.
        appended = run1_events + run2_events
        msgs_appended = messages.rebuild(appended)
        user_contents = [m["content"] for m in msgs_appended if m["role"] == "user"]
        self.assertEqual(user_contents, ["hello", "continue"])
        assistant_contents = [m["content"] for m in msgs_appended if m["role"] == "assistant"]
        self.assertEqual(assistant_contents, ["Hi there", "Sure"])

        # BUG scenario (regression guard): truncating before run 2 wipes turn 1.
        msgs_truncated = messages.rebuild(run2_events)
        user_truncated = [m["content"] for m in msgs_truncated if m["role"] == "user"]
        self.assertEqual(
            user_truncated,
            ["continue"],
            "truncating run 2's window must be detectable so the regression "
            "would re-fail this assertion",
        )

    def test_frontend_readonly_auth_and_no_key_modal(self) -> None:
        """After removing web-keys.json persistence, the UI must:
        - Not have set-key / delete-key endpoints
        - Not have an API key modal
        - Show a toast on auth failure instead of prompting for a key
        - Settings page is read-only (no key input / Save / Delete)
        """
        router_text = (
            Path(__file__).resolve().parents[3] / "web" / "routers" / "agent.py"
        ).read_text(encoding="utf-8")
        agents_js = (
            Path(__file__).resolve().parents[3] / "web" / "static" / "js" / "agents.js"
        ).read_text(encoding="utf-8")
        index_html = (
            Path(__file__).resolve().parents[3] / "web" / "static" / "index.html"
        ).read_text(encoding="utf-8")
        app_js = (
            Path(__file__).resolve().parents[3] / "web" / "static" / "js" / "app.js"
        ).read_text(encoding="utf-8")

        self.assertNotIn('@router.post("/set-key")', router_text)
        self.assertNotIn('@router.delete("/key")', router_text)
        self.assertNotIn('data-testid="agent-api-key-modal"', index_html)
        self.assertNotIn("requestApiKey", agents_js)
        self.assertNotIn("saveApiKeyPrompt", agents_js)
        self.assertNotIn("cancelApiKeyPrompt", agents_js)
        self.assertNotIn("openAgentKeyPrompt", agents_js)
        self.assertIn("if (data.has_auth) return true;", agents_js)
        self.assertIn("_showToast", agents_js)
        self.assertNotIn("saveSettingsKey", app_js)
        self.assertNotIn("deleteSettingsKey", app_js)
        self.assertIn("read-only", index_html)
        self.assertIn("envVar: st.env_var || be.envVar", app_js)
        self.assertIn("be.envVar", index_html)
        self.assertIn("data.env_var || envVars[backend]", agents_js)
        self.assertIn("fixedCliModelBackends", agents_js)
        self.assertIn("hasSelectableModel", agents_js)
        self.assertIn("hasSelectableModel(newAgentBackend)", index_html)
        self.assertIn("hasSelectableModel(loopCreateChatProfile.backend)", index_html)

    def test_session_persists_max_mode_for_resume(self) -> None:
        """Session must round-trip ``max_mode`` so that ``runner.submit``
        can default to the same launch flag the chat was created with.

        Regression guard: before this fix, the schema dropped ``max_mode``
        and a follow-up turn on a max-mode-required model (e.g.
        ``gpt-5.5-high``) would fail with ``Max Mode Required`` because
        the resume path defaulted to ``False``.
        """
        with tempfile.TemporaryDirectory() as tmp:
            old_dir = store.AGENTS_DIR
            store.AGENTS_DIR = Path(tmp)
            try:
                sess = store.Session(
                    agent_id="web-mm",
                    backend="cursor-cli",
                    backend_session_id="sid",
                    state=store.STATE_RUNNING,
                    model="gpt-5.5-high",
                    workspace="/tmp/workspace",
                    created_at=store.now_iso(),
                    max_mode=True,
                )
                store.save(sess)
                loaded = store.load("web-mm")
                assert loaded is not None
                self.assertTrue(loaded.max_mode)
            finally:
                store.AGENTS_DIR = old_dir

    def test_session_load_defaults_max_mode_for_legacy_session_json(self) -> None:
        """Legacy session.json without ``max_mode`` must still load
        (defaults to ``False``) so the rehydration of pre-fix sessions
        does not break."""
        import json

        with tempfile.TemporaryDirectory() as tmp:
            old_dir = store.AGENTS_DIR
            store.AGENTS_DIR = Path(tmp)
            try:
                d = store.AGENTS_DIR / "web-legacy-mm"
                d.mkdir()
                (d / "session.json").write_text(
                    json.dumps(
                        {
                            "agent_id": "web-legacy-mm",
                            "state": "completed",
                            "model": "opus",
                            "workspace": "/tmp",
                            "created_at": store.now_iso(),
                            "backend": "cursor-cli",
                        }
                    ),
                    encoding="utf-8",
                )
                loaded = store.load("web-legacy-mm")
                assert loaded is not None
                self.assertFalse(loaded.max_mode)
            finally:
                store.AGENTS_DIR = old_dir

    def test_runner_submit_inherits_persisted_max_mode_when_caller_omits_it(self) -> None:
        """The HTTP submit path doesn't always carry ``max_mode``; the
        runner must fall back to the persisted Session value so a
        gpt-5.5-high chat keeps resuming with --max-mode after the
        first turn."""
        import asyncio

        from web.agents import runner

        scheduled: list[object] = []

        class DummyTask:
            def done(self) -> bool:
                return False

        def capture_task(coro):  # type: ignore[no-untyped-def]
            scheduled.append(coro)
            return DummyTask()

        with tempfile.TemporaryDirectory() as tmp:
            old_dir = store.AGENTS_DIR
            store.AGENTS_DIR = Path(tmp)
            try:
                sess = store.Session(
                    agent_id="web-resume-mm",
                    backend="cursor-cli",
                    backend_session_id="sid",
                    state=store.STATE_COMPLETED,
                    model="gpt-5.5-high",
                    workspace="/tmp/workspace",
                    created_at=store.now_iso(),
                    max_mode=True,
                )
                store.save(sess)
                with mock.patch.object(runner.asyncio, "create_task", capture_task):
                    result = asyncio.run(
                        runner.submit(
                            agent_id="web-resume-mm",
                            prompt="follow up",
                            max_mode=None,
                        )
                    )
                self.assertTrue(result.max_mode)
                loaded = store.load("web-resume-mm")
                assert loaded is not None
                self.assertTrue(loaded.max_mode)
                self.assertEqual(len(scheduled), 1)
            finally:
                for coro in scheduled:
                    if hasattr(coro, "close"):
                        coro.close()
                store.AGENTS_DIR = old_dir

    def test_runner_submit_precreates_session_without_waiting_for_cli_init(self) -> None:
        """Submitting a follow-up from the web UI must return after the
        user turn is registered, not after cursor-cli emits ``system.init``.

        The old path awaited ``_spawn`` directly, so any slow Cursor CLI
        resume/init handshake blocked the HTTP request and made the Send
        button look stuck. The new path pre-registers the session and
        schedules backend startup in the event loop.
        """
        import asyncio

        from web.agents import runner

        async def slow_sync_spawn(**_kwargs):  # type: ignore[no-untyped-def]
            await asyncio.sleep(60)
            raise AssertionError("submit must not await the synchronous spawn path")

        scheduled: list[object] = []

        class DummyTask:
            def done(self) -> bool:
                return False

            def cancel(self) -> None:
                return None

        def capture_task(coro):  # type: ignore[no-untyped-def]
            scheduled.append(coro)
            return DummyTask()

        async def _drive() -> store.Session:
            return await asyncio.wait_for(
                runner.submit(agent_id="web-fast-submit", prompt="follow up"),
                timeout=0.2,
            )

        with tempfile.TemporaryDirectory() as tmp:
            old_dir = store.AGENTS_DIR
            store.AGENTS_DIR = Path(tmp)
            try:
                sess = store.Session(
                    agent_id="web-fast-submit",
                    backend="cursor-cli",
                    backend_session_id="sid",
                    state=store.STATE_COMPLETED,
                    model="gpt-5.5-high",
                    workspace="/tmp/workspace",
                    created_at=store.now_iso(),
                    max_mode=True,
                )
                store.save(sess)
                with (
                    mock.patch.object(runner, "_monitor_precreated_start", slow_sync_spawn),
                    mock.patch.object(runner.asyncio, "create_task", capture_task),
                ):
                    result = asyncio.run(_drive())

                self.assertEqual(result.agent_id, "web-fast-submit")
                self.assertEqual(result.state, store.STATE_RUNNING)
                self.assertEqual(result.backend_session_id, "sid")
                self.assertEqual(len(scheduled), 1)
                stdout = store.stdout_file("web-fast-submit").read_text(encoding="utf-8")
                self.assertIn('"type": "web_user"', stdout)
                self.assertIn("follow up", stdout)
            finally:
                for coro in scheduled:
                    coro.close()
                store.AGENTS_DIR = old_dir

    def test_settings_page_shows_cli_login_status_for_both_backends(self) -> None:
        """Settings page must surface CLI login status and env var presence."""
        router_text = (
            Path(__file__).resolve().parents[3] / "web" / "routers" / "agent.py"
        ).read_text(encoding="utf-8")
        app_js = (
            Path(__file__).resolve().parents[3] / "web" / "static" / "js" / "app.js"
        ).read_text(encoding="utf-8")
        index_html = (
            Path(__file__).resolve().parents[3] / "web" / "static" / "index.html"
        ).read_text(encoding="utf-8")

        self.assertIn("cli_logged_in", router_text)
        self.assertIn("cli_email", router_text)
        self.assertIn("cliLoggedIn", app_js)
        self.assertIn("cli_logged_in", app_js)
        self.assertIn("hasEnvKey", app_js)
        self.assertIn("has_env_key", app_js)
        self.assertIn("be.cliLoggedIn", index_html)
        self.assertIn("be.hasEnvKey", index_html)

    def test_pump_stdout_with_ts_survives_lines_larger_than_64kib(self) -> None:
        """An agent CLI may emit a single stream-json event larger than
        ``asyncio.StreamReader``'s default 64 KiB limit (e.g. an
        ``assistant`` event that quotes a long file or system-prompt
        echo). ``pump_stdout_with_ts`` must drain the stream without
        crashing or stalling — otherwise the upstream PIPE fills,
        the wrapper blocks on ``write()``, ``proc.wait()`` never
        returns, and the Session sits in ``state=running`` forever
        (observed in production as ``web-9244ef390f5a``).
        """
        import asyncio

        from web.agents import spawn

        big_payload = "x" * (200 * 1024)
        line = ('{"type":"assistant","subtype":"text","content":"' + big_payload + '"}\n').encode(
            "utf-8"
        )
        small_line = b'{"type":"assistant","subtype":"text","content":"tail"}\n'

        async def _drive() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / "stdout.log"
                reader = asyncio.StreamReader()
                pump = asyncio.create_task(spawn.pump_stdout_with_ts(reader, target))
                reader.feed_data(line)
                reader.feed_data(small_line)
                reader.feed_eof()
                await asyncio.wait_for(pump, timeout=2.0)

                contents = target.read_bytes().splitlines()
                self.assertEqual(len(contents), 2, contents[:1])
                import json as _json

                rec0 = _json.loads(contents[0])
                rec1 = _json.loads(contents[1])
                self.assertEqual(rec0["content"], big_payload)
                self.assertEqual(rec1["content"], "tail")
                self.assertIn("_ts", rec0)
                self.assertIn("_ts", rec1)

        asyncio.run(_drive())

    def test_pump_stream_survives_lines_larger_than_64kib(self) -> None:
        """``pump_stream`` (used for stderr) already reads in chunks and
        must continue to survive arbitrarily long lines without
        truncation; pin the invariant so a future refactor cannot
        regress stderr pumping into a ``readline()``-based bug."""
        import asyncio

        from web.agents import spawn

        payload = b"y" * (200 * 1024) + b"\n"

        async def _drive() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / "stderr.log"
                reader = asyncio.StreamReader()
                pump = asyncio.create_task(spawn.pump_stream(reader, target))
                reader.feed_data(payload)
                reader.feed_eof()
                await asyncio.wait_for(pump, timeout=2.0)
                self.assertEqual(target.read_bytes(), payload)

        asyncio.run(_drive())


if __name__ == "__main__":
    unittest.main()
