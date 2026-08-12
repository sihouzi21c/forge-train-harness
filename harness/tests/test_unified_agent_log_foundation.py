"""Foundation tests for unifying all spawned agents under the web-agents/<id>/ branch.

These tests pin the SSOT contract before the agent-loop.sh refactor lands:

1. ``web.agents.store.Session`` carries ``parent_agent_id`` / ``loop_id`` /
   ``kind`` so the dashboard can group dev / review / stage2 subagents
   under their owning loop wrapper.
2. ``web.agents.spawn`` exposes the low-level spawn / init / pump /
   monitor primitives that ``web.agents.runner`` already uses internally,
   so a CLI-side helper (``harness/tools/spawn_managed_agent.py``) can
   reuse the exact same write path without duplicating it.
3. ``web.agents.backends`` knows about a synthetic ``loop-wrapper``
   backend (no command, no spawn) so the loop wrapper agent entry can be
   created on disk without tripping backend validation.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from web.agents import store


class TestSessionSchemaUnifiedFields(unittest.TestCase):
    """``Session`` carries the parent/loop/kind triple used to thread
    dev/review/subagent rows back to their owning loop wrapper."""

    def test_session_defaults_for_new_fields(self) -> None:
        sess = store.Session(
            agent_id="web-x",
            state=store.STATE_RUNNING,
            model="opus",
            workspace="/tmp/ws",
            created_at=store.now_iso(),
        )
        # Defaults: chat agent, no parent, no loop.
        self.assertEqual(sess.kind, "chat")
        self.assertIsNone(sess.parent_agent_id)
        self.assertIsNone(sess.loop_id)

    def test_session_round_trips_through_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            original = store.AGENTS_DIR
            store.AGENTS_DIR = Path(tmp)
            try:
                sess = store.Session(
                    agent_id="web-dev-1",
                    state=store.STATE_RUNNING,
                    model="opus",
                    workspace="/tmp/ws",
                    created_at=store.now_iso(),
                    parent_agent_id="loop-abcdef",
                    loop_id="abcdef",
                    kind="loop_dev_round",
                )
                store.save(sess)
                loaded = store.load("web-dev-1")
                assert loaded is not None
                self.assertEqual(loaded.parent_agent_id, "loop-abcdef")
                self.assertEqual(loaded.loop_id, "abcdef")
                self.assertEqual(loaded.kind, "loop_dev_round")
            finally:
                store.AGENTS_DIR = original

    def test_session_load_tolerates_legacy_session_json_without_new_fields(self) -> None:
        """Older session.json on disk must still load (defaults applied)."""
        import json

        with tempfile.TemporaryDirectory() as tmp:
            original = store.AGENTS_DIR
            store.AGENTS_DIR = Path(tmp)
            try:
                d = store.AGENTS_DIR / "web-legacy"
                d.mkdir()
                # Schema as it existed before the unified-agent-log change.
                (d / "session.json").write_text(
                    json.dumps(
                        {
                            "agent_id": "web-legacy",
                            "state": "completed",
                            "model": "opus",
                            "workspace": "/tmp",
                            "created_at": store.now_iso(),
                            "backend": "cursor-cli",
                            "backend_session_id": "old-sid",
                        }
                    ),
                    encoding="utf-8",
                )
                loaded = store.load("web-legacy")
                assert loaded is not None
                self.assertEqual(loaded.kind, "chat")
                self.assertIsNone(loaded.parent_agent_id)
                self.assertIsNone(loaded.loop_id)
            finally:
                store.AGENTS_DIR = original


class TestSpawnLibraryExtracted(unittest.TestCase):
    """``web.agents.spawn`` is the SSOT spawn implementation.

    Both ``web.agents.runner`` (in-process, for the Agents tab HTTP API)
    and ``harness/tools/spawn_managed_agent.py`` (out-of-process, for
    agent-loop.sh) MUST go through these primitives so the on-disk
    record format stays identical for every code path.
    """

    def test_spawn_module_exposes_primitives(self) -> None:
        from web.agents import spawn

        for name in (
            "read_until_init",
            "pump_stream",
            "monitor_exit",
            "spawn_session",
        ):
            self.assertTrue(
                hasattr(spawn, name),
                f"web.agents.spawn must expose {name!r}; got {dir(spawn)!r}",
            )

    def test_runner_delegates_to_spawn_library(self) -> None:
        """runner must call into spawn primitives rather than carry its
        own copy of the init / pump / monitor logic."""
        import inspect

        from web.agents import runner

        src = inspect.getsource(runner)
        self.assertTrue(
            "spawn.precreate_web_session" in src or "spawn.spawn_session" in src,
            "runner must delegate to web.agents.spawn primitives (SSOT for the spawn write path).",
        )

    def test_spawn_session_closes_child_stdin(self) -> None:
        """Backend CLIs must not inherit the web server's stdin.

        Codex treats a piped/closed stdin as extra prompt input in exec mode;
        leaving stdin inherited made no-reload/nohup web chats disconnect
        before ``response.completed``.
        """
        import asyncio

        from web.agents import spawn

        captured: dict[str, object] = {}

        class FakeProc:
            pid = 12345
            stdout = None
            stderr = None
            returncode = None

        async def fake_exec(*_cmd, **kwargs):
            captured.update(kwargs)
            return FakeProc()

        async def fake_read_until_init(_proc, _backend, _expected_session_id):
            return "backend-session", [], ""

        with tempfile.TemporaryDirectory() as tmp:
            original = store.AGENTS_DIR
            store.AGENTS_DIR = Path(tmp) / "agents"
            try:
                with (
                    mock.patch.object(spawn.asyncio, "create_subprocess_exec", fake_exec),
                    mock.patch.object(spawn, "read_until_init", fake_read_until_init),
                ):
                    asyncio.run(
                        spawn.spawn_session(
                            backend_name="cursor-cli",
                            model="",
                            prompt="ls",
                            workspace=tmp,
                        )
                    )
            finally:
                store.AGENTS_DIR = original

        self.assertIs(captured.get("stdin"), asyncio.subprocess.DEVNULL)


class TestLoopWrapperBackendRegistered(unittest.TestCase):
    """A ``loop-wrapper`` backend exists so the loop's own session.json
    can carry a valid backend label without ever spawning a CLI."""

    def test_backend_registry_includes_loop_wrapper(self) -> None:
        from web.agents import backends

        # Synthetic backend is hidden from the default New-chat dropdown
        # but MUST be addressable by name via get_backend.
        wrapper = backends.get_backend("loop-wrapper")
        self.assertTrue(wrapper.is_synthetic)
        default_listing = {b["name"] for b in backends.list_backends()}
        self.assertNotIn("loop-wrapper", default_listing)
        full_listing = {b["name"] for b in backends.list_backends(include_synthetic=True)}
        self.assertIn("loop-wrapper", full_listing)

    def test_loop_wrapper_backend_refuses_to_build_command(self) -> None:
        """``loop-wrapper`` is synthetic: build_command MUST raise so a
        future bug that tries to spawn it as a CLI fails fast at the
        boundary instead of producing a half-broken process."""
        from web.agents import backends

        wrapper = backends.get_backend("loop-wrapper")
        with self.assertRaises(ValueError):
            wrapper.build_command(
                model="",
                prompt="",
                workspace="/tmp",
                resume_session_id=None,
                max_mode=False,
            )


class TestForestLayout(unittest.TestCase):
    """Per-loop forest: child agents nest under their wrapper tree root,
    grouped by stage/round and ordered by zero-padded seq.

    Contract pinned by docs/forest-agent-log-layout.md:
    child agent_id == ``loop-<loop_id>_<stage>_r<RRR>_<SS>_web-<uuid>`` and
    ``session_dir`` resolves it to
    ``<AGENTS_DIR>/loop-<loop_id>/agents/<stage>/r<RRR>/<full-id>/``.
    """

    def _swap_agents_dir(self, path: Path):
        original = store.AGENTS_DIR
        store.AGENTS_DIR = path
        self.addCleanup(lambda: setattr(store, "AGENTS_DIR", original))

    # --- session_dir resolution -----------------------------------------

    def test_session_dir_wrapper_and_chat_stay_top_level(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._swap_agents_dir(Path(tmp))
            self.assertEqual(store.session_dir("loop-X"), Path(tmp) / "loop-X")
            self.assertEqual(store.session_dir("web-Z"), Path(tmp) / "web-Z")

    def test_session_dir_child_nests_under_stage_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._swap_agents_dir(Path(tmp))
            cid = "loop-X_stage1_r003_02_web-Y"
            self.assertEqual(
                store.session_dir(cid),
                Path(tmp) / "loop-X" / "agents" / "stage1" / "r003" / cid,
            )

    def test_session_dir_underscore_prefix_sentinel_is_flat(self) -> None:
        # _draft_new_agent never reaches the store, but the defensive
        # ``not root`` branch must keep it flat rather than emit
        # ``AGENTS_DIR//agents/...``.
        with tempfile.TemporaryDirectory() as tmp:
            self._swap_agents_dir(Path(tmp))
            self.assertEqual(
                store.session_dir("_draft_new_agent"),
                Path(tmp) / "_draft_new_agent",
            )

    # --- new_agent_id encoding ------------------------------------------

    def test_new_agent_id_child_is_ordered_and_zero_padded(self) -> None:
        cid = store.new_agent_id(loop_id="X", stage="stage1", round_no=3, seq=2)
        self.assertTrue(cid.startswith("loop-X_stage1_r003_02_web-"), cid)

    def test_new_agent_id_flat_without_loop(self) -> None:
        aid = store.new_agent_id()
        self.assertTrue(aid.startswith("web-"))
        self.assertNotIn("_", aid)

    def test_new_agent_id_ordering_is_lexical(self) -> None:
        ids = [
            store.new_agent_id(loop_id="X", stage="stage1", round_no=r, seq=s)
            for (r, s) in [(2, 1), (10, 1), (2, 2)]
        ]
        # The order tag (stage_rNNN_SS) must sort the same as (round, seq).
        tags = [i.split("_", 1)[1].rsplit("_web-", 1)[0] for i in ids]
        self.assertEqual(
            sorted(tags),
            ["stage1_r002_01", "stage1_r002_02", "stage1_r010_01"],
        )

    # --- list_sessions / delete over the forest -------------------------

    def _seed(self, aid: str, kind: str, loop_id: str | None) -> None:
        store.save(
            store.Session(
                agent_id=aid,
                state=store.STATE_COMPLETED,
                model="m",
                workspace="/tmp/ws",
                created_at=store.now_iso(),
                loop_id=loop_id,
                kind=kind,
            )
        )

    def test_list_sessions_enumerates_nested_and_top_level(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._swap_agents_dir(Path(tmp))
            self._seed("loop-X", "loop_wrapper", "X")
            self._seed("loop-X_stage1_r001_01_web-dev", "loop_dev_round", "X")
            self._seed("loop-X_stage1_r001_02_web-rev", "loop_review", "X")
            self._seed("web-chat", "chat", None)
            ids = {s.agent_id for s in store.list_sessions()}
            self.assertEqual(
                ids,
                {
                    "loop-X",
                    "loop-X_stage1_r001_01_web-dev",
                    "loop-X_stage1_r001_02_web-rev",
                    "web-chat",
                },
            )

    def test_list_sessions_reads_legacy_flat_children(self) -> None:
        # Old archives store children flat (id has no '_'); they must
        # still enumerate from the top level.
        with tempfile.TemporaryDirectory() as tmp:
            self._swap_agents_dir(Path(tmp))
            self._seed("loop-X", "loop_wrapper", "X")
            self._seed("web-legacychild", "loop_dev_round", "X")
            ids = {s.agent_id for s in store.list_sessions()}
            self.assertEqual(ids, {"loop-X", "web-legacychild"})

    def test_delete_wrapper_removes_whole_subtree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._swap_agents_dir(Path(tmp))
            self._seed("loop-X", "loop_wrapper", "X")
            self._seed("loop-X_stage1_r001_01_web-dev", "loop_dev_round", "X")
            self.assertTrue(store.delete("loop-X"))
            self.assertEqual(store.list_sessions(), [])

    def test_spawn_session_nests_loop_child(self) -> None:
        """spawn_session threads loop/stage/round/seq into the child id so
        the session lands at the nested forest path."""
        import asyncio

        from web.agents import spawn

        class FakeProc:
            pid = 222
            stdout = None
            stderr = None
            returncode = None

        async def fake_exec(*_cmd, **_kw):
            return FakeProc()

        async def fake_read_until_init(_p, _b, _e):
            return "bsid", [], ""

        with tempfile.TemporaryDirectory() as tmp:
            self._swap_agents_dir(Path(tmp))
            with (
                mock.patch.object(spawn.asyncio, "create_subprocess_exec", fake_exec),
                mock.patch.object(spawn, "read_until_init", fake_read_until_init),
            ):
                result = asyncio.run(
                    spawn.spawn_session(
                        backend_name="cursor-cli",
                        model="m",
                        prompt="p",
                        workspace=tmp,
                        loop_id="X",
                        kind="loop_dev_round",
                        stage="stage1",
                        round_no=4,
                        seq=1,
                    )
                )
            aid = result.session.agent_id
            self.assertTrue(aid.startswith("loop-X_stage1_r004_01_web-"), aid)
            self.assertTrue(
                (Path(tmp) / "loop-X" / "agents" / "stage1" / "r004" / aid / "session.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
