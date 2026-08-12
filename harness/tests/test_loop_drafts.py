"""Stale-draft reaper for the Loop Create surface.

Opening the Loop Create tab POSTs ``/api/loop/drafts`` which calls
``_provision_workspace`` and copies ``harness/`` into
``.artifacts/forge_train/<loop_id>/workspace/``. Drafts that are never
started (user closes the tab, reloads the browser, etc.) accumulate
on disk and in the ``_instances`` map.

The reaper drops every draft whose ``session.json`` has been
idle (no writes) for more than ``_DRAFT_STALE_SECONDS`` (4 hours).
The "idle" semantics specifically use mtime - not creation time -
so a draft that the user is actively editing (every TOML save and
every chat turn calls ``_save_session`` and bumps mtime) keeps
itself alive without any explicit ping endpoint.

Reaping runs at exactly two well-defined moments:

1. Module import (right after ``_load_history()``) so a server
   restart self-cleans before the first list request.
2. The top of ``create_loop_draft`` so back-to-back tab opens
   cannot accumulate.

We never reap from inside ``list_loops`` - GET should not perform
filesystem mutations.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


class _DraftHarness:
    """Isolate ``web.routers.loop`` state for a single test.

    Patches ``FORGE_TRAIN_DIR`` to a tempdir and clears ``_instances`` so
    nothing leaks across tests or into the real ``.artifacts/``.
    """

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

    def make_draft(
        self,
        loop_id: str,
        *,
        status: str = "draft",
        idle_seconds: float = 0.0,
    ) -> Path:
        """Materialize a fake draft instance on disk + in ``_instances``.

        ``idle_seconds`` rewinds ``session.json`` mtime so the reaper
        sees an old draft without us having to actually wait.
        """
        instance_dir = self.loops_dir / loop_id
        workspace = instance_dir / "workspace"
        (workspace / "config").mkdir(parents=True)
        (workspace / "config" / "agent.toml").write_text(
            '[agent]\nbackend = "cursor-cli"\n', encoding="utf-8"
        )
        inst = self.loop_router.LoopInstance(
            loop_id=loop_id,
            mode="managed",
            status=status,
            workspace_dir=str(workspace),
        )
        self.loop_router._instances[loop_id] = inst
        # Persist a real session.json so the reaper can stat() it.
        self.loop_router._save_session(inst)
        if idle_seconds > 0:
            session_path = self.loop_router._session_file(loop_id)
            past = time.time() - idle_seconds
            os.utime(session_path, (past, past))
        return instance_dir


class TestDraftLastActiveIncludesAgentActivity(unittest.TestCase):
    """A draft hosts a Loop Create chat agent. The agent's
    transcript / session updates do NOT touch the draft's
    own ``session.json``, so reaping purely off
    ``_session_file(loop_id).stat().st_mtime`` would evict a
    draft that the user is still actively chatting with.

    The reaper MUST treat any managed agent whose ``workspace``
    points into the draft's ``workspace_dir`` as evidence of
    activity. Specifically, the latest mtime among:

        * ``_session_file(loop_id)``        (TOML edits)
        * each agent's ``session.json``     (state transitions)
        * each agent's ``stdout.log``       (chat streamed reply)

    where ``agent.workspace`` resolves into the draft's
    instance directory.

    This is the SSOT for "the user is still using this draft" —
    no per-event ping endpoint is required.
    """

    def setUp(self) -> None:
        self.h = _DraftHarness()
        self.addCleanup(self.h.close)
        # Re-point the agent store at a tempdir for the duration
        # of each test so we never write into real .artifacts/.
        from web import paths as web_paths
        from web.agents import store as agent_store

        self.agent_store = agent_store
        self._agents_dir = self.h.loops_dir.parent / "web-agents"
        self._agents_dir.mkdir()
        # Patch both the canonical location (web.paths.AGENTS_DIR,
        # which _draft_last_active imports) AND the agent_store
        # re-export so any helper consulting either sees the temp dir.
        self._paths_patch = mock.patch.object(web_paths, "AGENTS_DIR", self._agents_dir)
        self._paths_patch.start()
        self.addCleanup(self._paths_patch.stop)
        self._agents_patch = mock.patch.object(agent_store, "AGENTS_DIR", self._agents_dir)
        self._agents_patch.start()
        self.addCleanup(self._agents_patch.stop)

    def _make_agent_for_draft(
        self,
        *,
        agent_id: str,
        draft_workspace: Path,
        idle_seconds: float,
    ) -> Path:
        """Materialize a fake managed agent that points at the draft."""
        adir = self._agents_dir / agent_id
        adir.mkdir(parents=True)
        (adir / "session.json").write_text(
            f'{{"agent_id":"{agent_id}","workspace":"{draft_workspace}"}}',
            encoding="utf-8",
        )
        (adir / "stdout.log").write_text("chat reply\n", encoding="utf-8")
        if idle_seconds:
            past = time.time() - idle_seconds
            for f in adir.iterdir():
                os.utime(f, (past, past))
        return adir

    def test_draft_with_recent_chat_survives_even_when_session_is_old(
        self,
    ) -> None:
        """Reaper must NOT evict a draft whose session.json is old
        if any associated chat agent has recently appended output."""
        threshold = self.h.loop_router._DRAFT_STALE_SECONDS
        # session.json is OLD — way past the threshold.
        instance_dir = self.h.make_draft("draft-with-chat", idle_seconds=threshold + 600)
        draft_workspace = Path(self.h.loop_router._instances["draft-with-chat"].workspace_dir)
        # But there's a chat agent on this draft that wrote to its
        # stdout.log five minutes ago - the user IS using this draft.
        self._make_agent_for_draft(
            agent_id="agent-fresh",
            draft_workspace=draft_workspace,
            idle_seconds=5 * 60,
        )

        reaped = self.h.loop_router._reap_stale_drafts()

        self.assertEqual(
            reaped,
            [],
            "draft with a recent-active chat agent must survive even "
            "when its session.json mtime is past the threshold",
        )
        self.assertIn("draft-with-chat", self.h.loop_router._instances)
        self.assertTrue(instance_dir.is_dir())

    def test_draft_with_only_old_chat_is_still_reaped(self) -> None:
        """If every signal (session.json AND every agent) is stale,
        the draft is genuinely dead and should be reaped."""
        threshold = self.h.loop_router._DRAFT_STALE_SECONDS
        instance_dir = self.h.make_draft("stale-with-stale-chat", idle_seconds=threshold + 600)
        draft_workspace = Path(self.h.loop_router._instances["stale-with-stale-chat"].workspace_dir)
        self._make_agent_for_draft(
            agent_id="agent-stale",
            draft_workspace=draft_workspace,
            idle_seconds=threshold + 60,
        )

        reaped = self.h.loop_router._reap_stale_drafts()

        self.assertEqual(reaped, ["stale-with-stale-chat"])
        self.assertNotIn("stale-with-stale-chat", self.h.loop_router._instances)
        self.assertFalse(instance_dir.exists())

    def test_unrelated_agent_activity_does_not_save_a_stale_draft(
        self,
    ) -> None:
        """A fresh agent whose workspace is in some OTHER directory
        (not this draft) is not evidence of this draft being used."""
        threshold = self.h.loop_router._DRAFT_STALE_SECONDS
        instance_dir = self.h.make_draft("unrelated-draft", idle_seconds=threshold + 600)
        # Agent points at a sibling workspace, not at this draft.
        other_workspace = self.h.loops_dir / "some-other-loop" / "workspace"
        other_workspace.mkdir(parents=True)
        self._make_agent_for_draft(
            agent_id="agent-elsewhere",
            draft_workspace=other_workspace,
            idle_seconds=5,
        )

        reaped = self.h.loop_router._reap_stale_drafts()

        self.assertEqual(reaped, ["unrelated-draft"])
        self.assertFalse(instance_dir.exists())


class TestReapStaleDrafts(unittest.TestCase):
    """``_reap_stale_drafts`` only touches idle drafts."""

    def setUp(self) -> None:
        self.h = _DraftHarness()
        self.addCleanup(self.h.close)

    def test_returns_empty_when_no_instances(self) -> None:
        self.assertEqual(self.h.loop_router._reap_stale_drafts(), [])

    def test_drops_draft_idle_past_threshold(self) -> None:
        threshold = self.h.loop_router._DRAFT_STALE_SECONDS
        instance_dir = self.h.make_draft("aaaaaaaaaaaa", idle_seconds=threshold + 60)
        self.assertTrue(instance_dir.is_dir())

        reaped = self.h.loop_router._reap_stale_drafts()

        self.assertEqual(reaped, ["aaaaaaaaaaaa"])
        self.assertNotIn("aaaaaaaaaaaa", self.h.loop_router._instances)
        self.assertFalse(
            instance_dir.exists(),
            "stale draft workspace must be removed from disk",
        )

    def test_keeps_recently_touched_draft(self) -> None:
        threshold = self.h.loop_router._DRAFT_STALE_SECONDS
        # Idle for half the threshold - well within the active window.
        instance_dir = self.h.make_draft("bbbbbbbbbbbb", idle_seconds=threshold / 2)

        reaped = self.h.loop_router._reap_stale_drafts()

        self.assertEqual(reaped, [])
        self.assertIn("bbbbbbbbbbbb", self.h.loop_router._instances)
        self.assertTrue(instance_dir.is_dir())

    def test_never_touches_non_draft_status(self) -> None:
        # A 1-week-old completed loop must stay forever.
        ancient = 7 * 24 * 60 * 60
        for status in ("running", "completed", "failed", "stopped"):
            with self.subTest(status=status):
                loop_id = f"keep-{status}"
                instance_dir = self.h.make_draft(loop_id, status=status, idle_seconds=ancient)

                self.h.loop_router._reap_stale_drafts()

                self.assertIn(loop_id, self.h.loop_router._instances)
                self.assertTrue(
                    instance_dir.is_dir(),
                    f"{status} loop must not be reaped",
                )

    def test_reaps_draft_even_if_session_file_is_missing(self) -> None:
        """A draft whose session.json was already deleted is by
        definition more stale than the threshold; reap it instead
        of raising on the missing stat()."""
        threshold = self.h.loop_router._DRAFT_STALE_SECONDS
        instance_dir = self.h.make_draft("cccccccccccc", idle_seconds=threshold + 60)
        self.h.loop_router._session_file("cccccccccccc").unlink()

        reaped = self.h.loop_router._reap_stale_drafts()

        self.assertEqual(reaped, ["cccccccccccc"])
        self.assertNotIn("cccccccccccc", self.h.loop_router._instances)
        self.assertFalse(instance_dir.exists())

    def test_threshold_is_four_hours(self) -> None:
        # Pin the chosen policy so future drift is visible in diffs.
        self.assertEqual(self.h.loop_router._DRAFT_STALE_SECONDS, 4 * 60 * 60)


class TestCreateLoopDraftReapsBeforeCreating(unittest.TestCase):
    """``POST /api/loop/drafts`` evicts stale siblings before
    provisioning a new workspace, so the list never piles up."""

    def setUp(self) -> None:
        self.h = _DraftHarness()
        self.addCleanup(self.h.close)

    def test_stale_draft_is_reaped_when_new_draft_is_created(self) -> None:
        threshold = self.h.loop_router._DRAFT_STALE_SECONDS
        stale_dir = self.h.make_draft("dddddddddddd", idle_seconds=threshold + 60)
        fresh_dir = self.h.make_draft("eeeeeeeeeeee", idle_seconds=threshold / 4)

        captured: dict[str, Path] = {}

        def _fake_provision(loop_id: str, **_kwargs) -> Path:
            workspace = self.h.loops_dir / loop_id / "workspace"
            (workspace / "config").mkdir(parents=True)
            (workspace / "config" / "agent.toml").write_text(
                '[agent]\nbackend = "cursor-cli"\n', encoding="utf-8"
            )
            captured["workspace"] = workspace
            return workspace

        with mock.patch.object(self.loop_module, "_provision_workspace", _fake_provision):
            snapshot = asyncio.run(self.loop_module.create_loop_draft())

        self.assertEqual(snapshot["status"], "draft")
        new_loop_id = snapshot["loop_id"]

        self.assertNotIn("dddddddddddd", self.h.loop_router._instances)
        self.assertFalse(stale_dir.exists())

        self.assertIn("eeeeeeeeeeee", self.h.loop_router._instances)
        self.assertTrue(fresh_dir.is_dir())

        self.assertIn(new_loop_id, self.h.loop_router._instances)
        self.assertEqual(captured["workspace"].parent.name, new_loop_id)

    @property
    def loop_module(self):
        return self.h.loop_router


class TestLoadHistoryReapsOnStartup(unittest.TestCase):
    """``_load_history()`` followed by ``_reap_stale_drafts()`` cleans
    drafts that survived a previous process. The two functions chain
    at module import time - we exercise that contract here.
    """

    def setUp(self) -> None:
        self.h = _DraftHarness()
        self.addCleanup(self.h.close)

    def _seed_on_disk(self, loop_id: str, *, status: str, idle_seconds: float) -> Path:
        instance_dir = self.h.loops_dir / loop_id
        workspace = instance_dir / "workspace"
        (workspace / "config").mkdir(parents=True)
        (workspace / "config" / "agent.toml").write_text(
            '[agent]\nbackend = "cursor-cli"\n', encoding="utf-8"
        )
        session = instance_dir / "session.json"
        import json

        session.write_text(
            json.dumps(
                {
                    "loop_id": loop_id,
                    "label": "",
                    "mode": "managed",
                    "status": status,
                    "workspace_dir": str(workspace),
                }
            ),
            encoding="utf-8",
        )
        past = time.time() - idle_seconds
        os.utime(session, (past, past))
        return instance_dir

    def test_startup_reaper_drops_stale_draft_from_disk(self) -> None:
        threshold = self.h.loop_router._DRAFT_STALE_SECONDS
        stale_dir = self._seed_on_disk("ffffffffffff", status="draft", idle_seconds=threshold + 60)
        fresh_dir = self._seed_on_disk("gggggggggggg", status="draft", idle_seconds=threshold / 4)
        completed_dir = self._seed_on_disk(
            "hhhhhhhhhhhh",
            status="completed",
            idle_seconds=7 * 24 * 60 * 60,
        )

        # Reload from disk, then reap. This mirrors module-import order:
        # _load_history() first, _reap_stale_drafts() right after.
        self.h.loop_router._load_history()
        self.h.loop_router._reap_stale_drafts()

        self.assertNotIn("ffffffffffff", self.h.loop_router._instances)
        self.assertFalse(stale_dir.exists())

        self.assertIn("gggggggggggg", self.h.loop_router._instances)
        self.assertTrue(fresh_dir.is_dir())

        self.assertIn("hhhhhhhhhhhh", self.h.loop_router._instances)
        self.assertTrue(completed_dir.is_dir())


class TestListLoopsOmitsDrafts(unittest.TestCase):
    """``GET /api/loop`` is the user-visible loop list. Drafts are the
    Loop Create tab's internal scratch state and MUST NOT appear there.

    The draft surface has its own SSOT (``loopCreateLoopId`` carried by
    the front-end and the ``/drafts`` + ``/configs`` endpoints), so the
    list endpoint omitting drafts removes a redundant rendering path
    without losing any capability.
    """

    def setUp(self) -> None:
        self.h = _DraftHarness()
        self.addCleanup(self.h.close)

    def test_list_returns_only_non_draft_instances(self) -> None:
        self.h.make_draft("draft-active", status="draft")
        self.h.make_draft("running-loop", status="running")
        self.h.make_draft("completed-loop", status="completed")
        self.h.make_draft("failed-loop", status="failed")
        self.h.make_draft("stopped-loop", status="stopped")

        result = asyncio.run(self.h.loop_router.list_loops())

        returned_ids = {item["loop_id"] for item in result["loops"]}
        self.assertNotIn("draft-active", returned_ids)
        self.assertEqual(
            returned_ids,
            {"running-loop", "completed-loop", "failed-loop", "stopped-loop"},
        )
        # Counters reflect the same filtered view so the UI's
        # "{N} instance(s)" badge stays consistent with the list.
        self.assertEqual(result["total"], 4)
        self.assertEqual(result["running"], 1)

    def test_list_excludes_drafts_even_when_they_are_the_only_instances(self) -> None:
        self.h.make_draft("only-draft-1", status="draft")
        self.h.make_draft("only-draft-2", status="draft")

        result = asyncio.run(self.h.loop_router.list_loops())

        self.assertEqual(result["loops"], [])
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["running"], 0)

    def test_drafts_remain_addressable_individually(self) -> None:
        """List omits drafts but ``GET /api/loop/{loop_id}`` and the
        ``/configs`` family must still work - the Loop Create tab
        relies on those direct lookups to resume an in-progress draft.
        """
        self.h.make_draft("draft-resume", status="draft")

        snapshot = asyncio.run(self.h.loop_router.get_loop("draft-resume"))
        self.assertEqual(snapshot["loop_id"], "draft-resume")
        self.assertEqual(snapshot["status"], "draft")


if __name__ == "__main__":
    unittest.main()
