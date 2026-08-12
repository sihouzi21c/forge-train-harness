"""``GET /api/loop/{loop_id}/active`` resolves the currently-driven child agent.

This endpoint powers the frontend "Follow Live" mode when the user
opens a loop tab mid-run: the scheduler must locate which child
agent the wrapper is currently shelling out to without waiting for
the next ``spawn_child`` SSE frame.

The handler walks the wrapper's stdout.log in reverse looking for the
most recent ``spawn_child`` whose target Session is still in a live
state. Children that have already terminated are skipped.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from web.agents import spawn, store


class TestLoopActiveEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_agents_dir = store.AGENTS_DIR
        store.AGENTS_DIR = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(lambda: setattr(store, "AGENTS_DIR", self._orig_agents_dir))

        from web import server

        self.client = TestClient(server.app)

        from web.routers import loop as loop_router

        self.loop_router = loop_router

    def _seed_loop_in_router(self, loop_id: str) -> None:
        self.loop_router._instances[loop_id] = self.loop_router.LoopInstance(
            loop_id=loop_id, status="running", workspace_dir="/tmp/ws"
        )
        self.loop_router._instance_dir(loop_id).mkdir(parents=True, exist_ok=True)

    def _seed_wrapper(self, loop_id: str) -> None:
        spawn.init_wrapper_session(
            loop_id=loop_id,
            workspace="/tmp/ws",
            backend_label="cursor-cli",
            model="opus",
            stages="stage1",
        )

    def _seed_child(self, agent_id: str, *, state: str, loop_id: str, kind: str) -> None:
        sess = store.Session(
            agent_id=agent_id,
            state=state,
            model="opus",
            workspace="/tmp/ws",
            created_at=store.now_iso(),
            backend="cursor-cli",
            backend_session_id=agent_id,
            loop_id=loop_id,
            kind=kind,
        )
        store.save(sess)

    def test_returns_latest_running_child(self) -> None:
        loop_id = "deadbeef"
        self._seed_loop_in_router(loop_id)
        self._seed_wrapper(loop_id)

        # First child finished, second is still running. The endpoint
        # must skip the terminated one and return the running one.
        self._seed_child(
            "web-aaaaaaaa", state=store.STATE_COMPLETED, loop_id=loop_id, kind="loop_dev_round"
        )
        self._seed_child(
            "web-bbbbbbbb", state=store.STATE_RUNNING, loop_id=loop_id, kind="loop_review"
        )

        spawn.append_loop_event(
            loop_id,
            "spawn_child",
            {"agent_id": "web-aaaaaaaa", "kind": "loop_dev_round"},
        )
        spawn.append_loop_event(
            loop_id,
            "spawn_child",
            {"agent_id": "web-bbbbbbbb", "kind": "loop_review"},
        )

        res = self.client.get(f"/api/loop/{loop_id}/active")
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertEqual(data["loop_id"], loop_id)
        self.assertEqual(data["agent_id"], "web-bbbbbbbb")
        self.assertEqual(data["kind"], "loop_review")
        self.assertIsNotNone(data["since_ts"])

    def test_skips_terminated_child_and_returns_null_when_idle(self) -> None:
        loop_id = "deadbeef"
        self._seed_loop_in_router(loop_id)
        self._seed_wrapper(loop_id)

        self._seed_child(
            "web-cccccccc", state=store.STATE_COMPLETED, loop_id=loop_id, kind="loop_dev_round"
        )
        spawn.append_loop_event(
            loop_id,
            "spawn_child",
            {"agent_id": "web-cccccccc", "kind": "loop_dev_round"},
        )

        res = self.client.get(f"/api/loop/{loop_id}/active")
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertIsNone(data["agent_id"])
        self.assertIsNone(data["kind"])

    def test_missing_child_session_is_skipped(self) -> None:
        # spawn_child landed in the wrapper log but the child Session was
        # cleaned up (e.g. the agent dir got rm -rf'd). The endpoint must
        # tolerate the gap and not 500.
        loop_id = "deadbeef"
        self._seed_loop_in_router(loop_id)
        self._seed_wrapper(loop_id)
        spawn.append_loop_event(
            loop_id,
            "spawn_child",
            {"agent_id": "web-vanished", "kind": "loop_dev_round"},
        )
        res = self.client.get(f"/api/loop/{loop_id}/active")
        self.assertEqual(res.status_code, 200, res.text)
        self.assertIsNone(res.json()["agent_id"])

    def test_unknown_loop_id_is_404(self) -> None:
        res = self.client.get("/api/loop/missingloop/active")
        self.assertEqual(res.status_code, 404)


if __name__ == "__main__":
    unittest.main()
