"""Loop wrapper is queryable through the same /api/agent/* surface as any other Session.

After the unified-agent-log refactor, the loop wrapper agent
(``loop-<id>``) is a real Session row on disk with
``kind=loop_wrapper``, ``backend=loop-wrapper``, and a stream-json
``stdout.log`` produced by ``agent-loop.sh`` via the
``loop_wrapper_event`` helper.

These tests pin that the existing agent HTTP API (``/api/agent/{id}/messages``)
correctly surfaces a loop wrapper's events as ``loop_event`` rows, so
the frontend Loop tab can drop its bespoke ``/api/loop/{id}/output``
ring-buffer scraping and just consume the unified pipeline.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from web.agents import spawn, store


class TestLoopWrapperViaAgentApi(unittest.TestCase):
    """A loop-<id> wrapper Session is a first-class chat target."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_agents_dir = store.AGENTS_DIR
        store.AGENTS_DIR = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(lambda: setattr(store, "AGENTS_DIR", self._orig_agents_dir))

        from web import server  # imported here to honor the patched AGENTS_DIR

        self.client = TestClient(server.app)

    def _seed_wrapper(self, loop_id: str = "abc12345") -> str:
        """Initialize a wrapper Session + append a few loop_event records."""
        spawn.init_wrapper_session(
            loop_id=loop_id,
            workspace="/tmp/ws",
            backend_label="cursor-cli",
            model="opus",
            stages="stage1 stage2",
            pid=os.getpid(),
        )
        spawn.append_loop_event(loop_id, "stage_start", {"stage": "stage1"})
        spawn.append_loop_event(loop_id, "round_start", {"stage": "stage1", "round": 1})
        spawn.append_loop_event(
            loop_id, "spawn_child", {"agent_id": "web-xxxxxxxx", "kind": "loop_dev_round"}
        )
        spawn.append_loop_event(
            loop_id, "review_verdict", {"stage": "stage1", "round": 1, "verdict": "PASS"}
        )
        return f"loop-{loop_id}"

    def test_messages_endpoint_renders_wrapper_events(self) -> None:
        agent_id = self._seed_wrapper()
        res = self.client.get(f"/api/agent/{agent_id}/messages")
        self.assertEqual(res.status_code, 200, res.text)
        payload = res.json()
        self.assertEqual(payload["agent"]["agent_id"], agent_id)
        self.assertEqual(payload["agent"]["backend"], "loop-wrapper")
        self.assertEqual(payload["agent"]["kind"], "loop_wrapper")

        msgs = payload["messages"]
        subtypes = [m.get("subtype") for m in msgs if m.get("role") == "loop_event"]
        # Wrapper-only transcript: every message is a loop_event row.
        self.assertEqual(
            [m.get("role") for m in msgs],
            ["loop_event"] * 4,
            f"unexpected non-loop_event rows: {msgs}",
        )
        self.assertEqual(
            subtypes,
            [
                "stage_start",
                "round_start",
                "spawn_child",
                "review_verdict",
            ],
        )
        # spawn_child must surface agent_id at top level so the frontend
        # can link to the child agent's chat without parsing a nested envelope.
        spawn_child_msg = next(m for m in msgs if m["subtype"] == "spawn_child")
        self.assertEqual(spawn_child_msg["agent_id"], "web-xxxxxxxx")
        self.assertEqual(spawn_child_msg["kind"], "loop_dev_round")

    def test_get_agent_endpoint_returns_wrapper_snapshot(self) -> None:
        agent_id = self._seed_wrapper("def67890")
        res = self.client.get(f"/api/agent/{agent_id}")
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertEqual(data["agent_id"], agent_id)
        self.assertEqual(data["loop_id"], "def67890")
        self.assertEqual(data["kind"], "loop_wrapper")
        self.assertEqual(data["state"], "running")

    def test_milestone_advanced_event_promotes_milestone_payload(self) -> None:
        # The Stage 1 per-milestone loop emits milestone_advanced events
        # whose top-level payload (stage/round/from/to/max/commit) must
        # land at the message root, since the Loop tab renders the
        # current milestone badge without descending into nested envelopes.
        loop_id = "mile01234567"
        spawn.init_wrapper_session(
            loop_id=loop_id,
            workspace="/tmp/ws",
            backend_label="cursor-cli",
            model="opus",
            stages="stage1",
            pid=os.getpid(),
        )
        spawn.append_loop_event(
            loop_id,
            "round_start",
            {
                "stage": "stage1",
                "round": 7,
                "milestone": "bitwise-multicard",
            },
        )
        spawn.append_loop_event(
            loop_id,
            "milestone_advanced",
            {
                "stage": "stage1",
                "round": 7,
                "from": "bitwise-multicard",
                "to": "bitwise-perf",
                "max": "production",
                "commit": "deadbeefcafe",
            },
        )
        res = self.client.get(f"/api/agent/loop-{loop_id}/messages")
        self.assertEqual(res.status_code, 200, res.text)
        msgs = res.json()["messages"]
        subtypes = [m.get("subtype") for m in msgs if m.get("role") == "loop_event"]
        self.assertIn("milestone_advanced", subtypes)

        round_msg = next(m for m in msgs if m.get("subtype") == "round_start")
        self.assertEqual(
            round_msg["milestone"],
            "bitwise-multicard",
            "round_start.milestone must surface for the active milestone badge",
        )

        adv = next(m for m in msgs if m.get("subtype") == "milestone_advanced")
        self.assertEqual(adv["stage"], "stage1")
        self.assertEqual(adv["round"], 7)
        self.assertEqual(adv["from"], "bitwise-multicard")
        self.assertEqual(adv["to"], "bitwise-perf")
        self.assertEqual(adv["max"], "production")
        self.assertEqual(adv["commit"], "deadbeefcafe")

    def test_pending_approval_event_promotes_gate_payload(self) -> None:
        # The meta loop human gate emits a pending_approval event carrying the
        # gate_file (whose path carries the loop_id) and review_bundle. Both must
        # land at the message root so the Loop tab's Approve/Reject buttons can
        # target the right loop and link the bundle — without promotion the
        # detail-view gate pill renders no actionable loop_id.
        loop_id = "gate01234567"
        gate_file = f"/x/.artifacts/meta_forge_train/{loop_id}/state/harness_configs.gate"
        bundle = "/x/workload/out/harness_config"
        spawn.init_wrapper_session(
            loop_id=loop_id,
            workspace="/tmp/ws",
            backend_label="claude-code",
            model="opus",
            stages="harness_configs",
            pid=os.getpid(),
        )
        spawn.append_loop_event(
            loop_id,
            "pending_approval",
            {"stage": "harness_configs", "gate_file": gate_file, "review_bundle": bundle},
        )
        res = self.client.get(f"/api/agent/loop-{loop_id}/messages")
        self.assertEqual(res.status_code, 200, res.text)
        msgs = res.json()["messages"]
        gate = next(m for m in msgs if m.get("subtype") == "pending_approval")
        self.assertEqual(gate["stage"], "harness_configs")
        self.assertEqual(gate["gate_file"], gate_file)
        self.assertEqual(gate["review_bundle"], bundle)


class TestAgentSseTreatsExternalSessionAsLiveUntilTerminal(unittest.TestCase):
    """The SSE event loop must not declare a loop-wrapper finished just
    because ``runner.is_running`` returns False (the wrapper is owned
    by agent-loop.sh, not the in-process runner)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_agents_dir = store.AGENTS_DIR
        store.AGENTS_DIR = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(lambda: setattr(store, "AGENTS_DIR", self._orig_agents_dir))

    def test_event_loop_continues_while_external_session_alive(self) -> None:
        from web.routers import agent as agent_router

        loop_id = "live01234567"
        spawn.init_wrapper_session(
            loop_id=loop_id,
            workspace="/tmp/ws",
            backend_label="cursor-cli",
            model="opus",
            stages="stage1",
            pid=os.getpid(),
        )
        # Seed at least one loop_event BEFORE the stream starts so the
        # initial messages frame is non-empty and we can assert it does
        # NOT include an "end" event for a still-running wrapper.
        spawn.append_loop_event(loop_id, "stage_start", {"stage": "stage1"})

        async def _drive() -> list[str]:
            collected: list[str] = []

            async def is_disconnected() -> bool:
                # Disconnect after we've collected enough frames so the
                # SSE generator exits cleanly on its own.
                return len(collected) >= 4

            stream = agent_router._event_loop(f"loop-{loop_id}", is_disconnected)
            # Pull until generator either yields end/exhausts or
            # is_disconnected fires. Each anext awaits the next yield
            # naturally, so we never cancel mid-flight.
            for _ in range(6):
                try:
                    chunk = await asyncio.wait_for(anext(stream), timeout=2.0)
                except (TimeoutError, StopAsyncIteration):
                    break
                collected.append(chunk)
            return collected

        frames = asyncio.run(_drive())
        joined = "".join(frames)
        # The wrapper session is still ``running`` on disk, so the
        # SSE must NOT have emitted an ``end`` event.
        self.assertNotIn(
            "event: end", joined, f"SSE prematurely ended a live external wrapper:\n{joined}"
        )
        # And the seeded loop_event must have been rendered as a row.
        self.assertIn(
            '"subtype": "stage_start"', joined, f"loop_event row was not streamed:\n{joined}"
        )

    def test_event_loop_ends_after_external_session_marked_terminal(self) -> None:
        from web.routers import agent as agent_router

        loop_id = "term01234567"
        spawn.init_wrapper_session(
            loop_id=loop_id,
            workspace="/tmp/ws",
            backend_label="cursor-cli",
            model="opus",
            stages="stage1",
            pid=88888,
        )
        spawn.finalize_wrapper_session(
            loop_id=loop_id,
            exit_code=0,
            state=store.STATE_COMPLETED,
        )

        async def _drive() -> list[str]:
            collected: list[str] = []

            async def is_disconnected() -> bool:
                return False

            stream = agent_router._event_loop(f"loop-{loop_id}", is_disconnected)
            for _ in range(6):
                try:
                    chunk = await asyncio.wait_for(anext(stream), timeout=1.0)
                except (TimeoutError, StopAsyncIteration):
                    break
                collected.append(chunk)
            return collected

        frames = asyncio.run(_drive())
        self.assertIn(
            "event: end", "".join(frames), f"terminal external session must close stream:\n{frames}"
        )


class TestAgentSseDeltaPayload(unittest.TestCase):
    """Delta SSE mode should not resend the whole rebuilt transcript."""

    def test_delta_payload_replaces_only_mutating_tail_message(self) -> None:
        from web.agents import messages
        from web.routers import agent as agent_router

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stdout.log"
            records = [
                {"type": "web_user", "message": {"content": [{"type": "text", "text": "one"}]}},
                {"type": "assistant", "message": {"content": [{"type": "text", "text": "first"}]}},
                {"type": "result", "subtype": "success"},
                {"type": "web_user", "message": {"content": [{"type": "text", "text": "two"}]}},
                {"type": "assistant", "message": {"content": [{"type": "text", "text": "sec"}]}},
            ]
            path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            builder = messages.Rebuilder()
            first_payload, offset, advanced = agent_router._read_feed_encode(
                path,
                builder,
                0,
                delta=True,
                previous_message_count=0,
            )
            first = json.loads(first_payload)
            self.assertEqual(advanced, len(records))
            self.assertEqual(offset, path.stat().st_size)
            self.assertEqual(first["replace_from"], 0)
            self.assertEqual(
                [m["content"] for m in first["messages"] if m["role"] == "user"], ["one", "two"]
            )

            with path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "type": "assistant",
                            "message": {"content": [{"type": "text", "text": "ond"}]},
                        }
                    )
                    + "\n"
                )

            second_payload, _offset, advanced = agent_router._read_feed_encode(
                path,
                builder,
                offset,
                delta=True,
                previous_message_count=len(first["messages"]),
            )
            second = json.loads(second_payload)
            self.assertEqual(advanced, 1)
            self.assertEqual(second["replace_from"], 3)
            self.assertEqual(len(second["messages"]), 1)
            self.assertEqual(second["messages"][0]["role"], "assistant")
            self.assertEqual(second["messages"][0]["content"], "second")
            self.assertNotIn("one", json.dumps(second["messages"]))


if __name__ == "__main__":
    unittest.main()
