"""Frontend renderer learns ``loop_event`` so wrapper-side orchestration
text shows up in the same chat view as cursor-cli / claude assistant
output.

A loop wrapper Session (``backend=loop-wrapper``, ``kind=loop_wrapper``)
owns a ``stdout.log`` whose lines are:

* one ``{"type":"system","subtype":"init", ...}`` header (the seed
  written by :mod:`harness.tools.loop_wrapper_init`)
* one or more ``{"type":"loop_event","subtype":"<sub>", ...}`` records
  appended by :mod:`harness.tools.loop_wrapper_event` from
  ``agent-loop.sh``

``messages.rebuild`` must turn each ``loop_event`` into one well-typed
message the JS layer can render as a section header / spawn-child card
/ verdict pill, instead of dropping it or shoving it into an
``assistant`` text content field.
"""

from __future__ import annotations

import unittest

from web.agents import messages


def _evt(subtype: str, **payload: object) -> dict:
    return {"type": "loop_event", "subtype": subtype, "ts": 0.0, **payload}


class TestLoopEventRendering(unittest.TestCase):
    def test_stage_and_round_boundaries_render_as_dedicated_messages(self) -> None:
        events = [
            {"type": "system", "subtype": "init", "session_id": "loop-abc"},
            _evt("stage_start", stage="stage1"),
            _evt("round_start", stage="stage1", round=1),
        ]
        rebuilt = messages.rebuild(events)
        roles = [m.get("role") for m in rebuilt]
        # The system init is intentionally silent; loop_events are not.
        self.assertEqual(roles, ["loop_event", "loop_event"])
        self.assertEqual(rebuilt[0]["subtype"], "stage_start")
        self.assertEqual(rebuilt[0]["stage"], "stage1")
        self.assertEqual(rebuilt[1]["subtype"], "round_start")
        self.assertEqual(rebuilt[1]["round"], 1)

    def test_spawn_child_carries_link_metadata(self) -> None:
        events = [
            {"type": "system", "subtype": "init", "session_id": "loop-abc"},
            _evt("spawn_child", agent_id="web-deadbeef", kind="loop_dev_round"),
        ]
        rebuilt = messages.rebuild(events)
        self.assertEqual(len(rebuilt), 1)
        msg = rebuilt[0]
        self.assertEqual(msg["role"], "loop_event")
        self.assertEqual(msg["subtype"], "spawn_child")
        # The renderer MUST surface agent_id at the top level so the JS
        # side can render a clickable child-agent card without parsing
        # any nested payload key.
        self.assertEqual(msg["agent_id"], "web-deadbeef")
        self.assertEqual(msg["kind"], "loop_dev_round")

    def test_review_verdict_pass_and_fail(self) -> None:
        events = [
            {"type": "system", "subtype": "init", "session_id": "loop-abc"},
            _evt("review_verdict", stage="stage1", round=2, verdict="PASS"),
            _evt("review_verdict", stage="stage1", round=3, verdict="FAIL"),
        ]
        rebuilt = messages.rebuild(events)
        self.assertEqual(len(rebuilt), 2)
        self.assertEqual(rebuilt[0]["verdict"], "PASS")
        self.assertEqual(rebuilt[1]["verdict"], "FAIL")
        self.assertEqual(rebuilt[1]["round"], 3)

    def test_loop_events_do_not_pollute_assistant_turn(self) -> None:
        """A wrapper-only transcript MUST NOT produce an assistant
        message with empty text just because loop_events arrived without
        a preceding ``user`` event. The old code would have opened an
        assistant turn on any event; loop_events must stay neutral."""
        events = [
            {"type": "system", "subtype": "init", "session_id": "loop-abc"},
            _evt("stage_start", stage="stage1"),
            _evt("round_start", stage="stage1", round=1),
            _evt("loop_exit", state="completed", exit_code=0),
        ]
        rebuilt = messages.rebuild(events)
        roles = {m.get("role") for m in rebuilt}
        self.assertEqual(
            roles,
            {"loop_event"},
            f"wrapper transcript must contain only loop_event rows; got {roles}",
        )

    def test_info_subtype_carries_text(self) -> None:
        events = [
            {"type": "system", "subtype": "init", "session_id": "loop-abc"},
            _evt("info", text="anything not yet typed"),
        ]
        rebuilt = messages.rebuild(events)
        self.assertEqual(rebuilt[0]["subtype"], "info")
        self.assertEqual(rebuilt[0]["text"], "anything not yet typed")


if __name__ == "__main__":
    unittest.main()
