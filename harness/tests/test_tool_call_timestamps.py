"""Tool call entries carry ``ts`` (start time) when the raw event has ``_ts``.

The stdout pump injects ``_ts`` into each JSON line as it arrives. The
message rebuilder surfaces that as a ``ts`` field on each tool call
entry so the frontend can display it as an HH:MM:SS badge.
"""

from __future__ import annotations

import unittest

from web.agents import messages


class TestToolCallTimestamp(unittest.TestCase):
    def test_started_event_ts_surfaces_on_tool_entry(self) -> None:
        events = [
            {"type": "user", "message": {"content": [{"type": "text", "text": "hi"}]}},
            {
                "type": "tool_call",
                "subtype": "started",
                "call_id": "tc-1",
                "tool_call": {"shellToolCall": {"args": {"command": "ls"}}},
                "_ts": 1716600937.123,
            },
            {
                "type": "tool_call",
                "subtype": "completed",
                "call_id": "tc-1",
                "tool_call": {
                    "shellToolCall": {
                        "args": {"command": "ls"},
                        "result": {"success": {"stdout": "foo"}},
                    }
                },
                "_ts": 1716600940.0,
            },
        ]
        rebuilt = messages.rebuild(events)
        assistant = next(m for m in rebuilt if m["role"] == "assistant")
        tc = assistant["_toolCalls"][0]
        self.assertEqual(tc["ts"], 1716600937.123)

    def test_completed_only_event_uses_completed_ts(self) -> None:
        """When only a completed event arrives (no started), ts comes from the completed event."""
        events = [
            {"type": "user", "message": {"content": [{"type": "text", "text": "hi"}]}},
            {
                "type": "tool_call",
                "subtype": "completed",
                "call_id": "tc-2",
                "tool_call": {
                    "readToolCall": {
                        "args": {"path": "/tmp/x"},
                        "result": {"success": {"contents": "data"}},
                    }
                },
                "_ts": 1716601000.5,
            },
        ]
        rebuilt = messages.rebuild(events)
        assistant = next(m for m in rebuilt if m["role"] == "assistant")
        tc = assistant["_toolCalls"][0]
        self.assertEqual(tc["ts"], 1716601000.5)

    def test_missing_ts_yields_none(self) -> None:
        """Legacy events without _ts gracefully produce ts=None."""
        events = [
            {"type": "user", "message": {"content": [{"type": "text", "text": "hi"}]}},
            {
                "type": "tool_call",
                "subtype": "started",
                "call_id": "tc-3",
                "tool_call": {"shellToolCall": {"args": {"command": "pwd"}}},
            },
        ]
        rebuilt = messages.rebuild(events)
        assistant = next(m for m in rebuilt if m["role"] == "assistant")
        tc = assistant["_toolCalls"][0]
        self.assertIsNone(tc["ts"])


if __name__ == "__main__":
    unittest.main()
