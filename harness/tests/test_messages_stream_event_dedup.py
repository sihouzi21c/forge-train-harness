"""Claude Code ``--include-partial-messages`` emits BOTH ``stream_event``
deltas AND periodic ``assistant`` snapshot messages for the same content.
The Rebuilder must not double-count the text.

Without the fix, every ``assistant`` snapshot appends its full accumulated
text on top of what the deltas already contributed, producing a transcript
like "Hello WorldHello World" instead of "Hello World".
"""

from __future__ import annotations

import unittest

from web.agents import messages


def _stream_text_delta(text: str, index: int = 0) -> dict:
    return {
        "type": "stream_event",
        "event": {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": text},
        },
    }


def _stream_block_start_text(index: int = 0) -> dict:
    return {
        "type": "stream_event",
        "event": {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "text", "text": ""},
        },
    }


def _stream_block_stop(index: int = 0) -> dict:
    return {
        "type": "stream_event",
        "event": {"type": "content_block_stop", "index": index},
    }


def _assistant_snapshot(text: str) -> dict:
    return {
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": text}]},
    }


def _stream_tool_start(tool_id: str = "tool-1", index: int = 1) -> dict:
    return {
        "type": "stream_event",
        "event": {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "tool_use", "id": tool_id, "name": "Bash"},
        },
    }


def _stream_tool_args(partial_json: str, index: int = 1) -> dict:
    return {
        "type": "stream_event",
        "event": {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": partial_json},
        },
    }


def _assistant_tool_snapshot(tool_id: str = "tool-1") -> dict:
    return {
        "type": "assistant",
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": "Bash",
                    "input": {"command": "ls /tmp"},
                },
            ],
        },
    }


class TestStreamEventAssistantDedup(unittest.TestCase):
    """Verify that stream_event deltas + assistant snapshot do not duplicate."""

    def test_deltas_then_assistant_snapshot_no_duplication(self) -> None:
        events = [
            {"type": "user", "message": {"content": [{"type": "text", "text": "hi"}]}},
            _stream_block_start_text(),
            _stream_text_delta("Hello "),
            _stream_text_delta("World"),
            _stream_block_stop(),
            _assistant_snapshot("Hello World"),
        ]
        rebuilt = messages.rebuild(events)
        assistant = next(m for m in rebuilt if m["role"] == "assistant")
        self.assertEqual(assistant["content"].strip(), "Hello World")

    def test_multiple_assistant_snapshots_no_duplication(self) -> None:
        events = [
            {"type": "user", "message": {"content": [{"type": "text", "text": "hi"}]}},
            _stream_block_start_text(),
            _stream_text_delta("Hello "),
            _assistant_snapshot("Hello "),
            _stream_text_delta("World"),
            _assistant_snapshot("Hello World"),
            _stream_block_stop(),
            _assistant_snapshot("Hello World"),
        ]
        rebuilt = messages.rebuild(events)
        assistant = next(m for m in rebuilt if m["role"] == "assistant")
        self.assertEqual(assistant["content"].strip(), "Hello World")

    def test_assistant_only_still_works(self) -> None:
        """Without stream_event deltas, assistant events append normally."""
        events = [
            {"type": "user", "message": {"content": [{"type": "text", "text": "hi"}]}},
            _assistant_snapshot("Hello World"),
        ]
        rebuilt = messages.rebuild(events)
        assistant = next(m for m in rebuilt if m["role"] == "assistant")
        self.assertEqual(assistant["content"].strip(), "Hello World")

    def test_cursor_cli_incremental_assistant_events(self) -> None:
        """Cursor-cli emits non-delta assistant events per chunk — append."""
        events = [
            {"type": "user", "message": {"content": [{"type": "text", "text": "hi"}]}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Hello "}]}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "World"}]}},
        ]
        rebuilt = messages.rebuild(events)
        assistant = next(m for m in rebuilt if m["role"] == "assistant")
        self.assertEqual(assistant["content"].strip(), "Hello World")

    def test_cursor_cli_delta_with_timestamp_plus_final_snapshot(self) -> None:
        """Cursor CLI emits per-token delta events (each carrying ``timestamp_ms``)
        followed by ONE final snapshot event whose ``message.content[].text`` is
        the full accumulated text and which carries NO ``timestamp_ms``.

        The snapshot is a turn-completion marker, not new content. The deltas
        have already filled the bucket; appending the snapshot's text again
        would produce a doubled message like "HelloHello".
        """
        events = [
            {"type": "user", "message": {"content": [{"type": "text", "text": "hi"}]}},
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "H"}]},
                "timestamp_ms": 1001,
            },
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "e"}]},
                "timestamp_ms": 1002,
            },
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "l"}]},
                "timestamp_ms": 1003,
            },
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "l"}]},
                "timestamp_ms": 1004,
            },
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "o"}]},
                "timestamp_ms": 1005,
            },
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Hello"}]}},
            {"type": "result", "subtype": "success"},
        ]
        rebuilt = messages.rebuild(events)
        assistant = next(m for m in rebuilt if m["role"] == "assistant")
        self.assertEqual(assistant["content"].strip(), "Hello")

    def test_cursor_cli_delta_then_snapshot_across_two_turns(self) -> None:
        """The ``_cursor_delta_seen`` marker must reset between turns so that
        a fresh turn does not inherit the previous turn's dedup state."""
        events = [
            {"type": "user", "message": {"content": [{"type": "text", "text": "q1"}]}},
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "A"}]},
                "timestamp_ms": 1001,
            },
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "1"}]},
                "timestamp_ms": 1002,
            },
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "A1"}]}},
            {"type": "result", "subtype": "success"},
            {"type": "user", "message": {"content": [{"type": "text", "text": "q2"}]}},
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "B"}]},
                "timestamp_ms": 2001,
            },
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "2"}]},
                "timestamp_ms": 2002,
            },
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "B2"}]}},
            {"type": "result", "subtype": "success"},
        ]
        rebuilt = messages.rebuild(events)
        assistants = [m for m in rebuilt if m["role"] == "assistant"]
        self.assertEqual(len(assistants), 2)
        self.assertEqual(assistants[0]["content"].strip(), "A1")
        self.assertEqual(assistants[1]["content"].strip(), "B2")

    def test_tool_use_in_assistant_snapshot_after_stream_events(self) -> None:
        """tool_use blocks inside assistant snapshots must still register."""
        events = [
            {"type": "user", "message": {"content": [{"type": "text", "text": "hi"}]}},
            _stream_block_start_text(),
            _stream_text_delta("Checking..."),
            _stream_block_stop(),
            _assistant_snapshot("Checking..."),
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "Checking..."},
                        {"type": "tool_use", "name": "Shell", "input": {"command": "ls"}},
                    ],
                },
            },
        ]
        rebuilt = messages.rebuild(events)
        assistant = next(m for m in rebuilt if m["role"] == "assistant")
        self.assertIn("Checking...", assistant["content"])
        self.assertNotIn("Checking...Checking...", assistant["content"])
        self.assertEqual(len(assistant["_toolCalls"]), 1)

    def test_tool_use_stream_then_assistant_snapshot_no_duplication(self) -> None:
        """Claude Code emits tool_use stream events plus assistant snapshots."""
        events = [
            {"type": "user", "message": {"content": [{"type": "text", "text": "hi"}]}},
            _stream_tool_start(),
            _stream_tool_args('{"command": "ls /tmp"}'),
            _stream_block_stop(index=1),
            _assistant_tool_snapshot(),
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-1",
                            "content": "done",
                        },
                    ]
                },
            },
        ]
        rebuilt = messages.rebuild(events)
        assistant = next(m for m in rebuilt if m["role"] == "assistant")
        self.assertEqual(len(assistant["_toolCalls"]), 1)
        self.assertEqual(assistant["_toolCalls"][0]["id"], "tool-1")
        self.assertEqual(assistant["_toolCalls"][0]["name"], "Bash")
        self.assertEqual(assistant["_toolCalls"][0]["result"], "done")

    def test_tool_use_snapshot_before_block_stop_no_duplication(self) -> None:
        """Snapshot can arrive BEFORE ``content_block_stop`` for the tool_use.

        Observed in real Claude Code logs when a turn ends on a tool call:
        ``content_block_start`` → ``input_json_delta`` → ``assistant`` snapshot
        → ``content_block_stop``. The snapshot must dedupe against the
        already-opened tool, and the trailing stop must finalize args on
        the existing entry instead of appending a second one.
        """
        events = [
            {"type": "user", "message": {"content": [{"type": "text", "text": "hi"}]}},
            _stream_tool_start(),
            _stream_tool_args('{"command": "ls /tmp"}'),
            _assistant_tool_snapshot(),
            _stream_block_stop(index=1),
        ]
        rebuilt = messages.rebuild(events)
        assistant = next(m for m in rebuilt if m["role"] == "assistant")
        self.assertEqual(len(assistant["_toolCalls"]), 1)
        self.assertEqual(assistant["_toolCalls"][0]["id"], "tool-1")
        self.assertEqual(assistant["_toolCalls"][0]["name"], "Bash")
        self.assertEqual(assistant["_toolCalls"][0]["args"], {"command": "ls /tmp"})

    def test_cursor_duplicate_tool_call_started_no_duplication(self) -> None:
        """Cursor CLI sometimes emits ``tool_call started`` twice with the
        same ``call_id`` (observed on the final tool of a turn) before the
        matching ``completed``. The second ``started`` must dedupe."""
        started = {
            "type": "tool_call",
            "subtype": "started",
            "call_id": "tc-1",
            "tool_call": {"shellToolCall": {"args": {"command": "ls"}}},
        }
        completed = {
            "type": "tool_call",
            "subtype": "completed",
            "call_id": "tc-1",
            "tool_call": {
                "shellToolCall": {
                    "args": {"command": "ls"},
                    "result": {"success": {"stdout": "ok\n"}},
                },
            },
        }
        events = [
            {"type": "user", "message": {"content": [{"type": "text", "text": "hi"}]}},
            started,
            started,
            completed,
        ]
        rebuilt = messages.rebuild(events)
        assistant = next(m for m in rebuilt if m["role"] == "assistant")
        self.assertEqual(len(assistant["_toolCalls"]), 1)
        self.assertEqual(assistant["_toolCalls"][0]["id"], "tc-1")
        self.assertEqual(assistant["_toolCalls"][0]["status"], "completed")
        self.assertEqual(assistant["_toolCalls"][0]["result"], "ok\n")


if __name__ == "__main__":
    unittest.main()
