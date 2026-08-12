"""Tests for ``web.routers.agent._transcript_messages_cached``.

The cache is keyed by ``(path, mtime_ns, size)`` so:
  - repeated reads of an unchanged file hit the cache (no rebuild),
  - appending bytes (the SSE writer path) misses on the next call,
  - a missing file falls through to the canonical pipeline and
    returns an empty list, never raises.
"""

from __future__ import annotations

import json
import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from web.routers import agent as agent_router


def _user_event(text: str) -> dict:
    return {"type": "user", "message": {"content": [{"type": "text", "text": text}]}}


def _write_jsonl(path: Path, events: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(e) + "\n" for e in events),
        encoding="utf-8",
    )


class TranscriptMessagesCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        # Each test starts with a clean cache so prior entries don't leak.
        with agent_router._transcript_cache_lock:
            agent_router._transcript_cache.clear()
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.log = Path(self._tmp.name) / "stdout.log"

    def test_returns_rebuilt_messages_on_first_call(self) -> None:
        _write_jsonl(self.log, [_user_event("hello")])
        result = agent_router._transcript_messages_cached(self.log)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], "user")

    def test_second_call_returns_cached_instance(self) -> None:
        _write_jsonl(self.log, [_user_event("hi")])
        first = agent_router._transcript_messages_cached(self.log)
        second = agent_router._transcript_messages_cached(self.log)
        # Same object identity → cache hit (no rebuild produced a new list).
        self.assertIs(first, second)

    def test_append_invalidates_cache(self) -> None:
        _write_jsonl(self.log, [_user_event("first")])
        first = agent_router._transcript_messages_cached(self.log)
        # macOS / Linux mtime_ns has nanosecond resolution but APFS can
        # round; force a divergent mtime so the key truly changes.
        future = time.time() + 1.0
        os.utime(self.log, (future, future))
        with self.log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_user_event("second")) + "\n")
        second = agent_router._transcript_messages_cached(self.log)
        self.assertIsNot(first, second)
        self.assertEqual(len(second), 2)

    def test_missing_file_returns_empty_without_raising(self) -> None:
        missing = Path(self._tmp.name) / "does-not-exist.log"
        result = agent_router._transcript_messages_cached(missing)
        self.assertEqual(result, [])

    def test_cache_cap_evicts_oldest(self) -> None:
        # Shrink the cap for this test then restore it.
        original_cap = agent_router._TRANSCRIPT_CACHE_CAP
        agent_router._TRANSCRIPT_CACHE_CAP = 2
        try:
            files: list[Path] = []
            for i in range(3):
                p = Path(self._tmp.name) / f"log-{i}.jsonl"
                _write_jsonl(p, [_user_event(f"msg-{i}")])
                files.append(p)
                agent_router._transcript_messages_cached(p)
            with agent_router._transcript_cache_lock:
                self.assertEqual(len(agent_router._transcript_cache), 2)
                paths_in_cache = {key[0] for key in agent_router._transcript_cache}
            # The first inserted file must have been evicted.
            self.assertNotIn(str(files[0]), paths_in_cache)
            self.assertIn(str(files[1]), paths_in_cache)
            self.assertIn(str(files[2]), paths_in_cache)
        finally:
            agent_router._TRANSCRIPT_CACHE_CAP = original_cap


if __name__ == "__main__":
    unittest.main()
