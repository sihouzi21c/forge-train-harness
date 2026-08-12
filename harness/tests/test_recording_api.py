"""HTTP contract tests for the per-loop browser-recording API.

Validates the recorder/chunk/event lifecycle that the rrweb-based
frontend recorder depends on:

* ``POST /start`` creates ``recording/recorders/<id>/info.json``
* ``POST /chunks`` appends monotonically-numbered chunk files
* Multiple ``recorder_id``s can write the same loop concurrently
  without overwriting each other's chunks
* ``GET /events`` merges every recorder's events sorted by rrweb
  ``timestamp`` and de-dupes adjacent identical events
* ``DELETE`` wipes the recording subtree
* ``loop_id`` deletion cascades into the recording subtree via the
  loop router's ``store.delete(f"loop-{id}")`` call

The tests are importer-style (no real running uvicorn) so they exercise
the Python request handlers in isolation. The recording subtree lives
under ``store.AGENTS_DIR`` which we redirect to a tmp directory, the
same trick :mod:`test_loop_wrapper_messages_api` uses.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from web.agents import store
from web.routers import recording


class _RecordingApiBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_agents_dir = store.AGENTS_DIR
        store.AGENTS_DIR = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(lambda: setattr(store, "AGENTS_DIR", self._orig_agents_dir))

        from web import server  # imported here so the patched AGENTS_DIR sticks

        self.client = TestClient(server.app)
        # web/routers/loop.py keeps an in-memory _instances mapping that the
        # delete endpoint walks; seed it directly so the delete test
        # exercises the cascade without a full bootstrap.
        from web.routers import loop as loop_router

        self._loop_router = loop_router

    def _recording_dir(self, loop_id: str) -> Path:
        return recording._recording_dir(loop_id)

    def _recorders_dir(self, loop_id: str) -> Path:
        return recording._recorders_dir(loop_id)


class TestRecordingLifecycle(_RecordingApiBase):
    def test_start_creates_meta_and_recorder_dir(self) -> None:
        loop_id = "abc12345"
        res = self.client.post(
            f"/api/loop/{loop_id}/recording/start",
            json={
                "recorder_id": "tab-A",
                "viewport": {"width": 1280, "height": 720, "dpr": 2},
                "user_agent": "test-agent/1.0",
            },
        )
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertEqual(data["recorder_id"], "tab-A")
        self.assertGreater(data["started_at"], 0)

        meta_path = self._recording_dir(loop_id) / "meta.json"
        self.assertTrue(meta_path.is_file())

        info_path = self._recorders_dir(loop_id) / "tab-A" / "info.json"
        self.assertTrue(info_path.is_file())
        info = json.loads(info_path.read_text(encoding="utf-8"))
        self.assertEqual(info["recorder_id"], "tab-A")
        self.assertIsNone(info["ended_at"])
        self.assertEqual(info["chunk_count"], 0)
        self.assertEqual(info["viewport"]["width"], 1280)

    def test_chunks_append_monotonically(self) -> None:
        loop_id = "abc12345"
        self.client.post(
            f"/api/loop/{loop_id}/recording/start",
            json={"recorder_id": "tab-A"},
        )
        for i in range(3):
            res = self.client.post(
                f"/api/loop/{loop_id}/recording/chunks",
                json={
                    "recorder_id": "tab-A",
                    "events": [{"type": 0, "timestamp": 100 + i}],
                },
            )
            self.assertEqual(res.status_code, 200, res.text)
            self.assertEqual(res.json()["chunk_index"], i + 1)
            self.assertEqual(res.json()["event_count"], 1)

        rec_dir = self._recorders_dir(loop_id) / "tab-A"
        chunks = sorted(rec_dir.glob("chunk-*.json"))
        self.assertEqual(
            [p.name for p in chunks], ["chunk-0001.json", "chunk-0002.json", "chunk-0003.json"]
        )

        info = json.loads((rec_dir / "info.json").read_text(encoding="utf-8"))
        self.assertEqual(info["chunk_count"], 3)

    def test_chunks_lazily_create_recorder_dir_when_start_skipped(self) -> None:
        # Real-world flaky network: a tab reloads, posts /chunks before
        # /start makes it through. Server must accept and bootstrap the
        # recorder slot lazily so events are not orphaned.
        loop_id = "abc12345"
        res = self.client.post(
            f"/api/loop/{loop_id}/recording/chunks",
            json={"recorder_id": "lazy-tab", "events": [{"type": 4, "timestamp": 1}]},
        )
        self.assertEqual(res.status_code, 200, res.text)
        info = json.loads(
            (self._recorders_dir(loop_id) / "lazy-tab" / "info.json").read_text(encoding="utf-8")
        )
        self.assertEqual(info["chunk_count"], 1)

    def test_events_endpoint_merges_recorders_by_timestamp(self) -> None:
        loop_id = "abc12345"
        self.client.post(
            f"/api/loop/{loop_id}/recording/start",
            json={"recorder_id": "A"},
        )
        self.client.post(
            f"/api/loop/{loop_id}/recording/start",
            json={"recorder_id": "B"},
        )
        # Interleaved timestamps across two recorders.
        self.client.post(
            f"/api/loop/{loop_id}/recording/chunks",
            json={
                "recorder_id": "A",
                "events": [
                    {"type": 4, "timestamp": 100, "data": {"x": 1}},
                    {"type": 4, "timestamp": 300, "data": {"x": 3}},
                ],
            },
        )
        self.client.post(
            f"/api/loop/{loop_id}/recording/chunks",
            json={
                "recorder_id": "B",
                "events": [
                    {"type": 4, "timestamp": 200, "data": {"y": 2}},
                    # Identical to A's timestamp 100 to exercise dedup.
                    {"type": 4, "timestamp": 100, "data": {"x": 1}},
                ],
            },
        )

        res = self.client.get(f"/api/loop/{loop_id}/recording/events")
        self.assertEqual(res.status_code, 200, res.text)
        events = res.json()["events"]
        self.assertEqual(
            [(ev["timestamp"], ev.get("data", {})) for ev in events],
            [(100, {"x": 1}), (200, {"y": 2}), (300, {"x": 3})],
            f"unexpected merge order or dedup: {events}",
        )

    def test_meta_reports_recorders_and_chunk_total(self) -> None:
        loop_id = "abc12345"
        self.client.post(f"/api/loop/{loop_id}/recording/start", json={"recorder_id": "A"})
        self.client.post(f"/api/loop/{loop_id}/recording/start", json={"recorder_id": "B"})
        self.client.post(
            f"/api/loop/{loop_id}/recording/chunks",
            json={"recorder_id": "A", "events": [{"type": 4, "timestamp": 1}]},
        )
        self.client.post(
            f"/api/loop/{loop_id}/recording/chunks",
            json={"recorder_id": "A", "events": [{"type": 4, "timestamp": 2}]},
        )
        self.client.post(
            f"/api/loop/{loop_id}/recording/chunks",
            json={"recorder_id": "B", "events": [{"type": 4, "timestamp": 3}]},
        )

        res = self.client.get(f"/api/loop/{loop_id}/recording/meta")
        data = res.json()
        self.assertTrue(data["exists"])
        self.assertEqual(data["chunk_count"], 3)
        rec_ids = sorted(r["recorder_id"] for r in data["recorders"])
        self.assertEqual(rec_ids, ["A", "B"])
        self.assertFalse(data["has_mp4"])

    def test_stop_stamps_ended_at(self) -> None:
        loop_id = "abc12345"
        self.client.post(f"/api/loop/{loop_id}/recording/start", json={"recorder_id": "A"})
        res = self.client.post(
            f"/api/loop/{loop_id}/recording/stop",
            json={"recorder_id": "A"},
        )
        self.assertEqual(res.status_code, 200, res.text)
        info = json.loads(
            (self._recorders_dir(loop_id) / "A" / "info.json").read_text(encoding="utf-8")
        )
        self.assertIsNotNone(info["ended_at"])

    def test_delete_endpoint_wipes_recording(self) -> None:
        loop_id = "abc12345"
        self.client.post(f"/api/loop/{loop_id}/recording/start", json={"recorder_id": "A"})
        self.assertTrue(self._recording_dir(loop_id).is_dir())
        res = self.client.delete(f"/api/loop/{loop_id}/recording")
        self.assertEqual(res.status_code, 200, res.text)
        self.assertFalse(self._recording_dir(loop_id).is_dir())

    def test_recorder_id_validation_rejects_bad_chars(self) -> None:
        res = self.client.post(
            "/api/loop/abc12345/recording/start",
            json={"recorder_id": "../escape"},
        )
        self.assertEqual(res.status_code, 400, res.text)

    def test_chunks_endpoint_rejects_oversized_body(self) -> None:
        # Backstop for the recorder-side 4 MiB cap: even if a buggy
        # client ships a multi-megabyte body the server must 413
        # rather than block the uvicorn worker on json parsing.
        from web.routers import recording as recording_router

        loop_id = "abc12345"
        oversize = "a" * (recording_router._MAX_CHUNK_BODY_BYTES + 1024)
        res = self.client.post(
            f"/api/loop/{loop_id}/recording/chunks",
            data='{"recorder_id":"X","events":["' + oversize + '"]}',
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res.status_code, 413, res.text)

    def test_mp4_render_rejected_when_no_chunks(self) -> None:
        # /start creates meta.json but the recorder has no chunks yet,
        # so an mp4 render request must not silently kick off and waste
        # a Chromium spin-up.
        loop_id = "abc12345"
        self.client.post(f"/api/loop/{loop_id}/recording/start", json={"recorder_id": "A"})
        res = self.client.post(
            f"/api/loop/{loop_id}/recording/mp4",
            json={"speed": 4},
        )
        self.assertEqual(res.status_code, 409, res.text)

    def test_mp4_render_rejects_unsupported_concurrency(self) -> None:
        # The renderer fans out N headless Chromium instances; only
        # 1/4/8 are wired into the dashboard control. Anything else
        # must 400 so a typo cannot accidentally spin up 16 browsers.
        loop_id = "abc12345"
        self.client.post(f"/api/loop/{loop_id}/recording/start", json={"recorder_id": "A"})
        self.client.post(
            f"/api/loop/{loop_id}/recording/chunks",
            json={
                "recorder_id": "A",
                "events": [
                    {"type": 4, "timestamp": 1},
                    {"type": 2, "timestamp": 2},
                ],
            },
        )
        res = self.client.post(
            f"/api/loop/{loop_id}/recording/mp4",
            json={"speed": 4, "concurrency": 3},
        )
        self.assertEqual(res.status_code, 400, res.text)

    def test_events_endpoint_serves_persisted_segment(self) -> None:
        # The MP4 renderer fans out per-segment headless Chromium
        # instances that fetch /events?segment=<idx>. Verify the
        # round-trip: write a segment file directly, then read it
        # back through the public HTTP endpoint.
        loop_id = "abc12345"
        recording._segments_dir(loop_id).mkdir(parents=True, exist_ok=True)
        seg_path = recording._segment_file(loop_id, 0)
        seg_path.write_text(
            json.dumps(
                {
                    "events": [
                        {"type": 4, "timestamp": 100},
                        {"type": 2, "timestamp": 101},
                    ]
                }
            ),
            encoding="utf-8",
        )
        res = self.client.get(f"/api/loop/{loop_id}/recording/events?segment=0")
        self.assertEqual(res.status_code, 200, res.text)
        body = res.json()
        self.assertEqual(body["segment"], 0)
        self.assertFalse(body["truncated"])
        self.assertEqual([ev["timestamp"] for ev in body["events"]], [100, 101])

    def test_events_endpoint_404s_unknown_segment(self) -> None:
        loop_id = "abc12345"
        # No segment files exist yet — any index must 404.
        res = self.client.get(f"/api/loop/{loop_id}/recording/events?segment=7")
        self.assertEqual(res.status_code, 404, res.text)


class TestSplitIntoSegments(unittest.TestCase):
    """Unit tests for ``recording._split_into_segments``.

    The splitter is the only piece of MP4-render logic that runs
    without Playwright / ffmpeg / Chromium, so it carries the bulk
    of correctness assertions for the segment-fan-out pipeline.
    """

    def test_empty_input_returns_empty_list(self) -> None:
        self.assertEqual(recording._split_into_segments([]), [])

    def test_single_small_recording_stays_in_one_segment(self) -> None:
        events = [
            {"type": 4, "timestamp": 1},
            {"type": 2, "timestamp": 2},
            {"type": 3, "timestamp": 3, "data": {"x": 1}},
        ]
        segs = recording._split_into_segments(events, max_bytes=10_000)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0], events)

    def test_split_at_fullsnapshot_when_over_cap(self) -> None:
        # Two synthetic FullSnapshot boundaries with bulky
        # IncrementalSnapshots between them. Cap small enough that
        # the second FullSnapshot must open a new segment.
        bulky = {"type": 3, "timestamp": 10, "data": {"blob": "x" * 2000}}
        events = [
            {"type": 4, "timestamp": 1},
            {"type": 2, "timestamp": 2},
            bulky,
            {"type": 4, "timestamp": 20},
            {"type": 2, "timestamp": 21},
            {"type": 3, "timestamp": 22, "data": {"y": 1}},
        ]
        segs = recording._split_into_segments(events, max_bytes=1500)
        self.assertEqual(len(segs), 2, f"expected 2 segments, got {len(segs)}: {segs}")
        # Segment 1 must start with the original Meta + FullSnapshot.
        self.assertEqual(segs[0][0]["type"], 4)
        self.assertEqual(segs[0][1]["type"], 2)
        # Segment 2 must also bootstrap with a Meta + FullSnapshot
        # so rrweb-player can render it in isolation.
        self.assertEqual(segs[1][0]["type"], 4)
        self.assertEqual(segs[1][1]["type"], 2)

    def test_split_prepends_last_meta_when_none_at_boundary(self) -> None:
        # Realistic stream: one Meta at the start, periodic
        # FullSnapshots (from checkoutEveryNms) without their own
        # Meta. Splitter must reuse the cached Meta so each segment
        # still bootstraps.
        meta = {"type": 4, "timestamp": 0, "data": {"width": 1920, "height": 1080}}
        fs1 = {"type": 2, "timestamp": 1, "data": {"node": "root1"}}
        bulky = {"type": 3, "timestamp": 10, "data": {"blob": "y" * 3000}}
        fs2 = {"type": 2, "timestamp": 20, "data": {"node": "root2"}}
        events = [meta, fs1, bulky, fs2, {"type": 3, "timestamp": 30}]
        segs = recording._split_into_segments(events, max_bytes=2000)
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0][0], meta)
        self.assertEqual(segs[0][1], fs1)
        # Segment 2 must have the same Meta re-injected so the
        # headless renderer has viewport context before FullSnapshot.
        self.assertEqual(segs[1][0], meta)
        self.assertEqual(segs[1][1], fs2)

    def test_split_never_breaks_off_non_fullsnapshot(self) -> None:
        # If the only big event is a non-FullSnapshot, the splitter
        # has no safe boundary and must keep everything in one
        # segment rather than producing an unbootable mid-Increment
        # split. Defensive fall-through; checkoutEveryNms makes
        # this unlikely in practice.
        events = [
            {"type": 4, "timestamp": 1},
            {"type": 2, "timestamp": 2},
            {"type": 3, "timestamp": 3, "data": {"blob": "z" * 50_000}},
        ]
        segs = recording._split_into_segments(events, max_bytes=1000)
        self.assertEqual(len(segs), 1)


class TestDeleteLoopCascadesRecording(_RecordingApiBase):
    def test_delete_loop_removes_wrapper_session_and_recording(self) -> None:
        # Simulate a completed loop on disk: a stopped LoopInstance entry
        # in the in-memory map plus a recording subtree on disk.
        from web.routers import loop as loop_router

        loop_id = "abc12345"
        loop_router._instances[loop_id] = loop_router.LoopInstance(
            loop_id=loop_id, status="completed", workspace_dir="/tmp/ws"
        )
        loop_router._instance_dir(loop_id).mkdir(parents=True, exist_ok=True)
        # Wrapper session row + recording chunks under AGENTS_DIR.
        from web.agents import spawn

        spawn.init_wrapper_session(
            loop_id=loop_id,
            workspace="/tmp/ws",
            backend_label="cursor-cli",
            model="opus",
            stages="stage1",
        )
        spawn.finalize_wrapper_session(
            loop_id=loop_id,
            exit_code=0,
            state=store.STATE_COMPLETED,
        )
        self.client.post(f"/api/loop/{loop_id}/recording/start", json={"recorder_id": "A"})
        self.client.post(
            f"/api/loop/{loop_id}/recording/chunks",
            json={"recorder_id": "A", "events": [{"type": 4, "timestamp": 1}]},
        )
        wrapper_dir = store.session_dir(f"loop-{loop_id}")
        self.assertTrue(wrapper_dir.is_dir())
        self.assertTrue((wrapper_dir / "recording").is_dir())

        res = self.client.delete(f"/api/loop/{loop_id}")
        self.assertEqual(res.status_code, 200, res.text)
        self.assertFalse(
            wrapper_dir.is_dir(), "wrapper session dir must be wiped along with the loop"
        )
        self.assertNotIn(loop_id, loop_router._instances)


if __name__ == "__main__":
    unittest.main()
