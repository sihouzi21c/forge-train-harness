"""Behavioral tests for ``tools.mfu_backfill_events``.

In ssh/devspace mode ``tools.mfu_record`` runs on the REMOTE host, so the
``mfu_record`` loop_event it emits lands in the remote wrapper stdout.log
and never reaches the local dashboard SSE stream — only the
``mfu_history.jsonl`` FILE is rsynced back by the wrapper's background
puller. This helper re-emits a LOCAL ``mfu_record`` event for every history
entry not yet emitted locally, advancing a cursor so repeated 30 s polls
never duplicate events. Tests pin:

* One event emitted per new history entry; payload carries the badge fields
  (the same shape ``tools.mfu_record._emit_event`` ships).
* A cursor file makes repeated runs idempotent — no new entries → no events.
* Appended entries emit only the delta on the next run.
* Malformed lines / a missing history file are best-effort no-ops.
* A mid-batch emit failure advances the cursor only past what was emitted,
  so the remainder retries on the next poll instead of being dropped.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

# Make the in-tree ``tools`` package importable at test time, mirroring the
# ``PYTHONPATH=$WORKSPACE`` agent-loop.sh sets in production (see
# test_mfu_record.py for the same shim).
_HARNESS_SRC = Path(__file__).resolve().parents[2]
if str(_HARNESS_SRC) not in sys.path:
    sys.path.insert(0, str(_HARNESS_SRC))

from tools import mfu_backfill_events as mbe  # noqa: E402

_WRITER = Path("/fake/loop_wrapper_event.py")


def _entry(avg_mfu: float, milestone: str = "bitwise-perf", suite: str = "perf-bitwise") -> dict:
    return {
        "ts": 1.0,
        "loop_id": "abc",
        "milestone": milestone,
        "suite": suite,
        "avg_mfu": avg_mfu,
        "mfu_target": 10.0,
        "mfu_pass": True,
        "precision_pass": True,
        "world_size": 2,
        "num_steps": 50,
        "agent_id": None,
    }


class _Recorder:
    """Stand-in for ``subprocess.run`` that records each emit invocation and
    can be told to raise on the Nth call (simulating a wedged writer)."""

    def __init__(self, fail_at: int | None = None) -> None:
        self.calls: list[list[str]] = []
        self._fail_at = fail_at

    def __call__(self, cmd, **kwargs):
        if self._fail_at is not None and len(self.calls) == self._fail_at:
            raise OSError("boom")
        self.calls.append(cmd)
        return None

    def payloads(self) -> list[dict]:
        out = []
        for cmd in self.calls:
            out.append(json.loads(cmd[cmd.index("--payload") + 1]))
        return out


def _write_history(loop_dir: Path, entries: list[dict]) -> None:
    with (loop_dir / "mfu_history.jsonl").open("a", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")


class MfuBackfillEventsTest(unittest.TestCase):
    def _dir(self) -> Path:
        td = TemporaryDirectory()
        self.addCleanup(td.cleanup)
        return Path(td.name)

    def test_emits_one_event_per_new_entry(self):
        d = self._dir()
        _write_history(d, [_entry(10.0), _entry(14.3)])
        rec = _Recorder()
        n = mbe.backfill(loop_dir=d, loop_id="abc", writer=_WRITER, runner=rec)
        self.assertEqual(n, 2)
        self.assertEqual(len(rec.calls), 2)
        self.assertEqual((d / mbe.CURSOR_FILENAME).read_text().strip(), "2")
        # Each call targets the canonical writer with subtype mfu_record.
        self.assertIn("mfu_record", rec.calls[0])
        self.assertIn("--loop-id", rec.calls[0])
        # Payload carries the badge fields the frontend reads.
        p0 = rec.payloads()[0]
        self.assertEqual(p0["avg_mfu"], 10.0)
        self.assertEqual(p0["milestone"], "bitwise-perf")
        self.assertEqual(p0["suite"], "perf-bitwise")
        self.assertIn("mfu_pass", p0)
        self.assertIn("precision_pass", p0)

    def test_idempotent_when_no_new_entries(self):
        d = self._dir()
        _write_history(d, [_entry(10.0)])
        mbe.backfill(loop_dir=d, loop_id="abc", writer=_WRITER, runner=_Recorder())
        rec2 = _Recorder()
        n = mbe.backfill(loop_dir=d, loop_id="abc", writer=_WRITER, runner=rec2)
        self.assertEqual(n, 0)
        self.assertEqual(rec2.calls, [])

    def test_appended_entries_emit_only_delta(self):
        d = self._dir()
        _write_history(d, [_entry(10.0)])
        mbe.backfill(loop_dir=d, loop_id="abc", writer=_WRITER, runner=_Recorder())
        _write_history(d, [_entry(14.3), _entry(20.0)])
        rec = _Recorder()
        n = mbe.backfill(loop_dir=d, loop_id="abc", writer=_WRITER, runner=rec)
        self.assertEqual(n, 2)
        self.assertEqual([p["avg_mfu"] for p in rec.payloads()], [14.3, 20.0])
        self.assertEqual((d / mbe.CURSOR_FILENAME).read_text().strip(), "3")

    def test_missing_history_is_noop(self):
        d = self._dir()
        n = mbe.backfill(loop_dir=d, loop_id="abc", writer=_WRITER, runner=_Recorder())
        self.assertEqual(n, 0)
        self.assertFalse((d / "mfu_history.jsonl").exists())

    def test_malformed_line_is_skipped(self):
        d = self._dir()
        (d / "mfu_history.jsonl").write_text(
            json.dumps(_entry(10.0)) + "\n{not json}\n" + json.dumps(_entry(14.3)) + "\n",
            encoding="utf-8",
        )
        rec = _Recorder()
        n = mbe.backfill(loop_dir=d, loop_id="abc", writer=_WRITER, runner=rec)
        self.assertEqual(n, 2)
        self.assertEqual([p["avg_mfu"] for p in rec.payloads()], [10.0, 14.3])

    def test_midbatch_failure_advances_cursor_partially(self):
        d = self._dir()
        _write_history(d, [_entry(10.0), _entry(14.3), _entry(20.0)])
        rec = _Recorder(fail_at=1)  # first emit ok, second raises
        n = mbe.backfill(loop_dir=d, loop_id="abc", writer=_WRITER, runner=rec)
        self.assertEqual(n, 1)
        self.assertEqual((d / mbe.CURSOR_FILENAME).read_text().strip(), "1")
        # Next poll resumes from the un-emitted remainder.
        rec2 = _Recorder()
        n2 = mbe.backfill(loop_dir=d, loop_id="abc", writer=_WRITER, runner=rec2)
        self.assertEqual(n2, 2)
        self.assertEqual([p["avg_mfu"] for p in rec2.payloads()], [14.3, 20.0])


if __name__ == "__main__":
    unittest.main()
