"""Unit tests for the resume startup-time gate.

Pure-logic tests — no torch / GPU / data. A fake clock and two fake loaders
(replay-by-discard vs O(1) cursor seek) drive the shared timing path so the
gate is shown to fail the linear-replay loader and pass the O(1) seek, which
is exactly the behaviour the real ``resume-startup-90`` gate enforces.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.resume_startup_gate import (  # noqa: E402
    emit_resume_startup_marker,
    evaluate_resume_startup,
    measure_resume_startup,
    parse_resume_startup_seconds,
)

# Mirror the real gate scenario: resume from step 90 at grad_accum=8.
SAVE_STEP = 90
GRAD_ACCUM = 8
CONSUMED = SAVE_STEP * GRAD_ACCUM  # 720 micro-batches to replay
BUDGET_S = 15.0
# Measured replay cost on the real engine dataloader (~21-26 ms / micro-batch
# single-threaded tokenize+stream); use the lower end so the test is a
# conservative reproduction of the failure.
REPLAY_S_PER_MB = 0.021


class _FakeClock:
    """Monotonic clock the fake loaders advance deterministically."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class _ReplayLoader:
    """advance() cost grows with ``count`` — the bad replay-by-discard loader."""

    def __init__(self, clock: _FakeClock, per_mb: float) -> None:
        self._clock = clock
        self._per_mb = per_mb
        self._clock.t += 0.3  # construction (open streams, fill shuffle buffer)

    def advance(self, count: int) -> None:
        if count < 0:
            raise ValueError("count must be >= 0")
        self._clock.t += count * self._per_mb

    def __next__(self):
        self._clock.t += 0.1
        return {"tokens": None}


class _O1Loader:
    """advance() is O(1) — restores the saved cursor, no re-stream."""

    def __init__(self, clock: _FakeClock) -> None:
        self._clock = clock
        self._clock.t += 0.3

    def advance(self, count: int) -> None:
        if count < 0:
            raise ValueError("count must be >= 0")
        self._clock.t += 0.05  # constant regardless of count

    def __next__(self):
        self._clock.t += 0.1
        return {"tokens": None}


class TestEvaluateResumeStartup(unittest.TestCase):
    def test_pass_under_budget(self):
        v = evaluate_resume_startup(10.0, BUDGET_S, save_step=SAVE_STEP)
        self.assertTrue(v.passed)
        self.assertIn("within", v.summary)

    def test_fail_over_budget(self):
        v = evaluate_resume_startup(18.9, BUDGET_S, save_step=SAVE_STEP)
        self.assertFalse(v.passed)
        self.assertIn("exceeds", v.summary)

    def test_boundary_is_inclusive(self):
        self.assertTrue(evaluate_resume_startup(15.0, 15.0).passed)

    def test_zero_budget_disables_gate(self):
        v = evaluate_resume_startup(999.0, 0.0)
        self.assertTrue(v.passed)
        self.assertIn("disabled", v.summary)

    def test_negative_startup_raises(self):
        with self.assertRaises(ValueError):
            evaluate_resume_startup(-1.0, BUDGET_S)


class TestMeasureResumeStartup(unittest.TestCase):
    def test_consumed_negative_raises(self):
        clock = _FakeClock()
        with self.assertRaises(ValueError):
            measure_resume_startup(lambda: _O1Loader(clock), -1, clock=clock)

    def test_replay_scales_with_consumed(self):
        def measure(consumed: int) -> float:
            clock = _FakeClock()
            return measure_resume_startup(
                lambda: _ReplayLoader(clock, REPLAY_S_PER_MB), consumed, clock=clock
            )

        self.assertLess(measure(160), measure(720))

    def test_gate_fails_replay_loader(self):
        """The end-to-end point: at save_step=90 the replay loader blows the
        15 s budget, so the gate fails it."""
        clock = _FakeClock()
        startup = measure_resume_startup(
            lambda: _ReplayLoader(clock, REPLAY_S_PER_MB), CONSUMED, clock=clock
        )
        verdict = evaluate_resume_startup(
            startup, BUDGET_S, save_step=SAVE_STEP, consumed_microbatches=CONSUMED
        )
        self.assertGreater(startup, BUDGET_S)
        self.assertFalse(verdict.passed)

    def test_gate_passes_o1_loader(self):
        """An O(1) cursor seek resumes from step 90 well under budget."""
        clock = _FakeClock()
        startup = measure_resume_startup(lambda: _O1Loader(clock), CONSUMED, clock=clock)
        verdict = evaluate_resume_startup(
            startup, BUDGET_S, save_step=SAVE_STEP, consumed_microbatches=CONSUMED
        )
        self.assertLess(startup, BUDGET_S)
        self.assertTrue(verdict.passed)


class TestParseResumeStartupSeconds(unittest.TestCase):
    def test_happy(self):
        out = "noise\n[RESUME_STARTUP] seconds=18.51\nmore\n"
        self.assertAlmostEqual(parse_resume_startup_seconds(out), 18.51)

    def test_missing_raises(self):
        with self.assertRaises(ValueError):
            parse_resume_startup_seconds("no marker here")

    def test_duplicate_raises(self):
        out = "[RESUME_STARTUP] seconds=1.0\n[RESUME_STARTUP] seconds=2.0\n"
        with self.assertRaises(ValueError):
            parse_resume_startup_seconds(out)


class TestEmitResumeStartupMarker(unittest.TestCase):
    """launch_dp fans the runner into world_size processes sharing one
    ours.log, while the parser requires exactly one marker — so emission
    must be rank-guarded (loop f6b9c438e05f round 6: DP=2 emitted two
    markers and the gate failed structurally)."""

    def test_rank0_emits_one_parseable_line(self):
        lines: list[str] = []
        emit_resume_startup_marker(18.51, rank=0, emit=lines.append)
        self.assertEqual(len(lines), 1)
        self.assertAlmostEqual(parse_resume_startup_seconds(lines[0]), 18.51)

    def test_nonzero_rank_is_silent(self):
        lines: list[str] = []
        emit_resume_startup_marker(18.51, rank=1, emit=lines.append)
        self.assertEqual(lines, [])

    def test_dp2_shared_log_parses(self):
        # Both ranks measure and emit into the same log; only rank 0's
        # marker must survive so the strict parser sees exactly one.
        log: list[str] = []
        for rank in (0, 1):
            emit_resume_startup_marker(18.51, rank=rank, emit=log.append)
        self.assertAlmostEqual(parse_resume_startup_seconds("\n".join(log)), 18.51)


if __name__ == "__main__":
    unittest.main(verbosity=2)
