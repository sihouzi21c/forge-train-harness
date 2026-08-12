"""Behavioral tests for ``tools.mfu_record``.

The helper is the single writer for the per-loop ``mfu_history.jsonl``
file that drives the web Loop Progress badge. Tests pin:

* MFU value absence is a no-op (suite did not measure MFU).
* History is appended on every measured run, regardless of precision
  outcome — a low-MFU run with a failed precision gate is still a real
  data point worth keeping.
* ``precision_pass`` is derived from ``correctness_pass`` for bitwise
  suites and ``loss_pass`` for statistical suites; absent both, it is
  ``None`` (suite has no precision gate).
* Loop id resolved from env or from workspace path layout fallback.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

# Make the in-tree ``tools`` package importable at test time. The
# production code path runs with ``PYTHONPATH=$WORKSPACE`` (set by
# agent-loop.sh) so ``tools`` is top-level there; here we replicate
# that by adding the ``harness/`` source root to sys.path.
_HARNESS_SRC = Path(__file__).resolve().parents[2]
if str(_HARNESS_SRC) not in sys.path:
    sys.path.insert(0, str(_HARNESS_SRC))

from tools import mfu_record  # noqa: E402


def _make_result(
    *,
    avg_mfu: float | None = 42.5,
    mfu_target: float | None = 25.0,
    mfu_pass: bool | None = True,
    correctness_pass: bool | None = True,
    loss_pass: bool | None = None,
    milestone: str = "bitwise-perf",
    suite: str = "perf-bitwise",
) -> dict:
    metrics: dict = {
        "avg_mfu_e2e_standard": avg_mfu,
        "mfu_target": mfu_target,
        "mfu_pass": mfu_pass,
        "world_size": 8,
        "num_steps": 200,
    }
    if correctness_pass is not None:
        metrics["correctness_pass"] = correctness_pass
    if loss_pass is not None:
        metrics["loss_pass"] = loss_pass
    return {
        "schema_version": 1,
        "status": "passed" if (correctness_pass and mfu_pass) else "failed",
        "suite": suite,
        "summary": "synthetic",
        "metrics": metrics,
        "details": {"config": {"milestone": milestone}},
    }


class _Layout:
    """Synthetic ``forge_train/<loop_id>/workspace`` layout."""

    def __init__(self, root: Path, loop_id: str = "abc123") -> None:
        self.train_dir = root / "forge_train"
        self.loop_id = loop_id
        self.repo_root = self.train_dir / loop_id / "workspace"
        self.repo_root.mkdir(parents=True)

    @property
    def env(self) -> dict:
        # Path-layout fallback only; env contains no LOOP_ID/FORGE_TRAIN_DIR.
        return {}

    @property
    def env_with_overrides(self) -> dict:
        return {"LOOP_ID": self.loop_id, "FORGE_TRAIN_DIR": str(self.train_dir)}

    def history_path(self) -> Path:
        return self.train_dir / self.loop_id / mfu_record.HISTORY_FILENAME

    def read_history(self) -> list[dict]:
        p = self.history_path()
        if not p.is_file():
            return []
        return [
            json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()
        ]


class TestMfuRecord(unittest.TestCase):
    def test_no_avg_mfu_is_noop(self) -> None:
        with TemporaryDirectory() as tmp:
            layout = _Layout(Path(tmp))
            result = _make_result(avg_mfu=None)
            entry = mfu_record.record(result, layout.repo_root, env=layout.env_with_overrides)
            self.assertIsNone(entry)
            self.assertFalse(layout.history_path().exists())

    def test_history_appended_on_every_measured_run(self) -> None:
        with TemporaryDirectory() as tmp:
            layout = _Layout(Path(tmp))
            mfu_record.record(
                _make_result(avg_mfu=30.0, correctness_pass=False, mfu_pass=False),
                layout.repo_root,
                env=layout.env_with_overrides,
                now=1000.0,
            )
            mfu_record.record(
                _make_result(avg_mfu=42.5, correctness_pass=True, mfu_pass=True),
                layout.repo_root,
                env=layout.env_with_overrides,
                now=2000.0,
            )
            history = layout.read_history()
            self.assertEqual([h["avg_mfu"] for h in history], [30.0, 42.5])
            self.assertEqual([h["precision_pass"] for h in history], [False, True])

    def test_precision_pass_derives_from_correctness(self) -> None:
        with TemporaryDirectory() as tmp:
            layout = _Layout(Path(tmp))
            mfu_record.record(
                _make_result(avg_mfu=99.0, correctness_pass=False),
                layout.repo_root,
                env=layout.env_with_overrides,
                now=1.0,
            )
            mfu_record.record(
                _make_result(avg_mfu=42.5, correctness_pass=True),
                layout.repo_root,
                env=layout.env_with_overrides,
                now=2.0,
            )
            history = layout.read_history()
            self.assertEqual([h["precision_pass"] for h in history], [False, True])

    def test_precision_pass_derives_from_loss_for_statistical_suites(self) -> None:
        with TemporaryDirectory() as tmp:
            layout = _Layout(Path(tmp))
            entry = mfu_record.record(
                _make_result(
                    avg_mfu=48.0,
                    correctness_pass=None,
                    loss_pass=True,
                    milestone="long-horizon",
                    suite="long-train",
                ),
                layout.repo_root,
                env=layout.env_with_overrides,
                now=1.0,
            )
            self.assertIsNotNone(entry)
            self.assertTrue(entry["precision_pass"])

    def test_no_precision_field_records_none(self) -> None:
        with TemporaryDirectory() as tmp:
            layout = _Layout(Path(tmp))
            entry = mfu_record.record(
                _make_result(avg_mfu=70.0, correctness_pass=None, loss_pass=None),
                layout.repo_root,
                env=layout.env_with_overrides,
                now=1.0,
            )
            self.assertIsNotNone(entry)
            self.assertIsNone(entry["precision_pass"])
            self.assertEqual(len(layout.read_history()), 1)

    def test_loop_id_falls_back_to_workspace_path_layout(self) -> None:
        # No LOOP_ID / FORGE_TRAIN_DIR in env: helper must derive both
        # from ``<train_dir>/<loop_id>/workspace`` layout.
        with TemporaryDirectory() as tmp:
            layout = _Layout(Path(tmp), loop_id="deadbeef99")
            entry = mfu_record.record(
                _make_result(avg_mfu=33.3),
                layout.repo_root,
                env=layout.env,
                now=1.0,
            )
            self.assertIsNotNone(entry)
            self.assertEqual(entry["loop_id"], "deadbeef99")
            self.assertEqual(layout.read_history()[0]["loop_id"], "deadbeef99")

    def test_outside_loop_layout_is_noop(self) -> None:
        with TemporaryDirectory() as tmp:
            standalone = Path(tmp) / "elsewhere"
            standalone.mkdir()
            entry = mfu_record.record(
                _make_result(avg_mfu=42.0),
                standalone,
                env={},
            )
            self.assertIsNone(entry)


if __name__ == "__main__":
    unittest.main()
