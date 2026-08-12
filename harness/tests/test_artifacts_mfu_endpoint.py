"""``GET /api/artifacts/mfu`` returns per-loop MFU summary.

The endpoint is the hydration surface for the web Loop Progress badge.
It reads the per-loop SSOT file written by ``tools.mfu_record``:

* ``$FORGE_TRAIN_DIR/<loop_id>/mfu_history.jsonl`` — per-loop history.

The per-loop best (``loop_best``) is derived server-side from the
history file and is what the badge's big number binds to, so switching
between loops changes the displayed value.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

_HARNESS_SRC = Path(__file__).resolve().parents[2]
if str(_HARNESS_SRC) not in sys.path:
    sys.path.insert(0, str(_HARNESS_SRC))

from tools import mfu_record  # noqa: E402


class TestMfuEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.train_dir = Path(self._tmp.name) / "forge_train"
        self.train_dir.mkdir()

        # Repoint both ``tools.mfu_record`` and the web router at the
        # synthetic forge_train tree. The router reads from
        # ``web.paths.FORGE_TRAIN_DIR`` directly, but ``artifacts.py``
        # imports the module-level constant at import time, so we
        # patch the router's own binding.
        from web.routers import artifacts as artifacts_router

        self._orig_train_dir = artifacts_router.FORGE_TRAIN_DIR
        artifacts_router.FORGE_TRAIN_DIR = self.train_dir
        self.addCleanup(lambda: setattr(artifacts_router, "FORGE_TRAIN_DIR", self._orig_train_dir))

        from web import server

        self.client = TestClient(server.app)

    def _result(
        self, *, avg_mfu: float, milestone: str = "bitwise-perf", correctness: bool = True
    ) -> dict:
        return {
            "schema_version": 1,
            "status": "passed" if correctness else "failed",
            "suite": "perf-bitwise",
            "summary": "synthetic",
            "metrics": {
                "avg_mfu_e2e_standard": avg_mfu,
                "mfu_target": 25.0,
                "mfu_pass": True,
                "correctness_pass": correctness,
                "world_size": 8,
                "num_steps": 200,
            },
            "details": {"config": {"milestone": milestone}},
        }

    def _record_in(self, loop_id: str, **kwargs) -> None:
        repo_root = self.train_dir / loop_id / "workspace"
        repo_root.mkdir(parents=True, exist_ok=True)
        mfu_record.record(
            self._result(**kwargs),
            repo_root,
            env={"LOOP_ID": loop_id, "FORGE_TRAIN_DIR": str(self.train_dir)},
        )

    def test_empty_returns_null_loop_fields(self) -> None:
        resp = self.client.get("/api/artifacts/mfu", params={"loop_id": "nope"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIsNone(body["latest_in_loop"])
        self.assertIsNone(body["loop_best"])
        self.assertEqual(body["loop_by_milestone"], {})
        self.assertEqual(body["history_count"], 0)

    def test_returns_per_loop_summary(self) -> None:
        self._record_in("loop1", avg_mfu=30.0)
        self._record_in("loop1", avg_mfu=42.5)

        resp = self.client.get("/api/artifacts/mfu", params={"loop_id": "loop1"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["loop_best"]["avg_mfu"], 42.5)
        self.assertEqual(body["loop_best"]["milestone"], "bitwise-perf")
        self.assertEqual(body["loop_by_milestone"]["bitwise-perf"]["avg_mfu"], 42.5)
        self.assertEqual(body["latest_in_loop"]["avg_mfu"], 42.5)
        self.assertEqual(body["history_count"], 2)

    def test_failed_precision_does_not_promote_to_loop_best(self) -> None:
        self._record_in("loop1", avg_mfu=99.0, correctness=False)
        resp = self.client.get("/api/artifacts/mfu", params={"loop_id": "loop1"})
        body = resp.json()
        # History was written, but the per-loop best filter refused promotion.
        self.assertEqual(body["history_count"], 1)
        self.assertEqual(body["latest_in_loop"]["avg_mfu"], 99.0)
        self.assertIsNone(body["loop_best"])
        self.assertEqual(body["loop_by_milestone"], {})

    def test_loop_summary_is_scoped_to_requested_loop(self) -> None:
        """Switching ``loop_id`` MUST swap the entire response — loops may
        run different tasks, so cross-loop comparison is meaningless and
        the endpoint never exposes another loop's values."""
        self._record_in("loop1", avg_mfu=30.0)
        self._record_in("loop1", avg_mfu=42.5)
        self._record_in("loop2", avg_mfu=55.0, milestone="long-horizon")
        self._record_in("loop2", avg_mfu=50.0, milestone="long-horizon")

        body1 = self.client.get("/api/artifacts/mfu", params={"loop_id": "loop1"}).json()
        body2 = self.client.get("/api/artifacts/mfu", params={"loop_id": "loop2"}).json()

        self.assertEqual(body1["loop_best"]["avg_mfu"], 42.5)
        self.assertEqual(body1["loop_best"]["milestone"], "bitwise-perf")
        self.assertEqual(body2["loop_best"]["avg_mfu"], 55.0)
        self.assertEqual(body2["loop_best"]["milestone"], "long-horizon")

        # Per-loop, per-milestone bests are scoped to the queried loop.
        self.assertEqual(body1["loop_by_milestone"]["bitwise-perf"]["avg_mfu"], 42.5)
        self.assertNotIn("long-horizon", body1["loop_by_milestone"])
        self.assertEqual(body2["loop_by_milestone"]["long-horizon"]["avg_mfu"], 55.0)
        self.assertNotIn("bitwise-perf", body2["loop_by_milestone"])

    def test_loop_best_picks_max_not_latest(self) -> None:
        """When the loop's history has a higher earlier value, ``loop_best``
        keeps it even after a lower precision-pass run lands later."""
        self._record_in("loop1", avg_mfu=80.0)
        self._record_in("loop1", avg_mfu=40.0)
        body = self.client.get("/api/artifacts/mfu", params={"loop_id": "loop1"}).json()
        self.assertEqual(body["loop_best"]["avg_mfu"], 80.0)
        self.assertEqual(body["latest_in_loop"]["avg_mfu"], 40.0)


if __name__ == "__main__":
    unittest.main()
