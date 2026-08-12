"""Cross-config / cross-gate invariant checks in render_gate_configs (problem 2).

The renderer is the one place that sees every gate's resolved shape, so it
enforces invariants the per-file schema can't:
  C1  a `<gate>-smoke` must share the base gate's shape (only num_steps /
      gate_window / warmup_steps may differ).
  C2  gate_window must fit the run (half-open, <= num_steps+1).
  C8  ours determinism-off must drop the determinism-forcing env.
  C9  a bitwise gate must carry zero tolerances.

These are unit tests over the pure validators plus an integration pass that every
committed suite renders clean (so the fix to dense_training_qwen3's smoke holds).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import render_gate_configs as rgc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _gate_render as gr  # noqa: E402


class SmokePairTest(unittest.TestCase):
    def _pair(self, base_ours: dict, smoke_ours: dict) -> None:
        rgc._validate_smoke_pairs(
            {
                "long-train": {"ours": base_ours},
                "long-train-smoke": {"ours": smoke_ours},
            }
        )

    def test_matching_shape_passes(self):
        base = {
            "micro_batch_size": 4,
            "global_batch_size": 80,
            "world_size": 2,
            "num_steps": 200,
            "gate_window": [101, 201],
        }
        smoke = {
            "micro_batch_size": 4,
            "global_batch_size": 80,
            "world_size": 2,
            "num_steps": 20,
            "gate_window": [5, 20],
            "warmup_steps": 5,
        }
        self._pair(base, smoke)  # must not raise

    def test_divergent_ours_mbs_fails(self):
        base = {"micro_batch_size": 4, "global_batch_size": 80, "world_size": 2, "num_steps": 200}
        smoke = {"micro_batch_size": 10, "global_batch_size": 80, "world_size": 2, "num_steps": 20}
        with self.assertRaisesRegex(rgc.RenderError, "diverges"):
            self._pair(base, smoke)

    def test_smoke_without_base_fails(self):
        with self.assertRaisesRegex(rgc.RenderError, "no base gate"):
            rgc._validate_smoke_pairs({"long-train-smoke": {"ours": {"micro_batch_size": 4}}})


class ResolvedInvariantTest(unittest.TestCase):
    def test_window_out_of_range_fails(self):
        with self.assertRaisesRegex(rgc.RenderError, "gate_window"):
            rgc._validate_resolved("g", "ref", {"num_steps": 20, "gate_window": [5, 30]})

    def test_window_at_num_steps_plus_one_ok(self):
        rgc._validate_resolved("g", "ref", {"num_steps": 200, "gate_window": [101, 201]})

    def test_bitwise_nonzero_tol_fails(self):
        with self.assertRaisesRegex(rgc.RenderError, "gate_bitwise"):
            rgc._validate_resolved("g", "ref", {"gate_bitwise": True, "gate_atol": 1e-5})

    def test_bitwise_zero_tol_ok(self):
        rgc._validate_resolved("g", "ref", {"gate_bitwise": True, "gate_atol": 0})

    def test_det_off_with_leaked_env_fails(self):
        with self.assertRaisesRegex(rgc.RenderError, "determinism env"):
            rgc._validate_resolved(
                "g", "ours", {"deterministic": False, "cublas_workspace_config": ":4096:8"}
            )

    def test_det_off_clean_ok(self):
        rgc._validate_resolved("g", "ours", {"deterministic": False})


class CommittedSuitesRenderCleanTest(unittest.TestCase):
    """Every committed suite must pass the new validators (covers the qwen3 fix)."""

    def test_all_variants_render(self):
        for variant in (
            "dense_training",
            "dense_training_1b",
            "dense_training_qwen3",
            "dense_training_8b",
        ):
            if gr.missing_render_inputs(variant):
                continue  # axis inputs absent in this checkout — skip, don't fail
            with self.subTest(variant=variant):
                rendered = gr.render_variant(variant)  # raises on RenderError
                self.assertIsNotNone(rendered)
                rendered.cleanup()


if __name__ == "__main__":
    unittest.main()
