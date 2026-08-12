from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestReviewPrompts(unittest.TestCase):
    """Lock the structural invariants of the post-rewrite review prompts.

    Mechanical Common Checks A-E and Stage<N> Checks A-D were removed
    when the review system flipped from "rerun grep patterns" to
    "open-ended semantic review backed by `harness run anti-proxy`
    (dev side hard gate)". These tests pin the new contract:

    - review_common.md is the semantic-review SSOT;
    - review_stage1.md / review_stage2.md host only stage FINISH 判定;
    - retired mechanical-check anchors must not silently come back.
    """

    def _prompt(self, stage: str) -> str:
        common = (REPO_ROOT / "prompt" / "review_prompt" / "review_common.md").read_text(
            encoding="utf-8",
        )
        stage_text = (REPO_ROOT / "prompt" / "review_prompt" / f"review_{stage}.md").read_text(
            encoding="utf-8",
        )
        return common + "\n\n" + stage_text

    def test_agent_loop_loads_common_and_stage_review_prompts(self) -> None:
        text = (REPO_ROOT / "agent-loop.sh").read_text(encoding="utf-8")
        self.assertIn("common-review-template", text)
        self.assertIn("review-template", text)

    def test_stage_review_prompts_are_stage_specific(self) -> None:
        # The stage-specific files exist to layer the FINISH 判定 on top
        # of review_common.md, not to re-derive the shared protocol.
        # After the rewrite stage files are ~30-50 lines vs common's
        # ~100 — the 2x cap still catches accidental wholesale
        # copy-paste of the common protocol into a stage file.
        common = (REPO_ROOT / "prompt" / "review_prompt" / "review_common.md").read_text(
            encoding="utf-8",
        )
        cap = 2 * len(common)
        for stage in ("stage1", "stage2"):
            stage_text = (REPO_ROOT / "prompt" / "review_prompt" / f"review_{stage}.md").read_text(
                encoding="utf-8",
            )
            with self.subTest(stage=stage):
                self.assertLess(len(stage_text), cap)

    def test_review_common_anchors_semantic_review(self) -> None:
        """review_common.md is the SSOT for semantic review; mechanical
        pattern lint lives in `harness run anti-proxy` (dev side hard
        gate). The common prompt must reference both.
        """
        text = (REPO_ROOT / "prompt" / "review_prompt" / "review_common.md").read_text(
            encoding="utf-8",
        )
        self.assertIn("Semantic review", text)
        self.assertIn("anti-proxy", text)
        # Output contract must be verbatim — agent-loop.sh greps it.
        self.assertIn("REVIEW_VERDICT:", text)
        self.assertIn("STAGE_STATUS:", text)
        # Candidate engine scope must be enumerated (so the review
        # agent reads the right files).
        self.assertIn("workload/src/training_engine_tensor", text)
        self.assertIn("workload/ops/", text)

    def test_stage_files_are_finish_judgement_only(self) -> None:
        """Both stage review files host only stage FINISH 判定; the
        retired mechanical Common / Stage<N> Checks must not return.
        """
        retired_anchors = (
            "Repository write-surface isolation",
            "On non-op branches",
            "Ref-script gate preset tampering",
            "Framework guard bypass",
            "Banned framework imports introduced",
            "Rule files weakened",
            "Reference-state injection in ours-side gate scripts",
            "Batch tiling / replay hack",
            "M5 / M6 threshold tampering",
            "Harness-visible config tampering",
            "Triton / PyTorch-native as kernel impl",
            "Per-op branch path isolation",
            "Drop-in I/O contract violation",
        )
        for stage in ("stage1", "stage2"):
            with self.subTest(stage=stage):
                text = (REPO_ROOT / "prompt" / "review_prompt" / f"review_{stage}.md").read_text(
                    encoding="utf-8"
                )
                self.assertIn(
                    f"{'Stage 1' if stage == 'stage1' else 'Stage 2'} FINISH Decision",
                    text,
                )
                self.assertIn("STAGE_STATUS: finished", text)
                for anchor in retired_anchors:
                    self.assertNotIn(
                        anchor,
                        text,
                        f"retired mechanical-check anchor {anchor!r} "
                        f"re-appeared in review_{stage}.md",
                    )


if __name__ == "__main__":
    unittest.main()
