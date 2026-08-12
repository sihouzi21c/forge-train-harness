"""Anti-regression: the ref launcher must default to MBS=4, not MBS=10.

Naive fp32-logits CE in `train_pure_mup_mtp.py` (post-rollback of
chunked CE — see `test_ref_no_chunked_ce.py`) allocates a transient
`[B*S, V]` fp32 tensor inside `masked_ce`. At V=73448 / S=4096 the
allocation is ~1.12 GiB per sample; combined with bf16 model (1 GB),
bf16 gradients (1 GB), Adam fp32 master + m + v (~8 GB), recompute
residuals (~2 GB) and the duplicate logits tensor for the MTP branch,
the per-GPU peak crosses 80 GiB once MBS ≥ 7. Empirically on H100
80GB:

    MBS=4 → ~73 GB peak (OK)
    MBS=5 → ~78 GB peak (OK)
    MBS=6 → ~80 GB peak (OK, tight)
    MBS=7 → OOM (7.85 GiB alloc fails)
    MBS=8 → OOM (8.97 GiB alloc fails)
    MBS=10 → OOM (11.21 GiB alloc fails)

`MICRO_BATCH_SIZE_OVERRIDE:-10` would therefore make any direct
invocation of the .sh script (without harness overriding via env)
OOM. The harness already overrides to MBS=4 in practice; this test
locks the script default to match, so a fresh standalone run works
without per-caller knowledge.

GBS=80 presets (multistep / perf-bitwise / long-train / op-long /
loss-gate-200 / resume-gate-20) keep GBS=80; grad_accum becomes
10 (80 / (4 × DP=2)). WORLD=1 presets (forward-align /
backward-align / multistep-1gpu) drop GBS to 4 so grad_accum stays
1 (alignment contract: one micro-batch, no accumulation).
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

LAUNCHER = (
    Path(__file__).resolve().parents[2] / "ref" / "reference" / "run_16gpu_1000step_pure_mup_mtp.sh"
)


class TestRefLauncherMbsDefault(unittest.TestCase):
    def setUp(self) -> None:
        self.text = LAUNCHER.read_text(encoding="utf-8")

    def test_no_mbs_default_of_10(self) -> None:
        """`MICRO_BATCH_SIZE_OVERRIDE:-10` must not appear anywhere."""
        offenders = re.findall(r"MICRO_BATCH_SIZE_OVERRIDE:-10\b", self.text)
        self.assertEqual(
            offenders,
            [],
            msg=(
                "Launcher reintroduced `MICRO_BATCH_SIZE_OVERRIDE:-10`; "
                "naive fp32 logits OOM at MBS=10 on H100 80GB (see "
                "module docstring). Use MBS=4 instead."
            ),
        )

    def test_mbs_default_is_4_everywhere(self) -> None:
        """Every `MICRO_BATCH_SIZE_OVERRIDE:-N` defaults to 4."""
        defaults = re.findall(r"MICRO_BATCH_SIZE_OVERRIDE:-(\d+)\b", self.text)
        self.assertTrue(defaults, "no MBS defaults found in launcher")
        for d in defaults:
            self.assertEqual(
                d,
                "4",
                msg=f"found MICRO_BATCH_SIZE_OVERRIDE:-{d}, expected 4",
            )

    def test_world1_presets_gbs_matches_mbs(self) -> None:
        """forward-align / backward-align / multistep-1gpu use GBS=MBS=4
        (grad_accum=1 — the alignment contract requires a single
        micro-batch with no accumulation)."""
        for preset in ("forward-align|backward-align", "multistep-1gpu"):
            # Match the case-arm block (preset header up to the first ;;).
            m = re.search(
                rf"{re.escape(preset)}\)(.+?);;",
                self.text,
                re.DOTALL,
            )
            self.assertIsNotNone(m, f"preset arm {preset!r} not found")
            arm = m.group(1)
            mbs = re.search(r"MICRO_BATCH_SIZE_OVERRIDE:-(\d+)", arm)
            gbs = re.search(r"GLOBAL_BATCH_SIZE_OVERRIDE:-(\d+)", arm)
            self.assertIsNotNone(mbs, f"{preset}: no MBS default")
            self.assertIsNotNone(gbs, f"{preset}: no GBS default")
            self.assertEqual(
                mbs.group(1),
                gbs.group(1),
                msg=(
                    f"preset {preset!r}: GBS={gbs.group(1)} must equal "
                    f"MBS={mbs.group(1)} (WORLD=1, grad_accum=1)."
                ),
            )


if __name__ == "__main__":
    unittest.main()
