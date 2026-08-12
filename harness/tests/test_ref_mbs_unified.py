"""Ref-side micro-batch size is unified to 4 across every det-off gate.

The deterministic bitwise gates already drive the ref script at MBS=4. The
det-off perf gates historically drove the ref at MBS=5; ref/ours align on a
shared GBS=80 (each side derives its own grad_accum) rather than on equal
MBS, so the ref side runs MBS=4 (grad_accum=10) with no change to the
comparison contract. The ours side keeps MBS=10 on the perf gates (the
long-horizon milestone's reason to exist), which must NOT be unified.

Post flat→directory migration: per-gate shape lives in
``gate_config/<gate>.toml`` and is read from the rendered products. ``op-long``
is D5-exempt (product-less) and keeps its values inline in the registry.
"""

from __future__ import annotations

import unittest

from harness.tests._gate_render import (
    REPO_ROOT,
    missing_render_inputs,
    render_variant,
)

# Det-off perf gates that share the GBS=80 / ref-MBS=4 / ours-MBS=10 shape.
_PRODUCT_PERF_GATES = ("long-train", "long-train-smoke", "loss-gate-200")
# op-long is product-less (D5-exempt) — its values stay inline in the registry.
_INLINE_PERF_GATES = ("op-long",)
_GATE_CONFIG_DIR = REPO_ROOT / "config" / "eval" / "dense_training" / "gate_config"


@unittest.skipIf(missing_render_inputs(), f"render inputs absent: {missing_render_inputs()}")
class TestRefMbsUnifiedToFour(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rendered = render_variant("dense_training")
        cls.evals = cls.rendered.registry["evals"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.rendered.cleanup()

    def test_product_perf_gates_ref_mbs_is_four(self) -> None:
        for gate in _PRODUCT_PERF_GATES:
            with self.subTest(gate=gate):
                self.assertEqual(int(self.rendered.product(gate, "ref").get("micro_batch_size")), 4)

    def test_product_perf_gates_ours_mbs_still_ten(self) -> None:
        for gate in _PRODUCT_PERF_GATES:
            with self.subTest(gate=gate):
                self.assertEqual(
                    int(self.rendered.product(gate, "ours").get("micro_batch_size")), 10
                )

    def test_product_perf_gates_ref_grad_accum_is_ten(self) -> None:
        # GBS=80 / (MBS=4 * WS=2) = 10 — integer, so the ref shape is valid.
        for gate in _PRODUCT_PERF_GATES:
            with self.subTest(gate=gate):
                prod = self.rendered.product(gate, "ref")
                ws = int(prod.get("world_size"))
                mbs = int(prod.get("micro_batch_size"))
                gbs = int(prod.get("global_batch_size"))
                self.assertEqual(gbs % (mbs * ws), 0)
                self.assertEqual(gbs // (mbs * ws), 10)
                self.assertEqual(int(prod.get("grad_accum_steps")), 10)

    def test_op_long_inline_ref_mbs_four_ours_ten(self) -> None:
        # op-long is product-less; its ref/ours MBS stay inline in the registry.
        for gate in _INLINE_PERF_GATES:
            with self.subTest(gate=gate):
                self.assertEqual(self.evals[gate]["ref_env"]["MICRO_BATCH_SIZE_OVERRIDE"], "4")
                self.assertEqual(self.evals[gate]["ours_env"]["MICRO_BATCH_SIZE_OVERRIDE"], "10")

    def test_no_ref_mbs_five_in_gate_config_or_registry(self) -> None:
        # Whole-source sweep: no MICRO_BATCH_SIZE override of 5 survives.
        offenders: list[str] = []
        sources = list(_GATE_CONFIG_DIR.glob("*.toml"))
        sources.append(REPO_ROOT / "config" / "eval" / "dense_training" / "dense_training.toml")
        for src in sources:
            for line in src.read_text(encoding="utf-8").splitlines():
                low = line.strip().lower()
                if "micro_batch_size" in low and "=" in line:
                    val = line.split("=", 1)[1].strip().strip('"')
                    if val == "5":
                        offenders.append(f"{src.name}: {line.strip()}")
        self.assertEqual(offenders, [], f"stray ref MBS=5: {offenders}")


if __name__ == "__main__":
    unittest.main()
