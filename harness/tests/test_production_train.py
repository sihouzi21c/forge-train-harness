"""Tests for the ``production`` milestone ``production-train`` gate (ours-only
long train + periodic checkpoint, thin-dispatcher form).

Covers the contract seams production-train keeps on this machine:

  1. Registration: the registry routes production-train through the generic
     scripted executor (``ours_runner`` sh + ``evals.verdicts.production``)
     and ``run_schema.RUNNER_METRICS`` declares its metric contract.
  2. Crash-resume discovery: ``production_ckpt.latest_checkpoint`` scans the
     save root and returns the highest completed ``step_<N>`` so a restarted
     run continues from the last checkpoint instead of step 0.
  3. Shape validity per eval variant, via the same resolver the verdict uses
     (``evals.verdicts.production._production_shape``).

The legacy dispatcher-orchestrated 4-segment loop tests (segment env
composition, resume-skips, per-segment crash) were retired with the handler:
that behavior now lives in ``evals/scripts/run_ours_production.sh`` and is
verified end-to-end on the GPU devspace, not by subprocess mocks here.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from harness import run_schema
from harness._compat import tomllib
from harness.tests import _gate_render

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = REPO_ROOT / "evals" / "scripts"
for _p in (str(REPO_ROOT), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import production_ckpt  # noqa: E402

from evals.verdicts import production  # noqa: E402

_REGISTRY = REPO_ROOT / "config" / "eval" / "dense_training" / "dense_training.toml"


def _seed_checkpoints(save_root: Path, steps: list[int]) -> None:
    for s in steps:
        d = save_root / f"step_{s}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "training_state.pt").write_bytes(b"x")


class TestRegistration(unittest.TestCase):
    def test_runner_kind_registered(self) -> None:
        # production-train routes through the generic scripted executor:
        # ours runner sh + in-process verdict module, with runner_kind kept
        # only as the RUNNER_METRICS contract key.
        cfg = tomllib.loads(_REGISTRY.read_text(encoding="utf-8"))
        entry = cfg["evals"]["production-train"]
        self.assertEqual(entry["ours_runner"], "run_ours_production")
        self.assertEqual(entry["verdict"], "production")
        self.assertIn("production-train", run_schema.RUNNER_METRICS)

    def test_runner_metrics_entry_exists(self) -> None:
        self.assertIn("production-train", run_schema.RUNNER_METRICS)

    def test_metrics_keys_appear_in_verdict_body(self) -> None:
        # Mirrors test_runner_metrics_contract: every declared metric key
        # must be emitted as a string literal in the verdict module.
        src = (REPO_ROOT / "evals" / "verdicts" / "production.py").read_text(encoding="utf-8")
        for key in run_schema.RUNNER_METRICS["production-train"]:
            self.assertIn(f'"{key}"', src, f"metric {key!r} not emitted in verdict")


class TestLatestCheckpointDiscovery(unittest.TestCase):
    """production_ckpt.latest_checkpoint — the crash-resume primitive shared
    by the sh segment loop and the production verdict."""

    def test_empty_root_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            resume_dir, done_step = production_ckpt.latest_checkpoint(Path(tmp))
            self.assertIsNone(resume_dir)
            self.assertEqual(done_step, 0)

    def test_missing_root_returns_zero(self) -> None:
        resume_dir, done_step = production_ckpt.latest_checkpoint(
            Path("/nonexistent/checkpoints/xyz")
        )
        self.assertIsNone(resume_dir)
        self.assertEqual(done_step, 0)

    def test_picks_highest_nonempty_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _seed_checkpoints(root, [4, 8])
            resume_dir, done_step = production_ckpt.latest_checkpoint(root)
            self.assertEqual(done_step, 8)
            self.assertEqual(resume_dir, root / "step_8")

    def test_ignores_empty_checkpoint_dir(self) -> None:
        # A half-written step dir with no non-empty *.pt must not be
        # treated as a completed checkpoint (fail-safe resume).
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _seed_checkpoints(root, [4])
            (root / "step_8").mkdir()  # empty — crashed mid-write
            resume_dir, done_step = production_ckpt.latest_checkpoint(root)
            self.assertEqual(done_step, 4)
            self.assertEqual(resume_dir, root / "step_4")


# Every supported model-size eval variant ships its own config/eval/*.toml.
# production-train must exist for each: the runner reads the run shape
# straight out of the rendered ours product, so a size that lacks
# the suite simply has no production milestone. This guards the exact
# omission of adding production-train to
# the 0.5B config but not the 1B one.
_EVAL_VARIANTS = (
    "dense_training/dense_training.toml",
    "dense_training_1b/dense_training_1b.toml",
    "dense_training_qwen3/dense_training_qwen3.toml",
)

# Model-agnostic wiring: identical across sizes. forge_init_ones / seed /
# seq_length / shape are no longer inline (they live in gate_config → product)
# and are checked via the rendered product (TestForgeInitOnesPerSuite /
# test_each_variant_shape_is_valid); only the registry identity keys stay here.
_SHARED_KEYS = ("stage", "runner_kind", "milestone", "script", "launcher")


class TestEvalConfigParity(unittest.TestCase):
    def _load(self, name: str) -> dict:
        path = REPO_ROOT / "config" / "eval" / name
        return tomllib.loads(path.read_text(encoding="utf-8"))

    def test_production_train_present_in_every_variant(self) -> None:
        for name in _EVAL_VARIANTS:
            with self.subTest(variant=name):
                evals = self._load(name).get("evals", {})
                self.assertIn(
                    "production-train",
                    evals,
                    f"{name} is missing the [evals.production-train] suite",
                )

    def test_shared_wiring_identical_across_variants(self) -> None:
        cfgs = {n: self._load(n)["evals"]["production-train"] for n in _EVAL_VARIANTS}
        ref_name, ref_cfg = next(iter(cfgs.items()))
        for name, cfg in cfgs.items():
            for key in _SHARED_KEYS:
                with self.subTest(variant=name, key=key):
                    self.assertEqual(
                        cfg.get(key),
                        ref_cfg.get(key),
                        f"{name}[{key}] != {ref_name}[{key}] — model-agnostic "
                        "wiring must match across eval variants",
                    )

    def test_each_variant_shape_is_valid(self) -> None:
        # Reuse the production resolver so the test enforces the SAME
        # invariants the verdict does (grad_accum>=1, num_steps divisible
        # by the segment count). The shape now lives in the rendered ours
        # product, so render each variant and resolve from it.
        for variant in ("dense_training", "dense_training_1b", "dense_training_qwen3"):
            with self.subTest(variant=variant):
                if _gate_render.missing_render_inputs(variant):
                    self.skipTest(f"{variant}: render unavailable")
                rendered = _gate_render.render_variant(variant)
                try:
                    cfg = rendered.registry["evals"]["production-train"]
                    segments = production._resolve_segments(
                        cfg, rendered.workspace, "production-train"
                    )
                    ws, num_steps, mbs, grad_accum = production._production_shape(
                        cfg, rendered.workspace, "production-train", segments
                    )
                    self.assertGreaterEqual(segments, 1)
                    self.assertGreaterEqual(grad_accum, 1)
                    self.assertEqual(num_steps % segments, 0)
                    self.assertGreater(mbs * ws, 0)
                finally:
                    rendered.cleanup()


# ``forge_init_ones`` SSOT for the per-suite init regime.
#
# Every suite that touches canonical_state_fp32.pt must declare which
# init regime it loads. The field is REQUIRED (no default in dispatcher
# / launcher), so a typo or omission must surface as a test failure
# rather than as a silent ones-init fallback during a bitwise gate.
_FORGE_INIT_ONES_MATRIX: dict[str, int] = {
    "forward-align": 0,
    "backward-align": 0,
    "multistep-1gpu": 0,
    "multistep": 0,
    "perf-bitwise": 0,
    "resume-gate-20": 0,
    "long-train": 1,
    "long-train-smoke": 1,
    "loss-gate-200": 1,
    "op-long": 1,
    "production-train": 1,
}


_FORGE_VARIANTS = ("dense_training", "dense_training_1b", "dense_training_qwen3")
# op-long is D5-exempt (product-less): its forge_init_ones stays inline in the
# registry; every other canonical-loading gate carries it in its product.
_FORGE_INLINE_GATES = {"op-long"}


@unittest.skipIf(
    _gate_render.missing_render_inputs(),
    f"render inputs absent: {_gate_render.missing_render_inputs()}",
)
class TestForgeInitOnesPerSuite(unittest.TestCase):
    """Enforce per-gate ``forge_init_ones`` SSOT across every eval variant.

    The dispatcher / launcher refuse to default-init; this test guarantees the
    per-suite value matches the canonical milestone matrix, so a regression
    that flips a bitwise gate to ones init (or vice versa) trips here. The
    value now lives in each gate's rendered product (gate_config single
    source); product-less op-long keeps it inline in the registry.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.rendered = {v: _gate_render.render_variant(v) for v in _FORGE_VARIANTS}

    @classmethod
    def tearDownClass(cls) -> None:
        for r in cls.rendered.values():
            if r is not None:
                r.cleanup()

    def test_every_canonical_loading_suite_declares_field(self) -> None:
        for variant in _FORGE_VARIANTS:
            rendered = self.rendered[variant]
            if rendered is None:
                self.skipTest(f"{variant}: render unavailable")
            evals = rendered.registry.get("evals", {})
            for suite, expected in _FORGE_INIT_ONES_MATRIX.items():
                with self.subTest(variant=variant, suite=suite):
                    self.assertIn(
                        suite,
                        evals,
                        f"{variant}: matrix expects [evals.{suite}] but suite is "
                        "missing — update _FORGE_INIT_ONES_MATRIX if a gate was renamed.",
                    )
                    if suite in _FORGE_INLINE_GATES:
                        value = evals[suite].get("forge_init_ones")
                    else:
                        # ours product exists for every product gate (incl.
                        # ours-only production-train); ref may not.
                        value = rendered.product(suite, "ours").get("forge_init_ones")
                    self.assertIsNotNone(
                        value, f"{variant}: gate {suite!r} declares no forge_init_ones"
                    )
                    self.assertEqual(
                        int(value),
                        expected,
                        f"{variant}: gate {suite!r} forge_init_ones = {value!r} but matrix "
                        f"expects {expected!r}. Flipping this changes which canonical the "
                        "gate loads — confirm intentional before updating the matrix.",
                    )


if __name__ == "__main__":
    unittest.main()
