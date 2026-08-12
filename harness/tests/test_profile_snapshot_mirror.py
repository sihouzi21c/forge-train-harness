"""profile-snapshot mirrors the gate suite named by its label milestone.

profile-snapshot owns no shape of its own. It declares
``mirror_gate = { "bitwise-perf" = "perf-bitwise", "long-horizon" = "long-train" }``
and — since the thin-dispatcher migration — the mirror is resolved at RENDER
time: ``tools.render_gate_configs._synthesize_mirror_variants`` copies the
mirrored gate's resolved ours dict into an ours-only variant product
``workload/src/config/profile-snapshot@<milestone>.toml`` (judge-only keys
stripped, profile's own num_steps/warmup_steps/nsys overlays applied). The
ours sh launches from that variant and ``evals.verdicts.profile`` reads the
run shape back out of it.

This test renders the directory-form registry + gate_config (the same path
the lease/freeze hook uses) and checks the variant products against the
mirrored gates' ours products; the label→milestone fail-fast semantics moved
to ``profile._resolve_milestone`` and are checked there.
"""

from __future__ import annotations

import sys
import tomllib
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.verdicts import profile  # noqa: E402
from harness.tests import _gate_render  # noqa: E402

_REGISTRY = REPO_ROOT / "config" / "eval" / "dense_training" / "dense_training.toml"

# Milestone → mirrored gate, as declared in the registry.
_MIRROR = {"bitwise-perf": "perf-bitwise", "long-horizon": "long-train"}

# Shape keys the variant must copy verbatim from the mirrored gate's ours
# product (the exact keys the profile verdict reads, plus the launch shape).
# forge_init_ones is a shape key: resolve_deploy derives the variant's
# CHECKPOINT_ROOT ones/no1 subdir from it — stripping it left the profile
# run with a bare root and no canonical to load.
_MIRRORED_SHAPE_KEYS = (
    "world_size",
    "micro_batch_size",
    "global_batch_size",
    "grad_accum_steps",
    "deterministic",
    "seq_length",
    "forge_init_ones",
)

# Judge-only keys the renderer strips from mirror variants
# (tools/render_gate_configs._MIRROR_STRIP_KEYS).
_STRIPPED_KEYS = (
    "gate_window",
    "gate_bitwise",
    "gate_atol",
    "hash_capture_level",
    "mfu_e2e_target",
    "loss_rel_threshold",
    "loss_abs_threshold",
    "grad_norm_abs_threshold",
    "max_avg_relative_loss_diff",
    "resume_save_step",
)


@unittest.skipIf(
    _gate_render.missing_render_inputs(),
    f"render inputs absent: {_gate_render.missing_render_inputs()}",
)
class TestProfileSnapshotMirrorVariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rendered = _gate_render.render_variant("dense_training")
        assert cls.rendered is not None

    @classmethod
    def tearDownClass(cls) -> None:
        cls.rendered.cleanup()

    def _variant(self, milestone: str):
        return self.rendered.product(f"profile-snapshot@{milestone}", "ours")

    def test_variant_products_exist(self) -> None:
        cfg_dir = self.rendered.workspace / "workload" / "src" / "config"
        for milestone in _MIRROR:
            with self.subTest(milestone=milestone):
                self.assertTrue(
                    (cfg_dir / f"profile-snapshot@{milestone}.toml").exists(),
                    f"renderer did not synthesize the {milestone} mirror variant",
                )

    def test_variant_shape_matches_mirrored_gate_product(self) -> None:
        for milestone, gate in _MIRROR.items():
            variant = self._variant(milestone)
            mirrored = self.rendered.product(gate, "ours")
            for key in _MIRRORED_SHAPE_KEYS:
                with self.subTest(milestone=milestone, key=key):
                    self.assertEqual(
                        variant.get(key),
                        mirrored.get(key),
                        f"variant profile-snapshot@{milestone} must carry {gate}'s ours {key}",
                    )

    def test_bitwise_perf_variant_mirrors_perf_bitwise(self) -> None:
        variant = self._variant("bitwise-perf")
        # perf-bitwise: WS=2, MBS=4, GBS=80, det ON.
        self.assertEqual(variant.get("world_size"), 2)
        self.assertEqual(variant.get("micro_batch_size"), 4)
        self.assertEqual(variant.get("global_batch_size"), 80)
        self.assertEqual(variant.get("grad_accum_steps"), 10)  # 80 / (4 * 2)
        self.assertIs(variant.get("deterministic"), True)

    def test_long_horizon_variant_mirrors_long_train(self) -> None:
        variant = self._variant("long-horizon")
        # long-train: ours MBS=10, GBS=80, det OFF.
        self.assertEqual(variant.get("world_size"), 2)
        self.assertEqual(variant.get("micro_batch_size"), 10)
        self.assertEqual(variant.get("global_batch_size"), 80)
        self.assertEqual(variant.get("grad_accum_steps"), 4)  # 80 / (10 * 2)
        self.assertIs(variant.get("deterministic"), False)

    def test_profile_keeps_own_num_steps_not_gate_steps(self) -> None:
        # The variant must NOT pull the gate's 50/200 step count;
        # profile-snapshot owns a small diagnostic num_steps.
        self.assertEqual(self._variant("bitwise-perf").get("num_steps"), 12)
        self.assertEqual(self._variant("long-horizon").get("num_steps"), 12)

    def test_profile_overlays_warmup_and_nsys(self) -> None:
        for milestone in _MIRROR:
            variant = self._variant(milestone)
            with self.subTest(milestone=milestone):
                self.assertEqual(variant.get("warmup_steps"), 5)
                self.assertIs(variant.get("nsys_profile"), True)

    def test_judge_only_keys_stripped(self) -> None:
        for milestone in _MIRROR:
            variant = self._variant(milestone)
            for key in _STRIPPED_KEYS:
                with self.subTest(milestone=milestone, key=key):
                    self.assertIsNone(
                        variant.get(key),
                        f"judge-only key {key!r} must be stripped from the "
                        "mirror variant (the profile run grades nothing)",
                    )


class TestProfileMilestoneResolution(unittest.TestCase):
    """Label→milestone fail-fast, now in evals.verdicts.profile."""

    @classmethod
    def setUpClass(cls) -> None:
        with _REGISTRY.open("rb") as fh:
            cls.cfg = tomllib.load(fh)["evals"]["profile-snapshot"]

    def test_known_labels_resolve(self) -> None:
        self.assertEqual(
            profile._resolve_milestone(self.cfg, "profile-snapshot", "bitwise-perf_round0"),
            ("bitwise-perf", "perf-bitwise"),
        )
        self.assertEqual(
            profile._resolve_milestone(self.cfg, "profile-snapshot", "long-horizon_round3"),
            ("long-horizon", "long-train"),
        )

    def test_unknown_milestone_fails_fast(self) -> None:
        with self.assertRaises(ValueError):
            profile._resolve_milestone(self.cfg, "profile-snapshot", "bogus_round0")

    def test_label_without_milestone_prefix_fails_fast(self) -> None:
        with self.assertRaises(ValueError):
            profile._resolve_milestone(self.cfg, "profile-snapshot", "round0")


class TestProfileSnapshotConfigShape(unittest.TestCase):
    """The registry declares the mirror map and owns no per-side shape."""

    @classmethod
    def setUpClass(cls) -> None:
        with _REGISTRY.open("rb") as fh:
            cls.cfg = tomllib.load(fh)["evals"]["profile-snapshot"]

    def test_declares_mirror_gate(self) -> None:
        self.assertEqual(
            self.cfg.get("mirror_gate"),
            {"bitwise-perf": "perf-bitwise", "long-horizon": "long-train"},
        )

    def test_owns_num_steps(self) -> None:
        self.assertEqual(self.cfg.get("num_steps"), 12)

    def test_no_self_owned_shape(self) -> None:
        self.assertNotIn("ref_env", self.cfg)
        self.assertNotIn("ours_env", self.cfg)
        self.assertNotIn("deterministic", self.cfg)


if __name__ == "__main__":
    unittest.main()
