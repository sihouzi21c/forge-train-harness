"""Parity between the product readers and the renderer's resolved truth.

The gate-config refactor adds two product readers — ``evals.gate_product``
(transport: [cli]/[env]) and ``evals.gate_shape`` (the dispatcher's shape
source that retires the metadata round-trip). This test locks the invariant
that makes the round-trip safe to drop: **the value a consumer reads from a
rendered product is byte-identical to what the renderer resolved from the
single source.** One source, two readers, no drift.

It is a CPU/off-GPU check. The end-to-end bitwise equivalence of the legacy
env path vs. the product path is the GPU job (devspace); this guards the
read-side contract those products are consumed through.

The fixture assembles a config home from the committed axis templates + the
target-state registry + its gate_config single sources, renders both products
into a tmp workspace, then compares each product against an independent
``render_one`` / ``split_transport`` of the same single source.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path

from evals import dispatcher_stage2
from evals.gate_product import load_gate_product
from evals.gate_shape import load_gate_shape
from harness.config_runtime import _MODEL_REQUIRED_KEYS, _OPTIM_REQUIRED_KEYS
from tools import product_env
from tools import render_gate_configs as rgc

# Repo layout anchors (this file lives at harness/harness/tests/).
_HARNESS = Path(__file__).resolve().parents[2]
_CONFIG = _HARNESS / "config"
_SUITE = _CONFIG / "eval" / "dense_training_1b"
_GATE_CONFIG = _SUITE / "gate_config"

# A valid axis combo for the target-state suite. Only infra-env values depend
# on this choice; the cli shape (what the parity assertions check) does not.
_AXES = {
    "model": _CONFIG / "model" / "minicpm5_1b.toml",
    "optim": _CONFIG / "optim" / "minicpm5_1b.toml",
    "ref": _CONFIG / "ref" / "torch_1b.toml",
    "data": _CONFIG / "data" / "ultra_fineweb.toml",
}

# Integer shape keys gate_shape exposes under their legacy metadata names.
_SHAPE_KEYS = (
    "num_steps",
    "world_size",
    "seed",
    "micro_batch_size",
    "seq_length",
    "grad_accum_steps",
    "global_batch_size",
    "gate_window_start",
    "gate_window_end",
)


def _missing_inputs() -> list[str]:
    missing = [str(p) for p in _AXES.values() if not p.exists()]
    if not _GATE_CONFIG.is_dir():
        missing.append(str(_GATE_CONFIG))
    return missing


@unittest.skipIf(_missing_inputs(), f"render inputs absent: {_missing_inputs()}")
class GateTransportParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.cfg_dir = root / "config"
        cls.cfg_dir.mkdir()
        # Seed the axis files + registry under the committed basenames.
        for axis, src in _AXES.items():
            shutil.copy(src, cls.cfg_dir / f"{axis}.toml")
        shutil.copy(_SUITE / "dense_training_1b.toml", cls.cfg_dir / "eval.toml")
        shutil.copytree(_GATE_CONFIG, cls.cfg_dir / "gate_config")

        cls.workspace = root / "workspace"
        rc = rgc.main(["--workspace", str(cls.workspace), "--config-dir", str(cls.cfg_dir)])
        assert rc == 0, f"render returned {rc}"

        # Independent re-resolve of the same single sources for comparison.
        model = rgc._load(cls.cfg_dir / "model.toml")
        optim = rgc._load(cls.cfg_dir / "optim.toml")
        ref = rgc._load(cls.cfg_dir / "ref.toml")
        data = rgc._load(cls.cfg_dir / "data.toml")
        registry = rgc._load(cls.cfg_dir / "eval.toml")
        cls.baseline = rgc._build_baseline(model, optim, ref, data, registry)
        cls.env_whitelist = {k.upper() for k in rgc.ENV_INFRA} | {
            k.upper() for k in registry.get("env", {})
        }
        cls.gates = sorted(p.stem for p in (cls.cfg_dir / "gate_config").glob("*.toml"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _expected(self, gate: str, side: str) -> tuple[dict, dict]:
        gate_src = rgc._load(self.cfg_dir / "gate_config" / f"{gate}.toml")
        resolved = rgc.render_one(gate_src, side, self.baseline)
        return rgc.split_transport(resolved, self.env_whitelist)

    def test_at_least_the_known_gates_rendered(self) -> None:
        self.assertIn("multistep", self.gates)
        self.assertIn("long-train", self.gates)

    def test_product_matches_renderer(self) -> None:
        """load_gate_product == renderer's split for every gate/side."""
        for gate in self.gates:
            gate_src = rgc._load(self.cfg_dir / "gate_config" / f"{gate}.toml")
            for side in ("ref", "ours"):
                if side not in gate_src:
                    continue
                with self.subTest(gate=gate, side=side):
                    exp_cli, exp_env = self._expected(gate, side)
                    product = load_gate_product(self.workspace, side, gate)
                    self.assertEqual(product.cli, exp_cli)
                    self.assertEqual(product.env, exp_env)

    def test_product_reader_matches_raw_toml(self) -> None:
        """The reader does not transform the on-disk product."""
        for gate in self.gates:
            gate_src = rgc._load(self.cfg_dir / "gate_config" / f"{gate}.toml")
            for side in ("ref", "ours"):
                if side not in gate_src:
                    continue
                with self.subTest(gate=gate, side=side):
                    product = load_gate_product(self.workspace, side, gate)
                    with open(product.path, "rb") as fh:
                        doc = tomllib.load(fh)
                    self.assertEqual(product.cli, doc.get("cli", {}))
                    self.assertEqual(product.env, doc.get("env", {}))

    def test_shape_matches_renderer_ints(self) -> None:
        """gate_shape.metadata_int == int(resolved value) for present keys.

        This is the dispatcher's shape-source contract: reading the ref
        product yields exactly the integers the metadata round-trip used to
        report.
        """
        for gate in self.gates:
            gate_src = rgc._load(self.cfg_dir / "gate_config" / f"{gate}.toml")
            if "ref" not in gate_src:
                continue
            resolved = rgc.render_one(gate_src, "ref", self.baseline)
            shape = load_gate_shape(self.workspace, gate, "ref")
            for key in _SHAPE_KEYS:
                with self.subTest(gate=gate, key=key):
                    if key in ("gate_window_start", "gate_window_end"):
                        window = resolved.get("gate_window")
                        if not (isinstance(window, (list, tuple)) and len(window) == 2):
                            self.assertFalse(shape.has(key))
                            continue
                        expected = window[0] if key.endswith("start") else window[1]
                        self.assertEqual(shape.metadata_int(key), int(expected))
                    elif key in resolved:
                        self.assertTrue(shape.has(key))
                        self.assertEqual(shape.metadata_int(key), int(resolved[key]))
                    else:
                        self.assertFalse(shape.has(key))

    def test_ours_product_covers_all_axis_keys(self) -> None:
        """Collapse completeness: each gate's ours product carries every
        [model] + [optim] axis key in its [cli] table, so the engine's
        runtime_config.load() finds every FORGE_* hyperparameter it requires
        sourced from the single product — no dispatcher-side injection.
        """
        required = set(_MODEL_REQUIRED_KEYS) | set(_OPTIM_REQUIRED_KEYS)
        for gate in self.gates:
            gate_src = rgc._load(self.cfg_dir / "gate_config" / f"{gate}.toml")
            if "ours" not in gate_src:
                continue
            with self.subTest(gate=gate):
                product = load_gate_product(self.workspace, "ours", gate)
                missing = required - set(product.cli)
                self.assertEqual(missing, set(), f"{gate}: ours product missing {missing}")

    def test_verdict_thresholds_come_from_product(self) -> None:
        """Collapse D1: ``overlay_product_verdict`` lifts the verdict
        thresholds + ``forge_init_ones`` from the rendered product onto the
        gate cfg — the product, not the registry, is the verdict SSOT.
        (Lives in evals.verdicts._shared since the handler migration.)
        """
        from evals.verdicts import _shared

        for gate in self.gates:
            gate_src = rgc._load(self.cfg_dir / "gate_config" / f"{gate}.toml")
            if "ref" not in gate_src:
                continue
            product = load_gate_product(self.workspace, "ref", gate)
            verdict_in_product = {
                k: product.get(k)
                for k in _shared.PRODUCT_VERDICT_KEYS
                if product.get(k) is not None
            }
            if not verdict_in_product:
                continue
            with self.subTest(gate=gate):
                # Start from a cfg whose verdict values are deliberately wrong;
                # the overlay must replace them with the product's truth.
                stale_cfg = {k: "STALE" for k in verdict_in_product}
                merged = _shared.overlay_product_verdict(stale_cfg, self.workspace, gate)
                for key, val in verdict_in_product.items():
                    self.assertEqual(merged[key], val)


@unittest.skipIf(_missing_inputs(), f"render inputs absent: {_missing_inputs()}")
class StageAShapeSourceTest(unittest.TestCase):
    """The dispatcher's shape source is the rendered ref product, no toggle.

    Collapse: ``FORGE_GATE_SHAPE_SOURCE`` is gone — every Stage-1 gate carries
    a ref product, so ``_maybe_gate_shape`` returns a real ``GateShape``; the
    only ``None`` case is a product-less (D5-exempt) suite like ``op-long``.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cfg_dir = root / "config"
        cfg_dir.mkdir()
        for axis, src in _AXES.items():
            shutil.copy(src, cfg_dir / f"{axis}.toml")
        shutil.copy(_SUITE / "dense_training_1b.toml", cfg_dir / "eval.toml")
        shutil.copytree(_GATE_CONFIG, cfg_dir / "gate_config")
        cls.workspace = root / "workspace"
        assert rgc.main(["--workspace", str(cls.workspace), "--config-dir", str(cfg_dir)]) == 0

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_product_is_unconditional_shape_source(self) -> None:
        # No toggle, no env: the rendered ref product is read unconditionally
        # and yields the product's value (8 for multistep).
        shape = dispatcher_stage2._maybe_gate_shape("multistep", self.workspace)
        self.assertIsNotNone(shape)
        self.assertEqual(shape.metadata_int("num_steps"), 8)

    def test_product_less_suite_returns_none(self) -> None:
        # op-long keeps its shape flat in the registry (no gate_config product)
        # → no GateShape, dispatcher falls back to its legacy metadata path.
        self.assertIsNone(dispatcher_stage2._maybe_gate_shape("op-long", self.workspace))


# StageBOursProductTest was retired with the FORGE_GATE/FORGE_OURS_CONFIG_DIR
# env pair: GateInputs now takes an explicit ``--config <product>`` path and the
# per-run env-over-product precedence changed with it. The current transport
# semantics are locked by harness/tests/test_gate_entry_config.py.


@unittest.skipIf(_missing_inputs(), f"render inputs absent: {_missing_inputs()}")
class StageBRefTransportTest(unittest.TestCase):
    """The generic projection of the rendered product is the complete ref SSOT.

    ``tools/product_env.py`` has ZERO key knowledge, so completeness is a
    property of the PRODUCT itself: every launcher-required parameter (shape +
    model geometry + optimizer + init) must be carried by the rendered product
    and surface in the export map under its generic upper-cased name — the
    exact name the launcher's fail-fast ``${VAR:?}`` probe reads.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cfg_dir = root / "config"
        cfg_dir.mkdir()
        for axis, src in _AXES.items():
            shutil.copy(src, cfg_dir / f"{axis}.toml")
        shutil.copy(_SUITE / "dense_training_1b.toml", cfg_dir / "eval.toml")
        shutil.copytree(_GATE_CONFIG, cfg_dir / "gate_config")
        cls.workspace = root / "workspace"
        assert rgc.main(["--workspace", str(cls.workspace), "--config-dir", str(cfg_dir)]) == 0
        cls.ref_cfg = cls.workspace / "ref" / "config"

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_product_projection_carries_complete_ref_ssot(self) -> None:
        # (product == gate_config single source is verified by
        # GateProductParityTest.test_product_matches_renderer.)
        checked = 0
        for product in sorted(self.ref_cfg.glob("*.toml")):
            gate = product.stem
            with self.subTest(gate=gate):
                exported = product_env.export_map(product)
                for key in (
                    "WORLD_SIZE",
                    "NUM_STEPS",
                    "MICRO_BATCH_SIZE",
                    "GLOBAL_BATCH_SIZE",
                    "GRAD_ACCUM_STEPS",
                    "SEQ_LENGTH",
                    "SEED",
                    "NUM_LAYERS",
                    "HIDDEN_SIZE",
                    "NUM_ATTENTION_HEADS",
                    "INIT_METHOD_STD",
                    "FORGE_INIT_ONES",
                    "LR",
                    "WEIGHT_DECAY",
                ):
                    self.assertIn(key, exported, f"{gate}: projection missing {key}")
                checked += 1
        self.assertGreater(checked, 0, "no rendered ref product to check")


class RefConsumerNameConsistencyTest(unittest.TestCase):
    """Launcher consumer name == product key upper-cased — no translation layer.

    Every fail-fast ``${VAR:?}`` probe in a ref launcher must name either the
    upper-cased form of a renderer ``[cli]`` key (what the generic projection
    exports) or a documented runtime/deployment injection. Any other name
    would need a per-key translation layer between product and launcher —
    the registry this refactor deleted. This pins the door shut.
    """

    _LAUNCHERS = (
        _HARNESS / "ref" / "reference" / "run_16gpu_1000step_pure_mup_mtp.sh",
        _HARNESS / "ref" / "reference" / "run_qwen3_dense.sh",
        _HARNESS / "ref" / "reference" / "run_minicpm4_8b_dptp.sh",
        _HARNESS / "ref" / "reference" / "train_minicpm4_0.5b_fineweb_modelbestsdk.sh",
        _HARNESS / "ref" / "reference" / "train_minicpm4_0.5b_gsm8k.sh",
    )

    # B-class runtime/deployment values injected by runtime_env.py /
    # _ref_script_path_env (or derived in-launcher before the probe), never
    # product [cli] keys.
    _RUNTIME_INJECTED = frozenset({"DATA_CONF", "DATA_PATH", "MEGATRON_ROOT", "TOKENIZER_MODEL"})

    def test_probe_names_are_product_export_names(self) -> None:
        product_names = {k.upper() for k in rgc._CLI_ORDER}
        for sh in self._LAUNCHERS:
            text = sh.read_text(encoding="utf-8")
            probes = set(re.findall(r"\$\{([A-Z0-9_]+):\?", text))
            unknown = probes - product_names - self._RUNTIME_INJECTED
            with self.subTest(launcher=sh.name):
                self.assertFalse(
                    unknown,
                    f"{sh.name}: fail-fast probes with no product-key source "
                    f"(translation layer?): {sorted(unknown)}",
                )


class RefLauncherFailFastEnvTest(unittest.TestCase):
    """Collapse: the generic-projection torch launchers take ALL gate
    parameters from the caller-projected environment (ref-generic-projection
    plan, Step 2) — no in-launcher registry sourcing survives, and a missing
    required key dies on the first ``:?`` probe instead of falling back to a
    baked preset. run_qwen3_dense.sh's twin contract is pinned by
    test_qwen3_ref.TestQwen3Launcher.
    """

    _LAUNCHERS = (
        _HARNESS / "ref" / "reference" / "run_16gpu_1000step_pure_mup_mtp.sh",
        _HARNESS / "ref" / "reference" / "run_minicpm4_8b_dptp.sh",
        _HARNESS / "ref" / "reference" / "train_minicpm4_0.5b_fineweb_modelbestsdk.sh",
        _HARNESS / "ref" / "reference" / "train_minicpm4_0.5b_gsm8k.sh",
    )

    def test_no_in_launcher_registry_sourcing(self) -> None:
        for sh in self._LAUNCHERS:
            text = sh.read_text(encoding="utf-8")
            with self.subTest(launcher=sh.name):
                self.assertNotIn("gate_product_to_shell", text)
                # Registry-emitted shape names must be gone. DATA_PATH_OVERRIDE
                # is exempt: a documented caller knob (local-data sanity
                # override), never product-carried or emitter-named.
                self.assertNotRegex(
                    text,
                    r"(NUM_STEPS|MICRO_BATCH_SIZE|GLOBAL_BATCH_SIZE)_OVERRIDE",
                )

    @unittest.skipUnless(shutil.which("bash"), "bash required for launcher exec")
    def test_launcher_exits_nonzero_without_projected_env(self) -> None:
        for sh in self._LAUNCHERS:
            with self.subTest(launcher=sh.name):
                proc = subprocess.run(
                    ["bash", str(sh)],
                    # GPUS_PER_NODE pinned so the nvidia-smi autodetect (absent
                    # on CPU-only CI) can't abort before the required-key probes.
                    env={
                        "PATH": os.environ.get("PATH", ""),
                        "LOCAL_MODE": "1",
                        "GPUS_PER_NODE": "2",
                    },
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn("num_steps", proc.stderr)


class BaselineDataKeyRenderTest(unittest.TestCase):
    """No data-axis value is EVER rendered into the product.

    The data axis is a separate SSOT (config/data.toml). data_path is resolved
    at run time by the dispatcher sourcing the data_conf (--data-path-file);
    data_loader is read straight from [data].data_loader via FORGE_DATA_TOML
    (ref/sisters --data-config, ours runtime_config.data_loader()). A product
    copy of either would be a second source of truth, so the renderer projects
    neither — even when data.toml declares them.
    """

    def test_absent_data_keys_are_omitted(self) -> None:
        baseline = rgc._build_baseline({}, {}, {}, {}, {})
        self.assertNotIn("data_loader", baseline)
        self.assertNotIn("data_path", baseline)

    def test_declared_data_keys_are_not_projected(self) -> None:
        data = {"data": {"data_loader": "hf", "data_path": "0.5 a/*.parquet"}}
        baseline = rgc._build_baseline({}, {}, {}, data, {})
        self.assertNotIn("data_loader", baseline)
        self.assertNotIn("data_path", baseline)


if __name__ == "__main__":
    unittest.main()
