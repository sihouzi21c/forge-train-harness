"""Contract tests: the engine-side runtime_config SSOT loader must
stay in sync with the harness-side [model]/[optim] axis schema.

The engine template at ``harness/workload/src/training_engine_tensor/
runtime_config.py`` reads the gate's [model]/[optim] scalars from the
rendered ours product (``workload/src/config/<gate>.toml`` ``[cli]``),
located via the ``FORGE_GATE`` / ``FORGE_OURS_CONFIG_DIR`` pointers the
harness injects — NOT from ``FORGE_<KEY>`` value env vars. Its dataclass
field names must match the ``[model]`` / ``[optim]`` axis schema keys
(``_MODEL_GEOMETRY_KEYS`` / ``_OPTIM_REQUIRED_KEYS``) exactly — those
field names ARE the product ``[cli]`` keys, so a mismatch means the
product would carry a value the engine never reads, or the engine would
demand a key the renderer never emits. The renderer
(``tools/render_gate_configs.py``) is the sole projection of the axis
values into each gate product.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

from harness.config_runtime import (
    _MODEL_GEOMETRY_KEYS,
    _OPTIM_REQUIRED_KEYS,
    repo_root,
)


def _forge_keys(axis_keys) -> set[str]:
    """Project axis schema keys into the FORGE_<UPPER> env namespace."""
    return {f"FORGE_{k.upper()}" for k in axis_keys}


def _load_engine_runtime_config():
    path = repo_root() / "workload" / "src" / "training_engine_tensor" / "runtime_config.py"
    mod_name = "_engine_runtime_config_under_test"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


# Product ``[cli]`` values (bare lowercase keys — the renderer's naming
# contract; the dataclass field names match these one-for-one). Written
# unquoted so tomllib types them as int/float.
_OPTIM_VALUES = {
    "lr": "3.0e-4",
    "min_lr": "0.0",
    "lr_warmup_iters": "2000",
    "lr_decay_iters": "250000",
    "lr_wsd_decay_iters": "0",
    "weight_decay": "0.1",
    "adam_beta1": "0.9",
    "adam_beta2": "0.95",
    "clip_grad": "1.0",
}
_MODEL_VALUES = {
    "num_layers": "24",
    "hidden_size": "1024",
    "ffn_hidden_size": "4096",
    "num_attention_heads": "16",
    "num_query_groups": "2",
    "head_dim": "64",
    "seq_length": "4096",
    "max_position_embeddings": "4096",
    "padded_vocab_size": "73448",
    "rotary_base": "10000",
    "norm_epsilon": "1e-6",
    "init_method_std": "0.1",
    "mup_base_hidden_size": "256",
    "mup_emb_scale": "12.0",
    "mup_depth_scale": "1.4",
    "mtp_num_layers": "1",
    "mtp_loss_weight": "0.3",
}


def _full_cli() -> dict[str, str]:
    return {**_OPTIM_VALUES, **_MODEL_VALUES}


@contextlib.contextmanager
def _product(cli: dict[str, str]):
    """Write a rendered ours product with the given ``[cli]`` and point the
    engine at it via FORGE_GATE / FORGE_OURS_CONFIG_DIR (pointer-only env use).
    """
    keys = ("FORGE_GATE", "FORGE_OURS_CONFIG_DIR")
    snapshot = {k: os.environ.get(k) for k in keys}
    with tempfile.TemporaryDirectory() as d:
        gate = "unit-gate"
        body = "\n".join(["[cli]"] + [f"{k} = {v}" for k, v in cli.items()]) + "\n"
        (Path(d) / f"{gate}.toml").write_text(body)
        os.environ["FORGE_GATE"] = gate
        os.environ["FORGE_OURS_CONFIG_DIR"] = d
        try:
            yield
        finally:
            for k, v in snapshot.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


class RuntimeConfigContract(unittest.TestCase):
    def setUp(self) -> None:
        self.rc = _load_engine_runtime_config()

    def test_optim_keys_match_harness_projection(self) -> None:
        """Every FORGE_* key the harness can inject for [optim] is read
        by the engine, and the engine reads no extra optim keys.
        """
        engine_keys = {
            f"FORGE_{name.upper()}" for name in self.rc.OptimHParams.__dataclass_fields__
        }
        self.assertEqual(engine_keys, _forge_keys(_OPTIM_REQUIRED_KEYS))

    def test_model_keys_match_harness_projection(self) -> None:
        engine_keys = {
            f"FORGE_{name.upper()}" for name in self.rc.ModelHParams.__dataclass_fields__
        }
        self.assertEqual(engine_keys, _forge_keys(_MODEL_GEOMETRY_KEYS))

    def test_load_from_product_succeeds(self) -> None:
        with _product(_full_cli()):
            cfg = self.rc.load()
        self.assertEqual(cfg.optim.lr, 3.0e-4)
        self.assertEqual(cfg.optim.weight_decay, 0.1)
        self.assertEqual(cfg.model.hidden_size, 1024)
        self.assertEqual(cfg.model.head_dim, 64)
        self.assertEqual(cfg.model.mup_emb_scale, 12.0)

    def test_missing_required_key_raises(self) -> None:
        cli = _full_cli()
        del cli["lr"]
        with _product(cli):
            with self.assertRaises(self.rc.MissingRuntimeConfigError) as ctx:
                self.rc.load()
        self.assertIn("lr", str(ctx.exception))

    def test_missing_product_pointer_raises(self) -> None:
        keys = ("FORGE_GATE", "FORGE_OURS_CONFIG_DIR")
        snapshot = {k: os.environ.pop(k, None) for k in keys}
        try:
            with self.assertRaises(self.rc.MissingRuntimeConfigError):
                self.rc.load()
        finally:
            for k, v in snapshot.items():
                if v is not None:
                    os.environ[k] = v

    def test_no_silent_defaults_in_dataclass(self) -> None:
        """OptimHParams and ModelHParams must not declare field
        defaults — defaults would silently mask a missing harness
        injection, which is the M2 R4 failure pattern.
        """
        import dataclasses as _dc

        for cls in (self.rc.OptimHParams, self.rc.ModelHParams):
            for name, field in cls.__dataclass_fields__.items():
                self.assertIs(field.default, _dc.MISSING, msg=f"{cls.__name__}.{name}")
                self.assertIs(
                    field.default_factory,
                    _dc.MISSING,
                    msg=f"{cls.__name__}.{name}",
                )


if __name__ == "__main__":
    unittest.main()
