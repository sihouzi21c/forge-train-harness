"""Shared test helper: render a directory-form eval registry to products.

The flat eval registries are gone; per-gate shape / thresholds / init now
live in ``config/eval/<variant>/gate_config/<gate>.toml`` and are resolved
into ``ref/config/<gate>.toml`` + ``workload/src/config/<gate>.toml`` by
``tools.render_gate_configs`` at lease/freeze time. Tests that need the
effective per-gate values render the same way and read the products.

Usage::

    rendered = render_variant("dense_training")     # keep the handle alive
    if rendered is None: self.skipTest(...)
    prod = rendered.product("perf-bitwise", "ref")  # GateProduct
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_CONFIG = REPO_ROOT / "config"

# A valid axis combo per variant. Only infra-env values depend on the choice;
# the cli shape the assertions check does not.
_VARIANT_AXES = {
    "dense_training": {
        "model": "model/minicpm4_0.5b.toml",
        "optim": "optim/default.toml",
        "ref": "ref/torch_minicpm4_0.5b.toml",
        "data": "data/gsm8k_hf.toml",
    },
    "dense_training_1b": {
        "model": "model/minicpm5_1b.toml",
        "optim": "optim/minicpm5_1b.toml",
        "ref": "ref/torch_1b.toml",
        "data": "data/ultra_fineweb.toml",
    },
    "dense_training_qwen3": {
        "model": "model/qwen3_0.6b.toml",
        "optim": "optim/qwen3_0.6b.toml",
        "ref": "ref/torch_qwen3_0.6b.toml",
        "data": "data/gsm8k_hf.toml",
    },
    "dense_training_8b": {
        "model": "model/minicpm4_8b.toml",
        "optim": "optim/default.toml",
        "ref": "ref/torch_minicpm4_8b.toml",
        "data": "data/ultra_fineweb.toml",
    },
}

try:
    from tools import render_gate_configs as _rgc

    _IMPORT_ERROR = ""
except Exception as exc:  # tomli_w absent, etc.
    _rgc = None
    _IMPORT_ERROR = repr(exc)


def missing_render_inputs(variant: str = "dense_training") -> list[str]:
    """Return the inputs that block rendering (empty when render is possible)."""
    miss: list[str] = []
    if _rgc is None:
        miss.append(f"render import failed: {_IMPORT_ERROR}")
    suite = _CONFIG / "eval" / variant
    if not (suite / f"{variant}.toml").exists():
        miss.append(str(suite / f"{variant}.toml"))
    if not (suite / "gate_config").is_dir():
        miss.append(str(suite / "gate_config"))
    for rel in _VARIANT_AXES.get(variant, {}).values():
        if not (_CONFIG / rel).exists():
            miss.append(str(_CONFIG / rel))
    return miss


@dataclass
class RenderedVariant:
    """Rendered products + the parsed registry for one variant."""

    workspace: Path
    registry: dict
    _tmp: tempfile.TemporaryDirectory

    def product(self, gate: str, side: str):
        from evals.gate_product import load_gate_product

        return load_gate_product(self.workspace, side, gate)

    def cleanup(self) -> None:
        self._tmp.cleanup()


def render_variant(variant: str = "dense_training") -> RenderedVariant | None:
    """Render *variant*'s gate_config into products; return a handle or None
    when render inputs are absent. Caller MUST keep the handle alive (its
    TemporaryDirectory backs ``workspace``) and call ``cleanup()``."""
    if missing_render_inputs(variant):
        return None
    suite = _CONFIG / "eval" / variant
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    cfg_dir = root / "config"
    cfg_dir.mkdir()
    for axis, rel in _VARIANT_AXES[variant].items():
        shutil.copy(_CONFIG / rel, cfg_dir / f"{axis}.toml")
    shutil.copy(suite / f"{variant}.toml", cfg_dir / "eval.toml")
    shutil.copytree(suite / "gate_config", cfg_dir / "gate_config")
    workspace = root / "workspace"
    rc = _rgc.main(["--workspace", str(workspace), "--config-dir", str(cfg_dir)])
    if rc != 0:
        tmp.cleanup()
        raise AssertionError(f"render returned {rc} for variant {variant!r}")
    with (cfg_dir / "eval.toml").open("rb") as fh:
        registry = tomllib.load(fh)
    return RenderedVariant(workspace=workspace, registry=registry, _tmp=tmp)
