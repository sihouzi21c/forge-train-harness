"""Per-operator version router for Stage 2.

The name ``stage2_op_router`` distinguishes this Stage-2 runtime selector
from ``train_engine/src/training_engine_tensor/op_dispatcher.py``, the
*product*-side dispatcher in the produced training engine. The two
serve different concerns; the distinct filename keeps the boundary
visible at every import site.

Each operator has an environment variable ``OP_<NAME>`` that selects which
implementation to use.  Values are ``baseline`` (stage1 frozen), ``v1``,
``v2``, etc.

Registration is automatic: each operator directory contains a
``register.toml`` file with its env_var, default version, and available
versions.  This module scans ``workload/ops/*/register.toml`` at import
time and builds the registry.  No manual editing of this file is needed.

Usage in forward.py / backward.py::

    from tools.stage2_op_router import get_op_version
    if get_op_version("attention") == "baseline":
        ...  # original TE path
    else:
        from workload.ops.attention.kernel import attention_fwd
        ...
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

_BOOTSTRAP_ROOT = Path(__file__).resolve().parent.parent
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from harness._compat import tomllib
from harness.config_runtime import op_register_path, ops_root_path
from harness.config_runtime import repo_root as _repo_root


@dataclass(frozen=True)
class OpEntry:
    env_var: str
    default: str
    available: tuple[str, ...]


# Resolve once at import; ``op_register_path`` / ``ops_root_path`` keep
# the on-disk layout SSOT inside ``harness.config_runtime``.
_REPO_ROOT = _repo_root()
_OPS_DIR = ops_root_path(_REPO_ROOT)


def _load_registry() -> dict[str, OpEntry]:
    """Scan every ``register.toml`` under the ops root and build the registry.

    Any malformed ``register.toml`` raises immediately so the broken
    registration is visible rather than silently skipped.
    """
    registry: dict[str, OpEntry] = {}
    if not _OPS_DIR.is_dir():
        return registry
    for op_path in sorted(p for p in _OPS_DIR.iterdir() if p.is_dir()):
        op_name = op_path.name
        if op_name.startswith("_"):
            continue
        reg_file = op_register_path(_REPO_ROOT, op_name)
        if not reg_file.exists():
            continue
        with open(reg_file, "rb") as f:
            data = tomllib.load(f)
        registry[op_name] = OpEntry(
            env_var=data.get("env_var", f"OP_{op_name.upper()}"),
            default=data.get("default", "baseline"),
            available=tuple(data.get("available", ["baseline"])),
        )
    return registry


_REGISTRY: dict[str, OpEntry] = _load_registry()


def get_op_version(op_name: str) -> str:
    """Return the active version string for *op_name*.

    Resolution order:
    1. Environment variable ``OP_<NAME>`` (uppercase, hyphens->underscores)
    2. Registry default
    3. ``"baseline"`` if op_name is unknown
    """
    entry = _REGISTRY.get(op_name)
    if entry is None:
        env_key = f"OP_{op_name.upper().replace('-', '_')}"
        return os.environ.get(env_key, "baseline")
    return os.environ.get(entry.env_var, entry.default)


def list_ops() -> dict[str, dict[str, str | tuple[str, ...]]]:
    """Return a snapshot of all registered operators and their active versions."""
    result = {}
    for name, entry in sorted(_REGISTRY.items()):
        result[name] = {
            "env_var": entry.env_var,
            "default": entry.default,
            "active": get_op_version(name),
            "available": entry.available,
        }
    return result


if __name__ == "__main__":
    import json

    print(json.dumps(list_ops(), indent=2, default=str))
