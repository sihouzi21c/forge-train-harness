"""Read a rendered per-gate product (the [cli]/[env] consumption file).

The renderer (``tools.render_gate_configs``) turns each single-source
``gate_config/<gate>.toml`` into two self-contained products::

    <workspace>/ref/config/<gate>.toml             (ref side, frozen)
    <workspace>/workload/src/config/<gate>.toml     (ours side, agent-writable)

Each product carries the FULL effective value set for that side, split by
transport into ``[cli]`` (passed as flags / kwargs) and ``[env]`` (UPPERCASE
environment variables). This module is the single reader both the dispatcher
(shape) and the per-side entries (transport) build on, so there is one place
that knows the product's on-disk shape.

Additive step of the gate-config refactor: nothing here replaces the legacy
env round-trip — consumers opt in via the ``FORGE_GATE_*`` toggles. Reading a
product never runs the renderer; the product is written at lease/freeze time
and is present before either subprocess starts.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Product layout per side, relative to the workspace root.
_SIDE_SUBDIR = {
    "ref": "ref/config",
    "ours": "workload/src/config",
}


class GateProductError(RuntimeError):
    """Raised when a product is missing or malformed."""


@dataclass(frozen=True)
class GateProduct:
    """One side's rendered product: the transport-split effective values."""

    gate: str
    side: str
    cli: dict[str, Any]
    env: dict[str, Any]
    path: Path

    def get(self, key: str, default: Any = None) -> Any:
        """Look a bare key up across both transports.

        ``[cli]`` keys are bare (snake_case); ``[env]`` keys are UPPERCASE.
        A bare lookup checks the cli table first, then the env table under the
        upper-cased name, so callers need not know which transport carries a
        given value.
        """
        if key in self.cli:
            return self.cli[key]
        return self.env.get(key.upper(), default)


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_scalar(v) for v in value) + "]"
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _dump_section(name: str, table: dict[str, Any]) -> list[str]:
    lines = [f"[{name}]"]
    for key, value in table.items():
        lines.append(f"{key} = {_toml_scalar(value)}")
    return lines


def gate_product_path(workspace: Path, side: str, gate: str) -> Path:
    """Resolve the on-disk product path for ``side`` / ``gate``."""
    try:
        sub = _SIDE_SUBDIR[side]
    except KeyError as exc:
        raise GateProductError(f"unknown product side {side!r}") from exc
    return Path(workspace) / sub / f"{gate}.toml"


def load_gate_product(workspace: Path, side: str, gate: str) -> GateProduct:
    """Load the rendered product for ``side`` / ``gate`` under ``workspace``."""
    path = gate_product_path(workspace, side, gate)
    if not path.exists():
        raise GateProductError(f"missing gate product: {path}")
    with open(path, "rb") as fh:
        doc = tomllib.load(fh)
    return GateProduct(
        gate=gate,
        side=side,
        cli=dict(doc.get("cli", {})),
        env=dict(doc.get("env", {})),
        path=path,
    )
