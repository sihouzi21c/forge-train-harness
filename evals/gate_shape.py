"""Resolve a gate's run shape directly from its rendered product.

This retired the ``ref → gate_metadata.json → ours`` round-trip. Historically
the dispatcher learned the gate shape (steps, world_size, batch geometry, gate
window, seed) by reading the metadata the ref subprocess wrote *after it ran*.
The same shape is fully determined up front by the single source the renderer
already resolved into the ref product, so the dispatcher reads it without
waiting on ref.

``load_gate_shape`` reads the ref product — the frozen truth both sides are
rendered from. It is now the unconditional shape source for every Stage-1
gate; there is no toggle. The only remaining ``gate_metadata.json`` consumer
is the product-less, D5-exempt ``op-long`` suite.

The accessors mirror the integer keys the dispatcher pulled from metadata
(``num_steps``/``world_size``/``seed``/``micro_batch_size``/``seq_length``/
``grad_accum_steps``/``global_batch_size``/``gate_window_start``/
``gate_window_end``/``resume_save_step``) so a caller can swap a
``_ref_metadata_int(ref_run, key)`` call for ``shape.metadata_int(key)`` with
no name translation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from evals.gate_product import GateProduct, GateProductError, load_gate_product

if TYPE_CHECKING:
    from pathlib import Path

# Metadata int keys whose value lives behind a differently-shaped product
# field: gate_window is a single [start, end] list in the product, split here
# into the two scalar keys the dispatcher reads.
_WINDOW_KEYS = ("gate_window_start", "gate_window_end")


@dataclass(frozen=True)
class GateShape:
    """The run shape for one gate, sourced from a rendered product."""

    gate: str
    side: str
    product: GateProduct

    def _raw(self, key: str) -> Any:
        if key in _WINDOW_KEYS:
            window = self.product.get("gate_window")
            if not isinstance(window, (list, tuple)) or len(window) != 2:
                raise GateProductError(
                    f"{self.product.path}: gate_window must be a [start, end] "
                    f"pair to resolve {key!r}, got {window!r}"
                )
            return window[0] if key == "gate_window_start" else window[1]
        value = self.product.get(key)
        if value is None:
            raise GateProductError(
                f"{self.product.path}: gate {self.gate!r} product has no shape key {key!r}"
            )
        return value

    def metadata_int(self, key: str) -> int:
        """Return an integer shape value by its legacy metadata key name.

        Drop-in for ``dispatcher_stage2._ref_metadata_int(ref_run, key)``: same keys,
        same int coercion, same fail-fast on a missing/non-integer value.
        """
        raw = self._raw(key)
        try:
            return int(raw)
        except (TypeError, ValueError) as exc:
            raise GateProductError(
                f"{self.product.path}: shape key {key!r} is not an integer: {raw!r}"
            ) from exc

    def has(self, key: str) -> bool:
        """Whether an (optional) shape key is present, e.g. resume_save_step."""
        if key in _WINDOW_KEYS:
            window = self.product.get("gate_window")
            return isinstance(window, (list, tuple)) and len(window) == 2
        return self.product.get(key) is not None


def load_gate_shape(workspace: Path, gate: str, side: str = "ref") -> GateShape:
    """Load the run shape for ``gate`` from its rendered product.

    Defaults to the ref side — the frozen truth the metadata round-trip used
    to report. Pass ``side="ours"`` only where a per-side override (e.g. the
    op-long ours MBS) means the two products differ in shape.
    """
    product = load_gate_product(workspace, side, gate)
    return GateShape(gate=gate, side=side, product=product)
