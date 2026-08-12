"""SSOT for the M1 capture-mode artifact contract.

Sister module of :mod:`harness.wire_format` — that module pins the
*line-oriented* protocol every gate script speaks on stdout; this one
pins the *file-oriented* protocol the M1 capture flow speaks on disk.

The dispatcher (``evals/dispatcher.py::_run_align_capture_diff``) consumes
two ``CaptureArtifacts`` objects per gate (one from the L0 reference,
one from the candidate engine) and diffs the intersected tensor dict.
Both producers and the consumer route through this module so the
on-disk contract is enforced at exactly one place.

Public surface
--------------
``CaptureArtifacts``
    Resolved bundle of files a single capture run produces.

``ref_capture_paths`` / ``candidate_capture_paths``
    Build a ``CaptureArtifacts`` from the configured basenames + the
    capture run's destination dir. The basenames live in
    ``config/eval.toml [defaults]`` and are surfaced via
    :func:`harness.config_runtime.ref_capture_basename` /
    :func:`harness.config_runtime.candidate_capture_basename` so a
    project that wants a different filename convention flips one
    config key, not every callsite.

Naming conventions
------------------
* ``<basename>``                — JSON hash records ``{hash, shape,
                                  dtype}`` per captured tensor (NOT raw
                                  tensors, despite a ``.pt`` basename);
                                  keys are ``fwd.<module_fqn>`` and
                                  ``grad.<param_fqn>``. Written by the
                                  same ``_dump.dump_capture_files`` as the
                                  bitwise path.
* ``<basename>.graph.json``     — execution-order trace recorded by
                                  the same forward hooks. Optional in
                                  principle (older producers may not
                                  emit it); the dispatcher surfaces
                                  the path in result details when
                                  present.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


__all__ = [
    "CaptureArtifacts",
    "candidate_capture_paths",
    "graph_sibling_path",
    "load_merged_capture",
    "ref_capture_paths",
]


@dataclasses.dataclass(frozen=True)
class CaptureArtifacts:
    """Files a single capture run produces.

    ``tensor_dump`` is the capture dump the hook injector writes — a
    JSON hash-record file (``{hash, shape, dtype}`` per tensor), even
    though its configured basename carries a legacy ``.pt`` extension.
    ``graph_dump`` is the sibling JSON; ``None`` only when the producer
    is an older revision that didn't emit it (the dispatcher's
    ``details`` payload still records ``None`` so the drift is visible).
    """

    dump_dir: Path
    tensor_dump: Path
    graph_dump: Path | None

    @property
    def tensor_dump_exists(self) -> bool:
        return self.tensor_dump.exists()


def graph_sibling_path(tensor_dump: Path) -> Path:
    """Return the sibling graph JSON path for a tensor dump.

    Single source of truth for the ``.graph.json`` suffix. Both
    producers (the standard hook's dump-path resolution +
    ``training_engine_tensor.train_loop``'s capture path) and
    consumers (dispatcher result rendering) route through here so a
    future rename only touches one place.
    """
    return tensor_dump.with_name(tensor_dump.name + ".graph.json")


def _capture_paths(dump_dir: Path, basename: str) -> CaptureArtifacts:
    tensor_dump = dump_dir / basename
    graph_dump = graph_sibling_path(tensor_dump)
    return CaptureArtifacts(
        dump_dir=dump_dir,
        tensor_dump=tensor_dump,
        graph_dump=graph_dump if graph_dump.exists() else None,
    )


def ref_capture_paths(dump_dir: Path, basename: str) -> CaptureArtifacts:
    """Resolve the reference-side capture paths under ``dump_dir``."""
    return _capture_paths(dump_dir, basename)


def candidate_capture_paths(dump_dir: Path, basename: str) -> CaptureArtifacts:
    """Resolve the candidate-side capture paths under ``dump_dir``."""
    return _capture_paths(dump_dir, basename)


def load_merged_capture(tensor_dump: Path) -> dict:
    """Load a capture dump, merging every per-rank ``<file>.rank<r>`` shard.

    Each rank writes its own ``<tensor_dump>.rank<r>`` file whose keys carry
    the ``rank<r>.`` namespace, so the shards are disjoint and merge cleanly
    into one dict spanning all ranks — every rank / microbatch / call is
    compared, not just rank 0's. Falls back to the unsuffixed ``tensor_dump``
    when no per-rank shards exist (single-rank / legacy single-file dumps).
    """
    import json

    # ``.rank*`` also matches the ``.rank<r>.graph.json`` siblings — exclude
    # them; they are execution-order lists, not the hash-record dicts.
    shards = sorted(
        f
        for f in tensor_dump.parent.glob(tensor_dump.name + ".rank*")
        if not f.name.endswith(".graph.json")
    )
    if shards:
        merged: dict = {}
        for shard in shards:
            merged.update(json.loads(shard.read_text(encoding="utf-8")))
        return merged
    return json.loads(tensor_dump.read_text(encoding="utf-8"))
