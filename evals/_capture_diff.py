"""Generic alignment–resume alignment compare driver — model-agnostic.

Both alignment.forward (``forward-align``) and alignment.backward (``backward-align``) — and
the per-step trajectory diffs added for bitwise-singlecard / bitwise-multicard / bitwise-perf / resume — reduce
to the same shape: each side produces a ``dict[str, dict]`` of hash
records (``{hash, shape, dtype}`` per FQN; see
:func:`evals.harness_hook._dump.hash_tensor`) and the gate PASSes iff
every key present on both sides has an identical hash record.

This module owns that intersect-and-diff loop. Nothing here knows
about any specific framework or model architecture; the dispatcher
(``evals.dispatcher._run_align_capture_diff`` and the new Phase-4 in
``_run_bitwise_trajectory`` / ``_run_resume_gate``) owns:

* how to build each side (one subprocess per side — ref bridge for
  the baseline, ``training_engine_tensor.train_loop`` subprocess for
  the candidate);
* what to dump (forward activations under ``fwd.*``, full-backward
  signals under ``bwd.*``, parameter gradients under
  ``grad.<fqn>.{pre,post}allreduce``, manual captures under
  ``loss.*`` — all optionally prefixed by ``step_<n>.`` in bitwise-singlecard–resume
  persistent mode);
* under what FQN (the baseline publishes whatever its framework's
  ``named_modules()`` / ``named_parameters()`` naturally yields — no
  rename table; the candidate is responsible for matching those keys
  if it wants its records compared; keys present on only one side
  are silently skipped, so structural drift surfaces via the graph
  sibling files rather than this loop).

Keeping the diff loop architecture-free means a new model can reuse
the same gate skeleton by writing a new bridge that interposes
:func:`evals.harness_hook.install` into its training stack (see
``evals/harness_hook/recipes/README.md``) — the dispatcher and this
driver stay untouched.

Padded-vocab policy
-------------------
The harness deliberately does **no** comparator-side trim of
padded-vocab weights. The ours engine is responsible for padding
``embedding.word_embeddings.weight`` / ``output_layer.weight``
along the vocab dim to match the ref's
``make-vocab-size-divisible-by`` rounding, so the hash records line
up byte-equal. A shape mismatch surfaced by ``compare_records`` is a
real FAIL signal pointing at engine-side padding drift, not a noise
artifact to be swept under the rug. (The previous
``truncate_padded_to_candidate`` helper was a tensor-era workaround;
hash one-way-discards the bytes so the trim can't happen at compare
time anyway.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class CompareEntry:
    """Outcome of a single hash-record-vs-record compare."""

    name: str
    passed: bool
    actual_hash: str | None = None
    expected_hash: str | None = None
    shape: list[int] | None = None
    dtype: str | None = None
    reason: str | None = None


def compare_records(
    name: str,
    actual: dict,
    expected: dict,
) -> CompareEntry:
    """Compare two hash records.

    PASS iff ``actual["hash"] == expected["hash"]``. A shape or dtype
    mismatch is reported as FAIL with the offending axes spelled out
    in ``reason`` — both differences guarantee a hash mismatch anyway
    (different byte counts), but surfacing the structural difference
    explicitly turns a "two opaque digests differ" failure into an
    actionable one.
    """
    actual_shape = actual.get("shape")
    expected_shape = expected.get("shape")
    actual_dtype = actual.get("dtype")
    expected_dtype = expected.get("dtype")
    actual_hash = actual.get("hash")
    expected_hash = expected.get("hash")

    if actual_shape != expected_shape:
        return CompareEntry(
            name=name,
            passed=False,
            actual_hash=actual_hash,
            expected_hash=expected_hash,
            shape=expected_shape,
            dtype=expected_dtype,
            reason=f"shape mismatch: actual={actual_shape}, expected={expected_shape}",
        )
    if actual_dtype != expected_dtype:
        return CompareEntry(
            name=name,
            passed=False,
            actual_hash=actual_hash,
            expected_hash=expected_hash,
            shape=expected_shape,
            dtype=expected_dtype,
            reason=f"dtype mismatch: actual={actual_dtype}, expected={expected_dtype}",
        )
    return CompareEntry(
        name=name,
        passed=actual_hash == expected_hash,
        actual_hash=actual_hash,
        expected_hash=expected_hash,
        shape=expected_shape,
        dtype=expected_dtype,
    )


def format_entry(entry: CompareEntry) -> str:
    """One-line human-readable rendering of a compare entry."""
    status = "PASS" if entry.passed else "FAIL"
    if entry.reason is not None:
        return f"  [{status}] {entry.name}: {entry.reason}"
    short_actual = (entry.actual_hash or "")[:12]
    short_expected = (entry.expected_hash or "")[:12]
    return (
        f"  [{status}] {entry.name}: "
        f"hash actual={short_actual} expected={short_expected}  "
        f"shape={entry.shape}  dtype={entry.dtype}"
    )


def diff_capture_dicts(
    candidate: Mapping[str, dict],
    baseline: Mapping[str, dict],
    *,
    key_prefix: str | tuple[str, ...] | None = None,
    require_baseline_complete: bool = False,
) -> list[CompareEntry]:
    """Compare keys present on both sides, optionally filtered by prefix.

    Entries are returned in *baseline* iteration order, which for
    forward hooks corresponds to graph execution order and gives a
    natural top-down failure trace.

    When ``key_prefix`` is supplied (str or tuple of str), only keys in
    that tensor *family* are considered on the baseline side. The family
    marker (``fwd.`` / ``bwd.`` / ``grad.``) sits after the namespace
    prefixes (``{step_<n>.}rank<r>.{mb<m>.}``), so a key matches when it
    ``startswith`` the marker (legacy unprefixed dumps) OR contains it as
    a ``.``-delimited segment (``.fwd.`` in ``step_0.rank0.mb0.fwd.x#0``).
    The forward-align gate passes ``"fwd."``, the backward-align gate
    passes ``("grad.", "bwd.")``, and the trajectory / resume diffs
    typically pass ``None`` to compare every record their dumps contain.

    ``require_baseline_complete`` controls what happens to an in-scope
    baseline key the candidate never emitted:

    * ``False`` (default) — the key is skipped, so the compared set is
      whatever *both* publishers chose to emit under matching names
      (the legacy intersection behaviour; trajectory / resume diffs and
      forward-align rely on it).
    * ``True`` — the baseline is authoritative: a missing candidate key
      becomes a FAILED entry. This closes the hole where a candidate
      hides a divergence by simply not dumping the diverging tensor
      (e.g. an ``alignment.backward`` candidate that omits ``bwd.output``).
    """
    if key_prefix is None:
        prefixes: tuple[str, ...] | None = None
    elif isinstance(key_prefix, str):
        prefixes = (key_prefix,)
    else:
        prefixes = tuple(key_prefix)
    entries: list[CompareEntry] = []
    for key in baseline:
        if prefixes is not None and not any(key.startswith(p) or f".{p}" in key for p in prefixes):
            continue
        if key in candidate:
            entries.append(compare_records(key, candidate[key], baseline[key]))
        elif require_baseline_complete:
            entries.append(
                CompareEntry(
                    name=key,
                    passed=False,
                    expected_hash=baseline[key].get("hash"),
                    shape=baseline[key].get("shape"),
                    dtype=baseline[key].get("dtype"),
                    reason="missing in candidate dump (baseline captured "
                    "this key, candidate did not)",
                )
            )
        # else: baseline-only key, not required -> skipped (legacy intersect)
    return entries


@dataclass(frozen=True)
class GateOutcome:
    """Verdict of a forward/backward-align capture gate over two dumps."""

    passed: bool
    bitwise_count: int
    total: int
    no_overlap: bool
    entries: list[CompareEntry]


def capture_gate_outcome(
    candidate: Mapping[str, dict],
    baseline: Mapping[str, dict],
    *,
    key_prefix: str | tuple[str, ...] | None = None,
    require_baseline_complete: bool = False,
    returncode: int = 0,
) -> GateOutcome:
    """The forward/backward-align gate verdict over two merged capture dicts.

    Single source of truth for the dispatcher's Phase-3 decision, so the gate
    logic is testable in isolation:

    * **No in-family key overlaps** between baseline and candidate →
      ``passed=False`` with ``no_overlap=True``. This is the load-bearing
      anti-silent-pass rule: a candidate that emits wrong-namespaced keys
      (e.g. bare ``fwd.x`` instead of ``rank<r>.mb<m>.fwd.x#<call>``) produces
      zero overlap and FAILS the gate rather than trivially "passing" nothing.
    * Otherwise the gate passes iff every compared entry is hash-equal AND the
      candidate process exited 0 (``returncode == 0``).
    """
    entries = diff_capture_dicts(
        candidate,
        baseline,
        key_prefix=key_prefix,
        require_baseline_complete=require_baseline_complete,
    )
    if not entries:
        return GateOutcome(passed=False, bitwise_count=0, total=0, no_overlap=True, entries=entries)
    bitwise = sum(1 for e in entries if e.passed)
    return GateOutcome(
        passed=(bitwise == len(entries) and returncode == 0),
        bitwise_count=bitwise,
        total=len(entries),
        no_overlap=False,
        entries=entries,
    )
