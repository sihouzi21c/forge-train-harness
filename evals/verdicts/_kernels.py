"""Comparison kernels shared by the verdict modules.

One place for the arithmetic + accounting every trajectory/hash/MFU gate
repeats; the verdict modules keep ONLY what is genuinely gate-specific
(structural assertions, window shapes, summary strings, metrics keys).

Kernels:

* ``compare_series`` / ``step_checks``   — per-step abs-diff over a window
  (bitwise loss+grad loops, resume dual-trajectory, wsd-sft per-phase)
* ``hash_dump_gate`` + ``capture_entry_details`` — two-dump per-FQN hash
  diff (bitwise gate 1c, resume) and the shared checks-detail rendering
  (also align)
* ``filtered_avg``                       — warmup-filtered wire-metric
  average (MFU / step time)
* ``ref_failure_result`` / ``ref_missing_steps_result`` — the identical
  ref-side preflight failures of bitwise / loss_gate / long_train
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from evals._capture_diff import diff_capture_dicts, format_entry
from evals._common import classify_ref_failure, missing_window_steps
from evals.verdicts._shared import RefPhase, read_text, tail
from harness import run_schema
from harness.capture_artifacts import load_merged_capture

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Per-step series comparison
# ──────────────────────────────────────────────────────────────────────


@dataclass
class StepCompare:
    """One window step of a baseline-vs-candidate scalar series."""

    step: int
    baseline: float | None
    ours: float | None
    abs_diff: float | None  # None when either side is missing


def compare_series(
    baseline_by_step: Mapping[int, Any],
    ours_by_step: Mapping[int, Any],
    steps: Iterable[int],
) -> list[StepCompare]:
    """Absolute per-step diff of two scalar series over a step window.

    A ``None`` value (or absent key) on either side yields a record with
    ``abs_diff=None`` — the caller decides whether that is a missing-step
    failure or an inline failed check.
    """
    records: list[StepCompare] = []
    for step in steps:
        b = baseline_by_step.get(step)
        o = ours_by_step.get(step)
        if b is None or o is None:
            records.append(
                StepCompare(
                    step=step,
                    baseline=None if b is None else float(b),
                    ours=None if o is None else float(o),
                    abs_diff=None,
                )
            )
            continue
        b_f, o_f = float(b), float(o)
        records.append(StepCompare(step=step, baseline=b_f, ours=o_f, abs_diff=abs(o_f - b_f)))
    return records


def step_checks(
    records: list[StepCompare],
    *,
    atol: float,
    kind: str,
    missing_reason: str,
) -> tuple[list[dict[str, Any]], int, int, bool]:
    """Render a compared series into per-step check records.

    Returns ``(checks, bitwise_count, within_count, all_ok)`` with the
    legacy check schema: ``step_<n>_<kind>`` names, ``inf`` diff +
    ``reason`` for missing entries, ``ours``/``baseline`` otherwise.
    """
    checks: list[dict[str, Any]] = []
    bitwise_count = 0
    within_count = 0
    all_ok = True
    for rec in records:
        if rec.abs_diff is None:
            all_ok = False
            checks.append(
                {
                    "name": f"step_{rec.step}_{kind}",
                    "max_abs_diff": float("inf"),
                    "passed": False,
                    "reason": missing_reason,
                }
            )
            continue
        passed = rec.abs_diff <= atol
        checks.append(
            {
                "name": f"step_{rec.step}_{kind}",
                "max_abs_diff": rec.abs_diff,
                "passed": passed,
                "ours": rec.ours,
                "baseline": rec.baseline,
            }
        )
        if rec.abs_diff == 0.0:
            bitwise_count += 1
        if passed:
            within_count += 1
        else:
            all_ok = False
    return checks, bitwise_count, within_count, all_ok


# ──────────────────────────────────────────────────────────────────────
# Two-dump per-FQN hash gate
# ──────────────────────────────────────────────────────────────────────


@dataclass
class HashGateResult:
    entries: list[Any]
    rendered: list[str]
    passed: bool
    summary: str

    @property
    def bitwise_count(self) -> int:
        return sum(1 for e in self.entries if e.passed)


def hash_dump_gate(
    candidate_path: Path,
    baseline_path: Path,
    *,
    baseline_label: str = "ref",
    candidate_label: str = "ours",
) -> HashGateResult:
    """Diff two on-disk capture dumps (candidate vs baseline), fail-closed.

    Missing dump / empty overlap both fail with the legacy summary strings
    (labels parameterize the ``ref=… ours=…`` / ``ref=… res=…`` variants).
    """
    try:
        baseline = load_merged_capture(baseline_path)
    except FileNotFoundError:
        baseline = None
    try:
        candidate = load_merged_capture(candidate_path)
    except FileNotFoundError:
        candidate = None
    if baseline is None or candidate is None:
        return HashGateResult(
            entries=[],
            rendered=[],
            passed=False,
            summary=(
                f"hash dump missing — {baseline_label}={baseline_path.exists()} "
                f"{candidate_label}={candidate_path.exists()}"
            ),
        )
    entries = diff_capture_dicts(candidate, baseline, key_prefix=None)
    rendered = [format_entry(e) for e in entries]
    bitwise = sum(1 for e in entries if e.passed)
    total = len(entries) if entries else 1
    if entries:
        summary = f"hash {bitwise}/{total} equal"
    else:
        summary = (
            f"hash dump empty — {baseline_label}={len(baseline)} keys, "
            f"{candidate_label}={len(candidate)} keys, no overlap"
        )
    return HashGateResult(
        entries=entries,
        rendered=rendered,
        passed=bool(entries) and all(e.passed for e in entries),
        summary=summary,
    )


def capture_entry_details(entries: list[Any]) -> list[dict[str, Any]]:
    """The shared ``hash_checks`` / ``checks`` details rendering."""
    return [
        {
            "name": e.name,
            "passed": e.passed,
            "actual_hash": e.actual_hash,
            "expected_hash": e.expected_hash,
            "shape": e.shape,
            "dtype": e.dtype,
            "reason": e.reason,
        }
        for e in entries
    ]


# ──────────────────────────────────────────────────────────────────────
# Warmup-filtered wire-metric average (MFU / step time)
# ──────────────────────────────────────────────────────────────────────


def filtered_avg(
    loss_steps: list[dict[str, Any]],
    *,
    warmup_steps: int,
    key: str = "mfu_e2e_standard",
) -> tuple[float, int]:
    """Average ``key`` over post-warmup steps that carry it.

    Returns ``(avg, count)``; ``avg`` is 0.0 when no step qualifies —
    callers decide whether an empty sample fails the gate.
    """
    values = [float(s[key]) for s in loss_steps if s["step"] >= warmup_steps and key in s]
    if not values:
        return 0.0, 0
    return sum(values) / len(values), len(values)


# ──────────────────────────────────────────────────────────────────────
# Ref-side preflight failures (identical strings in bitwise / loss_gate /
# long_train)
# ──────────────────────────────────────────────────────────────────────


def ref_failure_result(
    *,
    suite_key: str,
    label: str,
    ref: RefPhase,
    ref_dir: Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Failed result for a ref side that exited dirty / timed out."""
    ref_stdout = tail(read_text(ref_dir / "ref.log"))
    failure_class = classify_ref_failure(
        timed_out=bool(ref.status["timed_out"]),
        returncode=int(ref.status["returncode"]),
        stdout_text=ref_stdout,
    )
    exit_desc = "timed out" if ref.status["timed_out"] else f"returncode={ref.status['returncode']}"
    return run_schema.make_failed_result(
        suite=suite_key,
        summary=(f"{label}: ref script {failure_class} — {exit_desc} after {ref.elapsed_s:.0f}s."),
        metrics={"ref_failure_class": failure_class},
        details={
            "config": cfg,
            "ref_stdout": ref_stdout,
            "ref_failure_class": failure_class,
        },
    )


def ref_missing_steps_result(
    *,
    suite_key: str,
    label: str,
    baseline_by_step: Mapping[int, Any],
    gate_steps: range,
    cfg: dict[str, Any],
) -> dict[str, Any] | None:
    """Failed result when the ref trajectory misses window steps, else None."""
    missing_ref = missing_window_steps(baseline_by_step, gate_steps)
    if not missing_ref:
        return None
    return run_schema.make_failed_result(
        suite=suite_key,
        summary=(
            f"{label}: ref-script trajectory missing steps in window "
            f"[{gate_steps.start}, {gate_steps.stop}). First missing: {missing_ref[:10]}"
        ),
        metrics={"ref_steps_observed": len(baseline_by_step)},
        details={"config": cfg},
    )
