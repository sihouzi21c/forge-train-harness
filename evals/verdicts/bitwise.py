"""Bitwise-trajectory verdict (bitwise-singlecard / bitwise-multicard / bitwise-perf).

File-based port of the legacy ``dispatcher._run_bitwise_trajectory`` judgment
(Gates 1 / 1b / 1c / 2, identical semantics). Reads:

* ``<run_dir>/ref/status.json`` + ``<run_dir>/ref/ref.log``
  + ``<run_dir>/ref/dump/ref_loss.txt``   — ref phase exit + trajectory
* ``<run_dir>/ref/ref_hash_dump.json``    — ref hash dump (level > 0)
* ``<run_dir>/ours/status.json`` + ``<run_dir>/ours/ours.log``
  — ours phase exit + [LOSS] trajectory
* ``<run_dir>/ours/ours_hash_dump.json``  — ours hash dump (level > 0)
* the rendered ref product                — shape (gate window / steps /
  world_size) and verdict thresholds

Gates (all must hold):
  1   per-step loss |ours − ref| ≤ gate_atol (max of loss/grad abs thresholds)
  1b  per-step grad_norm |ours − ref| ≤ grad_norm_abs_threshold; a ref
      trajectory with NO grad baseline fails fast
  1c  per-FQN per-step hash diff, when hash_capture_level > 0
  2   avg MFU(standard) ≥ mfu_e2e_target, when the target is > 0
plus ours returncode == 0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evals._common import missing_window_steps
from evals.gate_product import GateProductError
from evals.gate_shape import load_gate_shape
from evals.verdicts._kernels import (
    capture_entry_details,
    compare_series,
    filtered_avg,
    hash_dump_gate,
    ref_failure_result,
    ref_missing_steps_result,
    step_checks,
)
from evals.verdicts._shared import (
    overlay_product_verdict,
    resolve_ours_phase,
    resolve_ref_phase,
    tail,
    verdict_cli,
)
from harness import run_schema

if TYPE_CHECKING:
    from pathlib import Path


def run(
    *,
    suite_key: str,
    run_dir: Path,
    repo_root: Path,
    workload_config: dict[str, Any],
) -> dict[str, Any]:
    cfg = overlay_product_verdict(
        dict(workload_config.get("evals", {}).get(suite_key) or {}),
        repo_root,
        suite_key,
    )
    milestone = str(cfg.get("milestone", suite_key))
    summary_prefix = f"{milestone} {suite_key}"

    gate_atol = max(
        float(cfg.get("loss_abs_threshold", 0)),
        float(cfg.get("grad_norm_abs_threshold", 0)),
    )
    grad_atol = float(cfg.get("grad_norm_abs_threshold", 0))
    mfu_target = float(cfg.get("mfu_e2e_target", 0))
    has_mfu_gate = mfu_target > 0
    warmup_steps = int(cfg.get("warmup_steps", 1))
    hash_capture_level = int(cfg.get("hash_capture_level", 0))

    ref_dir = run_dir / "ref"
    ours_dir = run_dir / "ours"
    ref_hash_dump = ref_dir / "ref_hash_dump.json"
    ours_hash_dump = ours_dir / "ours_hash_dump.json"

    # ── Shape: the rendered ref product is the unconditional source ──
    try:
        shape = load_gate_shape(repo_root, suite_key)
        gate_steps = range(
            shape.metadata_int("gate_window_start"),
            shape.metadata_int("gate_window_end"),
        )
        num_steps = shape.metadata_int("num_steps")
        world_size = shape.metadata_int("world_size")
    except GateProductError as exc:
        return run_schema.make_failed_result(
            suite=suite_key,
            summary=f"{summary_prefix}: gate shape unavailable from product: {exc}",
            metrics={},
            details={"config": cfg},
        )
    gate_lo, gate_hi = gate_steps.start, gate_steps.stop

    # ── Ref phase: status + trajectory from on-disk artifacts ──
    ref = resolve_ref_phase(run_dir, hash_capture_level=hash_capture_level)
    ref_elapsed_s = ref.elapsed_s
    baseline_by_step = ref.baseline_by_step
    grad_baseline_by_step = ref.grad_baseline_by_step

    if not ref.succeeded:
        return ref_failure_result(
            suite_key=suite_key, label=summary_prefix, ref=ref, ref_dir=ref_dir, cfg=cfg
        )

    ref_missing = ref_missing_steps_result(
        suite_key=suite_key,
        label=summary_prefix,
        baseline_by_step=baseline_by_step,
        gate_steps=gate_steps,
        cfg=cfg,
    )
    if ref_missing is not None:
        return ref_missing

    # ── Ours phase ──
    ours_phase = resolve_ours_phase(run_dir)
    returncode = ours_phase.returncode
    output = ours_phase.output
    if ours_phase.timed_out:
        run_timeout = ours_phase.timeout_s
        return {
            "status": "failed",
            "suite": suite_key,
            "summary": f"{summary_prefix}: ours timed out after {run_timeout}s",
            "metrics": {"world_size": world_size},
            "details": {"config": cfg, "output_tail": tail(output)},
        }

    script_succeeded = returncode == 0
    loss_steps = ours_phase.loss_steps
    if not loss_steps:
        return {
            "status": "failed",
            "suite": suite_key,
            "summary": (
                f"{summary_prefix}: ours emitted no [LOSS] lines; returncode={returncode}."
            ),
            "metrics": {"world_size": world_size},
            "details": {"config": cfg, "output_tail": tail(output)},
        }

    ours_by_step = ours_phase.ours_by_step
    missing = missing_window_steps(ours_by_step, gate_steps)
    if missing:
        return {
            "status": "failed",
            "suite": suite_key,
            "summary": (
                f"{summary_prefix}: ours missing [LOSS] for {len(missing)} "
                f"window steps. First: {missing[:10]}."
            ),
            "metrics": {"world_size": world_size, "steps_observed": len(loss_steps)},
            "details": {"config": cfg, "output_tail": tail(output)},
        }

    # ── Gate 1: per-step bitwise loss comparison ──
    loss_records = compare_series(
        baseline_by_step,
        {step: ours_by_step[step]["global_loss"] for step in gate_steps},
        gate_steps,
    )
    checks, bitwise_count, gate_pass_count, correctness_pass = step_checks(
        loss_records,
        atol=gate_atol,
        kind="loss",
        missing_reason="missing baseline trajectory entry",
    )
    total = len(checks)

    # ── Gate 1b: per-step bitwise grad_norm comparison ──
    # A ref trajectory with no grad_norm baseline must FAIL FAST rather than
    # silently grading on loss alone (same policy as the legacy handler).
    grad_total = 0
    grad_bitwise_count = 0
    grad_gate_pass_count = 0
    if not grad_baseline_by_step:
        correctness_pass = False
        checks.append(
            {
                "name": "grad_norm_baseline",
                "max_abs_diff": float("inf"),
                "passed": False,
                "reason": "ref trajectory carries no grad_norm baseline",
            }
        )
    else:
        grad_records = compare_series(
            grad_baseline_by_step,
            {step: ours_by_step[step].get("grad_norm") for step in gate_steps},
            gate_steps,
        )
        grad_checks, grad_bitwise_count, grad_gate_pass_count, grad_ok = step_checks(
            grad_records,
            atol=grad_atol,
            kind="grad_norm",
            missing_reason="missing grad_norm baseline or ours entry",
        )
        checks.extend(grad_checks)
        grad_total = len(grad_records)
        correctness_pass = correctness_pass and grad_ok

    # ── Gate 1c: per-FQN per-step hash diff (hash_capture_level > 0) ──
    hash_entries: list[Any] = []
    hash_rendered: list[str] = []
    hash_pass = True
    hash_summary: str | None = None
    if hash_capture_level > 0:
        hash_gate = hash_dump_gate(
            ours_hash_dump, ref_hash_dump, baseline_label="ref", candidate_label="ours"
        )
        hash_entries = hash_gate.entries
        hash_rendered = hash_gate.rendered
        hash_pass = hash_gate.passed
        hash_summary = hash_gate.summary

    # ── Gate 2: MFU (mfu_e2e_target > 0 only) ──
    avg_mfu = 0.0
    mfu_pass = True
    mfu_status: str | None = None
    if has_mfu_gate:
        avg_mfu, mfu_samples = filtered_avg(loss_steps, warmup_steps=warmup_steps)
        if not mfu_samples:
            mfu_pass = False
            mfu_status = (
                f"MFU gate configured (target {mfu_target:.1f}%) but ours emitted "
                "no mfu_e2e_standard values — refusing to silently pass."
            )
        else:
            mfu_pass = avg_mfu >= mfu_target
            mfu_status = (
                f"MFU(standard) {avg_mfu:.1f}% {'≥' if mfu_pass else '<'} {mfu_target:.1f}%"
            )

    overall = correctness_pass and hash_pass and mfu_pass and script_succeeded
    summary_parts = [
        f"correctness {bitwise_count}/{total} bitwise, "
        f"{gate_pass_count}/{total} within gate {gate_atol:.0e}"
    ]
    if grad_baseline_by_step:
        summary_parts.append(
            f"grad_norm {grad_bitwise_count}/{grad_total} bitwise, "
            f"{grad_gate_pass_count}/{grad_total} within gate {grad_atol:.0e}"
        )
    else:
        summary_parts.append("grad_norm baseline missing")
    if hash_summary is not None:
        summary_parts.append(hash_summary)
    if mfu_status is not None:
        summary_parts.append(mfu_status)
    if not script_succeeded:
        summary_parts.append(f"ours returncode={returncode}")

    return {
        "status": "passed" if overall else "failed",
        "suite": suite_key,
        "summary": (
            f"{summary_prefix} (DP={world_size}, {num_steps} steps "
            f"vs ref-script {ref_elapsed_s:.0f}s): {'; '.join(summary_parts)}."
        ),
        "metrics": {
            "checks_total": total,
            "bitwise_match": bitwise_count,
            "gate_pass": gate_pass_count,
            "gate_atol": gate_atol,
            "grad_checks_total": grad_total,
            "grad_bitwise_match": grad_bitwise_count,
            "grad_gate_pass": grad_gate_pass_count,
            "grad_gate_atol": grad_atol,
            "world_size": world_size,
            "num_steps": num_steps,
            "steps_observed": len(loss_steps),
            "ref_steps_observed": len(baseline_by_step),
            "ref_elapsed_s": ref_elapsed_s,
            "gate_window": [gate_lo, gate_hi],
            "avg_mfu_e2e_standard": avg_mfu if has_mfu_gate else None,
            "mfu_target": mfu_target if has_mfu_gate else None,
            "mfu_pass": mfu_pass if has_mfu_gate else None,
            "correctness_pass": correctness_pass,
            "hash_capture_level": hash_capture_level,
            "hash_pass": hash_pass if hash_capture_level > 0 else None,
            "hash_checks_total": len(hash_entries) if hash_capture_level > 0 else None,
            "hash_bitwise_match": (
                sum(1 for e in hash_entries if e.passed) if hash_capture_level > 0 else None
            ),
        },
        "details": {
            "config": cfg,
            "checks": checks,
            "ref_dump_dir": str(ref_dir / "dump"),
            "trajectory": loss_steps,
            "output_tail": tail(output),
            "hash_checks": (capture_entry_details(hash_entries) if hash_capture_level > 0 else []),
            "hash_rendered": hash_rendered if hash_capture_level > 0 else [],
            "ref_hash_dump": str(ref_hash_dump) if hash_capture_level > 0 else None,
            "ours_hash_dump": str(ours_hash_dump) if hash_capture_level > 0 else None,
        },
    }


def main(argv: list[str] | None = None) -> int:
    return verdict_cli(run, argv, doc=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
