"""Loss-gate verdict (loss-gate-200).

File-based port of the legacy ``dispatcher._run_loss_gate`` judgment
(identical semantics and summary strings). Reads:

* ``<run_dir>/ref/status.json`` + ``<run_dir>/ref/ref.log``
  + ``<run_dir>/ref/dump/ref_loss.txt``   — ref phase exit + trajectory
* ``<run_dir>/ours/status.json`` + ``<run_dir>/ours/ours.log``
  — ours phase exit + [LOSS] trajectory
* the rendered ref product                — shape (gate window / steps /
  world_size) and verdict thresholds

Gate: average relative loss diff over the gate window
< max_avg_relative_loss_diff (REQUIRED key — a missing threshold raises
at the runner boundary rather than silently passing), plus ours
returncode == 0. Avg MFU is reported as diagnostic only, never gated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evals._common import missing_window_steps, window_loss_diff_metrics
from evals.gate_product import GateProductError
from evals.gate_shape import load_gate_shape
from evals.verdicts._kernels import (
    filtered_avg,
    ref_failure_result,
    ref_missing_steps_result,
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

    max_avg_rel = float(cfg["max_avg_relative_loss_diff"])

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
            summary=f"{milestone} {suite_key}: gate shape unavailable from product: {exc}",
            metrics={},
            details={"config": cfg},
        )
    gate_start, gate_end = gate_steps.start, gate_steps.stop

    # ── Ref phase: status + trajectory from on-disk artifacts ──
    ref = resolve_ref_phase(run_dir)
    baseline_by_step = ref.baseline_by_step

    if not ref.succeeded:
        return ref_failure_result(
            suite_key=suite_key,
            label=f"{milestone} {suite_key}",
            ref=ref,
            ref_dir=run_dir / "ref",
            cfg=cfg,
        )

    ref_missing = ref_missing_steps_result(
        suite_key=suite_key,
        label=f"{milestone} {suite_key}",
        baseline_by_step=baseline_by_step,
        gate_steps=gate_steps,
        cfg=cfg,
    )
    if ref_missing is not None:
        return ref_missing

    # ── Ours phase ──
    ours = resolve_ours_phase(run_dir)
    returncode = ours.returncode
    output = ours.output
    if ours.timed_out:
        run_timeout = ours.timeout_s
        return {
            "status": "failed",
            "suite": suite_key,
            "summary": f"{milestone} {suite_key}: timed out after {run_timeout}s",
            "metrics": {"world_size": world_size},
            "details": {"config": cfg, "output_tail": tail(output)},
        }

    script_succeeded = returncode == 0
    loss_steps = ours.loss_steps
    if not loss_steps:
        return {
            "status": "failed",
            "suite": suite_key,
            "summary": f"{milestone} {suite_key}: no [LOSS] lines parsed. script returncode={returncode}.",
            "metrics": {"world_size": world_size},
            "details": {"config": cfg, "output_tail": tail(output)},
        }

    ours_by_step = ours.ours_by_step
    missing_steps = missing_window_steps(ours_by_step, gate_steps)
    if missing_steps:
        return {
            "status": "failed",
            "suite": suite_key,
            "summary": f"{milestone} {suite_key}: engine emitted no [LOSS] for {len(missing_steps)} steps. First: {missing_steps[:10]}.",
            "metrics": {"world_size": world_size, "steps_observed": len(loss_steps)},
            "details": {"config": cfg, "output_tail": tail(output)},
        }

    # Apply relative-loss gate
    ours_loss_by_step = {step: float(entry["global_loss"]) for step, entry in ours_by_step.items()}
    try:
        loss_metrics = window_loss_diff_metrics(
            baseline_by_step,
            ours_loss_by_step,
            gate_steps,
        )
    except ValueError as exc:
        return {
            "status": "failed",
            "suite": suite_key,
            "summary": f"{milestone} {suite_key}: {exc}.",
            "metrics": {"world_size": world_size},
            "details": {"config": cfg},
        }

    avg_rel_diff = float(loss_metrics["mean_rel_diff"])
    max_rel_diff = float(loss_metrics["max_rel_diff"])
    loss_pass = avg_rel_diff < max_avg_rel

    warmup_steps = int(cfg.get("warmup_steps", 0))
    # Wire format: ``mfu_e2e_standard`` is already in 0–100 percent scale;
    # this gate's summary reports it as-is (no pass/fail gate, diagnostic only).
    avg_mfu, _mfu_samples = filtered_avg(loss_steps, warmup_steps=warmup_steps)

    overall_pass = loss_pass and script_succeeded
    summary_parts = [
        f"avg_rel_diff={avg_rel_diff * 100:.3f}% {'<' if loss_pass else '≥'} {max_avg_rel * 100:.2f}%",
        f"max_rel_diff={max_rel_diff * 100:.3f}%",
        f"signed_rel={loss_metrics['signed_mean_rel'] * 100:+.4f}% (signed_mean={loss_metrics['signed_mean']:+.4e})",
        f"gate=[{gate_start},{gate_end})",
        f"avg_mfu={avg_mfu:.1f}%",
    ]
    if loss_metrics.get("drift_warning"):
        summary_parts.append(f"⚠ DRIFT: {loss_metrics['drift_warning']}")
    if not script_succeeded:
        summary_parts.append(f"script returncode={returncode}")

    return {
        "status": "passed" if overall_pass else "failed",
        "suite": suite_key,
        "summary": f"{milestone} {suite_key} (DP={world_size}, {num_steps} steps vs ref-script {ref.elapsed_s:.0f}s): {'; '.join(summary_parts)}.",
        "metrics": {
            "world_size": world_size,
            "num_steps": num_steps,
            "steps_observed": len(loss_steps),
            "ref_steps_observed": len(baseline_by_step),
            "ref_elapsed_s": ref.elapsed_s,
            "gate_window": [gate_start, gate_end],
            "avg_relative_loss_diff": avg_rel_diff,
            "max_relative_loss_diff": max_rel_diff,
            "signed_mean": loss_metrics["signed_mean"],
            "signed_mean_rel": loss_metrics["signed_mean_rel"],
            "compared_steps": loss_metrics["compared_steps"],
            "gate_threshold": max_avg_rel,
            "loss_pass": loss_pass,
            "avg_mfu_e2e_standard": avg_mfu,
            "passed": overall_pass,
            "buckets": loss_metrics.get("buckets", []),
            "drift_warning": loss_metrics.get("drift_warning"),
        },
        "details": {
            "config": cfg,
            "ref_dump_dir": str(run_dir / "ref" / "dump"),
            "step_diffs_tail": loss_metrics["step_diffs"][-20:],
            "trajectory": loss_steps,
            "output_tail": tail(output),
        },
    }


def main(argv: list[str] | None = None) -> int:
    return verdict_cli(run, argv, doc=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
