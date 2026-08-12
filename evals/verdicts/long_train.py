"""Long-train verdict (long-train / long-train-smoke).

File-based port of the legacy ``dispatcher._run_long_train`` judgment
(identical semantics and summary strings). Reads:

* ``<run_dir>/ref/status.json`` + ``<run_dir>/ref/ref.log``
  + ``<run_dir>/ref/dump/ref_loss.txt``   — ref phase exit + trajectory
* ``<run_dir>/ours/status.json`` + ``<run_dir>/ours/ours.log``
  — ours phase exit + [LOSS] trajectory
* the rendered ref product                — shape (gate window / steps /
  world_size) and verdict thresholds

Gates:
  1  pointwise mean relative loss diff over the gate window
     < loss_rel_threshold
  2  avg MFU(standard) ≥ mfu_e2e_target, ONLY when the suite declares a
     positive floor ("0 = off"); avg MFU is still measured and reported
     ungated — the long-horizon throughput bar is then enforced by the
     review agent instead (see prompt/review_prompt/review_stage1.md)
plus ours returncode == 0.
"""

from __future__ import annotations

import sys
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


def _elastic_mfu_pass(avg_mfu: float, mfu_target: float, repo_root: Path) -> bool:
    """Hidden elastic relaxation for a dev-visible ``mfu_e2e_target``.

    Called only on a strict miss (``avg_mfu < mfu_target``). Passes when
    the current run plus the prior long-train history form a stable
    plateau inside the tolerance band — policy read from the gate SOURCE
    toml (renderer-stripped keys; see tools/mfu_elastic_check). The knob
    values must never be echoed into the summary / result.json.

    Best-effort: any failure to load policy or history keeps the strict
    verdict (returns False) — the relaxation can only widen the gate,
    never break it.
    """
    try:
        from tools import mfu_elastic_check

        policy = mfu_elastic_check.load_policy()
        if policy is None:
            return False
        history_path = mfu_elastic_check.default_history_path(repo_root)
        if history_path is None:
            return False
        samples, _had_file = mfu_elastic_check.load_samples(history_path)
        # This run's own measurement is not yet in the history (the
        # telemetry record is appended by the runner after this result
        # is finalized) — judge on history + current.
        samples.append(float(avg_mfu))
        passed, reason = mfu_elastic_check.evaluate(samples, float(mfu_target), policy)
        return passed and reason == "ELASTIC_PASS"
    except Exception as exc:  # pragma: no cover - defensive
        print(f"long-train: elastic MFU check skipped: {exc}", file=sys.stderr)
        return False


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
    summary_label = f"{milestone} {suite_key}"

    loss_rel_threshold = float(cfg.get("loss_rel_threshold", 0.025))
    # MFU floor follows the "0 = off" convention shared with the bitwise
    # trajectory gates (see module docstring).
    mfu_target = float(cfg.get("mfu_e2e_target", 0))
    warmup_steps = int(cfg.get("warmup_steps", 50))

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
            summary=f"{summary_label}: gate shape unavailable from product: {exc}",
            metrics={},
            details={"config": cfg},
        )
    gate_lo, gate_hi = gate_steps.start, gate_steps.stop

    # ── Ref phase: status + trajectory from on-disk artifacts ──
    ref = resolve_ref_phase(run_dir)
    baseline_by_step = ref.baseline_by_step

    if not ref.succeeded:
        return ref_failure_result(
            suite_key=suite_key, label=summary_label, ref=ref, ref_dir=run_dir / "ref", cfg=cfg
        )

    ref_missing = ref_missing_steps_result(
        suite_key=suite_key,
        label=summary_label,
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
        long_timeout = ours.timeout_s
        return {
            "status": "failed",
            "suite": suite_key,
            "summary": f"{summary_label}: ours-side timed out after {long_timeout}s",
            "metrics": {"world_size": world_size},
            "details": {"config": cfg, "output_tail": tail(output)},
        }

    script_succeeded = returncode == 0
    loss_steps = ours.loss_steps
    if not loss_steps:
        return {
            "status": "failed",
            "suite": suite_key,
            "summary": f"{summary_label}: no [LOSS] lines parsed. script returncode={returncode}.",
            "metrics": {"world_size": world_size},
            "details": {"config": cfg, "output_tail": tail(output)},
        }

    ours_by_step = ours.ours_by_step
    missing_steps = missing_window_steps(ours_by_step, gate_steps)
    if missing_steps:
        return {
            "status": "failed",
            "suite": suite_key,
            "summary": f"{summary_label}: ours emitted no [LOSS] for {len(missing_steps)} steps. First: {missing_steps[:10]}.",
            "metrics": {"world_size": world_size, "steps_observed": len(loss_steps)},
            "details": {"config": cfg, "output_tail": tail(output)},
        }

    # Gate 1: pointwise relative loss
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
            "summary": f"{summary_label}: {exc}.",
            "metrics": {"world_size": world_size},
            "details": {"config": cfg},
        }

    pointwise_mean_rel = float(loss_metrics["mean_rel_diff"])
    max_rel_diff = float(loss_metrics["max_rel_diff"])
    loss_pass = pointwise_mean_rel < loss_rel_threshold

    # Gate 2: MFU (only when the suite declares a floor — see above)
    #
    # When a floor IS set, the verdict additionally honors the hidden
    # elastic relaxation (stable plateau just under the bar) whose knobs
    # live in the gate SOURCE toml only — renderer-stripped, never in this
    # suite's cfg / rendered product / result.json (see
    # tools/mfu_elastic_check + tools/render_gate_configs REVIEW_ONLY_KEYS).
    #
    # ``mfu_e2e_standard`` on the wire is already in 0–100 percent scale,
    # same as ``mfu_e2e_target``.  No re-scaling.
    mfu_gated = mfu_target > 0
    avg_mfu, mfu_samples = filtered_avg(loss_steps, warmup_steps=warmup_steps)
    if mfu_samples:
        mfu_pass = avg_mfu >= mfu_target if mfu_gated else True
        if mfu_gated and not mfu_pass:
            # Hidden elastic relaxation: a stable plateau just under the
            # bar passes. Knobs come from the gate SOURCE toml only; the
            # summary/metrics never disclose the band or the reason.
            mfu_pass = _elastic_mfu_pass(avg_mfu, mfu_target, repo_root)
    else:
        mfu_pass = not mfu_gated

    overall_pass = loss_pass and mfu_pass and script_succeeded
    if mfu_gated:
        if mfu_pass and avg_mfu < mfu_target:
            # Elastic pass: no comparison line — printing ``≥ target``
            # would be false, and printing the band would disclose it.
            mfu_summary = f"MFU {avg_mfu:.1f}% (PASS)"
        else:
            mfu_summary = f"MFU {avg_mfu:.1f}% {'≥' if mfu_pass else '<'} {mfu_target:.1f}%"
    else:
        mfu_summary = f"MFU {avg_mfu:.1f}%"
    summary_parts = [
        f"loss_rel(point) {pointwise_mean_rel * 100:.3f}% {'<' if loss_pass else '≥'} {loss_rel_threshold * 100:.2f}%",
        f"signed_rel={loss_metrics['signed_mean_rel'] * 100:+.4f}% (signed_mean={loss_metrics['signed_mean']:+.4e})",
        mfu_summary,
    ]
    if loss_metrics.get("drift_warning"):
        summary_parts.append(f"⚠ DRIFT: {loss_metrics['drift_warning']}")
    if not script_succeeded:
        summary_parts.append(f"ours returncode={returncode}")

    return {
        "status": "passed" if overall_pass else "failed",
        "suite": suite_key,
        "summary": f"{summary_label} (DP={world_size}, {num_steps} steps vs ref-script {ref.elapsed_s:.0f}s): {'; '.join(summary_parts)}.",
        "metrics": {
            "world_size": world_size,
            "num_steps": num_steps,
            "steps_observed": len(loss_steps),
            "ref_steps_observed": len(baseline_by_step),
            "ref_elapsed_s": ref.elapsed_s,
            "gate_window": [gate_lo, gate_hi],
            "pointwise_mean_rel": pointwise_mean_rel,
            "max_rel_diff": max_rel_diff,
            "signed_mean": loss_metrics["signed_mean"],
            "signed_mean_rel": loss_metrics["signed_mean_rel"],
            "compared_steps": loss_metrics["compared_steps"],
            "loss_rel_threshold": loss_rel_threshold,
            "loss_pass": loss_pass,
            "avg_mfu_e2e_standard": avg_mfu,
            "mfu_target": mfu_target if mfu_gated else None,
            "mfu_pass": mfu_pass if mfu_gated else None,
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
