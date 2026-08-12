from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evals._common import (
    classify_ref_failure,
    missing_window_steps,
    output_tail_limit,
    run_via_ref_script,
    write_ref_capture_status,
)
from harness import run_schema


@dataclass(frozen=True, slots=True)
class RefTrajectory:
    ref_result: dict[str, Any] | None
    ref_run: Any | None
    loss_by_step: dict[int, float]
    grad_norm_by_step: dict[int, float] = field(default_factory=dict)
    gate_steps: range | None = None
    failure: dict[str, Any] | None = None


def resolve_ref_trajectory(
    *,
    repo_root: Path,
    suite_key: str,
    result_suite: str,
    summary_prefix: str,
    cfg: dict[str, Any],
    artifact_dir: Path,
    gate_steps: range | None = None,
    metrics: dict[str, Any] | None = None,
    hash_capture_level: int = 0,
    hash_output: Path | None = None,
    persistent: bool = False,
) -> RefTrajectory:
    """Run the live ref script and validate the requested comparison window.

    ``hash_capture_level`` / ``hash_output`` / ``persistent`` are
    typed passthroughs into :func:`run_via_ref_script`; when
    ``hash_capture_level > 0`` the bridge is used instead of the
    bare L0 ref script so the standard hook gets interposed and a
    persistent hash dump lands at ``hash_output``.
    """
    base_metrics = dict(metrics or {})
    try:
        ref_result = run_via_ref_script(
            repo_root=repo_root,
            suite_key=suite_key,
            cfg=cfg,
            artifact_dir=artifact_dir,
            hash_capture_level=hash_capture_level,
            hash_output=hash_output,
            persistent=persistent,
        )
    except Exception as exc:
        return RefTrajectory(
            ref_result=None,
            ref_run=None,
            loss_by_step={},
            failure=run_schema.make_failed_result(
                suite=result_suite,
                summary=f"{summary_prefix}: ref script invocation failed: {exc}",
                metrics=base_metrics,
                details={"config": cfg},
            ),
        )

    ref_run = ref_result["ref_run"]
    loss_by_step: dict[int, float] = ref_result["loss_by_step"]
    grad_norm_by_step: dict[int, float] = ref_result.get("grad_norm_by_step", {})
    # Structural ref-phase exit status for the meta harness_configs gate
    # (ref_side); written on both the pass and fail paths since ref_run exists.
    # dump_present is a REAL stat of the level-appropriate persistent hash dump
    # (hash_output) — not ref_run.succeeded, which was a misnomer that stated no
    # file. level 0 gates write no dump, so dump_present is trivially false there
    # and the meta gate defers to the trajectory check.
    _dump = Path(hash_output) if (hash_output and hash_capture_level > 0) else None
    write_ref_capture_status(
        artifact_dir,
        ref_run,
        dump_present=bool(_dump and _dump.exists()),
        hash_capture_level=hash_capture_level,
        graph_present=bool(_dump and _dump.with_name(_dump.name + ".graph.json").exists()),
        loss_by_step=loss_by_step,
    )
    if not ref_run.succeeded:
        stdout = ""
        if ref_run.stdout_path.exists():
            stdout = ref_run.stdout_path.read_text(errors="replace")[-output_tail_limit() :]
        # Classification (#21): mark the failure bucket so agent-loop /
        # subagents can pick the right remediation strategy (retry on
        # ``network``; ask for data-prep on ``data_missing``; surface to
        # human on ``script_setup``; etc.).
        failure_class = classify_ref_failure(
            timed_out=ref_run.timed_out,
            returncode=ref_run.returncode,
            stdout_text=stdout,
        )
        return RefTrajectory(
            ref_result=ref_result,
            ref_run=ref_run,
            loss_by_step=loss_by_step,
            failure=run_schema.make_failed_result(
                suite=result_suite,
                summary=(
                    f"{summary_prefix}: ref script {failure_class} — "
                    f"{'timed out' if ref_run.timed_out else f'returncode={ref_run.returncode}'}"
                    f" after {ref_run.elapsed_s:.0f}s."
                ),
                metrics={**base_metrics, "ref_failure_class": failure_class},
                details={
                    "config": cfg,
                    "ref_stdout": stdout,
                    "ref_failure_class": failure_class,
                },
            ),
        )

    resolved_gate_steps = gate_steps
    if resolved_gate_steps is None:
        metadata = getattr(ref_run, "metadata", {}) or {}
        try:
            start = int(metadata["gate_window_start"])
            end = int(metadata["gate_window_end"])
        except (KeyError, TypeError, ValueError) as exc:
            return RefTrajectory(
                ref_result=ref_result,
                ref_run=ref_run,
                loss_by_step=loss_by_step,
                failure=run_schema.make_failed_result(
                    suite=result_suite,
                    summary=f"{summary_prefix}: ref script did not emit gate window metadata: {exc}",
                    metrics=base_metrics,
                    details={"config": cfg, "metadata": metadata},
                ),
            )
        resolved_gate_steps = range(start, end)

    missing = missing_window_steps(loss_by_step, resolved_gate_steps)
    if missing:
        metrics_with_steps = {**base_metrics, "ref_steps_observed": len(loss_by_step)}
        return RefTrajectory(
            ref_result=ref_result,
            ref_run=ref_run,
            loss_by_step=loss_by_step,
            gate_steps=resolved_gate_steps,
            failure=run_schema.make_failed_result(
                suite=result_suite,
                summary=(
                    f"{summary_prefix}: ref-script trajectory missing steps in window "
                    f"[{resolved_gate_steps.start}, {resolved_gate_steps.stop}). First missing: {missing[:10]}"
                ),
                metrics=metrics_with_steps,
                details={"config": cfg},
            ),
        )

    return RefTrajectory(
        ref_result=ref_result,
        ref_run=ref_run,
        loss_by_step=loss_by_step,
        grad_norm_by_step=grad_norm_by_step,
        gate_steps=resolved_gate_steps,
    )
