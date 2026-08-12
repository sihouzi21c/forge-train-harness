"""Resume verdict (resume-gate-20).

File-based port of the legacy ``dispatcher._run_resume_gate`` judgment
(identical semantics and summary strings). Ours-only: the runner
(``eval_resume_train.py``) produces the uninterrupted [LOSS_REF] and the
save→resume [LOSS_RES] trajectories in one process; there is no ref
subprocess. Reads:

* ``<run_dir>/ours/status.json`` + ``<run_dir>/ours/ours.log``
  — ours phase exit + both trajectories
* ``<run_dir>/ours/ours_hash_dump.json.ref.json`` / ``.res.json``
  — per-FQN hash dumps (hash_capture_level > 0 only; the engine appends
  ``.ref.json`` / ``.res.json`` to the ``--hash-output`` base the generic
  runner passes)
* the rendered ref product — gate shape (num_steps / gate window /
  resume_save_step / world_size)

Gates: per-step ``|loss_ref − loss_res| == 0`` and
``|grad_norm_ref − grad_norm_res| == 0`` over the gate window, plus the
hash diff (when captured), plus ours returncode == 0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evals._common import missing_window_steps
from evals.gate_product import GateProductError
from evals.gate_shape import load_gate_shape
from evals.verdicts._kernels import (
    capture_entry_details,
    compare_series,
    hash_dump_gate,
)
from evals.verdicts._shared import (
    overlay_product_verdict,
    read_side_status,
    read_text,
    tail,
    verdict_cli,
)
from harness import run_schema
from harness.wire_format import parse_loss_lines

if TYPE_CHECKING:
    from pathlib import Path

# The generic ours runner passes ``--hash-output <run_dir>/ours_hash_dump.json``;
# the resume engine convention appends the phase suffix to that base.
OURS_HASH_BASENAME = "ours_hash_dump.json"


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

    # ── Shape: the rendered ref product is the unconditional source ──
    try:
        shape = load_gate_shape(repo_root, suite_key)
    except GateProductError:
        return run_schema.make_failed_result(
            suite=suite_key,
            summary=f"{milestone} {suite_key}: missing rendered ref product for gate shape",
            metrics={},
            details={"config": cfg},
        )
    try:
        num_steps = shape.metadata_int("num_steps")
        gate_start = shape.metadata_int("gate_window_start")
        gate_end = shape.metadata_int("gate_window_end")
        save_step = shape.metadata_int("resume_save_step")
        world_size = shape.metadata_int("world_size")
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        return run_schema.make_failed_result(
            suite=suite_key,
            summary=f"{milestone} {suite_key}: ref script gate metadata failed: {exc}",
            metrics={},
            details={"config": cfg},
        )

    hash_capture_level = int(cfg.get("hash_capture_level", 0))
    ours_dir = run_dir / "ours"
    hash_base = ours_dir / OURS_HASH_BASENAME
    hash_ref_dump = hash_base.with_name(hash_base.name + ".ref.json")
    hash_res_dump = hash_base.with_name(hash_base.name + ".res.json")

    # ── Ours phase: status + stdout from on-disk artifacts ──
    status = read_side_status(ours_dir)
    output = read_text(ours_dir / "ours.log")
    returncode = int(status["returncode"])
    if bool(status["timed_out"]):
        run_timeout = status.get("timeout_s")
        return {
            "status": "failed",
            "suite": suite_key,
            "summary": f"{milestone} {suite_key}: timed out after {run_timeout}s",
            "metrics": {"world_size": world_size},
            "details": {"config": cfg, "output_tail": tail(output)},
        }

    script_succeeded = returncode == 0
    ref_steps = parse_loss_lines(output, tag="LOSS_REF")
    res_steps = parse_loss_lines(output, tag="LOSS_RES")
    if not ref_steps or not res_steps:
        return {
            "status": "failed",
            "suite": suite_key,
            "summary": f"{milestone} {suite_key}: missing trajectories — [LOSS_REF]={len(ref_steps)} [LOSS_RES]={len(res_steps)} lines; script returncode={returncode}.",
            "metrics": {"world_size": world_size},
            "details": {"config": cfg, "output_tail": tail(output)},
        }

    ref_by_step = {s["step"]: s for s in ref_steps}
    res_by_step = {s["step"]: s for s in res_steps}

    missing_steps = sorted(
        set(missing_window_steps(ref_by_step, range(gate_start, gate_end)))
        | set(missing_window_steps(res_by_step, range(gate_start, gate_end)))
    )
    if missing_steps:
        return {
            "status": "failed",
            "suite": suite_key,
            "summary": f"{milestone} {suite_key}: gate window [{gate_start},{gate_end}) missing {len(missing_steps)} step(s). First: {missing_steps[:10]}.",
            "metrics": {
                "world_size": world_size,
                "ref_steps_observed": len(ref_steps),
                "res_steps_observed": len(res_steps),
            },
            "details": {"config": cfg, "output_tail": tail(output)},
        }

    # Bitwise gate
    window = range(gate_start, gate_end)
    loss_records = compare_series(
        {s: e["global_loss"] for s, e in ref_by_step.items()},
        {s: e["global_loss"] for s, e in res_by_step.items()},
        window,
    )
    grad_records = compare_series(
        {s: e["grad_norm"] for s, e in ref_by_step.items()},
        {s: e["grad_norm"] for s, e in res_by_step.items()},
        window,
    )
    step_diffs: list[dict[str, Any]] = [
        {
            "step": rl.step,
            "ref_loss": rl.baseline,
            "res_loss": rl.ours,
            "ref_grad_norm": rg.baseline,
            "res_grad_norm": rg.ours,
            "loss_abs_diff": rl.abs_diff,
            "grad_norm_abs_diff": rg.abs_diff,
        }
        for rl, rg in zip(loss_records, grad_records, strict=False)
    ]

    max_loss_diff = max((r.abs_diff for r in loss_records), default=0.0)
    max_grad_diff = max((r.abs_diff for r in grad_records), default=0.0)
    bitwise_pass = max_loss_diff == 0.0 and max_grad_diff == 0.0

    # ── Phase 4: per-FQN per-step hash diff (when hash_capture_level > 0) ──
    # resume ours emits two dumps (un-resumed = .ref.json,
    # resumed = .res.json); we compare them against each other.
    hash_entries: list[Any] = []
    hash_rendered: list[str] = []
    hash_pass = True
    hash_summary: str | None = None
    if hash_capture_level > 0:
        hash_gate = hash_dump_gate(
            hash_res_dump, hash_ref_dump, baseline_label="ref", candidate_label="res"
        )
        hash_entries = hash_gate.entries
        hash_rendered = hash_gate.rendered
        hash_pass = hash_gate.passed
        hash_summary = hash_gate.summary

    overall_pass = bitwise_pass and hash_pass and script_succeeded

    summary_parts = [
        f"max_abs_diff(loss)={max_loss_diff:.3e}",
        f"max_abs_diff(grad_norm)={max_grad_diff:.3e}",
        f"gate=[{gate_start},{gate_end})",
        f"({'==' if bitwise_pass else '!='} 0)",
    ]
    if hash_summary is not None:
        summary_parts.append(hash_summary)
    if not script_succeeded:
        summary_parts.append(f"script returncode={returncode}")

    first_violation: dict[str, Any] | None = None
    for d in step_diffs:
        if d["loss_abs_diff"] != 0.0 or d["grad_norm_abs_diff"] != 0.0:
            first_violation = d
            break

    return {
        "status": "passed" if overall_pass else "failed",
        "suite": suite_key,
        "summary": f"{milestone} {suite_key} (DP={world_size}, save@{save_step}, bitwise: ref vs resume): {'; '.join(summary_parts)}.",
        "metrics": {
            "world_size": world_size,
            "num_steps": num_steps,
            "ref_steps_observed": len(ref_steps),
            "res_steps_observed": len(res_steps),
            "gate_window": [gate_start, gate_end],
            "save_step": save_step,
            "max_abs_diff_loss": max_loss_diff,
            "max_abs_diff_grad_norm": max_grad_diff,
            "bitwise_pass": bitwise_pass,
            "passed": overall_pass,
            "hash_capture_level": hash_capture_level,
            "hash_pass": hash_pass if hash_capture_level > 0 else None,
            "hash_checks_total": len(hash_entries) if hash_capture_level > 0 else None,
            "hash_bitwise_match": (
                sum(1 for e in hash_entries if e.passed) if hash_capture_level > 0 else None
            ),
        },
        "details": {
            "config": cfg,
            "first_violation": first_violation,
            "step_diffs_tail": step_diffs[-20:],
            "ref_trajectory": ref_steps,
            "res_trajectory": res_steps,
            "output_tail": tail(output),
            "hash_checks": (capture_entry_details(hash_entries) if hash_capture_level > 0 else []),
            "hash_rendered": hash_rendered if hash_capture_level > 0 else [],
            "hash_ref_dump": str(hash_ref_dump) if hash_capture_level > 0 else None,
            "hash_res_dump": str(hash_res_dump) if hash_capture_level > 0 else None,
        },
    }


def main(argv: list[str] | None = None) -> int:
    return verdict_cli(run, argv, doc=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
