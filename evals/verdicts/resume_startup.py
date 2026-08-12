"""Resume-startup verdict (resume-startup-90).

File-based port of the legacy ``dispatcher._run_resume_startup`` judgment
(identical semantics and summary strings). Ours-only: no ref trajectory —
the runner (``eval_resume_startup.py``) resumes from ``resume_save_step``
on the engine's real streaming dataloader, times the seek window, and
emits ``[RESUME_STARTUP] seconds=<f>``. Reads:

* ``<run_dir>/ours/status.json`` + ``<run_dir>/ours/ours.log``
  — ours phase exit + the [RESUME_STARTUP] marker
* the rendered OURS product — world_size / resume_save_step /
  grad_accum_steps (consumed micro-batches for the verdict)
* the frozen registry cfg — ``resume_startup_budget_s`` (ours-only verdict
  threshold, deliberately NOT carried in the product)

The verdict (``evals.resume_startup_gate``) fails the replay-by-discard
loader (seek O(consumed)) and passes an O(1) cursor seek.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evals.gate_product import GateProductError, load_gate_product
from evals.resume_startup_gate import (
    evaluate_resume_startup,
    parse_resume_startup_seconds,
)
from evals.verdicts._shared import (
    overlay_product_verdict,
    read_side_status,
    read_text,
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

    # Shape source is the rendered ours product (ours-only gate; no ref
    # trajectory). The verdict needs only world_size (metric) and
    # save_step·grad_accum (consumed micro-batches); every other shape scalar
    # the runner reads from the product itself. The budget threshold is an
    # ours-only verdict knob and stays in the frozen registry.
    try:
        product = load_gate_product(repo_root, "ours", suite_key)
        world_size = int(product.get("world_size"))
        save_step = int(product.get("resume_save_step"))
        grad_accum = int(product.get("grad_accum_steps"))
    except (GateProductError, TypeError, ValueError) as exc:
        return run_schema.make_failed_result(
            suite=suite_key,
            summary=f"{milestone} {suite_key}: missing/invalid rendered ours product shape: {exc}",
            metrics={},
            details={"config": cfg},
        )
    budget_s = float(cfg.get("resume_startup_budget_s", 0))

    ours_dir = run_dir / "ours"
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

    # The gate verdict is parsed from the [RESUME_STARTUP] marker (the SIGABRT
    # some teardowns raise at interpreter exit makes the exit code unreliable —
    # same stdout-parsed contract the bitwise resume gate uses for LOSS lines).
    try:
        startup_s = parse_resume_startup_seconds(output)
    except ValueError as exc:
        return {
            "status": "failed",
            "suite": suite_key,
            "summary": (
                f"{milestone} {suite_key}: no [RESUME_STARTUP] marker ({exc}); "
                f"returncode={returncode}."
            ),
            "metrics": {"world_size": world_size},
            "details": {"config": cfg, "output_tail": tail(output)},
        }

    consumed = save_step * grad_accum
    verdict = evaluate_resume_startup(
        startup_s, budget_s, save_step=save_step, consumed_microbatches=consumed
    )
    return {
        "status": "passed" if verdict.passed else "failed",
        "suite": suite_key,
        "summary": f"{milestone} {suite_key}: {verdict.summary}",
        "metrics": {
            "resume_startup_s": startup_s,
            "budget_s": budget_s,
            "consumed_microbatches": consumed,
            "world_size": world_size,
        },
        "details": {"config": cfg, "output_tail": tail(output)},
    }


def main(argv: list[str] | None = None) -> int:
    return verdict_cli(run, argv, doc=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
