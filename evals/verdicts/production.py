"""Production-train verdict (production-train).

File-based port of the legacy ``dispatcher._run_production_train`` judgment
(identical semantics and summary strings). production is ours-only — there
is never a ref phase, no loss/MFU gate, and the verdict is checkpoint-only:
every expected ``step_<N>`` dir under the STABLE checkpoint root must hold a
non-empty ``*.pt``. Reads:

* ``<run_dir>/checkpoint_root.txt``          — stable root the runner used
  (fallback: the ``<.artifacts>/production_train/<suite>/checkpoints``
  convention formula, identical to the runner's default)
* ``<run_dir>/ours/segments.jsonl``          — one record per attempted
  segment: start/end/returncode/timed_out/timeout_s/elapsed_s
* ``<run_dir>/ours/resumed_from.txt``        — abs step the runner resumed at
* ``<run_dir>/ours/training_output_step_<N>.log`` — per-segment stdout
* ``<run_dir>/ours/status.json``             — whole-runner exit
* the rendered ours product                  — shape (world_size / num_steps /
  MBS / GBS) and segmentation (production_save_segments)

Checkpoint completeness shares its SSOT with the segment loop:
``evals/scripts/production_ckpt.py`` applies the same non-empty-``*.pt``
rule when it gates each segment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.gate_product import GateProductError, load_gate_product
from evals.verdicts._shared import (
    overlay_product_verdict,
    read_side_status,
    read_text,
    tail,
    verdict_cli,
)
from harness import run_schema

# Legacy dispatcher constant _PRODUCTION_SAVE_SEGMENTS — the default when
# neither the product nor the registry carries production_save_segments.
DEFAULT_SAVE_SEGMENTS = 4


def _checkpoint_dirname(abs_step: int) -> str:
    """SSOT spelling of the per-checkpoint directory name (``step_<N>``)."""
    return f"step_{abs_step}"


def _checkpoint_complete(step_dir: Path) -> bool:
    """Complete iff the dir holds a non-empty ``*.pt`` (mirrors production_ckpt.py)."""
    return step_dir.is_dir() and any(p.stat().st_size > 0 for p in step_dir.glob("*.pt"))


def _production_shape(
    cfg: dict[str, Any],
    repo_root: Path,
    suite_key: str,
    segments: int,
) -> tuple[int, int, int, int]:
    """Resolve (world_size, num_steps, micro_batch_size, grad_accum).

    Verbatim port of the legacy ``dispatcher._production_shape``: an inline
    ``cfg.shape`` table (flat / tests) wins; otherwise the rendered ours
    product. ``grad_accum`` derives as ``GBS / (MBS * WS)``.
    """
    shape = cfg.get("shape")
    if isinstance(shape, dict):
        try:
            world_size = int(shape["world_size"])
            num_steps = int(shape["num_steps"])
            micro_batch_size = int(shape["micro_batch_size"])
            global_batch_size = int(shape["global_batch_size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{suite_key} shape is malformed: {exc}") from exc
    else:
        try:
            prod = load_gate_product(repo_root, "ours", suite_key)
            world_size = int(prod.get("world_size"))
            num_steps = int(prod.get("num_steps"))
            micro_batch_size = int(prod.get("micro_batch_size"))
            global_batch_size = int(prod.get("global_batch_size"))
        except (GateProductError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{suite_key}: no inline shape and no rendered ours product: {exc}"
            ) from exc
    if micro_batch_size * world_size == 0:
        raise ValueError(f"{suite_key} MBS * WORLD_SIZE must be > 0")
    grad_accum = global_batch_size // (micro_batch_size * world_size)
    if grad_accum < 1:
        raise ValueError(
            f"{suite_key} GBS {global_batch_size} / (MBS {micro_batch_size} * "
            f"WS {world_size}) = {grad_accum} < 1"
        )
    if num_steps % segments != 0:
        raise ValueError(
            f"{suite_key} num_steps={num_steps} must be divisible by "
            f"{segments} so each segment ends on a checkpoint"
        )
    return world_size, num_steps, micro_batch_size, grad_accum


def _resolve_segments(cfg: dict[str, Any], repo_root: Path, suite_key: str) -> int:
    """Segment count: product key > registry key > legacy constant 4."""
    try:
        prod = load_gate_product(repo_root, "ours", suite_key)
        val = prod.get("production_save_segments")
        if val is not None:
            return int(val)
    except (GateProductError, TypeError, ValueError):
        pass
    try:
        return int(cfg.get("production_save_segments", DEFAULT_SAVE_SEGMENTS))
    except (TypeError, ValueError):
        return DEFAULT_SAVE_SEGMENTS


def _read_segment_records(ours_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    path = ours_dir / "segments.jsonl"
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            records.append(rec)
    return records


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
    ours_dir = run_dir / "ours"

    segments = _resolve_segments(cfg, repo_root, suite_key)
    try:
        world_size, num_steps, _mbs, _grad_accum = _production_shape(
            cfg, repo_root, suite_key, segments
        )
    except (KeyError, TypeError, ValueError, GateProductError) as exc:
        return run_schema.make_failed_result(
            suite=suite_key,
            summary=f"{summary_label}: shape config invalid: {exc}",
            metrics={},
            details={"config": cfg},
        )
    seg_steps = num_steps // segments
    expected_steps = [seg_steps * (k + 1) for k in range(segments)]

    # Stable checkpoint root: prefer the runner's own record; the formula
    # fallback is byte-identical to the runner's default.
    root_txt = run_dir / "checkpoint_root.txt"
    if root_txt.exists():
        save_root = Path(root_txt.read_text(encoding="utf-8").strip())
    else:
        save_root = run_dir.parent.parent / "production_train" / suite_key / "checkpoints"

    resumed_txt = ours_dir / "resumed_from.txt"
    try:
        resumed_from = int(resumed_txt.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        resumed_from = 0

    status = read_side_status(ours_dir)
    records = _read_segment_records(ours_dir)

    def _seg_tail(abs_end: int) -> str:
        return tail(read_text(ours_dir / f"training_output_step_{abs_end}.log"))

    # Per-segment failure judgment, in segment order (mirrors the legacy loop).
    for rec in records:
        start_step = int(rec.get("start", 0))
        abs_end = int(rec.get("end", 0))
        returncode = int(rec.get("returncode", -1))
        if rec.get("timed_out"):
            seg_timeout = rec.get("timeout_s")
            return run_schema.make_failed_result(
                suite=suite_key,
                summary=f"{summary_label}: segment [{start_step},{abs_end}) "
                f"timed out after {seg_timeout}s",
                metrics={"world_size": world_size, "num_steps": num_steps},
                details={"config": cfg, "output_tail": _seg_tail(abs_end)},
            )
        step_dir = save_root / _checkpoint_dirname(abs_end)
        if returncode != 0 or not _checkpoint_complete(step_dir):
            return run_schema.make_failed_result(
                suite=suite_key,
                summary=f"{summary_label}: segment [{start_step},{abs_end}) failed "
                f"(returncode={returncode}, checkpoint "
                f"{'present' if _checkpoint_complete(step_dir) else 'MISSING'}).",
                metrics={
                    "world_size": world_size,
                    "num_steps": num_steps,
                    "save_interval": seg_steps,
                    "resumed_from_step": resumed_from,
                },
                details={"config": cfg, "output_tail": _seg_tail(abs_end)},
            )

    # Whole-runner timeout (executor SIGKILL): the segment that was in
    # flight never wrote its record — report the first unaccounted segment.
    if status["timed_out"]:
        done_ends = {int(r.get("end", 0)) for r in records}
        pending = [s for s in expected_steps if s > resumed_from and s not in done_ends]
        if pending:
            abs_end = pending[0]
            return run_schema.make_failed_result(
                suite=suite_key,
                summary=f"{summary_label}: segment [{abs_end - seg_steps},{abs_end}) "
                f"timed out after {status['timeout_s']}s",
                metrics={"world_size": world_size, "num_steps": num_steps},
                details={"config": cfg, "output_tail": _seg_tail(abs_end)},
            )

    # Runner died outside the segment loop (product missing, prefetch error,
    # segmentation guard, …) — no legacy twin (the old dispatcher raised);
    # fail with the runner log so the cause is visible.
    if int(status["returncode"]) != 0 and not records:
        return run_schema.make_failed_result(
            suite=suite_key,
            summary=f"{summary_label}: ours runner failed before any segment "
            f"(returncode={status['returncode']}).",
            metrics={"world_size": world_size, "num_steps": num_steps},
            details={
                "config": cfg,
                "output_tail": tail(read_text(ours_dir / "ours.log")),
            },
        )

    # Final checkpoint-only verdict: every expected step_<N> must be a
    # complete checkpoint on disk.
    present = [_checkpoint_complete(save_root / _checkpoint_dirname(s)) for s in expected_steps]
    saved_checkpoints = sum(present)
    passed = saved_checkpoints == segments
    last_output = _seg_tail(int(records[-1]["end"])) if records else ""
    return run_schema.make_run_result(
        status="passed" if passed else "failed",
        suite=suite_key,
        summary=f"{summary_label} (DP={world_size}, {num_steps} steps, save every "
        f"{seg_steps}): {saved_checkpoints}/{segments} checkpoints "
        f"saved at steps {expected_steps}"
        + (f"; resumed from step {resumed_from}" if resumed_from else "")
        + ".",
        metrics={
            "world_size": world_size,
            "num_steps": num_steps,
            "save_interval": seg_steps,
            "saved_checkpoints": saved_checkpoints,
            "resumed_from_step": resumed_from,
            "passed": passed,
        },
        details={
            "config": cfg,
            "checkpoint_dirs": [str(save_root / _checkpoint_dirname(s)) for s in expected_steps],
            "output_tail": last_output,
        },
    )


def main(argv: list[str] | None = None) -> int:
    return verdict_cli(run, argv, doc=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
