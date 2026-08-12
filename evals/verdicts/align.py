"""Single-step tensor-capture alignment verdict (forward-align / backward-align).

File-based port of the legacy ``dispatcher._run_align_capture_diff`` judgment
(alignment.forward / alignment.backward, identical semantics). Reads:

* ``<run_dir>/ref/status.json`` + ``<run_dir>/ref/ref.log``   — bridge exit
* ``<run_dir>/ref/ref_hash_dump.json`` (+ ``.graph.json``)    — ref capture
* ``<run_dir>/ours/status.json`` + ``<run_dir>/ours/ours.log`` — engine exit
* ``<run_dir>/ours/ours_hash_dump.json`` (+ ``.graph.json``)  — ours capture
* the rendered ref product — world_size shape + verdict thresholds

``forward-align`` diffs ``fwd.*`` keys (intermediate activations);
``backward-align`` diffs ``grad.*`` (param gradients) AND ``bwd.*``
(per-module activation gradients), ref-authoritative (a ``bwd.*`` the ref
captured but the candidate omitted FAILS). The two gates are told apart by
the VALUE of ``milestone`` — no gate-name registry in this module.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from evals._common import write_ref_capture_status
from evals.gate_product import GateProductError
from evals.gate_shape import load_gate_shape
from evals.verdicts._kernels import capture_entry_details
from evals.verdicts._shared import (
    overlay_product_verdict,
    read_side_status,
    read_text,
    resolve_ours_phase,
    tail,
    verdict_cli,
)
from harness import run_schema
from harness.capture_artifacts import (
    candidate_capture_paths,
    load_merged_capture,
    ref_capture_paths,
)

if TYPE_CHECKING:
    from pathlib import Path

REF_DUMP_BASENAME = "ref_hash_dump.json"
OURS_DUMP_BASENAME = "ours_hash_dump.json"


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
    default_milestone = "alignment.backward" if "backward" in suite_key else "alignment.forward"
    milestone = str(cfg.get("milestone", default_milestone))
    summary_prefix = f"{milestone} {suite_key}"
    hash_capture_level = int(cfg.get("hash_capture_level", 2))
    # Comparison selector, keyed on the milestone VALUE (legacy: per runner
    # kind). backward gates BOTH grad.* and bwd.* and is ref-authoritative.
    if milestone == "alignment.backward":
        key_prefix: str | tuple[str, ...] = ("grad.", "bwd.")
        require_baseline_complete = True
    else:
        key_prefix = "fwd."
        require_baseline_complete = False

    failure_metrics = {
        "checks_total": 0,
        "bitwise_match": 0,
        "hash_capture_level": hash_capture_level,
    }

    ref_dir = run_dir / "ref"
    ours_dir = run_dir / "ours"

    # ── Phase 1: reference capture (bridge exit + on-disk dump) ──
    ref_status = read_side_status(ref_dir)
    ref_artifacts = ref_capture_paths(ref_dir, REF_DUMP_BASENAME)
    write_ref_capture_status(
        run_dir,
        SimpleNamespace(
            returncode=int(ref_status["returncode"]),
            timed_out=bool(ref_status["timed_out"]),
        ),
        dump_present=ref_artifacts.tensor_dump_exists,
        hash_capture_level=hash_capture_level,
        graph_present=ref_artifacts.graph_dump is not None,
    )
    if not ref_artifacts.tensor_dump_exists or int(ref_status["returncode"]) != 0:
        # Clean-exit contract: the ref capture PASSes its precondition iff it
        # produced the tensor dump AND returned 0 (a dirty exit — e.g. the
        # NCCL/flash_attn destructor-order SIGABRT — must not be diffed
        # against a half-produced baseline).
        stdout = tail(read_text(ref_dir / "ref.log"))
        if ref_status["timed_out"]:
            why = "timed out"
        elif int(ref_status["returncode"]) != 0:
            why = f"returncode={ref_status['returncode']}"
        else:
            why = "dump missing"
        return run_schema.make_failed_result(
            suite=suite_key,
            summary=(
                f"{summary_prefix}: ref-script capture failed — {why}"
                f" after {float(ref_status.get('elapsed_s') or 0.0):.0f}s; dump at "
                f"{ref_artifacts.tensor_dump}"
                f"{' (missing)' if not ref_artifacts.tensor_dump_exists else ''}."
            ),
            metrics=failure_metrics,
            details={"config": cfg, "ref_stdout": stdout},
        )

    # Shape precondition: the rendered ref product is the unconditional source.
    try:
        shape = load_gate_shape(repo_root, suite_key)
    except GateProductError:
        return run_schema.make_failed_result(
            suite=suite_key,
            summary=f"{summary_prefix}: missing rendered ref product for gate shape",
            metrics=failure_metrics,
            details={"config": cfg},
        )
    try:
        shape.metadata_int("world_size")
    except (KeyError, TypeError, ValueError) as exc:
        return run_schema.make_failed_result(
            suite=suite_key,
            summary=f"{summary_prefix}: ref product missing gate shape: {exc}",
            metrics=failure_metrics,
            details={"config": cfg},
        )

    # ── Phase 2: candidate capture (engine exit + on-disk dump) ──
    ours_phase = resolve_ours_phase(run_dir)
    returncode = ours_phase.returncode
    output = ours_phase.output
    candidate_artifacts = candidate_capture_paths(ours_dir, OURS_DUMP_BASENAME)
    if ours_phase.timed_out:
        return run_schema.make_failed_result(
            suite=suite_key,
            summary=f"{summary_prefix}: ours-side timed out after {ours_phase.timeout_s}s",
            metrics=failure_metrics,
            details={"config": cfg, "output_tail": tail(output)},
        )
    if not candidate_artifacts.tensor_dump_exists:
        return run_schema.make_failed_result(
            suite=suite_key,
            summary=(
                f"{summary_prefix}: ours-side returncode={returncode} "
                f"but did not produce {candidate_artifacts.tensor_dump}."
            ),
            metrics=failure_metrics,
            details={"config": cfg, "output_tail": tail(output)},
        )

    # ── Phase 3: diff intersected keys (the actual gate verdict) ──
    from evals._capture_diff import capture_gate_outcome, format_entry

    baseline = load_merged_capture(ref_artifacts.tensor_dump)
    candidate = load_merged_capture(candidate_artifacts.tensor_dump)

    outcome = capture_gate_outcome(
        candidate,
        baseline,
        key_prefix=key_prefix,
        require_baseline_complete=require_baseline_complete,
        returncode=returncode,
    )
    entries = outcome.entries
    rendered = [format_entry(e) for e in entries]
    bitwise_count = outcome.bitwise_count
    total = outcome.total if outcome.total else 1

    if outcome.no_overlap:
        return run_schema.make_failed_result(
            suite=suite_key,
            summary=(
                f"{summary_prefix}: no overlapping {key_prefix!r}-prefixed keys "
                f"between baseline ({len(baseline)} records) and candidate "
                f"({len(candidate)} records)."
            ),
            metrics=failure_metrics,
            details={
                "config": cfg,
                "baseline_keys": sorted(baseline.keys())[:20],
                "candidate_keys": sorted(candidate.keys())[:20],
                "output_tail": tail(output),
            },
        )

    overall = outcome.passed
    return {
        "status": "passed" if overall else "failed",
        "suite": suite_key,
        "summary": (
            f"{summary_prefix}: {bitwise_count}/{total} hash-equal"
            + (f" (ours returncode={returncode})" if returncode != 0 else "")
        ),
        "metrics": {
            "checks_total": total,
            "bitwise_match": bitwise_count,
            "hash_capture_level": hash_capture_level,
        },
        "details": {
            "config": cfg,
            "checks": capture_entry_details(entries),
            "rendered_checks": rendered,
            "ref_dump_dir": str(ref_artifacts.dump_dir),
            "ours_dump_dir": str(candidate_artifacts.dump_dir),
            "ref_graph": (str(ref_artifacts.graph_dump) if ref_artifacts.graph_dump else None),
            "ours_graph": (
                str(candidate_artifacts.graph_dump) if candidate_artifacts.graph_dump else None
            ),
            "output_tail": tail(output),
        },
    }


def main(argv: list[str] | None = None) -> int:
    return verdict_cli(run, argv, doc=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
