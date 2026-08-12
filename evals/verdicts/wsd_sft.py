"""WSD-SFT verdict (wsd-sft-70 / production-resume-70).

File-based port of the legacy ``_run_wsd_sft`` judgment (identical semantics,
summary strings and metric set). Two registrations share this module:

* ``wsd-sft-70`` (resume milestone) — plain 3-phase run. Verdict = structural
  ``[PHASE]`` switch assertions + per-step loss compare vs the clean baseline.
* ``production-resume-70`` (production milestone, step 1) — the SAME 3-phase
  line SIGKILLed at each crash step and resumed. Verdict = the crashed run's
  stitched trajectory must equal the never-crashed clean ours baseline
  BITWISE (``gate_atol=0``), plus the same structural assertions.

Reads (all written by ``evals/scripts/run_ours_wsd_sft.sh``):

* ``<run_dir>/ours/wsdsft_clean_baseline_output.log`` — baseline trajectory
* ``<run_dir>/ours/wsdsft_trajectory_output.log``     — main-run trajectory
* ``<run_dir>/ours/clean_returncode.txt`` / ``main_returncode.txt``
* ``<run_dir>/ours/status.json``                      — whole-runner exit
* the rendered ours product — gate shape (phase steps / gate_atol /
  crash_steps / comparison_basis); the registry entry carries only
  routing/transport keys.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from evals.gate_product import GateProductError, load_gate_product
from evals.verdicts._kernels import compare_series
from evals.verdicts._shared import read_side_status, read_text, tail
from harness import run_schema
from harness.wire_format import parse_loss_lines

if TYPE_CHECKING:
    from pathlib import Path

_WSD_PHASES = ("stable", "decay", "sft")

_PHASE_BANNER_RE = re.compile(
    r"\[PHASE\]\s+name=(?P<name>\S+)\s+start_step=(?P<start_step>\S+)\s+"
    r"lr=(?P<lr>\S+)\s+min_lr=(?P<min_lr>\S+)\s+warmup=(?P<warmup>\S+)\s+"
    r"decay=(?P<decay>\S+)\s+wsd_decay=(?P<wsd_decay>\S+)\s+"
    r"init_weights_only=(?P<iwo>\S+)\s+data_path=(?P<data_path>.*)$"
)


def _segment_trajectory_by_phase(
    text: str,
) -> tuple[list[dict[str, str]], dict[str, dict[int, dict[str, Any]]]]:
    """Split captured stdout into per-phase ``[LOSS]`` buckets keyed by ``[PHASE]``.

    Returns ``(banners_in_order, {phase_name: {step: loss_dict}})``. Each clean
    ``[PHASE] name=…`` line opens a new bucket; subsequent lines accrue to it and
    are parsed per-bucket with the strict ``[LOSS]`` grammar, so the stable
    step-1..S1 and sft step-1..S3 windows do not collide in one flat dict.
    """
    banners: list[dict[str, str]] = []
    order: list[str] = []
    chunks: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        m = _PHASE_BANNER_RE.match(line.strip())
        if m:
            fields = m.groupdict()
            banners.append(fields)
            current = fields["name"]
            if current not in chunks:
                order.append(current)
                chunks[current] = []
            continue
        if current is not None:
            chunks[current].append(line)
    by_phase: dict[str, dict[int, dict[str, Any]]] = {}
    for name in order:
        steps = parse_loss_lines("\n".join(chunks.get(name, [])), tag="LOSS")
        by_phase[name] = {s["step"]: s for s in steps}
    return banners, by_phase


def _read_rc(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def run(
    *,
    suite_key: str,
    run_dir: Path,
    repo_root: Path,
    workload_config: dict[str, Any],
) -> dict[str, Any]:
    cfg = dict(workload_config.get("evals", {}).get(suite_key) or {})
    milestone = str(cfg.get("milestone", suite_key))
    ours_dir = run_dir / "ours"

    # ── Gate shape from the rendered ours product (flat per-phase keys). ──
    try:
        product = load_gate_product(repo_root, "ours", suite_key)
    except GateProductError as exc:
        return run_schema.make_failed_result(
            suite=suite_key,
            summary=f"{milestone} {suite_key}: missing rendered ours product: {exc}",
            metrics={},
            details={"config": cfg},
        )
    try:
        world_size = int(product.get("world_size"))
        stable_steps = int(product.get("stable_steps"))
        decay_steps = int(product.get("decay_steps"))
        sft_steps = int(product.get("sft_steps"))
        gate_atol = float(product.get("gate_atol"))
        crash_steps = [int(c) for c in (product.get("crash_steps") or [])]
        comparison_basis = str(product.get("comparison_basis", "self"))
    except (TypeError, ValueError) as exc:
        return run_schema.make_failed_result(
            suite=suite_key,
            summary=f"{milestone} {suite_key}: bad/missing gate shape in ours product: {exc}",
            metrics={},
            details={"config": cfg},
        )
    if comparison_basis != "self":
        return run_schema.make_failed_result(
            suite=suite_key,
            summary=(
                f"{milestone} {suite_key}: comparison_basis={comparison_basis!r} is not "
                "available on this line (the pure-torch L0 ref has no 3-phase wrapper); "
                'use "self".'
            ),
            metrics={},
            details={"config": cfg},
        )

    status = read_side_status(ours_dir)
    clean_output = read_text(ours_dir / "wsdsft_clean_baseline_output.log")
    output = read_text(ours_dir / "wsdsft_trajectory_output.log")

    if status["timed_out"]:
        return run_schema.make_failed_result(
            suite=suite_key,
            summary=f"{milestone} {suite_key}: ours timed out after {status['timeout_s']}s",
            metrics={"world_size": world_size},
            details={"config": cfg, "output_tail": tail(output or clean_output)},
        )

    clean_rc = _read_rc(ours_dir / "clean_returncode.txt")
    if clean_rc is None:
        # The runner died before the clean baseline (product/env preflight).
        return run_schema.make_failed_result(
            suite=suite_key,
            summary=f"{milestone} {suite_key}: ours runner failed before the clean "
            f"baseline (returncode={status['returncode']}).",
            metrics={"world_size": world_size},
            details={"config": cfg, "output_tail": tail(read_text(ours_dir / "ours.log"))},
        )
    if clean_rc != 0:
        return run_schema.make_failed_result(
            suite=suite_key,
            summary=f"{milestone} {suite_key}: clean baseline returncode={clean_rc}.",
            metrics={"world_size": world_size},
            details={"config": cfg, "output_tail": tail(clean_output)},
        )
    _clean_banners, ref_by_phase = _segment_trajectory_by_phase(clean_output)
    baseline_desc = "ours-clean"

    main_rc = _read_rc(ours_dir / "main_returncode.txt")
    if main_rc is None:
        return run_schema.make_failed_result(
            suite=suite_key,
            summary=f"{milestone} {suite_key}: ours runner died before the main run "
            f"(returncode={status['returncode']}).",
            metrics={"world_size": world_size},
            details={"config": cfg, "output_tail": tail(read_text(ours_dir / "ours.log"))},
        )
    script_succeeded = main_rc == 0
    ours_banners, ours_by_phase = _segment_trajectory_by_phase(output)

    # ── Layer 1: per-phase per-step loss diff vs the baseline. Step windows are
    #    ABSOLUTE step numbers as emitted: stable counts 1..S1, decay continues
    #    S1+1..S1+S2 (full resume carries the counter), sft resets to 1..S3. ──
    phase_windows = {
        "stable": range(1, stable_steps + 1),
        "decay": range(stable_steps + 1, stable_steps + decay_steps + 1),
        "sft": range(1, sft_steps + 1),
    }
    max_loss_diff = 0.0
    step_diffs: list[dict[str, Any]] = []
    first_violation: dict[str, Any] | None = None
    missing: list[dict[str, Any]] = []
    for name in _WSD_PHASES:
        records = compare_series(
            {s: e["global_loss"] for s, e in ref_by_phase.get(name, {}).items()},
            {s: e["global_loss"] for s, e in ours_by_phase.get(name, {}).items()},
            phase_windows[name],
        )
        for r in records:
            if r.abs_diff is None:
                missing.append(
                    {
                        "phase": name,
                        "step": r.step,
                        "ref": r.baseline is not None,
                        "ours": r.ours is not None,
                    }
                )
                continue
            max_loss_diff = max(max_loss_diff, r.abs_diff)
            rec = {
                "phase": name,
                "step": r.step,
                "ref_loss": r.baseline,
                "ours_loss": r.ours,
                "loss_abs_diff": r.abs_diff,
            }
            step_diffs.append(rec)
            if r.abs_diff > gate_atol and first_violation is None:
                first_violation = rec
    tolerance_pass = not missing and max_loss_diff <= gate_atol

    # ── Layer 2: structural [PHASE] switch assertions on the OURS banners.
    #    These are what stop a degenerate "trained stable the whole time" run
    #    from passing a self-comparison. ──
    struct_reasons: list[str] = []
    ours_banner_by_name: dict[str, dict[str, str]] = {}
    for b in ours_banners:
        ours_banner_by_name.setdefault(b["name"], b)

    # Dedup repeated names before the order check: under crash-resume each phase
    # emits a banner on both its first launch AND every restart (and completed
    # phases emit none on later attempts), so the raw banner stream is e.g.
    # [stable, stable, decay, decay, sft, sft]. First-occurrence order is the
    # invariant; dict.fromkeys preserves it.
    observed_order = list(dict.fromkeys(b["name"] for b in ours_banners))[: len(_WSD_PHASES)]
    if observed_order != list(_WSD_PHASES):
        struct_reasons.append(f"phase order {observed_order} != {list(_WSD_PHASES)}")

    b_stable = ours_banner_by_name.get("stable")
    b_decay = ours_banner_by_name.get("decay")
    b_sft = ours_banner_by_name.get("sft")

    if b_stable is None:
        struct_reasons.append("missing ours [PHASE] banner for stable")
    else:
        if int(b_stable["start_step"]) != 0:
            struct_reasons.append(f"stable start_step {b_stable['start_step']} != 0")
        if b_stable["iwo"] != "0":
            struct_reasons.append("stable init_weights_only != 0")

    if b_decay is None:
        struct_reasons.append("missing ours [PHASE] banner for decay")
    else:
        if int(b_decay["start_step"]) != stable_steps:
            struct_reasons.append(
                f"decay start_step {b_decay['start_step']} != {stable_steps} "
                "(must carry the stable counter on full resume)"
            )
        if b_decay["iwo"] != "0":
            struct_reasons.append("decay init_weights_only != 0 (must be full resume)")
        if b_stable is not None and b_decay["data_path"] == b_stable["data_path"]:
            struct_reasons.append("decay data_path did not swap from stable")

    if b_sft is None:
        struct_reasons.append("missing ours [PHASE] banner for sft")
    else:
        if int(b_sft["start_step"]) != 0:
            struct_reasons.append(f"sft start_step {b_sft['start_step']} != 0 (must reset)")
        if b_sft["iwo"] != "1":
            struct_reasons.append("sft init_weights_only != 1 (must re-init the optimizer)")
        if b_decay is not None and float(b_sft["lr"]) >= float(b_decay["lr"]):
            struct_reasons.append(
                f"sft lr {b_sft['lr']} did not drop below decay lr {b_decay['lr']}"
            )
        if b_decay is not None and b_sft["data_path"] == b_decay["data_path"]:
            struct_reasons.append("sft data_path did not swap from decay")

    struct_pass = not struct_reasons
    overall_pass = tolerance_pass and struct_pass and script_succeeded

    summary_parts = [
        f"max_abs_diff(loss)={max_loss_diff:.3e} (atol={gate_atol:.1e})",
        f"phases stable/decay/sft={stable_steps}/{decay_steps}/{sft_steps}",
        f"tolerance={'PASS' if tolerance_pass else 'FAIL'}",
        f"structural={'PASS' if struct_pass else 'FAIL'}",
    ]
    if crash_steps:
        summary_parts.append(f"crash_steps={crash_steps}")
    if missing:
        summary_parts.append(f"missing {len(missing)} step(s), first {missing[0]}")
    if struct_reasons:
        summary_parts.append(f"switch: {struct_reasons[0]}")
    if not script_succeeded:
        summary_parts.append(f"ours returncode={main_rc}")

    return {
        "status": "passed" if overall_pass else "failed",
        "suite": suite_key,
        "summary": f"{milestone} {suite_key} (DP={world_size}, ours vs {baseline_desc}): {'; '.join(summary_parts)}.",
        "metrics": {
            "world_size": world_size,
            "stable_steps": stable_steps,
            "decay_steps": decay_steps,
            "sft_steps": sft_steps,
            "gate_atol": gate_atol,
            "max_abs_diff_loss": max_loss_diff,
            "tolerance_pass": tolerance_pass,
            "structural_pass": struct_pass,
            "missing_steps": len(missing),
            "passed": overall_pass,
        },
        "details": {
            "config": cfg,
            "first_violation": first_violation,
            "missing": missing[:20],
            "struct_reasons": struct_reasons,
            "ours_banners": ours_banners,
            "step_diffs_tail": step_diffs[-20:],
            "output_tail": tail(output),
        },
    }
