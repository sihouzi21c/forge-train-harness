"""Profile-snapshot verdict (profile-snapshot).

File-based port of the legacy ``dispatcher._run_profile_snapshot`` post-
processing (identical semantics and summary strings). ours-only diagnostic —
nothing is graded; the "verdict" is that the nsys capture rendered into
``summary.md`` + ``profile.json`` under the label's notes dir. Reads:

* ``args[0]`` = out_label (``<milestone>_round<N>``), ``args[1]`` = optional
  explicit prev_label for the Δ comparison
* the VARIANT ours product ``workload/src/config/<suite>@<milestone>.toml``
  (renderer mirror synthesis) — the exact shape the ours sh launched with
* ``<run_dir>/ours/status.json`` + ``ours.log``  — engine exit + [LOSS] lines
* ``<run_dir>/ours/profile.nsys-rep``            — run-dir公约 nsys report

Writes (via ``tools.profile_render.render``):
``<repo>/workload/notes/profile/<out_label>/{summary.md,profile.json}``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from evals.gate_product import GateProductError, load_gate_product
from evals.verdicts._kernels import filtered_avg
from evals.verdicts._shared import (
    overlay_product_verdict,
    read_side_status,
    read_text,
    tail,
)
from harness import run_schema
from harness.wire_format import parse_loss_lines

_MILESTONE_PREFIX = re.compile(r"^(.+)_round\d+$")


def _resolve_milestone(cfg: dict[str, Any], suite_key: str, out_label: str) -> tuple[str, str]:
    """Resolve (milestone, mirrored_gate) from the label — legacy messages."""
    mirror_gate = cfg.get("mirror_gate")
    if not isinstance(mirror_gate, dict):
        raise ValueError(
            f"{suite_key}: [evals.{suite_key}].mirror_gate table is required "
            "(maps milestone -> gate suite name)."
        )
    match = _MILESTONE_PREFIX.match(out_label)
    if match is None:
        raise ValueError(
            f"{suite_key}: out_label {out_label!r} must be '<milestone-name>_round<N>' "
            "(e.g. 'bitwise-perf_round0' or 'long-horizon_round3') so the gate to "
            "mirror can be resolved."
        )
    milestone = match.group(1)
    gate_name = mirror_gate.get(milestone)
    if gate_name is None:
        raise ValueError(
            f"{suite_key}: milestone {milestone!r} (from out_label {out_label!r}) "
            f"has no mirror_gate entry; known: {sorted(mirror_gate)}."
        )
    return milestone, str(gate_name)


def _resolve_prev_dir(profile_root: Path, out_label: str, prev_label: str | None) -> Path | None:
    """Verbatim port of ``dispatcher._resolve_profile_prev_dir``."""
    if prev_label:
        candidate = profile_root / prev_label
        return candidate if (candidate / "profile.json").exists() else None
    if not profile_root.exists():
        return None
    siblings = sorted(p.name for p in profile_root.iterdir() if p.is_dir() and p.name != out_label)
    earlier = [name for name in siblings if name < out_label]
    for name in reversed(earlier):
        if (profile_root / name / "profile.json").exists():
            return profile_root / name
    return None


def run(
    *,
    suite_key: str,
    run_dir: Path,
    repo_root: Path,
    workload_config: dict[str, Any],
    args: list[str],
) -> dict[str, Any]:
    out_label = args[0]
    prev_label = args[1] if len(args) > 1 else None
    if "/" in out_label or out_label.startswith("."):
        raise ValueError(
            f"profile-snapshot out_label {out_label!r} must be a plain directory "
            "name (no '/', no leading '.')."
        )
    cfg = overlay_product_verdict(
        dict(workload_config.get("evals", {}).get(suite_key) or {}),
        repo_root,
        suite_key,
    )
    milestone, _gate_name = _resolve_milestone(cfg, suite_key, out_label)
    summary_label = f"{milestone} {suite_key}:{out_label}"

    # Shape from the VARIANT product the ours sh actually launched with
    # (renderer mirror synthesis moved dispatcher._resolve_profile_shape to
    # render time; the variant already carries the mirrored gate's shape).
    variant_key = f"{suite_key}@{milestone}"
    try:
        variant = load_gate_product(repo_root, "ours", variant_key)
    except GateProductError as exc:
        raise ValueError(
            f"{suite_key}: variant ours product for milestone {milestone!r} not "
            f"rendered (render gate-configs first): {exc}"
        ) from exc

    def _int(key: str) -> int:
        value = variant.get(key)
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{suite_key}: variant product {variant_key!r} missing/invalid "
                f"shape key {key!r}: {value!r}"
            ) from exc

    world_size = _int("world_size")
    micro_batch_size = _int("micro_batch_size")
    grad_accum = _int("grad_accum_steps")
    num_steps = _int("num_steps")
    seq_length = int(
        variant.get("seq_length")
        or cfg.get("seq_length", workload_config.get("ref", {}).get("seq_length", 4096))
    )

    ours_dir = run_dir / "ours"
    status = read_side_status(ours_dir)
    output = read_text(ours_dir / "ours.log")
    if status["timed_out"]:
        return run_schema.make_failed_result(
            suite=suite_key,
            summary=f"{summary_label}: ours timed out after {status['timeout_s']}s",
            metrics={"world_size": world_size},
            details={"config": cfg, "output_tail": tail(output)},
        )
    returncode = int(status["returncode"])
    if returncode != 0:
        return run_schema.make_failed_result(
            suite=suite_key,
            summary=f"{summary_label}: training returncode={returncode}",
            metrics={"world_size": world_size, "returncode": returncode},
            details={"config": cfg, "output_tail": tail(output)},
        )

    # Run-dir公约: run_ours.sh exports FORGE_NSYS_RANK0_OUTPUT=$RUN_DIR/profile
    # when the variant carries nsys_profile=true; launch_dp appends .nsys-rep.
    nsys_rep = ours_dir / "profile.nsys-rep"
    if not nsys_rep.exists():
        return run_schema.make_failed_result(
            suite=suite_key,
            summary=(
                f"{summary_label}: training finished but nsys did not write "
                f"{nsys_rep}. Confirm nsys is installed on the GPU host."
            ),
            metrics={"world_size": world_size},
            details={"config": cfg, "output_tail": tail(output)},
        )

    profile_root = repo_root / "workload" / "notes" / "profile"
    out_dir = profile_root / out_label
    out_dir.mkdir(parents=True, exist_ok=True)
    prev_dir = _resolve_prev_dir(profile_root, out_label, prev_label)

    from tools.profile_render import render as render_profile

    loss_steps = parse_loss_lines(output)
    profiled_steps = len(loss_steps)
    step_time_ms = 0.0
    mfu_e2e_standard = 0.0
    if loss_steps:
        warmup = int(cfg.get("warmup_steps", 0))
        avg_time_s, _n = filtered_avg(loss_steps, warmup_steps=warmup, key="time_s")
        step_time_ms = avg_time_s * 1000.0
        mfu_e2e_standard, _n = filtered_avg(loss_steps, warmup_steps=warmup)

    run_meta = {
        "suite": suite_key,
        "world_size": world_size,
        "micro_batch_size": micro_batch_size,
        "seq_length": seq_length,
        "grad_accum_steps": grad_accum,
        "out_label": out_label,
    }
    try:
        render_profile(
            nsys_rep,
            out_dir,
            suite=suite_key,
            prev_dir=prev_dir,
            run_meta=run_meta,
            step_time_ms=step_time_ms,
            mfu_e2e_standard=mfu_e2e_standard,
            profiled_steps=profiled_steps,
        )
    except FileNotFoundError as exc:
        return run_schema.make_failed_result(
            suite=suite_key,
            summary=f"{summary_label}: profile_render failed: {exc}",
            metrics={"world_size": world_size},
            details={"config": cfg, "output_tail": tail(output)},
        )

    summary_md = out_dir / "summary.md"
    profile_json = out_dir / "profile.json"
    if not summary_md.exists() or not profile_json.exists():
        return run_schema.make_failed_result(
            suite=suite_key,
            summary=(
                f"{summary_label}: profile_render did not write summary.md + "
                "profile.json — check renderer error output."
            ),
            metrics={"world_size": world_size},
            details={"config": cfg, "output_tail": tail(output)},
        )

    return run_schema.make_run_result(
        status="passed",
        suite=suite_key,
        summary=(
            f"{summary_label} (DP={world_size}, {num_steps} steps): "
            f"summary.md + profile.json under {out_dir.relative_to(repo_root).as_posix()}"
            + (f"; Δ from {prev_dir.name}" if prev_dir is not None else "")
        ),
        metrics={
            "world_size": world_size,
            "num_steps": num_steps,
            "profiled_steps": profiled_steps,
            "step_time_ms": step_time_ms,
            "mfu_e2e_standard": mfu_e2e_standard,
            "out_dir": str(out_dir),
            "prev_dir": str(prev_dir) if prev_dir is not None else None,
        },
        details={
            "config": cfg,
            "summary_md": str(summary_md),
            "profile_json": str(profile_json),
            "nsys_rep": str(nsys_rep),
            "output_tail": tail(output),
        },
    )


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    from evals._common import _load_workload_config_for_ref

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("out_label")
    parser.add_argument("prev_label", nargs="?", default=None)
    args = parser.parse_args(argv)

    repo_root = Path.cwd()
    extra = [args.out_label] + ([args.prev_label] if args.prev_label else [])
    result = run(
        suite_key=args.gate,
        run_dir=args.run_dir,
        repo_root=repo_root,
        workload_config=_load_workload_config_for_ref(repo_root),
        args=extra,
    )
    json.dump(result, sys.stdout, indent=2, default=str)
    print()
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
