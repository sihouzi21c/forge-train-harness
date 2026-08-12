"""Shared helpers for verdict modules (file-based gate judgment)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from evals._common import output_tail_limit, write_ref_capture_status
from evals.gate_product import GateProductError, load_gate_product
from harness.wire_format import (
    parse_loss_dump_file,
    parse_loss_dump_with_grad,
    parse_loss_lines,
    parse_stdout_loss,
)

STATUS_FILENAME = "status.json"

# Verdict / threshold + init keys whose authoritative per-gate value is the
# rendered product ``[cli]``. Same list as the legacy
# ``dispatcher._PRODUCT_VERDICT_KEYS`` (which dies with the old handlers in
# the final migration step).
PRODUCT_VERDICT_KEYS = (
    "gate_bitwise",
    "gate_atol",
    "hash_capture_level",
    "mfu_e2e_target",
    "warmup_steps",
    "loss_rel_threshold",
    "loss_abs_threshold",
    "grad_norm_abs_threshold",
    "max_avg_relative_loss_diff",
    "forge_init_ones",
)


def overlay_product_verdict(
    cfg: dict[str, Any], repo_root: Path, suite_key: str, *, side: str = "ref"
) -> dict[str, Any]:
    """Return a cfg copy with verdict thresholds from the rendered product."""
    if not isinstance(cfg, dict):
        return cfg
    try:
        product = load_gate_product(repo_root, side, suite_key)
    except GateProductError:
        return cfg
    merged = dict(cfg)
    for key in PRODUCT_VERDICT_KEYS:
        val = product.get(key)
        if val is not None:
            merged[key] = val
    return merged


def read_side_status(side_dir: Path) -> dict[str, Any]:
    """Read a side's ``status.json`` written by the generic executor.

    Fail-closed: a missing or unreadable status file reads as a failed,
    zero-duration invocation — the side subprocess never reached a verdict.
    """
    fallback = {
        "returncode": -1,
        "elapsed_s": 0.0,
        "timed_out": False,
        "timeout_s": None,
        "cached": False,
    }
    path = side_dir / STATUS_FILENAME
    if not path.exists():
        return fallback
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return fallback
    return {**fallback, **raw}


def tail(text: str, limit: int | None = None) -> str:
    """Return at most *limit* trailing characters (default: output_tail_limit)."""
    cap = limit if limit is not None else output_tail_limit()
    return text[-cap:] if len(text) > cap else text


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def verdict_cli(run_fn: Any, argv: list[str] | None = None, *, doc: str | None = None) -> int:
    """The shared ``python -m evals.verdicts.<module> <gate> <run_dir>`` CLI."""
    import argparse
    import json
    import sys

    from evals._common import _load_workload_config_for_ref

    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument("gate")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)

    repo_root = Path.cwd()
    result = run_fn(
        suite_key=args.gate,
        run_dir=args.run_dir,
        repo_root=repo_root,
        workload_config=_load_workload_config_for_ref(repo_root),
    )
    json.dump(result, sys.stdout, indent=2, default=str)
    print()
    return 0 if result.get("status") == "passed" else 1


# ──────────────────────────────────────────────────────────────────────
# Phase resolution — mechanical artifact reading shared by every
# trajectory-comparing verdict (bitwise / long_train / loss_gate).
# No judgment here: each verdict builds its OWN failure results so the
# legacy per-handler summary strings survive the port verbatim.
# ──────────────────────────────────────────────────────────────────────


@dataclass
class RefPhase:
    """Ref side of a finished run, resolved from ``<run_dir>/ref/``."""

    status: dict[str, Any]
    elapsed_s: float
    baseline_by_step: dict[int, float]
    grad_baseline_by_step: dict[int, float]
    succeeded: bool


@dataclass
class OursPhase:
    """Ours side of a finished run, resolved from ``<run_dir>/ours/``."""

    status: dict[str, Any]
    returncode: int
    timed_out: bool
    timeout_s: Any
    output: str
    loss_steps: list[dict[str, Any]] = field(default_factory=list)
    ours_by_step: dict[int, dict[str, Any]] = field(default_factory=dict)


def resolve_ref_phase(run_dir: Path, *, hash_capture_level: int = 0) -> RefPhase:
    """Read the ref side's status + trajectory and record the capture status.

    Mirrors the legacy ``gate_common.resolve_ref_trajectory`` artifact
    handling: loss dump first (``dump/ref_loss.txt``), stdout fallback
    (``ref.log``), and the structural ``write_ref_capture_status`` record
    the meta harness_configs gate reads — written on both pass and fail.
    """
    ref_dir = run_dir / "ref"
    ref_hash_dump = ref_dir / "ref_hash_dump.json"
    status = read_side_status(ref_dir)

    loss_file = ref_dir / "dump" / "ref_loss.txt"
    baseline_by_step: dict[int, float] = {}
    grad_baseline_by_step: dict[int, float] = {}
    if loss_file.exists():
        baseline_by_step = parse_loss_dump_file(loss_file)
        grad_baseline_by_step = {
            step: entry["grad_norm"] for step, entry in parse_loss_dump_with_grad(loss_file).items()
        }
    if not baseline_by_step and (ref_dir / "ref.log").exists():
        baseline_by_step = parse_stdout_loss(ref_dir / "ref.log")

    write_ref_capture_status(
        run_dir,
        SimpleNamespace(
            returncode=int(status["returncode"]),
            timed_out=bool(status["timed_out"]),
        ),
        dump_present=bool(hash_capture_level > 0 and ref_hash_dump.exists()),
        hash_capture_level=hash_capture_level,
        graph_present=bool(ref_hash_dump.with_name(ref_hash_dump.name + ".graph.json").exists()),
        loss_by_step=baseline_by_step,
    )

    return RefPhase(
        status=status,
        elapsed_s=float(status.get("elapsed_s") or 0.0),
        baseline_by_step=baseline_by_step,
        grad_baseline_by_step=grad_baseline_by_step,
        succeeded=(int(status["returncode"]) == 0 and not bool(status["timed_out"])),
    )


def resolve_ours_phase(run_dir: Path) -> OursPhase:
    """Read the ours side's status + stdout and parse its [LOSS] trajectory."""
    ours_dir = run_dir / "ours"
    status = read_side_status(ours_dir)
    output = read_text(ours_dir / "ours.log")
    loss_steps = parse_loss_lines(output)
    return OursPhase(
        status=status,
        returncode=int(status["returncode"]),
        timed_out=bool(status["timed_out"]),
        timeout_s=status.get("timeout_s"),
        output=output,
        loss_steps=loss_steps,
        ours_by_step={s["step"]: s for s in loss_steps},
    )
