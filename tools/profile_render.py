"""Render an nsys profile snapshot into summary.md + profile.json.

Stage 1 ``perf-bitwise`` / ``long-train`` only. Called from the ``profile-snapshot`` dispatcher
runner after the rank-0 ``nsys profile`` wrap (see
``evals/scripts/launch_dp.py``) finishes producing a ``.nsys-rep``
file. The function is deterministic — no LLM, no network — and
stdlib-only at import time.

Inputs
------
``nsys_rep`` — absolute path to the rank-0 ``.nsys-rep`` produced by
``nsys profile``. The renderer shells out to ``nsys stats`` to extract
four preset CSV reports (kernels, memops, CUDA-API summary, OS-runtime
summary) and parses them into a single in-memory snapshot. Every
window-cumulative time is normalized to a single training step by
dividing by ``profiled_steps`` (the count of steps run under nsys).

Outputs
-------
``<out_dir>/summary.md`` — header (per-step totals + profiled_steps),
top-N GPU kernels, memcpy table, top-N CUDA API calls, top-N OS-runtime
calls (host-side blocking: distinguishes comm-wait from GPU-wait), and
the Δ-vs-prev block.

``<out_dir>/profile.json`` — machine schema for the next round's
``prev_dir`` to diff against. Shape mismatch (different MBS / GBS /
DP / seq_len) refuses the diff loudly instead of silently lying.

Why nsys instead of torch.profiler
----------------------------------
The torch.profiler renderer that lived here previously depended on
the agent wrapping every step body with 11 ``record_function`` names.
long-horizon round 8-11 (loop ``efcdff23``) demonstrated the failure mode:
when transformer-layer wrap sites were missing, 60-65% of the run
landed in the ``uncategorized`` bucket and the renderer's
"missing wrap site" hint was the loudest signal — useless for
optimization. Nsys captures every kernel by name unconditionally,
so there is no wrap-site contract to miss and no ``uncategorized``
bucket to chase.
"""

from __future__ import annotations

import csv
import io
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# nsys stats reports we extract. Stable preset names — supported
# across nsys 2023.x / 2024.x. Each maps to a CSV with a fixed
# header that ``_parse_csv`` keys by column name (defensive against
# minor schema bumps).
_REPORTS: tuple[str, ...] = (
    "cuda_gpu_kern_sum",
    "cuda_gpu_mem_time_sum",
    "cuda_api_sum",
    "osrt_sum",
)

# Shape keys that must match between two snapshots before a Δ block
# can be rendered. Mismatched shape is loud, not silent — the agent
# should not chase a fake regression because they changed MBS.
_SHAPE_KEYS: tuple[str, ...] = (
    "suite",
    "world_size",
    "micro_batch_size",
    "seq_length",
    "grad_accum_steps",
)

# Top-N rows surfaced in summary.md. Bigger N hurts skimmability;
# smaller N risks dropping the kernel that actually moved.
_TOP_KERNELS = 15
_TOP_API = 10


@dataclass
class KernelRow:
    name: str
    per_step_ms: float
    instances: int
    pct: float


@dataclass
class MemopRow:
    operation: str
    per_step_ms: float
    count: int
    pct: float


@dataclass
class ApiRow:
    name: str
    per_step_ms: float
    calls: int
    pct: float


@dataclass
class OsrtRow:
    name: str
    per_step_ms: float
    calls: int
    pct: float


@dataclass
class ProfileSnapshot:
    run_meta: dict[str, Any]
    step_time_ms: float
    mfu_e2e_standard: float
    profiled_steps: int
    gpu_kernel_per_step_ms: float
    gpu_memop_per_step_ms: float
    cuda_api_per_step_ms: float
    os_runtime_per_step_ms: float
    # GPU compute-engine idle per step. Derived as
    # ``step_time_ms - gpu_kernel_per_step_ms`` under the SINGLE-STREAM
    # assumption: when all compute kernels share one stream, the kernel
    # duration sum equals the compute engine's busy wall-clock, so the
    # remainder is genuine idle (CPU-side stall — dataloader / sync tail
    # / launch starvation / true idle). If the workload later introduces
    # a compute stream B (e.g. wgrad ↔ dgrad overlap), this derivation
    # over-counts active because two streams' kernels can run
    # concurrently — switch to nsys cuda_gpu_trace + interval merge in
    # that case.
    gpu_idle_per_step_ms: float = 0.0
    top_kernels: list[KernelRow] = field(default_factory=list)
    memops: list[MemopRow] = field(default_factory=list)
    top_api: list[ApiRow] = field(default_factory=list)
    top_osrt: list[OsrtRow] = field(default_factory=list)


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------


def render(
    nsys_rep: Path,
    out_dir: Path,
    *,
    suite: str,
    prev_dir: Path | None = None,
    run_meta: dict[str, Any] | None = None,
    step_time_ms: float = 0.0,
    mfu_e2e_standard: float = 0.0,
    profiled_steps: int = 0,
    _nsys_runner: Any = None,
) -> None:
    """Parse *nsys_rep*, write summary.md + profile.json under *out_dir*.

    *nsys_rep* must point at an existing ``.nsys-rep`` file produced by
    ``nsys profile``. ``_nsys_runner`` is a test seam — callers pass a
    callable ``(report: str, nsys_rep: Path) -> str`` (CSV body) instead
    of shelling out to the real binary.

    Every window-cumulative time (the three GPU/API/OSRT totals and each
    per-row time) is normalized to a single training step by dividing by
    *profiled_steps* — the count of steps actually executed under nsys.
    ``profiled_steps == 0`` (OOM / empty trajectory) yields 0.0 for every
    per-step field rather than dividing by zero.
    """
    nsys_rep = Path(nsys_rep)
    if not nsys_rep.exists():
        raise FileNotFoundError(f"nsys report not found: {nsys_rep}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(run_meta or {})
    meta.setdefault("suite", suite)
    meta["nsys_rep_path"] = str(nsys_rep)

    runner = _nsys_runner or _run_nsys_stats
    csv_payloads = {report: runner(report, nsys_rep) for report in _REPORTS}

    kernels = _parse_kernels(csv_payloads["cuda_gpu_kern_sum"], profiled_steps)
    memops = _parse_memops(csv_payloads["cuda_gpu_mem_time_sum"], profiled_steps)
    api = _parse_api(csv_payloads["cuda_api_sum"], profiled_steps)
    osrt = _parse_osrt(csv_payloads["osrt_sum"], profiled_steps)

    gpu_kernel_ms = round(sum(k.per_step_ms for k in kernels), 3)
    # Clamp at 0 so a degenerate input (step_time < kernel sum, e.g.
    # multi-stream where the assumption is violated) does not surface
    # as a negative idle that misleads downstream readers.
    gpu_idle_ms = round(max(0.0, step_time_ms - gpu_kernel_ms), 3)
    snapshot = ProfileSnapshot(
        run_meta=meta,
        step_time_ms=round(step_time_ms, 3),
        mfu_e2e_standard=round(mfu_e2e_standard, 4),
        profiled_steps=int(profiled_steps),
        gpu_kernel_per_step_ms=gpu_kernel_ms,
        gpu_memop_per_step_ms=round(sum(m.per_step_ms for m in memops), 3),
        cuda_api_per_step_ms=round(sum(a.per_step_ms for a in api), 3),
        os_runtime_per_step_ms=round(sum(o.per_step_ms for o in osrt), 3),
        gpu_idle_per_step_ms=gpu_idle_ms,
        top_kernels=kernels[:_TOP_KERNELS],
        memops=memops,
        top_api=api[:_TOP_API],
        top_osrt=osrt[:_TOP_API],
    )

    prev_snapshot, prev_reason = _load_prev_snapshot(prev_dir, meta)

    (out_dir / "profile.json").write_text(
        json.dumps(_snapshot_to_json(snapshot), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.md").write_text(
        _render_summary_md(snapshot, prev_snapshot, prev_reason, prev_dir),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# nsys stats invocation
# ---------------------------------------------------------------------


def _run_nsys_stats(report: str, nsys_rep: Path) -> str:
    """Shell out to ``nsys stats`` and return the CSV body as text.

    Output goes to stdout; nsys prints a 4-5 line preamble before the
    CSV header, so ``_parse_csv`` skips lines until it sees a row whose
    first field is the known header token.
    """
    proc = subprocess.run(  # nosec B603 — fixed argv, no shell
        [
            "nsys",
            "stats",
            "--report",
            report,
            "--format",
            "csv",
            "--force-export=true",
            str(nsys_rep),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


# ---------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------


def _parse_csv(payload: str, required_header_token: str) -> list[dict[str, str]]:
    """Parse a nsys-stats CSV blob, skipping any preamble before the header.

    *required_header_token* is a column name that uniquely identifies the
    real header row (nsys prints "Generating report... etc" lines first).
    """
    lines = payload.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if required_header_token in line and "," in line:
            header_idx = i
            break
    if header_idx is None:
        return []
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_idx:])))
    return [dict(row) for row in reader if any(row.values())]


def _ns_to_ms(value: str) -> float:
    """Convert an nsys time field (nanoseconds, may contain commas) to ms."""
    cleaned = (value or "0").replace(",", "").strip()
    try:
        return float(cleaned) / 1_000_000.0
    except ValueError:
        return 0.0


def _to_int(value: str) -> int:
    cleaned = (value or "0").replace(",", "").strip()
    try:
        return int(float(cleaned))
    except ValueError:
        return 0


def _to_float(value: str) -> float:
    cleaned = (value or "0").replace(",", "").replace("%", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def _per_step(total_ms: float, profiled_steps: int) -> float:
    """Normalize a window-cumulative ms value to a single step.

    ``profiled_steps == 0`` (OOM / empty trajectory) returns 0.0 — no
    samples to average over, surfaced as 0 rather than a ZeroDivisionError.
    """
    if profiled_steps <= 0:
        return 0.0
    return total_ms / profiled_steps


def _parse_kernels(payload: str, profiled_steps: int) -> list[KernelRow]:
    rows = _parse_csv(payload, "Name")
    out: list[KernelRow] = []
    for r in rows:
        total_ms = _ns_to_ms(r.get("Total Time (ns)", "0"))
        out.append(
            KernelRow(
                name=str(r.get("Name", "")).strip(),
                per_step_ms=round(_per_step(total_ms, profiled_steps), 3),
                instances=_to_int(r.get("Instances", "0")),
                pct=round(_to_float(r.get("Time (%)", "0")), 3),
            )
        )
    out.sort(key=lambda k: k.per_step_ms, reverse=True)
    return out


def _parse_memops(payload: str, profiled_steps: int) -> list[MemopRow]:
    rows = _parse_csv(payload, "Operation")
    out: list[MemopRow] = []
    for r in rows:
        total_ms = _ns_to_ms(r.get("Total Time (ns)", "0"))
        out.append(
            MemopRow(
                operation=str(r.get("Operation", "")).strip(),
                per_step_ms=round(_per_step(total_ms, profiled_steps), 3),
                count=_to_int(r.get("Count", "0")),
                pct=round(_to_float(r.get("Time (%)", "0")), 3),
            )
        )
    out.sort(key=lambda m: m.per_step_ms, reverse=True)
    return out


def _parse_api(payload: str, profiled_steps: int) -> list[ApiRow]:
    rows = _parse_csv(payload, "Name")
    out: list[ApiRow] = []
    for r in rows:
        total_ms = _ns_to_ms(r.get("Total Time (ns)", "0"))
        out.append(
            ApiRow(
                name=str(r.get("Name", "")).strip(),
                per_step_ms=round(_per_step(total_ms, profiled_steps), 3),
                calls=_to_int(r.get("Num Calls", "0")),
                pct=round(_to_float(r.get("Time (%)", "0")), 3),
            )
        )
    out.sort(key=lambda a: a.per_step_ms, reverse=True)
    return out


def _parse_osrt(payload: str, profiled_steps: int) -> list[OsrtRow]:
    rows = _parse_csv(payload, "Name")
    out: list[OsrtRow] = []
    for r in rows:
        total_ms = _ns_to_ms(r.get("Total Time (ns)", "0"))
        out.append(
            OsrtRow(
                name=str(r.get("Name", "")).strip(),
                per_step_ms=round(_per_step(total_ms, profiled_steps), 3),
                calls=_to_int(r.get("Num Calls", "0")),
                pct=round(_to_float(r.get("Time (%)", "0")), 3),
            )
        )
    out.sort(key=lambda o: o.per_step_ms, reverse=True)
    return out


# ---------------------------------------------------------------------
# Δ / prev snapshot
# ---------------------------------------------------------------------


def _load_prev_snapshot(
    prev_dir: Path | None,
    cur_meta: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    if prev_dir is None:
        return None, "no prior snapshot"
    prev_path = Path(prev_dir) / "profile.json"
    if not prev_path.exists():
        return None, f"no profile.json under {prev_dir}"
    try:
        prev = json.loads(prev_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"prev profile.json unreadable: {exc}"
    prev_meta = prev.get("run_meta", {})
    mismatched = [k for k in _SHAPE_KEYS if str(prev_meta.get(k)) != str(cur_meta.get(k))]
    if mismatched:
        return None, f"shape mismatch on {mismatched}"
    return prev, "ok"


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------


def _render_summary_md(
    cur: ProfileSnapshot,
    prev: dict[str, Any] | None,
    prev_reason: str,
    prev_dir: Path | None,
) -> str:
    meta = cur.run_meta
    lines: list[str] = []
    lines.append(f"# profile snapshot — {meta.get('suite', '?')}")
    lines.append("")
    lines.append(
        f"step_time_ms={cur.step_time_ms:.3f}  "
        f"mfu_e2e_standard={cur.mfu_e2e_standard:.4f}  "
        f"profiled_steps={cur.profiled_steps}  "
        f"gpu_kernel_per_step_ms={cur.gpu_kernel_per_step_ms:.3f}  "
        f"gpu_idle_per_step_ms={cur.gpu_idle_per_step_ms:.3f}  "
        f"gpu_memop_per_step_ms={cur.gpu_memop_per_step_ms:.3f}  "
        f"cuda_api_per_step_ms={cur.cuda_api_per_step_ms:.3f}  "
        f"os_runtime_per_step_ms={cur.os_runtime_per_step_ms:.3f}"
    )
    # Single-stream assumption: gpu_idle = step - gpu_kernel. If a future
    # round introduces compute stream parallelism (e.g. wgrad/dgrad on
    # separate streams), this derivation over-counts active because
    # concurrent kernels are summed independently; switch to interval
    # merge over `cuda_gpu_trace` then. cuda_api_per_step_ms and
    # os_runtime_per_step_ms are CPU-side sums (concurrent with GPU and
    # cross-thread for os_runtime); they are NOT addable to step_time.
    lines.append(
        "# gpu_idle = step - gpu_kernel (valid only when compute kernels share one stream);"
        " cuda_api / os_runtime are CPU time, do NOT subtract from step_time."
    )
    shape_bits = "  ".join(f"{k}={meta.get(k, '?')}" for k in _SHAPE_KEYS if k != "suite")
    if shape_bits:
        lines.append(shape_bits)
    git_sha = meta.get("git_sha")
    if git_sha:
        lines.append(f"git_sha={git_sha}")
    lines.append(f"nsys_rep={meta.get('nsys_rep_path', '?')}")
    lines.append("")

    lines.append(f"## top {_TOP_KERNELS} GPU kernels by total time")
    lines.append("")
    lines.append("| kernel | per_step_ms | pct | instances |")
    lines.append("|---|---:|---:|---:|")
    for k in cur.top_kernels:
        name = k.name if len(k.name) <= 80 else k.name[:77] + "..."
        lines.append(f"| {name} | {k.per_step_ms:.3f} | {k.pct:.2f}% | {k.instances} |")
    lines.append("")

    lines.append("## GPU memory operations")
    lines.append("")
    lines.append("| op | per_step_ms | pct | count |")
    lines.append("|---|---:|---:|---:|")
    for m in cur.memops:
        lines.append(f"| {m.operation} | {m.per_step_ms:.3f} | {m.pct:.2f}% | {m.count} |")
    lines.append("")

    lines.append(f"## top {_TOP_API} CUDA API calls by total time")
    lines.append("")
    lines.append("| api | per_step_ms | pct | calls |")
    lines.append("|---|---:|---:|---:|")
    for a in cur.top_api:
        lines.append(f"| {a.name} | {a.per_step_ms:.3f} | {a.pct:.2f}% | {a.calls} |")
    lines.append("")

    lines.append(f"## top {_TOP_API} OS runtime calls by total time")
    lines.append("")
    lines.append("| os call | per_step_ms | pct | calls |")
    lines.append("|---|---:|---:|---:|")
    for o in cur.top_osrt:
        lines.append(f"| {o.name} | {o.per_step_ms:.3f} | {o.pct:.2f}% | {o.calls} |")
    lines.append("")

    lines.append("## Δ from previous snapshot")
    lines.append("")
    if prev is None:
        prev_name = prev_dir.name if prev_dir is not None else "n/a"
        lines.append(f"Δ from previous: unavailable ({prev_reason}; prev_dir={prev_name})")
    else:
        prev_name = prev_dir.name if prev_dir is not None else "?"
        lines.append(f"Δ from {prev_name}")
        lines.append("")
        d_step = cur.step_time_ms - float(prev.get("step_time_ms", 0.0))
        d_mfu = cur.mfu_e2e_standard - float(prev.get("mfu_e2e_standard", 0.0))
        d_steps = cur.profiled_steps - int(prev.get("profiled_steps", 0))
        d_gpu = cur.gpu_kernel_per_step_ms - float(prev.get("gpu_kernel_per_step_ms", 0.0))
        d_idle = cur.gpu_idle_per_step_ms - float(prev.get("gpu_idle_per_step_ms", 0.0))
        d_mem = cur.gpu_memop_per_step_ms - float(prev.get("gpu_memop_per_step_ms", 0.0))
        d_api = cur.cuda_api_per_step_ms - float(prev.get("cuda_api_per_step_ms", 0.0))
        d_osrt = cur.os_runtime_per_step_ms - float(prev.get("os_runtime_per_step_ms", 0.0))
        lines.append(
            f"Δ step_time_ms={d_step:+.3f}  "
            f"Δ mfu_e2e_standard={d_mfu:+.4f}  "
            f"Δ profiled_steps={d_steps:+d}  "
            f"Δ gpu_kernel_per_step_ms={d_gpu:+.3f}  "
            f"Δ gpu_idle_per_step_ms={d_idle:+.3f}  "
            f"Δ gpu_memop_per_step_ms={d_mem:+.3f}  "
            f"Δ cuda_api_per_step_ms={d_api:+.3f}  "
            f"Δ os_runtime_per_step_ms={d_osrt:+.3f}"
        )
        lines.append("")
        prev_kernels = {k["name"]: k for k in prev.get("top_kernels", [])}
        cur_kernels = {k.name: k for k in cur.top_kernels}
        lines.append("| kernel | Δ per_step_ms | Δ pct |  state |")
        lines.append("|---|---:|---:|---|")
        for k in cur.top_kernels:
            pk = prev_kernels.get(k.name)
            if pk is None:
                lines.append(f"| {_truncate_name(k.name)} | (new) | (new) | new |")
                continue
            d_ms = k.per_step_ms - float(pk.get("per_step_ms", 0.0))
            d_pct = k.pct - float(pk.get("pct", 0.0))
            lines.append(f"| {_truncate_name(k.name)} | {d_ms:+.3f} | {d_pct:+.2f}% | tracked |")
        for name, pk in prev_kernels.items():
            if name not in cur_kernels:
                lines.append(
                    f"| {_truncate_name(name)} | -{float(pk.get('per_step_ms', 0.0)):.3f} | -{float(pk.get('pct', 0.0)):.2f}% | dropped |"
                )
    lines.append("")
    return "\n".join(lines)


def _truncate_name(name: str, limit: int = 80) -> str:
    return name if len(name) <= limit else name[: limit - 3] + "..."


def _snapshot_to_json(snap: ProfileSnapshot) -> dict[str, Any]:
    return {
        "run_meta": snap.run_meta,
        "step_time_ms": snap.step_time_ms,
        "mfu_e2e_standard": snap.mfu_e2e_standard,
        "profiled_steps": snap.profiled_steps,
        "gpu_kernel_per_step_ms": snap.gpu_kernel_per_step_ms,
        "gpu_idle_per_step_ms": snap.gpu_idle_per_step_ms,
        "gpu_memop_per_step_ms": snap.gpu_memop_per_step_ms,
        "cuda_api_per_step_ms": snap.cuda_api_per_step_ms,
        "os_runtime_per_step_ms": snap.os_runtime_per_step_ms,
        "top_kernels": [asdict(k) for k in snap.top_kernels],
        "memops": [asdict(m) for m in snap.memops],
        "top_api": [asdict(a) for a in snap.top_api],
        "top_osrt": [asdict(o) for o in snap.top_osrt],
    }
