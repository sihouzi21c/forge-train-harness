"""SSOT writer for MFU measurements emitted by gate runs.

Single writer used by :func:`evals.runner.run_request` after every
suite result is finalized. Persists one artifact under
``$FORGE_TRAIN_DIR``:

* ``<loop_id>/mfu_history.jsonl`` — append-only per-loop history;
  every measured run lands here regardless of precision outcome.

Best-effort: any I/O / schema problem is logged to stderr and
swallowed. The gate's own pass/fail signal MUST never be affected by
telemetry persistence failing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

__all__ = [
    "HISTORY_FILENAME",
    "record",
]

HISTORY_FILENAME = "mfu_history.jsonl"


def record(
    result: dict[str, Any],
    repo_root: Path,
    *,
    env: dict[str, str] | None = None,
    now: float | None = None,
) -> dict[str, Any] | None:
    """Persist one MFU record. Returns the entry written or ``None`` if
    nothing was recorded — the suite didn't measure MFU, or the loop
    layout couldn't be resolved.

    Failures during persistence are caught and printed to stderr; this
    function never raises.
    """
    env_map = dict(env if env is not None else os.environ)
    metrics = result.get("metrics") or {}
    avg_mfu = metrics.get("avg_mfu_e2e_standard")
    if avg_mfu is None:
        return None

    loop_id = _resolve_loop_id(repo_root, env_map)
    train_dir = _resolve_forge_train_dir(repo_root, env_map)
    if loop_id is None or train_dir is None:
        # Loud skip: the gate verdict must stay unaffected, but a gate-
        # critical consumer (mfu_elastic_check / the long-horizon review
        # override) reads this history — a silent miss here deadlocked
        # loop 9f03ffd324af for 5 rounds. One stderr line makes the gap
        # visible in the run log.
        print(
            "mfu_record: SKIPPED — loop layout unresolved (set LOOP_ID / "
            "FORGE_TRAIN_DIR); this run will be missing from mfu_history.jsonl",
            file=sys.stderr,
        )
        return None

    precision = _precision_pass(metrics)
    entry: dict[str, Any] = {
        "ts": float(now if now is not None else time.time()),
        "loop_id": loop_id,
        "milestone": _milestone_for(result),
        "suite": str(result.get("suite") or ""),
        "avg_mfu": float(avg_mfu),
        "mfu_target": metrics.get("mfu_target"),
        "mfu_pass": metrics.get("mfu_pass"),
        "precision_pass": precision,
        "world_size": metrics.get("world_size"),
        "num_steps": metrics.get("num_steps"),
        "agent_id": env_map.get("FORGE_AGENT_ID"),
    }

    try:
        loop_dir = train_dir / loop_id
        loop_dir.mkdir(parents=True, exist_ok=True)
        history_path = loop_dir / HISTORY_FILENAME
        with history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
    except Exception as exc:  # best-effort
        print(f"mfu_record: persist failed: {exc}", file=sys.stderr)
        return None

    _emit_event(loop_id, entry, env_map)
    return entry


def _resolve_loop_id(repo_root: Path, env: dict[str, str]) -> str | None:
    cand = env.get("LOOP_ID") or env.get("FORGE_TRAIN_LOOP_ID")
    if cand:
        return cand
    # ``<forge_train_dir>/<loop_id>/workspace`` layout fallback.
    if repo_root.name == "workspace":
        return repo_root.parent.name
    return None


def _resolve_forge_train_dir(repo_root: Path, env: dict[str, str]) -> Path | None:
    cand = env.get("FORGE_TRAIN_DIR")
    if cand:
        return Path(cand)
    if repo_root.name == "workspace":
        return repo_root.parent.parent
    return None


def _precision_pass(metrics: dict[str, Any]) -> bool | None:
    """Return True/False if a precision-style field is present and
    populated, else ``None``. Bitwise suites use ``correctness_pass``;
    statistical suites use ``loss_pass``. ``None`` means the suite has
    no precision gate (e.g. perf-only).
    """
    if metrics.get("correctness_pass") is not None:
        return bool(metrics["correctness_pass"])
    if metrics.get("loss_pass") is not None:
        return bool(metrics["loss_pass"])
    return None


def _milestone_for(result: dict[str, Any]) -> str:
    cfg = (result.get("details") or {}).get("config") or {}
    ms = cfg.get("milestone")
    if isinstance(ms, str) and ms:
        return ms
    return str(result.get("suite") or "unknown")


def _emit_event(loop_id: str, entry: dict[str, Any], env: dict[str, str]) -> None:
    """Best-effort: shell out to the canonical ``loop_wrapper_event``
    writer so the SSE stream feeding the web UI sees a typed
    ``mfu_record`` event in real time.

    The writer lives in the original source tree, not the workspace
    copy — ``FORGE_SOURCE_ROOT`` points there. Without it (e.g. in
    unit tests) we silently skip the event; the persisted file is
    enough for the REST hydration path.
    """
    source_root = env.get("FORGE_SOURCE_ROOT")
    if not source_root:
        return
    writer = Path(source_root) / "harness" / "tools" / "loop_wrapper_event.py"
    if not writer.is_file():
        return
    payload = {
        "milestone": entry["milestone"],
        "suite": entry["suite"],
        "avg_mfu": entry["avg_mfu"],
        "mfu_target": entry["mfu_target"],
        "mfu_pass": entry["mfu_pass"],
        "precision_pass": entry["precision_pass"],
    }
    try:
        subprocess.run(
            [
                sys.executable,
                str(writer),
                "--loop-id",
                loop_id,
                "--subtype",
                "mfu_record",
                "--payload",
                json.dumps(payload),
            ],
            check=False,
            timeout=5,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:  # best-effort
        print(f"mfu_record: event emit failed: {exc}", file=sys.stderr)
