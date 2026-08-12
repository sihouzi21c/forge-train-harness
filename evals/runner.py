"""Single suite-side runner that the harness transport invokes as a subprocess.

The harness transport invokes this module as a console command:

    python3 -m evals.runner doctor <repo_root>
        reads ``{"workload_config": {...}}`` from stdin, emits a doctor result

    python3 -m evals.runner run <request.json> <repo_root>
        executes the run_request and emits its result

Results are written to ``<repo_root>/<artifact_relpath>/result.json`` and also
echoed to stdout wrapped by fixed BEGIN/END markers so the harness can locate
them unambiguously regardless of any framework warnings printed alongside.

The markers are the single wire protocol between harness and evals.
Changing them is a cross-layer contract change.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from evals import dispatcher
from harness import run_schema
from tools import mfu_record

__all__ = ["doctor", "main", "run_request"]

_RESULT_BEGIN = run_schema.RESULT_BEGIN
_RESULT_END = run_schema.RESULT_END


def doctor(workload_config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """Check local runtime (python, torch/cuda).

    Returns a structured ``{status, summary, checks}`` dict. The workload
    config is kept in the signature for future per-workload checks, though
    the current implementation does not consume it.
    """
    del workload_config, repo_root
    checks: list[dict[str, str]] = [
        {"name": "python", "status": "ready", "detail": sys.executable},
    ]

    try:
        import torch

        checks.append(
            {
                "name": "torch",
                "status": "ready",
                "detail": f"torch {torch.__version__}",
            }
        )
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            checks.append(
                {
                    "name": "cuda",
                    "status": "ready",
                    "detail": f"{props.name} ({props.total_memory // (1024**2)} MiB)",
                }
            )
        else:
            checks.append(
                {
                    "name": "cuda",
                    "status": "degraded",
                    "detail": "CUDA not available",
                }
            )
    except ImportError:
        checks.append({"name": "torch", "status": "failed", "detail": "torch not installed"})
        return {"status": "failed", "summary": "PyTorch not installed", "checks": checks}

    return {"status": "ready", "summary": "All checks passed", "checks": checks}


def run_request(request: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """Dispatch a run request into the evals orchestrator.

    Exceptions escaping ``evals.dispatcher.run_suite`` are translated into a structured
    failure result at this single boundary site (per Fail-Fast guideline
    "boundary layers may translate exceptions into structured failures");
    the caller reads ``status`` and optionally ``details.traceback``.
    """
    run_schema.validate_run_request(request)
    artifact_relpath = request["artifact_relpath"]
    artifact_dir = repo_root / artifact_relpath
    artifact_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = dispatcher.run_suite(request, repo_root, artifact_dir)
    except Exception as exc:  # boundary — see docstring
        result = run_schema.make_failed_result(
            suite=request.get("suite", "unknown"),
            summary=f"Runner error: {exc}",
            details={"traceback": traceback.format_exc()},
        )

    result = run_schema.validate_run_result(result)
    (artifact_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    # Best-effort MFU telemetry: persist to the per-loop history and the
    # cross-loop leaderboard. The helper returns ``None`` when the suite
    # did not measure MFU or the loop layout cannot be resolved (e.g.
    # ad-hoc evaluation outside an agent-loop run).
    mfu_record.record(result, repo_root)
    return result


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(f"\n{_RESULT_BEGIN}\n{json.dumps(payload)}\n{_RESULT_END}\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals.runner")
    sub = parser.add_subparsers(dest="action", required=True)

    doctor_p = sub.add_parser("doctor", help="Run remote doctor (reads JSON from stdin)")
    doctor_p.add_argument("repo_root", type=Path)

    run_p = sub.add_parser("run", help="Execute a run request")
    run_p.add_argument("request", type=Path)
    run_p.add_argument("repo_root", type=Path)

    args = parser.parse_args(argv)

    if args.action == "doctor":
        payload = json.loads(sys.stdin.read())
        result = doctor(payload["workload_config"], args.repo_root.resolve())
    else:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        try:
            result = run_request(request, args.repo_root.resolve())
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2

    _emit(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
