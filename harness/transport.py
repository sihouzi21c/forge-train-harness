from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from harness import config_runtime, run_schema

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "TargetConfig",
    "Transport",
    "TransportTimeoutError",
    "create_transport",
]


_RESULT_BEGIN = run_schema.RESULT_BEGIN
_RESULT_END = run_schema.RESULT_END

_RUNNER_MODULE = "evals.runner"
_GPU_VALUE_RE = re.compile(r"^[0-9,\-\s]+$")


class Transport(Protocol):
    def doctor(self, workload_config: dict[str, Any], repo_root: Path) -> dict[str, Any]: ...
    def run(
        self,
        request: dict[str, Any],
        repo_root: Path,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]: ...


@dataclass
class LocalTargetConfig:
    gpu: str | None = None
    env: dict[str, str] = field(default_factory=dict)


TargetConfig = LocalTargetConfig


def create_transport(
    harness_config: dict[str, Any],
    target: TargetConfig | None = None,
) -> Transport:
    return _LocalTransport(harness_config, target=target)


class _LocalTransport:
    """Workload execution on the local machine (direct subprocess)."""

    def __init__(
        self,
        harness_config: dict[str, Any],
        *,
        target: LocalTargetConfig | None = None,
    ) -> None:
        self._config = harness_config
        self._target = target or LocalTargetConfig()

    def doctor(
        self,
        workload_config: dict[str, Any],
        repo_root: Path,
    ) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        checks.append(_check_local_gpu())
        checks.append(_check_local_python())

        payload_json = json.dumps({"workload_config": workload_config})
        command, env = _build_bridge_invocation(
            repo_root=repo_root,
            action_args=["doctor"],
            gpu=self._target.gpu,
            env=self._target.env,
        )
        completed = _run(
            command,
            input_text=payload_json,
            check=False,
            cwd=repo_root,
            env=env,
        )

        if completed.returncode == 0:
            inner = _extract_bracketed_result(completed.stdout)
            if isinstance(inner, dict):
                checks.extend(inner.get("checks", []))
                inner_status = inner.get("status", "ready")
                # Aggregate the local pre-flight checks (GPU / Python
                # interpreter) into the top-level status so a failing
                # local check (e.g. Python < 3.11) is not masked by an
                # otherwise-passing inner doctor subprocess.
                local_failed = any(c.get("status") == "failed" for c in checks)
                status = "failed" if local_failed else inner_status
                summary = inner.get("summary", "Checks passed")
                if local_failed and status == "failed":
                    summary = "Local environment check failed; see checks for details."
                return {
                    "status": status,
                    "summary": summary,
                    "checks": checks,
                }

        stderr_tail = (completed.stderr or "").strip()[-500:]
        checks.append(
            {
                "name": "local_doctor",
                "status": "failed",
                "detail": (
                    f"Doctor script failed (rc={completed.returncode}). "
                    f"stderr: {stderr_tail or '(empty)'}"
                ),
            }
        )
        return {
            "status": "failed",
            "summary": (
                f"Doctor failed (rc={completed.returncode}). Run with --report json for details."
            ),
            "checks": checks,
        }

    def run(
        self,
        request: dict[str, Any],
        repo_root: Path,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        run_schema.validate_run_request(request)
        artifact_relpath = request["artifact_relpath"]
        artifact_dir = repo_root / artifact_relpath
        request_path = artifact_dir / "request.json"

        command, env = _build_bridge_invocation(
            repo_root=repo_root,
            action_args=["run", str(request_path)],
            gpu=self._target.gpu,
            env=self._target.env,
        )
        completed = _run(
            command,
            check=False,
            cwd=repo_root,
            env=env,
            timeout=timeout,
        )
        return _resolve_run_result(
            artifact_dir=artifact_dir,
            stdout_or_logs=completed.stdout,
            failure_prefix=(
                f"Local run failed (exit {completed.returncode}). "
                f"stderr: {completed.stderr.strip()}"
            )
            if completed.returncode != 0
            else None,
        )


# ------------------------------------------------------------------
# Bridge command builder
# ------------------------------------------------------------------


def _validate_gpu_value(gpu: str | None) -> str | None:
    if gpu:
        normalized = gpu.strip()
        if not normalized or _GPU_VALUE_RE.fullmatch(normalized) is None:
            raise ValueError("--gpu must be a CUDA_VISIBLE_DEVICES list such as '0' or '0,1'")
        return normalized
    return None


def _build_bridge_invocation(
    *,
    repo_root: Path,
    action_args: list[str],
    gpu: str | None,
    env: dict[str, str] | None,
) -> tuple[list[str], dict[str, str]]:
    validated_gpu = _validate_gpu_value(gpu)
    run_env = config_runtime.build_subprocess_env(
        repo_root=repo_root,
        extra=env,
        cuda_visible_devices=validated_gpu,
    )

    command = [
        sys.executable,
        "-m",
        _RUNNER_MODULE,
        *action_args,
        str(repo_root),
    ]
    return command, run_env


# ------------------------------------------------------------------
# Local environment checks
# ------------------------------------------------------------------


def _check_local_gpu() -> dict[str, Any]:
    result = _run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        check=False,
    )
    if result.returncode != 0:
        return {"name": "gpu", "status": "warning", "detail": "nvidia-smi not available"}
    gpus = result.stdout.strip().split("\n")
    return {
        "name": "gpu",
        "status": "ready",
        "detail": f"{len(gpus)} GPU(s): {gpus[0].strip()}",
    }


_MIN_PYTHON_VERSION: tuple[int, int] = (3, 11)


def _check_local_python() -> dict[str, str]:
    """Verify the Python interpreter satisfies ``pyproject.toml`` requires-python.

    The harness contract is ``requires-python = ">=3.11"`` and several
    leaf modules use 3.11-only stdlib (notably ``tomllib`` — see
    :mod:`harness._compat`). The previous implementation always returned
    ``status="ready"`` regardless of interpreter version, which made
    ``harness doctor`` give a false-positive on an under-spec interpreter
    until a much later import failure surfaced. This is now an explicit
    fail-fast check.
    """
    interpreter = sys.version.split()[0]
    detail = f"Python {interpreter}"
    if sys.version_info < _MIN_PYTHON_VERSION:
        required = ".".join(str(p) for p in _MIN_PYTHON_VERSION)
        return {
            "name": "python",
            "status": "failed",
            "detail": (
                f"{detail}: harness requires Python >= {required} "
                "(see pyproject.toml requires-python)."
            ),
        }
    return {"name": "python", "status": "ready", "detail": detail}


# ------------------------------------------------------------------
# Result parsing
# ------------------------------------------------------------------


def _resolve_run_result(
    *,
    artifact_dir: Path,
    stdout_or_logs: str,
    failure_prefix: str | None,
) -> dict[str, Any]:
    result_path = artifact_dir / "result.json"
    if result_path.exists():
        return _validate_run_result(json.loads(result_path.read_text(encoding="utf-8")))

    payload = _extract_bracketed_result(stdout_or_logs)
    if isinstance(payload, dict):
        payload = _validate_run_result(payload)
        config_runtime.write_json(result_path, payload)
        return payload

    if failure_prefix:
        raise RuntimeError(failure_prefix)
    raise RuntimeError(
        f"Run finished without result.json in {artifact_dir}. stdout: {stdout_or_logs[:500]}"
    )


def _extract_bracketed_result(text: str) -> dict[str, Any] | None:
    """Extract the JSON payload emitted by ``evals.runner``.

    The runner prints its result wrapped in fixed BEGIN/END markers so we
    can locate it deterministically even when stdout is mixed with
    framework warnings. If multiple blocks are present (shouldn't happen,
    but defend against it), we return the last one.
    """
    end_idx = text.rfind(_RESULT_END)
    if end_idx < 0:
        return None
    begin_idx = text.rfind(_RESULT_BEGIN, 0, end_idx)
    if begin_idx < 0:
        return None
    payload = text[begin_idx + len(_RESULT_BEGIN) : end_idx].strip()
    try:
        obj = json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _validate_run_result(payload: dict[str, Any]) -> dict[str, Any]:
    return run_schema.validate_run_result(payload)


# ------------------------------------------------------------------
# Subprocess runner
# ------------------------------------------------------------------


def _run(
    command: list[str],
    *,
    input_text: str | None = None,
    check: bool = True,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run *command* and capture stdout/stderr.

    When ``timeout`` is ``None``, behaves identically to ``subprocess.run``
    — the legacy path used by ``doctor`` / ``nvidia-smi`` callers that
    have no per-suite budget contract.

    When ``timeout`` is set (seconds), enforces a hard wall-clock kill:

    1. ``Popen(start_new_session=True)`` puts the child in its own
       process group so a SIGKILL on the group reaps any descendants
       it spawned (the L0 ref scripts fork sub-workers; without
       ``killpg`` they survive the parent SIGTERM and burn budget).
    2. ``wait(timeout)`` blocks until completion or expiry.
    3. On ``TimeoutExpired``: ``killpg(SIGTERM)`` → up to 30 s grace →
       ``killpg(SIGKILL)``. Raises :class:`TransportTimeoutError` with
       the underlying ``returncode`` (typically ``-SIGKILL``) and any
       captured stdout/stderr so the caller can stamp them into
       ``result.json`` for post-mortem.
    """
    if timeout is None:
        completed = subprocess.run(
            command,
            cwd=cwd,
            input=input_text,
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )
        if check and completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
            raise RuntimeError(
                f"Command failed ({completed.returncode}): {' '.join(command)}\n{detail}"
            )
        return completed

    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdin=subprocess.PIPE if input_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(input=input_text, timeout=timeout)
    except subprocess.TimeoutExpired:
        stdout, stderr = _kill_process_group(process, grace_seconds=30.0)
        raise TransportTimeoutError(
            command=command,
            timeout_s=timeout,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
        ) from None

    completed = subprocess.CompletedProcess(
        args=command,
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n{detail}"
        )
    return completed


def _kill_process_group(
    process: subprocess.Popen[str],
    *,
    grace_seconds: float,
) -> tuple[str, str]:
    """SIGTERM → grace → SIGKILL the process's session group.

    The Popen was started with ``start_new_session=True`` so the child's
    PID == its process-group ID. Sending SIGKILL to ``-pgid`` reaps any
    descendants (sub-workers, helper daemons) that would otherwise leak
    past the timeout. Returns whatever stdout/stderr was buffered.
    """
    pgid = process.pid
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGTERM)

    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        time.sleep(0.1)

    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGKILL)

    try:
        stdout, stderr = process.communicate(timeout=5.0)
    except subprocess.TimeoutExpired:
        stdout, stderr = "", ""
    return stdout or "", stderr or ""


class TransportTimeoutError(RuntimeError):
    """Raised when ``_run`` exceeds its wall-clock ``timeout`` and the
    process group has been reaped via SIGTERM → 30 s grace → SIGKILL.

    Carries the captured stdout/stderr buffers so the upper layer can
    surface them in ``result.json`` instead of losing them with the
    killed subprocess. ``returncode`` is the underlying ``Popen``'s
    final code — typically ``-signal.SIGKILL`` when the grace elapsed.
    """

    def __init__(
        self,
        *,
        command: list[str],
        timeout_s: float,
        returncode: int | None,
        stdout: str,
        stderr: str,
    ) -> None:
        super().__init__(
            f"transport budget exceeded after {timeout_s}s "
            f"(returncode={returncode}): {' '.join(command)}"
        )
        self.command = command
        self.timeout_s = timeout_s
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
