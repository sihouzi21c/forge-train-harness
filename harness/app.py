from __future__ import annotations

import hashlib
import os
import shutil
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from harness import config_runtime, resources, run_schema, transport, workspace_contract

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "budget_command",
    "doctor_command",
    "echo_config_command",
    "env_probe_command",
    "info_command",
    "resources_provision_command",
    "run_command",
]


# Env var honoured by ``app.run_command`` as a lower-priority alternative
# to the CLI ``--timeout`` flag. Centralised here so tests, the
# ``tools/remote_run.sh`` wrapper, and the agent prompt all quote the
# same literal name instead of redefining it per-site.
TIMEOUT_OVERRIDE_ENV = "HARNESS_RUN_TIMEOUT_S"

# Extra wall-clock headroom on top of the dispatcher-owned budget. The
# transport-layer kill is a defensive backstop for cases where the
# dispatcher itself wedges (e.g. between subprocess exit and result.json
# write); 60 s comfortably covers the longest natural teardown observed
# in ``evals.runner`` while still bounding a runaway dispatcher. Adjust
# only after observing a real teardown that exceeds it — bumping this
# silently masks bugs in the dispatcher's own kill path.
_TRANSPORT_BUDGET_BUFFER_S = 60


def info_command(
    *,
    config_path: str | None = None,
    json_output: bool = False,
    path_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    harness_config = config_runtime.load_harness_config()
    resolved_config_path, workload_config = config_runtime.load_workload_config(
        config_path,
        path_overrides=path_overrides,
    )
    # ``workload.id`` / ``display_name`` / ``requires_cuda`` are all
    # *required* keys — ``config_runtime.validate_workload_config`` (called
    # inside ``load_workload_config``) raises if any is missing, so a
    # ``.get(..., default)`` here would be a dead branch that could mask a
    # later contract regression. Fail-fast via direct subscripting.
    workload_meta = workload_config["workload"]
    suites = config_runtime.suite_metadata(workload_config)
    return {
        "command": "info",
        "status": "ready",
        "report": _report(harness_config, json_output),
        "paths": {
            "repo_root": str(config_runtime.repo_root()),
            "workload_config": str(resolved_config_path),
            "harness_config": str(config_runtime.harness_config_path()),
        },
        "payload": {
            "workload": {
                "id": workload_meta["id"],
                "display_name": workload_meta["display_name"],
                "requires_cuda": bool(workload_meta["requires_cuda"]),
                "supported_suites": sorted(suites.keys()),
                "suites_by_stage": config_runtime.suites_by_stage(
                    workload_config, include_local=True
                ),
                "suite_metadata": suites,
            },
            "contracts": {
                "run": run_schema.schema_metadata(),
                "suites": _suite_contracts(workload_config),
            },
        },
    }


def budget_command(
    *,
    suite: str,
    config_path: str | None = None,
    json_output: bool = False,
    path_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Print the SSOT wall-clock budget (seconds) for *suite*.

    Single-purpose query against ``config_runtime.suite_timeout_s``. The
    payload always reports the SSOT value — never reflects a per-run
    ``--timeout`` override. Override is a run-time decision; budget is a
    config-time derivation.
    """
    harness_config = config_runtime.load_harness_config()
    _, workload_config = config_runtime.load_workload_config(
        config_path,
        path_overrides=path_overrides,
    )
    timeout_s = config_runtime.suite_timeout_s(workload_config, suite, harness_config)
    source = (
        f"evals.{suite}.timeout_s"
        if suite in workload_config.get("evals", {})
        and "timeout_s" in workload_config["evals"][suite]
        else "defaults.default_timeout_s"
    )
    return {
        "command": "budget",
        "status": "ready",
        "report": _report(harness_config, json_output),
        "payload": {
            "suite": suite,
            "timeout_s": timeout_s,
            "source": source,
        },
    }


def doctor_command(
    *,
    config_path: str | None = None,
    json_output: bool = False,
    gpu: str | None = None,
    path_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    harness_config = config_runtime.load_harness_config()
    _, workload_config = config_runtime.load_workload_config(
        config_path,
        path_overrides=path_overrides,
    )
    report = _report(harness_config, json_output)

    resolved_gpu = gpu or config_runtime.runtime_defaults(harness_config).gpu
    target = transport.LocalTargetConfig(gpu=resolved_gpu)
    t = transport.create_transport(harness_config, target)
    payload = t.doctor(workload_config, config_runtime.repo_root())

    return {
        "command": "doctor",
        "status": payload["status"],
        "report": report,
        "targets": [
            {
                "target": "local",
                "status": payload["status"],
                "payload": payload,
            }
        ],
    }


def run_command(
    *,
    suite: str,
    suite_args: list[str] | None = None,
    config_path: str | None = None,
    json_output: bool = False,
    gpu: str | None = None,
    path_overrides: dict[str, str] | None = None,
    override_timeout_s: int | None = None,
) -> dict[str, Any]:
    harness_config = config_runtime.load_harness_config()
    report = _report(harness_config, json_output)
    resolved_gpu = gpu or config_runtime.runtime_defaults(harness_config).gpu
    resolved_args = list(suite_args or [])

    _, workload_config = config_runtime.load_workload_config(
        config_path,
        path_overrides=path_overrides,
    )
    supported_suites = sorted(workload_config.get("evals", {}).keys())
    local_suites = set(workload_config.get("local_suites", {}).keys())

    if suite not in supported_suites and suite not in local_suites:
        raise ValueError(
            f"Unsupported suite '{suite}'. "
            f"Supported: {', '.join(sorted(set(supported_suites) | local_suites))}"
        )
    # Merge `evals` and `local_suites` so the same args_min / args_max /
    # args_unbounded / args_usage metadata gates apply uniformly to
    # local suites; inspecting only `evals` would silently accept any
    # number of args for ``guard`` / ``unit``.
    _all_suites = {
        **workload_config.get("evals", {}),
        **workload_config.get("local_suites", {}),
    }
    _check_suite_arg_arity(suite, resolved_args, _all_suites)

    if suite in local_suites:
        if override_timeout_s is not None:
            # Local suites bypass the dispatcher / transport budget path
            # entirely (they run in-process via ``_LOCAL_SUITE_RUNNERS``),
            # so accepting `--timeout` here would silently no-op. Fail
            # fast instead — the user should remove the flag or pick a
            # remote suite.
            raise ValueError(
                f"--timeout is not supported for local suite '{suite}' "
                "(local suites do not use the budget pipeline)."
            )
        result = _run_local_suite(
            suite,
            config_runtime.repo_root(),
            workload_config.get("local_suites", {}).get(suite, {}),
        )
        return {
            "command": "run",
            "status": result["status"],
            "report": report,
            "targets": [
                {
                    "target": "local",
                    "status": result["status"],
                    "payload": result,
                }
            ],
        }

    result = _run_gpu_suite(
        suite=suite,
        suite_args=resolved_args,
        report=report,
        harness_config=harness_config,
        workload_config=workload_config,
        gpu=resolved_gpu,
        override_timeout_s=override_timeout_s,
    )
    return {
        "command": "run",
        "status": result["status"],
        "report": report,
        "targets": [
            {
                "target": "local",
                "status": result["status"],
                "artifact_dir": result.get("artifact_dir"),
                "payload": result["payload"],
            }
        ],
    }


def _local_cuda_available() -> bool:
    """Return True when an NVIDIA GPU is visible to this host.

    Probes ``nvidia-smi`` (returncode 0) rather than importing torch — the
    guard must run on a CPU-only Mac where torch may not even be installed,
    and the dispatch decision only needs "is there a usable GPU here".
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _assert_runnable_locally(workload_config: dict[str, Any], suite: str) -> None:
    """Fail fast when a GPU suite is launched locally on a remote-only host.

    When ``[remote].kind`` is ``ssh`` / ``devspace`` the GPU suites belong
    on the remote host, reached via ``tools/remote_run.sh``. Running them
    through the local transport on a CPU-only box would dispatch the suite
    and crash deep inside torch with an opaque CUDA error. We do NOT
    auto-SSH — local/remote separation is intentional (see
    prompt/develop_prompt/remote-execution.md) — so instead we raise an
    actionable redirect. The guard is a no-op when ``kind = "local"`` or
    when a local GPU is actually present (the remote host runs the same
    synced config but does have CUDA).
    """
    remote = workload_config.get("remote") or {}
    kind = str(remote.get("kind", "local") or "local")
    if kind == "local":
        return
    if _local_cuda_available():
        return
    raise RuntimeError(
        f"Suite '{suite}' requires a GPU but [remote].kind = '{kind}' and no "
        "local CUDA device was found (nvidia-smi unavailable). GPU suites must "
        "run on the remote host — launch it with "
        f"`tools/remote_run.sh {suite}` after `bin/harness sync push`, not "
        "`bin/harness run` on this machine. This harness never auto-SSHes; "
        "see prompt/develop_prompt/remote-execution.md."
    )


def _run_gpu_suite(
    *,
    suite: str,
    suite_args: list[str],
    report: str,
    harness_config: dict[str, Any],
    workload_config: dict[str, Any],
    gpu: str | None,
    override_timeout_s: int | None = None,
) -> dict[str, Any]:
    _assert_runnable_locally(workload_config, suite)
    ssot_timeout = config_runtime.suite_timeout_s(workload_config, suite, harness_config)
    timeout_decision = _resolve_effective_timeout(
        ssot_default=ssot_timeout,
        cli_override=override_timeout_s,
        env=os.environ,
    )

    request = _build_run_request(
        suite=suite,
        suite_args=suite_args,
        report=report,
        harness_config=harness_config,
        workload_config=workload_config,
        effective_timeout_s=timeout_decision["effective_timeout_s"],
    )
    artifact_dir = config_runtime.repo_root() / request["artifact_relpath"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    config_runtime.write_json(artifact_dir / "request.json", request)

    target = transport.LocalTargetConfig(gpu=gpu)
    t = transport.create_transport(harness_config, target)
    outer_timeout = float(timeout_decision["effective_timeout_s"] + _TRANSPORT_BUDGET_BUFFER_S)
    try:
        result = t.run(request, config_runtime.repo_root(), timeout=outer_timeout)
    except transport.TransportTimeoutError as exc:
        result = _synthesize_transport_killed_result(suite, exc)
        config_runtime.write_json(artifact_dir / "result.json", result)

    _stamp_timeout_classification(result, timeout_decision, artifact_dir)

    return {
        "status": result["status"],
        "artifact_dir": str(artifact_dir),
        "payload": result,
    }


def _synthesize_transport_killed_result(
    suite: str,
    exc: transport.TransportTimeoutError,
) -> dict[str, Any]:
    """Build a structured ``run_result`` for a transport-killed run.

    The transport's killpg path raises before the runner can write
    ``result.json``, so the upper layer would otherwise see a bare
    ``RuntimeError`` and report no provenance. This synthesizes the
    same shape the runner produces, with ``details.classification``
    pinned to ``transport_killed_by_budget`` and the captured stdout
    tail preserved for post-mortem.
    """
    stdout_tail = (exc.stdout or "")[-2000:]
    stderr_tail = (exc.stderr or "")[-2000:]
    return run_schema.make_failed_result(
        suite=suite,
        summary=(
            f"transport budget exceeded after {exc.timeout_s}s; "
            "process group killed by transport backstop."
        ),
        details={
            "classification": {
                "reason": "transport_killed_by_budget",
                "transport_timeout_s": exc.timeout_s,
                "subprocess_returncode": exc.returncode,
            },
            "output_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        },
    )


def _resolve_effective_timeout(
    *,
    ssot_default: int,
    cli_override: int | None,
    env: dict[str, str] | os._Environ[str],
) -> dict[str, Any]:
    """Single decision point for ``effective_timeout_s``.

    Priority: ``--timeout`` CLI > ``HARNESS_RUN_TIMEOUT_S`` env > SSOT
    (``config_runtime.suite_timeout_s``). Each override path validates
    its input fail-fast: zero, negative, or non-integer values raise
    ``ValueError`` so a stray ``HARNESS_RUN_TIMEOUT_S=abc`` in a shell
    rc file cannot silently fall back to the default and mask itself.

    Returns a dict with ``effective_timeout_s`` / ``timeout_source``
    (``"override"`` | ``"env"`` | ``"config"``) / ``timeout_origin_argv``
    so callers can stamp the metadata into result.json for audit.
    """
    if cli_override is not None:
        if not isinstance(cli_override, int) or isinstance(cli_override, bool):
            raise ValueError("--timeout must be an integer number of seconds")
        if cli_override <= 0:
            raise ValueError(f"--timeout must be a positive integer, got {cli_override}")
        return {
            "effective_timeout_s": cli_override,
            "timeout_source": "override",
            "timeout_origin_argv": f"--timeout {cli_override}",
        }

    env_raw = env.get(TIMEOUT_OVERRIDE_ENV)
    if env_raw is not None and env_raw != "":
        try:
            env_value = int(env_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{TIMEOUT_OVERRIDE_ENV} must be an integer, got {env_raw!r}") from exc
        if env_value <= 0:
            raise ValueError(f"{TIMEOUT_OVERRIDE_ENV} must be a positive integer, got {env_value}")
        return {
            "effective_timeout_s": env_value,
            "timeout_source": "env",
            "timeout_origin_argv": f"{TIMEOUT_OVERRIDE_ENV}={env_value}",
        }

    return {
        "effective_timeout_s": int(ssot_default),
        "timeout_source": "config",
        "timeout_origin_argv": None,
    }


def _stamp_timeout_classification(
    result: dict[str, Any],
    decision: dict[str, Any],
    artifact_dir: Path,
) -> None:
    """Record the effective-timeout decision into ``result.details``.

    Mutates ``result`` in place AND rewrites ``result.json`` on disk so
    a post-mortem reader sees the same provenance the live caller sees.
    The classification block lives under ``details.classification`` and
    is additive — pre-existing classification fields (e.g. a future
    transport-kill reason) are preserved.
    """
    details = dict(result.get("details") or {})
    classification = dict(details.get("classification") or {})
    classification["timeout_source"] = decision["timeout_source"]
    classification["effective_timeout_s"] = decision["effective_timeout_s"]
    if decision["timeout_origin_argv"] is not None:
        classification["timeout_origin_argv"] = decision["timeout_origin_argv"]
    details["classification"] = classification
    result["details"] = details
    run_schema.validate_run_result(result)
    config_runtime.write_json(artifact_dir / "result.json", result)


def _run_guard_suite(repo_root: Path) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "harness.framework_guard", "--with-path-guard"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    passed = result.returncode == 0
    return run_schema.make_run_result(
        status="passed" if passed else "failed",
        suite="guard",
        summary="Framework guard passed." if passed else "Framework guard found violations.",
        metrics={"violations": 0 if passed else 1},
        details={"stderr": result.stderr.strip()},
    )


def _run_anti_proxy_suite(repo_root: Path) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "harness.anti_proxy_guard"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    passed = result.returncode == 0
    return run_schema.make_run_result(
        status="passed" if passed else "failed",
        suite="anti-proxy",
        summary=("Anti-proxy guard passed." if passed else "Anti-proxy guard found violations."),
        metrics={"violations": 0 if passed else 1},
        details={"stderr": result.stderr.strip()},
    )


def _run_unit_suite(repo_root: Path) -> dict[str, Any]:
    env = config_runtime.build_subprocess_env(
        repo_root=repo_root,
        remove_workload_src=True,
    )
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "harness/tests", "-v"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    passed = result.returncode == 0
    output = result.stderr.strip()
    run_line = next(
        (line for line in output.split("\n") if line.startswith("Ran ")),
        "",
    )
    return run_schema.make_run_result(
        status="passed" if passed else "failed",
        suite="unit",
        summary=run_line or ("Unit tests passed." if passed else "Unit tests failed."),
        details={"output": output},
    )


_LOCAL_SUITE_RUNNERS = {
    "guard": _run_guard_suite,
    "unit": _run_unit_suite,
    "anti-proxy": _run_anti_proxy_suite,
}


def _run_local_suite(
    suite: str,
    repo_root: Path,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runner_kind = str((cfg or {}).get("runner_kind", suite))
    try:
        runner = _LOCAL_SUITE_RUNNERS[runner_kind]
    except KeyError as exc:
        raise ValueError(f"Unknown local suite runner_kind: {runner_kind}") from exc
    result = runner(repo_root)
    if result["suite"] != suite:
        result = {**result, "suite": suite}
        return run_schema.validate_run_result(result)
    return result


def _report(harness_config: dict[str, Any], json_output: bool) -> str:
    if json_output:
        return "json"
    return config_runtime.runtime_defaults(harness_config).report


def _metrics_by_runner_kind() -> dict[str, list[str]]:
    return run_schema.RUNNER_METRICS


def _suite_contracts(workload_config: dict[str, Any]) -> dict[str, Any]:
    contracts: dict[str, Any] = {}
    for suite, cfg in config_runtime.suite_metadata(workload_config).items():
        args = {
            "usage": cfg.get("args_usage", ""),
            "min": int(cfg.get("args_min", 0)),
            "unbounded": bool(cfg.get("args_unbounded", False)),
        }
        if "args_max" in cfg:
            args["max"] = int(cfg["args_max"])
        runner_kind = str(cfg.get("runner_kind", suite))
        artifacts = [] if cfg.get("local") else run_schema.schema_metadata()["artifact_files"]
        contracts[suite] = {
            "runner_kind": runner_kind,
            "args": args,
            "env_inputs": list(cfg.get("env_inputs", [])),
            "artifacts": artifacts,
            "metrics": _metrics_by_runner_kind().get(runner_kind, []),
        }
    return contracts


def _check_suite_arg_arity(
    suite: str,
    args: list[str],
    suites: dict[str, Any],
) -> None:
    """Runtime check on ``harness run <suite> <args...>`` arg count.

    This is the *runtime* counterpart of
    :func:`harness.config_runtime._validate_suite_args_metadata` —
    config_runtime validates the **schema** of the args metadata (types,
    args_min ≤ args_max, unbounded XOR max), this function enforces the
    declared arity against the user's actual CLI invocation. Naming them
    apart prevents the two-look-alike-validators trap that an earlier
    audit flagged.
    """
    cfg = suites.get(suite, {})
    if not isinstance(cfg, dict):
        return
    args_min = int(cfg.get("args_min", 0))
    if len(args) < args_min:
        usage = cfg.get("args_usage", "")
        suffix = f" Usage: harness run {suite} {usage}".rstrip() if usage else ""
        raise ValueError(f"Suite {suite} expects at least {args_min} arg(s).{suffix}")
    if cfg.get("args_unbounded"):
        return
    if "args_max" not in cfg:
        return
    args_max = int(cfg["args_max"])
    if len(args) > args_max:
        usage = cfg.get("args_usage", "")
        suffix = f" Usage: harness run {suite} {usage}".rstrip() if usage else ""
        raise ValueError(f"Suite {suite} expects at most {args_max} arg(s).{suffix}")


def _build_run_request(
    *,
    suite: str,
    suite_args: list[str] | None = None,
    report: str,
    harness_config: dict[str, Any],
    workload_config: dict[str, Any],
    effective_timeout_s: int | None = None,
) -> dict[str, Any]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")  # noqa: UP017
    run_id = f"{suite}-{stamp}-{uuid.uuid4().hex[:6]}"
    runs_root = config_runtime.artifact_subtree(harness_config, "runs")
    # Sliding-window retention: before the new run dir is allocated,
    # rmtree the oldest entries of this suite until at most ``keep - 1``
    # remain (the new dir takes the last slot). ``run_id`` embeds a UTC
    # stamp so lexicographic order == chronological order.
    _prune_old_runs(runs_root, suite, harness_config)
    artifact_dir = runs_root / run_id
    snapshot = _workload_config_snapshot(workload_config, suite)
    if effective_timeout_s is not None and suite in snapshot.get("evals", {}):
        # Inject the resolved effective_timeout into the runner-bound
        # snapshot so dispatcher's ``cfg.get("timeout_s", ...)`` sees the
        # override (CLI / env) instead of the SSOT default. This is the
        # single seam between the app-layer decision and the dispatcher
        # — no other site is allowed to overwrite timeout_s.
        snapshot["evals"][suite] = {
            **snapshot["evals"][suite],
            "timeout_s": int(effective_timeout_s),
        }
    request = {
        "schema_version": run_schema.RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "suite": suite,
        "args": list(suite_args or []),
        "report": report,
        "target": "local",
        "gpu_count": None,
        "artifact_relpath": artifact_dir.relative_to(config_runtime.repo_root()).as_posix(),
        "requested_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "requested_from": socket.gethostname(),
        "workload_config": snapshot,
    }
    # Content-addressed sidecar: write the canonical bytes of *snapshot*
    # to ``<artifact_dir>/workload_config.json`` and embed the path +
    # sha into the request. The dispatcher/runner can keep reading
    # ``request["workload_config"]`` for now (no consumer churn this
    # commit), but the on-disk file is the contractual SSOT going
    # forward — any cross-process consumer can re-verify the parent's
    # view via ``config_runtime.load_workload_config_from_file``.
    config_path, config_sha = config_runtime.dump_workload_config(
        snapshot,
        artifact_dir / "workload_config.json",
    )
    request["workload_config_path"] = config_path.relative_to(config_runtime.repo_root()).as_posix()
    request["workload_config_sha"] = config_sha
    return run_schema.validate_run_request(request)


def _prune_old_runs(
    runs_root: Path,
    suite: str,
    harness_config: dict[str, Any],
) -> None:
    """Sliding-window retention for ``.artifacts/runs/<suite>-*``.

    Reads ``[artifacts.retention].per_suite`` from ``harness_config``;
    ``0`` (or missing/malformed) disables pruning. When the cap is
    ``keep``, this leaves at most ``keep - 1`` existing run dirs of
    *suite* on disk so the about-to-be-created run lands as the
    ``keep``-th entry. ``rmtree`` errors are swallowed silently — the
    new run must not fail to start just because an old dir is locked.
    """
    keep_raw = harness_config.get("artifacts", {}).get("retention", {}).get("per_suite", 0)
    try:
        keep = int(keep_raw)
    except (TypeError, ValueError):
        return
    if keep <= 0 or not runs_root.exists():
        return

    prefix = f"{suite}-"
    existing = sorted(
        (p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith(prefix)),
        key=lambda p: p.name,
    )
    excess = len(existing) - (keep - 1)
    if excess <= 0:
        return
    for stale in existing[:excess]:
        shutil.rmtree(stale, ignore_errors=True)


def _workload_config_snapshot(workload_config: dict[str, Any], suite: str) -> dict[str, Any]:
    """Build the runner-bound serialization of the validated workload config.

    Includes EVERY axis the validator emits (workload, ref, env, runtime,
    stage1, stage2, model, optim, …) so the child sees the same shape
    the parent sees. The only intentional shrink is ``[evals]`` →
    ``{suite: cfg}`` (the child only ever needs the one suite it's
    running) and dropping ``[local_suites]`` + ``[automation]`` (harness-
    only surfaces — local suites bypass the runner entirely; automation
    is consumed by ``agent-loop.sh`` and Stage 2 fan-out helpers, never
    by a gate runner).

    The previous implementation cherry-picked a whitelist of axes
    (``workload`` / ``ref`` / ``env`` / ``runtime`` / ``stage2``),
    silently dropping any axis added later (``[optim]`` and ``[model]``
    were both lost this way). Downstream FORGE_* env-var injection
    masked the drop until a single unset variable forced a hardcoded
    fallback, costing ~60 min/recurrence in agent debug time. The fix
    is to enumerate ONLY what we want to exclude, never to enumerate
    what we want to include.
    """
    excluded = {"local_suites", "automation", "evals"}
    snapshot: dict[str, Any] = {
        key: value for key, value in workload_config.items() if key not in excluded
    }
    evals_cfg = workload_config.get("evals", {})
    snapshot["evals"] = {suite: evals_cfg[suite]} if suite in evals_cfg else {}
    return snapshot


def env_probe_command() -> int:
    """Run workspace-contract invariants; return shell exit code.

    Exit 0 = every invariant passed. Exit 2 = a contract violation was
    detected; the violation message (with executable fix instructions)
    is written to stderr. The dedicated exit code lets agent-loop.sh /
    web `_provision_workspace` distinguish infra failure from a regular
    CLI usage error (which exits 1).

    Layer-DAG note: this function is the single seam between the CLI
    surface (``harness.cli``, which is restricted from importing
    ``harness.workspace_contract`` directly) and the contract module.
    """
    try:
        workspace_contract.run_all(config_runtime.repo_root())
    except workspace_contract.WorkspaceContractError as exc:
        sys.stderr.write(f"WORKSPACE CONTRACT VIOLATION: {exc}\n")
        return 2
    sys.stderr.write("env-probe: workspace contract invariants passed\n")
    return 0


def resources_provision_command() -> int:
    """Provision every entry in the workspace's external resource manifest.

    Symlinks declared sources to ``<workspace>/.resources/...`` after
    verifying the declared content hash. Exit 0 on success (or no-op
    when the manifest is empty); exit 2 when any entry fails to
    provision — the failure enumerates every attempted source so the
    operator knows exactly what to fix or populate.

    Layer-DAG note: same seam pattern as :func:`env_probe_command` —
    the CLI calls here instead of importing :mod:`harness.resources`
    directly, keeping ``harness.cli`` a thin wrapper.
    """
    try:
        provisioned = resources.provision(config_runtime.repo_root())
    except (resources.ResourceProvisionError, ValueError) as exc:
        sys.stderr.write(f"RESOURCE PROVISION FAILED: {exc}\n")
        return 2
    if not provisioned:
        sys.stderr.write("resources: manifest is empty (no entries declared)\n")
    else:
        sys.stderr.write(
            f"resources: provisioned {len(provisioned)} entr(y/ies): "
            f"{[r.name for r in provisioned]}\n"
        )
    return 0


def echo_config_command(
    *,
    config_path: str | None = None,
    path_overrides: dict[str, str] | None = None,
) -> int:
    """Emit the canonical workload-config bytes to stdout + sha to stderr.

    Used by the ``subprocess_config_isomorphic`` workspace-contract
    invariant: the in-process loader produces some bytes; this
    subprocess produces some bytes; if they differ, the workspace has a
    PYTHONPATH / cwd / env divergence that would silently corrupt every
    downstream run.

    Stdout carries the raw canonical bytes (so the parent can hash
    without re-deserialising). Stderr carries the sha hex and the
    resolved config path for human triage. Exits 0 on success, 1 on
    any load/validation error (the message goes to stderr).
    """
    try:
        resolved_path, workload_config = config_runtime.load_workload_config(
            config_path,
            path_overrides=path_overrides,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        sys.stderr.write(f"echo-config: load failed: {exc}\n")
        return 1
    payload = config_runtime.canonical_workload_config_bytes(workload_config)
    digest = hashlib.sha256(payload).hexdigest()
    sys.stdout.buffer.write(payload)
    sys.stdout.flush()
    sys.stderr.write(f"echo-config: sha256={digest} source={resolved_path}\n")
    return 0


def prefetch_command() -> int:
    """Stage the production long-train corpus into the executing host's data dir.

    Workspace-native seam for the background corpus prefetch: wherever
    ``bin/harness prefetch`` runs — locally, or ``ssh … 'cd $WORKDIR &&
    bin/harness prefetch'`` on the remote execution host — the bytes land
    local to that host, symmetric with ``bin/harness run``. ``agent-loop.sh``
    backgrounds this at loop start when ``[data].prefetch_target_gb > 0``;
    the production-train runner blocks on the sentinel before its first segment.

    Network: clusters behind a whitelist egress proxy (e.g. shandong) allow
    the HF mirror for metadata but DROP the Xet/LFS CDN where the parquet
    bytes live (cas-bridge.xethub.hf.co). Unsetting the proxy reaches both
    the mirror and the CDN directly, and ``--curl`` follows the
    resolve→CDN redirect where the hf_hub httpx/Xet client stalls.
    ``HF_ENDPOINT`` selects the mirror (default hf-mirror.com; a
    preexisting value is honoured). This proxy/endpoint handling used to
    live inline in ``agent-loop.sh``; owning it here keeps both the local
    and remote call sites trivial. The mutation is reverted before
    returning so the in-process CLI dispatch never leaks it into the
    parent. Same dedicated exit-code bypass as :func:`env_probe_command`:
    the return value is ``tools.prefetch_data.main``'s code verbatim, no
    render payload.
    """
    from tools import prefetch_data

    proxy_keys = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")
    saved = {k: os.environ.get(k) for k in (*proxy_keys, "HF_ENDPOINT")}
    try:
        for k in proxy_keys:
            os.environ.pop(k, None)
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        return prefetch_data.main(["--repo-root", str(config_runtime.repo_root()), "--curl"])
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
