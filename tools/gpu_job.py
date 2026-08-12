"""Ephemeral GPU-job runner for ``[remote].kind = "job"``.

In the job kind the per-loop devspace is claimed with **0
GPUs** — a pure filesystem gateway that stays cheap and is never
auto-reclaimed at a GPU idle-deadline. Each GPU suite is instead
submitted as a short-lived ``cctl pytorchjob`` — the single submission
path for both topologies: ``[remote].nodes`` pods (default 1) each
requesting ``[remote].gpu_count`` GPUs, so ``world_size = nodes x
gpu_count``. At ``nodes = 1`` this degenerates to the exact single-pod
behaviour the retired ``cctl job`` (BATCH) path had. Multi-node pods get
the platform-injected ``RANK / WORLD_SIZE(=nodes) / MASTER_ADDR /
MASTER_PORT / GPUS_PER_NODE`` rendezvous env. Every pod runs ``harness
run <suite>`` on the **same user shared filesystem** where ``bin/harness
sync push`` already landed the code, writes its ``result.json`` /
``mfu_history.jsonl`` back to that shared volume, and exits — releasing
the GPUs immediately. This is what lifts the cluster GPU-utilisation
floor versus holding GPUs for the whole loop.

This module owns the job lifecycle only: build the create args, submit,
poll to a terminal phase under a **caller-supplied** wall-clock deadline,
dump logs, and map the terminal phase to a process exit code.

``tools/remote_run.sh`` is the sole caller and the SSOT for the
wall-clock budget: it reads ``harness budget <suite>``, layers the
documented buffers, and passes the result as ``--outer``. This module
MUST NOT compute or hardcode any suite budget — it only enforces the
deadline it is handed (mirrors the "budget lives in remote_run.sh" rule
the no-hardcoded-budgets lint pins).

The ``cctl pytorchjob get/logs/stop`` invocation surface is symmetric
with ``cctl job`` / ``cctl devspace`` (all bridge the v2 tasks API).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from tools import cctl_common, remote_workspace

__all__ = [
    "build_canonical_create_args",
    "build_create_args",
    "main",
    "poll_suite",
    "run_suite",
]

# Reserved suite name for the canonical-state preflight job. It is not a
# real ``harness run`` suite — it runs ``evals.canonical_preflight`` on GPUs
# — but it reuses the per-suite handle store, re-attach, and poll machinery,
# so it needs a stable handle key distinct from every real suite.
_CANONICAL_SUITE = "canonical-preflight"

# Poll cadence while waiting for the job to reach a terminal phase. This
# is a polling interval, NOT a suite budget (that is owned by
# remote_run.sh and passed in via --outer).
_POLL_INTERVAL_S = 30

# Multi-node only: how long a worker pod's init container waits for the
# master pod's DNS before giving up (--init-wait-seconds). Workers and the
# master queue for their GPUs independently, so one can be scheduled long
# before the other. Fixed constant for now (matches the validated 16-GPU
# 2-node miles run); promote to a [remote] key only when a real run needs
# a different window.
_INIT_WAIT_S = 3000

# Per-loop job-handle store (cross-round re-attach safety net). Lives
# beside ``mfu_history.jsonl`` at the loop root (``<loop_dir>/``, the
# parent of the per-loop config dir) — tool-written per-loop control
# state, NOT under the wrapper-owned agent-loop-state dir, and not inside
# the synced workspace (it tracks LOCAL cctl submissions).
_HANDLE_SUBDIR = "gpu_jobs"

_SUCCESS_PHASES = frozenset({"succeeded"})
# Terminal-but-failed = every terminal phase that is not a success.
_FAIL_PHASES = cctl_common.TERMINAL_PHASES - _SUCCESS_PHASES

# Process exit code used when the job overruns the caller's --outer
# window (mirrors GNU ``timeout``'s 124).
_TIMEOUT_RC = 124
# Submit refused because a non-consumed job for this (suite, commit) is
# already in flight — the caller must monitor it with ``--poll`` instead of
# resubmitting. Distinct from a real gate FAIL (rc 1) so the prompt can
# tell them apart.
_INFLIGHT_RC = 3
# ``--poll`` found no in-flight job recorded for this suite.
_NO_JOB_RC = 4


def _handle_path(suite: str) -> Path | None:
    """``<loop_dir>/gpu_jobs/<suite>.json`` — the cross-round job handle.

    ``loop_dir`` is the parent of ``FORGE_CONFIG_DIR`` (the per-loop config
    dir), i.e. the loop root that already holds ``mfu_history.jsonl``.
    Returns None when the anchor is unavailable — the handle is a
    best-effort safety net, never required for correctness.
    """
    cfg = os.environ.get("FORGE_CONFIG_DIR")
    if not cfg:
        return None
    return Path(cfg).parent / _HANDLE_SUBDIR / f"{suite}.json"


def _read_handle(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return rec if isinstance(rec, dict) else None


def _write_handle(path: Path | None, record: dict) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(record), encoding="utf-8")
        tmp.replace(path)  # atomic swap
    except OSError as exc:
        print(f"gpu_job: failed to persist job handle {path}: {exc}", file=sys.stderr)


def _workspace_commit() -> str:
    """HEAD of the LOCAL per-loop workspace git repo — the exact code state
    this suite verifies (``<loop_dir>/workspace``, sibling of the config
    dir). Empty string when unresolvable; an empty key only matches another
    empty-key submission, which is safe (it just means "re-attach disabled
    for this run" rather than a false match across different code).
    """
    cfg = os.environ.get("FORGE_CONFIG_DIR")
    if not cfg:
        return ""
    workspace = Path(cfg).parent / "workspace"
    try:
        proc = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _stop_job(job_id: str, *, reason: str) -> None:
    try:
        cctl_common.run_cctl(["pytorchjob", "stop", str(job_id), "--reason", reason])
    except cctl_common.CctlError as exc:
        print(f"gpu_job: stop failed for job {job_id}: {exc}", file=sys.stderr)


class GpuJobError(RuntimeError):
    """Misconfiguration or an unrecoverable job-submission failure."""


def _require(remote: dict, key: str) -> str:
    val = str(remote.get(key, "") or "").strip()
    if not val:
        raise GpuJobError(
            f'[remote].{key} is empty in remote.toml — required for a GPU job (kind = "job")'
        )
    return val


def _nodes(remote: dict) -> int:
    """``[remote].nodes`` — pod count for the pytorchjob (default 1).

    ``gpu_count`` stays per-node (cctl ``--gpu`` semantics), so
    ``world_size = nodes x gpu_count``. Anything that is not a positive
    integer is a config error, never coerced."""
    raw = remote.get("nodes", 1)
    if raw in (None, ""):
        return 1
    try:
        nodes = int(raw)
    except (TypeError, ValueError) as exc:
        raise GpuJobError(f"[remote].nodes must be a positive integer, got {raw!r}") from exc
    if nodes < 1:
        raise GpuJobError(f"[remote].nodes must be >= 1, got {nodes}")
    return nodes


def _load_remote(config_dir: str) -> dict:
    path = Path(config_dir) / "remote.toml"
    if not path.exists():
        raise GpuJobError(f"remote.toml not found under FORGE_CONFIG_DIR: {path}")
    with path.open("rb") as fh:
        cfg = tomllib.load(fh).get("remote", {})
    if not isinstance(cfg, dict):
        raise GpuJobError("[remote] must be a table in remote.toml")
    return cfg


def _resolve_workdir(remote: dict, loop_id: str) -> str:
    # The ephemeral GPU job runs in a DIFFERENT pod from the gateway
    # devspace that ``sync push`` wrote to, so the workdir MUST live on a
    # shared cluster path (e.g. /user/<username>) — the pod-local $HOME
    # (=/root on the release image) would be invisible to the job. Enforce
    # the shared-root invariant via the SSOT validator (fail fast, no
    # silent "$HOME" fallback).
    try:
        return remote_workspace.resolve_remote_workdir(remote, loop_id)
    except ValueError as exc:
        raise GpuJobError(f"job mode: {exc}") from exc


def _job_env_prefix(*, workdir: str, loop_id: str) -> str:
    """The env-setup preamble shared by every kind=job entry command.

    Mirrors remote_run.sh's remote_cmd (persistent Triton/Inductor caches
    + cd + exec) and remote-execution.md Step 2's env exports
    (LOOP_ID / FORGE_TRAIN_DIR / FORGE_CONFIG_DIR) so MFU attribution and
    the config SSOT resolve exactly as they do on a GPU devspace. ``$HOME``
    and ``$workdir`` (which may contain a literal ``$HOME``) expand on the
    node. The caller appends the actual ``python3 -m ...`` target after the
    trailing ``exec``.

    The target is always invoked as ``python3 -m <module>`` with ``cd
    <workdir>`` + ``PYTHONPATH=<workdir>`` (never a bare ``harness`` /
    ``bin/`` shim): the cctl job pod is a clean non-login shell that does
    NOT have ``<workdir>/bin`` on ``PATH`` (unlike a provisioned devspace's
    ssh shell), so ``-m`` + ``PYTHONPATH`` make the packages importable
    regardless of ``PATH``.
    """
    return (
        'mkdir -p "$HOME/.cache/triton-persistent" "$HOME/.cache/inductor-persistent" '
        '&& export TRITON_CACHE_DIR="$HOME/.cache/triton-persistent" '
        'TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/inductor-persistent" '
        f"&& export LOOP_ID={loop_id} "
        f"FORGE_TRAIN_DIR={workdir}/.artifacts/forge_train "
        f"FORGE_CONFIG_DIR={workdir}/config "
        f"PYTHONPATH={workdir}"
        "${PYTHONPATH:+:$PYTHONPATH} "
        f"&& cd {workdir} && exec "
    )


def _build_entry(*, workdir: str, loop_id: str, suite: str, extra: list[str]) -> str:
    """The single shell command a per-suite GPU job runs: ``harness run
    <suite>``. ``extra`` is appended raw, matching remote_run.sh's ``$*``."""
    extra_str = (" " + " ".join(extra)) if extra else ""
    return (
        _job_env_prefix(workdir=workdir, loop_id=loop_id)
        + f"python3 -m harness.cli run {suite}{extra_str}"
    )


def _build_canonical_entry(*, workdir: str, loop_id: str) -> str:
    """The canonical-state preflight command a GPU job runs (kind=job).

    Runs the SAME ``evals.canonical_preflight`` the ssh/devspace path runs
    on its GPU host — here inside an ephemeral cctl job because the job-kind
    gateway holds 0 GPUs. It materializes the missing forge_init_ones
    canonicals on the shared volume, then exits."""
    return (
        _job_env_prefix(workdir=workdir, loop_id=loop_id) + "python3 -m evals.canonical_preflight"
    )


def _create_args_for_entry(remote: dict, *, entry: str) -> list[str]:
    """``cctl pytorchjob create`` argv requesting ``nodes`` pods of
    ``gpu_count`` GPUs each, running ``entry``. Shared by the per-suite and
    canonical-preflight jobs — the only difference between them is the
    ``--entry`` command.

    Reuses the same project/cluster/resource_pool/image/gpu_model/priority
    /billing fields as the gateway-devspace claim — the GPU job is the
    compute a held (kind=devspace) box used to host, just made ephemeral.
    ``--nodes`` is always explicit; ``--init-wait-seconds`` (worker waits
    for master DNS) only exists for a real multi-node topology.
    """
    nodes = _nodes(remote)
    args = [
        "pytorchjob",
        "create",
        "-o",
        "json",
        "--project",
        _require(remote, "project"),
        "--cluster",
        _require(remote, "cluster"),
        "--resource-pool",
        _require(remote, "resource_pool"),
        "--image",
        _require(remote, "image"),
        "--gpu",
        _require(remote, "gpu_count"),
        "--gpu-model",
        _require(remote, "gpu_model"),
        "--nodes",
        str(nodes),
        "--priority",
        str(remote.get("priority", "NORMAL") or "NORMAL"),
    ]
    if nodes > 1:
        args += ["--init-wait-seconds", str(_INIT_WAIT_S)]
    billing = str(remote.get("billing_account_id", "") or "").strip()
    if billing:
        args += ["--billing-account-id", billing]
    # --entry carries the long, space-laden shell command; keep it last so
    # neither the real argv nor the dry-run print can be misread as
    # trailing flags belonging to ``harness run``.
    args += ["--entry", entry]
    return args


def build_create_args(
    remote: dict, *, suite: str, workdir: str, loop_id: str, extra: list[str]
) -> list[str]:
    """``cctl pytorchjob create`` argv for one ``harness run <suite>`` GPU job."""
    return _create_args_for_entry(
        remote, entry=_build_entry(workdir=workdir, loop_id=loop_id, suite=suite, extra=extra)
    )


def build_canonical_create_args(remote: dict, *, workdir: str, loop_id: str) -> list[str]:
    """``cctl pytorchjob create`` argv for the canonical-state preflight GPU job."""
    return _create_args_for_entry(
        remote, entry=_build_canonical_entry(workdir=workdir, loop_id=loop_id)
    )


def _job_phase(job_id: str) -> str:
    proc = cctl_common.run_cctl(["pytorchjob", "get", f"tasks/{job_id}", "-o", "json"])
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise GpuJobError(
            f"cctl pytorchjob get tasks/{job_id} returned non-JSON: {proc.stdout!r}"
        ) from exc
    return str(payload.get("status") or "").lower()


def _dump_logs(job_id: str) -> None:
    """Best-effort: print the job's pod logs (all pods, so multi-node
    worker output is captured too) so the round notes / wrapper stdout
    carry them even though the pods are ephemeral. Never raises — the
    suite's own ``.artifacts/runs/.../*.log`` on the shared volume is the
    authoritative record the agent tails via the 0-GPU devspace."""
    try:
        proc = cctl_common.run_cctl(["pytorchjob", "logs", str(job_id), "--pod", "all"])
    except cctl_common.CctlError as exc:
        print(f"gpu_job: log fetch failed for job {job_id}: {exc}", file=sys.stderr)
        return
    sys.stdout.write(proc.stdout)


def _poll_to_terminal(handle: Path | None, rec: dict, *, outer: int, poll_interval_s: float) -> int:
    """Poll ``rec['job_id']`` to a terminal phase; return an exit code.

    Shared by the submit path (:func:`run_suite`) and the explicit
    :func:`poll_suite`. ``outer`` (the exec wall-clock window) is anchored
    at the first ``running`` poll — recorded as wall-clock ``running_since``
    in the handle — so an unbounded Queued phase never counts against it AND
    the deadline keeps counting across separate ``--poll`` invocations
    (each of which may be cut short by the caller's Bash foreground
    timeout). On exec overrun the job is stopped; the handle is marked
    ``consumed`` only once a verdict is actually obtained.
    """
    job_id = str(rec["job_id"])
    rc = 1
    while True:
        phase = _job_phase(job_id)
        if phase in _SUCCESS_PHASES:
            rc = 0
            break
        if phase in _FAIL_PHASES:
            print(f"gpu_job: job tasks/{job_id} ended in phase {phase!r}", file=sys.stderr)
            rc = 1
            break
        # Non-terminal: still Queued (unbounded) or Running. Anchor the exec
        # budget at the first Running poll and log the queue latency.
        if phase in cctl_common.READY_PHASES and not rec.get("running_since"):
            rec["running_since"] = time.time()
            rec["status"] = "running"
            _write_handle(handle, rec)
            sub = rec.get("submitted_at")
            queued = (rec["running_since"] - sub) if isinstance(sub, (int, float)) else 0.0
            print(
                f"gpu_job: job tasks/{job_id} started running after "
                f"{queued:.0f}s queued; exec budget {outer}s",
                file=sys.stderr,
            )
        running_since = rec.get("running_since")
        if running_since and (time.time() - running_since) >= outer:
            print(
                f"gpu_job: job tasks/{job_id} exceeded the {outer}s exec "
                f"wall-clock window (counted from running); stopping it",
                file=sys.stderr,
            )
            _stop_job(job_id, reason="exec wall-clock budget exceeded")
            rc = _TIMEOUT_RC
            break
        time.sleep(poll_interval_s)

    rec["status"] = "consumed"
    _write_handle(handle, rec)
    _dump_logs(job_id)
    return rc


def run_suite(
    remote: dict,
    *,
    suite: str,
    workdir: str,
    loop_id: str,
    extra: list[str],
    outer: int,
    poll_interval_s: float = _POLL_INTERVAL_S,
    entry: str | None = None,
) -> int:
    """Submit the suite as a fresh GPU job and poll its first window.

    ``entry`` overrides the shell command the job runs (used by the
    canonical-state preflight, which runs ``evals.canonical_preflight``
    instead of ``harness run <suite>``); when None it is built from
    ``suite``/``extra``. ``suite`` is still the handle key either way, so
    re-attach / in-flight-dedup work identically.

    Submit is a **one-shot** event. If a non-consumed handle for this exact
    ``(suite, commit)`` already exists, this **refuses to submit a
    duplicate** (returns ``_INFLIGHT_RC``) and points the caller at
    ``--poll`` — there is deliberately no silent re-attach; monitoring an
    in-flight job is an explicit, separate operation (:func:`poll_suite`).
    A handle at a *different* commit means the code changed: its stale job
    is stopped and a fresh one submitted. The handle is persisted (status
    ``submitted``) before polling, so a later ``--poll`` can always find the
    job even if this process is killed at the Bash foreground timeout
    mid-poll.
    """
    handle = _handle_path(suite)
    commit = _workspace_commit()
    rec = _read_handle(handle)

    if rec and rec.get("status") != "consumed" and rec.get("job_id"):
        if rec.get("commit_sha") == commit:
            print(
                f"gpu_job: an in-flight job tasks/{rec['job_id']} already exists "
                f"for suite={suite} @ commit {commit[:8] or '?'} — NOT resubmitting. "
                f"Monitor it with `tools/remote_run.sh --poll {suite}`.",
                file=sys.stderr,
            )
            return _INFLIGHT_RC
        # Different commit → the code changed; stop the stale job first.
        _stop_job(str(rec["job_id"]), reason="superseded by new commit")

    if entry is not None:
        create_args = _create_args_for_entry(remote, entry=entry)
    else:
        create_args = build_create_args(
            remote, suite=suite, workdir=workdir, loop_id=loop_id, extra=extra
        )
    proc = cctl_common.run_cctl(create_args)
    job_id = cctl_common.task_id_from_create(proc.stdout)
    print(
        f"gpu_job: submitted suite={suite} as pytorchjob tasks/{job_id} "
        f"(nodes={_nodes(remote)}, gpu-per-node={remote.get('gpu_count')})",
        file=sys.stderr,
    )
    rec = {
        "suite": suite,
        "job_id": job_id,
        "commit_sha": commit,
        "submitted_at": time.time(),
        "running_since": None,
        "status": "submitted",
    }
    _write_handle(handle, rec)
    return _poll_to_terminal(handle, rec, outer=outer, poll_interval_s=poll_interval_s)


def poll_suite(*, suite: str, outer: int, poll_interval_s: float = _POLL_INTERVAL_S) -> int:
    """Poll the in-flight job recorded for ``suite`` to terminal — the
    explicit continuation of a submit whose foreground poll timed out.

    Reads the persisted handle and NEVER submits. Returns ``_NO_JOB_RC`` if
    there is no non-consumed handle to poll. If this process is killed at
    the Bash foreground timeout mid-poll, the handle stays non-consumed, so
    simply re-running ``--poll`` resumes where it left off.
    """
    handle = _handle_path(suite)
    rec = _read_handle(handle)
    if not rec or rec.get("status") == "consumed" or not rec.get("job_id"):
        print(f"gpu_job: no in-flight job for suite={suite} to poll", file=sys.stderr)
        return _NO_JOB_RC
    print(
        f"gpu_job: polling in-flight job tasks/{rec['job_id']} for suite={suite}",
        file=sys.stderr,
    )
    return _poll_to_terminal(handle, rec, outer=outer, poll_interval_s=poll_interval_s)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tools.gpu_job",
        description="Run one harness suite as an ephemeral cctl pytorchjob (kind=job).",
    )
    p.add_argument("suite", nargs="?", help="suite name (omit when --canonical)")
    p.add_argument(
        "--outer", type=int, required=True, help="wall-clock window (s), from remote_run.sh"
    )
    p.add_argument(
        "--budget", type=int, default=0, help="effective suite budget (s), informational"
    )
    p.add_argument(
        "--canonical",
        action="store_true",
        help="run the canonical-state preflight (evals.canonical_preflight) "
        "as the GPU job instead of a harness suite",
    )
    p.add_argument(
        "--poll",
        action="store_true",
        help="poll the in-flight job recorded for <suite> to terminal (no submit)",
    )
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    # parse_known_args (not a REMAINDER positional) so --outer/--budget are
    # parsed regardless of position while the suite's pass-through extras
    # (e.g. "attention", "--timeout 12") fall through to ``extra``.
    args, extra = _build_parser().parse_known_args(argv)
    extra = [a for a in extra if a != "--"]

    config_dir = os.environ.get("FORGE_CONFIG_DIR")
    loop_id = os.environ.get("LOOP_ID") or os.environ.get("LOOP_WEB_ID") or ""
    if not config_dir:
        print("gpu_job: FORGE_CONFIG_DIR required (run via remote_run.sh)", file=sys.stderr)
        return 2
    if not loop_id:
        print("gpu_job: LOOP_ID required (run via remote_run.sh)", file=sys.stderr)
        return 2

    # The canonical-state preflight (--canonical) runs evals.canonical_preflight
    # on GPUs instead of a harness suite, under a reserved handle key; every
    # real suite comes in as the positional. Resolve the handle key + whether
    # to override the entry once, up front.
    if args.canonical:
        suite = _CANONICAL_SUITE
        extra = []
    else:
        suite = args.suite
        if not suite:
            print("gpu_job: a suite name is required (or pass --canonical)", file=sys.stderr)
            return 2

    # --poll monitors an already-submitted job from the persisted handle;
    # it never touches remote.toml / submits, so it is resolved before the
    # submit-only setup below.
    if args.poll:
        return poll_suite(suite=suite, outer=args.outer)

    try:
        remote = _load_remote(config_dir)
        workdir = _resolve_workdir(remote, loop_id)
        if args.canonical:
            entry = _build_canonical_entry(workdir=workdir, loop_id=loop_id)
            create_args = build_canonical_create_args(remote, workdir=workdir, loop_id=loop_id)
        else:
            entry = None
            create_args = build_create_args(
                remote, suite=suite, workdir=workdir, loop_id=loop_id, extra=extra
            )
    except GpuJobError as exc:
        print(f"gpu_job: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        # One field per line so the unit test can parse without depending
        # on shell word-splitting of the embedded entry command.
        print("mode=job")
        print(f"suite={suite}")
        print(f"gpu={remote.get('gpu_count')}")
        print(f"nodes={_nodes(remote)}")
        print(f"gpu_model={remote.get('gpu_model')}")
        print(f"outer={args.outer}")
        print(f"budget={args.budget}")
        print(f"workdir={workdir}")
        # shlex.quote each token so the printed command is a faithful,
        # copy-pasteable rendering of the real argv (the --entry value,
        # which contains spaces, is shown single-quoted as one argument).
        print(
            "create_cmd=" + " ".join(shlex.quote(a) for a in [cctl_common.cctl_bin(), *create_args])
        )
        return 0

    try:
        return run_suite(
            remote,
            suite=suite,
            workdir=workdir,
            loop_id=loop_id,
            extra=extra,
            outer=args.outer,
            entry=entry,
        )
    except (GpuJobError, cctl_common.CctlError) as exc:
        print(f"gpu_job: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
