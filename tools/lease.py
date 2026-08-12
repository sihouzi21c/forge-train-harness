"""Devspace lease registry — exclusive ``cctl devspace`` allocator.

Two loops launching in the same second used to inherit the same
``[remote].hostname`` from the shared ``harness/config/remote.toml``;
both dev agents then sed-substituted the same ``@@REMOTE_SSH_HOST@@``
into their prompts and physically contended for the same GPU.

This module is the fix: every loop's ``agent-loop.sh`` calls
``claim("devspace", spec=..., loop_id=...)`` during bootstrap, which
``cctl devspace create``s a fresh devspace from the loop's configured
spec (project / cluster / resource-pool / GPU / image / priority), gets
back a server-assigned task id, derives the exclusive host
``ds-<new-id>``, resolves the new node's real Teleport FQDN via
``tsh ls`` and synthesizes a reachable ``~/.ssh/config`` stanza, then the
wrapper rewrites the loop's frozen ``remote.toml`` hostname before
chmod-ing the per-loop config dir read-only. The EXIT trap calls
``release("devspace", loop_id=...)`` to ``cctl devspace stop`` the copy.

Storage:
    .artifacts/lease/devspace/
        .registry.lock     ← fcntl.flock guard for atomic registry writes
        registry.json      ← {host: {loop_id, task_id, claimed_at}}
        <loop_id>.host     ← reverse-lookup file for ``release``

The whole module shells out to ``cctl`` (env-var ``CCTL_BIN`` overrides
the binary, defaulting to ``cctl`` on $PATH) and ``tsh`` (env-var
``TSH_BIN``, default ``tsh``) — tests inject fakes for both.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import fcntl
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from tools import cctl_common

__all__ = [
    "DevspaceSpec",
    "LeaseBusyError",
    "LeaseError",
    "LeaseTimeoutError",
    "claim",
    "main",
    "rebind",
    "release",
]

_REGISTRY_FILENAME = "registry.json"
_LOCK_FILENAME = ".registry.lock"
_FLOCK_TIMEOUT_S = 60
_POLL_INTERVAL_S = 10
# Wall-clock cap on a single ``cctl`` shell-out (SSOT in cctl_common):
# bounds the EXIT-trap ``release`` path so a hung Teleport tunnel behind
# ``cctl devspace stop`` cannot block the wrapper indefinitely.
_CCTL_CALL_TIMEOUT_S = cctl_common.CCTL_CALL_TIMEOUT_S
# Post-stanza SSH reachability probe. ``tsh ls`` listing the node name does
# NOT mean its reverse tunnel carries traffic yet, so ``claim`` proves a real
# SSH round-trip before returning (see ``_probe_ssh_ready``). Cadence + the
# per-attempt subprocess cap; the overall budget reuses ``claim``'s timeout.
_SSH_PROBE_INTERVAL_S = 8
_SSH_PROBE_CONNECT_TIMEOUT_S = 20

# Teleport proxy the devspace SSH aliases route through. Every managed
# stanza shares one ProxyCommand template; only the per-host FQDN differs.
_TELEPORT_PROXY = "teleport.cybertron.modelbest.co:443"
_TELEPORT_CLUSTER = "teleport.cybertron.modelbest.co"


@dataclass(frozen=True)
class DevspaceSpec:
    """Resources for ``cctl devspace create``. Mirrors the configurable
    ``[remote]`` fields of ``config/remote/devspace.toml`` one-to-one.
    """

    project: str
    cluster: str
    resource_pool: str
    image: str
    gpu_count: int
    gpu_model: str
    priority: str = "NORMAL"
    billing_account_id: str = ""

    def to_create_args(self) -> list[str]:
        args = [
            "devspace",
            "create",
            "-o",
            "json",
            "--project",
            self.project,
            "--cluster",
            self.cluster,
            "--resource-pool",
            self.resource_pool,
            "--image",
            self.image,
            "--gpu",
            str(self.gpu_count),
            "--gpu-model",
            self.gpu_model,
            "--priority",
            self.priority,
        ]
        # Server enforces BILLING_ACCOUNT_REQUIRED — only emit when set so
        # clusters that don't require it stay unaffected.
        if self.billing_account_id:
            args += ["--billing-account-id", self.billing_account_id]
        return args


class LeaseError(RuntimeError):
    """``cctl`` returned non-zero or another subprocess failure."""


class LeaseTimeoutError(LeaseError):
    """Polling for ``Ready`` exceeded the configured timeout."""


class LeaseBusyError(LeaseError):
    """``fcntl.flock`` could not be acquired within ``_FLOCK_TIMEOUT_S``."""


# ──────────────────────────────────────────────────────────────────────
# Path helpers
# ──────────────────────────────────────────────────────────────────────


def _repo_root() -> Path:
    # Late import so tests that flip ``FORGE_REPO_ROOT`` see the new value.
    from harness import config_runtime

    return config_runtime.repo_root()


def _registry_base() -> Path:
    """Root that holds the shared ``.artifacts/lease/`` registry.

    The registry's whole purpose is to stop two loops from claiming the
    same devspace, so it must be ONE shared file — not a per-loop copy.
    ``claim`` runs at launch, while ``release`` (EXIT trap) and ``rebind``
    (dev agent) run later from the per-loop workspace cwd with a different
    ``FORGE_REPO_ROOT``; anchoring to ``repo_root()`` would split them
    across distinct files and ``release`` would never find what ``claim``
    wrote, leaking the devspace. ``FORGE_SOURCE_ROOT`` is the single root
    every loop shares — honor it first, falling back to ``repo_root()``
    for standalone CLI / unit-test invocations that do not set it.
    """
    source_root = os.environ.get("FORGE_SOURCE_ROOT")
    if source_root:
        return Path(source_root)
    return _repo_root()


def _lease_root(resource: str) -> Path:
    root = _registry_base() / ".artifacts" / "lease" / resource
    root.mkdir(parents=True, exist_ok=True)
    return root


def _registry_path(resource: str) -> Path:
    return _lease_root(resource) / _REGISTRY_FILENAME


def _lock_path(resource: str) -> Path:
    return _lease_root(resource) / _LOCK_FILENAME


def _reverse_path(resource: str, loop_id: str) -> Path:
    return _lease_root(resource) / f"{loop_id}.host"


def _cctl_bin() -> str:
    return cctl_common.cctl_bin()


def _tsh_bin() -> str:
    return os.environ.get("TSH_BIN", "tsh")


def _ssh_bin() -> str:
    # Overridable (like CCTL_BIN / TSH_BIN) so the post-claim SSH reachability
    # probe can be driven by a fake binary in tests — no real network.
    return os.environ.get("SSH_BIN", "ssh")


# ──────────────────────────────────────────────────────────────────────
# Locking
# ──────────────────────────────────────────────────────────────────────


class _RegistryLock:
    def __init__(self, resource: str, *, timeout: int = _FLOCK_TIMEOUT_S) -> None:
        self._path = _lock_path(resource)
        self._timeout = timeout
        self._fd: int | None = None

    def __enter__(self) -> _RegistryLock:
        fd = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o644)
        deadline = time.monotonic() + self._timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._fd = fd
                return self
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    os.close(fd)
                    raise LeaseBusyError(
                        f"Could not acquire {self._path} within {self._timeout}s"
                    ) from None
                time.sleep(0.2)

    def __exit__(self, *_exc) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None


def _read_registry(resource: str) -> dict[str, dict[str, str]]:
    path = _registry_path(resource)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LeaseError(f"Corrupt lease registry at {path}: {exc}") from exc


def _write_registry(resource: str, payload: dict[str, dict[str, str]]) -> None:
    path = _registry_path(resource)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


# ──────────────────────────────────────────────────────────────────────
# cctl shell-out
# ──────────────────────────────────────────────────────────────────────


def _run_cctl(args: list[str]) -> subprocess.CompletedProcess[str]:
    # Delegate to the shared primitive; translate its error into the
    # lease-specific class so callers' ``except LeaseError`` (and the
    # EXIT-trap ``contextlib.suppress(LeaseError)``) keep working.
    try:
        return cctl_common.run_cctl(args, timeout=_CCTL_CALL_TIMEOUT_S)
    except cctl_common.CctlError as exc:
        raise LeaseError(str(exc)) from exc


def _run_cctl_retry_not_found(
    args: list[str], *, tries: int, sleep_s: float = _POLL_INTERVAL_S
) -> subprocess.CompletedProcess[str]:
    """_run_cctl with bounded retries on the server's transient ``not_found``.

    The cybertron API intermittently answers ``not_found`` ("Resource not
    found: ") for a resource that exists — observed both on ``devspace
    create`` (a server-side lookup flake; a FAILED create makes no task, so
    re-issuing is safe) and on the first ``devspace get`` right after a
    successful create (read-after-write lag; the task shows up seconds
    later). Only that error signature is retried — anything else re-raises
    immediately.
    """
    last: LeaseError | None = None
    for attempt in range(1, tries + 1):
        try:
            return _run_cctl(args)
        except LeaseError as exc:
            if '"code": "not_found"' not in str(exc).replace("\\n", "\n").replace(
                '\\"', '"'
            ):
                raise
            last = exc
            if attempt < tries:
                time.sleep(sleep_s)
    assert last is not None
    raise last


_READY_PHASES = cctl_common.READY_PHASES
_TERMINAL_PHASES = cctl_common.TERMINAL_PHASES


def _task_id_from_create(stdout: str) -> str:
    """Extract the server-assigned task id from ``devspace create`` JSON."""
    try:
        return cctl_common.task_id_from_create(stdout)
    except cctl_common.CctlError as exc:
        raise LeaseError(str(exc)) from exc


def _devspace_phase(task_id: str) -> str:
    """Return the lower-cased flat ``status`` string for a devspace.

    Retries the transient server-side ``not_found`` — observed on the first
    ``get`` right after a successful ``create`` (read-after-write lag: the
    same id resolves seconds later)."""
    proc = _run_cctl_retry_not_found(
        ["devspace", "get", f"tasks/{task_id}", "-o", "json"], tries=6
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise LeaseError(
            f"cctl devspace get tasks/{task_id} returned non-JSON: {proc.stdout!r}"
        ) from exc
    return str(payload.get("status") or "").lower()


def _poll_ready(task_id: str, *, timeout: int) -> None:
    # timeout <= 0 => wait indefinitely. A busy cluster can keep a multi-GPU
    # devspace Queued for a long time; callers that would rather wait than fail
    # the whole loop pass 0 (meta_harness default).
    infinite = timeout <= 0
    deadline = time.monotonic() + timeout
    while True:
        phase = _devspace_phase(task_id)
        if phase in _READY_PHASES:
            return
        if phase in _TERMINAL_PHASES:
            raise LeaseError(
                f"devspace tasks/{task_id} entered terminal phase {phase!r} before becoming ready"
            )
        if not infinite and time.monotonic() >= deadline:
            raise LeaseTimeoutError(
                f"devspace tasks/{task_id} did not become ready within {timeout}s "
                f"(last phase: {phase!r})"
            )
        time.sleep(_POLL_INTERVAL_S)


# ──────────────────────────────────────────────────────────────────────
# SSH config
# ──────────────────────────────────────────────────────────────────────


def _ssh_config_path() -> Path:
    return Path(os.environ.get("HOME", str(Path.home()))) / ".ssh" / "config"


def _teleport_username() -> str:
    """Resolve the logged-in Teleport username via ``tsh status --format json``.

    The synthesized SSH stanza must point ``IdentityFile`` /
    ``CertificateFile`` at the per-user key material under
    ``~/.tsh/keys/<cluster>/<user>``; the username is not derivable from
    the node name alone (the project segment can collide), so read it
    from the authoritative ``tsh status`` profile.
    """
    proc = subprocess.run(
        [_tsh_bin(), "status", "--format", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise LeaseError(
            f"`tsh status` failed (rc={proc.returncode}); cannot resolve the "
            f"Teleport username for the SSH stanza. stderr: {proc.stderr!r}"
        )
    try:
        status = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise LeaseError(f"tsh status returned non-JSON: {proc.stdout!r}") from exc
    username = (status.get("active") or {}).get("username", "")
    if not username:
        raise LeaseError(f"tsh status carried no active.username: {proc.stdout!r}")
    return str(username)


def _teleport_node_name(task_id: str, *, timeout: int) -> str:
    """Resolve a freshly-created devspace's real Teleport node name.

    ``cctl devspace create`` has no base SSH stanza to clone, so the
    reachable hostname must be discovered. ``tsh ls --format json`` lists
    every Teleport node; the devspace registers as
    ``devspace-<user>-<project>-<task_id>`` (the id is the trailing
    segment). Registration into the Teleport tunnel can lag ``cctl``'s
    ``Running`` status, so poll until the node appears or ``timeout``.
    """
    infinite = timeout <= 0  # see _poll_ready: 0 => wait indefinitely
    deadline = time.monotonic() + timeout
    suffix = f"-{task_id}"
    while True:
        proc = subprocess.run(
            [_tsh_bin(), "ls", "--format", "json"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            try:
                nodes = json.loads(proc.stdout or "[]")
            except json.JSONDecodeError as exc:
                raise LeaseError(f"tsh ls returned non-JSON: {proc.stdout!r}") from exc
            for node in nodes:
                hostname = (node.get("spec") or {}).get("hostname", "")
                if hostname.endswith(suffix):
                    return hostname
        if not infinite and time.monotonic() >= deadline:
            raise LeaseTimeoutError(
                f"devspace task {task_id} never appeared in `tsh ls` within {timeout}s "
                f"(node name ending {suffix!r})"
            )
        time.sleep(_POLL_INTERVAL_S)


def _write_synthesized_stanza(host: str, teleport_node: str) -> None:
    """Write a reachable ``~/.ssh/config`` stanza for ``host`` from the
    discovered Teleport node name, using the fixed Teleport
    ``ProxyCommand`` template. Idempotent: a no-op when ``host`` already
    has a stanza.
    """
    cfg = _ssh_config_path()
    cfg.parent.mkdir(parents=True, exist_ok=True)
    existing = cfg.read_text(encoding="utf-8") if cfg.exists() else ""
    needle = f"Host {host}"
    if any(line.strip() == needle for line in existing.splitlines()):
        return
    fqdn = f"{teleport_node}.{_TELEPORT_CLUSTER}"
    # Teleport's SSH cert auth needs the per-user key material and the
    # SSH-listener port (3022); a stanza with only HostName+ProxyCommand
    # authenticates as the wrong identity and the remote rejects it with
    # "Permission denied (publickey)". The key paths follow Teleport's
    # fixed on-disk layout under ~/.tsh/keys/<cluster>/<user>.
    user = _teleport_username()
    tsh_dir = Path(os.environ.get("HOME", str(Path.home()))) / ".tsh"
    known_hosts = tsh_dir / "known_hosts"
    identity_file = tsh_dir / "keys" / _TELEPORT_CLUSTER / user
    cert_file = (
        tsh_dir / "keys" / _TELEPORT_CLUSTER / f"{user}-ssh" / f"{_TELEPORT_CLUSTER}-cert.pub"
    )
    stanza = (
        f"Host {host}\n"
        f"  HostName {fqdn}\n"
        "  User root\n"
        "  Port 3022\n"
        "  ConnectTimeout 15\n"
        "  ServerAliveInterval 30\n"
        "  ServerAliveCountMax 4\n"
        "  StrictHostKeyChecking accept-new\n"
        f'  UserKnownHostsFile "{known_hosts}"\n'
        f'  IdentityFile "{identity_file}"\n'
        f'  CertificateFile "{cert_file}"\n'
        f'  ProxyCommand "tsh" proxy ssh --cluster={_TELEPORT_CLUSTER} '
        f"--proxy={_TELEPORT_PROXY} %r@%h:%p\n"
    )
    with cfg.open("a", encoding="utf-8") as fh:
        fh.write("\n# autoskill-lease: managed devspace alias\n" + stanza)


def _probe_ssh_ready(host: str, *, timeout: int) -> None:
    """Block until an actual SSH round-trip to ``host`` succeeds.

    ``_teleport_node_name`` only proves the node NAME is registered in
    ``tsh ls``; the reverse tunnel can stay unconnected for a short window
    after that. The first thing a loop does with the claimed host is the
    canonical-preflight ``harness.cli sync push`` (rsync over the Teleport
    ProxyCommand) — issued milliseconds after ``claim`` returns, it raced the
    tunnel and died with ``no node reverse tunnel found ... agent is offline``,
    aborting the whole loop (exit 2). Poll a lightweight ``ssh <host> true``
    through the freshly-synthesized stanza until the tunnel genuinely carries
    traffic, so ``claim`` never hands back a host that cannot yet be reached.
    Bounded by the caller's ``timeout`` (``<= 0`` waits indefinitely, matching
    the other poll helpers); raises ``LeaseTimeoutError`` on exhaustion.
    """
    infinite = timeout <= 0
    deadline = time.monotonic() + timeout
    last_err = ""
    per_attempt = _SSH_PROBE_CONNECT_TIMEOUT_S + 15
    while True:
        try:
            proc = subprocess.run(
                [
                    _ssh_bin(),
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    f"ConnectTimeout={_SSH_PROBE_CONNECT_TIMEOUT_S}",
                    host,
                    "true",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=per_attempt,
            )
            if proc.returncode == 0:
                return
            last_err = (proc.stderr or proc.stdout or "").strip().replace("\n", " ")
        except subprocess.TimeoutExpired:
            last_err = f"ssh probe exceeded {per_attempt}s (ProxyCommand hang)"
        if not infinite and time.monotonic() >= deadline:
            raise LeaseTimeoutError(
                f"devspace {host} SSH tunnel not reachable within {timeout}s; "
                f"last ssh error: {last_err!r}"
            )
        time.sleep(_SSH_PROBE_INTERVAL_S)


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────


def claim(resource: str, *, spec: DevspaceSpec, loop_id: str, timeout: int = 600) -> str:
    """Create an exclusive devspace for ``loop_id`` from ``spec``.

    ``spec`` carries the ``cctl devspace create`` resources. The new task
    is server-named, so the derived host is ``ds-<new-id>`` where
    ``new-id`` is the task id ``cctl`` assigns. The real Teleport node name
    is discovered via ``tsh ls`` and turned into a reachable SSH stanza.
    Returns the derived hostname. Raises ``LeaseError`` /
    ``LeaseTimeoutError`` / ``LeaseBusyError`` on failure.
    """
    if resource != "devspace":
        raise LeaseError(f"Unknown lease resource: {resource!r}")
    if not loop_id:
        raise LeaseError("loop_id must be non-empty")

    # Idempotent retry: if this loop already created a still-live devspace,
    # reuse it instead of spawning a second one. The stanza is already on
    # disk from the original claim; nothing to re-synthesize.
    rev = _reverse_path(resource, loop_id)
    if rev.exists():
        existing_host = rev.read_text(encoding="utf-8").strip()
        existing_id = existing_host[len("ds-") :] if existing_host.startswith("ds-") else ""
        if existing_id and _devspace_phase(existing_id) not in _TERMINAL_PHASES:
            return existing_host

    # Step 1: cctl create. Slow (minutes) — done outside the registry lock
    # so other claimers can read/write the registry while we wait. cctl
    # assigns the new task id; we cannot choose it. The server intermittently
    # rejects a well-formed create with a transient ``not_found`` (a lookup
    # flake — the identical command succeeds seconds later); a create that
    # errors this way makes no task, so bounded re-issue is safe.
    proc = _run_cctl_retry_not_found(spec.to_create_args(), tries=3)
    new_id = _task_id_from_create(proc.stdout)
    host = f"ds-{new_id}"

    # Step 2: poll until the devspace is ready.
    _poll_ready(new_id, timeout=timeout)

    # Step 3: discover the Teleport node name and synthesize a reachable
    # SSH stanza. Tunnel registration can lag `Running`, so this polls.
    teleport_node = _teleport_node_name(new_id, timeout=timeout)
    _write_synthesized_stanza(host, teleport_node)

    # Step 3.5: the node NAME being in `tsh ls` does not mean its reverse
    # tunnel carries traffic yet. Prove a real SSH round-trip before returning
    # so the loop's first `sync push` rsync cannot race an unready tunnel
    # ("no node reverse tunnel found" → loop abort). See _probe_ssh_ready.
    _probe_ssh_ready(host, timeout=timeout)

    # Step 4: registry write under flock.
    with _RegistryLock(resource):
        registry = _read_registry(resource)
        registry[host] = {
            "loop_id": loop_id,
            "task_id": new_id,
            "claimed_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        }
        _write_registry(resource, registry)
        rev.write_text(host + "\n", encoding="utf-8")

    return host


def release(resource: str, *, loop_id: str) -> None:
    """Stop ``loop_id``'s leased devspace copy and drop its registry row.

    Best-effort: missing reverse-lookup file or already-stopped devspace
    returns silently.
    """
    if resource != "devspace":
        raise LeaseError(f"Unknown lease resource: {resource!r}")

    rev = _reverse_path(resource, loop_id)
    if not rev.exists():
        return
    host = rev.read_text(encoding="utf-8").strip()

    # Drop registry row first so a concurrent claim with the same loop_id
    # cannot see a stale row. Capture the task id before the row is gone.
    with _RegistryLock(resource):
        registry = _read_registry(resource)
        row = registry.pop(host, None)
        _write_registry(resource, registry)
        with contextlib.suppress(FileNotFoundError):
            rev.unlink()

    task_id = (row or {}).get("task_id")
    if not task_id and host.startswith("ds-"):
        task_id = host[len("ds-") :]

    # Stop outside the lock — cctl call can be slow. Best-effort: the
    # registry row is already gone; logging is left to the wrapper / caller.
    # ``--yes`` is required since cctl ≥ 0.0.15 gates ``devspace stop`` on
    # an interactive confirmation prompt; without it the call fails with
    # ``confirmation_required`` and the suppress below silently leaks the
    # box (registry cleared but cctl-side machine keeps running).
    if task_id:
        with contextlib.suppress(LeaseError):
            _run_cctl(["devspace", "stop", "--yes", f"tasks/{task_id}"])


def rebind(resource: str, *, loop_id: str, host: str) -> None:
    """Re-point ``loop_id``'s lease at a recovered devspace ``host``.

    When a leased devspace drops mid-run, the dev agent re-creates it with
    ``cctl devspace create`` and rewrites ``[remote].hostname``. Without
    this call the lease registry + reverse-lookup file still name the
    *dead* original, so the wrapper's EXIT-trap ``release`` would stop the
    wrong (already-gone) machine and leak the live recovered copy.
    ``rebind`` swaps the registry row + reverse-lookup file to ``host`` so
    ``release`` targets the machine actually in use.

    It does NOT stop the old machine: recovery is triggered precisely
    because the original already dropped (preemption / reclaim / restart),
    so a ``cctl devspace stop`` on it would only add a failing call.
    """
    if resource != "devspace":
        raise LeaseError(f"Unknown lease resource: {resource!r}")
    if not loop_id:
        raise LeaseError("loop_id must be non-empty")
    if not host:
        raise LeaseError("host must be non-empty")

    new_id = host[len("ds-") :] if host.startswith("ds-") else host
    rev = _reverse_path(resource, loop_id)

    with _RegistryLock(resource):
        registry = _read_registry(resource)
        old_host = rev.read_text(encoding="utf-8").strip() if rev.exists() else None
        if old_host is not None:
            registry.pop(old_host, None)
        registry[host] = {
            "loop_id": loop_id,
            "task_id": new_id,
            "claimed_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        }
        _write_registry(resource, registry)
        rev.write_text(host + "\n", encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness.tools.lease",
        description="Allocate/release exclusive cctl devspace leases per loop.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    claim_p = sub.add_parser("claim", help="Claim a lease")
    claim_p.add_argument("resource", choices=["devspace"])
    claim_p.add_argument("--loop-id", required=True, dest="loop_id")
    claim_p.add_argument("--project", required=True)
    claim_p.add_argument("--cluster", required=True)
    claim_p.add_argument("--resource-pool", required=True, dest="resource_pool")
    claim_p.add_argument("--image", required=True)
    claim_p.add_argument("--gpu", required=True, type=int, dest="gpu_count")
    claim_p.add_argument("--gpu-model", required=True, dest="gpu_model")
    claim_p.add_argument("--priority", default="NORMAL")
    claim_p.add_argument("--billing-account-id", default="", dest="billing_account_id")
    claim_p.add_argument("--timeout", type=int, default=600)

    release_p = sub.add_parser("release", help="Release a lease")
    release_p.add_argument("resource", choices=["devspace"])
    release_p.add_argument("--loop-id", required=True, dest="loop_id")

    rebind_p = sub.add_parser("rebind", help="Re-point a lease at a recovered host")
    rebind_p.add_argument("resource", choices=["devspace"])
    rebind_p.add_argument("--loop-id", required=True, dest="loop_id")
    rebind_p.add_argument("--host", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "claim":
            host = claim(
                args.resource,
                spec=DevspaceSpec(
                    project=args.project,
                    cluster=args.cluster,
                    resource_pool=args.resource_pool,
                    image=args.image,
                    gpu_count=args.gpu_count,
                    gpu_model=args.gpu_model,
                    priority=args.priority,
                    billing_account_id=args.billing_account_id,
                ),
                loop_id=args.loop_id,
                timeout=args.timeout,
            )
            sys.stdout.write(host + "\n")
            return 0
        if args.command == "release":
            release(args.resource, loop_id=args.loop_id)
            return 0
        if args.command == "rebind":
            rebind(args.resource, loop_id=args.loop_id, host=args.host)
            return 0
    except LeaseError as exc:
        sys.stderr.write(f"lease {args.command} failed: {exc}\n")
        return 2
    parser.error(f"Unknown command: {args.command}")
    return 2  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
