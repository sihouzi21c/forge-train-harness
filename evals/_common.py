"""Shared helpers for all evaluation suites (Stage 1 / Stage 2).

This module is the single source of truth for cross-cutting concerns:
  - subprocess environment construction
  - ref-script-as-gate execution bridge
  - stdout wire-format parsers ([LOSS], [LOSS_REF], [LOSS_RES])
  - torchrun command builder
  - missing-script result template
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
from typing import TYPE_CHECKING, Any

from harness import config_runtime
from harness.capture_artifacts import (
    CaptureArtifacts,
    candidate_capture_paths,
    ref_capture_paths,
)
from harness.wire_format import (
    parse_key_values,
    parse_loss_lines,
    parse_loss_lines_to_dict,
)

# Module-level (not symbol-level) imports for the ref-script runner so
# that ``mock.patch("tools.ref_script_runner.<name>", ...)`` in tests
# affects what ``run_via_ref_script`` actually calls. The static
# dependency edge ``evals._common → tools.ref_script_runner`` is still
# visible to the layer-DAG linter via this import.
from tools import ref_script_runner as _ref_script_runner

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def output_tail_limit() -> int:
    """Current value of ``[defaults].output_tail_limit`` (re-read each call).

    Reading from the harness defaults SSOT at every call instead of
    snapshotting at module-import time lets tests (and
    ``config/ref.toml``) override the value without re-importing
    every consumer module.
    """
    return config_runtime.default_output_tail_limit()


__all__ = [
    "CaptureArtifacts",
    "base_env",
    "build_suite_env",
    "candidate_capture_paths",
    "classify_ref_failure",
    "distributed_env",
    "missing_script_result",
    "missing_window_steps",
    "output_tail_limit",
    "parse_key_values",
    "parse_loss_lines",
    "parse_loss_lines_to_dict",
    "ref_capture_paths",
    "run_streaming_subprocess",
    "run_via_ref_script",
    "stage2_runtime_inputs",
    "suite_process_env",
    "torchrun_cmd",
    "window_loss_diff_metrics",
]


# ======================================================================
# Environment construction
# ======================================================================


def base_env(repo_root: Path) -> dict[str, str]:
    """Base environment for subprocess execution on the GPU host."""
    return config_runtime.build_subprocess_env(
        repo_root=repo_root,
        prepend_workload_src=True,
    )


def build_suite_env(
    repo_root: Path,
    cfg: dict[str, Any],
    workload_config: dict[str, Any],
    *,
    extra: dict[str, str] | None = None,
    env_inputs: tuple[str, ...] | list[str] | None = None,
) -> dict[str, str]:
    """Build subprocess env from suite config + workload-level [env].

    Used by Stage 1 suites (alignment–long-horizon) which pass most config via env vars.
    Reads CHECKPOINT_ROOT and MEGATRON_ROOT from cfg with workload [env]
    fallback; merges workload [env] as defaults; applies caller extras last.

    ``env_inputs`` declares which suite-config keys are LIFTED from
    ``cfg`` / workload ``[env]`` into the subprocess environment. It
    is purely a schema-lifting declaration — callers may also pass
    additional ``extra`` env vars (e.g. ``FORGE_NSYS_RANK0_OUTPUT``)
    that are forwarded without needing to appear in the declaration.
    """
    effective_cfg = {**workload_config.get("ref", {}), **cfg}
    # Resolve tokenizer + data assets (download from HF on cache miss).
    # Mutates os.environ so the data_conf.sh source step below sees
    # FORGE_DATA_DIR via its local_env= reference.
    config_runtime.resolve_assets(
        workload_config.get("ref", {}),
        workload_config.get("data", {}),
    )
    declared_inputs = {str(raw_key).upper() for raw_key in env_inputs or ()}
    # DATA_PATH is still resolved from the data_conf at run time. DATA_LOADER is
    # NOT transported via env at all: the loader kind lives in config/data.toml
    # `[data].data_loader` and every side reads it from that file via
    # FORGE_DATA_TOML (ref/sisters --data-config, ours runtime_config.data_loader()).
    # It is never a gate env_input nor a rendered product value.
    needs_path = "DATA_PATH" in declared_inputs and not effective_cfg.get("data_path")
    if needs_path and effective_cfg.get("data_conf"):
        resolved_env = config_runtime._resolve_data_env_from_conf(str(effective_cfg["data_conf"]))
        effective_cfg["data_path"] = resolved_env["data_path"]
    extra_env = {str(k).upper(): str(v) for k, v in (extra or {}).items()}

    env = base_env(repo_root)
    for k, v in workload_config.get("env", {}).items():
        env[k.upper()] = str(v)
    empty_deployment_paths: list[str] = []
    for key in declared_inputs:
        cfg_key = key.lower()
        if cfg_key in effective_cfg:
            value = str(effective_cfg[cfg_key])
            if not value and cfg_key in config_runtime.DEPLOYMENT_PATH_KEYS:
                empty_deployment_paths.append(cfg_key)
                continue
            env[key] = value
        elif key in env or key in extra_env:
            continue
        else:
            raise ValueError(
                f"Declared env input {key} has no source in suite/default config, "
                "workload [env], or runtime extra env"
            )
    if empty_deployment_paths:
        raise config_runtime.DeploymentPathError(empty_deployment_paths)
    env.update(extra_env)
    return env


_MASTER_PORT_AUTO_SENTINELS = frozenset({"", "auto"})


def _allocate_free_port() -> int:
    """Ask the kernel for a currently-free TCP port on localhost.

    The bind / getsockname / close idiom returns a port in the OS
    ephemeral range. There is a small TOCTOU window between releasing
    the socket and torch.distributed's TCPStore re-binding it, but on
    a single-host devspace the next `bind` typically lands on the
    same port the kernel just handed out, and even on the rare race
    the failure mode is identical to today's fixed-29500 collision —
    only stochastic instead of deterministic.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _resolve_master_port(value: str) -> str:
    """Resolve a master_port string, expanding the ``"auto"`` sentinel.

    A literal ``"auto"`` (or the empty string) means "pick a fresh
    free ephemeral port now". Any other value is treated as an
    already-validated numeric port and returned unchanged.
    """
    if value in _MASTER_PORT_AUTO_SENTINELS:
        return str(_allocate_free_port())
    return str(value)


def distributed_env(
    world_size: int,
    *,
    master_addr: str | None = None,
    master_port: str | None = None,
) -> dict[str, str]:
    """Return the standard single-node distributed env contract.

    When *master_addr* / *master_port* are ``None``, values are resolved
    from ``runtime_env_defaults()`` (→ dense_training.toml SSOT).

    The sentinel ``master_port == "auto"`` (or the empty string) resolves
    to a freshly-allocated free ephemeral port at call time. This avoids
    the TIME_WAIT / co-tenant collisions that a fixed default like
    ``29500`` produces on shared devspaces and on rapid retries after a
    SIGKILL'd suite. The port is allocated once per call; downstream
    rank children (``launch_dp.py`` env-copies, ref bash scripts read
    ``${MASTER_PORT}``) all see the same value.
    """
    if master_addr is None or master_port is None:
        defaults = config_runtime.runtime_env_defaults()
        master_addr = master_addr or defaults.master_addr
        master_port = master_port or defaults.master_port
    return {
        "NUM_PROCS": str(world_size),
        "MASTER_ADDR": master_addr,
        "MASTER_PORT": _resolve_master_port(master_port),
    }


def suite_process_env(
    repo_root: Path,
    cfg: dict[str, Any],
    workload_config: dict[str, Any],
    *,
    world_size: int | None = None,
    master_port: str | None = None,
    extra: dict[str, str] | None = None,
    suite_key: str | None = None,
) -> dict[str, str]:
    """Build the env for a suite subprocess from one centralized entrypoint."""
    merged_extra: dict[str, str] = {}
    # Point the ours entry at its rendered product (the sole source for the
    # gate's model/optim/muP/MTP shape, read via _gate_entry.GateInputs).
    # ``extra`` (merged last) still wins, so a caller can override these.
    if suite_key:
        merged_extra["FORGE_GATE"] = suite_key
        merged_extra["FORGE_OURS_CONFIG_DIR"] = str(repo_root / "workload" / "src" / "config")
    # Data-value SSOT pointer: the ours engine reads [data].data_loader from
    # config/data.toml via FORGE_DATA_TOML (runtime_config.data_loader()), the
    # same axis the ref launcher reads — no DATA_LOADER env transport, no
    # per-gate [env] copy. Symmetric with the ref paths (run_via_ref_script /
    # run_ref_capture) that set this pointer too.
    merged_extra["FORGE_DATA_TOML"] = str(config_runtime._data_config_path())
    if world_size is not None:
        defaults = config_runtime.runtime_env_defaults(workload_config)
        merged_extra.update(
            distributed_env(
                world_size,
                master_addr=defaults.master_addr,
                master_port=master_port or defaults.master_port,
            )
        )
    # forge_init_ones CHECKPOINT_ROOT redirect: retired. The ours product now
    # carries the freeze-filled repo-relative CHECKPOINT_ROOT, and the runner
    # derives the ones/ or no1/ subdir itself via GateInputs.checkpoint_root()
    # — the value is consumer-derived, not dispatcher-injected.
    if extra:
        merged_extra.update(extra)
    env_inputs = cfg.get("env_inputs")
    return build_suite_env(
        repo_root,
        cfg,
        workload_config,
        extra=merged_extra,
        env_inputs=env_inputs if isinstance(env_inputs, list) else [],
    )


def stage2_runtime_inputs(workload_config: dict[str, Any]) -> dict[str, str]:
    """Return only Stage 2 process-boundary inputs declared in workload config.

    Merges ``[ref]`` then ``[stage2]`` so shared keys like
    ``checkpoint_root`` / ``megatron_root`` need only appear in
    ``[ref]`` when the value is the same across stages.
    """
    ref = workload_config.get("ref", {})
    stage2 = workload_config.get("stage2", {})
    if not isinstance(stage2, dict):
        raise ValueError("config/eval.toml [stage2] must be a table")
    merged = {**(ref if isinstance(ref, dict) else {}), **stage2}
    inputs: dict[str, str] = {}
    for key in ("checkpoint_root", "megatron_root", "data_path"):
        if key not in merged:
            raise ValueError(f"config/eval.toml [ref]/[stage2].{key} is required")
        inputs[key.upper()] = str(merged[key])
    return inputs


# ======================================================================
# Ref-as-gate SSOT bridge - see README.md "SSOT: Ref Script as Gate"
# ======================================================================


# ``forge_init_ones`` routing: the value lives ONLY in the rendered gate
# product. Ref side: the generic projection exports it as FORGE_INIT_ONES
# (the launcher probes it fail-fast; see model_pure_mup_mtp.py::init_weights
# for the engine contract). Ours side: the runner derives the ones/ or no1/
# canonical subdir itself via GateInputs.checkpoint_root().


def _load_workload_config_for_ref(repo_root: Path) -> dict[str, Any]:
    """Best-effort load of eval.toml (or the template) for ref-script helpers.

    Returns ``{}`` when the config cannot be loaded (e.g. unit tests
    operating on a synthetic repo root). Production gate paths always
    have a valid config — the empty fallback is for test scaffolding.

    User-layer TOMLs (``config/ref.toml``, ``config/optim.toml``, …)
    are merged only when *repo_root* matches the real harness repo
    root. Tests pass a synthetic tempdir as *repo_root*; merging real
    user layers there would leak host-specific paths and FORGE_* knobs
    into a supposedly hermetic fixture and break the assertions that
    pin the exact set of injected env vars.
    """
    user_copy = repo_root / "config" / "eval.toml"
    template = repo_root / "config" / "eval" / "dense_training" / "dense_training.toml"
    path = str(user_copy) if user_copy.exists() else str(template)
    try:
        include_user_config = repo_root.resolve() == config_runtime.repo_root().resolve()
    except Exception:
        include_user_config = False
    try:
        _, workload_config = config_runtime.load_workload_config(
            path, include_user_config=include_user_config
        )
    except Exception:
        return {}
    return workload_config


def _ref_script_path_env(
    repo_root: Path,
    cfg: dict[str, Any],
    *,
    workload_config: dict[str, Any] | None = None,
    dump_dir: Path | None = None,
) -> dict[str, str]:
    """Resolve PATH-class env vars the harness pushes into the L0 ref script.

    Two categories of variables are emitted, both with the same goal:
    keep the L0 ref script free of any individual-user filesystem
    defaults.

    * **Deployment paths** — pulled from the workload SSOT (suite
      override first, then ``[defaults]``):

      - ``MEGATRON_ROOT`` (``[defaults].megatron_root``)
      - ``DATA_PATH``    (``[defaults].data_path``)

      Tokenizer + ultra_fineweb paths flow via the separate
      ``resolve_assets`` channel (``FORGE_TOKENIZER_DIR`` /
      ``FORGE_DATA_DIR``) — set on ``os.environ`` by ``build_suite_env``
      so subprocesses inherit them.

      Empty / missing values are skipped on purpose — the ref script
      uses ``${VAR:?…}`` fail-fast guards so an unconfigured machine
      hits a clear error instead of falling back to a baked-in path.

    * **Gate-private artifact paths** — auto-derived from *dump_dir* so
      every gate has its own subtree and so the ref script never has to
      write a default (the user-specific SAVE_PATH / TENSORBOARD_DIR
      defaults that previously crashed read-only-mount machines):

      - ``SAVE_PATH``       = ``<dump_dir>/save``
      - ``TENSORBOARD_DIR`` = ``<dump_dir>/tensorboard``

      ``dump_dir`` is None for callers that don't have one; in that
      case the two gate-private paths are simply omitted and the ref script falls
      back to its repo-relative ``.artifacts/ref_local/<basename>/{save,
      tensorboard}`` auto-derive.

    Model/optim/muP/MTP shape is NOT injected here — the generic product
    projection (``tools/product_env.py``) exports every product key
    straight from the rendered product, which is the sole source (§D7).
    This helper only sets the B-class deployment values
    (MEGATRON_ROOT / DATA_PATH / DATA_CONF) the renderer cannot know
    statically, plus the optional dump-dir save/tensorboard paths.
    """
    overrides: dict[str, str] = {}
    if workload_config is None:
        workload_config = _load_workload_config_for_ref(repo_root)
    ref = workload_config.get("ref", {}) if isinstance(workload_config, dict) else {}
    data = workload_config.get("data", {}) if isinstance(workload_config, dict) else {}

    # Resolve tokenizer + data assets (download from HF on cache miss) and
    # surface FORGE_TOKENIZER_DIR / FORGE_DATA_DIR into the child shell.
    # Idempotent: cache-hit branch fires on re-entry.
    overrides.update(config_runtime.resolve_assets(ref, data))

    def _resolve(key: str) -> str | None:
        if isinstance(cfg, dict) and cfg.get(key):
            return cfg.get(key)
        if isinstance(ref, dict):
            return ref.get(key)
        return None

    # B-class deployment values the renderer cannot know statically: the
    # ref-script launcher reads them from the env (they appear in the gate
    # product [env] as the "<runtime>" sentinel). The A-class model/optim
    # shape is NOT injected here — the generic projection (product_env.py)
    # exports it straight from the rendered product [cli]/[env], which is
    # the sole source (§D7).
    for cfg_key, env_key in (
        ("megatron_root", "MEGATRON_ROOT"),
        ("data_path", "DATA_PATH"),
        ("data_conf", "DATA_CONF"),
    ):
        value = _resolve(cfg_key)
        if isinstance(value, str) and value:
            overrides[env_key] = value

    if dump_dir is not None:
        overrides["SAVE_PATH"] = str(dump_dir / "save")
        overrides["TENSORBOARD_DIR"] = str(dump_dir / "tensorboard")

    return overrides


def _resolve_customer_ref_script(repo_root: Path, ref_script_name: str) -> Path:
    """Resolve a ``ref/reference/<basename>`` path; reject directory-traversal."""
    if not ref_script_name or "/" in ref_script_name or "\\" in ref_script_name:
        raise ValueError(
            f"ref_script_name must be a plain basename under ref/reference/ "
            f"(got {ref_script_name!r})"
        )
    customer_path = (repo_root / "ref" / "reference" / ref_script_name).resolve()
    if not customer_path.is_file():
        raise FileNotFoundError(
            f"L0 reference script not found: {customer_path}. "
            "(Configured via config/eval.toml [defaults].ref_script.)"
        )
    return customer_path


# ======================================================================
# Ref-trajectory caching
# ======================================================================
#
# The L0 ref script is deterministic given the same inputs (script
# content + env overrides). Long-horizon gates (long-horizon long-train, op-long)
# take ~30 min per ref run. Caching the trajectory avoids redundant
# re-runs when only the ours-side engine is being iterated on.
#
# Cache key = SHA-256(suite_key, ref_script_content, deterministic env).
# Volatile output paths (DUMP_DIR, LOSS_DUMP_FILE, SAVE_PATH,
# TENSORBOARD_DIR) are excluded from the key.
#
# Disable: ``FORGE_REF_CACHE=0``.
# Invalidate: ``rm -rf .artifacts/ref_cache/``.

_VOLATILE_ENV_KEYS = frozenset(
    {
        "DUMP_DIR",
        "LOSS_DUMP_FILE",
        "SAVE_PATH",
        "TENSORBOARD_DIR",
        # MASTER_ADDR / MASTER_PORT are rendezvous-only inputs to
        # torch.distributed; they have zero effect on the captured
        # trajectory. The ``master_port = "auto"`` sentinel allocates a
        # fresh ephemeral port per run, so excluding both keys is what
        # keeps the ref-cache hit rate independent of port churn.
        "MASTER_ADDR",
        "MASTER_PORT",
    }
)


def _ref_cache_key(
    suite_key: str,
    ref_script_path: Path,
    merged_env: dict[str, str],
    *,
    repo_root: Path,
) -> str:
    """Compute a stable cache key for a ref-script invocation.

    Hashes the suite preset name, the ref-script file content, all
    non-volatile env overrides, AND an environment fingerprint
    (torch build + harness git sha). Any change to those produces a
    different key.

    The fingerprint guards two real bug classes seen in loop
    ``e180edc7``: (a) the ref script ``source``s sibling shells under
    ``ref/reference/``, so a git-sha bump catches edits to those even
    though the script-content hash does not; (b) torch / CUDA
    upgrades shift kernel selection, so reusing a cached trajectory
    captured against an older stack produces phantom bit-exact
    regressions in downstream gates.
    """
    hasher = hashlib.sha256()
    hasher.update(f"suite:{suite_key}\n".encode())
    hasher.update(ref_script_path.read_bytes())
    for k in sorted(merged_env):
        if k not in _VOLATILE_ENV_KEYS:
            hasher.update(f"env:{k}={merged_env[k]}\n".encode())
    hasher.update(f"fingerprint:{_environment_fingerprint(repo_root)}\n".encode())
    return hasher.hexdigest()[:16]


_FINGERPRINT_CACHE: dict[Path, str] = {}


def _environment_fingerprint(repo_root: Path) -> str:
    """Stable string identifying the binary stack + source-tree git sha.

    Memoised per resolved repo path so the gate dispatcher does not
    fork ``git`` once per suite. Both halves use explicit ``no-*``
    sentinels on failure rather than empty strings: an empty value
    would let two distinct unknown environments collide on the same
    cache key.
    """
    key = repo_root.resolve()
    cached = _FINGERPRINT_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        import torch

        torch_version: str = torch.__version__
    except ImportError:
        torch_version = "no-torch"
    try:
        sha = (
            subprocess.check_output(
                ["git", "-C", str(key), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            or "no-git"
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha = "no-git"
    fp = f"torch={torch_version};git={sha}"
    _FINGERPRINT_CACHE[key] = fp
    return fp


def _ref_cache_base(repo_root: Path) -> Path:
    return repo_root / ".artifacts" / "ref_cache"


def _try_load_ref_cache(
    cache_dir: Path,
) -> tuple[dict[int, float], dict[int, float], dict[str, object], float] | None:
    """Try to load a cached ref trajectory.

    Returns ``(loss_by_step, grad_norm_by_step, metadata, elapsed_s)`` on
    hit, ``None`` on miss. ``grad_norm_by_step`` is empty for legacy
    caches written before the grad-norm baseline was persisted; the
    Stage 1 bitwise gates fail fast on an empty grad baseline so a stale
    cache surfaces as a re-run rather than a silent loss-only grade.
    """
    manifest_path = cache_dir / "cache_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        loss_by_step = {int(k): float(v) for k, v in raw["loss_by_step"].items()}
        grad_norm_by_step = {int(k): float(v) for k, v in raw.get("grad_norm_by_step", {}).items()}
        metadata = raw["metadata"]
        elapsed_s = float(raw["elapsed_s"])
        return loss_by_step, grad_norm_by_step, metadata, elapsed_s
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _save_ref_cache(
    cache_dir: Path,
    loss_by_step: dict[int, float],
    grad_norm_by_step: dict[int, float],
    ref_run: Any,
) -> None:
    """Persist a successful ref trajectory to the on-disk cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "loss_by_step": {str(k): v for k, v in loss_by_step.items()},
        "grad_norm_by_step": {str(k): v for k, v in grad_norm_by_step.items()},
        "metadata": dict(ref_run.metadata) if ref_run.metadata else {},
        "elapsed_s": ref_run.elapsed_s,
    }
    (cache_dir / "cache_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if ref_run.stdout_path.exists():
        shutil.copy2(ref_run.stdout_path, cache_dir / "ref_stdout.log")


def _ref_cache_enabled() -> bool:
    return os.environ.get("FORGE_REF_CACHE", "1") != "0"


def run_via_ref_script(
    *,
    repo_root: Path,
    suite_key: str,
    cfg: dict[str, Any],
    artifact_dir: Path,
    capture_loss_trace: bool = True,
    extra_env: dict[str, str] | None = None,
    extra_ref_args: list[str] | tuple[str, ...] | None = None,
    hash_capture_level: int = 0,
    hash_output: Path | None = None,
    persistent: bool = False,
) -> dict[str, Any]:
    """Shell-exec the L0 ref script (or the capture bridge) and return
    ``{loss_by_step, ref_run, dump_dir, hash_output}``.

    Thin adapter over :func:`tools.ref_script_runner.run_ref_script` so
    dispatcher code keeps speaking suite vocabulary (cfg keys,
    world_size). Capture-mode runs go through :func:`run_ref_capture`
    instead for the alignment single-step path — this function owns the
    plain ``[LOSS]``-trajectory path for bitwise-singlecard / bitwise-multicard / bitwise-perf / resume, and
    optionally piggybacks a persistent hash-record dump on it when
    ``hash_capture_level > 0`` (the bridge is invoked instead of the
    bare L0 script in that case so the standard hook gets interposed
    into the L0 training loop without modifying the read-only ref
    script).

    There is no frozen SHA anchor in this path; running the ref script
    is the baseline contract. Caller maps ``ref_run.succeeded == False``
    into a gate failure result. When the ref script's deterministic
    inputs (content + env overrides) match a previous successful run,
    the cached trajectory is returned without re-executing the script
    (only when ``hash_capture_level == 0`` — capture runs always live).
    Disable with ``FORGE_REF_CACHE=0``.

    Env precedence (low → high; later wins):

    1. Generic product projection (``tools/product_env.export_map`` of the
       rendered ref product — §D7 SSOT for the gate shape +
       model/optim/muP/MTP reals; upper-cased [cli] keys, [env] verbatim).
    2. Harness path shim (``_ref_script_path_env``).
    3. Product-dir pointer (``FORGE_REF_CONFIG_DIR``) + data SSOT pointer
       (``FORGE_DATA_TOML``).
    4. Caller ``extra_env``.

    Collapse: the legacy ``[evals.<name>.optim_overrides]`` and
    ``[evals.<name>.ref_env]`` env-override overlays are gone — the rendered
    product is the sole physical source for those values.

    Hash-capture wire (when ``hash_capture_level > 0``):

    * ``ref_script_path`` switches from the bare L0 script
      (``config_runtime.ref_script``) to the bridge
      (``config_runtime.ref_capture_script``); the bridge sed-patches
      the L0 launcher to insert ``ref/bridges/interposer.py`` as the
      torchrun entry, which calls
      ``harness_hook.install(... hash_capture_level=N,
      persistent=True)`` before delegating back to the L0 entry via
      ``runpy``.
    * ``--hash-capture-level <N> --hash-output <hash_output>
      --persistent`` are appended to the bridge subprocess argv via
      ``extra_ref_args`` (typed CLI wire, no env var).
    """
    workload_config = _load_workload_config_for_ref(repo_root)

    if hash_capture_level > 0:
        # Route through the capture bridge so the standard hook gets
        # interposed into the L0 training loop. The bridge replaces
        # the torchrun Python entry with ref/bridges/interposer.py
        # and forwards "$@" verbatim, so our --hash-* args reach the
        # interposer's argparse.
        raw_bridge = config_runtime.ref_capture_script(workload_config)
        if not raw_bridge:
            raise RuntimeError(_REF_CAPTURE_BRIDGE_MISSING_MSG)
        ref_script_path = _resolve_ref_capture_script(repo_root, raw_bridge)
    else:
        ref_script_name = config_runtime.ref_script(workload_config)
        ref_script_path = _resolve_customer_ref_script(repo_root, ref_script_name)

    dump_dir = artifact_dir / f"ref_dump__{suite_key}"
    timeout_s = config_runtime.suite_ref_timeout_s(workload_config, suite_key) or None
    # Generic product projection — LOWEST env layer. This path bypasses
    # ref/run_gate.sh (stage2 op gates via gate_common.resolve_ref_trajectory),
    # so like tools/bootstrap_canonical.py it must be its own projection layer:
    # the generic-projection launchers (run_qwen3_dense.sh, pure_mup_mtp) take
    # ALL gate parameters from the caller-projected environment. Everything
    # below (path shim, pointers, caller extra_env) is applied after and
    # wins, mirroring run_gate.sh's "product first, runtime last" order.
    # Meta gates without a rendered ref product skip.
    merged_extra_env: dict[str, str] = {}
    _ref_product = repo_root / "ref" / "config" / f"{suite_key}.toml"
    if _ref_product.is_file():
        from tools import product_env as _product_env

        merged_extra_env.update(_product_env.export_map(_ref_product))
        # Deployment-key suppression, mirroring runtime_env.py's ref branch:
        # a freeze-filled product carries a CONCRETE MASTER_PORT which the
        # generic projection just exported; the runtime resolution (fresh
        # free port for "auto") must win, as it does in run_gate.sh where
        # runtime_env is eval'd after product_env.
        _defaults = config_runtime.runtime_env_defaults(workload_config)
        merged_extra_env["MASTER_ADDR"] = _defaults.master_addr
        merged_extra_env["MASTER_PORT"] = _resolve_master_port(_defaults.master_port)
    merged_extra_env.update(
        _ref_script_path_env(repo_root, cfg, workload_config=workload_config, dump_dir=dump_dir)
    )
    # Point the L0 launcher at its rendered product dir. Applies symmetrically
    # to the bitwise-singlecard–long-horizon trajectory launcher and the
    # alignment capture bridge (whose interposed launcher reads the same
    # product). Set before the caller extra_env merge so tests can override.
    merged_extra_env["FORGE_REF_CONFIG_DIR"] = str(repo_root / "ref" / "config")
    # The loader kind (+ data_path for meta bundles) lives in data.toml (the
    # data-value SSOT), read via FORGE_DATA_TOML → the per-loop config dir's
    # data.toml. Every side reads it from this one file: meta DP×TP bundles, the
    # native torch ref + sisters (--data-config → data_loader_from_config), and
    # ours (runtime_config.data_loader()). No DATA_LOADER env value anywhere.
    merged_extra_env["FORGE_DATA_TOML"] = str(config_runtime._data_config_path())
    # forge_init_ones travels inside the product projection above (the
    # forge_init_ones [cli] key → FORGE_INIT_ONES env) — no separate
    # injection channel; the launcher probes FORGE_INIT_ONES fail-fast.
    if extra_env:
        merged_extra_env.update(extra_env)
    if hash_capture_level > 0:
        # Bridge needs FORGE_BACKEND to pick its sed-patch needle.
        merged_extra_env.setdefault("FORGE_BACKEND", config_runtime.ref_backend(workload_config))
        # The bridge must interpose the ACTIVE model's L0 launcher — the same
        # [ref].ref_script the bitwise-singlecard–long-horizon ref gates run — not a hardcoded
        # pure_mup_mtp script. Without this, qwen3's alignment forward/backward-align
        # would capture a minicpm reference (wrong architecture). The bridge's
        # torch branch reads FORGE_BRIDGE_REF_SCRIPT and derives PY_ENTRY from
        # the launcher, so every model's alignment matches its bitwise-singlecard–long-horizon reference.
        merged_extra_env.setdefault(
            "FORGE_BRIDGE_REF_SCRIPT", config_runtime.ref_script(workload_config)
        )
        # Back-compat env: HOOK_OUTPUT_FILE is read by interposer.py
        # when --hash-output is not passed, so legacy bridges still
        # see a non-empty value. New CLI arg overrides it.
        if hash_output is not None:
            merged_extra_env.setdefault("HOOK_OUTPUT_FILE", str(hash_output))
    # Suite-level ``ref_extra_args`` (list of strings) appends bare CLI
    # tokens after the bash ref-script invocation, which the L0 script
    # forwards via ``"$@"`` to its torchrun entry. The dispatcher uses
    # this for the long-horizon ``--no-deterministic`` toggle so the ref preset
    # stays frozen as the SSOT and only the suite cfg owns the flip;
    # explicit caller-supplied ``extra_ref_args`` (programmatic callers)
    # still take precedence so tests can override per-invocation.
    cfg_ref_extra_args = cfg.get("ref_extra_args") if isinstance(cfg, dict) else None
    if extra_ref_args is None and cfg_ref_extra_args:
        extra_ref_args = list(cfg_ref_extra_args)
    if hash_capture_level > 0:
        hash_args: list[str] = [
            "--hash-capture-level",
            str(int(hash_capture_level)),
        ]
        if hash_output is not None:
            hash_args += ["--hash-output", str(hash_output)]
        if persistent:
            hash_args.append("--persistent")
        extra_ref_args = list(extra_ref_args or []) + hash_args

    # ── Ref-trajectory cache ─────────────────────────────────────────
    # Capture runs always go live: the hash dump path and persistent
    # flag both vary per gate, and a cached run wouldn't have the
    # right artifact on disk.
    use_cache = _ref_cache_enabled() and capture_loss_trace and hash_capture_level == 0
    cache_dir: Path | None = None
    if use_cache:
        cache_key = _ref_cache_key(
            suite_key,
            ref_script_path,
            merged_extra_env,
            repo_root=repo_root,
        )
        cache_dir = _ref_cache_base(repo_root) / cache_key
        cached = _try_load_ref_cache(cache_dir)
        if cached is not None:
            loss_by_step, grad_norm_by_step, metadata, elapsed_s = cached
            print(
                f"[ref-cache] HIT for {suite_key} — reusing cached trajectory "
                f"({len(loss_by_step)} steps, {elapsed_s:.0f}s original runtime)"
            )
            cached_stdout = cache_dir / "ref_stdout.log"
            ref_run = _ref_script_runner.RefRun(
                returncode=0,
                stdout_path=cached_stdout,
                dump_dir=cache_dir,
                loss_file=None,
                elapsed_s=elapsed_s,
                timed_out=False,
                metadata=metadata,
            )
            return {
                "loss_by_step": loss_by_step,
                "grad_norm_by_step": grad_norm_by_step,
                "ref_run": ref_run,
                "dump_dir": cache_dir,
            }

    # ── Cache miss — live ref-script execution ───────────────────────
    ref_run = _ref_script_runner.run_ref_script(
        repo_root=repo_root,
        gate_name=suite_key,
        dump_dir=dump_dir,
        ref_script_path=ref_script_path,
        timeout_s=timeout_s,
        extra_env=merged_extra_env,
        extra_ref_args=extra_ref_args,
        capture_loss_trace=capture_loss_trace,
    )

    loss_by_step = {}
    grad_norm_by_step = {}
    if ref_run.loss_file is not None:
        loss_by_step = _ref_script_runner.parse_loss_dump(ref_run.loss_file)
        # Recover the grad_norm baseline for the Stage 1 bitwise gates
        # (bitwise-singlecard/bitwise-multicard/bitwise-perf). Only the loss-dump file carries grad_norm; the
        # Megatron stdout fallback below does not, which is why a
        # stdout-only ref surfaces as an empty grad baseline (the
        # bitwise gate fails fast rather than grading on loss alone).
        grad_norm_by_step = {
            step: entry["grad_norm"]
            for step, entry in _ref_script_runner.parse_loss_dump_with_grad(
                ref_run.loss_file
            ).items()
        }
    if not loss_by_step:
        loss_by_step = _ref_script_runner.parse_stdout_loss(ref_run.stdout_path)

    if use_cache and cache_dir is not None and ref_run.succeeded and loss_by_step:
        _save_ref_cache(cache_dir, loss_by_step, grad_norm_by_step, ref_run)
        print(f"[ref-cache] SAVED for {suite_key} ({len(loss_by_step)} steps)")

    return {
        "loss_by_step": loss_by_step,
        "grad_norm_by_step": grad_norm_by_step,
        "ref_run": ref_run,
        "dump_dir": dump_dir,
    }


# ──────────────────────────────────────────────────────────────────────
# alignment capture-mode high-level APIs
# ──────────────────────────────────────────────────────────────────────
#
# These are the only two functions ``evals/dispatcher.py`` needs to
# call to drive the alignment.forward / alignment.backward flow. They hide the bridge
# resolution (which agent-generated script to bash) and the env
# wiring (``HOOK_OUTPUT_FILE`` for the ref side,
# ``FORGE_CAPTURE_OUTPUT_FILE`` for the candidate side) so the
# dispatcher only sees ``CaptureArtifacts`` in / out. Pointing the
# gates at a new customer entry is a pure
# ``[defaults].ref_capture_script`` config change — neither these
# helpers nor the dispatcher needs editing.


_REF_CAPTURE_BRIDGE_MISSING_MSG = (
    "config/eval.toml [defaults].ref_capture_script is empty — the "
    "alignment capture bridge has not been generated yet for this customer "
    "entry. The bridge is a script that, when bashed with "
    "``HOOK_OUTPUT_FILE=<abs path>`` set, calls "
    "``evals.harness_hook.install(model, optimizer, output_file=...)`` "
    "in the per-rank Python process so the standard hook writes "
    "``<HOOK_OUTPUT_FILE>`` + ``<HOOK_OUTPUT_FILE>.graph.json``. See "
    "evals/harness_hook/recipes/README.md for the contract and patterns; "
    "set [defaults].ref_capture_script to your bridge's path."
)


def _resolve_ref_capture_script(repo_root: Path, raw: str) -> Path:
    """Resolve ``[defaults].ref_capture_script`` to an absolute path.

    Unlike ``[defaults].ref_script`` (a basename under
    ``ref/reference/``), the alignment capture bridge can live anywhere the
    agent puts it. We accept absolute paths verbatim and resolve
    relative paths against the repo root, then assert the file exists.
    """
    from pathlib import Path as _Path

    candidate = _Path(raw)
    if not candidate.is_absolute():
        candidate = (repo_root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(
            f"[defaults].ref_capture_script points at {candidate}, but "
            "no file is present there. Generate the bridge first; see "
            "evals/harness_hook/recipes/README.md."
        )
    return candidate


def write_ref_capture_status(
    artifact_dir: Path,
    ref_run: Any,
    *,
    dump_present: bool,
    hash_capture_level: int | None = None,
    graph_present: bool | None = None,
    loss_by_step: dict[int, float] | None = None,
) -> None:
    """Persist the ref subprocess's exit status next to ``result.json``.

    The meta ``harness_configs`` gate (``ref_side.run_ref_side``) asserts the
    ref-PHASE returncode structurally from this file — not by string-matching
    the result summary, and not from the overall ``bin/harness run`` exit code
    (which is dominated by the expected ours-stub failure at gate-authoring
    time). Written by every ref-capture site (alignment ``run_ref_capture`` +
    trajectory ``resolve_ref_trajectory``); a consumer treats a MISSING file as
    a ref failure (fail-closed — the ref subprocess never reached a verdict).

    ``dump_present`` MUST reflect a real stat of the level-appropriate dump
    file (not merely ``ref_run.succeeded`` — that was a trajectory-path
    misnomer). The optional fields let the meta gate's per-gate judgment
    (``ref_side._ref_verdict`` conditions 2 & 5) read authoritative facts the
    harness alone can compute: the ``hash_capture_level`` the ref actually ran
    at, whether the ``.graph.json`` sibling landed, and the parsed loss
    trajectory health (step count / finiteness / non-degeneracy).
    """
    status: dict[str, Any] = {
        "returncode": int(getattr(ref_run, "returncode", -1)),
        "timed_out": bool(getattr(ref_run, "timed_out", False)),
        "dump_present": bool(dump_present),
    }
    if hash_capture_level is not None:
        status["hash_capture_level"] = int(hash_capture_level)
    if graph_present is not None:
        status["graph_present"] = bool(graph_present)
    if loss_by_step is not None:
        losses = [float(v) for v in loss_by_step.values()]
        finite = [v for v in losses if math.isfinite(v)]
        status["loss_num_steps"] = len(losses)
        status["loss_all_finite"] = len(finite) == len(losses) and bool(losses)
        status["loss_constant"] = bool(finite) and len(set(finite)) == 1
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "ref_capture_status.json").write_text(json.dumps(status))


# ======================================================================
# Subprocess execution helpers
# ======================================================================


def run_streaming_subprocess(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
    out_file_path: Path,
) -> tuple[int, str]:
    """Run a subprocess streaming stdout to both the console and a log file.

    Returns (returncode, output_text). On timeout, kills the process and
    returns (-1, partial_output).
    """
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )

    def _relay_output() -> None:
        with open(out_file_path, "w", encoding="utf-8") as out_fh:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                out_fh.write(line)
                out_fh.flush()

    reader = threading.Thread(target=_relay_output, daemon=True)
    reader.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            proc.kill()
        proc.wait()
        reader.join(timeout=1)
        if proc.stdout is not None:
            proc.stdout.close()
        output = (
            out_file_path.read_text(encoding="utf-8", errors="replace")
            if out_file_path.exists()
            else ""
        )
        return -1, output
    reader.join(timeout=1)
    if proc.stdout is not None:
        proc.stdout.close()
    output = (
        out_file_path.read_text(encoding="utf-8", errors="replace")
        if out_file_path.exists()
        else ""
    )
    return proc.returncode, output


def torchrun_cmd(
    nproc: int,
    script: str,
    *,
    master_addr: str | None = None,
    master_port: str | None = None,
) -> list[str]:
    """Build a torchrun command line for single-node DP.

    When *master_addr* / *master_port* are ``None``, values are resolved
    from ``runtime_env_defaults()`` (→ dense_training.toml SSOT).
    The ``"auto"`` sentinel (or an empty string) on *master_port*
    resolves to a freshly-allocated free ephemeral port — same contract
    as :func:`distributed_env`.
    """
    if master_addr is None or master_port is None:
        defaults = config_runtime.runtime_env_defaults()
        master_addr = master_addr or defaults.master_addr
        master_port = master_port or defaults.master_port
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(nproc),
        "--nnodes",
        "1",
        "--master_addr",
        master_addr,
        "--master_port",
        _resolve_master_port(master_port),
        script,
    ]


# ======================================================================
# Stdout wire-format parsers
# ----------------------------------------------------------------------
# Re-exported from :mod:`harness.wire_format`, the SSOT for line-grammar
# regexes and trajectory parsing. ``parse_key_values``,
# ``parse_loss_lines``, ``parse_loss_lines_to_dict`` are imported at
# the top of this module so existing callers (``evals.dispatcher``,
# ``evals.gate_common``, suite scripts) keep their public import path.
# ======================================================================


def missing_window_steps(observed_by_step: Mapping[int, Any], steps: range) -> list[int]:
    """Return gate-window steps absent from a parsed trajectory mapping."""
    return [step for step in steps if step not in observed_by_step]


def window_loss_diff_metrics(
    baseline_by_step: dict[int, float],
    ours_by_step: dict[int, float],
    steps: range,
) -> dict[str, Any]:
    """Compute shared relative/signed loss-diff metrics for gate windows.

    In addition to ``mean_rel_diff`` / ``max_rel_diff`` / ``signed_mean``,
    splits the gate window into 4 equal buckets and emits a
    ``drift_warning`` flag when systematic bias (not run-to-run jitter) is
    detected. Faithful det-off training produces ``signed_mean ≈ 0`` per
    bucket; significant deviation or monotonic same-sign growth across the
    window is real numerical bias and must be attributed before
    ``STAGE_STATUS: finished``. See long-horizon.md §Loss-drift warning policy.
    """
    rel_diffs: list[float] = []
    signed_diffs: list[float] = []
    step_diffs: list[dict[str, float | int]] = []
    for step in steps:
        baseline = baseline_by_step[step]
        ours = ours_by_step[step]
        if abs(baseline) < 1e-12:
            raise ValueError(f"baseline loss is 0 at step {step}")
        signed = ours - baseline
        rel = abs(signed) / abs(baseline)
        rel_diffs.append(rel)
        signed_diffs.append(signed)
        step_diffs.append(
            {
                "step": step,
                "our_loss": ours,
                "baseline_loss": baseline,
                "rel_diff": rel,
                "signed_diff": signed,
            }
        )
    if not rel_diffs:
        raise ValueError("no comparable steps in gate window")

    # Bucket the gate window into 4 equal slices and compute per-bucket
    # signed_mean + |signed| / |baseline| ratio. Skip when window is too
    # short to give ≥5 steps per bucket.
    buckets: list[dict[str, Any]] = []
    drift_warning: str | None = None
    n_steps = len(step_diffs)
    if n_steps >= 20:
        chunk = n_steps // 4
        for i in range(4):
            lo = i * chunk
            hi = (i + 1) * chunk if i < 3 else n_steps
            slc = step_diffs[lo:hi]
            if not slc:
                continue
            b_signed = sum(d["signed_diff"] for d in slc) / len(slc)
            b_baseline_mag = sum(abs(d["baseline_loss"]) for d in slc) / len(slc)
            buckets.append(
                {
                    "step_range": [slc[0]["step"], slc[-1]["step"]],
                    "signed_mean": b_signed,
                    "rel_to_baseline": abs(b_signed) / max(b_baseline_mag, 1e-12),
                }
            )
        # Trigger 1: late-window signed bias > 1% of late-window |baseline|
        # — this is the systematic bias signature (det-off jitter has
        # |signed_mean| < ~0.5% of |baseline|).
        if buckets and buckets[-1]["rel_to_baseline"] > 0.01:
            last = buckets[-1]
            drift_warning = (
                f"signed_mean={last['signed_mean']:+.4f} is "
                f"{last['rel_to_baseline'] * 100:.2f}% of late-window "
                f"|baseline|; systematic bias, not det-off jitter"
            )
        # Trigger 2: monotonic same-sign growth across all 4 buckets with
        # the last bucket > 1.5× the first — compounding bias signature.
        elif len(buckets) == 4:
            signs = [b["signed_mean"] for b in buckets]
            if all(s > 0 for s in signs) and signs[3] > 1.5 * signs[0] > 0:
                drift_warning = (
                    "signed_mean grows monotonically positive across the "
                    "gate window (compounding bias)"
                )
            elif all(s < 0 for s in signs) and signs[3] < 1.5 * signs[0] < 0:
                drift_warning = (
                    "signed_mean grows monotonically negative across the "
                    "gate window (compounding bias)"
                )

    return {
        "compared_steps": len(rel_diffs),
        "mean_rel_diff": sum(rel_diffs) / len(rel_diffs),
        "max_rel_diff": max(rel_diffs),
        "signed_mean": sum(signed_diffs) / len(signed_diffs),
        # ``signed_mean`` normalised by the window's mean ``|baseline|``,
        # so callers can compare the signed drift against the same
        # ``loss_rel_threshold`` (2.5% fraction) ``mean_rel_diff`` is
        # judged against. ``abs(baseline) < 1e-12`` was already rejected
        # above, so the denominator is strictly > 0.
        "signed_mean_rel": (
            (sum(signed_diffs) / len(signed_diffs))
            / (sum(abs(d["baseline_loss"]) for d in step_diffs) / len(step_diffs))
        ),
        "step_diffs": step_diffs,
        "buckets": buckets,
        "drift_warning": drift_warning,
    }


# ======================================================================
# Result templates
# ======================================================================


def classify_ref_failure(
    *,
    timed_out: bool,
    returncode: int,
    stdout_text: str,
) -> str:
    """Classify a failed ref-script invocation into an actionable bucket.

    Returns one of:

    * ``"timeout"`` — wall-clock exceeded; usually means a slow GPU
      or an OOM hang. Agent-loop should NOT auto-retry on the same host.
    * ``"data_missing"`` — preprocessed dataset binary not found
      (``ERROR: Megatron binary data not found``); agent-loop should
      surface a one-time data-prep instruction instead of retrying.
    * ``"script_setup"`` — the ref script aborted before training
      started (preset error, missing path, bad CLI). Retrying without
      a config change won't help; surface the tail to the operator.
    * ``"network"`` — torch.distributed / NCCL rendezvous failures
      (port collision, peer hang). Agent-loop MAY retry once with a
      fresh MASTER_PORT offset.
    * ``"training_failure"`` — training started and crashed mid-run
      (CUDA OOM, NaN loss, asserts). Distinct from ``script_setup`` so
      the loop can offer different remediation prompts.
    * ``"unknown"`` — fallback when none of the heuristics matched.

    Heuristics deliberately favour false-negatives (``"unknown"``) over
    false-positives so the agent-loop never claims a transient is a
    permanent failure.
    """
    if timed_out:
        return "timeout"
    text = stdout_text or ""
    if "ERROR: Megatron binary data not found" in text:
        return "data_missing"
    # Same bucket for the alternative L0 ref stacks: pure-torch sibling
    # bails with "DATA_PATH_FILE not found", and both siblings bail with
    # "Data conf not found" when the [data].conf_path file is missing.
    if "ERROR: DATA_PATH_FILE not found" in text:
        return "data_missing"
    if "ERROR: Data conf not found" in text:
        return "data_missing"
    if "ERROR: unknown FORGE_GATE preset" in text:
        return "script_setup"
    lower = text.lower()
    network_markers = (
        "address already in use",
        "torch.distributed: rendezvous timeout",
        "nccl error",
        "watchdog caught collective operation timeout",
    )
    if any(marker in lower for marker in network_markers):
        return "network"
    training_markers = (
        "cuda out of memory",
        "torch.cuda.outofmemoryerror",
        "loss is nan",
        "assertionerror",
        "traceback (most recent call last):",
    )
    if any(marker in lower for marker in training_markers):
        return "training_failure"
    if returncode != 0:
        return "script_setup"
    return "unknown"


def missing_script_result(suite: str, script: Path) -> dict[str, Any]:
    return {
        "status": "failed",
        "suite": suite,
        "summary": f"Script not found: {script}",
        "metrics": {},
        "details": {"missing_script": str(script)},
    }


# ======================================================================
# Shared suite-script helpers
# ----------------------------------------------------------------------
# Dataloader construction is intentionally NOT a harness concern.  The
# ours-side dataloader lives inside
# :mod:`training_engine_tensor` (the self-developed engine), reached
# through :func:`training_engine_tensor.train_loop.run_training_loop`;
# the Megatron-side dataloader is owned by the L0 ref script
# (``ref/reference/${ref_script}``).  Gate scripts must
# not synthesise a third dataloader at the harness layer — see
# ``prompt/develop_prompt/stage1.md`` §"Dataloader 说明".
# ======================================================================
