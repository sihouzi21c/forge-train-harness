"""Thin gate dispatcher — generic 3-step scripted executor.

The suite list and per-gate routing live in the workload config
(``config/eval.toml``); every stage1 gate routes through
``_run_scripted_suite`` (registry keys
``needs_ref`` / ``ours_runner`` / ``verdict``): ① ``ref/run_gate.sh``
(optional, trajectory-cached) → ② ``evals/scripts/<ours_runner>.sh`` →
③ in-process ``evals.verdicts.<verdict>.run``. The dispatcher understands
no gate-specific parameters — both side scripts read everything from the
rendered products (``ref/config/<gate>.toml`` /
``workload/src/config/<gate>.toml``).

The exempt stage2 op-* suites keep their legacy in-process handlers in
``evals.dispatcher_stage2`` (``RUNNER_KINDS`` re-exported here).

SSOT model — Ref Script as Gate
-------------------------------
Per ``README.md``, the L0 ref script (basename owned by
``config/ref.toml [ref].ref_script`` and resolved via
``harness.config_runtime.ref_script``) is the execution
authority for every gate that needs a baseline trajectory.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

from evals._common import (
    _environment_fingerprint,
    _ref_cache_base,
    _ref_cache_enabled,
    _resolve_customer_ref_script,
    base_env,
    missing_script_result,
    run_streaming_subprocess,
)

# Stage-2 op-* legacy routing table (thin-dispatcher refactor EXEMPT, plan
# §10 step 7) — handlers live in evals.dispatcher_stage2; re-exported here
# so callers and tests keep addressing dispatcher.RUNNER_KINDS.
from evals.dispatcher_stage2 import RUNNER_KINDS
from evals.gate_product import GateProductError, load_gate_product
from harness import config_runtime, run_schema

__all__ = ["RUNNER_KINDS", "run_suite", "stage_groups"]

Runner = Any

# Scripted ref runner + the artifacts a trajectory-mode (hash_capture_level==0)
# run caches for reuse. Hash-capture runs (level > 0, e.g. the alignment
# gates) are never cached, so only the trajectory artifacts are listed here.
_SCRIPTED_REF_RUNNER = "ref/run_gate.sh"
_SCRIPTED_REF_CACHE_FILES = (
    "status.json",
    "ref.log",
    "dump/ref_loss.txt",
)


def _gate_hash_capture_level(
    cfg: dict[str, Any], repo_root: Path, suite_key: str, default: int
) -> int:
    """Resolve a gate's hash_capture_level: inline cfg wins (flat / tests),
    else the rendered product (directory form), else *default*.

    The capture level decides which tensors the gate hashes (0 none, 1 loss +
    grad, 2 + module fwd/bwd), so it must track the gate's single source, not a
    bare default — a wrong level silently weakens the bitwise comparison.
    """
    if "hash_capture_level" in cfg:
        return int(cfg["hash_capture_level"])
    try:
        product = load_gate_product(repo_root, "ref", suite_key)
    except GateProductError:
        return default
    raw = product.get("hash_capture_level")
    return int(raw) if raw is not None else default


def _write_side_status(
    side_dir: Path,
    *,
    returncode: int,
    elapsed_s: float,
    timed_out: bool,
    timeout_s: int | None,
    cached: bool = False,
) -> None:
    side_dir.mkdir(parents=True, exist_ok=True)
    (side_dir / "status.json").write_text(
        json.dumps(
            {
                "returncode": returncode,
                "elapsed_s": elapsed_s,
                "timed_out": timed_out,
                "timeout_s": timeout_s,
                "cached": cached,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


# Harness-side scripts on the ref execution path, beyond the runner itself:
# run_gate.sh sources the shared library and evals the two projection tools,
# so their bytes shape the trajectory just like the runner's own.
_REF_RUNNER_DEPENDENCY_FILES = (
    _SCRIPTED_REF_RUNNER,
    "evals/scripts/gate_runner_common.sh",
    "evals/scripts/runtime_env.py",
    "tools/product_env.py",
)


def _ref_execution_dep_files(
    repo_root: Path, workload_config: dict[str, Any]
) -> list[tuple[str, Path]]:
    """Every file whose bytes shape a scripted trajectory-mode ref run.

    The environment fingerprint's git half is ``no-git`` inside loop
    workspaces (provisioned without ``.git``), so file-content changes MUST
    be hashed explicitly — the fingerprint alone cannot invalidate the
    cache there. The set = the harness-side runner chain
    (``_REF_RUNNER_DEPENDENCY_FILES``) + the whole read-only
    ``ref/reference/`` tree (the L0 launcher plus everything it sources or
    imports: train/model .py, data-conf .sh, loaders) + the resolved
    launcher when it lives outside that tree (customer ref scripts). The
    capture bridge is deliberately absent: hash-capture runs
    (``hash_capture_level > 0``) are never cached.
    """
    deps: list[tuple[str, Path]] = [(rel, repo_root / rel) for rel in _REF_RUNNER_DEPENDENCY_FILES]
    reference_dir = repo_root / "ref" / "reference"
    if reference_dir.is_dir():
        for path in sorted(reference_dir.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                deps.append((str(path.relative_to(repo_root)), path))
    try:
        launcher = Path(
            _resolve_customer_ref_script(repo_root, config_runtime.ref_script(workload_config))
        )
        if launcher not in {p for _, p in deps}:
            deps.append((f"ref_script:{launcher.name}", launcher))
    except Exception:
        # Unresolvable launcher config: keep the legacy behavior (compute a
        # key, run the sh, let it fail loudly) — hash a sentinel instead.
        deps.append(("ref_script:<unresolved>", repo_root / "__no_such_file__"))
    return deps


def _scripted_ref_cache_dir(
    repo_root: Path,
    suite_key: str,
    dep_files: list[tuple[str, Path]],
    product_path: Path,
) -> Path:
    """Cache key = SHA-256(gate, every script on the ref execution path,
    ref product bytes, env fingerprint).

    ``dep_files`` comes from :func:`_ref_execution_dep_files`; each entry is
    hashed with its label so moving identical bytes between files still
    changes the key.
    """
    hasher = hashlib.sha256()
    hasher.update(f"suite:{suite_key}\n".encode())
    for label, path in dep_files:
        hasher.update(f"file:{label}\n".encode())
        hasher.update(path.read_bytes() if path.is_file() else b"<missing>")
    hasher.update(b"product:\n")
    hasher.update(product_path.read_bytes())
    hasher.update(f"fingerprint:{_environment_fingerprint(repo_root)}\n".encode())
    return _ref_cache_base(repo_root) / f"scripted_{hasher.hexdigest()[:16]}"


def _copy_cache_files(src: Path, dst: Path) -> None:
    for rel in _SCRIPTED_REF_CACHE_FILES:
        src_file = src / rel
        if src_file.exists():
            dst_file = dst / rel
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)


def _run_scripted_ref(
    workload_config: dict[str, Any],
    repo_root: Path,
    ref_dir: Path,
    env: dict[str, str],
    *,
    suite_key: str,
) -> dict[str, Any] | None:
    """Run the ref step (with trajectory cache). Returns a failure result
    only for infrastructure errors (missing sh); gate-semantic failures are
    left on disk for the verdict module to judge."""
    ref_sh = repo_root / _SCRIPTED_REF_RUNNER
    if not ref_sh.exists():
        return missing_script_result(suite_key, ref_sh)
    ref_dir.mkdir(parents=True, exist_ok=True)

    cfg = workload_config["evals"][suite_key]
    ref_timeout = config_runtime.suite_ref_timeout_s(workload_config, suite_key) or 24 * 3600
    # Cache only trajectory-mode runs — hash-capture runs (level > 0) write
    # per-run dumps a cached copy would not have (parity with the legacy
    # run_via_ref_script policy).
    hash_level = _gate_hash_capture_level(cfg, repo_root, suite_key, 0)
    ref_product = repo_root / "ref" / "config" / f"{suite_key}.toml"
    use_cache = _ref_cache_enabled() and hash_level == 0 and ref_product.exists()
    cache_dir: Path | None = None
    if use_cache:
        cache_dir = _scripted_ref_cache_dir(
            repo_root,
            suite_key,
            _ref_execution_dep_files(repo_root, workload_config),
            ref_product,
        )
        if (cache_dir / "status.json").exists():
            _copy_cache_files(cache_dir, ref_dir)
            print(f"[ref-cache] HIT (scripted) for {suite_key} — reusing {cache_dir}")
            return None

    start = time.monotonic()
    returncode, _output = run_streaming_subprocess(
        ["bash", str(ref_sh), suite_key, str(ref_dir)],
        cwd=repo_root,
        env=env,
        timeout=ref_timeout,
        out_file_path=ref_dir / "ref.log",
    )
    _write_side_status(
        ref_dir,
        returncode=returncode,
        elapsed_s=time.monotonic() - start,
        timed_out=returncode == -1,
        timeout_s=ref_timeout,
    )

    loss_dump = ref_dir / "dump" / "ref_loss.txt"
    if use_cache and cache_dir is not None and returncode == 0 and loss_dump.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        _copy_cache_files(ref_dir, cache_dir)
        print(f"[ref-cache] SAVED (scripted) for {suite_key}")
    return None


def _run_scripted_suite(
    workload_config: dict[str, Any],
    repo_root: Path,
    artifact_dir: Path,
    *,
    suite_key: str,
    args: list[str] | None = None,
) -> dict[str, Any]:
    """Generic 3-step gate executor: ① ref sh, ② ours sh, ③ verdict module.

    ``args`` are the run request's extra positional args (e.g. the
    profile-snapshot out-label): passed to the ours sh as ``$3..`` and to the
    verdict as an ``args`` kwarg — only when non-empty, so plain gates keep
    the verified 4-kwarg verdict signature.
    """
    cfg = workload_config["evals"][suite_key]
    args = list(args or [])
    if cfg.get("requires_label", False) and not args:
        raise ValueError(
            f"suite {suite_key!r} requires an out-label argument, e.g. "
            f"'harness run {suite_key} long-horizon_round3 [<prev_label>]'."
        )
    env = base_env(repo_root)

    # ── ① ref side (optional) ──
    if cfg.get("needs_ref", False):
        failure = _run_scripted_ref(
            workload_config, repo_root, artifact_dir / "ref", env, suite_key=suite_key
        )
        if failure is not None:
            return failure

    # ── ② ours side ──
    runner = str(cfg["ours_runner"])
    if not runner or "/" in runner or "\\" in runner:
        raise ValueError(
            f"suite {suite_key!r}: ours_runner must be a plain basename under "
            f"evals/scripts/ (got {runner!r})"
        )
    ours_sh = repo_root / "evals" / "scripts" / f"{runner}.sh"
    if not ours_sh.exists():
        return missing_script_result(suite_key, ours_sh)
    ours_dir = artifact_dir / "ours"
    ours_dir.mkdir(parents=True, exist_ok=True)
    run_timeout = config_runtime.suite_timeout_s(workload_config, suite_key)
    start = time.monotonic()
    returncode, _output = run_streaming_subprocess(
        ["bash", str(ours_sh), suite_key, str(ours_dir), *args],
        cwd=repo_root,
        env=env,
        timeout=run_timeout,
        out_file_path=ours_dir / "ours.log",
    )
    _write_side_status(
        ours_dir,
        returncode=returncode,
        elapsed_s=time.monotonic() - start,
        timed_out=returncode == -1,
        timeout_s=run_timeout,
    )

    # ── ③ harness-side verdict (in-process; reads files only) ──
    verdict = str(cfg["verdict"])
    if not re.fullmatch(r"[a-z][a-z0-9_]*", verdict):
        raise ValueError(
            f"suite {suite_key!r}: verdict must be a bare evals.verdicts module "
            f"name (got {verdict!r})"
        )
    module = importlib.import_module(f"evals.verdicts.{verdict}")
    verdict_kwargs: dict[str, Any] = {}
    if args:
        verdict_kwargs["args"] = args
    return module.run(
        suite_key=suite_key,
        run_dir=artifact_dir,
        repo_root=repo_root,
        workload_config=workload_config,
        **verdict_kwargs,
    )


def _run_scripted_suite_kind(
    workload_config: dict[str, Any],
    repo_root: Path,
    artifact_dir: Path,
    suite: str,
    args: list[str],
) -> dict[str, Any]:
    return _run_scripted_suite(workload_config, repo_root, artifact_dir, suite_key=suite, args=args)


def run_suite(
    request: dict[str, Any],
    repo_root: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    suite = request["suite"]
    args = request.get("args", [])
    workload_config = request["workload_config"]
    suite_cfg = workload_config.get("evals", {}).get(suite)
    if not isinstance(suite_cfg, dict):
        raise ValueError(f"Unsupported suite: {suite}")
    # Routing: ``ours_runner`` selects the generic scripted executor
    # (every stage1 gate); ``runner_kind`` remains only for the exempt
    # stage2 op-* handlers.
    if isinstance(suite_cfg.get("ours_runner"), str) and suite_cfg["ours_runner"]:
        if not isinstance(suite_cfg.get("verdict"), str) or not suite_cfg["verdict"]:
            raise ValueError(f"Suite {suite} declares ours_runner but no verdict")
        invoker: Runner = _run_scripted_suite_kind
    else:
        runner_kind = suite_cfg.get("runner_kind")
        if not isinstance(runner_kind, str):
            raise ValueError(f"Suite {suite} does not declare runner_kind")
        try:
            invoker = RUNNER_KINDS[runner_kind]
        except KeyError as exc:
            raise ValueError(f"Unsupported runner_kind for suite {suite}: {runner_kind}") from exc

    result = _invoke_runner(invoker, workload_config, repo_root, artifact_dir, suite, args)
    if "schema_version" not in result:
        result = {"schema_version": run_schema.RUN_SCHEMA_VERSION, **result}
    return run_schema.validate_run_result(result)


def _invoke_runner(
    invoker: Any,
    workload_config: dict[str, Any],
    repo_root: Path,
    artifact_dir: Path,
    suite: str,
    args: list[str],
) -> dict[str, Any]:
    """Call runner kind, translating DeploymentPathError into a structured result."""
    try:
        return invoker(workload_config, repo_root, artifact_dir, suite, args)
    except config_runtime.DeploymentPathError as exc:
        return {
            "status": "failed",
            "suite": suite,
            "summary": str(exc),
            "metrics": {},
            "details": {
                "classification": "deployment_paths_missing",
                "missing_paths": exc.missing,
            },
        }


def stage_groups(workload_config: dict[str, Any]) -> dict[str, list[str]]:
    """Derive suite stage groups from the workload config metadata."""
    return config_runtime.suites_by_stage(workload_config)
