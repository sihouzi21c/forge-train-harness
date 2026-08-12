"""Legacy ``RUNNER_KINDS`` suite handlers — thin-dispatcher refactor EXEMPT.

The op-inventory / op-long / op-status gates keep their legacy in-process
handler shape (plan §10 step 7). These are the only suites still routed via
``RUNNER_KINDS`` — every other gate (stage1) goes through the generic
scripted executor in ``evals.dispatcher`` (registry keys ``needs_ref`` /
``ours_runner`` / ``verdict``). This module owns the legacy handlers plus
the helpers only they still use (``suite_process_env`` env lifting,
ref-trajectory resolution, product-shape reads, the stage-2 GPU flock).

``evals.dispatcher`` re-exports ``RUNNER_KINDS`` from here, so external
callers/tests keep addressing ``dispatcher.RUNNER_KINDS``.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import subprocess
from pathlib import Path
from typing import Any

from evals._common import (
    missing_script_result,
    missing_window_steps,
    output_tail_limit,
    parse_loss_lines_to_dict,
    run_streaming_subprocess,
    stage2_runtime_inputs,
    suite_process_env,
    torchrun_cmd,
    window_loss_diff_metrics,
)
from evals.gate_common import resolve_ref_trajectory
from evals.gate_product import GateProductError
from evals.gate_shape import GateShape, load_gate_shape
from harness import config_runtime

__all__ = ["RUNNER_KINDS"]


def _tail_output(output: str, limit: int | None = None) -> str:
    """Return at most *limit* trailing characters (defaults to ``output_tail_limit()``)."""
    cap = limit if limit is not None else output_tail_limit()
    return output[-cap:] if len(output) > cap else output


def _run_op_long_kind(
    workload_config: dict[str, Any],
    repo_root: Path,
    artifact_dir: Path,
    suite: str,
    args: list[str],
) -> dict[str, Any]:
    del suite
    # Subagent-driven op-long calls are concurrency-safe by virtue of this
    # advisory file lock: multiple ``harness run op-long`` invocations queue
    # up on the same lock file and run strictly serially per GPU slice. The
    # lock is held for the full duration of the gate (ref-trajectory +
    # ours-trajectory), since both phases occupy the GPUs in the slice.
    #
    # GPU slicing (#17): when ``HARNESS_OP_LONG_GPU_SLICE`` is set
    # (usually a stable identifier derived from CUDA_VISIBLE_DEVICES,
    # e.g. ``"0,1,2,3"``), the lock name embeds it so two callers on
    # disjoint GPU slices run in parallel instead of blocking each other.
    # Falls back to a single global lock when the env is unset.
    slice_id = os.environ.get("HARNESS_OP_LONG_GPU_SLICE", "").strip()
    if not slice_id:
        slice_id = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if slice_id:
        sanitized = "".join(c if c.isalnum() else "_" for c in slice_id)[:64]
        lock_name = f"stage2-gpu-{sanitized}.lock"
    else:
        lock_name = "stage2-gpu.lock"
    with _stage2_file_lock(repo_root, lock_name):
        return _run_op_long(workload_config, repo_root, artifact_dir, op_names=args)


@contextlib.contextmanager
def _stage2_file_lock(repo_root: Path, lock_name: str):
    """POSIX advisory file lock for serializing Stage 2 GPU resources.

    ``stage2-gpu.lock`` serializes ``op-long`` invocations (the GPU is a
    shared exclusive resource). The main-branch atomicity needed at merge
    time is **not** owned by harness — subagents serialize their own
    ``git merge`` via ``flock <common-dir>/locks/stage2-main.lock`` in a
    shell wrapper (see ``prompt/develop_prompt/_shared/stage2-subagent-playbook.md``,
    "3b PASS 后：自驱合入主干" section).

    Lock directory placement: lives under the **git common dir**
    (``git rev-parse --git-common-dir``), which is the main worktree's
    ``.git/`` for every worktree of the same repository. Every
    ``ops_worktree/<op>/`` subagent therefore sees the same lock file and
    can contend with the main worktree's ``op-long`` invocation. The
    legacy ``<worktree>/.artifacts/locks/`` location used to silently
    create per-worktree lock files, so concurrent subagents thought
    they were each holding the lone GPU lock and stomped on each other.
    """
    lock_dir = _git_common_dir(repo_root) / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / lock_name
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _git_common_dir(repo_root: Path) -> Path:
    """Return the absolute path of the git common dir for ``repo_root``.

    For the main worktree this is ``<repo_root>/.git``; for a worktree
    created via ``git worktree add`` it is the main repo's ``.git/`` path
    (same on disk across every worktree of the same clone). Used by
    ``_stage2_file_lock`` so that flock contention is shared across all
    worktrees of a single repository.
    """
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=True,
    )
    common_dir = Path(result.stdout.strip())
    if not common_dir.is_absolute():
        common_dir = (repo_root / common_dir).resolve()
    return common_dir


def _maybe_gate_shape(suite_key: str, workspace: Path, side: str = "ref") -> GateShape | None:
    """Return the rendered product shape for this gate/side, or ``None``.

    Collapse Phase 1: the legacy ``FORGE_GATE_SHAPE_SOURCE`` toggle is gone —
    the rendered product is now the *unconditional* shape source. Every
    Stage-1 gate carries ref + ours products, so callers get a real
    ``GateShape``. The only ``None`` case left is a suite with no gate_config
    product — the D5-exempt Stage-2 ``op-long`` — which transparently stays on
    its legacy ``ref → gate_metadata.json`` path. ``side="ours"`` reads the
    ours product (its micro_batch_size may differ from ref's).
    """
    try:
        return load_gate_shape(workspace, suite_key, side)
    except GateProductError:
        return None


def _ref_metadata_int(ref_run: Any, key: str, shape: GateShape | None = None) -> int:
    # Collapse: Stage-1 gates always pass a product-backed ``shape`` — read the
    # value straight from the rendered ref product. The ``ref_run.metadata``
    # fallback below now serves ONLY the product-less, D5-exempt op-long suite.
    if shape is not None and shape.has(key):
        return shape.metadata_int(key)
    metadata = getattr(ref_run, "metadata", {}) or {}
    try:
        return int(metadata[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"L0 ref script did not emit integer gate metadata: {key}") from exc


def _suite_ours_batch_shape(
    ref_run: Any,
    cfg: dict[str, Any],
    shape: GateShape | None = None,
    ours_shape: GateShape | None = None,
) -> tuple[int, int, int]:
    """Return (micro_batch_size, global_batch_size, grad_accum_steps) for ours.

    The ours micro_batch_size may differ from ref's (a gate can run ours at a
    larger MBS while keeping the ref trajectory's GBS so both sides stay
    aligned on global batch size; grad_accum is recomputed). Source order for
    the ours MBS: the rendered ours product (``ours_shape``, the migration
    target) → inline ``cfg.ours_env.MICRO_BATCH_SIZE_OVERRIDE`` (flat) → the
    ref MBS (gates with no ours override).
    """
    world_size = _ref_metadata_int(ref_run, "world_size", shape)
    global_batch_size = _ref_metadata_int(ref_run, "global_batch_size", shape)
    ours_env = cfg.get("ours_env")
    if ours_shape is not None and ours_shape.has("micro_batch_size"):
        micro_batch_size = ours_shape.metadata_int("micro_batch_size")
    elif ours_env is not None:
        try:
            micro_batch_size = int(ours_env["MICRO_BATCH_SIZE_OVERRIDE"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("suite ours_env must declare MICRO_BATCH_SIZE_OVERRIDE") from exc
    else:
        micro_batch_size = _ref_metadata_int(ref_run, "micro_batch_size", shape)
    if micro_batch_size * world_size == 0:
        raise ValueError("MBS * WORLD_SIZE must be > 0")
    grad_accum = global_batch_size // (micro_batch_size * world_size)
    if grad_accum < 1:
        raise ValueError(
            f"GBS {global_batch_size} / (MBS {micro_batch_size} * "
            f"WS {world_size}) = {grad_accum} < 1"
        )
    return micro_batch_size, global_batch_size, grad_accum


def _ref_gate_steps(ref_run: Any, shape: GateShape | None = None) -> range:
    return range(
        _ref_metadata_int(ref_run, "gate_window_start", shape),
        _ref_metadata_int(ref_run, "gate_window_end", shape),
    )


def _run_op_inventory(
    workload_config: dict[str, Any],
    repo_root: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    """M1 gate — validate scout-produced artifacts (no enumeration).

    See ``prompt/develop_prompt/stage2.md`` M1.6: this gate only checks
    that the main agent has produced a coherent set of registry / per-op
    scaffolding / dispatcher wiring / worktree state. It does NOT enumerate
    operators on its own — operator selection is the main agent's
    responsibility in M1.1-M1.5.
    """
    del workload_config, artifact_dir
    return _run_stage2_validation(
        repo_root=repo_root, validate_worktrees=True, suite="op-inventory"
    )


def _run_stage2_validation(
    *, repo_root: Path, validate_worktrees: bool, suite: str
) -> dict[str, Any]:
    from harness._compat import tomllib

    errors: list[str] = []
    registry_path = config_runtime.ops_registry_path(repo_root)
    if not registry_path.exists():
        errors.append(f"registry missing: {registry_path.relative_to(repo_root)}")
        return _stage2_validation_result(suite, errors, op_names=[], registry_path=registry_path)

    with open(registry_path, "rb") as fh:
        registry = tomllib.load(fh)
    operators = registry.get("operators", {})
    if not operators:
        errors.append("registry has no [operators.*] entries")
        return _stage2_validation_result(suite, errors, op_names=[], registry_path=registry_path)

    op_names = sorted(operators.keys(), key=lambda n: operators[n].get("priority", 999))
    for op_name in op_names:
        errors.extend(_validate_op_artifacts(repo_root, op_name))
        errors.extend(_validate_op_prompt_reference(repo_root, op_name))
        if validate_worktrees:
            errors.extend(_validate_op_worktree(repo_root, op_name))

    errors.extend(_validate_dispatcher_wiring(repo_root))
    return _stage2_validation_result(suite, errors, op_names=op_names, registry_path=registry_path)


def _validate_op_artifacts(repo_root: Path, op_name: str) -> list[str]:
    from harness._compat import tomllib

    errors: list[str] = []
    op_path = config_runtime.op_dir(repo_root, op_name)
    register_path = config_runtime.op_register_path(repo_root, op_name)
    # ``register.toml`` is included via the SSOT helper rather than a
    # second literal — same name, single source.
    required = (
        "PROMPT.md",
        "BASELINE.md",
        "kernel.py",
        register_path.name,
        "test_op.py",
        "notes.md",
        "__init__.py",
    )
    for fname in required:
        if not (op_path / fname).exists():
            errors.append(f"{op_name}: missing {fname}")
    if register_path.exists():
        try:
            with open(register_path, "rb") as fh:
                reg = tomllib.load(fh)
        except Exception as exc:
            errors.append(f"{op_name}: register.toml parse error: {exc}")
            return errors
        if not isinstance(reg.get("env_var"), str) or not reg["env_var"]:
            errors.append(f"{op_name}: register.toml missing non-empty env_var")
        default = reg.get("default")
        if default != "baseline":
            errors.append(
                f"{op_name}: register.toml default must be 'baseline' at M1 (got {default!r})"
            )
        available = reg.get("available")
        if not isinstance(available, list) or "baseline" not in available:
            errors.append(f"{op_name}: register.toml available must include 'baseline'")
    return errors


def _reference_path_for_op(repo_root: Path, op_name: str) -> Path | None:
    """Map a Stage 2 op-name to its domain reference file.

    Stage 2 only optimises FlashAttention + the various GEMM call-sites
    (``stage2.md`` "概述"). ``attention`` is verbatim attention.md; every
    ``gemm_*`` site shares the same gemm.md. Returns ``None`` for any
    op-name outside that whitelist; callers fail the op-inventory gate
    from ``_validate_op_prompt_reference`` on a ``None`` mapping.
    """
    if op_name == "attention":
        return repo_root / "prompt" / "develop_prompt" / "_shared" / "reference" / "attention.md"
    if op_name.startswith("gemm_") or op_name == "gemm":
        return repo_root / "prompt" / "develop_prompt" / "_shared" / "reference" / "gemm.md"
    return None


def _validate_op_prompt_reference(repo_root: Path, op_name: str) -> list[str]:
    """Ensure ``PROMPT.md`` embeds the full ``reference/<op>.md`` verbatim.

    Stage 2 M1.5.1 mandates that the "optimization direction" section of
    every ``workload/ops/<name>/PROMPT.md`` is a byte-for-byte copy of
    the matching reference file (``reference/attention.md`` for
    ``attention``; ``reference/gemm.md`` for any ``gemm_*`` site). This
    is the only stable, reproducible way to feed identical domain
    knowledge to every subagent.
    """
    errors: list[str] = []
    ref_path = _reference_path_for_op(repo_root, op_name)
    if ref_path is None:
        errors.append(
            f"{op_name}: out-of-scope for Stage 2 — only 'attention' and "
            "'gemm_*' sites are allowed (see stage2.md '概述')"
        )
        return errors
    if not ref_path.exists():
        errors.append(f"{op_name}: reference file missing: {ref_path.relative_to(repo_root)}")
        return errors
    prompt_path = repo_root / "workload" / "ops" / op_name / "PROMPT.md"
    if not prompt_path.exists():
        return errors
    try:
        ref_text = ref_path.read_text(encoding="utf-8")
        prompt_text = prompt_path.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"{op_name}: cannot read PROMPT.md or reference: {exc}")
        return errors
    if ref_text not in prompt_text:
        errors.append(
            f"{op_name}: PROMPT.md does not embed "
            f"{ref_path.relative_to(repo_root)} verbatim "
            "(stage2.md M1.5.1 requires a byte-for-byte cat of the full "
            "reference file into the 'optimization direction' section)"
        )
    return errors


def _validate_op_worktree(repo_root: Path, op_name: str) -> list[str]:
    errors: list[str] = []
    wt_path = config_runtime.op_worktree_path(repo_root, op_name)
    if not wt_path.exists():
        errors.append(f"{op_name}: {wt_path.relative_to(repo_root).as_posix()} missing")
        return errors
    branch = _git_text(["rev-parse", "--abbrev-ref", "HEAD"], cwd=wt_path)
    expected_branch = f"stage2/op/{op_name}"
    if branch != expected_branch:
        errors.append(f"{op_name}: worktree HEAD on {branch!r}, expected {expected_branch!r}")
    sparse_list = _git_text(["sparse-checkout", "list"], cwd=wt_path).splitlines()
    sparse_set = {line.strip() for line in sparse_list if line.strip()}
    # Sparse-checkout patterns are POSIX paths relative to repo root; the
    # SSOT helpers return them as ``Path`` objects so we normalise here.
    op_target = config_runtime.op_dir(repo_root, op_name).relative_to(repo_root).as_posix()
    if op_target not in sparse_set:
        errors.append(f"{op_name}: sparse-checkout missing {op_target}")
    workload_src_target = (
        config_runtime.workload_src_path(repo_root).relative_to(repo_root).as_posix()
    )
    for required in ("harness", workload_src_target):
        if required not in sparse_set:
            errors.append(f"{op_name}: sparse-checkout missing read-only background {required}")
    return errors


def _validate_dispatcher_wiring(repo_root: Path) -> list[str]:
    """Require at least one ``get_op_version(`` call somewhere in workload/src.

    Stage 2 only optimises FlashAttention + the various GEMM call-sites; those
    live in a subset of ``forward.py`` / ``backward.py`` / ``kernels.py``
    (``triton_kernels.py`` typically hosts RoPE/RMSNorm/SwiGLU/etc. which are
    **out of scope** for Stage 2). A dispatcher branch in every file is
    therefore not required — we just need evidence that wiring exists at
    all. The M1.3 smoke (``OP_*=baseline`` forward) is responsible for
    proving the wiring actually routes to baseline.
    """
    errors: list[str] = []
    src_dir = config_runtime.workload_src_path(repo_root) / "training_engine_tensor"
    if not src_dir.exists():
        return [f"dispatcher: {src_dir.relative_to(repo_root).as_posix()} missing"]
    total_hits = 0
    for fpath in sorted(src_dir.rglob("*.py")):
        try:
            content = fpath.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"dispatcher: cannot read {fpath.relative_to(repo_root)}: {exc}")
            continue
        total_hits += content.count("get_op_version(")
    if total_hits == 0:
        errors.append(
            "dispatcher: no get_op_version() call found anywhere under "
            f"{src_dir.relative_to(repo_root).as_posix()}/ (Stage 2 wiring missing)"
        )
    return errors


def _stage2_validation_result(
    suite: str, errors: list[str], *, op_names: list[str], registry_path: Path
) -> dict[str, Any]:
    passed = not errors
    return {
        "status": "passed" if passed else "failed",
        "suite": suite,
        "summary": (
            f"M1 inventory: {len(op_names)} operator(s) validated, no errors."
            if passed
            else f"M1 inventory: {len(errors)} validation error(s) across {len(op_names)} op(s)."
        ),
        "metrics": {
            "operators_registered": len(op_names),
            "validation_errors": len(errors),
            "registry_path": str(registry_path),
        },
        "details": {
            "errors": errors,
            "operators": op_names,
        },
    }


def _git_text(args: list[str], *, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _git_branch_exists(repo_root: Path, branch: str) -> bool:
    result = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _run_op_long(
    workload_config: dict[str, Any],
    repo_root: Path,
    artifact_dir: Path,
    *,
    op_names: list[str] | None = None,
) -> dict[str, Any]:
    if op_names is None:
        op_names = []
    suite_cfg = workload_config.get("evals", {}).get("op-long", {})
    cfg = {
        **workload_config.get("stage2", {}),
        **suite_cfg,
    }
    rel_threshold = float(cfg.get("rel_diff_threshold", 0.01))

    script = repo_root / "evals" / "scripts" / "op_long_ours.py"
    if not script.exists():
        return missing_script_result("op-long", script)

    names_str = ",".join(op_names) if op_names else "all"

    ref = resolve_ref_trajectory(
        repo_root=repo_root,
        suite_key="op-long",
        result_suite=f"op-long:{names_str}",
        summary_prefix="op-long",
        cfg=cfg,
        artifact_dir=artifact_dir,
    )
    if ref.failure is not None:
        return ref.failure
    ref_result = ref.ref_result
    ref_run = ref.ref_run
    baseline_by_step = ref.loss_by_step
    assert ref_run is not None
    shape = _maybe_gate_shape("op-long", repo_root)
    gate_steps = _ref_gate_steps(ref_run, shape)
    gate_lo, gate_hi = gate_steps.start, gate_steps.stop
    num_steps = _ref_metadata_int(ref_run, "num_steps", shape)
    world_size = _ref_metadata_int(ref_run, "world_size", shape)
    seed = _ref_metadata_int(ref_run, "seed", shape)
    seq_length = _ref_metadata_int(ref_run, "seq_length", shape)
    micro_batch_size, gbs, grad_accum = _suite_ours_batch_shape(
        ref_run, suite_cfg, shape, _maybe_gate_shape("op-long", repo_root, side="ours")
    )

    # Phase 2: ours trajectory
    op_long_extra = {
        **stage2_runtime_inputs(workload_config),
        "OP_NAMES": names_str,
        "NUM_STEPS": str(num_steps),
        "GLOBAL_BATCH_SIZE": str(gbs),
        "MICRO_BATCH_SIZE": str(micro_batch_size),
        "SEED": str(seed),
        "SEQ_LENGTH": str(seq_length),
        "GRAD_ACCUM_STEPS": str(grad_accum),
    }
    env = suite_process_env(
        repo_root,
        suite_cfg,
        workload_config,
        world_size=world_size,
        suite_key="op-long",
        extra=op_long_extra,
    )
    # Per-op routing (#18): for every op in OP_NAMES, export
    # ``OP_<NAME>=<latest_available variant>`` so the engine's
    # ``get_op_version`` dispatch picks the optimised path; ops outside
    # the list inherit their registry default ("baseline" pre-merge).
    # These keys are dynamic (depend on workload/ops/_registry.toml) and
    # therefore intentionally NOT declared in env_inputs — they go in
    # post-validation so a new operator added to the registry doesn't
    # require a dense_training.toml edit just to switch its variant on.
    env.update(_per_op_variant_env(repo_root, op_names))
    # GPU slicing (#17): if the caller pinned CUDA_VISIBLE_DEVICES on the
    # parent shell (typical when running multiple op-long invocations on
    # disjoint GPU slices), forward it to the ours subprocess so the
    # ref/ours pair stay on the same device set. The flock contract
    # outside this function honours ``HARNESS_OP_LONG_GPU_SLICE`` to
    # serialize per-slice instead of globally.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        env["CUDA_VISIBLE_DEVICES"] = cvd

    timeout = config_runtime.suite_timeout_s(workload_config, "op-long")
    out_file_path = artifact_dir / "op_long_output.log"
    returncode, output = run_streaming_subprocess(
        torchrun_cmd(
            world_size,
            str(script),
            master_addr=env["MASTER_ADDR"],
            master_port=env["MASTER_PORT"],
        ),
        cwd=repo_root,
        env=env,
        timeout=timeout,
        out_file_path=out_file_path,
    )
    if returncode == -1:
        return {
            "status": "failed",
            "suite": f"op-long:{names_str}",
            "summary": f"op-long: ours-side timed out after {timeout}s",
            "metrics": {"world_size": world_size},
            "details": {"config": cfg, "output_tail": _tail_output(output)},
        }

    ours_by_step = parse_loss_lines_to_dict(output)
    missing_steps = missing_window_steps(ours_by_step, gate_steps)
    if missing_steps:
        return {
            "status": "failed",
            "suite": f"op-long:{names_str}",
            "summary": f"op-long: ours emitted no [LOSS] for {len(missing_steps)} steps. First: {missing_steps[:10]}.",
            "metrics": {"world_size": world_size, "ours_steps_observed": len(ours_by_step)},
            "details": {"config": cfg, "output_tail": _tail_output(output)},
        }

    try:
        diff_metrics = window_loss_diff_metrics(
            baseline_by_step,
            ours_by_step,
            gate_steps,
        )
    except ValueError as exc:
        return {
            "status": "failed",
            "suite": f"op-long:{names_str}",
            "summary": f"op-long: {exc}.",
            "metrics": {"world_size": world_size},
            "details": {"config": cfg},
        }
    mean_rel = float(diff_metrics["mean_rel_diff"])
    max_rel = float(diff_metrics["max_rel_diff"])
    signed_mean = float(diff_metrics["signed_mean"])
    signed_mean_rel = float(diff_metrics["signed_mean_rel"])
    # Boundary semantics are shared with ``long-train`` and ``loss-gate``:
    # all three encode the "≤ 1% relative loss diff" verbal contract as a
    # *strict* less-than, so a trajectory landing exactly on the threshold
    # fails. The cross-suite invariant is asserted by
    # ``test_loss_gate_threshold_directions_are_consistent`` in
    # test_dispatcher_behavior.
    passed = mean_rel < rel_threshold and returncode == 0

    return {
        "status": "passed" if passed else "failed",
        "suite": f"op-long:{names_str}",
        "summary": f"op-long [{names_str}]: [{gate_lo},{gate_hi}) mean_rel_diff={mean_rel:.6f} {'<' if mean_rel < rel_threshold else '≥'} {rel_threshold}, signed_rel={signed_mean_rel * 100:+.4f}% (signed_mean={signed_mean:+.4e})",
        "metrics": {
            "op_names": op_names,
            "world_size": world_size,
            "num_steps": num_steps,
            "gate_window": [gate_lo, gate_hi],
            "compared_steps": diff_metrics["compared_steps"],
            "mean_rel_diff": mean_rel,
            "max_rel_diff": max_rel,
            "signed_mean": signed_mean,
            "signed_mean_rel": signed_mean_rel,
            "rel_threshold": rel_threshold,
            "ref_elapsed_s": ref_run.elapsed_s,
            "passed": passed,
        },
        "details": {
            "config": cfg,
            "ref_dump_dir": str(ref_result["dump_dir"]) if ref_result is not None else None,
            "step_diffs_tail": diff_metrics["step_diffs"][-20:],
            "output_tail": _tail_output(output),
        },
    }


def _per_op_variant_env(repo_root: Path, op_names: list[str]) -> dict[str, str]:
    """Return ``OP_<NAME>=<latest_available variant>`` env vars for *op_names*.

    Implements the per-op routing contract referenced by
    ``evals/scripts/op_long_ours.py``: every op listed in ``OP_NAMES``
    gets its registry-declared variant switch (``register.toml`` →
    ``env_var`` + last entry of ``available``) flipped on, while ops
    outside the list inherit their registry default ("baseline" pre-merge).

    Skips silently on:
      * empty ``op_names`` (no per-op switches needed; baseline run)
      * missing registry (caller surfaces the failure via op-inventory)
      * unknown op-name (caller surfaces via summary; we don't fabricate
        a variant switch, the gate will run the registry default).
    """
    if not op_names:
        return {}
    from harness._compat import tomllib

    registry_path = config_runtime.ops_registry_path(repo_root)
    if not registry_path.exists():
        return {}
    try:
        with open(registry_path, "rb") as fh:
            registry = tomllib.load(fh)
    except Exception:
        return {}
    operators = registry.get("operators", {})
    out: dict[str, str] = {}
    for op_name in op_names:
        if op_name not in operators:
            continue
        register_path = config_runtime.op_register_path(repo_root, op_name)
        if not register_path.exists():
            continue
        try:
            with open(register_path, "rb") as fh:
                reg = tomllib.load(fh)
        except Exception:  # nosec B112 — registry parse failure is non-fatal; skip and continue
            continue
        env_var = reg.get("env_var")
        available = reg.get("available", [])
        if (
            isinstance(env_var, str)
            and env_var
            and isinstance(available, list)
            and len(available) >= 2
        ):
            # Pick the last non-baseline variant; with M1.5 register.toml
            # convention this is the optimised path under test.
            target = next(
                (v for v in reversed(available) if isinstance(v, str) and v != "baseline"),
                None,
            )
            if target is not None:
                out[env_var] = target
    return out


def _run_op_status(
    workload_config: dict[str, Any],
    repo_root: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    """M2 gate — aggregate per-op state from git + register.toml truth.

    Status is derived without reading signal files. The truth sources are:
    1. ``workload/ops/<op>/register.toml`` ``default`` field (post-merge marker)
    2. Existence of ``ops_worktree/<op>/`` directory
    3. Existence of ``stage2/op/<op>`` branch
    """
    del workload_config, artifact_dir

    from harness._compat import tomllib

    registry_path = config_runtime.ops_registry_path(repo_root)
    if not registry_path.exists():
        return {
            "status": "failed",
            "suite": "op-status",
            "summary": "registry missing — run op-inventory first.",
            "metrics": {"operators": [], "summary_counts": {}},
            "details": {},
        }

    with open(registry_path, "rb") as fh:
        registry = tomllib.load(fh)
    operators = registry.get("operators", {})

    entries: list[dict[str, Any]] = []
    for op_name in sorted(operators.keys(), key=lambda n: operators[n].get("priority", 999)):
        entries.append(_status_entry_for_op(repo_root, op_name, operators[op_name]))

    status_keys = ("merged", "failed", "in_progress", "not_started", "inconsistent")
    summary_counts = {key: sum(1 for e in entries if e["status"] == key) for key in status_keys}

    summary_line = f"{len(entries)} operator(s): " + ", ".join(
        f"{key}={summary_counts[key]}" for key in status_keys if summary_counts[key]
    )
    if all(v == 0 for v in summary_counts.values()):
        summary_line = f"{len(entries)} operator(s) registered."

    # Suite-level ``status`` mirrors the failure-bearing terminal
    # counts so the agent-loop review gate can read a single boolean
    # without having to special-case op-status, while the per-op
    # verdict still lives in ``metrics.operators[]`` /
    # ``metrics.summary_counts``. Lifecycle states (``not_started`` /
    # ``in_progress``) intentionally do NOT trip the gate — the M2
    # round expects to start in those states and only progresses when
    # subagents drive ops to ``merged``.
    failure_count = summary_counts["failed"] + summary_counts["inconsistent"]
    suite_status = "failed" if failure_count > 0 else "passed"

    return {
        "status": suite_status,
        "suite": "op-status",
        "summary": summary_line,
        "metrics": {
            "operators": entries,
            "summary_counts": summary_counts,
        },
        "details": {},
    }


def _status_entry_for_op(repo_root: Path, op_name: str, op_info: dict[str, Any]) -> dict[str, Any]:
    from harness._compat import tomllib

    register_path = config_runtime.op_register_path(repo_root, op_name)
    wt_path = config_runtime.op_worktree_path(repo_root, op_name)
    branch = f"stage2/op/{op_name}"
    branch_exists = _git_branch_exists(repo_root, branch)

    default = "baseline"
    available: list[str] = ["baseline"]
    if register_path.exists():
        try:
            with open(register_path, "rb") as fh:
                reg = tomllib.load(fh)
            default = str(reg.get("default", "baseline"))
            available = list(reg.get("available", ["baseline"]))
        except Exception:  # nosec B110 — best-effort metadata lookup; missing registry is non-fatal
            pass

    has_worktree = wt_path.exists()
    if default != "baseline" and not has_worktree and not branch_exists:
        status = "merged"
    elif has_worktree and branch_exists:
        # subagent crashed / op-long FAIL / merge aborted — worktree retained for review
        status = "failed"
    elif default == "baseline" and not has_worktree and not branch_exists:
        status = "not_started"
    else:
        status = "inconsistent"

    entry: dict[str, Any] = {
        "name": op_name,
        "status": status,
        "category": op_info.get("category", "unknown"),
        "priority": op_info.get("priority", 999),
        "default": default,
        "available": available,
    }
    if has_worktree:
        entry["worktree"] = str(wt_path.relative_to(repo_root))
    if branch_exists:
        entry["branch"] = branch
    return entry


# Legacy routing table — stage2 op-* only. Consumed by ``evals.dispatcher``
# (re-exported there); a suite lands here ONLY when its registry entry
# declares no ``ours_runner``.
RUNNER_KINDS: dict[str, Any] = {
    "op-inventory": lambda wc, rr, ad, suite, args: _run_op_inventory(wc, rr, ad),
    "op-long": _run_op_long_kind,
    "op-status": lambda wc, rr, ad, suite, args: _run_op_status(wc, rr, ad),
}
