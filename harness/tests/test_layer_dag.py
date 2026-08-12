"""Architectural linter: lock the harness/ layer DAG statically.

This is a tiny AST-based import checker that replaces a full
`import-linter` dependency. It guarantees:
  - The layer graph is acyclic and one-directional across
    ``harness/`` (leaf), ``evals/`` (control plane), and the
    control-plane portion of ``tools/``.
  - `harness/` is a leaf package: it imports ONLY from the standard
    library plus its own intra-package modules (no `workload`, `tools`,
    `evals`, or any third-party package).
  - `evals/` may import `harness/` and `tools/` (one-way, top-down)
    and its own intra-package modules; nothing else.
  - Control-plane `tools/*.py` may import `harness/` and other
    control-plane tools, but not `evals/` or `workload/`.

A breach here means someone violated the UNIDIRECTIONAL DEPENDENCY or
MINIMAL PUBLIC SURFACE red line silently — exactly what the existing
framework_guard cannot catch.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_DIR = REPO_ROOT / "harness"
EVALS_DIR = REPO_ROOT / "evals"
TOOLS_DIR = REPO_ROOT / "tools"

# Allowed intra-harness dependency edges. A module listed here may ONLY
# import from the set of modules given; anything else is a violation.
LAYER_ALLOWED_DEPS: dict[str, frozenset[str]] = {
    "harness": frozenset(),
    "harness._compat": frozenset(),
    # config_runtime re-exports validate_safe_relative_path from run_schema
    # (the canonical owner of schema-level vocabulary) so we keep exactly
    # one definition of that validator. run_schema remains a leaf.
    "harness.config_runtime": frozenset({"harness._compat", "harness.run_schema"}),
    "harness.run_schema": frozenset(),
    # Per-loop directory layout SSOT (config/workspace topology under
    # <forge_train_dir>/<id>/). Pure-stdlib leaf consumed by the web
    # layer and resolved from agent-loop.sh so the path formula has one
    # authoritative source.
    "harness.loop_layout": frozenset(),
    # Wire-format SSOT: stdout/dump line grammars + trajectory parsers.
    # Pure-stdlib leaf so both ``evals/`` and ``tools/`` can depend on
    # it without inducing a cycle.
    "harness.wire_format": frozenset(),
    # Capture-artifact wire-format SSOT (M1.fwd / M1.bwd file-oriented
    # protocol; sibling of ``harness.wire_format`` for the
    # line-oriented protocol). Pure-stdlib leaf consumed by
    # ``evals._common`` so the dispatcher and the per-baseline ref
    # scripts pin one disk contract.
    "harness.capture_artifacts": frozenset(),
    "harness.transport": frozenset({"harness.config_runtime", "harness.run_schema"}),
    "harness.app": frozenset(
        {
            "harness.config_runtime",
            "harness.resources",
            "harness.run_schema",
            "harness.transport",
            "harness.workspace_contract",
        }
    ),
    "harness.presentation": frozenset(),
    # cli is the top-level dispatcher; besides delegating to app /
    # presentation / remote_sync it bootstraps process-global data-axis
    # env defaults via config_runtime.ensure_forge_data_env_defaults()
    # before any command loads workload config. config_runtime is the
    # lowest leaf, so this is a strictly downward edge (no cycle).
    "harness.cli": frozenset(
        {"harness.app", "harness.presentation", "harness.remote_sync", "harness.config_runtime"}
    ),
    # Workspace-contract invariants run before the agent starts. The
    # bare-``harness`` import is intentional — the contract reads
    # ``harness.__file__`` to verify that the imported package lives
    # under the current workspace (catching bitwise-singlecard/bitwise-perf editable-install
    # shadowing). config_runtime supplies the canonical workload-config
    # loader + sha used by the ``subprocess_config_isomorphic`` invariant
    # to compare parent vs. ``harness echo-config`` child bytes.
    "harness.workspace_contract": frozenset({"harness", "harness.config_runtime"}),
    # Banned-import scanner + path isolation guard. Depends on
    # config_runtime for the canonical repo_root() resolution.
    "harness.framework_guard": frozenset({"harness.config_runtime"}),
    # Sister of framework_guard: scans workload/src/ + workload/ops/
    # for proxy / synthesis patterns. Reuses framework_guard's
    # GuardViolation / IGNORED_DIR_NAMES / repo_root primitives so it
    # depends on framework_guard directly.
    "harness.anti_proxy_guard": frozenset({"harness.framework_guard"}),
    "harness.remote_sync": frozenset({"harness.config_runtime"}),
    # Declared external-resource manifest. Depends on
    # config_runtime.repo_root() to locate the manifest at workspace
    # root; otherwise stdlib-only (hashlib / subprocess for git
    # rev-parse). Layered ABOVE config_runtime — config_runtime
    # imports it lazily inside the legacy ``_resolve_*`` bridges so
    # there is no circular import.
    "harness.resources": frozenset({"harness._compat", "harness.config_runtime"}),
}

# Allowed edges for control-plane modules under ``evals/`` (excluding
# ``evals/scripts/`` data-plane scripts, which run as subprocesses and
# may freely depend on ``training_engine_tensor`` etc.). Each entry maps
# a module to its permitted ``harness.*``, ``evals.*``, and ``tools.*``
# imports; standard library is always allowed.
EVALS_LAYER_ALLOWED_DEPS: dict[str, frozenset[str]] = {
    "evals": frozenset(),
    "evals._common": frozenset(
        {
            # gate_product / product_env: the ref-side env assembly reads the
            # rendered ref product and projects it via the generic
            # tools/product_env export_map (evals -> tools is the allowed
            # one-way direction; both are stdlib-only leaves).
            "evals.gate_product",
            "harness.capture_artifacts",
            "harness.config_runtime",
            "harness.wire_format",
            "tools.product_env",
            "tools.ref_script_runner",
        }
    ),
    # Generic M1 alignment compare driver. Imports ``torch`` only (not
    # tracked here) — no harness/evals/tools dependencies, since it is
    # the architecture-agnostic intersect-and-diff loop shared by the
    # M1.fwd / M1.bwd gate scripts under ``evals/scripts/``.
    "evals._capture_diff": frozenset(),
    # Candidate-importable async D2H hash-capture offload (torch-only leaf).
    "evals.capture_offload": frozenset(),
    # Rendered gate-product loader (stdlib-only leaf) + its shape accessor.
    "evals.gate_product": frozenset(),
    "evals.gate_shape": frozenset({"evals.gate_product"}),
    # Bundled DP / DP+TP reference harness packages (torch-only; dptp reuses
    # the harness_hook capture surface).
    "evals.harness_dp": frozenset(),
    "evals.harness_dptp": frozenset({"evals.harness_hook"}),
    # M5 resume startup-time gate arithmetic (stdlib-only leaf).
    "evals.resume_startup_gate": frozenset(),
    "evals.gate_common": frozenset({"evals._common", "harness.run_schema"}),
    "evals.dispatcher": frozenset(
        {
            "evals._common",
            # Stage-2 op-* legacy handlers (thin-dispatcher refactor exempt);
            # the dispatcher re-exports RUNNER_KINDS from here.
            "evals.dispatcher_stage2",
            "evals.gate_product",
            "harness.config_runtime",
            "harness.run_schema",
        }
    ),
    "evals.dispatcher_stage2": frozenset(
        {
            "evals._common",
            "evals.gate_common",
            "evals.gate_product",
            "evals.gate_shape",
            "harness._compat",
            "harness.config_runtime",
            "harness.run_schema",
        }
    ),
    "evals.runner": frozenset({"evals.dispatcher", "harness.run_schema", "tools.mfu_record"}),
    # Canonical-state preflight: reuses the dispatcher's ref-run env
    # assembly (evals._common._ref_script_path_env) to run
    # tools/bootstrap_canonical once, idempotently, before the first dev
    # round. Lives in evals precisely because it depends on tools — tools
    # must not depend back on evals (that would close a dependency cycle).
    "evals.canonical_preflight": frozenset(
        {
            "evals._common",
            "harness.config_runtime",
            "tools.bootstrap_canonical",
        }
    ),
    # M1 capture hook package — the public API and its three internal
    # modules. The whole package depends only on ``torch`` (lazily
    # imported), no evals/harness/tools edges by design.
    "evals.harness_hook": frozenset(),
    "evals.harness_hook._module_hook": frozenset(),
    "evals.harness_hook._grad_collector": frozenset(),
    "evals.harness_hook._dump": frozenset(),
    "evals.harness_hook._canonical_state": frozenset(),
    "evals.harness_hook._megatron_argparse_shim": frozenset(),
    "evals.harness_hook._op_inventory": frozenset(),
    # Verdict modules (thin-dispatcher refactor): file-based gate judgment.
    # They read run artifacts + rendered products and reuse the comparison
    # helpers; they never launch subprocesses.
    "evals.verdicts": frozenset(),
    "evals.verdicts._shared": frozenset(
        {"evals._common", "evals.gate_product", "harness.wire_format"}
    ),
    # Shared comparison kernels: the arithmetic/accounting layer verdict
    # modules compose (per-step diff series, hash-dump gate, warmup-filtered
    # MFU average, ref-preflight failure results).
    "evals.verdicts._kernels": frozenset(
        {
            "evals._capture_diff",
            "evals._common",
            "evals.verdicts._shared",
            "harness.run_schema",
            "harness.capture_artifacts",
        }
    ),
    "evals.verdicts.bitwise": frozenset(
        {
            "evals._common",
            "evals.gate_product",
            "evals.gate_shape",
            "evals.verdicts._kernels",
            "evals.verdicts._shared",
            "harness.run_schema",
        }
    ),
    "evals.verdicts.align": frozenset(
        {
            "evals._common",
            "evals._capture_diff",
            "evals.gate_product",
            "evals.gate_shape",
            "evals.verdicts._kernels",
            "evals.verdicts._shared",
            "harness.run_schema",
            "harness.capture_artifacts",
        }
    ),
    # long_train has a tools edge: the hidden elastic MFU relaxation reads
    # its policy via tools.mfu_elastic_check (evals -> tools allowed direction).
    "evals.verdicts.long_train": frozenset(
        {
            "evals._common",
            "evals.gate_product",
            "evals.gate_shape",
            "evals.verdicts._kernels",
            "evals.verdicts._shared",
            "harness.run_schema",
            "tools.mfu_elastic_check",
        }
    ),
    "evals.verdicts.loss_gate": frozenset(
        {
            "evals._common",
            "evals.gate_product",
            "evals.gate_shape",
            "evals.verdicts._kernels",
            "evals.verdicts._shared",
            "harness.run_schema",
        }
    ),
    "evals.verdicts.resume": frozenset(
        {
            "evals._common",
            "evals.gate_product",
            "evals.gate_shape",
            "evals.verdicts._kernels",
            "evals.verdicts._shared",
            "harness.run_schema",
            "harness.wire_format",
        }
    ),
    "evals.verdicts.wsd_sft": frozenset(
        {
            "evals.gate_product",
            "evals.verdicts._kernels",
            "evals.verdicts._shared",
            "harness.run_schema",
            "harness.wire_format",
        }
    ),
    "evals.verdicts.resume_startup": frozenset(
        {
            "evals.gate_product",
            "evals.resume_startup_gate",
            "evals.verdicts._shared",
            "harness.run_schema",
        }
    ),
    "evals.verdicts.production": frozenset(
        {
            "evals.gate_product",
            "evals.verdicts._shared",
            "harness.run_schema",
        }
    ),
    # profile is the one verdict with a tools edge: it post-processes the
    # nsys-rep via tools.profile_render (evals -> tools is the allowed
    # direction of the one-way layering).
    "evals.verdicts.profile": frozenset(
        {
            "evals._common",
            "evals.gate_product",
            "evals.verdicts._kernels",
            "evals.verdicts._shared",
            "harness.run_schema",
            "harness.wire_format",
            "tools.profile_render",
        }
    ),
}

# Allowed edges for control-plane modules under ``tools/``. Excludes:
#   * tools.profile_step / tools.capture_profile_batch — agent-writable
#     data-plane scripts that import ``training_engine_tensor`` and
#     therefore live outside this DAG.
TOOLS_LAYER_ALLOWED_DEPS: dict[str, frozenset[str]] = {
    "tools": frozenset(),
    "tools.ref_script_runner": frozenset({"harness.wire_format"}),
    "tools.stage2_op_router": frozenset({"harness._compat", "harness.config_runtime"}),
    "tools.stage2_config": frozenset({"harness._compat", "harness.config_runtime"}),
    "tools.agent_loop_config": frozenset(
        {"harness.config_runtime", "tools.codex_config", "tools.stage2_config"}
    ),
    "tools.codex_config": frozenset(),
    "tools.cursor_max_mode": frozenset(),
    "tools.mfu_record": frozenset(),
    # Devspace lease registry — ``harness.config_runtime`` for repo-root
    # resolution and ``tools.cctl_common`` for the shared cctl shell-out
    # primitives (run/timeout/task-id parse). The wrapper shells out to
    # ``cctl`` via subprocess, so there is no other harness/web import edge.
    "tools.lease": frozenset({"harness.config_runtime", "tools.cctl_common"}),
    # Remote-execution control plane (on-demand cctl GPU jobs + devspace
    # gateway). cctl_common/remote_workspace are stdlib-only SSOT helpers;
    # gpu_job composes both. None import harness/web.
    "tools.cctl_common": frozenset(),
    "tools.remote_workspace": frozenset(),
    "tools.gpu_job": frozenset({"tools.cctl_common", "tools.remote_workspace"}),
    # Generic product→shell exporter for the thin-dispatcher side scripts.
    # Fully generic (zero key-name knowledge), stdlib-only by design.
    "tools.product_env": frozenset(),
    "tools.render_gate_configs": frozenset(),
    # MFU SSOT backfill — replays gate run events into mfu_history;
    # stdlib-only (no tracked harness/tools edge at import time).
    "tools.mfu_backfill_events": frozenset(),
    # One-shot canonical_state bootstrap — resolves repo paths via
    # harness.config_runtime, then shells out to the ref bridge.
    # bootstrap_canonical projects the gate's rendered ref product into its
    # subprocess env via tools/product_env (stdlib-only leaf).
    "tools.bootstrap_canonical": frozenset({"harness.config_runtime", "tools.product_env"}),
    # Background production corpus prefetch — reads the [data] axis via
    # harness.config_runtime; the curl/reader path is otherwise stdlib.
    "tools.prefetch_data": frozenset({"harness.config_runtime"}),
    # Stdlib-only control-plane leaves: loop forking ("轨迹续跑"), offline
    # loop reporting, the review-side elastic MFU checker, gate-product
    # deployment resolution, and train-recipe env projection.
    "tools.fork_loop": frozenset(),
    "tools.loop_report": frozenset(),
    "tools.mfu_elastic_check": frozenset(),
    "tools.resolve_deploy": frozenset(),
    "tools.train_recipe_to_env": frozenset(),
    # Deterministic profiler-snapshot renderer for M4 / M6 iteration
    # suites. Stdlib-only at import time; ``torch`` is imported lazily
    # inside the memory probe so the static DAG sees no harness/torch
    # edges.
    "tools.profile_render": frozenset(),
    # Bitwise self-consistency check for the streaming HF dataloader.
    # Imports the ref-side ``hf_stream_dataloader`` via a runtime
    # sys.path insertion of ``ref/reference`` (a bare top-level name,
    # not a harness/evals/tools edge), so the static DAG sees no
    # tracked dependencies.
    "tools.hf_data_bitwise_check": frozenset(),
    # Unified-agent-log spawn helpers cross out to the web/agents
    # package (the SSOT spawn/init/pump library that the in-process
    # web runner also uses) via a runtime sys.path insertion. The
    # static DAG check whitelists the bare ``web`` name; the at-import
    # bootstrap inserts ``FORGE_REPO_ROOT`` so the resolution works
    # both from the source checkout and from a provisioned workspace.
    "tools.spawn_managed_agent": frozenset({"web", "web.agents.spawn", "web.auth"}),
    "tools.loop_wrapper_init": frozenset({"web", "web.agents.spawn", "web.agents.store"}),
    "tools.loop_wrapper_event": frozenset({"web", "web.agents.spawn", "web.agents.store"}),
}

# tools/*.py files that are intentionally outside the static control-plane
# DAG (data-plane / agent-writable). They are excluded from
# ``TestControlPlaneToolsLayerDag`` but still subject to the broader
# "no cross-layer cycles" sanity checks elsewhere.
DATA_PLANE_TOOLS_FILES: frozenset[str] = frozenset(
    {
        "tools/profile_step.py",
        "tools/capture_profile_batch.py",
    }
)

# Top-level packages that harness/ is allowed to depend on beyond the
# Python standard library and harness itself. huggingface_hub powers the
# HF Hub fallback for tokenizer/dataset paths in config_runtime.py.
ALLOWED_EXTERNAL_PACKAGES: frozenset[str] = frozenset(
    {"tomli", "tomllib", "tools", "huggingface_hub"}
)

_STDLIB_TOP_LEVEL: frozenset[str] = frozenset(sys.stdlib_module_names)


def _module_name_for(file: Path) -> str:
    rel = file.relative_to(REPO_ROOT).with_suffix("")
    parts = rel.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _iter_imports(source: str) -> set[str]:
    """Return every fully-qualified module name imported by `source`.

    `from pkg import mod` is resolved to `pkg.mod` when `pkg.mod` exists
    as a real module on disk; otherwise we fall back to recording `pkg`
    (the imported name is just an attribute, not a module).
    """
    tree = ast.parse(source)
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                result.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or node.module is None:
                continue
            base = node.module
            for alias in node.names:
                candidate_path = REPO_ROOT / Path(*base.split(".")) / f"{alias.name}.py"
                if candidate_path.exists():
                    result.add(f"{base}.{alias.name}")
                else:
                    result.add(base)
    return result


class TestHarnessLayerDag(unittest.TestCase):
    """Each harness/*.py may only import modules in its allowed layer set."""

    def test_intra_harness_imports_are_bounded(self) -> None:
        for py_file in sorted(HARNESS_DIR.rglob("*.py")):
            if "__pycache__" in py_file.parts or "tests" in py_file.relative_to(HARNESS_DIR).parts:
                continue
            module_name = _module_name_for(py_file)
            allowed = LAYER_ALLOWED_DEPS.get(module_name)
            if allowed is None:
                self.fail(
                    f"Unknown harness module '{module_name}'. "
                    f"Register it in harness/tests/test_layer_dag.py::LAYER_ALLOWED_DEPS "
                    f"with its permitted dependencies."
                )
            source = py_file.read_text(encoding="utf-8")
            imports = _iter_imports(source)
            harness_imports = {
                imp for imp in imports if imp == "harness" or imp.startswith("harness.")
            }
            illegal = harness_imports - allowed - {module_name}
            self.assertFalse(
                illegal,
                f"{module_name} has forbidden harness-internal imports: "
                f"{sorted(illegal)}. Allowed: {sorted(allowed)}.",
            )

    def test_harness_is_leaf_package(self) -> None:
        """harness/ may import ONLY stdlib + harness.* + ALLOWED_EXTERNAL_PACKAGES.

        Catches outward-bound dependencies (workload/, tools/,
        third-party libs) that the layer-DAG check above does not see.
        """
        permitted_top_level = _STDLIB_TOP_LEVEL | ALLOWED_EXTERNAL_PACKAGES | {"harness"}
        for py_file in sorted(HARNESS_DIR.rglob("*.py")):
            if "__pycache__" in py_file.parts or "tests" in py_file.relative_to(HARNESS_DIR).parts:
                continue
            source = py_file.read_text(encoding="utf-8")
            imports = _iter_imports(source)
            illegal = {imp for imp in imports if imp.split(".", 1)[0] not in permitted_top_level}
            self.assertFalse(
                illegal,
                f"{py_file.relative_to(REPO_ROOT)} imports forbidden top-level "
                f"packages: {sorted(illegal)}. harness/ must remain a leaf "
                f"package (stdlib + harness.* only). To add a new external "
                f"dependency, register it in "
                f"harness/tests/test_layer_dag.py::ALLOWED_EXTERNAL_PACKAGES.",
            )

    def test_control_plane_does_not_import_training_engine_config(self) -> None:
        offenders: list[str] = []
        for directory in (REPO_ROOT / "harness", REPO_ROOT / "evals"):
            for py_file in sorted(directory.rglob("*.py")):
                rel = py_file.relative_to(REPO_ROOT)
                if len(rel.parts) >= 2 and rel.parts[:2] == ("evals", "scripts"):
                    continue
                if len(rel.parts) >= 2 and rel.parts[:2] == ("harness", "tests"):
                    continue
                imports = _iter_imports(py_file.read_text(encoding="utf-8"))
                if any(
                    imp == "training_engine_tensor" or imp.startswith("training_engine_tensor.")
                    for imp in imports
                ):
                    offenders.append(str(rel))
        self.assertEqual(
            offenders,
            [],
            "harness control plane must not import workload/src internals",
        )


class TestEvalsLayerDag(unittest.TestCase):
    """``evals/*.py`` (excluding ``evals/scripts/``) bounded by the SSOT DAG."""

    def test_intra_evals_imports_are_bounded(self) -> None:
        for py_file in sorted(EVALS_DIR.rglob("*.py")):
            if "__pycache__" in py_file.parts:
                continue
            rel = py_file.relative_to(REPO_ROOT)
            if len(rel.parts) >= 2 and rel.parts[1] == "scripts":
                continue
            module_name = _module_name_for(py_file)
            allowed = EVALS_LAYER_ALLOWED_DEPS.get(module_name)
            if allowed is None:
                self.fail(
                    f"Unknown evals module '{module_name}'. Register it in "
                    f"harness/tests/test_layer_dag.py::EVALS_LAYER_ALLOWED_DEPS "
                    f"with its permitted dependencies."
                )
            source = py_file.read_text(encoding="utf-8")
            imports = _iter_imports(source)
            tracked = {
                imp
                for imp in imports
                if imp.startswith(("harness.", "evals.", "tools.")) or imp in {"harness", "tools"}
            }
            illegal = tracked - allowed - {module_name}
            self.assertFalse(
                illegal,
                f"{module_name} has forbidden imports: {sorted(illegal)}. "
                f"Allowed: {sorted(allowed)}.",
            )


class TestControlPlaneToolsLayerDag(unittest.TestCase):
    """``tools/*.py`` (excluding data-plane scripts) bounded by the SSOT DAG."""

    def test_intra_tools_imports_are_bounded(self) -> None:
        for py_file in sorted(TOOLS_DIR.glob("*.py")):
            if "__pycache__" in py_file.parts:
                continue
            rel = py_file.relative_to(REPO_ROOT)
            if rel.as_posix() in DATA_PLANE_TOOLS_FILES:
                continue
            module_name = _module_name_for(py_file)
            allowed = TOOLS_LAYER_ALLOWED_DEPS.get(module_name)
            if allowed is None:
                self.fail(
                    f"Unknown tools module '{module_name}'. Register it in "
                    f"harness/tests/test_layer_dag.py::TOOLS_LAYER_ALLOWED_DEPS "
                    f"with its permitted dependencies (or add it to "
                    f"DATA_PLANE_TOOLS_FILES if it's an agent-writable "
                    f"data-plane script)."
                )
            source = py_file.read_text(encoding="utf-8")
            imports = _iter_imports(source)
            tracked = {
                imp
                for imp in imports
                if imp.startswith(("harness.", "evals.", "tools.")) or imp in {"harness", "tools"}
            }
            illegal = tracked - allowed - {module_name}
            self.assertFalse(
                illegal,
                f"{module_name} has forbidden imports: {sorted(illegal)}. "
                f"Allowed: {sorted(allowed)}.",
            )

    def test_control_plane_tools_do_not_import_evals(self) -> None:
        """tools/ must never import evals/ (single-direction layering)."""
        offenders: list[str] = []
        for py_file in sorted(TOOLS_DIR.glob("*.py")):
            if "__pycache__" in py_file.parts:
                continue
            rel = py_file.relative_to(REPO_ROOT)
            if rel.as_posix() in DATA_PLANE_TOOLS_FILES:
                continue
            imports = _iter_imports(py_file.read_text(encoding="utf-8"))
            if any(imp == "evals" or imp.startswith("evals.") for imp in imports):
                offenders.append(str(rel))
        self.assertEqual(
            offenders,
            [],
            "tools/ must not import evals/ (one-way layer: evals/ → tools/, not the reverse).",
        )


class TestStage2OpPathSSOT(unittest.TestCase):
    """Static guard: only ``harness.config_runtime`` may build Stage 2 op paths.

    The on-disk layout (``workload/ops/<op>/register.toml`` and
    ``ops_worktree/<op>/``) is owned by these helpers in
    :mod:`harness.config_runtime`:

      * ``ops_root_path`` / ``ops_registry_path``
      * ``op_dir`` / ``op_register_path``
      * ``ops_worktree_root`` / ``op_worktree_path``

    Any other Python file rebuilding those paths from string literals
    silently re-creates the SSOT drift these helpers exist to fix
    (audit finding 2.3). This guard fails the build the moment a
    fresh ``"workload/ops"`` / ``"ops_worktree"`` / ``"register.toml"``
    appears in the source tree outside the allowlist.
    """

    # Owners of the SSOT (the helpers themselves) plus tests / docs that
    # legitimately reference the literal layout for audit purposes.
    ALLOWED_OWNERS: frozenset[Path] = frozenset(
        {
            Path("harness/config_runtime.py"),
            Path("harness/framework_guard.py"),
            # Sister boundary lint of framework_guard: legitimately
            # owns the "workload/ops" / "workload/src" literals as
            # its CANDIDATE_SCAN_ROOTS — same rationale as
            # framework_guard.py self-ownership above.
            Path("harness/anti_proxy_guard.py"),
            Path("harness/tests/test_layer_dag.py"),
            Path("harness/tests/test_dispatcher_behavior.py"),
            Path("harness/tests/test_harness_core.py"),
            Path("harness/tests/test_prompt_guidelines.py"),
            Path("harness/tests/test_suite_env_and_ref_as_gate.py"),
        }
    )

    SUSPECT_LITERALS: tuple[tuple[str, str], ...] = (
        # (literal, description)
        ('"ops_worktree"', "ops_worktree directory"),
        ("'ops_worktree'", "ops_worktree directory"),
        ('"register.toml"', "per-op register filename"),
        ("'register.toml'", "per-op register filename"),
        ('"workload/ops"', "Stage 2 ops root"),
        ("'workload/ops'", "Stage 2 ops root"),
    )

    def test_stage2_op_paths_have_single_owner(self) -> None:
        offenders: list[str] = []
        for py_file in sorted(REPO_ROOT.rglob("*.py")):
            if "__pycache__" in py_file.parts:
                continue
            try:
                rel = py_file.relative_to(REPO_ROOT)
            except ValueError:
                continue
            if rel in self.ALLOWED_OWNERS:
                continue
            text = py_file.read_text(encoding="utf-8", errors="replace")
            for needle, label in self.SUSPECT_LITERALS:
                if needle in text:
                    offenders.append(f"{rel}: literal {needle!r} ({label})")
                    break
        self.assertFalse(
            offenders,
            "These files build Stage 2 operator paths from string literals "
            "instead of routing through harness.config_runtime helpers "
            "(op_dir / op_register_path / ops_worktree_root / op_worktree_path); "
            "this re-introduces the SSOT drift fixed by audit finding 2.3:\n  "
            + "\n  ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
