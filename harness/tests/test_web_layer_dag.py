"""Architectural linter: lock the web/ layer DAG and SSOT statically.

Mirrors ``test_layer_dag.py`` for the web layer. Guarantees:

  - Shared constants (PROJECT_ROOT, FORGE_TRAIN_DIR, CONFIG_AXES,
    HARNESS_DIR, AGENTS_DIR, wrapper_agent_id, pid_alive,
    SIGNAL_EXIT_CODES) live in exactly one module: ``web.paths``.
  - ``web.agents`` never imports from ``web.routers`` (lower → upper
    dependency).
  - No router imports private symbols from another router.
  - No ``sys.path`` manipulation inside ``web/``.
  - Dead code identified by the audit stays removed.
  - Pydantic request models follow the ``*Request`` suffix convention.
  - ``wrapper_agent_id`` is never inlined in ``web/`` or ``harness/tools/``.
  - ``store.AGENTS_DIR`` is never mutated at runtime.
  - Key modules export ``__all__``.
"""

from __future__ import annotations

import ast
import re
import sys
import unittest
import unittest.mock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WEB_DIR = REPO_ROOT / "web"
WEB_ROUTERS_DIR = WEB_DIR / "routers"
WEB_AGENTS_DIR = WEB_DIR / "agents"
HARNESS_TOOLS_DIR = REPO_ROOT / "harness" / "tools"

_STDLIB_TOP_LEVEL: frozenset[str] = frozenset(sys.stdlib_module_names)


def _iter_imports(source: str, *, source_file: Path | None = None) -> set[str]:
    """Return every fully-qualified module name imported by ``source``.

    When *source_file* is given, relative imports (level > 0) are resolved
    against the file's package hierarchy so the DAG checker catches
    ``from ..routers import X`` style violations.
    """
    tree = ast.parse(source)
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                result.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module is None:
                    continue
                base = node.module
            else:
                base = _resolve_relative(node, source_file)
                if base is None:
                    continue
            for alias in node.names:
                candidate = REPO_ROOT / Path(*base.split(".")) / f"{alias.name}.py"
                if candidate.exists():
                    result.add(f"{base}.{alias.name}")
                else:
                    result.add(base)
    return result


def _resolve_relative(node: ast.ImportFrom, source_file: Path | None) -> str | None:
    """Resolve a relative import to a fully-qualified module name."""
    if source_file is None:
        return None
    try:
        rel = source_file.relative_to(REPO_ROOT)
    except ValueError:
        return None
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    # Go up `level` packages
    up = node.level
    if up > len(parts):
        return None
    package_parts = parts[:-up] if up > 0 else parts
    if node.module:
        package_parts = [*package_parts, *node.module.split(".")]
    return ".".join(package_parts)


def _iter_lazy_imports(source: str, *, source_file: Path | None = None) -> set[str]:
    """Return imports that appear inside function bodies (lazy imports)."""
    tree = ast.parse(source)
    result: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Import):
                for alias in child.names:
                    result.add(alias.name)
            elif isinstance(child, ast.ImportFrom):
                if child.level == 0:
                    if child.module is None:
                        continue
                    base = child.module
                else:
                    base = _resolve_relative(child, source_file)
                    if base is None:
                        continue
                for alias in child.names:
                    candidate = REPO_ROOT / Path(*base.split(".")) / f"{alias.name}.py"
                    if candidate.exists():
                        result.add(f"{base}.{alias.name}")
                    else:
                        result.add(base)
    return result


class TestWebPathsSSOT(unittest.TestCase):
    """``web.paths`` is the single source of truth for shared constants."""

    def test_web_paths_module_exists(self) -> None:
        self.assertTrue(
            (WEB_DIR / "paths.py").is_file(),
            "web/paths.py must exist as the SSOT for shared constants",
        )

    def test_no_duplicate_project_root_in_routers(self) -> None:
        """No router may define its own PROJECT_ROOT / WORKSPACE_ROOT."""
        pattern = re.compile(
            r"^(?:PROJECT_ROOT|WORKSPACE_ROOT|_PROJECT_ROOT)\s*=\s*Path\(",
            re.MULTILINE,
        )
        offenders: list[str] = []
        for py_file in sorted(WEB_ROUTERS_DIR.glob("*.py")):
            source = py_file.read_text(encoding="utf-8")
            if pattern.search(source):
                offenders.append(py_file.name)
        self.assertEqual(
            offenders,
            [],
            f"Routers defining own root path (must import from web.paths): {offenders}",
        )

    def test_no_duplicate_project_root_in_agents(self) -> None:
        """No agents module may define its own PROJECT_ROOT / WORKSPACE_ROOT."""
        pattern = re.compile(
            r"^(?:PROJECT_ROOT|WORKSPACE_ROOT|_PROJECT_ROOT)\s*=\s*Path\(",
            re.MULTILINE,
        )
        offenders: list[str] = []
        for py_file in sorted(WEB_AGENTS_DIR.glob("*.py")):
            if py_file.name == "__init__.py":
                continue
            source = py_file.read_text(encoding="utf-8")
            if pattern.search(source):
                offenders.append(py_file.name)
        self.assertEqual(
            offenders,
            [],
            f"Agent modules defining own root path (must import from web.paths): {offenders}",
        )

    def test_no_duplicate_forge_train_dir(self) -> None:
        """FORGE_TRAIN_DIR only in web/paths.py."""
        pattern = re.compile(r"^(?:FORGE_TRAIN_DIR|_FORGE_TRAIN_DIR)\s*=\s*", re.MULTILINE)
        offenders: list[str] = []
        for py_file in sorted(WEB_DIR.rglob("*.py")):
            if py_file.name in ("paths.py", "__pycache__"):
                continue
            if "__pycache__" in py_file.parts:
                continue
            source = py_file.read_text(encoding="utf-8")
            if pattern.search(source):
                offenders.append(str(py_file.relative_to(WEB_DIR)))
        self.assertEqual(
            offenders,
            [],
            f"Files defining own FORGE_TRAIN_DIR (must import from web.paths): {offenders}",
        )

    def test_no_duplicate_config_axes(self) -> None:
        """CONFIG_AXES only in web/paths.py."""
        pattern = re.compile(r"^_?CONFIG_AXES\s*=\s*", re.MULTILINE)
        offenders: list[str] = []
        for py_file in sorted(WEB_DIR.rglob("*.py")):
            if py_file.name in ("paths.py", "__pycache__"):
                continue
            if "__pycache__" in py_file.parts:
                continue
            source = py_file.read_text(encoding="utf-8")
            if pattern.search(source):
                offenders.append(str(py_file.relative_to(WEB_DIR)))
        self.assertEqual(
            offenders,
            [],
            f"Files defining own CONFIG_AXES (must import from web.paths): {offenders}",
        )

    def test_no_duplicate_harness_dir(self) -> None:
        """HARNESS_DIR / _HARNESS_DIR only in web/paths.py."""
        pattern = re.compile(r"^(?:HARNESS_DIR|_HARNESS_DIR)\s*=\s*", re.MULTILINE)
        offenders: list[str] = []
        for py_file in sorted(WEB_DIR.rglob("*.py")):
            if py_file.name in ("paths.py", "__pycache__"):
                continue
            if "__pycache__" in py_file.parts:
                continue
            source = py_file.read_text(encoding="utf-8")
            if pattern.search(source):
                offenders.append(str(py_file.relative_to(WEB_DIR)))
        self.assertEqual(
            offenders,
            [],
            f"Files defining own HARNESS_DIR (must import from web.paths): {offenders}",
        )

    def test_no_duplicate_agents_dir(self) -> None:
        """AGENTS_DIR only in web/paths.py (never redefined elsewhere in web/)."""
        pattern = re.compile(r"^(?:AGENTS_DIR|_AGENTS_DIR)\s*=\s*", re.MULTILINE)
        offenders: list[str] = []
        for py_file in sorted(WEB_DIR.rglob("*.py")):
            if py_file.name in ("paths.py", "__pycache__"):
                continue
            if "__pycache__" in py_file.parts:
                continue
            source = py_file.read_text(encoding="utf-8")
            if pattern.search(source):
                offenders.append(str(py_file.relative_to(WEB_DIR)))
        self.assertEqual(
            offenders,
            [],
            f"Files defining own AGENTS_DIR (must import from web.paths): {offenders}",
        )

    def test_no_inline_wrapper_agent_id(self) -> None:
        """f"loop-{loop_id}" must route through web.paths.wrapper_agent_id.

        Scans web/ (Python + JS), and harness/tools/ to catch SSOT leaks
        across the layer boundary.
        """
        py_pattern = re.compile(r'f"loop-\{loop_id\}"')
        js_pattern = re.compile(r"`loop-\$\{")
        offenders: list[str] = []
        scan_roots = [
            (WEB_DIR, WEB_DIR),
            (HARNESS_TOOLS_DIR, HARNESS_TOOLS_DIR),
        ]
        for scan_dir, _rel_base in scan_roots:
            for py_file in sorted(scan_dir.rglob("*.py")):
                if py_file.name == "paths.py":
                    continue
                if "__pycache__" in py_file.parts:
                    continue
                source = py_file.read_text(encoding="utf-8")
                if py_pattern.search(source):
                    offenders.append(str(py_file.relative_to(REPO_ROOT)))
        for js_file in sorted(WEB_DIR.rglob("*.js")):
            if "node_modules" in js_file.parts:
                continue
            source = js_file.read_text(encoding="utf-8")
            if js_pattern.search(source):
                offenders.append(str(js_file.relative_to(REPO_ROOT)))
        self.assertEqual(
            offenders,
            [],
            f"Files with inline wrapper_agent_id (must use wrapper_agent_id helper): {offenders}",
        )

    def test_no_agents_dir_mutation(self) -> None:
        """store.AGENTS_DIR must never be assigned at runtime.

        The FORGE_AGENTS_DIR env override is handled by web.paths; no
        caller should mutate the module attribute directly.
        """
        pattern = re.compile(r"store\.AGENTS_DIR\s*=\s*")
        offenders: list[str] = []
        scan_dirs = [WEB_DIR, HARNESS_TOOLS_DIR]
        for scan_dir in scan_dirs:
            for py_file in sorted(scan_dir.rglob("*.py")):
                if "__pycache__" in py_file.parts:
                    continue
                source = py_file.read_text(encoding="utf-8")
                if pattern.search(source):
                    offenders.append(str(py_file.relative_to(REPO_ROOT)))
        self.assertEqual(
            offenders,
            [],
            f"Files that mutate store.AGENTS_DIR (must use web.paths env override): {offenders}",
        )

    def test_pid_alive_has_single_definition(self) -> None:
        """pid_alive helper defined only in web/paths.py."""
        pattern = re.compile(r"^def _?pid_alive\(", re.MULTILINE)
        offenders: list[str] = []
        for py_file in sorted(WEB_DIR.rglob("*.py")):
            if py_file.name == "paths.py":
                continue
            if "__pycache__" in py_file.parts:
                continue
            source = py_file.read_text(encoding="utf-8")
            if pattern.search(source):
                offenders.append(str(py_file.relative_to(WEB_DIR)))
        self.assertEqual(
            offenders,
            [],
            f"Files with duplicate pid_alive (must import from web.paths): {offenders}",
        )

    def test_signal_exit_codes_has_single_definition(self) -> None:
        """SIGNAL_EXIT_CODES defined only in web/paths.py."""
        pattern = re.compile(r"_?SIGNAL_EXIT_CODES\s*=\s*\{128", re.MULTILINE)
        offenders: list[str] = []
        for py_file in sorted(WEB_DIR.rglob("*.py")):
            if py_file.name == "paths.py":
                continue
            if "__pycache__" in py_file.parts:
                continue
            source = py_file.read_text(encoding="utf-8")
            if pattern.search(source):
                offenders.append(str(py_file.relative_to(WEB_DIR)))
        self.assertEqual(
            offenders,
            [],
            f"Files with duplicate SIGNAL_EXIT_CODES (must import from web.paths): {offenders}",
        )

    def test_anthropic_base_url_only_in_paths(self) -> None:
        """The ANTHROPIC_BASE_URL default must live only in web/paths.py."""
        pattern = re.compile(r"llm-center\.ali\.modelbest\.cn")
        offenders: list[str] = []
        for py_file in sorted(WEB_DIR.rglob("*.py")):
            if py_file.name == "paths.py":
                continue
            if "__pycache__" in py_file.parts:
                continue
            source = py_file.read_text(encoding="utf-8")
            if pattern.search(source):
                offenders.append(str(py_file.relative_to(WEB_DIR)))
        self.assertEqual(
            offenders,
            [],
            "Files with hardcoded ANTHROPIC_BASE_URL default "
            f"(must import from web.paths): {offenders}",
        )

    def test_proxy_env_keys_only_in_paths(self) -> None:
        """Proxy environment key lists must be defined only in web/paths.py."""
        pattern = re.compile(
            r"""["']http_proxy["'],\s*["']https_proxy["'],\s*["']HTTP_PROXY["'],\s*["']HTTPS_PROXY["']"""
        )
        offenders: list[str] = []
        for py_file in sorted(WEB_DIR.rglob("*.py")):
            if py_file.name == "paths.py":
                continue
            if "__pycache__" in py_file.parts:
                continue
            source = py_file.read_text(encoding="utf-8")
            if pattern.search(source):
                offenders.append(str(py_file.relative_to(WEB_DIR)))
        self.assertEqual(
            offenders,
            [],
            f"Files with inline proxy key lists (must import from web.paths): {offenders}",
        )

    def test_store_does_not_reexport_path_constants(self) -> None:
        """store.__all__ must not re-export AGENTS_DIR or REPO_ROOT."""
        source = (WEB_AGENTS_DIR / "store.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        all_value: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    all_value.append(elt.value)
        forbidden = {"AGENTS_DIR", "REPO_ROOT"}
        found = forbidden & set(all_value)
        self.assertEqual(
            found,
            set(),
            f"store.__all__ re-exports path constants (import from web.paths instead): {found}",
        )

    def test_no_project_root_alias_in_routers(self) -> None:
        """Routers must use REPO_ROOT directly, not alias it to PROJECT_ROOT."""
        pattern = re.compile(r"REPO_ROOT\s+as\s+PROJECT_ROOT")
        offenders: list[str] = []
        for py_file in sorted(WEB_ROUTERS_DIR.glob("*.py")):
            if "__pycache__" in py_file.parts:
                continue
            source = py_file.read_text(encoding="utf-8")
            if pattern.search(source):
                offenders.append(py_file.name)
        self.assertEqual(
            offenders,
            [],
            f"Routers aliasing REPO_ROOT to PROJECT_ROOT (use REPO_ROOT directly): {offenders}",
        )

    def test_config_categories_derived_from_config_axes(self) -> None:
        """config.py _CATEGORIES keys must be programmatically derived from CONFIG_AXES."""
        source = (WEB_ROUTERS_DIR / "config.py").read_text(encoding="utf-8")
        self.assertNotIn(
            '"ref":\n',
            source.replace(" ", ""),
            "config.py still hardcodes category keys; derive from CONFIG_AXES",
        )

    def test_max_mode_script_uses_harness_dir(self) -> None:
        """backends.py must derive max-mode script from paths.HARNESS_DIR, not __file__."""
        source = (WEB_AGENTS_DIR / "backends.py").read_text(encoding="utf-8")
        self.assertNotIn(
            "parents[2]",
            source,
            "backends.py still uses Path(__file__).parents[2] for max-mode script "
            "(must import from web.paths)",
        )

    def test_harness_config_only_in_paths(self) -> None:
        """HARNESS_CONFIG must be defined in paths.py, not locally in routers."""
        pattern = re.compile(r"^_?HARNESS_CONFIG\s*=\s*", re.MULTILINE)
        offenders: list[str] = []
        for py_file in sorted(WEB_DIR.rglob("*.py")):
            if py_file.name in ("paths.py", "__pycache__"):
                continue
            if "__pycache__" in py_file.parts:
                continue
            source = py_file.read_text(encoding="utf-8")
            if pattern.search(source):
                offenders.append(str(py_file.relative_to(WEB_DIR)))
        self.assertEqual(
            offenders,
            [],
            f"Files defining own HARNESS_CONFIG (must import from web.paths): {offenders}",
        )

    def test_agent_loop_state_subpath_only_in_paths(self) -> None:
        """The agent-loop-state subpath must not be hardcoded outside paths.py."""
        offenders: list[str] = []
        for py_file in sorted(WEB_DIR.rglob("*.py")):
            if py_file.name in ("paths.py", "__pycache__"):
                continue
            if "__pycache__" in py_file.parts:
                continue
            source = py_file.read_text(encoding="utf-8")
            if "agent-loop-state" in source:
                offenders.append(str(py_file.relative_to(WEB_DIR)))
        self.assertEqual(
            offenders,
            [],
            f"Files hardcoding 'agent-loop-state' subpath (must import from web.paths): {offenders}",
        )

    def test_stale_threshold_env_read_in_paths(self) -> None:
        """FORGE_WEB_AGENT_STALE_SECONDS env read must be in paths.py."""
        offenders: list[str] = []
        for py_file in sorted(WEB_DIR.rglob("*.py")):
            if py_file.name in ("paths.py", "__pycache__"):
                continue
            if "__pycache__" in py_file.parts:
                continue
            source = py_file.read_text(encoding="utf-8")
            if "FORGE_WEB_AGENT_STALE_SECONDS" in source:
                offenders.append(str(py_file.relative_to(WEB_DIR)))
        self.assertEqual(
            offenders,
            [],
            f"Files reading FORGE_WEB_AGENT_STALE_SECONDS env "
            f"(must import from web.paths): {offenders}",
        )


class TestWebLayerDAG(unittest.TestCase):
    """web/agents must not import from web/routers (lower → upper)."""

    def test_agents_do_not_import_routers(self) -> None:
        offenders: list[str] = []
        for py_file in sorted(WEB_AGENTS_DIR.glob("*.py")):
            if "__pycache__" in py_file.parts:
                continue
            source = py_file.read_text(encoding="utf-8")
            all_imports = _iter_imports(source, source_file=py_file) | _iter_lazy_imports(
                source, source_file=py_file
            )
            router_imports = {imp for imp in all_imports if imp.startswith("web.routers")}
            if router_imports:
                offenders.append(f"{py_file.name}: {sorted(router_imports)}")
        self.assertEqual(
            offenders,
            [],
            "web/agents modules import from web/routers (DAG violation):\n  "
            + "\n  ".join(offenders),
        )

    def test_no_sys_path_manipulation(self) -> None:
        """web/ must not manipulate sys.path to reach harness internals.

        The sole exemption is the top-level ``web/__init__.py`` bootstrap:
        it inserts ``<repo>/harness`` onto sys.path so a bare
        ``python3 -m web.server`` resolves the sibling-module bare-name
        imports (``from tools import …``) without a PYTHONPATH shim. That
        injection must run before any ``web/`` submodule imports, so it
        cannot be expressed any other way — it reads its path from the
        ``web.paths.HARNESS_DIR`` SSOT rather than recomputing it.
        """
        pattern = re.compile(r"sys\.path\.(insert|append)\(")
        offenders: list[str] = []
        for py_file in sorted(WEB_DIR.rglob("*.py")):
            if "__pycache__" in py_file.parts:
                continue
            if py_file == WEB_DIR / "__init__.py":
                continue
            source = py_file.read_text(encoding="utf-8")
            if pattern.search(source):
                offenders.append(str(py_file.relative_to(WEB_DIR)))
        self.assertEqual(
            offenders,
            [],
            f"web/ files manipulating sys.path (must not reach into harness/): {offenders}",
        )

    def test_routers_do_not_use_private_agent_symbols(self) -> None:
        """Routers must not call _-prefixed symbols from web.agents modules."""
        pattern = re.compile(r"\brunner\._\w+|store\._\w+|spawn\._\w+|backends\._\w+")
        offenders: list[str] = []
        for py_file in sorted(WEB_ROUTERS_DIR.glob("*.py")):
            if "__pycache__" in py_file.parts:
                continue
            source = py_file.read_text(encoding="utf-8")
            matches = pattern.findall(source)
            if matches:
                offenders.append(f"{py_file.name}: {sorted(set(matches))}")
        self.assertEqual(
            offenders,
            [],
            "Routers using private symbols from web.agents:\n  " + "\n  ".join(offenders),
        )

    def test_routers_do_not_import_private_from_sibling_router(self) -> None:
        """No router imports another router's private symbols."""
        offenders: list[str] = []
        for py_file in sorted(WEB_ROUTERS_DIR.glob("*.py")):
            if "__pycache__" in py_file.parts:
                continue
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if node.level != 0 or node.module is None:
                    continue
                if not node.module.startswith("web.routers."):
                    continue
                sibling_module = node.module.split(".")[-1]
                if sibling_module == py_file.stem:
                    continue
                for alias in node.names:
                    if alias.name.startswith("_"):
                        offenders.append(f"{py_file.name} imports {node.module}.{alias.name}")
        self.assertEqual(
            offenders,
            [],
            "Routers importing private symbols from siblings:\n  " + "\n  ".join(offenders),
        )


class TestWebDoesNotImportHarness(unittest.TestCase):
    """web/ must never import harness/ — the dependency is one-way (harness → web)."""

    def test_no_harness_import_in_web(self) -> None:
        offenders: list[str] = []
        pattern = re.compile(r"^\s*(?:from|import)\s+harness\b", re.MULTILINE)
        for py_file in sorted(WEB_DIR.rglob("*.py")):
            if "__pycache__" in py_file.parts:
                continue
            source = py_file.read_text(encoding="utf-8")
            if pattern.search(source):
                offenders.append(str(py_file.relative_to(WEB_DIR)))
        self.assertEqual(
            offenders,
            [],
            f"web/ files importing harness (forbidden — harness depends on web, not reverse): {offenders}",
        )


class TestWebDoesNotImportTrainEngine(unittest.TestCase):
    """web/ must never import train_engine/ — they are sibling packages."""

    def test_no_train_engine_import_in_web(self) -> None:
        offenders: list[str] = []
        pattern = re.compile(r"^\s*(?:from|import)\s+train_engine\b", re.MULTILINE)
        for py_file in sorted(WEB_DIR.rglob("*.py")):
            if "__pycache__" in py_file.parts:
                continue
            source = py_file.read_text(encoding="utf-8")
            if pattern.search(source):
                offenders.append(str(py_file.relative_to(WEB_DIR)))
        self.assertEqual(
            offenders,
            [],
            f"web/ files importing train_engine (forbidden sibling dependency): {offenders}",
        )


class TestDeadCodeRemoved(unittest.TestCase):
    """Specific dead symbols identified by the audit must stay removed."""

    def test_loop_logs_router_removed(self) -> None:
        self.assertFalse(
            (WEB_ROUTERS_DIR / "loop_logs.py").is_file(),
            "web/routers/loop_logs.py must be removed (6 endpoints, zero frontend callers)",
        )

    def test_server_does_not_import_loop_logs(self) -> None:
        source = (WEB_DIR / "server.py").read_text(encoding="utf-8")
        self.assertNotIn(
            "loop_logs",
            source,
            "server.py must not reference the removed loop_logs router",
        )

    def test_dead_functions_removed(self) -> None:
        dead_symbols = {
            "agents/messages.py": ["first_user_preview"],
            "agents/runner.py": [
                "invalidate_models_cache",
                "_is_externally_active",
                "_spawn",
                "_is_pid_alive",
            ],
            "agents/spawn.py": ["run_to_completion"],
            "routers/recording.py": ["mark_recording_ended"],
        }
        for rel_path, functions in dead_symbols.items():
            py_file = WEB_DIR / rel_path
            if not py_file.is_file():
                continue
            source = py_file.read_text(encoding="utf-8")
            for func_name in functions:
                pattern = re.compile(rf"^def {func_name}\(", re.MULTILINE)
                self.assertIsNone(
                    pattern.search(source),
                    f"{rel_path} still defines dead function {func_name}",
                )

    def test_rerun_endpoint_removed(self) -> None:
        source = (WEB_ROUTERS_DIR / "loop.py").read_text(encoding="utf-8")
        self.assertNotIn(
            "rerun_loop",
            source,
            "loop.py must not define the permanently-410 rerun_loop endpoint",
        )

    def test_rerun_request_model_removed(self) -> None:
        source = (WEB_ROUTERS_DIR / "loop.py").read_text(encoding="utf-8")
        self.assertNotIn(
            "class RerunRequest",
            source,
            "RerunRequest model must be removed with the rerun endpoint",
        )

    def test_delete_recording_function_removed(self) -> None:
        source = (WEB_ROUTERS_DIR / "recording.py").read_text(encoding="utf-8")
        pattern = re.compile(r"^def delete_recording\(", re.MULTILINE)
        self.assertIsNone(
            pattern.search(source),
            "recording.py still defines dead function delete_recording",
        )

    def test_dead_constants_removed(self) -> None:
        """Constants identified as dead by the audit must stay removed."""
        dead_constants = {
            "agents/runner.py": ["_EXTERNAL_ACTIVE_THRESHOLD"],
        }
        for rel_path, constants in dead_constants.items():
            py_file = WEB_DIR / rel_path
            if not py_file.is_file():
                continue
            source = py_file.read_text(encoding="utf-8")
            for name in constants:
                pattern = re.compile(rf"^{re.escape(name)}\s*[:=]", re.MULTILINE)
                self.assertIsNone(
                    pattern.search(source),
                    f"{rel_path} still defines dead constant {name}",
                )

    def test_dead_pydantic_models_removed(self) -> None:
        """Pydantic models with zero callers must stay removed."""
        dead_models = {
            "routers/recording.py": ["RecordingChunkRequest"],
        }
        for rel_path, models in dead_models.items():
            py_file = WEB_DIR / rel_path
            if not py_file.is_file():
                continue
            source = py_file.read_text(encoding="utf-8")
            for name in models:
                pattern = re.compile(rf"^class {name}\(", re.MULTILINE)
                self.assertIsNone(
                    pattern.search(source),
                    f"{rel_path} still defines dead model {name}",
                )


class TestImportSideEffects(unittest.TestCase):
    """Module-level side effects must be deferred to startup."""

    def test_loop_module_no_toplevel_load_history(self) -> None:
        source = (WEB_ROUTERS_DIR / "loop.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                func = node.value.func
                name = ""
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                self.assertNotIn(
                    name,
                    ("_load_history", "_reap_stale_drafts"),
                    f"loop.py calls {name}() at module level (must move to startup)",
                )


class TestModuleSurface(unittest.TestCase):
    """Key agent modules must declare __all__ to bound public surface."""

    def test_key_modules_have_all(self) -> None:
        """Superseded by TestMinimalPublicSurface.test_all_key_modules_have_all."""

    def test_workspace_root_alias_removed(self) -> None:
        """store.py must not re-export REPO_ROOT as WORKSPACE_ROOT."""
        source = (WEB_AGENTS_DIR / "store.py").read_text(encoding="utf-8")
        self.assertNotIn(
            "WORKSPACE_ROOT",
            source,
            "store.py still re-exports REPO_ROOT as WORKSPACE_ROOT "
            "(callers should import REPO_ROOT from web.paths directly)",
        )

    def test_loop_router_public_startup_hooks(self) -> None:
        """loop.py startup hooks consumed by server.py must be public (no _ prefix)."""
        source = (WEB_ROUTERS_DIR / "loop.py").read_text(encoding="utf-8")
        self.assertTrue(
            re.search(r"^async def reattach_running\(", source, re.MULTILINE),
            "loop.py must expose reattach_running (no _ prefix) — "
            "server.py imports it as a public startup hook",
        )
        self.assertFalse(
            re.search(r"^async def _reattach_running\(", source, re.MULTILINE),
            "loop.py still defines _reattach_running with _ prefix — "
            "rename to reattach_running (it is a public startup hook)",
        )


class TestMinimalPublicSurface(unittest.TestCase):
    """Public surface (__all__) must be as tight as possible."""

    _MODULES_REQUIRING_ALL = (
        "paths.py",
        "auth.py",
        "agents/store.py",
        "agents/runner.py",
        "agents/spawn.py",
        "agents/backends.py",
        "agents/messages.py",
        "agents/transcript.py",
        "routers/loop.py",
        "routers/config.py",
        "routers/files.py",
        "routers/artifacts.py",
    )

    def test_all_key_modules_have_all(self) -> None:
        missing: list[str] = []
        for rel in self._MODULES_REQUIRING_ALL:
            py_file = WEB_DIR / rel
            if not py_file.is_file():
                continue
            source = py_file.read_text(encoding="utf-8")
            if not re.search(r"^__all__\s*=\s*", source, re.MULTILINE):
                missing.append(rel)
        self.assertEqual(
            missing,
            [],
            f"Modules missing __all__ (must declare public surface): {missing}",
        )

    def test_store_all_excludes_internal_helpers(self) -> None:
        """store.__all__ must not export write helpers only used by spawn/runner."""
        source = (WEB_AGENTS_DIR / "store.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        all_value: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    all_value.append(elt.value)
        internal_only = {
            "new_agent_id",
            "now_iso",
            "save",
            "session_file",
            "stderr_file",
            "update",
            "STATE_COMPLETED",
            "STATE_FAILED",
            "STATE_INTERRUPTED",
            "STATE_RUNNING",
            "STALE_OUTPUT_THRESHOLD_SECONDS",
        }
        leaked = internal_only & set(all_value)
        self.assertEqual(
            leaked,
            set(),
            f"store.__all__ exports internal helpers (should be private): {sorted(leaked)}",
        )

    def test_spawn_all_excludes_runner_internal_helpers(self) -> None:
        """spawn.__all__ must only export harness-contract symbols."""
        source = (WEB_AGENTS_DIR / "spawn.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        all_value: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    all_value.append(elt.value)
        runner_only = {
            "precreate_web_session",
            "start_precreated_web_session",
            "SpawnResult",
            "kill_process_group",
        }
        leaked = runner_only & set(all_value)
        self.assertEqual(
            leaked,
            set(),
            f"spawn.__all__ exports runner-internal symbols (not part of harness contract): {sorted(leaked)}",
        )

    def test_backends_all_excludes_spawn_internal_helpers(self) -> None:
        """backends.__all__ must not export symbols only used by spawn."""
        source = (WEB_AGENTS_DIR / "backends.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        all_value: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    all_value.append(elt.value)
        internal_only = {"synthetic_user_event", "AgentBackend"}
        leaked = internal_only & set(all_value)
        self.assertEqual(
            leaked,
            set(),
            f"backends.__all__ exports internal symbols: {sorted(leaked)}",
        )

    def test_runner_all_excludes_pid_alive_wrapper(self) -> None:
        """runner must not re-export a pid_alive wrapper."""
        source = (WEB_AGENTS_DIR / "runner.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        all_value: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    all_value.append(elt.value)
        self.assertNotIn(
            "is_pid_alive",
            all_value,
            "runner.__all__ must not re-export is_pid_alive "
            "(callers should use web.paths.pid_alive directly)",
        )

    def test_runner_auth_status_has_no_dead_api_key_param(self) -> None:
        """runner.auth_status must not accept an unused api_key parameter."""
        source = (WEB_AGENTS_DIR / "runner.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "auth_status":
                param_names = [a.arg for a in node.args.args]
                self.assertNotIn(
                    "api_key",
                    param_names,
                    "runner.auth_status still accepts dead parameter 'api_key'",
                )
                break


class TestSSOTAgentsDirAccess(unittest.TestCase):
    """AGENTS_DIR must only be accessed via web.paths, never via store attribute."""

    def test_loop_router_does_not_read_agents_dir_via_store(self) -> None:
        """loop.py must use web.paths.AGENTS_DIR, not agent_store.AGENTS_DIR."""
        source = (WEB_ROUTERS_DIR / "loop.py").read_text(encoding="utf-8")
        pattern = re.compile(r"agent_store\.AGENTS_DIR|store\.AGENTS_DIR")
        self.assertIsNone(
            pattern.search(source),
            "loop.py reads AGENTS_DIR via store module attribute "
            "(must import from web.paths instead)",
        )

    def test_agent_loop_sh_does_not_mutate_store_agents_dir(self) -> None:
        """agent-loop.sh inline Python must not mutate store.AGENTS_DIR."""
        agent_loop_sh = REPO_ROOT / "harness" / "agent-loop.sh"
        if not agent_loop_sh.is_file():
            return
        source = agent_loop_sh.read_text(encoding="utf-8")
        pattern = re.compile(r"store\.AGENTS_DIR\s*=\s*")
        self.assertIsNone(
            pattern.search(source),
            "agent-loop.sh mutates store.AGENTS_DIR at runtime "
            "(FORGE_AGENTS_DIR env should be exported before import)",
        )


class TestAuthImportNaming(unittest.TestCase):
    """Import alias for web.auth must not use the outdated cursor_auth name."""

    def test_no_cursor_auth_alias(self) -> None:
        """Routers must not alias ``web.auth`` as ``cursor_auth``."""
        pattern = re.compile(r"as\s+cursor_auth\b")
        offenders: list[str] = []
        for py_file in sorted(WEB_DIR.rglob("*.py")):
            if "__pycache__" in py_file.parts:
                continue
            source = py_file.read_text(encoding="utf-8")
            if pattern.search(source):
                offenders.append(str(py_file.relative_to(WEB_DIR)))
        self.assertEqual(
            offenders,
            [],
            f"Files using outdated 'cursor_auth' alias (rename to 'auth'): {offenders}",
        )


class TestBuildEnvDelegation(unittest.TestCase):
    """loop._build_env must delegate base env construction to backends."""

    def test_loop_does_not_import_anthropic_base_url_default(self) -> None:
        """loop.py must delegate ANTHROPIC_BASE_URL setup to backends.build_env."""
        source = (WEB_ROUTERS_DIR / "loop.py").read_text(encoding="utf-8")
        self.assertNotIn(
            "ANTHROPIC_BASE_URL_DEFAULT",
            source,
            "loop.py still imports ANTHROPIC_BASE_URL_DEFAULT "
            "(must delegate to backends.build_env instead)",
        )


class TestNamingConsistency(unittest.TestCase):
    """Logger variable naming must be consistent across routers."""

    def test_router_logger_uses_underscore_prefix(self) -> None:
        """All routers must name their module logger ``_log``, not ``log``."""
        offenders: list[str] = []
        pattern = re.compile(r"^log\s*=\s*logging\.getLogger\(", re.MULTILINE)
        for py_file in sorted(WEB_ROUTERS_DIR.glob("*.py")):
            if "__pycache__" in py_file.parts:
                continue
            source = py_file.read_text(encoding="utf-8")
            if pattern.search(source):
                offenders.append(py_file.name)
        self.assertEqual(
            offenders,
            [],
            f"Routers using public 'log' instead of '_log' for module logger: {offenders}",
        )


class TestLoopEventSchema(unittest.TestCase):
    """The loop_event NDJSON protocol must have a schema SSOT in web/agents/messages.py."""

    def test_loop_event_subtypes_defined(self) -> None:
        """messages.py must define LOOP_EVENT_SUBTYPES as the SSOT for valid subtypes."""
        source = (WEB_AGENTS_DIR / "messages.py").read_text(encoding="utf-8")
        self.assertTrue(
            re.search(r"^LOOP_EVENT_SUBTYPES\s*[:=]", source, re.MULTILINE),
            "web/agents/messages.py must define LOOP_EVENT_SUBTYPES as the protocol SSOT",
        )

    def test_loop_event_subtypes_not_duplicated(self) -> None:
        """LOOP_EVENT_SUBTYPES must not be redefined elsewhere in web/."""
        pattern = re.compile(r"^_?LOOP_EVENT_SUBTYPES\s*[:=]", re.MULTILINE)
        offenders: list[str] = []
        for py_file in sorted(WEB_DIR.rglob("*.py")):
            if py_file.name == "messages.py":
                continue
            if "__pycache__" in py_file.parts:
                continue
            source = py_file.read_text(encoding="utf-8")
            if pattern.search(source):
                offenders.append(str(py_file.relative_to(WEB_DIR)))
        self.assertEqual(
            offenders,
            [],
            f"Files redefining LOOP_EVENT_SUBTYPES (SSOT is messages.py): {offenders}",
        )

    def test_loop_event_promoted_keys_defined(self) -> None:
        """messages.py must define LOOP_EVENT_PROMOTED_KEYS as the SSOT."""
        source = (WEB_AGENTS_DIR / "messages.py").read_text(encoding="utf-8")
        self.assertTrue(
            re.search(r"^LOOP_EVENT_PROMOTED_KEYS\s*[:=]", source, re.MULTILINE),
            "web/agents/messages.py must define LOOP_EVENT_PROMOTED_KEYS as the protocol SSOT",
        )


class TestSSEEventTypeSSOT(unittest.TestCase):
    """SSE event type strings must be defined as constants in messages.py."""

    def test_sse_event_types_defined_in_messages(self) -> None:
        """messages.py must define SSE_EVENT_TYPES as the SSOT."""
        source = (WEB_AGENTS_DIR / "messages.py").read_text(encoding="utf-8")
        self.assertTrue(
            re.search(r"^SSE_EVENT_TYPES\s*[:=]", source, re.MULTILINE),
            "web/agents/messages.py must define SSE_EVENT_TYPES",
        )

    def test_sse_event_types_not_duplicated(self) -> None:
        """SSE_EVENT_TYPES must not be redefined elsewhere in web/."""
        pattern = re.compile(r"^_?SSE_EVENT_TYPES\s*[:=]", re.MULTILINE)
        offenders: list[str] = []
        for py_file in sorted(WEB_DIR.rglob("*.py")):
            if py_file.name == "messages.py":
                continue
            if "__pycache__" in py_file.parts:
                continue
            source = py_file.read_text(encoding="utf-8")
            if pattern.search(source):
                offenders.append(str(py_file.relative_to(WEB_DIR)))
        self.assertEqual(
            offenders,
            [],
            f"Files redefining SSE_EVENT_TYPES (SSOT is messages.py): {offenders}",
        )

    def test_agent_router_uses_sse_event_types_constant(self) -> None:
        """agent.py must import SSE_EVENT_TYPES from messages, not hardcode strings."""
        source = (WEB_ROUTERS_DIR / "agent.py").read_text(encoding="utf-8")
        self.assertIn(
            "SSE_EVENT_TYPES",
            source,
            "agent.py must reference SSE_EVENT_TYPES from messages module",
        )


class TestRegisterEndpointAuth(unittest.TestCase):
    """POST /api/loop/register must validate the X-Loop-Source header."""

    def test_register_requires_source_header(self) -> None:
        """The register endpoint must check X-Loop-Source header."""
        source = (WEB_ROUTERS_DIR / "loop.py").read_text(encoding="utf-8")
        self.assertIn(
            "x-loop-source",
            source.lower(),
            "loop.py register endpoint must validate X-Loop-Source header",
        )


class TestNamingConventions(unittest.TestCase):
    """Pydantic request models follow the *Request suffix convention."""

    def test_pydantic_request_models_use_request_suffix(self) -> None:
        """All Pydantic BaseModel subclasses used as request bodies end
        with ``Request``."""
        forbidden_names = {"SaveConfigBody", "SaveConfigRequest", "AgentCreate", "AgentSubmit"}
        for py_file in sorted(WEB_DIR.rglob("*.py")):
            if "__pycache__" in py_file.parts:
                continue
            source = py_file.read_text(encoding="utf-8")
            for name in forbidden_names:
                pattern = re.compile(rf"^class {name}\(", re.MULTILINE)
                self.assertIsNone(
                    pattern.search(source),
                    f"{py_file.relative_to(WEB_DIR)} still uses old name {name} "
                    f"(rename to *Request)",
                )

    def test_all_request_models_end_with_request(self) -> None:
        """Every Pydantic BaseModel in web/routers must end with 'Request'."""
        offenders: list[str] = []
        for py_file in sorted(WEB_ROUTERS_DIR.glob("*.py")):
            if "__pycache__" in py_file.parts:
                continue
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                for base in node.bases:
                    base_name = ""
                    if isinstance(base, ast.Name):
                        base_name = base.id
                    elif isinstance(base, ast.Attribute):
                        base_name = base.attr
                    if base_name == "BaseModel" and not node.name.endswith("Request"):
                        offenders.append(f"{py_file.name}::{node.name}")
        self.assertEqual(
            offenders,
            [],
            f"Pydantic models not ending with 'Request': {offenders}",
        )


class TestCrossModuleUsageCoverage(unittest.TestCase):
    """web.agents modules must declare __all__ covering all external usage."""

    def test_spawn_has_all(self) -> None:
        source = (WEB_AGENTS_DIR / "spawn.py").read_text(encoding="utf-8")
        self.assertIn("__all__", source, "web/agents/spawn.py must declare __all__")

    def test_store_has_all(self) -> None:
        source = (WEB_AGENTS_DIR / "store.py").read_text(encoding="utf-8")
        self.assertIn("__all__", source, "web/agents/store.py must declare __all__")

    def test_spawn_all_covers_harness_contract(self) -> None:
        """Harness-contract symbols used by harness/tools/ must be in __all__."""
        source = (WEB_AGENTS_DIR / "spawn.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        declared: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    declared.add(elt.value)

        harness_contract = {
            "spawn_session",
            "monitor_exit",
            "init_wrapper_session",
            "append_loop_event",
            "finalize_wrapper_session",
        }
        missing = harness_contract - declared
        self.assertEqual(
            missing,
            set(),
            f"spawn.__all__ missing harness-contract symbols: {sorted(missing)}",
        )


class TestJSDeadCodeRemoved(unittest.TestCase):
    """Dead JS symbols identified by the architecture audit must stay removed."""

    _JS_DIR = WEB_DIR / "static" / "js"

    def _read_js(self, filename: str) -> str:
        return (self._JS_DIR / filename).read_text(encoding="utf-8")

    def test_app_js_dead_state_properties_removed(self) -> None:
        """VM state props that are assigned but never read must be removed."""
        source = self._read_js("app.js")
        dead_props = ["showLoopForm", "loopCreateEditorTab"]
        for prop in dead_props:
            self.assertNotIn(
                f"{prop}:",
                source.split("init(")[0],
                f"app.js still initializes dead state property '{prop}'",
            )

    def test_app_js_dead_counter_removed(self) -> None:
        source = self._read_js("app.js")
        self.assertNotIn(
            "_loopLogMessageCount",
            source,
            "app.js still contains dead counter _loopLogMessageCount (assigned but never read)",
        )

    def test_wrapper_bus_ended_flag_removed(self) -> None:
        source = self._read_js("app.js")
        self.assertNotIn(
            "bus.ended",
            source,
            "WrapperBus still sets bus.ended flag that is never read",
        )

    def test_recorder_dead_symbols_removed(self) -> None:
        source = self._read_js("recorder.js")
        self.assertNotIn(
            "state.starting",
            source,
            "recorder.js still sets state.starting (assigned but never read)",
        )
        self.assertNotIn(
            "currentRecorderId",
            source,
            "recorder.js still exports currentRecorderId (never called)",
        )

    def test_agents_dead_state_removed(self) -> None:
        source = self._read_js("agents.js")
        dead_state = ["managedLoading", "apiKeyPrompt"]
        for prop in dead_state:
            # Check the defaultState() return object for dead properties
            self.assertNotIn(
                f"{prop}:",
                source.split("function attach")[0],
                f"agents.js defaultState still contains dead property '{prop}'",
            )

    def test_agents_dead_methods_removed(self) -> None:
        source = self._read_js("agents.js")
        dead_methods = [
            "stopManagedRefresh",
            "managedPendingLabel",
            "isManagedChatStopping",
            "isManagedChatSending",
            "isManagedAgentResponding",
            "apiKeyPromptLabel",
        ]
        for method in dead_methods:
            pattern = re.compile(rf"vm\.{method}\s*=\s*function")
            self.assertIsNone(
                pattern.search(source),
                f"agents.js still defines dead method vm.{method}",
            )


class TestJSMinimalExportSurface(unittest.TestCase):
    """window.* exports must be as tight as possible."""

    _JS_DIR = WEB_DIR / "static" / "js"

    def _read_js(self, filename: str) -> str:
        return (self._JS_DIR / filename).read_text(encoding="utf-8")

    def test_agents_export_surface(self) -> None:
        """window.Agents must only export attach and renderMessages."""
        source = self._read_js("agents.js")
        match = re.search(r"return\s*\{([^}]+)\};\s*\n\}\)\(\);", source)
        self.assertIsNotNone(match, "Cannot find Agents IIFE return block")
        exports = {s.strip().rstrip(",") for s in match.group(1).split(",") if s.strip()}
        forbidden = exports - {"attach", "renderMessages"}
        self.assertEqual(
            forbidden,
            set(),
            f"window.Agents exports symbols not consumed externally: {sorted(forbidden)}. "
            "Only attach and renderMessages are used by app.js.",
        )


class TestMinimalPublicSurfaceExtended(unittest.TestCase):
    """Extended __all__ content checks for modules not yet covered."""

    def test_agent_router_has_all(self) -> None:
        source = (WEB_ROUTERS_DIR / "agent.py").read_text(encoding="utf-8")
        self.assertIn("__all__", source, "agent.py must declare __all__")

    def test_recording_router_has_all(self) -> None:
        source = (WEB_ROUTERS_DIR / "recording.py").read_text(encoding="utf-8")
        self.assertIn("__all__", source, "recording.py must declare __all__")

    def test_agent_router_all_only_exports_router(self) -> None:
        """agent.py __all__ must only export 'router'."""
        source = (WEB_ROUTERS_DIR / "agent.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        all_value: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    all_value.append(elt.value)
        self.assertEqual(
            all_value,
            ["router"],
            f"agent.py __all__ must be ['router'], got {all_value}",
        )

    def test_recording_router_all_only_exports_router(self) -> None:
        """recording.py __all__ must only export 'router'."""
        source = (WEB_ROUTERS_DIR / "recording.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        all_value: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    all_value.append(elt.value)
        self.assertEqual(
            all_value,
            ["router"],
            f"recording.py __all__ must be ['router'], got {all_value}",
        )

    def test_messages_all_does_not_leak_internals(self) -> None:
        """messages.__all__ must not export private rebuild helpers."""
        source = (WEB_AGENTS_DIR / "messages.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        all_value: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    all_value.append(elt.value)
        internal_only = {"_handle_loop_event", "_handle_tool_call", "_parse_stream_line"}
        leaked = internal_only & set(all_value)
        self.assertEqual(
            leaked,
            set(),
            f"messages.__all__ exports internal helpers: {sorted(leaked)}",
        )


class TestPerLoopConfigDirSSOT(unittest.TestCase):
    """The per-loop config dir must be derived through one web helper.

    ``loop.py`` historically computed ``<forge_train>/<id>/config`` two
    incompatible ways: id-based (``_loop_config_dir``) and workspace-based
    (``Path(workspace_dir).parent / "config"``) in four separate sites.
    Both resolve to the same directory only because ``workspace_dir`` is
    invariably ``<forge_train>/<id>/workspace`` — an unstated coupling.
    Pin the SSOT: ``_loop_config_dir(loop_id)`` is the single owner, and
    no ``.parent / "config"`` re-derivation may reappear.
    """

    _LOOP_PY = WEB_ROUTERS_DIR / "loop.py"

    def test_no_workspace_parent_config_derivation(self) -> None:
        source = self._LOOP_PY.read_text(encoding="utf-8")
        pattern = re.compile(r'\.parent\s*/\s*"config"')
        offenders = pattern.findall(source)
        self.assertEqual(
            offenders,
            [],
            "loop.py re-derives the per-loop config dir via "
            '`.parent / "config"`; route every site through '
            "`_loop_config_dir(loop_id)` (the web SSOT).",
        )

    def test_loop_config_dir_is_sole_definition(self) -> None:
        source = self._LOOP_PY.read_text(encoding="utf-8")
        defs = re.findall(r"^def _loop_config_dir\(", source, re.MULTILINE)
        self.assertEqual(
            len(defs),
            1,
            f"loop.py must define exactly one _loop_config_dir helper (found {len(defs)}).",
        )


class TestWebConfigDirMatchesHarnessLayout(unittest.TestCase):
    """web's config-dir formula must mirror ``harness.loop_layout``.

    The DAG forbids ``web`` from importing ``harness`` (harness depends on
    web, not the reverse — see ``TestWebDoesNotImportHarness``), so web
    cannot reuse ``harness.loop_layout.loop_config_dir`` by import. Like
    ``CONFIG_AXES``, the formula is therefore duplicated across the layer
    boundary by necessity. This test imports BOTH sides and asserts they
    produce byte-identical paths, so the two copies cannot silently drift.
    """

    def test_web_and_harness_config_dir_agree(self) -> None:
        from web.routers import loop as loop_router

        from harness.loop_layout import loop_config_dir as harness_config_dir

        forge_train = Path("/t/forge_train")
        loop_id = "abc123def456"
        with unittest.mock.patch.object(loop_router, "FORGE_TRAIN_DIR", forge_train):
            web_dir = loop_router._loop_config_dir(loop_id)
        self.assertEqual(web_dir, harness_config_dir(forge_train, loop_id))


if __name__ == "__main__":
    unittest.main()
