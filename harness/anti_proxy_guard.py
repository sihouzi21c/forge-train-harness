"""Anti-proxy guard: lint the candidate engine for proxy / synthesis smells.

Stage 1 dev agent owns ``workload/src/training_engine_tensor/`` and
(Stage 2) ``workload/ops/``. Any sign that the candidate engine is
shell-out'ing to the reference scripts, importing reference-side helpers
(``ref_script_runner`` / ``evals.harness_hook``), hardcoding synthetic
metric kwargs (``mfu_e2e_*`` / ``global_loss`` / ``grad_norm``), or
using ``_synthetic_`` / ``_fake_`` / ``_ref_proxy_`` / ``_run_ref_``
named code paths is a proxy pattern: the candidate is leaning on the
reference to pass bitwise / perf gates instead of computing the work
itself.

Why a separate guard from :mod:`harness.framework_guard`
--------------------------------------------------------

The framework guard scans the *entire* repo for banned framework
imports (``megatron`` / ``deepspeed`` / ``torch.nn`` / ...) with a
broad ALLOWLIST. The anti-proxy guard scans **only** the candidate
engine write surface (``workload/src/`` and ``workload/ops/``) because
dispatchers, ref-as-gate helpers (``tools/ref_script_runner.py``),
and bridges legitimately use ``subprocess`` and reference paths;
flagging them would be noise. The pattern list lives here, not in any
review prompt — prompts only invoke ``harness run anti-proxy`` and read
the verdict, so the rule set has exactly one authoritative source.

Why we skip docstrings / comments
---------------------------------

The candidate stub legitimately *describes* the reference contract in
its module docstring (``ref/reference/<ref_script>``,
``evals.harness_hook``, "reference run", ...). Those are architectural
prose, not engine code. We parse the file with ``ast`` to identify
module / function / class docstring ranges and skip them, and we also
skip lines starting with ``#``. Anything outside those windows — bare
string literals, imports, function names, kwarg literals — is treated
as engine code and checked against the pattern set.
"""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from harness.framework_guard import (
    IGNORED_DIR_NAMES,
    GuardViolation,
    repo_root,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


CANDIDATE_SCAN_ROOTS = ("workload/src", "workload/ops")
SCAN_SUFFIXES = frozenset({".py"})


# (substring, reason). One substring → one violation; multiple substrings
# matching the same line each produce their own violation entry (a line
# like ``Path("ref") / "bridges" / "bridge.sh"`` reports both ``ref/``
# and ``bridge.sh``, which is the correct postmortem signal).
ANTI_PROXY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("ref/", "candidate references ref/ path literal"),
    ("/reference", "candidate references reference/ path segment"),
    ('"reference"', 'candidate quotes "reference" path segment'),
    ("'reference'", "candidate quotes 'reference' path segment"),
    ("bridge.sh", "candidate references ref bridge script"),
    ("pretrain.py", "candidate references reference pretrain script"),
    ("ref_script_runner", "candidate imports ref_script_runner helper"),
    ("evals.harness_hook", "candidate imports evals.harness_hook"),
    ("subprocess.run", "candidate uses subprocess.run (proxy-to-ref smell)"),
    ("subprocess.Popen", "candidate uses subprocess.Popen"),
    ("subprocess.call", "candidate uses subprocess.call"),
    ("subprocess.check_output", "candidate uses subprocess.check_output"),
    ("subprocess.check_call", "candidate uses subprocess.check_call"),
    ("os.system", "candidate uses os.system"),
    ("os.execv", "candidate uses os.execv*"),
    ("os.execl", "candidate uses os.execl*"),
    ("os.popen", "candidate uses os.popen"),
    ("_synthetic_", "candidate name contains _synthetic_"),
    ("_fake_", "candidate name contains _fake_"),
    ("_ref_proxy_", "candidate name contains _ref_proxy_"),
    ("_run_ref_", "candidate name contains _run_ref_"),
)


# Hardcoded metric kwargs: ``mfu_e2e_standard=99.0`` baked into a call
# site instead of computed from a real timing measurement. Excludes
# obvious placeholder defaults (0 / 0.0 / 1 / 1.0) so signatures with
# ``def f(grad_norm=0.0)`` don't trip the rule.
HARDCODED_METRIC_PATTERN = re.compile(
    r"\b(mfu_e2e_\w+|global_loss|grad_norm)\s*=\s*"
    r"(?!(?:0(?:\.0+)?|1(?:\.0+)?)\b)"
    r"([+-]?\d+(?:\.\d+)?)\b"
)


def scan_line(line: str) -> list[str]:
    """Return all reasons matched by *line* (may be empty)."""
    reasons: list[str] = []
    for substr, reason in ANTI_PROXY_PATTERNS:
        if substr in line:
            reasons.append(reason)
    metric_match = HARDCODED_METRIC_PATTERN.search(line)
    if metric_match:
        kwarg, value = metric_match.group(1), metric_match.group(2)
        reasons.append(f"candidate hardcodes {kwarg}={value} (synthetic metric kwarg)")
    return reasons


def _docstring_line_ranges(source: str) -> set[int]:
    """Return line numbers covered by module / func / class docstrings.

    Anything else (assignments, bare string statements that are not the
    first body element, ``__doc__ = "..."``, ...) is treated as engine
    code and stays in the scan.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    skip: set[int] = set()
    nodes = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for node in ast.walk(tree):
        if not isinstance(node, nodes):
            continue
        if not getattr(node, "body", None):
            continue
        first = node.body[0]
        if not isinstance(first, ast.Expr):
            continue
        value = first.value
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            continue
        start = first.lineno
        end = getattr(first, "end_lineno", start) or start
        skip.update(range(start, end + 1))
    return skip


def scan_file(path: Path) -> list[GuardViolation]:
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: failed to decode as UTF-8") from exc

    skip_lines = _docstring_line_ranges(source)
    violations: list[GuardViolation] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        if line_number in skip_lines:
            continue
        if line.lstrip().startswith("#"):
            continue
        for reason in scan_line(line):
            violations.append(
                GuardViolation(
                    path=path,
                    line_number=line_number,
                    line=line.strip(),
                    reason=reason,
                )
            )
    return violations


def _walk_candidate_files(root: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIR_NAMES]
        for name in filenames:
            candidate = Path(dirpath) / name
            if candidate.suffix in SCAN_SUFFIXES and candidate.is_file():
                yield candidate.resolve()


def scan_paths(raw_paths: Sequence[str]) -> list[GuardViolation]:
    """Scan the candidate engine write surface.

    With empty *raw_paths* walks the configured ``CANDIDATE_SCAN_ROOTS``
    under the repo root (the production code path the CLI uses); with
    explicit paths scans those instead (the unit-test code path with
    tempdir fixtures).
    """
    targets: list[Path] = []
    if raw_paths:
        for raw in raw_paths:
            p = Path(raw)
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            if p.is_dir():
                targets.extend(_walk_candidate_files(p))
            elif p.is_file() and p.suffix in SCAN_SUFFIXES:
                targets.append(p.resolve())
    else:
        root = repo_root()
        for scan_root in CANDIDATE_SCAN_ROOTS:
            d = root / scan_root
            if d.is_dir():
                targets.extend(_walk_candidate_files(d))

    seen: set[Path] = set()
    violations: list[GuardViolation] = []
    for path in targets:
        if path in seen:
            continue
        seen.add(path)
        violations.extend(scan_file(path))
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        violations = scan_paths(args)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    if not violations:
        return 0

    print(
        f"Anti-proxy violations in the candidate engine ({len(violations)} found):",
        file=sys.stderr,
    )
    print(file=sys.stderr)
    for violation in violations:
        print(f"  {violation.render()}", file=sys.stderr)
    print(file=sys.stderr)
    print(
        "Candidate engine (workload/src/, workload/ops/) must implement "
        "forward / backward / optimizer / loss / metric in-process. "
        "Shell-out to ref/ scripts, imports of ref-side helpers, "
        "synthetic function names, and hardcoded metric kwargs are all "
        "proxy patterns; see harness/anti_proxy_guard.py for the "
        "authoritative list.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
