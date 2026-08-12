"""Step 9: anti-regression lint preventing reintroduction of hardcoded
timeout budgets anywhere outside the SSOT.

The SSOT redesign establishes one rule: per-suite wall-clock budgets
live in ``harness/config/eval/dense_training.toml`` under
``[evals.<suite>].timeout_s``. Any literal ``timeout NNN`` shell
fragment, any hardcoded fallback in Python, or any prose reference to a
specific budget number in agent-facing prompts re-introduces the
"agent guesses a number" failure mode that produced the original
75-minute backward-align incident.

This lint sweeps the repo for the regression patterns and fails the
suite immediately when a new offender appears. The whitelists are
intentionally narrow — the SSOT itself, the wrapper that ferries the
SSOT into ssh, and the tests that exercise the wrapper.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ── Sweep scope ─────────────────────────────────────────────────────────
#
# We restrict the sweep to the directories agents read/write during a
# loop run so the lint has well-defined coverage. Adding a new scope here
# is fine; removing one without replacement is a regression.
SWEEP_DIRS = [
    REPO_ROOT / "harness",  # Python core + CLI
    REPO_ROOT / "evals",  # dispatcher + runners
    REPO_ROOT / "prompt",  # agent-facing prompts
    REPO_ROOT / "tools",  # ssh wrappers, helper scripts
]


def _iter_files(*, extensions: set[str]) -> list[Path]:
    out: list[Path] = []
    for root in SWEEP_DIRS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in extensions:
                continue
            # Skip __pycache__, .venv, .ruff_cache, etc.
            if any(part.startswith(".") or part == "__pycache__" for part in path.parts):
                continue
            # Skip test files universally — they intentionally exercise
            # and document the very patterns this lint forbids. The lint
            # is about production code + agent-facing prompts.
            if path.name.startswith("test_") or "/tests/" in str(path):
                continue
            out.append(path)
    return out


class TestNoOuterTimeoutWrapperInShellOrPrompt(unittest.TestCase):
    """``timeout NNN`` / ``gtimeout NNN`` shell prefixes around a
    ``harness run`` invocation are the exact violation that triggered the
    SSOT redesign. The wrapper at ``tools/remote_run.sh`` is the only
    sanctioned site (it reads the SSOT and computes its window)."""

    # Tokens followed by a literal seconds count. We deliberately match
    # both bare ``timeout 900`` and the GNU coreutils variants
    # (``gtimeout``, ``coreutils-timeout``). The negative lookbehind on
    # ``-`` excludes the sanctioned ``--timeout NN`` override flag (a
    # CLI argument to ``harness run``, not an outer wrapper). The
    # pattern requires a numeric token immediately after, with optional
    # unit suffix so ``timeout 90s`` and ``timeout 1m`` are caught too.
    PATTERN = re.compile(r"(?<![-\w])(?:g?timeout|coreutils-timeout)\s+\d+[smh]?\b")

    # Single-line whitelist by absolute path → set of substrings on
    # the offending line that mark a sanctioned occurrence.
    LINE_WHITELIST = {
        # The wrapper itself runs ``exec timeout --kill-after=30 ...``
        # with ``${outer}s`` — there is no numeric literal on that line,
        # so the regex won't fire. The whitelist entry exists as
        # documentation of intent and a safety net if the script ever
        # gains a literal default for the buffer constant.
        str(REPO_ROOT / "tools" / "remote_run.sh"): set(),
    }

    def test_no_outer_timeout_wrapper_anywhere(self) -> None:
        offenders: list[str] = []
        for path in _iter_files(extensions={".sh", ".md", ".py", ".toml"}):
            # The lint test files themselves talk about the pattern.
            if path.name.startswith("test_no_hardcoded_budgets"):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for lineno, line in enumerate(text.splitlines(), 1):
                if not self.PATTERN.search(line):
                    continue
                if path.suffix == ".py" and self._is_python_fixture_use(line):
                    continue
                # Whitelist match (must include any of the marker substrings).
                wl_markers = self.LINE_WHITELIST.get(str(path))
                if wl_markers is not None:
                    # Empty marker set means "no offender expected on any line"
                    # — fall through and flag if one appears.
                    if not wl_markers or any(m in line for m in wl_markers):
                        continue
                offenders.append(f"{path}:{lineno}: {line.strip()}")
        self.assertEqual(
            offenders,
            [],
            "outer `timeout NNN` wrapper detected — route through "
            "tools/remote_run.sh instead:\n  " + "\n  ".join(offenders),
        )

    @staticmethod
    def _is_python_fixture_use(line: str) -> bool:
        # Lines inside Python test fixtures often build mock shell
        # commands like ``["timeout", "2", "bash", ...]``. Those are
        # legitimate uses of the GNU ``timeout`` binary for the test
        # subject, not the wrapper-style abuse this lint targets.
        stripped = line.lstrip()
        if stripped.startswith("#"):
            return True
        # Bracketed/comma-delimited list form OR explicit subprocess args
        # form: both characterise unit-test fixtures, not production
        # invocations.
        return ("[" in line and "," in line) or "subprocess" in line.lower()


class TestNoHardcodedTimeoutDefaultInPython(unittest.TestCase):
    """``cfg.get("timeout_s", <literal int>)`` re-introduces a second
    default source for the same suite budget. The SSOT helpers
    ``config_runtime.suite_timeout_s`` (candidate side) and
    ``config_runtime.suite_ref_timeout_s`` (ref side) are the only
    sanctioned readers."""

    PATTERN = re.compile(r'cfg\.get\(\s*["\']timeout_s["\']\s*,\s*[^)]+\)')

    # The only sanctioned ``cfg.get("timeout_s", ...)`` sites are the SSOT
    # helpers themselves, where it is the *inner* fallback for the
    # ``suite_timeout_s`` / ``suite_ref_timeout_s`` resolution. Every
    # runtime caller (``evals/_common.py`` ref capture included) reads the
    # budget through those helpers, not via its own ``cfg.get`` fallback.
    SENTINEL_WHITELIST = {
        str(REPO_ROOT / "harness" / "config_runtime.py"): {
            "suite_timeout_s",
            "suite_ref_timeout_s",
            'cfg.get("timeout_s", fallback)',
        },
    }

    def test_no_cfg_get_timeout_s_with_default(self) -> None:
        offenders: list[str] = []
        for path in _iter_files(extensions={".py"}):
            text = path.read_text(encoding="utf-8", errors="replace")
            for lineno, line in enumerate(text.splitlines(), 1):
                if not self.PATTERN.search(line):
                    continue
                # Comments are documentation, not live code.
                if line.lstrip().startswith("#"):
                    continue
                wl = self.SENTINEL_WHITELIST.get(str(path), set())
                if any(marker in line for marker in wl):
                    continue
                offenders.append(f"{path}:{lineno}: {line.strip()}")
        self.assertEqual(
            offenders,
            [],
            "Use config_runtime.suite_timeout_s(workload_config, suite) "
            "instead of cfg.get(...,fallback):\n  " + "\n  ".join(offenders),
        )


class TestNoWallClockTableInPrompts(unittest.TestCase):
    """The deleted prompt table ("M3≈5–10min / M4≈10–20min / ...") is the
    canonical example of prose-level dual-sourcing — agents read it,
    internalize the number, and then hand-write a ``timeout NNN``. The
    only sanctioned reference is the helper ``harness budget <suite>``.
    """

    # Catch resurrected approximations like ``≈ 5 min`` / ``~30 min`` /
    # ``about 1.5 h`` adjacent to a suite name. We keep the pattern
    # specific to "minute"/"hour" tokens so generic timing prose in
    # other docs is not affected.
    APPROX_PATTERN = re.compile(
        r"(?:≈|~|about|approximately)\s*\d+(?:\.\d+)?\s*(?:min|minute|hour|h\b)",
        re.IGNORECASE,
    )

    def test_no_wall_clock_approximations_in_remote_execution_prompt(self) -> None:
        target = REPO_ROOT / "prompt" / "develop_prompt" / "remote-execution.md"
        if not target.exists():
            self.skipTest("remote-execution.md not present in this checkout")
        text = target.read_text(encoding="utf-8")
        offenders: list[str] = []
        for lineno, line in enumerate(text.splitlines(), 1):
            if self.APPROX_PATTERN.search(line):
                # The "Long-running commands" section legitimately
                # mentions approximate runtimes as motivation for the
                # foreground vs. background pattern choice. The SSOT
                # itself is queried via `harness budget`, not pulled
                # from these approximations.
                if "background" in line.lower() or "foreground" in line.lower():
                    continue
                offenders.append(f"{target}:{lineno}: {line.strip()}")
        self.assertEqual(
            offenders,
            [],
            "wall-clock approximations re-introduced in agent prompt — "
            "point agents at `harness budget <suite>` instead:\n  " + "\n  ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
