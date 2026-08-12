"""Regression: prompts and READMEs must never call the bare ``harness`` CLI.

Commit B deleted the global ``[project.scripts]`` harness entrypoint —
every workspace now ships its own ``<workspace>/bin/harness`` shim. If a
prompt or README instructs the agent to run ``harness run multistep-1gpu``
the agent will hit ``command not found`` on the remote (no global script
exists) and waste rounds re-deriving the correct invocation.

This test fails if any markdown under ``harness/prompt`` or the four
top-level READMEs still contains a bare ``harness <subcommand>`` token.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# A bare ``harness <subcommand>`` call: ``harness`` preceded by a delimiter
# that is NOT a path character (so ``bin/harness`` and ``harness.cli`` do
# not match), and followed by whitespace + a recognised subcommand.
BARE_CLI_RE = re.compile(r"(^|[\s`$(])harness\s+(run|sync|doctor|info|budget)\b")

SCAN_FILES: list[Path] = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "README_zh.md",
    REPO_ROOT / "harness" / "README.md",
    REPO_ROOT / "harness" / "README_zh.md",
]
SCAN_GLOBS: list[tuple[Path, str]] = [
    (REPO_ROOT / "harness" / "prompt", "**/*.md"),
]


def _gather_targets() -> list[Path]:
    files = [p for p in SCAN_FILES if p.is_file()]
    for root, pattern in SCAN_GLOBS:
        files.extend(sorted(root.rglob(pattern)))
    return files


class TestNoBareHarnessCLI(unittest.TestCase):
    def test_no_bare_harness_cli_in_prompts_or_readmes(self) -> None:
        violations: list[str] = []
        for path in _gather_targets():
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if BARE_CLI_RE.search(line):
                    rel = path.relative_to(REPO_ROOT)
                    violations.append(f"{rel}:{lineno}: {line.strip()}")
        self.assertFalse(
            violations,
            "Prompts/READMEs reference the deprecated bare `harness` CLI. "
            "Replace with `bin/harness <subcommand>` (workspace shim) or "
            "`python3 -m harness.cli <subcommand>` (manual checkout):\n" + "\n".join(violations),
        )


if __name__ == "__main__":
    unittest.main()
