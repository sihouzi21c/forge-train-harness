"""Guard: no undefined names (pyflakes rule F821) anywhere in harness/.

Why this exists as a TEST and not only as the pre-commit ruff hook: the
thin-dispatcher step-7 extraction moved a module-constant block out of
``evals/dispatcher.py`` while its in-function consumers stayed behind — an
import-clean, unittest-green NameError that only fired at gate runtime on
the GPU host (every ``needs_ref`` gate died). Behavior tests mock above the
broken frame and clones without pre-commit hooks installed never ran ruff,
so nothing static stood between the bug and a commit. F821 catches that
whole class in under a second.

Skips (with an actionable message) when ruff is not on PATH — the loop
workspaces' minimal envs don't carry dev tooling; dev machines and CI
should (``pip install -e harness/[dev]``).
"""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

# harness/tests/test_lint_undefined_names.py → harness/ (the ruff config root).
LINT_ROOT = Path(__file__).resolve().parents[2]


class TestNoUndefinedNames(unittest.TestCase):
    def test_ruff_f821_clean(self) -> None:
        ruff = shutil.which("ruff")
        if ruff is None:
            self.skipTest(
                "ruff not on PATH — install dev tooling "
                "(pip install -e harness/[dev]) to run the F821 guard"
            )
        proc = subprocess.run(
            [ruff, "check", "--select", "F821", str(LINT_ROOT)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            proc.returncode,
            0,
            msg=(
                "undefined names in the harness python surface "
                "(ruff F821):\n" + proc.stdout + proc.stderr
            ),
        )


if __name__ == "__main__":
    unittest.main()
