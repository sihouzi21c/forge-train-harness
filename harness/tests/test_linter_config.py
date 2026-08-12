"""Local lint/architecture hooks should not depend on undeclared tools."""

from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MONOREPO_ROOT = REPO_ROOT.parent


class TestLinterConfig(unittest.TestCase):
    def test_precommit_does_not_call_import_linter(self) -> None:
        text = (MONOREPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
        self.assertNotIn("importlinter", text)
        self.assertNotIn("import-linter", text)


if __name__ == "__main__":
    unittest.main()
