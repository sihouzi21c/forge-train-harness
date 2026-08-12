"""Step 8: every dispatcher suite reads its budget through
``config_runtime.suite_timeout_s``, never via a private derive.

The pre-fix dispatcher had two derivation branches
(``max(_DEFAULT_TIMEOUT, num_steps * 10 + 600)`` for ``loss-gate-200``
and ``max(_DEFAULT_TIMEOUT, num_steps * 20 + 600)`` for
``resume-gate-20``). Those branches re-introduced the same "default
value lives in two places" failure mode the SSOT redesign exists to
prevent. This test pins the dispatcher to a single source: the toml.
"""

from __future__ import annotations

import os
import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["FORGE_REPO_ROOT"] = str(REPO_ROOT)

DISPATCHER = REPO_ROOT / "evals" / "dispatcher.py"


class TestDispatcherUsesSsotHelper(unittest.TestCase):
    def test_no_derive_branches_remain(self) -> None:
        src = DISPATCHER.read_text(encoding="utf-8")
        # The two historical derive expressions, plus any future variant
        # that multiplies num_steps to manufacture a budget.
        offenders = re.findall(r"max\s*\(\s*_DEFAULT_TIMEOUT\s*,\s*num_steps\s*\*", src)
        self.assertEqual(offenders, [], "dispatcher must not derive timeout from num_steps")

    def test_no_remaining_cfg_get_timeout_s_fallback(self) -> None:
        src = DISPATCHER.read_text(encoding="utf-8")
        # ``cfg.get("timeout_s", ...)`` re-introduces a second default
        # source. The helper is the only sanctioned reader.
        offenders = re.findall(r'cfg\.get\(\s*"timeout_s"', src)
        self.assertEqual(
            offenders,
            [],
            "dispatcher must call config_runtime.suite_timeout_s instead",
        )

    def test_default_timeout_constant_removed(self) -> None:
        src = DISPATCHER.read_text(encoding="utf-8")
        # The historical module-level fallback ``_DEFAULT_TIMEOUT = ...``
        # has no callers anymore; leaving it around invites the next
        # author to wire a second source.
        self.assertNotIn("_DEFAULT_TIMEOUT", src)


class TestTomlPinsAllPreviouslyDerivedSuites(unittest.TestCase):
    """If ``timeout_s`` disappears from the toml for a suite the
    dispatcher previously derived for, the SSOT helper will fall back
    to ``default_timeout_s()`` — silently overriding the frozen value.
    Pin the two suites explicitly in the committed template (the
    active ``harness/config/eval.toml`` is a gitignored cp of this
    template, so we read the template directly to avoid depending on
    the developer's local copy)."""

    @classmethod
    def setUpClass(cls) -> None:
        import tomllib

        template = REPO_ROOT / "config" / "eval" / "dense_training" / "dense_training.toml"
        with template.open("rb") as fh:
            cls.wl = tomllib.load(fh)

    def test_resume_gate_20_pins_timeout_s(self) -> None:
        self.assertIn("timeout_s", self.wl["evals"]["resume-gate-20"])
        # The historical derive landed at 1800; freeze that value.
        self.assertEqual(self.wl["evals"]["resume-gate-20"]["timeout_s"], 1800)

    def test_loss_gate_200_pins_timeout_s(self) -> None:
        self.assertIn("timeout_s", self.wl["evals"]["loss-gate-200"])
        # ``max(1800, 200*10+600) == 2600`` — the historical derive value.
        self.assertEqual(self.wl["evals"]["loss-gate-200"]["timeout_s"], 2600)


if __name__ == "__main__":
    unittest.main()
