"""Unit coverage for the Stage 1 milestone helper used by agent-loop.sh.

The helper at ``harness/tools/agent_loop_milestone.sh`` owns the pure
I/O contracts the agent loop depends on to advance Stage 1 milestones:

* ``parse_commit_milestone_pass`` reads ``git log --format=%B`` output
  on stdin and emits the furthest-along milestone NAME declared via a
  literal ``MILESTONE_STATUS: <name> PASS`` marker.
* ``validate_milestone`` enforces membership in the milestone order.
* ``read_stage_milestone_file`` / ``write_stage_milestone_file`` do the
  file I/O, defaulting to the first milestone and garbage-resetting.

The ordering SSOT is injected via ``FORGE_MILESTONE_ORDER`` (a
space-separated list of milestone names) — agent-loop.sh resolves it from
the active ``eval.toml`` ``[stage1].milestone_order``; the helpers stay
python-free so each can be unit-tested in isolation by setting that env.

These are the seams the loop uses to decide "which milestone's MD do I
splice into the next dev prompt?", so regressions here change the prompt
the dev agent sees the very next round. Wrapper-side glue in
``agent-loop.sh`` is covered by the source-text class at the bottom.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
HARNESS_DIR = REPO_ROOT / "harness"
HELPER = HARNESS_DIR / "tools" / "agent_loop_milestone.sh"
AGENT_LOOP_SH = HARNESS_DIR / "agent-loop.sh"

# Canonical Stage 1 milestone order mirrored by the test (the production
# SSOT is eval.toml [stage1].milestone_order; agent-loop.sh injects it).
ORDER = [
    "alignment",
    "bitwise-singlecard",
    "bitwise-multicard",
    "bitwise-perf",
    "resume",
    "long-horizon",
    "production",
]
ORDER_ENV = " ".join(ORDER)


def _run(args: list[str], stdin: str = "") -> subprocess.CompletedProcess:
    env = {**os.environ, "FORGE_MILESTONE_ORDER": ORDER_ENV}
    return subprocess.run(
        ["bash", str(HELPER), *args],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
        env=env,
    )


class TestValidateMilestone(unittest.TestCase):
    """validate_milestone must accept names in the order and reject the rest."""

    def test_helper_script_exists(self) -> None:
        self.assertTrue(HELPER.is_file(), f"expected helper at {HELPER}")

    def test_accepts_every_milestone_in_order(self) -> None:
        for m in ORDER:
            with self.subTest(m=m):
                self.assertEqual(_run(["validate", m]).returncode, 0)

    def test_rejects_unknown_and_garbage(self) -> None:
        for bogus in (
            "M1",
            "M8",
            "bitwise",
            "",
            "alignment.forward",
            "stage1",
            "production ",
            " resume",
        ):
            with self.subTest(bogus=bogus):
                self.assertNotEqual(_run(["validate", bogus]).returncode, 0)

    def test_fails_fast_without_order_env(self) -> None:
        # No FORGE_MILESTONE_ORDER -> the progression cannot be resolved,
        # so validate must NOT silently accept anything.
        res = subprocess.run(
            ["bash", str(HELPER), "validate", "alignment"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            env={k: v for k, v in os.environ.items() if k != "FORGE_MILESTONE_ORDER"},
        )
        self.assertNotEqual(res.returncode, 0)


class TestParseCommitMilestonePass(unittest.TestCase):
    """parse_commit_milestone_pass owns the marker-shape contract."""

    def test_empty_input_emits_nothing(self) -> None:
        res = _run(["parse"], stdin="")
        self.assertEqual(res.stdout, "")
        # Returncode MUST be 0 even on no match. agent-loop.sh sources
        # this helper into a `set -euo pipefail` shell; if the inner
        # grep's no-match exit 1 ever surfaces, advance_stage_milestone
        # aborts the wrapper on every round before any milestone PASS.
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_parse_under_strict_pipefail_survives_no_match(self) -> None:
        # Real-world call shape from agent-loop.sh:advance_stage_milestone.
        # The inner pipeline must not propagate `grep` exit 1 to the
        # caller, or the wrapper -- running under `set -euo pipefail`
        # -- terminates the entire loop on the very first Stage 1 round.
        script = (
            "set -euo pipefail\n"
            f"export FORGE_MILESTONE_ORDER='{ORDER_ENV}'\n"
            f"source '{HELPER}'\n"
            "out=$(printf 'unrelated\\n' | parse_commit_milestone_pass)\n"
            'printf \'rc=%s out=%q\\n\' "$?" "$out"\n'
        )
        res = subprocess.run(
            ["bash", "-c", script],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("rc=0 out=''", res.stdout)

    def test_returns_furthest_along_declared(self) -> None:
        log = (
            "\n".join(
                [
                    "MILESTONE_STATUS: alignment PASS",
                    "unrelated commit body",
                    "MILESTONE_STATUS: bitwise-multicard PASS",
                    "MILESTONE_STATUS: bitwise-singlecard PASS",
                ]
            )
            + "\n"
        )
        # bitwise-multicard (index 2) is further along than singlecard (1)
        # and alignment (0), regardless of commit order.
        self.assertEqual(_run(["parse"], stdin=log).stdout.strip(), "bitwise-multicard")

    def test_single_declaration_round_trips(self) -> None:
        for m in ORDER:
            with self.subTest(m=m):
                log = f"MILESTONE_STATUS: {m} PASS\n"
                self.assertEqual(_run(["parse"], stdin=log).stdout.strip(), m)

    def test_ignores_prose_with_pass_word(self) -> None:
        # Strict-marker check: anything that doesn't look exactly like
        # `^MILESTONE_STATUS: <name> PASS$` (modulo trailing whitespace)
        # must NOT count as a milestone declaration.
        log = (
            "\n".join(
                [
                    "Refactor: tweak bitwise-multicard PASS gate handling",
                    "fix(bitwise-singlecard): bug",
                    "milestone_status: bitwise-multicard pass",  # lowercase
                    "MILESTONE_STATUS: bitwise-multicard PASSED",  # wrong terminator
                    "MILESTONE_STATUS:bitwise-multicard PASS",  # no whitespace after colon
                    "  MILESTONE_STATUS: bitwise-multicard PASS",  # leading whitespace
                    "MILESTONE_STATUS: bitwise-multicard PASS extra",  # trailing junk
                ]
            )
            + "\n"
        )
        self.assertEqual(_run(["parse"], stdin=log).stdout, "")

    def test_tolerates_multi_space_and_tabs(self) -> None:
        # The marker uses `[[:space:]]+` between fields so extra
        # spaces / tabs in well-formed commit messages still count.
        log = "MILESTONE_STATUS:\tbitwise-perf \t PASS\t\n"
        self.assertEqual(_run(["parse"], stdin=log).stdout.strip(), "bitwise-perf")

    def test_ignores_names_not_in_order(self) -> None:
        # Well-formed markers naming a milestone outside the order must be
        # ignored so a malformed commit can't escape the linear plan.
        log = (
            "\n".join(
                [
                    "MILESTONE_STATUS: M8 PASS",
                    "MILESTONE_STATUS: bogus PASS",
                ]
            )
            + "\n"
        )
        self.assertEqual(_run(["parse"], stdin=log).stdout, "")


class TestNextMilestoneAfter(unittest.TestCase):
    """next_milestone_after computes the next active milestone for advance_stage_milestone.

    A ``MILESTONE_STATUS: <name> PASS`` commit declares that <name>'s gate
    has passed, so the next active milestone is its successor in the order
    (capped at the last entry). The helper returns empty when no advance is
    warranted (already at or past the successor).
    """

    def test_first_pass_advances_to_second(self) -> None:
        self.assertEqual(
            _run(["next-after", "alignment", "alignment"]).stdout, "bitwise-singlecard"
        )

    def test_already_advanced_returns_empty(self) -> None:
        self.assertEqual(_run(["next-after", "alignment", "bitwise-singlecard"]).stdout, "")
        self.assertEqual(_run(["next-after", "bitwise-singlecard", "bitwise-multicard"]).stdout, "")

    def test_jump_ahead_skips_intermediate_milestones(self) -> None:
        # Declaring bitwise-multicard PASS while persisted is alignment means
        # the first three are passed; next active = bitwise-perf.
        self.assertEqual(
            _run(["next-after", "bitwise-multicard", "alignment"]).stdout, "bitwise-perf"
        )

    def test_penultimate_pass_advances_to_terminal(self) -> None:
        self.assertEqual(_run(["next-after", "long-horizon", "resume"]).stdout, "production")

    def test_terminal_pass_caps_at_terminal(self) -> None:
        # production is the last milestone. Declaring it PASS from
        # long-horizon advances to production (final). STAGE_STATUS=finished,
        # not the milestone counter, is what ends stage1.
        self.assertEqual(_run(["next-after", "production", "long-horizon"]).stdout, "production")

    def test_terminal_pass_when_already_terminal_is_noop(self) -> None:
        self.assertEqual(_run(["next-after", "production", "production"]).stdout, "")

    def test_unknown_high_milestone_does_not_advance(self) -> None:
        # parse_commit_milestone_pass already filters names outside the
        # order, but if a bad value sneaks through, never advance on it.
        self.assertEqual(_run(["next-after", "bogus", "alignment"]).stdout, "")


class TestStageMilestoneFileIO(unittest.TestCase):
    """File-level read/write helpers."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)
        self.target = self.tmpdir / "stage1.milestone"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_read_missing_file_returns_first_milestone(self) -> None:
        res = _run(["read", str(self.target)])
        self.assertEqual(res.returncode, 0)
        self.assertEqual(res.stdout, "alignment")

    def test_write_then_read_round_trips(self) -> None:
        for m in ORDER:
            with self.subTest(m=m):
                wres = _run(["write", str(self.target), m])
                self.assertEqual(wres.returncode, 0, wres.stderr)
                rres = _run(["read", str(self.target)])
                self.assertEqual(rres.stdout, m)

    def test_write_rejects_invalid_value(self) -> None:
        # Write must validate and refuse to clobber the file with junk.
        self.target.write_text("bitwise-multicard\n", encoding="utf-8")
        wres = _run(["write", str(self.target), "bogus"])
        self.assertNotEqual(wres.returncode, 0)
        self.assertIn("bogus", wres.stderr)
        self.assertEqual(self.target.read_text().strip(), "bitwise-multicard")

    def test_write_leaves_no_stray_tmp_files(self) -> None:
        wres = _run(["write", str(self.target), "bitwise-perf"])
        self.assertEqual(wres.returncode, 0, wres.stderr)
        siblings = sorted(p.name for p in self.tmpdir.iterdir())
        self.assertEqual(siblings, [self.target.name], f"unexpected files in state dir: {siblings}")

    def test_write_failure_does_not_truncate_existing_file(self) -> None:
        self.target.write_text("resume\n", encoding="utf-8")
        wres = _run(["write", str(self.target), "garbage"])
        self.assertNotEqual(wres.returncode, 0)
        self.assertEqual(self.target.read_text().strip(), "resume")
        siblings = sorted(p.name for p in self.tmpdir.iterdir())
        self.assertEqual(siblings, [self.target.name])

    def test_read_resets_garbage_to_first_milestone(self) -> None:
        # Corrupt content must self-heal so the loop never crashes on a
        # tampered state file; the warning routes to stderr so the
        # agent-loop wrapper can elevate it to an info loop_event.
        self.target.write_text("not-a-milestone\n", encoding="utf-8")
        res = _run(["read", str(self.target)])
        self.assertEqual(res.stdout, "alignment")
        self.assertIn("corrupt milestone file", res.stderr)
        self.assertEqual(self.target.read_text().strip(), "alignment")


class TestAgentLoopShWiring(unittest.TestCase):
    """Pin the agent-loop.sh integration so the helpers actually run.

    These are intentionally source-text checks — they're cheap, they
    catch the common regressions (forgotten source line, dropped
    advance_stage_milestone call, prompt header that lost the milestone
    block), and they don't require provisioning a workspace.
    """

    def setUp(self) -> None:
        self.wrapper = AGENT_LOOP_SH.read_text(encoding="utf-8")

    def test_wrapper_sources_milestone_helper(self) -> None:
        self.assertIn("tools/agent_loop_milestone.sh", self.wrapper)

    def test_wrapper_exports_milestone_order(self) -> None:
        # The python-free helpers need the order injected; the wrapper must
        # resolve it from agent_loop_config.py and export it.
        self.assertIn("FORGE_MILESTONE_ORDER", self.wrapper)
        self.assertIn("export FORGE_MILESTONE_ORDER", self.wrapper)

    def test_stage_milestone_state_file_lives_under_state_dir(self) -> None:
        self.assertIn(
            "printf '%s/%s.milestone\\n'",
            self.wrapper,
            "stage_milestone_file must compose <state_dir>/<stage>.milestone",
        )

    def test_advance_stage_milestone_runs_after_stage_status_write(self) -> None:
        status_write_idx = self.wrapper.index('write_stage_status "$stage" "$stage_status"')
        advance_idx = self.wrapper.index('advance_stage_milestone "$stage"')
        self.assertLess(
            status_write_idx,
            advance_idx,
            "advance_stage_milestone must follow write_stage_status",
        )

    def test_build_prompt_called_inside_round_loop_with_milestone(self) -> None:
        self.assertIn(
            'build_prompt "$stage" "$current_milestone"',
            self.wrapper,
        )
        self.assertNotIn(
            'PROMPT_PATH="$LOG_DIR/${stage}_prompt.md"',
            self.wrapper,
            "the stage-level PROMPT_PATH was replaced by per-round paths",
        )

    def test_round_start_event_carries_active_milestone(self) -> None:
        self.assertIn(
            'milestone "$current_milestone"',
            self.wrapper,
        )

    def test_reset_state_also_clears_milestone_files(self) -> None:
        self.assertIn(
            "*.milestone",
            self.wrapper,
            "--reset-state must clear *.milestone alongside *.status",
        )

    def test_stage_has_milestones_helper_exists(self) -> None:
        self.assertIn("stage_has_milestones() {", self.wrapper)
        self.assertIn("stage-milestones", self.wrapper)

    def test_run_stage_guards_milestone_read_with_helper(self) -> None:
        self.assertIn(
            'if stage_has_milestones "$stage"; then',
            self.wrapper,
        )
        helper_idx = self.wrapper.index("stage_has_milestones() {")
        guard_idx = self.wrapper.index('if stage_has_milestones "$stage"; then')
        self.assertLess(helper_idx, guard_idx)
        self.assertIn('build_prompt "$stage" "$current_milestone"', self.wrapper)
        self.assertIn('milestone "$current_milestone"', self.wrapper)

    def test_advance_stage_milestone_uses_helper(self) -> None:
        advance_block_start = self.wrapper.index("advance_stage_milestone() {")
        advance_block_end = self.wrapper.index("\n}\n", advance_block_start)
        advance_body = self.wrapper[advance_block_start:advance_block_end]
        self.assertIn(
            'stage_has_milestones "$stage" || return 0',
            advance_body,
        )


if __name__ == "__main__":
    unittest.main()
