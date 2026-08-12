"""Unit + e2e coverage for the dev-round retry decision in agent-loop.sh.

Background: a real run died after a single 25-minute dev round because
Cursor CLI emitted ``AI Model Not Found Model name is not valid:
"claude-opus-4-7"`` and the wrapper's retry path used an allow-list of
transient patterns. Anything outside the list collapsed the loop on the
first failure, even though ``agent_round_tries = 8`` was configured.

The fix inverts the policy: retry by DEFAULT on any non-zero exit;
give up ONLY for codes that mean explicit user intent (SIGINT/SIGTERM/
SIGKILL) or a configuration bug in our own spawn helper
(EX_USAGE / EX_CONFIG from ``spawn_managed_agent.py``). This locks
that contract in code so future patches cannot regress it.

Two layers of coverage:

1. ``TestShouldRetryAfterAgentFailure`` drives the standalone helper
   ``harness/tools/agent_loop_retry.sh`` over the full rc matrix so
   the decision table is reviewable in one place.
2. ``TestRetryEventEmission`` checks that ``agent-loop.sh`` sources
   the helper rather than reintroducing an inline allow-list, and
   that the wrapper emits a ``round_retry`` event when retrying so
   diagnostics survive the next post-mortem.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
HARNESS_DIR = REPO_ROOT / "harness"
RETRY_HELPER = HARNESS_DIR / "tools" / "agent_loop_retry.sh"
AGENT_LOOP_SH = HARNESS_DIR / "agent-loop.sh"
SPAWN_HELPER = HARNESS_DIR / "tools" / "spawn_managed_agent.py"


def _retry_rc(rc: int) -> int:
    """Run the helper as a script and return its exit code (0 = retry)."""
    result = subprocess.run(
        ["bash", str(RETRY_HELPER), str(rc)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    return result.returncode


class TestShouldRetryAfterAgentFailure(unittest.TestCase):
    """Pin the rc → retry-decision matrix.

    Exit-code semantics on POSIX:
      * ``0``      → success; caller must not invoke this helper.
      * ``130``    → SIGINT (Ctrl-C).
      * ``137``    → SIGKILL.
      * ``143``    → SIGTERM (web dashboard ``stop_loop`` / kill -15).
      * ``64``     → ``EX_USAGE`` from spawn_managed_agent.py (bad CLI args).
      * ``75``     → ``EX_TEMPFAIL`` from spawn_managed_agent.py
                     (cursor-cli died before ``system.init`` — a
                     transient that MUST be retried).
      * ``78``     → ``EX_CONFIG`` from spawn_managed_agent.py
                     (missing API key, ``--resume`` target gone).
      * anything else → "the CLI / model registry hiccupped"; retry.
    """

    def test_helper_script_exists_and_is_executable(self) -> None:
        self.assertTrue(
            RETRY_HELPER.is_file(),
            f"expected retry helper at {RETRY_HELPER}",
        )

    def test_success_is_not_a_retry_candidate(self) -> None:
        # Caller bug if we ever ask: success shouldn't be retried.
        self.assertEqual(_retry_rc(0), 1)

    def test_sigint_sigterm_sigkill_do_not_retry(self) -> None:
        for rc in (130, 143, 137):
            with self.subTest(rc=rc):
                self.assertEqual(
                    _retry_rc(rc),
                    1,
                    f"rc={rc} signals user-intent stop; must not retry",
                )

    def test_spawn_helper_config_errors_do_not_retry(self) -> None:
        for rc in (64, 78):
            with self.subTest(rc=rc):
                self.assertEqual(
                    _retry_rc(rc),
                    1,
                    f"rc={rc} (EX_USAGE/EX_CONFIG) is a config bug; must not retry",
                )

    def test_round_timeout_does_not_retry(self) -> None:
        # 124 (EX_TIMEOUT) means the round blew its wall-clock cap and the
        # agent's process group was killed. agent-loop.sh rolls the round
        # back; retrying would just re-burn the same budget. Defence in
        # depth: even though the wrapper intercepts 124 before consulting
        # this helper, the policy table must agree that 124 = give up.
        self.assertEqual(
            _retry_rc(124),
            1,
            "rc=124 (EX_TIMEOUT) is a wall-clock cap hit; must not retry",
        )

    def test_generic_nonzero_exits_retry_by_default(self) -> None:
        # The "AI Model Not Found" case manifests as rc=1; the original
        # allow-list missed this string and the loop died. Cover both
        # rc=1 and a handful of other arbitrary non-zero codes to lock
        # in the "retry by default" inversion.
        for rc in (1, 2, 3, 99, 125, 126, 127, 200, 255):
            with self.subTest(rc=rc):
                self.assertEqual(
                    _retry_rc(rc),
                    0,
                    f"rc={rc} should retry (allow-list inversion contract)",
                )

    def test_ex_tempfail_retries(self) -> None:
        # Pre-init cursor-cli deaths (model-registry blip, network
        # jitter during the init handshake) used to be mapped to rc=78
        # alongside hard config errors, so the helper refused to retry
        # them. spawn_managed_agent.py now emits rc=75 (EX_TEMPFAIL)
        # for this codepath; the helper MUST treat it as a transient
        # and let the round retry. Regressing this fuses the policy
        # back into the "any spawn-side failure is fatal" bug.
        self.assertEqual(
            _retry_rc(75),
            0,
            "rc=75 (EX_TEMPFAIL) is a transient pre-init CLI death; must retry",
        )


class TestRetryEventEmission(unittest.TestCase):
    """Guard the wrapper-side wiring around the helper.

    The decision helper is useless if agent-loop.sh forgets to call it
    or routes the retry path past its info-event emission. These tests
    grep the wrapper source so future refactors can't silently revert.
    """

    def setUp(self) -> None:
        self.wrapper_src = AGENT_LOOP_SH.read_text(encoding="utf-8")

    def test_wrapper_sources_the_retry_helper(self) -> None:
        # The wrapper must delegate the decision rather than reintroduce
        # an inline allow-list (the very bug this whole change repairs).
        self.assertIn("tools/agent_loop_retry.sh", self.wrapper_src)
        self.assertIn("should_retry_after_agent_failure", self.wrapper_src)

    def test_wrapper_no_longer_has_inline_transient_allowlist(self) -> None:
        # The legacy function name and its hardcoded grep pattern must
        # both be gone; the helper is the single source of truth.
        self.assertNotIn(
            "is_transient_cursor_agent_failure",
            self.wrapper_src,
            "legacy allow-list function must be removed; helper owns the policy",
        )
        self.assertNotIn(
            "AI Model Not Found|Model name is not valid",
            self.wrapper_src,
            "hardcoded transient pattern must move out of agent-loop.sh",
        )

    def test_wrapper_emits_retry_decision_info_event(self) -> None:
        # The original post-mortem was blocked because the wrapper went
        # straight from spawn_child to loop_exit with no breadcrumb. The
        # info event makes the decision visible in the wrapper chat.
        self.assertIn("retry_decision", self.wrapper_src)

    def test_spawn_helper_splits_runtime_error_into_ex_tempfail(self) -> None:
        # Regression: spawn_managed_agent used to bucket both
        # RuntimeError (CLI dies before init) and FileNotFoundError
        # (--resume target gone) into rc=78. That conflated a
        # transient backend hiccup with a permanent config bug and
        # made the retry policy give up on the wrong cases. The fix
        # is to map RuntimeError onto rc=75 (EX_TEMPFAIL) while
        # FileNotFoundError stays on rc=78 (EX_CONFIG).
        src = SPAWN_HELPER.read_text(encoding="utf-8")
        self.assertIn(
            "return 75",
            src,
            "spawn_managed_agent must emit rc=75 (EX_TEMPFAIL) for transient pre-init CLI failures",
        )
        self.assertNotIn(
            "except (RuntimeError, FileNotFoundError)",
            src,
            "RuntimeError and FileNotFoundError must be caught in "
            "separate except clauses so they can return different rcs",
        )
        # Sanity: the EX_CONFIG path must still exist for the
        # FileNotFoundError (resume-target-gone) case.
        self.assertIn(
            "return 78",
            src,
            "spawn_managed_agent must still emit rc=78 (EX_CONFIG) "
            "for hard config errors (FileNotFoundError on --resume)",
        )

    def test_spawn_helper_mirrors_stderr_to_loop_dir(self) -> None:
        # When agent-loop.sh runs under the web router, its stderr is
        # /dev/null. Without an explicit mirror, the actual
        # "spawn failed: ..." message that drives every rc=75 / rc=78
        # would vanish, leaving post-mortems with only a bare exit
        # code. The helper must mirror its diagnostic into the loop
        # wrapper's agent dir so the message survives the wrapper's
        # exit.
        src = SPAWN_HELPER.read_text(encoding="utf-8")
        self.assertIn("spawn_errors.log", src)
        self.assertIn("_emit_spawn_error", src)

    def test_spawn_helper_does_not_reenable_errexit_before_returning_rc(self) -> None:
        # Regression for loops that died immediately after spawn_child:
        # _spawn_child_agent used to run `set -e` after capturing the
        # child rc, then `return $rc` with rc=1 triggered the EXIT trap
        # before the caller could execute retry_decision / round_retry.
        match = re.search(
            r"_spawn_child_agent\(\) \{(?P<body>.*?)\n\}",
            self.wrapper_src,
            flags=re.S,
        )
        self.assertIsNotNone(match, "_spawn_child_agent function not found")
        body = match.group("body") if match else ""
        after_capture = body.split("local rc=$?", 1)[1]
        before_return = after_capture.rsplit("return $rc", 1)[0]
        self.assertIsNone(
            re.search(r"^\s*set\s+-e\b", before_return, flags=re.M),
            "_spawn_child_agent must leave errexit disabled until caller captures rc",
        )

    def test_loop_event_is_internally_infallible(self) -> None:
        match = re.search(
            r"_loop_event\(\) \{(?P<body>.*?)\n\}",
            self.wrapper_src,
            flags=re.S,
        )
        self.assertIsNotNone(match, "_loop_event function not found")
        body = match.group("body")
        self.assertIn(
            "|| true",
            body,
            "_loop_event must swallow errors internally (|| true) so "
            "it can never kill the script regardless of set -e state",
        )

    def test_no_loop_event_payload_command_substitutions(self) -> None:
        self.assertNotIn(
            "_loop_event_payload",
            self.wrapper_src,
            "_loop_event_payload has been removed; _loop_event accepts "
            "key-value pairs directly — no $() command substitution "
            "needed at call sites",
        )

    def test_retry_block_keeps_errexit_disabled_through_retry_verdict(self) -> None:
        match = re.search(
            r"while \(\( attempt <= CURSOR_AGENT_ROUND_TRIES \)\); do"
            r"(?P<body>.*?)"
            r"\n\s*done",
            self.wrapper_src,
            flags=re.S,
        )
        self.assertIsNotNone(match, "retry while loop not found")
        body = match.group("body")
        block = body.split("agent_ec=$?", 1)[1]
        block = block.split("retry_verdict=$?", 1)[0]
        self.assertIsNone(
            re.search(r"^\s*set\s+-e\b", block, flags=re.M),
            "set -e must not appear between agent_ec=$? and "
            "retry_verdict=$?; both exit-code captures need set +e",
        )

    def test_classify_agent_failure_uses_no_multi_stage_pipeline(self) -> None:
        match = re.search(
            r"_classify_agent_failure\(\) \{(?P<body>.*?)\n\}",
            self.wrapper_src,
            flags=re.S,
        )
        self.assertIsNotNone(match)
        body = match.group("body")
        self.assertNotRegex(
            body,
            r"grep.*\|.*tail.*\|.*head",
            "_classify_agent_failure must not use a grep|tail|head "
            "pipeline; tail -n1 reads from file-end without SIGPIPE risk",
        )


class TestReloadSessionFromDisk(unittest.TestCase):
    """Guard the external-process SSOT contract in _monitor_exit.

    External loops are NOT our child processes, so os.waitpid cannot
    retrieve their exit code.  agent-loop.sh's EXIT trap writes the
    authoritative session.json; _reload_session_from_disk must adopt
    those fields verbatim instead of fabricating state.
    """

    def _make_inst(self, loop_id: str) -> loop_mod.LoopInstance:
        return loop_mod.LoopInstance(loop_id=loop_id, mode="external")

    def setUp(self) -> None:
        import tempfile

        sys.path.insert(0, str(REPO_ROOT))
        global loop_mod
        from web.routers import loop as loop_mod

        self._tmp = tempfile.TemporaryDirectory()
        self._original = loop_mod.FORGE_TRAIN_DIR
        loop_mod.FORGE_TRAIN_DIR = Path(self._tmp.name)

    def tearDown(self) -> None:
        loop_mod.FORGE_TRAIN_DIR = self._original
        self._tmp.cleanup()

    def test_adopts_failed_state_from_disk(self) -> None:
        d = Path(self._tmp.name) / "t1"
        d.mkdir()
        (d / "session.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "exit_code": 1,
                    "ended_at": 12345.0,
                }
            )
        )
        inst = self._make_inst("t1")
        loop_mod._reload_session_from_disk(inst)
        self.assertEqual(inst.status, "failed")
        self.assertEqual(inst.exit_code, 1)
        self.assertEqual(inst.ended_at, 12345.0)

    def test_adopts_stopped_state_from_disk(self) -> None:
        d = Path(self._tmp.name) / "t2"
        d.mkdir()
        (d / "session.json").write_text(
            json.dumps(
                {
                    "status": "stopped",
                    "exit_code": -15,
                    "ended_at": 99.0,
                }
            )
        )
        inst = self._make_inst("t2")
        loop_mod._reload_session_from_disk(inst)
        self.assertEqual(inst.status, "stopped")
        self.assertEqual(inst.exit_code, -15)

    def test_adopts_completed_state_from_disk(self) -> None:
        d = Path(self._tmp.name) / "t3"
        d.mkdir()
        (d / "session.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "exit_code": 0,
                    "ended_at": 42.0,
                }
            )
        )
        inst = self._make_inst("t3")
        loop_mod._reload_session_from_disk(inst)
        self.assertEqual(inst.status, "completed")
        self.assertEqual(inst.exit_code, 0)

    def test_falls_back_to_failed_on_missing_file(self) -> None:
        inst = self._make_inst("nonexistent")
        loop_mod._reload_session_from_disk(inst)
        self.assertEqual(inst.status, "failed")
        self.assertIsNone(inst.exit_code)

    def test_falls_back_to_failed_on_bad_json(self) -> None:
        d = Path(self._tmp.name) / "t4"
        d.mkdir()
        (d / "session.json").write_text("not json")
        inst = self._make_inst("t4")
        loop_mod._reload_session_from_disk(inst)
        self.assertEqual(inst.status, "failed")

    def test_clamps_invalid_status_to_failed(self) -> None:
        d = Path(self._tmp.name) / "t5"
        d.mkdir()
        (d / "session.json").write_text(
            json.dumps(
                {
                    "status": "running",
                    "exit_code": 1,
                }
            )
        )
        inst = self._make_inst("t5")
        loop_mod._reload_session_from_disk(inst)
        self.assertEqual(inst.status, "failed")


if __name__ == "__main__":
    unittest.main()
