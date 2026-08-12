#!/usr/bin/env bash
# Single source of truth for the dev-round retry decision used by
# agent-loop.sh's per-round attempt loop.
#
# Policy: retry by DEFAULT on any non-zero exit. Only refuse to retry
# for exit codes that indicate either (a) explicit user intent to stop
# (SIGINT / SIGTERM / SIGKILL) or (b) a configuration bug surfaced by
# our own spawn helper (EX_USAGE / EX_CONFIG from
# harness/tools/spawn_managed_agent.py).
#
# Earlier versions of agent-loop.sh allow-listed transient patterns
# inside the agent's stdout.log (grep for "AI Model Not Found",
# "rate limit", etc.). That approach was too narrow: any new failure
# string Cursor CLI introduced collapsed the loop on the first
# attempt, even though `agent_round_tries = 8` was configured.
# A real run lost 25 minutes of dev work when the CLI raised
# `AI Model Not Found Model name is not valid: "claude-opus-4-7"`
# from a model-resolution path the allow-list happened to cover but
# the wrapper still exited without retrying, because the allow-list
# check is fragile to silent codepath changes inside the CLI binary.
# Inverting the policy makes the wrapper robust to that whole class
# of "the CLI hiccupped" errors.
#
# Returns 0 (retry) or 1 (give up) so the caller can drive it as the
# rhs of an `&&` test exactly like the previous helper.

set -uo pipefail

should_retry_after_agent_failure() {
  local rc="$1"
  case "$rc" in
    0)
      # Success isn't a retry candidate. If a caller asks anyway, say
      # no so we don't accidentally re-spawn a healthy agent.
      return 1
      ;;
    130|143|137)
      # 130 = 128 + SIGINT (Ctrl-C from terminal)
      # 143 = 128 + SIGTERM (web dashboard stop_loop endpoint, kill -15)
      # 137 = 128 + SIGKILL (oom-killer, manual kill -9)
      # All three mean "stop now", not "this was a flake". (The
      # post-result silence watchdog that briefly used rc=143 + a
      # ``watchdog_kill`` event to mark a retry-eligible 143 was
      # removed alongside the watchdog itself — see web/agents/spawn.py
      # header note.)
      return 1
      ;;
    64|78)
      # spawn_managed_agent.py exits with these codes when the CLI
      # invocation itself is malformed (EX_USAGE rc=64) or a hard
      # config error - missing API key, --resume target that no
      # longer exists - surfaced as EX_CONFIG (rc=78). Retrying just
      # burns more attempts on the same bad config.
      #
      # NB: spawn_managed_agent used to also map ``RuntimeError`` from
      # ``spawn.spawn_session`` (cursor-cli dying before system.init)
      # into rc=78, which made transient CLI startup hiccups
      # (model-registry blips, network jitter during the init
      # handshake) un-retriable. That codepath now exits with rc=75
      # (EX_TEMPFAIL) and falls through to the default ``*`` branch
      # below so the wrapper retries by default.
      return 1
      ;;
    124)
      # 124 = EX_TIMEOUT from spawn_managed_agent.py: the round blew its
      # wall-clock cap (ROUND_TIMEOUT_S) and the agent's process group was
      # SIGTERM->SIGKILL'd. agent-loop.sh rolls the round back to its
      # pre-spawn baseline and starts a fresh round; retrying in place
      # would just re-burn the same budget on the same wedged work.
      return 1
      ;;
    *)
      # Everything else - rc=1 from cursor-cli's "AI Model Not Found",
      # rc=75 EX_TEMPFAIL from a pre-init CLI death, arbitrary
      # non-zero codes from a crashed subprocess - is treated as a
      # transient hiccup. The outer loop bounds total attempts via
      # ``agent_round_tries``, so retrying here is safe.
      return 0
      ;;
  esac
}

# Allow direct invocation as `bash agent_loop_retry.sh <rc>` so the
# unit test can drive the matrix without sourcing bash functions.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  should_retry_after_agent_failure "${1:-0}"
fi
