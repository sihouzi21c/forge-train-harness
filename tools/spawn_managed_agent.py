#!/usr/bin/env python3
"""Synchronous CLI wrapper around :mod:`web.agents.spawn`.

Used by ``agent-loop.sh`` (dev round / review / stage2 subagent) so every
backend CLI subprocess spawned by this project lands as a Session row
under ``$FORGE_AGENTS_DIR`` (default ``.artifacts/web-agents/``), with a
real stream-json stdout.log the frontend chat renderer can display.

Lifecycle:

1. Spawn the requested backend CLI via :func:`spawn.spawn_session`,
   blocking until the ``system.init`` event lands (or the subprocess
   dies pre-init).
2. Print ``AGENT_ID=<id>`` on our own stdout so the calling shell can
   capture the assigned agent id and emit a ``spawn_child`` loop_event.
3. Await the subprocess to completion, pumping stdout/stderr into the
   agent's per-id directory.
4. Exit with the same exit code the backend CLI exited with, so the
   shell's transient-failure detection and retry logic still work
   verbatim.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import signal
import sys
from datetime import UTC
from pathlib import Path


def _bootstrap_path() -> None:
    """Ensure ``web.agents`` is importable when invoked from any cwd."""
    here = Path(__file__).resolve()
    # harness/tools/spawn_managed_agent.py -> repo root is parents[2]
    repo_root = here.parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_bootstrap_path()


def _apply_agents_dir_override() -> None:
    """Honor ``FORGE_AGENTS_DIR`` env override before any spawn writes.

    The override is handled by ``web.paths.AGENTS_DIR`` which reads
    ``FORGE_AGENTS_DIR`` at import time. This function is kept as a
    no-op entry point so callers do not need structural changes.
    """


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="spawn_managed_agent",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--backend", required=True, choices=["cursor-cli", "claude-code", "codex"])
    p.add_argument("--model", required=True)
    p.add_argument("--workspace", required=True)
    p.add_argument(
        "--prompt-file",
        required=True,
        help="Path to a file containing the prompt; passed as the CLI's positional prompt argument.",
    )
    p.add_argument("--resume", default=None, help="Existing agent_id to resume.")
    p.add_argument(
        "--parent-agent", default=None, help="Parent agent_id (loop wrapper or dev round)."
    )
    p.add_argument("--loop-id", default=None, help="Loop id this agent belongs to.")
    p.add_argument(
        "--kind",
        default="chat",
        choices=["chat", "loop_dev_round", "loop_review", "loop_subagent"],
    )
    # Forest layout: when all three are given for a loop child, the spawned
    # session nests under the wrapper at agents/<stage>/r<RRR>/ with an
    # ordered composite id. Absent (plain chat / legacy callers) → flat
    # web-<uuid>. See docs/forest-agent-log-layout.md.
    p.add_argument("--stage", default=None, help="Loop stage (e.g. stage1) for forest nesting.")
    p.add_argument(
        "--round",
        type=int,
        default=None,
        dest="round_no",
        help="Round number (per stage) for forest nesting + lexical ordering.",
    )
    p.add_argument(
        "--seq",
        type=int,
        default=None,
        help="Per-round spawn sequence (dev=1, review=2) for ordering.",
    )
    p.add_argument(
        "--max-mode",
        action="store_true",
        help=(
            "Enable cursor-cli max mode: writes maxMode=true to "
            "~/.cursor/cli-config.json for the duration of the CLI init "
            "handshake. No effect on claude-code/codex (use --effort there)."
        ),
    )
    p.add_argument(
        "--effort",
        default="",
        help=(
            "claude-code/codex reasoning-effort level "
            "(low / medium / high / xhigh / max). Passed to the backend "
            "CLI when non-empty; codex clamps xhigh/max to high. "
            "Ignored by cursor-cli (use --max-mode there)."
        ),
    )
    p.add_argument(
        "--thinking-display",
        default="",
        choices=["", "summarized", "omitted"],
        help=(
            "claude-code thinking-stream knob, forwarded as "
            "`--thinking-display=<value>`. Opus 4.8 emits no chain-of-"
            "thought into stream-json logs unless this is `summarized`; "
            "opus-pinning agent profiles set [agent].thinking_display = "
            '"summarized". Empty omits the flag (CLI default). Ignored '
            "by cursor-cli/codex."
        ),
    )
    p.add_argument(
        "--api-key-env",
        default="",
        help=(
            "Env var name to read the backend API key from (e.g. "
            "CURSOR_API_KEY). Empty means rely on web.auth's persisted key."
        ),
    )
    p.add_argument(
        "--timeout-s",
        type=int,
        default=0,
        help=(
            "Wall-clock cap in seconds for this agent's turn. When >0 and "
            "the agent is still running at expiry, its process group is "
            "SIGTERM -> 30s grace -> SIGKILL'd and the helper exits 124 "
            "(EX_TIMEOUT) so agent-loop.sh can roll the round back. 0 "
            "(default) disables the cap."
        ),
    )
    return p.parse_args(argv)


def _resolve_api_key(backend: str, api_key_env: str) -> str | None:
    if api_key_env:
        val = os.environ.get(api_key_env, "").strip()
        if val:
            return val
    # Fall back to whatever web.auth's persisted key store provides
    # (same source of truth the web server uses for the Agents tab).
    from web import auth as cursor_auth

    return cursor_auth.api_key(backend)


def _emit_spawn_error(loop_id: str | None, message: str) -> None:
    """Surface a spawn failure to both stderr and the wrapper's log dir.

    The web router runs ``agent-loop.sh`` with ``stderr=DEVNULL``, so
    anything we ``print(..., file=sys.stderr)`` from inside this helper
    used to vanish - a post-mortem on a "rc=78 / rc=75, no spawn_child
    event" loop_exit had nothing to grep. Mirror the message into
    ``$FORGE_AGENTS_DIR/loop-<loop_id>/spawn_errors.log`` (best-effort,
    never raises) so the diagnostic survives the wrapper's exit.
    """
    line = f"spawn_managed_agent: spawn failed: {message}"
    print(line, file=sys.stderr)
    loop_id = (loop_id or "").strip()
    if not loop_id:
        return
    from web.paths import AGENTS_DIR, wrapper_agent_id

    target_dir = AGENTS_DIR / wrapper_agent_id(loop_id)
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        with (target_dir / "spawn_errors.log").open("a", encoding="utf-8") as fh:
            from datetime import datetime

            fh.write(f"{datetime.now(UTC).isoformat()} {line}\n")
    except OSError:
        pass


async def _run(args: argparse.Namespace) -> int:
    from web.agents import spawn

    prompt_path = Path(args.prompt_file)
    if not prompt_path.is_file():
        print(f"spawn_managed_agent: --prompt-file does not exist: {prompt_path}", file=sys.stderr)
        return 64  # EX_USAGE
    prompt = prompt_path.read_text(encoding="utf-8")

    api_key = _resolve_api_key(args.backend, args.api_key_env)

    try:
        result = await spawn.spawn_session(
            backend_name=args.backend,
            model=args.model,
            prompt=prompt,
            workspace=args.workspace,
            api_key=api_key,
            max_mode=args.max_mode,
            effort=args.effort,
            thinking_display=args.thinking_display,
            resume_agent_id=args.resume,
            parent_agent_id=args.parent_agent,
            loop_id=args.loop_id,
            kind=args.kind,
            stage=args.stage,
            round_no=args.round_no,
            seq=args.seq,
        )
    except FileNotFoundError as exc:
        # A --resume target that no longer exists is a caller bug, not a
        # transient backend hiccup. Stay on EX_CONFIG so agent-loop.sh's
        # retry helper keeps refusing to burn attempts on it.
        _emit_spawn_error(args.loop_id, f"FileNotFoundError: {exc}")
        return 78  # EX_CONFIG
    except RuntimeError as exc:
        # ``spawn.spawn_session`` raises RuntimeError when the backend
        # CLI dies before ``system.init`` (cursor-cli model-resolution
        # hiccups, network blips during the init handshake, sandbox
        # spawn glitches, etc.). These are transient by nature - the
        # exact same argv often succeeds on the next attempt - so they
        # must NOT be lumped into EX_CONFIG, which the retry helper
        # treats as a permanent config bug. Use EX_TEMPFAIL (75) so
        # the helper's "retry by default for unknown rc" branch picks
        # it up and the round can heal itself.
        _emit_spawn_error(args.loop_id, f"RuntimeError: {exc}")
        return 75  # EX_TEMPFAIL

    # Forward stop signals to the backend CLI's process group. The CLI was
    # spawned with ``start_new_session=True`` (see spawn.spawn_session), so
    # it sits in its OWN session/group — the /stop endpoint's SIGTERM to the
    # agent-loop.sh process group reaches THIS helper but not the CLI. Relay
    # it so ``proc.wait()`` returns and ``monitor_exit`` below runs to
    # completion, writing the child's terminal state (interrupted) to
    # session.json. Without this relay the helper dies mid-asyncio, the CLI
    # leaks as an orphan, and the session stays "running"/responding forever.
    loop = asyncio.get_running_loop()
    for _sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            _sig,
            spawn.kill_process_group,
            result.proc,
            _sig,
        )

    # Print AGENT_ID after init lands so the calling shell can read it
    # back even when the agent runs for hours. ``flush=True`` is critical
    # because the shell may be parsing our stdout line-by-line.
    print(f"AGENT_ID={result.session.agent_id}", flush=True)

    # Write spawn_child to the wrapper's stdout.log NOW, while the child
    # is still running. The old flow deferred this to _spawn_child_agent
    # in agent-loop.sh which only ran after monitor_exit returned — i.e.
    # after the child had already exited. That made the /active endpoint
    # and the Follow Live SSE scheduler unable to see the currently
    # running child, since its spawn_child event didn't exist yet.
    if args.loop_id:
        spawn_payload: dict[str, object] = {
            "agent_id": result.session.agent_id,
            "kind": args.kind,
        }
        # Surface round/seq on the spawn_child card so the wrapper log
        # carries the same ordering the directory does (best-effort enrich).
        if args.round_no is not None:
            spawn_payload["round"] = args.round_no
        if args.seq is not None:
            spawn_payload["seq"] = args.seq
        spawn.append_loop_event(args.loop_id, "spawn_child", spawn_payload)

    # No watchdog: monitor_exit blocks on proc.wait() and returns the rc.
    # The historical post-result silence watchdog was removed because
    # ``result`` is a turn boundary, not a session boundary, and a dev
    # agent using ScheduleWakeup legitimately sleeps tens of minutes
    # between turns while polling a remote GPU gate — the watchdog
    # could not distinguish that from a stuck CLI and false-killed
    # legitimate dev rounds (see web/agents/spawn.py header note).
    # Lease leaks now have to be caught by per-suite ``timeout_s`` in
    # dispatcher.py (which already bounds every gate's wall-clock) and
    # by manual review of long-running rounds.
    #
    # ``--timeout-s`` adds an OPT-IN per-round wall-clock cap on top of
    # this (default 5h, wired from agent-loop.sh's ROUND_TIMEOUT_S). It
    # is NOT the old silence watchdog: it bounds the agent's total turn
    # time, not inter-event silence, so a legitimately slow round that
    # keeps making progress is still killed once it blows the budget —
    # which is exactly what the round cap is for. agent-loop.sh rolls the
    # workspace back to the round baseline when it sees rc=124.
    monitor_task = asyncio.ensure_future(
        spawn.monitor_exit(
            result.session.agent_id,
            result.proc,
            result.pump_tasks,
        )
    )
    if args.timeout_s and args.timeout_s > 0:
        try:
            return await asyncio.wait_for(asyncio.shield(monitor_task), timeout=args.timeout_s)
        except TimeoutError:
            _emit_spawn_error(
                args.loop_id,
                f"round wall-clock cap of {args.timeout_s}s exceeded; killing "
                "agent process group (SIGTERM -> 30s grace -> SIGKILL)",
            )
            # shield keeps monitor_task alive across the cancellation so it
            # still reconciles session.json once proc.wait() returns.
            spawn.kill_process_group(result.proc, signal.SIGTERM)
            try:
                await asyncio.wait_for(asyncio.shield(monitor_task), timeout=30)
            except TimeoutError:
                spawn.kill_process_group(result.proc, signal.SIGKILL)
                with contextlib.suppress(asyncio.TimeoutError, TimeoutError):
                    await asyncio.wait_for(asyncio.shield(monitor_task), timeout=5)
            return 124  # EX_TIMEOUT — mirrors GNU coreutils `timeout`
    return await monitor_task


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    _apply_agents_dir_override()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
