#!/usr/bin/env python3
"""Append one ``loop_event`` NDJSON record to a loop wrapper's stdout.log.

The single writer for all wrapper-side orchestration text. Replaces the
old ``tee -a "$LOG_DIR/summary.log"`` echo pattern so the frontend gets
typed events instead of raw bash output mixed with LLM NDJSON.

Subtype conventions (consumed by :mod:`web.agents.messages`):

* ``stage_start`` / ``stage_skip`` / ``stage_done`` — payload: ``stage``
* ``round_start`` / ``round_retry`` — payload: ``stage``, ``round``,
  optional ``attempt``
* ``spawn_child`` — payload: ``agent_id``, ``kind`` (``loop_dev_round``,
  ``loop_review``, ``loop_subagent``)
* ``review_verdict`` — payload: ``stage``, ``round``, ``verdict``
  (``PASS``/``FAIL``)
* ``stage_status`` — payload: ``stage``, ``status``
  (``in-progress``/``finished``)
* ``info`` — payload: ``text`` (fallback for anything not yet typed)
* ``loop_exit`` — payload: ``state``, ``exit_code``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap_path() -> None:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_bootstrap_path()


def _apply_agents_dir_override() -> None:
    """No-op: FORGE_AGENTS_DIR is handled by web.paths.AGENTS_DIR at import time."""


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="loop_wrapper_event", description=__doc__)
    p.add_argument("--loop-id", required=True)
    p.add_argument(
        "--subtype",
        required=True,
        help="One of stage_start/round_start/spawn_child/review_verdict/info/...",
    )
    p.add_argument(
        "--payload",
        default="",
        help="JSON object merged into the event record (caller's typed fields).",
    )
    p.add_argument(
        "kv_pairs",
        nargs="*",
        help="key value key value ... — positional pairs after '--' are "
        "merged into the payload as typed fields (integers auto-detected).",
    )
    return p.parse_args(argv)


def _kv_to_dict(pairs: list[str]) -> dict[str, object]:
    keys = pairs[0::2]
    vals = pairs[1::2]
    out: dict[str, object] = {}
    for k, v in zip(keys, vals, strict=False):
        try:
            out[k] = int(v)
        except ValueError:
            out[k] = v
    return out


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    _apply_agents_dir_override()
    payload: dict[str, object] = {}
    if args.payload:
        try:
            parsed = json.loads(args.payload)
        except json.JSONDecodeError as exc:
            print(
                f"loop_wrapper_event: invalid --payload JSON: {exc}: {args.payload!r}",
                file=sys.stderr,
            )
            return 64
        if not isinstance(parsed, dict):
            print(
                f"loop_wrapper_event: --payload must be a JSON object; got {type(parsed).__name__}",
                file=sys.stderr,
            )
            return 64
        payload = parsed
    if args.kv_pairs:
        payload.update(_kv_to_dict(args.kv_pairs))
    from web.agents import spawn

    spawn.append_loop_event(args.loop_id, args.subtype, payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
