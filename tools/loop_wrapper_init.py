#!/usr/bin/env python3
"""Create the synthetic loop-wrapper Session entry for a running loop.

Invoked once by ``agent-loop.sh`` at startup. Writes
``$FORGE_AGENTS_DIR/loop-<loop_id>/session.json`` with
``backend=loop-wrapper`` and seeds ``stdout.log`` with a
``system.init``-shape line so the frontend chat renderer picks it up as
a real session.

After this runs, the shell appends one ``loop_event`` NDJSON record per
orchestration boundary (stage_start, round_start, spawn_child,
review_verdict, ...) via :mod:`harness.tools.loop_wrapper_event`.
"""

from __future__ import annotations

import argparse
import os
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
    p = argparse.ArgumentParser(prog="loop_wrapper_init", description=__doc__)
    p.add_argument("--loop-id", required=True)
    p.add_argument("--workspace", required=True)
    p.add_argument(
        "--backend", required=True, help="Loop-level backend label (cursor-cli/claude-code)."
    )
    p.add_argument("--model", required=True)
    p.add_argument("--stages", required=True, help="Space-separated stage list.")
    p.add_argument("--pid", type=int, default=None, help="agent-loop.sh PID; defaults to parent.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    _apply_agents_dir_override()
    from web.agents import spawn

    spawn.init_wrapper_session(
        loop_id=args.loop_id,
        workspace=args.workspace,
        backend_label=args.backend,
        model=args.model,
        stages=args.stages,
        pid=args.pid if args.pid is not None else os.getppid(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
