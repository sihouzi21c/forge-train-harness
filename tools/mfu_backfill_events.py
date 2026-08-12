"""Re-emit ``mfu_record`` loop_events locally for remote-pulled MFU entries.

In ``ssh`` / ``devspace`` mode the dev agent's ``harness run`` executes on
the remote host, so :mod:`tools.mfu_record` writes ``mfu_history.jsonl`` and
emits its ``mfu_record`` loop_event *on the remote*. The local dashboard SSE
stream reads the *local* wrapper ``stdout.log``, so the remote-emitted event
is lost — the wrapper's background puller only rsyncs the history FILE back,
not the event. The badge then never live-updates; only the one-shot REST
hydration (``/api/artifacts/mfu``) ever reflects MFU.

This helper closes the gap: after each pull, it emits a LOCAL ``mfu_record``
event for every history entry not yet emitted locally, shelling out to the
canonical writer (``tools/loop_wrapper_event.py``, the SSOT for the event
shape + ``stdout.log`` location). A cursor file under the loop dir makes the
30 s poll idempotent — only the appended delta is emitted each round.

The emitted payload mirrors :func:`tools.mfu_record._emit_event` exactly so
the frontend ``_handleMfuRecordEvent`` reads the same fields whether the run
was local (event emitted in-process) or remote (event re-emitted here).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

__all__ = ["CURSOR_FILENAME", "EVENT_PAYLOAD_KEYS", "HISTORY_FILENAME", "backfill"]

HISTORY_FILENAME = "mfu_history.jsonl"
CURSOR_FILENAME = ".mfu_events_cursor"
# The badge fields the frontend reads — kept identical to the payload
# tools.mfu_record._emit_event ships so local + remote runs are uniform.
EVENT_PAYLOAD_KEYS = (
    "milestone",
    "suite",
    "avg_mfu",
    "mfu_target",
    "mfu_pass",
    "precision_pass",
)


def _read_history(history_path: Path) -> list[dict]:
    """Parse the append-only history, skipping malformed lines (best-effort
    — a half-written final line from an in-flight rsync must not crash the
    poll)."""
    if not history_path.is_file():
        return []
    entries: list[dict] = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            entries.append(obj)
    return entries


def _read_cursor(cursor_path: Path) -> int:
    try:
        return int(cursor_path.read_text(encoding="utf-8").strip() or "0")
    except (OSError, ValueError):
        return 0


def backfill(*, loop_dir: Path, loop_id: str, writer: Path, runner=subprocess.run) -> int:
    """Emit a local ``mfu_record`` event per not-yet-emitted history entry.

    Returns the number of events emitted. Advances the cursor by exactly
    that count, so a mid-batch emit failure leaves the remainder for the
    next poll instead of dropping it.
    """
    history_path = loop_dir / HISTORY_FILENAME
    cursor_path = loop_dir / CURSOR_FILENAME
    entries = _read_history(history_path)
    cursor = _read_cursor(cursor_path)
    pending = entries[cursor:]

    emitted = 0
    for entry in pending:
        payload = {key: entry.get(key) for key in EVENT_PAYLOAD_KEYS}
        try:
            runner(
                [
                    sys.executable,
                    str(writer),
                    "--loop-id",
                    loop_id,
                    "--subtype",
                    "mfu_record",
                    "--payload",
                    json.dumps(payload),
                ],
                check=False,
                timeout=5,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:  # best-effort; retry the remainder next poll
            print(f"mfu_backfill_events: emit failed: {exc}", file=sys.stderr)
            break
        emitted += 1

    if emitted:
        cursor_path.write_text(str(cursor + emitted), encoding="utf-8")
    return emitted


def _resolve_writer() -> Path | None:
    """The canonical event writer lives in the source tree (not the
    workspace copy) — same resolution tools.mfu_record._emit_event uses."""
    source_root = os.environ.get("FORGE_SOURCE_ROOT")
    if not source_root:
        return None
    writer = Path(source_root) / "harness" / "tools" / "loop_wrapper_event.py"
    return writer if writer.is_file() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mfu_backfill_events", description=__doc__)
    parser.add_argument("--loop-dir", required=True, help="Per-loop dir holding mfu_history.jsonl")
    parser.add_argument("--loop-id", required=True)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    writer = _resolve_writer()
    if writer is None:
        # No source-tree writer (e.g. a bare checkout) — the REST hydration
        # path still surfaces MFU; nothing to do here.
        return 0
    backfill(loop_dir=Path(args.loop_dir), loop_id=args.loop_id, writer=writer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
