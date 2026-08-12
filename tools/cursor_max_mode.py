"""Lock-guarded Cursor CLI max-mode toggling.

The Cursor CLI reads ``maxMode`` from ``~/.cursor/cli-config.json`` at
startup and snapshots it before model selection.  There is no CLI flag to
set it, and ``--model`` causes a write-back that resets it to ``false``.

This module provides:

1. A context-manager (``cursor_max_mode_guard``) for Python callers.
2. A ``wrap`` CLI subcommand for bash callers — holds the file lock from
   config write until the wrapped process emits its init event.

Adapted from the autoskill ``cursor_max_mode`` module.
"""

from __future__ import annotations

__all__ = ["cursor_max_mode_guard"]

import contextlib
import fcntl
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

_LOCK_PATH = Path(tempfile.gettempdir()) / "harness-cursor-maxmode.lock"


def _config_path() -> Path:
    return Path.home() / ".cursor" / "cli-config.json"


def _atomic_set_max_mode(value: bool) -> None:
    cfg_path = _config_path()
    # When running as root (or any user that has never launched the Cursor
    # agent CLI), ``~/.cursor/cli-config.json`` and possibly ``~/.cursor/``
    # itself do not exist yet. Treat that as an empty baseline rather than
    # an OSError — otherwise ``wrap_command``'s ``except OSError`` swallows
    # the FileNotFoundError and mislabels it as "Failed to start command".
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        cfg = {}
    cfg["maxMode"] = value
    cfg.setdefault("model", {})["maxMode"] = value
    fd, tmp = tempfile.mkstemp(dir=str(cfg_path.parent), suffix=".tmp")
    try:
        os.write(fd, json.dumps(cfg, indent=2, ensure_ascii=False).encode())
        os.close(fd)
        os.replace(tmp, str(cfg_path))
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


@contextlib.contextmanager
def cursor_max_mode_guard(max_mode: bool) -> Iterator[None]:
    """Hold a file lock while setting maxMode, yield, then release."""
    lock_fd = os.open(str(_LOCK_PATH), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        _atomic_set_max_mode(max_mode)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(lock_fd)


def wrap_command(max_mode: bool, cmd: list[str]) -> int:
    """Run *cmd*, holding the maxMode lock until the agent emits its init event."""
    child: subprocess.Popen[str] | None = None

    def _forward_signal(signum: int, _frame: object) -> None:
        if child is not None:
            try:
                os.killpg(os.getpgid(child.pid), signum)
            except (ProcessLookupError, PermissionError, OSError):
                with contextlib.suppress(OSError):
                    child.terminate()

    prev_sigterm = signal.signal(signal.SIGTERM, _forward_signal)
    prev_sigint = signal.signal(signal.SIGINT, _forward_signal)

    guard = cursor_max_mode_guard(max_mode)
    try:
        guard.__enter__()
        child = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
    except OSError as exc:
        guard.__exit__(None, None, None)
        print(f"Failed to start command: {exc}", file=sys.stderr)
        return 1

    lock_released = False
    assert child.stdout is not None
    for line in child.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if not lock_released and '"type"' in line and '"system"' in line:
            guard.__exit__(None, None, None)
            lock_released = True

    if not lock_released:
        guard.__exit__(None, None, None)

    rc = child.wait()
    signal.signal(signal.SIGTERM, prev_sigterm)
    signal.signal(signal.SIGINT, prev_sigint)
    return rc


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(
            "Usage: cursor_max_mode.py {set|reset|wrap -- cmd...}",
            file=sys.stderr,
        )
        return 2

    if args[0] == "set":
        _atomic_set_max_mode(True)
        return 0
    if args[0] == "reset":
        _atomic_set_max_mode(False)
        return 0
    if args[0] == "wrap":
        if "--" not in args:
            print("Usage: cursor_max_mode.py wrap -- cmd args...", file=sys.stderr)
            return 2
        sep = args.index("--")
        cmd = args[sep + 1 :]
        if not cmd:
            print("No command specified after '--'", file=sys.stderr)
            return 2
        return wrap_command(max_mode=True, cmd=cmd)

    print(f"Unknown subcommand: {args[0]}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
