"""Shell-exec a ref-script and capture its stdout / loss trajectory.

This module is the dispatcher's *bash-exec primitive* for the
ref-as-gate contract:

  * The L0 ref script is the baseline truth for every gate. Period.
  * No frozen JSON, no SHA anchor — running the ref script IS the
    gate's reference.

It owns:

  * Building the subprocess environment (``FORGE_GATE``,
    ``DUMP_DIR``, ``LOSS_DUMP_FILE``, caller-supplied ``extra_env``).
  * Streaming stdout to ``<dump_dir>/ref_stdout.log``.
  * Loading the optional ``gate_metadata.json`` the ref-script writes
    when ``PRINT_GATE_METADATA=1``.
  * Returning a structured :class:`RefRun`.

It does **not** own:

  * Hook wiring. The dispatcher's M1 caller bashes an agent-generated
    bridge (see ``evals/harness_hook/recipes/README.md``) that itself
    calls :func:`evals.harness_hook.install`. This runner is just a
    bash subprocess primitive; it has no framework knowledge.
  * Loss-trajectory parsing. See :mod:`harness.wire_format`.
"""

from __future__ import annotations

import dataclasses
import json
import os
import signal
import subprocess
from typing import TYPE_CHECKING

from harness.wire_format import (
    parse_loss_dump_file,
    parse_loss_dump_with_grad,
    parse_stdout_loss,
)

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "RefRun",
    "parse_loss_dump",
    "parse_loss_dump_with_grad",
    "parse_stdout_loss",
    "run_ref_script",
]


# Back-compat alias — historical callers know this helper as
# ``parse_loss_dump``. The single source of truth for the line grammar
# lives in :mod:`harness.wire_format`.
parse_loss_dump = parse_loss_dump_file


@dataclasses.dataclass
class RefRun:
    """Outcome of a ref-script invocation; consumed by dispatchers."""

    returncode: int
    stdout_path: Path
    dump_dir: Path
    loss_file: Path | None
    elapsed_s: float
    timed_out: bool
    # Launcher-emitted ``gate_metadata.json`` contents. Consumed ONLY by the
    # D5-exempt, product-less ``op-long`` suite; empty ``{}`` for every
    # Stage-1 gate (which reads its shape from the rendered product).
    metadata: dict[str, object]

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def _killpg(proc: subprocess.Popen) -> None:
    """SIGKILL the whole process group rooted at ``proc``.

    ``proc`` is launched with ``start_new_session=True`` so its PID is the
    group leader; ``killpg`` then reaps the bash launcher together with the
    torchrun rank-worker grandchildren. Best-effort: a group that has
    already exited (``ProcessLookupError``) is fine, and on the rare OSError
    we fall back to killing just the direct child.
    """
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        proc.kill()


def run_ref_script(
    repo_root: Path,
    *,
    gate_name: str,
    dump_dir: Path,
    ref_script_path: Path,
    timeout_s: int | None = None,
    extra_env: dict[str, str] | None = None,
    extra_ref_args: list[str] | tuple[str, ...] | None = None,
    capture_loss_trace: bool = True,
) -> RefRun:
    """``bash`` the given ref-script with gate env + caller env additions.

    Parameters
    ----------
    repo_root:
        Repo root used as ``cwd`` for the subprocess. Resolves any
        relative paths the ref-script references.
    gate_name:
        Forwarded to the ref-script as ``FORGE_GATE``. The L0
        script's ``apply_harness_gate_preset`` reads this to select
        training-shape overrides (one preset per gate).
    dump_dir:
        Gate-private artifact directory. Forwarded as ``DUMP_DIR`` and
        used as the parent for ``ref_stdout.log`` /
        ``ref_loss.txt`` / ``gate_metadata.json``.
    ref_script_path:
        Absolute path to the ``.sh`` (or other bash-executable) to
        run. May be the customer ref-script (M2-M6 trajectory path)
        or an agent-written M1 capture bridge.
    timeout_s:
        Optional subprocess timeout. ``None`` means wait forever.
    extra_env:
        Caller-supplied env additions, applied last (highest
        precedence). M1 dispatch rides ``HOOK_OUTPUT_FILE`` here;
        per-suite ``[evals.<name>.ref_env]`` overrides ride the same
        channel.
    extra_ref_args:
        Extra CLI args appended after ``bash <ref_script_path>``. The
        customer ref-script forwards ``"$@"`` straight to its
        ``torchrun`` invocation, so these land on the entry's argv
        unchanged.
    capture_loss_trace:
        When ``True``, sets ``LOSS_DUMP_FILE=<dump_dir>/ref_loss.txt``
        so the ref-script's loss hook can write a parsable trajectory.
    """
    import time

    ref_script_path = ref_script_path.resolve()
    if not ref_script_path.is_file():
        raise FileNotFoundError(
            f"Ref script not found: {ref_script_path}. The ref-as-gate "
            "SSOT model requires this file to exist."
        )

    dump_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = dump_dir / "ref_stdout.log"
    loss_file = dump_dir / "ref_loss.txt" if capture_loss_trace else None

    env = os.environ.copy()
    env["LOCAL_MODE"] = "1"
    env["FORGE_GATE"] = gate_name
    env["DUMP_DIR"] = str(dump_dir)
    if loss_file is not None:
        env["LOSS_DUMP_FILE"] = str(loss_file)
    if extra_env:
        env.update(extra_env)

    cmd = ["bash", str(ref_script_path)]
    if extra_ref_args:
        cmd.extend(str(a) for a in extra_ref_args)

    started = time.monotonic()
    timed_out = False
    # The ref script bashes ``torchrun``, which forks rank-worker
    # grandchildren; in ``--persistent`` capture mode a worker holds the
    # GPU until it is explicitly reaped. ``subprocess.run(timeout=…)`` only
    # signals the direct child (the bash/torchrun launcher) on timeout,
    # orphaning the grandchildren — they keep pinning GPU memory and OOM
    # the next gate. So launch in a dedicated process group
    # (``start_new_session=True``) and SIGKILL the whole group on every
    # exit path (timeout AND normal/error return), matching
    # ``evals._common.run_streaming_subprocess``.
    with stdout_path.open("w", encoding="utf-8") as out_fh:
        proc = subprocess.Popen(
            cmd,
            cwd=repo_root,
            env=env,
            stdout=out_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            returncode = -1
            timed_out = True
            _killpg(proc)
            proc.wait()
        finally:
            # Reap any grandchild left behind even on a clean launcher exit
            # (a persistent worker that outlived its parent). No-op once the
            # group is already gone.
            _killpg(proc)
    elapsed = time.monotonic() - started

    # Post-collapse, ``gate_metadata.json`` is read back ONLY for the
    # D5-exempt Stage-2 ``op-long`` suite, which has no rendered gate product
    # and therefore still sources its run shape from the launcher-emitted
    # metadata (see ``dispatcher_stage2._ref_metadata_int``'s ``shape is None``
    # fallback). Every Stage-1 gate reads its shape from the rendered product
    # instead and never consults this field.
    metadata_path = dump_dir / "gate_metadata.json"
    metadata: dict[str, object] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return RefRun(
        returncode=returncode,
        stdout_path=stdout_path,
        dump_dir=dump_dir,
        loss_file=loss_file,
        elapsed_s=elapsed,
        timed_out=timed_out,
        metadata=metadata,
    )
