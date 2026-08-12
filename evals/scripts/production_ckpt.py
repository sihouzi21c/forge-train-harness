"""Production checkpoint helper for ``run_ours_production.sh`` (thin dispatcher).

Verbatim port of the dispatcher's crash-resume primitives so the segment
loop can live in sh while the "what counts as a complete checkpoint" and
"where do we resume" judgments stay in one Python SSOT (shared with
``evals.verdicts.production`` via the same completeness rule):

    latest <save_root>     print ``<abs_step>\t<dir>`` of the highest
                           COMPLETE ``step_<N>`` checkpoint (``0\t`` when
                           none / root missing — a fresh start).
    check <step_dir>       exit 0 iff the dir is a complete checkpoint
                           (holds a non-empty ``*.pt``); 1 otherwise.
    prefetch-wait          block until the background corpus prefetch is
                           staged when ``[data].prefetch_target_gb`` is set
                           (no-op otherwise); nonzero exit on error/timeout.

Lives under evals/scripts/ (NOT tools/): ``prefetch-wait`` imports
``evals._common`` for the workload-config load, and the layer DAG is
one-way evals/ -> tools/.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Default ceiling for how long production waits on the background corpus
# prefetch before failing fast (the alignment–long-horizon dev loop normally
# gives it a long head start, so this is a backstop, not the expected wait).
# Overridable via ``[data].prefetch_wait_timeout_s``.
_PREFETCH_WAIT_TIMEOUT_S = 6 * 3600


def checkpoint_complete(step_dir: Path) -> bool:
    """A checkpoint dir counts as complete iff it holds a non-empty ``*.pt``.

    A dir that exists but has only zero-byte / no ``*.pt`` (a crash mid
    write) is NOT complete, so a restarted run re-runs that segment
    instead of resuming from a half-written state.
    """
    return step_dir.is_dir() and any(p.stat().st_size > 0 for p in step_dir.glob("*.pt"))


def latest_checkpoint(save_root: Path) -> tuple[Path | None, int]:
    """Return ``(dir, abs_step)`` of the highest completed checkpoint.

    Scans ``save_root`` for ``step_<N>`` directories that pass
    :func:`checkpoint_complete` and returns the one with the largest ``N``.
    Returns ``(None, 0)`` when the root is missing or holds no complete
    checkpoint — i.e. a fresh start from step 0.
    """
    if not save_root.is_dir():
        return None, 0
    best_dir: Path | None = None
    best_step = 0
    for child in save_root.glob("step_*"):
        suffix = child.name[len("step_") :]
        if not suffix.isdigit():
            continue
        step = int(suffix)
        if step > best_step and checkpoint_complete(child):
            best_dir = child
            best_step = step
    return best_dir, best_step


def prefetch_wait() -> None:
    """Block until the background corpus prefetch is staged, if enabled.

    No-op when ``[data].prefetch_target_gb`` is unset/0. Otherwise reads the
    ``tools.prefetch_data`` sentinel under ``[data].forge_data_dir`` and
    waits for ``status == "ok"``, raising on prefetch error or timeout so a
    misconfigured run fails before training on a half-staged corpus.
    """
    from evals._common import _load_workload_config_for_ref
    from harness import config_runtime

    workload_config = _load_workload_config_for_ref(Path.cwd())
    data = workload_config.get("data") or {}
    if not isinstance(data, dict):
        return
    try:
        target_gb = float(data.get("prefetch_target_gb", 0) or 0)
    except (TypeError, ValueError):
        target_gb = 0.0
    if target_gb <= 0:
        return
    forge_data_dir = str(data.get("forge_data_dir") or "").strip()
    if not forge_data_dir:
        raise ValueError(
            "[data].prefetch_target_gb is set but [data].forge_data_dir is "
            "empty — the prefetch sentinel has no home"
        )
    timeout_s = float(data.get("prefetch_wait_timeout_s", _PREFETCH_WAIT_TIMEOUT_S))
    from tools import prefetch_data

    # Resolve the same way the prefetch writer does: a relative dir lands
    # under the workspace repo root, NOT cwd, so the sentinel we wait on is
    # the one the prefetch wrote.
    resolved_dir = config_runtime.resolve_forge_data_dir(forge_data_dir)
    status = prefetch_data.wait_for_prefetch(resolved_dir, timeout_s=timeout_s)
    print(
        f"[production-train] prefetch ready: {status.get('bytes', 0) / 1024**3:.1f}GB "
        f"staged under {resolved_dir}",
        flush=True,
    )


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    cmd, args = argv[0], argv[1:]
    if cmd == "latest":
        if len(args) != 1:
            print("usage: production_ckpt.py latest <save_root>", file=sys.stderr)
            return 2
        ckpt_dir, step = latest_checkpoint(Path(args[0]))
        print(f"{step}\t{ckpt_dir if ckpt_dir is not None else ''}")
        return 0
    if cmd == "check":
        if len(args) != 1:
            print("usage: production_ckpt.py check <step_dir>", file=sys.stderr)
            return 2
        return 0 if checkpoint_complete(Path(args[0])) else 1
    if cmd == "prefetch-wait":
        prefetch_wait()
        return 0
    print(f"unknown subcommand: {cmd!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
