"""Per-loop directory layout SSOT.

Single authoritative source for the on-disk topology of a forge-train
loop: where its frozen per-loop config dir and its isolated workspace
live under ``<forge_train_dir>/<loop_id>/``.

Every consumer — the ``new-looptask`` skill, ``agent-loop.sh``, and the
``web`` layer — MUST resolve these paths through this module rather than
re-deriving ``<forge_train_dir>/<id>/config`` by hand. The functions are
pure (no env reads, no cwd) so they stay trivially testable and reusable
from any layer above ``harness``.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["loop_config_dir", "loop_workspace_dir"]


def loop_config_dir(forge_train_dir: str | Path, loop_id: str) -> Path:
    """Frozen per-loop config dir: ``<forge_train_dir>/<loop_id>/config``.

    Sibling of the workspace so it survives a workspace re-provision and
    can be ``chmod -R a-w`` frozen independently once the loop launches.
    """
    return Path(forge_train_dir) / loop_id / "config"


def loop_workspace_dir(forge_train_dir: str | Path, loop_id: str) -> Path:
    """Isolated per-loop workspace: ``<forge_train_dir>/<loop_id>/workspace``."""
    return Path(forge_train_dir) / loop_id / "workspace"
