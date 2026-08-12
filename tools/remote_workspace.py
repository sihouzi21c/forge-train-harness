"""Shared resolver/validator for the devspace remote-root (``[remote].workspace``).

Single source of truth for one rule the launch flow depends on: a
``kind = "devspace"`` or ``kind = "job"`` loop MUST point
``[remote].workspace`` at a **shared, persistent cluster path** (e.g.
``/user/<username>`` on GPFS) that is rw-mounted in BOTH the devspace
(held box / 0-GPU gateway) AND any GPU job pod. The pod-local ``$HOME``
(which resolves to ``/root`` on the release image) is neither shared
across pods nor persistent, so:

* in ``kind = "devspace"`` it is fragile (lost on devspace restart), and
* in ``kind = "job"`` it is **broken** — ``sync push`` lands the code on
  the gateway pod's ``/root`` while the ephemeral GPU job has its own,
  different ``/root`` and cannot see it.

Both ``tools/agent_loop_lease.sh`` (claim-time gate, both kinds) and
``tools/gpu_job.py`` (kind=job run, defense-in-depth) enforce this via
this module; the module depends on neither, so the layering stays a DAG.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["main", "resolve_remote_workdir", "validate_shared_workspace"]


def validate_shared_workspace(workspace: str | None) -> str:
    """Return the cleaned shared remote-root or raise ``ValueError``.

    Rejects the values that silently resolve to the pod-local ``/root``
    (empty / ``$HOME`` / ``~``) and any non-absolute or ``/root`` path.
    The message names the fix so the wrapper/agent can surface it.
    """
    ws = (workspace or "").strip()
    hint = (
        "set it to a shared cluster path rw-mounted in both the devspace and "
        "any GPU job, e.g. /user/<username> (the new-looptask devspace intake "
        "collects <username> for exactly this)"
    )
    if not ws:
        raise ValueError(
            f"empty — defaults to the remote $HOME, which is the pod-local "
            f"/root on the release image (not shared across pods, not "
            f"persistent); {hint}"
        )
    if ws in ("$HOME", "~"):
        raise ValueError(
            f"{ws!r} resolves to the pod-local /root (not shared, not persistent); {hint}"
        )
    if not ws.startswith("/"):
        raise ValueError(f"{ws!r} is not an absolute path; {hint}")
    if ws == "/root" or ws.startswith("/root/"):
        raise ValueError(
            f"{ws!r} is the pod-local home (not shared across pods, not persistent); {hint}"
        )
    return ws


def resolve_remote_workdir(remote: Mapping[str, object], loop_id: str) -> str:
    """Resolve the per-loop remote workdir under the validated shared root.

    ``<workspace>/.forge_train/<loop_id>`` — identical layout to the ssh
    transport, but with the shared-root invariant enforced.
    """
    ws = validate_shared_workspace(str(remote.get("workspace", "") or ""))
    return f"{ws}/.forge_train/{loop_id}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="tools.remote_workspace",
        description="Validate a devspace [remote].workspace shared remote-root.",
    )
    p.add_argument("--validate", metavar="WORKSPACE", required=True)
    args = p.parse_args(argv)
    try:
        validate_shared_workspace(args.validate)
    except ValueError as exc:
        print(f"invalid [remote].workspace: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
