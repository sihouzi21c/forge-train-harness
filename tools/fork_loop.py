#!/usr/bin/env python3
"""Fork a loop from a passed-milestone boundary ("轨迹续跑").

Builds a NEW loop instance dir whose state equals the source loop at the
moment a given milestone was accepted, so the normal ``--loop-id`` resume
contract continues from the NEXT milestone instead of from scratch. The
source loop is never mutated.

Anchor resolution (design: harness/docs/loop-trajectory-resume-design.md):

* Primary: the ``milestone_advanced`` loop_event in the source wrapper's
  ``stdout.log`` — carries the accepted declaration ``commit`` (anchor
  SHA), ``to`` (next active milestone) and ``round`` in one record.
* Fallback (missing/damaged event stream): ``git log --grep`` for the
  ``MILESTONE_STATUS: <name> PASS`` declaration. Ambiguous (multiple
  declarations, e.g. a review-vetoed retry) → refuse and ask for
  ``--at-sha``.

Usage:
    python3 harness/tools/fork_loop.py \\
        --src-loop-id <id> --at-milestone <name> \\
        [--at-sha <sha>] [--new-loop-id <id>] [--dry-run]

The tool prints the launch command but never starts the loop itself —
the gap between fork and launch is where the operator fixes whatever
made the original trajectory go wrong (configs, gates, devspace).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

try:  # py3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

_LOOP_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_DEFAULT_STATE_DIR = ".artifacts/agent-loop-state"


def _repo_root() -> Path:
    """Anchor of the ``.artifacts`` registry (env overrides win)."""
    here = Path(__file__).resolve()
    root = here.parents[2]
    if (root / "harness").is_dir():
        return root
    # Running from inside a provisioned workspace copy
    # (<registry>/<loop_id>/workspace/tools/fork_loop.py): the registry
    # root is three levels above the workspace.
    ws = here.parents[1]
    candidate = ws.parents[1]
    if (ws / "agent-loop.sh").is_file() and candidate.name == "forge_train":
        return candidate.parents[1]
    raise SystemExit(
        "fork_loop: cannot locate repo root; set FORGE_TRAIN_DIR and FORGE_AGENTS_DIR explicitly."
    )


def _train_dir() -> Path:
    env = os.environ.get("FORGE_TRAIN_DIR", "").strip()
    return Path(env) if env else _repo_root() / ".artifacts" / "forge_train"


def _agents_dir() -> Path:
    env = os.environ.get("FORGE_AGENTS_DIR", "").strip()
    return Path(env) if env else _repo_root() / ".artifacts" / "web-agents"


def _decl_pattern(milestone: str) -> str:
    """ERE identical to advance_stage_milestone's acceptance grep."""
    return rf"^MILESTONE_STATUS:[[:space:]]+{re.escape(milestone)}[[:space:]]+PASS[[:space:]]*$"


def _decl_re(milestone: str) -> re.Pattern[str]:
    return re.compile(
        rf"^MILESTONE_STATUS:[ \t]+{re.escape(milestone)}[ \t]+PASS[ \t]*$",
        re.MULTILINE,
    )


def _git(workspace: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(workspace), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def _commit_declares(workspace: Path, sha: str, milestone: str) -> bool:
    try:
        body = _git(workspace, "show", "-s", "--format=%B", sha)
    except subprocess.CalledProcessError:
        return False
    return bool(_decl_re(milestone).search(body))


def _load_events(agents_dir: Path, loop_id: str) -> list[dict]:
    # SSOT: the wrapper Session id convention lives in web.paths.
    if str(_repo_root()) not in sys.path:
        sys.path.insert(0, str(_repo_root()))
    from web.paths import wrapper_agent_id

    log = agents_dir / wrapper_agent_id(loop_id) / "stdout.log"
    events: list[dict] = []
    if not log.is_file():
        return events
    with open(log, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and rec.get("type") == "loop_event":
                events.append(rec)
    return events


@dataclass
class Anchor:
    sha: str
    next_milestone: str
    round: int
    stage: str
    source: str  # "event" | "git-fallback" | "explicit-sha"


def _milestone_successor(src_workspace: Path, config_dir: Path, name: str) -> str:
    """Successor of *name* in the active milestone_order, capped at last.

    Mirrors next_milestone_after: "<name> PASS" means <name>'s gate has
    passed, so the active milestone becomes the one after it.
    """
    env = dict(os.environ)
    env["FORGE_CONFIG_DIR"] = str(config_dir)
    out = subprocess.run(
        [
            sys.executable,
            str(src_workspace / "tools" / "agent_loop_config.py"),
            "stage-milestones",
            "stage1",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(src_workspace),
    )
    order = [m for m in out.stdout.split() if m]
    if out.returncode != 0 or name not in order:
        raise SystemExit(
            f"fork_loop: cannot resolve milestone_order from the source loop "
            f"config (rc={out.returncode}, order={order!r}); the event stream "
            f"is also unusable, so pass --at-sha AND ensure the config dir is "
            f"intact."
        )
    idx = order.index(name)
    return order[idx + 1] if idx + 1 < len(order) else order[-1]


def resolve_anchor(
    *,
    src_workspace: Path,
    src_config: Path,
    events: list[dict],
    milestone: str,
    at_sha: str = "",
) -> Anchor:
    advanced = [
        ev for ev in events if ev.get("subtype") == "milestone_advanced" and ev.get("commit")
    ]
    matching = [
        ev for ev in advanced if _commit_declares(src_workspace, str(ev["commit"]), milestone)
    ]

    if at_sha:
        if not _commit_declares(src_workspace, at_sha, milestone):
            raise SystemExit(
                f"fork_loop: --at-sha {at_sha} does not exist in the source "
                f"workspace or its message carries no "
                f"'MILESTONE_STATUS: {milestone} PASS' declaration."
            )
        full = _git(src_workspace, "rev-parse", at_sha)
        for ev in reversed(matching):
            if _git(src_workspace, "rev-parse", str(ev["commit"])) == full:
                return Anchor(
                    full,
                    str(ev["to"]),
                    int(ev.get("round", 0)),
                    str(ev.get("stage", "stage1")),
                    "explicit-sha",
                )
        nxt = _milestone_successor(src_workspace, src_config, milestone)
        print(
            "fork_loop: WARN — no matching milestone_advanced event for "
            "--at-sha; round unknown, writing 0 (display-only).",
            file=sys.stderr,
        )
        return Anchor(full, nxt, 0, "stage1", "explicit-sha")

    if matching:
        ev = matching[-1]  # the accepted (latest) advance for this milestone
        return Anchor(
            _git(src_workspace, "rev-parse", str(ev["commit"])),
            str(ev["to"]),
            int(ev.get("round", 0)),
            str(ev.get("stage", "stage1")),
            "event",
        )

    # git fallback: no usable event stream.
    out = _git(
        src_workspace,
        "log",
        "-E",
        f"--grep={_decl_pattern(milestone)}",
        "--format=%H",
    )
    shas = [s for s in out.splitlines() if s]
    if not shas:
        raise SystemExit(
            f"fork_loop: milestone '{milestone}' has no accepted PASS "
            f"declaration in the source loop (neither event stream nor git "
            f"history). It was never passed — nothing to fork from."
        )
    if len(shas) > 1:
        raise SystemExit(
            f"fork_loop: {len(shas)} commits declare "
            f"'MILESTONE_STATUS: {milestone} PASS' and no event stream "
            f"disambiguates which one was accepted (review-gated milestones "
            f"may carry vetoed declarations). Pick one and pass --at-sha:\n  " + "\n  ".join(shas)
        )
    nxt = _milestone_successor(src_workspace, src_config, milestone)
    print(
        "fork_loop: WARN — anchored via git fallback (no event stream); "
        "round unknown, writing 0 (display-only).",
        file=sys.stderr,
    )
    return Anchor(shas[0], nxt, 0, "stage1", "git-fallback")


def _refuse_if_running(instance_dir: Path) -> None:
    sess = instance_dir / "session.json"
    if not sess.is_file():
        print(
            "fork_loop: WARN — source loop has no session.json; assuming not running.",
            file=sys.stderr,
        )
        return
    try:
        data = json.loads(sess.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise SystemExit(f"fork_loop: unreadable source session.json: {exc}") from exc
    if data.get("status") != "running":
        return
    pid = data.get("pid")
    if isinstance(pid, int) and pid > 0:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            print(
                f"fork_loop: WARN — session.json says running but pid {pid} "
                f"is gone (stale record after a crash); continuing.",
                file=sys.stderr,
            )
            return
        except PermissionError:
            pass  # exists, owned by someone else — treat as alive
        raise SystemExit(
            f"fork_loop: source loop is RUNNING (pid {pid}). Stop it first — "
            f"forking a live loop reads half-written state."
        )


def _state_dir_rel(config_dir: Path) -> str:
    agent_toml = config_dir / "agent.toml"
    if tomllib is not None and agent_toml.is_file():
        try:
            cfg = tomllib.loads(agent_toml.read_text(encoding="utf-8"))
            raw = str(cfg.get("agent", {}).get("state_dir", "")).strip()
            if raw:
                return raw
        except Exception:  # nosec B110 — optional agent.toml override; fall back to default
            pass
    return _DEFAULT_STATE_DIR


def _clear_devspace_hostname(remote_toml: Path) -> bool:
    """Reset hostname for lease-derived kinds so launch claims a fresh one.

    Returns True when a rewrite happened. Static-ssh / local hostnames are
    kept: they are not per-loop resources (remote dirs are loop_id-scoped).
    """
    if not remote_toml.is_file():
        return False
    text = remote_toml.read_text(encoding="utf-8")
    kind_m = re.search(r'^kind\s*=\s*"([^"]*)"', text, re.MULTILINE)
    if not kind_m or kind_m.group(1) not in ("devspace", "job"):
        return False
    new_text, n = re.subn(r'^hostname\s*=\s*".*"', 'hostname = ""', text, flags=re.MULTILINE)
    if n:
        remote_toml.write_text(new_text, encoding="utf-8")
    return bool(n)


def _make_writable(root: Path) -> None:
    for p in [root, *root.rglob("*")]:
        with contextlib.suppress(OSError):
            p.chmod(p.stat().st_mode | stat.S_IWUSR)


def build_fork(
    *,
    train_dir: Path,
    src_id: str,
    new_id: str,
    anchor: Anchor,
    milestone: str,
    dry_run: bool = False,
) -> Path:
    src_dir = train_dir / src_id
    new_dir = train_dir / new_id
    if new_dir.exists():
        raise SystemExit(f"fork_loop: target {new_dir} already exists.")

    src_ws = src_dir / "workspace"
    src_cfg = src_dir / "config"
    new_ws = new_dir / "workspace"
    new_cfg = new_dir / "config"
    state_rel = _state_dir_rel(src_cfg)
    state_dir = new_ws / state_rel

    plan = [
        f"anchor        : {anchor.sha}  (source: {anchor.source})",
        f"copy workspace: {src_ws} -> {new_ws}",
        f"git           : reset --hard {anchor.sha[:12]} && clean -fd",
        f"reset runtime : rm -rf {new_ws / '.artifacts'}",
        f"state files   : {state_dir}/{anchor.stage}.milestone={anchor.next_milestone} "
        f".round={anchor.round} .status=in-progress",
        f"copy config   : {src_cfg} -> {new_cfg} (unfreeze; devspace/job "
        f"hostname cleared -> launch claims a fresh lease)",
        f"provenance    : {new_dir / 'forked_from.json'}",
    ]
    if dry_run:
        print("fork_loop: DRY RUN — planned actions:")
        for line in plan:
            print(f"  {line}")
        return new_dir

    new_dir.mkdir(parents=True)
    try:
        subprocess.run(["cp", "-a", str(src_ws), str(new_ws)], check=True)
        # A launched source workspace carries chmod a-w frozen trees
        # (ref/config, rendered gate products); cp -a preserves the
        # read-only bits and git reset/clean cannot rewrite them.
        _make_writable(new_ws)
        subprocess.run(
            ["git", "-C", str(new_ws), "reset", "--hard", anchor.sha],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(new_ws), "clean", "-fd"],
            check=True,
            capture_output=True,
            text=True,
        )
        # Source runtime residue (logs, gpu-job handles, old loop state)
        # must not leak into the new trajectory: anchor state == the pure
        # committed state (design §3.4/§4.1).
        runtime = new_ws / ".artifacts"
        if runtime.exists():
            _make_writable(runtime)
            shutil.rmtree(runtime)

        # A custom [agent].state_dir outside .artifacts would survive the
        # runtime wipe above — the copied stage files (possibly incl. a
        # stage2 entered by the source) must never leak into the fork.
        if state_dir.exists():
            _make_writable(state_dir)
            shutil.rmtree(state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / f"{anchor.stage}.milestone").write_text(
            anchor.next_milestone + "\n", encoding="utf-8"
        )
        (state_dir / f"{anchor.stage}.round").write_text(f"{anchor.round}\n", encoding="utf-8")
        (state_dir / f"{anchor.stage}.status").write_text("in-progress\n", encoding="utf-8")

        if src_cfg.is_dir():
            subprocess.run(["cp", "-a", str(src_cfg), str(new_cfg)], check=True)
            _make_writable(new_cfg)
            _clear_devspace_hostname(new_cfg / "remote.toml")
        else:
            print(
                f"fork_loop: WARN — source has no config dir at {src_cfg}; "
                f"seed {new_cfg} manually before launch.",
                file=sys.stderr,
            )

        provenance = {
            "loop_id": src_id,
            "milestone": milestone,
            "anchor_sha": anchor.sha,
            "anchor_source": anchor.source,
            "forked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        (new_dir / "forked_from.json").write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except BaseException:
        _make_writable(new_dir)
        shutil.rmtree(new_dir, ignore_errors=True)
        raise

    for line in plan:
        print(f"fork_loop: {line}")
    return new_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fork_loop", description=__doc__)
    p.add_argument("--src-loop-id", required=True)
    p.add_argument(
        "--at-milestone", required=True, help="passed milestone whose end is the fork anchor"
    )
    p.add_argument("--at-sha", default="", help="explicit anchor commit (disambiguation fallback)")
    p.add_argument("--new-loop-id", default="")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    train_dir = _train_dir()
    src_dir = train_dir / args.src_loop_id
    src_ws = src_dir / "workspace"
    if not (src_ws / ".git").exists():
        raise SystemExit(
            f"fork_loop: no workspace git repo under {src_ws} — is "
            f"'{args.src_loop_id}' a provisioned loop id?"
        )

    _refuse_if_running(src_dir)

    new_id = args.new_loop_id or uuid.uuid4().hex[:12]
    if not _LOOP_ID_RE.match(new_id):
        raise SystemExit(f"fork_loop: invalid --new-loop-id {new_id!r}")

    events = _load_events(_agents_dir(), args.src_loop_id)
    anchor = resolve_anchor(
        src_workspace=src_ws,
        src_config=src_dir / "config",
        events=events,
        milestone=args.at_milestone,
        at_sha=args.at_sha,
    )

    new_dir = build_fork(
        train_dir=train_dir,
        src_id=args.src_loop_id,
        new_id=new_id,
        anchor=anchor,
        milestone=args.at_milestone,
        dry_run=args.dry_run,
    )

    verb = "would fork" if args.dry_run else "forked"
    print(
        f"\nfork_loop: {verb} '{args.src_loop_id}' @ milestone "
        f"'{args.at_milestone}' -> new loop '{new_id}'."
    )
    print("fork_loop: launch with:\n")
    print(f"  bash harness/agent-loop.sh --loop-id {new_id} --config-dir {new_dir / 'config'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
