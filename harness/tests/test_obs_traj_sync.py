"""Tests for scripts/obs_traj_push.sh / obs_traj_pull.sh (append-only OBS sync).

obsutil is stubbed with a local-filesystem implementation so the suite runs
offline.  The stub mirrors the nesting semantics probed against the real
service: uploading a directory nests its basename under the destination
prefix; downloading a remote folder nests the folder name under the local
destination unless -flat is given; -u copies only new/changed files.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[3]
PUSH = REPO / "scripts" / "obs_traj_push.sh"
PULL = REPO / "scripts" / "obs_traj_pull.sh"

PREFIX = "obs://train-datasets/ForgeTrainData/trajectories"

OBSUTIL_STUB = '''#!/usr/bin/env python3
import json, os, shutil, sys

ROOT = os.environ["FAKE_OBS_ROOT"]
LOG = os.environ["FAKE_OBS_LOG"]


def log(entry):
    with open(LOG, "a") as f:
        f.write(json.dumps(entry) + "\\n")


def to_local(url):
    return os.path.join(ROOT, url[len("obs://"):].rstrip("/"))


def newer(src, dst):
    return (
        not os.path.exists(dst)
        or os.path.getsize(src) != os.path.getsize(dst)
        or os.path.getmtime(src) > os.path.getmtime(dst) + 1e-6
    )


def copy_tree(src, dst, update, copied):
    for base, _dirs, files in os.walk(src):
        rel = os.path.relpath(base, src)
        tdir = dst if rel == "." else os.path.join(dst, rel)
        os.makedirs(tdir, exist_ok=True)
        for name in files:
            s, d = os.path.join(base, name), os.path.join(tdir, name)
            if not update or newer(s, d):
                shutil.copy2(s, d)
                copied.append(os.path.normpath(os.path.join(rel, name)))


def main():
    args = sys.argv[1:]
    cmd = args[0] if args else ""
    flags = {a.split("=")[0] for a in args if a.startswith("-")}
    pos = [a for a in args[1:] if not a.startswith("-")]
    if cmd == "stat":
        # real obsutil: bucket stat works bare; folder stat REQUIRES a
        # trailing slash; file stat must not have one
        url = pos[0]
        p = to_local(url)
        if "/" not in url[len("obs://"):].rstrip("/"):
            rc = 0 if os.path.isdir(p) else 1
        elif url.endswith("/"):
            rc = 0 if os.path.isdir(p) else 1
        else:
            rc = 0 if os.path.isfile(p) else 1
        log({"argv": args, "rc": rc})
        return rc
    if cmd == "cp":
        src, dst, update, copied, rc = pos[0], pos[1], "-u" in flags, [], 0
        if src.startswith("obs://"):
            s = to_local(src)
            if not os.path.exists(s):
                rc = 1
            elif os.path.isdir(s):
                tgt = dst if "-flat" in flags else os.path.join(dst, os.path.basename(s))
                copy_tree(s, tgt, update, copied)
            else:
                d = os.path.join(dst, os.path.basename(s)) if os.path.isdir(dst) else dst
                if not update or newer(s, d):
                    os.makedirs(os.path.dirname(d), exist_ok=True)
                    shutil.copy2(s, d)
                    copied.append(d)
        else:
            d = to_local(dst)
            if os.path.isdir(src):
                copy_tree(src, os.path.join(d, os.path.basename(src.rstrip("/"))), update, copied)
            else:
                if dst.endswith("/"):
                    d = os.path.join(d, os.path.basename(src))
                os.makedirs(os.path.dirname(d), exist_ok=True)
                if not update or newer(src, d):
                    shutil.copy2(src, d)
                    copied.append(d)
        log({"argv": args, "rc": rc, "copied": copied})
        return rc
    log({"argv": args, "rc": 0})
    return 0


sys.exit(main())
'''


@pytest.fixture()
def env(tmp_path):
    fake_root = tmp_path / "obs"
    (fake_root / "train-datasets").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "obsutil"
    stub.write_text(OBSUTIL_STUB)
    stub.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    agents = tmp_path / "web-agents"
    agents.mkdir()
    e = os.environ.copy()
    e.update(
        {
            "FAKE_OBS_ROOT": str(fake_root),
            "FAKE_OBS_LOG": str(tmp_path / "obs.log"),
            "OBSUTIL_BIN": str(stub),
            "OBS_TRAJ_PREFIX": PREFIX,
            "FORGE_AGENTS_DIR": str(agents),
            "HOME": str(home),
            "OBS_PUSH_RETRY_DELAY": "0",
            "OBS_PUSH_USER": "alice",
            "CLAUDE_PROJECTS_DIR": str(home / ".claude" / "projects"),
        }
    )
    remote = fake_root / "train-datasets" / "ForgeTrainData" / "trajectories"
    return SimpleNamespace(
        env=e,
        agents=agents,
        remote=remote,
        home=home,
        log=tmp_path / "obs.log",
        tmp=tmp_path,
    )


def make_tree(agents_dir, tree_id, state="completed", files=None):
    d = agents_dir / tree_id
    d.mkdir(parents=True)
    session = {
        "agent_id": tree_id,
        "state": state,
        "started_at": "2026-07-15T01:00:00Z",
        "backend_session_id": "bs-" + tree_id,
        "kind": "loop_wrapper" if tree_id.startswith("loop-") else "chat",
    }
    (d / "session.json").write_text(json.dumps(session))
    (d / "stdout.log").write_text("event line\n")
    for rel, content in (files or {}).items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return d


def run(script, ctx, *args):
    return subprocess.run(
        ["bash", str(script), *args], env=ctx.env, capture_output=True, text=True
    )


def log_entries(ctx):
    if not ctx.log.exists():
        return []
    return [json.loads(x) for x in ctx.log.read_text().splitlines()]


def cp_entries_for(ctx, needle):
    return [
        e
        for e in log_entries(ctx)
        if e["argv"][0] == "cp" and any(needle in a for a in e["argv"])
    ]


class TestPush:
    def test_terminal_tree_pushed_with_meta_last(self, env):
        make_tree(env.agents, "loop-aaa111", files={"agents/stage1/r001/x/session.json": "{}"})
        r = run(PUSH, env)
        assert r.returncode == 0, r.stderr
        dest = env.remote / "alice" / "loop-aaa111"
        assert (dest / "stdout.log").read_text() == "event line\n"
        assert (dest / "agents/stage1/r001/x/session.json").exists()
        meta = json.loads((dest / "_push_meta.json").read_text())
        assert meta["pushed_by"] == "alice"
        assert meta["agent_id"] == "loop-aaa111"
        assert meta["state"] == "completed"
        assert meta["fingerprint"]
        # meta upload must be the last cp touching this tree
        tree_cps = cp_entries_for(env, "loop-aaa111")
        assert any("_push_meta.json" in a for a in tree_cps[-1]["argv"])

    def test_running_tree_skipped_by_default(self, env):
        make_tree(env.agents, "loop-run111", state="running")
        r = run(PUSH, env)
        assert r.returncode == 0
        assert not (env.remote / "alice" / "loop-run111").exists()

    def test_running_tree_pushed_with_flag_but_no_meta(self, env):
        make_tree(env.agents, "loop-run222", state="running")
        r = run(PUSH, env, "--include-running")
        assert r.returncode == 0
        dest = env.remote / "alice" / "loop-run222"
        assert (dest / "stdout.log").exists()
        assert not (dest / "_push_meta.json").exists()

    def test_repush_unchanged_is_noop(self, env):
        make_tree(env.agents, "loop-bbb222")
        assert run(PUSH, env).returncode == 0
        before = len(log_entries(env))
        assert run(PUSH, env).returncode == 0
        added = log_entries(env)[before:]
        assert not [e for e in added if e["argv"][0] == "cp"]

    def test_incremental_append_only_new_file(self, env):
        d = make_tree(env.agents, "loop-ccc333")
        assert run(PUSH, env).returncode == 0
        (d / "extra.txt").write_text("late file\n")
        assert run(PUSH, env).returncode == 0
        tree_cps = [
            e
            for e in cp_entries_for(env, "loop-ccc333")
            if "_push_meta" not in " ".join(e["argv"])
        ]
        assert tree_cps[-1]["copied"] == ["extra.txt"]
        assert (env.remote / "alice" / "loop-ccc333" / "extra.txt").exists()

    def test_fingerprint_conflict_never_overwrites(self, env):
        make_tree(env.agents, "loop-ddd444")
        rdir = env.remote / "alice" / "loop-ddd444"
        rdir.mkdir(parents=True)
        (rdir / "stdout.log").write_text("SOMEONE ELSES DATA\n")
        (rdir / "_push_meta.json").write_text(json.dumps({"fingerprint": "different"}))
        r = run(PUSH, env)
        assert r.returncode == 0
        assert (rdir / "stdout.log").read_text() == "SOMEONE ELSES DATA\n"
        conflicts = list((env.remote / "alice").glob("loop-ddd444__conflict-*"))
        assert len(conflicts) == 1
        assert (conflicts[0] / "stdout.log").read_text() == "event line\n"
        # conflict resolution is idempotent: rerun creates no second copy
        assert run(PUSH, env).returncode == 0
        assert len(list((env.remote / "alice").glob("loop-ddd444__conflict-*"))) == 1

    def test_claude_transcripts_uploaded_into_tree(self, env):
        make_tree(env.agents, "loop-98179bd97008")
        tdir = env.home / ".claude" / "projects" / "-x-forge-98179bd97008-workspace"
        tdir.mkdir(parents=True)
        (tdir / "thinking.jsonl").write_text('{"t":1}\n')
        assert run(PUSH, env).returncode == 0
        dest = env.remote / "alice" / "loop-98179bd97008" / "transcripts" / "thinking.jsonl"
        assert dest.read_text() == '{"t":1}\n'

    def test_run_record_uploaded(self, env):
        make_tree(env.agents, "loop-eee555")
        assert run(PUSH, env).returncode == 0
        records = list((env.remote / "_records" / "alice").glob("*.json"))
        assert len(records) == 1
        rec = json.loads(records[0].read_text())
        assert "loop-eee555" in json.dumps(rec["pushed"])

    def test_meta_repo_from_provenance_file(self, env):
        prov = {"branch": "develop", "commit": "c" * 40, "created_from": "origin/main"}
        make_tree(
            env.agents, "loop-prov01", files={"repo_provenance.json": json.dumps(prov)}
        )
        assert run(PUSH, env).returncode == 0
        meta = json.loads(
            (env.remote / "alice" / "loop-prov01" / "_push_meta.json").read_text()
        )
        assert meta["repo"]["created_from"] == "origin/main"
        assert meta["repo"]["branch"] == "develop"

    def test_legacy_flat_web_tree_pushed_as_is(self, env):
        make_tree(env.agents, "web-0123abcd")
        assert run(PUSH, env).returncode == 0
        assert (env.remote / "alice" / "web-0123abcd" / "stdout.log").exists()

    def test_dry_run_pushes_nothing(self, env):
        make_tree(env.agents, "loop-fff666")
        r = run(PUSH, env, "--dry-run")
        assert r.returncode == 0
        assert "loop-fff666" in r.stdout
        assert not (env.remote / "alice").exists()

    def test_missing_obsutil_soft_skips(self, env):
        make_tree(env.agents, "loop-ggg777")
        env.env["OBSUTIL_BIN"] = "/nonexistent/obsutil"
        r = run(PUSH, env)
        assert r.returncode == 0
        assert "skip" in (r.stdout + r.stderr).lower()

    def test_unreachable_bucket_soft_skips(self, env):
        make_tree(env.agents, "loop-hhh888")
        env.env["OBS_TRAJ_PREFIX"] = "obs://no-such-bucket/x/trajectories"
        r = run(PUSH, env)
        assert r.returncode == 0
        assert not (env.remote / "alice").exists()


class TestPull:
    def seed_remote(self, env):
        for user, tree, with_meta in [
            ("alice", "loop-aaa111", True),
            ("bob", "web-flat01", False),
        ]:
            d = env.remote / user / tree
            d.mkdir(parents=True)
            (d / "stdout.log").write_text(f"{user} data\n")
            if with_meta:
                (d / "_push_meta.json").write_text(json.dumps({"fingerprint": "f"}))

    def test_pull_mirrors_and_appends(self, env):
        self.seed_remote(env)
        dest = env.tmp / "mirror"
        r = run(PULL, env, "--dest", str(dest))
        assert r.returncode == 0, r.stderr
        assert (dest / "alice" / "loop-aaa111" / "stdout.log").exists()
        assert (dest / "bob" / "web-flat01" / "stdout.log").exists()
        # local extra data survives, remote additions arrive
        extra = dest / "alice" / "loop-aaa111" / "local-only.txt"
        extra.write_text("keep me\n")
        (env.remote / "alice" / "loop-aaa111" / "late.txt").write_text("new\n")
        assert run(PULL, env, "--dest", str(dest)).returncode == 0
        assert extra.read_text() == "keep me\n"
        assert (dest / "alice" / "loop-aaa111" / "late.txt").exists()

    def test_pull_single_user(self, env):
        self.seed_remote(env)
        dest = env.tmp / "mirror"
        assert run(PULL, env, "--dest", str(dest), "--user", "alice").returncode == 0
        assert (dest / "alice" / "loop-aaa111" / "stdout.log").exists()
        assert not (dest / "bob").exists()

    def test_link_into_only_complete_trees(self, env):
        self.seed_remote(env)
        dest = env.tmp / "mirror"
        forest = env.tmp / "forest"
        forest.mkdir()
        # a pre-existing tree with the same id must not be overwritten
        (forest / "loop-aaa111").mkdir()
        r = run(PULL, env, "--dest", str(dest), "--link-into", str(forest))
        assert r.returncode == 0
        assert not (forest / "web-flat01").exists()  # no _push_meta -> not linked
        assert not (forest / "loop-aaa111").is_symlink()  # existing kept

    def test_agent_loop_exit_hook_contract(self):
        """The wrapper EXIT trap ships the freshly-terminal tree: the push
        fires after finalize_wrapper_session (so this loop's own tree is
        terminal), default-on, env-gated, backgrounded."""
        text = (REPO / "harness" / "agent-loop.sh").read_text()
        assert "FORGE_OBS_PUSH_ON_EXIT:-1" in text  # default-on, 0 disables
        handler = text[
            text.index("_loop_exit_handler()") : text.index(
                "trap _loop_exit_handler EXIT"
            )
        ]
        assert handler.index("finalize_wrapper_session") < handler.index(
            "obs_traj_push.sh"
        )
        hook_line = [l for l in handler.splitlines() if "obs_traj_push.sh" in l][0]
        assert "--quiet" in hook_line
        assert ") &" in handler[handler.index("FORGE_OBS_PUSH_ON_EXIT"):]

    def test_push_lock_held_by_live_process_exits_zero(self, env):
        make_tree(env.agents, "loop-lock01")
        lock = env.home / ".forge_obs_push_state.json.lock"
        lock.mkdir()
        (lock / "pid").write_text(str(os.getpid()))  # alive holder
        r = run(PUSH, env)
        assert r.returncode == 0
        assert "another push" in (r.stdout + r.stderr).lower()
        assert not (env.remote / "alice").exists()

    def test_push_steals_stale_lock(self, env):
        make_tree(env.agents, "loop-lock02")
        proc = subprocess.run(["true"])  # any surely-dead pid
        lock = env.home / ".forge_obs_push_state.json.lock"
        lock.mkdir()
        (lock / "pid").write_text("99999999")
        del proc
        r = run(PUSH, env)
        assert r.returncode == 0
        assert (env.remote / "alice" / "loop-lock02" / "stdout.log").exists()
        assert not lock.exists()  # released after the run

    def test_agent_loop_startup_hook_contract(self):
        """agent-loop.sh fires the push on startup: default-on, env-gated,
        backgrounded, and tolerant of a workspace copy without scripts/."""
        text = (REPO / "harness" / "agent-loop.sh").read_text()
        hook = [l for l in text.splitlines() if "obs_traj_push.sh" in l]
        assert hook, "startup push hook missing from agent-loop.sh"
        assert "FORGE_OBS_PUSH_ON_START:-1" in text  # default-on, 0 disables
        assert "scripts/obs_traj_sync.py" in text  # guarded on script presence
        assert "--quiet" in hook[0]
        block = text[text.index("FORGE_OBS_PUSH_ON_START"):]
        assert ") &" in block.split("fi")[0]  # fire-and-forget subshell

    def test_link_into_links_new_tree(self, env):
        self.seed_remote(env)
        dest = env.tmp / "mirror"
        forest = env.tmp / "forest"
        forest.mkdir()
        assert run(PULL, env, "--dest", str(dest), "--link-into", str(forest)).returncode == 0
        link = forest / "loop-aaa111"
        assert link.is_symlink()
        assert (link / "stdout.log").read_text() == "alice data\n"
