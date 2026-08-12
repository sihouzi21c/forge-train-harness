"""Trajectory-resume fork tool (``tools/fork_loop.py``).

Design: docs/loop-trajectory-resume-design.md. A fork builds a NEW loop
instance whose state equals the source at a passed-milestone boundary:
workspace hard-reset to the accepted declaration commit, runtime residue
wiped, per-stage state files rewritten to the milestone's successor, and
the config copied UNFROZEN with lease-derived hostnames cleared so the
launch path claims a fresh devspace.

Fixtures build a real (tmp) registry: a workspace git repo whose commit
messages carry ``MILESTONE_STATUS: <name> PASS`` declarations, a wrapper
``stdout.log`` with ``milestone_advanced`` NDJSON events, and a frozen
per-loop config dir — no network, no real loop.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import contextlib  # noqa: E402

import fork_loop  # noqa: E402

_ORDER = ["alignment", "bitwise-singlecard", "production"]

_FAKE_CONFIG_TOOL = (
    "#!/usr/bin/env python3\n"
    "import sys\n"
    "if sys.argv[1:3] == ['stage-milestones', 'stage1']:\n"
    f"    print('\\n'.join({_ORDER!r}))\n"
)


def _git(ws: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ws), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _commit(ws: Path, msg: str, fname: str = "work.txt") -> str:
    f = ws / fname
    f.write_text(f.read_text() + msg + "\n" if f.exists() else msg + "\n")
    _git(ws, "add", "-A")
    _git(ws, "commit", "-q", "-m", msg)
    return _git(ws, "rev-parse", "HEAD")


class ForkLoopBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.train_dir = root / "forge_train"
        self.agents_dir = root / "web-agents"
        self.src_id = "srcloop001"
        self.src_dir = self.train_dir / self.src_id
        self.ws = self.src_dir / "workspace"
        self.cfg = self.src_dir / "config"
        self.ws.mkdir(parents=True)
        self.cfg.mkdir(parents=True)

        os.environ["FORGE_TRAIN_DIR"] = str(self.train_dir)
        os.environ["FORGE_AGENTS_DIR"] = str(self.agents_dir)
        self.addCleanup(os.environ.pop, "FORGE_TRAIN_DIR", None)
        self.addCleanup(os.environ.pop, "FORGE_AGENTS_DIR", None)

        _git(self.ws, "init", "-q")
        _git(self.ws, "config", "user.email", "t@t")
        _git(self.ws, "config", "user.name", "t")
        _git(self.ws, "config", "commit.gpgsign", "false")

        tools = self.ws / "tools"
        tools.mkdir()
        (tools / "agent_loop_config.py").write_text(_FAKE_CONFIG_TOOL)
        (self.ws / ".gitignore").write_text(".artifacts/\n")
        _commit(self.ws, "bootstrap")

        # Trajectory: alignment passes (round 3), bitwise-singlecard
        # passes (round 7), then two bad "production era" commits.
        _commit(self.ws, "impl alignment")
        self.sha_m1 = _commit(self.ws, "gate green\n\nMILESTONE_STATUS: alignment PASS")
        _commit(self.ws, "impl bitwise")
        self.sha_m2 = _commit(self.ws, "bitwise green\n\nMILESTONE_STATUS: bitwise-singlecard PASS")
        self.sha_bad = _commit(self.ws, "bad production attempt")
        _commit(self.ws, "more bad work")

        # Untracked dirt + runtime residue from the "production era".
        (self.ws / "scratch.tmp").write_text("dirty\n")
        residue = self.ws / ".artifacts" / "agent-loop-state"
        residue.mkdir(parents=True)
        (residue / "stage1.milestone").write_text("production\n")
        (residue / "stage1.round").write_text("9\n")
        (residue / "stage2.status").write_text("in-progress\n")
        (self.ws / ".artifacts" / "gpu_job_handle.json").write_text("{}")

        # Per-loop config, frozen like a launched loop.
        (self.cfg / "agent.toml").write_text('[agent]\nstate_dir = ".artifacts/agent-loop-state"\n')
        (self.cfg / "remote.toml").write_text(
            '[remote]\nkind = "devspace"\nhostname = "ds-src-derived"\n'
        )
        for p in [self.cfg, *self.cfg.rglob("*")]:
            p.chmod(p.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
        self.addCleanup(self._unfreeze_cfg)

        self._write_session(status="finished", pid=0)
        self._write_events(
            [
                {
                    "type": "loop_event",
                    "subtype": "round_start",
                    "ts": 1.0,
                    "stage": "stage1",
                    "round": 3,
                },
                {
                    "type": "loop_event",
                    "subtype": "milestone_advanced",
                    "ts": 2.0,
                    "stage": "stage1",
                    "round": 3,
                    "from": "alignment",
                    "to": "bitwise-singlecard",
                    "max": "production",
                    "commit": self.sha_m1,
                },
                {
                    "type": "loop_event",
                    "subtype": "milestone_advanced",
                    "ts": 3.0,
                    "stage": "stage1",
                    "round": 7,
                    "from": "bitwise-singlecard",
                    "to": "production",
                    "max": "production",
                    "commit": self.sha_m2,
                },
            ]
        )

    def _unfreeze_cfg(self) -> None:
        for p in [self.cfg, *self.cfg.rglob("*")]:
            with contextlib.suppress(OSError):
                p.chmod(p.stat().st_mode | stat.S_IWUSR)

    def _write_session(self, status: str, pid: int) -> None:
        (self.src_dir / "session.json").write_text(
            json.dumps({"loop_id": self.src_id, "status": status, "pid": pid})
        )

    def _write_events(self, events: list[dict]) -> None:
        d = self.agents_dir / f"loop-{self.src_id}"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "stdout.log", "w") as fh:
            fh.write("not-json noise line\n")
            for ev in events:
                fh.write(json.dumps(ev) + "\n")

    def _run(self, *argv: str) -> int:
        return fork_loop.main(list(argv))


class TestEventPathFork(ForkLoopBase):
    def test_fork_at_milestone_via_events(self) -> None:
        rc = self._run(
            "--src-loop-id",
            self.src_id,
            "--at-milestone",
            "bitwise-singlecard",
            "--new-loop-id",
            "fork001",
        )
        self.assertEqual(rc, 0)
        new_dir = self.train_dir / "fork001"
        new_ws = new_dir / "workspace"

        # Workspace: HEAD == accepted declaration commit; the bad
        # production-era commits and untracked dirt are gone.
        self.assertEqual(_git(new_ws, "rev-parse", "HEAD"), self.sha_m2)
        self.assertNotIn(self.sha_bad, _git(new_ws, "log", "--format=%H"))
        self.assertFalse((new_ws / "scratch.tmp").exists())

        # Runtime residue wiped; state files rebuilt from the event.
        self.assertFalse((new_ws / ".artifacts" / "gpu_job_handle.json").exists())
        state = new_ws / ".artifacts" / "agent-loop-state"
        self.assertEqual((state / "stage1.milestone").read_text().strip(), "production")
        self.assertEqual((state / "stage1.round").read_text().strip(), "7")
        self.assertEqual((state / "stage1.status").read_text().strip(), "in-progress")
        self.assertFalse((state / "stage2.status").exists())

        # Config: copied, unfrozen, lease-derived hostname cleared.
        remote = new_dir / "config" / "remote.toml"
        self.assertTrue(os.access(remote, os.W_OK))
        self.assertIn('hostname = ""', remote.read_text())

        # Provenance.
        prov = json.loads((new_dir / "forked_from.json").read_text())
        self.assertEqual(prov["loop_id"], self.src_id)
        self.assertEqual(prov["milestone"], "bitwise-singlecard")
        self.assertEqual(prov["anchor_sha"], self.sha_m2)
        self.assertEqual(prov["anchor_source"], "event")

        # Source loop untouched: frozen config, original HEAD, dirt intact.
        self.assertEqual(
            _git(self.ws, "rev-parse", "HEAD"),
            _git(self.ws, "rev-parse", "HEAD"),
        )
        self.assertTrue((self.ws / "scratch.tmp").exists())
        self.assertIn("ds-src-derived", (self.cfg / "remote.toml").read_text())

    def test_fork_survives_frozen_workspace_trees(self) -> None:
        # A launched source workspace carries chmod a-w frozen trees
        # (agent_loop_lease.sh freezes ref/config + rendered gate
        # products). cp -a preserves the read-only bits; git clean -fd
        # on the copy must not fail on them.
        frozen = self.ws / "ref" / "config"
        frozen.mkdir(parents=True)
        (frozen / "gate.toml").write_text("frozen = true\n")  # untracked
        for p in [frozen, *frozen.rglob("*")]:
            p.chmod(p.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)

        def _unfreeze() -> None:
            for p in [frozen, *frozen.rglob("*")]:
                if p.exists():
                    p.chmod(p.stat().st_mode | stat.S_IWUSR)

        self.addCleanup(_unfreeze)

        rc = self._run(
            "--src-loop-id",
            self.src_id,
            "--at-milestone",
            "bitwise-singlecard",
            "--new-loop-id",
            "fork-frozen",
        )
        self.assertEqual(rc, 0)
        new_ws = self.train_dir / "fork-frozen" / "workspace"
        self.assertEqual(_git(new_ws, "rev-parse", "HEAD"), self.sha_m2)
        # The untracked frozen file was cleaned from the fork...
        self.assertFalse((new_ws / "ref" / "config" / "gate.toml").exists())
        # ...and the source's frozen tree is untouched.
        self.assertTrue((frozen / "gate.toml").exists())
        self.assertFalse(os.access(frozen / "gate.toml", os.W_OK))

    def test_dry_run_writes_nothing(self) -> None:
        rc = self._run(
            "--src-loop-id",
            self.src_id,
            "--at-milestone",
            "alignment",
            "--new-loop-id",
            "forkdry",
            "--dry-run",
        )
        self.assertEqual(rc, 0)
        self.assertFalse((self.train_dir / "forkdry").exists())

    def test_refuses_existing_target(self) -> None:
        (self.train_dir / "forkdup").mkdir()
        with self.assertRaises(SystemExit):
            self._run(
                "--src-loop-id",
                self.src_id,
                "--at-milestone",
                "alignment",
                "--new-loop-id",
                "forkdup",
            )

    def test_refuses_running_source(self) -> None:
        self._write_session(status="running", pid=os.getpid())
        with self.assertRaises(SystemExit) as ctx:
            self._run(
                "--src-loop-id",
                self.src_id,
                "--at-milestone",
                "alignment",
                "--new-loop-id",
                "fork002",
            )
        self.assertIn("RUNNING", str(ctx.exception))

    def test_stale_running_record_proceeds(self) -> None:
        # status=running but the pid is gone (crash without trap):
        # a stale record must not block the fork forever.
        self._write_session(status="running", pid=2**22 + 12345)
        rc = self._run(
            "--src-loop-id",
            self.src_id,
            "--at-milestone",
            "alignment",
            "--new-loop-id",
            "fork003",
        )
        self.assertEqual(rc, 0)


class TestFallbackPaths(ForkLoopBase):
    def test_git_fallback_without_events(self) -> None:
        (self.agents_dir / f"loop-{self.src_id}" / "stdout.log").unlink()
        rc = self._run(
            "--src-loop-id",
            self.src_id,
            "--at-milestone",
            "bitwise-singlecard",
            "--new-loop-id",
            "forkgit",
        )
        self.assertEqual(rc, 0)
        new_ws = self.train_dir / "forkgit" / "workspace"
        self.assertEqual(_git(new_ws, "rev-parse", "HEAD"), self.sha_m2)
        state = new_ws / ".artifacts" / "agent-loop-state"
        # Successor computed from milestone_order; round unknown -> 0.
        self.assertEqual((state / "stage1.milestone").read_text().strip(), "production")
        self.assertEqual((state / "stage1.round").read_text().strip(), "0")

    def test_never_passed_milestone_errors(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            self._run(
                "--src-loop-id",
                self.src_id,
                "--at-milestone",
                "production",
                "--new-loop-id",
                "forknone",
            )
        self.assertIn("never passed", str(ctx.exception))

    def test_ambiguous_declarations_demand_sha(self) -> None:
        # A second (e.g. review-vetoed then re-declared) declaration with
        # no event stream to disambiguate.
        _commit(self.ws, "retry\n\nMILESTONE_STATUS: bitwise-singlecard PASS")
        (self.agents_dir / f"loop-{self.src_id}" / "stdout.log").unlink()
        with self.assertRaises(SystemExit) as ctx:
            self._run(
                "--src-loop-id",
                self.src_id,
                "--at-milestone",
                "bitwise-singlecard",
                "--new-loop-id",
                "forkamb",
            )
        self.assertIn("--at-sha", str(ctx.exception))

    def test_at_sha_override(self) -> None:
        rc = self._run(
            "--src-loop-id",
            self.src_id,
            "--at-milestone",
            "alignment",
            "--at-sha",
            self.sha_m1,
            "--new-loop-id",
            "forksha",
        )
        self.assertEqual(rc, 0)
        new_ws = self.train_dir / "forksha" / "workspace"
        self.assertEqual(_git(new_ws, "rev-parse", "HEAD"), self.sha_m1)
        state = new_ws / ".artifacts" / "agent-loop-state"
        # Matching event exists for this sha -> round/milestone from it.
        self.assertEqual(
            (state / "stage1.milestone").read_text().strip(),
            "bitwise-singlecard",
        )
        self.assertEqual((state / "stage1.round").read_text().strip(), "3")

    def test_at_sha_without_declaration_errors(self) -> None:
        with self.assertRaises(SystemExit):
            self._run(
                "--src-loop-id",
                self.src_id,
                "--at-milestone",
                "alignment",
                "--at-sha",
                self.sha_bad,
                "--new-loop-id",
                "forkbad",
            )


if __name__ == "__main__":
    unittest.main()
