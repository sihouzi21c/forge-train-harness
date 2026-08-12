"""Tests for web/agents/provenance.py.

Git provenance of the source checkout a loop was launched from
(branch/commit, including the branch a git worktree was created from),
captured BEFORE the workspace copy strips .git, folded into the wrapper
tree by init_wrapper_session, and referenced from both provision sites.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from web.agents import provenance, spawn, store

REPO = Path(__file__).resolve().parents[3]


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture()
def repo(tmp_path):
    src = tmp_path / "main"
    src.mkdir()
    _git(src, "init", "-b", "develop")
    _git(src, "config", "user.email", "t@example.com")
    _git(src, "config", "user.name", "tester")
    (src / "f.txt").write_text("x\n")
    _git(src, "add", "-A")
    _git(src, "commit", "-m", "init")
    return src


class TestCapture:
    def test_plain_repo(self, repo):
        p = provenance.capture(repo)
        assert p["branch"] == "develop"
        assert len(p["commit"]) == 40
        assert p["is_worktree"] is False
        assert p["dirty"] is False and p["dirty_files"] == 0

    def test_worktree_records_created_from(self, repo, tmp_path):
        wt = tmp_path / "wt"
        _git(repo, "worktree", "add", "-b", "loop/x", str(wt), "develop")
        p = provenance.capture(wt)
        assert p["branch"] == "loop/x"
        assert p["is_worktree"] is True
        assert p["created_from"] == "develop"
        assert p["main_worktree_branch"] == "develop"

    def test_dirty_counted(self, repo):
        (repo / "f.txt").write_text("changed\n")
        (repo / "new.txt").write_text("n\n")
        p = provenance.capture(repo)
        assert p["dirty"] is True and p["dirty_files"] == 2

    def test_non_repo_returns_none(self, tmp_path):
        plain = tmp_path / "plain"
        plain.mkdir()
        assert provenance.capture(plain) is None
        assert provenance.capture(tmp_path / "missing") is None


class TestWrite:
    def test_write_and_first_capture_wins(self, repo, tmp_path):
        inst = tmp_path / "inst"
        out = provenance.write(repo, inst)
        assert json.loads(out.read_text())["branch"] == "develop"
        # a later re-provision (resume) must NOT rewrite the origin story
        _git(repo, "checkout", "-b", "later")
        assert provenance.write(repo, inst) == out
        assert json.loads(out.read_text())["branch"] == "develop"

    def test_write_non_repo_is_noop(self, tmp_path):
        plain = tmp_path / "plain"
        plain.mkdir()
        assert provenance.write(plain, tmp_path / "inst") is None


class TestWrapperIntegration:
    def test_init_wrapper_session_folds_provenance(self, repo, tmp_path, monkeypatch):
        monkeypatch.setattr(store, "AGENTS_DIR", tmp_path / "agents")
        inst = tmp_path / "forge_train" / "lp1"
        workspace = inst / "workspace"
        workspace.mkdir(parents=True)
        provenance.write(repo, inst)
        sess = spawn.init_wrapper_session(
            loop_id="lp1",
            workspace=str(workspace),
            backend_label="claude",
            model="m",
            stages="stage1",
        )
        sd = store.session_dir(sess.agent_id)
        data = json.loads((sd / "repo_provenance.json").read_text())
        assert data["branch"] == "develop"
        assert sess.source_branch == "develop"

    def test_init_without_provenance_still_works(self, tmp_path, monkeypatch):
        monkeypatch.setattr(store, "AGENTS_DIR", tmp_path / "agents")
        workspace = tmp_path / "ft" / "lp2" / "workspace"
        workspace.mkdir(parents=True)
        sess = spawn.init_wrapper_session(
            loop_id="lp2",
            workspace=str(workspace),
            backend_label="claude",
            model="m",
            stages="stage1",
        )
        assert sess.source_branch == ""

    def test_both_provision_sites_write_provenance(self):
        assert "provenance.write" in (REPO / "web" / "routers" / "loop.py").read_text()
        assert "provenance" in (REPO / "harness" / "agent-loop.sh").read_text()
