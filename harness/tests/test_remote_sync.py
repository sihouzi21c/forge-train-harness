"""Tests for ``harness sync push`` — remote workspace synchronization."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["FORGE_REPO_ROOT"] = str(REPO_ROOT)

from harness import remote_sync  # noqa: E402


class TestBuildRsyncCommand(unittest.TestCase):
    def test_stage1_includes_delete_and_standard_excludes(self) -> None:
        cmd = remote_sync._build_rsync_command(
            source="/local/workspace/",
            host="ds-403009",
            dest="/remote/workdir/",
            stage="stage1",
            verbose=False,
        )
        self.assertIn("--delete", cmd)
        self.assertIn("--exclude=.git", cmd)
        self.assertIn("--exclude=__pycache__", cmd)
        self.assertIn("--exclude=.artifacts", cmd)
        self.assertIn("--exclude=.venv", cmd)
        self.assertIn("--exclude=.pytest_cache", cmd)
        self.assertIn("--exclude=.ruff_cache", cmd)
        self.assertIn("--exclude=workload/profile", cmd)
        # Remote-born profile verdict artifacts (workload/notes/profile/…)
        # must survive the mirror --delete.
        self.assertIn("--exclude=workload/notes", cmd)
        # Source ends with /
        self.assertTrue(any(c.endswith("/") and "/local/" in c for c in cmd))
        # Destination is host:path
        self.assertIn("ds-403009:/remote/workdir/", cmd)

    def test_stage2_omits_git_exclude(self) -> None:
        cmd = remote_sync._build_rsync_command(
            source="/local/workspace/",
            host="ds-403009",
            dest="/remote/workdir/",
            stage="stage2",
            verbose=False,
        )
        self.assertIn("--delete", cmd)
        self.assertNotIn("--exclude=.git", cmd)
        self.assertIn("--exclude=__pycache__", cmd)

    def test_verbose_adds_v_flag(self) -> None:
        cmd = remote_sync._build_rsync_command(
            source="/local/workspace/",
            host="ds-403009",
            dest="/remote/workdir/",
            stage="stage1",
            verbose=True,
        )
        self.assertIn("-avz", cmd)

    def test_quiet_uses_az(self) -> None:
        cmd = remote_sync._build_rsync_command(
            source="/local/workspace/",
            host="ds-403009",
            dest="/remote/workdir/",
            stage="stage1",
            verbose=False,
        )
        self.assertIn("-az", cmd)
        self.assertNotIn("-avz", cmd)


class TestBuildConfigPushCommand(unittest.TestCase):
    """The active per-loop config lives in the SIBLING ``FORGE_CONFIG_DIR``
    (``.artifacts/forge_train/<id>/config``), which the main rsync excludes
    via ``.artifacts``. The remote ``harness run`` reads config from
    ``<workdir>/config`` (FORGE_CONFIG_DIR is never exported across the ssh
    hop), so without a dedicated push the active ``*.toml`` never reach the
    remote — the agent hand-pushed them and the next ``--delete`` rsync
    orphaned them. ``sync push`` must re-ship the active config every time.
    """

    def test_config_push_targets_remote_config_dir(self) -> None:
        cmd = remote_sync._build_config_push_command(
            host="ds-403009",
            dest="/remote/workdir",
            config_dir="/local/loop/config",
        )
        # Source is the active config dir (trailing slash → contents).
        self.assertTrue(any(c == "/local/loop/config/" for c in cmd))
        # Destination is <dest>/config/ on the remote host.
        self.assertIn("ds-403009:/remote/workdir/config/", cmd)

    def test_config_push_ships_only_toml_and_never_deletes(self) -> None:
        cmd = remote_sync._build_config_push_command(
            host="ds-403009",
            dest="/remote/workdir",
            config_dir="/local/loop/config",
        )
        # Only top-level *.toml — leave the template subdirs (config/eval,
        # config/agent, …) that the main rsync already placed untouched.
        self.assertIn("--include=*.toml", cmd)
        self.assertIn("--exclude=*", cmd)
        # NEVER --delete: that would wipe the synced template subdirs.
        self.assertNotIn("--delete", cmd)


class TestBuildCleanupCommand(unittest.TestCase):
    def test_cleanup_removes_pycache_on_remote(self) -> None:
        cmd = remote_sync._build_cleanup_command(
            host="ds-403009",
            dest="/remote/workdir/",
        )
        self.assertIn("ssh", cmd[0])
        self.assertIn("ds-403009", cmd)
        joined = " ".join(cmd)
        self.assertIn("__pycache__", joined)
        self.assertIn("rm -rf", joined)


class TestBuildValidationCommand(unittest.TestCase):
    def test_validation_checks_required_paths(self) -> None:
        cmd = remote_sync._build_validation_command(
            host="ds-403009",
            dest="/remote/workdir/",
        )
        joined = " ".join(cmd)
        self.assertIn("pyproject.toml", joined)
        self.assertIn("harness/config/defaults.toml", joined)
        self.assertIn("config/eval", joined)


class TestSyncPush(unittest.TestCase):
    @mock.patch.dict(os.environ, {"LOOP_ID": "deadbeef99"}, clear=False)
    @mock.patch("harness.remote_sync._resolve_remote_params")
    @mock.patch("subprocess.run")
    def test_sync_push_calls_rsync_cleanup_validation(
        self, mock_run: mock.Mock, mock_params: mock.Mock
    ) -> None:
        mock_params.return_value = (
            "/local/workspace/",
            "ds-403009",
            "/remote/workdir/",
            "/local/loop/config",
        )
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

        remote_sync.sync_push(stage="stage1", verbose=False)

        # mkdir + rsync(tree) + rsync(config) + cleanup + validation +
        # mfu pull-back = 6 calls.
        self.assertEqual(mock_run.call_count, 6)
        # First call: ssh mkdir
        mkdir_call = mock_run.call_args_list[0]
        self.assertIn("ssh", mkdir_call.args[0][0])
        self.assertIn("mkdir", " ".join(mkdir_call.args[0]))
        # Second call: rsync (workspace tree)
        rsync_call = mock_run.call_args_list[1]
        self.assertIn("rsync", rsync_call.args[0][0])
        # Third call: rsync (active per-loop config)
        config_call = mock_run.call_args_list[2]
        self.assertIn("rsync", config_call.args[0][0])
        self.assertIn("ds-403009:/remote/workdir/config/", config_call.args[0])
        # Fourth call: ssh cleanup
        cleanup_call = mock_run.call_args_list[3]
        self.assertIn("ssh", cleanup_call.args[0][0])
        # Fifth call: ssh validation
        validation_call = mock_run.call_args_list[4]
        self.assertIn("ssh", validation_call.args[0][0])
        # Sixth call: rsync pull-back of mfu_history.jsonl
        pull_call = mock_run.call_args_list[5]
        self.assertIn("rsync", pull_call.args[0][0])
        self.assertIn(
            "ds-403009:/remote/workdir/.artifacts/forge_train/deadbeef99/mfu_history.jsonl",
            pull_call.args[0],
        )

    @mock.patch("harness.remote_sync._resolve_remote_params")
    @mock.patch("subprocess.run")
    def test_sync_push_raises_on_rsync_failure(
        self, mock_run: mock.Mock, mock_params: mock.Mock
    ) -> None:
        mock_params.return_value = (
            "/local/workspace/",
            "ds-403009",
            "/remote/workdir/",
            "/local/loop/config",
        )
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=12, stdout="", stderr="rsync error"
        )

        with self.assertRaises(RuntimeError) as ctx:
            remote_sync.sync_push(stage="stage1", verbose=False)
        self.assertIn("rsync", str(ctx.exception).lower())

    @mock.patch("harness.remote_sync._resolve_remote_params")
    @mock.patch("subprocess.run")
    def test_sync_push_raises_on_validation_failure(
        self, mock_run: mock.Mock, mock_params: mock.Mock
    ) -> None:
        mock_params.return_value = (
            "/local/workspace/",
            "ds-403009",
            "/remote/workdir/",
            "/local/loop/config",
        )

        def side_effect(cmd, **kwargs):
            if "rsync" in cmd[0]:
                return subprocess.CompletedProcess(args=cmd, returncode=0)
            if "mkdir" in " ".join(cmd):
                return subprocess.CompletedProcess(args=cmd, returncode=0)
            if "find" in " ".join(cmd):
                return subprocess.CompletedProcess(args=cmd, returncode=0)
            # validation fails
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="MISSING: pyproject.toml"
            )

        mock_run.side_effect = side_effect

        with self.assertRaises(RuntimeError) as ctx:
            remote_sync.sync_push(stage="stage1", verbose=False)
        self.assertIn("validation", str(ctx.exception).lower())


class TestMfuPullBack(unittest.TestCase):
    """``mfu_history.jsonl`` lives on the remote (``tools/mfu_record.py``
    runs inside the gate subprocess). ``sync_push`` is the
    high-frequency local-side hook that pulls it back so the web UI's
    ``/api/artifacts/mfu`` does not silently lag. The pull is
    best-effort: failure here MUST NEVER turn a successful sync into a
    raised exception.
    """

    def test_build_mfu_pull_targets_nested_remote_layout(self) -> None:
        # Remote layout (verified on ds-429964):
        # <dest>/.artifacts/forge_train/<loop_id>/mfu_history.jsonl
        # The nested ``forge_train/<loop_id>`` comes from the
        # FORGE_TRAIN_DIR export in remote-execution.md §Step 2.
        cmd = remote_sync._build_mfu_pull_command(
            host="ds-403009",
            dest="/user/me/.forge_train/deadbeef99",
            loop_id="deadbeef99",
            local_dest="/local/loop/mfu_history.jsonl",
        )
        self.assertEqual(cmd[0], "rsync")
        # ``-az`` so we match the rest of the wrapper's style and rsync
        # writes through a tempfile + atomic rename (web reader never
        # sees a half-written file).
        self.assertIn("-az", cmd)
        self.assertIn(
            "ds-403009:/user/me/.forge_train/deadbeef99/"
            ".artifacts/forge_train/deadbeef99/mfu_history.jsonl",
            cmd,
        )
        self.assertIn("/local/loop/mfu_history.jsonl", cmd)

    @mock.patch.dict(os.environ, {"LOOP_ID": "deadbeef99"}, clear=False)
    @mock.patch("subprocess.run")
    def test_pull_writes_to_local_loop_dir_next_to_workspace(self, mock_run: mock.Mock) -> None:
        # source = ``<loop_dir>/workspace`` ⇒ local mfu_history sits in
        # ``<loop_dir>/`` (sibling of workspace), matching the layout
        # ``web/routers/artifacts.py:get_mfu`` already reads.
        remote_sync._pull_mfu_history(
            host="ds-403009",
            dest="/remote/workdir",
            source="/local/forge_train/deadbeef99/workspace",
        )
        self.assertEqual(mock_run.call_count, 1)
        cmd = mock_run.call_args.args[0]
        self.assertIn("rsync", cmd[0])
        self.assertIn("/local/forge_train/deadbeef99/mfu_history.jsonl", cmd)

    @mock.patch.dict(os.environ, {"LOOP_ID": ""}, clear=True)
    @mock.patch("subprocess.run")
    def test_pull_skipped_when_loop_id_unavailable(self, mock_run: mock.Mock) -> None:
        # Without LOOP_ID we can't compute the remote path; skip
        # silently rather than guess and pull garbage.
        remote_sync._pull_mfu_history(
            host="ds-403009",
            dest="/remote/workdir",
            source="/local/workspace",
        )
        mock_run.assert_not_called()

    @mock.patch.dict(os.environ, {"LOOP_ID": "deadbeef99"}, clear=False)
    @mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired("rsync", 30))
    def test_pull_failure_does_not_raise(self, _mock_run: mock.Mock) -> None:
        # The whole point of the best-effort contract: telemetry I/O
        # failure must not propagate. ``sync_push`` callers downstream
        # treat a raised exception as a hard remote-sync failure.
        try:
            remote_sync._pull_mfu_history(
                host="ds-403009",
                dest="/remote/workdir",
                source="/local/forge_train/deadbeef99/workspace",
            )
        except Exception as exc:  # pragma: no cover — would mean a real regression
            self.fail(f"_pull_mfu_history raised: {exc!r}")

    @mock.patch.dict(os.environ, {"LOOP_ID": "deadbeef99"}, clear=False)
    @mock.patch("harness.remote_sync._resolve_remote_params")
    @mock.patch("subprocess.run")
    def test_sync_push_succeeds_even_if_pull_fails(
        self, mock_run: mock.Mock, mock_params: mock.Mock
    ) -> None:
        mock_params.return_value = (
            "/local/workspace/",
            "ds-403009",
            "/remote/workdir/",
            "/local/loop/config",
        )

        # First 5 calls succeed (mkdir + rsync tree + rsync config +
        # cleanup + validation); the 6th call is the mfu pull-back and
        # we make it throw.
        call_count = {"n": 0}

        def side_effect(cmd, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 6:
                raise OSError("simulated rsync transport failure")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        mock_run.side_effect = side_effect

        # Must not raise — telemetry pull failure is best-effort.
        result = remote_sync.sync_push(stage="stage1", verbose=False)
        self.assertEqual(result["status"], "ready")


class TestSyncRender(unittest.TestCase):
    """The sync payload must render without raising — the CLI prints it
    outside the try/except, so a missing branch crashes a *successful*
    sync into a non-zero exit and misleads the agent into retrying."""

    def _payload(self) -> dict:
        return {
            "command": "sync",
            "report": "text",
            "status": "ready",
            "payload": {
                "action": "push",
                "host": "ds-403009",
                "dest": "/remote/workdir",
                "stage": "stage1",
            },
        }

    def test_render_text_handles_sync(self) -> None:
        from harness import presentation

        out = presentation.render(self._payload(), "text")
        self.assertIn("ready", out)
        self.assertIn("ds-403009", out)
        self.assertIn("/remote/workdir", out)
        self.assertIn("stage1", out)

    def test_render_json_handles_sync(self) -> None:
        import json

        from harness import presentation

        out = json.loads(presentation.render(self._payload(), "json"))
        self.assertEqual(out["command"], "sync")
        self.assertEqual(out["status"], "ready")


class TestSyncPushCLI(unittest.TestCase):
    def test_cli_parser_accepts_sync_push(self) -> None:
        from harness import cli

        parser = cli._build_parser()
        args = parser.parse_args(["sync", "push"])
        self.assertEqual(args.command, "sync")
        self.assertEqual(args.sync_action, "push")

    def test_cli_parser_accepts_stage2_flag(self) -> None:
        from harness import cli

        parser = cli._build_parser()
        args = parser.parse_args(["sync", "push", "--stage2"])
        self.assertTrue(args.stage2)

    def test_cli_parser_accepts_verbose_flag(self) -> None:
        from harness import cli

        parser = cli._build_parser()
        args = parser.parse_args(["sync", "push", "--verbose"])
        self.assertTrue(args.verbose)


if __name__ == "__main__":
    unittest.main()
