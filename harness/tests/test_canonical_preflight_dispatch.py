"""Unit coverage for ``tools/canonical_preflight.sh`` — the local-vs-remote
dispatcher agent-loop.sh uses to run the canonical-state preflight on the
host where the ref gates actually execute.

For ``[remote].kind = local`` the canonicals are materialized on this
machine. For ssh / devspace loops the tokenizer + data mounts (e.g.
``/opt/forge-data``) and the GPUs live ONLY on the remote, so running the
preflight locally crashes (``PermissionError: /opt/forge-data``) — it MUST
run on the SSH host after seeding it with the current workspace.

The dispatcher is extracted from agent-loop.sh so the command construction
is testable without a real ssh round-trip: ``CANONICAL_PREFLIGHT_PRINT=1``
makes it print the resolved command(s) it would run, one per line, instead
of executing them.
"""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

# Test path: harness/harness/tests/<file>. parents[2] is the flat harness/
# import root that carries tools/canonical_preflight.sh.
HARNESS_DIR = Path(__file__).resolve().parents[2]
HELPER = HARNESS_DIR / "tools" / "canonical_preflight.sh"


def _run(env: dict[str, str]) -> subprocess.CompletedProcess:
    base = {"CANONICAL_PREFLIGHT_PRINT": "1", "PYTHON": "python3", "PATH": "/usr/bin:/bin"}
    base.update(env)
    return subprocess.run(
        ["bash", str(HELPER)],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
        env=base,
    )


class TestCanonicalPreflightDispatch(unittest.TestCase):
    def test_helper_exists(self) -> None:
        self.assertTrue(HELPER.is_file(), f"expected helper at {HELPER}")

    def test_local_runs_preflight_in_process(self) -> None:
        res = _run({"REMOTE_ENABLED": "false"})
        self.assertEqual(res.returncode, 0, res.stderr)
        lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
        # A local loop runs the preflight directly — one command, no ssh.
        self.assertEqual(len(lines), 1, res.stdout)
        self.assertIn("evals.canonical_preflight", lines[0])
        self.assertNotIn("ssh", lines[0])

    def test_remote_seeds_then_runs_on_ssh_host(self) -> None:
        res = _run(
            {
                "REMOTE_ENABLED": "true",
                "REMOTE_SSH_HOST": "ds-516904",
                "REMOTE_WORKDIR": "/user/x/.forge_train/abc",
                "REMOTE_CONFIG_DIR": "/user/x/.forge_train/abc/config",
            }
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
        # Two steps: (1) seed the remote workspace, (2) run the preflight
        # on the ssh host where the mounts + GPUs live.
        self.assertEqual(len(lines), 2, res.stdout)
        self.assertIn("sync push", lines[0])
        self.assertIn("ssh ds-516904", lines[1])
        self.assertIn("cd /user/x/.forge_train/abc", lines[1])
        self.assertIn("FORGE_CONFIG_DIR=/user/x/.forge_train/abc/config", lines[1])
        self.assertIn("evals.canonical_preflight", lines[1])

    def test_job_kind_submits_gpu_job_not_ssh(self) -> None:
        # kind=job: the devspace gateway holds 0 GPUs, so the 2-GPU canonical
        # bridge MUST be submitted as an ephemeral cctl pytorchjob (tools.gpu_job
        # --canonical) — NOT ssh'd onto the 0-GPU gateway (the old bug that
        # died with bridge.sh rc=6).
        res = _run(
            {
                "REMOTE_ENABLED": "true",
                "REMOTE_KIND": "job",
                "REMOTE_SSH_HOST": "ds-539213",  # 0-GPU gateway — must NOT run GPU work
                "REMOTE_WORKDIR": "/user/x/.forge_train/abc",
                "REMOTE_CONFIG_DIR": "/user/x/.forge_train/abc/config",
            }
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
        # Two steps: (1) seed the shared volume, (2) submit the canonical
        # bootstrap as a cctl GPU job locally (cctl/tsh live on the launcher).
        self.assertEqual(len(lines), 2, res.stdout)
        self.assertIn("sync push", lines[0])
        self.assertIn("tools.gpu_job", lines[1])
        self.assertIn("--canonical", lines[1])
        self.assertNotIn("ssh", lines[1])

    def test_ssh_kind_still_runs_on_ssh_host(self) -> None:
        # An explicit kind=ssh (GPU host reachable via ssh) keeps the
        # ssh-exec path — only kind=job diverts to a cctl job.
        res = _run(
            {
                "REMOTE_ENABLED": "true",
                "REMOTE_KIND": "ssh",
                "REMOTE_SSH_HOST": "ds-1",
                "REMOTE_WORKDIR": "/w",
                "REMOTE_CONFIG_DIR": "/w/config",
            }
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
        self.assertIn("ssh ds-1", lines[1])
        self.assertNotIn("tools.gpu_job", lines[1])

    def test_remote_missing_ssh_host_fails_fast(self) -> None:
        # kind unset/ssh with no host → ssh path fails fast (job mode never
        # needs REMOTE_SSH_HOST, so this stays scoped to the ssh path).
        res = _run(
            {
                "REMOTE_ENABLED": "true",
                "REMOTE_WORKDIR": "/user/x/.forge_train/abc",
                "REMOTE_CONFIG_DIR": "/user/x/.forge_train/abc/config",
            }
        )
        self.assertNotEqual(res.returncode, 0)


if __name__ == "__main__":
    unittest.main()
