"""Devspace lease lifecycle helper (``tools/agent_loop_lease.sh``).

Pins the missing half of the R12 lease fix: the claim + hostname
rewrite + per-loop config freeze must run for BOTH the CLI bootstrap
path and the web launch path. The logic previously lived only inside the
``FORGE_TRAIN_PROVISIONED != 1`` bootstrap block, so web-launched loops
(which set that sentinel) skipped it entirely — no GPU lease was claimed
(silent leak + anti-collision defeated) and no keepalive SSH stanza was
synthesized (a black-holed devspace then hung the loop for hours).

The helper keys off config-dir writability: a writable dir is a first
launch (claim + freeze); a read-only dir is a resume (no-op, so a
resumed loop never double-books a second devspace).

``cctl`` / ``tsh`` are mocked via the same ``CCTL_BIN`` / ``TSH_BIN``
overrides ``tools.lease`` honors — no real CLI, no network.
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
HELPER = REPO_ROOT / "tools" / "agent_loop_lease.sh"

_NEW_ID = "990099"

_DEVSPACE_TOML = (
    "[remote]\n"
    'kind = "devspace"\n'
    'project = "loopharness"\n'
    'cluster = "paratera_shandong"\n'
    'resource_pool = "faxin"\n'
    'image = "infra/autoloop-harness:train-engine"\n'
    "gpu_count = 2\n"
    'gpu_model = "h100"\n'
    'priority = "HIGH"\n'
    'workspace = "/user/liyuxuan"\n'
    'hostname = ""\n'
)


def _write_fake_cctl(tmp: Path) -> Path:
    bin_path = tmp / "fake-cctl"
    log_path = tmp / "fake-cctl.log"
    script = (
        "#!/usr/bin/env bash\n"
        "set -e\n"
        f'echo "$@" >> {log_path}\n'
        'if [ "$1" = "devspace" ] && [ "$2" = "create" ]; then\n'
        f'  printf \'{{"id": {_NEW_ID}, "name": "tasks/{_NEW_ID}", "status": "Running"}}\\n\'\n'
        'elif [ "$1" = "devspace" ] && [ "$2" = "get" ]; then\n'
        '  id="${3#tasks/}"\n'
        '  printf \'{"id": %s, "name": "tasks/%s", "status": "Running"}\\n\' "$id" "$id"\n'
        "fi\n"
        "exit 0\n"
    )
    bin_path.write_text(script, encoding="utf-8")
    bin_path.chmod(bin_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bin_path


def _write_fake_tsh(tmp: Path) -> Path:
    bin_path = tmp / "fake-tsh"
    log_path = tmp / "fake-tsh.log"
    node = f"devspace-liyuxuan-loopharness-{_NEW_ID}"
    payload = f'[{{"spec": {{"hostname": "{node}"}}}}]'
    # `_write_synthesized_stanza` resolves the per-user key material via
    # `tsh status --format json`; the fake must answer `status` too or the
    # claim aborts with "tsh status carried no active.username".
    status_payload = '{"active": {"username": "liyuxuan"}}'
    script = (
        "#!/usr/bin/env bash\n"
        f'echo "$@" >> {log_path}\n'
        'if [ "$1" = "ls" ]; then\n'
        f"  printf '{payload}\\n'\n"
        'elif [ "$1" = "status" ]; then\n'
        f"  printf '{status_payload}\\n'\n"
        "fi\n"
        "exit 0\n"
    )
    bin_path.write_text(script, encoding="utf-8")
    bin_path.chmod(bin_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bin_path


def _write_fake_ssh(tmp: Path) -> Path:
    # `claim` proves a real SSH round-trip (`ssh <host> true`) before
    # returning; the integration test must inject a fake ssh or the probe
    # would try to reach a non-existent devspace over the network.
    bin_path = tmp / "fake-ssh"
    script = "#!/usr/bin/env bash\nexit 0\n"
    bin_path.write_text(script, encoding="utf-8")
    bin_path.chmod(bin_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bin_path


class TestHelperShipped(unittest.TestCase):
    def test_helper_exists(self) -> None:
        self.assertTrue(HELPER.is_file(), f"missing {HELPER}")

    def test_helper_syntactically_valid(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(HELPER)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class TestClaimAndFreeze(unittest.TestCase):
    def _run(self, cfg_dir: Path, source_root: Path, tmp: Path) -> subprocess.CompletedProcess[str]:
        env = {
            **os.environ,
            "PYTHON": sys.executable,
            "LOOP_ID": "loop-test1234",
            "FORGE_SOURCE_ROOT": str(source_root),
            "CCTL_BIN": str(_write_fake_cctl(tmp)),
            "TSH_BIN": str(_write_fake_tsh(tmp)),
            "SSH_BIN": str(_write_fake_ssh(tmp)),
            "HOME": str(tmp / "home"),
        }
        (tmp / "home").mkdir(exist_ok=True)
        return subprocess.run(
            [
                "bash",
                "-c",
                f'source "{HELPER}"; _devspace_claim_and_freeze "$1"',
                "_",
                str(cfg_dir),
            ],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_writable_devspace_dir_claims_rewrites_and_freezes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            cfg = tmp / "config"
            cfg.mkdir()
            (cfg / "remote.toml").write_text(_DEVSPACE_TOML, encoding="utf-8")
            source_root = tmp / "src"
            source_root.mkdir()
            try:
                result = self._run(cfg, source_root, tmp)
                self.assertEqual(result.returncode, 0, result.stderr)

                text = (cfg / "remote.toml").read_text(encoding="utf-8")
                self.assertIn(f'hostname = "ds-{_NEW_ID}"', text)

                registry = source_root / ".artifacts" / "lease" / "devspace" / "registry.json"
                self.assertTrue(registry.exists(), "registry must land under FORGE_SOURCE_ROOT")
                self.assertIn(f"ds-{_NEW_ID}", json.loads(registry.read_text(encoding="utf-8")))

                self.assertFalse(
                    os.access(cfg, os.W_OK),
                    "config dir must be frozen (read-only) after claim",
                )
            finally:
                cfg.chmod(0o755)

    def test_job_kind_claims_zero_gpu_devspace(self) -> None:
        # kind = "job" makes the devspace a 0-GPU filesystem gateway: the
        # lease claim must pass --gpu 0 even though gpu_count = 2 (which
        # stays in remote.toml for the per-suite cctl GPU job submitter to
        # read).
        job_toml = _DEVSPACE_TOML.replace('kind = "devspace"\n', 'kind = "job"\n')
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            cfg = tmp / "config"
            cfg.mkdir()
            (cfg / "remote.toml").write_text(job_toml, encoding="utf-8")
            source_root = tmp / "src"
            source_root.mkdir()
            try:
                result = self._run(cfg, source_root, tmp)
                self.assertEqual(result.returncode, 0, result.stderr)
                cctl_log = (tmp / "fake-cctl.log").read_text(encoding="utf-8")
                # The create call requested 0 GPUs, not 2.
                self.assertIn("--gpu 0", cctl_log)
                self.assertNotIn("--gpu 2", cctl_log)
                # gpu_count is left untouched in the frozen config.
                self.assertIn("gpu_count = 2", (cfg / "remote.toml").read_text(encoding="utf-8"))
            finally:
                cfg.chmod(0o755)

    def test_job_empty_workspace_fails_fast_before_claim(self) -> None:
        # kind="job" MUST pin a shared persistent remote-root
        # (/user/<username>): its ephemeral GPU job runs on a DIFFERENT pod
        # than the 0-GPU gateway and can only see code synced to that shared
        # root, so an empty workspace (→ pod-local /root, invisible to the
        # job) is rejected before any cctl devspace create.
        job_empty = _DEVSPACE_TOML.replace('kind = "devspace"\n', 'kind = "job"\n').replace(
            'workspace = "/user/liyuxuan"\n', 'workspace = ""\n'
        )
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            cfg = tmp / "config"
            cfg.mkdir()
            (cfg / "remote.toml").write_text(job_empty, encoding="utf-8")
            source_root = tmp / "src"
            source_root.mkdir()
            try:
                result = self._run(cfg, source_root, tmp)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("workspace", result.stderr)
                # No devspace was created (validation precedes the claim).
                self.assertFalse((tmp / "fake-cctl.log").exists())
            finally:
                cfg.chmod(0o755)

    def test_devspace_empty_workspace_is_allowed(self) -> None:
        # kind="devspace" does NOT need a shared remote-root: the held box
        # runs suites on its OWN filesystem, so an empty workspace is fine
        # (it defaults to the remote $HOME). The shared-root fail-fast is
        # job-only, so the claim must proceed and rewrite the hostname.
        empty_ws = _DEVSPACE_TOML.replace('workspace = "/user/liyuxuan"\n', 'workspace = ""\n')
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            cfg = tmp / "config"
            cfg.mkdir()
            (cfg / "remote.toml").write_text(empty_ws, encoding="utf-8")
            source_root = tmp / "src"
            source_root.mkdir()
            try:
                result = self._run(cfg, source_root, tmp)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(
                    f'hostname = "ds-{_NEW_ID}"',
                    (cfg / "remote.toml").read_text(encoding="utf-8"),
                )
                # The devspace WAS created (no shared-root gate for devspace).
                self.assertTrue((tmp / "fake-cctl.log").exists())
            finally:
                cfg.chmod(0o755)

    def test_frozen_dir_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            cfg = tmp / "config"
            cfg.mkdir()
            (cfg / "remote.toml").write_text(_DEVSPACE_TOML, encoding="utf-8")
            source_root = tmp / "src"
            source_root.mkdir()
            # Pre-freeze: simulate a resume of an already-launched loop.
            cfg.chmod(0o555)
            try:
                result = self._run(cfg, source_root, tmp)
                self.assertEqual(result.returncode, 0, result.stderr)
                # No devspace claimed: registry never written, cctl never called.
                registry = source_root / ".artifacts" / "lease" / "devspace" / "registry.json"
                self.assertFalse(
                    registry.exists(),
                    "a frozen (resume) dir must NOT claim a second devspace",
                )
                cctl_log = tmp / "fake-cctl.log"
                self.assertFalse(cctl_log.exists(), "cctl must not be invoked on resume")
            finally:
                cfg.chmod(0o755)

    def test_local_kind_freezes_without_claim(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            cfg = tmp / "config"
            cfg.mkdir()
            (cfg / "remote.toml").write_text(
                '[remote]\nkind = "local"\nhostname = ""\n', encoding="utf-8"
            )
            source_root = tmp / "src"
            source_root.mkdir()
            try:
                result = self._run(cfg, source_root, tmp)
                self.assertEqual(result.returncode, 0, result.stderr)
                cctl_log = tmp / "fake-cctl.log"
                self.assertFalse(cctl_log.exists(), "local kind must not invoke cctl")
                self.assertFalse(
                    os.access(cfg, os.W_OK),
                    "local-kind config dir must still be frozen",
                )
            finally:
                cfg.chmod(0o755)

    def test_ssh_kind_freezes_without_claim(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            cfg = tmp / "config"
            cfg.mkdir()
            (cfg / "remote.toml").write_text(
                '[remote]\nkind = "ssh"\nhostname = "static-host"\n', encoding="utf-8"
            )
            source_root = tmp / "src"
            source_root.mkdir()
            try:
                result = self._run(cfg, source_root, tmp)
                self.assertEqual(result.returncode, 0, result.stderr)
                cctl_log = tmp / "fake-cctl.log"
                self.assertFalse(cctl_log.exists(), "ssh kind must not invoke cctl")
                # hostname untouched for ssh.
                self.assertIn(
                    'hostname = "static-host"',
                    (cfg / "remote.toml").read_text(encoding="utf-8"),
                )
                self.assertFalse(os.access(cfg, os.W_OK))
            finally:
                cfg.chmod(0o755)


if __name__ == "__main__":
    unittest.main()
