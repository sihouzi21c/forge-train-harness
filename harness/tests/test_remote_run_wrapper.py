"""Tests for ``harness/tools/remote_run.sh``.

The wrapper is the single client-side site authorised to layer an outer
``timeout`` on the ssh-side ``harness run`` invocation. These tests
exercise its ``--dry-run`` mode so we can verify the budget arithmetic
(SSOT, override priority, layered buffers) without an actual ssh hop.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "remote_run.sh"

# Mirrors ``app._TRANSPORT_BUDGET_BUFFER_S``.
_TRANSPORT_BUFFER = 60
# Local ssh-side teardown allowance (see remote_run.sh comments).
_SSH_TEARDOWN = 120


def _run_wrapper_with_stub_harness(
    args: list[str],
    *,
    stub_budget: int,
    stub_rc: int = 0,
) -> subprocess.CompletedProcess[str]:
    """Invoke ``remote_run.sh --dry-run`` with a stub ``harness`` on PATH
    that always returns *stub_budget* (or fails with *stub_rc*)."""
    with tempfile.TemporaryDirectory() as tmp:
        stub = Path(tmp) / "harness"
        stub.write_text(
            textwrap.dedent(
                f"""\
                #!/usr/bin/env bash
                # Stub harness for remote_run.sh tests. Only the ``budget``
                # subcommand is honoured; anything else triggers a loud failure
                # so a wrapper bug that bypasses the SSOT query is immediately
                # visible.
                if [[ "$1" != "budget" ]]; then
                  echo "stub harness: unexpected subcommand $1" >&2
                  exit 99
                fi
                exit_rc={stub_rc}
                if [[ "$exit_rc" != "0" ]]; then
                  echo "stub harness: simulated failure" >&2
                  exit "$exit_rc"
                fi
                echo {stub_budget}
                """
            )
        )
        stub.chmod(0o755)
        # Stub remote.toml — remote_run.sh derives host + workdir from
        # this SSOT (same source as harness/remote_sync.py). No env-var
        # override path exists; FORGE_CONFIG_DIR + LOOP_ID is the only
        # way the wrapper learns where to ssh.
        (Path(tmp) / "remote.toml").write_text(
            '[remote]\nkind = "ssh"\nhostname = "stub-host"\nworkspace = "/tmp/stub-workspace"\n'
        )
        env = os.environ.copy()
        env["PATH"] = f"{tmp}:{env['PATH']}"
        env["FORGE_CONFIG_DIR"] = tmp
        env["LOOP_ID"] = "stub-loop"
        return subprocess.run(
            ["bash", str(SCRIPT), "--dry-run", *args],
            env=env,
            capture_output=True,
            text=True,
        )


def _parse_dry_run(stdout: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in stdout.strip().split("\n") if "=" in line)


@unittest.skipUnless(shutil.which("bash"), "bash required")
class TestRemoteRunBudgetArithmetic(unittest.TestCase):
    def test_default_uses_ssot_budget_plus_buffers(self) -> None:
        completed = _run_wrapper_with_stub_harness(["op-status"], stub_budget=600)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        fields = _parse_dry_run(completed.stdout)
        self.assertEqual(fields["budget"], "600")
        self.assertEqual(fields["override"], "")
        self.assertEqual(fields["effective"], "600")
        self.assertEqual(
            int(fields["outer"]),
            600 + _TRANSPORT_BUFFER + _SSH_TEARDOWN,
        )
        self.assertEqual(fields["suite"], "op-status")

    def test_override_long_form_wins(self) -> None:
        completed = _run_wrapper_with_stub_harness(
            ["op-status", "--timeout", "12"],
            stub_budget=600,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        fields = _parse_dry_run(completed.stdout)
        self.assertEqual(fields["effective"], "12")
        self.assertEqual(
            int(fields["outer"]),
            12 + _TRANSPORT_BUFFER + _SSH_TEARDOWN,
        )
        # The flag is preserved in the extras so the remote harness
        # sees ``--timeout 12`` and stamps the result.json provenance.
        self.assertIn("--timeout", fields["extra"])
        self.assertIn("12", fields["extra"])

    def test_override_equals_form_wins(self) -> None:
        completed = _run_wrapper_with_stub_harness(
            ["op-status", "--timeout=20"],
            stub_budget=600,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        fields = _parse_dry_run(completed.stdout)
        self.assertEqual(fields["effective"], "20")

    def test_extra_args_passthrough(self) -> None:
        completed = _run_wrapper_with_stub_harness(
            ["op-long", "attention", "gemm_fc1"],
            stub_budget=7200,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        fields = _parse_dry_run(completed.stdout)
        self.assertEqual(fields["suite"], "op-long")
        self.assertIn("attention", fields["extra"])
        self.assertIn("gemm_fc1", fields["extra"])

    def test_harness_budget_failure_aborts(self) -> None:
        # If ``harness budget`` cannot answer, the wrapper MUST abort
        # rather than fall back to a guessed default — that fallback is
        # exactly the failure mode the SSOT redesign exists to prevent.
        completed = _run_wrapper_with_stub_harness(
            ["bogus-suite"],
            stub_budget=0,
            stub_rc=1,
        )
        self.assertNotEqual(completed.returncode, 0)


@unittest.skipUnless(shutil.which("bash"), "bash required")
class TestRemoteRunCompileCache(unittest.TestCase):
    """The wrapper must export persistent Triton/Inductor cache dirs on
    every ssh hop. The 3-min compile/launch tax on the first
    ``long-train`` run was the dominant per-kernel iteration cost in
    loop e180edc7 Round 5 (~10 min wall, ~3 min compile). Persisting
    the JIT cache across ssh sessions amortises that tax to a one-time
    cold start per devspace.
    """

    def test_remote_cmd_exports_persistent_compile_caches(self) -> None:
        completed = _run_wrapper_with_stub_harness(["long-train"], stub_budget=3600)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        fields = _parse_dry_run(completed.stdout)
        remote_cmd = fields["remote_cmd"]
        self.assertIn(
            'TRITON_CACHE_DIR="$HOME/.cache/triton-persistent"',
            remote_cmd,
        )
        self.assertIn(
            'TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/inductor-persistent"',
            remote_cmd,
        )
        # mkdir -p MUST precede the export so the first invocation on a
        # fresh devspace doesn't error out with a missing-dir failure
        # when Triton/Inductor try to write the cache.
        self.assertIn(
            'mkdir -p "$HOME/.cache/triton-persistent"',
            remote_cmd,
        )
        self.assertIn(
            "mkdir -p",
            remote_cmd.split("export")[0],
            "mkdir must come before any export to avoid first-write failure",
        )

    def test_remote_cmd_pins_workspace_shim_path(self) -> None:
        # The bare ``harness`` on the remote side is the per-workspace shim
        # at <workdir>/bin/harness; a non-interactive ssh shell does not
        # reliably provide it on PATH, so remote_cmd must pin it itself —
        # AFTER the cd (so $PWD is the workdir) and BEFORE the exec.
        completed = _run_wrapper_with_stub_harness(["long-train"], stub_budget=3600)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        remote_cmd = _parse_dry_run(completed.stdout)["remote_cmd"]
        pin = 'export PATH="$PWD/bin:$PATH"'
        self.assertIn(pin, remote_cmd)
        self.assertLess(remote_cmd.index("cd "), remote_cmd.index(pin))
        self.assertLess(remote_cmd.index(pin), remote_cmd.index("setsid harness run"))

    def test_remote_cmd_forwards_loop_telemetry_env(self) -> None:
        # LOOP_ID / FORGE_TRAIN_DIR must reach the remote runner or
        # tools/mfu_record.py silently skips the mfu_history.jsonl append —
        # the remote checkout is named <loop_id>, not "workspace", so the
        # recorder's layout fallback never fires there. Loop 9f03ffd324af
        # deadlocked in long-horizon on exactly this: five passing full
        # long-train runs, zero telemetry rows, review could never issue
        # MILESTONE_OVERRIDE.
        completed = _run_wrapper_with_stub_harness(["long-train"], stub_budget=3600)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        remote_cmd = _parse_dry_run(completed.stdout)["remote_cmd"]
        self.assertIn("export LOOP_ID=stub-loop", remote_cmd)
        self.assertIn(
            "FORGE_TRAIN_DIR=/tmp/stub-workspace/.forge_train/stub-loop"
            "/.artifacts/forge_train",
            remote_cmd,
        )
        self.assertLess(
            remote_cmd.index("LOOP_ID="),
            remote_cmd.index("setsid harness run"),
        )


def _run_wrapper_job(args: list[str], *, stub_budget: int) -> subprocess.CompletedProcess[str]:
    """Invoke ``remote_run.sh --dry-run`` with a ``kind = "job"``
    remote.toml so the wrapper delegates to ``tools/gpu_job.py``. The
    stub ``harness`` answers ``budget`` (the SSOT query) as before."""
    with tempfile.TemporaryDirectory() as tmp:
        stub = Path(tmp) / "harness"
        stub.write_text(
            textwrap.dedent(
                f"""\
                #!/usr/bin/env bash
                if [[ "$1" != "budget" ]]; then
                  echo "stub harness: unexpected subcommand $1" >&2
                  exit 99
                fi
                echo {stub_budget}
                """
            )
        )
        stub.chmod(0o755)
        (Path(tmp) / "remote.toml").write_text(
            "[remote]\n"
            'kind = "job"\n'
            'project = "loopharness"\n'
            'cluster = "paratera_shandong"\n'
            'resource_pool = "faxin"\n'
            'image = "infra/forge-train:tag"\n'
            "gpu_count = 2\n"
            'gpu_model = "h100"\n'
            'priority = "NORMAL"\n'
            'billing_account_id = ""\n'
            'workspace = "/user/lishangzhan"\n'
            'hostname = "ds-555"\n'
        )
        env = os.environ.copy()
        env["PATH"] = f"{tmp}:{env['PATH']}"
        env["FORGE_CONFIG_DIR"] = tmp
        env["LOOP_ID"] = "L1"
        return subprocess.run(
            ["bash", str(SCRIPT), "--dry-run", *args],
            env=env,
            capture_output=True,
            text=True,
        )


@unittest.skipUnless(shutil.which("bash"), "bash required")
class TestRemoteRunJobRouting(unittest.TestCase):
    """In kind=job the wrapper must hand the suite to tools/gpu_job.py (a
    cctl GPU job), passing the SSOT-derived budget as the outer window —
    never ssh-run it on the GPU-less gateway."""

    def test_job_delegates_to_gpu_job_with_ssot_budget(self) -> None:
        completed = _run_wrapper_job(["forward-align"], stub_budget=600)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        fields = _parse_dry_run(completed.stdout)
        # The output is gpu_job's dry-run dump, not the ssh remote_cmd.
        self.assertEqual(fields["mode"], "job")
        self.assertEqual(fields["gpu"], "2")
        self.assertEqual(fields["suite"], "forward-align")
        # SSOT budget flows through: outer = budget + 60 + 120.
        self.assertEqual(int(fields["outer"]), 600 + _TRANSPORT_BUFFER + _SSH_TEARDOWN)
        self.assertEqual(fields["budget"], "600")
        self.assertIn("pytorchjob create", fields["create_cmd"])
        self.assertIn("--nodes 1", fields["create_cmd"])
        self.assertIn("--gpu 2", fields["create_cmd"])

    def test_non_job_kind_uses_ssh_path(self) -> None:
        # Sanity: a non-job kind (ssh here) prints a remote_cmd, not a
        # gpu_job dump.
        completed = _run_wrapper_with_stub_harness(["op-status"], stub_budget=600)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        fields = _parse_dry_run(completed.stdout)
        self.assertEqual(fields["kind"], "ssh")
        self.assertIn("remote_cmd", fields)


@unittest.skipUnless(shutil.which("bash"), "bash required")
class TestRemoteRunUsage(unittest.TestCase):
    def test_missing_suite_returns_usage(self) -> None:
        # Usage check fires before any env-var lookup, so no
        # FORGE_CONFIG_DIR / LOOP_ID is required to reach it.
        completed = subprocess.run(
            ["bash", str(SCRIPT)],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("Usage:", completed.stderr)


if __name__ == "__main__":
    unittest.main()
