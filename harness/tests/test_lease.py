"""Unit tests for ``harness.tools.lease`` — devspace lease registry.

The lease module is the SSOT that prevents two simultaneously-launching
loops from inheriting the same devspace hostname. The tests cover:

* Registry file/directory layout under ``$FORGE_REPO_ROOT/.artifacts/lease/``.
* ``cctl`` shell-out is mocked through a ``CCTL_BIN`` env-var override that
  points at a fake script — no real ``cctl`` invocation, no network.
* ``tsh`` discovery is mocked through a ``TSH_BIN`` override: ``claim``
  resolves the freshly-created devspace's real Teleport node name from
  ``tsh ls --format json`` (the create path has no base stanza to clone).
* ``fcntl.flock`` blocks concurrent ``claim`` calls so two loops never
  appear to claim the same derived host.
* ``release`` cleans the registry row, calls ``cctl devspace stop``,
  and is idempotent on a missing/already-released loop.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


_NEW_TASK_ID = "990099"
_TELEPORT_DOMAIN = "teleport.cybertron.modelbest.co"


def _write_fake_cctl(
    tmp: Path, *, exit_code: int = 0, status: str = "Running", new_id: str = _NEW_TASK_ID
) -> Path:
    """Write a stub ``cctl`` executable mimicking the real CLI surface.

    Real ``cctl devspace create`` assigns the new task a server-side id and
    returns the created Task object. The stub mirrors that:

    * ``devspace create ... -o json`` → a task object whose ``id`` is
      ``new_id`` (so the derived host becomes ``ds-<new_id>``).
    * ``devspace get tasks/<id> -o json`` → flat ``status`` so the poll
      terminates.
    * ``devspace stop --yes tasks/<id>`` → logged for the release assertion.
    """
    bin_path = tmp / "fake-cctl"
    log_path = tmp / "fake-cctl.log"
    script = (
        "#!/usr/bin/env bash\n"
        "set -e\n"
        f'echo "$@" >> {log_path}\n'
        'if [ "$1" = "devspace" ] && [ "$2" = "create" ]; then\n'
        f'  printf \'{{"id": {new_id}, "name": "tasks/{new_id}", "status": "{status}"}}\\n\'\n'
        'elif [ "$1" = "devspace" ] && [ "$2" = "get" ]; then\n'
        '  id="${3#tasks/}"\n'
        f'  printf \'{{"id": %s, "name": "tasks/%s", "status": "{status}"}}\\n\' "$id" "$id"\n'
        "fi\n"
        f"exit {exit_code}\n"
    )
    bin_path.write_text(script, encoding="utf-8")
    bin_path.chmod(bin_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bin_path


def _write_fake_tsh(
    tmp: Path, *, new_id: str = _NEW_TASK_ID, project: str = "loopharness", lists: bool = True
) -> Path:
    """Write a stub ``tsh`` whose ``ls --format json`` lists one node whose
    ``spec.hostname`` ends with ``-<new_id>`` (mirroring the real Teleport
    node naming ``devspace-<user>-<project>-<id>``).

    When ``lists=False`` the node is absent, so ``_teleport_hostname_for``
    must time out — the case where a freshly-created devspace has not yet
    registered into the Teleport tunnel.
    """
    bin_path = tmp / "fake-tsh"
    log_path = tmp / "fake-tsh.log"
    if lists:
        node = f"devspace-liyuxuan-{project}-{new_id}"
        payload = f'[{{"spec": {{"hostname": "{node}"}}}}]'
    else:
        payload = "[]"
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


def _write_fake_ssh(tmp: Path, *, fail_first: int = 0) -> Path:
    """Write a stub ``ssh`` for the post-claim reachability probe.

    ``claim`` proves a real SSH round-trip (``ssh <host> true``) before
    returning, so the tests must inject a fake ssh or the probe would hit
    the network. ``fail_first`` returns non-zero for the first N calls
    (simulating a tunnel that is still coming up) then succeeds — used to
    prove the probe RETRIES rather than giving up on the first miss.
    """
    bin_path = tmp / "fake-ssh"
    count_path = tmp / "fake-ssh.count"
    script = (
        "#!/usr/bin/env bash\n"
        f"n=$(cat {count_path} 2>/dev/null || echo 0)\n"
        f"echo $((n+1)) > {count_path}\n"
        f'if [ "$n" -lt "{fail_first}" ]; then\n'
        '  echo "no node reverse tunnel found ... agent is offline" >&2\n'
        "  exit 255\n"
        "fi\n"
        "exit 0\n"
    )
    bin_path.write_text(script, encoding="utf-8")
    bin_path.chmod(bin_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bin_path


class _LeaseTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        # _looks_like_repo_root() requires these markers.
        (self.repo / "pyproject.toml").write_text("", encoding="utf-8")
        (self.repo / "harness" / "config").mkdir(parents=True)
        (self.repo / "harness" / "config" / "defaults.toml").write_text("", encoding="utf-8")
        (self.repo / "config" / "eval").mkdir(parents=True)
        self.home = self.tmp / "home"
        self.home.mkdir()

        self._saved = {
            "FORGE_REPO_ROOT": os.environ.get("FORGE_REPO_ROOT"),
            "FORGE_SOURCE_ROOT": os.environ.get("FORGE_SOURCE_ROOT"),
            "HOME": os.environ.get("HOME"),
            "CCTL_BIN": os.environ.get("CCTL_BIN"),
            "TSH_BIN": os.environ.get("TSH_BIN"),
            "SSH_BIN": os.environ.get("SSH_BIN"),
        }
        os.environ["FORGE_REPO_ROOT"] = str(self.repo)
        # Clear any ambient FORGE_SOURCE_ROOT (set when the suite runs
        # inside a provisioned workspace) so the registry root falls back
        # to FORGE_REPO_ROOT for the default-path tests. The dedicated
        # source-root test sets it explicitly.
        os.environ.pop("FORGE_SOURCE_ROOT", None)
        os.environ["HOME"] = str(self.home)

        # Late import — needs FORGE_REPO_ROOT set so repo_root() resolves.
        from harness import config_runtime

        config_runtime._resolve_repo_root.cache_clear()

        from tools import lease

        self.lease = lease

    def tearDown(self) -> None:
        for key, val in self._saved.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
        self._tmp.cleanup()

        from harness import config_runtime

        config_runtime._resolve_repo_root.cache_clear()

    def _spec(self, **over):
        params = dict(
            project="loopharness",
            cluster="paratera_shandong",
            resource_pool="faxin",
            image="infra/autoloop-harness:train-engine",
            gpu_count=2,
            gpu_model="h100",
            priority="HIGH",
        )
        params.update(over)
        return self.lease.DevspaceSpec(**params)

    def _install_fakes(
        self, *, new_id: str = _NEW_TASK_ID, lists: bool = True, ssh_fail_first: int = 0
    ) -> None:
        os.environ["CCTL_BIN"] = str(_write_fake_cctl(self.tmp, new_id=new_id))
        os.environ["TSH_BIN"] = str(_write_fake_tsh(self.tmp, new_id=new_id, lists=lists))
        os.environ["SSH_BIN"] = str(_write_fake_ssh(self.tmp, fail_first=ssh_fail_first))


class TestLeaseClaim(_LeaseTestBase):
    def test_claim_creates_registry_and_uses_server_assigned_host(self) -> None:
        self._install_fakes(new_id="990099")
        host = self.lease.claim("devspace", spec=self._spec(), loop_id="loop-aaaaaaaa", timeout=10)
        # Derived host is ds-<server-assigned id>, not a loop-id hash.
        self.assertEqual(host, "ds-990099")

        registry_path = self.repo / ".artifacts" / "lease" / "devspace" / "registry.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        self.assertIn(host, registry)
        self.assertEqual(registry[host]["loop_id"], "loop-aaaaaaaa")
        self.assertEqual(registry[host]["task_id"], "990099")
        self.assertIn("claimed_at", registry[host])

        # Reverse-lookup file exists under same directory.
        rev = self.repo / ".artifacts" / "lease" / "devspace" / "loop-aaaaaaaa.host"
        self.assertTrue(rev.exists())
        self.assertEqual(rev.read_text(encoding="utf-8").strip(), host)

    def test_claim_issues_cctl_create_with_spec_flags(self) -> None:
        self._install_fakes(new_id="990099")
        self.lease.claim("devspace", spec=self._spec(), loop_id="loop-flagsxxx", timeout=10)
        log = (self.tmp / "fake-cctl.log").read_text(encoding="utf-8")
        self.assertIn("devspace create", log)
        for fragment in (
            "--project loopharness",
            "--cluster paratera_shandong",
            "--resource-pool faxin",
            "--image infra/autoloop-harness:train-engine",
            "--gpu 2",
            "--gpu-model h100",
            "--priority HIGH",
        ):
            self.assertIn(fragment, log, f"create must forward {fragment!r}")

    def test_claim_synthesizes_ssh_stanza_from_teleport(self) -> None:
        self._install_fakes(new_id="990099")
        host = self.lease.claim("devspace", spec=self._spec(), loop_id="loop-cccccccc", timeout=10)
        ssh_cfg = self.home / ".ssh" / "config"
        text = ssh_cfg.read_text(encoding="utf-8")
        # The synthesized stanza carries the Teleport FQDN discovered via
        # `tsh ls` plus the fixed Teleport ProxyCommand — a bare
        # `HostName <host>` would never resolve through the proxy.
        self.assertIn(f"Host {host}", text)
        self.assertIn(
            f"HostName devspace-liyuxuan-loopharness-990099.{_TELEPORT_DOMAIN}",
            text,
        )
        derived_block = text.split(f"Host {host}", 1)[1]
        self.assertIn("ProxyCommand", derived_block)
        self.assertIn("User root", derived_block)
        # Connection keepalive so a black-holed devspace (half-open TCP, no
        # FIN/RST) makes ssh fail fast instead of hanging the loop forever.
        self.assertIn("ConnectTimeout 15", derived_block)
        self.assertIn("ServerAliveInterval 30", derived_block)
        self.assertIn("ServerAliveCountMax 4", derived_block)
        # Teleport SSH cert auth: without Port 3022 + the per-user
        # IdentityFile/CertificateFile the remote rejects with
        # "Permission denied (publickey)". Username comes from `tsh status`.
        self.assertIn("Port 3022", derived_block)
        self.assertIn(f"keys/{_TELEPORT_DOMAIN}/liyuxuan", derived_block.replace("\\", "/"))
        self.assertIn("CertificateFile", derived_block)
        self.assertIn("liyuxuan-ssh", derived_block)

        # Idempotent: re-synthesizing the same derived host is a no-op.
        self.lease._write_synthesized_stanza(host, "devspace-liyuxuan-loopharness-990099")
        self.assertEqual(
            text.count(f"Host {host}"),
            ssh_cfg.read_text(encoding="utf-8").count(f"Host {host}"),
            "Repeated SSH stanza synthesis must be a no-op",
        )

    def test_claim_is_idempotent_on_retry(self) -> None:
        self._install_fakes(new_id="990099")
        host1 = self.lease.claim("devspace", spec=self._spec(), loop_id="loop-retryxx", timeout=10)
        host2 = self.lease.claim("devspace", spec=self._spec(), loop_id="loop-retryxx", timeout=10)
        self.assertEqual(host1, host2)
        log = (self.tmp / "fake-cctl.log").read_text(encoding="utf-8")
        self.assertEqual(
            log.count("devspace create"), 1, "Retried claim must not create a second devspace"
        )

    def test_claim_propagates_cctl_failure(self) -> None:
        os.environ["CCTL_BIN"] = str(_write_fake_cctl(self.tmp, exit_code=1))
        os.environ["TSH_BIN"] = str(_write_fake_tsh(self.tmp))
        with self.assertRaises(self.lease.LeaseError):
            self.lease.claim("devspace", spec=self._spec(), loop_id="loop-dddddddd", timeout=10)

    def test_claim_times_out_when_teleport_never_lists_host(self) -> None:
        # cctl create + poll succeed, but the node never appears in the
        # Teleport tunnel — claim must surface a timeout rather than write a
        # broken (unresolvable) SSH stanza. Use a small POSITIVE timeout:
        # `timeout <= 0` means "wait indefinitely" in the poll helpers, so a
        # never-listing host would hang the test forever. Drive the poll
        # interval to 0 so the bounded wait resolves without real sleeps.
        self._install_fakes(new_id="990099", lists=False)
        saved = self.lease._POLL_INTERVAL_S
        self.lease._POLL_INTERVAL_S = 0
        try:
            with self.assertRaises(self.lease.LeaseTimeoutError):
                self.lease.claim("devspace", spec=self._spec(), loop_id="loop-notunnel", timeout=1)
        finally:
            self.lease._POLL_INTERVAL_S = saved

    def test_claim_probes_ssh_before_returning(self) -> None:
        # The node NAME being in `tsh ls` does not mean its reverse tunnel
        # carries traffic; claim must prove a real SSH round-trip so the
        # loop's first `sync push` cannot race an unready tunnel.
        self._install_fakes(new_id="990099")
        self.lease.claim("devspace", spec=self._spec(), loop_id="loop-probe", timeout=10)
        count = (self.tmp / "fake-ssh.count").read_text(encoding="utf-8").strip()
        self.assertGreaterEqual(int(count), 1, "claim must issue an SSH reachability probe")

    def test_claim_retries_ssh_probe_until_tunnel_ready(self) -> None:
        # First two probes fail with the exact "no node reverse tunnel found"
        # error that bricked loops d5bb90c24853 / 61f1904a70bb; the third
        # succeeds. claim must RETRY, not abort on the first miss.
        self._install_fakes(new_id="990099", ssh_fail_first=2)
        saved = self.lease._SSH_PROBE_INTERVAL_S
        self.lease._SSH_PROBE_INTERVAL_S = 0  # no real sleeps in the test
        try:
            host = self.lease.claim("devspace", spec=self._spec(), loop_id="loop-race", timeout=30)
        finally:
            self.lease._SSH_PROBE_INTERVAL_S = saved
        self.assertEqual(host, "ds-990099")
        count = int((self.tmp / "fake-ssh.count").read_text(encoding="utf-8").strip())
        self.assertEqual(count, 3, "probe must retry past transient tunnel failures")
        # A host that survived the probe is registered as reachable.
        registry_path = self.repo / ".artifacts" / "lease" / "devspace" / "registry.json"
        self.assertIn(host, json.loads(registry_path.read_text(encoding="utf-8")))

    def test_claim_times_out_when_ssh_never_reachable(self) -> None:
        # tsh lists the node but the tunnel never carries traffic — claim must
        # surface a timeout instead of returning an unreachable host that the
        # first sync push would then fail on.
        self._install_fakes(new_id="990099", ssh_fail_first=10_000)
        saved = self.lease._SSH_PROBE_INTERVAL_S
        self.lease._SSH_PROBE_INTERVAL_S = 0
        try:
            with self.assertRaises(self.lease.LeaseTimeoutError):
                self.lease.claim("devspace", spec=self._spec(), loop_id="loop-noreach", timeout=1)
        finally:
            self.lease._SSH_PROBE_INTERVAL_S = saved


class TestLeaseRelease(_LeaseTestBase):
    def test_release_removes_registry_row_and_stops_copy(self) -> None:
        self._install_fakes(new_id="990099")
        host = self.lease.claim("devspace", spec=self._spec(), loop_id="loop-eeeeeeee", timeout=10)
        self.lease.release("devspace", loop_id="loop-eeeeeeee")

        registry_path = self.repo / ".artifacts" / "lease" / "devspace" / "registry.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        self.assertNotIn(host, registry)

        rev = self.repo / ".artifacts" / "lease" / "devspace" / "loop-eeeeeeee.host"
        self.assertFalse(rev.exists())

        log = (self.tmp / "fake-cctl.log").read_text(encoding="utf-8")
        self.assertIn("devspace stop --yes tasks/990099", log)

    def test_release_is_idempotent_for_unknown_loop(self) -> None:
        self._install_fakes()
        # No prior claim — release must not raise.
        self.lease.release("devspace", loop_id="loop-doesnotexist")


class TestLeaseRebind(_LeaseTestBase):
    def test_rebind_points_release_at_recovered_host(self) -> None:
        """A devspace that drops mid-run is re-created by the dev agent;
        ``rebind`` re-points this loop's lease at the new host so the
        wrapper's EXIT-trap ``release`` stops the live machine, not the dead
        original.
        """
        self._install_fakes(new_id="990099")
        old_host = self.lease.claim(
            "devspace", spec=self._spec(), loop_id="loop-rebindxx", timeout=10
        )
        self.assertEqual(old_host, "ds-990099")

        new_host = "ds-771122"
        self.lease.rebind("devspace", loop_id="loop-rebindxx", host=new_host)

        registry_path = self.repo / ".artifacts" / "lease" / "devspace" / "registry.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        # Old row is gone; new row carries the recovered host + task id.
        self.assertNotIn(old_host, registry)
        self.assertIn(new_host, registry)
        self.assertEqual(registry[new_host]["loop_id"], "loop-rebindxx")
        self.assertEqual(registry[new_host]["task_id"], "771122")

        rev = self.repo / ".artifacts" / "lease" / "devspace" / "loop-rebindxx.host"
        self.assertEqual(rev.read_text(encoding="utf-8").strip(), new_host)

        # rebind must NOT stop the old machine — recovery is triggered
        # precisely because it already dropped.
        log = (self.tmp / "fake-cctl.log").read_text(encoding="utf-8")
        self.assertNotIn("devspace stop", log)

        # release now targets the recovered host.
        self.lease.release("devspace", loop_id="loop-rebindxx")
        log = (self.tmp / "fake-cctl.log").read_text(encoding="utf-8")
        self.assertIn("devspace stop --yes tasks/771122", log)
        self.assertNotIn("devspace stop --yes tasks/990099", log)

    def test_rebind_without_prior_claim_creates_row(self) -> None:
        """Recovery may fire before any registry row exists (e.g. claim
        raced a crash). rebind is the authoritative writer either way.
        """
        self._install_fakes()
        self.lease.rebind("devspace", loop_id="loop-norow", host="ds-555000")
        rev = self.repo / ".artifacts" / "lease" / "devspace" / "loop-norow.host"
        self.assertEqual(rev.read_text(encoding="utf-8").strip(), "ds-555000")


class TestLeaseRegistryRoot(_LeaseTestBase):
    """The lease registry is SHARED across all loops and must resolve to
    the SAME path regardless of which per-loop workspace cwd ``claim`` /
    ``release`` / ``rebind`` happen to run under.

    ``claim`` runs during launch with ``FORGE_REPO_ROOT`` pointing at one
    place; the EXIT-trap ``release`` and the dev-agent ``rebind`` run later
    with a different ``FORGE_REPO_ROOT`` (the per-loop workspace). If the
    registry followed ``FORGE_REPO_ROOT`` they would each touch a
    different file — ``release`` would never find what ``claim`` wrote and
    the devspace would leak (burning GPU quota). Anchoring the registry to
    the shared ``FORGE_SOURCE_ROOT`` keeps all three in agreement and
    makes anti-collision global rather than per-workspace.
    """

    def test_registry_anchored_to_source_root_not_cwd_repo(self) -> None:
        source_root = self.tmp / "source"
        source_root.mkdir()
        os.environ["FORGE_SOURCE_ROOT"] = str(source_root)
        self._install_fakes(new_id="990099")
        host = self.lease.claim("devspace", spec=self._spec(), loop_id="loop-srcroot", timeout=10)

        # Registry lands under FORGE_SOURCE_ROOT ...
        src_registry = source_root / ".artifacts" / "lease" / "devspace" / "registry.json"
        self.assertTrue(src_registry.exists(), "registry must live under FORGE_SOURCE_ROOT")
        self.assertIn(host, json.loads(src_registry.read_text(encoding="utf-8")))
        # ... and NOT under the per-loop FORGE_REPO_ROOT.
        repo_registry = self.repo / ".artifacts" / "lease" / "devspace" / "registry.json"
        self.assertFalse(
            repo_registry.exists(),
            "registry must NOT follow the per-loop FORGE_REPO_ROOT",
        )

    def test_release_finds_lease_under_different_repo_root(self) -> None:
        source_root = self.tmp / "source"
        source_root.mkdir()
        os.environ["FORGE_SOURCE_ROOT"] = str(source_root)
        self._install_fakes(new_id="990099")
        host = self.lease.claim("devspace", spec=self._spec(), loop_id="loop-releasex", timeout=10)

        # Simulate release running from a DIFFERENT cwd / repo root (the
        # per-loop workspace), as agent-loop.sh's EXIT trap does.
        other_workspace = self.tmp / "other-workspace"
        other_workspace.mkdir()
        os.environ["FORGE_REPO_ROOT"] = str(other_workspace)
        from harness import config_runtime

        config_runtime._resolve_repo_root.cache_clear()

        self.lease.release("devspace", loop_id="loop-releasex")

        src_registry = source_root / ".artifacts" / "lease" / "devspace" / "registry.json"
        self.assertNotIn(
            host,
            json.loads(src_registry.read_text(encoding="utf-8")),
            "release run from a different repo root must still drop the row",
        )
        log = (self.tmp / "fake-cctl.log").read_text(encoding="utf-8")
        self.assertIn("devspace stop --yes tasks/990099", log)


if __name__ == "__main__":
    unittest.main()
