"""``agent-loop.sh --config-dir <dir>`` stages the per-loop config dir.

Pins the bash-side half of Method F: the wrapper's pre-arg-scan must
capture a ``--config-dir <dir>`` pointing at a directory of axis TOMLs
and copy the seven known axes into
``$FORGE_TRAIN_DIR/$LOOP_ID/config/`` before the chmod -R a-w freeze,
so a user-authored config dir at any path (e.g. ``/tmp/myconfig/``) is
what the agent loop actually consumes.

These are structural tests against the bash source rather than
end-to-end runs: starting the wrapper would need a real harness env,
real tmux, and real agent CLI auth — none of which the test runner
has. The bash sketch is small and tightly contracted; reading it is
enough to catch the regressions that matter.
"""

from __future__ import annotations

import unittest
from pathlib import Path

_AGENT_LOOP_SH = Path(__file__).resolve().parents[3] / "harness" / "agent-loop.sh"


class TestAgentLoopConfigFlag(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = _AGENT_LOOP_SH.read_text(encoding="utf-8")

    def test_pre_scan_collects_config_dir(self) -> None:
        # The pre-arg-scan loop must recognise --config-dir <dir> and
        # stash it in CFG_DIR_OVERRIDE before exec.
        self.assertIn("CFG_DIR_OVERRIDE", self.text)
        self.assertIn('"--config-dir"', self.text)

    def test_per_loop_cfg_dir_matches_loop_layout_ssot(self) -> None:
        # The per-loop config dir is .artifacts/forge_train/<id>/config/
        # — sibling of workspace/, NOT inside it. Inside-workspace would
        # be wiped by re-provision; sibling survives. Bash can't import
        # harness.loop_layout without paying interpreter startup on every
        # launch, so the literal is kept native and pinned to the topology
        # SSOT by VALUE EQUIVALENCE here (same pattern as web/paths.py): if
        # loop_config_dir's formula ever changes, this assertion breaks.
        from harness.loop_layout import loop_config_dir

        expected = f'_cfg_dir="{loop_config_dir("$FORGE_TRAIN_DIR", "$LOOP_ID")}"'
        self.assertIn(expected, self.text)

    def test_config_dir_axes_are_copied_into_per_loop_dir(self) -> None:
        # Each of the seven known axes present in --config-dir must be
        # cp'd into the per-loop dir; the lease helper claims + freezes
        # it later in the shared post-bootstrap path.
        self.assertIn(
            'cp "$CFG_DIR_OVERRIDE/$_axis.toml" "$_cfg_dir/$_axis.toml"',
            self.text,
        )

    def test_axis_whitelist_is_seven_orthogonal_axes(self) -> None:
        # The axis list driving both the copy and the presence check must
        # match the seven orthogonal axes the config loader knows. Adding
        # an axis here without matching web/paths.py:CONFIG_AXES would
        # silently drop it.
        self.assertIn("ref data remote agent eval model optim", self.text)

    def test_missing_axis_fails_fast(self) -> None:
        # Half-staged per-loop dirs are the original R12 foot-gun —
        # the wrapper must exit non-zero, not silently launch with a
        # missing axis.
        self.assertIn("per-loop config missing:", self.text)

    def test_per_loop_dir_is_frozen_via_shared_lease_helper(self) -> None:
        # The claim + hostname rewrite + chmod -R a-w freeze now live in
        # tools/agent_loop_lease.sh so BOTH the CLI bootstrap and the web
        # launch run them exactly once (the web flow sets
        # FORGE_TRAIN_PROVISIONED=1 and skips the bootstrap block). The
        # wrapper must source and invoke it; the chmod itself is pinned in
        # test_agent_loop_lease.py.
        self.assertIn("tools/agent_loop_lease.sh", self.text)
        self.assertIn('_devspace_claim_and_freeze "$FORGE_CONFIG_DIR"', self.text)

    def test_forge_config_dir_is_exported_to_subprocess(self) -> None:
        # The redirect that makes the per-loop config dir actually
        # take effect: config_runtime._user_config_dir() reads this.
        self.assertIn('export FORGE_CONFIG_DIR="$_cfg_dir"', self.text)

    def test_lease_release_runs_on_exit(self) -> None:
        # EXIT trap chain must call lease release so a SIGKILL'd loop
        # leaves no orphaned cctl devspace.
        self.assertIn("tools.lease release devspace", self.text)

    def test_provision_only_exits_after_freeze_before_dev_loop(self) -> None:
        # The meta loop's harness_configs gate provisions the forge workspace
        # + renders gate config, then pre-runs each gate's ref side via
        # `bin/harness run` WITHOUT entering the dev loop. LOOP_PROVISION_ONLY=1
        # exits right after _devspace_claim_and_freeze (provision + render +
        # freeze done), BEFORE the config eval / backend auth / stage loop —
        # everything downstream is dev-loop-only.
        self.assertIn("LOOP_PROVISION_ONLY", self.text)
        freeze_at = self.text.index('_devspace_claim_and_freeze "$FORGE_CONFIG_DIR"')
        prov_at = self.text.index("LOOP_PROVISION_ONLY")
        # The dev-loop backend auth CALL (not the CLI-help mention earlier in
        # the file) is the first dev-loop-only step after the provision-only exit.
        backend_at = self.text.index('check-backend "$_check_cli"')
        self.assertLess(
            freeze_at, prov_at, "provision-only exit must come AFTER claim+freeze+render"
        )
        self.assertLess(
            prov_at, backend_at, "provision-only must exit BEFORE dev-loop backend auth"
        )


if __name__ == "__main__":
    unittest.main()
