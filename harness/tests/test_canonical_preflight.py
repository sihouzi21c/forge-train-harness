"""Unit tests for the canonical-state preflight.

Two units are covered:

* ``tools.bootstrap_canonical.canonical_path`` — the single source of
  truth for the ``<checkpoint_root>/{ones,no1}/canonical_state_fp32.pt``
  layout that both the CLI ``main`` and the loop preflight resolve
  through.
* ``tools.bootstrap_canonical.canonical_is_current`` — the fingerprint
  probe: a canonical is trusted only when its sidecar matches the hash of
  the CURRENT rendered ref product, so a canonical left behind by a loop
  with a different model axis is regenerated instead of poisoning the
  bitwise gates.
* ``evals.canonical_preflight`` — the idempotent, torch-only orchestration
  the agent-loop wrapper runs before the first dev round: skip when both
  canonicals are fingerprint-current, skip for non-torch backends, otherwise
  shell out to ``bootstrap_canonical.py`` once per missing/stale scheme with
  the ref-run env assembled, and fail-fast when a bootstrap subprocess
  errors.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
import unittest.mock as mock
from pathlib import Path

# ``harness/`` is the PYTHONPATH import root (tools/, evals/ live under it).
IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPORT_ROOT))


def _render_fake_ref_products(repo: Path) -> Path:
    """Minimal rendered ref/config: one gate per forge_init_ones scheme."""
    cfg = repo / "ref" / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "long-train.toml").write_text("[cli]\nforge_init_ones = 1\n")
    (cfg / "multistep.toml").write_text("[cli]\nforge_init_ones = 0\n")
    return cfg


def _stamp_current(ckpt: Path, ref_config_dir: Path, init_ones: int, ref_script: str = "") -> Path:
    """Create a canonical + a fingerprint matching the current products."""
    import json

    from tools import bootstrap_canonical

    subdir = "ones" if init_ones == 1 else "no1"
    target = ckpt / subdir / "canonical_state_fp32.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"x")
    bootstrap_canonical.fingerprint_path(target).write_text(
        json.dumps(
            bootstrap_canonical.expected_fingerprint(
                ref_config_dir, init_ones, ref_script=ref_script
            )
        )
    )
    return target


class TestCanonicalPath(unittest.TestCase):
    """Lock the {ones,no1}/canonical_state_fp32.pt path SSOT."""

    def _workload(self, ckpt: str) -> dict:
        return {"ref": {"backend": "torch", "checkpoint_root": ckpt}}

    def test_ones_subdir(self) -> None:
        from tools import bootstrap_canonical

        p = bootstrap_canonical.canonical_path(self._workload("/ckpt"), 1)
        self.assertEqual(p, Path("/ckpt/ones/canonical_state_fp32.pt"))

    def test_no1_subdir(self) -> None:
        from tools import bootstrap_canonical

        p = bootstrap_canonical.canonical_path(self._workload("/ckpt"), 0)
        self.assertEqual(p, Path("/ckpt/no1/canonical_state_fp32.pt"))

    def test_output_root_override_wins(self) -> None:
        from tools import bootstrap_canonical

        p = bootstrap_canonical.canonical_path(self._workload("/ckpt"), 1, output_root="/other")
        self.assertEqual(p, Path("/other/ones/canonical_state_fp32.pt"))

    def test_empty_checkpoint_root_fails_fast(self) -> None:
        from tools import bootstrap_canonical

        with self.assertRaises(SystemExit):
            bootstrap_canonical.canonical_path({"ref": {}}, 1)


class TestMissingInitOnes(unittest.TestCase):
    """Idempotency probe: only absent schemes are reported."""

    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt = Path(self._tmp.name)
        self.workload = {"ref": {"backend": "torch", "checkpoint_root": str(self.ckpt)}}

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _touch(self, subdir: str) -> None:
        target = self.ckpt / subdir / "canonical_state_fp32.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")

    def test_both_missing(self) -> None:
        from evals import canonical_preflight

        self.assertEqual(canonical_preflight.missing_init_ones(self.workload), [1, 0])

    def test_one_present(self) -> None:
        from evals import canonical_preflight

        self._touch("ones")
        self.assertEqual(canonical_preflight.missing_init_ones(self.workload), [0])

    def test_both_present(self) -> None:
        from evals import canonical_preflight

        self._touch("ones")
        self._touch("no1")
        self.assertEqual(canonical_preflight.missing_init_ones(self.workload), [])


class TestCanonicalIsCurrent(unittest.TestCase):
    """Fingerprint probe: trust a canonical only if its sidecar matches the
    current rendered ref product (stale/foreign canonicals must regenerate)."""

    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)
        self.cfg = _render_fake_ref_products(self.repo)
        self.ckpt = self.repo / "ckpt"
        self.workload = {"ref": {"backend": "torch", "checkpoint_root": str(self.ckpt)}}

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _is_current(self, init_ones: int) -> bool:
        from tools import bootstrap_canonical

        return bootstrap_canonical.canonical_is_current(
            self.workload, init_ones, ref_config_dir=self.cfg
        )

    def test_absent_canonical_is_not_current(self) -> None:
        self.assertFalse(self._is_current(0))

    def test_matching_fingerprint_is_current(self) -> None:
        _stamp_current(self.ckpt, self.cfg, 0)
        self.assertTrue(self._is_current(0))

    def test_canonical_without_sidecar_is_stale(self) -> None:
        # Pre-fingerprint deployment / hand-placed file: unverifiable → stale.
        target = self.ckpt / "no1" / "canonical_state_fp32.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        self.assertFalse(self._is_current(0))

    def test_rerendered_product_invalidates_fingerprint(self) -> None:
        # The wrong-structure-canonical failure mode: the canonical was dumped
        # from an earlier render (different model axis); the product changed.
        _stamp_current(self.ckpt, self.cfg, 0)
        (self.cfg / "multistep.toml").write_text("[cli]\nforge_init_ones = 0\nhidden_size = 4096\n")
        self.assertFalse(self._is_current(0))

    def test_ref_script_change_invalidates_fingerprint(self) -> None:
        # A canonical dumped under a different [ref].ref_script (the launcher
        # picks the model implementation) must be stale even when the rendered
        # product bytes happen to match — the MiniCPM→Qwen3 poisoning case.
        _stamp_current(self.ckpt, self.cfg, 0, ref_script="run_pure_mup_mtp.sh")
        self.workload["ref"]["ref_script"] = "run_qwen3_dense.sh"
        self.assertFalse(self._is_current(0))
        self.workload["ref"]["ref_script"] = "run_pure_mup_mtp.sh"
        self.assertTrue(self._is_current(0))

    def test_missing_init_ones_reports_stale_scheme(self) -> None:
        from evals import canonical_preflight

        _stamp_current(self.ckpt, self.cfg, 1)  # ones current; no1 absent
        self.assertEqual(
            canonical_preflight.missing_init_ones(self.workload, ref_config_dir=self.cfg),
            [0],
        )


class TestResolveCanonicalGate(unittest.TestCase):
    """FORGE_GATE selection: match forge_init_ones, first rendered wins."""

    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.cfg = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _product(self, name: str, forge_init_ones: int) -> None:
        (self.cfg / f"{name}.toml").write_text(f"[cli]\nforge_init_ones = {forge_init_ones}\n")

    def test_init_ones_0_picks_bitwise_gate(self) -> None:
        from tools import bootstrap_canonical

        self._product("multistep", 0)
        self._product("long-train", 1)
        self.assertEqual(bootstrap_canonical.resolve_canonical_gate(self.cfg, 0), "multistep")

    def test_init_ones_1_picks_production_gate(self) -> None:
        from tools import bootstrap_canonical

        self._product("multistep", 0)
        self._product("long-train", 1)
        self.assertEqual(bootstrap_canonical.resolve_canonical_gate(self.cfg, 1), "long-train")

    def test_mismatched_scheme_is_skipped(self) -> None:
        from tools import bootstrap_canonical

        # multistep rendered with the WRONG forge_init_ones is not usable for
        # scheme 0 (its product forge_init_ones would clobber our selection);
        # the resolver falls through to the next matching candidate.
        self._product("multistep", 1)
        self._product("multistep-1gpu", 0)
        self.assertEqual(
            bootstrap_canonical.resolve_canonical_gate(self.cfg, 0),
            "multistep-1gpu",
        )

    def test_no_matching_gate_fails_fast(self) -> None:
        from tools import bootstrap_canonical

        self._product("multistep", 0)  # only a scheme-0 gate exists
        with self.assertRaises(SystemExit):
            bootstrap_canonical.resolve_canonical_gate(self.cfg, 1)


class TestMainInjectsForgeGate(unittest.TestCase):
    """main() sets FORGE_GATE on the bridge subprocess env."""

    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)
        (self.repo / "ref" / "bridges").mkdir(parents=True)
        (self.repo / "ref" / "bridges" / "bridge.sh").write_text("#!/bin/sh\n")
        (self.repo / "ref" / "config").mkdir(parents=True)
        (self.repo / "ref" / "config" / "multistep.toml").write_text("[cli]\nforge_init_ones = 0\n")
        self.ckpt = self.repo / "ckpt"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_forge_gate_reaches_subprocess(self) -> None:
        from harness import config_runtime
        from tools import bootstrap_canonical

        workload = {
            "ref": {
                "backend": "torch",
                "checkpoint_root": str(self.ckpt),
                "ref_script": "run_qwen3_dense.sh",
            }
        }
        canonical = self.ckpt / "no1" / "canonical_state_fp32.pt"
        data_toml = self.repo / "data.toml"
        data_toml.write_text("[data]\n")

        def _fake_run(argv, **kwargs):
            canonical.parent.mkdir(parents=True, exist_ok=True)
            canonical.write_bytes(b"x")
            return subprocess.CompletedProcess(args=argv, returncode=0)

        # _data_config_path is mocked because the subprocess patch below lands
        # on the SHARED subprocess module: with a cold repo-root cache,
        # config_runtime._git_toplevel would otherwise receive the fake
        # CompletedProcess (stdout=None) and crash — a test-order dependency
        # that only bit when this module ran standalone.
        with (
            mock.patch.object(bootstrap_canonical, "_repo_root", return_value=self.repo),
            mock.patch.object(bootstrap_canonical, "_load_workload", return_value=workload),
            mock.patch.object(config_runtime, "_data_config_path", return_value=data_toml),
            mock.patch.object(
                bootstrap_canonical.subprocess, "run", side_effect=_fake_run
            ) as sp_run,
        ):
            rc = bootstrap_canonical.main(["--init-ones", "0"])
        self.assertEqual(rc, 0)
        self.assertEqual(sp_run.call_args.kwargs["env"]["FORGE_GATE"], "multistep")
        self.assertEqual(sp_run.call_args.kwargs["env"]["FORGE_INIT_ONES"], "0")
        # The ref axis must reach the bridge, or bridge.sh's torch branch
        # falls back to the MiniCPM launcher and the canonical comes out
        # wrong-structured for every non-MiniCPM model axis.
        self.assertEqual(
            sp_run.call_args.kwargs["env"]["FORGE_BRIDGE_REF_SCRIPT"],
            "run_qwen3_dense.sh",
        )
        # main() stamps the fingerprint sidecar so the preflight probe can
        # later verify this canonical against the rendered product bytes.
        self.assertTrue(bootstrap_canonical.fingerprint_path(canonical).exists())


class TestBuildBootstrapEnv(unittest.TestCase):
    """The bootstrap env is process env + the ref-run PATH/DATA overrides."""

    def test_merges_ref_env_over_process_env(self) -> None:
        from evals import canonical_preflight

        overrides = {"DATA_CONF": "/x/conf.sh", "FORGE_TOKENIZER_DIR": "/tok"}
        with (
            mock.patch.object(canonical_preflight, "_ref_script_path_env", return_value=overrides),
            mock.patch.dict("os.environ", {"PATH": "/usr/bin"}, clear=True),
        ):
            env = canonical_preflight.build_bootstrap_env(IMPORT_ROOT, {"ref": {}})
        self.assertEqual(env["DATA_CONF"], "/x/conf.sh")
        self.assertEqual(env["FORGE_TOKENIZER_DIR"], "/tok")
        # Process env is preserved alongside the ref overrides.
        self.assertEqual(env["PATH"], "/usr/bin")


class TestRunPreflight(unittest.TestCase):
    """The torch-only, idempotent, fail-fast orchestration."""

    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)
        self.cfg = _render_fake_ref_products(self.repo)
        self.ckpt = self.repo / "ckpt"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _workload(self, backend: str = "torch") -> dict:
        return {"ref": {"backend": backend, "checkpoint_root": str(self.ckpt)}}

    def _stamp(self, init_ones: int) -> None:
        _stamp_current(self.ckpt, self.cfg, init_ones)

    def _patch_common(self, workload: dict):
        """Patch repo root + config load + env assembly; subprocess mock ctx."""
        return (
            mock.patch.object(
                sys.modules["evals.canonical_preflight"],
                "_repo_root",
                return_value=self.repo,
            ),
            mock.patch.object(
                sys.modules["evals.canonical_preflight"].config_runtime,
                "load_workload_config",
                return_value=(Path("workload.toml"), workload),
            ),
            mock.patch.object(
                sys.modules["evals.canonical_preflight"],
                "_ref_script_path_env",
                return_value={"DATA_CONF": "/x/conf.sh"},
            ),
        )

    def test_skips_non_torch_backend(self) -> None:
        from evals import canonical_preflight

        p_root, p_cfg, p_env = self._patch_common(self._workload(backend="megatron"))
        with p_root, p_cfg, p_env, mock.patch.object(canonical_preflight, "subprocess") as sp:
            rc = canonical_preflight.run()
        self.assertEqual(rc, 0)
        sp.run.assert_not_called()

    def test_skips_when_both_present(self) -> None:
        from evals import canonical_preflight

        self._stamp(1)
        self._stamp(0)
        p_root, p_cfg, p_env = self._patch_common(self._workload())
        with p_root, p_cfg, p_env, mock.patch.object(canonical_preflight, "subprocess") as sp:
            rc = canonical_preflight.run()
        self.assertEqual(rc, 0)
        sp.run.assert_not_called()

    def test_generates_missing_with_data_conf_env(self) -> None:
        from evals import canonical_preflight

        p_root, p_cfg, p_env = self._patch_common(self._workload())
        completed = subprocess.CompletedProcess(args=[], returncode=0)
        with (
            p_root,
            p_cfg,
            p_env,
            mock.patch.object(
                canonical_preflight.subprocess, "run", return_value=completed
            ) as sp_run,
        ):
            rc = canonical_preflight.run()
        self.assertEqual(rc, 0)
        # One bootstrap invocation per missing scheme, in [1, 0] order.
        self.assertEqual(sp_run.call_count, 2)
        init_ones_args = [call.args[0][-1] for call in sp_run.call_args_list]
        self.assertEqual(init_ones_args, ["1", "0"])
        # The assembled ref env (DATA_CONF) reaches the subprocess.
        for call in sp_run.call_args_list:
            self.assertEqual(call.kwargs["env"]["DATA_CONF"], "/x/conf.sh")

    def test_only_generates_the_missing_scheme(self) -> None:
        from evals import canonical_preflight

        self._stamp(1)  # only no1 (--init-ones 0) is missing/stale
        p_root, p_cfg, p_env = self._patch_common(self._workload())
        completed = subprocess.CompletedProcess(args=[], returncode=0)
        with (
            p_root,
            p_cfg,
            p_env,
            mock.patch.object(
                canonical_preflight.subprocess, "run", return_value=completed
            ) as sp_run,
        ):
            rc = canonical_preflight.run()
        self.assertEqual(rc, 0)
        self.assertEqual(sp_run.call_count, 1)
        self.assertEqual(sp_run.call_args.args[0][-1], "0")

    def test_bootstrap_failure_is_fatal(self) -> None:
        from evals import canonical_preflight

        p_root, p_cfg, p_env = self._patch_common(self._workload())
        failed = subprocess.CompletedProcess(args=[], returncode=1)
        with (
            p_root,
            p_cfg,
            p_env,
            mock.patch.object(canonical_preflight.subprocess, "run", return_value=failed),
        ):
            with self.assertRaises(SystemExit):
                canonical_preflight.run()


if __name__ == "__main__":
    unittest.main()
