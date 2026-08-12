"""Tests for the unified M1 bridge (bridge.sh + interposer.py).

Covers:
* bridge.sh fail-fast on missing / unknown FORGE_BACKEND.
* Python heredoc sed-patch correctness for both backends.
* interposer.py dispatch and hook wiring for both backends.
* No sitecustomize / PYTHONPATH pollution in child processes.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BRIDGE_SH = _REPO_ROOT / "ref" / "bridges" / "bridge.sh"
_INTERPOSER_PY = _REPO_ROOT / "ref" / "bridges" / "interposer.py"
_MEGATRON_LAUNCHER = _REPO_ROOT / "ref" / "reference" / "train_minicpm4_0.5b_gsm8k.sh"
_TORCH_LAUNCHER = _REPO_ROOT / "ref" / "reference" / "run_16gpu_1000step_pure_mup_mtp.sh"


# ── bridge.sh env-validation tests ──────────────────────────────────


class TestBridgeShEnvValidation(unittest.TestCase):
    """bridge.sh must fail fast when FORGE_BACKEND is missing or unknown."""

    def _run_bridge(self, env_override: dict[str, str]) -> subprocess.CompletedProcess:
        env = {
            "PATH": os.environ["PATH"],
            "HOME": os.environ.get("HOME", "/tmp"),
            **env_override,
        }
        return subprocess.run(
            ["bash", str(_BRIDGE_SH)],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_missing_backend_exits_nonzero(self):
        result = self._run_bridge({})
        self.assertNotEqual(result.returncode, 0)

    def test_unknown_backend_exits_nonzero(self):
        result = self._run_bridge({"FORGE_BACKEND": "unknown"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown", result.stderr.lower())


# ── Sed-patch (Python heredoc replacement) tests ────────────────────


class TestMegatronDataPathStripping(unittest.TestCase):
    """bridge.sh strips the weighted-shard prefix from DATA_PATH for Megatron."""

    def _run_bash_strip(self, data_path: str) -> str:
        script = (
            f'DATA_PATH="{data_path}"; '
            'if [[ "${DATA_PATH:-}" =~ ^[0-9]+(\\.[0-9]+)?[[:space:]]+(.+)$ ]]; then '
            'DATA_PATH="${BASH_REMATCH[2]}"; fi; printf "%s" "$DATA_PATH"'
        )
        result = subprocess.run(
            ["bash", "-c", script],
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0)
        return result.stdout

    def test_weighted_shard_prefix_stripped(self):
        self.assertEqual(
            self._run_bash_strip("1.0 /data/gsm8k_megatron/gsm8k_train_text_document"),
            "/data/gsm8k_megatron/gsm8k_train_text_document",
        )

    def test_integer_weight_stripped(self):
        self.assertEqual(
            self._run_bash_strip("1 /data/prefix"),
            "/data/prefix",
        )

    def test_plain_path_unchanged(self):
        self.assertEqual(
            self._run_bash_strip("/data/gsm8k_megatron/gsm8k_train_text_document"),
            "/data/gsm8k_megatron/gsm8k_train_text_document",
        )


class TestSedPatchMegatron(unittest.TestCase):
    """The Megatron heredoc replaces 'pretrain_gpt.py' with the interposer."""

    def test_needle_exists_in_ref_launcher(self):
        self.assertTrue(
            _MEGATRON_LAUNCHER.exists(),
            f"Megatron ref launcher not found: {_MEGATRON_LAUNCHER}",
        )
        content = _MEGATRON_LAUNCHER.read_text()
        self.assertIn(
            "pretrain_gpt.py",
            content,
            "Megatron launcher must contain 'pretrain_gpt.py' needle",
        )

    def test_replacement_produces_interposer_path(self):
        content = _MEGATRON_LAUNCHER.read_text()
        interposer_abs = str(_INTERPOSER_PY)
        patched = content.replace("pretrain_gpt.py", interposer_abs, 1)
        self.assertIn(interposer_abs, patched)
        # First occurrence is replaced; the interposer path appears at least once
        self.assertEqual(patched.count(interposer_abs), 1)


class TestSedPatchTorch(unittest.TestCase):
    """The Torch heredoc rewrites whichever PY_ENTRY the ACTIVE launcher
    names to point at the interposer — toml-driven via FORGE_BRIDGE_REF_SCRIPT,
    not hardcoded to pure_mup_mtp. This is what lets qwen3's M1 capture
    interpose its own ``run_qwen3_dense.sh`` reference instead of a minicpm one.
    """

    # Mirror of the regex the bridge.sh torch heredoc uses.
    _PY_ENTRY_RE = re.compile(r'^PY_ENTRY="\$SCRIPT_DIR/(?P<entry>[^"/]+\.py)"', re.M)

    # Every torch L0 launcher routed through the bridge, with the train
    # entry each one is expected to name. Adding a model = adding a row.
    _LAUNCHERS = {
        "run_16gpu_1000step_pure_mup_mtp.sh": "train_pure_mup_mtp.py",
        "run_qwen3_dense.sh": "train_qwen3_dense.py",
    }

    def test_each_launcher_declares_matchable_py_entry(self):
        for launcher, entry in self._LAUNCHERS.items():
            with self.subTest(launcher=launcher):
                path = _REPO_ROOT / "ref" / "reference" / launcher
                self.assertTrue(path.exists(), f"ref launcher not found: {path}")
                m = self._PY_ENTRY_RE.search(path.read_text())
                self.assertIsNotNone(
                    m, f'{launcher} must declare a PY_ENTRY="$SCRIPT_DIR/*.py" line'
                )
                self.assertEqual(m.group("entry"), entry)

    def test_replacement_redirects_each_launcher_to_interposer(self):
        interposer_abs = str(_INTERPOSER_PY)
        for launcher in self._LAUNCHERS:
            with self.subTest(launcher=launcher):
                content = (_REPO_ROOT / "ref" / "reference" / launcher).read_text()
                m = self._PY_ENTRY_RE.search(content)
                patched = content[: m.start()] + f'PY_ENTRY="{interposer_abs}"' + content[m.end() :]
                self.assertIn(f'PY_ENTRY="{interposer_abs}"', patched)
                self.assertNotIn('PY_ENTRY="$SCRIPT_DIR/', patched)

    def test_bridge_threads_ref_script_to_torch_launcher_selection(self):
        # The dispatcher (evals/_common.run_ref_capture) must hand the active
        # [ref].ref_script to the bridge as FORGE_BRIDGE_REF_SCRIPT, and the
        # bridge's torch branch must consume it (default keeps 0.5B/1B).
        common_src = (_REPO_ROOT / "evals" / "_common.py").read_text()
        self.assertIn("FORGE_BRIDGE_REF_SCRIPT", common_src)
        self.assertIn("config_runtime.ref_script(workload_config)", common_src)
        bridge_src = _BRIDGE_SH.read_text()
        self.assertIn('FORGE_BRIDGE_REF_SCRIPT="${FORGE_BRIDGE_REF_SCRIPT:-', bridge_src)


class TestRunRefCaptureThreadsRefScript(unittest.TestCase):
    """The ref-side env projector (``runtime_env._ref_overrides`` — successor
    of the deleted ``run_ref_capture`` env plumbing) must thread the active
    ``[ref].ref_script`` into ``FORGE_BRIDGE_REF_SCRIPT``.

    Regression guard: without this, the bridge's torch branch falls back to
    its ``run_16gpu_1000step_pure_mup_mtp.sh`` (minicpm) default, so a qwen3
    config's M1.fwd / M1.bwd capture a minicpm reference (wrong
    architecture) and can never bitwise-align against a qwen3 candidate.
    """

    def test_forge_bridge_ref_script_is_active_ref_script(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "runtime_env_under_test",
            _REPO_ROOT / "evals" / "scripts" / "runtime_env.py",
        )
        runtime_env = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime_env)

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            bridge = repo_root / "ref" / "bridges" / "bridge.sh"
            bridge.parent.mkdir(parents=True, exist_ok=True)
            bridge.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            ref_script = repo_root / "ref" / "reference" / "run_qwen3_dense.sh"
            ref_script.parent.mkdir(parents=True, exist_ok=True)
            ref_script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            wc = {
                "ref": {
                    "backend": "torch",
                    "ref_script": "run_qwen3_dense.sh",
                    "ref_capture_script": "ref/bridges/bridge.sh",
                },
                # _ref_overrides now resolves MASTER_ADDR/MASTER_PORT from the
                # eval axis (mirroring _ours_overrides) so the runtime value
                # beats the freeze-filled product [env] value under the fully
                # generic projection.
                "runtime": {
                    "distributed": {
                        "master_addr": "127.0.0.1",
                        "master_port": "auto",
                    }
                },
            }
            with mock.patch.object(runtime_env._common, "_ref_script_path_env", return_value={}):
                env = runtime_env._ref_overrides(
                    repo_root, "forward-align", {"forge_init_ones": 0}, wc
                )

        self.assertEqual(env.get("FORGE_BRIDGE_REF_SCRIPT"), "run_qwen3_dense.sh")
        # Compare resolved: macOS tempdirs alias /var → /private/var.
        self.assertEqual(Path(env["FORGE_REF_SCRIPT_PATH"]).resolve(), ref_script.resolve())
        self.assertEqual(Path(env["FORGE_REF_CAPTURE_SCRIPT_PATH"]).resolve(), bridge.resolve())
        self.assertEqual(env.get("FORGE_BACKEND"), "torch")


# ── Torch ref family dispatch (persistent prepatch) tests ───────────


class TestTorchRefFamilyDispatch(unittest.TestCase):
    """interposer's persistent prepatch + model hook resolve per ref family
    from BRIDGE_ORIGINAL_ENTRY, not hardcoded to MiniCPM.

    This is the M2/M3 persistent-capture counterpart of the M1 toml-driven
    bridge fix: a qwen3 run (hash_capture_level=2 → --persistent) must
    prepatch ``train_qwen3_dense.py`` — which has NO ``--init-ones`` arg —
    instead of the MiniCPM ``train_pure_mup_mtp.py`` (whose required
    ``--init-ones`` flag crashed the hardcoded path). Pure source rewrite,
    so these run without torch.
    """

    @staticmethod
    def _load_interposer(entry_basename: str):
        import importlib.util

        saved_argv = list(sys.argv)
        try:
            with mock.patch.dict(
                os.environ,
                {
                    "BRIDGE_BACKEND": "torch",
                    "BRIDGE_ORIGINAL_ENTRY": f"/fake/ref/{entry_basename}",
                    "BRIDGE_REF_DIR": str(_REPO_ROOT / "ref" / "reference"),
                    "BRIDGE_REPO_ROOT": str(_REPO_ROOT),
                },
            ):
                spec = importlib.util.spec_from_file_location(
                    "interposer_family", str(_INTERPOSER_PY)
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
        finally:
            sys.argv = saved_argv
        return mod

    def test_registry_has_both_torch_families(self):
        mod = self._load_interposer("train_qwen3_dense.py")
        self.assertIn("train_pure_mup_mtp.py", mod._TORCH_REF_FAMILIES)
        self.assertIn("train_qwen3_dense.py", mod._TORCH_REF_FAMILIES)

    def test_resolve_picks_qwen3_family(self):
        mod = self._load_interposer("train_qwen3_dense.py")
        module_name, class_name, _prepatch = mod._resolve_torch_ref_family()
        self.assertEqual(module_name, "model_qwen3")
        self.assertEqual(class_name, "Qwen3Dense")

    def test_resolve_picks_minicpm_family(self):
        mod = self._load_interposer("train_pure_mup_mtp.py")
        module_name, class_name, _prepatch = mod._resolve_torch_ref_family()
        self.assertEqual(module_name, "model_pure_mup_mtp")
        self.assertEqual(class_name, "MiniCPM4MupMtp")

    def test_unregistered_entry_fails_fast(self):
        mod = self._load_interposer("train_qwen3_dense.py")
        mod.ORIGINAL_ENTRY = "/fake/ref/train_unknown.py"
        with self.assertRaises(SystemExit):
            mod._resolve_torch_ref_family()

    def test_qwen3_prepatch_applies_all_needles(self):
        mod = self._load_interposer("train_qwen3_dense.py")
        out_path = mod._prepatch_qwen3_dense_source(str(_REPO_ROOT))
        self.assertIsNotNone(
            out_path,
            "qwen3 prepatch returned None — a needle no longer matches "
            "train_qwen3_dense.py (see stderr 'missing needles')",
        )
        self.assertIn("train_qwen3_dense_patched_rank", out_path)
        patched = Path(out_path).read_text()
        self.assertIn("_HARNESS_SESSION = None", patched)
        self.assertIn("_HARNESS_SESSION.begin_step(step)", patched)
        self.assertIn("loss.per_token.preallreduce", patched)
        self.assertIn("grad.{_n}.postallreduce", patched)
        # The Qwen3 dense ref has no --init-ones required arg (the
        # MiniCPM-only flag that crashed the old hardcoded persistent path).
        self.assertNotIn("--init-ones", patched)

    def test_minicpm_prepatch_still_applies_all_needles(self):
        mod = self._load_interposer("train_pure_mup_mtp.py")
        out_path = mod._prepatch_train_pure_mup_mtp_source(str(_REPO_ROOT))
        self.assertIsNotNone(out_path, "minicpm prepatch regressed — a needle no longer matches")
        self.assertIn("train_pure_mup_mtp_patched_rank", out_path)
        patched = Path(out_path).read_text()
        self.assertIn("_HARNESS_SESSION.begin_step(step)", patched)


# ── Interposer dispatch tests ───────────────────────────────────────


class TestInterposerDispatchMegatron(unittest.TestCase):
    """interposer._install_megatron_hooks patches setup_model_and_optimizer."""

    def test_hook_wired_with_correct_grad_attrs(self):
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            hook_output = f.name

        fake_model = mock.MagicMock(name="model")
        fake_optimizer = mock.MagicMock(name="optimizer")

        fake_training = mock.MagicMock()
        original_fn = mock.MagicMock(return_value=([fake_model], fake_optimizer, mock.MagicMock()))
        fake_training.setup_model_and_optimizer = original_fn

        fake_arguments = mock.MagicMock()
        fake_arguments.parse_args = mock.MagicMock()

        fake_utils = mock.MagicMock()
        fake_utils.get_batch_on_this_tp_rank = None

        # Wire the module hierarchy so `from megatron.training import training`
        # resolves through attribute access on the parent mock.
        fake_megatron = mock.MagicMock()
        fake_megatron_training_pkg = mock.MagicMock()
        fake_megatron_training_pkg.training = fake_training
        fake_megatron_training_pkg.arguments = fake_arguments
        fake_megatron_training_pkg.utils = fake_utils
        fake_megatron.training = fake_megatron_training_pkg

        mock_install = mock.MagicMock()
        mock_install_canonical = mock.MagicMock()

        with (
            mock.patch.dict(
                sys.modules,
                {
                    "megatron": fake_megatron,
                    "megatron.training": fake_megatron_training_pkg,
                    "megatron.training.training": fake_training,
                    "megatron.training.arguments": fake_arguments,
                    "megatron.training.utils": fake_utils,
                },
            ),
            mock.patch.dict(
                os.environ,
                {
                    "BRIDGE_BACKEND": "megatron",
                    "BRIDGE_ORIGINAL_ENTRY": "/fake/pretrain_gpt.py",
                    "BRIDGE_REF_DIR": "/fake/megatron",
                    "BRIDGE_REPO_ROOT": str(_REPO_ROOT),
                    "HOOK_OUTPUT_FILE": hook_output,
                },
            ),
            mock.patch(
                "evals.harness_hook.install",
                mock_install,
            ),
            mock.patch(
                "evals.harness_hook.install_canonical_state_dump",
                mock_install_canonical,
            ),
        ):
            # Re-import to pick up mocked megatron
            import importlib

            spec = importlib.util.spec_from_file_location(
                "interposer_test",
                str(_INTERPOSER_PY),
            )
            mod = importlib.util.module_from_spec(spec)

            # Override module-level env reads
            mod.BACKEND = "megatron"
            mod.ORIGINAL_ENTRY = "/fake/pretrain_gpt.py"
            mod.REF_DIR = "/fake/megatron"
            mod.REPO_ROOT = str(_REPO_ROOT)
            mod.HOOK_OUTPUT = hook_output
            mod.CANONICAL_OUTPUT = None

            spec.loader.exec_module(mod)

            mod._install_megatron_hooks()

            # setup_model_and_optimizer should now be wrapped
            wrapped = fake_training.setup_model_and_optimizer
            self.assertIsNot(wrapped, original_fn)

            # Call the wrapper to trigger hook wiring
            wrapped()
            mock_install.assert_called_once()
            call_kwargs = mock_install.call_args
            self.assertEqual(
                call_kwargs.kwargs.get("grad_attrs") or call_kwargs[1].get("grad_attrs"),
                ("main_grad", "grad"),
            )


class TestInterposerDispatchTorch(unittest.TestCase):
    """interposer._install_torch_hooks patches model + optimizer inits."""

    def test_hook_wired_with_correct_grad_attrs(self):
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            hook_output = f.name

        mock_install = mock.MagicMock()

        fake_model_cls = type(
            "MiniCPM4MupMtp",
            (),
            {
                "__init__": lambda self, *a, **k: None,
            },
        )
        fake_ref_mod = mock.MagicMock()
        fake_ref_mod.MiniCPM4MupMtp = fake_model_cls

        import torch

        original_adamw_init = torch.optim.AdamW.__init__

        with (
            mock.patch.dict(
                sys.modules,
                {"model_pure_mup_mtp": fake_ref_mod},
            ),
            mock.patch.dict(
                os.environ,
                {
                    "BRIDGE_BACKEND": "torch",
                    "BRIDGE_ORIGINAL_ENTRY": "/fake/train_pure_mup_mtp.py",
                    "BRIDGE_REF_DIR": "/fake/ref",
                    "BRIDGE_REPO_ROOT": str(_REPO_ROOT),
                    "HOOK_OUTPUT_FILE": hook_output,
                },
            ),
            mock.patch(
                "evals.harness_hook.install",
                mock_install,
            ),
        ):
            import importlib

            spec = importlib.util.spec_from_file_location(
                "interposer_test_torch",
                str(_INTERPOSER_PY),
            )
            mod = importlib.util.module_from_spec(spec)
            mod.BACKEND = "torch"
            mod.ORIGINAL_ENTRY = "/fake/train_pure_mup_mtp.py"
            mod.REF_DIR = "/fake/ref"
            mod.REPO_ROOT = str(_REPO_ROOT)
            mod.HOOK_OUTPUT = hook_output
            mod.CANONICAL_OUTPUT = None

            spec.loader.exec_module(mod)

            mod._install_torch_hooks()

            # Simulate the two-phase capture: model init then optimizer init
            fake_ref_mod.MiniCPM4MupMtp()
            # AdamW needs real params
            param = torch.nn.Parameter(torch.zeros(2))
            torch.optim.AdamW([param], lr=0.01)

            mock_install.assert_called_once()
            call_kwargs = mock_install.call_args
            self.assertEqual(
                call_kwargs.kwargs.get("grad_attrs") or call_kwargs[1].get("grad_attrs"),
                ("grad",),
            )

        # Restore AdamW.__init__ to avoid polluting other tests
        torch.optim.AdamW.__init__ = original_adamw_init


# Need torch for the dispatch tests above
try:
    import importlib.util as _importlib_util

    _HAS_TORCH = _importlib_util.find_spec("torch") is not None
except (ImportError, ModuleNotFoundError):
    _HAS_TORCH = False

TestInterposerDispatchMegatron = unittest.skipUnless(_HAS_TORCH, "torch not installed")(
    TestInterposerDispatchMegatron
)
TestInterposerDispatchTorch = unittest.skipUnless(_HAS_TORCH, "torch not installed")(
    TestInterposerDispatchTorch
)


# ── No sitecustomize pollution test ─────────────────────────────────


class TestNoSitecustomizePollution(unittest.TestCase):
    """Subprocesses must NOT auto-load the interposer via sitecustomize."""

    def test_subprocess_does_not_import_interposer(self):
        env = os.environ.copy()
        env.pop("PYTHONSTARTUP", None)
        # Ensure no sitecustomize injection directory on PYTHONPATH
        pythonpath = env.get("PYTHONPATH", "")
        bridge_dir = str(_INTERPOSER_PY.parent)
        cleaned = os.pathsep.join(p for p in pythonpath.split(os.pathsep) if p != bridge_dir)
        env["PYTHONPATH"] = cleaned

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; print('interposer' in sys.modules)",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
