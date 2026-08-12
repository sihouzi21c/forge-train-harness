"""Unit tests for the ours-side ``--config`` transport (``_gate_entry``).

The runner sh passes the ours product's path as the single ``--config <path>``
CLI arg; ``GateInputs`` reads it from argv (so it resolves at import time,
before a runner's own argparse). The legacy ``FORGE_GATE`` +
``FORGE_OURS_CONFIG_DIR`` env pair is retired, and the os.environ fallback for
product-absent keys is restricted to the declared whitelist
(``_ENV_FALLBACK_KEYS``) plus explicit ``<runtime>`` sentinels. Pure CPU checks.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = REPO_ROOT / "evals" / "scripts"
for _p in (str(REPO_ROOT), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _gate_entry as ge  # noqa: E402

_PRODUCT = (
    "[cli]\n"
    'name = "minicpm"\n'
    "world_size = 1\n"
    "num_steps = 8\n"
    "forge_init_ones = 0\n"
    "\n"
    "[env]\n"
    'BACKEND = "torch"\n'
    'CHECKPOINT_ROOT = ".artifacts/checkpoints/torch"\n'
    'MASTER_ADDR = "localhost"\n'
    'MASTER_PORT = "29517"\n'
)


def _write_product(text: str = _PRODUCT) -> Path:
    d = Path(tempfile.mkdtemp())
    p = d / "multistep-1gpu.toml"
    p.write_text(text)
    return p


class ConfigArgvParsingTests(unittest.TestCase):
    def test_space_form(self) -> None:
        self.assertEqual(
            ge._config_path_from_argv(["--hash-capture-level", "1", "--config", "/a/b.toml"]),
            "/a/b.toml",
        )

    def test_equals_form(self) -> None:
        self.assertEqual(
            ge._config_path_from_argv(["--config=/x/y.toml", "--persistent"]), "/x/y.toml"
        )

    def test_absent(self) -> None:
        self.assertIsNone(ge._config_path_from_argv(["--persistent"]))


class GateInputsConfigTests(unittest.TestCase):
    def test_explicit_config_path(self) -> None:
        p = _write_product()
        gi = ge.GateInputs(config_path=str(p))
        self.assertEqual(gi.require("NUM_STEPS"), "8")
        self.assertEqual(gi.require("BACKEND"), "torch")
        self.assertEqual(gi.require("MASTER_PORT"), "29517")

    def test_config_read_from_argv(self) -> None:
        p = _write_product()
        old = sys.argv
        try:
            sys.argv = ["eval_train_steps.py", "--config", str(p), "--hash-capture-level", "0"]
            gi = ge.GateInputs()
            self.assertEqual(gi.require("NUM_STEPS"), "8")
        finally:
            sys.argv = old

    def test_legacy_env_pair_no_longer_accepted(self) -> None:
        import os

        p = _write_product()
        old = sys.argv
        os.environ["FORGE_GATE"] = "multistep-1gpu"
        os.environ["FORGE_OURS_CONFIG_DIR"] = str(p.parent)
        try:
            sys.argv = ["eval_train_steps.py"]  # no --config
            with self.assertRaises(RuntimeError):
                ge.GateInputs()
        finally:
            sys.argv = old
            del os.environ["FORGE_GATE"], os.environ["FORGE_OURS_CONFIG_DIR"]

    def test_missing_config_raises(self) -> None:
        old = sys.argv
        try:
            sys.argv = ["eval_train_steps.py"]
            with self.assertRaises(RuntimeError):
                ge.GateInputs()
        finally:
            sys.argv = old

    def test_world_size_is_env_sourced(self) -> None:
        import os

        p = _write_product()
        os.environ["WORLD_SIZE"] = "4"  # launch_dp per-process identity
        try:
            gi = ge.GateInputs(config_path=str(p))
            # runtime rendezvous key comes from env, NOT the product's world_size=1
            self.assertEqual(gi.require("WORLD_SIZE"), "4")
        finally:
            del os.environ["WORLD_SIZE"]


class EnvFallbackWhitelistTests(unittest.TestCase):
    """Product-absent keys read os.environ ONLY via the declared channels."""

    def test_stray_env_cannot_shadow_missing_product_key(self) -> None:
        import os

        p = _write_product()
        os.environ["SEQ_LENGTH"] = "9999"  # not whitelisted, not in product
        try:
            gi = ge.GateInputs(config_path=str(p))
            self.assertIsNone(gi.get("SEQ_LENGTH"))
            with self.assertRaises(RuntimeError):
                gi.require("SEQ_LENGTH")
        finally:
            del os.environ["SEQ_LENGTH"]

    def test_runtime_sentinel_defers_to_env(self) -> None:
        import os

        p = _write_product(_PRODUCT + 'MEGATRON_ROOT = "<runtime>"\n')
        os.environ["MEGATRON_ROOT"] = "/opt/megatron"
        try:
            gi = ge.GateInputs(config_path=str(p))
            self.assertEqual(gi.require("MEGATRON_ROOT"), "/opt/megatron")
        finally:
            del os.environ["MEGATRON_ROOT"]

    def test_whitelisted_key_falls_back_to_env(self) -> None:
        import os

        # DATA_PATH is never rendered into the product (data axis is SSOT);
        # runtime_env.py delivers it via env.
        p = _write_product()
        os.environ["DATA_PATH"] = "/data/corpus.bin"
        try:
            gi = ge.GateInputs(config_path=str(p))
            self.assertEqual(gi.require("DATA_PATH"), "/data/corpus.bin")
        finally:
            del os.environ["DATA_PATH"]

    def test_per_run_env_beats_product(self) -> None:
        import os

        # The production segment loop overrides the product's total NUM_STEPS
        # with this segment's slice.
        p = _write_product()
        os.environ["NUM_STEPS"] = "2"
        try:
            gi = ge.GateInputs(config_path=str(p))
            self.assertEqual(gi.require("NUM_STEPS"), "2")
        finally:
            del os.environ["NUM_STEPS"]
        gi = ge.GateInputs(config_path=str(p))
        self.assertEqual(gi.require("NUM_STEPS"), "8")


class CheckpointRootDirectReadTests(unittest.TestCase):
    """GateInputs.checkpoint_root: a straight read of the product's value.

    Freeze (``tools/resolve_deploy.py``) writes the FINAL CHECKPOINT_ROOT into
    each product — the repo-relative base with the gate's own ones/no1 subdir
    already appended. The runner does NOT lift it against a repo_root nor
    re-derive ones/no1 (design §0: ours only READS). The bare relative value
    resolves against the engine's cwd on the execution machine.
    """

    def _product(self, ckpt: str) -> Path:
        d = Path(tempfile.mkdtemp())
        p = d / "gate.toml"
        p.write_text(f'[cli]\nname = "minicpm"\n\n[env]\nCHECKPOINT_ROOT = "{ckpt}"\n')
        return p

    def test_relative_value_read_verbatim(self) -> None:
        p = self._product(".artifacts/checkpoints/torch/ones")
        gi = ge.GateInputs(config_path=str(p))
        self.assertEqual(gi.checkpoint_root(), ".artifacts/checkpoints/torch/ones")

    def test_no1_value_read_verbatim(self) -> None:
        p = self._product(".artifacts/checkpoints/torch/no1")
        gi = ge.GateInputs(config_path=str(p))
        self.assertEqual(gi.checkpoint_root(), ".artifacts/checkpoints/torch/no1")

    def test_absolute_value_read_verbatim(self) -> None:
        p = self._product("/abs/ckpt/ones")
        gi = ge.GateInputs(config_path=str(p))
        self.assertEqual(gi.checkpoint_root(), "/abs/ckpt/ones")

    def test_missing_checkpoint_root_raises(self) -> None:
        d = Path(tempfile.mkdtemp())
        p = d / "gate.toml"
        p.write_text('[cli]\nname = "minicpm"\n\n[env]\nBACKEND = "torch"\n')
        gi = ge.GateInputs(config_path=str(p))
        with self.assertRaises(RuntimeError):
            gi.checkpoint_root()


if __name__ == "__main__":
    unittest.main()
