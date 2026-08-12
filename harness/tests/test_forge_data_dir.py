"""Contract tests for the workspace-relative ``[data].forge_data_dir``.

The corpus prefetch (writer), the gate dataloader (reader, via the
exported ``FORGE_DATA_DIR`` env) and the production runner's
prefetch-sentinel wait (``evals/scripts/production_ckpt.py
prefetch-wait``) MUST all land on the SAME directory. ``config_runtime`` owns the one
resolver — ``resolve_forge_data_dir`` — that lifts a relative value to an
absolute path under the harness repo root (mirroring the
``checkpoint_root`` convention), and both harness-side consumers route
through it so a relative TOML value resolves identically regardless of
process cwd.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["FORGE_REPO_ROOT"] = str(REPO_ROOT)

from harness import config_runtime  # noqa: E402


class TestResolveForgeDataDir(unittest.TestCase):
    def test_relative_lifts_to_repo_root(self) -> None:
        root = Path("/ws")
        self.assertEqual(
            config_runtime.resolve_forge_data_dir(".artifacts/forge-data/uf", root),
            (root / ".artifacts/forge-data/uf").resolve(),
        )

    def test_absolute_is_left_untouched(self) -> None:
        self.assertEqual(
            config_runtime.resolve_forge_data_dir("/opt/forge-data/uf", Path("/ws")),
            Path("/opt/forge-data/uf"),
        )

    def test_defaults_root_to_repo_root(self) -> None:
        with mock.patch.object(config_runtime, "repo_root", return_value=Path("/ws")):
            self.assertEqual(
                config_runtime.resolve_forge_data_dir(".artifacts/d"),
                (Path("/ws") / ".artifacts/d").resolve(),
            )


class TestResolveAssetsExportsAbsoluteDataDir(unittest.TestCase):
    def test_relative_forge_data_dir_is_exported_absolute(self) -> None:
        root = Path("/ws")
        ref = {"forge_tokenizer_dir": "/tok", "tokenizer": "some/repo"}
        data = {
            "dataset": "openbmb/Ultra-FineWeb",
            "forge_data_dir": ".artifacts/forge-data/uf",
        }
        expected = str((root / ".artifacts/forge-data/uf").resolve())
        with (
            mock.patch.object(config_runtime, "repo_root", return_value=root),
            mock.patch.object(config_runtime, "_ensure_tokenizer"),
            mock.patch.object(config_runtime, "_ensure_dataset") as ensure,
            mock.patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("FORGE_DATA_DIR", None)
            out = config_runtime.resolve_assets(ref, data)
            # the in-process conf-sourcing reads it from os.environ
            self.assertEqual(os.environ.get("FORGE_DATA_DIR"), expected)
        self.assertEqual(out["FORGE_DATA_DIR"], expected)
        # the on-demand download must target the resolved absolute dir
        self.assertEqual(ensure.call_args.args[1], Path(expected))


class TestPrefetchWaitResolvesDir(unittest.TestCase):
    """``production_ckpt.prefetch_wait`` must wait on the SAME sentinel dir
    the prefetch writer used: a relative ``[data].forge_data_dir`` resolves
    against the harness repo root, never the process cwd."""

    def test_relative_dir_resolves_against_repo_root(self) -> None:
        scripts_dir = str(REPO_ROOT / "evals" / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import production_ckpt

        captured: dict[str, object] = {}

        def fake_wait(path, timeout_s):
            captured["path"] = path
            return {"status": "ok", "bytes": 0}

        wc = {
            "data": {
                "prefetch_target_gb": 1,
                "forge_data_dir": ".artifacts/forge-data/uf",
            }
        }
        root = Path("/ws")
        with (
            mock.patch.object(config_runtime, "repo_root", return_value=root),
            mock.patch("evals._common._load_workload_config_for_ref", return_value=wc),
            mock.patch("tools.prefetch_data.wait_for_prefetch", side_effect=fake_wait),
        ):
            production_ckpt.prefetch_wait()
        self.assertEqual(captured["path"], (root / ".artifacts/forge-data/uf").resolve())


if __name__ == "__main__":
    unittest.main()
