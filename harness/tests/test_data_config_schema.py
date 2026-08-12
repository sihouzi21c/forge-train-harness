"""[data] schema: inline data_path/data_loader is a first-class source (fix A).

Two data-source mechanisms coexist for different ref families:
  * meta bundles read inline [data].data_path / data_loader via FORGE_DATA_TOML;
  * native refs source a shell conf named by [data].conf_path.
The two must not be mixed in one [data] table (that is the "two paths" the fix
eliminates), so inline data_path and conf_path/conf_name are mutually exclusive.

This also pins the original loop-startup bug: a [data] with data_path +
data_loader (the materialized meta bundle) must validate, not raise
"unknown keys".
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness import config_runtime as cr  # noqa: E402

P = Path("data.toml")


class DataConfigSchemaTest(unittest.TestCase):
    def test_inline_data_path_and_loader_accepted(self):
        # The materialized meta bundle shape — must not raise (the bug fix).
        cr._validate_data_config(
            P,
            {
                "data_path": "0.667 en/*.parquet 0.333 zh/*.parquet",
                "data_loader": "ultra_fineweb",
                "dataset": "openbmb/Ultra-FineWeb",
                "forge_data_dir": ".artifacts/forge-data/ultra_fineweb",
            },
        )

    def test_pure_conf_path_accepted(self):
        # The native ref family — conf-sourced, no inline data_path.
        cr._validate_data_config(
            P,
            {
                "conf_path": "ref/reference/ultra_fineweb_data_conf.sh",
                "data_loader": "hf",
            },
        )

    def test_inline_data_path_with_conf_path_rejected(self):
        with self.assertRaisesRegex(ValueError, "one source|mutually exclusive"):
            cr._validate_data_config(
                P,
                {
                    "data_path": "0.5 a/*.parquet",
                    "conf_path": "ref/reference/x.sh",
                },
            )

    def test_inline_data_path_with_conf_name_rejected(self):
        with self.assertRaisesRegex(ValueError, "one source|mutually exclusive"):
            cr._validate_data_config(
                P,
                {
                    "data_path": "0.5 a/*.parquet",
                    "conf_name": "x",
                },
            )

    def test_empty_data_path_rejected(self):
        with self.assertRaisesRegex(ValueError, "data_path"):
            cr._validate_data_config(P, {"data_path": ""})

    def test_unknown_key_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            cr._validate_data_config(P, {"data_path": "0.5 a/*.parquet", "bogus": 1})


if __name__ == "__main__":
    unittest.main()
