"""Canonical-state bootstrap wiring on the torch ref capture path.

``harness_dp.install_from_args`` owns capture installation for the torch
refs (the bridge stopped routing torch through the interposer, whose
``elif CANONICAL_OUTPUT`` branch used to own canonical bootstrap). When no
capture output is requested but ``CANONICAL_STATE_OUTPUT_FILE`` is set, it
must install the one-shot FP32 master dump instead of a capture session.
"""

from __future__ import annotations

import argparse
import sys
import unittest
import unittest.mock as mock
from pathlib import Path

# ``harness/`` is the PYTHONPATH import root (evals/ lives under it).
IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPORT_ROOT))


def _args(hash_output: str = "") -> argparse.Namespace:
    return argparse.Namespace(hash_output=hash_output, hash_capture_level=0, persistent=False)


class TestInstallFromArgsCanonical(unittest.TestCase):
    def test_canonical_env_installs_master_dump(self) -> None:
        from evals import harness_dp

        model, optim = mock.MagicMock(), mock.MagicMock()
        with (
            mock.patch.dict(
                "os.environ",
                {"CANONICAL_STATE_OUTPUT_FILE": "/ckpt/ones/canonical_state_fp32.pt"},
                clear=True,
            ),
            mock.patch.object(harness_dp.harness_hook, "install_canonical_state_dump") as canon,
            mock.patch.object(harness_dp, "install") as capture,
        ):
            result = harness_dp.install_from_args(model, optim, _args())
        self.assertIsNone(result)
        canon.assert_called_once()
        self.assertEqual(
            canon.call_args.kwargs["output_file"],
            "/ckpt/ones/canonical_state_fp32.pt",
        )
        # The capture path must NOT run for a canonical bootstrap.
        capture.assert_not_called()

    def test_hash_output_wins_over_canonical(self) -> None:
        from evals import harness_dp

        model, optim = mock.MagicMock(), mock.MagicMock()
        with (
            mock.patch.dict(
                "os.environ",
                {"CANONICAL_STATE_OUTPUT_FILE": "/ckpt/ones/canonical_state_fp32.pt"},
                clear=True,
            ),
            mock.patch.object(harness_dp.harness_hook, "install_canonical_state_dump") as canon,
            mock.patch.object(harness_dp, "install") as capture,
        ):
            harness_dp.install_from_args(model, optim, _args(hash_output="/x/hash.json"))
        canon.assert_not_called()
        capture.assert_called_once()
        self.assertEqual(capture.call_args.kwargs["output_file"], "/x/hash.json")

    def test_no_output_is_strict_noop(self) -> None:
        from evals import harness_dp

        model, optim = mock.MagicMock(), mock.MagicMock()
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch.object(harness_dp.harness_hook, "install_canonical_state_dump") as canon,
            mock.patch.object(harness_dp, "install") as capture,
        ):
            harness_dp.install_from_args(model, optim, _args())
        canon.assert_not_called()
        capture.assert_called_once()
        self.assertIsNone(capture.call_args.kwargs["output_file"])


if __name__ == "__main__":
    unittest.main()
