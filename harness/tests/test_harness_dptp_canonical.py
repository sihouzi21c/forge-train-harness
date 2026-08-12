"""Canonical-state bootstrap wiring on the DP×TP ref capture path.

``harness_dptp.install_from_args`` owns capture installation for the 8B
DP×TP torch ref. When no capture output is requested but
``CANONICAL_STATE_OUTPUT_FILE`` is set, it must install the one-shot FP32
master dump (immediate mode — the 32-layer 8B only fits before the fp32
grad buffers and AdamW state materialize) instead of a capture session.
Without this branch the env var is silently ignored and the bootstrap runs
a REAL training step — the OOM that killed loop b8f4f63de47e.
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
    def test_canonical_env_installs_immediate_master_dump(self) -> None:
        from evals import harness_dptp

        model, optim = mock.MagicMock(), mock.MagicMock()
        with (
            mock.patch.dict(
                "os.environ",
                {"CANONICAL_STATE_OUTPUT_FILE": "/ckpt/ones/canonical_state_fp32.pt"},
                clear=True,
            ),
            mock.patch.object(harness_dptp.harness_hook, "install_canonical_state_dump") as canon,
            mock.patch.object(harness_dptp, "install") as capture,
        ):
            result = harness_dptp.install_from_args(model, optim, _args())
        self.assertIsNone(result)
        canon.assert_called_once()
        self.assertEqual(
            canon.call_args.kwargs["output_file"],
            "/ckpt/ones/canonical_state_fp32.pt",
        )
        # Immediate mode is load-bearing: hijacking optimizer.step would OOM
        # the full 8B once grad buffers + AdamW state materialize.
        self.assertTrue(canon.call_args.kwargs["immediate"])
        self.assertIs(
            canon.call_args.kwargs["writer_rank_predicate"],
            harness_dptp._default_writer_rank_predicate,
        )
        # The capture path must NOT run for a canonical bootstrap.
        capture.assert_not_called()

    def test_hash_output_wins_over_canonical(self) -> None:
        from evals import harness_dptp

        model, optim = mock.MagicMock(), mock.MagicMock()
        with (
            mock.patch.dict(
                "os.environ",
                {"CANONICAL_STATE_OUTPUT_FILE": "/ckpt/ones/canonical_state_fp32.pt"},
                clear=True,
            ),
            mock.patch.object(harness_dptp.harness_hook, "install_canonical_state_dump") as canon,
            mock.patch.object(harness_dptp, "install") as capture,
        ):
            harness_dptp.install_from_args(model, optim, _args(hash_output="/x/hash.json"))
        canon.assert_not_called()
        capture.assert_called_once()
        self.assertEqual(capture.call_args.kwargs["output_file"], "/x/hash.json")

    def test_no_output_is_strict_noop(self) -> None:
        from evals import harness_dptp

        model, optim = mock.MagicMock(), mock.MagicMock()
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch.object(harness_dptp.harness_hook, "install_canonical_state_dump") as canon,
            mock.patch.object(harness_dptp, "install") as capture,
        ):
            harness_dptp.install_from_args(model, optim, _args())
        canon.assert_not_called()
        capture.assert_called_once()
        self.assertIsNone(capture.call_args.kwargs["output_file"])


class TestImmediateCanonicalDump(unittest.TestCase):
    """``immediate=True`` dumps at install and never touches optimizer.step."""

    def test_immediate_dumps_at_install_without_hijack(self) -> None:
        from evals import harness_hook

        model, optim = mock.MagicMock(), mock.MagicMock()
        orig_step = optim.step
        harness_hook._INSTALLED[0] = False
        try:
            with (
                mock.patch.object(
                    harness_hook, "collect_canonical_state", return_value={}
                ) as collect,
                mock.patch.object(
                    harness_hook,
                    "_ordered_teardown_then_exit",
                    side_effect=SystemExit(0),
                ) as teardown,
                self.assertRaises(SystemExit),
            ):
                harness_hook.install_canonical_state_dump(
                    model,
                    optim,
                    output_file="/tmp/canonical_immediate_test.pt",
                    writer_rank_predicate=lambda: False,
                    immediate=True,
                )
            collect.assert_called_once()
            teardown.assert_called_once()
            self.assertIs(optim.step, orig_step)
        finally:
            harness_hook._INSTALLED[0] = False

    def test_default_still_hijacks_step(self) -> None:
        from evals import harness_hook

        model, optim = mock.MagicMock(), mock.MagicMock()
        orig_step = optim.step
        harness_hook._INSTALLED[0] = False
        try:
            with mock.patch.object(
                harness_hook, "collect_canonical_state", return_value={}
            ) as collect:
                harness_hook.install_canonical_state_dump(
                    model,
                    optim,
                    output_file="/tmp/canonical_hijack_test.pt",
                    writer_rank_predicate=lambda: False,
                )
            collect.assert_not_called()
            self.assertIsNot(optim.step, orig_step)
        finally:
            harness_hook._INSTALLED[0] = False


if __name__ == "__main__":
    unittest.main()
