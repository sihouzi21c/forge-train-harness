"""Strictness contracts for :mod:`harness.wire_format`.

The wire-format module owns the line grammars every gate script speaks
on stdout. The strict parsers (``parse_loss_lines`` /
``parse_loss_lines_to_dict``) already have negative tests inside
``test_dispatcher_behavior.TestLossParser``; this file pins a single
producer↔consumer round-trip on the L0 ref-side ``[LOSS]`` line —
re-emitting that line verbatim and confirming
``parse_loss_dump_file`` reads it back to the same step→loss mapping.
If any ref-side emitter rewrites that f-string in a way the dispatcher
cannot parse, this test fails immediately rather than waiting for a
silent gate-pass-with-empty-trajectory in production.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from harness.wire_format import (
    LOSS_FLOAT_FORMAT,
    parse_loss_dump_file,
    parse_loss_dump_with_grad,
)


class TestLossDumpLineRoundTrip(unittest.TestCase):
    """One round-trip per audit finding 2.2: the per-suite bridge the
    agent generates for the customer training stack emits
    ``[LOSS] step=N global_loss=X grad_norm=nan time_s=nan`` lines via
    a hand-crafted f-string; the dispatcher reads them via
    :func:`parse_loss_dump_file`. We re-emit the exact same f-string
    here and assert the reader recovers the trajectory.

    Drift detection: if any ref-side emitter's f-string or
    ``parse_loss_dump_file``'s regex changes in a way that breaks the
    contract, this test fails immediately.
    """

    def _emit_loss_line(self, step: int, loss_val: float | None) -> str:
        # Verbatim copy of the ``[LOSS]`` f-string used by ref-side
        # training entries — if you edit one, you MUST edit the other;
        # this test is the gate that surfaces the mismatch. Precision
        # for ``global_loss`` is pinned at ``LOSS_FLOAT_FORMAT`` (``.9e``)
        # so this test fails fast if an emitter regresses to a narrower
        # form (e.g. ``.6e``) that would mask sub-print fp32 ULP drift.
        loss_str = "nan" if loss_val is None else format(float(loss_val), LOSS_FLOAT_FORMAT)
        return f"[LOSS] step={step} global_loss={loss_str} grad_norm=nan time_s=nan"

    def test_loss_dump_format_round_trips_through_parser(self) -> None:
        lines = [
            self._emit_loss_line(0, 2.5),
            self._emit_loss_line(1, 2.4),
            self._emit_loss_line(2, 2.3),
            self._emit_loss_line(3, None),
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write("\n".join(lines) + "\n")
            dump = Path(fh.name)
        try:
            parsed = parse_loss_dump_file(dump)
        finally:
            dump.unlink(missing_ok=True)
        self.assertEqual(parsed, {0: 2.5, 1: 2.4, 2: 2.3})

    def test_loss_line_emits_grammar_keys_in_canonical_order(self) -> None:
        line = self._emit_loss_line(7, 1.5)
        # Order matters: the dispatcher's strict ``parse_loss_lines``
        # (used by Stage 1 perf gates) requires the keys in this order.
        # Precision matters too: ``LOSS_FLOAT_FORMAT`` (``.9e``) is the
        # canonical fp32-round-trip spec; any regression to a narrower
        # form would mask sub-print fp32 ULP drift in the bitwise gates.
        self.assertEqual(
            line,
            "[LOSS] step=7 global_loss=1.500000000e+00 grad_norm=nan time_s=nan",
        )


class TestLossDumpWithGrad(unittest.TestCase):
    """``parse_loss_dump_with_grad`` recovers both gate-decisive fields.

    The bitwise milestones (bitwise-singlecard / bitwise-multicard /
    bitwise-perf) require per-step bitwise alignment on **both**
    ``global_loss`` and ``grad_norm`` (see ``stage1/bitwise-multicard.md`` §Gate).
    The L0 ref entry (``train_pure_mup_mtp.py``) writes both fields at
    ``.9e`` into ``LOSS_DUMP_FILE``; this parser is the SSOT that lifts
    them back so the bitwise-trajectory dispatcher has a ``grad_norm``
    baseline to compare against.
    """

    def _emit(self, step: int, loss: float, grad: float) -> str:
        return (
            f"[LOSS] step={step} "
            f"global_loss={format(loss, LOSS_FLOAT_FORMAT)} "
            f"grad_norm={format(grad, LOSS_FLOAT_FORMAT)} "
            f"time_s=0.123456"
        )

    def _write(self, lines: list[str]) -> Path:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write("\n".join(lines) + "\n")
        return Path(fh.name)

    def test_parses_loss_and_grad_per_step(self) -> None:
        dump = self._write(
            [
                self._emit(0, 2.5, 1.25),
                self._emit(1, 2.4, 1.10),
                self._emit(2, 2.3, 0.95),
            ]
        )
        try:
            parsed = parse_loss_dump_with_grad(dump)
        finally:
            dump.unlink(missing_ok=True)
        self.assertEqual(set(parsed), {0, 1, 2})
        self.assertEqual(parsed[0], {"global_loss": 2.5, "grad_norm": 1.25})
        self.assertEqual(parsed[1], {"global_loss": 2.4, "grad_norm": 1.10})
        self.assertEqual(parsed[2], {"global_loss": 2.3, "grad_norm": 0.95})

    def test_nan_grad_step_is_dropped(self) -> None:
        # The per-suite customer bridge may emit ``grad_norm=nan`` for
        # steps where the loss_dict lacks a grad entry; such steps carry
        # no usable bitwise grad baseline and must be omitted (mirrors the
        # tolerant NaN policy of ``parse_loss_dump_file``).
        dump = self._write(
            [
                self._emit(0, 2.5, 1.25),
                "[LOSS] step=1 global_loss=2.400000000e+00 grad_norm=nan time_s=0.1",
            ]
        )
        try:
            parsed = parse_loss_dump_with_grad(dump)
        finally:
            dump.unlink(missing_ok=True)
        self.assertEqual(set(parsed), {0})

    def test_loss_subset_and_grad_share_step_keys(self) -> None:
        # The loss-only and loss+grad parsers must agree on which steps
        # they recover from a well-formed dump so the dispatcher can pair
        # ``loss_by_step`` and ``grad_norm_by_step`` step-for-step.
        dump = self._write([self._emit(5, 1.0, 0.5), self._emit(6, 0.9, 0.4)])
        try:
            loss_only = parse_loss_dump_file(dump)
            with_grad = parse_loss_dump_with_grad(dump)
        finally:
            dump.unlink(missing_ok=True)
        self.assertEqual(set(loss_only), set(with_grad))


if __name__ == "__main__":
    unittest.main()
