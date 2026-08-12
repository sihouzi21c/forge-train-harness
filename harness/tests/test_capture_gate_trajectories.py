"""End-to-end capture-gate trajectories.

Drives REAL capture dumps (produced by the actual ``harness_hook`` forward /
backward / grad hooks) through the exact gate path the dispatcher uses —
:func:`harness.capture_artifacts.load_merged_capture` (per-rank merge) +
:func:`evals._capture_diff.capture_gate_outcome` (family filter →
no-overlap-fails → all-hash-equal verdict) — across every scenario an actual
forward/backward-align gate encounters. These are the trajectories the
narrow single-sided key assertions kept missing.
"""

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from torch import nn

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

from evals._capture_diff import capture_gate_outcome
from harness.capture_artifacts import load_merged_capture

FWD = "fwd."
BWD_GRAD = ("grad.", "bwd.")


@unittest.skipUnless(_HAS_TORCH, "torch required")
class TestCaptureGateTrajectories(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.d = Path(self._tmp.name)

    def _shared_model(self):
        class SharedTwice(nn.Module):
            def __init__(s):
                super().__init__()
                s.pre = nn.Linear(4, 4)
                s.shared = nn.Linear(4, 4)  # fired twice -> #0 / #1
                s.head = nn.Linear(4, 2)

            def forward(s, x):
                h = s.pre(x)
                a = s.shared(h)
                b = s.shared(a)
                return s.head(b)

        return SharedTwice()

    def _real_capture(self, rank: str = "0", input_seed: int = 1) -> dict:
        """Real dump dict with rank<rank>.mb0.fwd/bwd + rank<rank>.grad keys."""
        from evals.harness_hook._dump import _resolve_future_records
        from evals.harness_hook._grad_collector import collect_param_gradients
        from evals.harness_hook._module_hook import (
            register_module_forward_hooks,
            register_module_full_backward_hooks,
        )

        torch.manual_seed(0)  # identical weights every call
        model = self._shared_model()
        records: dict = {}
        graph: list = []
        fb = f"rank{rank}.mb0."
        register_module_forward_hooks(
            model,
            captured_records=records,
            captured_graph=graph,
            step_prefix_getter=lambda: fb,
            async_hash=False,
        )
        register_module_full_backward_hooks(
            model,
            captured_records=records,
            step_prefix_getter=lambda: fb,
            async_hash=False,
        )
        torch.manual_seed(input_seed)
        x = torch.randn(2, 4, requires_grad=True)
        model(x).sum().backward()
        collect_param_gradients(
            model,
            captured_records=records,
            allreduce="post",
            grad_attrs=("grad",),
            step_prefix_getter=lambda: f"rank{rank}.",
        )
        _resolve_future_records(records)  # any async futures -> plain records
        return records

    def _write(self, name: str, records: dict) -> Path:
        p = self.d / name
        p.write_text(json.dumps(records), encoding="utf-8")
        return p

    # ── forward-align trajectories ──────────────────────────────────────

    def test_forward_align_identical_passes(self):
        ref = self._real_capture()
        cand = self._real_capture()  # same seed -> identical hashes
        out = capture_gate_outcome(cand, ref, key_prefix=FWD)
        self.assertFalse(out.no_overlap)
        self.assertTrue(out.passed)
        self.assertGreater(out.total, 0)
        # the shared module's BOTH calls are gated
        names = {e.name for e in out.entries}
        self.assertIn("rank0.mb0.fwd.shared#0", names)
        self.assertIn("rank0.mb0.fwd.shared#1", names)

    def test_forward_align_hash_divergence_fails(self):
        ref = self._real_capture(input_seed=1)
        cand = self._real_capture(input_seed=2)  # different input -> different hashes
        out = capture_gate_outcome(cand, ref, key_prefix=FWD)
        self.assertFalse(out.no_overlap)  # keys line up
        self.assertFalse(out.passed)  # but hashes differ -> FAIL

    def test_forward_align_wrong_namespace_fails_not_silent_pass(self):
        # THE c0f7 trap: candidate emits bare `fwd.x` (no rank/mb namespace).
        # The family filter still recognises them as `fwd.`, but the KEYS do
        # not match the ref's namespaced keys -> zero overlap -> the gate must
        # FAIL, never silently "pass" nothing.
        ref = self._real_capture()  # rank0.mb0.fwd.*
        bare = {
            k.split(".fwd.")[-1] and f"fwd.{k.split('.fwd.')[-1]}": v
            for k, v in ref.items()
            if ".fwd." in k
        }
        out = capture_gate_outcome(bare, ref, key_prefix=FWD)
        self.assertTrue(out.no_overlap)
        self.assertFalse(out.passed)

    def test_forward_align_nonzero_returncode_fails(self):
        ref = self._real_capture()
        cand = self._real_capture()  # identical hashes
        out = capture_gate_outcome(cand, ref, key_prefix=FWD, returncode=1)
        self.assertFalse(out.no_overlap)
        self.assertFalse(out.passed)  # ours crashed -> FAIL despite matches

    # ── backward-align trajectories (ref-authoritative) ─────────────────

    def test_backward_align_identical_passes(self):
        ref = self._real_capture()
        cand = self._real_capture()
        out = capture_gate_outcome(cand, ref, key_prefix=BWD_GRAD, require_baseline_complete=True)
        self.assertFalse(out.no_overlap)
        self.assertTrue(out.passed)
        names = {e.name for e in out.entries}
        self.assertTrue(any(".bwd." in n for n in names))
        self.assertTrue(any(".grad." in n for n in names))

    def test_backward_align_candidate_omits_key_fails(self):
        # A candidate that hides a divergence by simply not emitting a bwd key
        # must FAIL under require_baseline_complete (ref is authoritative).
        ref = self._real_capture()
        cand = self._real_capture()
        dropped = next(k for k in cand if ".bwd." in k)
        del cand[dropped]
        out = capture_gate_outcome(cand, ref, key_prefix=BWD_GRAD, require_baseline_complete=True)
        self.assertFalse(out.no_overlap)
        self.assertFalse(out.passed)
        missing = [e for e in out.entries if e.name == dropped]
        self.assertTrue(missing and not missing[0].passed)

    # ── multi-rank trajectories (per-rank file merge) ───────────────────

    def test_multirank_forward_align_all_ranks_compared_and_pass(self):
        # ref and candidate each write per-rank shards; the gate merges them
        # and compares EVERY rank, not just rank 0.
        for rank in ("0", "1"):
            self._write(f"cap.json.rank{rank}", self._real_capture(rank=rank))
            self._write(f"ref.json.rank{rank}", self._real_capture(rank=rank))
        ref = load_merged_capture(self.d / "ref.json")
        cand = load_merged_capture(self.d / "cap.json")
        out = capture_gate_outcome(cand, ref, key_prefix=FWD)
        self.assertTrue(out.passed)
        names = {e.name for e in out.entries}
        self.assertTrue(any(n.startswith("rank0.") for n in names))
        self.assertTrue(any(n.startswith("rank1.") for n in names))

    def test_multirank_forward_align_one_rank_diverges_fails(self):
        self._write("ref.json.rank0", self._real_capture(rank="0", input_seed=1))
        self._write("ref.json.rank1", self._real_capture(rank="1", input_seed=1))
        self._write("cap.json.rank0", self._real_capture(rank="0", input_seed=1))
        # rank1 candidate diverges (different input)
        self._write("cap.json.rank1", self._real_capture(rank="1", input_seed=9))
        ref = load_merged_capture(self.d / "ref.json")
        cand = load_merged_capture(self.d / "cap.json")
        out = capture_gate_outcome(cand, ref, key_prefix=FWD)
        self.assertFalse(out.no_overlap)
        self.assertFalse(out.passed)  # rank1 divergence caught
        bad = [e for e in out.entries if e.name.startswith("rank1.") and not e.passed]
        self.assertTrue(bad, "rank1 divergence not surfaced")


if __name__ == "__main__":
    unittest.main()
