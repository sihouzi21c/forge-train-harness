"""Smoke tests for :mod:`evals.harness_hook` — the standard M1–M5 hook.

Covers the public contract that the dispatcher and any bridge depend
on:

* :func:`evals.harness_hook.resolve_dump_path` env routing.
* :func:`evals.harness_hook.install` strict no-op when ``output_file``
  is ``None`` (production-safety guarantee — the same bridge code
  doubles as a vanilla trainer when the gate env is absent).
* :func:`evals.harness_hook.install` single-step (``persistent=False``)
  end-to-end: registers forward + backward hooks, hijacks
  ``optimizer.step``, harvests grads, writes JSON hash dict +
  ``.graph.json``, runs ordered teardown, and exits cleanly via
  ``raise SystemExit(0)``.
* :func:`evals.harness_hook.install` persistent (``persistent=True``)
  end-to-end: returns a :class:`CaptureSession`; bridge drives
  ``begin_step / capture / capture_grads / dump`` and the resulting
  JSON contains the expected step-prefixed records.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from torch import nn

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from evals.harness_hook import hash_tensor, install, resolve_dump_path


class TestResolveDumpPath(unittest.TestCase):
    def test_absolute_env_value_wins(self):
        self.assertEqual(
            resolve_dump_path("/tmp/x.pt", "default.pt", ""),
            Path("/tmp/x.pt"),
        )

    def test_relative_value_resolves_against_dump_dir(self):
        self.assertEqual(
            resolve_dump_path("sub/file.pt", "default.pt", "/tmp/dd"),
            Path("/tmp/dd/sub/file.pt"),
        )

    def test_dump_dir_supplies_default_basename(self):
        self.assertEqual(
            resolve_dump_path("", "hook_dump.pt", "/tmp/dd"),
            Path("/tmp/dd/hook_dump.pt"),
        )

    def test_no_env_yields_none(self):
        self.assertIsNone(resolve_dump_path("", "hook_dump.pt", ""))


@unittest.skipUnless(_HAS_TORCH, "torch not installed in this environment")
class TestHashTensor(unittest.TestCase):
    """``hash_tensor`` discriminates byte-equality across dtype / shape."""

    def test_byte_equal_tensors_hash_equal(self):
        t1 = torch.arange(100, dtype=torch.float32)
        t2 = torch.arange(100, dtype=torch.float32)
        self.assertEqual(hash_tensor(t1)["hash"], hash_tensor(t2)["hash"])

    def test_single_bit_flip_hashes_differ(self):
        t1 = torch.arange(100, dtype=torch.float32)
        t2 = t1.clone()
        t2[0] = t2[0] + 1e-8
        self.assertNotEqual(hash_tensor(t1)["hash"], hash_tensor(t2)["hash"])

    def test_padded_vs_unpadded_hashes_differ(self):
        # Same content, different shape (vocab-padding scenario):
        # the comparator must surface this as FAIL (hash + shape).
        full = torch.arange(120, dtype=torch.float32).reshape(20, 6)
        trimmed = full[:18]
        r_full = hash_tensor(full)
        r_trim = hash_tensor(trimmed)
        self.assertNotEqual(r_full["hash"], r_trim["hash"])
        self.assertNotEqual(r_full["shape"], r_trim["shape"])

    def test_record_carries_shape_dtype(self):
        r = hash_tensor(torch.zeros(3, 5, dtype=torch.bfloat16))
        self.assertEqual(r["shape"], [3, 5])
        self.assertEqual(r["dtype"], "bfloat16")
        self.assertEqual(len(r["hash"]), 32)  # 128 bit hex


@unittest.skipUnless(_HAS_TORCH, "torch not installed in this environment")
class TestInstallNoOp(unittest.TestCase):
    """``install(..., output_file=None)`` must not touch model/optimizer state."""

    def test_output_file_none_is_strict_noop(self):
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        install(model, optimizer, output_file=None)
        # install() replaces optimizer.step by assigning to the instance dict;
        # in strict-noop mode that override must NOT be installed, so the
        # name resolves through the class method, not the instance dict.
        self.assertNotIn(
            "step",
            optimizer.__dict__,
            "optimizer.step must be untouched when capture is disabled",
        )


@unittest.skipUnless(_HAS_TORCH, "torch not installed in this environment")
class TestInstallSingleStep(unittest.TestCase):
    """Drive single-step ``install`` against a tiny ``nn.Module`` and observe the dump."""

    def setUp(self):
        # Reset module-level singletons so multiple test methods can
        # exercise install() in sequence within one process.
        from evals.harness_hook import (
            _CAPTURED_GRAPH,
            _CAPTURED_RECORDS,
            _INSTALLED,
        )

        _CAPTURED_RECORDS.clear()
        _CAPTURED_GRAPH.clear()
        _INSTALLED[0] = False
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out_path = Path(self._tmp.name) / "ref_capture.json"

    def test_capture_dumps_fwd_grad_and_graph_then_exits(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 3, bias=False)
                self.fc2 = nn.Linear(3, 2, bias=False)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        model = Tiny()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        install(
            model,
            optimizer,
            output_file=self.out_path,
            writer_rank_predicate=lambda: True,
        )

        x = torch.randn(2, 4)
        y = torch.randn(2, 2)
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()

        # Clean-exit contract: the hook runs ordered teardown and raises
        # SystemExit(0) rather than os._exit(0). SystemExit is a
        # BaseException, so a framework's `except Exception` cannot swallow
        # it, and normal interpreter shutdown (atexit, flushes) still runs.
        with self.assertRaises(SystemExit) as ctx:
            optimizer.step()
        self.assertEqual(ctx.exception.code, 0)

        self.assertTrue(self.out_path.is_file(), "capture dump did not write file")
        graph_path = self.out_path.with_name(self.out_path.name + ".graph.json")
        self.assertTrue(graph_path.is_file(), "graph.json sibling missing")

        captured = json.loads(self.out_path.read_text(encoding="utf-8"))
        # Single-step alignment keys carry the uniform ``rank<r>.`` prefix
        # (r=0 single process); module fwd/bwd also carry ``mb0.`` and the
        # per-forward ``#<call>`` suffix; post-step grads carry ``rank0.`` only.
        self.assertIn("rank0.mb0.fwd.fc1#0", captured)
        self.assertIn("rank0.mb0.fwd.fc2#0", captured)
        self.assertIn("rank0.grad.fc1.weight.postallreduce", captured)
        self.assertIn("rank0.grad.fc2.weight.postallreduce", captured)
        for rec in captured.values():
            self.assertIn("hash", rec)
            self.assertIn("shape", rec)
            self.assertIn("dtype", rec)
            self.assertEqual(len(rec["hash"]), 32)

        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        fqns = [e["fqn"] for e in graph]
        # Execution order must contain fc1 *before* fc2 (forward order).
        self.assertEqual(fqns, ["fc1", "fc2"])
        for entry in graph:
            self.assertIn("class", entry)
            self.assertIn("input_shapes", entry)
            self.assertIn("output_shape", entry)

    def test_non_writer_rank_skips_file_but_still_exits(self):
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        install(
            model,
            optimizer,
            output_file=self.out_path,
            writer_rank_predicate=lambda: False,
        )

        loss = model(torch.randn(2, 4)).sum()
        loss.backward()

        with self.assertRaises(SystemExit) as ctx:
            optimizer.step()
        self.assertEqual(ctx.exception.code, 0)

        self.assertFalse(
            self.out_path.exists(),
            "non-writer ranks must not produce the dump file",
        )


@unittest.skipUnless(_HAS_TORCH, "torch not installed in this environment")
class TestInstallPersistent(unittest.TestCase):
    """Drive ``install(..., persistent=True)`` for M2–M5 multi-step hash dumps."""

    def setUp(self):
        from evals.harness_hook import _INSTALLED

        _INSTALLED[0] = False
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out_path = Path(self._tmp.name) / "ours_hash.json"

    def test_persistent_session_records_step_prefixed_keys(self):
        model = nn.Linear(4, 2, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        session = install(
            model,
            optimizer,
            output_file=self.out_path,
            hash_capture_level=1,  # loss + grad only, no module hooks
            persistent=True,
            writer_rank_predicate=lambda: True,
        )
        self.assertIsNotNone(session)

        # Simulate two training steps with pre / post all-reduce captures.
        for step in range(2):
            session.begin_step(step)
            x = torch.randn(2, 4)
            y = torch.randn(2, 2)
            loss_per_token = ((model(x) - y) ** 2).sum(dim=-1)  # [B]
            session.capture("loss.per_token.preallreduce", loss_per_token)
            loss = loss_per_token.mean()
            session.capture("loss.scalar.preallreduce", loss)
            loss.backward()
            session.capture_grads(allreduce="pre")
            # (no actual all-reduce since world_size=1 — just exercise the API)
            session.capture("loss.scalar.postallreduce", loss)
            session.capture_grads(allreduce="post")
            optimizer.step()
            optimizer.zero_grad()

        session.dump()

        self.assertTrue(self.out_path.is_file())
        records = json.loads(self.out_path.read_text(encoding="utf-8"))
        for step in (0, 1):
            self.assertIn(f"step_{step}.rank0.loss.per_token.preallreduce", records)
            self.assertIn(f"step_{step}.rank0.loss.scalar.preallreduce", records)
            self.assertIn(f"step_{step}.rank0.loss.scalar.postallreduce", records)
            self.assertIn(f"step_{step}.rank0.grad.weight.preallreduce", records)
            self.assertIn(f"step_{step}.rank0.grad.weight.postallreduce", records)

    def test_persistent_microbatch_namespaced_keys(self):
        # Two grad-accum microbatches in one step. Per-microbatch captures
        # (loss.per_token) must be namespaced ``step_<n>.mb<i>.`` and stay
        # distinct across microbatches; per-step captures issued AFTER
        # ``end_microbatch`` (loss.scalar, grads) must carry NO ``mb`` prefix.
        model = nn.Linear(4, 2, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        session = install(
            model,
            optimizer,
            output_file=self.out_path,
            hash_capture_level=1,  # loss + grad only, no module hooks
            persistent=True,
            writer_rank_predicate=lambda: True,
        )
        self.assertIsNotNone(session)

        session.begin_step(0)
        for mb in range(2):  # grad_accum = 2
            session.begin_microbatch(mb)
            x = torch.randn(2, 4)
            y = torch.randn(2, 2)
            loss_per_token = ((model(x) - y) ** 2).sum(dim=-1)
            session.capture("loss.per_token.preallreduce", loss_per_token)
            loss_per_token.mean().backward()  # accumulate grad across microbatches
        session.end_microbatch()
        session.capture("loss.scalar.preallreduce", torch.tensor(1.0))
        session.capture_grads(allreduce="pre")
        session.capture_grads(allreduce="post")
        session.dump()

        records = json.loads(self.out_path.read_text(encoding="utf-8"))
        # per-microbatch captures distinct under the rank+mb namespace
        self.assertIn("step_0.rank0.mb0.loss.per_token.preallreduce", records)
        self.assertIn("step_0.rank0.mb1.loss.per_token.preallreduce", records)
        # per-step captures after end_microbatch carry rank but NO mb prefix
        self.assertIn("step_0.rank0.loss.scalar.preallreduce", records)
        self.assertIn("step_0.rank0.grad.weight.preallreduce", records)
        self.assertIn("step_0.rank0.grad.weight.postallreduce", records)
        self.assertNotIn("step_0.rank0.mb1.grad.weight.preallreduce", records)
        self.assertNotIn("step_0.rank0.mb0.loss.scalar.preallreduce", records)

    def test_begin_step_resets_microbatch_namespace(self):
        # A new step clears any lingering microbatch namespace so the first
        # per-step capture of the next step is not accidentally mb-prefixed.
        model = nn.Linear(4, 2, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        session = install(
            model,
            optimizer,
            output_file=self.out_path,
            hash_capture_level=1,
            persistent=True,
            writer_rank_predicate=lambda: True,
        )
        session.begin_step(0)
        session.begin_microbatch(0)
        session.capture("loss.per_token.preallreduce", torch.tensor(0.5))
        session.begin_step(1)  # new step — must clear the mb namespace
        session.capture("loss.scalar.preallreduce", torch.tensor(0.5))
        session.dump()
        records = json.loads(self.out_path.read_text(encoding="utf-8"))
        self.assertIn("step_0.rank0.mb0.loss.per_token.preallreduce", records)
        self.assertIn("step_1.rank0.loss.scalar.preallreduce", records)
        self.assertNotIn("step_1.rank0.mb0.loss.scalar.preallreduce", records)

    def test_persistent_level_zero_is_silent_noop(self):
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        session = install(
            model,
            optimizer,
            output_file=self.out_path,
            hash_capture_level=0,  # all session methods become no-ops
            persistent=True,
        )
        self.assertIsNotNone(session)

        # Drive the session — at level 0 nothing should be captured.
        session.begin_step(0)
        session.capture("loss.scalar.preallreduce", torch.tensor(0.5))
        session.capture_grads(allreduce="pre")
        session.dump()

        # No file written when level==0.
        self.assertFalse(self.out_path.exists())


@unittest.skipUnless(_HAS_TORCH, "torch required")
class TestPerFireIndexing(unittest.TestCase):
    """Every hook fire is captured with a per-(step, fqn) ``#<idx>`` suffix.

    Locks the fix for the shared-module / grad-accum blind spot: a module
    re-used multiple times in one step (or fired once per microbatch) must
    record EVERY fire, not just the first.
    """

    def _shared_model(self):
        from torch import nn

        class SharedTwice(nn.Module):
            def __init__(self):
                super().__init__()
                self.pre = nn.Linear(4, 4)
                self.shared = nn.Linear(4, 4)
                self.head = nn.Linear(4, 2)

            def forward(self, x):
                # ``pre`` makes the first ``shared`` input require grad, so
                # BOTH shared invocations produce a real grad_input and the
                # backward hook fires twice (mirrors an output head shared
                # by the main + MTP paths).
                h = self.pre(x)
                a = self.shared(h)
                b = self.shared(a)  # same submodule fired a 2nd time
                return self.head(b)

        return SharedTwice()

    def test_shared_module_forward_records_both_fires(self):
        from evals.harness_hook._module_hook import register_module_forward_hooks

        model = self._shared_model()
        records: dict = {}
        graph: list = []
        register_module_forward_hooks(model, captured_records=records, captured_graph=graph)
        model(torch.randn(2, 4))
        self.assertIn("fwd.shared#0", records)
        self.assertIn("fwd.shared#1", records)  # 2nd fire NOT dropped
        self.assertIn("fwd.head#0", records)

    def test_shared_module_backward_records_both_fires(self):
        from evals.harness_hook._module_hook import (
            register_module_full_backward_hooks,
        )

        model = self._shared_model()
        records: dict = {}
        register_module_full_backward_hooks(model, captured_records=records)
        loss = model(torch.randn(2, 4)).sum()
        loss.backward()
        self.assertIn("bwd.shared#0", records)
        self.assertIn("bwd.shared#1", records)  # both dgrad paths kept

    def test_index_resets_per_step_prefix(self):
        from evals.harness_hook._module_hook import register_module_forward_hooks

        model = self._shared_model()
        records: dict = {}
        graph: list = []
        step = {"n": 0}
        register_module_forward_hooks(
            model,
            captured_records=records,
            captured_graph=graph,
            step_prefix_getter=lambda: f"step_{step['n']}.",
        )
        model(torch.randn(2, 4))
        step["n"] = 1
        model(torch.randn(2, 4))
        # Per-step namespacing: each step's shared fires restart at #0.
        self.assertIn("step_0.fwd.shared#0", records)
        self.assertIn("step_0.fwd.shared#1", records)
        self.assertIn("step_1.fwd.shared#0", records)
        self.assertIn("step_1.fwd.shared#1", records)

    def test_call_index_resets_per_microbatch(self):
        # mb and call are INDEPENDENT axes: microbatch is a ``mb<i>.`` prefix,
        # the per-forward call is the ``#<call>`` suffix. With a
        # microbatch-aware prefix the call index restarts each microbatch —
        # NOT a single flat counter that conflates the two.
        from evals.harness_hook._module_hook import register_module_forward_hooks

        model = self._shared_model()
        records: dict = {}
        graph: list = []
        state = {"mb": 0}
        register_module_forward_hooks(
            model,
            captured_records=records,
            captured_graph=graph,
            step_prefix_getter=lambda: f"step_0.mb{state['mb']}.",
        )
        model(torch.randn(2, 4))  # mb0: shared fires twice -> #0, #1
        state["mb"] = 1
        model(torch.randn(2, 4))  # mb1: call index restarts at #0
        self.assertIn("step_0.mb0.fwd.shared#0", records)
        self.assertIn("step_0.mb0.fwd.shared#1", records)
        self.assertIn("step_0.mb1.fwd.shared#0", records)
        self.assertIn("step_0.mb1.fwd.shared#1", records)


@unittest.skipUnless(_HAS_TORCH, "torch required")
class TestAsyncHash(unittest.TestCase):
    """Hook hashing is dispatched off-thread; digests are unchanged and
    resolved at dump time."""

    def test_async_digest_matches_sync(self):
        from evals.harness_hook._module_hook import _hash_or_submit

        t = torch.randn(3, 5)
        sync = _hash_or_submit(t, async_hash=False)
        fut = _hash_or_submit(t, async_hash=True)
        self.assertTrue(hasattr(fut, "result"), "async path must return a Future")
        self.assertEqual(fut.result(), sync, "async digest must equal sync digest")

    def test_dump_resolves_futures(self):
        from evals.harness_hook._dump import dump_capture_files
        from evals.harness_hook._module_hook import _hash_or_submit

        records = {"fwd.x#0": _hash_or_submit(torch.randn(2, 2), async_hash=True)}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "cap.json"
            dump_capture_files(out, records, [], is_writer=True)
            data = json.loads(out.read_text(encoding="utf-8"))
        self.assertIn("fwd.x#0", data)
        self.assertIn("hash", data["fwd.x#0"])  # Future was resolved to a record


class TestAllRankDump(unittest.TestCase):
    """Every rank persists its own ``<file>.rank<N>``; rank0 also writes the
    canonical unsuffixed file."""

    def _records(self):
        return {"fwd.x#0": {"hash": "a", "shape": [1], "dtype": "float32"}}

    def test_writer_rank_writes_canonical_and_rankfile(self):
        import os
        from unittest import mock

        from evals.harness_hook._dump import dump_capture_files

        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "cap.json"
            with mock.patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "2"}):
                dump_capture_files(out, self._records(), [], is_writer=True)
            self.assertTrue(out.is_file(), "canonical file (back-compat)")
            self.assertTrue((Path(d) / "cap.json.rank0").is_file(), "rank0 file")

    def test_nonwriter_rank_writes_only_its_rankfile(self):
        import os
        from unittest import mock

        from evals.harness_hook._dump import dump_capture_files

        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "cap.json"
            with mock.patch.dict(os.environ, {"RANK": "1", "WORLD_SIZE": "2"}):
                dump_capture_files(out, self._records(), [], is_writer=False)
            self.assertFalse(out.is_file(), "non-writer must not write canonical")
            self.assertTrue(
                (Path(d) / "cap.json.rank1").is_file(),
                "non-writer rank still persists its own capture",
            )


@unittest.skipUnless(_HAS_TORCH, "torch required")
class TestClosedLoopDiff(unittest.TestCase):
    """End-to-end closed loop: two identically-driven dumps go through the
    real merge loader (:func:`load_merged_capture`) + comparator
    (:func:`diff_capture_dicts`); every key overlaps and passes. This is the
    loop the earlier single-sided key assertions could not verify.
    """

    def _run_persistent(self, out_path):
        from evals.harness_hook import _INSTALLED

        _INSTALLED[0] = False
        torch.manual_seed(1234)  # identical weights both runs
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        session = install(
            model,
            optimizer,
            output_file=out_path,
            hash_capture_level=2,
            persistent=True,
            writer_rank_predicate=lambda: True,
        )
        torch.manual_seed(42)  # identical inputs both runs
        session.begin_step(0)
        for mb in range(2):
            session.begin_microbatch(mb)
            model(torch.randn(2, 4)).sum().backward()
        session.end_microbatch()
        session.capture_grads(allreduce="post")
        session.dump()

    def test_ref_candidate_full_overlap_and_pass(self):
        from evals._capture_diff import diff_capture_dicts
        from harness.capture_artifacts import load_merged_capture

        with tempfile.TemporaryDirectory() as d:
            ref_p = Path(d) / "ref.json"
            cand_p = Path(d) / "cand.json"
            self._run_persistent(ref_p)
            self._run_persistent(cand_p)
            ref = load_merged_capture(ref_p)
            cand = load_merged_capture(cand_p)

        self.assertGreater(len(ref), 0)
        # keys carry the full step_<n>.rank<r>.mb<m>.fwd.<fqn>#<call> grammar
        self.assertTrue(
            any(k.startswith("step_0.rank0.mb0.fwd.") and "#" in k for k in ref),
            f"no rank/mb/call-shaped key in {sorted(ref)[:4]}",
        )
        # closed loop through the real comparator: every key overlaps and passes
        entries = diff_capture_dicts(cand, ref)
        self.assertEqual(len(entries), len(ref), "comparator dropped keys")
        failed = [e.name for e in entries if not e.passed]
        self.assertFalse(failed, f"comparator FAIL on {failed[:5]}")

        # forward-align filters by the ``fwd.`` FAMILY, which now sits mid-key
        # after the rank/mb namespace (``step_0.rank0.mb0.fwd.x#0``). The
        # family filter must still match it — a plain ``startswith('fwd.')``
        # would drop every key and silently gate on nothing.
        fwd = diff_capture_dicts(cand, ref, key_prefix="fwd.")
        self.assertTrue(fwd, "fwd. family filter matched nothing on namespaced keys")
        self.assertTrue(all(".fwd." in e.name for e in fwd))
        self.assertTrue(all(e.passed for e in fwd))
        # backward-align passes ('grad.', 'bwd.') — both families present.
        bg = diff_capture_dicts(cand, ref, key_prefix=("grad.", "bwd."))
        self.assertTrue(any(".grad." in e.name for e in bg))
        self.assertTrue(any(".bwd." in e.name for e in bg))


if __name__ == "__main__":
    unittest.main()
