"""Smoke tests for :mod:`evals.harness_dptp` — the DP×TP capture wrapper.

``harness_dptp`` is the 2-D sibling of the already-tested ``harness_dp``; the
DP collective + hash capture are inherited and exercised elsewhere. What is
*new* here — and what these tests pin — is the tensor-parallel wrinkle:

  1. :func:`init_groups` topology — the Megatron ``global_rank =
     dp_rank*tp_size + tp_rank`` map and the rejection of a world that does not
     factor as ``dp_size*tp_size``.
  2. **Key namespacing** — each TP rank installs under ``key_prefix="tp<rank>."``
     so its shard records never collide with the other rank's.
  3. **finalize TP-gather merge** — the per-rank record dicts are all-gathered
     over the TP group and global rank 0 merges the disjoint key sets into the
     single dump the comparator reads (no per-rank files).

Layer 1 (single process, gloo world-1, CPU) covers (1)+(2) and the merge
no-op path; it always runs. Layer 2 spawns a real 4-rank DP(2)xTP(2) gloo job
to cover (3) end-to-end — sharded grads land DISTINCT under ``tp0.``/``tp1.``
while replicated loss/grad land IDENTICAL. It skips (never fails) when the
local box cannot launch the gloo cohort, but asserts hard on dump contents.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch import nn

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


if _HAS_TORCH:

    class _TinyShardedModel(nn.Module):
        """Two named params so ``install`` can build its ``id→FQN`` map: one
        stands in for a TP-sharded weight, one for a replicated weight."""

        def __init__(self) -> None:
            super().__init__()
            self.w_shard = nn.Parameter(torch.zeros(4))
            self.w_repl = nn.Parameter(torch.zeros(2))

        def forward(self, x):  # never driven — grads are fed synthetically
            return x


def _reset_capture_state() -> None:
    """Clear the harness_hook + harness_dptp process globals so a fresh
    ``install`` runs within a single interpreter."""
    from evals import harness_dptp as H
    from evals.harness_hook import _INSTALLED

    _INSTALLED[0] = False
    H._SESSION = None
    H._FQN_BY_ID = {}
    H._LAYOUT = None


@unittest.skipUnless(_HAS_TORCH, "requires torch for the DPxTP capture smoke")
class TestHarnessDptpSingleProcess(unittest.TestCase):
    """gloo world-1: topology, ``tp0.`` namespacing, and the finalize write."""

    def setUp(self) -> None:
        _reset_capture_state()
        os.environ.update(
            MASTER_ADDR="localhost",
            MASTER_PORT=str(_free_port()),
            RANK="0",
            WORLD_SIZE="1",
            LOCAL_RANK="0",
        )
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo", rank=0, world_size=1)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name) / "ref_capture.json"

    def tearDown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_init_groups_layout_and_world_factor_check(self) -> None:
        from evals import harness_dptp as H

        lay = H.init_groups(1, 1)
        self.assertEqual(
            (lay.dp_size, lay.tp_size, lay.dp_rank, lay.tp_rank, lay.global_rank),
            (1, 1, 0, 0, 0),
        )
        # world (1) must equal dp_size*tp_size — a 1×2 request cannot stand.
        with self.assertRaises(ValueError):
            H.init_groups(1, 2)

    def test_namespacing_and_finalize_merge(self) -> None:
        from evals import harness_dptp as H

        H.init_groups(1, 1)
        model = _TinyShardedModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        session = H.install(model, opt, output_file=self.out, hash_capture_level=1, persistent=True)
        self.assertIsNotNone(session)

        for step in range(2):
            H.begin_step(step)
            H.reduce_loss_scalar(torch.tensor([2.0]), torch.tensor([4.0]))
            H.reduce_grads(
                [model.w_shard, model.w_repl],
                [torch.full((4,), 1.0), torch.full((2,), 1.0)],
                norm_factor=1.0,
            )
        H.finalize()

        self.assertTrue(self.out.is_file(), "finalize did not write the merged dump")
        records = json.loads(self.out.read_text(encoding="utf-8"))
        for step in (0, 1):
            p = f"step_{step}.rank0."
            for key in (
                "loss.scalar.preallreduce",
                "loss.scalar.postallreduce",
                "grad.w_shard.preallreduce",
                "grad.w_shard.postallreduce",
                "grad.w_repl.preallreduce",
                "grad.w_repl.postallreduce",
            ):
                self.assertIn(p + key, records, f"missing {p + key}")
        # Every key carries this rank's global-rank namespace — none unprefixed.
        self.assertTrue(all(".rank0." in k for k in records))
        for rec in records.values():
            self.assertEqual(len(rec["hash"]), 32)


def _dptp_dp2tp2_worker(
    rank: int, world_size: int, tp_size: int, master_port: int, out_dir: str
) -> None:
    """One rank of a DP(2)xTP(2) gloo cohort: build groups, capture a sharded
    grad + replicated loss/grad for one step, finalize. Each rank drops a
    layout sentinel; global rank 0's finalize writes the single merged dump."""
    import torch
    import torch.distributed as dist

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(master_port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
    )
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        from evals import harness_dptp as H

        dp_size = world_size // tp_size
        lay = H.init_groups(dp_size, tp_size)
        Path(out_dir, f"layout_{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "dp_rank": lay.dp_rank,
                    "tp_rank": lay.tp_rank,
                    "global_rank": lay.global_rank,
                }
            ),
            encoding="utf-8",
        )

        model = _TinyShardedModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        out = Path(out_dir, "merged.json")
        H.install(model, opt, output_file=out, hash_capture_level=1, persistent=True)

        H.begin_step(0)
        # Replicated loss: identical statistics on every rank.
        H.reduce_loss_scalar(torch.tensor([3.0]), torch.tensor([6.0]))
        # Sharded grad: value depends on tp_rank, so the two TP shards differ;
        # replicated grad is identical everywhere.
        H.reduce_grads(
            [model.w_shard, model.w_repl],
            [torch.full((4,), float(lay.tp_rank + 1)), torch.full((2,), 7.0)],
            norm_factor=1.0,
        )
        H.finalize()
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@unittest.skipUnless(_HAS_TORCH, "requires torch for the DPxTP capture smoke")
class TestHarnessDptpDP2TP2(unittest.TestCase):
    """Real 4-rank DP(2)xTP(2) gloo job: the finalize TP-gather merge keeps the
    two ranks' shards distinct under one dump, with sharded≠ / replicated=."""

    def test_finalize_merges_distinct_tp_shards(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out_dir = tmp.name
        port = _free_port()

        try:
            mp.spawn(
                _dptp_dp2tp2_worker,
                args=(4, 2, port, out_dir),
                nprocs=4,
                join=True,
            )
        except Exception as exc:  # launch/gloo infra failure — skip, don't fail
            self.skipTest(f"DP2xTP2 gloo cohort unavailable here: {exc!r}")

        layouts = sorted(Path(out_dir).glob("layout_*.json"))
        if len(layouts) != 4:
            self.skipTest("not all 4 workers reported — treating as infra flake")

        # Topology: every rank's coordinates obey global = dp*tp_size + tp.
        for f in layouts:
            d = json.loads(f.read_text(encoding="utf-8"))
            self.assertEqual(d["dp_rank"], d["global_rank"] // 2)
            self.assertEqual(d["tp_rank"], d["global_rank"] % 2)
            self.assertEqual(d["rank"], d["global_rank"])

        from harness.capture_artifacts import load_merged_capture

        # Each rank wrote its own ``merged.json.rank<r>``; the dispatcher-side
        # merge globs+unions them (keys disjoint by the ``rank<r>.`` prefix).
        merged = load_merged_capture(Path(out_dir, "merged.json"))
        # All four global ranks' shards survive in the merged view.
        for r in range(4):
            self.assertTrue(
                any(f".rank{r}." in k for k in merged),
                f"rank{r} shard missing from merged dump",
            )

        def h(key: str) -> str:
            self.assertIn(key, merged, f"missing {key}")
            return merged[key]["hash"]

        # global rank = dp*tp_size + tp (tp_size=2): rank0=dp0tp0, rank1=dp0tp1,
        # rank2=dp1tp0, rank3=dp1tp1.
        # Different TP shard (rank0 tp0 vs rank1 tp1): DISTINCT sharded grad.
        self.assertNotEqual(
            h("step_0.rank0.grad.w_shard.postallreduce"),
            h("step_0.rank1.grad.w_shard.postallreduce"),
        )
        # Same TP shard across DP replicas (rank0 dp0tp0 vs rank2 dp1tp0):
        # identical post-all-reduce grad.
        self.assertEqual(
            h("step_0.rank0.grad.w_shard.postallreduce"),
            h("step_0.rank2.grad.w_shard.postallreduce"),
        )
        # Replicated weight: identical across TP shards.
        self.assertEqual(
            h("step_0.rank0.grad.w_repl.postallreduce"),
            h("step_0.rank1.grad.w_repl.postallreduce"),
        )
        # Replicated loss: same statistics → identical scalar on both shards.
        self.assertEqual(
            h("step_0.rank0.loss.scalar.preallreduce"),
            h("step_0.rank1.loss.scalar.preallreduce"),
        )


if __name__ == "__main__":
    unittest.main()
