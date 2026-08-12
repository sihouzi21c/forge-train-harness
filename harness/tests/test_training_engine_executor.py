"""CPU contract tests for ``training_engine_tensor.executor``.

Smoke-cover the static-graph forward+backward chain end-to-end on a
tiny CPU shape with the GQA math fallback enabled (the GPU + flash-attn
bitwise route is exercised by the ``forward-align`` /
``backward-align`` harness suites).

Gated on ``HARNESS_RUN_ENGINE_CONTRACTS=1`` and ``training_engine_tensor``
being importable, mirroring the rest of the engine-contract suite (see
``test_training_engine_forward.py`` for the rationale).
"""

from __future__ import annotations

import importlib
import os
import sys
import unittest
from pathlib import Path


def _ensure_engine_on_path() -> None:
    workspace = Path(__file__).resolve().parents[2]
    candidate = workspace / "workload" / "src"
    if candidate.is_dir():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


_ensure_engine_on_path()


def _can_import(name: str) -> bool:
    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True


@unittest.skipUnless(
    os.environ.get("HARNESS_RUN_ENGINE_CONTRACTS") == "1",
    "engine-contract tests gated on HARNESS_RUN_ENGINE_CONTRACTS=1",
)
@unittest.skipUnless(
    _can_import("training_engine_tensor.executor"),
    "training_engine_tensor.executor not importable",
)
class ExecutorEndToEndTests(unittest.TestCase):
    """End-to-end ``forward_capture`` -> ``backward_full`` smoke."""

    def test_forward_backward_smoke(self) -> None:
        import torch
        from training_engine_tensor import config as eng_config
        from training_engine_tensor.executor import (
            _masked_ce_forward_backward,
            backward_full,
            build_mtp_tensors,
            forward_capture,
        )
        from training_engine_tensor.forward import precompute_rope_freqs
        from training_engine_tensor.model import ModelState, MupConfig

        torch.manual_seed(0)
        mup = MupConfig(eagle_num_layers=1)
        state = ModelState(mup)
        # FP32 dtype so the cpu math fallback runs at full precision
        # and shape contracts are unaffected by bf16 absence on CPU.
        state.allocate(device="cpu", dtype=torch.float32)
        state.init_muP(init_std=0.1, seed=0)

        batch, seq = 1, 4
        input_ids = torch.randint(0, eng_config.VOCAB_SIZE, (batch, seq), dtype=torch.int64)
        labels = torch.randint(0, eng_config.VOCAB_SIZE, (batch, seq), dtype=torch.int64)
        loss_mask = torch.ones(batch, seq, dtype=torch.float32)
        mtp_in, mtp_lab, mtp_mask = build_mtp_tensors(input_ids, labels, loss_mask)
        rope = precompute_rope_freqs(seq, device="cpu")

        captured: dict[str, tuple[int, ...]] = {}

        def emit(name: str, t) -> None:
            captured[name] = tuple(t.shape)

        logits_main, logits_mtp, cache = forward_capture(
            state,
            input_ids,
            mtp_in,
            rope,
            allow_math_fallback=True,
            emit=emit,
        )
        self.assertEqual(tuple(logits_main.shape), (batch, seq, eng_config.VOCAB_SIZE))
        self.assertIsNotNone(logits_mtp)
        self.assertEqual(tuple(logits_mtp.shape), (batch, seq, eng_config.VOCAB_SIZE))

        # Per-module emission set matches ref's named_modules() FQNs.
        for layer_idx in range(eng_config.NUM_LAYERS):
            for sub in (
                "attention_norm",
                "wqkv",
                "wo",
                "ffn_norm",
                "wfc1",
                "w2",
            ):
                self.assertIn(f"layers.{layer_idx}.{sub}", captured)
            self.assertIn(f"layers.{layer_idx}", captured)
        for top in (
            "tok_embeddings",
            "norm",
            "output",
            "mtp.emb_input_layernorm",
            "mtp.hidden_input_layernorm",
            "mtp.eagle_fc",
            "mtp.final_layernorm",
            "mtp",
        ):
            self.assertIn(top, captured)
        for sub in ("attention_norm", "wqkv", "wo", "ffn_norm", "wfc1", "w2"):
            self.assertIn(f"mtp.layer.{sub}", captured)

        _, _, grad_main = _masked_ce_forward_backward(logits_main, labels, loss_mask, weight=1.0)
        _, _, grad_mtp = _masked_ce_forward_backward(logits_mtp, mtp_lab, mtp_mask, weight=0.3)

        grads = backward_full(
            state,
            cache,
            grad_main,
            grad_mtp,
            allow_math_fallback=True,
        )
        # One grad per spec'd parameter, each shape-matching the param.
        self.assertEqual(set(grads), set(state.params))
        for name, param in state.params.items():
            self.assertEqual(tuple(grads[name].shape), tuple(param.shape), name)
            self.assertEqual(grads[name].dtype, torch.float32, name)


if __name__ == "__main__":
    unittest.main()
