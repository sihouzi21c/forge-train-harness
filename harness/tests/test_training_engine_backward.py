"""CPU contract tests for ``training_engine_tensor.backward`` primitives.

Each primitive is checked against a finite-difference reference (for
gradients of the input) or against the closed-form derivation (for
``grad_weight`` / ``grad_embedding``).  The tests use float64 inputs
to keep the FD reference numerically meaningful while still exercising
the same FP32-internal math path the production code uses on bf16
inputs.

Skipped silently when the engine package is not on disk (the agent-loop
workspace gitignores ``workload/src/training_engine_tensor/*`` apart
from the package shell) so this file never breaks a clean clone.
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
        return True
    except (ImportError, ModuleNotFoundError):
        return False


_RUN_ENGINE_CONTRACTS = os.environ.get("HARNESS_RUN_ENGINE_CONTRACTS") == "1"
_HAS_TORCH = _can_import("torch")
_HAS_ENGINE = _can_import("training_engine_tensor.backward")


@unittest.skipUnless(
    _HAS_TORCH and _HAS_ENGINE and _RUN_ENGINE_CONTRACTS,
    "engine contract tests require torch + workload/src/training_engine_tensor + "
    "HARNESS_RUN_ENGINE_CONTRACTS=1",
)
class TestLinearBackward(unittest.TestCase):
    def test_matches_closed_form(self):
        import torch
        from training_engine_tensor.backward import linear_backward

        torch.manual_seed(0)
        x = torch.randn(2, 3, 4, dtype=torch.float64)
        w = torch.randn(5, 4, dtype=torch.float64)
        grad_out = torch.randn(2, 3, 5, dtype=torch.float64)

        grad_in, grad_w = linear_backward(grad_out, x, w)
        expected_in = grad_out.matmul(w)
        expected_w = grad_out.reshape(-1, 5).t().matmul(x.reshape(-1, 4))
        self.assertTrue(torch.allclose(grad_in, expected_in, atol=0.0, rtol=0.0))
        self.assertTrue(torch.allclose(grad_w, expected_w, atol=0.0, rtol=0.0))

    def test_rejects_shape_mismatch(self):
        import torch
        from training_engine_tensor.backward import linear_backward

        x = torch.randn(2, 4, dtype=torch.float32)
        w = torch.randn(5, 4, dtype=torch.float32)
        grad_out_bad = torch.randn(2, 7, dtype=torch.float32)
        with self.assertRaises(ValueError):
            linear_backward(grad_out_bad, x, w)


@unittest.skipUnless(
    _HAS_TORCH and _HAS_ENGINE and _RUN_ENGINE_CONTRACTS,
    "engine contract tests require torch + workload/src/training_engine_tensor + "
    "HARNESS_RUN_ENGINE_CONTRACTS=1",
)
class TestRmsNormBackward(unittest.TestCase):
    def test_matches_finite_difference(self):
        import torch
        from training_engine_tensor.backward import rms_norm_backward
        from training_engine_tensor.forward import rms_norm

        torch.manual_seed(1)
        hidden = torch.randn(1, 2, 8, dtype=torch.float64)
        weight = torch.randn(8, dtype=torch.float64)
        grad_out = torch.randn_like(hidden)
        eps = 1e-6

        grad_in, grad_w = rms_norm_backward(grad_out, hidden, weight, eps=eps)

        def loss(h, w):
            return (rms_norm(h, w, eps=eps) * grad_out).sum()

        step = 1e-4
        fd_in = torch.zeros_like(hidden)
        for idx in range(hidden.numel()):
            flat = hidden.reshape(-1)
            orig = flat[idx].item()
            flat[idx] = orig + step
            high = loss(hidden, weight)
            flat[idx] = orig - step
            low = loss(hidden, weight)
            flat[idx] = orig
            fd_in.reshape(-1)[idx] = (high - low) / (2 * step)

        fd_w = torch.zeros_like(weight)
        for idx in range(weight.numel()):
            orig = weight[idx].item()
            weight[idx] = orig + step
            high = loss(hidden, weight)
            weight[idx] = orig - step
            low = loss(hidden, weight)
            weight[idx] = orig
            fd_w[idx] = (high - low) / (2 * step)

        self.assertTrue(torch.allclose(grad_in, fd_in, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(grad_w, fd_w, atol=1e-6, rtol=1e-6))


@unittest.skipUnless(
    _HAS_TORCH and _HAS_ENGINE and _RUN_ENGINE_CONTRACTS,
    "engine contract tests require torch + workload/src/training_engine_tensor + "
    "HARNESS_RUN_ENGINE_CONTRACTS=1",
)
class TestSwigluIntermediateBackward(unittest.TestCase):
    def test_matches_finite_difference(self):
        import torch
        from training_engine_tensor.backward import (
            silu_swiglu_intermediate_backward,
        )

        torch.manual_seed(2)
        gate = torch.randn(1, 6, dtype=torch.float64)
        up = torch.randn_like(gate)
        grad_out = torch.randn_like(gate)

        grad_gate, grad_up = silu_swiglu_intermediate_backward(grad_out, gate, up)

        def loss(g, u):
            return (torch.sigmoid(g) * g * u * grad_out).sum()

        step = 1e-5
        fd_gate = torch.zeros_like(gate)
        for idx in range(gate.numel()):
            orig = gate.reshape(-1)[idx].item()
            gate.reshape(-1)[idx] = orig + step
            high = loss(gate, up)
            gate.reshape(-1)[idx] = orig - step
            low = loss(gate, up)
            gate.reshape(-1)[idx] = orig
            fd_gate.reshape(-1)[idx] = (high - low) / (2 * step)

        fd_up = torch.zeros_like(up)
        for idx in range(up.numel()):
            orig = up.reshape(-1)[idx].item()
            up.reshape(-1)[idx] = orig + step
            high = loss(gate, up)
            up.reshape(-1)[idx] = orig - step
            low = loss(gate, up)
            up.reshape(-1)[idx] = orig
            fd_up.reshape(-1)[idx] = (high - low) / (2 * step)

        self.assertTrue(torch.allclose(grad_gate, fd_gate, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(grad_up, fd_up, atol=1e-5, rtol=1e-5))


@unittest.skipUnless(
    _HAS_TORCH and _HAS_ENGINE and _RUN_ENGINE_CONTRACTS,
    "engine contract tests require torch + workload/src/training_engine_tensor + "
    "HARNESS_RUN_ENGINE_CONTRACTS=1",
)
class TestEmbeddingBackward(unittest.TestCase):
    def test_sums_repeated_token_grads(self):
        import torch
        from training_engine_tensor.backward import embedding_backward

        vocab = 7
        hidden = 3
        token_ids = torch.tensor([[1, 2], [1, 2]], dtype=torch.int64)
        grad_out = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]],
            dtype=torch.float32,
        )
        grad_embedding = embedding_backward(grad_out, token_ids, vocab_size=vocab)
        self.assertEqual(tuple(grad_embedding.shape), (vocab, hidden))
        # token 1 appears at flat positions 0 ([1,0,0]) and 2 ([0,0,1])
        self.assertTrue(
            torch.allclose(grad_embedding[1], torch.tensor([1.0, 0.0, 1.0]), atol=0.0, rtol=0.0)
        )
        # token 2 appears at flat positions 1 ([0,1,0]) and 3 ([1,0,0])
        self.assertTrue(
            torch.allclose(grad_embedding[2], torch.tensor([1.0, 1.0, 0.0]), atol=0.0, rtol=0.0)
        )
        # untouched rows stay zero
        zero_rows = [0, 3, 4, 5, 6]
        for row in zero_rows:
            self.assertTrue(bool((grad_embedding[row] == 0).all()))

    def test_rejects_out_of_vocab_id(self):
        import torch
        from training_engine_tensor.backward import embedding_backward

        token_ids = torch.tensor([[10]], dtype=torch.int64)
        grad_out = torch.zeros(1, 1, 2, dtype=torch.float32)
        with self.assertRaises(ValueError):
            embedding_backward(grad_out, token_ids, vocab_size=4)


@unittest.skipUnless(
    _HAS_TORCH and _HAS_ENGINE and _RUN_ENGINE_CONTRACTS,
    "engine contract tests require torch + workload/src/training_engine_tensor + "
    "HARNESS_RUN_ENGINE_CONTRACTS=1",
)
class TestApplyRopeBackward(unittest.TestCase):
    """Pin the closed-form chain rule of ``apply_rope`` against FD.

    RoPE is linear in ``tensor`` with frequency-only constants, so the
    backward must be exact (FD agreement is bounded only by FP64 FD
    truncation error).  This test acts as the contract enforcing that
    ``apply_rope_backward`` exists and stays bitwise-equivalent to the
    closed-form derivation in :func:`training_engine_tensor.backward.apply_rope_backward`.
    """

    def test_matches_finite_difference(self):
        import torch
        from training_engine_tensor.backward import apply_rope_backward
        from training_engine_tensor.forward import apply_rope, precompute_rope_freqs

        torch.manual_seed(3)
        seq = 4
        n_heads = 2
        head_dim = 8
        tensor = torch.randn(1, seq, n_heads, head_dim, dtype=torch.float64)
        freqs = precompute_rope_freqs(max_seq_len=seq, head_dim=head_dim).to(dtype=torch.float64)
        grad_out = torch.randn_like(tensor)

        grad_tensor = apply_rope_backward(grad_out, freqs)
        self.assertEqual(tuple(grad_tensor.shape), tuple(tensor.shape))
        self.assertEqual(grad_tensor.dtype, tensor.dtype)

        def loss(t):
            return (apply_rope(t, freqs) * grad_out).sum()

        step = 1e-5
        fd = torch.zeros_like(tensor)
        flat = tensor.reshape(-1)
        for idx in range(tensor.numel()):
            orig = flat[idx].item()
            flat[idx] = orig + step
            high = loss(tensor)
            flat[idx] = orig - step
            low = loss(tensor)
            flat[idx] = orig
            fd.reshape(-1)[idx] = (high - low) / (2 * step)

        self.assertTrue(torch.allclose(grad_tensor, fd, atol=1e-7, rtol=1e-7))

    def test_rejects_freqs_shape_mismatch(self):
        import torch
        from training_engine_tensor.backward import apply_rope_backward

        grad_out = torch.randn(1, 4, 2, 8, dtype=torch.float32)
        freqs_bad_dim = torch.zeros(4, 6, dtype=torch.float32)
        with self.assertRaises(ValueError):
            apply_rope_backward(grad_out, freqs_bad_dim)

        freqs_short = torch.zeros(2, 8, dtype=torch.float32)
        with self.assertRaises(ValueError):
            apply_rope_backward(grad_out, freqs_short)


@unittest.skipUnless(
    _HAS_TORCH and _HAS_ENGINE and _RUN_ENGINE_CONTRACTS,
    "engine contract tests require torch + workload/src/training_engine_tensor + "
    "HARNESS_RUN_ENGINE_CONTRACTS=1",
)
class TestProjectQkvBackward(unittest.TestCase):
    """Pin ``project_qkv_backward`` against FD on hidden + weight.

    The forward's per-kv-group interleaved slicing must invert exactly:
    a wrong gradient routing would silently corrupt backward-align bitwise
    even though shapes look fine.  This test acts as the contract.
    """

    def test_matches_finite_difference_for_hidden_and_weight(self):
        import torch
        from training_engine_tensor.backward import project_qkv_backward

        # Use a small synthetic config so the FD loop is tractable on CPU
        # while still exercising the nq_per_kv > 1 GQA slicing.
        n_kv = 2
        nq_per_kv = 4
        n_q = nq_per_kv * n_kv  # 8
        head_dim = 4
        hidden_size = 6
        qkv_width = (nq_per_kv + 2) * head_dim * n_kv

        # Direct local copies of the forward / backward closures so the
        # test does not depend on the production config constants.  We
        # use the same slicing math the module documents.
        def fwd(h, w):
            projected = h.matmul(w.t())
            leading = h.shape[:-1]
            grouped = projected.view(*leading, n_kv, (nq_per_kv + 2) * head_dim)
            q_block = grouped[..., : nq_per_kv * head_dim]
            k_block = grouped[..., nq_per_kv * head_dim : nq_per_kv * head_dim + head_dim]
            v_block = grouped[..., nq_per_kv * head_dim + head_dim :]
            return (
                q_block.reshape(*leading, n_q, head_dim),
                k_block.reshape(*leading, n_kv, head_dim),
                v_block.reshape(*leading, n_kv, head_dim),
            )

        torch.manual_seed(4)
        hidden = torch.randn(1, 3, hidden_size, dtype=torch.float64)
        weight = torch.randn(qkv_width, hidden_size, dtype=torch.float64)
        q_ref, k_ref, v_ref = fwd(hidden, weight)
        grad_q = torch.randn_like(q_ref)
        grad_k = torch.randn_like(k_ref)
        grad_v = torch.randn_like(v_ref)

        # Patch project_qkv to use this synthetic layout via direct
        # delegation: project_qkv uses module-level config, so we drive
        # project_qkv_backward end-to-end through the slicing logic by
        # constructing inputs that match its leading-dim and head-count
        # contract directly.
        grad_hidden, grad_weight = project_qkv_backward(grad_q, grad_k, grad_v, hidden, weight)
        self.assertEqual(tuple(grad_hidden.shape), tuple(hidden.shape))
        self.assertEqual(tuple(grad_weight.shape), tuple(weight.shape))

        def loss(h, w):
            q, k, v = fwd(h, w)
            return (q * grad_q).sum() + (k * grad_k).sum() + (v * grad_v).sum()

        step = 1e-5
        fd_hidden = torch.zeros_like(hidden)
        flat_h = hidden.reshape(-1)
        for idx in range(hidden.numel()):
            orig = flat_h[idx].item()
            flat_h[idx] = orig + step
            high = loss(hidden, weight)
            flat_h[idx] = orig - step
            low = loss(hidden, weight)
            flat_h[idx] = orig
            fd_hidden.reshape(-1)[idx] = (high - low) / (2 * step)

        fd_weight = torch.zeros_like(weight)
        flat_w = weight.reshape(-1)
        for idx in range(weight.numel()):
            orig = flat_w[idx].item()
            flat_w[idx] = orig + step
            high = loss(hidden, weight)
            flat_w[idx] = orig - step
            low = loss(hidden, weight)
            flat_w[idx] = orig
            fd_weight.reshape(-1)[idx] = (high - low) / (2 * step)

        self.assertTrue(torch.allclose(grad_hidden, fd_hidden, atol=1e-7, rtol=1e-7))
        self.assertTrue(torch.allclose(grad_weight, fd_weight, atol=1e-7, rtol=1e-7))

    def test_rejects_head_dim_mismatch(self):
        import torch
        from training_engine_tensor.backward import project_qkv_backward

        grad_q = torch.zeros(1, 2, 4, 8, dtype=torch.float32)
        grad_k = torch.zeros(1, 2, 2, 6, dtype=torch.float32)  # wrong head_dim
        grad_v = torch.zeros(1, 2, 2, 8, dtype=torch.float32)
        hidden = torch.zeros(1, 2, 12, dtype=torch.float32)
        weight = torch.zeros(1, 12, dtype=torch.float32)
        with self.assertRaises(ValueError):
            project_qkv_backward(grad_q, grad_k, grad_v, hidden, weight)

    def test_rejects_non_divisible_q_heads(self):
        import torch
        from training_engine_tensor.backward import project_qkv_backward

        # 5 Q heads not divisible by 2 KV heads
        grad_q = torch.zeros(1, 2, 5, 8, dtype=torch.float32)
        grad_k = torch.zeros(1, 2, 2, 8, dtype=torch.float32)
        grad_v = torch.zeros(1, 2, 2, 8, dtype=torch.float32)
        hidden = torch.zeros(1, 2, 12, dtype=torch.float32)
        weight = torch.zeros(((5 + 2 * 2) // 2) * 0 + 1, 12, dtype=torch.float32)
        with self.assertRaises(ValueError):
            project_qkv_backward(grad_q, grad_k, grad_v, hidden, weight)


@unittest.skipUnless(
    _HAS_TORCH and _HAS_ENGINE and _RUN_ENGINE_CONTRACTS,
    "engine contract tests require torch + workload/src/training_engine_tensor + "
    "HARNESS_RUN_ENGINE_CONTRACTS=1",
)
class TestGqaAttentionBackward(unittest.TestCase):
    """Pin ``gqa_attention_backward`` against FD on q / k / v.

    The math fallback path needs to match the analytical softmax +
    matmul derivation exactly (bounded only by FP64 FD truncation
    error).  This is the contract that lets the upcoming
    ``decoder_layer_backward`` orchestrator delegate the attention
    reverse leg to a static primitive.
    """

    def test_matches_finite_difference_on_math_fallback(self):
        import torch
        from training_engine_tensor.backward import gqa_attention_backward
        from training_engine_tensor.forward import _gqa_attention

        # Small synthetic GQA config so the FD loop is tractable on CPU
        # but still exercises both repeated KV heads (rep > 1) and a
        # causal mask interaction (seq > 1).
        batch = 1
        seq = 3
        n_kv = 2
        rep = 2
        n_q = n_kv * rep
        head_dim = 4

        torch.manual_seed(5)
        q = torch.randn(batch, seq, n_q, head_dim, dtype=torch.float64)
        k = torch.randn(batch, seq, n_kv, head_dim, dtype=torch.float64)
        v = torch.randn(batch, seq, n_kv, head_dim, dtype=torch.float64)
        out_ref = _gqa_attention(q, k, v, allow_math_fallback=True)
        grad_out = torch.randn_like(out_ref)

        grad_q, grad_k, grad_v = gqa_attention_backward(grad_out, q, k, v, allow_math_fallback=True)
        self.assertEqual(tuple(grad_q.shape), tuple(q.shape))
        self.assertEqual(tuple(grad_k.shape), tuple(k.shape))
        self.assertEqual(tuple(grad_v.shape), tuple(v.shape))

        def loss(q_in, k_in, v_in):
            return (_gqa_attention(q_in, k_in, v_in, allow_math_fallback=True) * grad_out).sum()

        step = 1e-5

        def fd_against(tensor: torch.Tensor) -> torch.Tensor:
            fd = torch.zeros_like(tensor)
            flat = tensor.reshape(-1)
            for idx in range(tensor.numel()):
                orig = flat[idx].item()
                flat[idx] = orig + step
                high = loss(q, k, v)
                flat[idx] = orig - step
                low = loss(q, k, v)
                flat[idx] = orig
                fd.reshape(-1)[idx] = (high - low) / (2 * step)
            return fd

        fd_q = fd_against(q)
        fd_k = fd_against(k)
        fd_v = fd_against(v)

        self.assertTrue(torch.allclose(grad_q, fd_q, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(grad_k, fd_k, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(grad_v, fd_v, atol=1e-6, rtol=1e-6))

    def test_cpu_without_opt_in_raises_not_implemented(self):
        import torch
        from training_engine_tensor.backward import gqa_attention_backward

        q = torch.randn(1, 2, 4, 8, dtype=torch.float32)
        k = torch.randn(1, 2, 2, 8, dtype=torch.float32)
        v = torch.randn(1, 2, 2, 8, dtype=torch.float32)
        grad_out = torch.randn_like(q)
        with self.assertRaises(NotImplementedError):
            gqa_attention_backward(grad_out, q, k, v)

    def test_rejects_grad_out_shape_mismatch(self):
        import torch
        from training_engine_tensor.backward import gqa_attention_backward

        q = torch.randn(1, 2, 4, 8, dtype=torch.float32)
        k = torch.randn(1, 2, 2, 8, dtype=torch.float32)
        v = torch.randn(1, 2, 2, 8, dtype=torch.float32)
        grad_bad = torch.randn(1, 2, 4, 6, dtype=torch.float32)
        with self.assertRaises(ValueError):
            gqa_attention_backward(grad_bad, q, k, v, allow_math_fallback=True)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
