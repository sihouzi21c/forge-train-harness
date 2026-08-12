"""CPU contract tests for ``training_engine_tensor.forward`` primitives.

These tests cover only the math layers that have no GPU / flash-attn
dependency: RoPE primitives, GQA QKV slicing, RMSNorm, SwiGLU.  The
GPU-only bitwise contract against the ref's flash-attn attention is
exercised separately by the ``forward-align`` harness suite.

The tests are gated on ``HARNESS_RUN_ENGINE_CONTRACTS=1`` (matching
the rest of ``test_training.py``) **and** on ``training_engine_tensor``
being importable — the engine source lives under
``workload/src/training_engine_tensor/``, which is gitignored by
default in the agent-loop workspace.  When that package is not
present (fresh harness checkout, no agent loop yet) the tests are
silently skipped, so this file never breaks CI on a clean clone.
"""

from __future__ import annotations

import importlib
import math
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
_HAS_ENGINE = _can_import("training_engine_tensor.forward")


@unittest.skipUnless(
    _HAS_TORCH and _HAS_ENGINE and _RUN_ENGINE_CONTRACTS,
    "engine contract tests require torch + workload/src/training_engine_tensor + "
    "HARNESS_RUN_ENGINE_CONTRACTS=1",
)
class TestRopePrimitives(unittest.TestCase):
    def test_freqs_shape_and_dtype_match_ref_contract(self):
        import torch
        from training_engine_tensor import config
        from training_engine_tensor.forward import precompute_rope_freqs

        freqs = precompute_rope_freqs(max_seq_len=128)
        self.assertEqual(tuple(freqs.shape), (128, config.HEAD_DIM))
        self.assertEqual(freqs.dtype, torch.float32)

    def test_freqs_first_row_is_zero(self):
        from training_engine_tensor.forward import precompute_rope_freqs

        freqs = precompute_rope_freqs(max_seq_len=4)
        self.assertTrue(bool((freqs[0] == 0).all()))

    def test_apply_rope_position_zero_is_identity(self):
        import torch
        from training_engine_tensor import config
        from training_engine_tensor.forward import (
            apply_rope,
            precompute_rope_freqs,
        )

        freqs = precompute_rope_freqs(max_seq_len=1)
        tensor = torch.randn(2, 1, config.NUM_HEADS, config.HEAD_DIM, dtype=torch.float32)
        rotated = apply_rope(tensor, freqs)
        self.assertTrue(torch.allclose(rotated, tensor, atol=0.0, rtol=0.0))

    def test_apply_rope_matches_explicit_half_rotate_formula(self):
        import torch
        from training_engine_tensor import config
        from training_engine_tensor.forward import (
            apply_rope,
            precompute_rope_freqs,
        )

        seq = 5
        torch.manual_seed(7)
        tensor = torch.randn(1, seq, config.NUM_HEADS, config.HEAD_DIM, dtype=torch.float32)
        freqs = precompute_rope_freqs(max_seq_len=seq)
        rotated = apply_rope(tensor, freqs)

        cos_ = torch.cos(freqs[:seq])[None, :, None, :]
        sin_ = torch.sin(freqs[:seq])[None, :, None, :]
        half = config.HEAD_DIM // 2
        x1 = tensor[..., :half]
        x2 = tensor[..., half:]
        explicit = tensor * cos_ + torch.cat((-x2, x1), dim=-1) * sin_
        self.assertTrue(torch.allclose(rotated, explicit, atol=0.0, rtol=0.0))


@unittest.skipUnless(
    _HAS_TORCH and _HAS_ENGINE and _RUN_ENGINE_CONTRACTS,
    "engine contract tests require torch + workload/src/training_engine_tensor + "
    "HARNESS_RUN_ENGINE_CONTRACTS=1",
)
class TestProjectQkvInterleavedLayout(unittest.TestCase):
    """``project_qkv`` must slice the packed wqkv weight per-kv-group.

    Verified against the slicing convention in
    ``ref/reference/model_pure_mup_mtp.py``: the packed projection of
    width ``NUM_KV_HEADS * (NUM_HEADS//NUM_KV_HEADS + 2) * HEAD_DIM``
    is viewed as ``[..., NUM_KV_HEADS, group_width]`` and split as
    ``[Q heads | K head | V head]`` inside each kv-group.
    """

    def test_q_k_v_shapes(self):
        import torch
        from training_engine_tensor import config
        from training_engine_tensor.forward import project_qkv

        qkv_width = config.NUM_HEADS * config.HEAD_DIM + 2 * config.NUM_KV_HEADS * config.HEAD_DIM
        hidden = torch.randn(2, 3, config.HIDDEN_SIZE, dtype=torch.float32)
        weight = torch.randn(qkv_width, config.HIDDEN_SIZE, dtype=torch.float32)
        q, k, v = project_qkv(hidden, weight)
        self.assertEqual(tuple(q.shape), (2, 3, config.NUM_HEADS, config.HEAD_DIM))
        self.assertEqual(tuple(k.shape), (2, 3, config.NUM_KV_HEADS, config.HEAD_DIM))
        self.assertEqual(tuple(v.shape), (2, 3, config.NUM_KV_HEADS, config.HEAD_DIM))

    def test_split_matches_ref_per_kv_group_layout(self):
        import torch
        from training_engine_tensor import config
        from training_engine_tensor.forward import project_qkv

        nkv = config.NUM_KV_HEADS
        nq_per_kv = config.NUM_HEADS // nkv
        d = config.HEAD_DIM
        qkv_width = (nq_per_kv + 2) * d * nkv

        torch.manual_seed(0)
        hidden = torch.randn(1, 2, config.HIDDEN_SIZE, dtype=torch.float32)
        weight = torch.randn(qkv_width, config.HIDDEN_SIZE, dtype=torch.float32)

        q, k, v = project_qkv(hidden, weight)
        projected = hidden.matmul(weight.t())
        grouped = projected.view(1, 2, nkv, (nq_per_kv + 2) * d)
        expected_q = grouped[..., : nq_per_kv * d].reshape(1, 2, config.NUM_HEADS, d)
        expected_k = grouped[..., nq_per_kv * d : nq_per_kv * d + d]
        expected_v = grouped[..., nq_per_kv * d + d :]

        self.assertTrue(torch.allclose(q, expected_q, atol=0.0, rtol=0.0))
        self.assertTrue(torch.allclose(k, expected_k, atol=0.0, rtol=0.0))
        self.assertTrue(torch.allclose(v, expected_v, atol=0.0, rtol=0.0))


@unittest.skipUnless(
    _HAS_TORCH and _HAS_ENGINE and _RUN_ENGINE_CONTRACTS,
    "engine contract tests require torch + workload/src/training_engine_tensor + "
    "HARNESS_RUN_ENGINE_CONTRACTS=1",
)
class TestRmsNormAndSwiglu(unittest.TestCase):
    def test_rms_norm_matches_explicit_formula(self):
        import torch
        from training_engine_tensor import config
        from training_engine_tensor.forward import rms_norm

        torch.manual_seed(11)
        hidden = torch.randn(2, 3, config.HIDDEN_SIZE, dtype=torch.float32)
        weight = torch.randn(config.HIDDEN_SIZE, dtype=torch.float32)
        out = rms_norm(hidden, weight)

        var = hidden.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden.float() * torch.rsqrt(var + config.NORM_EPSILON)
        expected = (normalized * weight.float()).to(hidden.dtype)
        self.assertTrue(torch.allclose(out, expected, atol=0.0, rtol=0.0))

    def test_rms_norm_bf16_matches_explicit_chain_with_nontrivial_weight(self):
        """``rms_norm`` must follow ``nn.RMSNorm``'s autograd chain bitwise on bf16
        inputs even when ``weight`` is not exactly 1.0.

        Regression for the multistep-1gpu step-2 forward divergence: the previous chain
        ``(normalized * weight.fp32).to(bf16)`` upcasts weight to fp32 then casts
        the product down, while ``nn.RMSNorm`` casts the normalized fp32 result
        to bf16 first and multiplies in the input dtype.  At canonical-state init
        every norm weight is exactly 1.0 so both chains coincide and forward-
        align passes.  After a single AdamW step the weights drift off 1.0 and the
        old chain diverges by 1 ULP per element, accumulating to a 0.2-bf16 logits
        gap by the LM head — which is exactly what multistep-1gpu hit.

        We compare against the explicit ``(x32 * r).to(bf16) * weight`` chain
        (the primitive decomposition ``nn.RMSNorm`` uses on CUDA — verified
        numerically on H100), not against ``nn.RMSNorm`` directly, because
        the CPU path of ``nn.RMSNorm`` collapses to a different fused kernel
        that rounds differently from the GPU path.  The training engine only
        runs on CUDA; the gate-relevant bit-equality is the explicit-chain one.
        """

        import torch
        from training_engine_tensor import config
        from training_engine_tensor.forward import rms_norm

        torch.manual_seed(17)
        hidden = torch.randn(2, 4, config.HIDDEN_SIZE, dtype=torch.bfloat16)
        weight = (1.0 + 0.01 * torch.randn(config.HIDDEN_SIZE)).to(torch.bfloat16)

        out = rms_norm(hidden, weight)

        x32 = hidden.float()
        r = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + config.NORM_EPSILON)
        expected = (x32 * r).to(torch.bfloat16) * weight
        max_abs = (out - expected).abs().max().item()
        self.assertEqual(
            max_abs,
            0.0,
            msg=(
                f"rms_norm bf16 forward must follow (x32*r).to(bf16)*weight chain; "
                f"got max_abs_diff={max_abs} (pre-fix value was 1 ULP per element)."
            ),
        )

    def test_swiglu_matches_explicit_formula(self):
        import torch
        from training_engine_tensor import config
        from training_engine_tensor.forward import mlp_swiglu

        torch.manual_seed(13)
        hidden = torch.randn(1, 2, config.HIDDEN_SIZE, dtype=torch.float32)
        fc1 = torch.randn(2 * config.FFN_HIDDEN_SIZE, config.HIDDEN_SIZE, dtype=torch.float32)
        fc2 = torch.randn(config.HIDDEN_SIZE, config.FFN_HIDDEN_SIZE, dtype=torch.float32)

        out = mlp_swiglu(hidden, fc1, fc2)
        gate_up = hidden.float().matmul(fc1.float().t())
        gate, up = gate_up.chunk(2, dim=-1)
        intermediate = gate * torch.sigmoid(gate) * up
        expected = intermediate.matmul(fc2.float().t()).to(hidden.dtype)
        self.assertTrue(torch.allclose(out, expected, atol=0.0, rtol=0.0))


@unittest.skipUnless(
    _HAS_TORCH and _HAS_ENGINE and _RUN_ENGINE_CONTRACTS,
    "engine contract tests require torch + workload/src/training_engine_tensor + "
    "HARNESS_RUN_ENGINE_CONTRACTS=1",
)
class TestDecoderLayerForwardSmoke(unittest.TestCase):
    """End-to-end shape / dtype / residual-non-degenerate smoke check.

    Uses the math-fallback path of :func:`_gqa_attention` (no flash_attn
    needed on CPU).  Bitwise alignment against the GPU ref happens in
    the ``forward-align`` harness suite.
    """

    def test_decoder_layer_preserves_shape_and_modifies_hidden(self):
        import torch
        from training_engine_tensor import config
        from training_engine_tensor.forward import (
            decoder_layer_forward,
            precompute_rope_freqs,
        )
        from training_engine_tensor.parameters import LayerParameterView

        seq = 4
        torch.manual_seed(17)
        hidden = torch.randn(1, seq, config.HIDDEN_SIZE, dtype=torch.float32)
        qkv_width = config.NUM_HEADS * config.HEAD_DIM + 2 * config.NUM_KV_HEADS * config.HEAD_DIM
        layer = LayerParameterView(
            index=0,
            input_norm_weight=torch.ones(config.HIDDEN_SIZE, dtype=torch.float32),
            qkv_weight=torch.randn(qkv_width, config.HIDDEN_SIZE, dtype=torch.float32) * 0.02,
            attention_proj_weight=torch.randn(
                config.HIDDEN_SIZE, config.HIDDEN_SIZE, dtype=torch.float32
            )
            * 0.02,
            pre_mlp_norm_weight=torch.ones(config.HIDDEN_SIZE, dtype=torch.float32),
            mlp_fc1_weight=torch.randn(
                2 * config.FFN_HIDDEN_SIZE, config.HIDDEN_SIZE, dtype=torch.float32
            )
            * 0.02,
            mlp_fc2_weight=torch.randn(
                config.HIDDEN_SIZE, config.FFN_HIDDEN_SIZE, dtype=torch.float32
            )
            * 0.02,
        )
        freqs = precompute_rope_freqs(max_seq_len=seq)
        depth_scale = 1.4 / math.sqrt(config.NUM_LAYERS)

        out = decoder_layer_forward(
            hidden,
            layer,
            freqs,
            depth_scale=depth_scale,
            allow_math_fallback=True,
        )
        self.assertEqual(tuple(out.shape), tuple(hidden.shape))
        self.assertEqual(out.dtype, hidden.dtype)
        self.assertFalse(torch.allclose(out, hidden))  # residual update must be non-trivial


@unittest.skipUnless(
    _HAS_TORCH and _HAS_ENGINE and _RUN_ENGINE_CONTRACTS,
    "engine contract tests require torch + workload/src/training_engine_tensor + "
    "HARNESS_RUN_ENGINE_CONTRACTS=1",
)
class TestGqaAttentionFailFast(unittest.TestCase):
    """Pin the CPU fail-fast contract on the math fallback gate.

    The math implementation is not bitwise against the ref's
    ``flash_attn`` kernel, so a CPU invocation without
    ``allow_math_fallback=True`` must raise — otherwise the
    ``forward-align`` capture path could silently take a non-bitwise
    code path and report a confusing PASS.
    """

    def test_cpu_without_opt_in_raises_not_implemented(self):
        import torch
        from training_engine_tensor.forward import _gqa_attention

        q = torch.randn(1, 2, 4, 8, dtype=torch.float32)
        k = torch.randn(1, 2, 2, 8, dtype=torch.float32)
        v = torch.randn(1, 2, 2, 8, dtype=torch.float32)
        with self.assertRaises(NotImplementedError):
            _gqa_attention(q, k, v)

    def test_cpu_with_opt_in_runs_math_fallback(self):
        import torch
        from training_engine_tensor.forward import _gqa_attention

        q = torch.randn(1, 2, 4, 8, dtype=torch.float32)
        k = torch.randn(1, 2, 2, 8, dtype=torch.float32)
        v = torch.randn(1, 2, 2, 8, dtype=torch.float32)
        out = _gqa_attention(q, k, v, allow_math_fallback=True)
        self.assertEqual(tuple(out.shape), tuple(q.shape))
        self.assertEqual(out.dtype, q.dtype)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
