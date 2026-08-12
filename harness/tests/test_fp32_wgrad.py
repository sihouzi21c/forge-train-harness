"""Bit-wise contract for the reference's fp32 weight-gradient accumulation.

The pure-PyTorch reference (``ref/reference/model_pure_mup_mtp.py``) routes
every weight matrix / embedding / RMSNorm through a custom autograd
Function so that, with ``WGRAD_ACCUM_FP32 = True`` (the default), the weight
gradient is accumulated in fp32 directly into ``param.main_grad`` — mirroring
the production training stack's fused-wgrad accumulation instead of stock
PyTorch's bf16 ``.grad``.

The three reference Functions are exercised in isolation (no full-model
forward — the backbone's attention path pulls in a CUDA-only kernel that is
unavailable on CPU/Mac CI, and the wgrad contract is per-op anyway). Three
guarantees are tested on small CPU tensors:

1. ``WGRAD_ACCUM_FP32 = False`` reproduces stock autograd *bit for bit*
   (dgrad AND wgrad), so the rewrite is a pure no-op in that mode — a
   regression fence around the manual backward math.
2. ``WGRAD_ACCUM_FP32 = True`` leaves ``.grad`` unset and exposes an fp32
   ``main_grad`` of the right shape on the weight (the surface the harness
   grad collector prefers via ``grad_attrs=("main_grad", "grad")``).
3. For a weight consumed by multiple call sites (the tied LM head / shared
   embedding), fp32 accumulation rounds once at the end rather than per
   source, so its value must differ from the bf16 ``.grad``.
"""

from __future__ import annotations

import importlib
import unittest
from pathlib import Path


def _can_import(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except (ImportError, ModuleNotFoundError):
        return False


_HAS_TORCH = _can_import("torch")


def _load_model_module():
    """Import ``model_pure_mup_mtp`` from the repo's ref/reference dir."""
    import os
    import sys

    # The model module reads its geometry fail-fast from os.environ at import
    # (generic product projection, no baked defaults); seed the 0.5B shape for
    # this standalone import. setdefault keeps a real projection authoritative.
    for k, v in {
        "NUM_LAYERS": "24",
        "HIDDEN_SIZE": "1024",
        "NUM_ATTENTION_HEADS": "16",
        "NUM_QUERY_GROUPS": "2",
        "HEAD_DIM": "64",
        "FFN_HIDDEN_SIZE": "4096",
        "PADDED_VOCAB_SIZE": "73448",
        "MAX_POSITION_EMBEDDINGS": "4096",
        "NORM_EPSILON": "1e-6",
        "ROTARY_BASE": "10000",
    }.items():
        os.environ.setdefault(k, v)

    repo_root = Path(__file__).resolve().parents[2]
    ref_dir = repo_root / "ref" / "reference"
    if str(ref_dir) not in sys.path:
        sys.path.insert(0, str(ref_dir))
    return importlib.import_module("model_pure_mup_mtp")


def _cases(mod):
    """(name, custom_apply, stock_apply, input, weight) per Function.

    ``stock_apply`` reuses the reference module's own functional alias
    (``mod.F``) so this test imports no banned framework submodule.
    """
    import torch

    f = mod.F
    n, k, b, s, eps = 24, 16, 2, 8, 1e-6
    g = torch.Generator().manual_seed(0)
    x_lin = torch.randn(b, s, k, generator=g, dtype=torch.bfloat16)
    w_lin = torch.randn(n, k, generator=g, dtype=torch.bfloat16)
    idx = torch.randint(0, n, (b, s), generator=g)
    w_emb = torch.randn(n, k, generator=g, dtype=torch.bfloat16)
    x_rms = torch.randn(b, s, k, generator=g, dtype=torch.bfloat16)
    w_rms = torch.randn(k, generator=g, dtype=torch.bfloat16)
    return [
        (
            "linear",
            lambda x, w: mod._LinearFn.apply(x, w),
            lambda x, w: f.linear(x, w),
            x_lin,
            w_lin,
        ),
        (
            "embedding",
            lambda i, w: mod._EmbeddingFn.apply(i, w),
            lambda i, w: f.embedding(i, w),
            idx,
            w_emb,
        ),
        (
            "rmsnorm",
            lambda x, w: mod._RMSNormFn.apply(x, w, eps),
            lambda x, w: f.rms_norm(x, (w.shape[0],), w, eps),
            x_rms,
            w_rms,
        ),
    ]


def _backward(mod, apply_fn, inp, weight, *, fp32):
    """Run one forward+backward; return (dx, dgrad_via_grad, main_grad)."""

    mod.WGRAD_ACCUM_FP32 = fp32
    x = inp.clone()
    w = mod.nn.Parameter(weight.clone())
    if x.is_floating_point():
        x.requires_grad_()
    out = apply_fn(x, w)
    out.float().pow(2).sum().backward()
    dx = x.grad.clone() if (x.is_floating_point() and x.grad is not None) else None
    dgrad = w.grad.clone() if w.grad is not None else None
    mg = getattr(w, "main_grad", None)
    return dx, dgrad, (mg.clone() if mg is not None else None)


@unittest.skipUnless(_HAS_TORCH, "requires torch")
class TestFp32Wgrad(unittest.TestCase):
    def setUp(self):
        self.mod = _load_model_module()
        self._saved_flag = self.mod.WGRAD_ACCUM_FP32

    def tearDown(self):
        self.mod.WGRAD_ACCUM_FP32 = self._saved_flag

    def test_bf16_mode_bitwise_matches_stock_autograd(self):
        import torch

        mod = self.mod
        for name, custom, stock, inp, weight in _cases(mod):
            dx_c, dw_c, mg_c = _backward(mod, custom, inp, weight, fp32=False)
            dx_s, dw_s, _ = _backward(mod, stock, inp, weight, fp32=False)
            self.assertIsNone(mg_c, f"{name}: bf16 mode must not create main_grad")
            self.assertIsNotNone(dw_c, f"{name}: custom .grad missing in bf16 mode")
            self.assertTrue(
                torch.equal(dw_c, dw_s),
                f"{name}: custom wgrad diverges from stock autograd "
                f"(maxabs={float((dw_c - dw_s).abs().max()):.3e})",
            )
            if dx_c is not None:
                self.assertTrue(
                    torch.equal(dx_c, dx_s),
                    f"{name}: custom dgrad diverges from stock autograd "
                    f"(maxabs={float((dx_c - dx_s).abs().max()):.3e})",
                )

    def test_fp32_mode_exposes_fp32_main_grad(self):
        import torch

        mod = self.mod
        for name, custom, _stock, inp, weight in _cases(mod):
            _dx, dgrad, mg = _backward(mod, custom, inp, weight, fp32=True)
            self.assertIsNotNone(mg, f"{name}: missing main_grad in fp32 mode")
            self.assertEqual(mg.dtype, torch.float32, name)
            self.assertEqual(tuple(mg.shape), tuple(weight.shape), name)
            self.assertIsNone(dgrad, f"{name}: .grad should be unset in fp32 mode")

    def test_fp32_main_grad_differs_from_bf16_for_multi_source_weight(self):
        """A weight summed from two call sites: fp32 rounds once, bf16 per-source."""
        import torch

        mod = self.mod
        _name, custom, _stock, inp, weight = _cases(mod)[0]  # linear
        x_a = inp.clone()
        x_b = (inp * 1.5).to(inp.dtype)

        def _two_site(fp32):
            mod.WGRAD_ACCUM_FP32 = fp32
            w = mod.nn.Parameter(weight.clone())
            obj = custom(x_a, w).float().pow(2).sum() + custom(x_b, w).float().pow(2).sum()
            obj.backward()
            return w.main_grad.clone() if fp32 else w.grad.clone()

        fp32_mg = _two_site(True)
        bf16_grad = _two_site(False)
        self.assertFalse(
            torch.equal(fp32_mg, bf16_grad.float()),
            "fp32 main_grad unexpectedly identical to bf16 .grad for multi-source weight",
        )


if __name__ == "__main__":
    unittest.main()
