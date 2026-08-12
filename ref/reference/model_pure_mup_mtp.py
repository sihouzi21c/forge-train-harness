"""MiniCPM4 0.5B — pure PyTorch + muP + MTP (no TE).

muP (Maximal Update Parameterization, depth + width):
    width_mult = hidden_size / mup_base_hidden_size
    1) init: matrix Linear weights ~ N(0, init_std / sqrt(width_mult))
       embedding / output_layer / residual-stream output (wo, w2): N(0, init_std)
    2) embedding output × mup_emb_scale
    3) hidden / width_mult before lm_head
    4) residual update: residual + branch_out * (mup_depth_scale / sqrt(num_layers))
    5) lr scaling for matrix weights handled in train_pure_mup_mtp.py

MTP / eagle layer (DeepSeek-style next-next-token prediction):
    bigram = [LN(mtp_token_emb) | LN(llm_hidden)]
    h_mtp  = LN_final( transformer_layer( eagle_fc(bigram) ) )
    logits_mtp = lm_head( h_mtp / width_mult )         # share lm_head with main
    mtp_loss   = CE(logits_mtp, mtp_labels)
    total_loss = lm_loss + eagle_ce_loss_weight * mtp_loss
"""
from __future__ import annotations

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Module-level architecture constants, read from the environment at import
# under the generic upper-cased product-key names (the [model] config axis →
# rendered gate product → tools/product_env.py projection). FAIL-FAST by
# design: a missing geometry key raises KeyError instead of silently falling
# back to a baked 0.5B default — with the per-key registry gone, this is the
# guarantee that a render/projection gap can never train the wrong
# architecture. Standalone imports must pre-project the product (or set the
# ten keys explicitly, as the torch-capable unit tests do).
NUM_LAYERS = int(os.environ["NUM_LAYERS"])
HIDDEN_SIZE = int(os.environ["HIDDEN_SIZE"])
NUM_HEADS = int(os.environ["NUM_ATTENTION_HEADS"])
NUM_KV_HEADS = int(os.environ["NUM_QUERY_GROUPS"])
# head_dim is an explicit knob: hidden_size // num_heads is NOT always the
# head dim (MiniCPM5-1B uses 128 with hidden=1536, heads=16 → 1536//16=96
# != 128), so it canNOT be derived — the product must carry it.
HEAD_DIM = int(os.environ["HEAD_DIM"])
FFN_HIDDEN_SIZE = int(os.environ["FFN_HIDDEN_SIZE"])
VOCAB_SIZE = int(os.environ["PADDED_VOCAB_SIZE"])
MAX_SEQ_LEN = int(os.environ["MAX_POSITION_EMBEDDINGS"])
NORM_EPS = float(os.environ["NORM_EPSILON"])
ROPE_THETA = float(os.environ["ROTARY_BASE"])


# ── fp32 weight-gradient accumulation ───────────────────────────────────
#
# The production training stack (Megatron-LM + TransformerEngine)
# accumulates the weight gradient (wgrad) directly in fp32 — the
# fused-wgrad GEMM writes fp32 straight into a pre-allocated
# ``param.main_grad`` buffer. Stock PyTorch autograd on a bf16 model
# instead produces a bf16 ``.grad`` (one rounding per source), which
# truncates every weight gradient and, for weights consumed by multiple
# call sites (the tied LM head shared by the main + MTP branches, the
# embedding used by both branches), rounds each contribution to bf16
# *before* summing.
#
# To mirror that industry-standard fp32 accumulation we route every
# weight matrix / embedding / RMSNorm weight through a custom autograd
# Function whose backward computes the weight gradient on the SAME bf16
# compute path used by forward / dgrad (cuBLAS bf16 GEMM with fp32
# multiply + fp32 accumulate + bf16 output for ``_LinearFn``; bf16
# elementwise + reduction for ``_RMSNormFn``; bf16
# ``aten.embedding_dense_backward`` for ``_EmbeddingFn``), and casts the
# bf16 result to fp32 once before adding to ``param.main_grad``. This
# gives:
#   * uniform cuBLAS algorithm / reduction-order across all three GEMMs
#     (forward, dgrad, wgrad share the same bf16 kernel dispatch),
#   * tied / multi-source weights still accumulate in fp32 — each
#     contribution is upcast to fp32 before being summed into
#     ``main_grad``, so the multi-source rounding that stock bf16
#     ``.grad`` suffers from is avoided,
#   * dgrad stays bf16 so forward/activation numerics are untouched
#     (RMSNorm replays its dgrad through autograd to stay
#     bitwise-identical).
# Set ``WGRAD_ACCUM_FP32 = False`` to fall back to stock bf16 ``.grad``
# (the manual backward is then bitwise-identical to autograd — see
# harness/tests/test_fp32_wgrad.py).
#
# Storage pattern: all three Functions save the weight ``nn.Parameter``
# via ``ctx.weight = weight`` (plain Python attribute) instead of
# ``ctx.save_for_backward``. Under
# ``torch.utils.checkpoint(use_reentrant=False)``, ``saved_tensors_hooks``
# rematerializes leaf parameters saved via ``save_for_backward`` as a
# storage-sharing but distinct Tensor object; any attribute side-effect
# (``_ensure_main_grad(weight).add_(wg)``) then lands on a temporary that
# is GC'd before ``optimizer.step`` — so the recomputed layers would
# silently receive zero weight-gradient. Storing the Parameter on ``ctx``
# directly bypasses the hook machinery and preserves Parameter identity,
# which is required for the main_grad side-channel to survive activation
# checkpointing.
WGRAD_ACCUM_FP32 = True


def _ensure_main_grad(weight: torch.Tensor) -> torch.Tensor:
    mg = getattr(weight, "main_grad", None)
    if mg is None:
        mg = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
        weight.main_grad = mg
    return mg


class _LinearFn(torch.autograd.Function):
    """y = x @ wᵀ (bias-free). dgrad stays bf16; wgrad accumulates fp32."""

    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x)
        ctx.weight = weight
        return torch.matmul(x, weight.t())

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        weight = ctx.weight
        n = weight.shape[0]
        k = weight.shape[1]
        dx = torch.matmul(grad_out, weight)
        g2 = grad_out.reshape(-1, n)
        x2 = x.reshape(-1, k)
        if WGRAD_ACCUM_FP32:
            wg = torch.matmul(g2.transpose(0, 1), x2)
            _ensure_main_grad(weight).add_(wg.float())
            return dx, None
        return dx, torch.matmul(g2.transpose(0, 1), x2)


class _EmbeddingFn(torch.autograd.Function):
    """Embedding lookup. wgrad via aten.embedding_dense_backward (fp32)."""

    @staticmethod
    def forward(ctx, idx, weight):
        ctx.save_for_backward(idx)
        ctx.num_embeddings = weight.shape[0]
        ctx.weight = weight
        return F.embedding(idx, weight)

    @staticmethod
    def backward(ctx, grad_out):
        (idx,) = ctx.saved_tensors
        weight = ctx.weight
        v = ctx.num_embeddings
        if WGRAD_ACCUM_FP32:
            wg = torch.ops.aten.embedding_dense_backward(
                grad_out, idx, v, -1, False
            )
            _ensure_main_grad(weight).add_(wg.float())
            return None, None
        wg = torch.ops.aten.embedding_dense_backward(grad_out, idx, v, -1, False)
        return None, wg


class _RMSNormFn(torch.autograd.Function):
    """RMSNorm. forward == nn.RMSNorm bitwise; dgrad replays autograd; wgrad fp32."""

    @staticmethod
    def forward(ctx, x, weight, eps):
        ctx.save_for_backward(x)
        ctx.weight = weight
        ctx.eps = eps
        return F.rms_norm(x, (weight.shape[0],), weight, eps)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        weight = ctx.weight
        shape = (weight.shape[0],)
        # dgrad: replay the exact op through autograd so the activation
        # gradient is bitwise-identical to stock nn.RMSNorm backward.
        with torch.enable_grad():
            xd = x.detach().requires_grad_(True)
            wd = weight.detach().requires_grad_(True)
            out = F.rms_norm(xd, shape, wd, ctx.eps)
            dx, dw = torch.autograd.grad(out, (xd, wd), grad_out)
        if WGRAD_ACCUM_FP32:
            normed = F.rms_norm(x.detach(), shape, None, ctx.eps)
            wg = (grad_out * normed).reshape(-1, weight.shape[0]).sum(0)
            _ensure_main_grad(weight).add_(wg.float())
            return dx, None, None
        return dx, dw, None


class _Linear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _LinearFn.apply(x, self.weight)


class _Embedding(nn.Embedding):
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return _EmbeddingFn.apply(idx, self.weight)


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _RMSNormFn.apply(x, self.weight, self.eps)


# ── RMSNorm ─────────────────────────────────────────────────────────────

def _make_rmsnorm(dim: int, eps: float = NORM_EPS) -> nn.Module:
    return _RMSNorm(dim, eps)


# ── RoPE ────────────────────────────────────────────────────────────────

def precompute_rope_freqs(max_seq_len: int = MAX_SEQ_LEN, device="cuda") -> torch.Tensor:
    inv_freq = 1.0 / (
        ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device=device) / HEAD_DIM)
    )
    seq = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(seq, inv_freq)
    return torch.cat((freqs, freqs), dim=-1)  # [S, D]


def apply_rope(t: torch.Tensor, freqs_full: torch.Tensor) -> torch.Tensor:
    cos_ = torch.cos(freqs_full).to(t.dtype)[None, :, None, :]
    sin_ = torch.sin(freqs_full).to(t.dtype)[None, :, None, :]
    x1 = t[..., : t.shape[-1] // 2]
    x2 = t[..., t.shape[-1] // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return t * cos_ + rotated * sin_


# ── Transformer layer ──────────────────────────────────────────────────

class TransformerLayer(nn.Module):
    """A standard MiniCPM4 layer.

    `depth_scale` is applied multiplicatively to the residual *output* of
    each branch (attention and MLP) — this is muP for depth.
    """
    def __init__(self, layer_idx: int, depth_scale: float = 1.0):
        super().__init__()
        self.layer_idx = layer_idx
        self.depth_scale = depth_scale
        qkv_dim = NUM_HEADS * HEAD_DIM + 2 * NUM_KV_HEADS * HEAD_DIM
        self.attention_norm = _make_rmsnorm(HIDDEN_SIZE)
        self.wqkv = _Linear(HIDDEN_SIZE, qkv_dim, bias=False)
        self.wo = _Linear(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False)
        self.ffn_norm = _make_rmsnorm(HIDDEN_SIZE)
        self.wfc1 = _Linear(HIDDEN_SIZE, 2 * FFN_HIDDEN_SIZE, bias=False)
        self.w2 = _Linear(FFN_HIDDEN_SIZE, HIDDEN_SIZE, bias=False)

    def forward(self, hidden: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        from flash_attn import flash_attn_func
        B, S, _ = hidden.shape
        normed = self.attention_norm(hidden)
        qkv = self.wqkv(normed)
        nkv = NUM_KV_HEADS
        nq_per_kv = NUM_HEADS // nkv
        d = HEAD_DIM
        qkv = qkv.view(B, S, nkv, (nq_per_kv + 2) * d)
        q = qkv[..., : nq_per_kv * d].reshape(B, S, NUM_HEADS, d)
        # flash_attn_func supports GQA natively (kv heads < q heads),
        # so K/V stay at NUM_KV_HEADS — no `repeat_interleave` needed.
        k = qkv[..., nq_per_kv * d : nq_per_kv * d + d].contiguous()
        v = qkv[..., nq_per_kv * d + d:].contiguous()
        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)
        # flash_attn_func layout: [B, S, H, D] (not [B, H, S, D]). The
        # `deterministic=True` flag forces the split-K reduce backward
        # so backward is bitwise across runs (verified at this shape on
        # H100 / torch 2.8 NGC / flash_attn 2.7.4.post1).
        attn = flash_attn_func(q, k, v, causal=True, deterministic=True)
        attn = attn.reshape(B, S, NUM_HEADS * d)
        hidden = hidden + self.wo(attn) * self.depth_scale

        normed2 = self.ffn_norm(hidden)
        gated = self.wfc1(normed2)
        y_1, y_2 = gated.chunk(2, dim=-1)
        intermediate = (F.silu(y_1.float()) * y_2.float()).to(y_1.dtype)
        hidden = hidden + self.w2(intermediate) * self.depth_scale
        return hidden


# ── MTP layer (Eagle) ──────────────────────────────────────────────────

class MTPLayer(nn.Module):
    """Megatron-LM-cpm_core_r0.15.0's MultiTokenPredictionLayer — single eagle layer.

        h_emb     = LN(mtp_token_emb)                      # self.emb_input_layernorm (enorm)
        h_hidden  = LN(llm_hidden)                         # self.hidden_input_layernorm (hnorm)
        h         = eagle_fc(concat(h_emb, h_hidden))      # 2h -> h, no bias
        h         = transformer_layer(h)                   # 1 standard layer
        h         = LN_final(h)                            # self.final_layernorm — applied before
                                                           # the shared LM head, matches Megatron
                                                           # MultiTokenPredictionLayer._postprocess()
                                                           # (multi_token_prediction.py:607-608).
    """
    def __init__(self, depth_scale: float):
        super().__init__()
        self.emb_input_layernorm = _make_rmsnorm(HIDDEN_SIZE)
        self.hidden_input_layernorm = _make_rmsnorm(HIDDEN_SIZE)
        self.eagle_fc = _Linear(2 * HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.layer = TransformerLayer(0, depth_scale=depth_scale)
        # Final RMSNorm before the shared LM head. In Megatron this is the
        # `self.final_layernorm` instantiated at MultiTokenPredictionLayer.__init__
        # (multi_token_prediction.py:480-485) and applied in `_postprocess`
        # (multi_token_prediction.py:607-608). Without it the mtp logits enter
        # the shared LM head un-normed, which (a) breaks the muP scale symmetry
        # between main and mtp paths (main feeds `self.norm(hidden) / width_mult`
        # while mtp would feed `transformer_out / width_mult`) and (b) introduces
        # a systematic offset in mtp_loss → total_loss → all-param grad relative
        # to the Megatron baseline.
        self.final_layernorm = _make_rmsnorm(HIDDEN_SIZE)

    def forward(self, mtp_token_emb: torch.Tensor, llm_hidden: torch.Tensor,
                rope_freqs: torch.Tensor) -> torch.Tensor:
        a = self.emb_input_layernorm(mtp_token_emb)
        b = self.hidden_input_layernorm(llm_hidden)
        h = self.eagle_fc(torch.cat([a, b], dim=-1))
        h = self.layer(h, rope_freqs)
        return self.final_layernorm(h)


# ── Full model with muP + optional MTP ─────────────────────────────────

class MiniCPM4MupMtp(nn.Module):
    """MiniCPM4 0.5B with muP (always on) and optional MTP."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        mup_base_hidden_size: int = 256,
        mup_emb_scale: float = 12.0,
        mup_depth_scale: float = 1.4,
        eagle_num_layers: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.width_mult = HIDDEN_SIZE / mup_base_hidden_size
        self.mup_emb_scale = mup_emb_scale
        self.mup_depth_scale = mup_depth_scale
        self.eagle_num_layers = eagle_num_layers

        # main depth scale = 1.4 / sqrt(N_main)
        depth_scale_main = mup_depth_scale / math.sqrt(NUM_LAYERS)
        # mtp depth scale uses N_main + N_eagle (matches Megatron's
        # num_layers_for_mup_scale = num_eagle_layers + num_layers)
        depth_scale_mtp = mup_depth_scale / math.sqrt(NUM_LAYERS + max(eagle_num_layers, 0))

        self.tok_embeddings = _Embedding(vocab_size, HIDDEN_SIZE)
        self.layers = nn.ModuleList([TransformerLayer(i, depth_scale=depth_scale_main)
                                     for i in range(NUM_LAYERS)])
        self.norm = _make_rmsnorm(HIDDEN_SIZE)
        self.output = _Linear(HIDDEN_SIZE, vocab_size, bias=False)

        if eagle_num_layers > 0:
            assert eagle_num_layers == 1, "Currently only 1-layer MTP is implemented"
            self.mtp = MTPLayer(depth_scale=depth_scale_mtp)
        else:
            self.mtp = None

    def _apply_blocks(self, hidden: torch.Tensor, rope_freqs: torch.Tensor,
                       recompute: bool,
                       recompute_num_layers: int = -1) -> torch.Tensor:
        """Apply backbone transformer layers, optionally with activation
        checkpointing.

        - ``recompute=False``: never checkpoint.
        - ``recompute=True`` + ``recompute_num_layers < 0``: checkpoint all
          ``NUM_LAYERS`` layers (full recompute, baseline behavior).
        - ``recompute=True`` + ``0 <= recompute_num_layers <= NUM_LAYERS``:
          checkpoint only the **last** ``recompute_num_layers`` layers
          (the first ``NUM_LAYERS - recompute_num_layers`` layers keep
          activations in memory). Trades activation memory for a
          recompute cost concentrated in the later layers.
        """
        from torch.utils.checkpoint import checkpoint as _ckpt
        if not recompute:
            for layer in self.layers:
                hidden = layer(hidden, rope_freqs)
            return hidden
        L = len(self.layers)
        n_ckpt = L if recompute_num_layers < 0 else min(max(recompute_num_layers, 0), L)
        n_no_ckpt = L - n_ckpt
        for i, layer in enumerate(self.layers):
            if i < n_no_ckpt:
                hidden = layer(hidden, rope_freqs)
            else:
                hidden = _ckpt(layer, hidden, rope_freqs, use_reentrant=False)
        return hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        rope_freqs: torch.Tensor,
        mtp_input_ids: torch.Tensor | None = None,
        recompute: bool = False,
        recompute_num_layers: int = -1,
    ):
        """Returns ``(main_logits, mtp_logits_or_None)``.

        Both elements are logits of shape ``[B, S, V]`` — i.e.
        ``self.output(hidden_normed / width_mult)`` for the main branch,
        and same for the MTP branch when MTP is on.
        """
        # main branch
        hidden = self.tok_embeddings(input_ids) * self.mup_emb_scale
        hidden = self._apply_blocks(hidden, rope_freqs, recompute, recompute_num_layers)
        # In Megatron's mcore, self.decoder includes final_layernorm,
        # so the hidden it returns is already normed. We apply final norm
        # here once and feed the SAME normed tensor to both lm-head and MTP.
        hidden_normed = self.norm(hidden)
        # muP: divide by width_mult before lm_head
        main_pre_head = hidden_normed / self.width_mult

        if self.mtp is None or mtp_input_ids is None:
            return self.output(main_pre_head), None

        # MTP branch — Megatron passes the (already normed) hidden directly
        # to MultiTokenPredictionLayer. The MTP wrapper itself owns BOTH the
        # two input LayerNorms (enorm / hnorm) AND a final_layernorm applied
        # after the inner transformer layer and before the shared LM head
        # (see Megatron-LM-cpm_core_r0.15.0
        # `multi_token_prediction.py:480-485` and `:607-608`). MTPLayer.forward
        # already applies `self.final_layernorm`, so `mtp_hidden` is normed
        # here, mirroring `hidden_normed` on the main path; both feed the
        # shared `output` after dividing by `width_mult` (muP scale symmetry).
        mtp_emb = self.tok_embeddings(mtp_input_ids) * self.mup_emb_scale
        mtp_hidden = self.mtp(mtp_emb, hidden_normed, rope_freqs)
        mtp_pre_head = mtp_hidden / self.width_mult

        return self.output(main_pre_head), self.output(mtp_pre_head)

    # ── muP-aware initialization ────────────────────────────────────
    def init_weights(self, init_std: float = 0.1, seed: int = 1234,
                     *, init_ones: bool = True):
        """muP init.

           - Linear matrix weight (qkv, fc1, eagle_fc): N(0, init_std/sqrt(width_mult))
           - embedding / output_layer / wo / w2: N(0, init_std) (standard init)
           - 1-D / norm weight: 1.0 if ``init_ones`` else 0.97

        ``init_ones=False`` makes every 1-D parameter strictly ≠ 1.0; the
        engine must produce bit-equal forward/backward for either value
        and must not branch on the parameter.
        """
        scaled_std = init_std / math.sqrt(self.width_mult)
        rng = torch.Generator().manual_seed(seed)
        for name, p in self.named_parameters():
            if p.ndim == 1:
                if init_ones:
                    nn.init.ones_(p)
                else:
                    p.data.fill_(0.97)
                continue
            tail = name.rsplit(".", 1)[-1]
            module_name = name.rsplit(".", 2)[-2] if "." in name else ""
            # standard init for: embedding, output_layer, wo, w2, eagle_fc input/output (eagle_fc init follows config.init_method which under muP = mup_init_method, so use scaled_std)
            if name == "tok_embeddings.weight" or name == "output.weight":
                std = init_std
            elif module_name in ("wo", "w2"):
                # Megatron sets output_layer_init_method = mup_init_method when use_mup,
                # which equals scaled_std (sigma/sqrt(width_mult)).
                std = scaled_std
            else:
                std = scaled_std
            p.data.copy_(
                torch.randn(p.shape, generator=rng, dtype=torch.float32)
                .mul_(std)
                .to(dtype=p.dtype, device=p.device)
            )
        if not init_ones:
            for n, p in self.named_parameters():
                assert not (p.detach().float() == 1.0).any(), (
                    f"init_ones=False but {n} contains 1.0"
                )

    # ── Per-param lr_mult for muP ──────────────────────────────────
    def mup_lr_groups(self, base_lr: float):
        """Return parameter groups with muP lr scaling.

        Megatron rule (training.py):
          - matrix weights (.weight, not embedding/output_layer/layernorm) → lr / width_mult
          - everything else → base_lr
        """
        scaled, unscaled = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_matrix = (
                name.endswith(".weight")
                and "norm" not in name.lower()
                and "tok_embeddings" not in name
                and "output.weight" not in name
            )
            if is_matrix:
                scaled.append(p)
            else:
                unscaled.append(p)
        return [
            {"params": scaled,   "lr": base_lr / self.width_mult},
            {"params": unscaled, "lr": base_lr},
        ]
