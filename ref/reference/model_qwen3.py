"""Qwen3 0.6B — pure PyTorch dense GQA + QK-Norm (no muP, no MTP, no TE).

This is the L0 reference architecture for the Qwen3 model axis. It is a
sibling of ``model_pure_mup_mtp.py`` (the MiniCPM4/5 reference) and reuses
that module's fp32 weight-gradient accumulation machinery verbatim
(``_LinearFn`` / ``_EmbeddingFn`` / ``_RMSNormFn``), but the transformer
geometry is Qwen3, which differs from MiniCPM in four structural ways:

    1. No muP. Qwen3 uses standard parameterization: no width_mult init
       rescale, no embedding output scale, no per-branch residual depth
       scale, no per-matrix lr split. ``init_method`` is a plain
       ``N(0, init_std)`` on every weight.
    2. No MTP / Eagle layer. ``forward`` returns main logits only.
    3. Separate q / k / v projections (NOT a single fused ``wqkv``), each
       bias-free (Qwen3 ``attention_bias=false``).
    4. QK-Norm: an RMSNorm over the per-head dimension applied to the query
       and key projections BEFORE RoPE (Qwen3's signature structure, absent
       in MiniCPM/Llama). The norm weight has shape ``[head_dim]`` and is
       shared across heads, matching HF ``Qwen3Attention.q_norm`` /
       ``k_norm``.
    5. Tied embeddings: the LM head reuses ``tok_embeddings.weight``
       (``tie_word_embeddings=true``), so there is no independent
       ``output`` matrix.

The attention math (RoPE on ``[B, S, H, D]`` then ``flash_attn_func`` with
native GQA and ``deterministic=True``) is identical to the MiniCPM
reference so the determinism / bitwise contract carries over unchanged.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the fp32-wgrad autograd primitives + RoPE helpers from the MiniCPM
# reference. These are pure architecture-agnostic building blocks (a
# bias-free Linear, an Embedding lookup, an RMSNorm, and the rotary
# embedding) — the muP / MTP specifics live only in MiniCPM4MupMtp, which
# we do NOT import. Importing keeps a single SSOT for the subtle fp32
# main_grad side-channel that survives activation checkpointing.
from model_pure_mup_mtp import (  # noqa: E402
    _Embedding,
    _Linear,
    _RMSNorm,
    apply_rope,
)

# Module-level architecture constants, read from the environment at import
# under the generic upper-cased product-key names (the [model] config axis →
# rendered gate product → tools/product_env.py projection). FAIL-FAST by
# design: a missing geometry key raises KeyError instead of silently falling
# back to a baked Qwen3-0.6B default — with the per-key registry gone, this
# is the guarantee that a render/projection gap can never train the wrong
# architecture. Standalone imports must pre-project the product (or set the
# ten keys explicitly, as the CUDA smoke test does).
NUM_LAYERS = int(os.environ["NUM_LAYERS"])
HIDDEN_SIZE = int(os.environ["HIDDEN_SIZE"])
NUM_HEADS = int(os.environ["NUM_ATTENTION_HEADS"])
NUM_KV_HEADS = int(os.environ["NUM_QUERY_GROUPS"])
# head_dim is an explicit knob — Qwen3 0.6B uses 128 with hidden=1024,
# heads=16 (1024//16=64 != 128), so it canNOT be derived from hidden//heads.
HEAD_DIM = int(os.environ["HEAD_DIM"])
FFN_HIDDEN_SIZE = int(os.environ["FFN_HIDDEN_SIZE"])
VOCAB_SIZE = int(os.environ["PADDED_VOCAB_SIZE"])
MAX_SEQ_LEN = int(os.environ["MAX_POSITION_EMBEDDINGS"])
NORM_EPS = float(os.environ["NORM_EPSILON"])
ROPE_THETA = float(os.environ["ROTARY_BASE"])
# Per-head QK-Norm and tied word embeddings are INTRINSIC to the Qwen3
# architecture — not tunable [model] knobs and not part of the shared
# FORGE_* model schema. They are hardcoded on here (a Qwen3 without them is
# simply not Qwen3). If a future Qwen variant ever needs to toggle either,
# promote it to a model knob then; until then keep them out of the schema so
# the MiniCPM templates and the engine ModelHParams contract stay untouched.
USE_QK_NORM = True
TIE_WORD_EMBEDDINGS = True


def _make_rmsnorm(dim: int, eps: float = NORM_EPS) -> nn.Module:
    return _RMSNorm(dim, eps)


# ── RoPE ────────────────────────────────────────────────────────────────
# Qwen3 uses the standard NeoX-style RoPE with a (large) rotary base. We
# reuse the MiniCPM reference's ``apply_rope`` (rotate-half on the full
# head_dim) and only re-derive the inverse-frequency table against the
# Qwen3 head_dim + theta.

def precompute_rope_freqs(max_seq_len: int = MAX_SEQ_LEN, device="cuda") -> torch.Tensor:
    inv_freq = 1.0 / (
        ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device=device) / HEAD_DIM)
    )
    seq = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(seq, inv_freq)
    return torch.cat((freqs, freqs), dim=-1)  # [S, D]


# ── Transformer layer ──────────────────────────────────────────────────

class TransformerLayer(nn.Module):
    """A standard Qwen3 dense decoder layer (GQA + QK-Norm + SwiGLU).

    No muP depth scale: the residual update is the plain ``x + branch(x)``
    of standard parameterization.
    """

    def __init__(self, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention_norm = _make_rmsnorm(HIDDEN_SIZE)
        # Separate q/k/v projections (Qwen3 attention_bias=false).
        self.wq = _Linear(HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, bias=False)
        self.wk = _Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.wv = _Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.wo = _Linear(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False)
        # QK-Norm: RMSNorm over the per-head dimension, shared across heads.
        if USE_QK_NORM:
            self.q_norm = _make_rmsnorm(HEAD_DIM)
            self.k_norm = _make_rmsnorm(HEAD_DIM)
        else:
            self.q_norm = None
            self.k_norm = None
        self.ffn_norm = _make_rmsnorm(HIDDEN_SIZE)
        # SwiGLU: fused gate+up projection (2*ffn), then down projection.
        self.wfc1 = _Linear(HIDDEN_SIZE, 2 * FFN_HIDDEN_SIZE, bias=False)
        self.w2 = _Linear(FFN_HIDDEN_SIZE, HIDDEN_SIZE, bias=False)

    def forward(self, hidden: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        from flash_attn import flash_attn_func
        B, S, _ = hidden.shape
        d = HEAD_DIM
        normed = self.attention_norm(hidden)
        q = self.wq(normed).view(B, S, NUM_HEADS, d)
        k = self.wk(normed).view(B, S, NUM_KV_HEADS, d)
        v = self.wv(normed).view(B, S, NUM_KV_HEADS, d)
        # QK-Norm BEFORE RoPE (Qwen3 contract): normalize each head vector.
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)
        # flash_attn_func supports GQA natively (kv heads < q heads), layout
        # [B, S, H, D]. deterministic=True forces the split-K reduce backward
        # so backward is bitwise across runs.
        attn = flash_attn_func(q, k.contiguous(), v.contiguous(), causal=True, deterministic=True)
        attn = attn.reshape(B, S, NUM_HEADS * d)
        hidden = hidden + self.wo(attn)

        normed2 = self.ffn_norm(hidden)
        gated = self.wfc1(normed2)
        y_1, y_2 = gated.chunk(2, dim=-1)
        intermediate = (F.silu(y_1.float()) * y_2.float()).to(y_1.dtype)
        hidden = hidden + self.w2(intermediate)
        return hidden


# ── Full model (dense, standard param, tied head) ───────────────────────

class Qwen3Dense(nn.Module):
    """Qwen3 0.6B dense decoder with QK-Norm and tied embeddings.

    Standard parameterization (no muP), no MTP. ``forward`` returns
    ``(main_logits, None)`` so the training entry's two-tuple unpacking is
    shared with the MiniCPM reference.
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, tie_word_embeddings: bool = TIE_WORD_EMBEDDINGS):
        super().__init__()
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings

        self.tok_embeddings = _Embedding(vocab_size, HIDDEN_SIZE)
        self.layers = nn.ModuleList([TransformerLayer(i) for i in range(NUM_LAYERS)])
        self.norm = _make_rmsnorm(HIDDEN_SIZE)
        if tie_word_embeddings:
            # LM head shares the embedding matrix — no independent weight.
            self.output = None
        else:
            self.output = _Linear(HIDDEN_SIZE, vocab_size, bias=False)

    def _lm_head(self, hidden_normed: torch.Tensor) -> torch.Tensor:
        if self.tie_word_embeddings:
            # x @ Eᵀ via the shared embedding matrix. Route through the same
            # fp32-wgrad Linear function used everywhere else so the tied
            # weight accumulates its (multi-source) gradient in fp32.
            from model_pure_mup_mtp import _LinearFn
            return _LinearFn.apply(hidden_normed, self.tok_embeddings.weight)
        return self.output(hidden_normed)

    def _apply_blocks(self, hidden: torch.Tensor, rope_freqs: torch.Tensor,
                      recompute: bool, recompute_num_layers: int = -1) -> torch.Tensor:
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
        """Returns ``(main_logits, None)``.

        ``mtp_input_ids`` is accepted (and ignored) so the training entry
        can call this model with the same signature as the MiniCPM MTP
        reference; Qwen3 has no MTP branch.
        """
        del mtp_input_ids  # Qwen3 has no MTP
        hidden = self.tok_embeddings(input_ids)
        hidden = self._apply_blocks(hidden, rope_freqs, recompute, recompute_num_layers)
        hidden_normed = self.norm(hidden)
        return self._lm_head(hidden_normed), None

    # ── Standard initialization (no muP rescale) ────────────────────
    def init_weights(self, init_std: float = 0.02, seed: int = 1234):
        """Standard init: every matrix weight ~ N(0, init_std); norms = 1.

        No muP width_mult rescale — Qwen3 uses ``initializer_range`` (0.02)
        applied uniformly. The tied LM head has no separate weight to init.
        """
        rng = torch.Generator().manual_seed(seed)
        for _name, p in self.named_parameters():
            if p.ndim == 1:
                nn.init.ones_(p)
                continue
            p.data.copy_(
                torch.randn(p.shape, generator=rng, dtype=torch.float32)
                .mul_(init_std)
                .to(dtype=p.dtype, device=p.device)
            )

    # ── Per-param lr groups (standard, no muP split) ────────────────
    def lr_groups(self, base_lr: float):
        """Return a single lr group — standard parameterization applies one
        lr to every parameter (no muP matrix/non-matrix split).

        Shaped like ``MiniCPM4MupMtp.mup_lr_groups`` (a list of
        ``{"params", "lr"}`` dicts) so the training entry's optimizer-group
        construction is shared.
        """
        params = [p for p in self.parameters() if p.requires_grad]
        return [{"params": params, "lr": base_lr}]

    # Alias so the training entry can call the same method name regardless
    # of which reference model it imports.
    mup_lr_groups = lr_groups


# Module-level width_mult shim. The MiniCPM training entry divides logits by
# ``model.width_mult``; Qwen3 has no muP so it is identically 1.0. Exposed as
# an attribute set in __init__ would require the entry to special-case; a
# class attribute of 1.0 keeps the shared entry path arithmetic-neutral.
Qwen3Dense.width_mult = 1.0
Qwen3Dense.mup_emb_scale = 1.0
