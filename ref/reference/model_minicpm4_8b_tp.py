"""MiniCPM4 8B — pure PyTorch + muP, no MTP, with self-written tensor parallel.

Derived from ``model_pure_mup_mtp.py`` (the 0.5B ref). Differences:

* **8B geometry** via the same ``FORGE_*`` env knobs (32 layers, hidden 4096,
  ffn 16384, 32 q-heads / 2 kv-heads GQA, head_dim 128, vocab 73448).
* **muP is kept**, with formulas byte-identical to the 0.5B ref — only
  ``width_mult = hidden_size / mup_base_hidden_size = 4096/256 = 16`` differs.
  The official ``mup_denominator`` is vestigial (the published forward uses
  standard ``1/sqrt(head_dim)`` attention and never references it), so there is
  no extra attention scaling here.
* **MTP / Eagle dropped** entirely (the official MiniCPM4-8B has no MTP head).
* **Tensor parallel (TP)** written from scratch on top of the same custom
  autograd Functions (``_LinearFn`` etc. — fp32 wgrad accumulation) so the
  bitwise contract survives. ``tp_size == 1`` degenerates to the dense path:
  every collective is a no-op and every shard spans the full dimension, so the
  single-card proxy and the DP+TP run share one code path.

TP layout (sharded across the TP group; ``tp_size`` must divide the sharded
dim):

* ``tok_embeddings``  — vocab-parallel embedding (mask out-of-range ids, local
  lookup, all-reduce over TP).
* ``wqkv``            — column-parallel. The fused QKV output is kv-group-major
  (``nkv * (nq_per_kv + 2) * head_dim``), so a contiguous output shard already
  carries whole GQA groups → local reshape works with ``nkv // tp`` kv heads.
* ``wo``              — row-parallel (input sharded by head, all-reduce output).
* ``wfc1``            — column-parallel with gate/up interleave: each rank holds
  ``gate[r_slice]`` and ``up[r_slice]`` of the *same* FFN sub-range so the local
  ``chunk(2)`` pairs matching neurons.
* ``w2``              — row-parallel (input sharded by FFN, all-reduce output).
* ``output`` (lm_head) — column-parallel by vocab; logits stay vocab-sharded and
  the loss is a vocab-parallel cross entropy (no full-logit gather).

RMSNorm / all 1-D params are **replicated** (full hidden on every rank); their
weight gradients are TP-all-reduced by ``harness_dptp`` (not here). Sharded
params are tagged ``param.tp_sharded = True`` so the reducer can tell them apart.
"""
from __future__ import annotations

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# Module-level architecture constants, read from the environment at import
# time. The rendered gate product is the single source of truth: the caller
# projects it via tools/product_env.py (every [model] key under its
# upper-cased name), so a missing key is a render/projection bug — fail fast
# rather than silently fall back to a baked-in shape.
NUM_LAYERS = int(os.environ["NUM_LAYERS"])
HIDDEN_SIZE = int(os.environ["HIDDEN_SIZE"])
NUM_HEADS = int(os.environ["NUM_ATTENTION_HEADS"])
NUM_KV_HEADS = int(os.environ["NUM_QUERY_GROUPS"])
HEAD_DIM = int(os.environ["HEAD_DIM"])
FFN_HIDDEN_SIZE = int(os.environ["FFN_HIDDEN_SIZE"])
VOCAB_SIZE = int(os.environ["PADDED_VOCAB_SIZE"])
MAX_SEQ_LEN = int(os.environ["MAX_POSITION_EMBEDDINGS"])
NORM_EPS = float(os.environ["NORM_EPSILON"])
ROPE_THETA = float(os.environ["ROTARY_BASE"])


# ── fp32 weight-gradient accumulation (verbatim from the 0.5B ref) ──────────
#
# The custom autograd Functions route every weight matrix / embedding / RMSNorm
# weight through a backward that computes wgrad on the same bf16 compute path as
# forward and upcasts to fp32 once before adding to ``param.main_grad``. This
# mirrors the Megatron + TransformerEngine fused-wgrad GEMM (fp32 accumulation),
# which is required for bitwise reproducibility. See the 0.5B ref for the full
# rationale (activation-checkpoint Parameter-identity note included).
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
            wg = torch.ops.aten.embedding_dense_backward(grad_out, idx, v, -1, False)
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


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _RMSNormFn.apply(x, self.weight, self.eps)


def _make_rmsnorm(dim: int, eps: float = NORM_EPS) -> nn.Module:
    return _RMSNorm(dim, eps)


# ── TP process-group helpers ────────────────────────────────────────────────

def _tp_world(group) -> int:
    if group is not None and dist.is_available() and dist.is_initialized():
        return dist.get_world_size(group)
    return 1


def _tp_rank(group) -> int:
    if group is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank(group)
    return 0


class _CopyToTP(torch.autograd.Function):
    """Identity forward; all-reduce(SUM) the gradient across TP on backward.

    Sits at the *input* of a column-parallel linear: each TP rank produces a
    partial input-gradient (its output shard's contribution), and they sum to
    the true dx. No-op when ``tp_size == 1``.
    """

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad_out):
        if _tp_world(ctx.group) > 1:
            grad_out = grad_out.contiguous()
            dist.all_reduce(grad_out, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_out, None


class _ReduceFromTP(torch.autograd.Function):
    """All-reduce(SUM) forward across TP; identity backward.

    Sits at the *output* of a row-parallel linear (or vocab-parallel embedding):
    each rank holds a partial output and they sum to the full activation. The
    gradient is already replicated downstream, so backward is identity. No-op
    when ``tp_size == 1``.
    """

    @staticmethod
    def forward(ctx, x, group):
        if _tp_world(group) > 1:
            x = x.contiguous()
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
        return x

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None


def _tag_sharded(p: nn.Parameter) -> nn.Parameter:
    """Mark a parameter as TP-sharded (per-rank distinct) so the DP+TP grad
    reducer skips the replicated-grad TP all-reduce for it."""
    p.tp_sharded = True
    return p


# ── TP linear / embedding layers ────────────────────────────────────────────

class _ColumnParallelLinear(nn.Module):
    """y = x @ Wᵀ with W sharded along the output dim.

    ``interleave`` > 1 splits the logical output into that many equal blocks and
    shards each block independently, then concatenates the per-rank slices —
    used by the gated MLP's fused ``[gate | up]`` so each rank's local
    ``chunk(2)`` recovers matching neurons. Forward is a plain local matmul; the
    ``_CopyToTP`` wrapper supplies the backward input-grad all-reduce.
    """

    def __init__(self, in_features: int, out_features: int, tp_group, *,
                 interleave: int = 1, std_kind: str = "matrix"):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = _tp_world(tp_group)
        self.tp_rank = _tp_rank(tp_group)
        self.in_features = in_features
        self.out_features = out_features
        self.interleave = interleave
        self.std_kind = std_kind
        assert out_features % (self.tp_size * interleave) == 0, (
            f"out_features={out_features} not divisible by "
            f"tp_size*interleave={self.tp_size * interleave}")
        self.out_per_rank = out_features // self.tp_size
        self.weight = _tag_sharded(
            nn.Parameter(torch.empty(self.out_per_rank, in_features)))

    def shard_full_weight(self, full: torch.Tensor) -> torch.Tensor:
        """Slice a full logical ``[out_features, in_features]`` weight into this
        rank's shard (used by init so ref/ours agree per shard)."""
        blocks = full.chunk(self.interleave, dim=0)
        per = self.out_features // (self.tp_size * self.interleave)
        r = self.tp_rank
        return torch.cat([b[r * per:(r + 1) * per] for b in blocks], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _CopyToTP.apply(x, self.tp_group)
        return _LinearFn.apply(x, self.weight)


class _RowParallelLinear(nn.Module):
    """y = sum_tp(x_shard @ W_shardᵀ) with W sharded along the input dim.

    Input is assumed already sharded along ``in_features`` (it comes from a
    matching column-parallel producer). Forward does a local matmul then
    all-reduces the partial outputs across TP via ``_ReduceFromTP``.
    """

    def __init__(self, in_features: int, out_features: int, tp_group, *,
                 std_kind: str = "matrix"):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = _tp_world(tp_group)
        self.tp_rank = _tp_rank(tp_group)
        self.in_features = in_features
        self.out_features = out_features
        self.std_kind = std_kind
        assert in_features % self.tp_size == 0, (
            f"in_features={in_features} not divisible by tp_size={self.tp_size}")
        self.in_per_rank = in_features // self.tp_size
        self.weight = _tag_sharded(
            nn.Parameter(torch.empty(out_features, self.in_per_rank)))

    def shard_full_weight(self, full: torch.Tensor) -> torch.Tensor:
        """Slice a full logical ``[out_features, in_features]`` weight along the
        input dim into this rank's column block."""
        per = self.in_per_rank
        r = self.tp_rank
        return full[:, r * per:(r + 1) * per].contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = _LinearFn.apply(x, self.weight)
        return _ReduceFromTP.apply(y, self.tp_group)


class _VocabParallelEmbedding(nn.Module):
    """Embedding with the vocabulary sharded across TP.

    Each rank owns ``vocab // tp`` rows; out-of-range ids are masked to a local
    zero lookup and their output rows are zeroed with a differentiable multiply
    (so the masked positions contribute no weight gradient), then the per-rank
    partial embeddings are summed with ``_ReduceFromTP``.
    """

    def __init__(self, vocab_size: int, dim: int, tp_group):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = _tp_world(tp_group)
        self.tp_rank = _tp_rank(tp_group)
        assert vocab_size % self.tp_size == 0, (
            f"vocab_size={vocab_size} not divisible by tp_size={self.tp_size}")
        self.vocab_per_rank = vocab_size // self.tp_size
        self.vocab_start = self.tp_rank * self.vocab_per_rank
        self.vocab_end = self.vocab_start + self.vocab_per_rank
        self.weight = _tag_sharded(
            nn.Parameter(torch.empty(self.vocab_per_rank, dim)))

    def shard_full_weight(self, full: torch.Tensor) -> torch.Tensor:
        return full[self.vocab_start:self.vocab_end].contiguous()

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return _EmbeddingFn.apply(idx, self.weight)
        mask = (idx < self.vocab_start) | (idx >= self.vocab_end)
        local_idx = (idx - self.vocab_start)
        local_idx = local_idx.masked_fill(mask, 0)
        y = _EmbeddingFn.apply(local_idx, self.weight)
        y = y * (~mask).unsqueeze(-1).to(y.dtype)
        return _ReduceFromTP.apply(y, self.tp_group)


class _VocabParallelCrossEntropy(torch.autograd.Function):
    """Cross entropy over vocab-sharded logits, without gathering full logits.

    Mirrors Megatron's vocab-parallel CE: global max via all-reduce(MAX), global
    denominator via all-reduce(SUM) of local exp-sums, target logit gathered on
    the owning rank and all-reduce(SUM)'d. Returns per-token NLL (identical on
    every TP rank). Backward gives the softmax-minus-one-hot dgrad on the local
    vocab shard. All collectives are no-ops at ``tp_size == 1``.
    """

    @staticmethod
    def forward(ctx, logits_shard, target, group, vocab_start, vocab_per_rank):
        # logits_shard: [N, vocab_per_rank] (fp32 for a numerically clean CE).
        logits_shard = logits_shard.float()
        logit_max = logits_shard.max(dim=-1).values
        if _tp_world(group) > 1:
            dist.all_reduce(logit_max, op=dist.ReduceOp.MAX, group=group)
        shifted = logits_shard - logit_max.unsqueeze(-1)
        exp = shifted.exp()
        sum_exp = exp.sum(dim=-1)
        if _tp_world(group) > 1:
            dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=group)

        vocab_end = vocab_start + vocab_per_rank
        tgt_mask = (target < vocab_start) | (target >= vocab_end)
        local_tgt = (target - vocab_start).masked_fill(tgt_mask, 0)
        pred_logit = logits_shard.gather(-1, local_tgt.unsqueeze(-1)).squeeze(-1)
        pred_logit = pred_logit.masked_fill(tgt_mask, 0.0)
        if _tp_world(group) > 1:
            dist.all_reduce(pred_logit, op=dist.ReduceOp.SUM, group=group)

        log_sum_exp = sum_exp.log() + logit_max
        loss = log_sum_exp - pred_logit  # [N]

        softmax = exp / sum_exp.unsqueeze(-1)
        ctx.save_for_backward(softmax, local_tgt, tgt_mask)
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        softmax, local_tgt, tgt_mask = ctx.saved_tensors
        grad = softmax.clone()
        # subtract one-hot at the (locally owned, unmasked) target column
        one_hot = torch.zeros_like(softmax)
        one_hot.scatter_(-1, local_tgt.unsqueeze(-1), 1.0)
        one_hot = one_hot * (~tgt_mask).unsqueeze(-1).to(one_hot.dtype)
        grad = grad - one_hot
        grad = grad * grad_loss.unsqueeze(-1)
        return grad, None, None, None, None


def vocab_parallel_cross_entropy(logits_shard, target, output_layer):
    """Per-token NLL for vocab-sharded ``logits_shard`` against global
    ``target`` ids, using the TP geometry of ``output_layer`` (a
    :class:`_ColumnParallelLinear` over vocab)."""
    return _VocabParallelCrossEntropy.apply(
        logits_shard, target, output_layer.tp_group,
        output_layer.vocab_start, output_layer.vocab_per_rank)


# ── RoPE ────────────────────────────────────────────────────────────────────

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


# ── Transformer layer (TP) ──────────────────────────────────────────────────

class TransformerLayer(nn.Module):
    """A MiniCPM4 layer with tensor-parallel attention + gated MLP.

    ``depth_scale`` multiplies each branch's residual output (muP for depth).
    """

    def __init__(self, layer_idx: int, tp_group, depth_scale: float = 1.0):
        super().__init__()
        self.layer_idx = layer_idx
        self.depth_scale = depth_scale
        self.tp_group = tp_group
        self.tp_size = _tp_world(tp_group)
        assert NUM_HEADS % self.tp_size == 0 and NUM_KV_HEADS % self.tp_size == 0, (
            f"NUM_HEADS={NUM_HEADS}/NUM_KV_HEADS={NUM_KV_HEADS} not divisible "
            f"by tp_size={self.tp_size}")
        self.nkv_local = NUM_KV_HEADS // self.tp_size
        self.nq_per_kv = NUM_HEADS // NUM_KV_HEADS
        self.nq_local = NUM_HEADS // self.tp_size

        qkv_dim = NUM_HEADS * HEAD_DIM + 2 * NUM_KV_HEADS * HEAD_DIM
        self.attention_norm = _make_rmsnorm(HIDDEN_SIZE)
        self.wqkv = _ColumnParallelLinear(HIDDEN_SIZE, qkv_dim, tp_group)
        self.wo = _RowParallelLinear(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, tp_group)
        self.ffn_norm = _make_rmsnorm(HIDDEN_SIZE)
        # gate+up fused → interleave=2 so each rank pairs gate[r]/up[r].
        self.wfc1 = _ColumnParallelLinear(
            HIDDEN_SIZE, 2 * FFN_HIDDEN_SIZE, tp_group, interleave=2)
        self.w2 = _RowParallelLinear(FFN_HIDDEN_SIZE, HIDDEN_SIZE, tp_group)

    def forward(self, hidden: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        from flash_attn import flash_attn_func
        B, S, _ = hidden.shape
        normed = self.attention_norm(hidden)
        qkv = self.wqkv(normed)
        d = HEAD_DIM
        nkv = self.nkv_local
        nq_per_kv = self.nq_per_kv
        qkv = qkv.view(B, S, nkv, (nq_per_kv + 2) * d)
        q = qkv[..., : nq_per_kv * d].reshape(B, S, self.nq_local, d)
        k = qkv[..., nq_per_kv * d: nq_per_kv * d + d].contiguous()
        v = qkv[..., nq_per_kv * d + d:].contiguous()
        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)
        attn = flash_attn_func(q, k, v, causal=True, deterministic=True)
        attn = attn.reshape(B, S, self.nq_local * d)
        hidden = hidden + self.wo(attn) * self.depth_scale

        normed2 = self.ffn_norm(hidden)
        gated = self.wfc1(normed2)
        y_1, y_2 = gated.chunk(2, dim=-1)
        intermediate = (F.silu(y_1.float()) * y_2.float()).to(y_1.dtype)
        hidden = hidden + self.w2(intermediate) * self.depth_scale
        return hidden


# ── Full model: muP + TP, no MTP ────────────────────────────────────────────

class MiniCPM4_8B_TP(nn.Module):
    """MiniCPM4 8B with muP (always on) and tensor parallel (tp_size>=1)."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        mup_base_hidden_size: int = 256,
        mup_emb_scale: float = 12.0,
        mup_depth_scale: float = 1.4,
        tp_group=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.width_mult = HIDDEN_SIZE / mup_base_hidden_size
        self.mup_emb_scale = mup_emb_scale
        self.mup_depth_scale = mup_depth_scale
        self.tp_group = tp_group
        self.tp_size = _tp_world(tp_group)
        self.tp_rank = _tp_rank(tp_group)

        depth_scale_main = mup_depth_scale / math.sqrt(NUM_LAYERS)

        self.tok_embeddings = _VocabParallelEmbedding(vocab_size, HIDDEN_SIZE, tp_group)
        self.layers = nn.ModuleList([
            TransformerLayer(i, tp_group, depth_scale=depth_scale_main)
            for i in range(NUM_LAYERS)
        ])
        self.norm = _make_rmsnorm(HIDDEN_SIZE)
        self.output = _ColumnParallelLinear(
            HIDDEN_SIZE, vocab_size, tp_group, std_kind="embed_output")
        # expose vocab geometry on the lm_head for the CE helper
        self.output.vocab_start = self.tp_rank * (vocab_size // self.tp_size)
        self.output.vocab_per_rank = vocab_size // self.tp_size

    def _apply_blocks(self, hidden, rope_freqs, recompute, recompute_num_layers=-1):
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

    def forward(self, input_ids, rope_freqs, recompute=False, recompute_num_layers=-1):
        """Return vocab-sharded logits ``[B, S, vocab // tp]``.

        Loss is computed by :func:`vocab_parallel_cross_entropy` against these
        logits in the training loop (no full-logit gather).
        """
        hidden = self.tok_embeddings(input_ids) * self.mup_emb_scale
        hidden = self._apply_blocks(hidden, rope_freqs, recompute, recompute_num_layers)
        hidden_normed = self.norm(hidden)
        main_pre_head = hidden_normed / self.width_mult
        return self.output(main_pre_head)

    # ── muP-aware, TP-consistent initialization ────────────────────────────
    def init_weights(self, init_std: float = 0.1, seed: int = 1234, *, init_ones: bool = True):
        """muP init that is bitwise-consistent across TP ranks.

        For every sharded weight the FULL logical tensor is drawn from a
        single seeded generator (identical draw order on every rank), then this
        rank's shard is sliced out via the layer's ``shard_full_weight``. Two
        runs at any ``tp_size`` thus agree per shard. std policy matches the
        0.5B ref: embedding / lm_head use ``init_std``; matrix weights use
        ``init_std / sqrt(width_mult)``; 1-D / norm weights are 1.0 (or 0.97
        when ``init_ones`` is False, the anti-cheat path the gate exercises).
        """
        scaled_std = init_std / math.sqrt(self.width_mult)
        rng = torch.Generator().manual_seed(seed)

        def draw_full(shape, std):
            return (torch.randn(shape, generator=rng, dtype=torch.float32)
                    .mul_(std))

        # Sharded weights — order is fixed (embedding, then per-layer attn/MLP,
        # then lm_head) so the RNG stream is identical on every rank.
        full = draw_full((self.vocab_size, HIDDEN_SIZE), init_std)
        self._set_shard(self.tok_embeddings, full)

        qkv_dim = NUM_HEADS * HEAD_DIM + 2 * NUM_KV_HEADS * HEAD_DIM
        for layer in self.layers:
            self._set_shard(layer.wqkv, draw_full((qkv_dim, HIDDEN_SIZE), scaled_std))
            self._set_shard(layer.wo, draw_full((HIDDEN_SIZE, NUM_HEADS * HEAD_DIM), scaled_std))
            self._set_shard(layer.wfc1, draw_full((2 * FFN_HIDDEN_SIZE, HIDDEN_SIZE), scaled_std))
            self._set_shard(layer.w2, draw_full((HIDDEN_SIZE, FFN_HIDDEN_SIZE), scaled_std))

        self._set_shard(self.output, draw_full((self.vocab_size, HIDDEN_SIZE), init_std))

        # Replicated 1-D / norm weights — identical on every rank.
        for name, p in self.named_parameters():
            if p.ndim == 1:
                if init_ones:
                    nn.init.ones_(p)
                else:
                    p.data.fill_(0.97)

        if not init_ones:
            for n, p in self.named_parameters():
                if p.ndim == 1:
                    assert not (p.detach().float() == 1.0).any(), (
                        f"init_ones=False but {n} contains 1.0")

    @staticmethod
    def _set_shard(layer, full: torch.Tensor) -> None:
        shard = layer.shard_full_weight(full)
        layer.weight.data.copy_(shard.to(dtype=layer.weight.dtype, device=layer.weight.device))

    # ── Per-param lr_mult for muP ──────────────────────────────────────────
    def mup_lr_groups(self, base_lr: float):
        """muP lr groups: matrix weights → base_lr/width_mult, else base_lr.

        Same rule as the 0.5B ref (Megatron training.py): a "matrix" weight is a
        ``.weight`` that is neither a norm nor the embedding / lm_head.
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
            (scaled if is_matrix else unscaled).append(p)
        return [
            {"params": scaled, "lr": base_lr / self.width_mult},
            {"params": unscaled, "lr": base_lr},
        ]
