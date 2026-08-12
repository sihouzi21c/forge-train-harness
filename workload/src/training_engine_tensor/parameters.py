"""Parameter management — LayerParameterView and model parameter containers.

The in-house engine owns bare ``torch.Tensor`` weight buffers directly
(no ``nn.Parameter``, no ``nn.Module``).  This module provides the
typed view that the forward / backward primitives navigate.

``LayerParameterView`` is a flat struct that groups the six weight
tensors of one transformer layer.  The top-level model container
(``ModelParameters``) holds the embedding, all layers, the final norm,
the output head, and the optional MTP layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from training_engine_tensor.config import (
    FFN_HIDDEN_SIZE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    NUM_HEADS,
    NUM_KV_HEADS,
    HEAD_DIM,
    VOCAB_SIZE,
)


@dataclass(frozen=True)
class LayerParameterView:
    """Flat weight-buffer view for one transformer layer.

    Each field is a ``torch.Tensor`` (bf16) that the forward/backward
    primitives read/write.  The tensor is the SSOT — no shadow copy.
    """

    index: int
    """Layer index (0-based)."""

    input_norm_weight: torch.Tensor
    qkv_weight: torch.Tensor
    attention_proj_weight: torch.Tensor
    pre_mlp_norm_weight: torch.Tensor
    mlp_fc1_weight: torch.Tensor
    mlp_fc2_weight: torch.Tensor

    @property
    def qkv_width(self) -> int:
        return NUM_HEADS * HEAD_DIM + 2 * NUM_KV_HEADS * HEAD_DIM


@dataclass(frozen=True)
class MTPParameterView:
    """Weight-buffer view for the optional Eagle MTP layer."""

    emb_input_norm_weight: torch.Tensor
    hidden_input_norm_weight: torch.Tensor
    eagle_fc_weight: torch.Tensor
    layer: LayerParameterView
    final_norm_weight: torch.Tensor


@dataclass(frozen=True)
class ModelParameters:
    """Top-level container for all model weight buffers.

    All tensors are bf16 (the compute dtype).  The optimizer owns a
    parallel FP32 master copy.
    """

    tok_embeddings_weight: torch.Tensor
    layers: list[LayerParameterView]
    final_norm_weight: torch.Tensor
    output_weight: torch.Tensor
    mtp: MTPParameterView | None


def create_layer_params(index: int, device: torch.device) -> LayerParameterView:
    """Create a layer's weight buffers filled with ones (placeholder).

    The actual weights are loaded from the canonical checkpoint by
    :func:`load_weights_from_checkpoint`; this factory exists only
    for shape/dtype discovery.
    """
    qkv_w = torch.empty(0, dtype=torch.bfloat16, device=device)
    attn_proj_w = torch.empty(0, dtype=torch.bfloat16, device=device)
    mlp_fc1_w = torch.empty(0, dtype=torch.bfloat16, device=device)
    mlp_fc2_w = torch.empty(0, dtype=torch.bfloat16, device=device)
    return LayerParameterView(
        index=index,
        input_norm_weight=qkv_w,
        qkv_weight=qkv_w,
        attention_proj_weight=attn_proj_w,
        pre_mlp_norm_weight=qkv_w,
        mlp_fc1_weight=mlp_fc1_w,
        mlp_fc2_weight=mlp_fc2_w,
    )


def _find_in_flat(flat_params: list[torch.Tensor], names: list[str],
                  name: str) -> torch.Tensor | None:
    """Look up a weight from the flat list by matching the name suffix."""
    for p, n in zip(flat_params, names):
        if n == name:
            return p
    return None


def _ref_fqn_to_internal(ref_fqn: str) -> str:
    """Translate the ref's FQN to the internal parameter name.

    The canonical checkpoint uses the ref's ``named_parameters()`` keys
    (e.g. ``tok_embeddings.weight``, ``layers.0.attention_norm.weight``).
    The internal engine uses flat names with underscores.
    """
    mapping = {
        "tok_embeddings.weight": "tok_embeddings_weight",
        "norm.weight": "final_norm_weight",
        "output.weight": "output_weight",
        "mtp.emb_input_layernorm.weight": "mtp.emb_input_norm_weight",
        "mtp.hidden_input_layernorm.weight": "mtp.hidden_input_norm_weight",
        "mtp.eagle_fc.weight": "mtp.eagle_fc_weight",
        "mtp.final_layernorm.weight": "mtp.final_norm_weight",
    }
    if ref_fqn in mapping:
        return mapping[ref_fqn]
    # Layer weights: layers.{i}.XXX.weight → layers.{i}.YYY_weight
    layer_map = {
        "attention_norm": "input_norm",
        "wqkv": "qkv",
        "wo": "attention_proj",
        "ffn_norm": "pre_mlp_norm",
        "wfc1": "mlp_fc1",
        "w2": "mlp_fc2",
    }
    parts = ref_fqn.rsplit(".", 2)
    if len(parts) == 3:
        prefix, module, _ = parts
        if module in layer_map:
            return f"{prefix}.{layer_map[module]}_weight"
    return ref_fqn  # fallback


def load_weights_from_checkpoint(
    checkpoint_root: str,
    device: torch.device,
    world_size: int,
    rank: int,
    init_std: float = 0.02,
    seed: int = 1234,
    init_ones: bool = True,
    mtp_num_layers: int = 1,
    mup_base_hidden_size: int = 256,
    mup_emb_scale: float = 12.0,
    mup_depth_scale: float = 1.4,
    resume_from: str | None = None,
    init_weights_only: bool = False,
) -> ModelParameters:
    """Load or initialize model weights.

    In alignment mode (no checkpoint exists), this function initializes
    weights using the same muP scheme as the ref's ``init_weights``.

    Returns a :class:`ModelParameters` container with all weight buffers
    on the given device in bf16.
    """
    # Weights are loaded from the canonical checkpoint.  The canonical
    # state is a flat ``canonical_state_fp32.pt`` at ``checkpoint_root``.
    # When that file is absent (first run in a fresh workspace), we
    # initialise the weights deterministically from the seed instead.
    import os
    from pathlib import Path

    ckpt_path = Path(checkpoint_root) / "canonical_state_fp32.pt"
    if ckpt_path.is_file():
        # Load and convert to bf16.  The canonical checkpoint uses the
        # ref's ``named_parameters()`` FQN keys; translate them to our
        # internal flat names.
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        flat_params = []
        flat_names = []
        for k in sorted(state.keys()):
            internal_name = _ref_fqn_to_internal(k)
            flat_params.append(state[k].to(device=device, dtype=torch.bfloat16))
            flat_names.append(internal_name)
    else:
        # Initialise weights using muP scheme (same as ref)
        # ref: matrix weights (qkv, fc1, eagle_fc) → N(0, init_std/sqrt(width_mult))
        #      embedding / output_layer / wo / w2 → N(0, init_std)
        #      1-D / norm weight → 1.0 if init_ones else 0.97
        _rng = torch.Generator().manual_seed(seed)
        width_mult = HIDDEN_SIZE / mup_base_hidden_size
        scaled_std = init_std / math.sqrt(width_mult)
        all_params = []
        all_names = []

        # Pre-create norm weight (1-D, ones or 0.97)
        norm_weight = torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
        if not init_ones:
            norm_weight.fill_(0.97)

        # tok_embeddings: std = init_std
        w = _randn([VOCAB_SIZE, HIDDEN_SIZE], _rng, device, std=init_std)
        all_params.append(w)
        all_names.append("tok_embeddings_weight")

        # Layers
        qkv_dim = NUM_HEADS * HEAD_DIM + 2 * NUM_KV_HEADS * HEAD_DIM
        for li in range(NUM_LAYERS):
            for suffix, std_val in [
                ("input_norm_weight", 0),      # norm → 1.0 or 0.97
                ("qkv_weight", scaled_std),    # matrix → scaled_std
                ("attention_proj_weight", init_std),  # wo → init_std
                ("pre_mlp_norm_weight", 0),    # norm → 1.0 or 0.97
                ("mlp_fc1_weight", scaled_std),  # matrix → scaled_std
                ("mlp_fc2_weight", init_std),  # w2 → init_std
            ]:
                if std_val == 0:
                    # 1-D / norm weight
                    w = norm_weight.clone()
                elif suffix == "qkv_weight":
                    w = _randn([qkv_dim, HIDDEN_SIZE], _rng, device, std=std_val)
                elif suffix == "attention_proj_weight":
                    w = _randn([HIDDEN_SIZE, NUM_HEADS * HEAD_DIM], _rng, device, std=std_val)
                elif suffix == "mlp_fc1_weight":
                    w = _randn([2 * FFN_HIDDEN_SIZE, HIDDEN_SIZE], _rng, device, std=std_val)
                elif suffix == "mlp_fc2_weight":
                    w = _randn([HIDDEN_SIZE, FFN_HIDDEN_SIZE], _rng, device, std=std_val)
                else:
                    w = _randn([HIDDEN_SIZE, HIDDEN_SIZE], _rng, device, std=std_val)
                all_params.append(w)
                all_names.append(f"layers.{li}.{suffix}")

        # final_norm_weight
        w = norm_weight.clone()
        all_params.append(w)
        all_names.append("final_norm_weight")

        # output_weight: std = init_std
        w = _randn([VOCAB_SIZE, HIDDEN_SIZE], _rng, device, std=init_std)
        all_params.append(w)
        all_names.append("output_weight")

        # MTP
        if mtp_num_layers > 0:
            # MTP norms — use struct field names as flat names
            for suffix in ["emb_input_norm_weight", "hidden_input_norm_weight"]:
                w = norm_weight.clone()
                all_params.append(w)
                all_names.append(f"mtp.{suffix}")

            # eagle_fc: matrix → scaled_std
            w = _randn([HIDDEN_SIZE, 2 * HIDDEN_SIZE], _rng, device, std=scaled_std)
            all_params.append(w)
            all_names.append("mtp.eagle_fc_weight")

            # MTP layer inner weights
            for suffix in ["input_norm_weight", "qkv_weight", "attention_proj_weight",
                           "pre_mlp_norm_weight", "mlp_fc1_weight", "mlp_fc2_weight"]:
                if suffix.endswith("norm_weight"):
                    w = norm_weight.clone()
                    all_params.append(w)
                    all_names.append(f"mtp.layer.{suffix}")
                else:
                    std_val = scaled_std if suffix in ("qkv_weight", "mlp_fc1_weight") else init_std
                    if suffix == "qkv_weight":
                        w = _randn([qkv_dim, HIDDEN_SIZE], _rng, device, std=std_val)
                    elif suffix == "attention_proj_weight":
                        w = _randn([HIDDEN_SIZE, NUM_HEADS * HEAD_DIM], _rng, device, std=std_val)
                    elif suffix == "mlp_fc1_weight":
                        w = _randn([2 * FFN_HIDDEN_SIZE, HIDDEN_SIZE], _rng, device, std=std_val)
                    elif suffix == "mlp_fc2_weight":
                        w = _randn([HIDDEN_SIZE, FFN_HIDDEN_SIZE], _rng, device, std=std_val)
                    else:
                        w = _randn([HIDDEN_SIZE, HIDDEN_SIZE], _rng, device, std=std_val)
                    all_params.append(w)
                    all_names.append(f"mtp.layer.{suffix}")

            # final_norm_weight
            w = norm_weight.clone()
            all_params.append(w)
            all_names.append("mtp.final_norm_weight")

        flat_params = all_params
        flat_names = all_names

    # Build ModelParameters from the flat list
    layers = []
    li = 0
    while li < NUM_LAYERS:
        in_norm = _find_in_flat(flat_params, flat_names, f"layers.{li}.input_norm_weight")
        qkv = _find_in_flat(flat_params, flat_names, f"layers.{li}.qkv_weight")
        attn_proj = _find_in_flat(flat_params, flat_names, f"layers.{li}.attention_proj_weight")
        mlp_norm = _find_in_flat(flat_params, flat_names, f"layers.{li}.pre_mlp_norm_weight")
        mlp_fc1 = _find_in_flat(flat_params, flat_names, f"layers.{li}.mlp_fc1_weight")
        mlp_fc2 = _find_in_flat(flat_params, flat_names, f"layers.{li}.mlp_fc2_weight")
        if all(v is not None for v in [in_norm, qkv, attn_proj, mlp_norm, mlp_fc1, mlp_fc2]):
            layers.append(LayerParameterView(
                index=li,
                input_norm_weight=in_norm,
                qkv_weight=qkv,
                attention_proj_weight=attn_proj,
                pre_mlp_norm_weight=mlp_norm,
                mlp_fc1_weight=mlp_fc1,
                mlp_fc2_weight=mlp_fc2,
            ))
        li += 1

    tok_emb = _find_in_flat(flat_params, flat_names, "tok_embeddings_weight")
    assert tok_emb is not None, "tok_embeddings_weight not found"
    final_norm = _find_in_flat(flat_params, flat_names, "final_norm_weight")
    assert final_norm is not None
    output_w = _find_in_flat(flat_params, flat_names, "output_weight")
    assert output_w is not None

    mtp = None
    if mtp_num_layers > 0:
        mtp = MTPParameterView(
            emb_input_norm_weight=_find_in_flat(flat_params, flat_names, "mtp.emb_input_norm_weight"),
            hidden_input_norm_weight=_find_in_flat(flat_params, flat_names, "mtp.hidden_input_norm_weight"),
            eagle_fc_weight=_find_in_flat(flat_params, flat_names, "mtp.eagle_fc_weight"),
            layer=LayerParameterView(
                index=0,
                input_norm_weight=_find_in_flat(flat_params, flat_names, "mtp.layer.input_norm_weight"),
                qkv_weight=_find_in_flat(flat_params, flat_names, "mtp.layer.qkv_weight"),
                attention_proj_weight=_find_in_flat(flat_params, flat_names, "mtp.layer.attention_proj_weight"),
                pre_mlp_norm_weight=_find_in_flat(flat_params, flat_names, "mtp.layer.pre_mlp_norm_weight"),
                mlp_fc1_weight=_find_in_flat(flat_params, flat_names, "mtp.layer.mlp_fc1_weight"),
                mlp_fc2_weight=_find_in_flat(flat_params, flat_names, "mtp.layer.mlp_fc2_weight"),
            ),
            final_norm_weight=_find_in_flat(flat_params, flat_names, "mtp.final_norm_weight"),
        )

    return ModelParameters(
        tok_embeddings_weight=tok_emb,
        layers=layers,
        final_norm_weight=final_norm,
        output_weight=output_w,
        mtp=mtp,
    )


def _randn(shape: list[int], rng: torch.Generator, device: torch.device,
           std: float = 1.0) -> torch.Tensor:
    """Create a random fp32 tensor, multiply by std in fp32, then convert to bf16.

    Matches the ref's ``torch.randn(shape, generator=rng, dtype=torch.float32).mul_(std).to(bf16)``.
    """
    return torch.randn(shape, generator=rng, dtype=torch.float32, device=device).mul_(std).to(dtype=torch.bfloat16)