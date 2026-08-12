"""Block 1 — model: structure + sizes from config, random init, muP width.

The structure *code* is NOT here. trust_remote_code makes transformers load the
official modeling_minicpm.py the config's auto_map points at — bundled offline in
hf_model/ next to this script (no Hub lookup). We only build from the config
object and override the few knobs that let an 8B fit one card.
"""
from __future__ import annotations

import torch


def install_transformers_compat_shim():
    """Let the 4.56-era MiniCPM4 remote modeling import under transformers 5.9.

    The remote modeling pulls two symbols newer transformers relocated/removed;
    re-providing them is enough. Must run BEFORE the remote module is imported
    (i.e. before from_config).
    """
    import transformers.utils.import_utils as iu
    import transformers.utils as U
    import transformers.pytorch_utils as pu

    if not hasattr(iu, "is_torch_fx_available"):
        iu.is_torch_fx_available = lambda: False
    if not hasattr(U, "is_torch_fx_available"):
        U.is_torch_fx_available = iu.is_torch_fx_available
    if not hasattr(pu, "is_torch_greater_or_equal_than_1_13"):
        pu.is_torch_greater_or_equal_than_1_13 = True


def build_model(model_id, num_layers, seq_length, attn_impl,
                gradient_checkpointing, seed, device):
    from transformers import AutoConfig, AutoModelForCausalLM

    install_transformers_compat_shim()

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    config.num_hidden_layers = num_layers          # fit one card
    config.rope_scaling = None                      # drop longrope → plain 4k RoPE
    config.max_position_embeddings = seq_length
    config.use_cache = False

    width_mult = getattr(config, "hidden_size", 4096) / getattr(config, "dim_model_base", 256)

    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(
        config, trust_remote_code=True, attn_implementation=attn_impl,
    )
    # bf16-master recipe: params are stored bf16; the optimizer keeps the fp32 master.
    model = model.to(device=device, dtype=torch.bfloat16)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    return model, config, width_mult
