"""SSOT runtime config for the self-developed training engine.

This is the **only** module in ``training_engine_tensor/`` that reads the
gate's hyperparameter values from outside the engine. It reads them from
the **rendered ours product** (``workload/src/config/<gate>.toml``),
located via the two pointers the harness injects into every gate
subprocess — ``FORGE_GATE`` and ``FORGE_OURS_CONFIG_DIR``. Every other
engine module must obtain its hyperparameters by calling :func:`load` and
threading the returned ``RuntimeHParams`` through its call graph.

The product is the SOLE source for the gate's ``[model]`` / ``[optim]``
scalars — same contract the gate runners use via
``evals/scripts/_gate_entry.py`` ("no env transport, the product is the
sole source"). The renderer (``tools/render_gate_configs.py``) projects
the ``[model]`` / ``[optim]`` axes into the product ``[cli]`` table as
bare lowercase keys, so the dataclass field names below ARE the product
keys.

Missing keys raise — there are **no defaults baked into the engine**. A
required key being missing means the render/materialize contract was not
honoured; silently falling back to a baked-in default is exactly the SSOT
violation this module exists to prevent.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

__all__ = ["ModelHParams", "OptimHParams", "RuntimeHParams", "data_loader", "load"]


@dataclass(frozen=True)
class OptimHParams:
    lr: float
    min_lr: float
    lr_warmup_iters: int
    lr_decay_iters: int
    lr_wsd_decay_iters: int
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    clip_grad: float


@dataclass(frozen=True)
class ModelHParams:
    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    num_query_groups: int
    head_dim: int
    seq_length: int
    max_position_embeddings: int
    padded_vocab_size: int
    rotary_base: int
    norm_epsilon: float
    init_method_std: float
    mup_base_hidden_size: int
    mup_emb_scale: float
    mup_depth_scale: float
    mtp_num_layers: int
    mtp_loss_weight: float


@dataclass(frozen=True)
class RuntimeHParams:
    optim: OptimHParams
    model: ModelHParams


class MissingRuntimeConfigError(RuntimeError):
    """Raised when the rendered ours product is absent or missing a key."""


# ``from __future__ import annotations`` makes ``field.type`` a string, so map
# the two scalar annotations the dataclasses use to their coercion callable.
_CASTS = {"int": int, "float": float}


def _product_cli() -> dict[str, object]:
    """Load the ``[cli]`` table of the rendered ours product for this gate.

    The product path is ``FORGE_OURS_CONFIG_DIR/<FORGE_GATE>.toml`` — both env
    vars are injected by ``evals._common.suite_process_env`` into every gate
    subprocess. This is a pointer-only use of the environment; the
    hyperparameter VALUES come from the product, never from ``FORGE_<KEY>`` env.
    """
    gate = os.environ.get("FORGE_GATE")
    cfg_dir = os.environ.get("FORGE_OURS_CONFIG_DIR")
    if not gate or not cfg_dir:
        raise MissingRuntimeConfigError(
            "runtime_config: FORGE_GATE / FORGE_OURS_CONFIG_DIR not set. The "
            "harness injects both into every gate subprocess (suite_process_env); "
            "the engine reads its hyperparameters from the rendered ours product "
            "they point at."
        )
    path = Path(cfg_dir) / f"{gate}.toml"
    if not path.is_file():
        raise MissingRuntimeConfigError(
            f"runtime_config: ours product not found: {path}. Render it with "
            "tools/render_gate_configs.py before launching the engine."
        )
    with open(path, "rb") as fh:
        doc = tomllib.load(fh)
    cli = doc.get("cli")
    if not isinstance(cli, dict):
        raise MissingRuntimeConfigError(f"runtime_config: {path} has no [cli] table.")
    return cli


def _coerce(cls, cli: dict[str, object]):
    """Build a frozen hparam dataclass from the product ``[cli]`` by field name.

    The field names ARE the product keys (the renderer's naming contract). A
    missing key raises rather than defaulting — the M2 R4 anti-pattern guard.
    """
    kwargs = {}
    for f in fields(cls):
        if f.name not in cli:
            raise MissingRuntimeConfigError(
                f"runtime_config: product [cli] missing {f.name!r} (required by "
                f"{cls.__name__}). The renderer must project every [model]/[optim] "
                "axis key into the gate product."
            )
        kwargs[f.name] = _CASTS[f.type](cli[f.name])
    return cls(**kwargs)


def load() -> RuntimeHParams:
    """Load runtime hyperparameters from the rendered ours product ``[cli]``."""
    cli = _product_cli()
    return RuntimeHParams(
        optim=_coerce(OptimHParams, cli),
        model=_coerce(ModelHParams, cli),
    )


def data_loader() -> str:
    """Read the dataloader kind from the data-value SSOT (config/data.toml).

    The kind lives in ``[data].data_loader`` of the file ``FORGE_DATA_TOML``
    points at — the per-loop config dir's data.toml, which the harness injects
    into every gate subprocess (``suite_process_env``). This is the SAME axis
    the ref launcher reads (``train_pure_mup_mtp`` ``--data-config``), so both
    sides select the loader from one source — no ``DATA_LOADER`` env transport,
    no per-gate ``[env]`` copy. The engine's ``dataloader.py`` calls this to
    dispatch its reader; a missing pointer/key raises (no baked-in default).
    """
    path = os.environ.get("FORGE_DATA_TOML")
    if not path:
        raise MissingRuntimeConfigError(
            "runtime_config: FORGE_DATA_TOML not set. The harness injects it "
            "into every gate subprocess (suite_process_env); the engine reads "
            "[data].data_loader from the data.toml it points at."
        )
    p = Path(path)
    if not p.is_file():
        raise MissingRuntimeConfigError(
            f"runtime_config: data.toml not found: {p} (FORGE_DATA_TOML)."
        )
    with open(p, "rb") as fh:
        data = tomllib.load(fh).get("data", {})
    kind = str(data.get("data_loader", "")).strip()
    if not kind:
        raise MissingRuntimeConfigError(
            f"runtime_config: {p} has no [data].data_loader. It is the SSOT for "
            "the loader kind (hf / modelbest / megatron_binary)."
        )
    return kind
