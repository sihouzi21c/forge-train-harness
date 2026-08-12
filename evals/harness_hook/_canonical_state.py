"""FP32 master-weight harvest for canonical-state bootstrap dumps.

Reference implementation. The collector and its ``master_weight_fn``
contract are stable; ref stacks whose FP32 master does not live on
``param.main_param`` / ``param.data`` (FSDP shards, DeepSpeed
``BF16_Optimizer``, TP>1 layouts that need cross-rank assembly, …)
override the callable when calling
:func:`evals.harness_hook.install_canonical_state_dump`. See that
function's docstring and ``recipes/README.md`` for the bridge
pattern.

A train-time companion to :mod:`._grad_collector`. Instead of capturing
gradients between backward and step, this module captures the FP32
master copy of every trainable parameter *as the optimizer would see
it* at the moment of the first ``optimizer.step()`` — i.e. the
canonical "starting state" any downstream training stack should
bootstrap from to reproduce the reference trajectory.

The artifact this produces is the ``canonical_state_fp32.pt`` file that
several harness suites (``op-long``, ``eval_train_steps``,
``eval_resume_train``, ``eval_long_train``, ``eval_capture_align``)
already expect under ``$FORGE_CHECKPOINT_ROOT``. There is no
shipped script that generates it — the bridge author wires
:func:`install_canonical_state_dump` (in the package ``__init__``)
into the reference launcher and runs it once.

Key naming policy
-----------------
Tensors are keyed by ``named_parameters()`` FQN **verbatim** — no
``weight.`` / ``master.`` / ``param.`` prefix, no rename table. This
mirrors what a Megatron consumer would see if it ``torch.load``-ed its
own checkpoint and read ``state_dict["model"]``. Downstream ours-side
loaders may still apply their own ``ref_name → ours_name`` map on top
(see ``train_engine/src/training_engine_tensor/parameters.py``'s
``_PARAM_NAME_MAP``), but that's not this module's concern.

Master-weight source
--------------------
Megatron's ``DistributedOptimizer`` attaches the FP32 master copy to
each ``nn.Parameter`` as ``.main_param`` (a CPU/GPU FP32 tensor that
mirrors ``.data`` but lives outside the BF16 model weight). Plain
PyTorch trainers have no master copy at all — ``.data`` *is* the
trainable tensor — so ``param.data.float()`` is the correct fallback.

The default :func:`default_master_weight_fn` tries ``main_param``
first then falls back to ``data``; bridge authors override the
callable for other frameworks (FSDP sharded master, DeepSpeed
``BF16_Optimizer``, …).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["collect_canonical_state", "default_master_weight_fn"]


def default_master_weight_fn(name: str, param: Any) -> Any:
    """Return the FP32 master copy of ``param`` as a CPU FP32 tensor.

    Tries Megatron's ``param.main_param`` first (populated by
    ``DistributedOptimizer`` when ``--use-distributed-optimizer`` is
    on), then falls back to ``param.data`` upcast to FP32. Returns
    ``None`` only when neither is a tensor — callers skip such
    parameters.
    """
    import torch

    master = getattr(param, "main_param", None)
    if not isinstance(master, torch.Tensor):
        master = getattr(param, "data", None)
    if not isinstance(master, torch.Tensor):
        return None
    return master.detach().to(torch.float32).cpu()


def collect_canonical_state(
    model: Any,
    *,
    master_weight_fn: Callable[[str, Any], Any] | None = None,
) -> dict[str, Any]:
    """Walk ``model.named_parameters()`` and return ``{fqn: fp32 tensor}``.

    Uses the same ``.module``-unwrap policy as the rest of
    ``harness_hook`` — DDP / Float16Module / FSDP-style wrappers are
    chased until a non-``.module`` attribute name is reached, so FQNs
    stay stable across wrap layers.

    Parameters
    ----------
    model:
        ``nn.Module`` (or wrapped). We walk ``named_parameters()`` of
        the innermost module.
    master_weight_fn:
        ``(name, param) -> fp32 cpu tensor | None``. Defaults to
        :func:`default_master_weight_fn`. Override for non-Megatron
        frameworks whose FP32 master lives somewhere other than
        ``param.main_param`` (e.g. FSDP's ``_local_shard``, DeepSpeed's
        ``BF16_Optimizer.single_partition_of_fp32_groups``).
    """
    if master_weight_fn is None:
        master_weight_fn = default_master_weight_fn

    inner = model
    while hasattr(inner, "module"):
        inner = inner.module

    out: dict[str, Any] = {}
    for name, p in inner.named_parameters():
        t = master_weight_fn(name, p)
        if t is None:
            continue
        out[name] = t
    return out
