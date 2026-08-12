"""Compose-in extra muP / Eagle argparse flags missing on the on-remote
``pretrain_minicpm.py`` argparse stack.

Background
----------
The L0 ref script ``train_minicpm4_0.5b_fineweb_modelbestsdk.sh`` passes
five flags that the on-remote Megatron checkout
(``cpm_core_r0.15.0``) does not register on its argparse parser:

    --mup-base-hidden-size <int>
    --mup-emb-scale        <float>
    --mup-depth-scale      <float>
    --eagle-num-layers     <int>
    --eagle-ce-loss-weight <float>

The values themselves flow into the ``model_provider`` callback via the
JSON config the ref script writes (``ref/reference/train_minicpm4_0.5b_fineweb_modelbestsdk.sh``
lines 240-261), not via ``get_args()`` — argparse only needs to *accept*
the tokens so the entry point reaches ``pretrain(...)`` without an
"unrecognized arguments" exit code 2.

What this module does
---------------------
Monkey-patches ``megatron.training.arguments.parse_args`` so that any
caller-supplied ``extra_args_provider`` is composed with our additive
provider that registers the five flags as plain scalars.  Idempotent;
strict no-op when the Megatron argparse module is not importable.

When to call
------------
From the unified M1 bridge interposer (``ref/bridges/interposer.py``),
before the original Megatron entry reaches ``initialize_megatron``
(which calls ``parse_args``).  Calling this multiple times is safe;
the first call wins and the subsequent ones are no-ops.

Why a shim instead of patching the customer entry / megatron clone
------------------------------------------------------------------
Both alternatives mutate state outside this repo: the customer
``pretrain_minicpm.py`` lives on the cluster and is owned by the model
team, and the on-remote megatron clone is provisioned by the
``[remote]`` config (user-owned).  This shim is reversible, repo-local,
and disappears entirely the moment the on-remote checkout is moved to a
fork that registers the flags natively.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable

_INSTALLED = [False]


def _register_mup_eagle_flags(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    existing_dests = {action.dest for action in parser._actions}
    group = parser.add_argument_group(
        title="muP / Eagle (harness-bridge argparse shim)",
        description=(
            "Registered by evals.harness_hook._megatron_argparse_shim so "
            "the ref script can pass these flags through the customer "
            "argparse stack without an exit-code-2 failure.  Values flow "
            "to the model provider via the JSON config the ref script "
            "writes; the argparse targets here are unused by Megatron."
        ),
    )

    if "mup_base_hidden_size" not in existing_dests:
        group.add_argument(
            "--mup-base-hidden-size",
            type=int,
            default=None,
            help="(shim) base hidden size for muP scaling",
        )
    if "mup_emb_scale" not in existing_dests:
        group.add_argument(
            "--mup-emb-scale",
            type=float,
            default=None,
            help="(shim) muP embedding scale",
        )
    if "mup_depth_scale" not in existing_dests:
        group.add_argument(
            "--mup-depth-scale",
            type=float,
            default=None,
            help="(shim) muP depth scale",
        )
    if "eagle_num_layers" not in existing_dests:
        group.add_argument(
            "--eagle-num-layers",
            type=int,
            default=None,
            help="(shim) MTP / Eagle head transformer layers",
        )
    if "eagle_ce_loss_weight" not in existing_dests:
        group.add_argument(
            "--eagle-ce-loss-weight",
            type=float,
            default=None,
            help="(shim) MTP / Eagle head cross-entropy loss weight",
        )

    return parser


def install_megatron_argparse_shim() -> bool:
    """Monkey-patch every reachable ``parse_args`` binding additively.

    Returns ``True`` if the patch was installed in this call, ``False``
    if it was already installed or the Megatron arguments module is not
    importable.
    """
    if _INSTALLED[0]:
        return False

    try:
        from megatron.training import arguments as mta
    except ImportError as exc:
        sys.stderr.write(
            "[harness_hook] muP/Eagle argparse shim: cannot import "
            f"megatron.training.arguments ({exc!r}); skipping\n"
        )
        return False

    original_parse_args = mta.parse_args

    def _patched_parse_args(
        extra_args_provider: Callable | None = None,
        ignore_unknown_args: bool = False,
    ):
        def _composed(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
            if extra_args_provider is not None:
                returned = extra_args_provider(parser)
                parser = returned if returned is not None else parser
            return _register_mup_eagle_flags(parser)

        return original_parse_args(
            extra_args_provider=_composed,
            ignore_unknown_args=ignore_unknown_args,
        )

    mta.parse_args = _patched_parse_args

    patched_modules = ["megatron.training.arguments"]
    for module_name, module in list(sys.modules.items()):
        if module is None or module is mta:
            continue
        bound = getattr(module, "parse_args", None)
        if bound is original_parse_args:
            module.parse_args = _patched_parse_args
            patched_modules.append(module_name)

    _INSTALLED[0] = True
    sys.stderr.write(f"[harness_hook] installed muP/Eagle argparse shim on {patched_modules}\n")
    return True
