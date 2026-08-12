"""bitwise-singlecard / bitwise-multicard / bitwise-perf ours-side runner — thin wrapper over the framework train CLI.

Used by every Stage 1 multi-step bitwise gate whose comparison wire
format is a stdout ``[LOSS]`` trajectory:

* bitwise-singlecard ``multistep-1gpu`` — single-card multi-step bitwise (DP=1);
* bitwise-multicard ``multistep`` — DP=2 multi-step bitwise;
* bitwise-perf ``perf-bitwise`` — DP=2 multi-step bitwise + MFU floor.

The alignment single-step tensor-capture path is owned by a sibling script
(``eval_capture_align.py``); ``evals/scripts/`` reads as a
per-milestone entry table — every milestone ``MX`` has exactly one
``eval_<...>.py`` here.

The Megatron-side baseline for every shape above is owned by the L0
ref script (``ref/reference/${ref_script}``, the L0 SSOT); this
script's only job is to lift the dispatcher-injected env into a
:class:`TrainLoopConfig` and call
:func:`training_engine_tensor.train_loop.run_training_loop`. The two
SSOTs (this engine entry + the ref script) feed Megatron's standard
``GPTDataset`` against the same ``DATA_PATH`` (preprocessed
to Megatron binary; see ``ref/reference/prepare_gsm8k_data.sh``);
identical seed + identical data path → identical per-step batches,
which is what makes the bitwise comparison meaningful from bitwise-singlecard
(single-card) through bitwise-perf (DP=2).

Why no in-process Megatron build?  The Megatron baseline trajectory
is owned by the L0 ref script — running a second Megatron init
inside this gate would fork that SSOT.  Why no in-process ours
loop?  The dataloader, model, optimizer, and checkpoint codec all
live inside :mod:`training_engine_tensor`; re-implementing them in
the gate would fork the ours-side SSOT in turn.  See
``prompt/develop_prompt/stage1.md`` §"Dataloader 说明".

Required env (dispatcher-injected via ``config/eval.toml``):
  RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT
                       torchrun-injected
  BACKEND              "megatron" | "torch" — selects which L0 ref stack
                       the candidate engine pairs with
  CHECKPOINT_ROOT      canonical_state_fp32.pt directory
  MEGATRON_ROOT        Megatron source root (required only when
                       BACKEND == "megatron")
  DATA_PATH            Data source the candidate dataloader consumes —
                       Megatron ``--data-path`` prefix when BACKEND ==
                       "megatron"; modelbest_sdk shard string (derived
                       from DATA_CONF) when BACKEND == "torch"
  NUM_STEPS            global training steps
  SEED                 base RNG seed
  MICRO_BATCH_SIZE     per-DP-rank micro batch size
  SEQ_LENGTH           sequence length
  GRAD_ACCUM_STEPS     microbatches per global step

Output protocol (see :func:`training_engine_tensor.train_loop.run_training_loop`):
  ``[LOSS] step=<N> global_loss=<float> grad_norm=<float>
           time_s=<float> mfu_e2e_standard=<float>``
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

from _gate_entry import gate_inputs
from _runner_utils import abort_if_gpu_dirty
from training_engine_tensor.train_loop import TrainLoopConfig, run_training_loop

_GATE = gate_inputs()


def _require(name: str) -> str:
    # Stage B: env transport (default) reads os.environ as before; products
    # transport reads gate-shape keys from the rendered ours product.
    return _GATE.require(name)


def _parse_hash_args() -> argparse.Namespace:
    """Parse dispatcher-injected hash-capture CLI args.

    Typed wire (per project plan SSOT — no env var on this hop):
        --hash-capture-level <int>   0 / 1 / 2 (see harness_hook docstring)
        --hash-output <abs path>     where the JSON hash dump lands
        --persistent                 set for bitwise-singlecard / bitwise-multicard / bitwise-perf / resume; unset for alignment
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--hash-capture-level", type=int, default=0)
    parser.add_argument("--hash-output", type=str, default="")
    parser.add_argument("--persistent", action="store_true")
    args, _remaining = parser.parse_known_args(sys.argv[1:])
    return args


def main() -> None:
    abort_if_gpu_dirty()
    hash_args = _parse_hash_args()
    backend = _require("BACKEND")
    config = TrainLoopConfig(
        num_steps=int(_require("NUM_STEPS")),
        micro_batch_size=int(_require("MICRO_BATCH_SIZE")),
        seq_length=int(_require("SEQ_LENGTH")),
        grad_accum_steps=int(_require("GRAD_ACCUM_STEPS")),
        seed=int(_require("SEED")),
        world_size=int(_require("WORLD_SIZE")),
        checkpoint_root=_GATE.checkpoint_root(),
        data_path=_require("DATA_PATH"),
        backend=backend,
        megatron_root=_require("MEGATRON_ROOT") if backend == "megatron" else "",
        hash_capture_level=hash_args.hash_capture_level,
        hash_output=hash_args.hash_output or None,
        persistent=hash_args.persistent,
        # Optimizer hyperparameters from the rendered product (config/optim.toml
        # + gate_config overrides, projected into the product's [cli] section).
        lr=float(_require("LR")),
        min_lr=float(_require("MIN_LR")),
        lr_warmup_iters=int(_require("LR_WARMUP_ITERS")),
        lr_decay_iters=int(_require("LR_DECAY_ITERS")),
        lr_wsd_decay_iters=int(_require("LR_WSD_DECAY_ITERS")),
    )
    run_training_loop(config, loss_tag="LOSS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        rank = os.environ.get("RANK", "?")
        print(f"[rank {rank}] FATAL: {exc}", flush=True)
        traceback.print_exc()
        sys.exit(1)
    # Clean exit is a gate requirement: returncode==0 AND valid artifact.
    # `run_training_loop` owns ordered teardown (drain CUDA, stop dataloader
    # threads, NCCL barrier + destroy_process_group, empty_cache) and returns
    # normally, leaving the interpreter with nothing to crash on. The runner
    # MUST NOT os._exit — a SIGABRT (returncode=-6) during shutdown is FAIL.
    sys.stdout.flush()
    sys.stderr.flush()
