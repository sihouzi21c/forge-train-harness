"""alignment ours-side runner — single-step forward+backward tensor capture.

Used by the only two Stage 1 gates whose comparison wire format is a
**tensor dump** rather than a stdout ``[LOSS]`` trajectory:

* alignment.forward ``forward-align`` — diffs the per-module forward activations
  (keys prefixed ``fwd.``) between the candidate engine and the
  Megatron reference;
* alignment.backward ``backward-align`` — diffs the per-parameter gradients
  (keys prefixed ``grad.``) on the same single step.

Architectural shape
-------------------
Symmetric with ``eval_train_steps.py`` (bitwise-singlecard–bitwise-perf multi-step trajectory
runner), but pinned to the alignment contract:

* exactly one global step (``num_steps=1``);
* exactly one micro-batch per step (``grad_accum_steps=1``);
* ``capture_output_file`` is required (the engine's single-step capture
  short-circuit; see ``TrainLoopConfig`` "alignment capture semantics"). The
  dispatcher computes this absolute path from
  ``[defaults].candidate_capture_basename`` in ``config/eval.toml``;
  this script never picks a basename itself.

Splitting alignment into its own entry — even though the body is a small
restriction of ``eval_train_steps.py`` — keeps the ``evals/scripts/``
directory readable as a per-milestone table of contents: every
milestone ``MX`` has exactly one ``eval_<...>.py`` script that the
dispatcher shells out to.

The reference side of the alignment capture is owned by
``[defaults].ref_capture_script`` from ``config/eval.toml`` —
an agent-generated bridge that interposes
:func:`evals.harness_hook.install` into whatever customer training
stack is in play (see ``evals/harness_hook/recipes/README.md``).
This script's only job is to lift the dispatcher-injected env into a
:class:`TrainLoopConfig` and call
:func:`training_engine_tensor.train_loop.run_training_loop`. Both
sides feed Megatron's standard ``GPTDataset`` against the same
``DATA_PATH`` (HF gsm8k preprocessed to Megatron binary; see
``ref/reference/prepare_gsm8k_data.sh``); identical seed + identical
data path → identical single-step batch, which is what makes the
tensor-by-name bitwise comparison meaningful.

Required env (dispatcher-injected via ``config/eval.toml``):
  RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT
                                torchrun-injected
  BACKEND                       "megatron" | "torch" — selects which L0
                                ref stack the candidate engine pairs with
  CHECKPOINT_ROOT               canonical_state_fp32.pt directory
  MEGATRON_ROOT                 Megatron source root (required only when
                                BACKEND == "megatron"; train_loop reads it
                                for ``initialize_megatron`` + mpu setup)
  DATA_PATH                     Data source string the candidate
                                dataloader consumes — Megatron
                                ``--data-path`` prefix when BACKEND ==
                                "megatron"; modelbest_sdk weighted-shard
                                string (resolved from DATA_CONF) when
                                BACKEND == "torch"
  SEED                          base RNG seed (must match ref-script seed)
  MICRO_BATCH_SIZE              per-DP-rank micro batch size
  SEQ_LENGTH                    sequence length
  FORGE_CAPTURE_OUTPUT_FILE     absolute path the engine writes its
                                tensor dump to (+ optional sibling
                                ``<path>.graph.json``); chosen by the
                                dispatcher from
                                ``[defaults].candidate_capture_basename``

Output protocol:
  None on stdout — the dispatcher consumes the on-disk capture file
  only. The engine MUST NOT emit ``[LOSS]`` lines in capture mode.
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
    # Stage B: env transport (default) reads os.environ; products transport
    # reads gate-shape keys from the rendered ours product.
    return _GATE.require(name)


def _parse_hash_args() -> argparse.Namespace:
    """Parse dispatcher-injected hash-capture CLI args (alignment single-step path).

    alignment always runs ``persistent=False`` — the engine dumps on first
    optimizer.step and exits. The level / output knobs still flow
    through the same typed CLI wire as bitwise-singlecard–resume so the dispatcher's
    ``[evals.<suite>].hash_capture_level`` TOML SSOT reaches install().
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--hash-capture-level", type=int, default=2)
    parser.add_argument("--hash-output", type=str, default="")
    parser.add_argument("--persistent", action="store_true")
    args, _remaining = parser.parse_known_args(sys.argv[1:])
    return args


def main() -> None:
    abort_if_gpu_dirty()
    hash_args = _parse_hash_args()
    backend = _require("BACKEND")
    config = TrainLoopConfig(
        num_steps=1,
        micro_batch_size=int(_require("MICRO_BATCH_SIZE")),
        seq_length=int(_require("SEQ_LENGTH")),
        grad_accum_steps=1,
        seed=int(_require("SEED")),
        world_size=int(_require("WORLD_SIZE")),
        checkpoint_root=_GATE.checkpoint_root(),
        data_path=_require("DATA_PATH"),
        backend=backend,
        megatron_root=_require("MEGATRON_ROOT") if backend == "megatron" else "",
        capture_output_file=_require("FORGE_CAPTURE_OUTPUT_FILE"),
        hash_capture_level=hash_args.hash_capture_level,
        hash_output=hash_args.hash_output or None,
        persistent=False,  # alignment is single-step by definition
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
