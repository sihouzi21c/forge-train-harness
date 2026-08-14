"""Iteration profile snapshot — wraps a 12-step long-train under nsys.

The dispatcher's ``profile-snapshot`` runner is the only caller. It sets
``FORGE_NSYS_RANK0_OUTPUT`` so ``evals/scripts/launch_dp.py`` wraps rank
0 with ``nsys profile``, then post-processes the ``.nsys-rep`` via
``tools.profile_render.render`` to emit ``summary.md`` + ``profile.json``
under the caller-supplied label directory.

NOT a commit gate — the verdict here is "did the trajectory complete +
did nsys produce a report"; correctness/MFU continue to be validated by
``long-train`` (commit gate) and ``long-train-smoke`` (smoke gate).
Reading the env contract / training shape is the same as
``eval_long_train.py``; the only deltas are NUM_STEPS pinned small (12)
and the surrounding nsys wrap.
"""

from __future__ import annotations

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


def main() -> None:
    abort_if_gpu_dirty()
    micro_batch_size = int(_require("MICRO_BATCH_SIZE"))
    world_size = int(_require("WORLD_SIZE"))
    backend = _require("BACKEND")
    # Enable the fused Triton kernels for long-horizon profiling (same as
    # eval_long_train.py).  These are gated by deterministic=False, so
    # bitwise gates always use the PyTorch paths.
    os.environ.setdefault("ENABLE_TRITON_RMSNORM_BWD", "1")
    os.environ.setdefault("ENABLE_TRITON_SWIGLU_BWD", "1")
    os.environ.setdefault("ENABLE_TRITON_SWIGLU_FWD", "1")
    os.environ.setdefault("ENABLE_TRITON_RMSNORM_FWD", "1")
    os.environ.setdefault("ENABLE_TRITON_CE_BWD", "1")
    os.environ.setdefault("ENABLE_TRITON_ROPE_FWD", "1")
    os.environ.setdefault("ENABLE_TRITON_ROPE_BWD", "1")
    config = TrainLoopConfig(
        num_steps=int(_require("NUM_STEPS")),
        micro_batch_size=micro_batch_size,
        seq_length=int(_require("SEQ_LENGTH")),
        grad_accum_steps=int(_require("GRAD_ACCUM_STEPS")),
        seed=int(_require("SEED")),
        world_size=world_size,
        checkpoint_root=_GATE.checkpoint_root(),
        data_path=_require("DATA_PATH"),
        backend=backend,
        megatron_root=_require("MEGATRON_ROOT") if backend == "megatron" else "",
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
