"""Production long-train runner — ours-only, one save/resume segment.

Thin black-box launcher, structurally the ours-side half of
``eval_resume_train.py`` (same ``abort_if_gpu_dirty`` + ``_require`` +
``TrainLoopConfig`` + ordered-teardown skeleton) with the ref/bitwise
machinery stripped out. The dispatcher (``_run_production_train``) owns
the multi-segment orchestration: it walks the full run in
``_PRODUCTION_SAVE_SEGMENTS`` equal segments and invokes this script once
per segment, passing ``START_STEP`` / ``FORGE_SAVE_PATH`` (and
``FORGE_RESUME_FROM`` from the 2nd segment on). This script therefore
trains exactly one segment: ``NUM_STEPS`` steps from absolute
``START_STEP``, saving a full resume checkpoint to ``FORGE_SAVE_PATH``
after the final step.

There is NO ref comparison and NO loss/MFU gate here — this is a
checkpoint-only run (the engine still emits ``[LOSS]`` lines, but
the dispatcher does not parse them). Crash-resume is a dispatcher
concern: a restarted run discovers the last completed checkpoint and
sets ``START_STEP`` / ``FORGE_RESUME_FROM`` so this script continues
from there rather than from step 0.

Required env (dispatcher-injected via ``config/eval.toml``):
  RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT
                       launch_dp-injected; train_loop reads them
  CHECKPOINT_ROOT      canonical_state_fp32.pt directory
  BACKEND              "megatron" | "torch"
  MEGATRON_ROOT        Megatron source root (required only when
                       BACKEND == "megatron")
  DATA_PATH            data source the candidate dataloader consumes
  NUM_STEPS            steps to train in THIS segment (= total / segments)
  START_STEP           absolute step this segment begins at
  SEED                 base RNG seed
  MICRO_BATCH_SIZE     per-DP-rank micro batch size
  SEQ_LENGTH           sequence length
  GRAD_ACCUM_STEPS     gradient-accumulation micro-steps per global step
  FORGE_SAVE_PATH      directory to persist this segment's full state into
                       (engine writes AFTER the final step — see
                       train_loop.run_training_loop "Save hook")

Optional:
  FORGE_RESUME_FROM    previous segment's checkpoint dir; absent on the
                       first segment, so it is read with a default (NOT
                       _require) — the dispatcher writes it into the run-config
                       from the 2nd segment on, so it is resolved through the
                       same ``--config`` product-first path as every other key.

All of the above are read from the single ``--config <run_config>`` the
dispatcher composes per segment (frozen product ⊕ the per-segment
``NUM_STEPS`` / ``START_STEP`` / ``FORGE_SAVE_PATH`` / ``FORGE_RESUME_FROM``);
nothing but that one path and the launch_dp rendezvous identity crosses the
process boundary.
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
        start_step=int(_require("START_STEP")),
        save_path=_require("FORGE_SAVE_PATH"),
        # Optional: only the 2nd+ segment resumes. Read product-first with an
        # explicit default ("") so the first segment (no prior checkpoint, no
        # FORGE_RESUME_FROM in its run-config) does not fail.
        resume_from=_GATE.get("FORGE_RESUME_FROM", "") or None,
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
    # Clean exit is a gate requirement: returncode==0 AND valid checkpoint.
    # `run_training_loop` owns ordered teardown (drain CUDA, stop dataloader
    # threads, NCCL barrier + destroy_process_group, empty_cache) and returns
    # normally, leaving the interpreter with nothing to crash on. The runner
    # MUST NOT os._exit — a SIGABRT (returncode=-6) during shutdown is FAIL.
    sys.stdout.flush()
    sys.stderr.flush()
