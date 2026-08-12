"""DP=2 1000-step ours-side runner for the Stage-2 op-long gate.

Under the ref-as-gate SSOT model (`README.md`), this script holds no
baseline of its own.  It is a thin wrapper over
:func:`training_engine_tensor.train_loop.run_training_loop` — the same
SSOT entry that the Stage 1 ``long-train`` gate, the
``resume-gate-20`` gate, and the auxiliary ``loss-gate-200``
suite use.
The candidate fused operators are activated upstream via the standard
``FUSE_*`` env switches that the engine reads at start-up; this
script does not need any operator-aware logic of its own.

The dispatcher (``evals.dispatcher.run_op_long``) shell-execs the L0
ref script in a sibling subprocess to obtain the baseline trajectory
and applies the rel-diff gate against the trajectories emitted from
both sides.

Required env (dispatcher-injected via ``config/eval.toml``):
  OP_NAMES              — comma-separated op list being tested.  The
                          dispatcher also exports the per-op variant
                          switch as ``OP_<NAME>=<latest_available>`` for
                          every op in this list (see
                          ``evals.dispatcher_stage2._run_op_long``); the engine
                          reads those env vars to pick the optimised path.
  NUM_STEPS             — global training steps
  WORLD_SIZE            — data-parallel world size
  GLOBAL_BATCH_SIZE     — global batch size (used to derive
                          GRAD_ACCUM_STEPS when not declared explicitly)
  MICRO_BATCH_SIZE      — per-rank micro batch size
  SEED                  — RNG seed (no fallback — the dispatcher must
                          inject the same seed used to capture the
                          ref trajectory; baked-in defaults would
                          desync ours-vs-ref bitwise gates).
  SEQ_LENGTH            — sequence length (no fallback — same desync
                          rationale as SEED above; kept matching the
                          ref-script ``--seq-length`` value).
  CHECKPOINT_ROOT       — canonical_state_fp32.pt directory
  BACKEND               — "megatron" | "torch" — selects which L0 ref
                          stack the candidate engine pairs with
  DATA_PATH             — Data source the candidate dataloader consumes —
                          Megatron --data-path prefix when BACKEND ==
                          "megatron"; modelbest_sdk shard string (derived
                          from DATA_CONF) when BACKEND == "torch"
  MEGATRON_ROOT         — Megatron source root (required only when
                          BACKEND == "megatron")
"""

from __future__ import annotations

import os
import sys
import traceback

from _runner_utils import abort_if_gpu_dirty
from training_engine_tensor.train_loop import TrainLoopConfig, run_training_loop


def _require_env(name: str) -> str:
    """Return Stage-2 runtime env var ``name`` or raise immediately.

    D5 exemption: op-long is explicitly OUT of the gate-product collapse — it
    has no rendered product (the renderer covers only Stage-1 gates), so it
    keeps reading its flat run shape straight from ``os.environ`` (injected by
    the dispatcher's ``suite_process_env``). See
    ``gate_param_collapse_design.md`` §D5(b).
    """
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set (required by op-long Stage-2 runner).")
    return value


def _grad_accum_steps(*, micro_batch_size: int, world_size: int) -> int:
    raw = os.environ.get("GRAD_ACCUM_STEPS")
    if raw:
        return int(raw)
    global_batch_size = int(_require_env("GLOBAL_BATCH_SIZE"))
    return global_batch_size // (micro_batch_size * world_size)


def main() -> None:
    abort_if_gpu_dirty()
    rank = int(os.environ.get("RANK", "0"))
    op_names = _require_env("OP_NAMES")
    world_size = int(_require_env("WORLD_SIZE"))
    micro_batch_size = int(_require_env("MICRO_BATCH_SIZE"))
    num_steps = int(_require_env("NUM_STEPS"))

    if rank == 0:
        print(
            f"=== op-long ours runner: [{op_names}], DP={world_size}, "
            f"MBS={micro_batch_size}, {num_steps} steps ===",
            flush=True,
        )

    backend = _require_env("BACKEND")
    config = TrainLoopConfig(
        num_steps=num_steps,
        micro_batch_size=micro_batch_size,
        seq_length=int(_require_env("SEQ_LENGTH")),
        grad_accum_steps=_grad_accum_steps(
            micro_batch_size=micro_batch_size,
            world_size=world_size,
        ),
        seed=int(_require_env("SEED")),
        world_size=world_size,
        checkpoint_root=_require_env("CHECKPOINT_ROOT"),
        data_path=_require_env("DATA_PATH"),
        backend=backend,
        megatron_root=_require_env("MEGATRON_ROOT") if backend == "megatron" else "",
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
