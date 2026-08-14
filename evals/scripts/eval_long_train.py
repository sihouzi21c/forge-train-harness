"""Long-horizon training gate runner — thin wrapper over the framework train CLI.

Used by the Stage 1 ``long-train`` hard gate and the auxiliary
``loss-gate-200`` suite (a loss-only subset of long-train without
the MFU gate).  The script is a black-box launcher: it reads the dispatcher-
injected env vars, builds a
:class:`training_engine_tensor.train_loop.TrainLoopConfig`, and
delegates the entire training loop (dataloader, model, optimizer,
checkpointing, MFU accounting, ``[LOSS]`` line emission) to
:func:`training_engine_tensor.train_loop.run_training_loop` — the SSOT
for the self-developed engine's training entry.

Why no training body here?
~~~~~~~~~~~~~~~~~~~~~~~~~~

Per the milestone contract, the self-developed engine owns the full
training stack from the multi-step bitwise gates onward (including the dataloader, see
``prompt/develop_prompt/stage1.md`` §"Dataloader 说明").  Building a
dataloader, model, or optimizer in this gate would silently fork that
SSOT — agents could change the framework while gates kept working
against a stale gate-local copy.  The matching Megatron-side SSOT is
the L0 ref script (``ref/reference/${ref_script}``), invoked by the
dispatcher via ``evals._common.run_via_ref_script``.

Required env (dispatcher-injected via ``config/eval.toml``):
  RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT
                       torchrun-injected; train_loop reads them
  CHECKPOINT_ROOT      canonical_state_fp32.pt directory
  BACKEND              "megatron" | "torch" — selects which L0 ref stack
                       the candidate engine pairs with
  MEGATRON_ROOT        Megatron source root (required only when BACKEND ==
                       "megatron"; train_loop reads it for
                       Megatron GPTDataset / mpu init)
  DATA_PATH            Data source the candidate dataloader consumes —
                       Megatron ``--data-path`` prefix when BACKEND ==
                       "megatron"; modelbest_sdk shard string (derived
                       from DATA_CONF) when BACKEND == "torch"
  NUM_STEPS            global training steps
  SEED                 base RNG seed
  MICRO_BATCH_SIZE     per-DP-rank micro batch size
  SEQ_LENGTH           sequence length
  GRAD_ACCUM_STEPS     gradient-accumulation micro-steps per global step
                       (always injected by the dispatcher; required —
                       ``test_runner_metrics_contract.py`` enforces this
                       on every long-train / loss-gate suite)

Output protocol (see :func:`training_engine_tensor.train_loop.run_training_loop`):
  ``[LOSS] step=<N> global_loss=<float> grad_norm=<float>
           time_s=<float> mfu_e2e_standard=<float>``
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
    # Enable the fused Triton RMSNorm backward kernel for long-horizon
    # performance (non-deterministic mode).  The bitwise gates (perf-bitwise,
    # multistep-1gpu, multistep) use a different script (eval_train_steps.py)
    # and do NOT set this flag, so they always use the PyTorch closed-form.
    os.environ.setdefault("ENABLE_TRITON_RMSNORM_BWD", "1")
    # Enable the fused Triton SwiGLU backward kernel for long-horizon
    # performance.  Also gated by deterministic=False, so bitwise gates
    # always use the PyTorch closed-form.
    os.environ.setdefault("ENABLE_TRITON_SWIGLU_BWD", "1")
    # Enable the fused Triton SwiGLU forward kernel for long-horizon
    # performance.  Also gated by deterministic=False, so bitwise gates
    # always use the PyTorch path.
    os.environ.setdefault("ENABLE_TRITON_SWIGLU_FWD", "1")
    # Enable the fused Triton RMSNorm forward kernel for long-horizon
    # performance.  Reads bf16 directly, computes in fp32, writes bf16
    # output — eliminates the internal dtype round-trip overhead.
    # Also gated by deterministic=False, so bitwise gates always use the
    # PyTorch F.rms_norm path.
    os.environ.setdefault("ENABLE_TRITON_RMSNORM_FWD", "1")
    # Enable the fused Triton cross-entropy backward kernel for long-horizon
    # performance.  Fuses the softmax forward + gradient backward into a
    # single kernel, eliminating the one_hot allocation and elementwise
    # operations.  Also gated by deterministic=False, so bitwise gates always
    # use the PyTorch path.
    os.environ.setdefault("ENABLE_TRITON_CE_BWD", "1")
    # Enable the fused Triton RoPE forward kernel for long-horizon
    # performance.  Fuses cos/sin computation, dtype conversion, and
    # the rotary operation into a single kernel, reading bf16 directly
    # and computing in fp32.  Also gated by deterministic=False, so
    # bitwise gates always use the PyTorch path.
    os.environ.setdefault("ENABLE_TRITON_ROPE_FWD", "1")
    # Enable the fused Triton RoPE backward kernel for long-horizon
    # performance.  Also gated by deterministic=False, so bitwise gates
    # always use the PyTorch path.
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
