"""Resume gate (``resume-gate-20``) — thin wrapper over the framework train CLI.

Validates checkpoint save / resume correctness by running THREE
back-to-back
:func:`training_engine_tensor.train_loop.run_training_loop` invocations
from the same initial state and asserting that the uninterrupted
reference trajectory matches the save → load resumed trajectory
bit-for-bit over the gate window:

  Phase R (Reference) — uninterrupted ``num_steps`` from initial
        checkpoint.  Emits ``[LOSS_REF] step=N …`` lines.

  Phase A (Pre-save)  — ``save_step`` steps from the same initial
        checkpoint, persisting the full state into a temp directory.
        Emits ``[LOSS_RES] step=N …`` lines for ``N ∈ [0, save_step)``.

  Phase B (Resume)    — load the Phase-A checkpoint, continue training
        ``num_steps - save_step`` more steps.  Emits ``[LOSS_RES]
        step=N …`` lines for ``N ∈ [save_step, num_steps)``.

The dispatcher parses both ``LOSS_REF`` and ``LOSS_RES`` line streams
and gates on per-step ``max_abs_diff(global_loss) == 0`` and
``max_abs_diff(grad_norm) == 0`` over the configured window (default
``[10, 20)`` — i.e. exactly Phase B's range).

Why no training body here?
~~~~~~~~~~~~~~~~~~~~~~~~~~

The dataloader, model, optimizer, and checkpoint codec all live
inside :mod:`training_engine_tensor` (the self-developed engine).
Re-implementing any of them in this gate would silently fork the
SSOT — agents could change the framework while the gate kept running
against a stale gate-local copy.  Likewise, the Megatron-side baseline
trajectory comes from the L0 ref script (``ref/reference/${ref_script}``)
via the dispatcher; this script is exclusively the ours-side runner.

Required env (dispatcher-injected via ``config/eval.toml``):
  RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT
                       torchrun-injected
  CHECKPOINT_ROOT      canonical_state_fp32.pt directory
  BACKEND              "megatron" | "torch" — selects which L0 ref stack
                       the candidate engine pairs with
  MEGATRON_ROOT        Megatron source root (required only when
                       BACKEND == "megatron")
  DATA_PATH            Data source the candidate dataloader consumes —
                       Megatron ``--data-path`` prefix when BACKEND ==
                       "megatron"; modelbest_sdk shard string (derived
                       from DATA_CONF) when BACKEND == "torch"
  NUM_STEPS            total global training steps per phase (e.g. 20)
  RESUME_SAVE_STEP     step at which Phase A saves and Phase B resumes
  SEED                 base RNG seed
  MICRO_BATCH_SIZE     per-DP-rank micro batch size
  SEQ_LENGTH           sequence length
  GRAD_ACCUM_STEPS     gradient-accumulation micro-steps per global step
  RESUME_SCRATCH_DIR   dispatcher-injected scratch root for the Phase-A
                       checkpoint; lives under this run's per-suite
                       ``.artifacts/runs/<...>/resume_scratch/`` on
                       persistent gpfs (NOT /tmp). Required, no fallback.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import shutil
import sys
import traceback

from _gate_entry import gate_inputs
from _runner_utils import abort_if_gpu_dirty
from training_engine_tensor.train_loop import TrainLoopConfig, run_training_loop

_GATE = gate_inputs()


def _cleanup_resume_dir(resume_dir: str) -> None:
    """Best-effort removal of the Phase-A checkpoint dir after Phase B.

    Per rank-coordination: the directory is shared across all DP ranks
    (they all read it at Phase B start), so only rank 0 removes it.
    Other ranks no-op. ``ignore_errors=True`` so a partial-state Phase A
    failure doesn't shadow the original exception with an FS error.

    The scratch dir lives under ``<workspace>/tmp/`` (dispatcher-injected
    ``RESUME_SCRATCH_DIR`` = absolute ``<repo_root>/tmp/resume_scratch_<run>``),
    on the loop's mounted volume — NOT ``.artifacts`` and NOT ``/tmp``.
    Cleanup still matters: a stray ~14 GiB checkpoint per run adds up,
    but the failure mode of skipping it is bounded to mounted-volume
    bloat, not pod reclamation. The legacy ``/tmp/`` default — and the
    later ``.artifacts/runs/<...>/`` path — both sat on the docker
    overlay and reclaimed pods that hit the 50 GiB ephemeral-storage
    cap; moving the scratch under ``tmp/`` on the mounted volume is the
    fix.
    """
    if int(os.environ.get("RANK", "0")) == 0:
        shutil.rmtree(resume_dir, ignore_errors=True)


def _require(name: str) -> str:
    # Stage B: env transport (default) reads os.environ; products transport
    # reads gate-shape keys from the rendered ours product.
    return _GATE.require(name)


def _parse_hash_args() -> argparse.Namespace:
    """Parse dispatcher-injected hash-capture CLI args.

    Resume-gate convention: the dispatcher passes a single ``--hash-output
    <base>`` and the engine writes ``<base>.ref.json`` (Phase R
    un-resumed trajectory) and ``<base>.res.json`` (Phase A+B resumed
    trajectory) so the dispatcher's Phase-4 hash diff can ours-vs-ours
    compare. Suffix derivation lives in :func:`main` below — the
    dispatcher knows the suffix convention; the engine just writes
    where it's told.
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
    hash_base = (hash_args.hash_output or "").strip() or None
    hash_ref_dump = f"{hash_base}.ref.json" if hash_base else None
    hash_res_dump = f"{hash_base}.res.json" if hash_base else None

    num_steps = int(_require("NUM_STEPS"))
    save_step = int(_require("RESUME_SAVE_STEP"))
    if save_step <= 0 or save_step >= num_steps:
        raise ValueError(
            f"RESUME_SAVE_STEP={save_step} must lie strictly between 0 and "
            f"NUM_STEPS={num_steps}; otherwise Phase A or Phase B would have "
            "no steps to run."
        )

    backend = _require("BACKEND")
    base = TrainLoopConfig(
        num_steps=num_steps,
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
        persistent=hash_args.persistent,
    )

    # ── Phase R: uninterrupted reference trajectory ────────────────
    run_training_loop(
        dataclasses.replace(base, hash_output=hash_ref_dump),
        loss_tag="LOSS_REF",
    )

    # ── Phase A: train save_step steps, save full state ────────────
    # Every rank must agree on the same checkpoint directory: rank 0
    # writes during Phase A, every rank reads during Phase B.  Using
    # ``tempfile.mkdtemp`` here would mint a private path per rank, so
    # ranks 1..N-1 would crash in Phase B on a non-existent file.
    # ``os.getppid()`` is launch_dp.py's PID — identical across all
    # ranks spawned by the same launcher — making the path deterministic
    # and per-run unique.  ``save_checkpoint`` itself creates the dir
    # on rank 0; no pre-creation needed here.
    #
    # ``RESUME_SCRATCH_DIR`` is dispatcher-injected and points at the
    # ABSOLUTE ``<repo_root>/tmp/resume_scratch_<run>`` scratch dir
    # (repo_root = the per-loop workspace), so it does not depend on a
    # rank's cwd — NOT ``.artifacts`` and NOT ``/tmp``. On a devspace the container
    # tree (workspace + ``.artifacts`` included) sits on the docker
    # overlay and counts against the 50 GiB ephemeral-storage cap, so a
    # ~14 GiB checkpoint there evicts the pod; ``tmp/`` lands on the
    # loop's mounted volume. Required, no fallback: a missing value
    # means the caller bypassed the dispatcher.
    scratch_root = _require("RESUME_SCRATCH_DIR")
    resume_dir = f"{scratch_root}/ckpt_{os.getppid()}"
    try:
        run_training_loop(
            dataclasses.replace(
                base,
                num_steps=save_step,
                start_step=0,
                save_path=resume_dir,
                hash_output=hash_res_dump,
            ),
            loss_tag="LOSS_RES",
        )

        # ── Phase B: load Phase-A state, continue to num_steps ─────────
        run_training_loop(
            dataclasses.replace(
                base,
                num_steps=num_steps - save_step,
                start_step=save_step,
                resume_from=resume_dir,
                hash_output=hash_res_dump,
            ),
            loss_tag="LOSS_RES",
        )
    finally:
        # Cleanup regardless of success / failure path — see
        # _cleanup_resume_dir docstring for the k8s-quota rationale.
        _cleanup_resume_dir(resume_dir)


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
    # For the three-phase resume gate the engine returns normally from each
    # phase, so the runner observes Phase R / A / B in sequence and the
    # process exits cleanly once only after the final phase.
    sys.stdout.flush()
    sys.stderr.flush()
