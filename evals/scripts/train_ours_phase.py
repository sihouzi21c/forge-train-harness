"""Ours-side SINGLE-phase entry for the WSD 3-phase bash driver.

Pure env-driven (NO argparse, NO GateInputs, NO harness suite coupling): the
bash driver ``train_ours_al.sh`` exports ONE phase's knobs, this reads them from
``os.environ`` into a :class:`TrainLoopConfig` and calls
:func:`run_training_loop` exactly once. Shared verbatim by the ``wsd-sft-70`` /
``production-resume-70`` gates (short 20/20/20 windows) and the single-job
production 3-phase run (the long recipe) — same contract, different step env.

Each phase is one ``torchrun`` of this env entry (the driver calls torchrun
directly, no bespoke DP spawner in between). ``RANK`` / ``LOCAL_RANK`` /
``WORLD_SIZE`` (global = nproc_per_node × nnodes) are injected by ``torchrun``,
NOT by this file; this reads ``WORLD_SIZE`` straight into ``config.world_size``.

Env contract (bash exports these per phase; see ``train_ours_al.sh``):
  NUM_STEPS START_STEP LR MIN_LR LR_WARMUP_ITERS LR_DECAY_ITERS
  LR_WSD_DECAY_ITERS RESUME_FROM INIT_WEIGHTS_ONLY SAVE_PATH DATA_PATH
    per-phase schedule + handoff (change every phase). RESUME_FROM is either a
    phase SAVE_PATH root (first-launch handoff) or a versioned ``step_<abs>/``
    dir (crash restart) — the engine globs ``*training_state.pt`` inside it.
  MICRO_BATCH_SIZE SEQ_LENGTH GRAD_ACCUM_STEPS SEED WORLD_SIZE
  CHECKPOINT_ROOT BACKEND MEGATRON_ROOT SAVE_INTERVAL
    shared geometry/runtime (constant across phases). SAVE_INTERVAL>0 makes the
    engine write versioned mid-phase ckpts (crash-resume; 0 = phase-end only).

NOTE — geometry (FORGE_NUM_LAYERS ...) and the optim/muP/MTP hyperparams are
read by the engine itself from the rendered ours product via
FORGE_GATE / FORGE_OURS_CONFIG_DIR (runtime_config.load()), NOT reconstructed
here — they are constant across all three phases, so they stay out of the
per-phase config. The gate path injects those pointers via suite_process_env;
the production path projects them via tools/ours_product_to_forge_env.py.
"""

from __future__ import annotations

import os
import sys
import traceback

from training_engine_tensor.train_loop import TrainLoopConfig, run_training_loop


def _req(name: str) -> str:
    val = os.environ.get(name)
    if val is None or val == "":
        raise KeyError(f"required env {name} is unset")
    return val


def _opt(name: str, default: str) -> str:
    val = os.environ.get(name)
    return default if (val is None or val == "") else val


def main() -> None:
    resume_from = _opt("RESUME_FROM", "")
    save_path = _opt("SAVE_PATH", "")

    cfg = TrainLoopConfig(
        num_steps=int(_req("NUM_STEPS")),
        start_step=int(_opt("START_STEP", "0")),
        micro_batch_size=int(_req("MICRO_BATCH_SIZE")),
        seq_length=int(_req("SEQ_LENGTH")),
        grad_accum_steps=int(_req("GRAD_ACCUM_STEPS")),
        seed=int(_req("SEED")),
        world_size=int(_req("WORLD_SIZE")),
        checkpoint_root=_req("CHECKPOINT_ROOT"),
        data_path=_req("DATA_PATH"),
        backend=_req("BACKEND"),
        megatron_root=_opt("MEGATRON_ROOT", ""),
        lr=float(_req("LR")),
        min_lr=float(_opt("MIN_LR", "0")),
        lr_warmup_iters=int(_opt("LR_WARMUP_ITERS", "0")),
        lr_decay_iters=int(_opt("LR_DECAY_ITERS", "0")),
        lr_wsd_decay_iters=int(_opt("LR_WSD_DECAY_ITERS", "0")),
        init_weights_only=_opt("INIT_WEIGHTS_ONLY", "0") == "1",
        resume_from=resume_from or None,
        save_path=save_path or None,
        # Explicit phase label so the engine's [PHASE] name= banner stays correct
        # under crash-restart (a resumed stable/sft phase has resume_from set and
        # would otherwise mis-infer as "decay", breaking phase segmentation).
        phase_name=_opt("PHASE_NAME", "") or None,
        # Periodic-save cadence in ABSOLUTE steps (0 = save only at phase end).
        # >0 → engine writes a versioned ckpt every interval to
        # <save_path>/step_<abs>/ AND one final ckpt at phase end. resume_from,
        # when set to such a step_<abs>/ dir, is loaded by globbing
        # *training_state.pt inside it. The gates inject 10 to force mid-phase
        # crash points; production sets the per-phase cadence.
        save_interval=int(_opt("SAVE_INTERVAL", "0")),
    )
    # no-load-data-state: the decay phase switches corpus and must NOT replay
    # the stable dataloader cursor (it still restores RNG + optimizer + step).
    # TrainLoopConfig deliberately has NO load_data_state field — the contract
    # is that the engine INFERS no-load when ``resume_from`` is set AND
    # ``data_path`` differs from the resumed run's corpus. So NO_LOAD_DATA_STATE
    # exported by train_ours_al.sh is informational only; correctness rides on
    # passing the decay/sft corpus in DATA_PATH (which the driver does).
    run_training_loop(cfg, loss_tag="LOSS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        rank = os.environ.get("RANK", "?")
        print(f"[rank {rank}] FATAL: {exc}", flush=True)
        traceback.print_exc()
        sys.exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
