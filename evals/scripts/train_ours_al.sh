#!/bin/bash
# Ours-side WSD 3-phase driver (stable -> decay -> sft) for the SELF-DEVELOPED
# engine (training_engine_tensor.run_training_loop): THREE sequential torchrun
# launches of a single-phase entry + directory-level checkpoint handoff. This is
# NOT single-process in-loop phase switching. This bash fills EVERY knob (its
# defaults are the production recipe from config/train/wsdsft_05b_prod.toml)
# and invokes train_ours_phase.py -> run_training_loop directly.
#
# Gate-scale manual run (2-card, MBS 2, GBS 32, 20/20/20, warmup 2/0/3).
# WORLD_SIZE is NODE count (torchrun semantics), so a single-node 2-GPU run
# sets GPUS_PER_NODE=2:
#   GPUS_PER_NODE=2 MICRO_BATCH_SIZE=2 GLOBAL_BATCH_SIZE=32 \
#   STABLE_ITERS=20 STABLE_WARMUP=2 DECAY_ITERS=20 DECAY_WARMUP=0 \
#   SFT_ITERS=20 SFT_WARMUP=3 \
#   CHECKPOINT_ROOT=<shared init dir> SAVE_ROOT=<scratch> \
#   bash evals/scripts/train_ours_al.sh
#
# Shared by THREE consumers, same script / different step env:
#   * wsd-sft-70 gate           : short 20/20/20, single-node 2-card, minutes.
#   * production-resume-70 gate : same, wrapped by train_ours_al_resume.sh
#                                 (crash injection); also the clean baseline.
#   * production                : the downscaled production recipe (defaults
#                                 below), launched by production_train.sh.
#
# Env driven (NO argparse): the top block defines the recipe as env; run_phase
# exports one phase's knobs and launches train_ours_phase.py, which reads them
# from os.environ into a TrainLoopConfig.
#
# Launcher is bash -> torchrun -> single-phase py entry so single- AND
# multi-node both work via torchrun rendezvous. launch_dp.py stays for OTHER
# suites — only THIS driver uses torchrun directly.
#
# Crash-restart is CHECKPOINT-driven: each phase inspects ITS OWN save dir for
# versioned step_<N>/ ckpts (latest_ckpt_step) and first-launches, resumes the
# remaining span, or skips accordingly. Re-running this script against the same
# SAVE_ROOT therefore continues where the previous run stopped.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HARNESS_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)   # .../harness (holds workload/, ref/)
OURS_ENTRY="$SCRIPT_DIR/train_ours_phase.py"
REF_REF_DIR="$HARNESS_ROOT/ref/reference"       # sibling data-conf files live here
[[ -f "$OURS_ENTRY" ]] || { echo "ERROR: missing $OURS_ENTRY" >&2; exit 1; }

set -euo pipefail

# ── PYTHONPATH: engine package + this dir (for entry resolution). ──
export PYTHONPATH="$HARNESS_ROOT/workload/src:$SCRIPT_DIR:${PYTHONPATH:-}"

# ── Distributed topology — torchrun does the single- AND multi-node
#    rendezvous and sets RANK/LOCAL_RANK/WORLD_SIZE(global) for each child, so
#    train_ours_phase reads config.world_size straight off torchrun's
#    WORLD_SIZE — this bash does NOT export it. A PyTorchJob operator injects
#    WORLD_SIZE=node count, RANK=node rank, MASTER_ADDR/PORT; a plain
#    single-node run leaves them unset and the defaults apply.
#    DP world = NNODES * GPUS_PER_NODE. ──
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
NNODES="${WORLD_SIZE:-1}"
NODE_RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-23456}"
DP_WORLD_SIZE=$(( NNODES * GPUS_PER_NODE ))

# ── Determinism env (CUDA line; mirrors the ref's enable_determinism stack —
#    the env half that must be set BEFORE CUDA init). The engine applies the
#    torch-API half itself when the rendered product says deterministic. ──
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

# ── Engine FORGE_* shape/optim/muP/MTP pointers — NOT set here. The ours
#    engine reads its hyperparameters from the rendered product via
#    FORGE_GATE / FORGE_OURS_CONFIG_DIR (runtime_config.load()):
#      * GATE path (dispatcher → this script): injected by suite_process_env
#        into the subprocess env.
#      * PRODUCTION path (production_train.sh → this script): exported by the
#        wrapper before this script runs (FORGE_GATE=production-train +
#        FORGE_OURS_CONFIG_DIR=<workload/src/config>).
#    FORGE_DATA_TOML is likewise a pointer the caller provides. ──

# ── Shared training shape (constant across phases). ──
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-10}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1280}"
export SEED="${SEED:-1234}"
export SEQ_LENGTH="${SEQ_LENGTH:-4096}"
export BACKEND="${BACKEND:-torch}"
export MEGATRON_ROOT="${MEGATRON_ROOT:-}"
# ── Periodic-save interval (engine contract). When SAVE_INTERVAL>0 the engine
#    writes a versioned ckpt every SAVE_INTERVAL ABSOLUTE steps to
#    <save_path>/step_<abs>/ AND one final ckpt (also under step_<phase_end>/)
#    at the phase end; the has_ckpt branches below resume from the latest such
#    step_<abs>/ dir (they glob *training_state.pt inside it). 0 = single
#    final ckpt at the save root (legacy). The gates inject 10 to force
#    mid-phase crash points; production sets a per-phase cadence. ──
export SAVE_INTERVAL="${SAVE_INTERVAL:-0}"
# Per-phase SAVE_INTERVAL override (production wants a different cadence per
# phase); each defaults to the global SAVE_INTERVAL so the gates (single value)
# and the legacy single-ckpt mode (0) are unchanged. The >0-ness is UNIFORM
# across phases in every real config, so handoff_dir's versioned-vs-plain
# decision is identical regardless of which phase's value is live.
STABLE_SAVE_INTERVAL="${STABLE_SAVE_INTERVAL:-$SAVE_INTERVAL}"
DECAY_SAVE_INTERVAL="${DECAY_SAVE_INTERVAL:-$SAVE_INTERVAL}"
SFT_SAVE_INTERVAL="${SFT_SAVE_INTERVAL:-$SAVE_INTERVAL}"
# Canonical init state dir (gate: dispatcher redirects CHECKPOINT_ROOT via
# forge_init_ones; production: the real from-scratch init dir).
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:?CHECKPOINT_ROOT must be set (ours stable init dir)}"

if (( GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * DP_WORLD_SIZE) != 0 )); then
    echo "ERROR: GBS=$GLOBAL_BATCH_SIZE not divisible by MBS=$MICRO_BATCH_SIZE * DP=$DP_WORLD_SIZE" >&2
    exit 1
fi
export GRAD_ACCUM_STEPS=$(( GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * DP_WORLD_SIZE) ))

# ── Phase recipe. Defaults = the DOWNSCALED production recipe (single source:
#    config/train/wsdsft_05b_prod.toml — values here must match it; the gates
#    inject their own 20/20/20 gate-scale window over these). The full-scale
#    reference recipe (stable 100000 / decay 19000 / sft 9500 @ 256 cards) is
#    recorded in that toml's header; this line's per-person card cap (2-4)
#    makes the downscaled recipe the production contract. ──
STABLE_ITERS="${STABLE_ITERS:-1800}"
STABLE_WARMUP="${STABLE_WARMUP:-40}"
STABLE_LR="${STABLE_LR:-1e-2}"
STABLE_MIN_LR="${STABLE_MIN_LR:-0}"

DECAY_ITERS="${DECAY_ITERS:-360}"
DECAY_WARMUP="${DECAY_WARMUP:-0}"
DECAY_LR="${DECAY_LR:-1e-2}"
DECAY_MIN_LR="${DECAY_MIN_LR:-5e-4}"

SFT_ITERS="${SFT_ITERS:-180}"
SFT_WARMUP="${SFT_WARMUP:-6}"
SFT_LR="${SFT_LR:-5e-4}"
SFT_MIN_LR="${SFT_MIN_LR:-1e-6}"
SFT_WSD_DECAY_ITERS="${SFT_WSD_DECAY_ITERS:-$(( SFT_ITERS - SFT_WARMUP ))}"

DECAY_TOTAL_ITERS=$(( STABLE_ITERS + DECAY_ITERS ))   # decay's absolute lr_decay_iters

# ── Per-phase data confs (each exports DATA_PATH; three DISTINCT corpora —
#    the wsd-sft gate's structural verdict asserts the swap at both handoffs). ──
STABLE_DATA_CONF="${STABLE_DATA_CONF:-$REF_REF_DIR/wsdsft_stable_data_conf.sh}"
DECAY_DATA_CONF="${DECAY_DATA_CONF:-$REF_REF_DIR/wsdsft_decay_data_conf.sh}"
SFT_DATA_CONF="${SFT_DATA_CONF:-$REF_REF_DIR/wsdsft_sft_data_conf.sh}"

# ── Checkpoint handoff layout. SAVE_ROOT is REQUIRED (no shared-literal
#    default — a shared path collides across loops/agents; production_train.sh
#    composes a per-loop persistent path, gates pass a scratch dir). ──
SAVE_ROOT="${SAVE_ROOT:?SAVE_ROOT must be set (per-loop persistent path in production; scratch in gates)}"
STABLE_SAVE="${STABLE_SAVE:-$SAVE_ROOT/stable}"
DECAY_SAVE="${DECAY_SAVE:-$SAVE_ROOT/decay}"
SFT_SAVE="${SFT_SAVE:-$SAVE_ROOT/sft}"   # sft is the deliverable
mkdir -p "$SAVE_ROOT"

DRY_RUN="${DRY_RUN:-0}"

# Multi-node handoff barrier: on >1 node the non-rank0 nodes must block until
# rank0 flushes the phase ckpt to the shared FS before their next-phase
# torchrun tries to load resume_from. Single node (the gates, the capped
# production job) skips this — torchrun already blocked until rank0 exited.
# Engine-format-agnostic on purpose: wait on ANY regular file under the dir.
wait_for_ckpt() {
    local dir="$1" waited=0
    until [[ -n "$(find "$dir" -type f -print -quit 2>/dev/null)" ]]; do
        sleep 5; waited=$(( waited + 5 ))
        (( waited % 300 == 0 )) && echo "[WAIT] node$NODE_RANK waiting for ckpt in $dir (${waited}s)" >&2
        (( waited > 7200 )) && { echo "ERROR: timed out waiting for ckpt in $dir" >&2; exit 1; }
    done
}

# latest_ckpt_step <phase_save_dir> — echo the max ABSOLUTE step N over versioned
# ckpt dirs <phase_save_dir>/step_<N>/ that actually contain a *training_state.pt
# (guards against a half-written dir). Empty output when none exist. Drives the
# has_ckpt first-launch/restart split below: on a clean start there is no step_*/
# so the phase first-launches; after a crash the phase restarts from this max N.
latest_ckpt_step() {
    local dir="$1" d n best=""
    [[ -d "$dir" ]] || return 0
    for d in "$dir"/step_*/; do
        [[ -d "$d" ]] || continue
        compgen -G "${d}*training_state.pt" >/dev/null || continue
        n="${d%/}"; n="${n##*/step_}"
        [[ "$n" =~ ^[0-9]+$ ]] || continue
        n=$(( 10#$n ))
        if [[ -z "$best" || "$n" -gt "$best" ]]; then best="$n"; fi
    done
    [[ -n "$best" ]] && echo "$best"
    return 0
}

# handoff_dir <phase_save> <phase_end_abs> — the dir the NEXT phase's first
# launch resumes from. SAVE_INTERVAL>0 writes the phase-final ckpt versioned
# under step_<end>/; SAVE_INTERVAL=0 writes a single training_state.pt at the
# save root. Keeps the cross-phase handoff correct in BOTH modes.
handoff_dir() {
    if (( SAVE_INTERVAL > 0 )); then echo "$1/step_$2"; else echo "$1"; fi
}

# run_phase <name> <data_conf>. All per-phase config is EXPORTED by the caller
# (NUM_STEPS/START_STEP/LR/.../RESUME_FROM/INIT_WEIGHTS_ONLY/SAVE_PATH); this
# only sources the corpus, then torchruns one job of the single-phase entry.
run_phase() {
    local name="$1" data_conf="$2"
    [[ -f "$data_conf" ]] || { echo "ERROR: [$name] DATA_CONF not found: $data_conf" >&2; exit 1; }
    # Fresh PYTHONPATH base each phase, then source the conf — keeps any
    # conf-side path prepends from accumulating across phases.
    export PYTHONPATH="$HARNESS_ROOT/workload/src:$SCRIPT_DIR:${PYTHONPATH:-}"
    # shellcheck disable=SC1090  # data_conf is resolved at runtime by design
    source "$data_conf"
    [[ -n "${DATA_PATH:-}" ]] || { echo "ERROR: [$name] DATA_CONF exported no DATA_PATH" >&2; exit 1; }
    export DATA_PATH
    # Authoritative phase label for the engine's [PHASE] name= banner (the
    # dispatcher segments the trajectory on it). Passing it explicitly keeps a
    # crash-restarted stable/sft phase correctly named instead of the engine's
    # resume_from-based inference collapsing it to "decay".
    export PHASE_NAME="$name"

    echo "==================== [PHASE $name] steps=$NUM_STEPS start=$START_STEP lr=$LR->$MIN_LR decay_iters=$LR_DECAY_ITERS wsd=$LR_WSD_DECAY_ITERS resume=${RESUME_FROM:-<none>} iwo=$INIT_WEIGHTS_ONLY save=${SAVE_PATH:-<none>} ====================" >&2

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "DRY_RUN[$name]: torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT $OURS_ENTRY" >&2
        return 0
    fi

    if [[ -n "${SAVE_PATH:-}" ]]; then mkdir -p "$SAVE_PATH"; fi
    torchrun \
        --nproc_per_node="$GPUS_PER_NODE" \
        --nnodes="$NNODES" \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        "$OURS_ENTRY"
    # torchrun blocks until all local ranks exit; on a single node the save is
    # flushed here. Multi-node handoff is enforced by wait_for_ckpt (below) on
    # the next phase's resume dir.
    echo "[PHASE $name] done." >&2
}

# ── ONE script, TWO entry modes per phase (has_ckpt first-launch/restart split).
#    Each phase inspects ITS OWN save dir for a versioned step_<N>/ ckpt:
#      * none present  -> FIRST LAUNCH: the original cross-phase handoff (fresh
#                         stable / decay-from-stable / sft-weights-from-decay).
#      * present, < end -> CRASH RESTART: full-resume from the phase's own latest
#                         step_<N>/ (params+optim(m/v/step)+cursor), same corpus,
#                         NUM_STEPS = remaining = phase_end_abs - latest. The
#                         engine restores the cursor because resumed data_path ==
#                         the ckpt's saved data_path.
#      * present, >=end -> phase already finished: SKIP (no torchrun, no [PHASE]
#                         banner) so a later attempt fast-forwards past phases
#                         completed in an earlier attempt.
#    NUM_STEPS is RELATIVE to START_STEP, so a restart passes only the remaining
#    span — re-passing the full span overshoots. Phase-end ABSOLUTE step:
#    stable=STABLE_ITERS, decay=DECAY_TOTAL_ITERS, sft=SFT_ITERS. ──

# ── PHASE 1 — STABLE (fresh init, lr plateau: wsd_decay 0). ──
export SAVE_INTERVAL="$STABLE_SAVE_INTERVAL"
export LR="$STABLE_LR" MIN_LR="$STABLE_MIN_LR" LR_WARMUP_ITERS="$STABLE_WARMUP" \
    LR_DECAY_ITERS="$STABLE_ITERS" LR_WSD_DECAY_ITERS=0 SAVE_PATH="$STABLE_SAVE"
stable_latest=$(latest_ckpt_step "$STABLE_SAVE")
if [[ -n "$stable_latest" && "$stable_latest" -ge "$STABLE_ITERS" ]]; then
    echo "[PHASE stable] already complete (latest=$stable_latest >= $STABLE_ITERS) — skip." >&2
elif [[ -n "$stable_latest" ]]; then
    echo "[PHASE stable] crash-restart from step_$stable_latest (remaining=$(( STABLE_ITERS - stable_latest )))." >&2
    export RESUME_FROM="$STABLE_SAVE/step_$stable_latest" INIT_WEIGHTS_ONLY=0 \
        START_STEP="$stable_latest" NUM_STEPS=$(( STABLE_ITERS - stable_latest ))
    run_phase stable "$STABLE_DATA_CONF"
else
    export RESUME_FROM="" INIT_WEIGHTS_ONLY=0 START_STEP=0 NUM_STEPS="$STABLE_ITERS"
    run_phase stable "$STABLE_DATA_CONF"
fi

# ── PHASE 2 — DECAY (first launch: full resume from stable's final step dir,
#    params+optim+step; absolute lr_decay_iters = STABLE+DECAY so the WSD tail
#    anchors at the continuation point; new corpus, cursor NOT reloaded). ──
export SAVE_INTERVAL="$DECAY_SAVE_INTERVAL"
export LR="$DECAY_LR" MIN_LR="$DECAY_MIN_LR" LR_WARMUP_ITERS="$DECAY_WARMUP" \
    LR_DECAY_ITERS="$DECAY_TOTAL_ITERS" LR_WSD_DECAY_ITERS="$DECAY_ITERS" \
    SAVE_PATH="$DECAY_SAVE"
decay_latest=$(latest_ckpt_step "$DECAY_SAVE")
if [[ -n "$decay_latest" && "$decay_latest" -ge "$DECAY_TOTAL_ITERS" ]]; then
    echo "[PHASE decay] already complete (latest=$decay_latest >= $DECAY_TOTAL_ITERS) — skip." >&2
elif [[ -n "$decay_latest" ]]; then
    echo "[PHASE decay] crash-restart from step_$decay_latest (remaining=$(( DECAY_TOTAL_ITERS - decay_latest )))." >&2
    export RESUME_FROM="$DECAY_SAVE/step_$decay_latest" INIT_WEIGHTS_ONLY=0 \
        START_STEP="$decay_latest" NUM_STEPS=$(( DECAY_TOTAL_ITERS - decay_latest ))
    run_phase decay "$DECAY_DATA_CONF"
else
    stable_handoff=$(handoff_dir "$STABLE_SAVE" "$STABLE_ITERS")
    (( NNODES > 1 )) && wait_for_ckpt "$stable_handoff"
    export RESUME_FROM="$stable_handoff" INIT_WEIGHTS_ONLY=0 \
        START_STEP="$STABLE_ITERS" NUM_STEPS="$DECAY_ITERS" NO_LOAD_DATA_STATE=1
    run_phase decay "$DECAY_DATA_CONF"
fi

# ── PHASE 3 — SFT (first launch: weights-only from decay's final step dir; fresh
#    optim, step 0, fresh warmup; new corpus; NO extraction bridge — the engine
#    reads weights straight from the decay ckpt dir via init_weights_only). ──
export SAVE_INTERVAL="$SFT_SAVE_INTERVAL"
export LR="$SFT_LR" MIN_LR="$SFT_MIN_LR" LR_WARMUP_ITERS="$SFT_WARMUP" \
    LR_DECAY_ITERS="$SFT_ITERS" LR_WSD_DECAY_ITERS="$SFT_WSD_DECAY_ITERS" \
    SAVE_PATH="$SFT_SAVE"
sft_latest=$(latest_ckpt_step "$SFT_SAVE")
if [[ -n "$sft_latest" && "$sft_latest" -ge "$SFT_ITERS" ]]; then
    echo "[PHASE sft] already complete (latest=$sft_latest >= $SFT_ITERS) — skip." >&2
elif [[ -n "$sft_latest" ]]; then
    echo "[PHASE sft] crash-restart from step_$sft_latest (remaining=$(( SFT_ITERS - sft_latest )))." >&2
    export RESUME_FROM="$SFT_SAVE/step_$sft_latest" INIT_WEIGHTS_ONLY=0 \
        START_STEP="$sft_latest" NUM_STEPS=$(( SFT_ITERS - sft_latest ))
    run_phase sft "$SFT_DATA_CONF"
else
    decay_handoff=$(handoff_dir "$DECAY_SAVE" "$DECAY_TOTAL_ITERS")
    (( NNODES > 1 )) && wait_for_ckpt "$decay_handoff"
    export RESUME_FROM="$decay_handoff" INIT_WEIGHTS_ONLY=1 \
        START_STEP=0 NUM_STEPS="$SFT_ITERS" NO_LOAD_DATA_STATE=1
    run_phase sft "$SFT_DATA_CONF"
fi

echo "[PIPELINE] all 3 phases done. stable=$STABLE_SAVE decay=$DECAY_SAVE sft=$SFT_SAVE" >&2
