#!/bin/bash
# MiniCPM4 0.5B dense + muP + MTP — pure PyTorch DP-only training.
# Defaults: DP=2 (single-node H100), GBS=80, MBS=4, 1000 steps.
#
# This is the alternative L0 reference stack to ``train_minicpm4_0.5b_gsm8k.sh``
# (Megatron-LM).  Either script may be wired in via ``[defaults].ref_script``
# in ``config/eval.toml``; the harness honors the same env contract for
# both (LOCAL_MODE / FORGE_GATE / DUMP_DIR / LOSS_DUMP_FILE /
# PRINT_GATE_METADATA, plus WORLD_SIZE / NUM_STEPS / MICRO_BATCH_SIZE /
# GLOBAL_BATCH_SIZE / LR_WARMUP_ITERS / GATE_WINDOW).
#
# Gate parameters arrive as ENVIRONMENT — the rendered ref product
# (ref/config/<gate>.toml), projected by the caller via the fully generic
# tools/product_env.py: every [cli] key under its upper-cased name, [env]
# verbatim. There is NO in-launcher projection and NO fallback cascade: a
# missing key is a render/projection bug and trips `:?` / `set -u`.
# Callers: ref/run_gate.sh (gate path), ref/bridges/bridge.sh (capture,
# env inherited from run_gate.sh), tools/bootstrap_canonical.py
# (canonical, projects the product itself). Standalone runs must
# pre-project:  eval "$(python3 tools/product_env.py ref/config/<gate>.toml)"
#
# Caller knobs:
#   DATA_CONF                  Path to a shell file that ``source``s into
#                              ``$DATA_PATH`` (a modelbest_sdk weighted
#                              shard string). Shared with the Megatron
#                              sibling — wired from ``[data].conf_path``
#                              via ``evals/_common.py:_ref_script_path_env``.
#                              The script materializes ``$DATA_PATH`` into
#                              a tmp textfile and passes it via
#                              ``--data-path-file`` to the python entry.
#   DATA_PATH_FILE             Direct shard-list textfile path (overrides
#                              the DATA_CONF-derived tmp file). Optional.
#   OUT_DIR                    Log + checkpoint root.
#   GPUS_PER_NODE              Per-node GPU count (default 2).
#   LOCAL_MODE=1               Single-node local run (skip cctl scheduler path).
#   FORGE_GATE=<name>          Gate name (metadata / banner only).
#   PRINT_GATE_METADATA=1      Write $DUMP_DIR/gate_metadata.json and exit
#                              without launching torchrun.
#   LOSS_DUMP_FILE             Mirror per-step `[LOSS] …` lines to this path.
#
# ``WORLD_SIZE`` has the same double meaning as in the Megatron sibling
# script: under LOCAL_MODE / PRINT_GATE_METADATA it is the DP world size
# (total ranks on a single node); in the cctl scheduler path it is the
# node count (RANK = node rank, total DP = WORLD_SIZE * GPUS_PER_NODE).
#
# Like the Megatron sibling, M1 forward / backward alignment is *not*
# driven through this script — the M1 agent writes a separate "capture
# bridge" that wires ``evals.harness_hook.install`` into a copy of the
# Python entry without touching this file.  See
# ``evals/harness_hook/recipes/README.md`` for the contract and patterns.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PY_ENTRY="$SCRIPT_DIR/train_pure_mup_mtp.py"

# ── Canonical-state bootstrap: collapse to a single (DP=1) replica ───
# A canonical dump (CANONICAL_STATE_OUTPUT_FILE set by
# tools/bootstrap_canonical.py) fires at the first optimizer.step(), which the
# harness hook REPLACES with a dump-and-exit — the real step never runs. The
# artifact is the INITIAL weights, invariant to the data-parallel degree, so
# one process (LOCAL_MODE WORLD=1) reproduces the byte-identical canonical using
# a single GPU instead of demanding the gate's full world_size_override worth of
# cards (the clobber that bricked the canonical preflight on a smaller box).
if [[ -n "${CANONICAL_STATE_OUTPUT_FILE:-}" ]]; then
    WORLD_SIZE=1
fi

# ── Distributed topology resolution ──────────────────────────────────
# LOCAL_MODE=1 / PRINT_GATE_METADATA=1: single-node, DP = $WORLD_SIZE
#   (preset sets WORLD_SIZE per gate; caller can override).
# Cctl scheduler path: WORLD_SIZE = node count, RANK = node rank,
#   DP = WORLD_SIZE * GPUS_PER_NODE.
# Autodetect the per-node GPU count from the exec host instead of a hardcoded
# default so the leased box's real GPU count is authoritative (an explicit
# GPUS_PER_NODE env still wins; numeric fallback covers hosts without nvidia-smi).
if [[ -z "${GPUS_PER_NODE:-}" ]]; then
    GPUS_PER_NODE="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
fi
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
(( GPUS_PER_NODE > 0 )) 2>/dev/null || GPUS_PER_NODE=2
if [[ "${LOCAL_MODE:-0}" == "1" || "${PRINT_GATE_METADATA:-0}" == "1" ]]; then
    DP_WORLD_SIZE="${WORLD_SIZE:-${GPUS_PER_NODE}}"
    if [[ "${LOCAL_MODE:-0}" == "1" ]] && (( DP_WORLD_SIZE > GPUS_PER_NODE )); then
        echo "ERROR: LOCAL_MODE=1 with DP_WORLD_SIZE=$DP_WORLD_SIZE > GPUS_PER_NODE=$GPUS_PER_NODE" >&2
        exit 1
    fi
    NPROC_PER_NODE="$DP_WORLD_SIZE"
    NNODES=1
    NODE_RANK=0
else
    NPROC_PER_NODE="$GPUS_PER_NODE"
    NNODES="${WORLD_SIZE:-1}"
    NODE_RANK="${RANK:-0}"
    DP_WORLD_SIZE=$(( NNODES * GPUS_PER_NODE ))
fi
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"23456"}

# ── Training shape (from the product; feeds the python CLI) ─────────
TRAIN_ITERS="${NUM_STEPS:?gate product must define num_steps}"
: "${MICRO_BATCH_SIZE:?gate product must define micro_batch_size}"
: "${GLOBAL_BATCH_SIZE:?gate product must define global_batch_size}"
: "${SEED:?gate product must define seed}"

# gate_window arrives as the generic space-joined list ("11 26"); split it
# into the start/end pair the metadata block reports.
if [[ -n "${GATE_WINDOW:-}" ]]; then
    read -r GATE_WINDOW_START GATE_WINDOW_END <<< "$GATE_WINDOW"
fi

# ── Optimizer (from the product [optim] section) ────────────────────
# The product is the sole source; the warmup clamp below is the only
# value the launcher still recomputes (gates may set TRAIN_ITERS=1).
: "${LR:?gate product must define lr}"
: "${MIN_LR:?gate product must define min_lr}"
: "${LR_WARMUP_ITERS:?gate product must define lr_warmup_iters}"
: "${LR_DECAY_ITERS:?gate product must define lr_decay_iters}"
: "${LR_WSD_DECAY_ITERS:?gate product must define lr_wsd_decay_iters}"
: "${WEIGHT_DECAY:?gate product must define weight_decay}"
: "${ADAM_BETA1:?gate product must define adam_beta1}"
: "${ADAM_BETA2:?gate product must define adam_beta2}"
: "${CLIP_GRAD:?gate product must define clip_grad}"
if (( LR_WARMUP_ITERS >= TRAIN_ITERS )); then
    LR_WARMUP_ITERS=0
fi

# ── Model architecture (from the product) ───────────────────────────
# model_pure_mup_mtp.py reads the same upper-cased geometry names from
# os.environ at import (the generic projection exported every product
# key). The `:?` probes below fail fast in-shell when a geometry key is
# missing, before torchrun spends GPU time on a KeyError. HEAD_DIM /
# PADDED_VOCAB_SIZE / ROTARY_BASE / NORM_EPSILON have no launcher
# consumer (model.py reads them directly), so no probe here.
: "${NUM_LAYERS:?gate product must define num_layers}"
: "${HIDDEN_SIZE:?gate product must define hidden_size}"
: "${FFN_HIDDEN_SIZE:?gate product must define ffn_hidden_size}"
: "${NUM_ATTENTION_HEADS:?gate product must define num_attention_heads}"
: "${NUM_QUERY_GROUPS:?gate product must define num_query_groups}"
: "${MAX_POSITION_EMBEDDINGS:?gate product must define max_position_embeddings}"
: "${INIT_METHOD_STD:?gate product must define init_method_std}"

# INIT_ONES — REQUIRED (product key forge_init_ones → env FORGE_INIT_ONES).
# Selects between standard ones init (1) and anti-cheat 0.97 init (0) for
# RMSNorm weights. The product always carries forge_init_ones, so a
# bitwise-gate run can never silently fall back to ones init. See
# train_pure_mup_mtp.py --init-ones and
# model_pure_mup_mtp.MiniCPM4MupMtp.init_weights for the contract.
INIT_ONES="${FORGE_INIT_ONES:?gate product must define forge_init_ones (1=production / 0=M1-M5 bitwise gates)}"
if [[ "$INIT_ONES" != "0" && "$INIT_ONES" != "1" ]]; then
    echo "ERROR: FORGE_INIT_ONES must be 0 or 1, got '$INIT_ONES'" >&2
    exit 1
fi

: "${MUP_BASE_HIDDEN_SIZE:?gate product must define mup_base_hidden_size}"
: "${MUP_EMB_SCALE:?gate product must define mup_emb_scale}"
: "${MUP_DEPTH_SCALE:?gate product must define mup_depth_scale}"
EAGLE_NUM_LAYERS="${MTP_NUM_LAYERS:?gate product must define mtp_num_layers}"
EAGLE_CE_LOSS_WEIGHT="${MTP_LOSS_WEIGHT:?gate product must define mtp_loss_weight}"

# grad_accum_steps / seq_length are renderer-derived and carried by the
# product; consuming them directly (instead of recomputing from the runtime
# DP world) keeps the product the single source of truth. The renderer
# fail-fasts when GBS does not factor through MBS · DP.
: "${GRAD_ACCUM_STEPS:?gate product must define grad_accum_steps}"
: "${SEQ_LENGTH:?gate product must define seq_length}"

# ── Paths ────────────────────────────────────────────────────────────
OUT_DIR="${OUT_DIR:-$REPO_ROOT/.artifacts/ref_local/pure_torch_16gpu_1000step_mup_mtp}"
mkdir -p "$OUT_DIR"
ENGINE_ROOT="${ENGINE_ROOT:-$REPO_ROOT}"

# Accept DATA_CONF (same env the Megatron sibling reads) so both ref
# scripts share a single weighted-shard authority. The Megatron sibling
# `source`s DATA_CONF to populate $DATA_PATH, then passes --data-path to
# Megatron; here we do the same `source`, then materialize $DATA_PATH
# into a tmp textfile that train_pure_mup_mtp.py reads via
# --data-path-file (its python entry calls .read().split() so any
# whitespace-separated "weight path" tokens are accepted).
if [[ -z "${DATA_PATH_FILE:-}" && -n "${DATA_CONF:-}" ]]; then
    if [[ ! -f "$DATA_CONF" ]]; then
        echo "ERROR: Data conf not found: $DATA_CONF" >&2
        exit 1
    fi
    source "$DATA_CONF"
    mkdir -p "$REPO_ROOT/tmp"
    DATA_PATH_FILE="$(mktemp "$REPO_ROOT/tmp/data_path.XXXXXX")"
    printf "%s\n" "$DATA_PATH" > "$DATA_PATH_FILE"
    trap 'rm -f "$DATA_PATH_FILE"' EXIT
fi
DATA_PATH_FILE="${DATA_PATH_FILE:-${ENGINE_ROOT}/workload/data/lizhen_0.5b_data_path.txt}"

write_gate_metadata() {
    local out_dir="$1"
    mkdir -p "$out_dir"
    cat > "$out_dir/gate_metadata.json" <<EOF
{
  "gate": "${FORGE_GATE:-}",
  "num_steps": ${TRAIN_ITERS},
  "world_size": ${DP_WORLD_SIZE},
  "seed": ${SEED},
  "micro_batch_size": ${MICRO_BATCH_SIZE},
  "global_batch_size": ${GLOBAL_BATCH_SIZE},
  "seq_length": ${SEQ_LENGTH},
  "grad_accum_steps": ${GRAD_ACCUM_STEPS},
  "gate_window_start": ${GATE_WINDOW_START:-0},
  "gate_window_end": ${GATE_WINDOW_END:-${TRAIN_ITERS}},
  "resume_save_step": ${RESUME_SAVE_STEP:-0},
  "lr": ${LR},
  "lr_warmup_iters": ${LR_WARMUP_ITERS},
  "lr_decay_iters": ${LR_DECAY_ITERS},
  "lr_wsd_decay_iters": ${LR_WSD_DECAY_ITERS},
  "num_layers": ${NUM_LAYERS},
  "hidden_size": ${HIDDEN_SIZE},
  "ffn_hidden_size": ${FFN_HIDDEN_SIZE},
  "num_attention_heads": ${NUM_ATTENTION_HEADS},
  "num_query_groups": ${NUM_QUERY_GROUPS},
  "init_method_std": ${INIT_METHOD_STD},
  "mup_base_hidden_size": ${MUP_BASE_HIDDEN_SIZE},
  "mup_emb_scale": ${MUP_EMB_SCALE},
  "mup_depth_scale": ${MUP_DEPTH_SCALE},
  "eagle_num_layers": ${EAGLE_NUM_LAYERS},
  "eagle_ce_loss_weight": ${EAGLE_CE_LOSS_WEIGHT}
}
EOF
}

# Metadata-only path: write gate_metadata.json and exit (no torchrun).
if [[ -n "${DUMP_DIR:-}" && "${PRINT_GATE_METADATA:-0}" == "1" ]]; then
    write_gate_metadata "$DUMP_DIR"
    exit 0
fi

# ── DATA_PATH_FILE sanity check ──────────────────────────────────────
if [[ ! -f "$DATA_PATH_FILE" ]]; then
    echo "ERROR: DATA_PATH_FILE not found: $DATA_PATH_FILE" >&2
    echo "       (set DATA_PATH_FILE env var or place the file at" >&2
    echo "        workload/data/lizhen_0.5b_data_path.txt under ENGINE_ROOT)" >&2
    exit 1
fi
if [[ ! -f "$PY_ENTRY" ]]; then
    echo "ERROR: train_pure_mup_mtp.py not found at $PY_ENTRY" >&2
    exit 1
fi

# ── Runtime env defaults ─────────────────────────────────────────────
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# DUMP_DIR + LOSS_DUMP_FILE forwarding to the python entry.
if [[ -n "${DUMP_DIR:-}" ]]; then
    mkdir -p "$DUMP_DIR"
    write_gate_metadata "$DUMP_DIR"
fi
export DUMP_DIR="${DUMP_DIR:-}"
export LOSS_DUMP_FILE="${LOSS_DUMP_FILE:-}"

# ── Launch ───────────────────────────────────────────────────────────
PY_ARGS=(
    --data-path-file "$DATA_PATH_FILE"
    --data-config "${FORGE_DATA_TOML:-}"
    --train-iters "$TRAIN_ITERS"
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --lr "$LR"
    --min-lr "$MIN_LR"
    --lr-warmup-iters "$LR_WARMUP_ITERS"
    --lr-decay-iters "$LR_DECAY_ITERS"
    --lr-wsd-decay-iters "$LR_WSD_DECAY_ITERS"
    --weight-decay "$WEIGHT_DECAY"
    --adam-beta1 "$ADAM_BETA1"
    --adam-beta2 "$ADAM_BETA2"
    --clip-grad "$CLIP_GRAD"
    --seed "$SEED"
    --init-method-std "$INIT_METHOD_STD"
    --init-ones "$INIT_ONES"
    --mup-base-hidden-size "$MUP_BASE_HIDDEN_SIZE"
    --mup-emb-scale "$MUP_EMB_SCALE"
    --mup-depth-scale "$MUP_DEPTH_SCALE"
    --eagle-num-layers "$EAGLE_NUM_LAYERS"
    --eagle-ce-loss-weight "$EAGLE_CE_LOSS_WEIGHT"
    --out-dir "$OUT_DIR"
    --log-interval 1
    --recompute
)

echo "=== Pure PyTorch MiniCPM4 0.5B + muP + MTP ==="
echo "  FORGE_GATE=${FORGE_GATE:-<none>}  TRAIN_ITERS=$TRAIN_ITERS"
echo "  NNODES=$NNODES NODE_RANK=$NODE_RANK NPROC_PER_NODE=$NPROC_PER_NODE  DP=$DP_WORLD_SIZE"
echo "  GBS=$GLOBAL_BATCH_SIZE MBS=$MICRO_BATCH_SIZE GRAD_ACCUM=$GRAD_ACCUM_STEPS  SEED=$SEED"
echo "  LR=$LR  init_std=$INIT_METHOD_STD  mup_emb_scale=$MUP_EMB_SCALE  mup_depth_scale=$MUP_DEPTH_SCALE"
echo "  init_ones=$INIT_ONES  eagle_num_layers=$EAGLE_NUM_LAYERS  ce_w=$EAGLE_CE_LOSS_WEIGHT"

set -x
torchrun \
    --nproc_per_node="$NPROC_PER_NODE" \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "$PY_ENTRY" \
    "${PY_ARGS[@]}" \
    "$@" \
    2>&1 | tee "$OUT_DIR/master_${NODE_RANK}.log"
