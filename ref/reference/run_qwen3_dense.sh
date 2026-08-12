#!/bin/bash
# Qwen3 0.6B dense + QK-Norm — pure PyTorch DP-only training.
# Defaults: DP=2 (single-node H100), GBS=80, MBS=4, 1000 steps.
#
# Sibling of ``run_16gpu_1000step_pure_mup_mtp.sh`` (the MiniCPM torch
# reference) for the Qwen3 model axis. It honors the same harness env
# contract (LOCAL_MODE / FORGE_GATE / DUMP_DIR / LOSS_DUMP_FILE /
# PRINT_GATE_METADATA, plus WORLD_SIZE / NUM_STEPS / MICRO_BATCH_SIZE /
# GLOBAL_BATCH_SIZE / LR_WARMUP_ITERS / GATE_WINDOW), but launches the
# Qwen3 entry (train_qwen3_dense.py) with NO muP / MTP CLI args.
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
# Caller knobs (identical to the MiniCPM sibling):
#   DATA_CONF / DATA_PATH_FILE   data source (modelbest weighted shards or
#                                hf:// tokens; sourced into $DATA_PATH).
#   OUT_DIR                      Log + checkpoint root.
#   GPUS_PER_NODE                Per-node GPU count (default 2).
#   LOCAL_MODE=1                 Single-node local run.
#   FORGE_GATE=<name>            Gate name (metadata / banner only).
#   PRINT_GATE_METADATA=1        Write $DUMP_DIR/gate_metadata.json and exit.
#   LOSS_DUMP_FILE               Mirror per-step `[LOSS] …` lines to this path.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PY_ENTRY="$SCRIPT_DIR/train_qwen3_dense.py"

# ── Canonical-state bootstrap: collapse to a single (DP=1) replica ───
# A canonical dump (CANONICAL_STATE_OUTPUT_FILE set by
# tools/bootstrap_canonical.py) fires at the first optimizer.step(), which the
# harness hook REPLACES with a dump-and-exit — the real step never runs. The
# artifact is the INITIAL weights, invariant to the data-parallel degree, so
# one process (WORLD=1) reproduces the byte-identical canonical using a single
# GPU instead of demanding the gate's full world_size_override worth of cards
# (the clobber that bricked the canonical preflight on a smaller box).
if [[ -n "${CANONICAL_STATE_OUTPUT_FILE:-}" ]]; then
    WORLD_SIZE=1
fi

# ── Distributed topology resolution ──────────────────────────────────
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
# model_qwen3.py reads the same upper-cased geometry names from os.environ
# at import (the generic projection exported every product key). The `:?`
# probes below fail fast in-shell when a geometry key is missing, before
# torchrun spends GPU time on a KeyError. ROTARY_BASE / NORM_EPSILON have
# no launcher consumer (model.py reads them directly), so no probe here.
: "${NUM_LAYERS:?gate product must define num_layers}"
: "${HIDDEN_SIZE:?gate product must define hidden_size}"
: "${FFN_HIDDEN_SIZE:?gate product must define ffn_hidden_size}"
: "${NUM_ATTENTION_HEADS:?gate product must define num_attention_heads}"
: "${NUM_QUERY_GROUPS:?gate product must define num_query_groups}"
: "${HEAD_DIM:?gate product must define head_dim}"
: "${MAX_POSITION_EMBEDDINGS:?gate product must define max_position_embeddings}"
: "${PADDED_VOCAB_SIZE:?gate product must define padded_vocab_size}"
: "${INIT_METHOD_STD:?gate product must define init_method_std}"
# QK-Norm and tied embeddings are intrinsic to the Qwen3 architecture, NOT
# tunable [model] knobs — they live entirely inside model_qwen3.py (which
# hardcodes both on). These two constants are display-only (gate_metadata
# echo below); they are NOT exported to the model and have no FORGE_* env.
USE_QK_NORM=1
TIE_WORD_EMBEDDINGS=1

# grad_accum_steps / seq_length are renderer-derived and carried by the
# product; consuming them directly (instead of recomputing from the runtime
# DP world) keeps the product the single source of truth. The renderer
# fail-fasts when GBS does not factor through MBS · DP.
: "${GRAD_ACCUM_STEPS:?gate product must define grad_accum_steps}"
: "${SEQ_LENGTH:?gate product must define seq_length}"

# ── Paths ────────────────────────────────────────────────────────────
OUT_DIR="${OUT_DIR:-$REPO_ROOT/.artifacts/ref_local/pure_torch_qwen3_dense}"
mkdir -p "$OUT_DIR"
ENGINE_ROOT="${ENGINE_ROOT:-$REPO_ROOT}"

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
  "head_dim": ${HEAD_DIM},
  "init_method_std": ${INIT_METHOD_STD},
  "use_qk_norm": "${USE_QK_NORM}",
  "tie_word_embeddings": "${TIE_WORD_EMBEDDINGS}"
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
    echo "       (set DATA_PATH_FILE / DATA_CONF env var)" >&2
    exit 1
fi
if [[ ! -f "$PY_ENTRY" ]]; then
    echo "ERROR: train_qwen3_dense.py not found at $PY_ENTRY" >&2
    exit 1
fi

# ── Runtime env defaults ─────────────────────────────────────────────
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
    --out-dir "$OUT_DIR"
    --log-interval 1
    --recompute
)

echo "=== Pure PyTorch Qwen3 0.6B dense + QK-Norm ==="
echo "  FORGE_GATE=${FORGE_GATE:-<none>}  TRAIN_ITERS=$TRAIN_ITERS"
echo "  NNODES=$NNODES NODE_RANK=$NODE_RANK NPROC_PER_NODE=$NPROC_PER_NODE  DP=$DP_WORLD_SIZE"
echo "  GBS=$GLOBAL_BATCH_SIZE MBS=$MICRO_BATCH_SIZE GRAD_ACCUM=$GRAD_ACCUM_STEPS  SEED=$SEED"
echo "  LR=$LR  init_std=$INIT_METHOD_STD  layers=$NUM_LAYERS kv_heads=$NUM_QUERY_GROUPS head_dim=$HEAD_DIM"
echo "  use_qk_norm=$USE_QK_NORM  tie_word_embeddings=$TIE_WORD_EMBEDDINGS  vocab=$PADDED_VOCAB_SIZE"

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
