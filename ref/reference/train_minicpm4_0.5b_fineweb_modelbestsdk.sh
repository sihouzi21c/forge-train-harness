#!/bin/bash
# ============================================================================
# MiniCPM4 0.5B dense (openbmb/MiniCPM4-0.5B) + fineweb2_top9 (minicpm4_0.5b)
#
# Authoritative reference for the cctl baseline run (the only baseline
# all suites in this repo align against). Per the ref-as-gate SSOT model
# (`README.md`) this script is BOTH the parameter
# authority AND the execution authority for every gate. Editing this file
# IS editing every gate's reference — there is no sibling baseline file
# to keep in sync, no audit table, no SHA anchor. PR review is the only
# guard.
#
# Configuration:
#   - DATA_CONF must be supplied by the caller as an ABSOLUTE PATH to a
#     Megatron data_conf .sh (this script sources it directly; no fallback,
#     no name-to-path reconstruction). Harness wires it from
#     config/eval.toml [defaults].data_conf via evals/_common.py.
#   - TOKENIZER_TYPE -> Llama2Tokenizer (SentencePiece)
#     (TOKENIZER_MODEL points to tokenizer.model under MiniCPM4-0.5B)
#   - MICRO_BATCH_SIZE default -> 10
#   - GLOBAL_BATCH_SIZE default -> 80
#
# ── Ref-as-gate env-var contract (see README.md) ─────────────────────
#
# Gates dispatched via `tools/ref_script_runner.py` set the variables below
# to drive a single-machine, short-horizon execution of this same script
# without forking a sibling. Any new override must use this contract and be
# documented here — do NOT hand-edit the body to inject test-only knobs.
#
#   LOCAL_MODE                  =1 → skip cctl scheduling logic, run locally
#   FORGE_GATE                  Gate name (metadata / banner only — the
#                               shape comes from the projected product env)
#   DATA_PATH_OVERRIDE          Override DATA_PATH (local-data sanity)
#   DUMP_DIR                    Where the gate runner expects loss / grad /
#                               hook tensors to be dumped
#   M1 capture is not driven through this customer .sh — the M1 agent
#   writes a separate bridge that loads `evals.harness_hook.install`
#   into a copy of the Python entry without touching this file. See
#   `evals/harness_hook/recipes/README.md` for the contract.
#   SEED                        Override --seed (default unchanged)
#   DETERMINISTIC               =0 → disable the bit-wise determinism
#                               stack (CUBLAS_WORKSPACE_CONFIG,
#                               NVTE_ALLOW_NONDETERMINISTIC_ALGO,
#                               PYTHONHASHSEED, --deterministic-mode).
#                               Default is 1 (ON) so M1-M5 alignment
#                               gates are not affected by ref-side
#                               nondeterminism; set to 0 to reproduce
#                               the original cctl run
#                               byte-for-byte.
#
# All overrides above except DETERMINISTIC are inert when unset; sourcing
# this file with DETERMINISTIC=0 from the cctl scheduler still produces
# the original invocation byte-for-byte.
# ============================================================================
set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1

# ── Bit-wise determinism stack ───────────────────────────────────────
# Default ON so M1-M5 alignment gates are not broken by the ref's own
# nondeterminism. Mirrors train_pure_mup_mtp.py's --deterministic
# default ON. The stack is:
#   * CUBLAS_WORKSPACE_CONFIG=:4096:8   (required by cuBLAS for det)
#   * NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 (TE fused attention path)
#   * PYTHONHASHSEED=0                  (python hash determinism)
#   * --deterministic-mode              (Megatron CLI flag — sets
#                                        torch.use_deterministic_algorithms,
#                                        cudnn.deterministic, and disables
#                                        all nondet fused kernels)
# Set DETERMINISTIC=0 to disable for throughput-oriented baseline
# reproduction (e.g. recreating the original cctl run
# byte-for-byte).
DETERMINISTIC="${DETERMINISTIC:-1}"
if [[ "$DETERMINISTIC" == "1" ]]; then
    export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
    export NVTE_ALLOW_NONDETERMINISTIC_ALGO="${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-0}"
    export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_BASENAME="$(basename "${BASH_SOURCE[0]}" .sh)"
FORGE_REPO_ROOT_GUESS="$(cd "$SCRIPT_DIR/../.." && pwd)"
REF_LOCAL_ROOT_DEFAULT="${FORGE_REPO_ROOT_GUESS}/.artifacts/ref_local/${SCRIPT_BASENAME}"
MEGATRON_ROOT="${MEGATRON_ROOT:-}"

# ── Gate parameters arrive as ENVIRONMENT (generic product projection) ──
# Every gate parameter — shape (WORLD_SIZE / NUM_STEPS / MICRO_BATCH_SIZE /
# GLOBAL_BATCH_SIZE / SEED / GATE_WINDOW / RESUME_SAVE_STEP), optimizer,
# model geometry, muP / MTP, init, determinism [env] — comes from this
# gate's rendered product (ref/config/<gate>.toml), projected into the
# environment BY THE CALLER via tools/product_env.py (every [cli] key under
# its upper-cased name, [env] verbatim). This script consumes the generic
# names fail-fast (`:?` / `set -u`) and has no in-launcher projection and
# no baked defaults: a missing key is a render/projection bug.
# Callers: ref/run_gate.sh (gate path), ref/bridges/bridge.sh (capture, env
# inherited from run_gate.sh), tools/bootstrap_canonical.py (canonical,
# projects the product itself). Standalone runs must pre-project:
#   eval "$(python3 tools/product_env.py ref/config/<gate>.toml)"

# In LOCAL_MODE, run single-node: the product's world_size IS the GPU count
# (the old registry emitter's megatron-only NPROC_PER_NODE seeding, moved
# in-launcher), falling back to the locally-detected count rather than
# whatever NNODES env-var the cctl scheduler injected.
if [[ "${LOCAL_MODE:-0}" == "1" ]]; then
    GPUS_PER_NODE="${WORLD_SIZE:-${GPUS_PER_NODE:-$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)}}"
    NNODES="1"
    NODE_RANK="0"
else
    GPUS_PER_NODE="${GPUS_PER_NODE:-$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)}"
    NNODES=${WORLD_SIZE:-"1"}
    NODE_RANK=${RANK:-"0"}
fi
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
DISTRIBUTED_TIMEOUT_MINUTES=${DISTRIBUTED_TIMEOUT_MINUTES:-"60"}

# ── Paths ────────────────────────────────────────────────────────────
# All user-supplied paths come from the harness path shim
# (dense_training.toml [defaults]) or the caller env; nothing is hardcoded.
TOKENIZER_MODEL="${TOKENIZER_MODEL:-}"
VOCAB_SIZE="${VOCAB_SIZE:-73448}"
SAVE_PATH="${SAVE_PATH:-${REF_LOCAL_ROOT_DEFAULT}/save}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${REF_LOCAL_ROOT_DEFAULT}/tensorboard}"
LOAD_PATH="${LOAD_PATH:-}"
TRAIN_ITERS="${NUM_STEPS:?gate product must define num_steps}"
: "${MICRO_BATCH_SIZE:?gate product must define micro_batch_size}"
: "${GLOBAL_BATCH_SIZE:?gate product must define global_batch_size}"

# gate_window arrives as the generic space-joined list ("11 26"); split it
# into the start/end pair the metadata block reports.
if [[ -n "${GATE_WINDOW:-}" ]]; then
    read -r GATE_WINDOW_START GATE_WINDOW_END <<< "$GATE_WINDOW"
fi

# Optimizer knobs come straight from the product [optim] section. The one
# exception is LR_DECAY_ITERS: megatron's historical default is $TRAIN_ITERS,
# preserved as the fallback when the product does not carry it (§D3).
: "${LR:?gate product must define lr}"
: "${MIN_LR:?gate product must define min_lr}"
LR_DECAY_ITERS="${LR_DECAY_ITERS:-$TRAIN_ITERS}"
: "${LR_WSD_DECAY_ITERS:?gate product must define lr_wsd_decay_iters}"
: "${LR_WARMUP_ITERS:?gate product must define lr_warmup_iters}"
: "${WEIGHT_DECAY:?gate product must define weight_decay}"
: "${ADAM_BETA1:?gate product must define adam_beta1}"
: "${ADAM_BETA2:?gate product must define adam_beta2}"
: "${CLIP_GRAD:?gate product must define clip_grad}"
LOG_INTERVAL=${LOG_INTERVAL:-"1"}
SAVE_INTERVAL=${SAVE_INTERVAL:-"10000"}
EVAL_INTERVAL=${EVAL_INTERVAL:-"1000"}
EVAL_ITERS=${EVAL_ITERS:-"10"}
LOG_TASK_LOSS_INTERVAL=${LOG_TASK_LOSS_INTERVAL:-"100"}

# ── Model architecture (from the product's generic projection) ──────
# megatron consumes geometry through the CLI flags built below; the
# generic names are probed fail-fast.
: "${NUM_LAYERS:?gate product must define num_layers}"
: "${HIDDEN_SIZE:?gate product must define hidden_size}"
: "${FFN_HIDDEN_SIZE:?gate product must define ffn_hidden_size}"
: "${NUM_ATTENTION_HEADS:?gate product must define num_attention_heads}"
: "${NUM_QUERY_GROUPS:?gate product must define num_query_groups}"
: "${MAX_POSITION_EMBEDDINGS:?gate product must define max_position_embeddings}"
: "${SEQ_LENGTH:?gate product must define seq_length}"
: "${GRAD_ACCUM_STEPS:?gate product must define grad_accum_steps}"
: "${ROTARY_BASE:?gate product must define rotary_base}"
: "${NORM_EPSILON:?gate product must define norm_epsilon}"
: "${PADDED_VOCAB_SIZE:?gate product must define padded_vocab_size}"
: "${INIT_METHOD_STD:?gate product must define init_method_std}"

# muP — always on (project convention), symmetric with the torch sibling.
: "${MUP_BASE_HIDDEN_SIZE:?gate product must define mup_base_hidden_size}"
: "${MUP_EMB_SCALE:?gate product must define mup_emb_scale}"
: "${MUP_DEPTH_SCALE:?gate product must define mup_depth_scale}"

# MTP / Eagle — always on (same flag names as the torch sibling).
EAGLE_NUM_LAYERS="${MTP_NUM_LAYERS:?gate product must define mtp_num_layers}"
EAGLE_CE_LOSS_WEIGHT="${MTP_LOSS_WEIGHT:?gate product must define mtp_loss_weight}"

write_gate_metadata() {
    local out_dir="$1"
    mkdir -p "$out_dir"
    cat > "$out_dir/gate_metadata.json" <<EOF
{
  "gate": "${FORGE_GATE:-}",
  "num_steps": ${TRAIN_ITERS},
  "world_size": ${GPUS_PER_NODE},
  "seed": ${SEED:-1234},
  "micro_batch_size": ${MICRO_BATCH_SIZE},
  "global_batch_size": ${GLOBAL_BATCH_SIZE},
  "seq_length": ${SEQ_LENGTH},
  "grad_accum_steps": ${GRAD_ACCUM_STEPS},
  "gate_window_start": ${GATE_WINDOW_START:-0},
  "gate_window_end": ${GATE_WINDOW_END:-${TRAIN_ITERS}},
  "resume_save_step": ${RESUME_SAVE_STEP:-0},
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

# ── Ref-as-gate metadata-only path ───────────────────────────────────
# Metadata is owned by this script and must be available without sourcing
# data configs or launching Megatron.
if [[ -n "${DUMP_DIR:-}" && "${PRINT_GATE_METADATA:-0}" == "1" ]]; then
    write_gate_metadata "$DUMP_DIR"
    exit 0
fi

# ── Data ────────────────────────────────────────────────────────────
# DATA_CONF must be the absolute path to a Megatron data_conf .sh — the
# harness orchestrator (evals/_common.py) is the single source of truth
# and always injects a full path; there is no fallback or name-to-path
# reconstruction here on purpose.
: "${DATA_CONF:?DATA_CONF must be set to an absolute path to a data_conf .sh}"
if [[ ! -f "$DATA_CONF" ]]; then
    echo "ERROR: Data conf not found: $DATA_CONF" >&2
    exit 1
fi
# Required deployment env. Same fail-fast contract as the gsm8k variant.
: "${MEGATRON_ROOT:?MEGATRON_ROOT is required (set via [defaults].megatron_root, FORGE_MEGATRON_ROOT env, or --megatron-root CLI flag)}"
: "${TOKENIZER_MODEL:?TOKENIZER_MODEL is required (set via [defaults].tokenizer_model, FORGE_TOKENIZER_MODEL env, or --tokenizer-model CLI flag)}"
source "$DATA_CONF"
# Local-mode override: gate runners point this at a small fixture so the
# ref script can run on a dev box without the full cctl data corpus.
if [[ -n "${DATA_PATH_OVERRIDE:-}" ]]; then
    DATA_PATH="$DATA_PATH_OVERRIDE"
fi

# ── Distributed ──────────────────────────────────────────────────────
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# ── Model: MiniCPM dense (FORGE_*-driven; defaults = MiniCPM4 0.5B) ─
MODEL_ARGS=(
    --use-mcore-models
    --vocab-size "$VOCAB_SIZE"
    --padded-vocab-size "$PADDED_VOCAB_SIZE"
    --make-vocab-size-divisible-by 1
    --disable-bias-linear
    --seq-length "$SEQ_LENGTH"
    --max-position-embeddings "$MAX_POSITION_EMBEDDINGS"
    --num-layers "$NUM_LAYERS"
    --hidden-size "$HIDDEN_SIZE"
    --ffn-hidden-size "$FFN_HIDDEN_SIZE"
    --num-attention-heads "$NUM_ATTENTION_HEADS"
    --group-query-attention
    --num-query-groups "$NUM_QUERY_GROUPS"
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base "$ROTARY_BASE"
    --norm-epsilon "$NORM_EPSILON"
    --init-method-std "$INIT_METHOD_STD"
    # muP (symmetric with the torch sibling).
    --mup-base-hidden-size "$MUP_BASE_HIDDEN_SIZE"
    --mup-emb-scale "$MUP_EMB_SCALE"
    --mup-depth-scale "$MUP_DEPTH_SCALE"
    # MTP / Eagle (always on).
    --eagle-num-layers "$EAGLE_NUM_LAYERS"
    --eagle-ce-loss-weight "$EAGLE_CE_LOSS_WEIGHT"
)

# ── Data loading ─────────────────────────────────────────────────────
TOKENIZER_TYPE="${TOKENIZER_TYPE:-Llama2Tokenizer}"
if [[ "$TOKENIZER_TYPE" == "NullTokenizer" ]]; then
    DATA_ARGS=(
        --tokenizer-type NullTokenizer
        --split 99990,8,2
        --use-modelbest-sdk
        --dataloader-type external
    )
else
    DATA_ARGS=(
        --tokenizer-type "$TOKENIZER_TYPE"
        --tokenizer-model "$TOKENIZER_MODEL"
        --split 99990,8,2
        --use-modelbest-sdk
        --dataloader-type external
    )
fi

# ── Training hyperparams ─────────────────────────────────────────────
TRAINING_ARGS=(
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --lr "$LR"
    --min-lr "$MIN_LR"
    --train-iters "$TRAIN_ITERS"
    --lr-decay-iters "$LR_DECAY_ITERS"
    --lr-decay-style WSD
    --lr-wsd-decay-style exponential
    --lr-wsd-decay-iters "$LR_WSD_DECAY_ITERS"
    --lr-warmup-iters "$LR_WARMUP_ITERS"
    --weight-decay "$WEIGHT_DECAY"
    --clip-grad "$CLIP_GRAD"
    --adam-beta1 "$ADAM_BETA1"
    --adam-beta2 "$ADAM_BETA2"
    --bf16
    --distributed-timeout-minutes "$DISTRIBUTED_TIMEOUT_MINUTES"
)
if [[ -n "${SEED:-}" ]]; then
    TRAINING_ARGS+=(--seed "$SEED")
fi
if [[ "$DETERMINISTIC" == "1" ]]; then
    TRAINING_ARGS+=(--deterministic-mode)
fi

# ── Parallelism ──────────────────────────────────────────────────────
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --use-distributed-optimizer
    --sequence-parallel
)

# ── Activation recompute ─────────────────────────────────────────────
RECOMPUTE_ARGS=(
    --recompute-granularity full
    --recompute-method block
    --recompute-num-layers 8
)

# ── Logging and checkpoint ───────────────────────────────────────────
LOGGING_ARGS=(
    --log-interval "$LOG_INTERVAL"
    --save-interval "$SAVE_INTERVAL"
    --eval-interval "$EVAL_INTERVAL"
    --eval-iters "$EVAL_ITERS"
    --log-task-loss-interval "$LOG_TASK_LOSS_INTERVAL"
    --ckpt-format torch
    --log-throughput
)

# ── Save / Load / Tensorboard ────────────────────────────────────────
SAVE_LOAD_ARGS=(
    --save "$SAVE_PATH"
    --tensorboard-dir "$TENSORBOARD_DIR"
    --no-load-data-state
)
if [[ -n "$LOAD_PATH" ]]; then
    SAVE_LOAD_ARGS+=(--load "$LOAD_PATH")
fi

# ── Ref-as-gate dump dir (gate runners parse files written here) ─────
# Inert when DUMP_DIR is unset (production / cctl path); when set, all
# gate-only side-effects (loss trace, grad-norm trace, hook tensors)
# go under this directory so the dispatcher can collect them.
if [[ -n "${DUMP_DIR:-}" ]]; then
    mkdir -p "$DUMP_DIR"
    export DUMP_DIR
    write_gate_metadata "$DUMP_DIR"
fi

# ── Launch ───────────────────────────────────────────────────────────
cd "$MEGATRON_ROOT"

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_minicpm.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${RECOMPUTE_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${SAVE_LOAD_ARGS[@]} \
    --data-path $DATA_PATH \
    "$@"
