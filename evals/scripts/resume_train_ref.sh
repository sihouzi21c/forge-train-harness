#!/bin/bash
# Ref-side RESUME-milestone short-train job entry — the per-pod command of the
# cctl PyTorchJob submitted by launch_resume.sh ref. Runs the L0 reference
# stack (ref/reference/train_pure_mup_mtp.py, pure-PyTorch muP+MTP) for a SHORT
# window (default 20 steps; config/train/resume_20.toml [train].iters) — the
# ref-side truth run of the same recipe resume_train_ours.sh trains, so the
# resume milestone has a job-submitted reference trajectory to eyeball the
# ours job against. Sibling: resume_train_ours.sh.
#
# Parameter sources (explicit precedence):
#   * geometry / muP / MTP / optimizer constants / init — the RENDERED REF
#     PRODUCT ref/config/<FORGE_RESUME_SUITE>.toml (default resume-gate-20),
#     projected via tools/gate_product_to_shell.py exactly like the tracked L0
#     launcher does. The product is a contract: it always wins for these.
#   * run shape (iters / MBS / GBS / seed / LR schedule) — env > recipe TOML.
#     Snapshotted into R_* BEFORE the product projection so the recipe wins
#     over the product's gate-scale num_steps/world_size for the RUN SHAPE
#     only (that is the point of this job being TOML-configurable).
#
# We torchrun the python entry directly (same pattern as train_ours_al.sh)
# instead of exec'ing run_16gpu_1000step_pure_mup_mtp.sh because that launcher
# re-sources the product AFTER any caller export, so a recipe-driven step count
# could never win there. The L0 launcher itself stays untouched (frozen truth).
#
# NOT a resume test on this side: the ref entry has no --resume-from primitive
# (see wsd-sft-70.toml header) — this is the short truth RUN; resume mechanics
# are proven on the ours side. NOT a gate: no verdict; verify by returncode 0
# and the [LOSS] lines in `cctl logs`.
#
# Env in: SAVE_ROOT (required; holds out-dir + materialized data-path file),
# RESUME_RECIPE_TOML / FORGE_RESUME_SUITE / DATA_CONF / GPUS_PER_NODE optional;
# WORLD_SIZE/RANK/MASTER_* from the PyTorchJob operator.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
REF_DIR="$REPO_ROOT/ref/reference"
PY_ENTRY="$REF_DIR/train_pure_mup_mtp.py"
[[ -f "$PY_ENTRY" ]] || { echo "FATAL: missing $PY_ENTRY" >&2; exit 1; }

set -euo pipefail

# ── 1) Recipe TOML (env-wins), then snapshot the recipe-owned run shape. ──
RESUME_RECIPE_TOML="${RESUME_RECIPE_TOML:-$REPO_ROOT/config/train/resume_20.toml}"
[[ -f "$RESUME_RECIPE_TOML" ]] || {
    echo "FATAL: resume recipe TOML not found: $RESUME_RECIPE_TOML" >&2
    exit 1
}
# eval-form, not `source <(...)` — old bash (mac 3.2) silently sources zero
# bytes from a process substitution; eval is version-robust.
eval "$(python3 "$REPO_ROOT/tools/train_recipe_to_env.py" "$RESUME_RECIPE_TOML")"
R_ITERS="${ITERS:-20}"
R_MBS="${MICRO_BATCH_SIZE:-4}"
R_GBS="${GLOBAL_BATCH_SIZE:-80}"
R_SEED="${SEED:-1234}"
R_LR="${LR:-1e-2}"
R_MIN_LR="${MIN_LR:-0}"
R_WARMUP="${WARMUP:-0}"
R_WSD="${WSD_DECAY_ITERS:-0}"
R_LR_DECAY_ITERS="${LR_DECAY_ITERS:-$R_ITERS}"

# ── 2) Topology — snapshot the OPERATOR's WORLD_SIZE (node count) before the
#    product projection clobbers WORLD_SIZE with the gate's DP world. ──
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
NNODES="${WORLD_SIZE:-1}"
NODE_RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-23456}"
DP_WORLD_SIZE=$(( NNODES * GPUS_PER_NODE ))

# ── 3) Rendered ref product: geometry / muP / MTP / optimizer / init SSOT. ──
FORGE_SUITE="${FORGE_RESUME_SUITE:-resume-gate-20}"
_GATE_PRODUCT="${FORGE_REF_CONFIG_DIR:-$REPO_ROOT/ref/config}/$FORGE_SUITE.toml"
[[ -f "$_GATE_PRODUCT" ]] || {
    echo "FATAL: rendered ref product not found: $_GATE_PRODUCT" >&2
    echo "       Render it (python -m tools.render_gate_configs) and sync push before submitting." >&2
    exit 1
}
eval "$(python3 "$REPO_ROOT/tools/gate_product_to_shell.py" "$_GATE_PRODUCT")"

# ── 4) Corpus: DATA_CONF exports DATA_PATH; default = the [data] axis conf.
#    Materialized into a textfile for --data-path-file (launcher pattern). ──
export FORGE_DATA_TOML="${FORGE_DATA_TOML:-$REPO_ROOT/config/data.toml}"
[[ -f "$FORGE_DATA_TOML" ]] || {
    echo "FATAL: FORGE_DATA_TOML not found: $FORGE_DATA_TOML" >&2
    exit 1
}
if [[ -z "${DATA_CONF:-}" ]]; then
    DATA_CONF="$REPO_ROOT/$(python3 -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["data"]["conf_path"])' "$FORGE_DATA_TOML")"
fi
[[ -f "$DATA_CONF" ]] || { echo "FATAL: DATA_CONF not found: $DATA_CONF" >&2; exit 1; }
# shellcheck disable=SC1090  # DATA_CONF is resolved at runtime by design
source "$DATA_CONF"
[[ -n "${DATA_PATH:-}" ]] || { echo "FATAL: DATA_CONF exported no DATA_PATH" >&2; exit 1; }

SAVE_ROOT="${SAVE_ROOT:?SAVE_ROOT must be a per-loop persistent shared-FS path (set by launch_resume.sh)}"
OUT_DIR="${OUT_DIR:-$SAVE_ROOT/ref_out}"
mkdir -p "$OUT_DIR"
DATA_PATH_FILE="${DATA_PATH_FILE:-$OUT_DIR/data_path_node${NODE_RANK}.txt}"
printf "%s\n" "$DATA_PATH" > "$DATA_PATH_FILE"

# ── 5) Shape checks + runtime env (mirror the L0 launcher). ──
if (( R_GBS % (R_MBS * DP_WORLD_SIZE) != 0 )); then
    echo "ERROR: GBS=$R_GBS not divisible by MBS=$R_MBS * DP=$DP_WORLD_SIZE" >&2
    exit 1
fi
if (( R_WARMUP >= R_ITERS )); then
    R_WARMUP=0
fi
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY_ARGS=(
    --data-path-file "$DATA_PATH_FILE"
    --data-config "$FORGE_DATA_TOML"
    --train-iters "$R_ITERS"
    --micro-batch-size "$R_MBS"
    --global-batch-size "$R_GBS"
    --lr "$R_LR"
    --min-lr "$R_MIN_LR"
    --lr-warmup-iters "$R_WARMUP"
    --lr-decay-iters "$R_LR_DECAY_ITERS"
    --lr-wsd-decay-iters "$R_WSD"
    --weight-decay "${WEIGHT_DECAY:?ref product must define weight_decay}"
    --adam-beta1 "${ADAM_BETA1:?ref product must define adam_beta1}"
    --adam-beta2 "${ADAM_BETA2:?ref product must define adam_beta2}"
    --clip-grad "${CLIP_GRAD:?ref product must define clip_grad}"
    --seed "$R_SEED"
    --init-method-std "${FORGE_INIT_METHOD_STD:?ref product must define init_method_std}"
    --init-ones "${INIT_ONES:?ref product must define forge_init_ones}"
    --mup-base-hidden-size "${FORGE_MUP_BASE_HIDDEN_SIZE:?ref product must define mup_base_hidden_size}"
    --mup-emb-scale "${FORGE_MUP_EMB_SCALE:?ref product must define mup_emb_scale}"
    --mup-depth-scale "${FORGE_MUP_DEPTH_SCALE:?ref product must define mup_depth_scale}"
    --eagle-num-layers "${FORGE_MTP_NUM_LAYERS:?ref product must define mtp_num_layers}"
    --eagle-ce-loss-weight "${FORGE_MTP_LOSS_WEIGHT:?ref product must define mtp_loss_weight}"
    --out-dir "$OUT_DIR"
    --log-interval 1
    --recompute
)

echo "=== resume_train_ref (node_rank=$NODE_RANK/$NNODES nodes, $GPUS_PER_NODE GPU/node, DP=$DP_WORLD_SIZE) ===" >&2
echo "  recipe = $RESUME_RECIPE_TOML  iters=$R_ITERS" >&2
echo "  GBS=$R_GBS MBS=$R_MBS seed=$R_SEED lr=$R_LR->$R_MIN_LR warmup=$R_WARMUP" >&2
echo "  product = $_GATE_PRODUCT  OUT_DIR=$OUT_DIR" >&2

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN: torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK $PY_ENTRY ${PY_ARGS[*]}" >&2
    exit 0
fi

exec torchrun \
    --nproc_per_node="$GPUS_PER_NODE" \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "$PY_ENTRY" \
    "${PY_ARGS[@]}"
