#!/bin/bash
# Block 4 — env/network launcher for the single-card HF MiniCPM4-8B trainer.
#
# Independent oracle (not a gate). Routes the Hub through hf-mirror.com (the
# devspace whitelist blocks huggingface.co), points the loader at the local
# Ultra-FineWeb + tokenizer mirror, sources the DATA_CONF to build DATA_PATH,
# and runs train.py on cuda:0.
#
# Required env (export before calling, or rely on the defaults below):
#   FORGE_DATA_DIR        local Ultra-FineWeb parquet mirror (en/ zh/ subdirs)
#   FORGE_TOKENIZER_DIR   dir with tokenizer.model (SentencePiece)
#   OUT_DIR               where to write loss log (default ./_out)
# Pass-through args after `--` go to train.py (e.g. --num-layers 16 --lr 1e-3).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Only the config.json + remote modeling .py are fetched (random init, no weights).
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

export FORGE_DATA_DIR="${FORGE_DATA_DIR:-/opt/forge-data/ultra_fineweb}"
export FORGE_TOKENIZER_DIR="${FORGE_TOKENIZER_DIR:-/opt/forge-data/tokenizer}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/_out}"
DATA_CONF="${DATA_CONF:-$SCRIPT_DIR/ultra_fineweb_data_conf.sh}"

mkdir -p "$OUT_DIR"

# DATA_CONF sets DATA_PATH (weighted en/zh parquet globs). data.py reads it
# directly — the whole data path lives in this directory.
# shellcheck disable=SC1090
source "$DATA_CONF"
DATA_PATH_FILE="$OUT_DIR/data_path.txt"
printf '%s\n' "$DATA_PATH" > "$DATA_PATH_FILE"

echo "  FORGE_DATA_DIR=$FORGE_DATA_DIR"
echo "  FORGE_TOKENIZER_DIR=$FORGE_TOKENIZER_DIR"
echo "  OUT_DIR=$OUT_DIR"

cd "$SCRIPT_DIR"
exec python3 train.py \
    --data-path-file "$DATA_PATH_FILE" \
    --out-dir "$OUT_DIR" \
    "$@"
