#!/bin/bash
# Launch the single-card HF-transformers MiniCPM4-8B mixed-precision trainer.
#
# Independent oracle (not a gate). Sources the Ultra-FineWeb DATA_CONF to build
# the DATA_PATH token string, points the loader at the local data + tokenizer
# mirror, and runs train_minicpm4_8b_hf_singlecard.py on cuda:0.
#
# Required env (export before calling, or rely on the defaults below):
#   FORGE_DATA_DIR        local Ultra-FineWeb parquet mirror (en/ zh/ subdirs)
#   FORGE_TOKENIZER_DIR   dir with tokenizer.model (SentencePiece)
#   OUT_DIR               where to write loss log (default ./_hf_singlecard)
# Pass-through args after `--` go to the python script (e.g. --num-layers 8).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The official MiniCPM4-8B modeling code is fetched from the Hub (config + the
# remote .py only — random init needs no weights). Direct huggingface.co is not
# whitelisted on the devspace proxy; hf-mirror.com is, so route through it.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

export FORGE_DATA_DIR="${FORGE_DATA_DIR:-/opt/forge-data/ultra_fineweb}"
export FORGE_TOKENIZER_DIR="${FORGE_TOKENIZER_DIR:-/opt/forge-data/tokenizer}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/_hf_singlecard}"
DATA_CONF="${DATA_CONF:-$SCRIPT_DIR/ultra_fineweb_data_conf.sh}"
# Data-value SSOT: the loader kind lives in [data].data_loader of this data.toml.
# Standalone default = the ultra_fineweb template; harness-driven runs override
# FORGE_DATA_TOML to the per-loop config dir's data.toml. No DATA_LOADER env value.
FORGE_DATA_TOML="${FORGE_DATA_TOML:-$SCRIPT_DIR/../../config/data/ultra_fineweb.toml}"

mkdir -p "$OUT_DIR"

# DATA_CONF sets DATA_PATH only; the loader kind is read from data.toml
# [data].data_loader via --data-config (no DATA_LOADER env value).
# shellcheck disable=SC1090
source "$DATA_CONF"
DATA_PATH_FILE="$OUT_DIR/data_path.txt"
printf '%s\n' "$DATA_PATH" > "$DATA_PATH_FILE"

echo "  FORGE_DATA_DIR=$FORGE_DATA_DIR"
echo "  FORGE_TOKENIZER_DIR=$FORGE_TOKENIZER_DIR"
echo "  FORGE_DATA_TOML=$FORGE_DATA_TOML"
echo "  OUT_DIR=$OUT_DIR"

cd "$SCRIPT_DIR"
exec python3 train_minicpm4_8b_hf_singlecard.py \
    --data-path-file "$DATA_PATH_FILE" \
    --data-config "$FORGE_DATA_TOML" \
    --out-dir "$OUT_DIR" \
    "$@"
