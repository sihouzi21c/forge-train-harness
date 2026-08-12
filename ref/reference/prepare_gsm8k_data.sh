#!/bin/bash
# ============================================================================
# Preprocess gsm8k (HuggingFace parquet) → Megatron binary format
#
# Usage:
#   bash ref/reference/prepare_gsm8k_data.sh
#
# Required env (no defaults — set per-machine via shell, harness CLI,
# or ``config/ref.toml`` / ``config/data.toml``):
#   GSM8K_DIR          HuggingFace gsm8k dataset dir (with train-*.parquet)
#   TOKENIZER_MODEL    Path to SentencePiece tokenizer.model
#   MEGATRON_ROOT      Megatron-LM source root (provides tools/preprocess_data.py)
#
# Optional env (auto-derived if unset):
#   OUTPUT_DIR         Where to write processed data
#                      (default: <repo>/.artifacts/data/gsm8k_megatron;
#                       gitignored under .artifacts/).
#
# After this script finishes, the printed ``DATA_PATH`` value is what
# ``config/eval.toml [defaults].data_path`` (or
# ``FORGE_DATA_PATH`` / ``--data-path``) should point at.
# ============================================================================
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FORGE_REPO_ROOT_GUESS="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Required user-supplied env. Empty defaults + ``${VAR:?…}`` fail-fast
# so a fresh-machine invocation surfaces the missing input by name
# instead of silently falling back to an author's home directory.
: "${GSM8K_DIR:?GSM8K_DIR is required (HuggingFace gsm8k parquet root, e.g. .../gsm8k/main)}"
: "${TOKENIZER_MODEL:?TOKENIZER_MODEL is required (SentencePiece tokenizer.model path; same value as dense_training.toml [defaults].tokenizer_model)}"
: "${MEGATRON_ROOT:?MEGATRON_ROOT is required (Megatron-LM source root; same value as dense_training.toml [defaults].megatron_root)}"

OUTPUT_DIR="${OUTPUT_DIR:-${FORGE_REPO_ROOT_GUESS}/.artifacts/data/gsm8k_megatron}"

JSONL_PATH="$OUTPUT_DIR/gsm8k_train.jsonl"
OUTPUT_PREFIX="$OUTPUT_DIR/gsm8k_train"

mkdir -p "$OUTPUT_DIR"

echo "=== Step 1: Convert gsm8k parquet → JSONL ==="
python3 -c "
import pandas as pd, json, sys

df = pd.read_parquet('${GSM8K_DIR}/train-00000-of-00001.parquet')
out_path = '${JSONL_PATH}'
count = 0
with open(out_path, 'w', encoding='utf-8') as f:
    for _, row in df.iterrows():
        text = f\"Question: {row['question']}\nAnswer: {row['answer']}\"
        f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
        count += 1
print(f'Wrote {count} samples to {out_path}')
"

echo "=== Step 2: Tokenize with Megatron preprocess_data.py ==="
cd "$MEGATRON_ROOT"

python3 tools/preprocess_data.py \
    --input "$JSONL_PATH" \
    --output-prefix "$OUTPUT_PREFIX" \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model "$TOKENIZER_MODEL" \
    --append-eod \
    --workers 4

echo "=== Done ==="
echo "Output files:"
ls -lh "$OUTPUT_DIR"/gsm8k_train*
echo ""
echo "Use in ref script:  DATA_PATH=$OUTPUT_PREFIX"
