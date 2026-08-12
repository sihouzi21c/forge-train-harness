#!/bin/bash
# ============================================================================
# WSD-SFT PHASE 3 (sft) DATA_CONF — the supervised-finetune corpus.
#
# The production sft phase feeds a pure SFT blend (align/ifeval-like sources,
# loss_mask active on non-answer tokens; see design_wsd_sft_production_train.md
# §1). This line's stand-in reuses the offline gsm8k Q/A parquet through the
# same streaming HF loader with the gsm8k_qa template — a genuinely
# instruction-shaped corpus, and a DATA_PATH string distinct from BOTH the
# stable and decay confs (the wsd-sft gate's structural verdict requires the
# sft corpus to swap from decay).
#
# Swap in the real production SFT mix by replacing the token below — the
# driver contract is only that this file exports DATA_PATH.
# ============================================================================
set -euo pipefail

DATA_PATH="1.0 hf://openai/gsm8k?config=main&split=train&template=gsm8k_qa&local_env=FORGE_DATA_DIR&glob=train-*.parquet"
export DATA_PATH
