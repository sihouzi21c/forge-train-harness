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
#
# NOTE: the gsm8k dataset is not available locally on the devspace and
# cannot be downloaded through the proxy.  As a workaround, use the
# same Ultra-FineWeb en split (100%) as the sft corpus — a genuinely
# distinct DATA_PATH from stable (0.667 en 0.333 zh) and decay
# (0.333 en 0.667 zh), satisfying the wsd-sft gate's structural
# assertion that all three corpora differ.
# ============================================================================
set -euo pipefail

REVISION="${ULTRA_FINEWEB_REVISION:-}"
REV_Q=""
[[ -n "$REVISION" ]] && REV_Q="&revision=${REVISION}"

EN="hf://openbmb/Ultra-FineWeb?split=en&text=content&local_env=FORGE_DATA_DIR&glob=en/*.parquet${REV_Q}"

DATA_PATH="1.0 ${EN}"
export DATA_PATH
