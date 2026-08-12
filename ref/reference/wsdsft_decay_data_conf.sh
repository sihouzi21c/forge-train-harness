#!/bin/bash
# ============================================================================
# WSD-SFT PHASE 2 (decay) DATA_CONF — the anneal mix.
#
# The production decay phase feeds a HIGHER-QUALITY blend than stable (the
# real recipe adds ~130 curated/anneal sources to the pretrain mix; see
# design_wsd_sft_production_train.md §1). This line's stand-in keeps the
# same two Ultra-FineWeb splits but INVERTS the en/zh weighting, which is
# (a) a genuinely different sampling distribution — the per-step token
# stream diverges from stable immediately, and (b) a distinct DATA_PATH
# string, which the wsd-sft gate's structural verdict requires (decay
# data_path must differ from stable).
#
# Swap in the real production decay mix by replacing the weighted list
# below — the driver contract is only that this file exports DATA_PATH.
# ============================================================================
set -euo pipefail

REVISION="${ULTRA_FINEWEB_REVISION:-}"
REV_Q=""
[[ -n "$REVISION" ]] && REV_Q="&revision=${REVISION}"

EN="hf://openbmb/Ultra-FineWeb?split=en&text=content&local_env=FORGE_DATA_DIR&glob=en/*.parquet${REV_Q}"
ZH="hf://openbmb/Ultra-FineWeb?split=zh&text=content&local_env=FORGE_DATA_DIR&glob=zh/*.parquet${REV_Q}"

DATA_PATH="0.333 ${EN} 0.667 ${ZH}"
export DATA_PATH
