#!/bin/bash
# ============================================================================
# WSD-SFT PHASE 1 (stable) DATA_CONF — the pretraining mix.
#
# One of the three per-phase corpora the 3-phase driver
# (evals/scripts/train_ours_al.sh) sources, one per phase. The wsd-sft gate's
# structural verdict asserts the DATA_PATH string DIFFERS across all three
# phases (corpus swap is a gate-checked switch fact), so these three confs
# must never collapse to the same DATA_PATH.
#
# stable = the standard Ultra-FineWeb pretraining mix — identical recipe to
# ultra_fineweb_data_conf.sh (60% en / 30% zh with the code slice dropped,
# normalized to 2:1 inside the loader). Kept as its own file so the phase →
# corpus mapping is one basename per phase and swapping in the real
# production stable mix later means editing THIS file only.
# ============================================================================
set -euo pipefail

REVISION="${ULTRA_FINEWEB_REVISION:-}"
REV_Q=""
[[ -n "$REVISION" ]] && REV_Q="&revision=${REVISION}"

EN="hf://openbmb/Ultra-FineWeb?split=en&text=content&local_env=FORGE_DATA_DIR&glob=en/*.parquet${REV_Q}"
ZH="hf://openbmb/Ultra-FineWeb?split=zh&text=content&local_env=FORGE_DATA_DIR&glob=zh/*.parquet${REV_Q}"

DATA_PATH="0.667 ${EN} 0.333 ${ZH}"
export DATA_PATH
