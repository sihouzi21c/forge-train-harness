#!/bin/bash
# ============================================================================
# DATA_CONF for the generic streaming HuggingFace dataloader, pointed at
# gsm8k — the small/offline dev+CI vehicle that exercises the SAME loader
# code path as Ultra-FineWeb (ingestion adapter + tokenize + mix + stream).
#
# Single weighted source. ``template=gsm8k_qa`` reproduces the reference
# document layout ("Question: {q}\nAnswer: {a}"). ``local_env=FORGE_DATA_DIR``
# streams offline from the baked parquet (``train-*.parquet`` under
# FORGE_DATA_DIR — the layout the Dockerfile bake produces); falls back
# to the Hub (openai/gsm8k, config main) only if FORGE_DATA_DIR is unset.
#
# Lets every M1–M6 torch-ref gate run on tiny baked data through the exact
# streaming loader that, repointed via ultra_fineweb_data_conf.sh, does
# real pretraining.
# ============================================================================
set -euo pipefail

DATA_PATH="1.0 hf://openai/gsm8k?config=main&split=train&template=gsm8k_qa&local_env=FORGE_DATA_DIR&glob=train-*.parquet"
export DATA_PATH
