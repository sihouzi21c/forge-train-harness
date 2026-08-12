#!/bin/bash
# ============================================================================
# DATA_CONF for the single-card Ultra-FineWeb oracle.
#
# DATA_PATH is "<weight> <glob> <weight> <glob> …" — each glob is a parquet
# pattern under FORGE_DATA_DIR (the local Ultra-FineWeb mirror). data.py reads
# it directly; weights are normalized inside the loader.
#
# en/zh = 0.667/0.333 ≈ 2:1, the authors' 60/30 recipe with the code slice
# dropped.
# ============================================================================
set -euo pipefail

DATA_PATH="0.667 en/*.parquet 0.333 zh/*.parquet"
export DATA_PATH
