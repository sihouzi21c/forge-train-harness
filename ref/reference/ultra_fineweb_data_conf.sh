#!/bin/bash
# ============================================================================
# DATA_CONF for the generic streaming HuggingFace dataloader, pointed at
# Ultra-FineWeb (openbmb/Ultra-FineWeb) — the real-pretraining target.
#
# Emits a weighted two-source DATA_PATH (en + zh) consumed by
# ``train_pure_mup_mtp.py::build_dataloader`` → ``hf_stream_dataloader``.
# Weights follow the authors' recipe with the code slice dropped:
# 60% en / 30% zh ≈ 2:1 (normalized inside the loader).
#
# Each source is an ``hf://`` token (no spaces; query-string params):
#   text=content   → Ultra-FineWeb text column
#   local_env=FORGE_DATA_DIR → if that env points at a local parquet mirror,
#                    stream offline from it (frozen sorted file list) instead
#                    of the Hub. The harness exports FORGE_DATA_DIR from
#                    [data].forge_data_dir.
#   glob=en/*.parquet / zh/*.parquet → per-split file pattern under the mirror
#
# To stream from the Hub instead of a mirror, leave FORGE_DATA_DIR unset
# (requires network + the dataset reachable). Pin a revision for
# reproducibility by appending ``&revision=<sha>`` to each token.
# ============================================================================
set -euo pipefail

REVISION="${ULTRA_FINEWEB_REVISION:-}"
REV_Q=""
[[ -n "$REVISION" ]] && REV_Q="&revision=${REVISION}"

EN="hf://openbmb/Ultra-FineWeb?split=en&text=content&local_env=FORGE_DATA_DIR&glob=en/*.parquet${REV_Q}"
ZH="hf://openbmb/Ultra-FineWeb?split=zh&text=content&local_env=FORGE_DATA_DIR&glob=zh/*.parquet${REV_Q}"

DATA_PATH="0.667 ${EN} 0.333 ${ZH}"
export DATA_PATH
