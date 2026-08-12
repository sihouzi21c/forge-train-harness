#!/bin/bash
# ============================================================================
# Bridge between the gsm8k Megatron-binary preparator and the pure-torch
# ref's DATA_CONF contract. Used by the torch backend
# (``config/ref/torch_minicpm4_0.5b.toml``).
#
# The pure-PyTorch L0 ref (``run_16gpu_1000step_pure_mup_mtp.sh`` +
# ``train_pure_mup_mtp.py``) takes a weighted-shard string of the form
# ``"<weight> <path> [<weight> <path> ...]"`` via the ``DATA_CONF`` →
# materialized ``--data-path-file`` pipeline.
#
# The gsm8k preparator (``prepare_gsm8k_data.sh``) emits a single
# Megatron binary prefix ``<OUTPUT_DIR>/gsm8k_train_text_document``
# (.bin / .idx pair). This script wraps that prefix as a one-shard
# weighted list (weight = 1.0) only to satisfy the unified
# ``--data-path-file`` CLI contract; it does NOT imply the file is
# read by ``modelbest_sdk``.
#
# Runtime dispatch (see ``train_pure_mup_mtp.py::build_dataloader``):
# the single-shard + ``.bin``/``.idx`` case is detected and routed to
# the in-file ``MegatronBinaryDataloader``, bypassing modelbest_sdk
# entirely. Multi-shard lists (e.g. the production sstable_251023
# entries in ``minicpm4_0.5.stable.sh``) are the only path that
# reaches ``ModelbestDataloader``. Net effect under this conf: the
# torch ref consumes the same physical ``.bin``/``.idx`` file as the
# Megatron sibling (``config/ref/megatron_minicpm4_0.5b.toml`` →
# ``train_minicpm4_0.5b_gsm8k.sh``), via two independent readers.
#
# Auto-prep: if the ``.idx`` sibling is missing — or is stale relative to a
# configured ``GSM8K_DIR`` parquet source (sequence_count != parquet rows, e.g.
# a hand-synthesized placeholder) — and the prep env contract is satisfied
# (``GSM8K_DIR`` + ``TOKENIZER_MODEL``), this script invokes
# the Megatron-free torch preparator ``gsm8k_prepare_torch.py`` (pure
# SentencePiece + numpy/struct, NO ``MEGATRON_ROOT`` dependency). Its
# output is bitwise-identical to the Megatron ``prepare_gsm8k_data.sh``
# (verified: same ``.bin``/``.idx`` sha256). When the env is incomplete
# the auto-prep is skipped silently and a downstream "file not found"
# error from the torch ref points the user at the manual prep step.
#
# Override ``GSM8K_DATA_PREFIX`` to point at a non-default binary
# location (e.g. a shared cluster scratch dir).
# ============================================================================
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FORGE_REPO_ROOT_GUESS="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Same default output prefix as prepare_gsm8k_data.sh's OUTPUT_DIR
# auto-derive, so the two scripts agree on where the binary lives
# without a separate SSOT for the path.
GSM8K_DATA_PREFIX="${GSM8K_DATA_PREFIX:-${FORGE_REPO_ROOT_GUESS}/.artifacts/data/gsm8k_megatron/gsm8k_train_text_document}"

# Decide whether the binary needs (re)building. Rebuild when the .idx is
# missing OR — when a real parquet source is configured via GSM8K_DIR — when
# the existing .idx does not match it (a stale or hand-synthesized placeholder
# .idx must not silently shadow the baked dataset). The match test compares
# the .idx ``sequence_count`` field (int64 LE at byte offset 18) against the
# parquet row count: one non-empty document per row, so they are equal for a
# faithfully prepared gsm8k binary (the stored ``document_count`` is n_seq+1,
# so we read sequence_count, not document_count). No GSM8K_DIR / no parquet →
# keep the legacy "build only if missing" behaviour untouched.
need_prep=0
if [[ ! -f "${GSM8K_DATA_PREFIX}.idx" ]]; then
    need_prep=1
elif [[ -n "${GSM8K_DIR:-}" ]]; then
    if ! python3 - "$GSM8K_DIR" "${GSM8K_DATA_PREFIX}.idx" <<'PY'
import sys, glob, struct

gsm8k_dir, idx = sys.argv[1], sys.argv[2]
parqs = sorted(glob.glob(f"{gsm8k_dir}/train-*.parquet"))
if not parqs:
    sys.exit(0)  # no real parquet source → don't second-guess the existing idx

import pyarrow.parquet as pq

rows = sum(pq.ParquetFile(p).metadata.num_rows for p in parqs)
with open(idx, "rb") as f:
    head = f.read(26)
if len(head) < 26 or head[:9] != b"MMIDIDX\x00\x00":
    sys.exit(1)  # foreign / corrupt / placeholder idx → rebuild
n_seq = struct.unpack("<q", head[18:26])[0]  # sequence_count (== gsm8k row count)
sys.exit(0 if n_seq == rows else 1)
PY
    then
        need_prep=1
    fi
fi

if [[ "$need_prep" == 1 ]]; then
    PREP_SCRIPT="${SCRIPT_DIR}/gsm8k_prepare_torch.py"
    if [[ -n "${GSM8K_DIR:-}" && -n "${TOKENIZER_MODEL:-}" && -f "$PREP_SCRIPT" ]]; then
        OUTPUT_DIR="$(dirname "$GSM8K_DATA_PREFIX")"
        export OUTPUT_DIR
        python3 "$PREP_SCRIPT" >&2
    fi
fi

# Weighted-shard format consumed by run_16gpu_1000step_pure_mup_mtp.sh:
# whitespace-separated "weight path" pairs. Single shard → weight 1.0.
DATA_PATH="1.0 ${GSM8K_DATA_PREFIX}"
export DATA_PATH
