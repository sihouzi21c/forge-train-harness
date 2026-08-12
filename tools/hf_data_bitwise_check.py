#!/usr/bin/env python3
"""Bitwise self-consistency check for the streaming HF dataloader.

For each Stage 1 torch-ref gate *shape* (the (world_size, micro_batch_size,
seq_length, seed) tuple each FORGE_GATE preset pins), build the streaming
dataloader, pull the first N micro-batches per rank, and sha256 each
``(tokens, labels)`` pair. Do this **twice** from fresh loader instances
and assert the hash streams are identical.

This proves the property the user asked for: *same seed → the torch-ref
data is bitwise-identical across two runs* — without needing a GPU or the
model (pure CPU tokenize + pack).

Usage (on a box with ``datasets`` + ``sentencepiece`` + the data + tokenizer)::

    # DATA_PATH defaults to the gsm8k hf token; override to test others.
    GSM8K_DIR=/opt/forge-data/gsm8k \
    TOKENIZER_MODEL=/opt/forge-data/tokenizer/tokenizer.model \
    python3 harness/tools/hf_data_bitwise_check.py [--batches 16]

Exit code 0 iff every shape is bitwise-consistent across the two runs.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

# Make the ref-side loader importable.
_REF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ref",
    "reference",
)
if _REF not in sys.path:
    sys.path.insert(0, _REF)

MAX_SEQ_LEN = 4096  # mirrors model_pure_mup_mtp.MAX_SEQ_LEN / [model].seq_length

# Default DATA_PATH: the gsm8k streaming-HF vehicle (offline via GSM8K_DIR).
_DEFAULT_DATA_PATH = (
    "1.0 hf://openai/gsm8k?config=main&split=train&template=gsm8k_qa"
    "&local_env=GSM8K_DIR&glob=train-*.parquet"
)

# (name, world_size, micro_batch_size, seed) — one row per FORGE_GATE preset.
# All Stage 1 presets pin mbs=4 and seed=1234; world_size is the only mover
# (1 for the *-1gpu / align presets, 2 for the multi-rank trajectory gates).
GATE_SHAPES = [
    ("forward-align", 1, 4, 1234),
    ("backward-align", 1, 4, 1234),
    ("multistep-1gpu", 1, 4, 1234),
    ("multistep", 2, 4, 1234),
    ("perf-bitwise", 2, 4, 1234),
    ("resume-gate-20", 2, 4, 1234),
    ("long-train", 2, 4, 1234),
]


def _parse_data_path(data_path: str):
    raw = data_path.strip().split()
    return [(float(raw[i]), raw[i + 1]) for i in range(0, len(raw), 2)]


def _hash_n_batches(weights_and_paths, world_size, mbs, seed, n):
    """Return {rank: [sha256(batch) for first n batches]} for one fresh build."""
    import hf_stream_dataloader

    out = {}
    for rank in range(world_size):
        dl = hf_stream_dataloader.build(
            weights_and_paths,
            dp_rank=rank,
            world_size=world_size,
            micro_batch_size=mbs,
            seq_length=MAX_SEQ_LEN,
            seed=seed,
        )
        hashes = []
        for _ in range(n):
            batch = next(dl)
            h = hashlib.sha256()
            h.update(batch["tokens"].numpy().tobytes())
            h.update(batch["labels"].numpy().tobytes())
            hashes.append(h.hexdigest())
        out[rank] = hashes
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--batches", type=int, default=16, help="micro-batches to sample per (shape, rank)"
    )
    ap.add_argument("--data-path", default=os.environ.get("DATA_PATH", _DEFAULT_DATA_PATH))
    args = ap.parse_args()

    weights_and_paths = _parse_data_path(args.data_path)
    print(f"DATA_PATH sources: {weights_and_paths}")
    print(f"sampling {args.batches} micro-batches per (shape, rank)\n")

    all_ok = True
    for name, world, mbs, seed in GATE_SHAPES:
        run1 = _hash_n_batches(weights_and_paths, world, mbs, seed, args.batches)
        run2 = _hash_n_batches(weights_and_paths, world, mbs, seed, args.batches)
        ok = run1 == run2
        all_ok &= ok
        # Per-rank first-hash fingerprint for the log.
        fp = ",".join(run1[r][0][:12] for r in sorted(run1))
        print(
            f"[{'PASS' if ok else 'FAIL'}] {name:<16} world={world} mbs={mbs} "
            f"seed={seed}  rank0..N first-hash={fp}"
        )
        if not ok:
            for r in sorted(run1):
                if run1[r] != run2[r]:
                    bad = next(i for i in range(len(run1[r])) if run1[r][i] != run2[r][i])
                    print(
                        f"    rank={r} diverges at batch {bad}: "
                        f"{run1[r][bad][:16]} != {run2[r][bad][:16]}"
                    )

    print()
    print("RESULT:", "ALL SHAPES BITWISE-CONSISTENT" if all_ok else "MISMATCH DETECTED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
