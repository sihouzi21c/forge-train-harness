"""Block 2 — data: weighted local Ultra-FineWeb parquet stream (single card).

Self-contained: depends only on this file plus external libraries (datasets,
sentencepiece, numpy, torch) — no project import from outside this directory.

Pipeline (the only path a single-card loss-trend oracle needs):
  open local parquet (frozen sorted file list, per corpus)
    → interleave_datasets(probabilities=weights)        # en/zh mix
    → shuffle(seed, buffer)
    → SentencePiece encode(text) + [eos], no BOS, skip empty docs
    → pack into fixed (mbs, seq+1) windows
    → {tokens, labels, loss_mask}, the shape the gate refs consume.

Deterministic given seed: sorted file list fixes document order; interleave and
shuffle are seeded (seed+epoch); single-threaded (no num_workers reordering).
"""
from __future__ import annotations

import glob
import itertools
import os
from pathlib import Path

import numpy as np
import torch


def _load_tokenizer():
    """SentencePiece tokenizer from FORGE_TOKENIZER_DIR/tokenizer.model."""
    d = os.environ.get("FORGE_TOKENIZER_DIR", "").strip()
    if not d:
        raise ValueError(
            "FORGE_TOKENIZER_DIR not set (dir holding tokenizer.model). "
            "run.sh exports it; configure [ref].forge_tokenizer_dir.")
    model = Path(d) / "tokenizer.model"
    if not model.is_file():
        raise FileNotFoundError(f"no tokenizer.model under FORGE_TOKENIZER_DIR={d}")
    import sentencepiece
    return sentencepiece.SentencePieceProcessor(model_file=str(model))


def _open_corpus(pattern):
    """Stream one local parquet corpus; `pattern` is a glob under FORGE_DATA_DIR.

    The sorted file list freezes document order → reproducible across runs.
    """
    from datasets import load_dataset

    root = os.environ.get("FORGE_DATA_DIR", "").strip()
    if not root or not os.path.isdir(root):
        raise FileNotFoundError(
            f"FORGE_DATA_DIR must point at the local Ultra-FineWeb mirror; got {root!r}")
    files = sorted(glob.glob(os.path.join(root, pattern)))
    if not files:
        raise FileNotFoundError(f"no parquet matches {pattern!r} under {root}")
    return load_dataset("parquet", data_files=files, split="train", streaming=True)


class UltraFineWebLoader:
    """Infinite, deterministic streaming packer over weighted local corpora."""

    def __init__(self, weights_and_globs, micro_batch_size, seq_length, seed,
                 text_col="content", buffer_size=10000):
        self._sources = weights_and_globs  # [(weight, glob), ...]
        self._mbs = int(micro_batch_size)
        self._seq_length = int(seq_length)
        self._seed = int(seed)
        self._text_col = text_col
        self._buffer_size = int(buffer_size)

        self._tok = _load_tokenizer()
        self._eos = self._tok.eos_id()
        if self._eos is None or self._eos < 0:
            raise ValueError(f"tokenizer eos_id invalid: {self._eos!r}")
        self._gen = self._batches()

    def _open_mixed(self, epoch):
        from datasets import interleave_datasets

        streams = [_open_corpus(g) for _, g in self._sources]
        if len(streams) == 1:
            mixed = streams[0]
        else:
            total = sum(w for w, _ in self._sources)
            probs = [w / total for w, _ in self._sources]
            mixed = interleave_datasets(
                streams, probabilities=probs, seed=self._seed + epoch,
                stopping_strategy="all_exhausted",
            )
        return mixed.shuffle(seed=self._seed + epoch, buffer_size=self._buffer_size)

    def _token_stream(self):
        """Infinite token stream; re-open with a new epoch seed when drained."""
        epoch = 0
        while True:
            produced = False
            for row in self._open_mixed(epoch):
                text = str(row.get(self._text_col, "") or "")
                ids = list(self._tok.encode(text))
                if not ids:  # skip empty docs
                    continue
                produced = True
                yield from ids
                yield self._eos
            if not produced:
                raise RuntimeError(
                    "data source produced no non-empty documents — check the "
                    f"text column {self._text_col!r} and the parquet files")
            epoch += 1

    def _batches(self):
        stream = self._token_stream()
        win = self._seq_length + 1
        while True:
            rows = []
            for _ in range(self._mbs):
                chunk = list(itertools.islice(stream, win))
                if len(chunk) < win:  # stream is infinite; only on empty corpus
                    return
                rows.append(chunk)
            arr = np.asarray(rows, dtype=np.int64)  # (mbs, win)
            yield {
                "tokens": torch.from_numpy(arr[:, :-1].copy()),
                "labels": torch.from_numpy(arr[:, 1:].copy()),
                "loss_mask": torch.ones(
                    (self._mbs, self._seq_length), dtype=torch.float32),
            }

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._gen)

    def close(self):
        """Release the streaming generator + pyarrow readers before finalize.

        Drop the parquet readers now, while CUDA/UCX are healthy: a pyarrow
        buffer collected during the post-main() finalize GC races the UCX
        malloc-hook unload on the HPCX image → SIGABRT / hang. Shrinking the
        pyarrow I/O pool + an explicit collect drains in-flight futures here.
        Idempotent; teardown-only, no effect on iteration order or numerics.
        """
        gen, self._gen = getattr(self, "_gen", None), None
        if gen is not None:
            try:
                gen.close()
            except Exception:
                pass
        try:
            import pyarrow as pa
            pa.set_io_thread_count(1)
            pa.set_cpu_count(1)
        except Exception:
            pass
        import gc
        gc.collect()


def build_stream(data_path_file, micro_batch_size, seq_length, seed):
    """Parse '<weight> <glob> <weight> <glob> …' (globs under FORGE_DATA_DIR)."""
    with open(data_path_file) as f:
        raw = f.read().strip().split()
    weights_and_globs = [
        (float(raw[i]), raw[i + 1]) for i in range(0, len(raw), 2)
    ]
    return UltraFineWebLoader(
        weights_and_globs, micro_batch_size=micro_batch_size,
        seq_length=seq_length, seed=seed,
    )
