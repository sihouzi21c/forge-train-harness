"""Generic streaming HuggingFace ``datasets`` dataloader (torch ref side).

Real-pretraining-grade data path for the pure-torch L0 ref
(``train_pure_mup_mtp.py``). One universal pipeline —
**tokenize + mix + stream** — driven by a per-dataset *ingestion
adapter* encoded in the ``DATA_PATH`` shard string. Adding a new dataset
is one ``hf://`` token, never a new dataloader.

Factoring (only the first item is per-dataset; the rest are universal):

    ingestion   per-dataset  -> which HF repo/split + which column / text template
    tokenize    universal    -> SentencePiece encode(text)+[eos], no BOS, skip empty
                               (the exact contract of gsm8k_prepare_torch.encode_documents)
    mix         universal    -> datasets.interleave_datasets(probabilities=weights)
    stream      universal    -> .shuffle(buffer_size) + split_dataset_by_node + pack

``DATA_PATH`` token grammar (whitespace-separated ``weight token`` pairs,
each token a query-string URI with NO spaces)::

    hf://<repo>?split=<sp>&text=<col>[&config=<cfg>][&local_env=<ENV>][&glob=<pat>][&revision=<sha>]
    hf://<repo>?split=<sp>&template=<name>[&config=<cfg>][&local_env=<ENV>][&glob=<pat>]

* ``text=<col>``     row -> ``str(row[col])``                 (Ultra-FineWeb: ``content``)
* ``template=<name>`` row -> named template (e.g. ``gsm8k_qa``) from ``_TEMPLATES``
* ``local_env=<ENV>`` if ``os.environ[ENV]`` is a non-empty dir, stream local
  parquet (``load_dataset("parquet", data_files=sorted(glob(root/glob)), streaming=True)``)
  instead of the Hub — keeps the harness offline-by-default. ``glob`` defaults
  to ``*.parquet``.
* ``revision=<sha>`` pins the Hub revision (only used on the network path).

Determinism contract (what the bitwise self-consistency check relies on):
given the same ``seed`` and the same physical files, two independent runs
of this loader produce **bitwise-identical** ``tokens``/``labels`` per
``(rank, micro-batch index)`` — because every stage is seeded
(``interleave``/``shuffle`` seed = ``seed + epoch``), single-threaded
(no ``num_workers`` reordering), and the document stream order is fixed by
a frozen sorted file list (local) or pinned revision (Hub).

Batch dict contract is identical to ``MegatronBinaryDataloader``:
``{"tokens": LongTensor(mbs, S), "labels": LongTensor(mbs, S),
"loss_mask": FloatTensor(mbs, S)}`` and no ``indexes``/``last_sample`` keys
(so ``train_pure_mup_mtp.next_batch``'s ``dl.update`` hook is skipped).

NOTE: this module lives on the **reference** side (allowlisted in
``framework_guard``) and may freely import ``datasets`` / ``sentencepiece``.
The ours-side engine has a separate, framework-native equivalent (no
``torch.utils.data``) — out of scope here.
"""

from __future__ import annotations

import hashlib
import itertools
import os
import sys
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

# Reuse the exact tokenizer load + text layout from the gsm8k preparator so
# the encode contract is a single SSOT (SentencePiece, encode+[eos], no BOS).
try:  # normal: ref/reference is on sys.path (sibling import)
    from gsm8k_prepare_torch import _load_tokenizer, format_gsm8k_text
except Exception:  # pragma: no cover - test/packaging fallback
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gsm8k_prepare_torch import _load_tokenizer, format_gsm8k_text


# ── Per-dataset ingestion adapter table ─────────────────────────────────
# Named text templates. Each maps a dataset row -> a single training string.
# Keep tiny and table-driven: a new dataset that needs special text layout
# adds ONE entry here, not a new loader.
_TEMPLATES = {
    # gsm8k: reproduce the reference parquet->jsonl layout exactly.
    "gsm8k_qa": lambda row: format_gsm8k_text(row["question"], row["answer"]),
}


@dataclass
class SourceSpec:
    """One weighted ingestion source parsed from a single ``hf://`` token."""

    weight: float
    repo: str
    split: str
    config: str | None = None
    text_col: str | None = None
    template: str | None = None
    local_env: str | None = None
    glob: str = "*.parquet"
    revision: str | None = None

    def text_of(self, row: dict) -> str:
        if self.template is not None:
            fn = _TEMPLATES.get(self.template)
            if fn is None:
                raise KeyError(
                    f"unknown text template {self.template!r}; "
                    f"known: {sorted(_TEMPLATES)}"
                )
            return fn(row)
        if self.text_col is not None:
            return str(row.get(self.text_col, "") or "")
        raise ValueError(
            f"source {self.repo} has neither text= nor template= in its hf:// token"
        )


def _parse_hf_token(weight: float, token: str) -> SourceSpec:
    """Parse ``hf://<repo>?k=v&...`` into a :class:`SourceSpec`."""
    if not token.startswith("hf://"):
        raise ValueError(f"not an hf:// token: {token!r}")
    parsed = urlparse(token)
    # urlparse("hf://openai/gsm8k?...") -> netloc="openai", path="/gsm8k"
    repo = (parsed.netloc + parsed.path).rstrip("/")
    q = {k: v[0] for k, v in parse_qs(parsed.query).items()}
    split = q.get("split")
    if not split:
        raise ValueError(f"hf:// token missing required split=: {token!r}")
    return SourceSpec(
        weight=weight,
        repo=repo,
        split=split,
        config=q.get("config"),
        text_col=q.get("text"),
        template=q.get("template"),
        local_env=q.get("local_env"),
        glob=q.get("glob", "*.parquet"),
        revision=q.get("revision"),
    )


def is_hf_data_path(weights_and_paths) -> bool:
    """True iff any shard path is an ``hf://`` token (routes to this loader)."""
    return any(str(p).startswith("hf://") for _, p in weights_and_paths)


def _open_stream(spec: SourceSpec):
    """Open one source as a streaming ``IterableDataset`` (local or Hub)."""
    from datasets import load_dataset

    local_root = os.environ.get(spec.local_env, "") if spec.local_env else ""
    if local_root and os.path.isdir(local_root):
        import glob as _glob

        files = sorted(_glob.glob(os.path.join(local_root, spec.glob)))
        if not files:
            raise FileNotFoundError(
                f"{spec.local_env}={local_root} set but no files match "
                f"{spec.glob!r} under it"
            )
        # Frozen sorted file list -> deterministic document order.
        return load_dataset(
            "parquet", data_files=files, split="train", streaming=True,
        )
    # Network path (offline-by-default harness: requires the data to be
    # reachable; pin revision for reproducibility).
    return load_dataset(
        spec.repo, name=spec.config, split=spec.split,
        streaming=True, revision=spec.revision,
    )


class HFStreamDataloader:
    """Infinite, deterministic streaming packer over weighted HF sources."""

    def __init__(self, specs, dp_rank, world_size, micro_batch_size,
                 seq_length, seed, buffer_size=10000):
        self._specs = specs
        self._dp_rank = int(dp_rank)
        self._world_size = int(world_size)
        self._mbs = int(micro_batch_size)
        self._seq_length = int(seq_length)
        self._seed = int(seed)
        self._buffer_size = int(buffer_size)

        self._tokenizer = _load_tokenizer(_resolve_tokenizer_file(_require_tokenizer_dir()))
        self._eos_id = self._tokenizer.eos_id()
        if self._eos_id is None or self._eos_id < 0:
            raise ValueError(f"tokenizer eos_id invalid: {self._eos_id!r}")

        # Optional bitwise self-check hook: when set, emit a per-batch hash.
        self._dump = os.environ.get("FORGE_HF_DATA_DUMP", "")
        self._batch_idx = 0
        self._gen = self._batches()

    # ── streaming pipeline (mix -> shuffle -> shard) ────────────────────
    def _open_mixed(self, epoch: int):
        from datasets import interleave_datasets
        from datasets.distributed import split_dataset_by_node

        streams = [_open_stream(s) for s in self._specs]
        if len(streams) == 1:
            mixed = streams[0]
        else:
            total = sum(s.weight for s in self._specs)
            probs = [s.weight / total for s in self._specs]
            mixed = interleave_datasets(
                streams, probabilities=probs, seed=self._seed + epoch,
                stopping_strategy="all_exhausted",
            )
        mixed = mixed.shuffle(seed=self._seed + epoch, buffer_size=self._buffer_size)
        if self._world_size > 1:
            mixed = split_dataset_by_node(
                mixed, rank=self._dp_rank, world_size=self._world_size,
            )
        return mixed

    def _token_stream(self):
        """Infinite token stream: re-open with a new epoch seed when drained."""
        epoch = 0
        while True:
            mixed = self._open_mixed(epoch)
            produced = False
            for row in mixed:
                ids = list(self._tokenizer.encode(self._specs_text(row, epoch)))
                if not ids:  # skip empty docs (gsm8k_prepare_torch contract)
                    continue
                produced = True
                yield from ids
                yield self._eos_id
            if not produced:
                raise RuntimeError(
                    "HF data source produced no non-empty documents — "
                    "check the text=/template= column and the data files"
                )
            epoch += 1

    def _specs_text(self, row, epoch):
        # With a single interleaved stream we cannot know which source a row
        # came from, so all sources must agree on how to extract text. We use
        # the first spec whose extractor succeeds (templates/columns are
        # per-corpus and mutually exclusive in practice).
        for spec in self._specs:
            try:
                return spec.text_of(row)
            except Exception:
                continue
        # Fall back to the first spec's extractor to surface a clear error.
        return self._specs[0].text_of(row)

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
            tokens = torch.from_numpy(arr[:, :-1].copy())
            labels = torch.from_numpy(arr[:, 1:].copy())
            loss_mask = torch.ones((self._mbs, self._seq_length), dtype=torch.float32)
            if self._dump:
                self._emit_hash(tokens, labels)
            self._batch_idx += 1
            yield {"tokens": tokens, "labels": labels, "loss_mask": loss_mask}

    def _emit_hash(self, tokens, labels):
        h = hashlib.sha256()
        h.update(tokens.numpy().tobytes())
        h.update(labels.numpy().tobytes())
        print(
            f"[DATAHASH] rank={self._dp_rank} batch={self._batch_idx} "
            f"sha256={h.hexdigest()}",
            flush=True,
        )

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._gen)

    def close(self):
        """Release the streaming generator chain (HF iterators + pyarrow readers).

        Called from the ref's teardown while CUDA / UCX are still healthy so the
        pyarrow parquet readers are reclaimed *now*, not during ``Py_Finalize``.
        On the HPCX/UCX image (``/opt/hpcx/ucx/lib/libucm.so`` is loaded into the
        process), a pyarrow buffer collected during the post-``main()`` finalize
        GC pass has its C++ destructor race the UCX malloc-hook unload, producing
        ``terminate called without an active exception`` → SIGABRT (returncode
        -6) — or, depending on thread timing, a finalize hang. Closing the
        generator here unwinds ``_batches`` → ``_token_stream`` → the HF
        ``IterableDataset`` iterators so those readers drop before finalize.
        Idempotent; teardown-only, no effect on iteration order or numerics.
        """
        gen = getattr(self, "_gen", None)
        self._gen = None
        if gen is not None:
            try:
                gen.close()
            except Exception:
                pass
        # ``gen.close()`` drops the parquet readers from Python land, but
        # pyarrow keeps a process-wide I/O ThreadPool whose worker may still
        # be mid ``arrow::Future::SetResult`` against a libarrow_python
        # binding — if the main thread races interpreter shutdown, that
        # worker fires the callback against a destroyed C++ object and
        # segfaults at offset 0x60 (NULL+field). Shrinking the pool to 1
        # thread + an explicit ``gc.collect`` synchronously drains in-flight
        # futures and reclaims the buffers here, while the GIL is still
        # well-behaved.
        try:
            import pyarrow as pa
            pa.set_io_thread_count(1)
            pa.set_cpu_count(1)
        except Exception:
            pass
        try:
            import gc
            gc.collect()
        except Exception:
            pass


def _require_tokenizer_dir() -> str:
    d = os.environ.get("FORGE_TOKENIZER_DIR", "").strip()
    if not d:
        raise ValueError(
            "FORGE_TOKENIZER_DIR not set. The harness exports this from "
            "[ref].forge_tokenizer_dir; configure that field in config/ref.toml."
        )
    return d


def _resolve_tokenizer_file(tokenizer_dir: str) -> str:
    """Pick the tokenizer artifact inside FORGE_TOKENIZER_DIR.

    ``tokenizer.model`` (SentencePiece) wins over ``tokenizer.json`` (HF
    fast) when both are present — same priority order as the legacy
    .resources fallback the rest of the code used.
    """
    from pathlib import Path as _Path
    d = _Path(tokenizer_dir)
    for basename in ("tokenizer.model", "tokenizer.json"):
        p = d / basename
        if p.is_file():
            return str(p)
    raise FileNotFoundError(
        f"no tokenizer.model or tokenizer.json under FORGE_TOKENIZER_DIR={tokenizer_dir}"
    )


def build(weights_and_paths, dp_rank, world_size,
          micro_batch_size, seq_length, seed, buffer_size=None):
    """Construct an :class:`HFStreamDataloader` from parsed shard tuples.

    ``weights_and_paths`` is the same ``[(weight, path), ...]`` list
    ``train_pure_mup_mtp.build_dataloader`` already parses from
    ``--data-path-file``; here every ``path`` is an ``hf://`` token.
    """
    specs = [_parse_hf_token(float(w), str(p)) for w, p in weights_and_paths]
    if buffer_size is None:
        buffer_size = int(os.environ.get("FORGE_HF_SHUFFLE_BUFFER", "10000"))
    return HFStreamDataloader(
        specs, dp_rank=dp_rank, world_size=world_size,
        micro_batch_size=micro_batch_size, seq_length=seq_length,
        seed=seed, buffer_size=buffer_size,
    )
