#!/usr/bin/env python3
"""Self-contained gsm8k → indexed-binary (.bin/.idx) preparator (torch path).

Produces the exact same on-disk ``.bin``/``.idx`` indexed-binary pair
(``MMIDIDX`` magic) that the pure-torch L0 ref's in-file dataloader in
``train_pure_mup_mtp.py`` consumes — but using **SentencePiece directly**
for tokenization instead of the reference training framework's
``tools/preprocess_data.py`` tooling. This removes the only remaining
external-source-tree dependency on the gsm8k torch data-prep path: the
build no longer needs an external training-framework checkout.

Tokenization fidelity
---------------------
Byte-for-byte matches the reference preprocessing
(``preprocess_data.py --tokenizer-type Llama2Tokenizer --append-eod``,
no ``--split-sentences``), which for each document emits::

    SentencePiece.encode(text) + [eos_id]

where ``eos_id`` comes from the SentencePiece model and ``eod == eos_id``.
**No leading BOS.** This was verified byte-for-byte on cpm_core_r0.15.0
(commit 8b4e79a5 / branch HEAD) by diffing against a freshly generated
reference dataset: although ``Llama2Tokenizer.tokenize`` nominally defaults
``bos=True``, the reference ``preprocess_data.py --append-eod`` run
deterministically emits ``encode(text) + [eod]`` with no BOS (confirmed
identical under --workers 1 and 4). The on-disk reference data is the
ground truth, so this writer matches it (see ``encode_documents``).

On-disk format (little-endian), as parsed by the reader
-------------------------------------------------------
``.idx``::

    magic   "MMIDIDX\\x00\\x00"   (9 bytes)
    version int64 = 1
    dtype   uint8                 (4 = int32; see DTYPE_CODE rationale)
    n_seq   int64
    n_docs  int64
    sizes   int32[n_seq]          (#tokens per sequence)
    ptrs    int64[n_seq]          (byte offset of each sequence in .bin)
    doc_idx int64[n_docs + 1]     (sequence index spanned by each document)

``.bin``: all token ids of all documents, concatenated, as ``dtype``.

DTYPE_CODE
----------
Fixed to ``4`` (int32). MiniCPM4-0.5B has ``padded_vocab_size = 73448``,
which overflows uint16 (max 65535); int32 is required to store the ids.

Usage (CLI)::

    GSM8K_DIR=/path/to/hf/gsm8k/main \\
    TOKENIZER_MODEL=/path/to/tokenizer.model \\
    python3 gsm8k_prepare_torch.py

Required env: ``GSM8K_DIR``, ``TOKENIZER_MODEL``.
Optional env: ``OUTPUT_DIR`` (default ``<repo>/.artifacts/data/gsm8k_megatron``).
"""

from __future__ import annotations

import os
import struct
import sys
from array import array
from pathlib import Path
from typing import Iterable, Protocol


# ── On-disk format constants ────────────────────────────────────────────
_INDEX_HEADER = b"MMIDIDX\x00\x00"  # 9 bytes
_INDEX_VERSION = 1
DTYPE_CODE = 4  # int32 — required: MiniCPM4 vocab (73448) > uint16 max (65535)
_DTYPE_ITEMSIZE = 4  # bytes per int32


class _Tokenizer(Protocol):
    """Minimal SentencePiece-shaped surface used by :func:`encode_documents`.

    Implemented by ``sentencepiece.SentencePieceProcessor`` and by test
    doubles alike — kept injectable so the encoding rule is testable
    without a real model file.
    """

    def bos_id(self) -> int: ...
    def eos_id(self) -> int: ...
    def encode(self, text: str) -> list[int]: ...


# ── Core (dependency-light, unit-testable) ──────────────────────────────
def encode_documents(
    texts: Iterable[str],
    tokenizer: _Tokenizer,
    add_bos: bool = False,
    add_eos: bool = True,
) -> list[list[int]]:
    """Tokenize each document as ``([bos] +) encode(text) + ([eos])``.

    Defaults (``add_bos=False, add_eos=True``) reproduce the **empirically
    verified** output of the reference preprocessing
    (``prepare_gsm8k_data.sh`` → ``preprocess_data.py --tokenizer-type
    Llama2Tokenizer --append-eod``) on cpm_core_r0.15.0: each document is
    ``encode(text) + [eod]`` with ``eod == eos_id`` and **no leading BOS**.

    NOTE on the no-BOS finding: ``_Llama2Tokenizer.tokenize`` *does* default
    ``bos=True`` in source (and adds BOS when called directly), yet the
    reference ``preprocess_data.py`` run deterministically produces no BOS
    (verified byte-for-byte against a freshly generated reference dataset,
    --workers 1 and 4 alike). The on-disk reference data is the ground
    truth, so the defaults here match the data, not the nominal source
    default. ``add_bos=True`` is exposed for callers that need it.

    Empty documents (no tokens) are skipped, matching the reference
    encoder's ``if len(doc_ids) > 0`` guard.
    """
    bos_id = tokenizer.bos_id() if add_bos else None
    eos_id = tokenizer.eos_id() if add_eos else None
    if add_bos and (bos_id is None or bos_id < 0):
        raise ValueError(f"add_bos=True but tokenizer bos_id is invalid ({bos_id!r})")
    if add_eos and (eos_id is None or eos_id < 0):
        raise ValueError(f"add_eos=True but tokenizer eos_id is invalid ({eos_id!r})")

    docs: list[list[int]] = []
    for text in texts:
        ids = list(tokenizer.encode(text))
        if not ids:
            continue
        if add_bos:
            ids = [bos_id, *ids]
        if add_eos:
            ids = [*ids, eos_id]
        docs.append(ids)
    return docs


def write_indexed_binary(path_prefix: str, documents: list[list[int]]) -> tuple[str, str]:
    """Write ``documents`` to ``<prefix>.bin`` / ``<prefix>.idx``.

    Each document becomes exactly one sequence. Pure stdlib (``array`` +
    ``struct``); no third-party dependency, so the format is verifiable
    anywhere. Returns the ``(.bin, .idx)`` paths.
    """
    bin_path = path_prefix + ".bin"
    idx_path = path_prefix + ".idx"

    n_seq = len(documents)
    sizes = [len(doc) for doc in documents]

    # Byte offset of each sequence within the flat .bin stream.
    pointers: list[int] = []
    running = 0
    for size in sizes:
        pointers.append(running)
        running += size * _DTYPE_ITEMSIZE

    # .bin — concatenated int32 token stream, little-endian.
    flat = array("i")  # 'i' == 4-byte signed int on all supported platforms
    for doc in documents:
        flat.extend(doc)
    if sys.byteorder == "big":
        flat.byteswap()
    with open(bin_path, "wb") as f:
        flat.tofile(f)

    # document_indices: one sequence per document → [0, 1, ..., n_seq].
    # document_count is the *length* of this array (n_seq + 1), matching
    # the reference _IndexWriter, which writes ``len(document_indices)``.
    doc_indices = list(range(n_seq + 1))

    # .idx — header + sizes + pointers + document_indices (all little-endian).
    with open(idx_path, "wb") as f:
        f.write(_INDEX_HEADER)
        f.write(struct.pack("<q", _INDEX_VERSION))
        f.write(struct.pack("<B", DTYPE_CODE))
        f.write(struct.pack("<q", n_seq))            # sequence_count
        f.write(struct.pack("<q", len(doc_indices)))  # document_count == n_seq + 1
        f.write(struct.pack(f"<{n_seq}i", *sizes))
        f.write(struct.pack(f"<{n_seq}q", *pointers))
        f.write(struct.pack(f"<{len(doc_indices)}q", *doc_indices))

    return bin_path, idx_path


def format_gsm8k_text(question: str, answer: str) -> str:
    """Document text layout — identical to the reference parquet→jsonl step."""
    return f"Question: {question}\nAnswer: {answer}"


# ── CLI glue (heavier deps imported lazily so the module stays importable) ─
def _read_gsm8k_texts(gsm8k_dir: str) -> list[str]:
    import pandas as pd  # lazy: only the CLI path needs pandas/pyarrow

    parquet_dir = Path(gsm8k_dir)
    candidates = sorted(parquet_dir.glob("train-*.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"no train-*.parquet under GSM8K_DIR={gsm8k_dir} "
            f"(expected e.g. train-00000-of-00001.parquet)"
        )
    texts: list[str] = []
    for path in candidates:
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            texts.append(format_gsm8k_text(row["question"], row["answer"]))
    return texts


class _HFTokenizerAdapter:
    """Wrap a HuggingFace ``tokenizers.Tokenizer`` in the SentencePiece-shaped
    surface :class:`_Tokenizer` (``bos_id`` / ``eos_id`` / ``encode``).

    A HuggingFace *fast* tokenizer (``tokenizer.json``, e.g. MiniCPM5-1B's
    130k BPE vocab) is the only tokenizer artifact some models publish — they
    deliberately ship no SentencePiece ``tokenizer.model``. This adapter lets
    such a ``.json`` flow through the exact same encode contract the rest of
    the torch-ref data path relies on (``encode(text) + [eos]``, no BOS —
    owned by :func:`encode_documents`).

    ``encode`` returns the raw token ids with **no special tokens injected**:
    :func:`encode_documents` is the single authority that appends ``[eos]``,
    so injecting here would double-append and corrupt the on-disk stream.

    ``bos_id`` / ``eos_id`` are resolved from the sibling
    ``tokenizer_config.json`` / ``special_tokens_map.json`` (the HF home for
    the bos/eos token *strings*) and mapped to ids via the tokenizer's own
    vocab. They fall back to the conventional ``<s>`` / ``</s>`` when the
    config files are absent. ``-1`` signals "unknown" (the SentencePiece
    convention), so ``encode_documents``'s ``eos_id < 0`` guard fires loudly
    rather than silently writing a bogus id.
    """

    def __init__(self, tokenizer, bos_id: int, eos_id: int) -> None:
        self._tok = tokenizer
        self._bos = bos_id
        self._eos = eos_id

    def bos_id(self) -> int:
        return self._bos

    def eos_id(self) -> int:
        return self._eos

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids


def _hf_special_id(tokenizer, sidecar_dir: Path, key: str, default_token: str) -> int:
    """Resolve a bos/eos token id for a HF ``tokenizer.json``.

    Reads the token *string* from ``tokenizer_config.json``'s ``<key>_token``
    (falling back to ``special_tokens_map.json`` then to *default_token*),
    then maps it to an id through the tokenizer's own vocab. Returns ``-1``
    when the token cannot be mapped, matching the SentencePiece "unknown"
    sentinel that the downstream ``< 0`` guards key off.
    """
    import json

    token: str | None = None
    for fname, field in (
        ("tokenizer_config.json", f"{key}_token"),
        ("special_tokens_map.json", f"{key}_token"),
    ):
        path = sidecar_dir / fname
        if not path.is_file():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8")).get(field)
        except (ValueError, OSError):
            continue
        # ``special_tokens_map.json`` may store either a bare string or an
        # ``AddedToken`` dict ``{"content": "<s>", ...}``.
        if isinstance(raw, dict):
            raw = raw.get("content")
        if isinstance(raw, str) and raw:
            token = raw
            break
    if token is None:
        token = default_token
    tid = tokenizer.token_to_id(token)
    return tid if isinstance(tid, int) else -1


def _load_tokenizer(tokenizer_model: str) -> _Tokenizer:
    """Load a tokenizer, dispatching on the basename suffix.

    * ``*.model`` → SentencePiece (``SentencePieceProcessor``), the
      byte-for-byte reference-aligned encoder.
    * ``*.json``  → HuggingFace fast tokenizer (``tokenizers.Tokenizer``)
      wrapped in :class:`_HFTokenizerAdapter` so it presents the same
      ``bos_id`` / ``eos_id`` / ``encode`` surface.

    Both lazy-import their backend so this module stays importable on hosts
    that have neither installed (the dependency-light core is what the unit
    tests exercise).
    """
    if tokenizer_model.endswith(".json"):
        from tokenizers import Tokenizer  # lazy: only the .json path needs it

        tok = Tokenizer.from_file(tokenizer_model)
        sidecar_dir = Path(tokenizer_model).resolve().parent
        bos_id = _hf_special_id(tok, sidecar_dir, "bos", "<s>")
        eos_id = _hf_special_id(tok, sidecar_dir, "eos", "</s>")
        return _HFTokenizerAdapter(tok, bos_id=bos_id, eos_id=eos_id)

    import sentencepiece  # lazy: only the .model path needs sentencepiece

    return sentencepiece.SentencePieceProcessor(model_file=tokenizer_model)


def main() -> int:
    gsm8k_dir = os.environ.get("GSM8K_DIR")
    tokenizer_model = os.environ.get("TOKENIZER_MODEL")
    if not gsm8k_dir:
        sys.stderr.write("ERROR: GSM8K_DIR is required (HuggingFace gsm8k parquet root)\n")
        return 2
    if not tokenizer_model:
        sys.stderr.write(
            "ERROR: TOKENIZER_MODEL is required "
            "(SentencePiece tokenizer.model or HuggingFace tokenizer.json path)\n"
        )
        return 2
    if not Path(tokenizer_model).is_file():
        sys.stderr.write(f"ERROR: TOKENIZER_MODEL not found: {tokenizer_model}\n")
        return 2

    repo_root_guess = Path(__file__).resolve().parents[2]
    output_dir = Path(
        os.environ.get("OUTPUT_DIR", repo_root_guess / ".artifacts" / "data" / "gsm8k_megatron")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = str(output_dir / "gsm8k_train_text_document")

    sys.stderr.write("=== Step 1: read gsm8k parquet → text ===\n")
    texts = _read_gsm8k_texts(gsm8k_dir)
    sys.stderr.write(f"loaded {len(texts)} documents\n")

    sys.stderr.write("=== Step 2: tokenize (SentencePiece, encode+[eos], no bos) ===\n")
    tokenizer = _load_tokenizer(tokenizer_model)
    documents = encode_documents(texts, tokenizer)

    sys.stderr.write("=== Step 3: write indexed binary (.bin/.idx) ===\n")
    bin_path, idx_path = write_indexed_binary(output_prefix, documents)
    total_tokens = sum(len(doc) for doc in documents)
    sys.stderr.write(
        f"wrote {len(documents)} sequences / {total_tokens} tokens\n"
        f"  {bin_path}\n  {idx_path}\n"
    )
    # Stdout: the DATA_PATH prefix the torch ref / data conf should point at.
    print(output_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
