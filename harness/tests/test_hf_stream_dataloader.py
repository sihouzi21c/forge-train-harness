"""Tests for the streaming HF dataloader (ref-side data path).

Pure parsing/routing tests run anywhere the module imports. The
end-to-end pack + determinism tests need ``datasets`` + ``pyarrow`` +
``torch`` (present in the repro image / devspace) and are skipped
otherwise via ``importorskip``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REF = Path(__file__).resolve().parents[2] / "ref" / "reference"
if str(_REF) not in sys.path:
    sys.path.insert(0, str(_REF))

# The module imports numpy/torch + gsm8k_prepare_torch at top level.
hf = pytest.importorskip("hf_stream_dataloader")


# ── pure parsing / routing (no heavy runtime) ───────────────────────────
def test_parse_hf_token_text_column():
    spec = hf._parse_hf_token(0.667, "hf://openbmb/Ultra-FineWeb?split=en&text=content")
    assert spec.repo == "openbmb/Ultra-FineWeb"
    assert spec.split == "en"
    assert spec.text_col == "content"
    assert spec.template is None
    assert spec.weight == 0.667


def test_parse_hf_token_template_and_local_env():
    tok = (
        "hf://openai/gsm8k?config=main&split=train&template=gsm8k_qa"
        "&local_env=GSM8K_DIR&glob=train-*.parquet"
    )
    spec = hf._parse_hf_token(1.0, tok)
    assert spec.repo == "openai/gsm8k"
    assert spec.config == "main"
    assert spec.template == "gsm8k_qa"
    assert spec.local_env == "GSM8K_DIR"
    assert spec.glob == "train-*.parquet"


def test_template_text_extraction():
    spec = hf._parse_hf_token(1.0, "hf://openai/gsm8k?split=train&template=gsm8k_qa")
    assert spec.text_of({"question": "1+1?", "answer": "2"}) == "Question: 1+1?\nAnswer: 2"


def test_is_hf_data_path():
    assert hf.is_hf_data_path([(1.0, "hf://openai/gsm8k?split=train&text=q")])
    assert not hf.is_hf_data_path([(1.0, "/abs/path/gsm8k_train_text_document")])


def test_parse_missing_split_raises():
    with pytest.raises(ValueError):
        hf._parse_hf_token(1.0, "hf://openai/gsm8k?text=content")


# ── end-to-end pack + determinism (needs datasets + pyarrow) ────────────
class _FakeTokenizer:
    """Deterministic stand-in for SentencePiece: char codes, eos=2."""

    def encode(self, text):
        return [(ord(c) % 97) + 3 for c in text][:64]

    def eos_id(self):
        return 2


@pytest.fixture
def _local_parquet(tmp_path, monkeypatch):
    pytest.importorskip("datasets")
    pytest.importorskip("pyarrow")
    import pandas as pd

    root = tmp_path / "corpus"
    root.mkdir()
    rows = [{"content": f"document number {i} with some words"} for i in range(200)]
    pd.DataFrame(rows).to_parquet(root / "part-00000.parquet")
    monkeypatch.setenv("CORPUS_DIR", str(root))
    monkeypatch.setenv("TOKENIZER_MODEL", "/dev/null")  # bypassed by fake
    monkeypatch.setattr(hf, "_load_tokenizer", lambda _m: _FakeTokenizer())
    return root


def _token(weight_irrelevant=False):
    return "hf://local/corpus?split=train&text=content&local_env=CORPUS_DIR&glob=*.parquet"


def test_pack_shapes(_local_parquet):
    import torch  # noqa: F401

    dl = hf.build(
        [(1.0, _token())], dp_rank=0, world_size=1, micro_batch_size=4, seq_length=16, seed=1234
    )
    b = next(dl)
    assert b["tokens"].shape == (4, 16)
    assert b["labels"].shape == (4, 16)
    assert b["loss_mask"].shape == (4, 16)
    # labels are tokens shifted by one within each packed window.
    assert b["tokens"].dtype.is_floating_point is False


def test_bitwise_determinism_same_seed(_local_parquet):
    def run():
        dl = hf.build(
            [(1.0, _token())], dp_rank=0, world_size=1, micro_batch_size=4, seq_length=16, seed=1234
        )
        return [next(dl)["tokens"].numpy().tobytes() for _ in range(8)]

    assert run() == run()


def test_two_source_interleave_determinism(tmp_path, monkeypatch):
    pytest.importorskip("datasets")
    pytest.importorskip("pyarrow")
    import pandas as pd

    root = tmp_path / "multi"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir(parents=True)
    pd.DataFrame([{"content": f"alpha {i}"} for i in range(150)]).to_parquet(
        root / "a" / "p.parquet"
    )
    pd.DataFrame([{"content": f"beta {i}"} for i in range(150)]).to_parquet(
        root / "b" / "p.parquet"
    )
    monkeypatch.setenv("MULTI_DIR", str(root))
    monkeypatch.setenv("TOKENIZER_MODEL", "/dev/null")
    monkeypatch.setattr(hf, "_load_tokenizer", lambda _m: _FakeTokenizer())

    wp = [
        (0.667, "hf://x/a?split=train&text=content&local_env=MULTI_DIR&glob=a/*.parquet"),
        (0.333, "hf://x/b?split=train&text=content&local_env=MULTI_DIR&glob=b/*.parquet"),
    ]

    def run():
        dl = hf.build(wp, dp_rank=0, world_size=1, micro_batch_size=2, seq_length=8, seed=7)
        return [next(dl)["tokens"].numpy().tobytes() for _ in range(8)]

    assert run() == run()


# ── explicit DATA_LOADER dispatch in train_pure_mup_mtp.build_dataloader ──
def test_build_dataloader_dispatch(monkeypatch):
    """build_dataloader routes on the explicit loader kind, not the path."""
    import os

    # The training entry imports model_pure_mup_mtp, which reads its geometry
    # fail-fast from os.environ at import (generic product projection, no baked
    # defaults) — a KeyError there would hard-fail past importorskip. Seed the
    # 0.5B shape; setdefault keeps a real projection authoritative.
    for k, v in {
        "NUM_LAYERS": "24",
        "HIDDEN_SIZE": "1024",
        "NUM_ATTENTION_HEADS": "16",
        "NUM_QUERY_GROUPS": "2",
        "HEAD_DIM": "64",
        "FFN_HIDDEN_SIZE": "4096",
        "PADDED_VOCAB_SIZE": "73448",
        "MAX_POSITION_EMBEDDINGS": "4096",
        "NORM_EPSILON": "1e-6",
        "ROTARY_BASE": "10000",
    }.items():
        os.environ.setdefault(k, v)

    tp = pytest.importorskip("train_pure_mup_mtp")  # needs torch + model module
    seen = {}
    monkeypatch.setattr(
        tp,
        "_build_megatron_binary_dataloader",
        lambda *a, **k: seen.setdefault("megatron_binary", True),
    )
    monkeypatch.setattr(
        tp, "_build_sstable_dataloader", lambda *a, **k: seen.setdefault("modelbest", True)
    )
    monkeypatch.setattr(hf, "build", lambda *a, **k: seen.setdefault("hf", True))

    tp.build_dataloader(["1.0", "hf://x/y?split=train&text=t"], 0, 1, 4, 16, 1234, loader="hf")
    tp.build_dataloader(["0.5", "/a", "0.5", "/b"], 0, 1, 4, 16, 1234, loader="modelbest")
    tp.build_dataloader(["1.0", "/p/prefix"], 0, 1, 4, 16, 1234, loader="megatron_binary")
    assert seen == {"hf": True, "modelbest": True, "megatron_binary": True}

    with pytest.raises(ValueError):
        tp.build_dataloader(["1.0", "/p"], 0, 1, 4, 16, 1234, loader="bogus")
