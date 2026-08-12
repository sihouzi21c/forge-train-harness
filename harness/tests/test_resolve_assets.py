"""Smoke test: TOML → ``FORGE_TOKENIZER_DIR`` / ``FORGE_DATA_DIR`` transmission.

Verifies that ``config_runtime.resolve_assets`` reads the TOML axis dicts,
writes the resolved paths into ``os.environ``, and returns them — so the
gate subprocess (``_resolve_data_env_from_conf``, ``build_suite_env``)
picks them up via inheritance.

Pre-populates the target dirs so the cache-hit branch fires; no network
or huggingface_hub import is exercised.
"""

from __future__ import annotations

import os

from harness import config_runtime


def test_resolve_assets_transmits_toml_to_env(monkeypatch, tmp_path):
    tok_dir = tmp_path / "tok"
    tok_dir.mkdir()
    (tok_dir / "tokenizer.model").write_bytes(b"x")

    data_dir = tmp_path / "data"
    (data_dir / "en").mkdir(parents=True)
    (data_dir / "en" / "shard.parquet").write_bytes(b"x")

    monkeypatch.delenv("FORGE_TOKENIZER_DIR", raising=False)
    monkeypatch.delenv("FORGE_DATA_DIR", raising=False)

    ref = {
        "forge_tokenizer_dir": str(tok_dir),
        "tokenizer": "vendor/tokenizer-repo",
    }
    data = {
        "forge_data_dir": str(data_dir),
        "dataset": "vendor/data-repo",
        "download_files": {"en/shard.parquet": "src/en/shard.parquet"},
    }

    out = config_runtime.resolve_assets(ref, data)

    assert out == {
        "FORGE_TOKENIZER_DIR": str(tok_dir),
        "FORGE_DATA_DIR": str(data_dir),
    }
    assert os.environ["FORGE_TOKENIZER_DIR"] == str(tok_dir)
    assert os.environ["FORGE_DATA_DIR"] == str(data_dir)


def test_ensure_dataset_seeds_from_baked_dir_without_network(tmp_path):
    # On image-baked clusters (and behind shandong's whitelist proxy where
    # the HF mirror stalls), the dev slice must come from the local baked
    # tree, not a re-download. The baked branch never imports hf_hub — and
    # since huggingface_hub is not installed here, a regression that fell
    # through to the download path would raise ModuleNotFoundError, so a
    # green copy is itself proof the baked branch was taken.
    baked = tmp_path / "baked"
    (baked / "en").mkdir(parents=True)
    (baked / "en" / "p1.parquet").write_bytes(b"BAKED")
    dest = tmp_path / "ws_data"

    config_runtime._ensure_dataset(
        "vendor/data-repo",
        dest,
        {"en/p1.parquet": "src/en/p1.parquet"},
        baked_dir=baked,
    )

    assert (dest / "en" / "p1.parquet").read_bytes() == b"BAKED"


def test_ensure_dataset_falls_back_to_hf_when_not_baked(tmp_path, monkeypatch):
    import sys
    import types

    cached = tmp_path / "cached.parquet"
    cached.write_bytes(b"HF")
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = lambda **_: str(cached)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    dest = tmp_path / "ws_data"
    config_runtime._ensure_dataset(
        "vendor/data-repo",
        dest,
        {"en/p1.parquet": "src/en/p1.parquet"},
        baked_dir=tmp_path / "nonexistent",
    )

    assert (dest / "en" / "p1.parquet").read_bytes() == b"HF"


def test_resolve_assets_seeds_data_from_baked_dir(tmp_path, monkeypatch):
    # baked covers every download_files entry → no hf_hub import at all.
    monkeypatch.delenv("FORGE_DATA_DIR", raising=False)

    tok_dir = tmp_path / "tok"
    tok_dir.mkdir()
    (tok_dir / "tokenizer.model").write_bytes(b"x")
    baked = tmp_path / "baked"
    (baked / "en").mkdir(parents=True)
    (baked / "en" / "p1.parquet").write_bytes(b"BAKED")
    dest = tmp_path / "ws_data"  # workspace forge_data_dir, initially empty

    ref = {"forge_tokenizer_dir": str(tok_dir), "tokenizer": "vendor/tokenizer-repo"}
    data = {
        "forge_data_dir": str(dest),
        "dataset": "vendor/data-repo",
        "download_files": {"en/p1.parquet": "src/en/p1.parquet"},
        "baked_data_dir": str(baked),
    }

    out = config_runtime.resolve_assets(ref, data)

    assert (dest / "en" / "p1.parquet").read_bytes() == b"BAKED"
    assert out["FORGE_DATA_DIR"] == str(dest)
