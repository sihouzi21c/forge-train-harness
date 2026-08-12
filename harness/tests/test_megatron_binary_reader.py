"""Tests for the Megatron binary (.bin/.idx) reader in train_pure_mup_mtp.py."""

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _ref_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "ref" / "reference"


def _import_reader():
    """Import the reader module by path (avoids needing it on sys.path)."""
    import importlib.util
    import os
    import sys

    # The training entry imports model_pure_mup_mtp, which reads its geometry
    # fail-fast from os.environ at import (generic product projection, no baked
    # defaults); seed the 0.5B shape for this standalone import. setdefault
    # keeps a real projection authoritative.
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

    ref_dir = str(_ref_dir())
    added = ref_dir not in sys.path
    if added:
        sys.path.insert(0, ref_dir)
    try:
        spec = importlib.util.spec_from_file_location(
            "train_pure_mup_mtp", _ref_dir() / "train_pure_mup_mtp.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if added and ref_dir in sys.path:
            sys.path.remove(ref_dir)


_INDEX_HEADER = b"MMIDIDX\x00\x00"


def _write_synthetic_dataset(prefix: str, tokens: list[int], seq_length: int):
    """Write a synthetic Megatron binary dataset (.bin + .idx).

    The dataset stores ``tokens`` as a single contiguous sequence of
    dtype uint16 (code=8). The .idx records one document covering the
    full sequence.
    """
    dtype_code = 8  # uint16
    arr = np.array(tokens, dtype=np.uint16)
    num_sequences = 1
    num_documents = 1
    seq_len = len(tokens)

    # .bin — flat-packed token data
    bin_path = prefix + ".bin"
    arr.tofile(bin_path)

    # .idx — header + sizes + pointers + document_indices
    idx_path = prefix + ".idx"
    with open(idx_path, "wb") as f:
        f.write(_INDEX_HEADER)  # 9 bytes magic
        f.write(struct.pack("<q", 1))  # 8 bytes version
        f.write(struct.pack("<B", dtype_code))  # 1 byte dtype
        f.write(struct.pack("<q", num_sequences))  # 8 bytes num_sequences
        f.write(struct.pack("<q", num_documents))  # 8 bytes num_documents
        # sizes: int32[num_sequences]
        f.write(np.array([seq_len], dtype=np.int32).tobytes())
        # pointers: int64[num_sequences]
        f.write(np.array([0], dtype=np.int64).tobytes())
        # document_indices: int64[num_documents + 1]
        f.write(np.array([0, num_sequences], dtype=np.int64).tobytes())


def _write_multi_seq_dataset(prefix: str, sequences: list[list[int]]):
    """Write dataset with multiple sequences."""
    dtype_code = 8  # uint16
    num_sequences = len(sequences)
    num_documents = 1

    # .bin — concatenated sequences
    all_tokens = []
    for seq in sequences:
        all_tokens.extend(seq)
    arr = np.array(all_tokens, dtype=np.uint16)
    bin_path = prefix + ".bin"
    arr.tofile(bin_path)

    # Compute sizes and byte pointers
    sizes = np.array([len(s) for s in sequences], dtype=np.int32)
    pointers = np.zeros(num_sequences, dtype=np.int64)
    offset = 0
    for i, s in enumerate(sequences):
        pointers[i] = offset
        offset += len(s) * 2  # uint16 = 2 bytes each

    # .idx
    idx_path = prefix + ".idx"
    with open(idx_path, "wb") as f:
        f.write(_INDEX_HEADER)
        f.write(struct.pack("<q", 1))
        f.write(struct.pack("<B", dtype_code))
        f.write(struct.pack("<q", num_sequences))
        f.write(struct.pack("<q", num_documents))
        f.write(sizes.tobytes())
        f.write(pointers.tobytes())
        f.write(np.array([0, num_sequences], dtype=np.int64).tobytes())


class TestFormatDetection(unittest.TestCase):
    """Test _is_megatron_binary() detection logic."""

    def test_returns_true_when_both_files_exist(self):
        mod = _import_reader()
        with tempfile.TemporaryDirectory() as d:
            prefix = str(Path(d) / "data")
            _write_synthetic_dataset(prefix, list(range(100)), 50)
            self.assertTrue(mod._is_megatron_binary(prefix))

    def test_returns_false_when_bin_missing(self):
        mod = _import_reader()
        with tempfile.TemporaryDirectory() as d:
            prefix = str(Path(d) / "data")
            # Only write .idx
            Path(prefix + ".idx").write_bytes(b"dummy")
            self.assertFalse(mod._is_megatron_binary(prefix))

    def test_returns_false_when_idx_missing(self):
        mod = _import_reader()
        with tempfile.TemporaryDirectory() as d:
            prefix = str(Path(d) / "data")
            Path(prefix + ".bin").write_bytes(b"dummy")
            self.assertFalse(mod._is_megatron_binary(prefix))

    def test_returns_false_when_neither_exists(self):
        mod = _import_reader()
        with tempfile.TemporaryDirectory() as d:
            prefix = str(Path(d) / "nonexistent")
            self.assertFalse(mod._is_megatron_binary(prefix))


class TestIndexParsing(unittest.TestCase):
    """Test that MegatronBinaryDataloader correctly parses .idx headers."""

    def test_single_sequence(self):
        mod = _import_reader()
        with tempfile.TemporaryDirectory() as d:
            prefix = str(Path(d) / "data")
            tokens = list(range(200))
            _write_synthetic_dataset(prefix, tokens, 50)
            dl = mod.MegatronBinaryDataloader(
                prefix,
                dp_rank=0,
                world_size=1,
                micro_batch_size=1,
                seq_length=49,
                seed=42,
            )
            self.assertEqual(dl._num_sequences, 1)
            self.assertEqual(dl._num_documents, 1)

    def test_multi_sequence(self):
        mod = _import_reader()
        with tempfile.TemporaryDirectory() as d:
            prefix = str(Path(d) / "data")
            seqs = [list(range(50)), list(range(50, 100)), list(range(100, 150))]
            _write_multi_seq_dataset(prefix, seqs)
            dl = mod.MegatronBinaryDataloader(
                prefix,
                dp_rank=0,
                world_size=1,
                micro_batch_size=1,
                seq_length=49,
                seed=42,
            )
            self.assertEqual(dl._num_sequences, 3)


class TestBatchFormat(unittest.TestCase):
    """Test that __next__() returns correctly shaped batch dicts."""

    def test_batch_keys_and_shapes(self):
        mod = _import_reader()
        with tempfile.TemporaryDirectory() as d:
            prefix = str(Path(d) / "data")
            # 500 tokens → enough for several windows of seq_length+1=50+1
            tokens = list(range(500))
            _write_synthetic_dataset(prefix, tokens, 50)
            mbs = 2
            seq_len = 49
            dl = mod.MegatronBinaryDataloader(
                prefix,
                dp_rank=0,
                world_size=1,
                micro_batch_size=mbs,
                seq_length=seq_len,
                seed=42,
            )
            batch = next(iter(dl))
            self.assertIn("tokens", batch)
            self.assertIn("labels", batch)
            self.assertIn("loss_mask", batch)
            self.assertEqual(batch["tokens"].shape, (mbs, seq_len))
            self.assertEqual(batch["labels"].shape, (mbs, seq_len))
            self.assertEqual(batch["loss_mask"].shape, (mbs, seq_len))

    def test_tokens_labels_shifted(self):
        mod = _import_reader()
        with tempfile.TemporaryDirectory() as d:
            prefix = str(Path(d) / "data")
            tokens = list(range(500))
            _write_synthetic_dataset(prefix, tokens, 50)
            dl = mod.MegatronBinaryDataloader(
                prefix,
                dp_rank=0,
                world_size=1,
                micro_batch_size=1,
                seq_length=49,
                seed=0,
            )
            batch = next(iter(dl))
            t = batch["tokens"][0].numpy()
            l = batch["labels"][0].numpy()
            # labels[i] == tokens[i+1] within the same window
            # The window is contiguous so labels = tokens shifted by 1
            self.assertTrue(np.array_equal(l[:-1], t[1:]))

    def test_loss_mask_all_ones(self):
        mod = _import_reader()
        with tempfile.TemporaryDirectory() as d:
            prefix = str(Path(d) / "data")
            tokens = list(range(500))
            _write_synthetic_dataset(prefix, tokens, 50)
            dl = mod.MegatronBinaryDataloader(
                prefix,
                dp_rank=0,
                world_size=1,
                micro_batch_size=2,
                seq_length=49,
                seed=42,
            )
            batch = next(iter(dl))
            self.assertTrue((batch["loss_mask"] == 1.0).all())

    def test_token_dtype_int64(self):
        mod = _import_reader()
        with tempfile.TemporaryDirectory() as d:
            prefix = str(Path(d) / "data")
            tokens = list(range(500))
            _write_synthetic_dataset(prefix, tokens, 50)
            dl = mod.MegatronBinaryDataloader(
                prefix,
                dp_rank=0,
                world_size=1,
                micro_batch_size=1,
                seq_length=49,
                seed=42,
            )
            batch = next(iter(dl))
            import torch

            self.assertEqual(batch["tokens"].dtype, torch.long)
            self.assertEqual(batch["labels"].dtype, torch.long)


class TestDistributedSharding(unittest.TestCase):
    """Test that different dp_ranks get non-overlapping data."""

    def test_rank0_and_rank1_no_overlap(self):
        mod = _import_reader()
        with tempfile.TemporaryDirectory() as d:
            prefix = str(Path(d) / "data")
            # 1000 tokens → ~20 windows of length 50
            tokens = list(range(1000))
            _write_synthetic_dataset(prefix, tokens, 50)
            seq_len = 49
            dl0 = mod.MegatronBinaryDataloader(
                prefix,
                dp_rank=0,
                world_size=2,
                micro_batch_size=1,
                seq_length=seq_len,
                seed=42,
            )
            dl1 = mod.MegatronBinaryDataloader(
                prefix,
                dp_rank=1,
                world_size=2,
                micro_batch_size=1,
                seq_length=seq_len,
                seed=42,
            )
            # Collect first few batches from each
            batches_0, batches_1 = [], []
            iter0, iter1 = iter(dl0), iter(dl1)
            for _ in range(5):
                b0 = next(iter0)
                b1 = next(iter1)
                batches_0.append(b0["tokens"][0].numpy().tobytes())
                batches_1.append(b1["tokens"][0].numpy().tobytes())
            # No overlap between any pair
            self.assertEqual(len(set(batches_0) & set(batches_1)), 0)

    def test_both_ranks_cover_full_data(self):
        mod = _import_reader()
        with tempfile.TemporaryDirectory() as d:
            prefix = str(Path(d) / "data")
            tokens = list(range(400))
            _write_synthetic_dataset(prefix, tokens, 50)
            seq_len = 49
            dl0 = mod.MegatronBinaryDataloader(
                prefix,
                dp_rank=0,
                world_size=2,
                micro_batch_size=1,
                seq_length=seq_len,
                seed=42,
            )
            dl1 = mod.MegatronBinaryDataloader(
                prefix,
                dp_rank=1,
                world_size=2,
                micro_batch_size=1,
                seq_length=seq_len,
                seed=42,
            )
            n0 = dl0._num_local_windows
            n1 = dl1._num_local_windows
            # Together should equal total windows
            total_windows = dl0._total_windows
            self.assertEqual(n0 + n1, total_windows)


class TestCycling(unittest.TestCase):
    """Test that the iterator cycles when data is exhausted."""

    def test_more_batches_than_windows(self):
        mod = _import_reader()
        with tempfile.TemporaryDirectory() as d:
            prefix = str(Path(d) / "data")
            # Only 200 tokens → ~4 windows of 50
            tokens = list(range(200))
            _write_synthetic_dataset(prefix, tokens, 50)
            dl = mod.MegatronBinaryDataloader(
                prefix,
                dp_rank=0,
                world_size=1,
                micro_batch_size=1,
                seq_length=49,
                seed=42,
            )
            it = iter(dl)
            # Should be able to pull many more batches than windows
            for _ in range(20):
                batch = next(it)
                self.assertEqual(batch["tokens"].shape[1], 49)


class TestAutoDispatch(unittest.TestCase):
    """Test that build_dataloader dispatches to the correct loader class."""

    def test_dispatches_to_megatron_binary(self):
        mod = _import_reader()
        with tempfile.TemporaryDirectory() as d:
            prefix = str(Path(d) / "data")
            tokens = list(range(500))
            _write_synthetic_dataset(prefix, tokens, 50)
            # data_path_args format: ["weight", "path", ...]
            data_path_args = ["1.0", prefix]
            dl = mod.build_dataloader(
                data_path_args,
                dp_rank=0,
                world_size=1,
                micro_batch_size=2,
                seq_length=49,
                seed=42,
            )
            self.assertIsInstance(dl, mod.MegatronBinaryDataloader)

    def test_raises_on_missing_megatron_binary(self):
        mod = _import_reader()
        with tempfile.TemporaryDirectory() as d:
            prefix = str(Path(d) / "missing_text_document")
            data_path_args = ["1.0", prefix]
            with self.assertRaises(FileNotFoundError) as ctx:
                mod.build_dataloader(
                    data_path_args,
                    dp_rank=0,
                    world_size=1,
                    micro_batch_size=2,
                    seq_length=49,
                    seed=42,
                )
            self.assertIn("Megatron binary data not found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
