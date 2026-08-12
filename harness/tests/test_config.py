"""External contract tests for MiniCPM4 0.5B architecture constants."""

import importlib
import os
import unittest

_HAS_TORCH = importlib.util.find_spec("torch") is not None
_RUN_ENGINE_CONTRACTS = os.environ.get("HARNESS_RUN_ENGINE_CONTRACTS") == "1"


@unittest.skipUnless(_HAS_TORCH and _RUN_ENGINE_CONTRACTS, "requires engine contract tests")
class TestMiniCPM4Config(unittest.TestCase):
    def test_architecture_constants(self):
        from training_engine_tensor.config import (
            FFN_HIDDEN_SIZE,
            HEAD_DIM,
            HIDDEN_SIZE,
            MAX_SEQ_LENGTH,
            NORM_EPSILON,
            NUM_HEADS,
            NUM_KV_HEADS,
            NUM_LAYERS,
            ROTARY_BASE,
            VOCAB_SIZE,
        )

        self.assertEqual(NUM_LAYERS, 24)
        self.assertEqual(HIDDEN_SIZE, 1024)
        self.assertEqual(NUM_HEADS, 16)
        self.assertEqual(NUM_KV_HEADS, 2)
        self.assertEqual(HEAD_DIM, 64)
        self.assertEqual(FFN_HIDDEN_SIZE, 4096)
        self.assertEqual(VOCAB_SIZE, 73448)
        self.assertEqual(MAX_SEQ_LENGTH, 4096)
        self.assertAlmostEqual(NORM_EPSILON, 1e-6)
        self.assertAlmostEqual(ROTARY_BASE, 10000.0)

    def test_gqa_projection_sizes(self):
        from training_engine_tensor.config import (
            _KV_PROJ_SIZE,
            _Q_PROJ_SIZE,
            HEAD_DIM,
            HIDDEN_SIZE,
            NUM_HEADS,
            NUM_KV_HEADS,
        )

        self.assertEqual(_Q_PROJ_SIZE, NUM_HEADS * HEAD_DIM)
        self.assertEqual(_Q_PROJ_SIZE, 1024)
        self.assertEqual(_KV_PROJ_SIZE, NUM_KV_HEADS * HEAD_DIM)
        self.assertEqual(_KV_PROJ_SIZE, 128)
        self.assertEqual(HEAD_DIM, HIDDEN_SIZE // NUM_HEADS)

    def test_heads_per_kv_group(self):
        from training_engine_tensor.config import NUM_HEADS, NUM_KV_HEADS

        self.assertEqual(NUM_HEADS % NUM_KV_HEADS, 0)
        self.assertEqual(NUM_HEADS // NUM_KV_HEADS, 8)

    def test_flop_counts_positive(self):
        from training_engine_tensor.config import (
            FORWARD_FLOPS_PER_TOKEN,
            STANDARD_FORWARD_FLOPS_PER_TOKEN,
            STANDARD_TRAINING_FLOPS_PER_TOKEN,
            TRAINING_FLOPS_PER_TOKEN,
        )

        self.assertGreater(FORWARD_FLOPS_PER_TOKEN, 0)
        self.assertGreater(STANDARD_FORWARD_FLOPS_PER_TOKEN, 0)
        self.assertEqual(TRAINING_FLOPS_PER_TOKEN, 3 * FORWARD_FLOPS_PER_TOKEN)
        self.assertEqual(STANDARD_TRAINING_FLOPS_PER_TOKEN, 3 * STANDARD_FORWARD_FLOPS_PER_TOKEN)

    def test_compute_training_flops(self):
        from training_engine_tensor.config import (
            STANDARD_TRAINING_FLOPS_PER_TOKEN,
            TRAINING_FLOPS_PER_TOKEN,
            compute_training_flops,
            compute_training_flops_standard,
        )

        tokens = 4096
        self.assertEqual(compute_training_flops(tokens), TRAINING_FLOPS_PER_TOKEN * tokens)
        self.assertEqual(
            compute_training_flops_standard(tokens),
            STANDARD_TRAINING_FLOPS_PER_TOKEN * tokens,
        )


if __name__ == "__main__":
    unittest.main()
