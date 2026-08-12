"""Contract + smoke tests for the MiniCPM4-8B muP DP×TP reference.

Layers, cheapest first (CPU-only CI exercises everything — unlike the
CUDA-gated Qwen3 smoke, the 8B attention imports ``flash_attn`` lazily so
it can be mocked with an SDPA reference and run on CPU):

  1. [model] axis — config/model/minicpm4_8b.toml matches the official
     openbmb/MiniCPM4-8B geometry + muP (no MTP) and validates against the
     shared [model] schema. This is the "extreme accuracy" drift guard.
  2. [ref] axis — config/ref/torch_minicpm4_8b.toml points at the DP×TP
     launcher + the MiniCPM4-8B tokenizer repo.
  3. Launcher — run_minicpm4_8b_dptp.sh consumes the generic product
     projection env (NUM_STEPS / bare geometry / MUP_*, fail-fast `:?`; no
     in-launcher registry sourcing) and threads the TP degree through
     --tensor-parallel-size into train_minicpm4_8b_tp.py.
  4. CPU forward/backward smoke — a tiny GQA-faithful (32q/2kv) config runs
     fwd + bwd under a size-1 TP group, emits vocab-sharded logits, and the
     fp32 main_grad accumulators fill on both an embedding and a matrix
     weight.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_HAS_TORCH = importlib.util.find_spec("torch") is not None


def _load_toml(path: Path) -> dict:
    from harness._compat import tomllib

    return tomllib.loads(path.read_text(encoding="utf-8"))


class TestMiniCPM4_8BModelAxis(unittest.TestCase):
    def setUp(self) -> None:
        self.path = REPO_ROOT / "config" / "model" / "minicpm4_8b.toml"
        self.model = _load_toml(self.path)["model"]

    def test_geometry_matches_official_minicpm4_8b(self) -> None:
        m = self.model
        self.assertEqual(m["name"], "minicpm4_8b")
        self.assertEqual(m["num_layers"], 32)
        self.assertEqual(m["hidden_size"], 4096)
        self.assertEqual(m["ffn_hidden_size"], 16384)
        self.assertEqual(m["num_attention_heads"], 32)
        # GQA: 2 KV groups shared across 32 query heads (16:1).
        self.assertEqual(m["num_query_groups"], 2)
        self.assertEqual(m["head_dim"], 128)
        self.assertEqual(m["head_dim"], m["hidden_size"] // m["num_attention_heads"])
        self.assertEqual(m["padded_vocab_size"], 73448)
        self.assertEqual(m["max_position_embeddings"], 4096)
        self.assertEqual(m["rotary_base"], 10000)
        self.assertEqual(m["norm_epsilon"], 1e-6)
        self.assertEqual(m["init_method_std"], 0.1)

    def test_mup_on_mtp_off(self) -> None:
        m = self.model
        # muP ON: width_mult = hidden / dim_model_base = 4096/256 = 16.
        self.assertEqual(m["mup_base_hidden_size"], 256)
        self.assertEqual(m["hidden_size"] // m["mup_base_hidden_size"], 16)
        self.assertEqual(m["mup_emb_scale"], 12.0)
        self.assertEqual(m["mup_depth_scale"], 1.4)
        # MTP OFF: the official 8B config has no eagle/nextn head.
        self.assertEqual(m["mtp_num_layers"], 0)

    def test_vestigial_mup_denominator_absent(self) -> None:
        # The official config's mup_denominator=32 is vestigial — the forward
        # uses standard 1/sqrt(head_dim) attention and never references it, so
        # it must NOT be carried as a [model] key.
        self.assertNotIn("mup_denominator", self.model)

    def test_validates_against_shared_schema(self) -> None:
        from harness.config_runtime import _validate_model_config

        _validate_model_config(self.path, dict(self.model))


class TestMiniCPM4_8BRefAxis(unittest.TestCase):
    def test_ref_toml_points_at_dptp_launcher_and_tokenizer(self) -> None:
        ref = _load_toml(REPO_ROOT / "config" / "ref" / "torch_minicpm4_8b.toml")["ref"]
        self.assertEqual(ref["backend"], "torch")
        self.assertEqual(ref["ref_script"], "run_minicpm4_8b_dptp.sh")
        self.assertIn("MiniCPM4-8B", ref["tokenizer"])
        self.assertTrue(ref["forge_tokenizer_dir"])

    def test_launcher_exists(self) -> None:
        self.assertTrue((REPO_ROOT / "ref" / "reference" / "run_minicpm4_8b_dptp.sh").is_file())


class TestMiniCPM4_8BLauncher(unittest.TestCase):
    def setUp(self) -> None:
        self.text = (REPO_ROOT / "ref" / "reference" / "run_minicpm4_8b_dptp.sh").read_text(
            encoding="utf-8"
        )

    def test_no_in_launcher_registry_sourcing(self) -> None:
        # Generic projection contract: the caller projects the product;
        # the launcher consumes fail-fast generic names only.
        self.assertNotIn("gate_product_to_shell.py", self.text)
        self.assertNotRegex(self.text, r"_OVERRIDE")
        self.assertNotRegex(self.text, r"FORGE_NUM_LAYERS|FORGE_HEAD_DIM|FORGE_MUP_")
        self.assertRegex(self.text, r"NUM_STEPS:\?")

    def test_threads_tp_degree(self) -> None:
        # TP degree comes from the product's optional tensor_parallel_size key
        # (generic projection name TENSOR_PARALLEL_SIZE, default 1 = dense
        # proxy) and is passed as the --tensor-parallel-size CLI flag.
        self.assertIn("TENSOR_PARALLEL_SIZE", self.text)
        self.assertIn("--tensor-parallel-size", self.text)

    def test_targets_8b_python_entry(self) -> None:
        self.assertIn("train_minicpm4_8b_tp.py", self.text)


@unittest.skipUnless(_HAS_TORCH, "requires torch for the 8B model smoke")
class TestMiniCPM4_8BModelForward(unittest.TestCase):
    """CPU smoke: a tiny GQA-faithful config runs fwd+bwd under a size-1 TP
    group; flash_attn is mocked with an SDPA causal reference."""

    def _import_model_module(self):
        ref_dir = REPO_ROOT / "ref" / "reference"
        if str(ref_dir) not in sys.path:
            sys.path.insert(0, str(ref_dir))
        # model_minicpm4_8b_tp reads its geometry fail-fast from os.environ at
        # import (generic product projection, bare upper-cased names); set a
        # tiny GQA-faithful (4 query heads / 2 KV groups) shape and re-import.
        # The bare names are shared with the 0.5B ref modules, so restore the
        # prior values on cleanup instead of leaking the tiny shape into other
        # tests' imports.
        geom = {
            "NUM_LAYERS": "2",
            "HIDDEN_SIZE": "32",
            "NUM_ATTENTION_HEADS": "4",
            "NUM_QUERY_GROUPS": "2",
            "HEAD_DIM": "8",
            "FFN_HIDDEN_SIZE": "16",
            "PADDED_VOCAB_SIZE": "40",
            "MAX_POSITION_EMBEDDINGS": "16",
            "NORM_EPSILON": "1e-6",
            "ROTARY_BASE": "10000",
        }
        saved = {k: os.environ.get(k) for k in geom}

        def _restore() -> None:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        self.addCleanup(_restore)
        os.environ.update(geom)
        sys.modules.pop("model_minicpm4_8b_tp", None)
        import model_minicpm4_8b_tp as M

        return M

    def _install_flash_attn_mock(self) -> None:
        import torch.nn.functional as F

        fa = types.ModuleType("flash_attn")

        def flash_attn_func(q, k, v, causal=True, deterministic=True):
            hq, hkv = q.shape[2], k.shape[2]
            rep = hq // hkv
            qt = q.transpose(1, 2)
            kt = k.transpose(1, 2).repeat_interleave(rep, dim=1)
            vt = v.transpose(1, 2).repeat_interleave(rep, dim=1)
            o = F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal)
            return o.transpose(1, 2)

        fa.flash_attn_func = flash_attn_func
        sys.modules["flash_attn"] = fa

    def test_forward_backward_vocab_sharded_and_fp32_wgrad(self) -> None:
        import torch
        import torch.distributed as dist

        self._install_flash_attn_mock()
        M = self._import_model_module()

        os.environ.update(
            MASTER_ADDR="localhost",
            MASTER_PORT="29699",
            RANK="0",
            WORLD_SIZE="1",
            LOCAL_RANK="0",
        )
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo", rank=0, world_size=1)
        try:
            from evals import harness_dptp

            lay = harness_dptp.init_groups(1, 1)
            self.assertEqual((lay.dp_size, lay.tp_size, lay.global_rank), (1, 1, 0))

            model = M.MiniCPM4_8B_TP(vocab_size=40, tp_group=lay.tp_group)
            model.init_weights(seed=1234, init_ones=False)
            # tp=1 → the output layer owns the whole vocab.
            self.assertEqual(model.output.vocab_per_rank, 40)
            self.assertEqual(model.output.vocab_start, 0)

            rope = M.precompute_rope_freqs(8, device="cpu")
            ids = torch.randint(0, 40, (2, 8))
            logits = model(ids, rope)
            # tp=1 → full vocab in the (only) shard.
            self.assertEqual(tuple(logits.shape), (2, 8, 40))

            labels = torch.randint(0, 40, (2, 8))
            nll = M.vocab_parallel_cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), model.output
            )
            nll.sum().backward()

            # fp32 wgrad accumulators must fill on both an embedding and a
            # matrix weight (the custom autograd Functions route main_grad).
            emb_g = getattr(model.tok_embeddings.weight, "main_grad", None)
            self.assertIsNotNone(emb_g)
            self.assertTrue(torch.isfinite(emb_g).all())
            wqkv = model.layers[0].wqkv.weight
            wqkv_g = getattr(wqkv, "main_grad", None)
            self.assertIsNotNone(wqkv_g)
            self.assertTrue(torch.isfinite(wqkv_g).all())
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
