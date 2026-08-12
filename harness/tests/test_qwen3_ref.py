"""Contract + smoke tests for the Qwen3 0.6B reference onboarding.

Three layers, cheapest first (so CPU-only CI exercises everything but the
GPU forward/backward):

  1. [model] axis — config/model/qwen3_0.6b.toml validates against the
     shared schema with muP/MTP set to the neutral values the Qwen3
     reference ignores (mtp_num_layers = 0). QK-Norm / tied embeddings are
     intrinsic to the Qwen3 reference (hardcoded in model_qwen3.py), so the
     template MUST NOT carry them as [model] keys.
  2. [ref] axis — config/ref/torch_qwen3_0.6b.toml points at the Qwen3
     launcher + a Qwen tokenizer repo.
  3. Launcher script — run_qwen3_dense.sh consumes its full parameter set
     (shape + Qwen3 geometry) from the ENVIRONMENT under the generic
     upper-cased product-key names (projected by the caller via
     tools/product_env.py — no in-launcher registry sourcing), and
     model_qwen3.py reads the same names from os.environ at import.
     No baked MBS default, no fallback.
  4. (CUDA-gated) model_qwen3 forward + backward runs and the LM head is
     genuinely tied to the input embedding (hardcoded on, no env needed).
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_HAS_TORCH = importlib.util.find_spec("torch") is not None
_HAS_CUDA = False
if _HAS_TORCH:
    import torch

    _HAS_CUDA = torch.cuda.is_available()


def _load_toml(path: Path) -> dict:
    from harness._compat import tomllib

    return tomllib.loads(path.read_text(encoding="utf-8"))


class TestQwen3ModelAxis(unittest.TestCase):
    def setUp(self) -> None:
        self.path = REPO_ROOT / "config" / "model" / "qwen3_0.6b.toml"
        self.model = _load_toml(self.path)["model"]

    def test_geometry_matches_hf_qwen3_0_6b(self) -> None:
        m = self.model
        self.assertEqual(m["name"], "qwen3_0.6b")
        self.assertEqual(m["num_layers"], 28)
        self.assertEqual(m["hidden_size"], 1024)
        self.assertEqual(m["ffn_hidden_size"], 3072)
        self.assertEqual(m["num_attention_heads"], 16)
        self.assertEqual(m["num_query_groups"], 8)
        self.assertEqual(m["head_dim"], 128)
        self.assertEqual(m["padded_vocab_size"], 151936)
        self.assertEqual(m["rotary_base"], 1000000)
        # head_dim is decoupled from hidden//heads (1024//16 = 64 != 128) —
        # the whole reason head_dim is an explicit knob.
        self.assertNotEqual(m["head_dim"], m["hidden_size"] // m["num_attention_heads"])

    def test_neutral_mup_mtp(self) -> None:
        m = self.model
        # muP/MTP OFF: mtp_num_layers = 0 means no Eagle layer / no MTP loss.
        self.assertEqual(m["mtp_num_layers"], 0)

    def test_qk_norm_and_tie_are_not_model_keys(self) -> None:
        # QK-Norm and tied embeddings are intrinsic to the Qwen3 reference
        # (hardcoded in model_qwen3.py), NOT shared [model] knobs. Keeping
        # them out of the template is what lets the MiniCPM templates and the
        # engine ModelHParams contract stay untouched.
        self.assertNotIn("use_qk_norm", self.model)
        self.assertNotIn("tie_word_embeddings", self.model)

    def test_validates_against_shared_schema(self) -> None:
        from harness.config_runtime import _validate_model_config

        # Must not raise — the shared [model] schema accepts the Qwen3
        # template (same numeric keys as MiniCPM, muP/MTP at neutral values).
        _validate_model_config(self.path, dict(self.model))


class TestQwen3RefAxis(unittest.TestCase):
    def test_ref_toml_points_at_qwen3_launcher_and_tokenizer(self) -> None:
        ref = _load_toml(REPO_ROOT / "config" / "ref" / "torch_qwen3_0.6b.toml")["ref"]
        self.assertEqual(ref["backend"], "torch")
        self.assertEqual(ref["ref_script"], "run_qwen3_dense.sh")
        # Qwen tokenizer (HF fast tokenizer.json — no SentencePiece model).
        self.assertIn("Qwen", ref["tokenizer"])
        self.assertTrue(ref["forge_tokenizer_dir"])

    def test_launcher_exists(self) -> None:
        self.assertTrue((REPO_ROOT / "ref" / "reference" / "run_qwen3_dense.sh").is_file())


class TestQwen3Launcher(unittest.TestCase):
    def setUp(self) -> None:
        self.text = (REPO_ROOT / "ref" / "reference" / "run_qwen3_dense.sh").read_text(
            encoding="utf-8"
        )

    def test_mbs_comes_from_product_no_baked_default(self) -> None:
        # Collapse: the gate product is the SOLE MBS source — no baked-in
        # ``:-4`` fallback default survives. MBS is a required product key,
        # consumed under its generic upper-cased name.
        self.assertNotRegex(self.text, r"MICRO_BATCH_SIZE:-\d+")
        self.assertRegex(self.text, r"MICRO_BATCH_SIZE:\?")

    def test_consumes_generic_env_names_no_registry(self) -> None:
        # Geometry arrives as environment under the upper-cased product-key
        # names (caller-side generic projection); the launcher performs NO
        # in-launcher registry sourcing and reads NO legacy translated names.
        self.assertNotIn("gate_product_to_shell", self.text)
        self.assertNotRegex(self.text, r"FORGE_NUM_LAYERS|FORGE_HEAD_DIM")
        self.assertNotRegex(self.text, r"_OVERRIDE")
        self.assertRegex(self.text, r"HEAD_DIM:\?")
        # QK-Norm / tie are intrinsic to model_qwen3.py (hardcoded), so the
        # launcher MUST NOT export them — there is no such knob.
        self.assertNotRegex(self.text, r"export (FORGE_)?USE_QK_NORM=")
        self.assertNotRegex(self.text, r"export (FORGE_)?TIE_WORD_EMBEDDINGS=")

    @unittest.skipUnless(shutil.which("bash"), "bash required")
    def test_fails_fast_without_projected_env(self) -> None:
        # With no projected product env, the launcher must die on the first
        # required-key probe (set -u + :?), never fall back to a baked shape.
        import subprocess

        proc = subprocess.run(
            ["bash", str(REPO_ROOT / "ref" / "reference" / "run_qwen3_dense.sh")],
            # GPUS_PER_NODE pinned so the nvidia-smi autodetect (absent on
            # CPU-only CI) can't abort the run before the required-key probes.
            env={
                "PATH": os.environ.get("PATH", ""),
                "LOCAL_MODE": "1",
                "GPUS_PER_NODE": "2",
            },
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("num_steps", proc.stderr)

    def test_targets_qwen3_python_entry(self) -> None:
        self.assertIn("train_qwen3_dense.py", self.text)


@unittest.skipUnless(_HAS_CUDA, "requires CUDA + flash_attn for the Qwen3 model")
class TestQwen3ModelForward(unittest.TestCase):
    """GPU smoke test: a tiny Qwen3 config runs fwd+bwd and the tied head
    shares the embedding matrix object identity.
    """

    def _import_model_module(self):
        # model_qwen3 reads geometry from the generic upper-cased names at
        # import (fail-fast, no defaults); set a tiny shape so the test is
        # fast and fits any GPU. Import via spec so repeated runs with
        # different env re-evaluate the module-level constants.
        import sys

        ref_dir = REPO_ROOT / "ref" / "reference"
        if str(ref_dir) not in sys.path:
            sys.path.insert(0, str(ref_dir))
        os.environ.update(
            NUM_LAYERS="2",
            HIDDEN_SIZE="128",
            NUM_ATTENTION_HEADS="4",
            NUM_QUERY_GROUPS="2",
            HEAD_DIM="32",
            FFN_HIDDEN_SIZE="256",
            PADDED_VOCAB_SIZE="512",
            MAX_POSITION_EMBEDDINGS="64",
            ROTARY_BASE="1000000",
            NORM_EPSILON="1e-6",
        )
        for mod in ("model_qwen3", "model_pure_mup_mtp"):
            sys.modules.pop(mod, None)
        import model_qwen3

        return model_qwen3

    def test_forward_backward_and_tied_head(self) -> None:
        mq = self._import_model_module()
        device = "cuda"
        model = mq.Qwen3Dense()
        model.init_weights(init_std=0.02, seed=0)
        model = model.to(device=device, dtype=torch.bfloat16)
        # Tied head: no independent output matrix.
        self.assertIsNone(model.output)
        self.assertTrue(model.tie_word_embeddings)

        rope = mq.precompute_rope_freqs(64, device=device)
        ids = torch.randint(0, mq.VOCAB_SIZE, (2, 16), device=device)
        logits, mtp = model(ids, rope)
        self.assertIsNone(mtp)  # Qwen3 has no MTP branch
        self.assertEqual(tuple(logits.shape), (2, 16, mq.VOCAB_SIZE))

        loss = logits.float().mean()
        loss.backward()
        # The tied embedding weight must have received a gradient (it is
        # consumed both as the input embedding and as the LM head).
        emb_w = model.tok_embeddings.weight
        grad = getattr(emb_w, "main_grad", None)
        if grad is None:
            grad = emb_w.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())


if __name__ == "__main__":
    unittest.main()
