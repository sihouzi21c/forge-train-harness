"""CPU contract tests for ``training_engine_tensor.parameters``.

These tests pin the canonical-state ↔ engine parameter-view contract:
``canonical_state_fp32.pt`` is produced by
:func:`evals.harness_hook.install_canonical_state_dump`, which keys
every tensor by ``model.named_parameters()`` FQN **verbatim** (see
``evals/harness_hook/_canonical_state.py``).  For the pure-torch ref
stack (``ref/reference/model_pure_mup_mtp.py``) those FQNs are::

    tok_embeddings.weight
    layers.<i>.attention_norm.weight
    layers.<i>.wqkv.weight
    layers.<i>.wo.weight
    layers.<i>.ffn_norm.weight
    layers.<i>.wfc1.weight
    layers.<i>.w2.weight
    norm.weight
    output.weight
    # MTP / Eagle (when --eagle-num-layers >= 1, the ref preset default)
    mtp.emb_input_layernorm.weight
    mtp.hidden_input_layernorm.weight
    mtp.eagle_fc.weight
    mtp.layer.attention_norm.weight
    mtp.layer.wqkv.weight
    mtp.layer.wo.weight
    mtp.layer.ffn_norm.weight
    mtp.layer.wfc1.weight
    mtp.layer.w2.weight
    mtp.final_layernorm.weight

Earlier revisions used an alien (non-ref) FQN scheme
(``decoder.layers.<i>.self_attention.linear_qkv.weight`` etc.); those
keys never appear in ``canonical_state_fp32.pt`` for this backend, so
the loader fail-fast on every Stage 1 gate.  These tests lock the contract
to the actual ref FQNs so the regression cannot recur silently.

Skipped when the engine package is absent (gitignored on a fresh
harness clone) or when ``HARNESS_RUN_ENGINE_CONTRACTS`` is not set,
matching the rest of ``test_training*.py``.
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path


def _ensure_engine_on_path() -> None:
    workspace = Path(__file__).resolve().parents[2]
    candidate = workspace / "workload" / "src"
    if candidate.is_dir():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


_ensure_engine_on_path()


def _can_import(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except (ImportError, ModuleNotFoundError):
        return False


_RUN_ENGINE_CONTRACTS = os.environ.get("HARNESS_RUN_ENGINE_CONTRACTS") == "1"
_HAS_TORCH = _can_import("torch")
_HAS_ENGINE = _can_import("training_engine_tensor.parameters")


def _make_canonical_state_dict(*, eagle_num_layers: int):
    """Build a ref-FQN canonical state with correct shapes (deterministic).

    Tensors are FP32 with values derived from a per-key seed so the test
    can also assert tensor identity round-trips through the loader.
    """

    import torch
    from training_engine_tensor import config

    qkv_dim = config.NUM_HEADS * config.HEAD_DIM + 2 * config.NUM_KV_HEADS * config.HEAD_DIM

    def _t(seed: int, shape):
        gen = torch.Generator().manual_seed(seed)
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    state: dict[str, torch.Tensor] = {
        "tok_embeddings.weight": _t(1, (config.VOCAB_SIZE, config.HIDDEN_SIZE)),
        "norm.weight": _t(2, (config.HIDDEN_SIZE,)),
        "output.weight": _t(3, (config.VOCAB_SIZE, config.HIDDEN_SIZE)),
    }
    for i in range(config.NUM_LAYERS):
        state[f"layers.{i}.attention_norm.weight"] = _t(100 + 6 * i, (config.HIDDEN_SIZE,))
        state[f"layers.{i}.wqkv.weight"] = _t(101 + 6 * i, (qkv_dim, config.HIDDEN_SIZE))
        state[f"layers.{i}.wo.weight"] = _t(
            102 + 6 * i, (config.HIDDEN_SIZE, config.NUM_HEADS * config.HEAD_DIM)
        )
        state[f"layers.{i}.ffn_norm.weight"] = _t(103 + 6 * i, (config.HIDDEN_SIZE,))
        state[f"layers.{i}.wfc1.weight"] = _t(
            104 + 6 * i, (2 * config.FFN_HIDDEN_SIZE, config.HIDDEN_SIZE)
        )
        state[f"layers.{i}.w2.weight"] = _t(
            105 + 6 * i, (config.HIDDEN_SIZE, config.FFN_HIDDEN_SIZE)
        )

    if eagle_num_layers >= 1:
        state["mtp.emb_input_layernorm.weight"] = _t(900, (config.HIDDEN_SIZE,))
        state["mtp.hidden_input_layernorm.weight"] = _t(901, (config.HIDDEN_SIZE,))
        state["mtp.eagle_fc.weight"] = _t(902, (config.HIDDEN_SIZE, 2 * config.HIDDEN_SIZE))
        state["mtp.layer.attention_norm.weight"] = _t(910, (config.HIDDEN_SIZE,))
        state["mtp.layer.wqkv.weight"] = _t(911, (qkv_dim, config.HIDDEN_SIZE))
        state["mtp.layer.wo.weight"] = _t(
            912, (config.HIDDEN_SIZE, config.NUM_HEADS * config.HEAD_DIM)
        )
        state["mtp.layer.ffn_norm.weight"] = _t(913, (config.HIDDEN_SIZE,))
        state["mtp.layer.wfc1.weight"] = _t(914, (2 * config.FFN_HIDDEN_SIZE, config.HIDDEN_SIZE))
        state["mtp.layer.w2.weight"] = _t(915, (config.HIDDEN_SIZE, config.FFN_HIDDEN_SIZE))
        state["mtp.final_layernorm.weight"] = _t(920, (config.HIDDEN_SIZE,))
    return state


@unittest.skipUnless(
    _HAS_TORCH and _HAS_ENGINE and _RUN_ENGINE_CONTRACTS,
    "engine contract tests require torch + workload/src/training_engine_tensor + "
    "HARNESS_RUN_ENGINE_CONTRACTS=1",
)
class TestRefPureTorchFqnContract(unittest.TestCase):
    """The loader must accept ref-pure-torch ``named_parameters()`` FQNs."""

    def _save_state(self, tmp: Path, state: dict) -> Path:
        import torch
        from training_engine_tensor.parameters import CANONICAL_STATE_FILENAME

        out = tmp / CANONICAL_STATE_FILENAME
        torch.save(state, out)
        return out

    def test_loader_accepts_ref_natural_fqns_with_eagle_default(self):
        import torch
        from training_engine_tensor.parameters import (
            build_parameter_view,
            load_canonical_parameters,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            state = _make_canonical_state_dict(eagle_num_layers=1)
            self._save_state(tmp, state)

            loaded = load_canonical_parameters(tmp, eagle_num_layers=1)
            self.assertEqual(set(loaded), set(state))
            for name, tensor in loaded.items():
                self.assertEqual(tensor.dtype, torch.float32)
                self.assertEqual(tuple(tensor.shape), tuple(state[name].shape))

            view = build_parameter_view(loaded, eagle_num_layers=1)
            self.assertIsNotNone(view.mtp)
            self.assertIs(view.token_embedding, loaded["tok_embeddings.weight"])
            self.assertIs(view.final_norm_weight, loaded["norm.weight"])
            self.assertIs(view.output_weight, loaded["output.weight"])
            self.assertIs(
                view.layers[0].input_norm_weight,
                loaded["layers.0.attention_norm.weight"],
            )
            self.assertIs(view.layers[0].qkv_weight, loaded["layers.0.wqkv.weight"])
            self.assertIs(view.layers[0].attention_proj_weight, loaded["layers.0.wo.weight"])
            self.assertIs(view.layers[0].pre_mlp_norm_weight, loaded["layers.0.ffn_norm.weight"])
            self.assertIs(view.layers[0].mlp_fc1_weight, loaded["layers.0.wfc1.weight"])
            self.assertIs(view.layers[0].mlp_fc2_weight, loaded["layers.0.w2.weight"])
            mtp = view.mtp
            assert mtp is not None
            self.assertIs(mtp.emb_input_norm_weight, loaded["mtp.emb_input_layernorm.weight"])
            self.assertIs(mtp.hidden_input_norm_weight, loaded["mtp.hidden_input_layernorm.weight"])
            self.assertIs(mtp.eagle_fc_weight, loaded["mtp.eagle_fc.weight"])
            self.assertIs(mtp.final_norm_weight, loaded["mtp.final_layernorm.weight"])
            self.assertIs(mtp.transformer_layer.qkv_weight, loaded["mtp.layer.wqkv.weight"])

    def test_loader_supports_no_mtp_for_smoke(self):
        from training_engine_tensor.parameters import (
            build_parameter_view,
            load_canonical_parameters,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            state = _make_canonical_state_dict(eagle_num_layers=0)
            self._save_state(tmp, state)

            loaded = load_canonical_parameters(tmp, eagle_num_layers=0)
            view = build_parameter_view(loaded, eagle_num_layers=0)
            self.assertIsNone(view.mtp)

    def test_loader_rejects_alien_fqn_scheme(self):
        """A dump with non-ref FQNs must fail-fast — those keys do not
        exist for the pure-torch backend.
        """

        import torch
        from training_engine_tensor import config
        from training_engine_tensor.parameters import (
            CANONICAL_STATE_FILENAME,
            load_canonical_parameters,
        )

        # Build a state with the previous (non-ref) FQN scheme.  The
        # loader must reject it as "missing" the ref-pure-torch keys.
        bogus = {
            "embedding.word_embeddings.weight": torch.zeros(
                config.VOCAB_SIZE, config.HIDDEN_SIZE, dtype=torch.float32
            ),
            "decoder.final_layernorm.weight": torch.zeros(config.HIDDEN_SIZE, dtype=torch.float32),
            "output_layer.weight": torch.zeros(
                config.VOCAB_SIZE, config.HIDDEN_SIZE, dtype=torch.float32
            ),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / CANONICAL_STATE_FILENAME
            torch.save(bogus, out)
            with self.assertRaises(ValueError) as ctx:
                load_canonical_parameters(Path(tmpdir), eagle_num_layers=1)
            msg = str(ctx.exception)
            self.assertIn("tok_embeddings.weight", msg)

    def test_loader_rejects_dtype_mismatch(self):
        import torch
        from training_engine_tensor.parameters import (
            CANONICAL_STATE_FILENAME,
            load_canonical_parameters,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            state = _make_canonical_state_dict(eagle_num_layers=1)
            # Force one tensor to bf16 — must fail-fast (FP32 spec).
            state["tok_embeddings.weight"] = state["tok_embeddings.weight"].to(torch.bfloat16)
            torch.save(state, tmp / CANONICAL_STATE_FILENAME)
            with self.assertRaises(ValueError):
                load_canonical_parameters(tmp, eagle_num_layers=1)

    def test_loader_rejects_extra_keys(self):
        import torch
        from training_engine_tensor.parameters import (
            CANONICAL_STATE_FILENAME,
            load_canonical_parameters,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            state = _make_canonical_state_dict(eagle_num_layers=1)
            state["legacy.helper.weight"] = torch.zeros(8, dtype=torch.float32)
            torch.save(state, tmp / CANONICAL_STATE_FILENAME)
            with self.assertRaises(ValueError):
                load_canonical_parameters(tmp, eagle_num_layers=1)


@unittest.skipUnless(
    _HAS_TORCH and _HAS_ENGINE and _RUN_ENGINE_CONTRACTS,
    "engine contract tests require torch + workload/src/training_engine_tensor + "
    "HARNESS_RUN_ENGINE_CONTRACTS=1",
)
class TestParameterViewIsZeroCopyOverState(unittest.TestCase):
    """``build_parameter_view`` must reference (not copy) the loaded tensors.

    The loader copies once into the destination device/dtype; the view
    just groups by static-execution-graph role.  Re-copying inside the
    view would silently desync forward / backward writes from the
    canonical master state.
    """

    def test_view_holds_same_tensors(self):
        from training_engine_tensor.parameters import (
            build_parameter_view,
            load_canonical_parameters,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            state = _make_canonical_state_dict(eagle_num_layers=1)

            import torch
            from training_engine_tensor.parameters import CANONICAL_STATE_FILENAME

            torch.save(state, tmp / CANONICAL_STATE_FILENAME)
            loaded = load_canonical_parameters(tmp, eagle_num_layers=1)
            view = build_parameter_view(loaded, eagle_num_layers=1)
            self.assertIs(view.layers[0].mlp_fc1_weight, loaded["layers.0.wfc1.weight"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
