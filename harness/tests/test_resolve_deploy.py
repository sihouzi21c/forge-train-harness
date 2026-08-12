"""Unit tests for the freeze-time deployment fill (``tools.resolve_deploy``).

Step 1 of the gate-config file-ification: a pure function that replaces the
renderer's ``<runtime>`` sentinels (MASTER_PORT / CHECKPOINT_ROOT /
MEGATRON_ROOT) with machine-independent deployment values, producing a
zero-``<runtime>`` frozen product. These are CPU/off-GPU checks; wiring into the
live freeze + the matching consumer change is a later step.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from harness.tests import _gate_render  # noqa: E402
from tools import resolve_deploy as rd  # noqa: E402


class DeriveMasterPortTests(unittest.TestCase):
    def test_deterministic_and_in_range(self) -> None:
        a = rd.derive_master_port("loop-abc123")
        b = rd.derive_master_port("loop-abc123")
        self.assertEqual(a, b)
        self.assertTrue(rd._PORT_BASE <= a < rd._PORT_BASE + rd._PORT_SPAN)

    def test_distinct_loops_usually_differ(self) -> None:
        self.assertNotEqual(rd.derive_master_port("loop-aaa"), rd.derive_master_port("loop-bbb"))

    def test_empty_loop_id_rejected(self) -> None:
        with self.assertRaises(rd.ResolveError):
            rd.derive_master_port("")


class FillProductTextTests(unittest.TestCase):
    _SAMPLE = (
        "[cli]\n"
        'name = "x"\n'
        "world_size = 1\n"
        "\n"
        "[env]\n"
        'BACKEND = "torch"\n'
        'CHECKPOINT_ROOT = "<runtime>"\n'
        'MASTER_ADDR = "localhost"\n'
        'MASTER_PORT = "<runtime>"\n'
    )

    def test_fills_env_only_and_reports_clean(self) -> None:
        res = {"CHECKPOINT_ROOT": ".artifacts/checkpoints/torch", "MASTER_PORT": "29517"}
        out, unresolved = rd.fill_product_text(self._SAMPLE, res)
        self.assertEqual(unresolved, [])
        self.assertNotIn(rd.RUNTIME, out)
        self.assertIn('CHECKPOINT_ROOT = ".artifacts/checkpoints/torch"', out)
        self.assertIn('MASTER_PORT = "29517"', out)
        # untouched real values survive
        self.assertIn('MASTER_ADDR = "localhost"', out)
        self.assertIn('BACKEND = "torch"', out)

    def test_unresolved_sentinel_reported(self) -> None:
        out, unresolved = rd.fill_product_text(self._SAMPLE, {"MASTER_PORT": "29517"})
        self.assertEqual(unresolved, ["CHECKPOINT_ROOT"])
        self.assertIn(rd.RUNTIME, out)  # left as-is when no resolution

    def test_cli_runtime_is_never_touched(self) -> None:
        # A <runtime> in [cli] (shouldn't happen, but guard the section scope).
        text = '[cli]\nfoo = "<runtime>"\n\n[env]\nMASTER_PORT = "<runtime>"\n'
        out, unresolved = rd.fill_product_text(text, {"MASTER_PORT": "1"})
        self.assertIn('foo = "<runtime>"', out)
        self.assertEqual(unresolved, [])


class ComputeResolutionsTests(unittest.TestCase):
    def _cfg_with_backend(self, backend: str) -> Path:
        import tempfile

        d = Path(tempfile.mkdtemp())
        (d / "ref.toml").write_text(f'[ref]\nbackend = "{backend}"\n')
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        return d

    def test_torch_backend(self) -> None:
        res = rd.compute_resolutions(self._cfg_with_backend("torch"), "loop-1")
        self.assertEqual(res["CHECKPOINT_ROOT"], ".artifacts/checkpoints/torch")
        self.assertEqual(res["MEGATRON_ROOT"], "")  # unused for torch
        self.assertTrue(res["MASTER_PORT"].isdigit())

    def test_megatron_backend_gets_submodule(self) -> None:
        res = rd.compute_resolutions(self._cfg_with_backend("megatron"), "loop-1")
        self.assertEqual(res["CHECKPOINT_ROOT"], ".artifacts/checkpoints/megatron")
        self.assertEqual(res["MEGATRON_ROOT"], rd._MEGATRON_SUBMODULE_REL)

    def test_missing_backend_rejected(self) -> None:
        import tempfile

        d = Path(tempfile.mkdtemp())
        (d / "ref.toml").write_text("[ref]\n")
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        with self.assertRaises(rd.ResolveError):
            rd.compute_resolutions(d, "loop-1")


class ProductResolutionsTests(unittest.TestCase):
    """Per-product ones/no1 suffixing of the base CHECKPOINT_ROOT."""

    _BASE = {
        "CHECKPOINT_ROOT": ".artifacts/checkpoints/torch",
        "MASTER_PORT": "29517",
        "MEGATRON_ROOT": "",
    }

    def _product(self, body: str) -> Path:
        import tempfile

        d = Path(tempfile.mkdtemp())
        p = d / "gate.toml"
        p.write_text(body)
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        return p

    def test_forge_init_ones_read(self) -> None:
        self.assertEqual(rd._forge_init_ones(self._product("[cli]\nforge_init_ones = 1\n")), 1)
        self.assertEqual(rd._forge_init_ones(self._product("[cli]\nforge_init_ones = 0\n")), 0)

    def test_forge_init_ones_absent_is_none(self) -> None:
        self.assertIsNone(rd._forge_init_ones(self._product('[cli]\nname = "x"\n')))

    def test_ones_appends_ones_subdir(self) -> None:
        res = rd._product_resolutions(self._BASE, self._product("[cli]\nforge_init_ones = 1\n"))
        self.assertEqual(res["CHECKPOINT_ROOT"], ".artifacts/checkpoints/torch/ones")
        # gate-independent keys pass through unchanged
        self.assertEqual(res["MASTER_PORT"], "29517")

    def test_no1_appends_no1_subdir(self) -> None:
        res = rd._product_resolutions(self._BASE, self._product("[cli]\nforge_init_ones = 0\n"))
        self.assertEqual(res["CHECKPOINT_ROOT"], ".artifacts/checkpoints/torch/no1")

    def test_absent_forge_init_ones_keeps_bare_root(self) -> None:
        res = rd._product_resolutions(self._BASE, self._product('[cli]\nname = "x"\n'))
        self.assertEqual(res["CHECKPOINT_ROOT"], ".artifacts/checkpoints/torch")

    def test_does_not_mutate_base(self) -> None:
        rd._product_resolutions(self._BASE, self._product("[cli]\nforge_init_ones = 1\n"))
        self.assertEqual(self._BASE["CHECKPOINT_ROOT"], ".artifacts/checkpoints/torch")


class EndToEndRenderThenFillTests(unittest.TestCase):
    """Render the real dense_training products, fill, assert zero <runtime>."""

    def test_fill_clears_all_sentinels(self) -> None:
        rendered = _gate_render.render_variant("dense_training")
        if rendered is None:
            self.skipTest(
                "render inputs absent: "
                + ", ".join(_gate_render.missing_render_inputs("dense_training"))
            )
        self.addCleanup(rendered.cleanup)
        ws = rendered.workspace
        cfg_dir = ws.parent / "config"

        # Pre-condition: products carry the three sentinels.
        dirs = rd._default_product_dirs(ws)
        before = "".join(p.read_text() for d in dirs for p in sorted(d.glob("*.toml")))
        self.assertIn(rd.RUNTIME, before)

        resolutions = rd.compute_resolutions(cfg_dir, "loop-e2e")
        rd.fill_products(dirs, resolutions, require_complete=True)  # raises if leftover

        base_ckpt = resolutions["CHECKPOINT_ROOT"]
        for d in dirs:
            for product in sorted(d.glob("*.toml")):
                txt = product.read_text()
                self.assertNotIn(rd.RUNTIME, txt, f"{product} still has a sentinel after fill")
                self.assertIn(f'MASTER_PORT = "{resolutions["MASTER_PORT"]}"', txt)
                # CHECKPOINT_ROOT is filled per-product: each gate's own
                # forge_init_ones appends the matching ones/no1 subdir (a gate
                # with no forge_init_ones keeps the bare base root).
                ones = rd._forge_init_ones(product)
                expected = (
                    base_ckpt if ones is None else (f"{base_ckpt}/{'ones' if ones == 1 else 'no1'}")
                )
                self.assertIn(f'CHECKPOINT_ROOT = "{expected}"', txt)


if __name__ == "__main__":
    unittest.main()
