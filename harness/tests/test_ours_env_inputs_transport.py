"""Transport contract: a gate that has a rendered product must NOT declare
any product-sourced scalar in its ours ``env_inputs``.

``env_inputs`` is the OURS-side "lift into the subprocess env" list (consumed
only by ``evals._common.suite_process_env``; the ref side gets its values from
the generic product projection ``tools/product_env.py`` and never reads
``env_inputs``). After the
gate-config refactor the gate shape + model/optim + verdict scalars live in the
rendered ours product (``workload/src/config/<gate>.toml``), which the engine
reads via ``_gate_entry.GateInputs`` / ``runtime_config`` — NOT from env. Listing
those scalars in ``env_inputs`` makes ``build_suite_env`` try to env-inject a
value that no longer has a cfg source → ``Declared env input X has no source``
(the crash loop 107d15ca hit on SEED / DATA_LOADER).

So: for every gate that HAS a ``gate_config/<gate>.toml`` (i.e. produces a
product), its ``env_inputs`` may carry only runtime/deployment keys, never a
product-sourced scalar. Product-less gates (op-* stage2, resume-startup) legitimately
keep env transport and are exempt (they have no gate_config). CPU-only check.
"""

from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

_HARNESS = Path(__file__).resolve().parents[2]
_EVAL_ROOT = _HARNESS / "config" / "eval"

# Scalars the renderer projects into the ours product (gate shape + data +
# verdict + backend). The engine reads these from the product, so they must not
# be env-injected to ours. (Model/optim hyperparameters are likewise product-only
# but never appear in env_inputs, so the explicit shape/data set suffices.)
_PRODUCT_SCALARS = frozenset(
    {
        "SEED",
        "MICRO_BATCH_SIZE",
        "SEQ_LENGTH",
        "DATA_LOADER",
        "NUM_STEPS",
        "GRAD_ACCUM_STEPS",
        "GATE_ATOL",
        "GLOBAL_BATCH_SIZE",
        "DETERMINISTIC",
        "BACKEND",
        "RESUME_SAVE_STEP",
    }
)


def _nested_suites() -> list[Path]:
    """Suite dirs whose registry declares a [suite] table + a gate_config dir."""
    out = []
    if not _EVAL_ROOT.is_dir():
        return out
    for d in sorted(_EVAL_ROOT.iterdir()):
        reg = d / f"{d.name}.toml"
        if not reg.is_file() or not (d / "gate_config").is_dir():
            continue
        with open(reg, "rb") as fh:
            if "suite" in tomllib.load(fh):
                out.append(d)
    return out


class OursEnvInputsTransport(unittest.TestCase):
    def test_product_gates_carry_no_product_scalar_in_env_inputs(self) -> None:
        suites = _nested_suites()
        self.assertTrue(suites, f"no nested eval suites under {_EVAL_ROOT}")
        for suite in suites:
            with open(suite / f"{suite.name}.toml", "rb") as fh:
                reg = tomllib.load(fh)
            gate_dir = suite / "gate_config"
            for gate, entry in reg.get("evals", {}).items():
                # Only gates that produce a rendered product are constrained;
                # product-less gates (op-*/resume-startup) keep env transport.
                if not (gate_dir / f"{gate}.toml").is_file():
                    continue
                env_inputs = {str(k).upper() for k in entry.get("env_inputs", [])}
                leaked = env_inputs & _PRODUCT_SCALARS
                with self.subTest(suite=suite.name, gate=gate):
                    self.assertEqual(
                        leaked,
                        set(),
                        f"{suite.name}/{gate}: env_inputs must not env-inject "
                        f"product-sourced scalars {sorted(leaked)} — ours reads "
                        "them from the rendered product.",
                    )


if __name__ == "__main__":
    unittest.main()
