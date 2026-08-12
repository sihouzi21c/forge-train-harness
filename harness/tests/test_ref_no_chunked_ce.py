"""Anti-regression: the dedicated reference must stay naive pre-resume.

Pre-resume the engine still operates under the bitwise-identical contract.
The earlier ``chunked_cross_entropy`` path in
``ref/reference/train_pure_mup_mtp.py`` split logits across the sequence
axis and accumulated ``output.weight.grad`` from 8 partials (4 chunks ×
{main, MTP}) via ``torch.utils.checkpoint`` + autograd. fp32 add is
non-associative, so any bitwise mirror in the engine had to reproduce
autograd's exact sequence-nr-driven traversal order — a brittle implicit
contract that previously cost ~22 min of dev-agent exploration per loop
just to discover.

The dedicated reference is by design the simplest possible trainer; it
does NOT need to support large micro batches or any memory optimization
pre-resume. So chunked CE is rolled back and the symbols below are pinned
absent. Any future change that needs chunked CE belongs to resume-onward work,
where the bitwise constraint is relaxed.
"""

from __future__ import annotations

import unittest
from pathlib import Path

REF_DIR = Path(__file__).resolve().parents[2] / "ref" / "reference"
TRAIN_PY = REF_DIR / "train_pure_mup_mtp.py"
MODEL_PY = REF_DIR / "model_pure_mup_mtp.py"

FORBIDDEN_TOKENS = (
    "chunked_cross_entropy",
    "_chunk_ce_forward",
    "--chunked-ce",
    "--no-chunked-ce",
    "--ce-chunk-size",
    "return_hidden",
)


class TestRefNoChunkedCE(unittest.TestCase):
    def test_train_script_has_no_chunked_ce(self) -> None:
        text = TRAIN_PY.read_text(encoding="utf-8")
        for token in FORBIDDEN_TOKENS:
            with self.subTest(token=token):
                self.assertNotIn(
                    token,
                    text,
                    msg=(
                        f"{TRAIN_PY.name} reintroduced {token!r}; the "
                        "dedicated reference must stay on the naive "
                        "masked_ce path pre-resume (see module docstring)."
                    ),
                )

    def test_model_script_has_no_return_hidden(self) -> None:
        text = MODEL_PY.read_text(encoding="utf-8")
        for token in ("return_hidden", "chunked_cross_entropy", "--chunked-ce"):
            with self.subTest(token=token):
                self.assertNotIn(
                    token,
                    text,
                    msg=(
                        f"{MODEL_PY.name} reintroduced {token!r}; the "
                        "model must always return logits, never raw "
                        "pre-LM-head hidden, pre-resume."
                    ),
                )


if __name__ == "__main__":
    unittest.main()
