"""Model architecture constants — read from environment at import time.

Mirrors ``ref/reference/model_pure_mup_mtp.py`` module-level constants:
geometry values are read from upper-cased env vars that the harness
injects from the rendered gate product.  Missing keys raise ``KeyError``
(no baked-in defaults) so a render/projection gap is always fail-fast.

Every other engine module imports these constants at module level; they
are never re-read from the environment during training.
"""

from __future__ import annotations

import os

NUM_LAYERS = int(os.environ["NUM_LAYERS"])
HIDDEN_SIZE = int(os.environ["HIDDEN_SIZE"])
NUM_HEADS = int(os.environ["NUM_ATTENTION_HEADS"])
NUM_KV_HEADS = int(os.environ["NUM_QUERY_GROUPS"])
HEAD_DIM = int(os.environ["HEAD_DIM"])
FFN_HIDDEN_SIZE = int(os.environ["FFN_HIDDEN_SIZE"])
VOCAB_SIZE = int(os.environ["PADDED_VOCAB_SIZE"])
MAX_SEQ_LEN = int(os.environ["MAX_POSITION_EMBEDDINGS"])
NORM_EPS = float(os.environ["NORM_EPSILON"])
ROPE_THETA = float(os.environ["ROTARY_BASE"])

# Aliases for test compatibility
NORM_EPSILON = NORM_EPS