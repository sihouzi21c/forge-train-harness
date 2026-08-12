"""Stage B transport for the ours-side gate runners (``eval_*.py``).

The rendered, freeze-filled ours product (``workload/src/config/<gate>.toml``)
is the SOLE source for every gate runner's inputs. The dispatcher passes the
product's path as the single CLI argument ``--config <path>`` — nothing else
about the gate's shape, thresholds, or deployment values crosses the process
boundary. This module loads that product and resolves each requested key from
it.

Boundary (§D7, "product [env] is SSOT"):

* ``[cli]`` (bare lowercase) carries gate shape + model/optim/muP/MTP scalars.
* ``[env]`` (UPPER) carries real values: the A-class ones the renderer knows
  statically (BACKEND, NUM_PROCS, MASTER_ADDR, determinism flags) plus the
  B-class deployment values (CHECKPOINT_ROOT, MEGATRON_ROOT, MASTER_PORT) that
  the freeze step (``tools/resolve_deploy.py``) fills before the config is
  frozen, so a shipped product carries no ``<runtime>`` sentinel.
* C-class process keys never live in the product at all (``_RUNTIME_ENV_KEYS``):
  the per-process torchrun rendezvous rank identity (RANK / LOCAL_RANK /
  WORLD_SIZE) that ``launch_dp.py`` assigns each spawned child — note the
  product's ``world_size`` is the gate's intended DP count, not the entry's
  process-group size.

os.environ fallback is NOT unconditional: a key absent from the product reads
the env only when the product explicitly deferred it (the ``<runtime>``
sentinel in ``[env]``) or when it is on the declared env-transport whitelist
(``_ENV_FALLBACK_KEYS`` — values the runner sh derives per run, plus DATA_PATH
whose SSOT is the data axis and which the renderer deliberately keeps out of
the product). Everything else resolves from the frozen product alone, so a
stray environment variable can never shadow a missing product key.

Same-directory import (``from _gate_entry import gate_inputs``), matching the
``_runner_utils`` convention — ``evals/scripts/`` has no package.
"""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

RUNTIME = "<runtime>"

# C-class per-process rendezvous keys: the rank identity ``launch_dp.py``
# assigns each spawned child. Never gate shape, never in the product, always
# env-sourced. Every other key is resolved from the product first (falling
# through to os.environ only when the product does not carry it).
_RUNTIME_ENV_KEYS = frozenset({"RANK", "LOCAL_RANK", "WORLD_SIZE"})

# Per-RUN override whitelist (thin-dispatcher §5): values the gate runner sh
# derives per run/segment and passes via env, WINNING over the frozen product.
# Only the production segment loop actually overrides a product-carried key
# (NUM_STEPS: product = total run steps, env = this segment's slice); the
# other three never appear in a product, so env-first == the old fallback.
# Every remaining key stays product-first — the frozen product is the SSOT
# for static shape/thresholds.
_PER_RUN_ENV_FIRST = frozenset({"NUM_STEPS", "START_STEP", "FORGE_SAVE_PATH", "FORGE_RESUME_FROM"})

# Declared env-transport keys: the ONLY names for which a product-absent read
# may fall through to os.environ (besides an explicit "<runtime>" sentinel in
# the product's [env]). Two sources:
#   * the per-run values above (the runner sh derives them per run/segment);
#   * run-dir pointers the runner sh exports (RESUME_SCRATCH_DIR,
#     FORGE_CAPTURE_OUTPUT_FILE) and DATA_PATH, whose SSOT is the data axis
#     (config/data.toml) — the renderer deliberately renders NO data value
#     into the product (see render_gate_configs._build_baseline), so
#     runtime_env.py delivers it via env.
_ENV_FALLBACK_KEYS = _PER_RUN_ENV_FIRST | frozenset(
    {"DATA_PATH", "RESUME_SCRATCH_DIR", "FORGE_CAPTURE_OUTPUT_FILE"}
)


def _is_runtime_key(name: str) -> bool:
    return name in _RUNTIME_ENV_KEYS


def _config_path_from_argv(argv: list[str]) -> str | None:
    """Extract ``--config <path>`` (or ``--config=<path>``) from ``argv``.

    Read from argv rather than a parsed Namespace so it resolves at import
    time (runners build their ``GateInputs`` at module load, before their own
    ``argparse`` runs). Extra runner args (``--hash-*``) are ignored here.
    """
    for i, tok in enumerate(argv):
        if tok == "--config":
            return argv[i + 1] if i + 1 < len(argv) else None
        if tok.startswith("--config="):
            return tok[len("--config=") :]
    return None


class GateInputs:
    """Resolve a gate runner's inputs from its rendered product (§D7)."""

    def __init__(self, config_path: str | None = None) -> None:
        if config_path is None:
            config_path = _config_path_from_argv(sys.argv[1:])
        if not config_path:
            raise RuntimeError("ours gate runners require --config <product.toml>.")
        path = Path(config_path)
        if not path.exists():
            raise RuntimeError(
                f"gate product not found: {path} "
                "(render it first with tools/render_gate_configs.py)."
            )
        with open(path, "rb") as fh:
            doc = tomllib.load(fh)
        self._cli: dict[str, object] = dict(doc.get("cli", {}))
        self._env: dict[str, object] = dict(doc.get("env", {}))
        self.product_path = path

    def get(self, name: str, default: str | None = None) -> str | None:
        if name in _PER_RUN_ENV_FIRST and name in os.environ:
            return os.environ[name]
        if _is_runtime_key(name):
            return os.environ.get(name, default)
        # name is the UPPER env-style key the runner asks for; product [env]
        # is UPPER, product [cli] is bare lowercase. A "<runtime>" sentinel in
        # [env] means the renderer explicitly deferred this value to
        # os.environ (§D7).
        declared_runtime = False
        if name in self._env:
            value = self._env[name]
            if value == RUNTIME:
                declared_runtime = True
            else:
                return str(value)
        else:
            low = name.lower()
            if low in self._cli:
                return str(self._cli[low])
        if declared_runtime or name in _ENV_FALLBACK_KEYS:
            return os.environ.get(name, default)
        return default

    def require(self, name: str) -> str:
        value = self.get(name)
        if not value:
            raise RuntimeError(
                f"{name} is not set. It must be a [cli]/[env] key of the ours "
                f"product ({self.product_path}), or a runtime key in the env."
            )
        return value

    def checkpoint_root(self) -> str:
        """Ours-side canonical checkpoint dir — read straight from the product.

        The product's ``[env].CHECKPOINT_ROOT`` is the FINAL repo-relative value
        the freeze step (``tools/resolve_deploy.py``) fills, already including
        the gate's ``ones``/``no1`` subdir (design §0: values are WRITTEN into
        the config at freeze, ours only READS them — no consumer-side lift or
        ones/no1 derivation). It resolves against the engine's ``cwd`` (=
        ``repo_root`` on the execution machine), so a bare relative string is
        exactly what the engine consumes.
        """
        return self.require("CHECKPOINT_ROOT")


def gate_inputs() -> GateInputs:
    """Build the resolver once per runner process."""
    return GateInputs()
