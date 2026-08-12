"""Stage 2 ``[automation.stage2]`` shell exports for the agent loop.

Stage 2 is dispatched through the standard ``run_agent_stage`` milestone
pipeline (see ``prompt/develop_prompt/stage2.md``); this module owns:

1. ``STAGE2_MAX_CONCURRENT`` / ``STAGE2_MAX_OP_LONG_FAILURES`` shell
   exports consumed by ``agent-loop.sh`` while building the M1/M2
   prompts.
2. A thin ``operator_names`` helper still used by tests; production code
   reads the registry via ``harness.config_runtime.ops_registry_path``.

``MODEL`` is owned by ``agent_loop_config.py`` (reads from the run-level
config ``config/agent.toml [agent].model``).

Each subagent process runs a single per-round linear decision tree and
exits; cross-round progression is driven by ``agent-loop.sh`` re-fan-out.
``max_op_long_failures`` is the per-op safety net that bounds runaway
op-long retries across rounds (see
``prompt/develop_prompt/_shared/stage2-subagent-playbook.md`` for the per-round
decision tree and the merge wrapper).
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from harness import config_runtime
from harness._compat import tomllib

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def _repo_root() -> Path:
    return config_runtime.repo_root()


REPO_ROOT = _repo_root()


def load_stage2_automation() -> dict[str, Any]:
    _, data = config_runtime.load_workload_config(None)
    cfg = data.get("automation", {}).get("stage2", {})
    if not isinstance(cfg, dict):
        raise ValueError("config/eval.toml [automation.stage2] must be a table")
    return cfg


def registry_path() -> Path:
    return config_runtime.ops_registry_path(REPO_ROOT)


def operator_names() -> list[str]:
    """Return registered operator names sorted by priority (helper for tests)."""
    path = registry_path()
    if not path.exists():
        raise FileNotFoundError(f"Registry not found: {path}")
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    operators = data.get("operators", {})
    return sorted(operators.keys(), key=lambda name: operators[name].get("priority", 999))


def shell_exports(env: Mapping[str, str] | None = None) -> str:
    cfg = load_stage2_automation()
    values = {
        "STAGE2_MAX_CONCURRENT": str(cfg["max_concurrent"]),
        "STAGE2_MAX_OP_LONG_FAILURES": str(cfg["max_op_long_failures"]),
    }
    return "\n".join(f"export {key}={shell_quote(value)}" for key, value in values.items())


def shell_quote(value: str) -> str:
    """Single-quote a value for safe shell interpolation."""
    return "'" + value.replace("'", "'\"'\"'") + "'"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if args == ["shell"]:
            print(shell_exports())
            return 0
        if args == ["operator-names"]:
            print("\n".join(operator_names()))
            return 0
        if args == ["operator-count"]:
            print(len(operator_names()))
            return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("Usage: stage2_config.py {shell|operator-names|operator-count}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
