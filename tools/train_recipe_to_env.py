"""Project a ``config/train/*.toml`` recipe into shell env assignments.

The training-recipe TOMLs (``config/train/wsdsft_05b_prod.toml``,
``config/train/resume_20.toml``) are the machine-read source for the remote
PyTorchJob scripts: the launchers read ``[launch]`` (card count / node count)
and the per-pod entries read the full recipe. This module is the transport —
it prints one shell line per recipe key:

    export VAR="${VAR:-<toml value>}"

i.e. **env-wins** semantics: an already-exported env var always beats the TOML
(so a dispatcher-injected gate env or an explicit operator override keeps
precedence and the TOML supplies the defaults). This is the opposite
precedence of ``gate_product_to_shell.py`` (where the rendered product must
always win) — recipes are defaults, products are contracts.

Section -> variable mapping (fixed, completeness-checked):
  [launch]      gpus_per_node -> GPUS_PER_NODE, workers -> WORKERS
  [common]      <key> -> <KEY upper>            (production 3-phase recipe)
  [train]       <key> -> <KEY upper>            (single-phase resume recipe)
  [phases.<p>]  <key> -> <P upper>_<KEY upper>  (stable/decay/sft phase table)

An unknown top-level section raises — a recipe key that silently never
reaches shell is the drift bug this tool exists to kill.

Usage (from a launcher / per-pod entry):
    source <(python3 tools/train_recipe_to_env.py config/train/<recipe>.toml)
    source <(python3 tools/train_recipe_to_env.py --section launch <recipe>)
"""

from __future__ import annotations

import argparse
import shlex
import sys
import tomllib
from pathlib import Path

_LAUNCH_KEYS = {
    "gpus_per_node": "GPUS_PER_NODE",
    "workers": "WORKERS",
}

# Flat-scalar sections whose keys map to their own uppercase name.
_FLAT_SECTIONS = ("launch", "common", "train")
_KNOWN_SECTIONS = (*_FLAT_SECTIONS, "phases")


def _fmt(value: object) -> str:
    if isinstance(value, bool):  # before int: bool is an int subclass
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return value
    raise TypeError(f"unsupported recipe value type: {value!r}")


def _emit(lines: list[str], var: str, value: object) -> None:
    quoted = shlex.quote(_fmt(value))
    lines.append(f'export {var}="${{{var}:-{quoted}}}"')


def project(recipe: dict, sections: list[str] | None = None) -> list[str]:
    lines: list[str] = []
    for section, table in recipe.items():
        if section not in _KNOWN_SECTIONS:
            raise KeyError(
                f"unknown recipe section [{section}] — map it in "
                f"tools/train_recipe_to_env.py or it never reaches shell"
            )
        if sections and section not in sections:
            continue
        if section == "phases":
            for phase, phase_table in table.items():
                prefix = phase.upper()
                for key, value in phase_table.items():
                    _emit(lines, f"{prefix}_{key.upper()}", value)
        else:
            for key, value in table.items():
                var = _LAUNCH_KEYS.get(key, key.upper()) if section == "launch" else key.upper()
                _emit(lines, var, value)
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path, help="config/train/<recipe>.toml")
    parser.add_argument(
        "--section",
        action="append",
        dest="sections",
        help="emit only this top-level section (repeatable); default: all",
    )
    args = parser.parse_args(argv)

    with args.recipe.open("rb") as fh:
        recipe = tomllib.load(fh)

    for line in project(recipe, args.sections):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
