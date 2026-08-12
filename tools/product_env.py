"""Generic product → shell-env projector (thin-dispatcher refactor, step 1).

Reads a rendered gate product (``ref/config/<gate>.toml`` or
``workload/src/config/<gate>.toml``) and prints ``export KEY=value`` lines so
a side sh script can ``eval "$(python3 tools/product_env.py <product>)"``.

FULLY GENERIC by design: this module has ZERO key-name knowledge. Any key
added to a product passes through without ever touching this file. The rules
are keyed on table membership and VALUE shape only:

* ``[env]`` keys are exported verbatim (already UPPERCASE by convention).
* ``[cli]`` keys are exported under their upper-cased name. On a collision
  with an ``[env]`` key, ``[env]`` wins (a warning goes to stderr when the
  values differ).
* Value coercion: ``bool`` → ``"1"``/``"0"``; ``list`` → space-joined scalar
  items; everything else → ``str(value)``. Output values are shell-quoted.
* Value sentinels (matched on the VALUE, never the key name):
  - ``"<runtime>"``   → the key is NOT exported; the process env (runtime
    injection) stays authoritative.
  - ``"<auto-port>"`` → a free ephemeral TCP port is allocated at eval time
    and exported in place of the sentinel.
* A key that is not a valid shell identifier (e.g. contains ``-``) fails
  fast — it can never round-trip through ``export``.

Query mode: ``--get KEY`` prints the coerced value of one bare key (cli first,
then env under the upper-cased name) instead of the export block. A missing
key or a ``"<runtime>"`` sentinel prints nothing (exit 0), so shell callers
can apply their own ``${VAR:-default}``.

Usage (from a side sh, cwd = workspace root):
    eval "$(python3 tools/product_env.py workload/src/config/$GATE.toml)"
    LEVEL=$(python3 tools/product_env.py --get hash_capture_level ref/config/$GATE.toml)
"""

from __future__ import annotations

import argparse
import re
import shlex
import socket
import sys
import tomllib
from pathlib import Path

RUNTIME_SENTINEL = "<runtime>"
AUTO_PORT_SENTINEL = "<auto-port>"

_SHELL_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _scalar(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, list):
        return " ".join(_scalar(v) for v in value)
    return str(value)


def _allocate_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _load(product_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    with open(product_path, "rb") as fh:
        doc = tomllib.load(fh)
    return dict(doc.get("cli", {})), dict(doc.get("env", {}))


def _merged_exports(cli: dict[str, object], env: dict[str, object]) -> dict[str, object]:
    """Merge the two transports into one export map ([env] wins on collision)."""
    merged: dict[str, object] = {}
    for key, value in cli.items():
        upper = key.upper()
        if not _SHELL_IDENT.match(upper):
            raise ValueError(
                f"[cli] key {key!r} is not a valid shell identifier after "
                "upper-casing; it cannot be exported"
            )
        merged[upper] = value
    for key, value in env.items():
        if not _SHELL_IDENT.match(key):
            raise ValueError(
                f"[env] key {key!r} is not a valid shell identifier; it cannot be exported"
            )
        if key in merged and _scalar(merged[key]) != _scalar(value):
            print(
                f"product_env: WARNING [env].{key} overrides [cli].{key.lower()} "
                f"({_scalar(merged[key])!r} -> {_scalar(value)!r})",
                file=sys.stderr,
            )
        merged[key] = value
    return merged


def export_map(product_path: Path) -> dict[str, str]:
    """The projection as a plain dict, for Python callers building a
    subprocess env (e.g. tools/bootstrap_canonical.py) — same rules as the
    shell export block, still zero key-name knowledge."""
    out: dict[str, str] = {}
    for key, value in _merged_exports(*_load(product_path)).items():
        text = _scalar(value)
        if text == RUNTIME_SENTINEL:
            continue
        if text == AUTO_PORT_SENTINEL:
            text = str(_allocate_free_port())
        out[key] = text
    return out


def export_lines(product_path: Path) -> list[str]:
    return [f"export {key}={shlex.quote(text)}" for key, text in export_map(product_path).items()]


def get_value(product_path: Path, key: str) -> str | None:
    """Resolve one bare key: [cli] first, then [env] under the UPPER name."""
    cli, env = _load(product_path)
    if key in cli:
        value = cli[key]
    elif key.upper() in env:
        value = env[key.upper()]
    else:
        return None
    text = _scalar(value)
    if text == RUNTIME_SENTINEL:
        return None
    if text == AUTO_PORT_SENTINEL:
        return str(_allocate_free_port())
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--get", metavar="KEY", help="print one key's value instead of exports")
    parser.add_argument("product", help="path to a rendered gate product TOML")
    args = parser.parse_args(argv)

    path = Path(args.product)
    if not path.exists():
        print(f"ERROR: gate product not found: {path}", file=sys.stderr)
        return 1

    if args.get:
        value = get_value(path, args.get)
        if value is not None:
            print(value)
        return 0

    for line in export_lines(path):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
