"""Freeze-time deployment fill: replace ``<runtime>`` sentinels in rendered
gate products with concrete, MACHINE-INDEPENDENT deployment values.

Runs at freeze (``tools/agent_loop_lease.sh``, on the LAUNCH machine) after
render, before ``chmod -R a-w``. The gate SUBPROCESS runs on the devspace
(``kind != "local"``), so freeze cannot fill any value that depends on the
devspace filesystem. Only machine-independent values are filled here:

  * ``MASTER_ADDR`` — ``localhost`` (single-node multi-GPU rendezvous). Already
    a real value at render time, so it is not a ``<runtime>`` sentinel.
  * ``MASTER_PORT`` — a fixed, per-loop-deterministic port. An exclusive
    devspace running gates serially needs no free-port probe, so a stable port
    replaces the dispatch-time ``_allocate_free_port``.
  * ``CHECKPOINT_ROOT`` / ``MEGATRON_ROOT`` — these are ``repo_root()``-derived
    on the execution machine, whose absolute path differs from the launch box.
    They are filled as a RELATIVE convention path; the ours engine / ref
    launcher resolves it against ITS OWN ``cwd`` (= ``repo_root`` on the exec
    machine). Physical storage is redirected via a lease-time symlink on the
    devspace (see memory ``gate-checkpoint-root-symlink``); the value written
    here stays machine-independent.
    ``CHECKPOINT_ROOT`` is filled PER PRODUCT: each gate's product carries its
    own ``[cli].forge_init_ones``, so the base ``.artifacts/checkpoints/
    <backend>`` gets an ``ones`` / ``no1`` subdir appended to match the
    canonical bytes that gate loads (meta gates with no ``forge_init_ones`` get
    the bare root). The ours engine reads this final value straight from the
    product — no consumer-side ones/no1 derivation (design §0: values are
    WRITTEN into the config at freeze, ours only READS them).

Deliberately NOT done here: ``DATA_PATH`` / ``DATA_LOADER`` source-conf
resolution and ``resolve_assets`` downloads — the download is an
execution-machine side-effect and is unused in practice (memory
``gate-asset-download-crossmachine``). Those remain on the dispatch / ref-fold
path.

This module is a PURE function set (plus a thin CLI). Wiring it into the live
freeze happens alongside the matching consumer change (emitter exports the
filled value; dispatcher stops injecting) so the two never double-set a key.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tomllib
from pathlib import Path
from typing import Any

# Must match tools/render_gate_configs.py RUNTIME.
RUNTIME = "<runtime>"

# The three product [env] keys the renderer leaves as RUNTIME sentinels.
_SENTINEL_KEYS = ("MASTER_PORT", "CHECKPOINT_ROOT", "MEGATRON_ROOT")

# Deterministic MASTER_PORT range. Ephemeral ports start at 32768 on Linux;
# stay below that and above the privileged range.
_PORT_BASE = 29500
_PORT_SPAN = 2000

# Vendored Megatron submodule (repo-relative) — mirrors
# config_runtime._resolve_megatron's submodule fallback. Filled only for the
# megatron backend; torch backends never read MEGATRON_ROOT.
_MEGATRON_SUBMODULE_REL = "harness/third_party/megatron/v15"


class ResolveError(RuntimeError):
    """Raised when a product still carries an unresolved sentinel."""


def derive_master_port(loop_id: str, *, base: int = _PORT_BASE, span: int = _PORT_SPAN) -> int:
    """A stable, per-loop MASTER_PORT.

    Deterministic in ``loop_id`` (never ``random``): a resumed loop re-derives
    the same port, and two concurrent loops on a shared launch box (``kind =
    "local"``) get different ports. Within one loop, gates run serially and
    reuse the port after the prior process frees it.
    """
    if not loop_id:
        raise ResolveError("loop_id required to derive MASTER_PORT")
    digest = hashlib.sha1(loop_id.encode(), usedforsecurity=False).hexdigest()
    return base + (int(digest, 16) % span)


def _read_backend(config_dir: Path) -> str:
    ref_path = config_dir / "ref.toml"
    if not ref_path.is_file():
        raise ResolveError(f"[ref] config not found: {ref_path}")
    with ref_path.open("rb") as fh:
        ref = tomllib.load(fh)
    refb = ref.get("ref", ref)
    backend = str(refb.get("backend", "")).strip()
    if not backend:
        raise ResolveError(f"{ref_path}: [ref].backend is required")
    return backend


def compute_resolutions(config_dir: Path, loop_id: str) -> dict[str, str]:
    """Machine-independent replacements for the product ``<runtime>`` sentinels.

    * ``MASTER_PORT`` — per-loop deterministic port (string, to match the
      renderer's string form for a fixed port).
    * ``CHECKPOINT_ROOT`` — repo-relative convention BASE ``.artifacts/
      checkpoints/<backend>``. ``fill_products`` appends each gate's ``ones``/
      ``no1`` subdir per product; resolved against the exec machine's cwd.
    * ``MEGATRON_ROOT`` — repo-relative submodule path for the megatron backend;
      empty for torch (unused there).
    """
    backend = _read_backend(config_dir)
    megatron_root = _MEGATRON_SUBMODULE_REL if backend == "megatron" else ""
    return {
        "MASTER_PORT": str(derive_master_port(loop_id)),
        "CHECKPOINT_ROOT": f".artifacts/checkpoints/{backend}",
        "MEGATRON_ROOT": megatron_root,
    }


def _forge_init_ones(product_path: Path) -> int | None:
    """Read ``[cli].forge_init_ones`` from a product, or ``None`` if absent."""
    with product_path.open("rb") as fh:
        doc = tomllib.load(fh)
    value = doc.get("cli", {}).get("forge_init_ones")
    return int(value) if value is not None else None


def _product_resolutions(base: dict[str, str], product_path: Path) -> dict[str, str]:
    """Per-product resolutions: append the ``ones``/``no1`` subdir to the base
    ``CHECKPOINT_ROOT`` from the product's own ``forge_init_ones``.

    A meta gate that declares no ``forge_init_ones`` keeps the bare base root.
    ``MASTER_PORT`` / ``MEGATRON_ROOT`` are gate-independent, so they pass
    through unchanged.
    """
    res = dict(base)
    ckpt = base.get("CHECKPOINT_ROOT")
    if ckpt:
        ones = _forge_init_ones(product_path)
        if ones is not None:
            res["CHECKPOINT_ROOT"] = f"{ckpt}/{'ones' if ones == 1 else 'no1'}"
    return res


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def fill_product_text(text: str, resolutions: dict[str, str]) -> tuple[str, list[str]]:
    """Replace ``<runtime>`` values in the ``[env]`` section only.

    Returns ``(new_text, unresolved)`` where ``unresolved`` lists env keys that
    were still ``<runtime>`` but had no entry in ``resolutions``. The ``[cli]``
    section is never touched (it carries no sentinels).
    """
    out: list[str] = []
    unresolved: list[str] = []
    in_env = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_env = stripped == "[env]"
            out.append(line)
            continue
        if in_env and "=" in stripped and not stripped.startswith("#"):
            key, _, raw = line.partition("=")
            if raw.strip() == f'"{RUNTIME}"':
                name = key.strip()
                if name in resolutions:
                    out.append(f"{key}= {_toml_scalar(resolutions[name])}")
                    continue
                unresolved.append(name)
        out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else ""), unresolved


def fill_products(
    product_dirs: list[Path],
    resolutions: dict[str, str],
    *,
    require_complete: bool = True,
) -> dict[str, list[str]]:
    """Fill every ``*.toml`` product under each dir. Returns ``{path: unresolved}``.

    With ``require_complete`` (the freeze default) a remaining sentinel is a
    hard error — a product must never ship a ``<runtime>`` into a frozen config.
    """
    report: dict[str, list[str]] = {}
    leftover: dict[str, list[str]] = {}
    for pdir in product_dirs:
        if not pdir.is_dir():
            continue
        for product in sorted(pdir.glob("*.toml")):
            per_product = _product_resolutions(resolutions, product)
            new_text, unresolved = fill_product_text(product.read_text(), per_product)
            product.write_text(new_text)
            report[str(product)] = unresolved
            if unresolved:
                leftover[str(product)] = unresolved
    if require_complete and leftover:
        detail = "; ".join(f"{p}: {keys}" for p, keys in sorted(leftover.items()))
        raise ResolveError(f"unresolved <runtime> sentinels after fill — {detail}")
    return report


def _default_product_dirs(workspace: Path) -> list[Path]:
    return [workspace / "ref" / "config", workspace / "workload" / "src" / "config"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="loop workspace (products under ref/config + workload/src/config)",
    )
    ap.add_argument(
        "--config-dir",
        type=Path,
        required=True,
        help="frozen axis config dir (reads ref.toml for backend)",
    )
    ap.add_argument(
        "--loop-id", required=True, help="loop id — seeds the deterministic MASTER_PORT"
    )
    ap.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="do not fail on leftover <runtime> (diagnostic)",
    )
    args = ap.parse_args(argv)

    resolutions = compute_resolutions(args.config_dir, args.loop_id)
    report = fill_products(
        _default_product_dirs(args.workspace),
        resolutions,
        require_complete=not args.allow_incomplete,
    )
    filled = sum(1 for _p in report)
    print(f"resolve_deploy: filled {filled} product(s) with {resolutions}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ResolveError as exc:
        print(f"resolve_deploy error: {exc}", file=sys.stderr)
        sys.exit(2)
