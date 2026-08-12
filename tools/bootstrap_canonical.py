"""One-shot canonical_state_fp32.pt bootstrap.

Dumps a canonical FP32 master-weight state by invoking the ref bridge
(``ref/bridges/bridge.sh``) once with ``CANONICAL_STATE_OUTPUT_FILE`` set
and an explicit ``--init-ones`` selection. Two canonicals must be
produced before any stage1 gate can run under the
``forge_init_ones`` scheme:

    python tools/bootstrap_canonical.py --init-ones 1   # → ones/canonical_state_fp32.pt
    python tools/bootstrap_canonical.py --init-ones 0   # → no1/canonical_state_fp32.pt

This is the same one-time-per-deployment bootstrap pattern documented in
``evals/harness_hook/recipes/README.md`` §3 ("not driven by the
dispatcher"). The dispatcher reads whichever subdir the per-gate
``forge_init_ones`` field selects.

Required environment (sourced from harness config_runtime):
  FORGE_BACKEND          ``torch`` or ``megatron`` (defaulted from
                         ``config/ref.toml``).
  Repo paths             Resolved from ``config_runtime``; the script
                         finds the bridge / interposer relative to the
                         workspace root automatically.

Knobs (CLI):
  --init-ones {0,1}      REQUIRED. 1 → ones/ (production); 0 → no1/ (bitwise gates).
  --backend NAME         Override the resolved ref backend.
  --output-root PATH     Override the parent of ``ones/`` / ``no1/``.
                         Defaults to ``[ref].checkpoint_root``.
  --world-size N         GPU count (default 2 to match the smallest
                         multi-GPU bitwise gate; 1 also works since the
                         interposer exits after the first optimizer step).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import tomllib
from pathlib import Path


def _free_port() -> int:
    """Return a currently-free ephemeral port (bind(0) / getsockname idiom).

    The ref launcher defaults ``MASTER_PORT`` to a fixed ``23456`` when the
    env does not set it. A canonical bootstrap that shares the exec host
    with any other torchrun — a prior gate run whose remote process leaked
    past its timeout, or a concurrent one — then dies with EADDRINUSE
    before the bridge starts. Handing torchrun a fresh free port per
    bootstrap sidesteps the collision entirely.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


# ── Canonical source-gate selection ─────────────────────────────────────
# Every launcher param comes from the rendered gate product — forge_init_ones
# included (projected as FORGE_INIT_ONES by product_env). The canonical for a
# given forge_init_ones scheme MUST be produced by a gate whose product
# carries the matching forge_init_ones.
# Both candidates must also reach the first optimizer.step() (the canonical
# dump fires there) — pure forward/backward alignment gates that never step
# are excluded. First rendered product whose forge_init_ones matches wins;
# no match → fail fast (a render / gate-set bug, not a silent fallback).
_CANONICAL_SOURCE_GATES: dict[int, tuple[str, ...]] = {
    1: ("long-train", "loss-gate-200", "long-train-smoke"),
    0: ("multistep", "multistep-1gpu", "resume-gate-20", "perf-bitwise"),
}


def _product_forge_init_ones(product: Path) -> int | None:
    """Read ``[cli].forge_init_ones`` from a rendered ref gate product."""
    try:
        with product.open("rb") as fh:
            cli = tomllib.load(fh).get("cli", {})
    except (OSError, tomllib.TOMLDecodeError):
        return None
    value = cli.get("forge_init_ones")
    return int(value) if value is not None else None


def resolve_canonical_gate(ref_config_dir: Path, init_ones: int) -> str:
    """Pick the FORGE_GATE whose product materializes ``init_ones``'s canonical."""
    candidates = _CANONICAL_SOURCE_GATES.get(init_ones)
    if candidates is None:
        raise SystemExit(f"init_ones must be 0 or 1; got {init_ones!r}.")
    for name in candidates:
        product = ref_config_dir / f"{name}.toml"
        if product.exists() and _product_forge_init_ones(product) == init_ones:
            return name
    raise SystemExit(
        f"canonical bootstrap: no rendered gate in {ref_config_dir} carries "
        f"forge_init_ones={init_ones} among {candidates}. Render the gate "
        "configs first (lease step / tools/render_gate_configs.py)."
    )


def _repo_root() -> Path:
    # tools/bootstrap_canonical.py → tools/ → harness/
    return Path(__file__).resolve().parent.parent


def _load_workload(repo_root: Path):
    sys.path.insert(0, str(repo_root))
    from harness import config_runtime

    _path, workload = config_runtime.load_workload_config()
    return workload


def _resolved_backend(workload, override: str | None) -> str:
    if override:
        return override
    backend = workload.get("ref", {}).get("backend", "")
    if not backend:
        raise SystemExit(
            "Cannot resolve ref backend: --backend not supplied and "
            "[ref].backend is missing from the workload config."
        )
    return str(backend)


def _resolved_output_root(workload, override: str | None) -> Path:
    if override:
        return Path(override).resolve()
    ckpt = workload.get("ref", {}).get("checkpoint_root", "")
    if not ckpt:
        raise SystemExit(
            "Cannot resolve checkpoint_root: --output-root not supplied "
            "and [ref].checkpoint_root is empty in the workload config."
        )
    return Path(ckpt).resolve()


CANONICAL_FILENAME = "canonical_state_fp32.pt"


def canonical_path(workload, init_ones: int, *, output_root: str | None = None) -> Path:
    """Absolute path of the canonical dump for one ``forge_init_ones`` scheme.

    Single source of truth for the
    ``<checkpoint_root>/{ones,no1}/canonical_state_fp32.pt`` layout: both
    ``main`` (the writer) and the loop-wrapper preflight (the idempotency
    probe in :mod:`evals.canonical_preflight`) resolve the path here so the
    two can never drift. ``init_ones == 1`` → ``ones/`` (production);
    ``init_ones == 0`` → ``no1/`` (anti-cheat bitwise gates).
    """
    subdir = "ones" if init_ones == 1 else "no1"
    return _resolved_output_root(workload, output_root) / subdir / CANONICAL_FILENAME


def fingerprint_path(canonical_file: Path) -> Path:
    """Sidecar recording WHICH rendered ref product produced a canonical."""
    return canonical_file.with_name(canonical_file.name + ".fingerprint.json")


def _workload_ref_script(workload) -> str:
    """Raw ``[ref].ref_script`` for fingerprinting (empty when absent)."""
    ref = workload.get("ref", {})
    return str(ref.get("ref_script", "")) if isinstance(ref, dict) else ""


def expected_fingerprint(
    ref_config_dir: Path, init_ones: int, *, ref_script: str
) -> dict[str, str]:
    """Identity a CURRENT bootstrap would stamp on ``init_ones``'s canonical.

    The canonical's tensor structure comes from the source gate's rendered
    ref product PLUS the L0 launcher the bridge execs ([ref].ref_script —
    that is what picks the model implementation), so both are the
    canonical's identity. A model-axis change, a re-render, or a
    checkpoint_root inherited from a previous loop all show up as a
    fingerprint mismatch — the wrong-structure-canonical class of failure
    (a MiniCPM-shaped canonical silently poisoning a Qwen3 loop's bitwise
    gates) that a bare ``exists()`` probe can never see.
    """
    gate = resolve_canonical_gate(ref_config_dir, init_ones)
    product = ref_config_dir / f"{gate}.toml"
    return {
        "gate": gate,
        "product_sha256": hashlib.sha256(product.read_bytes()).hexdigest(),
        "ref_script": ref_script,
    }


def canonical_is_current(
    workload,
    init_ones: int,
    *,
    ref_config_dir: Path,
    output_root: str | None = None,
) -> bool:
    """True iff the canonical exists AND matches the current rendered product."""
    target = canonical_path(workload, init_ones, output_root=output_root)
    if not target.exists():
        return False
    try:
        recorded = json.loads(fingerprint_path(target).read_text())
    except (OSError, ValueError):
        # No/unreadable sidecar: a pre-fingerprint deployment or a hand-placed
        # file — unverifiable, so treat as stale and regenerate once.
        return False
    return recorded == expected_fingerprint(
        ref_config_dir, init_ones, ref_script=_workload_ref_script(workload)
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--init-ones",
        type=int,
        choices=[0, 1],
        required=True,
        help="1 → standard ones init (non-bitwise gates); "
        "0 → anti-cheat 0.97 init (bitwise gates).",
    )
    parser.add_argument("--backend", default=None, help="Override ref backend.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Override the parent of ones/ / no1/ subdirs (default: [ref].checkpoint_root).",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
        help="GPU count for the bootstrap run (default 2).",
    )
    args = parser.parse_args(argv)

    repo_root = _repo_root()
    workload = _load_workload(repo_root)
    backend = _resolved_backend(workload, args.backend)
    if backend != "torch":
        raise SystemExit(
            f"bootstrap_canonical.py currently only supports backend=torch; got {backend!r}."
        )

    canonical_file = canonical_path(workload, args.init_ones, output_root=args.output_root)
    canonical_file.parent.mkdir(parents=True, exist_ok=True)

    bridge = repo_root / "ref" / "bridges" / "bridge.sh"
    if not bridge.exists():
        raise SystemExit(f"ref bridge missing: {bridge}")

    # Pick a gate whose product forge_init_ones matches the scheme we are
    # materializing, then project that product into the subprocess env
    # ourselves (tools/product_env.py, generic upper-cased-key names). The
    # launchers consume the generic projection only (ref-generic-projection
    # plan): outside run_gate.sh, WE are the projection layer. Explicit
    # assignments below come AFTER the product projection so the bootstrap's
    # own overrides (WORLD_SIZE / MASTER_PORT / FORGE_INIT_ONES / ...) stay
    # authoritative over freeze-filled product values.
    gate = resolve_canonical_gate(repo_root / "ref" / "config", args.init_ones)

    from harness import config_runtime
    from tools import product_env

    # bridge.sh's torch branch falls back to the MiniCPM launcher when
    # FORGE_BRIDGE_REF_SCRIPT is unset — the dispatcher normally sets it
    # from [ref].ref_script, but the bootstrap bypasses the dispatcher, so
    # it must project the ref axis itself or every canonical comes out
    # MiniCPM-shaped regardless of the model axis.
    ref_script_name = config_runtime.ref_script(workload)

    env = os.environ.copy()
    env.update(product_env.export_map(repo_root / "ref" / "config" / f"{gate}.toml"))
    env["FORGE_BACKEND"] = backend
    env["FORGE_GATE"] = gate
    env["FORGE_BRIDGE_REF_SCRIPT"] = ref_script_name
    # Scheme override: the picked gate's product carries a matching
    # forge_init_ones by construction, but the explicit export keeps the
    # bootstrap authoritative over the freeze-filled product value.
    env["FORGE_INIT_ONES"] = str(args.init_ones)
    env["CANONICAL_STATE_OUTPUT_FILE"] = str(canonical_file)
    env["LOCAL_MODE"] = "1"
    env["WORLD_SIZE"] = str(args.world_size)
    # Pin a fresh free MASTER_PORT so torchrun never falls back to the ref
    # launcher's fixed 23456 default (nor any shared value inherited from the
    # loop env) and collides with a leaked/concurrent torchrun on the exec
    # host — the observed EADDRINUSE that aborted the canonical preflight.
    # The bootstrap is single-node (LOCAL_MODE), so any free port works.
    env["MASTER_PORT"] = str(_free_port())
    # No HOOK_OUTPUT_FILE on purpose — interposer.py treats these two as
    # mutually exclusive and dispatches canonical_state_dump only when
    # HOOK_OUTPUT_FILE is unset (see interposer.py:_wire_hook).
    env.pop("HOOK_OUTPUT_FILE", None)
    # The ref launcher resolves [data].data_loader from config/data.toml via
    # --data-config "${FORGE_DATA_TOML:-}"; nothing else in the canonical
    # bootstrap path exports this pointer, so point it at the active per-loop
    # data.toml (a POINTER, never the loader VALUE — same as evals/_common.py).
    data_toml = config_runtime._data_config_path()
    if not data_toml.exists():
        raise SystemExit(
            f"canonical bootstrap: active data config not found at {data_toml}; "
            "config/data.toml must exist so the ref resolves [data].data_loader."
        )
    env["FORGE_DATA_TOML"] = str(data_toml)

    print(
        f"[bootstrap_canonical] backend={backend} gate={gate} "
        f"init_ones={args.init_ones} out={canonical_file}",
        flush=True,
    )
    # A pre-existing file here is one we were asked to REPLACE (the preflight
    # only dispatches missing/stale schemes). Drop it first so the post-run
    # existence check below proves the dump actually fired — otherwise a
    # silently-skipped interposer hook would leave the stale bytes in place
    # and this script would still report success.
    canonical_file.unlink(missing_ok=True)
    fingerprint_path(canonical_file).unlink(missing_ok=True)
    result = subprocess.run(["bash", str(bridge)], cwd=repo_root, env=env)
    if result.returncode != 0:
        raise SystemExit(
            f"bridge.sh exited with returncode={result.returncode}; "
            "canonical state was NOT written. Re-run with the failing "
            "command for diagnosis."
        )
    if not canonical_file.exists():
        raise SystemExit(
            f"bridge.sh returned 0 but canonical not at {canonical_file}; "
            "check that the interposer's install_canonical_state_dump hook "
            "fired (a missing optimizer.step in the ref launcher would "
            "prevent it)."
        )
    fingerprint_path(canonical_file).write_text(
        json.dumps(
            expected_fingerprint(
                repo_root / "ref" / "config",
                args.init_ones,
                ref_script=ref_script_name,
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        f"[bootstrap_canonical] OK: wrote {canonical_file} ({canonical_file.stat().st_size} bytes)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
