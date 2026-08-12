"""Runtime-env projector for the thin-dispatcher side scripts (step 1).

Prints ``export KEY=value`` lines for everything a side needs at run time
that is NOT carried by its rendered product: axis-resolved deployment paths,
asset dirs, SSOT pointers, and script-routing values. The side sh evals this
AFTER the product exports, so runtime values win — matching the legacy
dispatcher env layering (product/preset < path shim < pointers < runtime).

This is a thin wrapper over the existing Python resolution helpers
(``evals._common`` / ``harness.config_runtime``); the renderer and the
products are untouched. Values are per-GATE, never per-RUN — per-run paths
(DUMP_DIR, LOSS_DUMP_FILE, SAVE_PATH, ...) are derived by the side sh from
its ``<run_dir>`` argument.

Lives under evals/scripts/ (NOT tools/): it imports ``evals._common``,
and the layer DAG is one-way evals/ → tools/.

Usage (from a side sh, cwd = workspace root):
    eval "$(python3 evals/scripts/runtime_env.py ref  "$GATE")"
    eval "$(python3 evals/scripts/runtime_env.py ours "$GATE")"
"""

from __future__ import annotations

import argparse
import contextlib
import shlex
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evals import _common  # noqa: E402
from harness import config_runtime  # noqa: E402


def _ref_overrides(repo_root: Path, gate: str, cfg: dict, workload_config: dict) -> dict[str, str]:
    ov: dict[str, str] = {}
    # Deployment paths + assets (FORGE_TOKENIZER_DIR / FORGE_DATA_DIR /
    # MEGATRON_ROOT / DATA_PATH / DATA_CONF). dump_dir=None: the per-run
    # SAVE_PATH / TENSORBOARD_DIR are derived by run_gate.sh from its run_dir.
    ov.update(
        _common._ref_script_path_env(repo_root, cfg, workload_config=workload_config, dump_dir=None)
    )
    # SSOT pointers the L0 launcher / bridge read unconditionally.
    ov["FORGE_REF_CONFIG_DIR"] = str(repo_root / "ref" / "config")
    ov["FORGE_DATA_TOML"] = str(config_runtime._data_config_path())
    # Single-node rendezvous, mirroring _ours_overrides: the freeze-filled
    # product carries a concrete MASTER_PORT in [env] which the full generic
    # projection now exports — these runtime values are eval'd LAST by
    # project_env so they stay authoritative over the frozen product value.
    defaults = config_runtime.runtime_env_defaults(workload_config)
    ov["MASTER_ADDR"] = defaults.master_addr
    ov["MASTER_PORT"] = _common._resolve_master_port(defaults.master_port)
    # Script routing: the resolved L0 launcher and (optionally) the capture
    # bridge. run_gate.sh picks between them on the product's
    # hash_capture_level value — no gate-name branching.
    ref_script_name = config_runtime.ref_script(workload_config)
    ov["FORGE_REF_SCRIPT_PATH"] = str(
        _common._resolve_customer_ref_script(repo_root, ref_script_name)
    )
    raw_bridge = config_runtime.ref_capture_script(workload_config)
    if raw_bridge:
        ov["FORGE_REF_CAPTURE_SCRIPT_PATH"] = str(
            _common._resolve_ref_capture_script(repo_root, raw_bridge)
        )
    # Bridge inputs (consumed only on the hash-capture path).
    ov["FORGE_BACKEND"] = config_runtime.ref_backend(workload_config)
    ov["FORGE_BRIDGE_REF_SCRIPT"] = ref_script_name
    # Suite-level extra CLI tokens for the L0 script ("$@" passthrough),
    # pre-quoted so the sh can `eval "ARGS=($FORGE_REF_EXTRA_ARGS)"`.
    extra = cfg.get("ref_extra_args") if isinstance(cfg, dict) else None
    if isinstance(extra, list) and extra:
        ov["FORGE_REF_EXTRA_ARGS"] = " ".join(shlex.quote(str(t)) for t in extra)
    return ov


def _ours_overrides(repo_root: Path, gate: str, cfg: dict, workload_config: dict) -> dict[str, str]:
    ov: dict[str, str] = {}
    # Tokenizer / dataset assets (download on cache miss) — FORGE_TOKENIZER_DIR
    # / FORGE_DATA_DIR, same channel the legacy build_suite_env used.
    ov.update(
        config_runtime.resolve_assets(
            workload_config.get("ref", {}), workload_config.get("data", {})
        )
    )
    # Data-value SSOT pointer (loader kind lives in data.toml, not in env).
    ov["FORGE_DATA_TOML"] = str(config_runtime._data_config_path())
    # B-class deployment fallbacks from the axis. A frozen ours product
    # carries these itself (GateInputs reads the product first); these env
    # values only serve pre-freeze products holding "<runtime>" sentinels.
    effective = {**workload_config.get("ref", {}), **(cfg if isinstance(cfg, dict) else {})}
    for cfg_key, env_key in (
        ("megatron_root", "MEGATRON_ROOT"),
        ("checkpoint_root", "CHECKPOINT_ROOT"),
        ("data_path", "DATA_PATH"),
        ("data_conf", "DATA_CONF"),
    ):
        value = effective.get(cfg_key)
        if isinstance(value, str) and value:
            ov[env_key] = value
    if "DATA_PATH" not in ov and effective.get("data_conf"):
        resolved = config_runtime._resolve_data_env_from_conf(str(effective["data_conf"]))
        ov["DATA_PATH"] = resolved["data_path"]
    # Single-node rendezvous: master_port="auto" resolves to a fresh free
    # ephemeral port here (per sh eval), same contract as distributed_env.
    defaults = config_runtime.runtime_env_defaults(workload_config)
    ov["MASTER_ADDR"] = defaults.master_addr
    ov["MASTER_PORT"] = _common._resolve_master_port(defaults.master_port)
    # Entry routing from the suite registry (cfg["script"] / cfg["launcher"]).
    script = cfg.get("script") if isinstance(cfg, dict) else None
    launcher = cfg.get("launcher") if isinstance(cfg, dict) else None
    if script:
        ov["FORGE_OURS_ENTRY"] = str(repo_root / str(script))
    if launcher:
        ov["FORGE_OURS_LAUNCHER"] = str(repo_root / str(launcher))
    # Engine product locator (workload runtime_config._product_cli): the
    # engine reads its [cli] hyperparameters from the product these two
    # pointers name. GateInputs itself no longer accepts them (--config only).
    ov["FORGE_GATE"] = gate
    ov["FORGE_OURS_CONFIG_DIR"] = str(repo_root / "workload" / "src" / "config")
    return ov


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("side", choices=("ref", "ours"))
    parser.add_argument("gate")
    args = parser.parse_args(argv)

    repo_root = Path.cwd()
    workload_config = _common._load_workload_config_for_ref(repo_root)
    cfg = workload_config.get("evals", {}).get(args.gate, {})
    if not isinstance(cfg, dict):
        cfg = {}

    # The resolution helpers may print (asset downloads, ref-cache notes).
    # stdout is the eval wire — divert everything they emit to stderr so
    # only the export lines below reach the caller's eval.
    with contextlib.redirect_stdout(sys.stderr):
        if args.side == "ref":
            overrides = _ref_overrides(repo_root, args.gate, cfg, workload_config)
        else:
            overrides = _ours_overrides(repo_root, args.gate, cfg, workload_config)

    for key, value in overrides.items():
        print(f"export {key}={shlex.quote(str(value))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
