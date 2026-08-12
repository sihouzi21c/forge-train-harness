"""Render per-gate single-source TOML into ref/ours consumption products.

Single source: ``<config-dir>/<gate_config_dir>/<gate>.toml`` (the per-loop
config home, alongside the axis files) with semantic sections
(``[shared]`` / ``[shared.gate]`` / ``[optim]`` / ``[model]`` /
``[ref]`` / ``[ours]``).

Per gate × per side (ref/ours) the renderer:

  1. Seeds a flat baseline dict (bare names) from the active axis configs
     (model.toml + optim.toml + global seed default), the registry global
     ``[env]`` (library flags) and the infra env values (ref/data/runtime).
  2. Overlays each section low→high, stripping the ``_override`` suffix to a
     bare name and overwriting (last-wins). ``_override`` is purely an
     authoring marker — mechanically it is a bare-name overwrite and may
     target ANY lower-layer key (model/optim/seed/global-env/[shared]).
  3. Sweeps ``"@unset"`` tombstones — the key vanishes from the product
     entirely (gone from ``[env]`` ⇒ this side does not export it).
  4. Derives ``grad_accum_steps`` = GBS // (MBS · world_size) and
     ``NUM_PROCS`` = world_size.
  5. Splits by transport: bare name in the ENV whitelist → ``[env]``
     (UPPERCASE); everything else → ``[cli]``.

Products are self-contained: one file gives the full effective value set for
that side plus how each value is delivered. With no per-side override the two
products' ``[cli]``/``[env]`` are byte-identical.

This is step 2 of the gate-config refactor: it only WRITES products to an
output dir. Wiring into the lease/freeze hook and downstream consumers
(ref scripts, ours runtime_config) is a later step. See
``gate_config_render_plan.md``.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path
from typing import Any

# Fallback seed default if the registry has no [defaults].seed.
DEFAULT_SEED = 1234

# Infra env keys (7): values come from the ref/data/runtime axes, never a
# per-gate knob — they are NOT in <gate>.toml. DATA_LOADER is NOT here: the
# loader kind is read directly from config/data.toml [data].data_loader by both
# sides (ref --data-config, ours runtime_config.data_loader()), so it is never
# rendered into the product [env] nor transported via env var.
ENV_INFRA = (
    "BACKEND",
    "CHECKPOINT_ROOT",
    "MEGATRON_ROOT",
    "DATA_PATH",
    "NUM_PROCS",
    "MASTER_ADDR",
    "MASTER_PORT",
)

RUNTIME = "<runtime>"
_OVR = "_override"

# Review-side keys: consumed ONLY by the review-side throughput checker
# (tools/mfu_elastic_check.py) and the dispatcher's hidden elastic-verdict
# read — both read the gate SOURCE toml directly. They are stripped here so
# they never appear in a rendered (dev-readable) product; the long-horizon
# blinding depends on this strip.
REVIEW_ONLY_KEYS = (
    "review_mfu_target",
    "mfu_elastic_tolerance",
    "mfu_elastic_rounds",
    "mfu_elastic_eps",
    "mfu_screen_margin",
    "mfu_full_every_smokes",
)

# Stable in-section ordering so two sides diff cleanly.
_CLI_ORDER = [
    # shape
    "world_size",
    "tensor_parallel_size",
    "num_steps",
    "micro_batch_size",
    "global_batch_size",
    "grad_accum_steps",
    "seq_length",
    "seed",
    "gate_window",
    "resume_save_step",
    # determinism intent
    "deterministic",
    # thresholds / verdict
    "gate_bitwise",
    "gate_atol",
    "hash_capture_level",
    "mfu_e2e_target",
    "warmup_steps",
    "loss_rel_threshold",
    "loss_abs_threshold",
    "grad_norm_abs_threshold",
    "max_avg_relative_loss_diff",
    "forge_init_ones",
    # model geometry (17)
    "name",
    "num_layers",
    "hidden_size",
    "ffn_hidden_size",
    "num_attention_heads",
    "num_query_groups",
    "head_dim",
    "seq_length",
    "max_position_embeddings",
    "padded_vocab_size",
    "rotary_base",
    "norm_epsilon",
    "init_method_std",
    "mup_base_hidden_size",
    "mup_emb_scale",
    "mup_depth_scale",
    "mtp_num_layers",
    "mtp_loss_weight",
    # optim (9)
    "lr",
    "min_lr",
    "lr_warmup_iters",
    "lr_decay_iters",
    "lr_wsd_decay_iters",
    "weight_decay",
    "adam_beta1",
    "adam_beta2",
    "clip_grad",
]
_CLI_RANK = {k: i for i, k in enumerate(_CLI_ORDER)}


class RenderError(RuntimeError):
    """Raised on any fail-fast condition (missing source, ambiguity, …)."""


def _load(path: Path) -> dict[str, Any]:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def _overlay(resolved: dict[str, Any], table: dict[str, Any], *, where: str) -> None:
    """Overlay one section: strip ``_override`` → bare name, last-wins."""
    bare_seen: dict[str, str] = {}
    for key, val in table.items():
        bare = key[: -len(_OVR)] if key.endswith(_OVR) else key
        if bare in bare_seen:
            raise RenderError(
                f"{where}: ambiguous override — both {bare_seen[bare]!r} and "
                f"{key!r} resolve to bare key {bare!r}"
            )
        bare_seen[bare] = key
        resolved[bare] = val


def _build_baseline(
    model: dict[str, Any],
    optim: dict[str, Any],
    ref: dict[str, Any],
    data: dict[str, Any],
    registry: dict[str, Any],
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    resolved.update(model.get("model", model))
    resolved.update(optim.get("optim", optim))
    resolved["seed"] = registry.get("defaults", {}).get("seed", DEFAULT_SEED)
    # D3: every gate's product carries resume_save_step so the ref launcher
    # reads it without a `${RESUME_SAVE_STEP:-0}` floor (retired under set -u).
    # Non-resume gates keep this 0; the resume gate's source overrides it.
    resolved["resume_save_step"] = 0

    # Library flags from the registry global [env] — bare lowercase so an
    # [ours] override / @unset can target them; re-uppercased at split time.
    for key, val in registry.get("env", {}).items():
        resolved[key.lower()] = val

    dist = registry.get("runtime", {}).get("distributed", {})
    resolved["master_addr"] = dist.get("master_addr", "localhost")
    mp = str(dist.get("master_port", "auto"))
    resolved["master_port"] = RUNTIME if mp in ("", "auto") else mp

    refb = ref.get("ref", ref)
    resolved["backend"] = refb.get("backend", RUNTIME)
    resolved["megatron_root"] = refb.get("megatron_root", RUNTIME)
    resolved["checkpoint_root"] = refb.get("checkpoint_root", RUNTIME)

    # No data-axis value is rendered into the product — the data axis is a
    # separate SSOT (config/data.toml), read at run time via file pointers:
    #   * data_path — resolved by the dispatcher sourcing the data_conf named by
    #     [data].conf_path, then threaded in via --data-path-file. A product copy
    #     would be a second source AND (being truthy) would short-circuit the
    #     dispatcher's `needs_path` resolution, leaking a stale literal.
    #   * data_loader — read straight from [data].data_loader by every side
    #     (ref/sisters --data-config, ours runtime_config.data_loader(), meta
    #     ref_bundle_run.sh), so a product copy would be a second source of truth.
    return resolved


def render_one(gate_src: dict[str, Any], side: str, baseline: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(baseline)

    shared = gate_src.get("shared", {})
    shared_scalars = {k: v for k, v in shared.items() if k != "gate"}
    shared_gate = shared.get("gate", {})

    _overlay(resolved, shared_scalars, where="[shared]")
    _overlay(resolved, shared_gate, where="[shared.gate]")
    _overlay(resolved, gate_src.get("optim", {}), where="[optim]")
    _overlay(resolved, gate_src.get("model", {}), where="[model]")
    _overlay(resolved, gate_src.get(side, {}), where=f"[{side}]")

    # Tombstone sweep.
    resolved = {k: v for k, v in resolved.items() if v != "@unset"}

    # Review-side policy keys never reach a product (see REVIEW_ONLY_KEYS).
    resolved = {k: v for k, v in resolved.items() if k not in REVIEW_ONLY_KEYS}

    # Derived shape. With tensor parallelism the data-parallel world is
    # world_size // tensor_parallel_size (TP ranks process replicated data),
    # so grad_accum derives from dp_size, matching the launcher's recompute.
    if {"global_batch_size", "micro_batch_size", "world_size"} <= resolved.keys():
        gbs = resolved["global_batch_size"]
        mbs = resolved["micro_batch_size"]
        ws = resolved["world_size"]
        tp = resolved.get("tensor_parallel_size", 1)
        if tp == 0 or ws % tp != 0:
            raise RenderError(f"world_size={ws} not divisible by tensor_parallel_size={tp}")
        dp = ws // tp
        denom = mbs * dp
        if denom == 0 or gbs % denom != 0:
            raise RenderError(
                f"grad_accum not integral: GBS={gbs} / (MBS={mbs} · DP={dp}) "
                f"[world_size={ws}, tensor_parallel_size={tp}]"
            )
        resolved.setdefault("grad_accum_steps", gbs // denom)
    if "world_size" in resolved:
        resolved["num_procs"] = resolved["world_size"]

    return resolved


def split_transport(
    resolved: dict[str, Any], env_whitelist: set[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    cli: dict[str, Any] = {}
    env: dict[str, Any] = {}
    for key, val in resolved.items():
        if key.upper() in env_whitelist:
            env[key.upper()] = val
        else:
            cli[key] = val
    cli = dict(sorted(cli.items(), key=lambda kv: (_CLI_RANK.get(kv[0], 10_000), kv[0])))
    infra_rank = {k: i for i, k in enumerate(ENV_INFRA)}
    env = dict(sorted(env.items(), key=lambda kv: (infra_rank.get(kv[0], 10_000), kv[0])))
    return cli, env


def _toml_value(v: Any) -> str:
    # bool before int/float: isinstance(True, int) is True.
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"unsupported toml value {v!r} ({type(v).__name__})")


def _dump_table(name: str, table: dict[str, Any]) -> str:
    lines = [f"[{name}]"]
    lines += [f"{k} = {_toml_value(val)}" for k, val in table.items()]
    return "\n".join(lines)


def _serialize(cli: dict[str, Any], env: dict[str, Any], *, gate: str, side: str) -> str:
    writable = "agent-writable" if side == "ours" else "DO NOT EDIT (harness-frozen)"
    header = (
        f"# AUTO-GENERATED by tools/render_gate_configs.py — {writable}\n"
        f"# gate={gate} side={side}\n\n"
    )
    # Self-contained TOML emit (no tomli_w dep): flat [cli]/[env] scalar tables.
    return header + _dump_table("cli", cli) + "\n\n" + _dump_table("env", env) + "\n"


# ── Cross-config / cross-gate invariant checks ───────────────────────────────
# The renderer is the one place that sees every gate's resolved shape, so it is
# where structural invariants that span gates (or relate keys within a gate) are
# enforced. Without this the meta agent can hand-author divergent configs that
# render + launch fine but train wrong. The authoring rules live in the
# harness_configs milestone prompt (meta_harness/prompts/harness_configs/SKILL.md,
# "Gate-config invariants").

_SMOKE_SUFFIX = "-smoke"
# A `<gate>-smoke` is "the same gate, fewer steps": its entire shape +
# determinism profile must equal the base gate's; ONLY the step count / window /
# warmup may differ. Anything else (MBS, GBS, world_size, determinism, init,
# model/optim values) diverging is a bug (e.g. a smoke that exercises a
# different grad-accum path than the gate it is meant to proxy).
_SMOKE_ALLOWED_DIFF = {"num_steps", "gate_window", "warmup_steps"}
_BITWISE_ZERO_KEYS = ("gate_atol", "loss_abs_threshold", "grad_norm_abs_threshold")
_DETERMINISM_ENV = ("cublas_workspace_config", "nvte_allow_nondeterministic_algo")


def _validate_resolved(gate: str, side: str, resolved: dict[str, Any]) -> None:
    """Per-gate relational invariants on one resolved side (C2/C8/C9)."""
    # C2: gate_window must be a proper interval that fits the run. The window is
    # half-open [lo, hi) over 1-indexed steps, so hi may be num_steps+1 (grades
    # through the last step) — only a window reaching beyond that is broken.
    win = resolved.get("gate_window")
    ns = resolved.get("num_steps")
    if (
        win is not None
        and ns is not None
        and (
            not isinstance(win, list)
            or len(win) != 2
            or win[0] < 0
            or win[0] >= win[1]
            or win[1] > ns + 1
        )
    ):
        raise RenderError(
            f"{gate} ({side}): gate_window {win} must be [lo, hi) with "
            f"0 <= lo < hi <= num_steps+1 ({ns + 1})"
        )
    # C9: a bitwise gate cannot carry a non-zero tolerance.
    if resolved.get("gate_bitwise") is True:
        for k in _BITWISE_ZERO_KEYS:
            v = resolved.get(k)
            if v not in (None, 0, 0.0):
                raise RenderError(f"{gate} ({side}): gate_bitwise=true but {k}={v!r} (must be 0)")
    # C8 (ours side): determinism-off must drop the determinism-forcing env, or
    # the run is half-deterministic. Only the clearly-wrong direction is flagged
    # (det=false but the env still exported); a suite that never declares the env
    # is left alone.
    if side == "ours" and resolved.get("deterministic") is False:
        leaked = [k for k in _DETERMINISM_ENV if k in resolved]
        if leaked:
            raise RenderError(
                f"{gate} (ours): deterministic=false but determinism env still "
                f"exported: {leaked} (must be @unset)"
            )


def _validate_smoke_pairs(resolved_by_gate: dict[str, dict[str, dict[str, Any]]]) -> None:
    """C1: every `<gate>-smoke` must share the base gate's shape (per side)."""
    for gate, sides in resolved_by_gate.items():
        if not gate.endswith(_SMOKE_SUFFIX):
            continue
        base = gate[: -len(_SMOKE_SUFFIX)]
        if base not in resolved_by_gate:
            raise RenderError(f"{gate}: smoke variant has no base gate {base!r} to mirror")
        for side, smoke_resolved in sides.items():
            base_resolved = resolved_by_gate[base].get(side)
            if base_resolved is None:
                continue
            a = {k: v for k, v in base_resolved.items() if k not in _SMOKE_ALLOWED_DIFF}
            b = {k: v for k, v in smoke_resolved.items() if k not in _SMOKE_ALLOWED_DIFF}
            diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            if diff:
                detail = ", ".join(f"{k}: {base}={a.get(k)!r} vs {gate}={b.get(k)!r}" for k in diff)
                raise RenderError(
                    f"{gate} ({side}) diverges from base gate {base} on non-step "
                    f"keys — a smoke must share the base shape; only "
                    f"{sorted(_SMOKE_ALLOWED_DIFF)} may differ. Offenders: {detail}"
                )


# ── Mirror-variant synthesis (profile snapshots) ─────────────────────────────
# A registry suite carrying ``mirror_gate = { <milestone> = <gate> }`` has no
# gate_config source of its own: per milestone the renderer synthesizes an
# ours-only VARIANT product ``<suite>@<milestone>.toml`` from the mirrored
# gate's resolved ours dict, so a snapshot run profiles exactly the shape the
# mirrored gate trains (legacy dispatcher._resolve_profile_shape, moved to
# render time). Verdict/judgment keys are STRIPPED — a snapshot grades nothing
# (and the mirrored gate_window would violate C2 against the short snapshot
# num_steps; a leaked hash_capture_level would wrongly arm the hash wire).
# forge_init_ones is NOT stripped: it is a shape key, not a judgment key —
# resolve_deploy derives the CHECKPOINT_ROOT ones/no1 subdir from it, so a
# variant without it keeps the BARE root and the engine cannot find the
# canonical the mirrored gate loads.
_MIRROR_STRIP_KEYS = (
    "gate_window",
    "gate_bitwise",
    "gate_atol",
    "hash_capture_level",
    "mfu_e2e_target",
    "loss_rel_threshold",
    "loss_abs_threshold",
    "grad_norm_abs_threshold",
    "max_avg_relative_loss_diff",
    "resume_save_step",
)


def _synthesize_mirror_variants(
    registry: dict[str, Any],
    resolved_by_gate: dict[str, dict[str, dict[str, Any]]],
    env_whitelist: set[str],
) -> dict[str, dict[str, str]]:
    """Return ``{"<suite>@<milestone>": {"ours": product_text}}`` for mirror suites."""
    out: dict[str, dict[str, str]] = {}
    for suite_key, cfg in (registry.get("evals") or {}).items():
        mirror = cfg.get("mirror_gate")
        if not isinstance(mirror, dict):
            continue
        for milestone, mirrored_gate in mirror.items():
            mirrored = resolved_by_gate.get(str(mirrored_gate), {}).get("ours")
            if mirrored is None:
                # A partial render (--only without the mirrored gate) simply
                # skips the variant; the full render always covers it.
                print(
                    f"render: {suite_key}@{milestone}: mirrored gate "
                    f"{mirrored_gate!r} not in this render — variant skipped",
                    file=sys.stderr,
                )
                continue
            variant = dict(mirrored)
            for key in _MIRROR_STRIP_KEYS:
                variant.pop(key, None)
            variant["num_steps"] = int(cfg.get("num_steps", 12))
            variant["warmup_steps"] = int(cfg.get("warmup_steps", 0))
            variant["nsys_profile"] = True
            name = f"{suite_key}@{milestone}"
            _validate_resolved(name, "ours", variant)
            cli, env = split_transport(variant, env_whitelist)
            out[name] = {"ours": _serialize(cli, env, gate=name, side="ours")}
    return out


def render_gate(
    gate: str,
    gates_dir: Path,
    baseline_inputs: dict[str, Any],
    env_whitelist: set[str],
) -> dict[str, tuple[str, dict[str, Any]]]:
    """Render a gate to ``{side: (product_text, resolved_dict)}``.

    The resolved dict is returned alongside the serialized product so the caller
    can run cross-gate validation (``_validate_smoke_pairs``) before writing.
    """
    src_path = gates_dir / f"{gate}.toml"
    if not src_path.exists():
        raise RenderError(f"missing single-source gate file: {src_path}")
    gate_src = _load(src_path)

    sides = [s for s in ("ref", "ours") if s in gate_src]
    if not sides:
        raise RenderError(f"{gate}: no [ref] or [ours] section")

    out: dict[str, tuple[str, dict[str, Any]]] = {}
    for side in sides:
        resolved = render_one(gate_src, side, baseline_inputs["baseline"])
        _validate_resolved(gate, side, resolved)
        cli, env = split_transport(resolved, env_whitelist)
        out[side] = (_serialize(cli, env, gate=gate, side=side), resolved)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--workspace", type=Path, help="write products under <ws>/{ref,workload/src}/config"
    )
    ap.add_argument(
        "--config-dir",
        type=Path,
        required=True,
        help="active axis config dir (model/optim/ref/data/registry)",
    )
    ap.add_argument(
        "--gates-dir",
        type=Path,
        help="single-source <gate>.toml dir (default: <config-dir>/<gate_config_dir>)",
    )
    ap.add_argument(
        "--registry", type=Path, help="registry eval.toml (default: <config-dir>/eval.toml)"
    )
    ap.add_argument("--model", type=Path, help="model.toml (default: <config-dir>/model.toml)")
    ap.add_argument("--optim", type=Path, help="optim.toml (default: <config-dir>/optim.toml)")
    ap.add_argument("--ref", type=Path, help="ref.toml (default: <config-dir>/ref.toml)")
    ap.add_argument("--data", type=Path, help="data.toml (default: <config-dir>/data.toml)")
    ap.add_argument("--only", help="comma-separated gate names (default: all *.toml in gates-dir)")
    ap.add_argument(
        "--out-dir", type=Path, help="write products here instead of --workspace layout"
    )
    ap.add_argument("--dry-run", action="store_true", help="print to stdout, do not write")
    ap.add_argument(
        "--skip-if-absent",
        action="store_true",
        help="exit 0 (no-op) when no gate sources resolve — for legacy suites",
    )
    args = ap.parse_args(argv)

    cfg = args.config_dir
    model = _load(args.model or cfg / "model.toml")
    optim = _load(args.optim or cfg / "optim.toml")
    ref = _load(args.ref or cfg / "ref.toml")
    data = _load(args.data or cfg / "data.toml")
    registry = _load(args.registry or cfg / "eval.toml")

    baseline = _build_baseline(model, optim, ref, data, registry)
    env_lib = {k.upper() for k in registry.get("env", {})}
    env_whitelist = {k.upper() for k in ENV_INFRA} | env_lib

    # Resolve the gate-source dir: explicit --gates-dir wins; otherwise the
    # single sources live in the per-loop config home alongside the axis
    # files at <config-dir>/<gate_config_dir>. A loop carries exactly one
    # active suite (its eval.toml), so the gate sources are a flat sibling
    # of ref.toml/model.toml/… — no per-suite nesting and no dependency on
    # [suite].name or --workspace for path resolution (--workspace is still
    # the product write root).
    gates_dir = args.gates_dir
    if gates_dir is None:
        gcd = registry.get("suite", {}).get("gate_config_dir", "gate_config")
        gates_dir = cfg / gcd
    if gates_dir is None or not gates_dir.exists():
        msg = f"no gate sources to render (gates_dir={gates_dir})"
        if args.skip_if_absent:
            print(f"render: {msg} — skipped", file=sys.stderr)
            return 0
        raise RenderError(
            f"{msg}; pass --gates-dir or seed <config-dir>/{registry.get('suite', {}).get('gate_config_dir', 'gate_config')}/"
        )

    if args.only:
        gates = [g.strip() for g in args.only.split(",") if g.strip()]
    else:
        gates = sorted(p.stem for p in gates_dir.glob("*.toml"))

    # Resolve + per-gate validate ALL gates first, so cross-gate invariants run
    # before any product is written (a render error must not leave a partial,
    # half-validated product tree behind).
    rendered: dict[str, dict[str, str]] = {}
    resolved_by_gate: dict[str, dict[str, dict[str, Any]]] = {}
    for gate in gates:
        products = render_gate(gate, gates_dir, {"baseline": baseline}, env_whitelist)
        rendered[gate] = {side: text for side, (text, _r) in products.items()}
        resolved_by_gate[gate] = {side: r for side, (_t, r) in products.items()}
    _validate_smoke_pairs(resolved_by_gate)
    rendered.update(_synthesize_mirror_variants(registry, resolved_by_gate, env_whitelist))

    for gate, sides in rendered.items():
        for side, text in sides.items():
            if args.dry_run or not (args.out_dir or args.workspace):
                print(f"\n{'=' * 70}\n# {gate} :: {side}\n{'=' * 70}")
                print(text)
                continue
            if args.out_dir:
                dest = args.out_dir / ("ref" if side == "ref" else "ours") / f"{gate}.toml"
            else:
                sub = "ref/config" if side == "ref" else "workload/src/config"
                dest = args.workspace / sub / f"{gate}.toml"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text)
            print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RenderError as exc:
        print(f"render error: {exc}", file=sys.stderr)
        sys.exit(2)
