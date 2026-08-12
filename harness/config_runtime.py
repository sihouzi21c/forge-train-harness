from __future__ import annotations

import functools
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from harness._compat import tomllib

__all__ = [
    "BACKENDS",
    "DEPLOYMENT_PATH_KEYS",
    "DeploymentPathError",
    "RuntimeDefaults",
    "RuntimeEnvDefaults",
    "artifact_subtree",
    "build_subprocess_env",
    "candidate_capture_basename",
    "canonical_workload_config_bytes",
    "default_output_tail_limit",
    "default_timeout_s",
    "dump_workload_config",
    "harness_config_path",
    "load_harness_config",
    "load_workload_config",
    "load_workload_config_from_file",
    "op_dir",
    "op_register_path",
    "op_worktree_path",
    "ops_registry_path",
    "ops_root_path",
    "ops_worktree_root",
    "prepend_pythonpath",
    "ref_backend",
    "ref_capture_basename",
    "ref_capture_script",
    "ref_script",
    "remove_pythonpath_entry",
    "repo_root",
    "resolve_assets",
    "resolve_suite_config",
    "runtime_defaults",
    "runtime_env_defaults",
    "suite_metadata",
    "suite_ref_timeout_s",
    "suite_timeout_s",
    "suites_by_stage",
    "validate_safe_relative_path",
    "validate_workload_config",
    "workload_src_path",
    "write_json",
]

_PACKAGE_HARNESS_DIR = Path(__file__).resolve().parent
_REPO_ROOT_ENV_VAR = "FORGE_REPO_ROOT"

# Env-var → [ref] key mapping for path overrides.
# Priority chain: TOML < env vars < CLI args (``path_overrides``).
_REF_ENV_OVERRIDES: dict[str, str] = {
    "FORGE_MEGATRON_ROOT": "megatron_root",
    "FORGE_CHECKPOINT_ROOT": "checkpoint_root",
    "FORGE_DATA_PATH": "data_path",
    "FORGE_REF_SCRIPT": "ref_script",
    "FORGE_BACKEND": "backend",
}

_REF_SOURCE_KEYS: frozenset[str] = frozenset({"megatron", "megatron_branch", "tokenizer"})

# Required scalar fields of the [model] table. The renderer
# (tools/render_gate_configs.py) is now the SOLE projection of these
# into the gate products' [cli]/[env] tables; this set exists only so
# ``_validate_model_config`` can still schema-check the axis template.
#
# ``name`` is intentionally absent — it is identity metadata, not
# a model-shape knob (it is validated separately below).
_MODEL_GEOMETRY_KEYS: frozenset[str] = frozenset(
    {
        "num_layers",
        "hidden_size",
        "ffn_hidden_size",
        "num_attention_heads",
        "num_query_groups",
        # Explicit attention head dimension. Required because hidden_size //
        # num_attention_heads is NOT always the head dim: MiniCPM5-1B uses
        # head_dim=128 with hidden_size=1536, heads=16 (1536//16=96 != 128).
        # 0.5B happens to satisfy hidden//heads == head_dim, which hid the
        # coupling — every [model] template now pins head_dim explicitly.
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
    }
)

_MODEL_REQUIRED_KEYS: frozenset[str] = _MODEL_GEOMETRY_KEYS | {"name"}

# Required scalar fields of the [optim] table. Like ``_MODEL_GEOMETRY_KEYS``
# this set exists only so ``_validate_optim_config`` can schema-check the
# axis template; the renderer is the SOLE projection of these into the
# gate products.
_OPTIM_REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "lr",
        "min_lr",
        "lr_warmup_iters",
        "lr_decay_iters",
        "lr_wsd_decay_iters",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "clip_grad",
    }
)

# Allowed values for [ref].backend. Each value names a sibling L0
# ref stack:
#   * "megatron" — Megatron-LM ref scripts (gsm8k binary or
#     modelbest_sdk weighted shards via DATA_CONF), ours-side trains
#     against Megatron BlendedMegatronDatasetBuilder + GPTDataset.
#   * "torch"    — pure-PyTorch ref (train_pure_mup_mtp.py + muP + MTP)
#     with modelbest_sdk weighted shards via DATA_CONF, ours-side
#     trains against a framework-native dataloader (no Megatron).
BACKENDS: frozenset[str] = frozenset({"megatron", "torch"})

#
# Keys whose empty value at the process boundary is treated as a fatal
# schema misconfiguration (i.e. raise ``DeploymentPathError`` from
# ``build_suite_env`` BEFORE the gate subprocess is spawned).
#
# Currently empty: ``checkpoint_root`` used to live here but is now
# auto-derived in ``_resolve_ref`` from the conventional layout
# ``<repo>/.artifacts/checkpoints/<backend>/``, with
# ``FORGE_CHECKPOINT_ROOT`` / ``--checkpoint-root`` as escape hatches.
# Backend-specific keys (``megatron_root`` / ``data_path`` /
# ``tokenizer_model``) are intentionally absent — they are still listed
# in suite ``env_inputs`` and lifted into the subprocess env, but an
# empty value is allowed at schema time so that the ours-side gate
# script can fail with a precise ``_require("MEGATRON_ROOT")`` message
# (or branch onto an alternative backend that does not need them).
DEPLOYMENT_PATH_KEYS: frozenset[str] = frozenset()


class DeploymentPathError(RuntimeError):
    """Raised when required deployment paths resolve to empty at the process boundary."""

    def __init__(self, missing: list[str]) -> None:
        self.missing = sorted(missing)
        hint = (
            "Configure these in config/ref.toml (cp config/ref/megatron_minicpm4_0.5b.toml "
            "config/ref.toml), or via FORGE_* env vars."
        )
        super().__init__(f"Missing deployment paths: {self.missing}. {hint}")


# ──────────────────────────────────────────────────────────────────────
# Source / data declaration validation + resolution
# ──────────────────────────────────────────────────────────────────────


def _is_local_path(value: str) -> bool:
    """True if *value* looks like a filesystem path (exists on disk)."""
    p = Path(value)
    return p.is_dir() or p.is_file()


def _validate_ref_source_fields(path: Path, ref: dict[str, Any]) -> None:
    """Validate source-declaration fields within the [ref] table."""
    if "megatron" in ref and not isinstance(ref["megatron"], str):
        raise ValueError(f"{path}: [ref].megatron must be a string")
    if "megatron_branch" in ref:
        if "megatron" not in ref:
            raise ValueError(f"{path}: [ref].megatron_branch requires [ref].megatron")
        if not isinstance(ref["megatron_branch"], str):
            raise ValueError(f"{path}: [ref].megatron_branch must be a string")
    if "tokenizer" in ref and not isinstance(ref["tokenizer"], str):
        raise ValueError(f"{path}: [ref].tokenizer must be a string")


def _validate_model_config(path: Path, model: dict[str, Any]) -> None:
    """Validate the [model] table schema.

    Required keys: every field in ``_MODEL_GEOMETRY_KEYS`` plus
    ``name`` (identity metadata). Each field must be a scalar
    (str / int / float; bool rejected because env values are strings).
    Unknown keys are rejected so typos in a custom config/model.toml
    surface immediately rather than silently fall back to the default.
    """
    unknown = set(model) - _MODEL_REQUIRED_KEYS
    if unknown:
        raise ValueError(f"{path}: [model] has unknown keys: {sorted(unknown)}")
    missing = sorted(k for k in _MODEL_REQUIRED_KEYS if k not in model)
    if missing:
        raise ValueError(f"{path}: [model] missing required keys: {missing}")
    if not isinstance(model.get("name"), str) or not model["name"]:
        raise ValueError(f"{path}: [model].name must be a non-empty string")
    for key in _MODEL_GEOMETRY_KEYS:
        val = model[key]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(
                f"{path}: [model].{key} must be int or float (got {type(val).__name__})"
            )


def _validate_optim_config(path: Path, optim: dict[str, Any]) -> None:
    """Validate the [optim] table schema.

    Same shape as ``_validate_model_config`` but for optimizer knobs.
    Every field in ``_OPTIM_REQUIRED_KEYS`` must be present and a
    scalar number; bools and unknown keys are rejected.
    """
    unknown = set(optim) - _OPTIM_REQUIRED_KEYS
    if unknown:
        raise ValueError(f"{path}: [optim] has unknown keys: {sorted(unknown)}")
    missing = sorted(k for k in _OPTIM_REQUIRED_KEYS if k not in optim)
    if missing:
        raise ValueError(f"{path}: [optim] missing required keys: {missing}")
    for key in _OPTIM_REQUIRED_KEYS:
        val = optim[key]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(
                f"{path}: [optim].{key} must be int or float (got {type(val).__name__})"
            )


def _validate_data_config(path: Path, data: dict[str, Any]) -> None:
    """Validate [data] table schema.

    Two data-source mechanisms coexist for different ref families and must NOT be
    mixed in one table (the "two paths" SSOT hazard):
      * inline ``data_path`` (+ ``data_loader``) — read directly via
        FORGE_DATA_TOML (meta-generated bundles);
      * ``conf_path`` / ``conf_name`` — a shell conf the dispatcher sources for
        DATA_PATH (native refs).
    ``data_path`` is therefore mutually exclusive with ``conf_path``/``conf_name``
    so a loop's data value has exactly one source.
    """
    allowed_keys = {
        # inline source (meta bundles read these via FORGE_DATA_TOML)
        "data_path",
        "data_loader",
        # conf source (native refs source a shell conf for DATA_PATH)
        "conf_name",
        "conf_path",
        "dataset",
        "forge_data_dir",
        "download_files",
        # Image bake location of the dev slice; alignment–long-horizon seed the
        # workspace-relative forge_data_dir by copying from here instead of
        # re-downloading (see _ensure_dataset).
        "baked_data_dir",
        # production background corpus prefetch (tools/prefetch_data.py). 0/absent =
        # off (no behavior change); >0 stages that many GB of [data].dataset
        # into forge_data_dir before the production production long-train consumes it.
        "prefetch_target_gb",
        "prefetch_en_fraction",
        "prefetch_wait_timeout_s",
    }
    unknown = set(data) - allowed_keys
    if unknown:
        raise ValueError(f"{path}: [data] has unknown keys: {sorted(unknown)}")
    if "conf_name" in data and "conf_path" in data:
        raise ValueError(f"{path}: [data] conf_name and conf_path are mutually exclusive")
    # One source only: inline data_path XOR a conf (conf_path/conf_name).
    if "data_path" in data and ("conf_path" in data or "conf_name" in data):
        raise ValueError(
            f"{path}: [data] declares both an inline data_path and a "
            f"conf_path/conf_name — data must come from exactly one source"
        )
    if "data_path" in data and (not isinstance(data["data_path"], str) or not data["data_path"]):
        raise ValueError(f"{path}: [data].data_path must be a non-empty string")
    if "conf_name" in data and (not isinstance(data["conf_name"], str) or not data["conf_name"]):
        raise ValueError(f"{path}: [data].conf_name must be a non-empty string")
    if "conf_path" in data and (not isinstance(data["conf_path"], str) or not data["conf_path"]):
        raise ValueError(f"{path}: [data].conf_path must be a non-empty string")


def _resolve_megatron(root: Path, source: dict[str, Any]) -> str:
    """Resolve [ref].megatron → absolute path to Megatron source tree.

    Resolution order (post-Commit-D, hermetic):
    1. Local directory (value is a path that exists) → use as-is.
    2. ``<workspace>/.resources/megatron/*/`` entry from the
       external-resources manifest if any matches.
    3. Vendored submodule fallback → ``third_party/megatron/v15``.

    The legacy "git clone on demand" branch is gone: provisioning is
    now the operator's responsibility (``harness resources provision``)
    so a fresh workspace cannot silently spend 30 s cloning Megatron
    during the agent's first read.
    """
    raw = source.get("megatron", "")
    if raw and _is_local_path(raw):
        return str(Path(raw).resolve())

    for candidate in sorted((root / ".resources" / "megatron").glob("*")):
        if candidate.is_dir():
            return str(candidate.resolve())

    submodule = root / "harness" / "third_party" / "megatron" / "v15"
    if submodule.is_dir() and any(submodule.iterdir()):
        return str(submodule)
    return ""


def _resolve_data_conf(root: Path, ref: dict[str, Any], data_cfg: dict[str, Any]) -> str:
    """Resolve [data].conf_name or [data].conf_path to an absolute DATA_CONF path.

    conf_path is used as-is if absolute; relative paths are resolved against
    the harness repo root so in-repo data_conf scripts like ref/reference/*.sh
    are portable across checkouts. conf_name is resolved relative to the
    Megatron source tree:
    ${megatron_root}/examples/scaling/data_conf/{name}.sh
    """
    conf_path = data_cfg.get("conf_path", "")
    if conf_path:
        if not Path(conf_path).is_absolute():
            conf_path = str((root / conf_path).resolve())
        return conf_path

    conf_name = data_cfg.get("conf_name", "")
    if not conf_name:
        return ""
    megatron_root = ref.get("megatron_root", "")
    if not megatron_root:
        return ""
    return f"{megatron_root}/examples/scaling/data_conf/{conf_name}.sh"


def _resolve_data_env_from_conf(data_conf: str) -> dict[str, str]:
    """Source a data_conf shell file and return the resolved ``data_path``.

    The conf's sole contract is exporting ``DATA_PATH`` (the weighted shard
    token string). The loader kind is NOT lifted here: it lives in
    config/data.toml ``[data].data_loader`` and every side reads it from that
    file (ref/sisters ``--data-config``, ours ``runtime_config.data_loader()``,
    meta ``ref_bundle_run.sh``), never from a ``DATA_LOADER`` env value.
    """
    conf_path = Path(data_conf)
    if not conf_path.is_file():
        raise FileNotFoundError(f"DATA_CONF file not found: {data_conf}")
    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                'set -euo pipefail; source "$1"; '
                'if [[ -z "${DATA_PATH:-}" ]]; then '
                'echo "DATA_PATH is not set by DATA_CONF=$1" >&2; exit 1; '
                'fi; printf "%s" "$DATA_PATH"'
            ),
            "bash",
            str(conf_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {"data_path": result.stdout}


def _derive_checkpoint_root(root: Path, backend: str) -> Path:
    """Convention-based fallback for ``[ref].checkpoint_root``.

    Returns ``<repo>/.artifacts/checkpoints/<backend>/`` so the value
    never has to appear in any TOML. Explicit overrides (TOML field,
    ``FORGE_CHECKPOINT_ROOT`` env, ``--checkpoint-root`` CLI flag)
    still win because this helper only runs when the field is empty.
    A future ``[model]`` axis may extend the suffix
    (e.g. ``<model.name>_<backend>``); the convention lives here so
    every consumer reads it through ``_resolve_ref``.
    """
    return (root / ".artifacts" / "checkpoints" / backend).resolve()


def resolve_forge_data_dir(forge_data_dir: str, root: Path | None = None) -> Path:
    """Lift a (possibly relative) ``[data].forge_data_dir`` to an absolute path.

    The corpus prefetch (writer), the gate dataloader (reader, via the
    exported ``FORGE_DATA_DIR``) and the dispatcher's prefetch-sentinel
    wait must all agree on ONE location. A relative value is resolved
    against the harness repo root — NOT the process cwd — so every
    consumer lands on the same dir wherever it runs (e.g.
    ``.artifacts/forge-data/...`` under the workspace, which
    ``harness sync push`` already excludes from rsync). Mirrors the
    ``checkpoint_root`` convention in :func:`_resolve_ref`. An absolute
    value is returned verbatim (it may be a deliberate scratch mount such
    as ``/opt/forge-data`` whose symlinks must not be collapsed).

    NOTE: ``tools.prefetch_data`` — a lower DAG layer that must not import
    ``harness`` — re-implements this same one-liner against its own
    ``--repo-root``; keep the two in lock-step.
    """
    p = Path(forge_data_dir)
    if p.is_absolute():
        return p
    return ((root or repo_root()) / p).resolve()


def _resolve_ref(path: Path, data: dict[str, Any]) -> None:
    """Resolve source declarations + [data] conf into [ref] path values.

    Called after env/CLI overrides so user-specified paths always win.
    Only fills keys that are still empty in [ref].

    ``checkpoint_root`` is auto-derived from the convention
    ``<repo>/.artifacts/checkpoints/<backend>/`` when empty (see
    :func:`_derive_checkpoint_root`); explicit relative paths in [ref]
    are still lifted to an absolute path under the harness repo root.
    """
    root = repo_root()
    ref = data.setdefault("ref", {})
    data_cfg = data.get("data", {})

    ckpt = ref.get("checkpoint_root", "")
    if ckpt:
        if not Path(ckpt).is_absolute():
            ref["checkpoint_root"] = str((root / ckpt).resolve())
    else:
        backend = ref.get("backend") or "megatron"
        ref["checkpoint_root"] = str(_derive_checkpoint_root(root, str(backend)))

    has_source_fields = any(k in ref for k in _REF_SOURCE_KEYS)
    if has_source_fields:
        _validate_ref_source_fields(path, ref)
        if not ref.get("megatron_root"):
            resolved = _resolve_megatron(root, ref)
            if resolved:
                ref["megatron_root"] = resolved

    if isinstance(data_cfg, dict) and data_cfg:
        _validate_data_config(path, data_cfg)
        if "conf_name" in data_cfg or "conf_path" in data_cfg:
            if not ref.get("data_conf"):
                resolved = _resolve_data_conf(root, ref, data_cfg)
                if resolved:
                    ref["data_conf"] = resolved
            if not ref.get("data_path") and ref.get("data_conf"):
                resolved_env = _resolve_data_env_from_conf(str(ref["data_conf"]))
                ref["data_path"] = resolved_env["data_path"]


@dataclass(frozen=True, slots=True)
class RuntimeDefaults:
    report: str
    gpu: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeEnvDefaults:
    """Distributed runtime values from ``config/eval.toml [runtime.distributed]``.

    No hardcoded fallbacks — the TOML is the single source of truth.
    """

    master_addr: str
    master_port: str


def repo_root() -> Path:
    env_root = os.environ.get(_REPO_ROOT_ENV_VAR)
    cwd = str(Path.cwd().resolve())
    return _resolve_repo_root(cwd, env_root)


def harness_config_path() -> Path:
    return _PACKAGE_HARNESS_DIR / "config" / "defaults.toml"


def _user_config_dir() -> Path:
    env = os.environ.get("FORGE_CONFIG_DIR")
    if env:
        return Path(env)
    return repo_root() / "config"


def _default_workload_config_path(*, include_user_config: bool = True) -> Path:
    if include_user_config:
        user_copy = _user_config_dir() / "eval.toml"
        if user_copy.exists():
            return user_copy
    return repo_root() / "config" / "eval" / "dense_training" / "dense_training.toml"


def _local_harness_config_path() -> Path:
    return _user_config_dir() / "harness.local.toml"


def _agent_config_path() -> Path:
    """User's agent profile (gitignored): config/agent.toml.

    Created by copying one of the committed profiles under config/agent/*.toml.
    """
    return _user_config_dir() / "agent.toml"


def _ref_config_path() -> Path:
    """User's ref stack config (gitignored): config/ref.toml."""
    return _user_config_dir() / "ref.toml"


def _data_config_path() -> Path:
    """User's data source config (gitignored): config/data.toml."""
    return _user_config_dir() / "data.toml"


def _remote_config_path() -> Path:
    """User's execution topology config (gitignored): config/remote.toml."""
    return _user_config_dir() / "remote.toml"


def _model_config_path() -> Path:
    """User's model config (gitignored): config/model.toml.

    Required — must exist at the user's checkout. Created by copying
    one of the committed templates under config/model/*.toml. This
    file is the single source of truth for model architecture
    hyperparameters; both ref-side ``FORGE_*`` injection and engine-
    side ``runtime_config.load()`` read from it.
    """
    return _user_config_dir() / "model.toml"


def _optim_config_path() -> Path:
    """User's optimizer config (gitignored): config/optim.toml.

    Required — must exist at the user's checkout. This file is the
    single source of truth for optimizer hyperparameters; both ref-
    side ``FORGE_*`` injection and engine-side ``runtime_config.load()``
    read from it. There is intentionally no auto-template fallback and
    no "opt-in" branch: a missing file would mean ref and engine
    disagree on which defaults are in effect, which is the SSOT
    violation this axis exists to prevent.
    """
    return _user_config_dir() / "optim.toml"


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


# ──────────────────────────────────────────────────────────────────────
# Content-addressed workload-config protocol
# ──────────────────────────────────────────────────────────────────────
#
# Parent → child config delivery used to cherry-pick a hand-rolled subset
# of keys into ``request["workload_config"]`` (see ``harness.app``'s
# ``_workload_config_snapshot``). Any axis silently dropped from that
# subset — historically ``[optim]`` and ``[model]`` — fell back to
# FORGE_* env-var injection at the bash-launcher seam, and a single
# unset env var would silently switch the run to a hardcoded default
# without raising. That class of silent drift cost ~60 min of agent
# debug time on each occurrence.
#
# The fix is structural: serialize the full validated config as
# deterministic bytes, hash them, and deliver path + sha to the child.
# Any byte-level divergence (missed field, accidental override, stale
# cache) is caught by ``load_workload_config_from_file`` before the
# child ever consults the dict.

_WORKLOAD_CONFIG_FILENAME = "workload_config.json"


def canonical_workload_config_bytes(cfg: dict[str, Any]) -> bytes:
    """Serialize *cfg* to the canonical JSON form used for hashing.

    The serializer pins ``sort_keys=True`` (stable order across Python
    dict hash randomization), ``indent=2`` (human-greppable for triage),
    ``ensure_ascii=False`` (preserve non-ASCII model names verbatim),
    and a trailing newline (POSIX text-file convention so ``sha256sum``
    on disk matches what we hash in memory).
    """
    text = json.dumps(cfg, sort_keys=True, indent=2, default=str, ensure_ascii=False)
    return (text + "\n").encode("utf-8")


def dump_workload_config(cfg: dict[str, Any], dst: Path) -> tuple[Path, str]:
    """Write *cfg* to *dst* canonically and return ``(path, sha256_hex)``.

    Companion writes a ``<dst>.sha256`` sidecar in the ``sha256sum``
    format (``"<hex>  <basename>\\n"``) so external tools can verify
    the file without re-implementing the canonicalization.
    """
    payload = canonical_workload_config_bytes(cfg)
    digest = hashlib.sha256(payload).hexdigest()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(payload)
    sidecar = dst.with_suffix(dst.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {dst.name}\n", encoding="utf-8")
    return dst, digest


def load_workload_config_from_file(path: Path, *, expected_sha: str) -> dict[str, Any]:
    """Read *path* and assert its sha256 equals *expected_sha*.

    The first thing every subprocess that consumes a serialized config
    MUST do — guards against (a) the file being mutated between dump
    and read, (b) the child reading a different file than the parent
    intended (PATH or PYTHONPATH mismatch), and (c) accidental encoding
    drift.
    """
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha:
        raise ValueError(
            f"workload_config sha mismatch: {path} hashes to {actual} "
            f"but parent declared {expected_sha}. The file was modified "
            f"between dump and read, or the child resolved a different "
            f"path than the parent intended."
        )
    return json.loads(raw.decode("utf-8"))


def workload_src_path(root: Path) -> Path:
    return root / "workload" / "src"


# ──────────────────────────────────────────────────────────────────────
# Stage 2 per-operator path SSOT
# ──────────────────────────────────────────────────────────────────────
#
# Single source of truth for the on-disk layout of Stage 2 operators —
# directory structure, registry filename, register filename, worktree
# location. Every callsite under ``evals/`` and ``tools/`` MUST route
# through these helpers; ``test_layer_dag``'s sibling guard test
# ``TestStage2OpPathSSOT`` (in ``test_dispatcher_behavior.py``) refuses
# any new file that re-builds these paths from string literals.
#
# Layout owned here:
#
#   workload/ops/                        ← ``ops_root_path(root)``
#   workload/ops/_registry.toml          ← ``ops_registry_path(root)``
#   workload/ops/<op>/                   ← ``op_dir(root, op)``
#   workload/ops/<op>/register.toml      ← ``op_register_path(root, op)``
#   ops_worktree/                        ← ``ops_worktree_root(root)``
#   ops_worktree/<op>/                   ← ``op_worktree_path(root, op)``


def ops_root_path(root: Path) -> Path:
    """Directory holding every Stage 2 operator workspace."""
    return root / "workload" / "ops"


def ops_registry_path(root: Path) -> Path:
    """Path of the cross-operator registry TOML (M1 scout output)."""
    return ops_root_path(root) / "_registry.toml"


def op_dir(root: Path, op_name: str) -> Path:
    """Directory holding a single operator's workspace (PROMPT.md / kernel.py / …)."""
    return ops_root_path(root) / op_name


def op_register_path(root: Path, op_name: str) -> Path:
    """Per-op ``register.toml`` (env_var / default / available variants)."""
    return op_dir(root, op_name) / "register.toml"


def ops_worktree_root(root: Path) -> Path:
    """Directory holding all per-op git worktrees (M1 scout creates one per op)."""
    return root / "ops_worktree"


def op_worktree_path(root: Path, op_name: str) -> Path:
    """Per-op git worktree (sparse-checkout of ``stage2/op/<name>`` branch)."""
    return ops_worktree_root(root) / op_name


def prepend_pythonpath(env: dict[str, str], path: Path) -> dict[str, str]:
    updated = dict(env)
    existing_pythonpath = updated.get("PYTHONPATH", "")
    path_text = str(path)
    updated["PYTHONPATH"] = (
        f"{path_text}:{existing_pythonpath}" if existing_pythonpath else path_text
    )
    return updated


def remove_pythonpath_entry(env: dict[str, str], path: Path) -> dict[str, str]:
    updated = dict(env)
    path_text = str(path)
    pythonpath = [
        entry for entry in updated.get("PYTHONPATH", "").split(":") if entry and entry != path_text
    ]
    if pythonpath:
        updated["PYTHONPATH"] = ":".join(pythonpath)
    else:
        updated.pop("PYTHONPATH", None)
    return updated


def build_subprocess_env(
    *,
    repo_root: Path,
    source_env: dict[str, str] | None = None,
    extra: dict[str, str] | None = None,
    cuda_visible_devices: str | None = None,
    prepend_workload_src: bool = False,
    remove_workload_src: bool = False,
    prepend_repo_root: bool = True,
) -> dict[str, str]:
    """Build subprocess env at the single harness runtime boundary.

    ``prepend_repo_root`` (default True) puts the harness repo root on
    PYTHONPATH so evals/scripts/*.py subprocesses can ``from evals._common
    import ...`` without each script doing its own ``sys.path`` surgery.
    """
    env = dict(os.environ if source_env is None else source_env)
    workload_src = workload_src_path(repo_root)
    if prepend_workload_src:
        env = prepend_pythonpath(env, workload_src)
    if remove_workload_src:
        env = remove_pythonpath_entry(env, workload_src)
    if prepend_repo_root:
        env = prepend_pythonpath(env, repo_root)
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    return env


# Asset resolution: TOML carries the on-disk directory and the HF repo id;
# this layer downloads from the HF Hub on cache miss and writes two env vars
# (FORGE_TOKENIZER_DIR, FORGE_DATA_DIR) that the gate subprocess reads.
# The downloader has no knowledge of which tokenizer or dataset is in play —
# it only operates on the (repo, dir) pairs the TOML provides.


def _ensure_tokenizer(repo: str, dest: Path) -> None:
    """Cache-or-fetch the tokenizer artifacts. No-op when ``dest`` is non-empty.

    Pulls ``tokenizer*`` / ``special_tokens*`` only (the dataloader needs
    just the tokenizer file). Layout written by ``snapshot_download``
    matches the HF repo root, which is also where the bake places them.
    """
    if dest.is_dir() and any(dest.iterdir()):
        return
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(  # nosec B615 — dataset revision is pinned by the data config, not the API arg
        repo_id=repo,
        repo_type="model",
        local_dir=str(dest),
        allow_patterns=["tokenizer*", "special_tokens*"],
    )


def _ensure_dataset(
    repo: str,
    dest: Path,
    files: dict[str, str] | None,
    baked_dir: Path | None = None,
) -> None:
    """Per-file cache-or-fetch into the (workspace-relative) ``dest``.

    ``files`` is a ``{local_rel: hf_repo_path}`` map. For every entry whose
    local target is missing, seed ``dest / local_rel`` by COPYING from
    ``baked_dir / local_rel`` when the image baked the dev slice there
    (``[data].baked_data_dir``) — a fast local copy that avoids the HF
    mirror, which stalls behind whitelist-proxy clusters (shandong). Only
    entries with no baked copy fall back to ``hf_hub_download``. Copying
    (not symlinking) keeps ``dest`` a real, writable tree so the production
    prefetch can extend the same dir without writing through into the
    shared baked location. The local layout is fully declared by the
    caller (TOML), so this has no knowledge of asset names or repo layout.
    """
    if not files:
        return
    missing = [(k, v) for k, v in files.items() if not (dest / k).is_file()]
    if not missing:
        return
    import shutil

    hf_missing: list[tuple[str, str]] = []
    for local_rel, hf_path in missing:
        target = dest / local_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        baked = (baked_dir / local_rel) if baked_dir else None
        if baked is not None and baked.is_file():
            shutil.copyfile(baked, target)
        else:
            hf_missing.append((local_rel, hf_path))

    if hf_missing:
        from huggingface_hub import hf_hub_download

        for local_rel, hf_path in hf_missing:
            target = dest / local_rel
            cached = hf_hub_download(  # nosec B615 — same pinned-by-config rationale as snapshot_download above
                repo_id=repo,
                repo_type="dataset",
                filename=hf_path,
            )
            shutil.copyfile(cached, target)


def resolve_assets(ref: dict[str, Any], data: dict[str, Any]) -> dict[str, str]:
    """Return ``{FORGE_TOKENIZER_DIR, FORGE_DATA_DIR}`` for the gate subprocess.

    Triggers an HF-hub download on cache miss against the TOML-declared
    target dir. Fails fast on any missing required TOML field.

    Also writes the resolved values into ``os.environ`` so the in-process
    ``_resolve_data_env_from_conf`` (which sources the bash data_conf via
    a subprocess inheriting the parent env) sees them.
    """
    out: dict[str, str] = {}

    tok_dir = ((ref or {}).get("forge_tokenizer_dir") or "").strip()
    tok_repo = ((ref or {}).get("tokenizer") or "").strip()
    if not tok_dir:
        raise ValueError("[ref].forge_tokenizer_dir is required")
    if not tok_repo:
        raise ValueError("[ref].tokenizer is required (HF repo id)")
    _ensure_tokenizer(tok_repo, Path(tok_dir))
    out["FORGE_TOKENIZER_DIR"] = tok_dir

    ds_repo = ((data or {}).get("dataset") or "").strip()
    if ds_repo:
        ds_dir = ((data or {}).get("forge_data_dir") or "").strip()
        if not ds_dir:
            raise ValueError("[data].forge_data_dir is required when [data].dataset is set")
        ds_dir_abs = resolve_forge_data_dir(ds_dir)
        baked = ((data or {}).get("baked_data_dir") or "").strip()
        _ensure_dataset(
            ds_repo,
            ds_dir_abs,
            (data or {}).get("download_files") or None,
            baked_dir=Path(baked) if baked else None,
        )
        # Export the ABSOLUTE dir: the data_conf is sourced in a subprocess
        # (``local_env=FORGE_DATA_DIR``) whose cwd is not guaranteed, so a
        # relative value would resolve against the wrong directory.
        out["FORGE_DATA_DIR"] = str(ds_dir_abs)

    os.environ.update(out)
    return out


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into a shallow copy of *base*.

    - Dict values are merged recursively (override keys win).
    - Non-dict values in *override* replace *base* entirely.
    - Keys present only in *base* are preserved.
    """
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_harness_config() -> dict[str, Any]:
    """Load package defaults, then deep-merge any ``harness.local.toml`` on top."""
    config = _read_toml(harness_config_path())
    local = _load_local_harness_config()
    if local:
        config = _deep_merge(config, local)
    return config


def _load_local_harness_config() -> dict[str, Any]:
    path = _local_harness_config_path()
    if not path.exists():
        return {}
    return _read_toml(path)


def _apply_ref_env_overrides(data: dict[str, Any]) -> None:
    """Overlay environment variables onto [ref] path fields."""
    ref = data.get("ref")
    if isinstance(ref, dict):
        for env_var, cfg_key in _REF_ENV_OVERRIDES.items():
            value = os.environ.get(env_var)
            if value:
                ref[cfg_key] = value


def _apply_ref_path_overrides(
    data: dict[str, Any],
    overrides: dict[str, str],
) -> None:
    """Apply explicit CLI path overrides onto [ref]."""
    ref = data.get("ref")
    if not isinstance(ref, dict):
        return
    for key, value in overrides.items():
        if value:
            ref[key] = value


def load_workload_config(
    path_text: str | None = None,
    *,
    path_overrides: dict[str, str] | None = None,
    include_user_config: bool = True,
) -> tuple[Path, dict[str, Any]]:
    if path_text is None:
        path = _default_workload_config_path(include_user_config=include_user_config).resolve()
    else:
        candidate = Path(path_text)
        if not candidate.is_absolute():
            candidate = repo_root() / candidate
        path = candidate.resolve()
    data = _read_toml(path)
    if include_user_config:
        for layer_path in [
            _ref_config_path(),
            _data_config_path(),
            _remote_config_path(),
            _agent_config_path(),
        ]:
            if layer_path.exists():
                data = _deep_merge(data, _read_toml(layer_path))
        for required_path in (_model_config_path(), _optim_config_path()):
            if not required_path.exists():
                raise FileNotFoundError(
                    f"Required SSOT config missing: {required_path}. "
                    f"Copy a template from "
                    f"{repo_root() / 'config' / required_path.stem}/ into "
                    f"$FORGE_CONFIG_DIR (defaults to <repo>/config) — e.g. "
                    f"`cp {repo_root() / 'config' / required_path.stem}/default.toml "
                    f"{required_path}`. This axis is the single source of truth for "
                    f"its hyperparameters; both ref-side FORGE_* injection and "
                    f"engine-side runtime_config.load() depend on it."
                )
            data = _deep_merge(data, _read_toml(required_path))
        ref = data.setdefault("ref", {})
        for key in ("megatron_root", "data_path", "tokenizer_model"):
            ref.setdefault(key, "")
    _apply_ref_env_overrides(data)
    if path_overrides:
        _apply_ref_path_overrides(data, path_overrides)
    _resolve_ref(path, data)
    validate_workload_config(path, data)
    return path, data


def suite_metadata(workload_config: dict[str, Any]) -> dict[str, Any]:
    """Return configured suite metadata, including local suite annotations."""
    suites = {
        name: {**cfg, "local": False, "requires_cuda": bool(cfg.get("requires_cuda", True))}
        for name, cfg in workload_config.get("evals", {}).items()
    }
    for name, cfg in workload_config.get("local_suites", {}).items():
        suites[name] = {**cfg, "local": True, "requires_cuda": False}
    return suites


def resolve_suite_config(workload_config: dict[str, Any], suite: str) -> dict[str, Any]:
    suites = workload_config.get("evals", {})
    suite_cfg = suites.get(suite)
    if not isinstance(suite_cfg, dict):
        raise ValueError(f"Unknown suite: {suite}")
    resolved: dict[str, Any] = dict(workload_config.get("ref", {}))
    stage = suite_cfg.get("stage")
    if isinstance(stage, str):
        stage_defaults = workload_config.get(stage, {})
        if isinstance(stage_defaults, dict):
            resolved = _deep_merge(resolved, stage_defaults)
    return _deep_merge(resolved, suite_cfg)


def suites_by_stage(
    workload_config: dict[str, Any],
    *,
    include_local: bool = False,
) -> dict[str, list[str]]:
    """Group configured suite names by their declared stage."""
    suites = suite_metadata(workload_config) if include_local else workload_config.get("evals", {})
    grouped: dict[str, list[str]] = {}
    for suite, cfg in suites.items():
        stage = str(cfg.get("stage", "uncategorized"))
        grouped.setdefault(stage, []).append(suite)
    return {stage: sorted(names) for stage, names in sorted(grouped.items())}


def validate_workload_config(path: Path, data: dict[str, Any]) -> None:
    workload = data.get("workload")
    if not isinstance(workload, dict):
        raise ValueError(f"{path}: missing [workload]")
    for key in ("id", "display_name", "requires_cuda"):
        if key not in workload:
            raise ValueError(f"{path}: workload.{key} is required")

    env = data.get("env", {})
    if not isinstance(env, dict):
        raise ValueError(f"{path}: [env] must be a table")
    _validate_global_env_sources(path, data, env)
    runtime = data.get("runtime", {})
    if runtime is not None and not isinstance(runtime, dict):
        raise ValueError(f"{path}: [runtime] must be a table")
    distributed = runtime.get("distributed", {}) if isinstance(runtime, dict) else {}
    if not isinstance(distributed, dict):
        raise ValueError(f"{path}: [runtime.distributed] must be a table")
    for key in ("master_addr", "master_port"):
        if key not in distributed:
            raise ValueError(f"{path}: runtime.distributed.{key} is required")
        if not isinstance(distributed[key], str) or not distributed[key]:
            raise ValueError(f"{path}: runtime.distributed.{key} must be a non-empty string")
    # ``master_port`` accepts either a numeric string (legacy fixed
    # port like "29500") or the sentinel ``"auto"`` which expands at
    # call time to a freshly-allocated free ephemeral port. The
    # sentinel is the supported default on shared devspaces where a
    # literal port collides with co-tenants and lingers in TIME_WAIT
    # after SIGKILL. See ``evals._common._resolve_master_port``.
    master_port_value = distributed["master_port"]
    if master_port_value != "auto" and not master_port_value.isdigit():
        raise ValueError(
            f'{path}: runtime.distributed.master_port must be "auto" or a '
            f"numeric port string (got {master_port_value!r})"
        )

    valid_stages = {"stage1", "stage2"}
    suites = data.get("evals", {})
    if not isinstance(suites, dict):
        raise ValueError(f"{path}: [evals] must be a table")
    for suite, cfg in suites.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"{path}: evals.{suite} must be a table")
        stage = cfg.get("stage")
        if stage not in valid_stages:
            raise ValueError(f"{path}: evals.{suite}.stage must be one of {sorted(valid_stages)}")
        if "requires_cuda" in cfg and not isinstance(cfg["requires_cuda"], bool):
            raise ValueError(f"{path}: evals.{suite}.requires_cuda must be a boolean")
        if "execution_scope" in cfg and cfg["execution_scope"] != "suite-runner":
            raise ValueError(f"{path}: evals.{suite}.execution_scope must be 'suite-runner'")
        # env_inputs is a stage2-only key (suite_process_env lifting on the
        # op-* path); stage1 suites route through the scripted executor and
        # deliver values via rendered products + runtime_env.py, so the old
        # "script implies env_inputs" coupling is retired.
        if "env_inputs" in cfg and not isinstance(cfg["env_inputs"], list):
            raise ValueError(f"{path}: evals.{suite}.env_inputs must be a list")
        if "env_inputs" in cfg:
            for item in cfg["env_inputs"]:
                if not isinstance(item, str) or not item or item.upper() != item:
                    raise ValueError(
                        f"{path}: evals.{suite}.env_inputs entries must be uppercase strings"
                    )
        if not isinstance(cfg.get("runner_kind"), str) or not cfg.get("runner_kind"):
            raise ValueError(f"{path}: evals.{suite}.runner_kind must be declared")
        _validate_suite_args_metadata(path, f"evals.{suite}", cfg)
        _validate_suite_ref_env(path, suite, cfg)
        _validate_suite_ours_env(path, suite, cfg)
        _validate_suite_ref_extra_args(path, suite, cfg)

    if "op-long" in suites:
        op_long_env = set(suites["op-long"].get("env_inputs", []))
        required = {
            "CHECKPOINT_ROOT",
            "MEGATRON_ROOT",
            "DATA_PATH",
            "MICRO_BATCH_SIZE",
            "NUM_STEPS",
            "GLOBAL_BATCH_SIZE",
            "NUM_PROCS",
            "MASTER_ADDR",
            "MASTER_PORT",
            "OP_NAMES",
            # SEED / SEQ_LENGTH / GRAD_ACCUM_STEPS were promoted from
            # ``evals/scripts/op_long_ours.py`` defaults to dispatcher-
            # injected env (see issue #7 in the harness audit) so the
            # ours-side trajectory cannot drift from the ref-script
            # capture shape on a missing env var.
            "SEED",
            "SEQ_LENGTH",
            "GRAD_ACCUM_STEPS",
        }
        missing = sorted(required - op_long_env)
        if missing:
            raise ValueError(
                f"{path}: evals.op-long.env_inputs missing required entries: {missing}"
            )

    local_suites = data.get("local_suites", {})
    if not isinstance(local_suites, dict):
        raise ValueError(f"{path}: [local_suites] must be a table")
    for suite, cfg in local_suites.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"{path}: local_suites.{suite} must be a table")
        if cfg.get("stage") != "local":
            raise ValueError(f"{path}: local_suites.{suite}.stage must be 'local'")
        if not isinstance(cfg.get("runner_kind"), str) or not cfg.get("runner_kind"):
            raise ValueError(f"{path}: local_suites.{suite}.runner_kind must be declared")
        # Apply the same args_min / args_max / args_unbounded type
        # constraints to local suites so ``harness run guard <typo>``
        # fails identically to ``harness run forward-align <typo>``.
        _validate_suite_args_metadata(path, f"local_suites.{suite}", cfg)

    _validate_automation_config(path, data.get("automation", {}))
    _validate_agent_config(path, data.get("agent", {}))

    model = data.get("model")
    if isinstance(model, dict) and model:
        _validate_model_config(path, model)

    optim = data.get("optim")
    if isinstance(optim, dict) and optim:
        _validate_optim_config(path, optim)


def _validate_automation_config(path: Path, automation: Any) -> None:
    if automation is None:
        return
    if not isinstance(automation, dict):
        raise ValueError(f"{path}: [automation] must be a table")
    if "stage2" in automation:
        _validate_automation_stage2(path, automation["stage2"])


def _validate_automation_stage2(path: Path, cfg: Any) -> None:
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: [automation.stage2] must be a table")
    # Stage 2 orchestration is goal-driven: each subagent does one
    # per-round linear decision-tree step and exits; cross-round
    # progression is driven by `agent-loop.sh` re-fan-out, and the
    # merge happens once `harness run op-long` PASSes (see
    # prompt/develop_prompt/_shared/stage2-subagent-playbook.md).
    _require_positive_int(path, cfg, "automation.stage2.max_concurrent", "max_concurrent")
    # Per-op safety-net cap on cross-round op-long FAIL accumulation.
    # 0 = disabled (subagent never gives up regardless of how many
    # op-long FAILs accumulate; not recommended — the GPU is metered).
    _require_non_negative_int(
        path,
        cfg,
        "automation.stage2.max_op_long_failures",
        "max_op_long_failures",
    )


def _validate_agent_config(path: Path, agent: Any) -> None:
    """Validate the top-level ``[agent]`` table (agent-loop knobs)."""
    if agent is None or agent == {}:
        return
    if not isinstance(agent, dict):
        raise ValueError(f"{path}: [agent] must be a table")
    for key in (
        "poll_seconds",
        "agent_round_tries",
        "retry_base_sleep",
    ):
        if key in agent:
            _require_positive_int(path, agent, f"agent.{key}", key)
    if "runs_per_stage" in agent:
        _require_non_negative_int(path, agent, "agent.runs_per_stage", "runs_per_stage")
    if "max_consecutive_review_fails" in agent:
        _require_non_negative_int(
            path,
            agent,
            "agent.max_consecutive_review_fails",
            "max_consecutive_review_fails",
        )
    if "retry_continue_prompt" in agent:
        _require_non_empty_string(
            path,
            agent,
            "agent.retry_continue_prompt",
            "retry_continue_prompt",
        )
    if "state_dir" in agent:
        _require_non_empty_string(path, agent, "agent.state_dir", "state_dir")


def _require_non_empty_string(path: Path, cfg: dict[str, Any], label: str, key: str) -> str:
    value = cfg.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}: {label} must be a non-empty string")
    return value


def _require_positive_int(path: Path, cfg: dict[str, Any], label: str, key: str) -> int:
    value = cfg.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path}: {label} must be a positive integer")
    return value


def _require_non_negative_int(path: Path, cfg: dict[str, Any], label: str, key: str) -> int:
    """Like _require_positive_int but allows 0 (used for "unlimited" caps)."""
    value = cfg.get(key)
    # bool is a subclass of int — reject explicitly so True/False aren't
    # silently accepted as 1/0.
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path}: {label} must be a non-negative integer")
    return value


# Safe-relative-path validation is owned by :mod:`harness.run_schema`
# (a base-layer leaf module). We re-export it here so existing callers
# can keep importing it as ``config_runtime.validate_safe_relative_path``
# without breaking the API surface.
from harness.run_schema import validate_safe_relative_path  # noqa: E402


def _validate_suite_ref_env(path: Path, suite: str, cfg: dict[str, Any]) -> None:
    """Validate optional ``[evals.<name>.ref_env]`` knob.

    Lets a deployment hand the L0 ref script extra environment variables
    (``NUM_STEPS_OVERRIDE``, ``GATE_WINDOW_*``, ``WORLD_SIZE``, batch-size
    overrides, …) without editing either the ref script or harness Python.
    Schema: a flat table of ``UPPERCASE_KEY = "string|int|float"``. Bools
    are rejected so we never have to argue about ``true`` vs ``"1"`` in a
    shell env. Empty/missing table is the production default and means
    "use the L0 preset values byte-for-byte".
    """
    if "ref_env" not in cfg:
        return
    ref_env = cfg["ref_env"]
    if not isinstance(ref_env, dict):
        raise ValueError(f"{path}: evals.{suite}.ref_env must be a table")
    for key, val in ref_env.items():
        if not isinstance(key, str) or not key or key.upper() != key:
            raise ValueError(
                f"{path}: evals.{suite}.ref_env keys must be uppercase strings (got {key!r})"
            )
        if isinstance(val, bool) or not isinstance(val, (str, int, float)):
            raise ValueError(
                f"{path}: evals.{suite}.ref_env[{key!r}] must be str/int/float "
                f"(env values are strings); got {type(val).__name__}"
            )


def _validate_suite_ours_env(path: Path, suite: str, cfg: dict[str, Any]) -> None:
    """Validate optional ``[evals.<name>.ours_env]`` knob.

    Lets long-horizon suites run ours at a different per-rank micro-batch size than
    ref while keeping the same global batch size (``grad_accum`` is derived
    at dispatch time). Schema matches ``ref_env``: flat uppercase keys.
    """
    if "ours_env" not in cfg:
        return
    ours_env = cfg["ours_env"]
    if not isinstance(ours_env, dict):
        raise ValueError(f"{path}: evals.{suite}.ours_env must be a table")
    for key, val in ours_env.items():
        if not isinstance(key, str) or not key or key.upper() != key:
            raise ValueError(
                f"{path}: evals.{suite}.ours_env keys must be uppercase strings (got {key!r})"
            )
        if isinstance(val, bool) or not isinstance(val, (str, int, float)):
            raise ValueError(
                f"{path}: evals.{suite}.ours_env[{key!r}] must be str/int/float "
                f"(env values are strings); got {type(val).__name__}"
            )


def _validate_suite_ref_extra_args(path: Path, suite: str, cfg: dict[str, Any]) -> None:
    """Validate optional ``[evals.<name>].ref_extra_args`` knob.

    A flat list of bare-string CLI tokens the harness appends after the
    L0 ref script's ``bash <script>`` invocation. The L0 script's
    trailing ``"$@"`` forwards them to its torchrun entry — used by long-horizon
    to inject ``--no-deterministic`` without editing the frozen ref
    script. Empty/missing list means "no extra args" (the alignment–resume default).
    """
    if "ref_extra_args" not in cfg:
        return
    raw = cfg["ref_extra_args"]
    if not isinstance(raw, list):
        raise ValueError(f"{path}: evals.{suite}.ref_extra_args must be a list of strings")
    for token in raw:
        if not isinstance(token, str) or not token:
            raise ValueError(
                f"{path}: evals.{suite}.ref_extra_args entries must be non-empty "
                f"strings (got {token!r})"
            )


def _validate_suite_args_metadata(path: Path, suite_label: str, cfg: dict[str, Any]) -> None:
    """Validate args metadata.  *suite_label* is the dotted config path
    (e.g. ``"evals.long-train"`` or ``"local_suites.guard"``) so the
    error message points at the exact key the user must edit.
    """
    if "args_usage" in cfg and not isinstance(cfg["args_usage"], str):
        raise ValueError(f"{path}: {suite_label}.args_usage must be a string")
    for key in ("args_min", "args_max"):
        if key in cfg and not isinstance(cfg[key], int):
            raise ValueError(f"{path}: {suite_label}.{key} must be an integer")
        if key in cfg and cfg[key] < 0:
            raise ValueError(f"{path}: {suite_label}.{key} must be non-negative")
    if "args_unbounded" in cfg and not isinstance(cfg["args_unbounded"], bool):
        raise ValueError(f"{path}: {suite_label}.args_unbounded must be a boolean")
    if cfg.get("args_unbounded") and "args_max" in cfg:
        raise ValueError(f"{path}: {suite_label}.args_max conflicts with args_unbounded")
    if "args_min" in cfg and "args_max" in cfg and cfg["args_min"] > cfg["args_max"]:
        raise ValueError(f"{path}: {suite_label}.args_min must be <= args_max")


def _validate_global_env_sources(
    path: Path,
    data: dict[str, Any],
    env: dict[str, Any],
) -> None:
    """Keep process-boundary env vars owned by one config section.

    ``[env]`` is reserved for global runtime flags. Suite/default scalar
    inputs use their lower-case config keys and are injected only when the
    suite declares the corresponding ``env_inputs`` entry.
    """
    env_keys = {str(key).upper() for key in env}
    for section_name in ("ref", "stage2"):
        section = data.get(section_name, {})
        if not isinstance(section, dict):
            continue
        duplicates = sorted(key for key in env_keys if key.lower() in section)
        if duplicates:
            raise ValueError(
                f"{path}: [env] duplicates process env source(s) from "
                f"[{section_name}]: {duplicates}"
            )


def runtime_defaults(harness_config: dict[str, Any] | None = None) -> RuntimeDefaults:
    config = harness_config or load_harness_config()
    defaults = config.get("defaults", {})
    return RuntimeDefaults(
        report=str(defaults.get("report", "text")),
        gpu=defaults.get("gpu"),
    )


def default_timeout_s(harness_config: dict[str, Any] | None = None) -> int:
    """Return the global default suite timeout from ``defaults.toml``."""
    config = harness_config or load_harness_config()
    return int(config.get("defaults", {}).get("default_timeout_s", 1800))


def suite_timeout_s(
    workload_config: dict[str, Any],
    suite: str,
    harness_config: dict[str, Any] | None = None,
) -> int:
    """SSOT for the wall-clock budget of *suite* in seconds.

    Precedence: ``[evals.<suite>].timeout_s`` → ``[defaults].default_timeout_s``.
    Local suites fall back to the global default since they have no remote
    workload contract. Unknown suites raise ``ValueError``.

    Reads ``timeout_s`` directly off the suite table (NOT through
    ``resolve_suite_config``) so a stray ``timeout_s`` at ``[ref]`` /
    ``[stage1]`` / ``[stage2]`` cannot leak into the budget by deep-merge
    — the budget is a per-suite knob, not inheritable.

    All callers (``harness budget`` CLI, ``app._run_gpu_suite``, dispatcher
    ``_run_*`` functions, ``tools/remote_run.sh`` via the CLI) MUST go
    through this helper — no other site is allowed to read ``timeout_s``
    with its own fallback. This is the single seam that owns the
    config → runtime mapping.
    """
    evals = workload_config.get("evals", {})
    local_suites = workload_config.get("local_suites", {})
    fallback = default_timeout_s(harness_config)
    if suite in evals:
        cfg = evals[suite]
        return int(cfg.get("timeout_s", fallback))
    if suite in local_suites:
        return fallback
    raise ValueError(f"Unknown suite: {suite}")


def suite_ref_timeout_s(
    workload_config: dict[str, Any],
    suite: str,
    harness_config: dict[str, Any] | None = None,
) -> int:
    """SSOT for the ref-subprocess wall-clock budget of *suite* in seconds.

    Precedence: ``[evals.<suite>].ref_timeout_s`` →
    ``[evals.<suite>].timeout_s`` → ``[defaults].default_timeout_s``.

    This is the ref-side counterpart of :func:`suite_timeout_s`. The ref
    capture for some gates (bitwise-perf ``perf-bitwise`` runs the Megatron ref under
    per-FQN hash records, which is heavier than the ours-side replay) needs
    a longer or otherwise independent budget than the suite-wide
    ``timeout_s`` that bounds the candidate side; declaring
    ``ref_timeout_s`` on the suite lets a per-variant TOML own that number
    instead of forcing both sides through one knob. When unset it falls back
    to ``timeout_s`` so every gate that omits it keeps its current
    behaviour. Reads directly off the suite table (NOT through
    ``resolve_suite_config``) so a stray ``ref_timeout_s`` at ``[ref]`` /
    ``[stage1]`` / ``[stage2]`` cannot leak in by deep-merge.

    Both ref-capture sites in ``evals/_common.py`` MUST go through this
    helper — no other site is allowed to read ``ref_timeout_s`` with its own
    fallback.
    """
    evals = workload_config.get("evals", {})
    local_suites = workload_config.get("local_suites", {})
    fallback = default_timeout_s(harness_config)
    if suite in evals:
        cfg = evals[suite]
        return int(cfg.get("ref_timeout_s", cfg.get("timeout_s", fallback)))
    if suite in local_suites:
        return fallback
    raise ValueError(f"Unknown suite: {suite}")


def default_output_tail_limit(harness_config: dict[str, Any] | None = None) -> int:
    """Return the global default output tail limit from ``defaults.toml``."""
    config = harness_config or load_harness_config()
    return int(config.get("defaults", {}).get("output_tail_limit", 50000))


def runtime_env_defaults(workload_config: dict[str, Any] | None = None) -> RuntimeEnvDefaults:
    config = workload_config or load_workload_config(None)[1]
    runtime = config.get("runtime", {})
    distributed = runtime.get("distributed", {}) if isinstance(runtime, dict) else {}
    master_addr = distributed.get("master_addr")
    master_port = distributed.get("master_port")
    if not isinstance(master_addr, str) or not master_addr:
        raise ValueError("config/eval.toml [runtime.distributed].master_addr is required")
    if not isinstance(master_port, str) or not master_port:
        raise ValueError("config/eval.toml [runtime.distributed].master_port is required")
    return RuntimeEnvDefaults(master_addr=master_addr, master_port=master_port)


def ref_script(workload_config: dict[str, Any] | None = None) -> str:
    """Return ``[ref].ref_script`` basename (SSOT for the L0 ref script).

    The value must be a plain basename (no path separators, no ``..``)
    so callers can safely join it to ``<repo>/ref/reference/``.
    """
    config = workload_config or load_workload_config(None)[1]
    ref = config.get("ref", {})
    raw = ref.get("ref_script") if isinstance(ref, dict) else None
    if not isinstance(raw, str) or not raw:
        raise ValueError(
            "[ref].ref_script is required (the L0 ref-script basename "
            "under ref/reference/); set it in config/ref.toml "
            "(cp from config/ref/*.toml) or export FORGE_REF_SCRIPT."
        )
    if "/" in raw or "\\" in raw or raw in {".", ".."} or ".." in PurePosixPath(raw).parts:
        raise ValueError(
            f"[ref].ref_script must be a plain basename under ref/reference/ (got {raw!r})"
        )
    return raw


def ref_backend(workload_config: dict[str, Any] | None = None) -> str:
    """Return ``[ref].backend`` (SSOT for which L0 ref stack to use).

    Default: ``"megatron"`` (profiles that omit it get megatron).
    """
    config = workload_config or load_workload_config(None)[1]
    ref = config.get("ref", {})
    raw = ref.get("backend") if isinstance(ref, dict) else None
    if raw is None or raw == "":
        return "megatron"
    if not isinstance(raw, str):
        raise ValueError(
            f"config/ref.toml [ref].backend must be a string (got {type(raw).__name__})"
        )
    if raw not in BACKENDS:
        raise ValueError(
            f"config/ref.toml [ref].backend must be one of {sorted(BACKENDS)!r} (got {raw!r})"
        )
    return raw


def ref_capture_script(workload_config: dict[str, Any] | None = None) -> str:
    """Return ``[ref].ref_capture_script`` (SSOT for the alignment bridge path).

    Empty value is valid — means no bridge generated yet.
    """
    config = workload_config or load_workload_config(None)[1]
    ref = config.get("ref", {})
    raw = ref.get("ref_capture_script") if isinstance(ref, dict) else None
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raise ValueError(
            f"config/eval.toml [ref].ref_capture_script must be a string (got {type(raw).__name__})"
        )
    return raw


def _basename_only(raw: object, key: str) -> str:
    """Validate a plain-basename TOML field (no path separators, no ``..``)."""
    if not isinstance(raw, str) or not raw:
        raise ValueError(
            f"config/eval.toml [ref].{key} is required (a plain basename, e.g. 'ref_capture.pt')."
        )
    if "/" in raw or "\\" in raw or raw in {".", ".."} or ".." in PurePosixPath(raw).parts:
        raise ValueError(f"config/eval.toml [ref].{key} must be a plain basename (got {raw!r})")
    return raw


def ref_capture_basename(workload_config: dict[str, Any] | None = None) -> str:
    """Return ``[ref].ref_capture_basename`` (SSOT for the ref tensor dump filename)."""
    config = workload_config or load_workload_config(None)[1]
    ref = config.get("ref", {})
    raw = ref.get("ref_capture_basename") if isinstance(ref, dict) else None
    return _basename_only(raw, "ref_capture_basename")


def candidate_capture_basename(workload_config: dict[str, Any] | None = None) -> str:
    """Return ``[ref].candidate_capture_basename`` (SSOT for the candidate tensor dump filename)."""
    config = workload_config or load_workload_config(None)[1]
    ref = config.get("ref", {})
    raw = ref.get("candidate_capture_basename") if isinstance(ref, dict) else None
    return _basename_only(raw, "candidate_capture_basename")


def _artifact_root(harness_config: dict[str, Any]) -> Path:
    root_dir = repo_root()
    root = Path(harness_config["artifacts"]["root"])
    if not root.is_absolute():
        root = root_dir / root
    return root.resolve()


def artifact_subtree(harness_config: dict[str, Any], subtree: str) -> Path:
    subtrees = harness_config["artifacts"].get("subtrees", {})
    key = subtrees.get(subtree)
    if key is None:
        raise ValueError(
            f"Unknown artifact subtree '{subtree}'. "
            f"Register it in [artifacts.subtrees] in defaults.toml."
        )
    return _artifact_root(harness_config) / key


@functools.cache
def _resolve_repo_root(cwd_text: str, env_root_text: str | None) -> Path:
    if env_root_text:
        root = Path(env_root_text).resolve()
        if not _looks_like_repo_root(root):
            raise RuntimeError(f"FORGE_REPO_ROOT must point to a harness repo root: {root}")
        return root

    cwd = Path(cwd_text)
    discovered = _discover_repo_root(cwd)
    if discovered is not None:
        return discovered
    return _PACKAGE_HARNESS_DIR.parent


def _discover_repo_root(cwd: Path) -> Path | None:
    git_root = _git_toplevel(cwd)
    if git_root is not None and _looks_like_repo_root(git_root):
        return git_root
    for candidate in (cwd, *cwd.parents):
        if _looks_like_repo_root(candidate):
            return candidate.resolve()
    return None


def _git_toplevel(cwd: Path) -> Path | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        return None
    stdout = completed.stdout.strip()
    return Path(stdout).resolve() if stdout else None


def _looks_like_repo_root(candidate: Path) -> bool:
    root = candidate.resolve()
    return (
        (root / "pyproject.toml").exists()
        and (root / "harness" / "config" / "defaults.toml").exists()
        and (root / "config" / "eval").is_dir()
    )
