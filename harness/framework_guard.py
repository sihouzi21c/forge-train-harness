from __future__ import annotations

import ast
import io
import os
import re
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from harness.config_runtime import repo_root as _repo_root

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


def repo_root() -> Path:
    """Resolve the harness repo root lazily on every access.

    Indirects through :func:`harness.config_runtime.repo_root`, which
    is ``@functools.cache``-d on ``(cwd, FORGE_REPO_ROOT)`` so this
    stays cheap. Computing the value lazily (rather than at
    module-import time) means tests that mutate cwd or
    ``FORGE_REPO_ROOT`` between framework_guard imports observe the
    updated root on the next call.
    """
    return _repo_root()


# Directory names that are pruned from full-repo guard scans. These are
# build / cache / VCS artefact roots that legitimately contain copies of
# allowlisted source files (under ``__pycache__``) or generated output
# (``.artifacts/agent-logs/<ts>/transcript.md``); rescanning them on
# every ``harness run guard`` invocation wastes I/O without ever
# changing the verdict — anything that lands here is either a Python
# artefact whose source was already scanned, or external state the
# guard has no jurisdiction over. The prune is name-based (matches at
# any depth) rather than path-based to keep the rule cheap to reason
# about.
IGNORED_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".artifacts",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".tox",
        "node_modules",
        "third_party",
    }
)

TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".mdc",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_BASENAMES = {
    ".dockerignore",
    ".gitignore",
    "Dockerfile",
    "Makefile",
}

FULLY_BANNED = ("megatron", "deepspeed")
TORCH_BANNED_SUBMODULES = (
    "torch.nn",
    "torch.optim",
    "torch.autograd",
    "torch.utils.data",
    "torch.amp",
    "torch.cuda.amp",
)
# Autograd-engine entry points.  ``torch.autograd`` already covers
# direct submodule imports, but the autograd engine is reachable
# *without* naming the submodule via two Tensor-method paths:
#
#   1. ``tensor.backward()``      — backward pass
#   2. ``requires_grad=True`` /   — opts a raw tensor into the
#      ``requires_grad_(True)``    autograd graph
#
# Stage 1 explicitly forbids relying on autograd for backward
# (forward + backward must be a statically scheduled graph driven by
# ATen primitives such as ``torch.ops.aten.<op>_backward``).  Banning
# these literal tokens at the keyword scanner level makes the
# enforcement match the spec — otherwise the rule is only visible in
# prose and the agent can quietly call ``.backward()`` to short-circuit
# the entire static-graph requirement.
AUTOGRAD_BANNED_KEYWORDS = (
    ".backward(",
    "requires_grad=True",
    "requires_grad_(True)",
)
# NOTE: ``.backward(`` is banned UNCONDITIONALLY. There is deliberately no
# "this file defines its own ``def backward``" carve-out: such an exemption
# is trivially exploitable — an agent could short-circuit the entire
# static-graph requirement with a real ``loss.backward()`` autograd call and
# silence the scanner by parking any ``def backward(self, …)`` method in the
# same file. The engine's hand-coded backward MUST be named so it does not
# collide with the ``.backward(`` literal (e.g. ``def bwd(self, …)``); the
# scanner intentionally has no way to tell a Tensor autograd entry from a
# user-method call by substring alone, so we err on the side of the ban.
DOCUMENTATION_ALLOWLIST_FILES = {
    Path("README.md"),
    # Open-source release docs — legitimately mention the reference
    # framework (Megatron-LM) and alternatives (DeepSpeed) in the
    # "reference stack" sections; the banned-keyword scan is meant for
    # engine code, not user-facing prose.
    Path("README_zh.md"),
}
ALLOWLIST = {
    Path("harness/framework_guard.py"),
    # Sister guard to ``framework_guard``: scans
    # ``workload/src/`` + ``workload/ops/`` for proxy / synthesis
    # patterns (subprocess shell-out to ``ref/`` scripts,
    # ``ref_script_runner`` / ``evals.harness_hook`` imports,
    # ``_synthetic_`` / ``_ref_proxy_`` named functions,
    # hardcoded ``mfu_e2e_*`` metric kwargs).  Its docstring
    # legitimately names the banned framework families it does NOT
    # cover (``megatron`` / ``deepspeed`` / ``torch.nn``) to motivate
    # the split — same rationale as ``framework_guard.py`` self-allowlist.
    Path("harness/anti_proxy_guard.py"),
    Path("agent-loop.sh"),
    # Torch-image verifier. Legitimately names Megatron/HF as external
    # dependency context (which entries need the Megatron .idx format) —
    # same "dependency context, not engine code" rationale as
    # pyproject.toml / README.
    Path("env/verify_env_torch.sh"),
    # ``eval_train_steps.py`` is the ours-side runner for the bitwise-singlecard–bitwise-perf
    # multi-step bitwise gates (``multistep-1gpu`` / ``multistep`` /
    # ``perf-bitwise``).  It legitimately imports
    # ``training_engine_tensor.train_loop``, which transitively touches
    # Megatron-LM (Megatron is the engine SSOT — the ban only applies
    # to harness adapters that reimplement Megatron logic).
    Path("evals/scripts/eval_train_steps.py"),
    # alignment single-step tensor-capture sibling of ``eval_train_steps.py``
    # (forward-align / backward-align); same import surface
    # (``training_engine_tensor.train_loop``), separate file so
    # ``evals/scripts/`` reads as a per-milestone table of contents.
    Path("evals/scripts/eval_capture_align.py"),
    # resume / long-horizon suite scripts
    Path("evals/scripts/eval_long_train.py"),
    Path("evals/scripts/eval_resume_train.py"),
    # bitwise-perf / long-horizon profile-snapshot eval — mirrors eval_long_train.py shape
    # (same MEGATRON_ROOT env contract), wrapped by the dispatcher's
    # nsys-on-rank-0 path. Allowlisted for the same "harness entry
    # spells framework symbols" reason as its siblings above.
    Path("evals/scripts/eval_profile_snapshot.py"),
    # production production-train segmented runner: same harness-entrypoint shape as
    # resume / long-horizon train scripts, with backend literal strings used only to select
    # the external reference root env contract.
    Path("evals/scripts/eval_production_train.py"),
    # Active L0 ref scripts. fineweb_modelbestsdk and run_16gpu_pure_mup_mtp
    # are both live ref stacks (selected via ``[ref].ref_script`` in
    # ``config/ref/{megatron,torch}.toml``);
    # ``config/data/{modelbest,gsm8k}.toml`` are data variants.
    # Ref scripts legitimately name framework symbols
    # (``modelbest_sdk``, ``torch.optim``, etc.) the same way Megatron
    # itself legitimately uses ``megatron`` / ``deepspeed`` keywords —
    # allowlisted here so the guard's keyword sweep does not trip on
    # customer-entry code.
    Path("ref/reference/train_minicpm4_0.5b_fineweb_modelbestsdk.sh"),
    Path("ref/reference/train_minicpm4_0.5b_gsm8k.sh"),
    Path("ref/reference/prepare_gsm8k_data.sh"),
    # gsm8k → pure-torch DATA_CONF bridge: emits the modelbest_sdk
    # weighted-shard string the pure-torch ref expects from a single
    # Megatron .bin/.idx prefix. Legitimately spells "megatron" in
    # comments + the auto-prep fallback (calls ``prepare_gsm8k_data.sh``
    # which itself runs Megatron's ``tools/preprocess_data.py``) — same
    # rationale as the two siblings above.
    Path("ref/reference/gsm8k_data_conf.sh"),
    # gsm8k → pure-torch Megatron-free preparator. Emits the same
    # indexed-binary .bin/.idx as ``prepare_gsm8k_data.sh`` (bitwise-
    # verified) using SentencePiece directly, no MEGATRON_ROOT. The
    # torch ``gsm8k_data_conf.sh`` auto-prep calls this instead of the
    # Megatron tool. Allowlisted alongside its siblings so future
    # provenance comments naming the format do not trip the sweep.
    Path("ref/reference/gsm8k_prepare_torch.py"),
    # Pure-PyTorch reference stack (alternative L0 baseline switched in
    # via ``[defaults].ref_script``).  These three files are
    # customer-entry code, not engine code: they legitimately use
    # ``torch.nn`` / ``torch.optim`` (matrix Linear / AdamW) the same
    # way ``train_minicpm4_0.5b_gsm8k.sh`` legitimately runs Megatron,
    # and ``train_pure_mup_mtp.py`` imports ``modelbest_sdk`` as the
    # dataloader same as the gsm8k variant above.  Any harness alignment
    # capture wiring stays in an agent-generated bridge (see
    # ``evals/harness_hook/recipes/README.md``) — these three files
    # contain no hook logic.
    Path("ref/reference/model_pure_mup_mtp.py"),
    Path("ref/reference/train_pure_mup_mtp.py"),
    Path("ref/reference/run_16gpu_1000step_pure_mup_mtp.sh"),
    # Qwen3 0.6B pure-PyTorch dense L0 reference (alternative model axis,
    # sibling of model_pure_mup_mtp.py). Same "customer-entry reference,
    # not engine code" rationale: legitimately uses ``torch.nn`` /
    # ``torch.optim`` / ``.backward()`` for the dense GQA + QK-Norm
    # baseline. Added with the Qwen3 reference (commit d5412181) but
    # missed from this list until now.
    Path("ref/reference/model_qwen3.py"),
    Path("ref/reference/train_qwen3_dense.py"),
    Path("ref/reference/run_qwen3_dense.sh"),
    # MiniCPM4 8B pure-PyTorch TP/DP + HF-singlecard L0 references
    # (alternative model axis, siblings of model_pure_mup_mtp.py /
    # model_qwen3.py). Same "customer-entry reference, not engine code"
    # rationale: they legitimately drive ``torch.nn`` / ``torch.optim`` /
    # ``torch.autograd`` / ``.backward()`` for the baseline, and the
    # vendored HF modeling file ships upstream ``torch.nn`` idioms we do
    # not rewrite. Added with the 8B reference (merge a51c52dd) but
    # missed from this list until now — same gap class as the Qwen3
    # entries above.
    Path("ref/reference/model_minicpm4_8b_tp.py"),
    Path("ref/reference/train_minicpm4_8b_tp.py"),
    Path("ref/reference/train_minicpm4_8b_hf_singlecard.py"),
    Path("ref/reference/minicpm4_8b_hf_singlecard/optim.py"),
    Path("ref/reference/minicpm4_8b_hf_singlecard/train.py"),
    Path("ref/reference/minicpm4_8b_hf_singlecard/hf_model/modeling_minicpm.py"),
    # Deploy-source resolver: spells the vendored Megatron submodule
    # path and the ``"megatron"`` backend literal as string data (the
    # same selector-not-import class as ``tools/agent_loop_config.py``).
    Path("tools/resolve_deploy.py"),
    # Generic streaming HuggingFace dataloader (torch-ref data path) +
    # its two DATA_CONF bridges. Reference-side data plumbing, not engine
    # code: they import ``datasets`` / ``sentencepiece`` the same way
    # ``train_pure_mup_mtp.py`` imports ``modelbest_sdk``. Allowlisted so
    # the keyword sweep does not trip on provenance comments.
    Path("ref/reference/hf_stream_dataloader.py"),
    Path("ref/reference/ultra_fineweb_data_conf.sh"),
    Path("ref/reference/gsm8k_hf_data_conf.sh"),
    # Ref-as-gate SSOT helper — pure bash subprocess primitive. The
    # alignment capture bridge is agent-generated (see
    # ``evals/harness_hook/recipes/README.md``) and therefore not in
    # this list; the runner has no framework-specific knowledge.
    Path("tools/ref_script_runner.py"),
    # Agent-generated alignment capture bridge for the pure-PyTorch + muP + MTP
    # ref stack: monkey-patches MiniCPM4MupMtp + the ref's AdamW
    # construction to install ``evals.harness_hook.install`` (or the
    # canonical-state dumper) before the first ``optimizer.step``. Same
    # exception class as the ref entry itself (legitimately names the
    # high-level wrapper modules it interposes on, ``torch.optim`` /
    # ``torch.nn``).
    # Profile-batch capture goes through the same Megatron bootstrap as
    # the gate scripts (initialize_megatron + ``GPTDataset``) so it
    # legitimately imports Megatron; treated as a ref-as-gate helper.
    Path("tools/capture_profile_batch.py"),
    # Per-backend stage rule router: legitimately spells the literal
    # backend names ``"megatron"`` / ``"torch"`` as string data
    # (selects the right ``prompt/develop_prompt/<backend>/...`` overlay
    # from ``config/ref.toml [ref].backend``). This is the
    # selector itself — not an engine import — so it has to be
    # allowlisted out of the megatron keyword scan.
    Path("tools/agent_loop_config.py"),
    Path("harness/tests/test_harness_hook.py"),
    # Project metadata.  Documents the optional GPU-host extras bundle
    # (transformer_engine) and explains in a comment why no
    # internal-only dataloader is bundled — naturally mentions
    # ``Megatron`` and ``GPTDataset`` as external dependency context.
    Path("pyproject.toml"),
    # Wire-format SSOT: legitimately documents and parses the Megatron
    # stdout fallback (``iteration N/M … lm loss: X``) so harness gates
    # can fall back when the bridge does not emit ``[LOSS]`` lines.
    # Pure-stdlib leaf, no engine code.
    Path("harness/wire_format.py"),
    # Layer DAG linter — references control-plane module names as
    # string data, not as engine imports.
    Path("harness/tests/test_layer_dag.py"),
    # Dispatcher-behavior tests pin the alignment capture-diff contract; the
    # test bodies construct placeholder ``ref_capture.pt`` /
    # ``candidate_capture.pt`` dump files to exercise the dispatcher's
    # I/O paths. The filenames live in
    # ``[defaults].{ref,candidate}_capture_basename`` (single source of
    # truth for the dispatcher↔bridge wire format), not in the engine.
    Path("harness/tests/test_dispatcher_behavior.py"),
    # Asserts the configured ref-script value; the SSOT it pins
    # legitimately spells customer-specific filenames as string
    # literals.
    Path("harness/tests/test_ref_script_runner.py"),
    # Path-shim contract test — exercises the env keys the harness
    # pushes into the L0 ref script (``MEGATRON_ROOT`` /
    # ``DATA_PATH`` / ``TOKENIZER_MODEL`` / ``SAVE_PATH`` /
    # ``TENSORBOARD_DIR``) plus a repo-wide static scan that bans
    # individual-user absolute paths. The literal strings live in
    # the test on purpose.
    Path("harness/tests/test_ref_path_shim_and_no_user_paths.py"),
    # Source resolution tests reference megatron_root / tokenizer_model
    # as config key names (string data), not engine imports.
    Path("harness/tests/test_harness_core.py"),
    Path("harness/tests/test_megatron_binary_reader.py"),
    # Streaming HF dataloader dispatch tests. Legitimately spell the
    # ``megatron_binary`` loader-kind as string data (the explicit
    # DATA_LOADER value ``build_dataloader`` routes on) — config-key
    # strings, not an engine import — same rationale as the
    # megatron-binary-reader test above.
    Path("harness/tests/test_hf_stream_dataloader.py"),
    # Per-backend prompt-guideline tests legitimately iterate over the
    # backend name tuple ``("megatron", "torch")`` to assert each
    # backend's playbook + stage2.md carry the required invariants.
    Path("harness/tests/test_prompt_guidelines.py"),
    # Production-train env-contract tests pin string-valued suite config, not
    # engine imports.
    Path("harness/tests/test_production_train.py"),
    # Qwen3 reference parity test — exercises the ref model/train scripts
    # above, so it legitimately calls ``loss.backward()`` on the reference
    # graph. Same exception class as the ref files it tests.
    Path("harness/tests/test_qwen3_ref.py"),
    # 8B reference parity test — same exception class as test_qwen3_ref
    # (drives the allowlisted 8B ref graph with autograd + torch.nn).
    Path("harness/tests/test_minicpm4_8b_ref.py"),
    # DP/TP launcher tests build tiny torch.optim/autograd fixtures to
    # exercise the allowlisted 8B dptp launch path — ref-as-gate test
    # class, not engine code.
    Path("harness/tests/test_harness_dptp.py"),
    # Capture-gate trajectory tests construct a minimal autograd graph
    # (requires_grad / .backward) to exercise the capture hook plumbing.
    Path("harness/tests/test_capture_gate_trajectories.py"),
    # These three pin config-key string data (``backend="megatron"``,
    # ``MEGATRON_ROOT`` env contract, vendored submodule path) — the
    # same string-data-not-import class as test_harness_core.py.
    Path("harness/tests/test_canonical_preflight.py"),
    Path("harness/tests/test_gate_entry_config.py"),
    Path("harness/tests/test_resolve_deploy.py"),
} | DOCUMENTATION_ALLOWLIST_FILES
DOCUMENTATION_ALLOWLIST_DIRS = {
    Path(".cursor/rules"),
    Path("prompt"),
    Path(".artifacts"),
    Path("config"),
    # Design / plan documents — prose that legitimately names the
    # reference framework (Megatron resolver plans, backend tables).
    # Only documentation-style files are protected; ``.py`` / ``.sh``
    # under here would still be scanned per
    # CODE_SUFFIXES_NEVER_ALLOWLISTED_BY_DIR.
    Path("docs"),
    Path("ref"),
    Path("harness/tests"),
    Path("workload/notes"),
    Path("workload/ops"),
    Path("tools"),
}
ALLOWLIST |= {
    Path("config/eval/dense_training/dense_training.toml"),
    Path("harness/cli.py"),
    Path("harness/config_runtime.py"),
    Path("evals/dispatcher.py"),
    Path("evals/_common.py"),
    Path("evals/scripts/op_long_ours.py"),
    # alignment capture hook — the four files implement the standard hook
    # (``install(model, optimizer, output_file=...)`` over
    # ``torch.nn``) and legitimately enumerate the frameworks the hook
    # is designed to attach to (``Megatron`` / ``DeepSpeed`` /
    # ``Nanotron`` / ``torch.nn`` / ``torch.optim``) as documentation
    # surface so an alignment agent reading these files knows the contract.
    # No baseline plugin layer — bridges are agent-generated; see
    # ``evals/harness_hook/recipes/README.md``.
    Path("evals/harness_hook/__init__.py"),
    Path("evals/harness_hook/_module_hook.py"),
    Path("evals/harness_hook/_grad_collector.py"),
    Path("evals/harness_hook/_canonical_state.py"),
    Path("evals/harness_hook/_dump.py"),
    Path("evals/harness_hook/_megatron_argparse_shim.py"),
    Path("evals/harness_hook/recipes/README.md"),
    # Unified alignment capture bridge — dispatches to the interposer for both
    # Megatron and Torch backends. Legitimately names framework symbols
    # (``megatron``, ``torch.optim.AdamW``, ``setup_model_and_optimizer``)
    # because the interposer monkey-patches these entry points.
    Path("ref/bridges/bridge.sh"),
    Path("ref/bridges/interposer.py"),
    Path("harness/tests/test_m1_bridge.py"),
    Path("harness/tests/test_framework_guard.py"),
    # Bit-wise contract test for the reference stack's fp32 wgrad
    # accumulation. Exercises the ref Functions
    # (``ref/reference/model_pure_mup_mtp.py``) in isolation and so
    # legitimately drives the autograd engine (``.backward()`` /
    # ``requires_grad_``) and constructs ``nn.Parameter`` — the same
    # ref-as-gate exception class as ``test_m1_bridge.py``: it tests the
    # allowlisted reference, it is not itself agent-authored engine code.
    Path("harness/tests/test_fp32_wgrad.py"),
    # Self-developed engine's public SSOT entry contract.  The module
    # legitimately names ``Megatron`` / ``GPTDataset`` / ``torch.utils.data``
    # in its interface docstring (the long-lived contract that the
    # agent loop must satisfy when filling in the body); the package
    # __init__ re-exports the same symbols and references the same SSOT
    # boundary.  See ``train_loop.py``'s docstring for the rationale.
    Path("workload/src/training_engine_tensor/__init__.py"),
    Path("workload/src/training_engine_tensor/train_loop.py"),
}
ALLOWED_SUMMARY = (
    "Allowed: torch (bare), torch.cuda, torch.backends, torch.distributed, "
    "triton, cuBLAS, cuDNN FlashAttention, TransformerEngine, NCCL"
)
BANNED_SUMMARY = (
    "Banned: megatron, deepspeed, torch.nn, torch.optim, "
    "torch.autograd, torch.utils.data, torch.amp, torch.cuda.amp; "
    "autograd entry points .backward( / requires_grad=True / "
    "requires_grad_(True)"
)

# ``megatron`` / ``deepspeed`` are banned as framework *module
# references*, not as substrings. The token must be a standalone
# identifier — no identifier char on either side — so ``import megatron``
# / ``from megatron`` / ``megatron.core`` match while the Megatron
# *data-format* identifiers (``megatron_binary`` loader kind,
# ``MegatronBinaryDataloader``, ``_is_megatron_binary``) do NOT. The
# trailing ``_``/letter in those names keeps them out of the match — the
# distinction between "the framework" and "the .bin/.idx data layout the
# harness legitimately reads".
FORBIDDEN_PATTERNS = tuple(
    re.compile(rf"(?<![A-Za-z0-9_]){re.escape(prefix)}(?![A-Za-z0-9_])", re.IGNORECASE)
    for prefix in FULLY_BANNED
)
TORCH_BANNED_PATTERNS = tuple(
    re.compile(rf"\b{re.escape(mod)}(?:[-_./a-zA-Z0-9]*)?\b") for mod in TORCH_BANNED_SUBMODULES
)
# Autograd entry-point patterns are plain literal substring searches:
# the tokens are not module names, so the word-boundary regex used for
# import-style bans does not apply. ``re.escape`` keeps the parens
# literal so ``.backward(`` matches both ``obj.backward()`` and
# ``obj.backward(retain_graph=True)``.
AUTOGRAD_BANNED_PATTERNS = tuple(re.compile(re.escape(token)) for token in AUTOGRAD_BANNED_KEYWORDS)


@dataclass(frozen=True)
class GuardViolation:
    path: Path
    line_number: int
    line: str
    reason: str

    def render(self) -> str:
        relative_path = relative_to_repo(self.path)
        display_path: str | Path = relative_path if relative_path is not None else self.path
        return f"{display_path}:{self.line_number}: {self.reason}\n    {self.line}"


def iter_candidate_files(raw_paths: Sequence[str]) -> list[Path]:
    if not raw_paths:
        return sorted(_walk_text_candidates(repo_root()))

    candidates: list[Path] = []
    seen_paths: set[Path] = set()
    for raw_path in raw_paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            continue
        if path.is_dir():
            iterator = sorted(_walk_text_candidates(path))
        elif path.is_file():
            iterator = [path.resolve()]
        else:
            continue
        for candidate in iterator:
            if candidate in seen_paths:
                continue
            seen_paths.add(candidate)
            candidates.append(candidate)
    return candidates


def _walk_text_candidates(root: Path) -> Iterator[Path]:
    """Yield text-candidate files under *root*, pruning IGNORED_DIR_NAMES.

    Replaces the previous ``Path.rglob('*')`` walk, which descended into
    ``__pycache__`` / ``.git`` / ``.artifacts`` / cache directories on
    every guard invocation. Pruning at the directory level (via
    ``os.walk``'s in-place ``dirnames`` mutation) is the canonical way
    to avoid that cost.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        # In-place prune so os.walk does not descend into ignored trees.
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIR_NAMES]
        for filename in filenames:
            candidate = Path(dirpath) / filename
            if candidate.is_file() and is_text_candidate(candidate):
                resolved = candidate.resolve()
                # A symlink (e.g. workspace ``config/<gate>.toml`` → parent
                # ``.artifacts/...``) resolves to a target that the dirname
                # prune never saw. Apply the same prune to the resolved path
                # so a link into an ignored tree is not scanned or reported.
                if any(part in IGNORED_DIR_NAMES for part in resolved.parts):
                    continue
                yield resolved


def is_text_candidate(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_BASENAMES


def relative_to_repo(path: Path) -> Path | None:
    try:
        return path.resolve().relative_to(repo_root())
    except ValueError:
        return None


CODE_SUFFIXES_NEVER_ALLOWLISTED_BY_DIR = {".py", ".sh"}


def is_allowlisted(path: Path) -> bool:
    relative_path = relative_to_repo(path)
    if relative_path is None:
        return True
    if relative_path in ALLOWLIST:
        return True
    # Directory-level allowlist only protects documentation-style files —
    # executable code (Python / shell) must always be scanned, otherwise
    # allowlisting a directory that also contains source code silently
    # weakens the boundary.
    if path.suffix.lower() in CODE_SUFFIXES_NEVER_ALLOWLISTED_BY_DIR:
        return False
    return any(
        allowlisted_dir == relative_path or allowlisted_dir in relative_path.parents
        for allowlisted_dir in DOCUMENTATION_ALLOWLIST_DIRS
    )


def match_banned_reference(line: str) -> str | None:
    for pattern in FORBIDDEN_PATTERNS:
        match = pattern.search(line)
        if match:
            return f"banned keyword `{match.group()}`"
    for pattern in TORCH_BANNED_PATTERNS:
        match = pattern.search(line)
        if match:
            return f"banned torch submodule `{match.group()}`"
    for pattern in AUTOGRAD_BANNED_PATTERNS:
        match = pattern.search(line)
        if match:
            return (
                f"banned autograd entry point `{match.group()}` — Stage 1 "
                f"requires statically scheduled backward via "
                f"`torch.ops.aten.<op>_backward(...)`, not Tensor-method "
                f"autograd"
            )
    return None


def scan_file(path: Path) -> list[GuardViolation]:
    try:
        contents = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        relative_path = relative_to_repo(path)
        display_path: str | Path = relative_path if relative_path is not None else path
        raise ValueError(f"{display_path}: failed to decode text candidate as UTF-8") from exc

    raw_lines = contents.splitlines()
    scan_lines = _redact_python_for_scan(contents) if path.suffix.lower() == ".py" else raw_lines

    violations: list[GuardViolation] = []
    for line_number, (scan_line, raw_line) in enumerate(
        zip(scan_lines, raw_lines, strict=True), start=1
    ):
        reason = match_banned_reference(scan_line)
        if reason is None:
            continue
        violations.append(
            GuardViolation(
                path=path,
                line_number=line_number,
                line=raw_line.strip(),
                reason=reason,
            )
        )
    return violations


def _redact_python_for_scan(source: str) -> list[str]:
    """Per-line view of *source* with docstrings and ``#`` comments
    blanked to spaces while preserving line/column structure.

    Only module / class / function docstrings and ``#`` comment tokens
    are blanked. Ordinary string literals (``"torch.nn"``) are kept
    intact so dynamic-import patterns such as
    ``importlib.import_module("torch.nn")`` remain detectable.

    On any parse / tokenize failure (broken Python source), returns the
    raw lines unchanged — the guard MUST NOT silently weaken when it
    cannot understand the file.
    """
    raw_lines = source.splitlines()
    try:
        ranges = _collect_python_redaction_ranges(source)
    except (SyntaxError, tokenize.TokenError, ValueError, IndentationError):
        return raw_lines
    return _apply_redaction_ranges(raw_lines, ranges)


def _collect_python_redaction_ranges(
    source: str,
) -> list[tuple[int, int, int, int]]:
    tree = ast.parse(source)
    ranges: list[tuple[int, int, int, int]] = []

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            continue
        body = getattr(node, "body", None) or []
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
            and first.value.end_lineno is not None
            and first.value.end_col_offset is not None
        ):
            d = first.value
            assert d.end_lineno is not None and d.end_col_offset is not None
            ranges.append((d.lineno, d.col_offset, d.end_lineno, d.end_col_offset))

    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            (sl, sc), (el, ec) = tok.start, tok.end
            ranges.append((sl, sc, el, ec))

    return ranges


def _apply_redaction_ranges(lines: list[str], ranges: list[tuple[int, int, int, int]]) -> list[str]:
    buf = [list(ln) for ln in lines]
    for sl, sc, el, ec in ranges:
        for ln in range(sl, el + 1):
            if ln - 1 >= len(buf):
                break
            row = buf[ln - 1]
            lo = sc if ln == sl else 0
            hi = ec if ln == el else len(row)
            for i in range(lo, min(hi, len(row))):
                row[i] = " "
    return ["".join(row) for row in buf]


def scan_paths(raw_paths: Sequence[str]) -> list[GuardViolation]:
    violations: list[GuardViolation] = []
    for path in iter_candidate_files(raw_paths):
        if is_allowlisted(path):
            continue
        violations.extend(scan_file(path))
    return violations


def check_path_isolation() -> list[GuardViolation]:
    """L2 path guard.

    On stage2/op/* branches, only allow modifying the operator's own
    directory. On all other branches, enforce the repository write-surface
    contract: ``ref/`` baselines stay out of staged changes.

    Returns violations for any staged file outside the allowed set.
    """
    import subprocess as _sp

    root = repo_root()
    result = _sp.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(root),
    )
    branch = result.stdout.strip()

    diff_result = _sp.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(root),
    )
    changed_files = [f.strip() for f in diff_result.stdout.strip().split("\n") if f.strip()]

    if not branch.startswith("stage2/op/"):
        return _check_repo_write_surface(branch, changed_files)

    op_name = branch.split("stage2/op/", 1)[1]
    bad_paths = op_path_violations(op_name, changed_files)

    return [
        GuardViolation(
            path=Path(filepath),
            line_number=0,
            line="",
            reason=(
                f"path isolation: branch '{branch}' may only modify "
                f"workload/ops/{op_name}/, not '{filepath}'"
            ),
        )
        for filepath in bad_paths
    ]


def op_path_violations(op_name: str, changed_paths: list[str]) -> list[str]:
    """Return paths that violate per-operator path isolation (SSOT).

    On a stage2/op/* branch, only ``workload/ops/{op_name}/`` may be modified.
    Used by ``check_path_isolation`` (pre-commit staged files); subagents
    must invoke ``python -m harness.framework_guard --with-path-guard`` on
    the merge diff before running ``git merge`` to enforce the same rule
    at merge time (see ``prompt/develop_prompt/_shared/stage2-subagent-playbook.md``,
    "3b PASS 后: 自驱合入主干" section).
    """
    allowed_prefix = f"workload/ops/{op_name}/"
    return [path for path in changed_paths if path and not path.startswith(allowed_prefix)]


REPO_WRITE_SURFACE_PREFIXES = (
    "harness/",
    "evals/",
    "prompt/",
    # Platform mirrors. SSOT lives under .rules/; scripts/sync_skills.py
    # and scripts/sync_project_guide.py regenerate the per-platform
    # discovery files (Cursor / Claude Code / Codex / Copilot). Edits
    # here are gated by the sync hook so the auto-generated mirrors are
    # part of the legitimate write surface.
    ".cursor/rules/",
    ".cursor/skills/",
    ".claude/",
    ".rules/",
    ".github/",
    "scripts/",
    "workload/src/",
    "workload/ops/",
    "workload/notes/",
    "workload/profile/",
    "tools/",
    "config/",
    "web/",
)
REPO_WRITE_SURFACE_FILES = {
    "agent-loop.sh",
    "AGENTS.md",
    "CLAUDE.md",
    "README.md",
    "README_zh.md",
    "LICENSE",
    "pyproject.toml",
    ".gitignore",
    ".gitmodules",
    ".pre-commit-config.yaml",
    # Linter sidecar configs (same class as .pre-commit-config.yaml — they
    # tune the hook suite, not application behavior).
    ".codespellrc",
    ".secrets.baseline",
}
REPO_FROZEN_PREFIXES = ("ref/",)
REPO_FROZEN_FILES: set[str] = set()


def _is_repo_write_surface(filepath: str) -> bool:
    if filepath in REPO_WRITE_SURFACE_FILES:
        return True
    return any(filepath.startswith(prefix) for prefix in REPO_WRITE_SURFACE_PREFIXES)


def _check_repo_write_surface(branch: str, changed_files: list[str]) -> list[GuardViolation]:
    violations: list[GuardViolation] = []
    for filepath in changed_files:
        frozen = filepath in REPO_FROZEN_FILES or any(
            filepath.startswith(prefix) for prefix in REPO_FROZEN_PREFIXES
        )
        if not frozen and _is_repo_write_surface(filepath):
            continue
        violations.append(
            GuardViolation(
                path=Path(filepath),
                line_number=0,
                line="",
                reason=(
                    f"path isolation: branch '{branch}' may not modify "
                    f"read-only path '{filepath}' (see review_common.md Check E)"
                ),
            )
        )
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    with_path_guard = "--with-path-guard" in args
    if with_path_guard:
        args.remove("--with-path-guard")

    exit_code = 0

    try:
        violations = scan_paths(args)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    if violations:
        print("Prohibited framework references found:", file=sys.stderr)
        print(file=sys.stderr)
        for violation in violations:
            print(f"  {violation.render()}", file=sys.stderr)
        print(file=sys.stderr)
        print(ALLOWED_SUMMARY, file=sys.stderr)
        print(BANNED_SUMMARY, file=sys.stderr)
        exit_code = 1

    if with_path_guard:
        path_violations = check_path_isolation()
        if path_violations:
            print("Path isolation violations found:", file=sys.stderr)
            print(file=sys.stderr)
            for v in path_violations:
                print(f"  {v.reason}", file=sys.stderr)
            print(file=sys.stderr)
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
