"""Unified M1–M5 capture interposer for Megatron and Torch backends.

Runs as the torchrun entry point (replaces the original training
script via sed-patch in bridge.sh). Monkey-patches the framework
call that returns (model, optimizer), wires
``evals.harness_hook.install()`` or ``install_canonical_state_dump()``,
then delegates to the original training entry via ``runpy.run_path()``.

CLI wire (preferred — typed, no env var):
    --hash-capture-level <int>   0 / 1 / 2 (see harness_hook docstring)
    --hash-output <abs path>     where the JSON hash dump lands
    --persistent                 set for M2–M5; unset for M1 single-step

Env contract (back-compat — bridge.sh still exports these):
    BRIDGE_BACKEND              "megatron" | "torch"
    BRIDGE_ORIGINAL_ENTRY       path to the original Python entry
    BRIDGE_REF_DIR              directory of the original entry (for local imports)
    BRIDGE_REPO_ROOT            repo root (for harness imports)
    HOOK_OUTPUT_FILE            back-compat M1 capture output path; --hash-output overrides
    CANONICAL_STATE_OUTPUT_FILE canonical-state bootstrap output path

When ``--persistent`` is set the interposer installs a
:class:`evals.harness_hook.CaptureSession`, then patches the ref
training loop (``train_pure_mup_mtp.main`` for the torch backend) to
drive ``session.begin_step / capture / capture_grads / dump`` at the
hook points spelled out in the project plan. The monkey-patches use
``inspect.getsource`` + string replacement + ``exec`` (same pattern
as ``_install_gpt_dataset_batch_fix``).

No sitecustomize. No PYTHONPATH pollution. Subprocesses spawned by
transformer_engine (pip show, etc.) will NOT re-trigger this
interposer because it is loaded explicitly as the torchrun entry, not
via the site machinery.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evals.harness_hook import CaptureSession

# ── CLI args ─────────────────────────────────────────────────────────
#
# We use ``parse_known_args`` so our flags are stripped before the
# original L0 ref script sees its own argv. The L0 script's argparse
# (torch ref) or Megatron's parser would otherwise reject unknown
# ``--hash-*`` flags.


def _parse_cli() -> tuple[int, str, bool]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--hash-capture-level", type=int, default=0)
    parser.add_argument("--hash-output", type=str, default="")
    parser.add_argument("--persistent", action="store_true")
    args, remaining = parser.parse_known_args(sys.argv[1:])
    # Strip our flags from sys.argv so the L0 entry's argparse only
    # sees its own flags. ``sys.argv[0]`` (the interposer path / patched
    # launcher script) is preserved as the bash convention requires.
    sys.argv = [sys.argv[0], *remaining]
    return int(args.hash_capture_level), (args.hash_output or "").strip(), bool(args.persistent)


HASH_CAPTURE_LEVEL, _HASH_OUTPUT_CLI, PERSISTENT = _parse_cli()


# ── Env contract ─────────────────────────────────────────────────────


def _require_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise SystemExit(f"{name} must be set by bridge.sh")
    return value


BACKEND = _require_env("BRIDGE_BACKEND")
ORIGINAL_ENTRY = _require_env("BRIDGE_ORIGINAL_ENTRY")
REF_DIR = os.environ.get("BRIDGE_REF_DIR", "")
REPO_ROOT = os.environ.get("BRIDGE_REPO_ROOT", "")

# ``--hash-output`` takes precedence over the legacy ``HOOK_OUTPUT_FILE``
# env var; both are kept transitionally so old bridges keep working.
HOOK_OUTPUT = _HASH_OUTPUT_CLI or (os.environ.get("HOOK_OUTPUT_FILE", "").strip() or "")
HOOK_OUTPUT = HOOK_OUTPUT or None
CANONICAL_OUTPUT = os.environ.get("CANONICAL_STATE_OUTPUT_FILE", "").strip() or None

# Persistent session handle. ``None`` in single-step (M1) mode; populated
# after _wire_hook fires in persistent mode so the ref-loop monkey-patches
# can pick it up.
_SESSION: CaptureSession | None = None

# Pre-patched ref-train source (written to a temp file at interposer
# startup so runpy.run_path executes the patched version, not the
# read-only original). Populated by the active torch ref family's prepatch
# (``_TORCH_REF_FAMILIES``; MiniCPM muP+MTP or Qwen3 dense) whenever
# ``--persistent`` is requested. The patched source has
# ``_HARNESS_SESSION = None`` / ``_HARNESS_GRAD_NAMES = []`` injected
# at the top; the optim-init hook later writes the real values via
# ``sys.modules["__main__"]`` so the per-step patches inside
# ``main()`` see a populated session as the for-step loop runs.
_PATCHED_ENTRY: str | None = None


# ── Path bootstrap ───────────────────────────────────────────────────


def _bootstrap_paths() -> None:
    if REF_DIR and REF_DIR not in sys.path:
        sys.path.insert(0, REF_DIR)
    if REPO_ROOT and REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)


# ── Shared hook wiring ───────────────────────────────────────────────


def _wire_hook(
    model: object,
    optimizer: object,
    *,
    grad_attrs: tuple[str, ...],
) -> None:
    global _SESSION

    from evals.harness_hook import (
        install,
        install_canonical_state_dump,
    )

    if HOOK_OUTPUT:
        result = install(
            model,
            optimizer,
            output_file=HOOK_OUTPUT,
            hash_capture_level=HASH_CAPTURE_LEVEL,
            persistent=PERSISTENT,
            grad_attrs=grad_attrs,
        )
        if PERSISTENT:
            _SESSION = result
            # Hand the session + FQN list to the in-flight main() module
            # via sys.modules["__main__"], which runpy.run_path sets to
            # the executing patched file. The source patches added at
            # interposer startup (``_prepatch_train_pure_mup_mtp_source``)
            # read these two globals at every for-step iteration.
            if BACKEND == "torch":
                _install_torch_ref_loop_globals(model)
    elif CANONICAL_OUTPUT:
        install_canonical_state_dump(model, optimizer, output_file=CANONICAL_OUTPUT)


# ── Torch backend: ref training-loop patches ─────────────────────────
#
# When --persistent is set, the bridge needs to drive the
# CaptureSession through the ref training loop. The ref file
# ``train_pure_mup_mtp.py`` is read-only, so we pre-patch its source
# at interposer startup, write the patched copy to a temp file, and
# repoint runpy at the temp file (see
# ``_prepatch_train_pure_mup_mtp_source``). The patches reference two
# module-level globals — ``_HARNESS_SESSION`` and ``_HARNESS_GRAD_NAMES``
# — initialised to safe defaults at the top of the patched module.
# Once the AdamW optim-init hook fires (after the model is built),
# ``_install_torch_ref_loop_globals`` writes the live session and
# the per-FQN grad-name list into those globals via
# ``sys.modules["__main__"]``, and the per-step patches reading those
# globals become live for the rest of training.
#
# Patches injected (each guarded by ``if _HARNESS_SESSION is not None``
# so the patched module still runs cleanly when no session is active —
# e.g. a non-capture rerun of the same module):
#
#   * before each step's microbatch loop:
#       _HARNESS_SESSION.begin_step(step)
#   * on first microbatch's lm head (mb==0): per-token NLL captured
#       (recomputed cheaply from logits / labels / mask) under
#       ``loss.per_token.preallreduce``
#   * after all microbatches accumulate into fp32_grad_bufs but before
#       the DP all-reduce: walk fp32_grad_bufs + _HARNESS_GRAD_NAMES and
#       call session.capture("grad.<fqn>.preallreduce", buf) per param
#   * after the DP all-reduce on flat: same walk with
#       ``.postallreduce`` suffix
#   * after the local→global loss reduce: session.capture(
#       "loss.scalar.preallreduce" / "loss.scalar.postallreduce", t)
#   * an atexit handler dumps the session at process exit


def _prepatch_train_pure_mup_mtp_source(repo_root_dir: str) -> str | None:
    """Source-rewrite ``train_pure_mup_mtp.py``; return path to the patched copy.

    Read the read-only ref training-loop module's source, inject two
    module-level globals (``_HARNESS_SESSION`` / ``_HARNESS_GRAD_NAMES``)
    near the top, apply the 6 hook-point ``str.replace`` substitutions,
    and write the patched copy to a stable temp file. Returns the
    absolute path of the patched file (consumed by ``main()`` as the
    new ``ORIGINAL_ENTRY`` for runpy), or ``None`` if any needle didn't
    match — that case is reported to stderr and the interposer falls
    back to the un-patched original so the gate sees a clean Phase-1
    succeed (with only module fwd / bwd records, no loss / grad family).
    """
    from pathlib import Path as _Path

    src_path = _Path(repo_root_dir) / "ref" / "reference" / "train_pure_mup_mtp.py"
    if not src_path.is_file():
        sys.stderr.write(
            f"[interposer] persistent pre-patch skipped — source not found at {src_path}\n"
        )
        return None
    src = src_path.read_text(encoding="utf-8")

    # ── Patch 0: inject module-level globals near the top (after the
    # final ``__future__`` import; we anchor on the existing ``import F``
    # alias line which sits at module scope right after the stdlib
    # imports). The optim-init hook later overwrites these via
    # sys.modules["__main__"].
    needle_top = "import torch.nn.functional as F\n"
    patch_top = (
        "import torch.nn.functional as F\n"
        "\n"
        "# ── harness_hook persistent-mode globals (set by the M1-M5 interposer\n"
        "# after the optimizer is built; see ref/bridges/interposer.py).\n"
        "# Default values keep the patched body inert under a non-capture run.\n"
        "_HARNESS_SESSION = None\n"
        "_HARNESS_GRAD_NAMES: list = []\n"
    )

    # ── Patch 1: begin_step at the top of each iteration ──
    needle_begin = "    for step in range(args.train_iters):\n        torch.cuda.synchronize()\n"
    patch_begin = (
        "    for step in range(args.train_iters):\n"
        "        if _HARNESS_SESSION is not None:\n"
        "            _HARNESS_SESSION.begin_step(step)\n"
        "        torch.cuda.synchronize()\n"
    )

    # ── Patch 2: per-token NLL capture on first microbatch's lm head ──
    needle_pt_nll_a = (
        "                lm_sum, lm_n = masked_ce(logits, labels, loss_mask)\n"
        "                mtp_sum, mtp_n = masked_ce(logits_mtp, mtp_lab, mtp_mask)\n"
    )
    patch_pt_nll_a = (
        "                if _HARNESS_SESSION is not None and _mb == 0:\n"
        "                    _B, _S, _V = logits.shape\n"
        "                    _nll = F.cross_entropy(logits.reshape(-1, _V).float(), labels.reshape(-1), reduction='none').reshape(_B, _S)\n"
        "                    _HARNESS_SESSION.capture('loss.per_token.preallreduce', _nll)\n"
        "                lm_sum, lm_n = masked_ce(logits, labels, loss_mask)\n"
        "                mtp_sum, mtp_n = masked_ce(logits_mtp, mtp_lab, mtp_mask)\n"
    )
    needle_pt_nll_b = (
        "                lm_sum, lm_n = masked_ce(logits, labels, loss_mask)\n"
        "                obj = lm_sum\n"
    )
    patch_pt_nll_b = (
        "                if _HARNESS_SESSION is not None and _mb == 0:\n"
        "                    _B, _S, _V = logits.shape\n"
        "                    _nll = F.cross_entropy(logits.reshape(-1, _V).float(), labels.reshape(-1), reduction='none').reshape(_B, _S)\n"
        "                    _HARNESS_SESSION.capture('loss.per_token.preallreduce', _nll)\n"
        "                lm_sum, lm_n = masked_ce(logits, labels, loss_mask)\n"
        "                obj = lm_sum\n"
    )

    # ── Patch 3: pre-allreduce grad capture + loss.scalar.preallreduce ──
    needle_pre_grad = (
        "        if world_size > 1:\n"
        "            stats = torch.cat([local_lm_sum, local_lm_n, local_mtp_sum, local_mtp_n])\n"
        "            dist.all_reduce(stats, op=dist.ReduceOp.SUM)\n"
    )
    patch_pre_grad = (
        "        if _HARNESS_SESSION is not None:\n"
        "            _HARNESS_SESSION.capture('loss.scalar.preallreduce', local_lm_sum.float() / local_lm_n.float().clamp(min=1.0))\n"
        "            _HARNESS_SESSION.capture_batch([(f'grad.{_n}.preallreduce', _b) for _n, _b in zip(_HARNESS_GRAD_NAMES, fp32_grad_bufs)])\n"
        "        if world_size > 1:\n"
        "            stats = torch.cat([local_lm_sum, local_lm_n, local_mtp_sum, local_mtp_n])\n"
        "            dist.all_reduce(stats, op=dist.ReduceOp.SUM)\n"
    )

    # ── Patch 4: post-allreduce loss scalar capture ──
    needle_post_loss = "        reported_lm = g_lm_sum / max(g_lm_n, 1.0)\n"
    patch_post_loss = (
        "        reported_lm = g_lm_sum / max(g_lm_n, 1.0)\n"
        "        if _HARNESS_SESSION is not None:\n"
        "            import torch as _t\n"
        "            _HARNESS_SESSION.capture('loss.scalar.postallreduce', _t.tensor([reported_lm], dtype=_t.float32))\n"
    )

    # ── Patch 5: post-allreduce grad capture ──
    needle_post_grad = (
        "        grad_norm = torch.nn.utils.clip_grad_norm_(fp32_master, args.clip_grad).item()\n"
        "        optim.step()\n"
    )
    patch_post_grad = (
        "        if _HARNESS_SESSION is not None:\n"
        "            _HARNESS_SESSION.capture_batch([(f'grad.{_n}.postallreduce', _b) for _n, _b in zip(_HARNESS_GRAD_NAMES, fp32_grad_bufs)])\n"
        "        grad_norm = torch.nn.utils.clip_grad_norm_(fp32_master, args.clip_grad).item()\n"
        "        optim.step()\n"
    )

    needles_and_patches = [
        (needle_top, patch_top, "globals injection"),
        (needle_begin, patch_begin, "begin_step"),
        (needle_pt_nll_a, patch_pt_nll_a, "loss.per_token (mtp branch)"),
        (needle_pt_nll_b, patch_pt_nll_b, "loss.per_token (lm-only branch)"),
        (needle_pre_grad, patch_pre_grad, "grad.preallreduce + loss.scalar.preallreduce"),
        (needle_post_loss, patch_post_loss, "loss.scalar.postallreduce"),
        (needle_post_grad, patch_post_grad, "grad.postallreduce"),
    ]
    return _write_persistent_patched_copy(
        src,
        needles_and_patches,
        out_basename="train_pure_mup_mtp_patched_rank",
        shape_label="train_pure_mup_mtp.py",
    )


def _write_persistent_patched_copy(
    src: str,
    needles_and_patches: list[tuple[str, str, str]],
    *,
    out_basename: str,
    shape_label: str,
) -> str | None:
    """Apply ``(needle, patch, tag)`` substitutions to ``src`` and write the
    patched copy to a rank-keyed temp file.

    Shared by every torch ref family's persistent prepatch. Fail-closed: if
    any needle is absent (the ref training loop drifted) the patch set is
    refused and ``None`` is returned, so ``main()`` falls back to the
    un-patched original and the gate still produces a clean Phase-1 capture.
    """
    missing = [tag for needle, _, tag in needles_and_patches if needle not in src]
    if missing:
        sys.stderr.write(
            "[interposer] refusing to install persistent patches — "
            f"{shape_label} shape changed; missing needles: {missing}\n"
        )
        return None

    patched_src = src
    for needle, patch, _tag in needles_and_patches:
        patched_src = patched_src.replace(needle, patch, 1)

    # Stable temp file under <workspace>/tmp keyed on rank so DP ranks
    # don't fight for the same name. The interposer is the torchrun entry
    # (cwd = the per-loop workspace), so RANK is set per process and the
    # path lands on the loop's mounted volume, not the docker overlay.
    rank = os.environ.get("RANK", "0")
    out_dir = os.path.join(os.getcwd(), "tmp")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{out_basename}{rank}.py")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(patched_src)

    sys.stderr.write(
        f"[interposer] persistent pre-patch wrote {out_path} "
        f"({len(needles_and_patches)} hook points injected)\n"
    )
    return out_path


def _prepatch_qwen3_dense_source(repo_root_dir: str) -> str | None:
    """Source-rewrite ``train_qwen3_dense.py`` for persistent capture.

    Qwen3 dense shares ``train_pure_mup_mtp``'s loop skeleton (same
    ``masked_ce`` / ``fp32_grad_bufs`` / ``clip_grad_norm_(fp32_master, …)``
    / ``dist.all_reduce`` machinery) but has NO MTP branch and a shallower
    micro-batch indentation, so it needs its own needle set rather than the
    MiniCPM one. Qwen3 imports ``torch`` but not ``torch.nn.functional as
    F``; the per-token NLL hook therefore calls
    ``torch.nn.functional.cross_entropy`` directly and the module globals
    are anchored on the ``import torch.distributed as dist`` line.
    """
    src_path = Path(repo_root_dir) / "ref" / "reference" / "train_qwen3_dense.py"
    if not src_path.is_file():
        sys.stderr.write(
            f"[interposer] persistent pre-patch skipped — source not found at {src_path}\n"
        )
        return None
    src = src_path.read_text(encoding="utf-8")

    # ── Patch 0: module-level globals, anchored on the dist import ──
    needle_top = "import torch.distributed as dist\n"
    patch_top = (
        "import torch.distributed as dist\n"
        "\n"
        "# ── harness_hook persistent-mode globals (set by the M1-M5 interposer\n"
        "# after the optimizer is built; see ref/bridges/interposer.py).\n"
        "# Default values keep the patched body inert under a non-capture run.\n"
        "_HARNESS_SESSION = None\n"
        "_HARNESS_GRAD_NAMES: list = []\n"
    )

    # ── Patch 1: begin_step at the top of each iteration ──
    needle_begin = "    for step in range(args.train_iters):\n        torch.cuda.synchronize()\n"
    patch_begin = (
        "    for step in range(args.train_iters):\n"
        "        if _HARNESS_SESSION is not None:\n"
        "            _HARNESS_SESSION.begin_step(step)\n"
        "        torch.cuda.synchronize()\n"
    )

    # ── Patch 2: per-token NLL on the first microbatch (lm-only, 12-sp) ──
    needle_pt_nll = (
        "            lm_sum, lm_n = masked_ce(logits, labels, loss_mask)\n"
        "            obj = lm_sum\n"
    )
    patch_pt_nll = (
        "            if _HARNESS_SESSION is not None and _mb == 0:\n"
        "                _B, _S, _V = logits.shape\n"
        "                _nll = torch.nn.functional.cross_entropy(logits.reshape(-1, _V).float(), labels.reshape(-1), reduction='none').reshape(_B, _S)\n"
        "                _HARNESS_SESSION.capture('loss.per_token.preallreduce', _nll)\n"
        "            lm_sum, lm_n = masked_ce(logits, labels, loss_mask)\n"
        "            obj = lm_sum\n"
    )

    # ── Patch 3: pre-allreduce grad + loss.scalar capture (lm-only stats) ──
    needle_pre_grad = (
        "        if world_size > 1:\n"
        "            stats = torch.cat([local_lm_sum, local_lm_n])\n"
        "            dist.all_reduce(stats, op=dist.ReduceOp.SUM)\n"
    )
    patch_pre_grad = (
        "        if _HARNESS_SESSION is not None:\n"
        "            _HARNESS_SESSION.capture('loss.scalar.preallreduce', local_lm_sum.float() / local_lm_n.float().clamp(min=1.0))\n"
        "            _HARNESS_SESSION.capture_batch([(f'grad.{_n}.preallreduce', _b) for _n, _b in zip(_HARNESS_GRAD_NAMES, fp32_grad_bufs)])\n"
        "        if world_size > 1:\n"
        "            stats = torch.cat([local_lm_sum, local_lm_n])\n"
        "            dist.all_reduce(stats, op=dist.ReduceOp.SUM)\n"
    )

    # ── Patch 4: post-allreduce loss scalar ──
    needle_post_loss = "        reported_lm = g_lm_sum / max(g_lm_n, 1.0)\n"
    patch_post_loss = (
        "        reported_lm = g_lm_sum / max(g_lm_n, 1.0)\n"
        "        if _HARNESS_SESSION is not None:\n"
        "            import torch as _t\n"
        "            _HARNESS_SESSION.capture('loss.scalar.postallreduce', _t.tensor([reported_lm], dtype=_t.float32))\n"
    )

    # ── Patch 5: post-allreduce grad ──
    needle_post_grad = (
        "        grad_norm = torch.nn.utils.clip_grad_norm_(fp32_master, args.clip_grad).item()\n"
        "        optim.step()\n"
    )
    patch_post_grad = (
        "        if _HARNESS_SESSION is not None:\n"
        "            _HARNESS_SESSION.capture_batch([(f'grad.{_n}.postallreduce', _b) for _n, _b in zip(_HARNESS_GRAD_NAMES, fp32_grad_bufs)])\n"
        "        grad_norm = torch.nn.utils.clip_grad_norm_(fp32_master, args.clip_grad).item()\n"
        "        optim.step()\n"
    )

    needles_and_patches = [
        (needle_top, patch_top, "globals injection"),
        (needle_begin, patch_begin, "begin_step"),
        (needle_pt_nll, patch_pt_nll, "loss.per_token (lm-only branch)"),
        (needle_pre_grad, patch_pre_grad, "grad.preallreduce + loss.scalar.preallreduce"),
        (needle_post_loss, patch_post_loss, "loss.scalar.postallreduce"),
        (needle_post_grad, patch_post_grad, "grad.postallreduce"),
    ]
    return _write_persistent_patched_copy(
        src,
        needles_and_patches,
        out_basename="train_qwen3_dense_patched_rank",
        shape_label="train_qwen3_dense.py",
    )


# ── Torch ref family registry ────────────────────────────────────────
#
# Maps the original train-entry basename (set by bridge.sh from the active
# ``[ref].ref_script``) to the (model module, model class, persistent-mode
# source prepatch) it should hook. Adding a torch model = adding its
# run_*.sh / train_*.py / model_*.py plus one row here; no other branch in
# this file is model-specific.
_TORCH_REF_FAMILIES: dict[str, tuple[str, str, Any]] = {
    "train_pure_mup_mtp.py": (
        "model_pure_mup_mtp",
        "MiniCPM4MupMtp",
        _prepatch_train_pure_mup_mtp_source,
    ),
    "train_qwen3_dense.py": (
        "model_qwen3",
        "Qwen3Dense",
        _prepatch_qwen3_dense_source,
    ),
}


def _resolve_torch_ref_family() -> tuple[str, str, Any]:
    """Resolve the active torch ref family from ``ORIGINAL_ENTRY``.

    Fail-fast: an unregistered entry means a new torch model was wired into
    a run_*.sh without a matching ``_TORCH_REF_FAMILIES`` row, which would
    otherwise silently fall back to the MiniCPM hooks and mis-capture.
    """
    entry_base = os.path.basename(ORIGINAL_ENTRY)
    family = _TORCH_REF_FAMILIES.get(entry_base)
    if family is None:
        raise SystemExit(
            f"[interposer] no torch ref family registered for entry {entry_base!r}; "
            f"known: {sorted(_TORCH_REF_FAMILIES)}. Add a row to "
            "_TORCH_REF_FAMILIES (model module, model class, prepatch fn)."
        )
    return family


def _install_torch_ref_loop_globals(model: object) -> None:
    """Populate ``_HARNESS_SESSION`` / ``_HARNESS_GRAD_NAMES`` on the
    in-flight ``__main__`` module so the pre-patched ref training loop
    reads live values when its for-step loop starts.

    Called from the AdamW optim-init hook after ``_wire_hook`` produces
    a session. The patched module's pristine values
    (``None`` / ``[]``) are kept until this assignment fires; after
    that, every for-step iteration's hooks become active.
    """
    if _SESSION is None:
        return

    grad_names: list[str] = []
    inner = model
    while hasattr(inner, "module"):
        inner = inner.module  # type: ignore[assignment]
    for name, _p in inner.named_parameters():  # type: ignore[union-attr]
        grad_names.append(name)

    main_mod = sys.modules.get("__main__")
    if main_mod is None:
        sys.stderr.write(
            "[interposer] persistent globals injection skipped — no __main__ module in sys.modules\n"
        )
        return
    main_mod._HARNESS_SESSION = _SESSION  # type: ignore[attr-defined]
    main_mod._HARNESS_GRAD_NAMES = grad_names  # type: ignore[attr-defined]

    # Register an atexit handler to dump the session — covers normal
    # exit *and* exceptions raised from the training loop, both of
    # which still want the partial dump on disk for diagnosis.
    import atexit

    def _atexit_dump() -> None:
        try:
            if _SESSION is not None:
                _SESSION.dump()
        except Exception as exc:  # never mask the underlying training-loop exit
            sys.stderr.write(f"[interposer] atexit dump failed: {exc!r}\n")

    atexit.register(_atexit_dump)

    sys.stderr.write(
        "[interposer] persistent globals set on __main__ "
        f"(_HARNESS_SESSION populated, _HARNESS_GRAD_NAMES has {len(grad_names)} FQNs); "
        "atexit dump handler registered\n"
    )


# ── Megatron backend ─────────────────────────────────────────────────


def _install_megatron_hooks() -> None:
    from evals.harness_hook._megatron_argparse_shim import (
        install_megatron_argparse_shim,
    )

    install_megatron_argparse_shim()

    try:
        from megatron.training import training as megatron_training
    except ImportError as exc:
        sys.stderr.write(f"[interposer] cannot import megatron.training.training: {exc!r}\n")
        return

    original = megatron_training.setup_model_and_optimizer

    def _patched(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        model = result[0]
        optimizer = result[1]
        single = model[0] if isinstance(model, (list, tuple)) else model
        _wire_hook(single, optimizer, grad_attrs=("main_grad", "grad"))
        return result

    megatron_training.setup_model_and_optimizer = _patched
    sys.stderr.write(
        "[interposer] installed M1 hook on setup_model_and_optimizer; "
        f"HOOK_OUTPUT_FILE={HOOK_OUTPUT} "
        f"HASH_CAPTURE_LEVEL={HASH_CAPTURE_LEVEL} "
        f"PERSISTENT={PERSISTENT} "
        f"CANONICAL_STATE_OUTPUT_FILE={CANONICAL_OUTPUT}\n"
    )

    _install_gpt_dataset_batch_fix()


def _install_gpt_dataset_batch_fix() -> None:
    """Patch ``megatron.training.utils.get_batch_on_this_tp_rank``.

    The on-remote Megatron fork (cpm_core_r0.15.0) only reads
    ``data = next(data_iterator)`` inside ``if args.use_modelbest_sdk:``;
    under the GPTDataset path the variable is never assigned and the
    first ``forward_step`` raises ``UnboundLocalError``.
    """
    try:
        import inspect

        from megatron.training import utils as megatron_utils
    except ImportError as exc:
        sys.stderr.write(
            f"[interposer] cannot import megatron.training.utils for get_batch fix: {exc!r}\n"
        )
        return

    original = getattr(megatron_utils, "get_batch_on_this_tp_rank", None)
    if original is None:
        sys.stderr.write(
            "[interposer] megatron.training.utils has no "
            "get_batch_on_this_tp_rank; skipping GPTDataset fix\n"
        )
        return

    try:
        src = inspect.getsource(original)
    except (OSError, TypeError) as exc:
        sys.stderr.write(f"[interposer] cannot read get_batch_on_this_tp_rank source: {exc!r}\n")
        return

    needle = "        if args.use_modelbest_sdk:\n            data = next(data_iterator)\n"
    replacement = (
        "        if not args.use_modelbest_sdk:\n"
        "            data = next(data_iterator)\n"
        "        if args.use_modelbest_sdk:\n"
        "            data = next(data_iterator)\n"
    )
    if needle not in src:
        sys.stderr.write(
            "[interposer] get_batch_on_this_tp_rank source shape "
            "changed; refusing to patch (upstream fork drift)\n"
        )
        return

    patched_src = src.replace(needle, replacement, 1)
    try:
        exec(  # nosec B102
            compile(patched_src, "<interposer get_batch fix>", "exec"),
            megatron_utils.__dict__,
        )
    except Exception as exc:
        sys.stderr.write(
            f"[interposer] failed to exec patched get_batch_on_this_tp_rank: {exc!r}\n"
        )
        return

    inner_patched = megatron_utils.get_batch_on_this_tp_rank
    _FORK_ONLY_KEYS = ("dataset_id", "packed_seq_params")

    def _strip_fork_only_keys(batch: Any) -> Any:
        if isinstance(batch, dict):
            for key in _FORK_ONLY_KEYS:
                batch.pop(key, None)
        return batch

    def _wrap_tp_rank(data_iterator: Any) -> Any:
        return _strip_fork_only_keys(inner_patched(data_iterator))

    megatron_utils.get_batch_on_this_tp_rank = _wrap_tp_rank

    inner_cp = getattr(megatron_utils, "get_batch_on_this_cp_rank", None)
    if inner_cp is not None:

        def _wrap_cp_rank(batch: Any) -> Any:
            return _strip_fork_only_keys(inner_cp(batch))

        megatron_utils.get_batch_on_this_cp_rank = _wrap_cp_rank

    sys.stderr.write(
        "[interposer] patched megatron.training.utils."
        "get_batch_on_this_tp_rank / get_batch_on_this_cp_rank for "
        "GPTDataset path (read-iterator + drop dataset_id/packed_seq_params)\n"
    )


# ── Torch backend ────────────────────────────────────────────────────


def _install_torch_hooks() -> None:
    global _PATCHED_ENTRY
    import importlib

    import torch

    model_module_name, model_class_name, prepatch_fn = _resolve_torch_ref_family()

    # Pre-patch the ref training-loop source for persistent mode BEFORE the
    # customer entry runs (the original train_*.main has already entered its
    # frame by the time the optim-init hook fires, so source rewrites done
    # from inside that hook would be inert; pre-patching and re-pointing
    # runpy at the patched copy is the only way to drive ``main()``
    # line-by-line from the session). The prepatch is family-specific
    # (MiniCPM muP+MTP vs Qwen3 dense), resolved from the active ref entry.
    if PERSISTENT:
        _PATCHED_ENTRY = prepatch_fn(REPO_ROOT)

    _ref_model_mod = importlib.import_module(model_module_name)
    _model_cls = getattr(_ref_model_mod, model_class_name)

    state: dict[str, object] = {"model": None, "installed": False}

    real_model_init = _model_cls.__init__

    def _patched_model_init(self, *a: Any, **k: Any) -> None:  # type: ignore[no-untyped-def]
        real_model_init(self, *a, **k)
        state["model"] = self

    _model_cls.__init__ = _patched_model_init  # type: ignore[assignment]

    real_optim_init = torch.optim.AdamW.__init__

    def _patched_optim_init(self, *a: Any, **k: Any) -> None:  # type: ignore[no-untyped-def]
        real_optim_init(self, *a, **k)
        if state["installed"] or state["model"] is None:
            return
        state["installed"] = True
        # Mirror the megatron path's attr order: ``main_grad`` first, ``grad``
        # as fallback. The torch ref's default is ``WGRAD_ACCUM_FP32=True``
        # (ref/reference/model_pure_mup_mtp.py:67), which routes wgrad into
        # ``param.main_grad`` (fp32) and leaves ``param.grad`` as ``None``;
        # the legacy ``("grad",)`` order produced zero ``grad.*`` keys in
        # the M1.bwd capture against that default. Listing ``grad`` second
        # keeps ``WGRAD_ACCUM_FP32=False`` working (stock bf16 ``.grad``).
        _wire_hook(state["model"], self, grad_attrs=("main_grad", "grad"))

    torch.optim.AdamW.__init__ = _patched_optim_init  # type: ignore[assignment]

    sys.stderr.write(
        "[interposer] installed torch two-phase capture hooks "
        f"({model_class_name}.__init__ + AdamW.__init__); "
        f"HOOK_OUTPUT_FILE={HOOK_OUTPUT} "
        f"HASH_CAPTURE_LEVEL={HASH_CAPTURE_LEVEL} "
        f"PERSISTENT={PERSISTENT} "
        f"PATCHED_ENTRY={_PATCHED_ENTRY} "
        f"CANONICAL_STATE_OUTPUT_FILE={CANONICAL_OUTPUT}\n"
    )


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    _bootstrap_paths()

    if BACKEND == "megatron":
        _install_megatron_hooks()
    elif BACKEND == "torch":
        _install_torch_hooks()
    else:
        raise SystemExit(f"unknown BRIDGE_BACKEND={BACKEND!r} (expected megatron|torch)")

    # If persistent mode pre-patched the source, runpy the patched
    # copy instead of the read-only original so the for-step loop's
    # hook calls actually execute.
    entry = Path(_PATCHED_ENTRY) if _PATCHED_ENTRY else Path(ORIGINAL_ENTRY)
    if not entry.is_absolute():
        entry = Path.cwd() / entry
    sys.argv[0] = str(entry)
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
