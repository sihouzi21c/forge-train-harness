#!/usr/bin/env python3
"""Profile one training step to find time breakdown by operation category.

Inputs:
  --checkpoint-root  : directory holding the (large, external) model weights.
                       Defaults to env CHECKPOINT_ROOT (required).
  --captured-batch   : workspace-local input batch produced by
                       `tools/capture_profile_batch.py`.
                       Defaults to `<repo>/workload/profile/captured_batch.pt`.
                       Run the capture script once to populate it; no
                       fallback path is consulted.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("FUSED_OPS", "1")

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent

# Bootstrap PYTHONPATH so we can import the SSOT helper from harness.
# (This file is a data-plane script run as ``python tools/profile_step.py``
# rather than through the harness CLI, so sys.path is not pre-arranged.)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness.config_runtime import workload_src_path

sys.path.insert(0, str(workload_src_path(_REPO_ROOT)))


def _require_stage1_artifacts() -> None:
    """Fail fast when Stage 1 has not yet populated the engine submodules.

    ``profile_step`` is the Stage 2 M1 scout entry point and depends on
    the full Stage 1 ``training_engine_tensor`` surface
    (``forward`` / ``backward`` / ``optimizer`` / ``parameters`` /
    ``kernels`` / ``triton_kernels``; the authoritative submodule list
    lives in ``training_engine_tensor.train_loop`` docstring). A fresh
    checkout that has not run the Stage 1 agent loop will raise
    ``ImportError`` on the imports below — surface that as a single
    actionable message instead.
    """
    src_dir = Path(workload_src_path(_REPO_ROOT)) / "training_engine_tensor"
    required = (
        "config.py",
        "forward.py",
        "backward.py",
        "kernels.py",
        "optimizer.py",
        "parameters.py",
        "triton_kernels.py",
    )
    missing = [name for name in required if not (src_dir / name).exists()]
    if missing:
        rel = src_dir.relative_to(_REPO_ROOT)
        sys.stderr.write(
            f"\nERROR: tools/profile_step.py requires Stage 1 to be complete; "
            f"the following submodules under {rel}/ are missing:\n"
            f"  {', '.join(missing)}\n"
            f"  Drive the Stage 1 agent loop to landing first "
            f"(see prompt/develop_prompt/stage1.md).\n\n"
        )
        sys.exit(2)


_require_stage1_artifacts()

import torch
from training_engine_tensor import config
from training_engine_tensor.backward import backward_pass
from training_engine_tensor.forward import forward_pass_with_save
from training_engine_tensor.kernels import precompute_rope_freqs
from training_engine_tensor.optimizer import (
    MAX_GRAD_NORM,
    AdamState,
    adam_step,
    clip_gradients_fp32,
    compute_grad_norm_fp32,
    compute_lr,
    sync_params_from_master,
)
from training_engine_tensor.parameters import load_megatron_checkpoint, trainable_param_names

DEFAULT_CHECKPOINT_ROOT = os.environ.get("CHECKPOINT_ROOT", "")

# Workspace-local default. The captured input batch is small (a few MB of
# token IDs + labels + loss_mask) and must live inside the repo so the
# profile is reproducible without depending on any external trace-output
# directory. Generate it via `tools/capture_profile_batch.py`.
DEFAULT_CAPTURED_BATCH = str(_REPO_ROOT / "workload" / "profile" / "captured_batch.pt")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint-root",
        default=DEFAULT_CHECKPOINT_ROOT,
        help="External model checkpoint root (large weights).",
    )
    p.add_argument(
        "--captured-batch",
        default=DEFAULT_CAPTURED_BATCH,
        help="Workspace-local captured batch pt (default: workload/profile/captured_batch.pt).",
    )
    return p.parse_args()


def timed(name, events, fn, *args, **kwargs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn(*args, **kwargs)
    end.record()
    events.append((name, start, end))
    return result


def main():
    args = _parse_args()
    if not args.checkpoint_root:
        raise SystemExit("--checkpoint-root or CHECKPOINT_ROOT env var is required")
    device = "cuda:0"
    torch.cuda.set_device(device)
    _seed = int(os.environ.get("SEED", "1234"))
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)

    params = load_megatron_checkpoint(args.checkpoint_root, device=device)
    t_names = trainable_param_names()
    opt_state = AdamState(t_names, params, device=device)
    rope_freqs = precompute_rope_freqs(config.MAX_SEQ_LENGTH, device=device)

    if not os.path.exists(args.captured_batch):
        rel = os.path.relpath(args.captured_batch, _REPO_ROOT)
        sys.stderr.write(
            f"\nERROR: captured batch not found at {args.captured_batch}\n"
            f"  This file must live inside the workspace at {rel}.\n"
            f"  Generate it once with:\n"
            f"    python tools/capture_profile_batch.py\n"
            f"  (or override path with --captured-batch <path>).\n\n"
        )
        sys.exit(2)

    batch_data = torch.load(args.captured_batch, map_location="cpu", weights_only=False)  # nosec B614 — checkpoint payload contains non-tensor state (config/optim); local file
    input_ids = batch_data["tokens"].to(device)
    labels = batch_data["labels"].to(device)
    loss_mask = batch_data["loss_mask"].float().to(device)

    # Warmup
    for _ in range(2):
        logits, saved = forward_pass_with_save(params, input_ids, rope_freqs)
        if config.FUSED_OPS:
            from training_engine_tensor.triton_kernels import fused_cross_entropy_fwd_bwd

            _loss, _, d_logits = fused_cross_entropy_fwd_bwd(logits, labels, loss_mask)
        else:
            from training_engine_tensor.backward import cross_entropy_loss_backward

            _loss, _, d_logits = cross_entropy_loss_backward(logits, labels, loss_mask)
        grads = backward_pass(d_logits, params, saved, rope_freqs)
        fp32_grads = {n: g.float() if g.dtype != torch.float32 else g for n, g in grads.items()}
        gn = compute_grad_norm_fp32(fp32_grads, t_names)
        clip_gradients_fp32(fp32_grads, t_names, MAX_GRAD_NORM, gn)
        lr = compute_lr(opt_state.num_samples, 16)
        adam_step(opt_state, fp32_grads, lr)
        opt_state.num_samples += 16
        sync_params_from_master(params, opt_state)
        del saved, logits, d_logits, grads, fp32_grads
        torch.cuda.synchronize()

    # Profiled step
    events = []
    torch.cuda.synchronize()

    logits, saved = timed("forward", events, forward_pass_with_save, params, input_ids, rope_freqs)

    if config.FUSED_OPS:
        from training_engine_tensor.triton_kernels import fused_cross_entropy_fwd_bwd

        _loss, _, d_logits = timed(
            "cross_entropy", events, fused_cross_entropy_fwd_bwd, logits, labels, loss_mask
        )
    else:
        from training_engine_tensor.backward import cross_entropy_loss_backward

        _loss, _, d_logits = timed(
            "cross_entropy", events, cross_entropy_loss_backward, logits, labels, loss_mask
        )

    grads = timed("backward", events, backward_pass, d_logits, params, saved, rope_freqs)

    fp32_grads = {n: g.float() if g.dtype != torch.float32 else g for n, g in grads.items()}
    gn = timed("grad_norm", events, compute_grad_norm_fp32, fp32_grads, t_names)
    timed("clip_grads", events, clip_gradients_fp32, fp32_grads, t_names, MAX_GRAD_NORM, gn)
    lr = compute_lr(opt_state.num_samples, 16)
    timed("adam_step", events, adam_step, opt_state, fp32_grads, lr)
    timed("sync_params", events, sync_params_from_master, params, opt_state)

    torch.cuda.synchronize()

    print(f"{'Operation':<20} {'Time (ms)':>10}")
    print("=" * 32)
    total = 0
    for name, start, end in events:
        ms = start.elapsed_time(end)
        total += ms
        print(f"{name:<20} {ms:>10.2f}")
    print("=" * 32)
    print(f"{'Total':<20} {total:>10.2f}")


if __name__ == "__main__":
    main()
