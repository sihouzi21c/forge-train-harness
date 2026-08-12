#!/usr/bin/env python3
"""Capture one input batch and save it inside the workspace.

The profile harness (`tools/profile_step.py`) consumes a small
`captured_batch.pt` containing one micro-batch of {tokens, labels, loss_mask}.
This script materialises that file **inside the workspace** at
    workload/profile/captured_batch.pt
so profiling never depends on an out-of-tree artifact and can be
reproduced from a fresh worktree.  The model checkpoint itself
(multi-GB weights) intentionally stays external — only the small input
batch moves in-tree.

Two source modes (priority order):

  1. ``--source dataloader``  — preferred. Drives Megatron's standard
     ``GPTDataset`` against the HuggingFace gsm8k dataset preprocessed
     into Megatron binary format (see
     ``ref/reference/prepare_gsm8k_data.sh``) for one step and
     captures its output.  Talks to Megatron's native
     ``build_train_valid_test_data_iterators`` + ``pretrain_gpt.get_batch``
     directly so the captured batch matches the SSOT pipeline the L0
     ref script uses; no harness-layer wrapper is involved.  Requires
     a Megatron-LM checkout on ``$MEGATRON_ROOT`` and an active CUDA /
     NCCL environment (this is normally run on the GPU host through
     ``harness run``).

  2. ``--source synthetic``   — deterministic seeded random tokens.
     Useful for first-time bring-up on machines without the Megatron
     toolchain. Same shape / dtype as the real loader; loss_mask is
     all-ones.

Usage examples
--------------

    # On the GPU node, prefer real data:
    python tools/capture_profile_batch.py --source dataloader

    # Bootstrap on any host (no GPU):
    python tools/capture_profile_batch.py --source synthetic \
        --batch-size 4 --seq-len 4096

    # Custom output path:
    python tools/capture_profile_batch.py \
        --output workload/profile/captured_batch_dp4.pt --batch-size 16
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_DEFAULT_OUTPUT = _REPO_ROOT / "workload" / "profile" / "captured_batch.pt"

DEFAULT_BATCH_SIZE = 4
DEFAULT_SEQ_LEN = 4096
DEFAULT_VOCAB_SIZE = int(os.environ.get("VOCAB_SIZE", "73448"))
DEFAULT_SEED = int(os.environ.get("SEED", "1234"))


def _capture_synthetic(batch_size: int, seq_len: int, vocab_size: int, seed: int) -> dict:
    """Deterministic seeded random batch — fixed recipe used by the profile
    path so synthetic captures are reproducible across runs."""
    torch.manual_seed(seed + 42)
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    labels = tokens.clone()
    labels[:, :-1] = tokens[:, 1:]
    labels[:, -1] = tokens[:, 0]
    loss_mask = torch.ones(batch_size, seq_len, dtype=torch.float32)
    return {"tokens": tokens, "labels": labels, "loss_mask": loss_mask}


def _capture_from_dataloader(batch_size: int, seq_len: int, data_path: str) -> dict:
    """Drive Megatron's GPTDataset for one step.

    Calls Megatron's native ``build_train_valid_test_data_iterators``
    plus ``pretrain_gpt.get_batch`` directly — the same APIs the L0
    ref script uses through ``pretrain_gpt.py`` — so the captured
    batch matches what the engine sees at runtime when its own
    dataloader (inside ``training_engine_tensor``) yields its first
    micro-batch.  No harness-layer wrapper is involved.  Imports are
    deferred because Megatron / PyTorch distributed init is only
    available on the GPU host.
    """
    megatron_root = os.environ.get("MEGATRON_ROOT") or os.environ.get("FORGE_MEGATRON_ROOT")
    if not megatron_root:
        raise SystemExit(
            "ERROR: --source dataloader requires MEGATRON_ROOT (or "
            "FORGE_MEGATRON_ROOT) so the GPTDataset builder can find "
            "Megatron-LM.  Either run on the GPU host with the env set "
            "or use --source synthetic."
        )

    tokenizer_model = os.environ.get("TOKENIZER_MODEL")
    if not tokenizer_model:
        raise SystemExit(
            "ERROR: --source dataloader requires TOKENIZER_MODEL pointing "
            "at the Llama2 SentencePiece tokenizer.model used to "
            "preprocess the gsm8k binary.  See "
            "ref/reference/prepare_gsm8k_data.sh and the harness "
            "ref-as-gate env contract."
        )

    sys.path.insert(0, megatron_root)
    sys.path.insert(0, str(_REPO_ROOT))

    sys.argv = [
        sys.argv[0],
        "--micro-batch-size",
        str(batch_size),
        "--global-batch-size",
        str(batch_size),
        "--seq-length",
        str(seq_len),
        "--max-position-embeddings",
        str(seq_len),
        "--num-layers",
        "1",
        "--hidden-size",
        "8",
        "--num-attention-heads",
        "1",
        "--vocab-size",
        str(DEFAULT_VOCAB_SIZE),
        "--padded-vocab-size",
        str(DEFAULT_VOCAB_SIZE),
        "--data-path",
        data_path,
        "--split",
        "99990,8,2",
        "--tokenizer-type",
        "Llama2Tokenizer",
        "--tokenizer-model",
        tokenizer_model,
        "--bf16",
        "--seed",
        str(DEFAULT_SEED),
        "--no-save-optim",
        "--no-save-rng",
        "--train-iters",
        "1",
    ]
    try:
        from megatron.training import build_train_valid_test_data_iterators
        from megatron.training.initialize import initialize_megatron
    except ImportError as exc:
        raise SystemExit(
            f"--source dataloader requires Megatron-LM importable from "
            f"$MEGATRON_ROOT={megatron_root}; got: {exc}.  Use --source "
            f"synthetic for hosts without Megatron."
        ) from exc

    initialize_megatron()

    from pretrain_gpt import (
        get_batch as _get_batch,
    )
    from pretrain_gpt import (
        train_valid_test_datasets_provider,
    )

    if hasattr(train_valid_test_datasets_provider, "is_distributed"):
        train_valid_test_datasets_provider.is_distributed = True
    train_iter, _, _ = build_train_valid_test_data_iterators(train_valid_test_datasets_provider)
    tokens, labels, loss_mask, _attention_mask, _position_ids = _get_batch(train_iter)
    return {
        "tokens": tokens.cpu().long(),
        "labels": labels.cpu().long(),
        "loss_mask": loss_mask.cpu().float(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        choices=("dataloader", "synthetic"),
        default="synthetic",
        help="Where to draw the batch from (default: synthetic).",
    )
    p.add_argument(
        "--data-path",
        default=None,
        help=(
            "Megatron --data-path prefix — required when --source "
            "dataloader.  HuggingFace gsm8k dataset preprocessed into "
            "Megatron binary format (see ref/reference/prepare_gsm8k_data.sh)."
        ),
    )
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help="Synthetic vocab size (ignored for dataloader).",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Workspace-local output pt path "
        f"(default: {_DEFAULT_OUTPUT.relative_to(_REPO_ROOT)}).",
    )
    p.add_argument("--force", action="store_true", help="Overwrite even if output already exists.")
    args = p.parse_args()

    out_path = Path(args.output).resolve()

    # Hard guard: the captured batch MUST land inside the workspace. If the
    # caller passed a path outside the repo root we refuse — that would
    # re-introduce the very out-of-tree dependency this script removes.
    try:
        out_path.relative_to(_REPO_ROOT)
    except ValueError:
        sys.exit(
            f"ERROR: --output {out_path} is outside the workspace "
            f"({_REPO_ROOT}). The captured batch must live inside the "
            f"repo (e.g. workload/profile/captured_batch.pt)."
        )

    if out_path.exists() and not args.force:
        sys.exit(
            f"ERROR: {out_path} already exists. Re-run with --force to "
            f"overwrite, or pass --output <new path>."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Capturing batch from source={args.source} (B={args.batch_size}, N={args.seq_len})...")
    if args.source == "synthetic":
        batch = _capture_synthetic(args.batch_size, args.seq_len, args.vocab_size, args.seed)
    else:
        data_path = (
            args.data_path or os.environ.get("DATA_PATH") or os.environ.get("FORGE_DATA_PATH")
        )
        if not data_path:
            sys.exit(
                "ERROR: --source dataloader requires --data-path <prefix> "
                "(or DATA_PATH / FORGE_DATA_PATH env).  See "
                "config/eval.toml [defaults].data_path for the "
                "canonical value."
            )
        batch = _capture_from_dataloader(args.batch_size, args.seq_len, data_path)

    torch.save(batch, str(out_path))
    size_mb = out_path.stat().st_size / 1e6

    print(f"  tokens   : shape={tuple(batch['tokens'].shape)}, dtype={batch['tokens'].dtype}")
    print(f"  labels   : shape={tuple(batch['labels'].shape)}, dtype={batch['labels'].dtype}")
    print(f"  loss_mask: shape={tuple(batch['loss_mask'].shape)}, dtype={batch['loss_mask'].dtype}")
    print(f"Saved {out_path.relative_to(_REPO_ROOT)} ({size_mb:.1f} MB, source={args.source})")


if __name__ == "__main__":
    main()
