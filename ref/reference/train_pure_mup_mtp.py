"""Pure PyTorch training of MiniCPM4 0.5B with muP + MTP.

DP-only training entry. No Megatron, no TransformerEngine, no fused
attention kernels — just nn.Linear + scaled_dot_product_attention with
muP scaling and an optional Eagle-style MTP head.

Per-step progress is printed to stdout in two forms: a human-readable
``iteration …`` line and a machine-readable
``[LOSS] step=N global_loss=… grad_norm=… time_s=… mfu_e2e_standard=…``
line (also mirrored to ``$LOSS_DUMP_FILE`` when that env var is set).

Determinism (bit-wise reproducibility)
--------------------------------------
This entry enables a full bit-wise determinism stack by default
(``--no-deterministic`` to disable). The stack is:

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # before CUDA init
    random.seed / np.random.seed / torch.manual_seed / cuda.manual_seed_all
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

Notes:
- ``warn_only=False`` is required; ``warn_only=True`` is not enough
  because ``F.scaled_dot_product_attention`` backward otherwise stays on
  a non-deterministic path.
- Disabling flash / mem-efficient SDPA falls back to the math backend
  (5–10× slower on attention; overall step typically 1.3–2× slower).
- Pass ``--no-deterministic`` for normal throughput-oriented runs; keep
  it on when debugging numerical regressions or proving bit-wise
  equivalence.
"""
from __future__ import annotations

import argparse
import atexit
import gc
import math
import os
import queue
import random
import threading
import time
from pathlib import Path

# Must be set before CUDA initializes anywhere in the process; harmless
# when determinism is later disabled at runtime.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from model_pure_mup_mtp import (
    FFN_HIDDEN_SIZE,
    HEAD_DIM,
    HIDDEN_SIZE,
    MAX_SEQ_LEN,
    MiniCPM4MupMtp,
    NUM_HEADS,
    NUM_KV_HEADS,
    NUM_LAYERS,
    VOCAB_SIZE,
    precompute_rope_freqs,
)

# ── Capture wiring (harness_dp) ──────────────────────────────────────
# The reference carries its own M1–M5 capture call sites via harness_dp
# (install + begin_step + reduce_loss_scalar + reduce_grads + capture),
# replacing the runtime source-patching that ref/bridges/interposer.py used to
# do. Bootstrap the harness root onto sys.path so ``evals`` is importable when
# the launcher runs this entry directly (torchrun only puts ref/reference on
# the path).
import sys as _sys

_HARNESS_ROOT = str(Path(__file__).resolve().parents[2])
if _HARNESS_ROOT not in _sys.path:
    _sys.path.insert(0, _HARNESS_ROOT)
from evals import harness_dp

# H100 SXM5 BF16 peak throughput (dense), used as the MFU denominator.
# `mfu_e2e_standard` = achieved_flops / (peak * world_size) * 100, emitted on
# the wire already in 0–100 percentage scale (the harness gate compares it
# directly without re-scaling). Override via the env var so other deployments
# (PCIe / non-H100) can recompute against their own peak without forking.
H100_BF16_PEAK_FLOPS = 989.4e12


def enable_determinism(seed: int) -> None:
    """Apply the full bit-wise determinism stack. See module docstring."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)


# ── LR schedule (WSD) ───────────────────────────────────────────────────

def compute_lr(step: int, max_lr: float, min_lr: float,
               warmup: int, decay: int, wsd_decay: int) -> float:
    if step < warmup:
        return max_lr * step / warmup
    if step > decay:
        return min_lr
    wsd_anneal_start = decay - wsd_decay
    if step <= wsd_anneal_start:
        return max_lr
    wsd_steps = step - wsd_anneal_start
    wsd_ratio = float(wsd_steps) / float(wsd_decay)
    coeff = 2.0 * math.pow(0.5, wsd_ratio) - 1.0
    return min_lr + coeff * (max_lr - min_lr)


# ── Dataloader ──────────────────────────────────────────────────────────

_MEGATRON_IDX_HEADER = b"MMIDIDX\x00\x00"
_MEGATRON_DTYPE_MAP = {
    1: np.uint8, 2: np.int8, 3: np.int16,
    4: np.int32, 5: np.int64, 6: np.float64,
    7: np.float32, 8: np.uint16,
}


class MegatronBinaryDataloader:
    """Lightweight reader for Megatron indexed binary datasets (.bin/.idx).

    Pure struct + numpy + torch.tensor construction — no megatron import.

    Scope: GSM8K dev/CI convenience only. Exists so single-shard
    Megatron-binary runs can proceed without depending on
    ``modelbest_sdk`` being installed in the dev/CI environment.
    Lacks SSTable's weighted sampling, segment/packer abstractions,
    and resumable cursor state, so it is NOT a substitute for
    ``ModelbestDataloader`` on the multi-shard production path.
    """

    def __init__(self, path_prefix: str, dp_rank: int, world_size: int,
                 micro_batch_size: int, seq_length: int, seed: int):
        import struct

        self._mbs = micro_batch_size
        self._seq_length = seq_length

        # Parse .idx
        idx_path = path_prefix + ".idx"
        with open(idx_path, "rb") as f:
            magic = f.read(9)
            if magic != _MEGATRON_IDX_HEADER:
                raise ValueError(f"Invalid .idx magic: {magic!r}")
            _version = struct.unpack("<q", f.read(8))[0]
            dtype_code = struct.unpack("<B", f.read(1))[0]
            self._num_sequences = struct.unpack("<q", f.read(8))[0]
            self._num_documents = struct.unpack("<q", f.read(8))[0]
            sizes = np.frombuffer(
                f.read(self._num_sequences * 4), dtype=np.int32
            ).copy()
            pointers = np.frombuffer(
                f.read(self._num_sequences * 8), dtype=np.int64
            ).copy()

        # mmap .bin
        dtype = _MEGATRON_DTYPE_MAP[dtype_code]
        bin_path = path_prefix + ".bin"
        self._bin_data = np.memmap(bin_path, dtype=dtype, mode="r")

        # Concatenate all sequences into a flat token stream
        token_list = []
        for i in range(self._num_sequences):
            byte_offset = pointers[i]
            elem_offset = byte_offset // np.dtype(dtype).itemsize
            length = int(sizes[i])
            token_list.append(
                self._bin_data[elem_offset:elem_offset + length].astype(np.int64)
            )
        flat_tokens = np.concatenate(token_list)

        # Slice into non-overlapping windows of (seq_length + 1)
        window_size = seq_length + 1
        self._total_windows = len(flat_tokens) // window_size
        usable = self._total_windows * window_size
        self._windows = flat_tokens[:usable].reshape(self._total_windows, window_size)

        # Distributed sharding: shared shuffle then round-robin split
        all_indices = np.arange(self._total_windows)
        rng = np.random.RandomState(seed)
        rng.shuffle(all_indices)
        local_indices = all_indices[dp_rank::world_size]
        self._local_windows = self._windows[local_indices]
        self._num_local_windows = len(local_indices)

        self._seed = seed
        self._dp_rank = dp_rank
        self._world_size = world_size
        self._epoch = 0
        self._pos = 0

    def _reshuffle(self):
        self._epoch += 1
        rng = np.random.RandomState(
            self._seed + self._dp_rank + self._world_size + self._epoch * 1000
        )
        perm = rng.permutation(self._num_local_windows)
        self._local_windows = self._local_windows[perm]
        self._pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._num_local_windows == 0:
            raise StopIteration
        batch_windows = []
        for _ in range(self._mbs):
            if self._pos >= self._num_local_windows:
                self._reshuffle()
            batch_windows.append(self._local_windows[self._pos])
            self._pos += 1
        batch = np.stack(batch_windows)
        tokens = torch.from_numpy(batch[:, :-1].copy()).long()
        labels = torch.from_numpy(batch[:, 1:].copy()).long()
        loss_mask = torch.ones(tokens.shape, dtype=torch.float32)
        return {"tokens": tokens, "labels": labels, "loss_mask": loss_mask}


def _build_megatron_binary_dataloader(path_prefix, dp_rank, world_size,
                                       micro_batch_size, seq_length, seed):
    return MegatronBinaryDataloader(
        path_prefix, dp_rank, world_size,
        micro_batch_size, seq_length, seed,
    )


def _build_sstable_dataloader(weights_and_paths, dp_rank, world_size,
                               micro_batch_size, seq_length, seed):
    from modelbest_sdk.dataset.batch_packer.batch_packer_factory import MEGATRON_BATCH_PACKER
    from modelbest_sdk.dataset.modelbest_dataloader import ModelbestDataloader
    from modelbest_sdk.dataset.sampler.sampler_factory import WEIGHTED_MEGATRON_SAMPLER
    from modelbest_sdk.dataset.segment.segment_factory import CONDITIONAL_FIXED_LENGTH_SEGMENT
    from modelbest_sdk.dataset.thrift_wrapper.dataset_checkpoint import (
        DatasetInfo, DatasetInfoList,
    )
    from modelbest_sdk.dataset.thrift_wrapper.dataset_context import DatasetContext

    total_w = sum(w for w, _ in weights_and_paths)
    ds_info = DatasetInfoList([
        DatasetInfo(path=p, weight=w / total_w) for w, p in weights_and_paths
    ])
    ctx = DatasetContext(
        rank=dp_rank, world_size=world_size,
        tp_rank=0, tp_size=1, pp_rank=0, pp_size=1,
        num_workers=2, dataset_config_path="",
        dataset_checkpoint_path="",
        seed=seed + dp_rank + world_size,
    )
    return ModelbestDataloader(
        ctx, ds_info,
        batch_size=micro_batch_size, max_len=seq_length, chunk_size=64,
        cuda_prefetch=False,
        segment_type=CONDITIONAL_FIXED_LENGTH_SEGMENT,
        sampler_type=WEIGHTED_MEGATRON_SAMPLER,
        batch_packer_type=MEGATRON_BATCH_PACKER,
        dp_group=dist.group.WORLD if world_size > 1 else None,
        clear_sampler_state=False,
    )


def data_loader_from_config(data_config: str) -> str:
    """Read the loader kind from the data-value SSOT (config/data.toml).

    ``data_config`` is the FORGE_DATA_TOML path the launcher threads in via
    ``--data-config``. The loader kind lives in ``[data].data_loader`` — the
    single source every ref (this module + its sister train scripts), the
    in-house engine, and the meta DP×TP bundle (``ref_bundle_run.sh`` →
    ``train.py``) read. The sister refs import this helper alongside
    ``build_dataloader`` so all torch refs select the loader from one source,
    never from a ``DATA_LOADER`` env value. Empty string when no data axis is
    provided (standalone dev runs).
    """
    if not data_config:
        return ""
    import tomllib
    with open(data_config, "rb") as fh:
        doc = tomllib.load(fh)
    return str(doc.get("data", {}).get("data_loader", "")).strip()


def build_dataloader(data_path_args, dp_rank, world_size,
                     micro_batch_size, seq_length, seed, loader=""):
    # Dispatch on the EXPLICIT loader kind the selected data axis declares in
    # config/data.toml [data].data_loader ∈ {megatron_binary, modelbest, hf};
    # main() reads it from --data-config (FORGE_DATA_TOML). The loader is chosen
    # by which data.toml the user picked — NOT by sniffing the path format — so
    # the data path can later be provided uniformly without changing routing.
    weights_and_paths = [
        (float(data_path_args[i]), data_path_args[i + 1].strip())
        for i in range(0, len(data_path_args), 2)
    ]

    kind = (loader or "").strip().lower()
    if kind == "hf":
        import hf_stream_dataloader
        return hf_stream_dataloader.build(
            weights_and_paths, dp_rank, world_size,
            micro_batch_size, seq_length, seed,
        )
    if kind == "modelbest":
        return _build_sstable_dataloader(
            weights_and_paths, dp_rank, world_size,
            micro_batch_size, seq_length, seed,
        )
    if kind == "megatron_binary":
        _, path_prefix = weights_and_paths[0]
        return _build_megatron_binary_dataloader(
            path_prefix, dp_rank, world_size,
            micro_batch_size, seq_length, seed,
        )
    raise ValueError(
        f"data_loader={kind!r} is empty or unknown; config/data.toml must set "
        f"[data].data_loader to one of: megatron_binary, modelbest, hf"
    )


def next_batch(dl, device):
    data = next(dl)
    while (data["loss_mask"] == 0).all().item():
        data = next(dl)
    if hasattr(dl, "update"):
        dl.update(data.get("indexes"), data.get("last_sample"))
    return (
        data["tokens"].to(device),
        data["labels"].to(device),
        data["loss_mask"].float().to(device),
    )


class PrefetchedBatcher:
    """Background-thread prefetch + pinned non-blocking H2D for next_batch().

    Wraps any of the build_dataloader backends (HFStreamDataloader,
    MegatronBinaryDataloader, ModelbestDataloader) with one worker thread
    that runs the original next_batch() body: skip zero-loss-mask
    micro-batches, advance the Modelbest cursor when present, and pin to
    host memory. The main thread pulls from a bounded queue and issues
    non_blocking H2D copies, which overlap with the previous step's GPU
    compute on the default stream.

    Bitwise: preserves dl.__next__() order. The skip-mask loop is consumed
    in the worker in the same order the inline next_batch() consumed it,
    so per-(rank, mb_index) (tokens, labels, loss_mask) is unchanged.

    NOTE: sentencepiece >= 0.2.0 is thread-safe (C++ lock). For older
    versions instantiate one tokenizer per thread inside the HF
    dataloader's _token_stream — see hf_stream_dataloader.py.
    """

    def __init__(self, dl, device, queue_size: int = 2):
        self._dl = dl
        self._device = device
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        # Teardown control: the worker calls ``.pin_memory()`` (a CUDA
        # host-alloc) every iteration, so a daemon left spinning when the
        # interpreter releases the CUDA context is the SIGABRT
        # ("terminate called without an active exception") source. ``close()``
        # sets this event and joins the worker before context teardown. The
        # event is never set during a normal run, so the per-(rank, mb_index)
        # data order below is bitwise-unchanged.
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._worker, daemon=True)
        self._t.start()
        # Safety net for the M1 capture path: the harness hook hijacks
        # ``optimizer.step`` and exits via ``raise SystemExit(0)`` from inside
        # the first step, so ``main()``'s post-loop ``close()`` is skipped and
        # this daemon would otherwise still be spinning (touching CUDA) at
        # interpreter shutdown. Registering ``close`` with atexit stops the
        # worker on that path too. Idempotent with the explicit close().
        atexit.register(self.close)

    def _worker(self):
        try:
            while not self._stop.is_set():
                data = next(self._dl)
                if (data["loss_mask"] == 0).all().item():
                    continue
                if hasattr(self._dl, "update"):
                    self._dl.update(data.get("indexes"), data.get("last_sample"))
                item = (
                    data["tokens"].pin_memory(),
                    data["labels"].pin_memory(),
                    data["loss_mask"].float().pin_memory(),
                )
                # Stop-aware put: a bounded queue would otherwise block the
                # worker forever on a full queue during close(), so poll the
                # stop flag between attempts and drop the pinned item on stop.
                while not self._stop.is_set():
                    try:
                        self._q.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except StopIteration:
            try:
                self._q.put(None, timeout=0.1)
            except queue.Full:
                pass

    def __call__(self):
        batch = self._q.get()
        if batch is None:
            raise StopIteration
        tokens, labels, loss_mask = batch
        return (
            tokens.to(self._device, non_blocking=True),
            labels.to(self._device, non_blocking=True),
            loss_mask.to(self._device, non_blocking=True),
        )

    def close(self, timeout: float = 5.0):
        """Stop the prefetch worker so no thread touches CUDA at shutdown.

        Idempotent. Sets the stop flag, drains the queue to unblock a
        worker parked on a full ``put()``, then joins with a bounded
        timeout so a wedged worker surfaces as a hang rather than an
        indefinite block. After the worker joins, also close the inner
        dataloader so its pyarrow generator chain is dropped BEFORE
        ``Py_Finalize`` — on the HPCX/UCX image, leaving pyarrow buffers
        to the finalize GC pass races ``libucm.so`` unload (registered
        malloc-hook) and yields ``terminate called without an active
        exception`` → SIGABRT(-6). The M1 capture path raises
        ``SystemExit`` from inside ``optimizer.step``, so ``main``'s
        post-loop ``dl.close()`` never runs; this atexit-reachable close
        is the only safe site.
        """
        self._stop.set()
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        self._t.join(timeout=timeout)
        inner_close = getattr(self._dl, "close", None)
        if callable(inner_close):
            try:
                inner_close()
            except Exception:
                pass


def build_mtp_tensors(input_ids, labels, loss_mask):
    """Construct (mtp_input_ids, mtp_labels, mtp_loss_mask) by shifting.

    For MTP we predict the t+2 token from position t, conditioned on the
    main hidden at position t and the embedding of token t+1.
        mtp_input_ids[t]  = labels[t]            (= input_ids[t+1])
        mtp_labels[t]     = labels[t+1]
        mtp_loss_mask[t]  = loss_mask[t] * loss_mask[t+1]
    The very last position has no t+2 token; we mask it out.
    """
    mtp_input_ids = labels.clone()
    mtp_labels = torch.zeros_like(labels)
    mtp_labels[:, :-1] = labels[:, 1:]
    mtp_loss_mask = torch.zeros_like(loss_mask)
    mtp_loss_mask[:, :-1] = loss_mask[:, :-1] * loss_mask[:, 1:]
    return mtp_input_ids, mtp_labels, mtp_loss_mask


# ── Loss helpers ────────────────────────────────────────────────────────

def masked_ce(logits, labels, mask):
    """Returns (sum_loss [fp32], num_tokens [fp32])."""
    B, S, V = logits.shape
    nll = F.cross_entropy(logits.reshape(-1, V).float(),
                          labels.reshape(-1), reduction="none")
    m = mask.reshape(-1).float()
    return (nll * m).sum(), m.sum()


# ── Param accounting (for MFU) ──────────────────────────────────────────

def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


# ── Main ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path-file", required=True)
    p.add_argument("--data-config", default="",
                   help="Path to the data-value SSOT (config/data.toml, pointed "
                        "at by FORGE_DATA_TOML). The [data].data_loader key selects "
                        "the loader kind (hf / modelbest / megatron_binary). Empty "
                        "only for standalone dev invocations with no data axis.")
    p.add_argument("--train-iters", type=int, default=1000)
    p.add_argument("--micro-batch-size", type=int, default=10)
    p.add_argument("--global-batch-size", type=int, default=1280)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--lr-warmup-iters", type=int, default=2000)
    p.add_argument("--lr-decay-iters", type=int, default=150000)
    p.add_argument("--lr-wsd-decay-iters", type=int, default=0)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.95)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--init-method-std", type=float, default=0.1)
    # No default: every caller must declare which init regime they want.
    # Production launchers / bootstrap pass --init-ones 1 (textbook ones
    # init for RMSNorm); bitwise gates M1-M5 pass --init-ones 0 (anti-cheat
    # 0.97 fill — see MiniCPM4MupMtp.init_weights docstring). Making this
    # required prevents accidental ones-init in a bitwise-gate context.
    p.add_argument("--init-ones", type=int, choices=[0, 1], required=True,
                   help="Required. 1 = standard ones init (M6/M7/production), "
                        "0 = anti-cheat 0.97 init (M1-M5 bitwise gates). See "
                        "model_pure_mup_mtp.MiniCPM4MupMtp.init_weights "
                        "docstring for the engine contract.")
    p.add_argument("--mup-base-hidden-size", type=int, default=256)
    p.add_argument("--mup-emb-scale", type=float, default=12.0)
    p.add_argument("--mup-depth-scale", type=float, default=1.4)
    p.add_argument("--eagle-num-layers", type=int, default=1)
    p.add_argument("--eagle-ce-loss-weight", type=float, default=0.3)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--recompute-num-layers", type=int, default=8,
                   help="Number of layers to recompute (-1 = all 24 layers when --recompute is set; 8 = only last 8 layers, default)")
    p.add_argument("--deterministic", dest="deterministic",
                   action="store_true", default=True,
                   help="Enable full bit-wise determinism stack (default).")
    p.add_argument("--no-deterministic", dest="deterministic",
                   action="store_false",
                   help="Disable determinism for throughput-oriented runs.")
    harness_dp.add_capture_cli_args(p)
    args = p.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    if args.deterministic:
        enable_determinism(args.seed)
        if rank == 0:
            print("  determinism: ON (use --no-deterministic to disable)")
    elif rank == 0:
        print("  determinism: OFF")

    grad_accum_steps = args.global_batch_size // (args.micro_batch_size * world_size)
    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"=== Pure PyTorch (no TE) MiniCPM4 0.5B + muP + MTP ===")
        print(f"  world_size={world_size}, grad_accum={grad_accum_steps}")
        print(f"  lr={args.lr}, mup_emb_scale={args.mup_emb_scale}, "
              f"mup_depth_scale={args.mup_depth_scale}, base_hidden={args.mup_base_hidden_size}")
        print(f"  init_std={args.init_method_std}, eagle_num_layers={args.eagle_num_layers}, "
              f"ce_weight={args.eagle_ce_loss_weight}")
        print(f"  init_ones={args.init_ones} "
              f"({'ones (production)' if args.init_ones else 'anti-cheat 0.97 (M1-M5)'})")

    # ── Model ────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    model = MiniCPM4MupMtp(
        mup_base_hidden_size=args.mup_base_hidden_size,
        mup_emb_scale=args.mup_emb_scale,
        mup_depth_scale=args.mup_depth_scale,
        eagle_num_layers=args.eagle_num_layers,
    )
    model.init_weights(init_std=args.init_method_std, seed=args.seed,
                       init_ones=bool(args.init_ones))

    model = model.to(device=device, dtype=torch.bfloat16)
    if world_size > 1:
        for p_ in model.parameters():
            dist.broadcast(p_.data, src=0)

    rope_freqs = precompute_rope_freqs(MAX_SEQ_LEN, device=device)

    # Count params for MFU (trainable, includes embedding + eagle layer).
    num_params = count_trainable_params(model)

    # ── FP32 master + AdamW with muP lr groups × wd-by-dim ──────────
    # Megatron-aligned wd policy (see
    # prompt/develop_prompt/megatron/reference/megatron_bitwise_aligenment_textbook.md
    # §4.6): params with dim >= 2 use wd = args.weight_decay; params with
    # dim < 2 (LayerNorm/RMSNorm.weight, biases) use wd = 0. Cross with
    # the muP lr-mult split (matrix weights -> lr/width_mult, rest -> base_lr)
    # so we end up with up to four optim groups.
    bf16_params = [p_ for p_ in model.parameters() if p_.requires_grad]
    fp32_master = [p_.detach().float().clone().requires_grad_(True) for p_ in bf16_params]
    bf16_to_master = {id(b): m for b, m in zip(bf16_params, fp32_master)}
    mup_groups = model.mup_lr_groups(args.lr)
    optim_groups = []
    for g in mup_groups:
        wd_params, no_wd_params = [], []
        for p_ in g["params"]:
            master = bf16_to_master[id(p_)]
            (wd_params if master.dim() >= 2 else no_wd_params).append(master)
        if wd_params:
            optim_groups.append(
                {"params": wd_params, "lr": g["lr"], "weight_decay": args.weight_decay}
            )
        if no_wd_params:
            optim_groups.append(
                {"params": no_wd_params, "lr": g["lr"], "weight_decay": 0.0}
            )

    optim = torch.optim.AdamW(
        optim_groups,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=1e-8, weight_decay=args.weight_decay,
        fused=True,
    )

    # Install capture (strict no-op unless --hash-output is set). Single-step
    # M1 mode (persistent=False) hijacks optim.step and dumps on first step;
    # persistent mode returns a live session driven by the begin_step /
    # reduce_* / capture calls below and dumped at process exit.
    harness_dp.install_from_args(model, optim, args)

    base_lr_per_group = [g["lr"] for g in optim_groups]
    lr_mult_per_group = [lr / args.lr for lr in base_lr_per_group]
    wd_per_group = [g["weight_decay"] for g in optim_groups]
    if rank == 0:
        print(
            f"  optim has {len(optim_groups)} groups, "
            f"lr_mults={lr_mult_per_group}, wd={wd_per_group}"
        )
        print(f"  trainable params = {num_params / 1e6:.2f}M")

    # ── Data ─────────────────────────────────────────────────────────
    with open(args.data_path_file) as f:
        raw = f.read().strip().split()
    dl = build_dataloader(
        data_path_args=raw,
        dp_rank=rank, world_size=world_size,
        micro_batch_size=args.micro_batch_size,
        seq_length=MAX_SEQ_LEN, seed=args.seed,
        loader=data_loader_from_config(args.data_config),
    )

    # ── Harness-facing artefacts ─────────────────────────────────────
    loss_dump_path = os.environ.get("LOSS_DUMP_FILE", "")
    loss_dump_fh = None
    if rank == 0 and loss_dump_path:
        # absolute paths are honoured; otherwise resolve under DUMP_DIR
        path_obj = Path(loss_dump_path)
        if not path_obj.is_absolute():
            base = Path(os.environ.get("DUMP_DIR", args.out_dir))
            path_obj = base / loss_dump_path
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        loss_dump_fh = path_obj.open("w", encoding="utf-8", buffering=1)

    # ── Training loop ────────────────────────────────────────────────
    fp32_grad_bufs = [torch.zeros_like(p_) for p_ in fp32_master]
    use_mtp = (args.eagle_num_layers > 0)
    ce_w = args.eagle_ce_loss_weight
    tokens_per_step = args.global_batch_size * MAX_SEQ_LEN
    # MFU(standard) — physical-GEMM enumeration (closed-form, exact for
    # this fixed MiniCPM4 0.5B + muP + MTP geometry).
    #
    # Each GEMM ``[m, k] · [k, n]`` contributes ``2·m·k·n`` FLOPs per
    # forward pass (FMA = 2 ops). Training adds one dgrad + one wgrad
    # GEMM of the same arithmetic cost as the forward, so ``fwd + bwd``
    # totals ``3 · fwd`` per GEMM.
    #
    # Per-token forward FLOPs (one transformer block):
    #   QKV proj : Q(H→H_q) + K(H→H_kv) + V(H→H_kv) = 2H(H_q + 2 H_kv)
    #   Wo proj  : H_q → H                          = 2 H_q H
    #   attn     : causal Q·Kᵀ + causal attn·V      = 2 S H_q
    #              (each term ``2·S·H_q`` × ½ for the causal mask)
    #   MLP      : SwiGLU = 3 GEMMs of size [H, ffn] (gate / up / down) = 6 H·ffn
    #
    # Per-token forward LM-head GEMM:                                = 2 H V
    #
    # MTP (Eagle) additional (only when eagle_num_layers > 0):
    #   ``eagle_num_layers`` × one transformer block (same shape as main layers)
    #   ``eagle_num_layers`` × Eagle FC (Linear 2H → H, no bias)     = 2·(2H)·H = 4H²
    #   ``eagle_num_layers`` × second LM-head call (weight is tied to the main
    #                                               LM-head but the compute is
    #                                               independent)
    #                                                                = 2 H V
    #
    # The previous ``6·num_params + 12·L·H·S`` heuristic both
    # over-counts attention (it carries no causal ½ factor) and
    # under-counts MTP (``num_params`` only counts the tied LM-head
    # weight once even though it is consumed twice when MTP is on),
    # netting roughly -3% to +5% drift depending on geometry. This
    # explicit form is bitwise equivalent to Megatron-LM v15's
    # ``num_floating_point_operations`` (megatron/training/training.py
    # expansion_factor=12 path + the L438-448 MTP block) for the
    # MiniCPM4 + 1-layer-Eagle geometry.
    H_q = float(NUM_HEADS * HEAD_DIM)
    H_kv = float(NUM_KV_HEADS * HEAD_DIM)
    H = float(HIDDEN_SIZE)
    S = float(MAX_SEQ_LEN)
    ffn = float(FFN_HIDDEN_SIZE)
    V = float(VOCAB_SIZE)
    fwd_attn_proj_per_layer = 2.0 * H * (H_q + 2.0 * H_kv) + 2.0 * H_q * H
    fwd_attn_score_per_layer = 2.0 * S * H_q
    fwd_mlp_per_layer = 6.0 * H * ffn
    fwd_per_layer = (
        fwd_attn_proj_per_layer + fwd_attn_score_per_layer + fwd_mlp_per_layer
    )
    fwd_lm_head = 2.0 * H * V
    n_mtp = float(args.eagle_num_layers)
    fwd_mtp = n_mtp * (
        fwd_per_layer
        + 2.0 * (2.0 * H) * H  # Eagle FC: Linear(2H -> H, no bias)
        + fwd_lm_head           # second LM-head call (tied weight, separate compute)
    )
    fwd_per_token = float(NUM_LAYERS) * fwd_per_layer + fwd_lm_head + fwd_mtp
    train_per_token = 3.0 * fwd_per_token  # fwd + dgrad + wgrad
    flops_per_step = train_per_token * tokens_per_step
    peak_total = H100_BF16_PEAK_FLOPS * max(world_size, 1)

    next_batch_fn = PrefetchedBatcher(dl, device)

    for step in range(args.train_iters):
        harness_dp.begin_step(step)
        torch.cuda.synchronize()
        t0 = time.time()

        for buf in fp32_grad_bufs:
            buf.zero_()
        local_lm_sum = torch.zeros(1, device=device, dtype=torch.float64)
        local_lm_n   = torch.zeros(1, device=device, dtype=torch.float64)
        local_mtp_sum = torch.zeros(1, device=device, dtype=torch.float64)
        local_mtp_n   = torch.zeros(1, device=device, dtype=torch.float64)

        for _mb in range(grad_accum_steps):
            harness_dp.begin_microbatch(_mb)
            input_ids, labels, loss_mask = next_batch_fn()
            for p_ in bf16_params:
                p_.grad = None
                # The reference accumulates wgrad in fp32 directly into
                # ``main_grad`` (Megatron-style). Reset it per microbatch
                # so each backward starts from zero; the custom autograd
                # Functions re-create it lazily on first wgrad.
                if getattr(p_, "main_grad", None) is not None:
                    p_.main_grad = None

            if use_mtp:
                mtp_in, mtp_lab, mtp_mask = build_mtp_tensors(input_ids, labels, loss_mask)
                logits, logits_mtp = model(input_ids, rope_freqs,
                                          mtp_input_ids=mtp_in, recompute=args.recompute,
                                          recompute_num_layers=args.recompute_num_layers)
                if harness_dp.capturing():
                    _B, _S, _V = logits.shape
                    _nll = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, _V).float(), labels.reshape(-1), reduction='none'
                    ).reshape(_B, _S)
                    harness_dp.capture('loss.per_token.preallreduce', _nll)
                lm_sum, lm_n = masked_ce(logits, labels, loss_mask)
                mtp_sum, mtp_n = masked_ce(logits_mtp, mtp_lab, mtp_mask)
                obj = lm_sum + ce_w * mtp_sum
                local_mtp_sum += mtp_sum.detach().double()
                local_mtp_n += mtp_n.detach().double()
            else:
                logits, _ = model(input_ids, rope_freqs, recompute=args.recompute,
                                  recompute_num_layers=args.recompute_num_layers)
                if harness_dp.capturing():
                    _B, _S, _V = logits.shape
                    _nll = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, _V).float(), labels.reshape(-1), reduction='none'
                    ).reshape(_B, _S)
                    harness_dp.capture('loss.per_token.preallreduce', _nll)
                lm_sum, lm_n = masked_ce(logits, labels, loss_mask)
                obj = lm_sum

            obj.backward()
            local_lm_sum += lm_sum.detach().double()
            local_lm_n += lm_n.detach().double()
            with torch.no_grad():
                for p_bf16, buf in zip(bf16_params, fp32_grad_bufs):
                    # ``main_grad`` (fp32, Megatron-style) is authoritative
                    # when present; fall back to bf16 ``.grad`` otherwise.
                    g = getattr(p_bf16, "main_grad", None)
                    if g is None:
                        g = p_bf16.grad
                    if g is not None:
                        buf.add_(g)
        harness_dp.end_microbatch()

        reported_lm, g_lm_n, _mtp_reduced = harness_dp.reduce_loss_scalar(
            local_lm_sum, local_lm_n,
            aux=torch.cat([local_mtp_sum, local_mtp_n]),
        )
        g_mtp_sum, g_mtp_n = _mtp_reduced[0].item(), _mtp_reduced[1].item()
        local_lm = (local_lm_sum / local_lm_n.clamp(min=1.0)).item()
        reported_mtp = (g_mtp_sum / max(g_mtp_n, 1.0)) if use_mtp else 0.0

        norm_factor = 1.0 / max(g_lm_n, 1.0)
        harness_dp.reduce_grads(bf16_params, fp32_grad_bufs, norm_factor=norm_factor)

        for p_fp32, buf in zip(fp32_master, fp32_grad_bufs):
            p_fp32.grad = buf

        lr = compute_lr(step + 1, args.lr, args.min_lr,
                        args.lr_warmup_iters, args.lr_decay_iters, args.lr_wsd_decay_iters)
        for pg, mult in zip(optim.param_groups, lr_mult_per_group):
            pg["lr"] = lr * mult

        grad_norm = torch.nn.utils.clip_grad_norm_(fp32_master, args.clip_grad).item()
        optim.step()

        with torch.no_grad():
            for p_bf16, p_fp32 in zip(bf16_params, fp32_master):
                p_bf16.data.copy_(p_fp32.data)

        torch.cuda.synchronize()
        step_time = time.time() - t0
        total = reported_lm + ce_w * reported_mtp if use_mtp else reported_lm
        mfu = flops_per_step / (step_time * peak_total) * 100.0 if step_time > 0 else 0.0

        if rank == 0 and (step + 1) % args.log_interval == 0:
            mtp_str = f" | mtp_loss: {reported_mtp:.4e}" if use_mtp else ""
            # Human-readable progress line (kept for ad-hoc tailing).
            print(
                f" iteration {step + 1:6d} | total_loss: {total:.4e} | "
                f"lm_loss: {reported_lm:.4e}{mtp_str} | "
                f"local_lm: {local_lm:.4e} | grad norm: {grad_norm:.3f} | "
                f"lr: {lr:.6e} | time(ms): {step_time * 1000:.1f}",
                flush=True,
            )
            # Harness-facing wire-format line — MUST stay in sync with
            # ``evals/_common.parse_loss_lines`` and ``tools/ref_script_runner._LOSS_LINE``.
            # Float precision is pinned at ``.9e`` (fp32 round-trip; see
            # ``harness/wire_format.LOSS_FLOAT_FORMAT``). Lower precision
            # collapses sub-print fp32 ULP drift and lets it escape the
            # per-step ``max_abs_diff == 0`` gate. This module is standalone
            # (zero ``harness/`` imports) so the literal is restated here.
            loss_line = (
                f"[LOSS] step={step + 1} global_loss={total:.9e} "
                f"grad_norm={grad_norm:.9e} time_s={step_time:.6f} "
                f"mfu_e2e_standard={mfu:.6f}"
            )
            print(loss_line, flush=True)
            if loss_dump_fh is not None:
                loss_dump_fh.write(loss_line + "\n")
                loss_dump_fh.flush()

    if rank == 0:
        print("=== Training complete ===")
        if loss_dump_fh is not None:
            loss_dump_fh.close()

    # ── Ordered teardown (exit-0 contract) ──────────────────────────────
    # The artifact is already flushed above; everything below only releases
    # resources so the interpreter has nothing to crash on during its
    # destructor pass (the SIGABRT / "terminate called without an active
    # exception" race). Order: drain in-flight GPU work, stop the CUDA-
    # touching prefetch daemon, then coordinate the NCCL release across
    # ranks, then free the allocator cache. This is teardown only — no
    # numerical or training logic lives here.
    torch.cuda.synchronize()
    next_batch_fn.close()
    # Release the underlying streaming dataloader's pyarrow generator chain NOW,
    # while CUDA / UCX are still alive — not at Py_Finalize. On the HPCX/UCX image
    # a pyarrow reader collected during the post-main() finalize GC pass races the
    # UCX malloc-hook unload and aborts the process ("terminate called without an
    # active exception" → SIGABRT, returncode -6; or a finalize hang). Reclaiming
    # it here lets the interpreter finalize cleanly with NO os._exit bypass, so the
    # CUDA context is destroyed normally and no VRAM leaks. Teardown only.
    try:
        _dl_close = getattr(dl, "close", None)
        if callable(_dl_close):
            _dl_close()
    except Exception:
        pass
    try:
        del next_batch_fn, dl
    except Exception:
        pass
    gc.collect()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    torch.cuda.empty_cache()



if __name__ == "__main__":
    main()
