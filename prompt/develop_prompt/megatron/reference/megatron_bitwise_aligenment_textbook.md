# Megatron Bitwise Alignment Textbook

> **Scope**: bitwise alignment against **Megatron-LM** (the full TE FusedAdam +
> DistributedOptimizer + multi_tensor_l2norm + reduce_max_stat stack).
>
> **Pure-torch ref does not need any of this**: `torch.optim.AdamW` +
> `torch.nn.utils.clip_grad_norm_` has no fp32 squaring, no implicit fp32 cast,
> and no hidden functions like reduce_max_stat. As long as you match
> "same init + same batch + same optimizer hyperparameters +
> `torch.use_deterministic_algorithms(True)` + cuDNN deterministic" you are
> basically bitwise. The 19 pitfalls listed in §4 are **almost all** Megatron-specific.

---

## 1. Core Concepts

**bitwise equality** = `struct.pack('<f', a) == struct.pack('<f', b)` (fp32).
Note: `fp64=21.381580076148122` and `fp64=21.381580352783203` **are the same bit pattern
`0x41ab0d7a` at fp32 precision**. When you see a ~1e-7 difference, do not conclude
"the error is small" — it is either bitwise or it is not; there is no "close enough".

**Why bitwise is mandatory**: a 1-ULP drift accumulates per step, and after 10 steps will exceed any atol.
bitwise is one-shot, falsifiable, and regression-friendly.

**Environment differences that are inherently not bitwise**: across hardware, across major PyTorch/TE versions,
across DP counts (reduction order changes), across nproc. So bitwise always comes with environment locks.

---

## 2. Bisection Decision Tree

```
1. Pin down scope (forward / forward+backward / N-step trajectory / resume)
2. Confirm the ref itself is reproducible: does it bitwise-match itself on the same input twice?
   If not, fix the ref's own nondeterminism first
3. Add hook anchor points on the ref side (dump intermediate tensors to disk); see §5.4 for the anchors
4. On the candidate side, dump-and-diff each node along the dataflow (§3)
5. Re-test with multiple seeds (≥3) all passing before ship
```

**Discipline**: do not open bitwise-singlecard if alignment has not passed, and do not open bitwise-multicard if bitwise-singlecard has not passed.
"Approximately aligned" makes it impossible to find bugs later.

---

## 3. Precision-Sensitive Nodes in One Training Step

```
STEP N start
 │
 ├─ ①  Load initial weights
 │       master must be a bf16 → fp32 cast (lossy), not the raw fp32  [§4.5]
 │
 ├─ ②  RNG init (PYTHONHASHSEED + manual_seed + cudnn.deterministic
 │      + torch.use_deterministic_algorithms)                          [§4.17/4.18]
 │
 ├─ ③  Forward over micro-batches (embedding → attn → mlp → head)
 │
 ├─ ④  Backward
 │       main_grad must be a pre-allocated fp32 buffer, not .grad.float() [§4.14]
 │
 ├─ ⑤  DP all_reduce(grads)
 │       Even DP=1 goes through NCCL                                   [§4.3]
 │
 ├─ ⑥  grad_norm computation
 │       a. grads_for_norm = filter (exclude shared / no_grad_norm)    [§4.8]
 │       b. multi_tensor_l2norm(grads)                                  [§4.4 / §4.9]
 │       c. (norm ** 2.0) ← fp32 squaring, LOSSY                       [§4.12]
 │       d. all_reduce(SUM) [data_parallel_group]
 │       e. all_reduce(SUM) [grad_stats_parallel_group]                [§4.11]
 │       f. .item() ** 0.5  ← fp64 sqrt
 │       → outputs lossy_grad_norm (used for clip)
 │
 ├─ ⑦  clip_coeff = max_norm / (lossy_grad_norm + 1e-6)                [§4.10]
 │       if clip_coeff < 1.0: multi_tensor_scale(grads, clip_coeff)
 │
 ├─ ⑧  Inner FusedAdam.step
 │       capturable=False                                              [§4.7]
 │       Two parameter groups: 2D wd=0.1 / 1D wd=0.0                   [§4.6]
 │
 ├─ ⑨  master → bf16 model copy
 │
 └─ ⑩  Reporting (does not affect the next training step, only the log)
         reported_grad_norm =
           torch.tensor([lossy], fp32).all_reduce(MAX).item()          [§4.1 / §4.2]
         ↑ the fp32 cast snaps the lossy value back to the raw kernel output
```

**Key insights**:
- The `lossy_grad_norm` computed in ⑥ → used in ⑦ for clip
- The `reported_grad_norm` reported in ⑩ = `lossy` passed through one fp32 cast
- The two **have different numeric values but the same fp32 bit pattern**
- The candidate engine must **compute both values**: use `lossy` for clip, use the cast result for report

---

## 4. Pitfall Checklist (🟥 extremely subtle / 🟧 moderate / 🟨 easy to find / 🟩 obvious)

### §4.1 🟥 Implicit fp32 cast in `reduce_max_stat_across_model_parallel_group`

`megatron/training/utils.py:241`

```python
def reduce_max_stat_across_model_parallel_group(stat: float) -> float:
    stat = torch.tensor([stat], dtype=torch.float32, device=...)  # ←!!
    torch.distributed.all_reduce(stat, op=MAX, group=mpu.get_model_parallel_group())
    return stat.item()
```

The function name "reduce max stat" looks precision-irrelevant, but it is actually an fp64→fp32→fp64 round trip.
**This is the real culprit behind most 1-ULP grad_norm differences**.

### §4.2 🟥 Clip uses lossy / report uses raw — two values

```python
clip_value = _compute_grad_norm_fp32(...)      # lossy fp64
_clip_by(grads, clip_value)                    # ← use lossy
reported = _reported_grad_norm(clip_value)     # ← apply another fp32 cast
log(reported)
```

Forcing the two to match → divergence from step 2 onward.

### §4.3 🟧 `all_reduce` must go through NCCL even at world_size=1

```python
if torch.distributed.is_initialized():
    torch.distributed.all_reduce(total_norm, op=SUM)  # ← invoked on single card too
```

NCCL on a size-1 group still pulls in a CUDA stream sync + internal memcpy. Skipping it = 1-ULP off.

### §4.4 🟧 `multi_tensor_l2norm` is tree+final-sum, not atomicAdd

apex/TE uses "per-block shared-mem tree reduce inside the tile, writes a partial → a single block
accumulates the partials at fixed indices"; nothing is atomic and the result is independent of memory addresses.

The only input factors that affect the output: tensor list order, element count per tensor, element values.

→ Do not bother messing with the shared bucket memory layout; that is not the source of 1-ULP differences.

### §4.5 🟧 master must be cast from bf16 back to fp32 (lossy)

```python
# Wrong (retains 16 extra bits of precision; Adam diverges on step 1):
master = init_weights_pt[name].clone()

# Right (lossy bf16→fp32, matching Megatron):
master = bf16_model_param.detach().clone().float()
```

### §4.6 🟧 Split `weight_decay` into two groups by `param.dim()`

```python
optimizer = FusedAdam([
    {"params": [p for p in main_params if p.dim() >= 2], "weight_decay": 0.1},
    {"params": [p for p in main_params if p.dim() <  2], "weight_decay": 0.0},
], lr=lr, capturable=False)
```

Using a single wd → bias/LN gets an extra decay.

### §4.7 🟧 TE FusedAdam `capturable=False`

`capturable=True` goes through the CUDA-graph-compatible path, which is **a different kernel** from
`False` and produces different results.

### §4.8 🟧 `grads_for_norm` ≠ `model.parameters()`

Exclude parameters marked `param.shared` (to avoid double-counting tied embeddings) and those marked
`no_grad_norm`.

### §4.9 🟥 Bucket physical order ≠ `multi_tensor_l2norm` input list order

**Two different orderings; they must be kept straight when DP>1**:

|   | Order | Who decides it |
|---|---|---|
| **bucket physical layout** (where the reduce-scatter slice lives, where the padding lives) | `reversed(named_parameters)` | `for p in params[::-1]` inside `_ParamAndGradBuffer.__init__` |
| **`multi_tensor_l2norm` input list order** | **wd-first / no-wd-second** (FusedAdam's `param_groups` order) | `get_main_grads_for_grad_norm` walks `get_parameters() → param_groups` |

```python
# Megatron path:
shard_main_param.grad  # data: slices in bucket reverse-named order
get_parameters()       # traversal order: param_groups[0]=wd, param_groups[1]=no-wd
                       # so the list multi_tensor_l2norm sees is wd-first
```

**The correct approach on the candidate side**:

```python
# 1. Compute bucket offsets using reversed-named-parameters order (which determines each param's position in the bucket)
bucket_offsets = compute_bucket_layout(main_params_in_REVERSED_named_order)

# 2. But traverse the fragment list in wd-first / no-wd-second order
for p in main_params_WD_FIRST:        # ← the order Megatron's multi_tensor_l2norm sees
    offset = bucket_offsets[p]         # ← look up the offset from the bucket physical layout
    if rank_range overlaps param[offset]:
        fragment_list.append(slice)
```

**Why this is 🟥 extremely subtle**:

* Without reading the three nested layers `get_main_grads_for_grad_norm` +
  `MegatronOptimizer.get_parameters()` + `_get_param_groups`, you simply cannot tell that list order ≠ bucket order.
* Zero impact when DP=1 (rank 0 sees the whole thing on its own).
* When DP=8 **it depends on the seed**: if a given rank's slice happens to consist entirely of 2D weights (all wd),
  bucket order == wd order — by coincidence no problem; if the slice contains a 1D `layer_norm_weight`,
  the orders diverge.
* **Symptoms**: out of 8 steps, **one specific** `grad_norm` differs by 1 ULP, the rest are bitwise;
  loss is also entirely bitwise (because the 1-ULP clip_coeff difference is absorbed by the bf16 copy →
  the next step's bf16 model is still bitwise → no avalanche). **Distinguishing feature**: changing the seed
  shifts where the drift occurs (seed=1234 all pass; seed=42 step 3 differs; seed=7777 step 6 differs) —
  seed-dependent, single-point drift, loss bitwise — this is virtually a diagnosis of this pitfall.
  Tracking this down in bitwise-multicard in practice took 20+ rounds of dumping.

### §4.10 🟧 Two details of clip

```python
clip_coeff = max_norm / (total_norm + 1.0e-6)   # 1e-6 cannot be omitted (even when the norm is much larger)
if clip_coeff < 1.0:
    multi_tensor_applier(multi_tensor_scale, dummy_buf, [grads, grads], clip_coeff)
    # multi_tensor_scale is required; do not use for g in grads: g.mul_(clip_coeff)
```

### §4.11 🟨 Double all_reduce over two distributed groups

```python
if data_parallel_group is not None:
    torch.distributed.all_reduce(total_norm, op=SUM, group=data_parallel_group)
torch.distributed.all_reduce(total_norm, op=SUM, group=grad_stats_parallel_group)
```

Both are no-ops at TP=1/DP=1, but every NCCL call has a stream-sync side effect. One fewer call = 1-ULP off.

### §4.12 🟨 The `** 2 → reduce → ** 0.5` round trip itself

```python
total_norm = grad_norm ** norm_type              # fp32 ** float = fp32, LOSSY
torch.distributed.all_reduce(total_norm, op=SUM)
return total_norm.item() ** (1.0 / norm_type)    # fp64 sqrt
```

Simplifying to `return grad_norm.item()` skips the lossy path and differs from the ref by 1 ULP.

### §4.13 🟨 `norm_type=float(2.0)` must not be `int`

`tensor ** int_2` and `tensor ** float_2.0` go through different dispatch paths and differ in the lowest bit.

### §4.14 🟨 `main_grad` is a pre-allocated fp32 buffer

```python
# Wrong:
fp32_grad = param.grad.float()               # extra bf16→fp32 cast

# Right:
param.main_grad = torch.zeros(param.shape, dtype=torch.float32, device=...)
# TE fused-wgrad-accum accumulates fp32 directly into main_grad
fp32_grad = param.main_grad
```

### §4.15 🟨 Do not change `multi_tensor_applier` chunk_size

Default 65536; leave it at the apex/TE default. Changing it changes the result.

### §4.16 🟨 Loss reduce order: sum / num_microbatches

```python
averaged = sum_of_losses / num_microbatches     # not mean of means
```

### §4.17 🟩 `PYTHONHASHSEED` must be set before Python starts

```bash
PYTHONHASHSEED=0 python -m harness.cli run ...
```

### §4.18 🟩 model.init RNG consumption order is hard to reproduce

Most reliable: have the ref dump `init_weights.pt` and have the candidate load from that file, **skipping its own init**.

### §4.19 🟩 Use nccl as the `torch.distributed` backend, not gloo

gloo is CPU-side, nccl is GPU-side; the all_reduce result may differ by 1 ULP.

---

## 5. Diagnostic Methodology

### 5.1 Golden rule: dump-and-diff, do not guess

Whenever there is a 1-ULP difference, **the first response is always**: insert dumps at the corresponding intermediate
points on the ref and ours, then compare hex patterns.

**Do not**: guess the cause / change code and try / spend three hours reading Megatron source.
**Do**: dump → compare hex → binary-search down to the first divergence point → then read that part of the source.

### 5.2 Standard Dump Patterns

#### Tensor

```python
def dump_tensor(t, path):
    import torch
    t_cpu = t.detach().to("cpu").contiguous()
    torch.save(t_cpu.clone(), path)
```

#### Scalar (fp64 + hex)

```python
def dump_scalar(value: float, path):
    import struct
    open(path, 'w').write(
        f"{value!r}\n"
        f"hex_fp64: {struct.pack('<d', float(value)).hex()}\n"
        f"hex_fp32: {struct.pack('<f', float(value)).hex()}\n"
    )
```

#### Diff

```python
def diff_dumps(ref_dir, ours_dir):
    import torch
    from pathlib import Path
    for ref_pt in sorted(Path(ref_dir).glob("*.pt")):
        ours_pt = Path(ours_dir) / ref_pt.name
        a = torch.load(ref_pt, map_location='cpu', weights_only=False)
        b = torch.load(ours_pt, map_location='cpu', weights_only=False)
        diff = (a.float() - b.float()).abs().max().item()
        bitwise = bool((a == b).all().item())
        tag = "OK  " if diff == 0 else "DIFF"
        print(f"  {tag}: {ref_pt.name}  max_abs_diff={diff:.4e}  bitwise={bitwise}")
```

### 5.3 Incremental localization (bisect on the dataflow)

When a difference is found at some step → trace back along the dataflow, confirming bitwise at each node
until the first divergence point is found. **Nodes must not be skipped**; skipping them means losing the root cause.

### 5.4 Key Hook Points

| Hook point | How to patch |
|---|---|
| Initial weights | dump `model.state_dict()` after `setup_model_and_optimizer` |
| Per-microbatch input | wrap `get_batch` |
| Raw main_grad | wrap inner optimizer.step (at the start) |
| Clipped grads | wrap inner FusedAdam.step (at the start) |
| After master update | wrap inner FusedAdam.step (at the end) |
| Reported grad_norm | wrap `training_log` |
| Raw l2norm | wrap `clip_grads.multi_tensor_applier` (on the l2_norm_impl path) |
| Final grad_norm | wrap `clip_grads.get_grad_norm_fp32` **and** rebind it in the `optimizer` module |

**Key pitfall**: Megatron submodules use `from .foo import bar` (which copies the binding).
Patching one place is not enough; every module that uses it must be rebound:

```python
from megatron.core.optimizer import clip_grads as _cg
from megatron.core.optimizer import optimizer as _opt_mod
_cg.get_grad_norm_fp32 = wrapped
_opt_mod.get_grad_norm_fp32 = wrapped  # ← do not miss this
```

---

## 6. Code Templates (copy and adapt directly)

### 6.1 grad_norm (two values)

```python
def _compute_grad_norm_fp32(main_params, device) -> float:
    """Returns the LOSSY value (used for clip)."""
    import torch
    from transformer_engine.pytorch.optimizers.multi_tensor_apply import (
        multi_tensor_applier, multi_tensor_l2norm,
    )
    grads = [
        m.grad for m in main_params
        if m.grad is not None
        and not getattr(m, 'shared', False)
        and not getattr(m, 'no_grad_norm', False)
    ]
    if not grads:
        return 0.0
    dummy = torch.zeros(1, dtype=torch.int, device=device)
    grad_norm, _ = multi_tensor_applier(multi_tensor_l2norm, dummy, [grads], False)
    norm_type = 2.0
    total_norm = grad_norm ** norm_type            # LOSSY fp32 squaring
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.SUM)
    return float(total_norm.item()) ** (1.0 / norm_type)


def _reported_grad_norm(lossy: float, device) -> float:
    """Replicates the fp32 cast in reduce_max_stat_across_model_parallel_group."""
    import torch
    stat = torch.tensor([lossy], dtype=torch.float32, device=device)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(stat, op=torch.distributed.ReduceOp.MAX)
    return float(stat.item())
```

### 6.2 fp32 master

```python
def _build_fp32_master(model, init_weights_pt):
    """Returns (main_params, model_params), in reversed order (mimics the Megatron bucket)."""
    import torch
    named_pairs = list(reversed(list(model.named_parameters())))

    # 1. Load init weights (ensure the initialization source matches the ref)
    init_state = torch.load(init_weights_pt, map_location='cpu', weights_only=False)
    with torch.no_grad():
        for name, p in named_pairs:
            p.copy_(init_state[name].to(device=p.device, dtype=p.dtype))

    # 2. Pre-allocate fp32 main_grad
    for _, p in named_pairs:
        p.main_grad = torch.zeros(p.shape, dtype=torch.float32, device=p.device)

    # 3. master = bf16 → fp32 lossy cast
    main_params, model_params = [], []
    for _, p in named_pairs:
        master = p.detach().clone().float()
        master.requires_grad_(True)
        master.shared = getattr(p, 'shared', False)
        master.no_grad_norm = getattr(p, 'no_grad_norm', False)
        main_params.append(master)
        model_params.append(p)
    return main_params, model_params
```

### 6.3 clip + single-step training loop

```python
def _clip_grad(grads, max_norm, total_norm, device):
    import torch
    from transformer_engine.pytorch.optimizers.multi_tensor_apply import (
        multi_tensor_applier, multi_tensor_scale,
    )
    clip_coeff = max_norm / (total_norm + 1.0e-6)
    if clip_coeff < 1.0:
        dummy = torch.zeros(1, dtype=torch.int, device=device)
        multi_tensor_applier(multi_tensor_scale, dummy, [grads, grads], clip_coeff)


def _train_one_step(model, main_params, model_params, optimizer,
                    micro_batches, device, max_norm=1.0):
    import torch
    # 1. zero main_grad
    for p in model_params:
        p.main_grad.zero_()
    # 2. forward + backward
    losses = []
    for batch in micro_batches:
        loss = model(batch) / len(micro_batches)
        loss.backward()
        losses.append(loss.detach())
    # 3. DP all_reduce (invoked even at DP=1)
    if torch.distributed.is_initialized():
        for p in model_params:
            torch.distributed.all_reduce(p.main_grad,
                                         op=torch.distributed.ReduceOp.SUM)
    # 4. main_grad → master.grad
    for master, p in zip(main_params, model_params):
        master.grad = p.main_grad
    # 5. grad_norm (lossy)
    lossy = _compute_grad_norm_fp32(main_params, device)
    # 6. clip (use lossy)
    grads = [m.grad for m in main_params if m.grad is not None]
    _clip_grad(grads, max_norm, lossy, device)
    # 7. optimizer.step
    optimizer.step()
    # 8. master → bf16 model
    with torch.no_grad():
        for master, p in zip(main_params, model_params):
            p.copy_(master.to(p.dtype))
    # 9. Report (the fp32-cast value)
    return sum(l.item() for l in losses), _reported_grad_norm(lossy, device)
```

---

## 7. SOP

| Stage | Pass condition |
|---|---|
| alignment.forward | per-layer forward activations bitwise |
| alignment.backward | every main_grad bitwise |
| bitwise-singlecard trajectory | N-step loss + grad_norm all bitwise |
| Any milestone ship | at least 3 seeds (e.g. 1234/42/7777) all pass |

```bash
for s in 1234 42 7777; do SEED=$s python -m harness.cli run <suite> --gpu 0; done
```

---

## 8. Appendix

### 8.1 Glossary

| Term | Meaning |
|---|---|
| ULP | Unit in the Last Place |
| Lossy | A value that has been through precision losses such as fp32 squaring |
| Raw | Raw kernel output |
| main_grad | Pre-allocated fp32 buffer |
| master | fp32 weights held by the optimizer |
| Bridge | Glue script that wraps the ref to inject hooks |

### 8.2 Megatron source code locations (v0.15)

| Module | Path |
|---|---|
| Optimizer main entry | `megatron/core/optimizer/__init__.py` |
| Float16Optimizer | `megatron/core/optimizer/optimizer.py` |
| Grad clip / norm | `megatron/core/optimizer/clip_grads.py` |
| `training_log` | `megatron/training/training.py:1501` |
| `reduce_max_stat` | `megatron/training/utils.py:231` |
| Param grouping (wd) | `megatron/core/optimizer/__init__.py:_get_param_groups` |

### 8.3 Anti-cheat checklist

- [ ] candidate does **not** import any `megatron.*` symbol
- [ ] candidate setup phase does **not** read the ref's master / grad / activation
- [ ] candidate only reads: `init_weights.pt` (shared initialization source), `batch_<i>.pt` (input)
- [ ] candidate `_compute_grad_norm` does not cheat by reading the ref's value
- [ ] harness framework_guard passes

---

## 9. Quick-reference card (11 commandments)

```
┌──────────────────────────────────────────────────────────────────┐
│  1. master must be a bf16 → fp32 cast (lossy), not the raw fp32  │
│  2. weight_decay split by dim into two groups: 2D=0.1, 1D=0.0    │
│  3. FusedAdam capturable=False                                   │
│  4. main_grad is a pre-allocated fp32 buffer, do not .grad.float() │
│  5. NCCL all_reduce is required even at world_size=1             │
│  6. Compute two grad_norm values: lossy (for clip) + reported (for log) │
│  7. reported = torch.tensor([v], fp32).item() to fp32-snap       │
│  8. clip via multi_tensor_scale; the denominator +1e-6 must not be omitted │
│  9. bucket uses reversed(named_param), but the l2norm input list │
│     uses wd-first / no-wd-second (two orderings; must be kept    │
│     straight when DP>1, §4.9)                                    │
│ 10. When DP>1, force a single bucket: ref sets --ddp-bucket-size 2^30, │
│     ours hard-codes the same (sidestepping multi-bucket complexity) │
│ 11. Any 1-ULP difference → dump-and-diff, do not guess           │
└──────────────────────────────────────────────────────────────────┘
```

---

**Version**: 2026-05-19, summarized from experience with **alignment (forward+backward align) + bitwise-singlecard (DP=1 multistep)
+ bitwise-multicard (DP=8 multistep)**; only applicable to the Megatron-LM ref (the pure-torch
path does not need any of this).

**Main additions in bitwise-multicard**:
* §4.9 upgraded to 🟥 — bucket physical order ≠ `multi_tensor_l2norm` input order;
  the real roadblock of bitwise-multicard, only exposed when DP>1
* Forced single-bucket tactic (quick-reference card #10) — sidestep the complexity
  of Megatron's default 40M multi-bucket split; pinch both ours and ref into a single-bucket layout
  so the per-rank shard l2norm can bit-match

If new pitfalls show up in subsequent bitwise-perf–long-horizon, append to §4.
