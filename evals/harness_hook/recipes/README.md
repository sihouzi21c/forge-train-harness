# alignment–resume capture bridge — recipes

The harness ships **one** thing for alignment forward / backward alignment
and the per-step bitwise-singlecard / bitwise-multicard / bitwise-perf / resume hash-record diffs: the standard
hook at `evals.harness_hook.install(model, optimizer, *, output_file,
hash_capture_level, persistent, ...)`. It's pure-PyTorch (only depends
on `torch.nn`), and it produces the exact wire format the dispatcher
expects:

* `<output_file>` — **JSON dict** whose keys share a uniform grammar
  `{step_<N>.}rank<r>.{mb<m>.}<family>`:
  * single-step alignment (`persistent=False`): `rank<r>.mb0.fwd.<named_modules() fqn>#<call>`,
    `rank<r>.mb0.bwd.<fqn>#<call>`, `rank<r>.grad.<named_parameters() fqn>.postallreduce`;
  * persistent bitwise-singlecard–resume (`persistent=True`): the same keys
    additionally prefixed with `step_<N>.`, `mb<m>.` running over the
    grad-accumulation microbatches, plus
    `step_<N>.rank<r>.mb<m>.loss.per_token.preallreduce` (`[MBS, seq_len]`
    fp32 NLL, one per microbatch), `step_<N>.rank<r>.loss.scalar.preallreduce`,
    `step_<N>.rank<r>.loss.scalar.postallreduce`, and the grad family with
    both `.preallreduce` and `.postallreduce` flavours.

  Each rank writes its own `<output_file>.rank<r>` (`r` = global `RANK`); the
  dispatcher merges the per-rank files by their disjoint `rank<r>.` prefixes
  before diffing, so every rank / microbatch / call is compared — not just
  rank 0.

  **Key axes** (all four independent):
  * `rank<r>.` — the global `RANK`, on **every** key (uniform; `rank0.` for a
    single process). Replaces any tensor/data-parallel-specific scheme; the
    two shards of a column-parallel weight land under distinct `rank<r>.`
    prefixes and never collide.
  * `step_<N>.` — the training step (persistent mode only).
  * `mb<m>.` — the grad-accumulation microbatch. On `fwd`/`bwd`/`loss.per_token`.
    **Not** on `grad.*` / `loss.scalar.*` — those are post-accumulation.
  * `#<call>` — the per-forward CALL index (`fwd`/`bwd` only): how many times
    that module has already fired *under the current prefix*, so a shared
    module re-used within one forward records both fires (e.g. an `output`
    head on the main + MTP paths → `#0` / `#1`). Because `mb<m>.` is in the
    prefix, the call index restarts at `#0` each microbatch — microbatch is
    the prefix, call is the suffix, never one flat counter.

  Every value is a record `{"hash": "<32-hex>", "shape": [...],
  "dtype": "..."}` (blake2b-128 over the tensor's raw bytes, streamed
  in 8M-element chunks).

* `<output_file>.graph.json` — list of `{order, step_prefix, fqn,
  class, input_shapes, input_dtypes, output_shape, output_dtype}`
  entries in forward-execution order. Used for diagnosing structural
  drift when a key is present on only one side.

After dumping, single-step alignment mode runs ordered teardown (drain CUDA,
NCCL barrier + `destroy_process_group`, `empty_cache`) and **`raise
SystemExit(0)`**. Persistent mode does NOT exit — the training loop
continues; the bridge calls `session.dump()` at the end.

What the harness does **not** ship: any wiring that drops `install(...)`
into an arbitrary customer training script. The shape of customer
scripts varies too much (torchrun + upstream `pretrain_*.py`,
container-launched multi-node, custom Python entry, DeepSpeed YAML,
…) for a single in-repo wrapper to stay correct. Building that
wiring is the **bridge author's job** (alignment agent / dev-agent on the
ours side).

## Two modes, one knob — `hash_capture_level`

The per-suite TOML key `[evals.<suite>].hash_capture_level` is a
typed int (0 / 1 / 2) that flows through the dispatcher → subprocess
CLI args (`--hash-capture-level <N>`) → bridge argparse → `install()`
kwarg. **No env var carries it** anywhere on the wire. Levels:

| Level | Records captured per step |
|---|---|
| **0** | None — install is a no-op; existing stdout scalar comparison only |
| **1** | Loss family (`loss.per_token.preallreduce`, `loss.scalar.{pre,post}allreduce`) + grad family (`grad.<fqn>.{pre,post}allreduce`) |
| **2** | Level 1 + module forward hooks (`fwd.<fqn>#<call>`) + module full-backward hooks (`bwd.<fqn>#<call>`) |

Assignment in `harness/config/eval/dense_training.toml` and
`dense_training_1b.toml`:

| Suite | Milestone | Level |
|---|---|---|
| `forward-align` | alignment.forward | 2 |
| `backward-align` | alignment.backward | 2 |
| `multistep-1gpu` | bitwise-singlecard | 2 |
| `multistep` | bitwise-multicard | 2 |
| `perf-bitwise` | bitwise-perf | 1 |
| `resume-gate-20` | resume | 1 |
| `long-train` / `long-train-smoke` | long-horizon | 0 (absent) |

bitwise-perf uses level 1 (no per-module hooks) to bound the MFU overhead on
the MFU-measuring path; the 14.5% target is revisited after the
first measurement of the actual overhead — not pre-adjusted.

## The dispatcher's contract with the bridge

The dispatcher routes ref subprocesses through the **bridge**
(`ref/bridges/bridge.sh`) whenever `hash_capture_level > 0`. When 0
it stays on the bare L0 ref script.

The bridge subprocess receives:

* env vars (back-compat — kept for old bridges):
  `HOOK_OUTPUT_FILE`, `FORGE_BACKEND`, `BRIDGE_*`, `MEGATRON_ROOT`,
  `DATA_PATH`, suite `ref_env` overrides, …
* **CLI args** (the typed wire — new):
  `--hash-capture-level <N>  --hash-output <abs path>  [--persistent]`
  forwarded as `"$@"` through bash → patched launcher → `interposer.py`.

`interposer.py`'s argparse strips these flags from `sys.argv` before
`runpy.run_path` invokes the L0 entry, so the L0 script's own
argparse only sees its own flags.

The ours subprocess receives the same CLI args appended to its
command line — the launcher (`launch_dp.py`) forwards `sys.argv[1:]`
to every rank, where the per-runner script (`eval_capture_align.py`
/ `eval_train_steps.py` / `eval_resume_train.py`) parses them with
`argparse.parse_known_args` and threads the values into
`TrainLoopConfig(hash_capture_level=..., hash_output=...,
persistent=...)`.

## The minimum bridge — when the customer entry is a plain PyTorch `.py`

**Single-step alignment**: same as before — three lines after model and
optimizer are built, plus the new typed kwargs:

```python
import argparse, os
from evals.harness_hook import install

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--hash-capture-level", type=int, default=0)
_p.add_argument("--hash-output", type=str, default="")
_p.add_argument("--persistent", action="store_true")
_args, _rest = _p.parse_known_args()
import sys
sys.argv = [sys.argv[0], *_rest]  # strip our flags before customer parses

install(
    model, optimizer,
    output_file=_args.hash_output or os.environ.get("HOOK_OUTPUT_FILE"),
    hash_capture_level=_args.hash_capture_level,
    persistent=_args.persistent,
)
```

**Persistent bitwise-singlecard–resume**: `install(..., persistent=True)` returns a
`CaptureSession` that the customer training loop must drive
explicitly. Hook points (mirror the ref-side `train_pure_mup_mtp.py`
call sequence — see Layer 5 of the project plan):

```python
session = install(model, optim, output_file=path, hash_capture_level=2,
                  persistent=True)

for step in range(num_steps):
    session.begin_step(step)
    for mb in range(grad_accum):
        session.begin_microbatch(mb)   # -> step_<N>.rank<r>.mb<mb>. namespace
        loss_per_token = compute_per_token_nll(...)   # [MBS, seq_len] fp32
        session.capture("loss.per_token.preallreduce", loss_per_token)
        loss_local = (loss_per_token * mask).sum() / mask.sum()
        loss_local.backward()          # fwd/bwd hooks fire under rank<r>.mb<mb>.
        # ... accumulate grads ...
    session.end_microbatch()           # back to step_<N>.rank<r>. for per-step captures
    session.capture("loss.scalar.preallreduce", loss_local)
    session.capture_grads(allreduce="pre")
    dist.all_reduce(...)   # DP all-reduce on grads + loss scalar
    session.capture("loss.scalar.postallreduce", loss_global)
    session.capture_grads(allreduce="post")
    optim.step()

session.dump()
```

At `hash_capture_level == 0`, every session method is a no-op — the
above code paths zero-cost regardless of the configured level.

## Dev-agent contract — what the ours engine must satisfy

For the per-FQN / per-step hash diffs to produce useful signal,
the ours engine MUST mirror the ref side on four axes:

1. **Shape parity** — including vocab padding. The ref Megatron stack
   pads `embedding.word_embeddings.weight` /
   `output_layer.weight` along dim 0 to the next multiple of
   `make-vocab-size-divisible-by`. Ours **must** pad to the same
   shape so the hash records line up byte-equal. The harness does
   **no** comparator-side trim; shape mismatch in the hash diff is
   a real FAIL signal (the legacy `truncate_padded_to_candidate`
   helper has been deleted — hash discards the bytes so it cannot
   exist).

2. **FQN parity** — `named_modules()` / `named_parameters()` walks
   on both sides must produce identical key sets. Wrapper-unwrap
   policy in `_module_hook.py` chases `.module` chains automatically.
   The full module key is `{step_<N>.}rank<r>.{mb<m>.}fwd|bwd.<fqn>#<call>` —
   the `rank<r>.` prefix, the `mb<m>.` microbatch prefix, and the `#<call>`
   per-forward index must all match too, not just the `<fqn>` (see § key axes
   above).

3. **Hook-point parity** — ours must expose pre / post DP-allreduce
   hook points for both grad and loss scalar, plus the per-token
   NLL tensor before any reduction. The exact API for ours-side
   capture (manual `session.capture(...)` + automatic
   `session.capture_grads(...)`) matches the ref-side call sequence
   above 1:1.

4. **Call sequence** — `begin_step → for each microbatch: begin_microbatch
   → capture(loss.per_token) → forward/backward → end_microbatch →
   capture(loss.scalar.preallreduce) → capture_grads(pre) → DP all-reduce →
   capture_grads(post) → capture(loss.scalar.postallreduce) → optim.step →
   repeat → dump`. `begin_microbatch(i)` / `end_microbatch()` bracket the
   grad-accumulation loop so per-microbatch tensors land under `rank<r>.mb<m>.`
   and the post-accumulation grad / loss.scalar captures under `rank<r>.`
   only (no `mb`). Any deviation
   (wrong order, missing call, different key) surfaces as either an
   FQN-only-on-one-side silent skip or a hash mismatch — never the right
   thing silently.

The ref-side `ref/bridges/interposer.py` monkey-patches
`train_pure_mup_mtp.main` to drive this sequence (see Layer 5 of the
project plan). The same sequence MUST appear in the ours engine code.

## Shipped unified bridge — `ref/bridges/bridge.sh` + `interposer.py`

The repo ships a unified bridge that supports both the Megatron and
Torch backends out of the box. No agent-authored bridge code is
needed for these two frameworks.

**Shell dispatch** (`ref/bridges/bridge.sh`):

1. Reads `FORGE_BACKEND` (set by `_common.py:run_ref_capture` or
   `run_via_ref_script`).
2. Copies the ref launcher to a temp directory and sed-patches the
   Python entry point to `ref/bridges/interposer.py`.
3. Exports `BRIDGE_BACKEND`, `BRIDGE_ORIGINAL_ENTRY`, `BRIDGE_REF_DIR`,
   `BRIDGE_REPO_ROOT`.
4. `exec bash <patched_launcher> "$@"` — the original ref script is
   never modified, and the dispatcher's `--hash-*` CLI args flow
   through `"$@"` to the interposer.

**Python interposer** (`ref/bridges/interposer.py`):

Runs as the torchrun entry point in each rank. Parses
`--hash-capture-level / --hash-output / --persistent` via
`argparse.parse_known_args` and strips them from `sys.argv` before
`runpy.run_path` invokes the L0 entry. Per backend:

* **Megatron**: patches `megatron.training.training.setup_model_and_optimizer`
  (single-phase — the factory returns `(model_list, optimizer, scheduler)`),
  installs the argparse shim for muP/Eagle flags, and fixes the
  GPTDataset `get_batch_on_this_tp_rank` unbound-variable bug.
  `grad_attrs=("main_grad", "grad")`.
* **Torch**: two-phase capture — patches `MiniCPM4MupMtp.__init__`
  to record the model, patches `AdamW.__init__` to record the
  optimizer, wires the hook when both exist. When `--persistent`
  is set, additionally source-rewrites `train_pure_mup_mtp.main`
  to drive `session.begin_step / capture / capture_grads / dump`
  at the hook points enumerated above.

After hook installation, delegates to the original entry via
`runpy.run_path(BRIDGE_ORIGINAL_ENTRY, run_name="__main__")`.

## Things the bridge author should keep in mind

* **FQNs are SSOT.** `install(...)` keys tensors by
  `named_modules()` / `named_parameters()` verbatim — no rename.
  If `nn.Module` wrapping (DDP / FSDP / Megatron `Float16Module` / …)
  prefixes the names, unwrap before calling `install`. The standard
  hook already chases `.module` chains, but stops at non-`.module`
  attribute names.
* **Writer-rank predicate.** The default writes from `RANK == 0`. For
  any non-trivial parallel layout (TP > 1, PP > 1, FSDP shards, …)
  pass `writer_rank_predicate=lambda: <your zero-rank check>` so a
  single canonical dump appears on disk and non-writer ranks still run
  ordered teardown and exit 0 so the NCCL barrier stays balanced.
* **Gradient slot.** The default `grad_attrs=("main_grad", "grad")`
  covers Megatron's distributed-optimizer slot first and plain
  PyTorch's `.grad` as fallback. Override `grad_attrs=...` only if
  your framework accumulates somewhere else.
* **Atomic write.** The JSON dump and the `.graph.json` are written
  to a `.tmp` sibling and renamed; partial files never appear under
  `$HOOK_OUTPUT_FILE` / `--hash-output`.
* **No comparator-side trim.** The harness does NOT silently absorb
  shape mismatches. If your engine's vocab-padded weight gets a
  different shape than ref, the hash diff FAILs — fix the padding,
  don't ask the comparator to look the other way.

## Canonical-state bootstrap recipe

`canonical_state_fp32.pt` is the FP32 master state every harness
suite (`op-long`, `eval_train_steps`, `eval_resume_train`,
`eval_long_train`, `eval_capture_align`) and every ours-side
loader (`train_engine/src/training_engine_tensor/parameters.py`)
bootstraps from. It is a **one-shot** artifact, NOT a hash-record
dump — the loader needs the real fp32 bytes to initialize from.

`install_canonical_state_dump(model, optimizer, *, output_file, ...)`
remains the API. The dump is a single `torch.save` dict
`{fqn: fp32 cpu tensor}`, same as before. No change in this layer.
**`forge_init_ones`: TWO canonicals, not one.** Each gate declares a
`forge_init_ones` field (`0` or `1`) that selects between two init
regimes. Bootstrap must therefore be run **twice**, once per regime,
into sibling subdirs of `$HARNESS_CHECKPOINT_ROOT`:

```
$HARNESS_CHECKPOINT_ROOT/
  ones/canonical_state_fp32.pt   ← --init-ones 1
  no1/canonical_state_fp32.pt    ← --init-ones 0
```

The ours runner derives the matching subdir per gate from its product's
`forge_init_ones` (see `evals/scripts/_gate_entry.py::GateInputs.checkpoint_root`).
Missing either file → the gate fails fast at canonical-load time.

The shipped tool `tools/bootstrap_canonical.py` wraps the recipe below
and writes the file to the matching subdir:

```bash
python tools/bootstrap_canonical.py --init-ones 1   # → ones/canonical_state_fp32.pt
python tools/bootstrap_canonical.py --init-ones 0   # → no1/canonical_state_fp32.pt
```

The harness ships **one** thing for the actual dump:
`evals.harness_hook.install_canonical_state_dump(model, optimizer, *,
output_file, ...)`. Same shape as `install(...)`:

* `output_file=None` is a strict no-op (safe to leave in a
  production trainer).
* Idempotent.
* Hijacks `optimizer.step`; at the first `.step()` we walk
  `named_parameters()`, harvest each FP32 master copy
  (`param.main_param` if Megatron's `DistributedOptimizer` set it,
  otherwise `param.data.float()`), atomic-write a single
  `torch.save` dict, then run ordered teardown and `raise SystemExit(0)`
  on every rank.

Unlike the alignment capture path, this hook is **not driven by the
dispatcher**. There is no `[defaults].ref_canonical_state_script`
field — `canonical_state_fp32.pt` is a precondition you set up once
during bootstrap (see `README.md` §3), not a per-gate artifact.

### Output file schema

```
torch.save dict, keyed by named_parameters() FQN verbatim:
    "embedding.word_embeddings.weight"               → fp32 cpu tensor
    "decoder.layers.0.input_layernorm.weight"        → fp32 cpu tensor
    "decoder.layers.0.self_attention.linear_qkv.weight"
                                                     → fp32 cpu tensor
    …
    "output_layer.weight"                            → fp32 cpu tensor
```

No prefix, no rename. Downstream ours-side loaders apply their own
`ref_name → ours_name` map on top
(`_PARAM_NAME_MAP` in `train_engine/.../parameters.py` is the
reference example). This keeps the dump usable by any consumer that
already knows the reference framework's parameter naming.

### Minimum bridge — when the reference entry is a plain PyTorch `.py`

Add three lines right after model + optimizer are built:

```python
from evals.harness_hook import install_canonical_state_dump
install_canonical_state_dump(
    model, optimizer,
    output_file=os.environ.get("CANONICAL_STATE_OUTPUT_FILE"),
)
```

Then run the reference launcher once with
`CANONICAL_STATE_OUTPUT_FILE=$HARNESS_CHECKPOINT_ROOT/canonical_state_fp32.pt`.
The process self-terminates on the first `optimizer.step`, leaving
the file in place. Unset the env var on subsequent runs and the
bridge code is a strict no-op — the reference trainer behaves
normally.

### Reference entry you cannot modify — general recipe

Same interposition pattern as the alignment bridge above (copy launcher,
write an outside-of-customer-entry Python module, monkey-patch the
framework call that returns `(model, optimizer)`), with one line
swapped: call `install_canonical_state_dump(...)` instead of
`install(...)`. The two hooks share the same surface, so a single
bridge skeleton can carry both — see "Mutual exclusion" below for
the constraint.

### Mutual exclusion with `install(...)`

`install` (single-step alignment mode) and `install_canonical_state_dump`
share the process-wide installed-flag because both hijack
`optimizer.step`. Persistent mode (`install(..., persistent=True)`)
does NOT hijack `optimizer.step` and composes freely with neither.

Calling both `install` (single-step) and
`install_canonical_state_dump` in the same process is a no-op for
the second call. Bridges keep them in **separate** Python entries:

* one bridge for alignment forward / backward capture (registered as
  `ref_capture_script` in `config/eval.toml`);
* one bridge for canonical-state bootstrap (run manually once
  during setup; not registered with the dispatcher).
