# Stage 1 Constraints (boundary + spec + forbidden)

This file consolidates everything the dev agent **must obey** across stage1 milestones: low-level technology stack boundaries, backward / autograd boundary, FP32 precision specification, MFU specification, precision alignment requirements, and all cross-milestone forbidden behaviors. Per-milestone gates and any single-milestone constraints live in the per-milestone `<name>.md`. Backend differences are inlined; the active backend is selected by `@@FORGE_CONFIG_DIR@@/ref.toml [ref].backend`.

## Hard rule — no external framework reference

> **RED LINE — NO READING OUTSIDE `ref/` / `$MEGATRON_ROOT` *(megatron only)*** (ABSOLUTE ZERO TOLERANCE)
>
> The only framework code you may read for alignment guidance is `ref/` (the active backend's reference source under the current loop's own workspace) and `$MEGATRON_ROOT` *(megatron only)* (read-only baseline reference). **Every other engine implementation is off-limits — not just to copy from, but to read, list, grep, `cat`, or open in any way.**
>
> - **FORBIDDEN sources** (non-exhaustive):
>   - `.artifacts/forge_train/<other_loop_id>/workspace/...` — any sibling loop's workspace tree, including its `workload/`, `evals/`, `harness/`, `ref/` mirror, **and especially `workload/notes/perf_log.md`**. Exactly one `loop_id` is yours: the one named in your standing rules' workspace path. Reading any other is a hard violation, even a bare `ls` of its directory.
>   - `git show <branch>:path`, `git log -p path`, `git show <sha>:path`, `git stash show -p`, sibling worktrees listed by `git worktree list` — prior commits or other branches of `workload/src/training_engine_tensor/`, including this loop's own prior rounds. Each round must rederive its solution from `ref/` and the gate signal, not from history.
>   - Path-spelunking for engine source on the local disk: `find ~ -name forward.py`, `mdfind`, `locate model.py`, scanning `/Users/*/...` or `/home/*/...` for someone else's framework checkout.
>   - Third-party training engines (HuggingFace Transformers, vLLM, Megatron-LM forks outside `$MEGATRON_ROOT`, the active workload's open-source inference reference code, etc.).
>
> - **WHY**: Each round must be an independent derivation against `ref/` and the gate. Reading sibling work — even "just to compare file layout" or "just the perf_log to see which milestones passed" — silently anchors your next decisions to a structure you did not earn from `ref/`. The review agent **cannot reliably detect** this leakage from a finished round's diff (the structural choices look plausible against `ref/` alone), so the prohibition is enforced at the dev stage, not deferred to review.
>
> - **VIOLATION**: A single `Read`, `Bash ls/cat/find/grep/head/tail`, or `Glob` call whose target resolves under any forbidden source above. The round is **discarded** — roll back all uncommitted work in this round. Do not commit rationales of the form "I saw the file but did not copy from it"; reading is the violation, not copying. Repeating after a violation aborts the loop and surfaces a blocker to the user.
>
> - **ONLY ALLOWED REFERENCES**: `ref/` (current workspace, read-only), `$MEGATRON_ROOT` (read-only via `cat` / `ls` / `git show` within that path; baseline framework source for the megatron backend), and the per-stage prompt files this loop injects. If you are tempted to look anywhere else, that temptation itself is the bug — stop and rederive from `ref/`.

## Hard rule — every round must advance the active milestone's core work

> **RED LINE — EVERY ROUND ADVANCES CORE WORK** (ABSOLUTE ZERO TOLERANCE)
>
> The active milestone's **core objective** must move forward in every round. The core objective is the gate-decisive change the milestone exists to produce — e.g. for **bitwise-perf** a concrete MFU / memory optimization (kernel change, fused operator, scheduling/overlap change, recompute elimination, etc.), for **alignment** the next bitwise-alignment fix. Each milestone's own core objective is defined in its `<name>.md`.
>
> Preparation work — profiler infrastructure, diagnostic / tracing tools, helper scripts, test scaffolding, internal refactors, measurement hooks — is permitted **only as a direct unblocker** for this round's core change. Once the minimum prep needed to make the next core decision is in place, the core implementation MUST begin in the same round.
>
> - **MANDATORY**: Every round's commit must contain at least one substantive change to the active milestone's core target (defined in its `<name>.md`). E.g. for **alignment**, a real alignment fix in the workload's training-engine source.
> - **MANDATORY**: When prep is required, build the **minimum** prep needed to make the next core decision, then act on that decision in the same round. Cite the prep output (e.g., the profile snapshot path) in the commit message and follow it with the optimization the prep justified.
> - **FORBIDDEN**: Ending a round on prep alone — improving the profiler, adding metrics, polishing the trace pipeline, cleaning up the test harness, writing analysis docs — without an accompanying core-work change in the same commit, **even when the prep is genuinely needed**.
> - **FORBIDDEN**: Deferring bitwise-perf / long-horizon optimization to "first finish profiler / infra polish / tooling cleanup". Examples of disallowed framings: "this round I improved the nsys wrapper, next round I will optimize"; "the profiler still misses kernel X, I will add it before touching the hot path"; "I refactored the perf_log format so future rounds are clearer". All of these waste the round.
> - **VIOLATION**: A round whose commit diff under `workload/src/training_engine_tensor/` is empty or touches only logging / measurement / diagnostic code, while the milestone's core target (MFU, bitwise diff, memory, save/load) is unchanged.

## Hard rule — no self-imposed ceilings in `perf_log.md`

> **FORBIDDEN in `workload/notes/perf_log.md`**: any statement that claims a metric or threshold is "structurally unreachable", "ceiling ≈ N%", "physically impossible without breaking X", or any equivalent paraphrase, **unless every contributing kernel time is quoted from `summary.md` AND the floor is shown to be invariant across all allowed implementation swaps in `long-horizon.md` §long-horizon stage operator freeze constraint**. A self-imposed ceiling that a later round disproves wastes loop budget and pollutes attribution; if you are tempted to write one, write a bisect plan instead.

## Low-level technology stack boundaries

### Allowed (primitive layer, direct call, no wrapping)

The union of capabilities each backend's ref exposes. Items marked **(megatron only)** / **(torch only)** are only relevant when the corresponding backend is active; the others apply to both.

| Category               | Description                                                                |
| ---------------------- | -------------------------------------------------------------------------- |
| `torch` (bare import)    | tensor container, device management (`torch.empty` / `torch.zeros` / `torch.randn` / `.data_ptr()` / `.to()`); under the **torch backend**, matrix multiply `torch.matmul` / `torch.mm` / `torch.bmm` / `torch.addmm` (underlying cuBLAS) is the canonical GEMM entry |
| `torch.cuda`         | stream / event / device management                                         |
| `torch.backends`     | cuDNN and other backend configuration. **Under the torch backend** also covers any backend toggles required by ref's deterministic stack — read ref source for the exact set, since the concrete backend may change |
| `torch.distributed`  | DP communication (`all_reduce` / `reduce_scatter` / `all_gather`), shared NCCL communicator implementation for bitwise alignment |
| `torch.ops.aten.*` *(torch only)* | direct ATen op calls, used to implement Linear / Embedding / RMSNorm / SwiGLU / RoPE / Cross-Entropy / Attention etc.; the specific choice is up to the agent, as long as it is bitwise aligned with ref |
| `torch._foreach_*` / `torch._fused_adamw_` *(torch only)* | top-level ATen-style fused / foreach ops, used to implement AdamW / grad norm etc., replacing `torch.optim` |
| `torch.utils.checkpoint.checkpoint(use_reentrant=False)` *(torch only)* | recompute entrypoint (the ref enables recompute through this path; if the in-house side enables `--recompute` it must reproduce the same path to keep bitwise) |
| `triton`             | writing and calling self-developed kernels                                 |
| cuBLAS               | GEMM. Under **torch**, **triggered via `torch.matmul` / `torch.addmm`**; the `nn.Linear` / `F.linear` wrapper path is not allowed. Under **megatron**, GEMM goes through TransformerEngine fused operators (e.g. `te.LayerNormLinear`, `te.Linear`) which internally call cuBLAS/cublasLt — bare `torch.matmul` uses a different cuBLAS algorithm selection and will produce 1-ULP diffs vs the ref; do not use it for bitwise-aligned GEMM under megatron |
| cuDNN FlashAttention *(megatron only)* | may serve as a source of low-level capability, but the formal attention interface uniformly goes through the TransformerEngine fused attention mode (GQA mode) |
| TransformerEngine *(megatron only)*   | fused kernels (RMSNorm, SwiGLU, fused attention, GEMM). The in-house side must use the same TE operator the ref uses for each site (e.g. `te.LayerNormLinear` for fused norm+GEMM, `te.DotProductAttention` for fused attention) to achieve bitwise alignment. Forbidden under the torch backend (see Forbidden table) |
| NCCL                 | `allreduce` / `allgather` and other collectives                            |
| `modelbest_sdk` *(torch only)* | external dataloader dependency; its internal indirect use of `torch.utils.data` falls within external dependency scope and is not intercepted by framework_guard |
| tokenizer / checkpoint | data loading (directly reading files)                                    |

### Forbidden (framework layer, intercepted by pre-commit + framework_guard)

| Forbidden item                            | Reason                                                                                                |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `torch.nn` (incl. `torch.nn.functional`)  | Self-developed static model definition, no `Module` / `Linear` / `Embedding` / `RMSNorm` high-level abstractions; under the torch backend the ATen ops behind wrapper layers are called directly by the agent |
| `torch.optim`                  | Self-developed optimizer step orchestration. The in-house side replaces `torch.optim` with primitive-level ops; the **specific** primitive choice is whatever the active ref's optimizer path uses — read `ref/reference/${ref_script}` to confirm before implementing. **In bitwise milestones**, the in-house implementation must reproduce the ref's optimizer path exactly (same primitive set, same parameter grouping if ref splits parameters, e.g. by muP lr-group). On the megatron backend the ref typically calls fused kernels directly; on the torch backend the ref typically uses `torch._fused_adamw_` / `torch._foreach_*` and may split params by muP — verify in source before mirroring. |
| `torch.autograd` / `.backward(` / `requires_grad=True` / `requires_grad_(True)` | Static execution graph, backward order determined at compile time. The four tokens are all autograd-engine entrypoints; `torch.autograd` is scanned at module level, the other three at literal-string level by `framework_guard.py:AUTOGRAD_BANNED_KEYWORDS`. Legitimate static-backward path is described in §Backward and autograd boundary below |
| `torch.utils.data`             | Self-developed fixed-shape data pipeline. In-house code must not import directly; under the torch backend, `modelbest_sdk.ModelbestDataloader`'s indirect use as an external dependency is allowed (decoupled from the in-house path); under the megatron backend, indirect use via Megatron `GPTDataset` is similarly allowed |
| `torch.amp` / `torch.cuda.amp` | Precision is fixed at bf16, no mixed precision management needed                                          |
| Standard torch attention path *(megatron only)* | The formal attention implementation uniformly uses TransformerEngine fused attention (GQA mode)           |
| `megatron`                     | Completely forbidden as a direct framework dependency. Under **megatron** backend: only allowed as a baseline / alignment reference (the agent may read source code and run baseline scripts). Under **torch** backend: the prohibition is preserved even though this backend does not depend on Megatron, to prevent accidental introduction; the agent may still read Megatron source code for MFU formula reference |
| `deepspeed`                    | Completely forbidden                                                                                      |
| `transformer_engine` *(torch only)* | completely forbidden; under the torch backend, ref also does not depend on TE |

### Gate enforcement

- A single `pre-commit` hook scans all text files in the repo, intercepting forbidden keywords and forbidden torch framework-layer submodules.
- The gate rules are centralized in `harness/framework_guard.py`; only this single file needs to be modified to adjust the boundary.
- Whitelisted files (the gate script itself, this file, rules files) are managed through `ALLOWLIST`.

## Dataloader description (thin wrapper inside the self-developed framework)

**The dataloader's algorithm and data source do not need to be self-developed**, but its entry point belongs to the self-developed framework. Starting from the bitwise-singlecard single-card alignment milestone the in-house engine integrates the **real** dataloader, and every gate from then on uses that **same** dataloader. The in-house side only needs to match the **single** dataset / external dataloader the active **ref script** loads — read the ref to see which external dependency it imports and which data path / env var it reads, place that call entry inside the workload's `dataloader.py` (driven by `train_loop`), and reproduce the ref's windowing / per-step batching. Do **not** build a general loader that adapts to multiple dataset formats; match the active ref's choice only. `torch.utils.data` is on the forbidden list — the self-developed code must not import it directly, but indirect use of the external dependency is allowed.

**Strict SSOT boundary**: the harness layer (`evals/`, `harness/`, `tools/`) is forbidden from constructing a dataloader — from bitwise-singlecard onwards, the sole owner of the dataloader is the self-developed framework itself (`training_engine_tensor`). Gate scripts (`evals/scripts/eval_*.py`) only read env, construct `TrainLoopConfig`, and call `run_training_loop`; any code in a gate that reconstructs the dataloader, captures a batch, or self-builds an iterator is regarded as SSOT drift.

## Attention-specific rules

- During the bitwise-alignment milestones, read the **ref script's attention implementation** and use the **same attention backend the ref uses** (same call, same backend toggles); mirror whatever the ref does rather than assuming a stack or maintaining a backend-specific divergence. (The attention backend may later change under the performance milestone's operator-freeze rules — see `long-horizon.md`.)
- If special attention patches must be retained for debug / compare needs, isolate them behind an explicit switch, off by default; they must not pollute the performance hot path.

## Backward and autograd boundary

> The in-house backward must be a **statically-scheduled reverse computation graph**, not relying on the PyTorch autograd engine. This applies to both backends; the autograd-engine entrypoints below are intercepted by `framework_guard.py` regardless of which backend is active.

### Forbidden entrypoints (intercepted at the framework_guard keyword-scan level)

| Forbidden token                                                                | Why forbidden                                                                                            |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| Any `.backward(` call, e.g. `tensor.backward()` / `tensor.backward(retain_graph=True)` | autograd engine reverse entrypoint 1: triggers PyTorch to automatically derive the computation graph backward from leaves, running all `*_backward` ops, bypassing the "compile-time static backward-order determination" requirement |
| `requires_grad=True` constructor argument                                      | autograd engine reverse entrypoint 2: causes the original tensor to enter the autograd graph; all subsequent ops automatically record the forward graph |
| `requires_grad_(True)` setter                                                  | same as above, setter version                                                                            |
| `torch.autograd.grad(...)` / `torch.autograd.backward(...)`                    | autograd module-level API; already in the `torch.autograd` ban list                                       |
| Custom `torch.autograd.Function` subclasses                                    | even with manually-implemented fwd/bwd, this is essentially hooking backward into the autograd engine, violating the "static scheduling" spirit |
| `torch.func.vjp` / `torch.func.grad` / `torch.func.jacrev` (and the deprecated `functorch.*` aliases) | a thin functorch wrapper that internally calls `torch.autograd.grad` on a tape recorded by re-running the forward — still the dynamic autograd engine, just relabelled; same "compile-time static backward-order" violation |

`framework_guard.py` scans the first three tokens as literal strings in `AUTOGRAD_BANNED_KEYWORDS`; `torch.autograd` is scanned by module name in `TORCH_BANNED_SUBMODULES`. All fail-fast.

### Acceptance signals

- `bin/harness run guard` must pass (at the keyword-scan level).
- `bin/harness run forward-align` / `bin/harness run backward-align` bitwise PASS (`max_abs_diff == 0`).
- The review agent should check that no `.backward(` / `requires_grad=True` / `requires_grad_(True)` appears in the commit diff (any occurrence is a violation).

## FP32 precision specification (mandatory requirement under BF16 training)

Even if the main training precision is BF16, the following 9 operations **must use FP32**. This is an architecture-level constraint; it is not exempted just because numerical gates pass — gates test the observable output on the current test data; the FP32 specification guarantees the system's numerical stability on arbitrary data. Under the torch backend, the concrete implementation must align word-for-word with the same-named landing points in the ref script (the agent should confirm by reading the ref).

### Mandatory FP32 list

| #   | Operation                          | Requirement                | Rationale                                                                                  |
| --- | ---------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------ |
| 1   | Optimizer state (master weights, m, v) | All FP32                | BF16 has only ~7 significant digits; Adam updates of ~1e-6 would be rounded directly to zero. Torch ref: `fp32_master = [p.detach().float().clone()...]`, AdamW state defaults to the same dtype as param = fp32 |
| 2   | FP32 → BF16 parameter sync         | precision-aware delta copy (megatron) / **word-for-word identical** to ref sync path (torch) | Naive `master.bfloat16()` accumulates error. Megatron: `param += (master - param.float()).bf16()`. Torch: replicate the specific implementation inside the ref script after reading; arbitrary changes are not allowed |
| 3   | Cross-entropy / Softmax            | FP32 computation           | Over the vocab_size dimension, BF16 `exp()` will overflow/underflow. Torch ref `masked_ce` uses `F.cross_entropy(logits.reshape(-1, V).float(), labels.reshape(-1), reduction="none")` |
| 4   | Weight gradient (wgrad) dtype contract | **Megatron**: cuBLAS → FP32 output, written into the FP32 `main_grad` buffer. **Torch**: read `ref/reference/model_pure_mup_mtp.py` for the wgrad path; the in-house side must reproduce its dtype contract — FP32 wgrad output, FP32 `main_grad`, no per-microbatch BF16 round-trip. The underlying implementation (cuBLAS or in-house Triton) is a long-horizon optimization decision; see `long-horizon.md` §long-horizon stage operator freeze constraint for what is frozen vs swappable. | A BF16 GEMM with BF16 output produces ~3 ULP per-element error; per-microbatch `main_grad += wg` then sums BF16→BF16 with rounding errors that compound to systematic loss drift on long horizons (observable as `signed_mean > 0` on `loss-gate-200`). |
| 5   | Gradient accumulation buffer       | FP32 flat buffer           | Multiple-microbatch `+=` produces rounding errors each time in BF16. Megatron: aligns with baseline `accumulate_allreduce_grads_in_fp32=True`. Torch: completely consistent with ref `fp32_grad_bufs = [torch.zeros_like(p_) for p_ in fp32_master]` |
| 6   | AllReduce communication buffer     | FP32                       | Cross-rank summation in BF16 has precision loss amplified by `world_size`; the communication buffer dtype must match the dtype of ref's actual `dist.all_reduce` call |
| 7   | Gradient norm computation          | Compute L2 norm after `.float()` | Accumulating L2 norm over many parameters overflows in BF16. Torch ref uses `torch.nn.utils.clip_grad_norm_` (internally already reduces in fp32); the in-house side replicates the same math path using low-level ATen ops (`torch._foreach_norm` / `torch._foreach_mul_` etc.) |
| 8   | RMSNorm backward (rsigma)          | FP32                       | The division/subtraction in normalization backward is precision-sensitive; rsigma must be kept in FP32. Both torch ref `_RMSNorm` and `nn.RMSNorm` compute in fp32 |
| 9   | RoPE frequency precomputation      | FP32                       | Trigonometric precision requirement, and as a defense for future extension to longer sequences where precision becomes insufficient. Torch ref `precompute_rope_freqs` uses fp32 trigonometry |

### How to enforce the specification

- **Development constraint**: when modifying code that involves the above operations, the corresponding dtype must be ensured to be `torch.float32`.
- **Code review checklist**: when a PR involves the above operations, the reviewer must confirm the FP32 precision has not been downgraded.
- **No automated audit gate is added**: this specification is guaranteed by development constraints and review; no runtime or static analysis gate is added.

## MFU convention

MFU uniformly uses the standard formula (causal S/2, aligned with the baseline) and reports two conventions in the log (precise and standard). The gate decision is based on `standard`.

⚠️ **MFU scale (single unit invariant)**: every MFU value in this repo — the `mfu_e2e_standard` you emit on each `[LOSS]` line, the `mfu_e2e_target` gate threshold, and the MFU targets in prompts — lives on **one 0–100 percentage scale** (e.g. `36` means 36%), **never** the 0–1 decimal scale. The value you put on the wire must already fall in `[0, 100]`; every consumer (the harness gate, profile snapshot, mfu_record) reads it as-is and **does not re-scale**. This is purely a unit-alignment contract: ref and ours emit on this scale, and the gate compares against `mfu_e2e_target` directly. Emitting a `[0, 1]` fraction (or letting anyone multiply by 100 a second time) is the bug that turned a real ~17% MFU into `1742%`.

### MFU formula — torch backend (GEMM-enumeration closed-form, incl. MTP / Eagle)

Each GEMM `[m, k] · [k, n]` contributes `2·m·k·n` FLOPs (FMA = 2 ops); a full training step is `fwd + dgrad + wgrad = 3 · fwd` per GEMM:

```
H_q = NUM_HEADS · HEAD_DIM           # NUM_HEADS, HEAD_DIM, NUM_KV_HEADS, H, ffn, S, V, L from @@FORGE_CONFIG_DIR@@/model.toml — do NOT assume HEAD_DIM == H / NUM_HEADS (only an accident on some configs)
H_kv = NUM_KV_HEADS · HEAD_DIM

# Per-token forward FLOPs for one transformer block:
fwd_attn_proj  = 2 · H · (H_q + 2·H_kv) + 2 · H_q · H    # QKV + Wo
fwd_attn_score = 2 · S · H_q                              # causal: Q·K^T + attn·V each S·H_q
fwd_mlp_swiglu = 6 · H · ffn                              # SwiGLU three GEMMs (gate + up + down)
fwd_per_layer  = fwd_attn_proj + fwd_attn_score + fwd_mlp_swiglu

# Per-token forward total:
fwd_per_token = L · fwd_per_layer + 2·H·V                 # backbone + main LM head
              + eagle_num_layers · (
                    fwd_per_layer                         # Eagle transformer block
                  + 2·(2·H)·H                             # Eagle FC: Linear(2H -> H, no bias)
                  + 2·H·V                                 # second LM head (weight-tied, computation independent)
              )

train_per_token = 3 · fwd_per_token                       # fwd + dgrad + wgrad
flops_per_step  = train_per_token · (B · S)
```

`mfu_e2e_standard = flops_per_step / (peak_bf16 · world_size · step_time) * 100`, `peak_bf16 = 989.4 TFLOPS` (H100 SXM5 BF16 dense peak).

This form is numerically bitwise equivalent to Megatron-LM v15 `num_floating_point_operations` (`megatron/training/training.py:332+399+418+438`, including the L438–448 MTP norms/proj + the `(mtp_num_layers + 1)` logit factor at L448); you can copy that function's implementation directly, **`import megatron` is not allowed**. The L0 ref script `ref/reference/train_pure_mup_mtp.py` already implements the formula in this section; please keep the ours-side implementation consistent.

⚠️ **Do not fall back to the `12·L·H²·[bracket]` simplified form**: that form historically has three common deviations — (1) attention causal term written as `S/H` missing the ½ factor, (2) SwiGLU `gated_mult` taken as 2 instead of 3/2, (3) does not include MTP — net over-counting about 16% FLOPs, **making the reported MFU number too high**. Any change to the formula form must go through GEMM-enumeration numerical reconciliation.

### MFU formula — megatron backend

Aligned with the Megatron baseline's own MFU reporting; the same standard / precise dual-convention rule applies and the gate decision is based on `standard` with the 0–100 percentage scale described above.

## Precision alignment requirement (alignment–bitwise-perf mandatory bitwise; resume mandatory self-comparison bitwise)

**Milestones alignment to bitwise-perf must achieve bitwise exact match with the active backend's ref (`max_abs_diff == 0`).** Under megatron this is the Megatron baseline; under torch this is the pure-PyTorch ref. **The resume resume milestone must achieve reference vs resume self-comparison bitwise (`max_abs_diff == 0`)**, i.e., `save_checkpoint` + `load_resume_checkpoint` must be a lossless round-trip.

- Using 1e-7 or any numerical tolerance as a backstop is not allowed (alignment–resume are all mandatory).
- Using "FP32 to BF16 precision loss" as an excuse for alignment failure is not allowed.
- alignment–bitwise-perf must use exactly the same computation path, precision strategy, and kernel calling method as the active ref (under torch, including the complete switching sequence of the deterministic stack).
- If a discrepancy appears, it must be traced to the specific operator / computation step to find the root cause and fix it.
- Under megatron: TE kernels should use the same mode as the Megatron baseline.
- Recomputation (activation checkpointing) is **only a temporary backstop for OOM**, not a long-term optimization point. Under torch, recompute (`--recompute`) must follow the same path as ref to preserve bitwise during alignment–resume. The long-horizon memory-elimination targets and policy live in `long-horizon.md` §Secondary goal.
- The precision alignment process (alignment–bitwise-multicard) may ignore MFU; from bitwise-perf onwards an MFU gate is also required.

### Bitwise alignment debugging discipline

When a bitwise gate fails, dump→diff→bisect to the **first divergence** (the first point where `max_abs_diff != 0` in computation order); read the source only there. Never guess-and-edit. The set of failing keys is a **symptom map, not a diagnosis** — one upstream error contaminates every node downstream of it (tied embedding/LM-head amplifies a single upstream error across many keys), so never infer the culprit from which keys fail. Bisect along intermediate tensors that have a linear computation order. (Dump/diff code patterns: see the megatron alignment textbook §5.)


## Forbidden injection of baseline runtime state (alignment–bitwise-perf mandatory)

**In a bitwise gate script, the ref baseline and the self-developed implementation must run completely independently** (alignment–bitwise-perf all go through two completely independent subprocesses: the ref side runs `[ref].ref_capture_script`/`ref_script` separately via dispatcher bash, and the ours side is run in another subprocess by dispatcher pulling up `evals/scripts/eval_*.py`; the two sides compare via disk dump / stdout trajectories, and have no visibility into each other's runtime state). The following behaviors are forbidden:

- Taking baseline's `grad_norm` as the clip coefficient on the self-developed side.
- Copying baseline's bf16 weight snapshot into the self-developed model parameters or optimizer master weights.
- Injecting baseline's optimizer state (`m`/`v`/`step_count`) into the self-developed side.
- Feeding any baseline runtime intermediate result back into the self-developed side in any form to correct multi-step drift.

The two sides reach the same fp32 starting state via DIFFERENT paths: ref re-runs `init_weights(seed)` deterministically every launch, ours loads `canonical_state_fp32.pt`. Byte-equality at step 0 depends on the canonical having been dumped from the same `init_weights` version the ref currently runs.

The data path is the only **runtime** input genuinely shared: under megatron this is `DATA_PATH` (from alignment onwards both sides load the same Megatron binary independently via the Megatron `GPTDataset`; alignment uses single-step tensor dump, bitwise-singlecard+ uses multi-step trajectory); under torch this is `DATA_PATH_FILE` (modelbest_sdk weighted shard string; both sides independently load the same shard list via `ModelbestDataloader`; alignment single-step dump, bitwise-singlecard+ multi-step trajectory).

### `forge_init_ones` contract

Each gate in `@@FORGE_CONFIG_DIR@@/eval.toml` declares a required `forge_init_ones` field. `0` means no 1-D parameter element equals `1.0` (anti-cheat); `1` is textbook ones init. The engine must produce bit-equal output for any value of these parameters.

## Correctness script ref parameter consistency constraint

The parameters passed to the ref invocation in correctness scripts must satisfy:

**Megatron backend** (constructing the Megatron model):

1. `--use-distributed-optimizer`: must remain consistent (enabled).
2. `--micro-batch-size`: must come from the L0 ref script's `HARNESS_GATE` preset.
3. `--global-batch-size`: must come from the L0 ref script's `HARNESS_GATE` preset.
4. When modifying these Megatron parameters, the L0 ref script (`ref/reference/${ref_script}`) must be modified in sync; `@@FORGE_CONFIG_DIR@@/eval.toml` does not own the training spec.

**Torch backend** (invoking `train_pure_mup_mtp.py`):

1. `--micro-batch-size` / `--global-batch-size` / `--seed` / `--lr` / `--lr-warmup-iters` / `--lr-decay-iters` / `--lr-wsd-decay-iters` / `--weight-decay` / `--adam-beta1` / `--adam-beta2` / `--clip-grad` / `--init-method-std` / `--mup-base-hidden-size` / `--mup-emb-scale` / `--mup-depth-scale` / `--eagle-num-layers` / `--eagle-ce-loss-weight` / `--recompute` must come from the `HARNESS_GATE` preset of the L0 ref script (`run_16gpu_1000step_pure_mup_mtp.sh`).
2. alignment–resume bitwise gates must preserve the ref's default `--deterministic` (**`--no-deterministic` is not allowed**), which forces bitwise-singlecard–resume to use MBS=4 to avoid OOM under ref's deterministic stack; **in the long-horizon long-run performance optimization stage, deterministic mode is turned off on both the ref and ours sides** (this is a necessary condition for MBS=10 / GBS=80 to run on 80GB H100); long-train is a statistical gate (`mean rel < 1%`) robust to non-deterministic jitter; resume-gate-20 regression still runs per resume rules (keep deterministic + MBS=4).
3. When modifying these script parameters, synchronously modify the L0 ref script's `apply_harness_gate_preset` or the `PY_ARGS` array; `@@FORGE_CONFIG_DIR@@/eval.toml` does not own the training specification.
