# long-horizon — Long-running optimization (including operator fusion) + long-running statistical gate

> **Prereq**: resume PASS. **Postreq**: long-train PASS + resume-gate-20 PASS continuously → hand off to **production** (ours-only production long-train, the terminal stage1 milestone). long-horizon no longer ends stage1; it advances the loop to production.

## Goal

After resume completes the resume lossless round-trip, do the final long-running performance optimization, raising end-to-end `MFU(standard)` as high as you can, and validate via the long-train long-running statistical gate that the operator combination does not introduce systematic bias after many gradient updates. Whether this milestone has a dev-visible numeric MFU bar depends on the suite: if the rendered gate config (`workload/src/config/long-train.toml [cli]`) declares `mfu_e2e_target`, the long-train verdict gates on it; otherwise the gate verdict is loss-only — MFU is still measured and reported every run. In **both** modes, whether the achieved throughput is sufficient for hand-off is judged independently at PR review from harness-written telemetry — the loop advances the milestone on its own when that judgment passes (see §Finish signal); a dev-visible bar passing does not bypass or accelerate that review.

**long-horizon does not mandate end-to-end bitwise** (operator fusion changes the computation path); validation is jointly performed by the long-train long-running statistical gate + the constraint that the operator fusion precision path must not be downgraded (the fused operator's input/output dtype must be exactly the same as the baseline operator it replaces, see §long-horizon operator fusion precision path constraints below); `resume-gate-20` is also required to continuously pass bitwise (operator fusion must not break save/load).

## Secondary goal (memory)

**Optimize memory usage as much as possible and actively eliminate recomputation.** Activation recompute is only a compromise for deterministic + 80GB H100 OOM; it is not an optimization point to be retained long-term; on the premise that the long-train gate passes and MFU does not regress, **eliminating recomputation has higher priority than retaining the small amount of throughput recomputation buys.** The memory target is to run the full per-loop batch shape (`@@FORGE_CONFIG_DIR@@/eval.toml [evals.long-train].ours_env` MBS, with `grad_accum` derived per side) on 80GB H100 without relying on recomputation.

## Optimization ordering (default prior — profile may override)

The techniques below are a menu of *what* you may do; this section is
the default *order* in which to do them. The ordering principle is
simple: **changes move from "restructuring the whole hot path" to
"freezing the hot path", and later phases are progressively more
expensive to redo.** Doing them out of order forces rework (e.g.
re-capturing a CUDA graph after every fusion).

0. **Phase 0 — det-off flag audit (free MFU on the long-horizon det-off path).**
   long-horizon turns the deterministic stack OFF on both ref and ours
   (constraint 8), but library kernels frequently keep their own
   `deterministic` / safe-mode flag that defaults to ON and silently
   picks a slower reduction. Audit the in-use op stack for such
   flags and turn them off on the long-horizon det-off path.
1. **Phase 1 — measure-enable (memory + Python overhead).** Eliminate
   activation recompute and strip hot-path host-blocking calls
   (`.item()` / `.cpu()` / `torch.cuda.synchronize()`, blocking H2D →
   pinned + `non_blocking`; including the dataloader path). Goal: run
   `MBS=10` with **zero recompute**
   *and* a clean, trustworthy MFU measurement. The fusions that
   themselves unblock this goal (chunked/fused CE to kill the
   `[B·S, V]` fp32-logits peak; SwiGLU bwd-recompute to drop fp32
   intermediates) belong **here** — Phase 1 is defined by its *goal*,
   not by a "memory-only techniques" bucket. Until recompute is gone
   and host stalls are removed, the profile is polluted and every later
   decision rests on a bad measurement.
2. **Phase 2 — fuse and swap (operator fusion + Triton GEMM swap).**
   With measurement trustworthy, converge the kernel set: RMSNorm /
   RoPE / residual-add / remaining CE fusion, `_foreach_*` batching,
   **and replace the cuBLAS wgrad GEMM(s) (body and head) with an
   in-house Triton kernel preserving the dtype contract** (per §long-horizon
   stage operator freeze: bf16-in / fp32-acc / fp32-out; the Triton
   form reads bf16 directly and eliminates the `.float()` upcast tax
   that the cuBLAS-TF32 path pays). wgrad is usually the largest
   single non-attention kernel (15–25% of step time on the body + head
   combined). A Triton kernel's performance is **specific to the shape
   it was tiled for**: when applying an existing Triton kernel to a
   new shape with substantially different aspect ratio (e.g. M:K ratio
   inverted, M ≫ 10×K, strided vs contiguous contraction), do NOT
   infer "Triton loses on this shape" from a probe that reused the
   existing kernel. Design and probe a kernel tiled for the new
   shape's access pattern first. This must precede overlap and graph
   capture because both downstream phases are planned *around the
   specific set of kernels* — changing the kernel set after them
   invalidates their layout.
3. **Phase 3 — overlap (communication / compute).** Only meaningful once
   compute is already tight **and** a multi-GPU profile shows NCCL comm
   actually exposed on the critical path (see §Profiling cadence: read
   the OS-runtime block to distinguish real comm exposure from plain
   GPU-wait). Stream-level reorchestration of an un-converged kernel set
   is wasted work.
4. **Phase 4 — capture (CUDA graph).** Freeze the hot path last —
   only after the operator set has fully converged. This
   subsumes structural launch-overhead elimination (the ~50us ×
   hundreds-of-kernels-per-step Python/launch cost): once captured, any
   kernel change forces a re-capture, so graphing before the kernel set
   is stable means repeated rework.

**This ordering is a default prior, not a hard sequence — profile may
override it.** The §Profiling cadence rule is authoritative: if a
round's `summary.md` shows the current bottleneck sits in a later
phase (e.g. NCCL comm already exposed before fusion is exhausted), jump
to that phase. Never follow this order against profile evidence — that
just repeats the failure this section exists to prevent (burning rounds
on low-ROI work while the profile points elsewhere).

## Allowed optimization techniques (mix freely; the agent chooses the combination)

Not limited to the following categories; any solution that produces MFU benefit or memory benefit is encouraged. **Do not give up on small MFU lifts or memory benefits.**

1. **Operator fusion**: fuse adjacent operators into a single kernel. E.g., SwiGLU fusion, RMSNorm+Linear fusion, Fused Cross-Entropy (chunked), Residual-Add + RMSNorm, RoPE fusion, etc. **A forward-fused operator must be accompanied by an implementation of the same-named backward fusion** (see §long-horizon operator fusion precision path constraints below). The **operator freeze constraint** (only GEMM and the fused attention kernel) is in §long-horizon stage operator freeze constraint below; other operators may all participate in fusion.
2. **CUDA graph**: capture the hot path as a graph, eliminating Python-side overhead and launch overhead (~50us × hundreds of kernels per step). **Defer until #1/#7 are settled** — any kernel change after capture forces a re-capture. Multiple granularities can be captured separately — `forward` / `step` (fwd+bwd) / `step_full` (including optimizer) / `step_optimizer` (only optimizer + param sync) / `step_nccl_opt` (including NCCL comm). After startup, run N steps of warmup (after dataloader / autotune / NCCL handshake stabilizes) before capturing, to avoid capturing in warmup state. After capture, during replay, input tensors must be **written in place** to the same storage (cannot be reallocated); the dataloader / loss / param update must all comply.
3. **Communication / compute overlap**: let communication / compute / IO run concurrently on different streams. Common points —
   - **dgrad/wgrad ↔ grad reduce_scatter / allreduce concurrency**: NCCL grad allreduce / reduce_scatter is bucketed (~25 MB scale) + async hook; each layer's backward triggers this bucket's communication concurrently with the next layer's dgrad/wgrad upon completion.
   - **next-layer fwd ↔ wgrad GEMM concurrency**: the previous layer's wgrad GEMM and the next layer's forward run concurrently on different streams.
   - **ZeRO-1 param all_gather ↔ next-step fwd concurrency**: after optimizer step, the bf16 working copy's `all_gather` must overlap with the next step's forward — at minimum achieve "all_gather of layer i+1's param concurrent with fwd of layer i", and ideally hide the full step-level all_gather completely under the fwd critical path. Pairs with ZeRO-1 below.
   - **optimizer step ↔ dataloader prefetch / H2D concurrency**: the optimizer step runs concurrently with the next step's dataloader prefetch + H2D on different streams.
4. **Memory compression / eliminating recomputation**: drop saved tensors (e.g., SwiGLU change `silu.out` to bwd-internal recompute rather than saved; CE does not materialize V-dim logits; RMSNorm bwd uses `rstd` to back-derive rather than save normalized values); fused kernel keeps intermediate results in registers/SRAM; resident buffer pool + shape-stable static tensor allocation; remove redundant `.contiguous()` / temporary `.clone()`; fp32 grad accumulator pre-allocated + `zero_()` reuse; ZeRO-1 master state sharding saves `(DP-1)/DP` of fp32 state; dtype conversion path simplification (merge redundant bf16↔fp32 casts). **Lower the activation checkpoint layer count (`--recompute-num-layers`) whenever possible; close it down to 0 when possible**; this is the last valve of the memory budget, not an active optimization point.
5. **Python overhead reduction**: remove unnecessary `torch.cuda.synchronize()` / blocking CPU-GPU calls from the hot path (`.item()` / `.cpu()` / `torch.cuda.synchronize()`); pinned-memory + `non_blocking=True` H2D; cache scheduling metadata; merge redundant scheduling.
6. **ZeRO-1 distributed optimizer**: shard the fp32 master state (`master_weight + Adam m + Adam v`) onto DP ranks; each card holds only `1/DP_WORLD_SIZE` share; NCCL communication switches from grad allreduce to **`reduce_scatter` (grad → this rank's own shard) + `all_gather` (synchronize the updated bf16 working copy back to all ranks)**; the optimizer step only updates its own shard, saving `(DP-1)/DP` of master state HBM traffic + `(DP-1)/DP` of optimizer compute. Combined with `reduce_scatter` / `all_gather` overlap with fwd (see §3 Overlap above) + sharded fused Adam (per-shard fused clip + AdamW + master→bf16 cast). **Compatibility with `resume-gate-20`**: save/load must support shard reassembly back into the full master state; otherwise resume lossless round-trip will break.
7. **Kernel implementation swap**: replace cuBLAS GEMM with in-house Triton preserving the dtype contract (see §long-horizon stage operator freeze constraint).
8. **Dataloader async**: background-thread `next(dl)` + bounded queue to hide periodic shard refill (impact visible only at long horizon, smoke cannot see).

## Gate system (both must pass)

1. **Long-running statistical gate**: `bin/harness run long-train` — the training spec is defined by the L0 ref script's `HARNESS_GATE=long-train` preset; the authoritative shape (WORLD_SIZE / MICRO_BATCH_SIZE_OVERRIDE / GLOBAL_BATCH_SIZE_OVERRIDE / NUM_STEPS_OVERRIDE / GATE_WINDOW_*, with `grad_accum = GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE × WORLD_SIZE)`) is the SSOT under `@@FORGE_CONFIG_DIR@@/eval.toml [evals.long-train].ref_env` (`seq_length` / `seed` come from the L0 ref script's preset). Decision:
   - **Pointwise relative loss (the gate verdict)**: `mean(|ours_i - ref_i| / |ref_i|)` over the gate window must stay below `[evals.long-train].loss_rel_threshold` (the gate window is declared in `gate_metadata.json`).
   - **Average MFU**: from `warmup_steps` onward the gate computes `avg(mfu_e2e_standard)` and reports it in the summary and metrics. If the rendered gate config declares `mfu_e2e_target`, the verdict also gates on it; otherwise the harness does not compare it against a floor. Either way, throughput sufficiency for hand-off is judged at PR review on milestone declaration. Record the number in `workload/notes/perf_log.md` every round.
2. **Resume regression gate**: `bin/harness run resume-gate-20` must keep passing bitwise — any operator fusion or system-level optimization must not break the lossless round-trip of save/load.

baseline data: long-train obtains the ref trajectory by dispatcher `shell-exec`ing the L0 ref script (`ref/reference/${ref_script}`) each time the gate is called, comparing synchronously with the ours trajectory (see `README.md`). **There is no frozen JSON, no sha anchor, no audit table**. Modifying the ref script = modifying the baseline truth; must go through PR review.

## Required output

Every fusion point / every system-level change must be recorded independently in `workload/notes/perf_log.md`, explaining the change, the impact on MFU, and the regression results of `resume-gate-20` and `long-train`.

## Loss-drift warning policy

`loss-gate-200` and `long-train` summaries surface `signed_mean` and a `drift_warning` flag. `signed_mean ≈ 0` (within ±0.005 of `|baseline|`) is the expected det-off signature; substantial deviation or monotonic same-sign growth across the gate window is real numerical bias, not run-to-run jitter. Any commit whose summary shows `⚠ DRIFT` must either attribute the bias to a specific code change in this round's `perf_log.md` (with an argument why it is bounded under further training) or open a bisect TODO and revert the suspect change before declaring `STAGE_STATUS: finished`.

A small `signed_mean` does NOT imply small `pointwise_mean_rel`. Always cross-check both: `signed_mean` rules out directional bias; `pointwise_mean_rel` rules out symmetric high-frequency divergence.

## Negative-result discipline

A negative probe falsifies only the implementation × shape × kernel-mix tested, never the broader technique — phrasings like "this lever is dead" / "<category> is frozen" / "host overhead off the table" / "<path> at its floor" are FORBIDDEN in `perf_log.md` unless invariance across all allowed swaps is documented (mirror of `constraint.md` §FORBIDDEN positive ceilings).

## Profiling cadence (hard rule)

Same shape as bitwise-perf but the snapshot label is long-horizon-scoped. Every round whose
commit touches the perf hot path
(`workload/src/training_engine_tensor/{forward,backward,kernels,triton_kernels,optimizer,nccl,train_loop,dataloader,parameters,config}.py`)
MUST run, after the smoke gate has PASSed:

    bin/harness run profile-snapshot M6_round<N>

`profile-snapshot` is the only suite that wraps rank 0 with `nsys profile`.
The commit gates (`long-train`, `resume-gate-20`, `loss-gate-200`) never
carry profiler overhead — the gates' loss_rel verdict and reported MFU stay clean.

**CUDA graph × nsys caveat**: if the gate hot path uses a captured CUDA graph, verify `profile-snapshot`'s graph state matches the gate before trusting per-kernel rankings — default nsys CUPTI folds the graph into one opaque `cudaGraphLaunch` event, so engines that auto-disable the graph under nsys end up profiling a different kernel mix from what the commit gate actually runs. Either wrap nsys with `--cuda-graph-trace=node` to keep the graph on, or add a committed-path busy-probe (CUDA-event timed, no profiler) as the source of truth.

The long-horizon `summary.md` exposes signals not present in bitwise-perf: (a) the
top-15 GPU kernel ranking (whether comm / GEMM / attention dominates,
critical for §3 Overlap), (b) the GPU memory-op block (HtoD/DtoH timing,
critical for §Secondary goal — eliminate recompute), (c) per-kernel
Δ vs round N-1 (direct measure of this round's fusion/overlap benefit),
and (d) the top-N **OS-runtime** block (host-side blocking calls:
`pthread_cond_wait` / `poll` / `futex` etc.). All time columns are
**per-step** (window total divided by `profiled_steps`, shown in the
header) so they compare directly against `step_time_ms`; the row
`instances` / `calls` counts stay window-cumulative. Pick the category with the largest MFU headroom first by mapping profile
signals to §Allowed optimization techniques: non-frozen kernel > 15% step
→ #1 / #7; `cudaStreamSynchronize` > 100 ms/step → #5; `cudaLaunchKernel`
> 500 ms/step with GPU ≥ 90% busy → #2 (only after #1/#5/#7); OS-runtime
top `nccl*` → #3, only `pthread_cond_wait`/`poll` → NOT #3
(compute-bound); long-horizon periodic step spikes → #8.
Use these signals to choose the next round's direction; do not pick
fusion targets by intuition. Before committing this round, list ≥2 candidate levers with their estimated upside (ms saved → pp MFU) and pick the largest; an unestimated direction is the most common round-waste. Cite `summary.md` by path and quote at least
one `Δ from M6_round<N-1>` line in the round's `perf_log.md` entry.

## Constraints

1. The training spec must not be arbitrarily reduced (`grad_accum` is computed from `@@FORGE_CONFIG_DIR@@/eval.toml [evals.long-train].ref_env` as `GLOBAL_BATCH_SIZE_OVERRIDE / (MICRO_BATCH_SIZE_OVERRIDE × WORLD_SIZE)` and is a hard constraint aligned with the baseline).
2. "Raising `loss_rel_threshold` to let the gate pass" is not allowed — threshold modification must go through PR review, and `workload/notes/perf_log.md` must explain whether the baseline has drifted.
3. Changes to `job_id` and `config` blocks in `gate_metadata.json` = changes to gate semantics; the PR must surface this diff.
4. A full long-train run takes a substantial wall-clock budget × (DP from `@@FORGE_CONFIG_DIR@@/eval.toml [evals.long-train].ref_env.WORLD_SIZE`) × H100 (see `[evals.long-train].timeout_s`); ensure the GPU resource pool has sufficient quota; do not reduce `grad_accum` below the config-derived value for speed.
5. The compute dtype of fused operators must not be lower than the precision of the baseline operator they replace (see §long-horizon operator fusion precision path constraints below).
6. Forward-fused operators must have corresponding backward fusion.
7. Ref `micro_batch_size` follows `@@FORGE_CONFIG_DIR@@/eval.toml [evals.long-train].ref_env.MICRO_BATCH_SIZE_OVERRIDE`; ours follows `[evals.long-train].ours_env.MICRO_BATCH_SIZE_OVERRIDE`. Both sides keep the same `GLOBAL_BATCH_SIZE_OVERRIDE`; `grad_accum` is derived per side.
8. **(torch backend only)** When running training in long-horizon, deterministic mode must be off (both ref and ours turn off synchronously; otherwise ref itself cannot run the baseline); currently the torch ref script defaults to `--deterministic`, so either the agent injects `--no-deterministic` via a hook at the dispatcher / ref script layer, or the agent temporarily local-patches the ref script to run this round (the patch cannot be committed; the ref is the baseline truth). `resume-gate-20` still runs per resume rules (deterministic + the resume MBS) for bitwise regression.

## Finish signal

**long-horizon has no dev-side finish signal.** Do not emit `MILESTONE_STATUS: long-horizon PASS` — the line is retired and carries no effect. The hand-off decision is made entirely at PR review, from harness-written gate telemetry: between rounds the harness itself runs the full `long-train` gate every few smoke runs (its artifacts under `.artifacts/` are normal gate outputs — do not treat an unexpected long-train result as something you launched or must re-run), and the review agent independently judges whether the accumulated end-to-end throughput is acceptable for hand-off (see `review_stage1.md`). When it is, the loop advances to **production** on its own; if a round starts and the active milestone has become production, simply proceed under `production.md`. **long-horizon does not emit `STAGE_STATUS: finished`** — production is the terminal stage1 milestone and owns that signal.

Your job every round is therefore unchanged and simple: keep raising MFU while keeping the hand-off bundle green — `long-train` (loss), `resume-gate-20` bitwise, and `perf-bitwise` (no regression) — and keep recording the numbers in `workload/notes/perf_log.md`. A broken regression gate blocks the hand-off no matter how good the throughput is, so fixing one always outranks further optimization. There is no benefit to grinding past the point of diminishing returns (per §Profiling cadence, estimate your levers every round) — but the decision to stop is not yours to make; continue producing honest rounds until the milestone advances.

## long-horizon operator fusion precision path constraints

> **The compute dtype of the fused operator must not be lower than the precision of the baseline operator it replaces.**

The overall precision is jointly validated by `long-train` (pointwise relative loss under `[evals.long-train].loss_rel_threshold`) + `resume-gate-20` bitwise; no operator-level FP64 numerical validation is performed. **Forward-fused operators must implement the corresponding backward fusion of the same name.**

## long-horizon stage operator freeze constraint (only GEMM + Attention two classes)

After entering long-horizon, **only the calling interface and mathematical behavior of the following two classes of low-level operators are frozen and forbidden to modify** (this does not conflict with operator fusion — fusion points are all in their upper layer / bypass):

| Frozen operator | Location | Description |
| --- | --- | --- |
| GEMM | **Megatron**: the GEMM implementation used when resume passes (by default assumed to be the TE-routed cuBLAS matrix multiplication). **Torch**: the wgrad **dtype contract** at resume PASS — see `constraint.md` §FP32 spec item 4 and read the ref's model source for the exact path. The contract is FP32 wgrad output, FP32 `main_grad`, no per-microbatch BF16 round-trip. | **Frozen** (the call must keep numerically equivalent semantics): input/output dtype contract, transpose mode, accumulator dtype, the FP32 `main_grad` buffer semantics. **Allowed in long-horizon**: replacing the cuBLAS dispatch with an in-house Triton that produces a numerically equivalent result (same dtype contract, no regression on `loss-gate-200` / `resume-gate-20`). **Forbidden under torch**: introducing a new training-framework dependency (`transformer_engine`, xformers,cutlass, etc.; see `constraint.md` §Forbidden); changing the dtype contract itself (e.g. BF16 wgrad output, FP16 accumulator); any swap that fails `loss-gate-200` or `resume-gate-20`. |
| Attention | **Megatron**: the fused attention implementation used when resume passes (by default assumed to be the TE fused attention call); the GQA mode has been validated in alignment–bitwise-perf; replacing the attention implementation or changing its calling parameters is forbidden. **Torch**: the attention implementation in use at resume passing time; the attention kernel and GQA layout actually used in the repo at resume passing time are the long-horizon baseline; backend switching or fusion inside long-horizon is allowed as long as long-train + resume-gate-20 still PASS, but new training-framework dependencies (TE / xformers / a complete in-house attention stack, etc., non-already-used stack implementations) are not allowed |

> The "Location" column above lists the baseline default assumption; the authority is the **GEMM / Attention implementation actually used in the repo when resume passes** — during alignment–resume, the agent may replace the underlying call stack; at long-horizon it freezes based on the implementation measured at that time. Under torch: the GEMM interface is fully frozen; the SDPA backend can be switched as a whole but the implementation source is frozen.

The upper-layer code that calls these two classes of operators in long-horizon (layer assembly, training loop, optimizer orchestration, communication, checkpoint, fusion of other operators, etc.) is all within the writable scope of this milestone.
