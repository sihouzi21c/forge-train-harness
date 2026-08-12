# Stage 2 Subagent Playbook

This file is the common methodology, constraints, pitfalls, Resource Contract, and per-round single-round decision-tree protocol used by a **per-op subagent** when progressing through a single round of decision-tree in its own worktree. During M2.1 dispatch, the main agent cats this entire file into the subagent prompt — together with each op's own `PROMPT.md` (which contains the full text of `reference/<op>.md`), it constitutes the entire context of the subagent.

> This file **serves only the subagent**. stage2.md is the main agent's orchestration SSOT; reference/*.md is domain knowledge distributed by op type; this file is the cross-op shared engineering discipline + pitfalls + per-round single-round decision-tree protocol.

### Initial Files in the Worktree (created by the M1 main agent; you can read all of them)


| File              | Purpose                                                                            | Who reads                                                                |
| --------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `PROMPT.md`     | This op's work instructions: current baseline call location / I/O contract / profile numbers / domain reference (`reference/<op>.md` full text embedded) | You (main input) + review agent                                          |
| `BASELINE.md`   | Review-side SSOT, same-source as PROMPT.md "Current I/O Contract / Current Profile"; derived from `profile_result.json` | review agent (`review_stage2.md` Stage2 Check D) verifies whether your drop-in changed the I/O contract |
| `kernel.py`     | pass-through skeleton (directly re-exports baseline functions)                     | dispatcher's `else` branch; you replace it incrementally across rounds   |
| `register.toml` | `env_var` / `default = "baseline"` / `available`                                   | dispatcher reads on startup; in the merge wrapper you change `default` to the new version |
| `test_op.py`    | Basic test skeleton written by main agent (contig + real stride + FP64 comparison), passes via pass-through | You (run each round, can add cases)                                      |
| `notes.md`      | empty; you write assumptions / experimental results / decision basis each round    | You + review agent                                                       |
| `__init__.py`   | empty                                                                              |                                                                          |


> **Modification boundary**: you can only modify files under `workload/ops/<name>/` — `BASELINE.md` counts as review SSOT and generally is not modified; the others can be changed as needed. Modifying the "Current I/O Contract / Current Profile" fields of `BASELINE.md` is equivalent to **changing the drop-in contract** — it must strictly correspond to the actual behavioral change of your kernel; otherwise the review agent will judge it as contract drift. Other paths (`workload/src/` / `harness/` / `prompt/` / ...) are read-only.

## Tech Stack

**Must use CUDA C++** (via `torch.utils.cpp_extension`), CUTLASS or CuteDSL. TE / cuBLAS may serve as baseline reference; they must be replaced with an in-house CUDA kernel.

Do not use Triton or PyTorch native functions (`torch.mm`, `F.silu`, `F.cross_entropy`, `F.scaled_dot_product_attention`, etc.) as the optimized kernel implementation. The goal is to control thread/block/shared memory/Tensor Core directly.

## Drop-in Replacement Constraints (mandatory)

The optimized kernel **must** be a strict drop-in replacement:

1. **Input contract frozen**: the tensor parameters received by the wrapper function (shape, dtype, device, stride, contiguity) must be exactly identical to the baseline wrapper being replaced.
2. **Output contract frozen**: the returned tensor must have the same shape, dtype, device, and semantic meaning. Changing layout is not allowed.
3. **Saved tensor contract frozen**: tensors stored in the `saved` dict for backward use must keep the same key, shape, and dtype.
4. **Internal freedom**: thread/block/tile layout, shared memory, kernel launch count, FP32 accumulation strategy, register pressure — all free.
5. **Defensive I/O**:
  - All inputs `.contiguous()` before entering CUDA kernel; do not rely on `empty_like(non_contig)`.
  - Allocate output with `torch.empty(N, D, ...)` and then `.view()` / `.reshape()` to the target shape; do not use `empty_like`.
  - Unit tests cover real stride scenarios (e.g. the K / V slice in ref `TransformerLayer.forward` cut out from `qkv[..., nq_per_kv*d : nq_per_kv*d+d]` after `qkv.view(B, S, NUM_KV_HEADS, (nq_per_kv+2)*d)`: a **non-contig strided view** produced by last-dim offset slicing), not only `torch.randn().bf16()`.

### Wrapper Defensive Programming Template

```python
# Bad example (common source of bugs)
def optimized_op(x):
    out = torch.empty_like(x)              # behavior unpredictable when non-contig
    kernel[(N,)](x.reshape(N, D),          # may trigger implicit contiguous copy
                 out.reshape(N, D), ...)   # reshape may not share out's storage
    return out

# Good example
def optimized_op(x):
    x_in = x.contiguous()                  # explicit contig-ification
    N, D = x_in.numel() // x_in.shape[-1], x_in.shape[-1]
    out_2d = torch.empty(N, D, dtype=x.dtype, device=x.device)  # independent contig
    kernel[(N,)](x_in.reshape(N, D), out_2d, ...)
    return out_2d.view(*x_in.shape[:-1], D)
```

Three principles: (1) explicitly `.contiguous()` all inputs before entering kernel; (2) directly allocate an independent contig output with `torch.empty(N, D, ...)`; (3) avoid the `empty_like(x).reshape(...)` combination — the storage relationship is a hidden pitfall.

## test_op.py Hard Constraints


| Item        | Practice                          | Bad example                    |
| --------- | -------------------------------- | ----------------------------- |
| Baseline    | new kernel vs baseline signed diff | only compute each vs FP64     |
| Data source | real stride scenario (non-contig view) | only use `torch.randn().bfloat16()` |
| Stride coverage | test both contig + non-contig cases | only test contig          |
| Direction statistics | print `signed_mean` / positive-negative ratio | only print `abs_max` / `abs_mean` |
| State loop  | multiple calls + simulate optimizer modifying inputs | only call once          |


### FP64 Reference Cheatsheet

Do not derive the math formula from scratch; directly use PyTorch standard functions casting inputs to FP64:


| Operator category | FP64 reference syntax                                                       |
| --------- | ------------------------------------------------------------------------ |
| GEMM      | `torch.mm(a.double(), b.double())` (or `bmm` / `matmul`, aligned with the baseline call form) |
| Attention | `F.scaled_dot_product_attention(q, k, v, is_causal=True)` with all `.double()` |


Two lines of defense:

1. `_fp64_reference()` must faithfully reproduce the baseline math sequence, only with precision swapped to FP64.
2. Runtime self-check: `baseline_stats.mean < 0.01`. The pass-through kernel and FP64 reference should have a mean relative error at the 1e-3 level; > 0.01 almost certainly indicates the FP64 reference is computed incorrectly.

### Statistics helper inline

Comparison dataclasses like `PrecisionStats(mean, p95, max, signed_mean, num_elements)` should be written inline in `test_op.py`. The statistical contract ("the new kernel is no worse than the old kernel on `mean / P95 / max` relative errors and `abs(signed_mean)`") is your responsibility to implement — `test_op.py` itself decides `[PASS]` / `[FAIL]` and calls `sys.exit(1)` accordingly.

### Trust Model: local precision tests are not harness gates

`test_op.py` is **your own** development feedback loop — the harness does not read `[PASS]` / `[FAIL]` from stdout, does not read the exit code, and does not enforce any precision threshold. If you print a fake `[PASS]` line to stdout, there is no harness side effect — this path does not lead to merge.

What actually determines merging is `bin/harness run op-long <name>` (ref-script runs baseline trajectory, ours runs ours; the dispatcher itself parses `[LOSS] step=` lines and computes 200-step mean rel diff); you cannot insert instrumentation in between. So:

- **Write `test_op.py` seriously** — it is the sole local feedback tool in phase A (iteration) for you to judge whether this optimization path is actually viable.
- **Do not forge evidence** — forging a test pass only causes you to hit an unpassable commit when running op-long, burning GPU + triggering the safety net's cumulative op-long FAIL count, and ultimately being kicked out anyway.
- **After op-long FAIL, repair or exit depending on the situation**: each op-long FAIL increments the cross-round safety net count; once the cumulative count reaches `STAGE2_MAX_OP_LONG_FAILURES`, the next round's startup takes the 3a exit (worktree and branch preserved for review/human review, see "Failure Retention Policy" section below). In `notes.md`, write down each "self-judged convergence → op-long PASS / FAIL → decision" clearly so that the reviewer can retrace.

## Experiment Discipline

1. **Change only one thing at a time** — do not modify multiple variables simultaneously; isolate the source of impact, otherwise attribution is impossible.
2. **Correctness over performance** — a fast kernel that cannot pass the precision gate has no value.
3. **Identify bottlenecks before acting** — use profiling tools (NCU / nsys / torch.profiler) to localize bottlenecks; do not guess by intuition.
4. **Record experimental results promptly** — each round write to `notes.md`: what was changed, what the hypothesis was, what the result was, the next direction. This is the sole medium for cross-round knowledge transfer.
5. **Stay reproducible** — each experiment should be reproducible via the same command. Do not rely on temporary environment variables or manual operations.

The two operator categories in Stage 2 (FlashAttention + various GEMM sites) are both compute-bound matrix-multiply types; **the Tensor Core path must be reached** to achieve the target MFU. A naive kernel + thread/block tuning is a transitional phase (verifying the drop-in interface is correct), not the endpoint.

## Profiling Diagnostics

### NCU Key Metrics


| Metric                                | Meaning                              | Focus                                  |
| ------------------------------------- | -------------------------------- | ------------------------------------- |
| `smsp__inst_executed_pipe_tensor.sum` | Tensor Core instruction count    | > 0 means TC was actually used        |
| SM Throughput %                       | compute utilization              | low = possibly memory-bound or launch overhead |
| DRAM Throughput %                     | HBM bandwidth utilization        | high = memory-bound; optimize access patterns |
| Occupancy                             | concurrent warp ratio            | low = register or shared memory pressure |
| Stall reasons                         | long_sb / short_sb / math / wait | localize specific bottleneck type     |


### Memory-bound vs Compute-bound Judgment

- **Memory-bound** (DRAM high, SM low): reduce global memory accesses — operator fusion, shared memory caching, vectorized loads (float4)
- **Compute-bound** (SM high, DRAM low): improve compute efficiency — Tensor Core utilization, tile-size tuning, reduce redundant computation

## per-round single-round decision-tree protocol (a subagent process does one round only)

**Process model core constraint**: your subagent process **does one linear decision-tree round per startup and then exits**. Cross-round accumulated progress is driven by the outer `agent-loop.sh` main loop making multiple fan-outs. In-process you can only do one thing and leave; when the next main-loop round comes, the main agent restarts a fresh process for you to continue.

**Where the state lives**: after the subagent process exits, all state lands in two places —

1. **git history**: in `git log --oneline` are all your prior commits per round. HEAD is always the current best. **Strictly forbidden** to `git reset --hard <historical SHA>` to roll back a PASS-ed optimization; **strictly forbidden** to `git revert PASS commit`; to retain branch experiment snapshots, you can `git tag stage2/op/<name>/round-<N>-experiment`.
2. **`workload/ops/<name>/notes.md`**: the natural-language paragraphs you wrote in previous rounds. The first thing the next round's subagent does on startup is `cat notes.md` to rebuild state.

### Step 1 after startup: rebuild context and state

**Parse** `notes.md` **yourself** to rebuild state:

- **Cumulative op-long FAIL count**: count anchors like `OP_LONG: FAIL` in `notes.md`. This protocol does not mandate a row format, but it is recommended you always use a consistent searchable anchor (e.g. "`OP_LONG: FAIL #<N>`") for easy grep on each round's startup.
- **Last round's self-judged conclusion**: from the end of `notes.md`, extract a keyword — `STILL_ITERATING` (last round you thought there was still room) / `READY_FOR_OP_LONG` (last round you self-judged convergence and will run op-long this round) / `OP_LONG_FAILED` (last round you ran op-long but it FAILed; you need to repair this round) / does not exist (this is the first round).
- **Whether HEAD is waiting on op-long acceptance**: look at `git log` to see whether after the last `OP_LONG: ...` commit there is any non-`--allow-empty` real kernel commit; if no → HEAD is the unverified point left by last round's self-judged convergence.

### Step 2 after startup: three-choice decision tree

Choose one of the following three based on the state rebuilt in Step 1 to decide which branch to do this round:

#### 3a. Safety Net Triggered — cumulative op-long FAIL ≥ `STAGE2_MAX_OP_LONG_FAILURES`

The environment variable `$STAGE2_MAX_OP_LONG_FAILURES` is the cap (injected by agent-loop.sh from the SSOT `config/eval.toml` `[automation.stage2].max_op_long_failures`). When hit:

1. Write a paragraph in `notes.md`:
  ```
   ## SAFETY_NET_TRIGGERED (round <N>)

   Cumulative op-long FAIL <N> times (cap=<K>); abandoning this op; retaining worktree for human review.
   Brief attribution per failure:
   - FAIL #1: <one line>
   - FAIL #2: <one line>
   - FAIL #3: <one line>
  ```
2. `git commit --allow-empty -m "stage2 <op>: safety net triggered, abandoning"`.
3. **Do not actively merge** — exit directly. `bin/harness run op-status` sees `worktree still exists + register.toml.default == baseline` → marks `failed`; in the next round, the main-loop dev agent skips this op, awaiting review/human review.

#### 3b. Run op-long for acceptance — when ready to merge

Merge requirements (both must be met):

- Optimization space exhausted; nothing more can be optimized. Do not abandon any possibility that can still be optimized.
- End-to-end speed (`mfu_e2e_standard`) must exceed the baseline implementation (pass-through calling cuBLAS / SDPA). A kernel below baseline is never merged; go back to 3c and keep optimizing.

If the previous subagent has already finished the kernel, self-judged the optimization space exhausted, and left a commit waiting for this round to run op-long. This round you do:

1. **Entry self-check**: run `flock -s "$REPO_ROOT/.artifacts/locks/stage2-gpu.lock" python workload/ops/<name>/test_op.py` to confirm the local test on the current HEAD `[PASS]`. If FAIL (last round's subagent self-judgment was off), fall back to 3c for repair.
2. **Run op-long**: `bin/harness run op-long <name>` (the suite has `LOCK_EX` internally and waits for all sibling subagents' `LOCK_SH` to release).
3. **PASS** → immediately enter the "self-driven merge into main" section's merge wrapper below (`flock + git merge` + `register.toml.default` bump + worktree clean); after merge exit. This op exits the stage2 follow-up rounds.
4. **FAIL** → append to `notes.md`:
  ```
   ## OP_LONG_FAILED (round <N>)

   OP_LONG: FAIL #<new cumulative count>
   Measured [100,199] step mean rel diff = <value> (cap = dense_training.toml [evals.op-long].rel_diff_threshold)
   Loss deviation signal: <dispatcher output snippet>
   Failure-mode attribution: <systemic bias / numerical instability / certain token triggers / ...>
   Repair candidates for next round: <one or two specific next-step actions>

   Next-round self-judgment: STILL_ITERATING (needs repair)
  ```
   `git commit --allow-empty -m "stage2 <op>: op-long FAIL #<N>, will repair next round"` and exit. When the next main loop round comes, the subagent sees `OP_LONG: FAIL` cumulative count has not hit cap + self-judgment `STILL_ITERATING` → goes to 3c for repair.

#### 3c. Phased progress (default branch) — neither safety net triggered nor ready for op-long

Enter under the following situations: first round with no commits yet / last round self-judged `STILL_ITERATING` / after last round's op-long FAIL, this round is for repair / 3b entry self-check failed and fell back. This round do one "linear single-step progress":

1. **Optimize the kernel**.
2. **Write a paragraph in `notes.md`**:
  ```
   ## Round <N>

   What was done: <one or two sentences>
   Hypothesis: <which bottleneck targeted, expected benefit>
   Local test result: [PASS] / [FAIL]
   mfu_e2e: <value>% (last round <value>%; baseline <value>%)
   Key observations: <key changes observed via NCU/profiler>

   This round self-judgment: STILL_ITERATING (reason: still have <X> / <Y> / <Z> directions worth trying)
   OR
   This round self-judgment: READY_FOR_OP_LONG (reason: roofline already saturating / NCU TC utilization saturated / remaining candidates cost > benefit / ...)
  ```
3. **commit**:
  - Local test PASS → `git commit -m "stage2 <op> round-<N>: <short summary of what was done>"`.
  - Local test FAIL and cannot fix this round → `git stash` / `git checkout -- workload/ops/<name>/kernel.py` to revert kernel changes so HEAD falls back to last round's PASS state; still commit the `notes.md` paragraph (incl. this round's self-judgment `STILL_ITERATING`), `git commit --allow-empty -m "stage2 <op> round-<N>: failed attempt rolled back, will retry"`.
4. **Exit**.

### Step 3 — Exit

After the subagent process completes one of 3a / 3b / 3c, it **exits directly** — the dev agent `wait`s for all subagents, writes the op-status summary + commits + the review agent takes over. When the next main-loop round starts, in M2.1 the dev agent runs the op-status filter to determine whether to keep dispatching this op.

### After 3b PASS: self-driven merge into main (merge wrapper)

After op-long PASS, use the shell below to self-drive the merge:

```bash
set -euo pipefail
OP="<name>"
REPO_ROOT="$(cd "$(git rev-parse --git-common-dir)/.." && pwd)"
WT="$REPO_ROOT/ops_worktree/$OP"
LOCK="$REPO_ROOT/.artifacts/locks/stage2-main.lock"
mkdir -p "$(dirname "$LOCK")"

(
  flock -x 9                                       # global unique main lock

  # 1) 4th path allowlist: rescan diff before merge
  cd "$REPO_ROOT"
  MERGE_BASE="$(git merge-base HEAD "stage2/op/$OP")"
  git diff --name-only "$MERGE_BASE" "stage2/op/$OP" \
    | python -c "import sys; from harness.framework_guard import op_path_violations; \
paths=[l.strip() for l in sys.stdin if l.strip()]; \
v=op_path_violations('$OP', paths); \
print('\n'.join(v), file=sys.stderr) if v else None; sys.exit(1 if v else 0)"

  # 2) ff-merge into main; on failure fall back to no-edit merge commit
  git merge --ff-only "stage2/op/$OP" \
    || git merge --no-edit "stage2/op/$OP" -m "stage2: merge $OP best snapshot"

  # 3) In the same atomic commit, bump register.toml(default = <new_version>)
  #    Recommend tomllib (read) + tomli_w (write) to keep TOML structure stable;
  #    sed easily breaks multi-line [section] / comments / whitespace,
  #    triggering _validate_op_artifacts errors.
  OP_REG="workload/ops/$OP/register.toml"
  python - "$OP_REG" <<'PY'
import sys, tomllib, tomli_w, pathlib
p = pathlib.Path(sys.argv[1])
d = tomllib.loads(p.read_text())
new_ver = "v1"  # or the next unused vN
d["default"] = new_ver
if new_ver not in d.get("available", []):
    d["available"] = list(d.get("available", [])) + [new_ver]
p.write_text(tomli_w.dumps(d))
PY
  git add "$OP_REG"

  # 4) Clean up worktree and branch
  git worktree remove --force "$WT"
  git branch -D "stage2/op/$OP"
) 9>"$LOCK"
```

Key points:

- **One global lock** `stage2-main.lock`: any subagent queues via `flock -x 9` before merge; main ref advances atomically. Independent from the op-long lock.
- **Bump `register.toml` yourself**: refer to step 3 example above; `tomllib` (read) + `tomli_w` (write) to keep structure stable. Change `default` from `baseline` to `v1` (or the next unused vN), append the new version to `available`, `git add` + `git commit` together into the merge commit.
- **4th path allowlist**: `git diff … | python -c …` reuses `harness.framework_guard.op_path_violations` — the merge-time preflight after L1/L2/L3, catching `--no-verify` leaks; when violations are non-empty, `set -e` exits immediately, and the main lock is released as the subshell exits.
- The merge wrapper merges the current HEAD; between op-long PASS and merge you can do diagnostics, but you may no longer modify the kernel.

### Failure Retention Policy (3a triggered / any step of merge wrapper fails)

Any "non-merging exit" path — 3a safety net triggered, or any step of the merge wrapper above failing — requires retention of the worktree + branch for review/human review:

- Keep `ops_worktree/<name>` and the `stage2/op/<name>` branch as-is (do not delete the worktree; do not switch branches).
- `notes.md` leaves the complete cross-round iteration record + failure attribution (incl. dispatcher output snippets for each op-long failure, the `set -e` exit position / error message for the merge failure).
- Exit directly.

`bin/harness run op-status` sees `worktree still exists + register.toml.default == baseline` and automatically marks `failed`. Once marked failed, the main-loop dev agent skips this op in M2.1 dispatch (the M2.1 step-1 jq filter excludes `failed`), waiting for review / human review to decide the next step (re-run / abandon / change constraints).

### Resource Contract: local test_op.py and op-long are mutually exclusive

When fan-out is concurrent, multiple sibling subagents simultaneously run their own 3c (local tests) — being on different GPUs cuda:0/1/2 is OK, but when any subagent enters 3b to run op-long, it must grab all 8. The flock read-write lock protocol resolves this:

- **In 3c, wrap `test_op.py` with `LOCK_SH`** (shared lock):
  ```bash
  REPO_ROOT="$(cd "$(git rev-parse --git-common-dir)/.." && pwd)"
  mkdir -p "$REPO_ROOT/.artifacts/locks"
  flock -s "$REPO_ROOT/.artifacts/locks/stage2-gpu.lock" \
    python workload/ops/<name>/test_op.py
  ```
  Multiple subagents holding `LOCK_SH` simultaneously are not mutually exclusive; they each use the GPU specified by their own `CUDA_VISIBLE_DEVICES` and do not conflict.
- **In 3b, op-long internally already has `LOCK_EX`** (exclusive lock): op-long waits for all `LOCK_SH` holders to release before acquiring the lock and exclusively running 200 steps on 2 GPUs; during this period new `LOCK_SH` requests block. This is transparent to you — as long as you do not manually start `python test_op.py` while op-long is running (there is no need to), you will not collide.

If you do not follow this protocol (e.g. running `python test_op.py` not wrapped in flock) → when op-long launches torchrun to grab 2 GPUs, it will fight with your `test_op.py` for the same card's memory → CUDA OOM false positive.

Consequence: the `mfu_e2e` you see in 3c is a **single-GPU** number, **significantly higher than** the 2-GPU op-long measured MFU (which deducts all-reduce + micro-batch grad accum overhead).

- This number is **only used by the subagent itself for comparing between optimization paths**; it is not the 2-GPU training target.
- **Do not** prematurely write `READY_FOR_OP_LONG` in `notes.md` just because single-GPU MFU looks high — your `op-long` PASS is the sufficient condition, and the real 2-GPU MFU from op-long is the same order of magnitude as PROMPT.md "Current Profile".
- The baseline kernel's benchmark at the end of `test_op.py` also goes through the same single-GPU dimension, so the **relative** ratio `ours_mfu / baseline_mfu` is still meaningful; do not directly compare the absolute number to PROMPT.md "Current Profile".

## Experience and Pitfalls

> Summary based on many rounds of kernel development. Purpose: avoid "single-operator test PASS but op-long invalid" — op-long is Stage 2's sole unforgeable operator-level hard gate; single-operator tests are your own feedback loop.

### Why the single-operator gate leaks

An optimized kernel PASSes in isolated tests but fails when put into the real training loop. Three categories of typical blind spots:

#### Blind spot 1 — state change (cache-stale class)

A single-op test only invokes once → no state loop. In training, `optimizer.step()` modifies weights in place; if the kernel internally has a cache (keyed by `id(tensor)`, `torch.cat` independent-storage cache, etc.), step 2+ hits a stale cache.

Check: multiple invocations + simulated optimizer modifications to inputs and observe whether output is still correct.

#### Blind spot 2 — real storage layout (storage-aware class)

The underlying storage of synthetic data `torch.randn().bfloat16()` is independently newly allocated, completely different from the strided view layout produced by **last-dim offset slicing** like `qkv[..., nq_per_kv*d : nq_per_kv*d+d]` in real training in ref `TransformerLayer.forward`.

- The behavior of `empty_like(non_contig_tensor)` is unpredictable.
- `.reshape()` may not share storage with the original tensor → the kernel writes into a temporary object, returning garbage.

Check: must test with real stride scenarios (non-contig view); cannot only test contig.

### When cache / persistent state is involved

For any module with a cache, ask four questions:

1. **Does the cache key match the data lifecycle?** When `id(weight)` is used as the key, will the weight be modified in place?
2. **Is the cache value independent storage or a view?** If independent storage, who is responsible for keeping it in sync?
3. **Invalidation signal?** After `optimizer.step()` / checkpoint load / `.copy_()`, does the cache need to be refreshed?
4. **Safest practice**: cache an intermediary buffer + `.copy_()` sync on each forward. The performance cost is usually <1ms.

### Test Pyramid

```
              ┌─────────────────────────────────────┐
              │  op-long long-run training (200 steps DP=2)     │  ← self-driven in 3b (sole hard gate)
              │  signed_mean ≈ 0, +/- ratio ≈ 50/50     │     ref-script unforgeable;
              └─────────────────────────────────────┘     dispatcher computes loss diff itself
                       ↑
         ┌──────────────────────────────────────┐
         │  test_op.py unit test + real stride        │  ← run each round in 3c
         │  FP64 reference + signed diff check         │     harness does not read stdout / exit code
         │  + benchmark → mfu_e2e (subagent internal reference) │
         └──────────────────────────────────────┘
```

`test_op.py` PASS is the **necessary condition you give yourself** (don't write `READY_FOR_OP_LONG` in `notes.md` until green); op-long PASS is the sufficient condition given by harness (the merge bar). After op-long failure, immediately write an `OP_LONG_FAILED` paragraph + attribution + next-round repair direction in `notes.md`; otherwise when the next round's subagent rebuilds state it will only hit the same wall again.

## `notes.md` Self-describing State (your sole state carrier across rounds)

This protocol **does not enforce** any machine-parseable row format — the paragraphs in your `notes.md` are natural-language records for the next round's subagent (which is also you) and for human reviewers. But because the next round's subagent will grep `notes.md` to rebuild state on startup ("where did the last round get to? how many cumulative op-long FAILs?"), it is strongly recommended you always use the following consistent anchor strings, for easy grep:


| Anchor string                       | When it appears                  | Who reads                          |
| ------------------------------- | ------------------------------ | -------------------------------- |
| `OP_LONG: FAIL #<N>`            | when 3b op-long fails (`<N>` = cumulative count) | next round's subagent counts cumulative FAILs to decide whether 3a is triggered |
| `This round self-judgment: STILL_ITERATING`          | at the end of the 3c paragraph (you think there is still room) | next round's subagent decides to continue 3c progress |
| `This round self-judgment: READY_FOR_OP_LONG`        | at the end of the 3c paragraph (you think convergence is reached) | next round's subagent decides to go to 3b op-long |
| `## SAFETY_NET_TRIGGERED`       | when 3a is triggered           | human review                      |
| `## OP_LONG_FAILED (round <N>)` | when 3b FAILs                  | next round's subagent looks at failure-mode attribution to decide repair direction |
| `## Round <N>`                  | each time the 3c paragraph is written | yourself, reviewing history       |


- These anchors do not enter any harness parser — the harness only looks at git truth + op-long suite output. They are purely an engineering convention for the subagent's own cross-process communication.
- You may freely add extra natural-language context in the paragraph (NCU screenshot URL, roofline calculation, candidate cost-benefit); the denser, the easier for the next round's subagent to relay.
- Do not forge the `OP_LONG: FAIL` count — because the next round's subagent will grep it to decide whether 3a is triggered; forging will make you be wrongly marked failed and kicked out.
- HEAD is always the current best; `notes.md` is always the state SSOT; you are responsible for maintaining their consistency by committing both together.
