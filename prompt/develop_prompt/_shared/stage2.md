# MiniCPM4 0.5B Training Engine — Stage 2 Workload Rules

This file is the SSOT of the agent loop's stage2 standing rules, **serving only the main agent** (the one orchestrating M1/M2 in the repo-root workspace). `.rules/project-guide.md` is the project-level SSOT and is used in conjunction with this file.

**Sub-files / collaborative SSOT**:

- `prompt/develop_prompt/_shared/stage2-subagent-playbook.md` — engineering discipline + pitfalls + Drop-in / FP64 / test hard constraints + tech stack + **per-round single-round decision-tree protocol** for per-op subagents. The main agent cats this whole file into the subagent prompt during M2.1 dispatch.
- `prompt/develop_prompt/_shared/reference/{attention,gemm}.md` — domain knowledge distributed by op type. In M1.5 the main agent cats the corresponding file into the "Optimization Directions" section of the respective `workload/ops/<name>/PROMPT.md`.

**Stage task**: on top of the bitwise-aligned training engine from Stage 1, do deep CUDA optimization on the core operators. **Only two categories of operators are optimized**:

- **Attention** (fwd + bwd merged into one operator, corresponding to `workload/ops/attention/`; at Stage 1 passing time the baseline = `F.scaled_dot_product_attention(q, k, v, is_causal=True)` running on the math backend under the deterministic stack; the in-house side triggers via `torch.ops.aten._scaled_dot_product_`* or an equivalent hand-written kernel. The Stage 2 optimization direction is to replace it with a high-MFU FlashAttention implementation on H100).
- **Various GEMMs** (split by call site: `gemm_qkv_proj`, `gemm_attn_out_proj`, `gemm_fc1`, `gemm_fc2`, `gemm_output`; for each site, fwd + bwd dgrad + bwd wgrad are considered the same logical operator. At Stage 1 passing time, the baseline is the cuBLAS GEMM triggered through `torch.matmul` / `torch.addmm` in `kernels.py`, **not going through the `nn.Linear` / `F.linear` wrapper**). When MTP / Eagle is enabled (default `eagle_num_layers=1`), there is an additional candidate site `gemm_eagle_fc` (corresponding to ref `MTPLayer.eagle_fc`, with width `2*HIDDEN_SIZE → HIDDEN_SIZE = 2048 → 1024`); whether it is included in the Stage 2 optimization list depends on the measured time fraction at M1. Other ops (RoPE / RMSNorm / SwiGLU / CE / Adam / Embedding etc.) remain at baseline; this stage does not touch them.

The two milestones are walked through sequentially by the same main agent:

- **M1 — Scout**: in one round, complete profile / op selection / dispatcher wiring / per-op worktree creation / skeleton writing; the gate `bin/harness run op-inventory` validates the produced artifacts.
- **M2 — Optimize**: within each agent loop round, the main agent uses cursor agent CLI fan-out to launch several subagents; **each subagent process does one round of linear progress and then exits**; cross-round, `agent-loop.sh` main loop drives multiple fan-outs to accumulate progress; the gate `bin/harness run op-status` aggregates acceptance. The subagent rebuilds the previous round's progress by reading the self-describing paragraphs in `notes.md` ("still iterating" / "converged, ready for op-long" / "last op-long FAIL #N"); the safety net is `[automation.stage2].max_op_long_failures` capping the cumulative op-long FAIL count per op across rounds (default value in the SSOT `config/eval/dense_training.toml`). All implementation details of the subagent (decision tree, merge flow, failure retention policy, `notes.md` state anchors) are in `prompt/develop_prompt/_shared/stage2-subagent-playbook.md`.

> **Hardware scenarios and GPU allocation**:
>
>
> | Use                                                              | GPU count            | Process model                                                                                                                            |
> | ---------------------------------------------------------------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
> | `tools/profile_step.py` (M1 real training step profile)          | **2 × H100, DP=2**   | torchrun, running real forward/backward/optimizer; **must use MBS=10 / S=4096 / GBS=80 / GA=4**; explicitly **reject** synthetic init / single GPU / B<10 / shrunk GA |
> | subagent running local `test_op.py` in its own worktree (precision + single-GPU benchmark) | **1 × H100**         | single Python process; uses the GPU selected by the `CUDA_VISIBLE_DEVICES` injected by the main agent                                       |
> | `bin/harness run op-long <name>` (subagent self-driven after self-judging convergence) | **2 × H100, DP=2**   | torchrun, with built-in global `flock` serial queue                                                                                       |
> | Final full pre-training (agent loop does not participate)        | **64 × H100, DP=64** | manually triggered, only started after all gates pass on 2 GPUs                                                                          |
>
>
> **The "current profile" field in PROMPT.md is measured on 2 GPUs** (from `tools/profile_step.py`) and is the reference line for the op-long MFU target. When migrating to 64-GPU training, the communication overhead ratio, gradient accumulation config, global batch size, and MFU target all need to be re-validated; the 2-GPU numbers cannot be applied directly.

The op-long gate is a ref-as-gate evaluation (the dispatcher shell-execs the L0 ref script to get the baseline trajectory; `evals/scripts/op_long_ours.py` runs ours); see the "SSOT: Ref Script as Gate" section in `README.md` for implementation details.

## Command Cheatsheet

```bash
# M1 — Scout milestone
bin/harness run op-inventory           # M1 gate: validate registry / dispatcher wiring / worktree form

# M2 — Optimize milestone
bin/harness run op-status              # M2 gate: aggregate merged / failed status per operator
bin/harness run op-long <name>         # Operator-level hard gate (self-driven by subagent; DP=2 200 steps integration; built-in GPU flock)

# Infrastructure
bin/harness run guard                  # framework guard
bin/harness run anti-proxy             # anti-proxy lint (workload/src/ + workload/ops/)
bin/harness doctor                     # check local GPU environment
bin/harness info                       # show workload metadata
```

> **Local precision tests are not harness suites**. Stage 2 deliberately sets only `op-long` as the operator-level hard gate — single-GPU precision + MFU benchmark are run entirely by the subagent via `test_op.py` in its own worktree; `[PASS]` / `[FAIL]` in stdout is the subagent's own feedback to itself; the dispatcher does not read it and does not count it as a passing condition. Making single-operator precision a "fake gate" gives the subagent the illusion of being audited; better to let it focus on op-long, the sole unforgeable hard gate. For the detailed Trust Model see `stage2-subagent-playbook.md`.

The main agent self-drives worktree creation (M1.4 cheatsheet) and self-drives subagent fan-out each round (M2.1); M2 acceptance looks at `bin/harness run op-status` aggregation (M2.2).

## Gate System


| Milestone       | Gate suite                    | Scope                                                                                                                                                                                                              | Criteria                                                                                                                                                                                                                                                  |
| --------------- | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **M1.scout**    | `bin/harness run op-inventory`   | scout artifacts                                                                                                                                                                                                    | (a) `workload/ops/_registry.toml` valid; (b) each op's `PROMPT.md` embeds the full text of `reference/<op>.md`; (c) the 6 skeleton files under `workload/ops/<name>/` are complete; (d) `ops_worktree/<name>/` exists, HEAD is on `stage2/op/<name>`, sparse-checkout form is correct; (e) at least one `get_op_version()` wiring in `workload/src/` |
| **M2.optimize** | `bin/harness run op-status`      | all registered operators (i.e. `attention` + each `gemm_*`)                                                                                                                                                       | each op falls into either `merged` / `failed`; `attention` must be `merged` (the core benefit item); ≥ half of the GEMM sites are `merged`                                                                                                              |
| op-long         | `bin/harness run op-long <name>` | DP=2 200-step integration (self-driven by subagent after self-judging convergence); **same form as Stage 1 M6 long-train** (MBS=10 / GBS=80 / grad_accum=4), **deterministic mode off** (per the stage1.md §M6 deterministic-off decision; both ref and ours sides off in sync) — necessary condition for single-card 80GB H100 to run at MBS=10 | [100,199] step mean \|rel diff\| ≤ 1%; built-in flock; multiple subagents queue serially to exclusively use the GPU. **The sole unforgeable operator-level hard gate of Stage 2**                                                                        |


merge is not a harness gate; it is a normal git operation, self-driven by the subagent via `flock + git merge` after op-long PASS (see `stage2-subagent-playbook.md` for details). The SSOT for a successful merge is git truth (worktree/branch deleted + `register.toml.default ≠ baseline`), aggregated and judged by `bin/harness run op-status`.

## Operator Version Switching Mechanism (brief)

`tools/stage2_op_router.py` on startup scans `workload/ops/*/register.toml` to build the registry, and via the env var `OP_<NAME>=baseline|v1|v2|...` provides per-op switching: `baseline` always routes to the stage1 frozen implementation; after merging, the subagent changes `register.toml.default` in the same atomic commit; auto-discovery avoids multi-concurrent conflicts. The subagent does not manually modify `tools/stage2_op_router.py`.

---

# M1 — Scout milestone

In the repo-root workspace, the main agent completes the following six steps in one go, lands one or more git commits, and then enters review.

## M1.1 Runtime Probe (`tools/profile_step.py`)

All operator metadata (shape / dtype / stride / contiguity / saved tensor contract) and timing **must come from the runtime probe of the same real training step**. Reasons:

1. checkpoint pt can only tell you the weight's shape; it cannot tell you the real-call layout, dtype, stride of `q/k/v` (`split_qkv_interleaved` gives K/V as a strided view).
2. activation shape (B / N / micro_batch) is determined at runtime by dataloader + config.
3. The saved tensor contract (which keys, shape, dtype enter backward) can only be enumerated after forward has actually run once.
4. Timing and shape must be captured on the same real input.

The sole entrypoint: extend and run `tools/profile_step.py`. It already runs real forward/backward/optimizer; this round only needs to add a runtime probe layer to it.

### Extension Items

Extend on the basis of `profile_step.py` (modify this script itself; do not write a new one):

1. **Per-call-site tensor metadata probe**: only wrap a lightweight `_probe(name, *tensors, saved_keys=None)` around the two classes of call sites: **attention and each GEMM site**. Enable only at the timing step; disable during warmup steps. The same call site invoked multiple times only records the first 2 (preserves the contig / non-contig switch difference).
2. **torch.profiler kernel-level timing**: upgrade the existing `torch.cuda.Event` timing to `torch.profiler.profile(activities=[CUDA], record_shapes=True)`, aggregating `cuda_time_total_us` by `name + input_shapes`. Map kernel name back to logical operator:

   | kernel name keyword (contains-match)          | op_name              |
   | --------------------------------------------- | -------------------- |
   | `flash_attn` / `te_dpa` / `mha_fwd`           | `attention`          |
   | `gemm` and shape matches `qkv_proj` weight    | `gemm_qkv_proj`      |
   | `gemm` and shape matches `attn_out_proj`      | `gemm_attn_out_proj` |
   | `gemm` and shape matches `fc1`/`fc2`/`output` | `gemm_<that one>`    |

   Unmatched kernels are aggregated to `op_name = "uncategorized"`, used only for reporting the overall share.
3. **MFU and end-to-end time**: read `total_step_time` / `mfu_e2e` directly from the existing output of `workload/src/training_engine_tensor/train_loop.py`; do not recompute FLOPs.
4. **Multi-step aggregation**: run ≥ 5 timing steps (warmup ≥ 2 steps); timing takes the mean and discards the first step; the shape/stride probe is only captured once at the 1st timing step.
5. **JSON dump entry**: the current stub only prints the cuda-event timing table to stdout. M1.1 must add a `--output PATH` parameter to `tools/profile_step.py` (default `workload/profile/profile_result.json`), and aggregate items 1–4 above into a JSON written to `--output`; the schema is referenced by M1.2 / M1.4's references to `ops.<name>.{measured_cuda_time_ms, measured_pct_of_step, ...}`. All downstream stage2 milestones (including `BASELINE.md` / `_registry.toml`) take numbers from this one file.

### Run Entrypoint

Before first run, use `tools/capture_profile_batch.py` to generate the workspace-local `workload/profile/captured_batch.pt`; the second command below depends on extension #5 adding `--output`, and the stub form does not accept this parameter:

```bash
python tools/capture_profile_batch.py --source dataloader
python tools/profile_step.py --output workload/profile/profile_result.json
```

### Hard Constraints

- Using `torch.load(canonical_state_fp32.pt)` + dumping shape as contract basis → rejected.
- Using FLOP formula estimation instead of timing → rejected (FLOPs do not reflect the real cost of a memory-bound epilogue).
- Using synthetic data (`torch.randn` / `synthetic init_params_fp32`) instead of real checkpoint + dataloader → rejected.
- Splitting shape probe and timing probe into two separate scripts → rejected (they must produce the same single `profile_result.json` in one run).
- `profile shape == production shape (MBS=10)` → required; running profile with reduced batch will lead the subagent's autotune to pick the wrong block size, causing the entire stage2 to be redone → rejected.

## M1.2 Operator Grouping and Ordering

Merge call sites into logical operators:

- `attention` — SDPA's fwd + bwd (the Stage 1 baseline is the deterministic math-backend path; the Stage 2 optimization target is a high-MFU FlashAttention implementation on H100).
- `gemm_qkv_proj` / `gemm_attn_out_proj` / `gemm_fc1` / `gemm_fc2` / `gemm_output` — each GEMM site is independent; fwd + bwd dgrad + bwd wgrad are the same operator. The concrete number of sites and naming follow the GEMMs actually hit in `profile_result.json`.
- `gemm_eagle_fc` (optional) — exists when MTP / Eagle is enabled; shape is small (input 2*H=2048, output H=1024); time fraction may be low; whether to include it in the Stage 2 optimization list depends on M1 measurements.

Sort by `ops.<name>.pct_of_step` in `profile_result.json`:

- `attention` must be ranked at `priority=1` regardless of measured rank.
- Each GEMM site is ordered by measured time, descending, starting from `priority=10` and incrementing by 1.
- `_registry.toml` must fill in `measured_cuda_time_ms`, `measured_pct_of_step` (copied directly from `profile_result.json`; do not estimate/recompute).

## M1.3 Dispatcher Wiring

Only for the Stage 2 optimization list (`attention` + each `gemm_`*), insert branches at the corresponding call sites in `workload/src/training_engine_tensor/`:

```python
from tools.stage2_op_router import get_op_version

if get_op_version("gemm_qkv_proj") == "baseline":
    qkv = cublas_gemm(x, w_qkv)                     # original code untouched
else:
    from workload.ops.gemm_qkv_proj.kernel import gemm_fwd
    qkv = gemm_fwd(x, w_qkv)
```

- The `"baseline"` branch keeps the pre-modification code untouched — default behavior is unchanged.
- The `else` branch imports from `workload/ops/<name>/kernel.py`.
- Call sites outside the list are completely untouched; do not even add a `get_op_version()` call.
- After wiring is complete, these source files are frozen for the subagent (enforced by L2 pre-commit + L3 review).

After wiring is complete, run an "all `OP_*=baseline`" forward smoke to confirm that the dispatcher branches do not change baseline behavior. The op-inventory gate will re-validate this.

## M1.4 Worktree and Sparse-Checkout (cheatsheet)

Each operator has an independent worktree on branch `stage2/op/<name>`; sparse-checkout limited to this op's directory + read-only run context:

```bash
git worktree add "ops_worktree/$OP" -b "stage2/op/$OP"
# Sync: make the new worktree directly take the main worktree's current HEAD (including this round's M1 scout commit), to avoid the "worktree created before commit" order leaving the worktree stuck on the old HEAD unable to see the scout artifacts.
(cd "ops_worktree/$OP" && git reset --hard "$(git -C ../.. rev-parse HEAD)")
(
  cd "ops_worktree/$OP" && \
  git sparse-checkout init --no-cone && \
  git sparse-checkout set \
    pyproject.toml \
    .pre-commit-config.yaml \
    config/eval/dense_training.toml \
    harness evals tools prompt \
    workload/src \
    "workload/ops/$OP"
)
```

Write permissions: the subagent can only modify files under `workload/ops/<name>/`; all other paths are read-only (enforced in three layers: L1 sparse-checkout + L2 pre-commit hook + L3 review agent).

## M1.5 Per-op File Skeleton

Each op `workload/ops/<name>/` contains:

- `__init__.py` — empty.
- `PROMPT.md` — subagent prompt; the unified template is in M1.5.1.
- `BASELINE.md` — derived field-by-field from `profile_result.json` (call location; for each I/O tensor: role / shape / dtype / stride / is_contiguous / saved_keys / measured `cuda_time_ms` / `pct_of_step`). All numbers must be field-by-field traceable in `profile_result.json`. **The batch dimension of the I/O contract must = production MBS=10**; if BASELINE.md shows B<10 = profile cutting corners, op-inventory should reject.
- `kernel.py` — pass-through skeleton, directly re-exporting baseline functions. Makes the `else` branch immediately usable.
- `register.toml` — `env_var` / `default = "baseline"` / `available = ["baseline"]`.
- `test_op.py` — basic test (basic contig + real stride + FP64 comparison) runnable directly via `python test_op.py`. **Input tensor shape must = production (MBS=10 / S=4096)**; on OOM, prefer reducing S rather than B.
- `notes.md` — empty; for the subagent to record iterations.

Also maintain `workload/ops/_registry.toml` and `workload/ops/__init__.py`.

### M1.5.1 PROMPT.md Unified Template

Both op categories share the same writing process; the only differences are the domain reference embedded (`reference/attention.md` or `reference/gemm.md`) and the runtime numbers copied in. The main agent directly reads `profile_result.json` and the baseline source, and writes the real numbers into PROMPT.md in place.

Template:

```markdown
# Operator Optimization: <name>

You are optimizing the **<name>** operator for MiniCPM4 0.5B on H100.

## Remote Execution Reference (fill only when `[remote].kind` is `ssh` or `devspace`)
<Include remote execution path, GPU resources, op-long resources, GPU to use for unit tests, etc.>

## Current Baseline Implementation
<baseline call location (precise to workload/src/training_engine_tensor/<file>.py:L<lineno>), the kernel invoked:

## Current I/O Contract
<Copy the tensor metadata of this op entry in profile_result.json directly here: for each input role / shape / dtype / stride / is_contiguous / saved_keys; outputs in the same format. All numbers must be field-by-field traceable in profile_result.json.>

## Current Profile
<Paste the op entry's cuda_time_ms / pct_of_step from profile_result.json. This is the realistic reference line for the subagent's MFU optimization across rounds.>

## Optimization Directions (Domain Reference — embed full text)
<In M1.5, the main agent uses `cat prompt/develop_prompt/_shared/reference/<op>.md` to fully append the corresponding domain reference: `reference/attention.md` for the attention op, `reference/gemm.md` for all gemm_* sites. Byte-for-byte copy; excerpting / rewriting / translating / summarizing is not allowed.>
```

PROMPT.md **does not duplicate** content like tech stack, Drop-in replacement constraints, FP64 cheatsheet, single-round decision-tree protocol — these are sent to the subagent via the M2.1 dispatch cat-ing `stage2-subagent-playbook.md`, to avoid multiple copies.

Hard constraints:

- The three sections "Current Baseline Implementation", "Current I/O Contract", "Current Profile" **must be filled with real numbers**; no `<…>` placeholders.
- The "Optimization Directions" section **must cat the entire `reference/<op>.md` file in**, preserved byte-for-byte (op-inventory gate rejects any excerpts/rewrites). If `reference/*.md` itself is updated, simply re-cat the whole file to sync.
- For the same op, the "Current I/O Contract" / "Current Profile" fields in BASELINE.md and PROMPT.md must be aligned (derived from the same `profile_result.json`).

Mechanically write PROMPT.md (the main agent runs once in M1.5):

```bash
set -euo pipefail
OP="<name>"                                       # e.g. attention / gemm_qkv_proj
REF=$([[ "$OP" == "attention" ]] \
        && echo "prompt/develop_prompt/_shared/reference/attention.md" \
        || echo "prompt/develop_prompt/_shared/reference/gemm.md")

# The main agent writes the first 4 sections (title / baseline / I/O / profile) here:
$EDITOR "workload/ops/$OP/PROMPT.md"

# Then append the domain reference — entire, as-is:
{
  echo
  echo "## Optimization Directions (the following is cat'd from $REF)"
  echo
  cat "$REF"
} >> "workload/ops/$OP/PROMPT.md"
```

## M1.6 op-inventory Validation

`bin/harness run op-inventory` validates:

1. `workload/ops/_registry.toml` exists and is valid (each op contains `category` / `priority` / `env_var` / `measured_*`).
2. For each op `<name>` in the registry:
  - `workload/ops/<name>/{PROMPT.md, BASELINE.md, kernel.py, register.toml, test_op.py, notes.md, __init__.py}` all exist.
  - `register.toml` contains `env_var` / `default = "baseline"` / `available` contains `"baseline"`.
  - PROMPT.md embeds the full text of the corresponding `reference/<op>.md` (byte-for-byte).
  - `ops_worktree/<name>/` exists, HEAD is on `stage2/op/<name>`, sparse-checkout covers required paths.
3. `workload/src/training_engine_tensor/` has at least one `get_op_version(` call.
4. All `OP_*=baseline` smoke (optional forward-align 1 GPU) confirms dispatcher branches make zero behavioral change.

Passes → review_stage2.md M1 section check → **M1 done**.

---

# M2 — Optimize milestone

The main agent in the repo-root workspace runs one round of dev + review; in the dev phase, only dispatch is performed; the actual optimization is closed-looped by the subagent inside its own worktree.

## M2.1 Main Agent Dispatch Protocol (run once per round)

`agent-loop.sh` stage2 starts a fresh dev agent + review agent each round. The dev agent does the following in order in its own round:

1. `bin/harness run op-status --json` lists all operators not in `merged` and not in `failed` states (operators already `failed` are no longer fanned out; they wait for review / human review to decide the next step). **Note the distinction**: `register.toml.default = "baseline"` is an **un-optimized state** (incl. `not_started` / still iterating but not yet merged), **needs fan-out subagent to keep progressing**; `failed` state requires the subagent to **explicitly mark `STATUS: failed` in `notes.md`** (a still-existing worktree+branch alone is not sufficient) — they are different; do not skip an op still on baseline by mistaking it for failed.
2. Launch cursor agent CLI subprocesses in batches; each batch has at most `automation.stage2.max_concurrent` (the value is controlled by the SSOT `config/eval/dense_training.toml`; the main agent reads it via the `$STAGE2_MAX_CONCURRENT` environment variable).
3. Wait for all subagents of this batch to **exit** before processing the next batch (each subagent process automatically exits after completing one round of linear progress; see `stage2-subagent-playbook.md`).
4. After all dispatch completes, run `bin/harness run op-status` and write a summary into `workload/notes/stage2_op_opt.md`, then commit.
5. This round's dev phase ends here. The review agent then runs (see the "Stage2 Check FINISH" section of `review_stage2.md`); if op-status is not all merged → review reports `STAGE_STATUS: in-progress` → `agent-loop.sh` enters round=N+1, rerunning this flow.

Dispatch command template (subagent prompt = this op's PROMPT.md + subagent playbook):

```bash
mkdir -p .artifacts/locks
idx=0
for op in $(bin/harness run op-status --json \
            | jq -r '.targets[0].payload.metrics.operators[]
                     | select(.status != "merged" and .status != "failed") | .name'); do
  # 只有 [remote].kind 为 ssh 或 devspace 时才把 remote-execution.md overlay
  # 嵌入 subagent prompt；本地模式留空，避免 subagent 误以为该走 SSH。
  # CUDA_VISIBLE_DEVICES 由下方 `CUDA_VISIBLE_DEVICES=$idx ... spawn_managed_agent.py`
  # 注入 env（不在 prompt 里）；SSH host / workdir 等已经在 main agent loop
  # build_prompt() 里替换好 remote-execution.md 的 @@KEY@@ 占位符，
  # subagent 拿到的就是最终值。
  remote_overlay=""
  if [[ "${REMOTE_ENABLED:-false}" == "true" ]]; then
    remote_overlay="$(cat prompt/develop_prompt/remote-execution.md)"
  fi
  prompt="$(cat workload/ops/$op/PROMPT.md)

$(cat prompt/develop_prompt/_shared/stage2-subagent-playbook.md)

${remote_overlay}"
  # 每个并发 subagent 拿一张独占 GPU 做"本 round 的一轮线性推进"，
  # 然后退出；跨 round 推进由外层 agent-loop.sh 多次 fan-out 累计。
  # op-long 用 LOCK_EX 抢全部 8 卡，与本批 LOCK_SH 互斥
  # （flock 协议详见 stage2-subagent-playbook.md「资源协议」段）。
  #
  # spawn_managed_agent.py 把这个 subagent 注册为一行 Session：
  #   $FORGE_AGENTS_DIR/<agent_id>/{session.json, stdout.log, stderr.log}
  # 前端 Agents tab 会自动发现并和当前 loop 的 wrapper 链接（parent_agent_id
  # = loop-$LOOP_ID, kind = loop_subagent），所以不需要再写 .artifacts/
  # stage2-agent-logs/${op}.log；stdout.log 就是同样的 stream-json。
  prompt_file=$(mktemp)
  printf '%s' "$prompt" > "$prompt_file"
  CUDA_VISIBLE_DEVICES=$idx python "$FORGE_REPO_ROOT/harness/tools/spawn_managed_agent.py" \
    --backend cursor-cli --model "$MODEL" \
    --workspace "$(pwd)/ops_worktree/$op" \
    --prompt-file "$prompt_file" \
    --kind loop_subagent \
    --loop-id "$LOOP_ID" \
    --parent-agent "loop-$LOOP_ID" \
    --api-key-env CURSOR_API_KEY \
    > ".artifacts/locks/${op}.spawn.out" 2>&1 &
  idx=$((idx + 1))
  if (( idx >= STAGE2_MAX_CONCURRENT )); then
    wait
    idx=0
  fi
done
wait
```

The main agent does not parse the subagent's output JSON, nor read the subagent's notes — M2 acceptance is fully aggregated by `bin/harness run op-status`; the main agent only looks at the op-status summary.

> **Subagent implementation details** (what each subagent process does in a single run, how to walk the three-choice decision tree, how to self-drive `flock + git merge` to merge into main after op-long PASS, failure retention policy, `notes.md` self-describing state anchors) all live in `prompt/develop_prompt/_shared/stage2-subagent-playbook.md`, and are cat'd into the subagent prompt by the M2.1 dispatch template. The main agent does not need to and should not read it.

## M2.2 op-status Aggregation

`bin/harness run op-status --json` output (`.targets[0].payload` is the suite's actual payload; the jq paths are fully consistent with `review_stage2.md` Stage2 Check FINISH):

```json
{
  "targets": [
    {
      "payload": {
        "status": "passed",
        "suite": "op-status",
        "summary": "3 operator(s): merged=2, failed=1",
        "metrics": {
          "operators": [
            {"name": "attention",     "status": "merged", "default": "v1", "available": ["baseline","v1"]},
            {"name": "gemm_qkv_proj", "status": "merged", "default": "v1", "available": ["baseline","v1"]},
            {"name": "gemm_fc1",      "status": "failed", "default": "baseline", "available": ["baseline"], "worktree": "ops_worktree/gemm_fc1", "branch": "stage2/op/gemm_fc1"}
          ],
          "summary_counts": {"merged": 2, "failed": 1, "not_started": 0, "inconsistent": 0}
        },
        "details": {}
      }
    }
  ]
}
```

Fields used in dispatch / review: `.targets[0].payload.metrics.operators[]` is the per-op entry; `.targets[0].payload.metrics.summary_counts` is the status count.

Status determination (git truth + registry; corresponding to `_run_op_status` / `_status_entry_for_op`):


| Status         | Determination                                                                                                                       |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `merged`       | `register.toml.default ≠ "baseline"` and `ops_worktree/<name>` does not exist and branch does not exist                                                  |
| `failed`       | `ops_worktree/<name>` exists and `stage2/op/<name>` branch exists (subagent progressed in worktree but did not merge; see failure retention policy section in `stage2-subagent-playbook.md`) |
| `not_started`  | `register.toml.default == "baseline"` and both worktree / branch are absent (op is in registry but M1 has not yet created the worktree)                  |
| `inconsistent` | other unexplainable combinations (e.g. default ≠ baseline but worktree still present — main is merged but worktree/branch are not cleaned; goes to human review) |


op-status runs at the end of each main-loop round (the main agent has already `wait`-ed for all subagents dispatched this round); the states observed are post-exit steady states: either this round's linear progress left a new commit, or worktree failure retention was hit (`failed`), or a merge was completed (`merged`).

M2 gate judging rule (the hard conditions before review):

- All ops fall into `merged` or `failed`.
- `attention` must be `merged`.
- ≥ half of GEMM sites are `merged` (the threshold in `review_stage2.md` is determined by the total number of GEMM sites measured in M1.2).

Passes → review_stage2.md M2 section check → **M2 done**.

---

## Stage 2 Endgame Signal (the protocol for the dev agent to self-report STAGE_STATUS: finished)

`agent-loop.sh` is goal-driven (see `config/agent.toml [agent]` + `agent-loop.sh --help`): the loop runs until the review agent reports `STAGE_STATUS: finished` to end stage2. Stage 2's "completion" determination is provided by op-status as git truth.

**Preconditions** (hard, all required) for the dev agent (main agent) to write a sole-line `STAGE_STATUS: finished` at the end of the commit message:

1. `bin/harness run op-status --json | jq '.targets[0].payload.metrics.summary_counts'` shows `merged > 0` and `failed == 0` and `not_started == 0` and `inconsistent == 0`.
2. `attention` op status is `merged` (priority=1, must be merged).
3. All `gemm_*` sites have status `merged`.
4. The above three items come directly from the output of `bin/harness run op-status --json`; **no other source can substitute** — op-status aggregates git truth + registry and is the sole unforgeable "completion" criterion for Stage 2.
5. This round's commit is only a wrap-up nature (writing the op-status summary into `workload/notes/stage2_op_opt.md` / updating perf_log), and does not include new dispatcher wiring or new op registration — the dev agent self-reporting finished typically occurs in the round when the last op completes merge, and the dev agent runs op-status to see `merged_count == total`.

How to decide when conditions are not met:

- If `failed > 0`: there are still failed ops — decide whether to accept failure (write postmortem and remove non-attention/gemm failed ops from `_registry.toml` scope; attention failure must be human-reviewed and re-dispatched, with no fallback), or after review/human review intervenes, clean up the worktree + raise `max_op_long_failures` and re-dispatch. Either way, you **cannot** write finished this round.
- If `not_started > 0`: M1 did not create worktrees for all ops — go back to M1 to complete them.
- If `inconsistent > 0`: register.toml and worktree state are misaligned — human review intervenes; you **cannot** clean up yourself and then write finished (else you may hide a real merge failure).

After satisfying all 5 items, the commit message looks like:

```
chore(stage2): all ops merged; close stage 2

op-status: merged=4 failed=0 not_started=0 inconsistent=0
attention=merged gemm_qkv_proj=merged gemm_fc1=merged gemm_fc2=merged

STAGE_STATUS: finished
```

The review agent (`review_stage2.md` Stage2 Check FINISH) re-runs op-status and re-verifies conditions 1–4 before pass-through `STAGE_STATUS: finished` to the agent loop; when conditions do not hold, the review agent still outputs `STAGE_STATUS: in-progress` and the loop continues.

---

## Meta: Rule Evolution

This file is the SSOT of stage2 standing rules (**serving only the main agent's orchestration**). When it changes, in the same round synchronously update:

- `prompt/project-guide.md` (project-level SSOT)
- `prompt/review_prompt/review_stage2.md` (the review check items correspond one-to-one with this file's milestones / gates / state machine)
- `prompt/develop_prompt/_shared/stage2-subagent-playbook.md` (if it involves subagent engineering discipline / constraints / experience / Resource Contract / single-round decision-tree protocol)
- `prompt/develop_prompt/_shared/reference/{attention,gemm}.md` (if it involves domain knowledge — re-cat into all `workload/ops/<name>/PROMPT.md` to sync; the prose is currently aligned with the torch backend perspective (SDPA / `torch.matmul`), with megatron-side facts (`te.DotProductAttention` / TE GEMM) noted inline as union prose; if the active backend's API surface changes materially, the affected sections must be rewritten in the same round)
- `harness/framework_guard.py`'s `ALLOWLIST` / `REPO_WRITE_SURFACE_`* (if applicable) + `harness/tests/test_framework_guard.py`
- `evals/dispatcher.py` (if it involves `_validate_op_artifacts` / `_validate_op_prompt_reference` / `_validate_dispatcher_wiring` / `_run_op_long*` / `_run_op_status` / `_stage2_file_lock` lock name and other gate implementations)
- `config/eval/dense_training.toml` (if it involves `[automation.stage2]` fields or `[evals.op-inventory|long|status]` ref_env) + `RUNNER_METRICS` in `harness/run_schema.py`
