# Training Engine — Stage 1 Overview

This file is the entry point of the agent loop's stage1 standing rules under the active backend (selected by `@@FORGE_CONFIG_DIR@@/ref.toml [ref].backend`). `prompt/project-guide.md` is the project-level SSOT, derived into `.cursor/rules/project-guide.mdc`, and is used in conjunction with this file.

**Task for this stage**: Develop and optimize the training framework implementation under the workload's training-engine source (see `project-guide.md` LLM Permission Matrix), so that it passes all bitwise / statistical gates of the milestones declared in `@@FORGE_CONFIG_DIR@@/eval.toml` `[evals.*].milestone`. All forward/backward/optimizer/dataloader/kernel/checkpoint/train_loop paths fall within the writable, optimizable scope; per-milestone operator-freeze rules (e.g. the long-horizon GEMM + fused-attention freeze) live in the corresponding per-milestone `<name>.md` (see `long-horizon.md` §long-horizon stage operator freeze constraint for the concrete long-horizon case).

> **Backend selection**: This repo selects the L0 ref stack via `@@FORGE_CONFIG_DIR@@/ref.toml [ref].backend` (and the matching `[ref].ref_script`). The L0 ref entrypoint is `ref/reference/${ref_script}` (filename from `[ref].ref_script`). All "ref" references in this document refer to that ref.

> **SSOT Gate Model**: the L0 ref script (specified by `[ref].ref_script`, located under `ref/reference/`) is the **parameter authority** and **execution authority** for the baseline — all gates directly `shell-exec` the ref script as the reference; there is no frozen JSON, no sha anchor, no audit table. All gate execution specifications are defined by the ref script's `HARNESS_GATE` preset and `gate_metadata.json`. **Duplicating a second copy of the training spec in Python/TOML, or reverting to frozen JSON / sha anchors, is not allowed.**
>
> Before starting any milestone for the first time, the agent must read that section once to understand the upstream–downstream relationships of `ref/reference/${ref_script}` → `dispatcher` → `_run_bitwise_trajectory`. Skipping it leads to wrong attempts (e.g., trying to freeze `gate_metadata.json`, looking for a non-existent baseline JSON, trying to add `evaluator/baselines/...`).
>
> The same section also describes the "L0 ref script as gate" rule of long-running gates — `long-train` does not have any frozen JSON; the dispatcher synchronously `shell-exec`s the L0 ref script every time the gate runs to produce the ref trajectory. Modifying the ref script = modifying the baseline truth, and must go through PR review.

## Introduction

The goal is to use the real forward/backward of the ref (`ref/reference/${ref_script}`) directly as the alignment authority source, and to build a static-execution-graph-driven dedicated training framework for the active workload.

**This project does not use pre-captured traces for alignment.** Every milestone uses the ref-vs-ours subprocess form — the dispatcher runs the L0 ref script as one subprocess and the in-house engine as another, then compares their outputs. The two flavours of comparison wire format (single-step tensor-dump diff vs multi-step trajectory diff) and the per-milestone gate semantics (bitwise vs statistical) are documented in each per-milestone `<name>.md`. **There is no frozen JSON, no sha anchor, no audit table.** Both sides share static inputs only (checkpoint, seed, data path); the data-path env var and the external dataloader are whatever the active ref script uses (see `constraint.md` §Dataloader description).

## Command quick reference

The full per-suite parameters (window, step count, MFU/loss thresholds, `ref_env`, `runner_kind`, `timeout_s`) are owned by `@@FORGE_CONFIG_DIR@@/eval.toml` `[evals.<suite>]`; read that file directly when you need a number. The milestone → suite mapping in this stage:

| Milestone | Suite(s) | Form |
|-----------|----------|------|
| alignment | `forward-align` / `backward-align` | single-step tensor-dump bitwise (ref-vs-ours subprocess) |
| bitwise-singlecard | `multistep-1gpu` | single-GPU multi-step bitwise |
| bitwise-multicard | `multistep` | multi-GPU DP-only multi-step bitwise (smaller models) |
| bitwise-dptp | `dptp` | multi-GPU DP×TP multi-step bitwise (TP-sharded; 8B uses this instead of bitwise-multicard) |
| bitwise-perf | `perf-bitwise` | multi-card bitwise gate (bitwise-multicard or bitwise-dptp) + MFU gate |
| resume | `resume-gate-20` + `wsd-sft-70` | save/load round-trip self-comparison + WSD-SFT 3-phase switch correctness (stable→decay→sft handoffs) |
| long-horizon | `long-train` (main) + `resume-gate-20` (regression) | long-run statistical gate + operator-fusion regression |
| production | `production-resume-70` (step 1) + `production-train` (step 2) | crash-resume self-comp gate, then the ours-only 3-phase WSD-SFT production long-train as a cctl PyTorchJob (checkpoint-only: no ref, no loss·MFU threshold) — **terminal** stage1 milestone, owns `STAGE_STATUS: finished` |
| (auxiliary) | `loss-gate-200` | lightweight loss-only subset, for debugging only |

Local commands (`bin/harness run guard` / `anti-proxy`, `bin/harness doctor`) are inherited from the inline header at the top of this prompt.

## Core principles

1. **Reuse existing efficient kernels** — cuBLAS GEMM, the appropriate fused kernel path per backend, and NCCL are already good; the focus is the in-house scheduling layer, with kernels invoked directly; **do not introduce high-level wrappers such as `torch.nn` / `torch.optim`** (the in-house side stays at primitive level regardless of backend). Backend-specific kernel rules: **megatron** — TransformerEngine fused kernels (RMSNorm, SwiGLU, fused attention, GEMM via `te.LayerNormLinear` / `te.Linear` etc.) are allowed primitives and **must match the ref's calling mode** to achieve bitwise alignment (see `constraint.md`); bare `torch.matmul` uses a different cuBLAS algorithm selection than TE's internal GEMM path and will produce 1-ULP diffs — do not attempt to close the gap with precision flags, use the same TE operator the ref uses. **torch** — `torch.matmul` / `torch.addmm` (cuBLAS), SDPA math / flash / mem-efficient triplet; TransformerEngine is forbidden.
2. **Optimize only the fixed scenario** — this is a dedicated framework for the configuration declared in `@@FORGE_CONFIG_DIR@@/eval.toml` and the other axis TOMLs (`model.toml` / `optim.toml` / `ref.toml`); do not pursue any generality beyond it.
3. **In the bitwise correctness phase, uniformly align with the ref** — all correctness, compare, debug in the bitwise correctness milestones (including real dataloader) are based on the real forward/backward execution results of the ref. Pre-captured traces are not used. The agent needs to read the ref code (the ref source code under `ref/reference/`) itself to understand the behavior of the reference implementation.
4. **Build directly, do not restore old stacks** — no old training stack compatibility layer, no restoring historical training entrypoints; build the in-house executor directly around the computational dependencies already exposed in the ref training entrypoint.
5. **Static first** — things that can be determined statically should not be left to runtime, e.g. shape, buffer, execution order, bucket partitioning, and key tensor mapping relationships.
6. **Look at performance and correctness together** — every optimization must explain its impact on step time / MFU, while ensuring `loss`, key tensor events, parameter updates, and optimizer-phase behavior are aligned with the ref.
7. **Separate correctness paths from performance paths** — correctness / compare paths may explicitly attach extra verification logic; the performance path must be a real training process, may allow randomness from dropout, but except for necessary tensors (e.g. token embedding tensor), loading additional tensor assets is forbidden.
8. **Performance optimization prioritizes framework simplification** — the performance optimization process should minimize dynamic structures, temporary bridges, and unnecessary abstractions, prioritizing static definitions and direct scheduling.
9. **Development pace — one milestone per execution**. In each conversation, work on **at most one milestone** (any milestone in the active stage manifest). Once that milestone's gate passes (or you have made one meaningful, committed forward step toward it), end this round with a commit and return control to the loop. **Do not chain multiple milestones in a single conversation, and do not ask the user whether to continue.** The agent loop is goal-driven; the next round will pick up from the new state.
10. **No regression on prior milestones unless requested** — Each milestone only needs to pass its **own** gate. Do NOT re-run earlier milestones' gates as a regression sweep on every round; the per-milestone constraints in `constraint.md` and the M-files already document whatever invariants must survive into later milestones. Run a prior milestone's gate only when (a) the user explicitly asks for it, or (b) the active milestone's own gate names it as a regression check (e.g. some milestones declare a regression suite from a prior milestone). The agent loop reinforces this by only injecting the current milestone's MD into the dev prompt — prior milestones' MDs are intentionally omitted.

## In-house layer

The following items belong to the training backbone and must be directly controlled by the in-house execution path:

- Static model definition: fixed parameter layout — concrete dimensions / layer count / attention layout live in `@@FORGE_CONFIG_DIR@@/model.toml`. **Do not use high-level abstractions such as `torch.nn.Module` / `nn.Linear` / `nn.Embedding` / `nn.RMSNorm`** — own `torch.Tensor` parameters directly via `torch.empty`, and forward uses primitives like `torch.matmul` / `torch.addmm` / `torch.ops.aten.*`. (On the torch backend the ref itself uses high-level wrappers; the in-house side strictly stays at the primitive layer, but the final forward / backward / optimizer paths must be bitwise aligned with the ref.)
- Static execution graph: fixed shape / order / microbatch forward and backward execution order
- Static memory planning: size and reuse of buffer / activation / workspace, determined at compile time
- Communication scheduling: bucket partitioning, overlap timing, dispatch order, calling NCCL directly (`torch.distributed`)
- Optimizer step orchestration: replace `torch.optim` with primitive-level ops; the specific primitive choice and parameter grouping are determined by the active ref (see `constraint.md` §Low-level technology stack boundaries → `torch.optim` row).
- Training main loop: step entrypoint, scheduling order, failure semantics
- Data path: in-house entrypoint wrapping the external dataloader the active ref script uses, fixed-shape collate / prefetch / H2D; see `constraint.md` §Dataloader description.

## Permission to read ref code

The agent is **allowed and encouraged** to actively read ref source code (under `ref/reference/`) to understand:

- the execution order and computational logic of forward/backward
- parameter initialization and weight layout
- the implementation details of loss computation (cross-entropy)
- the precise behavior of the optimizer step
- the data loading and batch construction methods

Reading the ref source code or running the ref baseline script to obtain alignment references is not forbidden to the agent. What is forbidden is `import` of the ref's framework root inside the in-house code path (see `constraint.md` §Hard rule — no external framework reference).

## Per-round procedure

1. **Record results** — Write the changes and performance results into `workload/notes/perf_log.md`, in English. **Append new content at the end** for easy subsequent reading.

   **Observation vs. attribution — attribution must be earned by a fix.** In both `perf_log.md` entries and commit messages, separate what you *observed* from what *caused* it. State observations freely (which gate keys fail, measured numeric diffs, where failures cluster). Do **not** assert a **root cause** until a fix has empirically confirmed it — i.e. the change you made on that hypothesis actually moved the gate (more keys pass, or the diff drops to 0). A cause is proven by the repair, not by argument. Until then the strongest allowed phrasing is "candidate hypothesis — next round must verify before investing", and any queued fix plan must be labelled provisional. Rationale: an unverified "root cause located" handed to the next (memoryless) round becomes an anchor it will not re-question — a hypothesis can read as airtight yet contribute zero, and a fix that leaves the gate unchanged retroactively disproves the attribution.
2. **First align with the ref baseline, then measure performance** — Check at least `loss`, `grad norm`, parameter update result, step time, MFU, and clearly state the ref baseline configuration bound to this alignment.
3. **Effective optimizations need to be committed** — Every effective round needs to save and commit the code to the local repo.
4. **Decouple debug from the performance hot path** — Additional logic such as `debug`, `compare`, `checksum`, `dump` must be enabled by an explicit switch; by default, they must not pollute the performance path.
5. **Performance optimization must go through the real training entry point** — From token ids drive the real forward-backward logic; mounting a full set of intermediate tensor replays just to score is not allowed.
6. **Continue to keep things static during performance optimization** — Any performance change should prioritize removing useless framework structures, rather than stacking new dynamic layers.
7. **Actively read the ref code** — When encountering alignment problems, you should first read the ref source code to understand the precise behavior of the reference implementation, instead of guessing or relying on documentation.
8. **Milestone self-declaration in the commit message** — The advancement protocol (`MILESTONE_STATUS: <name> PASS`, where `<name>` is the active milestone, e.g. `alignment`, `bitwise-perf`, `production`) is documented in the inline header injected at the top of this prompt. Two stage1-specific notes:
   - When you declare `<name> PASS`, the loop advances the active milestone to its successor in the stage order (capped at the highest milestone in the stage manifest); declaring the successor's `PASS` would mean you've already passed the successor's gate, not just the current one's.
   - The terminal milestone in this stage may co-emit `MILESTONE_STATUS: <last> PASS` and `STAGE_STATUS: finished` in the same commit; the latter is the sole signal that ends stage1 (`review_stage1.md` owns that contract).
9. **Don't wait on a job by polling process liveness — wait on its output artifact, with a timeout.** `kill -0` / `ps` / `pgrep` still report a finished-but-unreaped **zombie** (`<defunct>`) as alive, so a `until ! kill -0 <pid>; do sleep N; done` loop hangs forever and silently deadlocks the round. Instead poll for the completion evidence (exit code, result JSON, artifact like `summary.md`), always bound the wait with `timeout`, and prefer `bin/harness run …` over hand-rolled `ssh` wait loops.

## Meta: rule evolution

This `stage1/` directory is the SSOT of stage1 standing rules under the active backend:

- `overview.md` — intro, commands, core principles, in-house layer, per-round procedure (this file)
- `constraint.md` — cross-milestone forbidden / spec / red-line clauses
- per-milestone files — see `harness/tools/agent_loop_config.py:_STAGE_DIR_MANIFEST_*` for the active manifest (each per-milestone `<name>.md` defines its own goals, gates, constraints)

When any file in this directory changes, synchronously update in the same round:

- `prompt/project-guide.md` (project-level SSOT, derived into `.cursor/rules/project-guide.mdc`)
- `ALLOWLIST` / banned-import lists in `harness/framework_guard.py` (if applicable)
- `harness/tests/test_framework_guard.py` (if applicable)

Do not duplicate another boundary definition long-term in other scripts, ad-hoc instructions, or extra rule files. To add a new stage1 standing rule, append to the appropriate file in this directory; `agent-loop.sh` loads `overview.md` + `constraint.md` + the **active milestone's MD** every round via `harness/tools/agent_loop_config.py:stage_rule_files` (prior and future milestones' MDs are intentionally omitted from each round's prompt — see Core principle 10).
