# Training Engine Harness

This file is the single source of truth for repository-scoped project guidance.

## IMPORTANT — Workload Rule Loading

Interactive planning/review tasks should read all relevant files under
`prompt/develop_prompt/` to understand the full repository contract.

`agent-loop.sh` uses stage-specific rule loading for autonomous rounds:
each stage prompt includes only the current stage rule file
(`prompt/develop_prompt/<backend>/<stage>.md`) plus the shared coding
guidelines. This keeps the Stage 1 / Stage 2 instructions focused.

## Overview

This is a unified harness covering two development stages for dense model training engine:

### Stage 1 — End-to-end Training Engine (alignment → long-horizon)

Aligns the custom engine against ref training, then implements
checkpoint save/load (resume) and finishes with long-horizon optimization
that includes operator fusion (long-horizon). alignment → resume require bitwise exact match
(max_abs_diff == 0); long-horizon switches to the long-horizon statistical gate
plus a resume-gate regression that operator fusion / system-level
optimizations must not break.

Every Stage 1 gate uses the same architectural shape — ref-vs-ours
subprocess, both sides driven by `evals/scripts/eval_<milestone>.py`
(ours) and the L0 ref script — with two flavours of
comparison wire format:

* **alignment (tensor-dump diff)** — single-step single-card. ref dumps the
  union of forward activations (`rank<r>.mb0.fwd.<fqn>#<call>` /
  `rank<r>.mb0.bwd.<fqn>#<call>`, where `rank<r>.` is the global rank
  (`rank0` single-card), `mb0` the single microbatch, and `#<call>` the
  per-forward call index that keeps a shared module's re-uses distinct —
  e.g. an `output` head on the main + MTP paths → `#0` / `#1`) and parameter
  gradients (`rank<r>.grad.*`) to disk via the agent-generated alignment
  capture bridge
  (`[defaults].ref_capture_script`; the bridge loads
  `evals.harness_hook.install` into the customer training stack — see
  `evals/harness_hook/recipes/README.md`).
  ours runs `training_engine_tensor.train_loop.run_training_loop` with
  `TrainLoopConfig.capture_output_file` set (the dispatcher passes the
  full path via `HARNESS_CAPTURE_OUTPUT_FILE`; the candidate never picks
  the basename) so the engine short-circuits after one fwd+bwd and
  writes the matching dict. The dispatcher diffs the two dicts on
  `fwd.` or `grad.` prefix.
* **bitwise-singlecard onwards (trajectory diff)** — multi-step. Each side runs the
  full window with the same dataloader and same dataset; per-step `[LOSS]` trajectories are compared
  bit-for-bit on a gate window pulled from `gate_metadata.json`.

| Milestone | Suite(s) | Description |
|-----------|----------|-------------|
| alignment | `forward-align`, `backward-align` | Single-step single-card forward+backward bitwise (ref-vs-ours subprocess + tensor-dump diff) |
| bitwise-singlecard | `multistep-1gpu` | Single-card multi-step training bitwise (DP=1, real dataloader and dataset) |
| bitwise-multicard | `multistep` | Multi-card multi-step training bitwise (DP=2, real dataloader and dataset) |
| bitwise-perf | `perf-bitwise` | DP=2 bitwise + MFU gate (real dataloader and dataset; shape + `mfu_e2e_target` in `config/eval.toml [evals.perf-bitwise]`) |
| resume | `resume-gate-20` + `resume-startup-90` | Resume bitwise (self-comparison: reference vs save@10→load) **and** resume-startup budget (`resume-startup-90`: resume from step 90, dataloader seek to checkpoint position ≤ `[evals.resume-startup-90].resume_startup_budget_s` — an O(1) cursor seek, NOT replaying prior steps) |
| long-horizon | `long-train` + `resume-gate-20` regression | Long-horizon optimization (incl. operator fusion + system-level): long-horizon statistical gate (`loss_rel_threshold` in `config/eval.toml [evals.long-train]`; MFU is measured and reported, throughput sufficiency is judged at PR review), resume gate must keep passing. Auxiliary `loss-gate-200` available as a non-gating loss-only subset. |

### Stage 2 — Per-Operator Kernel Optimization

Optimizes individual operators for H100 throughput. Operators must be strict drop-in replacements.

| Suite | Description |
|-------|-------------|
| `op-inventory` | M1 gate: validates scout artifacts (registry / scaffolding / worktrees / dispatcher wiring) |
| `op-long [names]` | M2 per-op hard gate: DP=2 200-step integration vs ref-script. **Only** unforgeable algo-level gate; each subagent self-drives once it judges (in its single per-round decision tree) that no further optimisation space remains |
| `op-status` | M2 aggregate verdict (merged / failed per op) |

Per-round single-op precision is **not** a harness gate. Subagents run `python workload/ops/<name>/test_op.py` directly inside their own worktree; harness ignores its stdout. The only algo-level pass/fail decision the harness owns is `op-long`. See `prompt/develop_prompt/_shared/stage2-subagent-playbook.md` "Trust Model" for rationale.

## SSOT — Ref Script as Gate

The **only** baseline truth is the L0 ref script (`ref/reference/${ref_script}`, determined by `config/ref.toml [ref].ref_script`). Gates shell-exec the ref script directly. No frozen JSON, no SHA anchor.

## Execution Model

All workloads execute **locally** on the machine's GPU(s). The `agent-loop.sh` script orchestrates two stages sequentially, each with configurable rounds.

## Commands

```bash
# Stage 1
bin/harness run forward-align     # alignment.forward
bin/harness run backward-align    # alignment.backward
bin/harness run multistep-1gpu    # bitwise-singlecard  (DP=1 real-dataloader bitwise)
bin/harness run multistep         # bitwise-multicard  (DP=2 real-dataloader bitwise)
bin/harness run perf-bitwise      # bitwise-perf  (DP=2 bitwise + MFU gate)
bin/harness run resume-gate-20   # resume + long-horizon regression
bin/harness run resume-startup-90 # resume startup-time budget (seek ≤ 15s, no replay)
bin/harness run long-train        # long-horizon (long-horizon statistical gate)
bin/harness run loss-gate-200    # auxiliary (long-horizon loss-only subset, no MFU gate)

# Stage 2
bin/harness run op-inventory          # M1 gate
bin/harness run op-long [names]       # M2 per-op hard gate (subagent self-drives once it self-judges convergence)
bin/harness run op-status             # M2 aggregate verdict
# Per-round local precision check is run by subagents as
#   flock -s .artifacts/locks/stage2-gpu.lock python workload/ops/<name>/test_op.py
# in their worktrees — not a harness gate (see stage2-subagent-playbook.md
# "Trust Model" + "Resource Contract" sections for the trust model and flock contract).

# Infrastructure
bin/harness run guard
bin/harness run anti-proxy
bin/harness run unit
bin/harness doctor
bin/harness info
bin/harness budget <suite>            # print SSOT wall-clock budget (seconds)
bin/harness sync push                 # rsync local → remote (when [remote].kind is ssh or devspace)
bin/harness sync push --stage2        # include .git (needed for op-status)
```

## Machine Contracts

`bin/harness info --json` exposes the run request/result contract under
`payload.contracts.run`, including schema version, required request/result
keys, artifact filenames, and stdout result markers. External automation
should read that metadata instead of copying schema details from source files.

## Agent Loop

The loop is **goal-driven**: it iterates each stage until the review agent
emits `STAGE_STATUS: finished` for that stage, then advances. The default
`runs_per_stage` in `dense_training.toml` is `0` (= unlimited rounds). Per-stage
status is persisted under `.artifacts/agent-loop-state/` so a restart
resumes from the first non-finished stage.

```bash
# One-shot: run stage1 then stage2 to completion (no round cap)
CURSOR_API_KEY=... bash agent-loop.sh

# Cap each stage at N rounds (debugging / cost-bounded)
RUNS_PER_STAGE=3 MODEL=<optional-agent-model-override> bash agent-loop.sh

# Force-restart, ignoring any previously persisted "finished" markers
bash agent-loop.sh --reset-state
```

Tunables (config/agent.toml `[agent]`):

- `runs_per_stage`: per-stage round cap; `0` = unlimited.
- `max_consecutive_review_fails`: consecutive `REVIEW_VERDICT: FAIL`
  rounds tolerated before aborting (value owned by `dense_training.toml` —
  see its `[agent]` comment for the default and the recommended
  override range). One-off FAILs get an extra dev round to self-heal;
  persistent failures still surface.
- `state_dir`: per-stage status file directory (gitignored).

### Unified Agent Logging

Every agent the loop spawns — dev, review, and Stage2 per-operator
subagents — runs through the same web-displayable transcript
pipeline. There is no separate plain-text log surface anymore.

- Each spawned agent gets a `web-agents/<agent_id>/` directory
  with a `session.json` (metadata) and `stdout.log` (stream-json).
  `harness/tools/spawn_managed_agent.py` is the single entrypoint —
  `agent-loop.sh` and the Stage2 fan-out templates both call it.
- `agent-loop.sh` itself owns a synthetic `loop-<id>` Session row
  written via `harness/tools/loop_wrapper_init.py` +
  `loop_wrapper_event.py`. Its `stdout.log` carries typed
  `loop_event` records (`stage_start`, `round_start`, `spawn_child`,
  `review_verdict`, `stage_status`, `loop_exit`, `info`) that the
  web frontend renders as compact orchestration pills inline with
  the agent chat content.
- The web Loop tab consumes the unified pipeline via
  `/api/agent/loop-<id>/{messages,events}` — the same endpoints any
  other chat agent uses. `spawn_child` rows in the wrapper
  transcript link to the child's own chat in the Agents tab.

## Build

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
bin/harness doctor
```

## Configuration

The repo's seven orthogonal config axes — `[ref]`, `[data]`, `[remote]`,
`[agent]`, `[eval]`, `[model]`, `[optim]` — and their `config/*.toml`
locations + templates are documented in `.rules/bootstrap-guide.md`
(also auto-mirrored to `CLAUDE.md` / `AGENTS.md` /
`.github/copilot-instructions.md`). That guide is the single source
of truth for the config layout; consult it instead of duplicating
the table here.

## Repository Layout

```
harness/        Framework CLI and control plane. pip console_script entry point.
evals/          Evaluation layer.
                ├── _common.py      shared helpers (config, parsers, ref-script bridge)
                ├── dispatcher.py   unified suite registry (all stages)
                ├── runner.py       subprocess boundary the harness invokes
                └── scripts/        frozen gate scripts — shell-out targets
                                    of the dispatcher, not edited by the
                                    agent. Agent-writable scout-phase utilities
                                    (profile_step.py / capture_profile_batch.py)
                                    live under tools/.
workload/       The implementation under test.
                ├── src/            training engine (forward/backward/optimizer/dataloader),
                │                   primary write surface for stage1 agents (alignment → long-horizon)
                ├── ops/            per-operator kernels, prompts, tests (stage2 surface)
                └── notes/          experiment log (perf_log.md)
tools/          Ref-as-gate helpers, stage2 op runtime, orchestration glue,
                and agent-writable profiling tools (profile_step.py,
                capture_profile_batch.py — Stage 2 M1 extends these in place)
                (agent_loop_config, stage2_*.sh, …).
ref/            Read-only ground truth (reference scripts + baseline).
config/         Configuration: project-level + user-local.
prompt/         Standing rules and prompts.
  review_prompt/  Review and coding guidelines prompts.
  develop_prompt/     Workload stage rules and references.
harness/tests/  Framework / contract unit tests.
```

## LLM Permission Matrix

The agent loop covers the **training engine** end-to-end. Active write surface
includes both the training framework implementation and the harness control
plane. The only "frozen" pieces are external reference data, the L0 ref
script's `HARNESS_GATE` preset values, and — only in Stage 1 long-horizon —
the frozen GEMM + attention call surface (see `long-horizon.md` §long-horizon stage operator freeze constraint).

Agent scopes (used by different sub-loops):

- **Stage 1 agent** (`stage1.md` prompt) — implements and optimizes the
  training engine end-to-end (alignment → long-horizon), including checkpoint save/load
  (resume) and long-horizon optimization with operator fusion (long-horizon). Write
  surface includes `workload/src/`, the harness control plane, and
  prompts/configs.
- **Stage 2 main agent** (`stage2.md` prompt) — runs the M1 scout milestone
  (dispatcher wiring at the FlashAttention + GEMM call-sites under
  `workload/src/training_engine_tensor/` — typically `forward.py` /
  `backward.py` / `kernels.py`; out-of-scope kernels such as those in
  `triton_kernels.py` carry no wiring — plus `workload/ops/<name>/`
  skeletons + per-op worktrees), then the M2 optimize milestone
  (fan-out per-op subagents via cursor CLI; verdict via
  `bin/harness run op-status`).
- **Stage 2 per-op subagent** (`stage2-subagent-playbook.md` + the op's
  `PROMPT.md`) — runs inside a sparse worktree on a `stage2/op/<name>`
  branch and may edit **only** `workload/ops/<name>/`. The broader codebase
  is readable context for that subagent, not its write surface. This
  isolation is enforced by pre-commit hook and review agent (see
  `review_stage2.md` Check C).

Per-path summary:

- `workload/src/` — read/write; training engine implementation
- `workload/ops/<name>/` — read/write; per-op kernel + tests + notes (each op writes its own helpers — no shared utility module)
- `workload/ops/*/PROMPT.md` — read/write; per-op subagent prompts
- `workload/notes/` — read/write; `perf_log.md` is a required output every round
- `workload/profile/` — runtime-generated profile artifacts; git-ignored
- `harness/`, `harness/tests/` — read/write; CLI / config / runtime / unit tests
- `evals/` — read/write; suite dispatcher, parsers, artifact/result contracts
- `prompt/`, `.cursor/rules/`, `README.md` — read/write; standing rules
- `config/eval.toml` — read/write; suite parameters and harness↔workload contract
- `config/{ref,data,remote,agent,model,optim}.toml` — read/write; the orthogonal axes (gitignored; cp from `config/<axis>/*.toml`). See `.rules/bootstrap-guide.md` for the full layout.
- `tools/` — read/write; ref-script helpers, stage2 op runtime, orchestration loops and stage automation
- `ref/` — **read-only**; baseline + reference scripts (the L0 SSOT for gates)
- `$MEGATRON_ROOT` (see `config/eval.toml [defaults].megatron_root`) — read-only; reference implementation
