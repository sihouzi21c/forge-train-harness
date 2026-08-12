# Forge-Train Harness

**A zero-touch agent loop that writes a real, trainable LLM training engine —
plus the development guideline that makes it reproducible.**

> Subproject of the [Forge-Train](../README.md) monorepo. The engine produced
> by this loop lives in [`../train_engine/`](../train_engine/).

[English](README.md) | [Chinese](README_zh.md)

This repository is a "code-self-driven" experiment: aside from a single
reference training script (any correct one works — this reproduction uses
Megatron-LM v0.15), the agent receives no human prompting. Once
`bash agent-loop.sh` starts, the coding agent — under strict gate constraints —
writes the entire MiniCPM4-0.5B training engine on its own, aligned bitwise
against the reference, and gated by long-horizon statistical tests
(long-train run vs the live ref baseline; loss rel diff < 1%; MFU
measured and audited at review). **The engine has subsequently been validated by a full
pretraining run** — not just gate-level smoke tests, but real end-to-end
MiniCPM4-0.5B pretraining.

> **Status**: validated by a full pretraining run; production-grade ready.
> Reproducing the Stage 1 long-horizon long-train / resume resume-gate-20 and Stage 2
> op-long gates end-to-end requires 8× H100; alignment / bitwise-singlecard single-card stages
> run on a single H100.

---

## Quick Start

### 1. Requirements


| Item                     | Version / Note                                                                                                                                    |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Python                   | 3.11+                                                                                                                                             |
| CUDA                     | 12.x (H100, sm_90a)                                                                                                                               |
| GPU                      | 8× H100 80GB for full Stage 1 long-horizon long-train / resume resume-gate-20 / Stage 2 op-long                                                                |
| GPU (minimal)            | 1× H100 for alignment / bitwise-singlecard single-card stages                                                                                                              |
| PyTorch                  | 2.3+ with CUDA 12 wheels (provides `torchrun` + NCCL)                                                                                             |
| Transformer Engine       | TE ≥ 1.7, matching whatever reference stack you use                                                                                               |
| Triton                   | bundled with the PyTorch CUDA wheel                                                                                                               |
| NVIDIA driver            | recent enough to expose `nvidia-smi`                                                                                                              |
| Cursor Agent CLI         | latest (`curl [https://cursor.com/install](https://cursor.com/install) -fsS                                                                       |


#### Dependency overview


| Layer           | Package                  | Purpose                                                                                                                  |
| --------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| **Core**        | `torch>=2.1`             | Training framework, provides `torchrun` and `torch.distributed`                                                          |
| **Core**        | `pandas`, `pyarrow`      | Data preprocessing (HuggingFace parquet → JSONL)                                                                         |
| **Megatron-LM** | from source              | Not pip-installed; clone source and point `MEGATRON_ROOT` at it                                                          |
| **Optional**    | `transformer_engine`     | TE ≥ 1.7, required for full gates (bitwise-multicard+, i.e. multi-card)                                                                |
| **Dev**         | `ruff`, `mypy`, `pytest` | Linting and testing                                                                                                      |


> **Note**: `pyproject.toml` declares core dependencies. `transformer_engine`
> is the only optional GPU-host extra (`pip install -e ".[full]"`); install it
> against your own CUDA / TE / reference-stack versions. Megatron-LM is used
> from source, not via pip. Real data is loaded through Megatron's standard
> `BlendedMegatronDatasetBuilder` + `GPTDataset` against the HuggingFace
> gsm8k dataset preprocessed by `ref/reference/prepare_gsm8k_data.sh` — no
> internal-only dataloader is required.

### 2. Install Harness

```bash
git clone https://github.com/OWNER/REPO.git training-engine-harness
cd training-engine-harness
python3 -m venv .venv
source .venv/bin/activate

# Base install (torch + pandas, enough for gsm8k benchmark)
pip install -e .

# Full install (with transformer_engine on the GPU host)
# pip install -e ".[full]"

# Dev tools
# pip install -e ".[dev]"

# NOTE: ``pip install -e .`` installs the Python package but does NOT
# create a global ``harness`` script. Each loop workspace ships its
# own ``<workspace>/bin/harness`` shim, auto-generated at provision
# time by agent-loop.sh and the web router with the workspace's
# absolute path baked in. From a manual checkout (no provisioned
# workspace), invoke the CLI as ``python3 -m harness.cli ...``.

python3 -m harness.cli doctor      # verifies CUDA / NCCL / TE / nvidia-smi
```

### 3. Prepare the reference training stack and data

The loop bootstraps from **any working LLM pretraining stack**.
The L0 ref script (determined by `config/ref.toml [ref].ref_script`,
e.g. `ref/reference/train_minicpm4_0.5b_gsm8k.sh`) does `cd $MEGATRON_ROOT`
and launches a real training process to produce the live baseline that every
gate compares against — **there is no cached baseline**, so a runnable
reference training script is required.

This reproduction uses Megatron-LM v0.15. Any other version, or any other
correct training script (DeepSpeed, custom, etc.), is adaptable — drop in your
own `ref/reference/*.sh` exposing the same env contract and the L0 SSOT will
follow automatically.

```bash
# 1. Bring up your reference training stack
git submodule update --init --recursive -- harness/third_party/megatron/v15

# 2. Create your run config — see CLAUDE.md "Bootstrap Guide" for the
#    seven orthogonal config axes (ref / data / remote / agent / eval /
#    model / optim). Each is `cp`'d from its `config/<axis>/*.toml`
#    template, e.g. `cp config/ref/megatron_minicpm4_0.5b.toml config/ref.toml`.

# 3. Preprocess data (gsm8k HuggingFace format → Megatron binary; output
#    defaults to .artifacts/data/gsm8k_megatron/)
export GSM8K_DIR=/path/to/hf/gsm8k/main
export TOKENIZER_MODEL=/path/to/MiniCPM4-0.5B/tokenizer.model
export MEGATRON_ROOT=$PWD/harness/third_party/megatron/v15
bash ref/reference/prepare_gsm8k_data.sh
```

Deployment paths (megatron_root, checkpoint_root, data_path, tokenizer_model)
live in the gitignored per-axis config files under `config/` (`ref.toml`,
`data.toml`, etc. — see `CLAUDE.md` Bootstrap Guide). Override priority
(low → high):

```
config/eval/dense_training.toml [defaults] (task-level fallback, no paths)
  < config/<axis>.toml               (gitignored; cp from config/<axis>/*.toml)
  < HARNESS_* env vars               (CI / one-shot runs)
  < CLI flags --megatron-root etc.
  < [source]/[data] auto-resolve     (fills remaining empty keys)
```
During gate runs the harness auto-derives gate-private SAVE_PATH /
TENSORBOARD_DIR under `.artifacts/ref_dump__<suite>/`; ad-hoc
`bash`-invocations land in `.artifacts/ref_local/<script-basename>/`.

`$HARNESS_CHECKPOINT_ROOT` is expected to contain
`canonical_state_fp32.pt` — the FP32 master state every harness suite
and every ours-side loader bootstraps from. The harness ships a
one-shot dumper for it:
`evals.harness_hook.install_canonical_state_dump(model, optimizer,
output_file=...)`. Wire it into the reference launcher once
(same bridge-author pattern as the alignment capture hook) and run the
launcher with `CANONICAL_STATE_OUTPUT_FILE=$HARNESS_CHECKPOINT_ROOT/canonical_state_fp32.pt`;
the process self-terminates on the first `optimizer.step` and leaves
the file in place. See
[`evals/harness_hook/recipes/README.md`](evals/harness_hook/recipes/README.md#canonical-state-bootstrap-recipe)
for the contract.

### 4. Prepare the evals, tools, and prompts

Beyond the reference training stack and data, the harness depends on three
companion directories to drive the agent loop to convergence. Together they
form the agent's "working environment" — evals define the gates, tools provide
the bridges, and prompts give targeted guidance.

#### Evals (`evals/`)

`evals/` contains every milestone's gate scripts and the dispatcher. The best
practice is to **avoid writing additional training scripts** and instead pull
new gates in via the declarative configuration in `config/eval.toml`
(env vars, gate thresholds, `runner_kind`, etc.), which keeps the SSOT — every
training spec lives only in the L0 ref script and `dense_training.toml`, never
duplicated in a third place.

To add a new gate suite:

1. Add an `[evals.<suite-name>]` section to `config/eval.toml`, declaring
   `stage`, `runner_kind`, `script`, `env_inputs`, gate thresholds, etc.
2. If needed, add the gate script to `evals/scripts/` (follow the existing
   output protocol: `[LOSS]`, `[PASS/FAIL]`, …).
3. Register the new `runner_kind` in `evals/dispatcher.py`'s `RUNNER_KINDS`
   (only when the kind is genuinely new).
4. Run `bin/harness info` to confirm the suite is registered, then verify with
   `bin/harness run <suite>`.

```bash
bin/harness info                  # confirm the eval registry
bin/harness info --json | jq .    # full suite contracts
```

#### Tools (`tools/`)

`tools/` holds the framework-agnostic operational scaffolding: the
ref-script subprocess runner, the per-op dispatcher, etc.

The alignment capture hook (`evals/harness_hook/install`) is the **only**
piece of alignment infrastructure the harness ships. It is pure-PyTorch and
makes no assumption about the customer's launcher. The wiring that
loads `install(...)` into the customer training stack — the so-called
"alignment capture bridge" — is **agent-generated**; see
`evals/harness_hook/recipes/README.md` for the contract and the
common interposition patterns.

In general, **stronger models need fewer tools** — a sufficiently
strong agent could write equivalent bridge code on its own.
Providing the following pieces materially improves reproducibility
and convergence speed:

- `tools/ref_script_runner.py` — bash-execs the script the dispatcher
  is told to run (either the customer ref script for bitwise-singlecard → long-horizon trajectory
  gates, or the agent-generated alignment bridge for forward / backward
  alignment) with the merged gate env and captures stdout / stderr /
  loss trace into a structured `RefRun`
- `evals/harness_hook/` — pure-PyTorch hooks for the two ref-side
  artifacts the harness consumes. Two public entries, no per-framework
  plugin layer:
  * `install(model, optimizer, output_file=...)` — alignment forward
    activations + per-param grads + execution-order graph;
  * `install_canonical_state_dump(model, optimizer, output_file=...)` —
    one-shot FP32 master-state dump, used to bootstrap
    `$HARNESS_CHECKPOINT_ROOT/canonical_state_fp32.pt`.
- `tools/stage2_op_router.py` — the runtime per-op version selector for Stage 2
- `tools/agent_loop_config.py` — SSOT for the stage list, rule file paths, etc.

If you switch to another training stack (e.g. DeepSpeed), you will need to
replace or adapt these tools accordingly.

#### Prompts (`prompt/`)

`prompt/` provides targeted agent guidance. Different training stacks may
have different soft constraints and code conventions, all reflected here:

- `prompt/review_prompt/coding-guidelines.md` — the core development contract
  (SSOT / DAG / Fail-Fast red lines), training-stack-agnostic
- `prompt/develop_prompt/_shared/stage1/` (alignment.md..long-horizon.md, overview.md,
  constraint.md) and `prompt/develop_prompt/_shared/stage2.md` — per-stage
  standing rules as backend-neutral union markdown. `stage1` covers the
  full alignment → production lifecycle (alignment forward/backward alignment → bitwise-singlecard single-card
  multi-step bitwise → bitwise-multicard DP=8 multi-step bitwise → bitwise-perf perf-bitwise
  (bitwise + MFU ≥ 14.5%) → resume resume bitwise → long-horizon long-horizon
  optimization including operator fusion). `stage2.md` is a thin
  orchestration SSOT paired with `stage2-subagent-playbook.md` containing
  all per-op subagent methodology / Drop-in constraints /
  experience-and-pitfalls; the playbook is `cat`-ed into each
  subagent prompt at bitwise-singlecard.1 dispatch time. Per-backend overrides may live
  at `prompt/develop_prompt/<backend>/<stage>/...` when a file resists
  sharing.
- `prompt/review_prompt/review_*.md` — review-agent audit protocols

If you target a different model architecture or training stack, update the
stage rules in `develop_prompt/` to match your specific constraints (e.g.
attention implementation differences, optimizer state layout, …).

### 5. Verify the harness

After install and configuration, sanity-check that the harness is working:

```bash
bin/harness info                  # human-readable workload metadata
bin/harness info --json | jq .    # machine-readable contract (suite registry, gate thresholds)
bin/harness run guard             # framework-import + path-isolation guard
bin/harness run unit              # unit tests
bin/harness sync push             # rsync local → remote (only when [remote].kind is ssh or devspace)
```

### 6. Run the Agent Loop — produce the training framework

This is the core step. One command starts the loop and the agent autonomously
iterates the full training engine source code into existence.

#### Minimal launch

```bash
export CURSOR_API_KEY=...           # required, Cursor Agent API Key
bash agent-loop.sh
```

#### Optional parameters


| Env var                          | Default                                      | Description                                                                                                                                                                                          |
| -------------------------------- | -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CURSOR_API_KEY`                 | (required)                                   | Cursor Agent API Key                                                                                                                                                                                 |
| `STAGES_OVERRIDE`                | all stages                                   | Pin to a subset of stages, comma-separated (e.g. `stage1`, `stage1,stage2`)                                                                                                                          |
| `RUNS_PER_STAGE`                 | `0` (unlimited)                              | Per-stage round cap. **`0` means unlimited** — the loop only stops when the review agent emits `STAGE_STATUS: finished`. Set a positive integer to cap a debugging session at N rounds. The default is set in `config/agent.toml` under `[agent].runs_per_stage`. |
| `MODEL`                          | (taken from `[agent]`)                       | Model used by the coding agent across both stages. Set in `config/agent.toml` under `[agent].model` (cp from `config/agent/*.toml`); override with `--model <slug>` or this env var. |
| `REVIEW_MODEL`                   | same as `MODEL`                              | Model used by the review agent (can be set independently)                                                                                                                                            |


#### Typical usage

```bash
# Goal-driven full pipeline: no round cap; auto stage1 → stage2,
# stops when the review agent emits STAGE_STATUS: finished.
export CURSOR_API_KEY=...
bash agent-loop.sh

# Run only Stage 1 (end-to-end training engine, alignment → production milestones,
# now including resume resume bitwise + long-horizon long-horizon optimization with
# operator fusion)
export CURSOR_API_KEY=...
export STAGES_OVERRIDE=stage1
bash agent-loop.sh

# Debugging: cap each stage at 3 rounds (cost-bounded)
export CURSOR_API_KEY=...
export RUNS_PER_STAGE=3
bash agent-loop.sh

# Reset finished-stage markers (force a re-run)
bash agent-loop.sh --reset-state

# Run only Stage 2 (per-op CUDA kernel optimization)
export CURSOR_API_KEY=...
export STAGES_OVERRIDE=stage2
bash agent-loop.sh
```

> **Note on workspace isolation**: each `bash agent-loop.sh` run executes
> inside an auto-provisioned copy of `harness/` at
> `.artifacts/forge_train/<loop_id>/workspace/` (the source tree is never
> mutated). The agent's per-round commits land in that workspace; to
> inspect, extract, or compare the generated engine across runs, look
> inside `.artifacts/forge_train/<loop_id>/workspace/workload/src/`.

#### Agent Loop runtime flow

Once `bash agent-loop.sh` starts, the framework runs the following procedure
automatically:

```
agent-loop.sh starts
      │
      ├─ read config/eval.toml + env vars
      ├─ resolve the list of stages to run
      │
      ▼
 ┌─ Per-stage loop ────────────────────────────────────┐
 │                                                     │
 │  1. Build the prompt                                │
 │     - coding-guidelines.md (core dev contract)      │
 │     - README.md (project-wide context)              │
 │     - prompt/develop_prompt/{backend}/{stage}.md (stg)│
 │                                                     │
 │  2. Invoke the Cursor Agent CLI                     │
 │     - agent reads prompt + bin/harness info → gates     │
 │     - agent writes code under workload/src/         │
 │     - agent runs bin/harness run <suite> to verify      │
 │     - on PASS, git commit                           │
 │     - on FAIL, exponential-backoff retry            │
 │                                                     │
 │  3. Review-Agent audit                              │
 │     - audits the coding agent's commit              │
 │     - checks SSOT / DAG / Fail-Fast red lines       │
 │     - emits REVIEW_VERDICT: PASS or FAIL            │
 │     - FAIL blocks subsequent rounds                 │
 │                                                     │
 │  4. Move to the next round                          │
 └─────────────────────────────────────────────────────┘
      │
      ▼
 Output: <workspace>/workload/src/training_engine_tensor/
         (a complete, trainable LLM training engine, where
          <workspace> = .artifacts/forge_train/<loop_id>/workspace/)
```

#### Outputs

After the run completes everything for this loop_id is rooted at
`.artifacts/forge_train/<loop_id>/`:


| Path (relative to `.artifacts/forge_train/<loop_id>/`)            | Description                                                                                  |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `workspace/workload/src/training_engine_tensor/`                  | The generated training-engine source (forward / backward / kernels / optimizer / dataloader / train_loop, …) |
| `workspace/workload/notes/perf_log.md`                            | Per-round performance log                                                                    |
| `workspace/.artifacts/agent-logs/<timestamp>/`                    | Full agent transcripts and review records                                                    |
| `session.json`                                                    | Per-loop registry record (status, pid, args, paths)                                          |


#### Stage breakdown

> **Note on the split**: GEMM and FlashAttention operator optimization is
> markedly harder than the rest, and different models behave very differently
> on this task, so per-op kernel optimization is carved out as Stage 2.
> Stage 1 covers everything else end-to-end (alignment forward/backward
> alignment, bitwise-singlecard single-card multi-step bitwise, bitwise-multicard DP=8 multi-step
> bitwise, bitwise-perf perf-bitwise (bitwise + MFU ≥ 14.5%), resume checkpoint
> save/load resume bitwise, long-horizon long-horizon optimization that
> includes operator fusion + system-level work). This split is one
> configuration that works, not the only valid one.


| Stage              | Content                                                                                                                                                                                                                                                  | Minimum GPUs                |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| **Stage 1 (alignment → production)** | End-to-end training engine: forward align → backward align → 1-GPU multi-step bitwise → DP=8 multi-step bitwise → perf-bitwise (bitwise + MFU ≥ 14.5%) → resume bitwise → long-horizon optimization (operator fusion + system-level) gated by `long-train` (loss rel < 1%; MFU reported, review-audited) **and** `resume-gate-20` regression | alignment / bitwise-singlecard: 1× H100; bitwise-multicard+: 8× H100 |
| **Stage 2**        | Per-operator CUDA kernel optimization: scout → per-op multi-round agent → best-MFU election → integration gate                                                                                                                                            | 8× H100                     |


---

## Two-stage gate pipeline

```
                bash agent-loop.sh   (zero human input from here on)
                          │
                          ▼
   ┌────────────────────────────────────────────────────────────┐
   │ Stage 1  alignment → production  end-to-end training engine                 │
   │   alignment  forward-align → backward-align (subprocess           │
   │       ref-vs-ours single-step tensor-dump diff via         │
   │       evals.harness_hook + agent-generated bridge)         │
   │   bitwise-singlecard  multistep-1gpu (DP=1, real GPTDataset,               │
   │       8-step ref-vs-ours bitwise)                          │
   │   bitwise-multicard  multistep      (DP=8, real GPTDataset,               │
   │       8-step ref-vs-ours bitwise)                          │
   │   bitwise-perf  perf-bitwise   (DP=8, real GPTDataset,               │
   │       50-step bitwise + MFU(standard) ≥ 14.5%)             │
   │   resume  resume-gate-20 (save/load bitwise round-trip,       │
   │       reference vs resume@10 self-comparison)              │
   │   long-horizon  long-horizon optimization (operator fusion +         │
   │       CUDA graph + comm/compute overlap + buffer reuse)    │
   │       Hard gates (both must pass):                         │
   │         long-train  (DP=8, loss rel < 1%;                  │
   │                       MFU reported)                        │
   │         resume-gate-20 regression (bitwise must hold)     │
   │       Aux (non-gating): loss-gate-200 (loss-only subset)  │
   └────────────────────────────────────────────────────────────┘
                          │
                          ▼
   ┌────────────────────────────────────────────────────────────┐
   │ Stage 2  Per-operator CUDA kernel optimization             │
   │   alignment scout wires dispatcher + scaffolds per-op worktrees   │
   │   bitwise-singlecard per-op subagent: per-round single-decision-tree       │
   │      iteration, op-status-driven cross-round progression   │
   │   op-long DP=8 long-train integration gate per merge       │
   └────────────────────────────────────────────────────────────┘
                          │
                          ▼
   the produced engine lands in ../train_engine/src/training_engine_tensor/
```

### Key invariants

- **Single source of truth for the gate**: every gate `shell-exec`s the L0
  reference script (`ref/reference/${ref_script}`, set by `config/ref.toml`).
  No frozen JSON, no SHA-anchored baseline — the reference can never
  silently drift.
- **Bitwise-first**: alignment → bitwise-perf require `max_abs_diff == 0` against the reference
  stack. No numerical fudge factors.
- **Evidence-driven rounds**: every round must commit and write
  `workload/notes/perf_log.md`. Stage 2 subagents
  additionally maintain a self-described cross-round trail in
  `workload/ops/<name>/notes.md` (per-round `## Round <N>` sections,
  `OP_LONG: FAIL #<N>` accumulation, self-judgement
  `STILL_ITERATING` / `READY_FOR_OP_LONG` markers) which is the only
  state carrier across subagent process restarts.
- **Path isolation enforced in code**: `harness/framework_guard.py` rejects
  out-of-surface commits (ref data, read-only spec summaries, etc.) at
  pre-commit time.

---

## Two core contributions

### 1. Zero-touch agent loop → a real training engine

`bash agent-loop.sh` orchestrates Coding-Agent rounds against the
per-stage SSOT prompt files in `prompt/develop_prompt/` and
`prompt/review_prompt/`. **No human in the loop.** The training engine
is produced entirely by the closed loop "run the gate → feed gate
evidence into the next round's prompt".

### 2. Development guideline

The core guideline lives in a single file —
**[`prompt/review_prompt/coding-guidelines.md`](prompt/review_prompt/coding-guidelines.md)**.
It is the actual contract that lets the agent loop converge on a real engine
without human steering. Hard rules:

- **SSOT / DAG / Fail-Fast** as three absolute zero-tolerance red lines.
- **Ref-script as gate** — every training-shape value comes from the L0 ref
  script; no second copy in TOML or Python.
- **Bitwise budget = 0** through bitwise-perf (vs reference) and resume (resume
  self-comparison). long-horizon keeps `resume-gate-20` at bitwise as a
  regression check; operator-fusion candidates introduced in long-horizon must
  not lower the compute dtype below the operator they replace, and
  the fused composition is validated end-to-end by the `long-train`
  statistical gate.
- **`workload/notes/perf_log.md`** is the only medium
  that carries knowledge across rounds.
- **Three-layer path isolation**: sparse-checkout (background) → pre-commit
  hook → review-agent re-audit.

The rest of `prompt/` (per-stage standing rules including the Stage 2 main
agent prompt + subagent playbook, review protocol) specializes the core
guideline above to specific stages and reviewers — supporting material, not
the contribution itself.

---

## Repository layout

```
agent-loop.sh             the orchestrator (stage loop, review pass, retry)
pyproject.toml            console_script entry point: `harness`
                          (NOTE: the monorepo's .pre-commit-config.yaml lives
                          one level up at the Forge-Train repo root and wires
                          ruff + framework_guard for deny-prohibited-framework-
                          references; it is shared with the rest of the monorepo)

harness/                  Framework CLI + control plane
  cli.py                  `harness` entry point (info / doctor / run);
                          `run` accepts a suite name — `guard`, `unit`, or any
                          gate suite from config/eval.toml [evals]
  app.py                  command dispatch, JSON output
  config_runtime.py       dense_training.toml + env + CLI override resolution (SSOT)
  framework_guard.py      banned-keyword scan + repo-write-surface path isolation
  presentation.py         human/JSON rendering of suite results
  transport.py            subprocess streaming bridge
  run_schema.py           harness-result JSON schema
  tests/                  ~260 unit tests (harness, dispatcher, prompt audit, …)

evals/                    Evaluation layer (gate registry + per-suite scripts)
  _common.py              shared helpers (config, log parsers, ref-bridge)
  dispatcher.py           unified suite registry across both stages,
                          including op-inventory / op-long / op-status
                          (Stage 2 gates; op-long is the sole algo-level
                          hard gate and uses an internal POSIX flock to
                          serialize GPU access. Per-round single-op
                          precision is run by subagents directly via
                          `python workload/ops/<name>/test_op.py`,
                          intentionally outside the harness gate set.
                          merge is plain `flock + git merge` written by
                          subagents — no `bin/harness run op-merge`)
  gate_common.py          common bitwise / loss gate utilities
  runner.py               subprocess boundary
  scripts/                frozen gate scripts — one ``eval_<...>.py``
                            per milestone (eval_capture_align.py for
                            alignment single-step tensor-capture diff,
                            eval_train_steps.py for bitwise-singlecard → bitwise-perf multi-step
                            trajectory, eval_resume_train.py for resume,
                            eval_long_train.py for long-horizon, plus the shared
                            launch_dp.py torchrun shim). Shell-out
                            targets of the dispatcher; never edited
                            by the agent. eval_train_steps.py is a
                            thin wrapper around
                            training_engine_tensor.train_loop.run_training_loop.

workload/                 Agent write surface — the actual training engine
  src/training_engine_tensor/
                          backward / forward / optimizer / parameters / nccl /
                          kernels / triton_kernels / dataloader / train_loop
                          (train_loop is the SSOT entry; gates import
                          `run_training_loop` from here. The submodule list
                          mirrors the `training_engine_tensor.train_loop`
                          docstring — that docstring is authoritative.)
                          (Stage 1 primary write surface across all of alignment → production;
                          dispatcher `get_op_version(...)` insertions in alignment
                          of Stage 2)
  ops/                    per-operator kernels + FP64 unit tests (Stage 2)
    _registry.toml          op registry maintained by the alignment scout milestone
    <op>/                   per-operator scaffolding: PROMPT.md / BASELINE.md /
                            kernel.py / register.toml / test_op.py / notes.md
                            (each op inlines its own FP64 reference + stat helper —
                            no shared `_shared/` module under workload/ops)
  notes/                  perf_log.md (per-iteration journal)
  profile/                in-tree profiling artifacts (captured_batch.pt, …)

tools/                    Harness ↔ reference-framework bridges + agent-loop glue
  ref_script_runner.py    dispatcher-side: env setup + ref-script bash-exec
                          primitive (framework-agnostic; the alignment
                          capture bridge that loads
                          evals.harness_hook.install is
                          agent-generated, see
                          evals/harness_hook/recipes/README.md)
  stage2_op_router.py     Stage 2 per-op runtime version selector
                          (reads workload/ops/*/register.toml)
  stage2_config.py        agent-loop shell exports for [automation.stage2]
                          (max_concurrent / max_op_long_failures)
  agent_loop_config.py    stage list / rule files / review templates (SSOT)
  profile_step.py         agent-writable scaffold: the stub prints a per-op
                          cuda-event timing table to stdout; the Stage 2 alignment
                          scout agent extends it (per stage2.md alignment.1) to add
                          kernel-name -> op-name mapping, runtime tensor probes
                          and a `--output` argument that writes
                          workload/profile/profile_result.json — the schema
                          consumed by every downstream Stage 2 milestone
  capture_profile_batch.py agent-writable: one-shot capture of a real micro-batch
                          into workload/profile/captured_batch.pt for profile_step

evals/harness_hook/       Standard alignment capture hook (SSOT)
  __init__.py             install(model, optimizer, *, output_file, ...)
                          Generic forward hook + grad collection + atomic
                          dump, then ordered teardown + SystemExit(0).
                          No framework imports.
  _module_hook.py         forward-hook walking named_modules() + execution
                          order trace.
  _grad_collector.py      per-param gradient harvest (main_grad → grad fallback)
  _dump.py                .pt + .graph.json atomic writer.
  recipes/README.md       contract + interposition patterns for the
                          agent-generated alignment capture bridge

ref/                      Frozen reference assets (REPO_FROZEN; SSOT)
  reference/train_minicpm4_0.5b_*.sh
                          L0 ref scripts (selected by config/ref.toml)
  reference/prepare_gsm8k_data.sh
                          gsm8k HuggingFace → Megatron binary preprocessor
  baseline/               reference run snapshots

prompt/                   Standing rules + agent prompts (the guideline itself)
  project-guide.md        master rule file (CLAUDE.md / .cursor/rules mirror it)
  develop_prompt/         per-stage agent prompts
    _shared/              backend-neutral union markdown (canonical)
      stage1/             alignment.md..long-horizon.md, overview.md, constraint.md
      stage2.md           thin main-agent orchestration SSOT
      stage2-subagent-playbook.md
                                          subagent methodology + per-round
                                          single-decision-tree protocol +
                                          notes.md self-described state
                                          anchors (cat-ed into each subagent
                                          prompt at bitwise-singlecard.1)
      reference/{attention,gemm}.md
                                          per-op-type domain knowledge,
                                          cat-ed verbatim into each
                                          workload/ops/<name>/PROMPT.md
    <backend>/            optional per-backend overrides (loaded only
                          when a same-named file is absent from _shared/)
  review_prompt/          review-agent prompts (review_common.md + per-stage)

config/                   dense_training.toml (task SSOT) + run/ profiles
.cursor/rules/            derived view of project-guide.md for Cursor IDE
CLAUDE.md                 auto-generated mirror of .rules/project-guide.md (pre-commit hook)
```

---

## Reproducibility

- **Stage 1** is model-tolerant: both GPT and Opus series have been
  observed to converge on the gates (alignment → bitwise-perf bitwise alignment, resume
  resume bitwise, and long-horizon long-horizon optimization). Pick whichever
  you have credit for.
- **Stage 2** currently reproduces stably only with Opus 4.7 (Claude). Other
  models have been observed to stall on the per-op kernel optimization loop.

Practical preconditions:

- Hardware: H100 (sm_90a). alignment / bitwise-singlecard run on a single GPU; everything else needs
  DP=8.
- Bootstrap: any working reference training stack at `$HARNESS_MEGATRON_ROOT`,
  the reference checkpoint (`canonical_state_fp32.pt`), and the gsm8k binary
  produced by `ref/reference/prepare_gsm8k_data.sh` — see
  [Prepare the reference training stack and data](#3-prepare-the-reference-training-stack-and-data).

---

## Limitations & future work

- **Milestones and test suites should be adapted to the model's capability.**
  The current alignment → production gates and their test suites are not necessary-and-sufficient
  conditions, but rather a configuration empirically verified on MiniCPM4-0.5B
  pretraining. The design principle is: milestones should reflect the smallest
  functional unit a model can stably achieve — stronger models can use sparser
  milestones (e.g. skipping intermediate alignment steps and directly verifying
  end-to-end loss convergence), while weaker models need finer-grained
  checkpoints to ensure correctness at every step. The specific milestone
  selection and gate thresholds should be tuned for the target model and
  training stack.
- **Stage 2 `attention.backward` does not yet reproduce stably.** This is the
  known weak point of the current per-op optimization loop.
- **The review agent's shape and value are not yet settled, and may evolve.**
  Its current form (a paired audit pass after each coding round) is one
  possible design among several; whether it should stay, be replaced by
  stronger mechanical checks, or fold into the gate contract itself is an
  open question that may evolve in later releases.
- **This repository is a sufficiency witness, not a minimum.** Some of the
  infrastructure shipped here (control-plane scripts, prompt files, unit
  tests, etc.) could itself be authored by the agent loop. The project shows
  that this configuration works end-to-end; it does not claim to be the
  minimal reproducible harness.

---

## License

Apache-2.0. See the monorepo top-level [LICENSE](../LICENSE).

Built on a reference training stack (Megatron-LM v0.15 in this reproduction)
and the Cursor Coding Agent; data and tokenizer follow MiniCPM4-0.5B upstream.
