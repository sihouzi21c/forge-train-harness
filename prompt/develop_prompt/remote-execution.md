# Remote Execution Overlay (injected when `[remote].kind` is `ssh` or `devspace`)

> This document is the **single source of truth** for the agent's
> remote-execution **transport** procedure (sync / launch / log
> retrieval), shared by both `ssh` and `devspace` protocols.
> `agent-loop.sh:build_prompt()` injects it into the dev / review
> system prompt **only** when `config/remote.toml [remote].kind` is
> `ssh` or `devspace`; when `kind = "local"` (or the table is absent),
> this file is not part of the prompt and the agent runs every command
> locally. For `kind = "devspace"` the lifecycle overlay
> `remote-execution-devspace.md` (drop recovery + lease rebind) is
> appended after this file.
>
> Placeholders below are substituted at injection time from the
> `[remote]` config table (see `config/remote/ssh.toml` and
> `config/remote/devspace.toml` for the commented templates). The
> `@@KEY@@` syntax is chosen so the substitution does not collide with
> `${var}` shell expansions inside the command examples —
> `agent-loop.sh` runs a single `sed` pass at inject time to replace
> each `@@KEY@@` literal.

> **RED LINE — SUITE TIMEOUT IS SSOT-OWNED; AD-HOC COMMANDS MUST SELF-GUARD**
> (ABSOLUTE ZERO TOLERANCE)
>
> Scope: the SSOT / no-override rules below govern **suite execution
> only** — any `bin/harness run <suite>`, run directly or via
> `tools/remote_run.sh`. Ad-hoc / exploratory commands (a bare `ssh`
> probe, a `find` / `cat` / `ls` round-trip, a one-off pipeline) are
> explicitly OUT of that scope and are governed by the self-guard
> clause at the bottom of this block.
>
> - **MANDATORY**: Each suite's wall-clock budget is determined
>   exclusively by `harness/config/eval/dense_training.toml`
>   `[evals.<suite>].timeout_s`. The runtime is enforced by
>   `bin/harness run` itself (dispatcher + transport `killpg` backstop).
>   Query the current budget via `bin/harness budget <suite>`; never
>   hardcode a suite's timeout seconds in prompt text, shell, or Python.
> - **FORBIDDEN**: Running a suite OUTSIDE the budgeted path —
>   hand-rolling an `ssh ... python -m ...` that reproduces a suite and
>   bypasses `bin/harness run`.
> - **FORBIDDEN**: OVERRIDING the preset budget — wrapping
>   `bin/harness run` / `tools/remote_run.sh` with an outer
>   `timeout NNN` / `gtimeout NNN` / `coreutils-timeout`, or using
>   `bin/harness run --timeout SECS` — UNLESS all of (a)–(d) hold after
>   **deep analytical thinking and repeated verification**: (a) read the
>   suite's `_run_*` function in `harness/evals/dispatcher.py` in full;
>   (b) read the complete log of the most recent run; (c) rule out — by
>   evidence, not guesswork — deadlock, infinite loop, performance
>   regression, dependency hang, and network flakiness; (d) confirm
>   beyond reasonable doubt that the **only** explanation is that the
>   toml budget is genuinely too small. Vague or speculative "maybe the
>   budget is too small" judgements are not allowed.
> - **MANDATORY (self-guard for out-of-scope commands)**: An ad-hoc
>   remote / exploratory command that is NOT a suite run MUST carry its
>   own bound so it can never hang forever. SSH probes inherit
>   connection keepalive (`ConnectTimeout` + `ServerAlive*`) from the
>   `ds-*` / `devspace-*` stanza in `~/.ssh/config`, and the agent MAY
>   additionally wrap such a non-suite command in a reasonable
>   `timeout NNN`. A deadlocked probe that stalls the loop is itself a
>   violation — we do NOT want unbounded probes.
> - **VIOLATION**: Running a suite outside the budgeted path, overriding
>   a suite budget without the (a)–(d) audit, OR letting an unbounded
>   ad-hoc command hang the loop = **immediately discard the current
>   round and roll back**.

## Remote target (resolved from `config/remote.toml [remote]`)

| Key | Value |
|---|---|
| SSH host alias (`~/.ssh/config`; from `[remote].hostname`) | `@@REMOTE_SSH_HOST@@` |
| Remote workdir (`<[remote].workspace or $HOME>/.forge_train/<loop_id>`) | `@@REMOTE_WORKDIR@@` |
| This loop's id (echoes `LOOP_ID`; used in env passthrough) | `@@REMOTE_LOOP_ID@@` |

The harness materializes each loop's remote checkout under
`<[remote].workspace>/.forge_train/<loop_id>` so sibling loops on the
same devspace never overlap on disk. An empty `workspace` defaults to
the literal string `$HOME`, which the remote shell expands at
command-execution time, so the workdir resolves on the target host
(e.g. `/user/<username>/.forge_train/a55a6f96`) without any
client-side `$HOME` knowledge.

> **`[remote].workspace` is user-owned config — do not modify it.**
> If you suspect the workspace path is wrong (collision with a
> sibling agent, devspace storage layout drift, OOM on a tmpfs
> mount), surface the issue as a blocker instead of editing
> `config/remote.toml`. The wrapper composes the per-loop
> `.forge_train/<loop_id>` suffix itself; the value you read in
> `@@REMOTE_WORKDIR@@` is already loop-isolated.

## Calling the harness CLI

Harness is **per-workspace**. Every workspace ships its own
`bin/harness` shim — auto-generated at provision time by
`agent-loop.sh` with the workspace's absolute path baked into the
`PYTHONPATH` export. The shim then `exec`s `python3 -m harness.cli`.

| Where | How to invoke |
|---|---|
| Local | `bin/harness <cmd>` |
| Remote (after `bin/harness sync push`) | `ssh @@REMOTE_SSH_HOST@@ 'cd @@REMOTE_WORKDIR@@ && bin/harness <cmd>'` |

The shim is part of the workspace tree, so every `bin/harness sync
push` rsyncs it to the remote alongside the source code — the same
`bin/harness <cmd>` form works on both sides.

**Do NOT** do any of the following — they are remnants of the
retired global-CLI path and will waste a round of debugging:

- `pip install -e .` (locally or remotely). The shim does not depend
  on a console-script entry point; installing the package wheel adds
  nothing.
- `which harness` / looking for `~/.local/bin/harness`. There is no
  global binary on purpose — each workspace pins its own harness
  version.
- `PYTHONPATH=$PWD python3 -m harness.cli ...`. The shim already
  does this; only use the long form in a manual git checkout that
  has no provisioned workspace.

## Runtime probing of CUDA-only packages

`flash_attn`, `apex`, `transformer_engine`, `megatron`, and any other
CUDA-only wheel are **not importable on the local Mac**. Do not loop
over local conda envs hunting for them — every attempt will fail with
`ModuleNotFoundError`.

To inspect a runtime signature / dtype / shape contract, ssh the
remote and let Python introspect it where it actually loads:

```bash
ssh @@REMOTE_SSH_HOST@@ \
  "python3 -c 'import inspect, flash_attn; print(inspect.getsourcefile(flash_attn._flash_attn_backward)); print(inspect.signature(flash_attn._flash_attn_backward))'"
```

To statically read source — when a local mirror exists, prefer that
over an ssh round-trip:

- `flash_attn`: `/local-example/Projects/flash-attention/` (read
  the `flash_attn/flash_attn_interface.py` file directly).
- Anything else without a local mirror: ssh + `cat` the file the
  remote interpreter resolves it to.

**Bootstrap assumption**: the remote host was prepared before the
agent loop starts:

- `~/.ssh/config` already has the `[remote].hostname` alias wired
  with whatever proxy / cert / port the deployment needs, so plain
  `ssh @@REMOTE_SSH_HOST@@ <cmd>` works without password / port /
  identity flags. For `kind = "devspace"`, the lease tool creates a
  per-loop devspace from the configured spec, discovers its Teleport
  node via `tsh ls`, and appends a matching `Host <new-alias>`
  stanza to `~/.ssh/config` before the wrapper freezes the per-loop
  config dir (see the appended devspace overlay). That synthesized
  stanza carries connection keepalive (`ConnectTimeout` +
  `ServerAliveInterval` / `ServerAliveCountMax`), so a bare
  `ssh @@REMOTE_SSH_HOST@@ <cmd>` fails fast if the devspace is
  black-holed (half-open TCP after a reclaim) instead of hanging
  forever — you do not need to add `-o` flags to inherit it.
- `rsync` (3.x) is on the remote non-interactive `$PATH` (e.g.
  conda-installed and the conda bin dir added to `~/.bashrc`).
- If the host auto-sets `http(s)_proxy` that blocks
  `pypi.org` / `github.com`, the proxy is already unset in the
  remote shell profile so `pip` / `curl` to public hosts work
  without per-command workarounds.

## Local↔remote contract

**You own local↔remote sync, remote command launch, and remote log
retrieval — no user intervention is required during a round.** The
local workspace is the SSOT for code (write / lint / unit-test /
commit); the remote is the GPU execution host.

### Where to run what

| Workload | Where | Why |
|---|---|---|
| `bin/harness run guard` / `bin/harness run anti-proxy` / `bin/harness run unit` / `pytest harness/tests/` / `bin/harness info` / `git ...` / `ruff ...` / code editing / commits | **Local** | CPU-only / no GPU needed; git truth lives locally |
| Every other `bin/harness run <suite>` — `forward-align`, `backward-align`, `multistep-1gpu`, `multistep`, `perf-bitwise`, `resume-gate-20`, `long-train`, `loss-gate-200`, `op-inventory`, `op-long`, `op-status` | **Remote** | GPU required, OR (for op-status) requires the remote git truth that other GPU suites mutate |
| `bin/harness doctor` when probing remote env | **Remote** | needs to see the remote CUDA / NCCL stack |
| `bash agent-loop.sh` itself | **Local** | the loop drives both sides; only the suites it launches go remote |

### Step 1 — Sync local → remote (after every code change, before any remote command)

```bash
bin/harness sync push
```

For stage2 (when `.git` must be present on the remote for `op-status`):

```bash
bin/harness sync push --stage2
```

Do **not** construct rsync commands manually — `bin/harness sync push` is the
SSOT for the exclude list, post-sync `__pycache__` cleanup, and remote
file validation. It will:

1. `rsync --delete` the workspace to the remote (excluding `.git`,
   `.venv`, `__pycache__`, `.artifacts`, caches, and `workload/profile`).
2. Re-ship the active per-loop config (`$FORGE_CONFIG_DIR/*.toml`) into
   the remote `<workdir>/config/` with a dedicated, **`--delete`-free**
   rsync. The active config lives in the sibling
   `.artifacts/forge_train/<id>/config/`, which step 1 excludes via
   `.artifacts`, and `$FORGE_CONFIG_DIR` never crosses the ssh hop — so
   this step is the ONLY thing that lands the live `ref/data/remote/
   agent/eval/model/optim.toml` on the remote, where `bin/harness run` reads
   them from `<workdir>/config`. **Never hand-push config `*.toml` to
   the remote yourself** — the next `sync push --delete` would orphan
   files that aren't in the source tree; let this step own it.
3. Purge **all** `__pycache__` directories on the remote (prevents stale
   bytecode from a previous round causing `ModuleNotFoundError`).
4. Validate that critical files exist on the remote (`pyproject.toml`,
   `harness/config/defaults.toml`, `config/eval/`, `evals/`, etc.) and
   fail-fast with a diagnostic message if any are missing.

### Step 2 — Run a remote command (fail-fast)

```bash
ssh @@REMOTE_SSH_HOST@@ \
  "set -euo pipefail \
   && export LOOP_ID=@@REMOTE_LOOP_ID@@ \
   && export FORGE_TRAIN_DIR=@@REMOTE_WORKDIR@@/.artifacts/forge_train \
   && export FORGE_CONFIG_DIR=@@REMOTE_CONFIG_DIR@@ \
   && cd @@REMOTE_WORKDIR@@ && <cmd>"
```

`set -euo pipefail` is mandatory — silent non-zero exits are forbidden.

`LOOP_ID` / `FORGE_TRAIN_DIR` are required: `bin/harness run` writes MFU
records via `harness/tools/mfu_record.py`, which uses them to
attribute the measurement to the right loop's
`<FORGE_TRAIN_DIR>/<LOOP_ID>/mfu_history.jsonl`. Without these, MFU
telemetry silently vanishes and the wrapper's MFU badge never
updates.

`FORGE_CONFIG_DIR` is required too: it points the remote
`config_runtime._user_config_dir()` at the `<workdir>/config` that
`bin/harness sync push` (Step 1.2) lands the active per-loop `*.toml`
into. Without it the remote falls back to `repo_root()/config`, which
happens to be the same path only because `cd @@REMOTE_WORKDIR@@`
precedes the command — exporting it explicitly removes that hidden
coupling so the remote reads the SSOT config regardless of cwd.

### Step 3 — Inspect remote files / artifacts in place (preferred over pull-back)

```bash
ssh @@REMOTE_SSH_HOST@@ \
  'tail -200 @@REMOTE_WORKDIR@@/.artifacts/runs/<run>/<file>.log'
ssh @@REMOTE_SSH_HOST@@ \
  'cat   @@REMOTE_WORKDIR@@/.artifacts/runs/<run>/result.json'
```

### Step 4 — Pull remote artifacts back to local (only when you must read large files locally)

```bash
rsync -avz \
  @@REMOTE_SSH_HOST@@:@@REMOTE_WORKDIR@@/.artifacts/runs/<run>/ \
  "${WORKSPACE}/.artifacts/runs/<run>/"
```

### One-time remote bootstrap

None. `bin/harness sync push` rsyncs the workspace's `bin/harness`
shim alongside the source tree, so the remote is immediately able to
run `ssh @@REMOTE_SSH_HOST@@ 'cd @@REMOTE_WORKDIR@@ && bin/harness
info'` after the first sync — no editable install, no PATH wiring.

### Long-running commands

The resume-gate, long-train, and op-long suites can run for tens of
minutes to multiple hours. Always size waiters from the SSOT via
`bin/harness budget <suite>` (prints integer seconds) — never hardcode a
duration here or in the foreground call. Two acceptable patterns:

- **Foreground**: read the budget with `B=$(bin/harness budget <suite>)`,
  then size `block_until_ms` on the foreground ssh call to
  `(B + a few minutes) * 1000`.
- **Background**: launch via
  `nohup <cmd> > ./.artifacts/remote_logs/<run>.log 2>&1 & echo $! > <pidfile>`,
  record the remote PID, and poll `ssh ... 'tail -100 ...'` on
  backoff (2s → 4s → 8s → 16s → 30s, then steady 30s thereafter).
  Recommended for the longest suites so the single ssh connection does
  not get killed by a proxy idle timeout.

### Fail-fast contract for remote calls

- Every ssh / `bin/harness sync push` call must propagate the remote exit
  code (use `set -euo pipefail` inside the ssh command body; check `$?`
  after `bin/harness sync push`).
- Never wrap remote calls with `|| true`. SSH transport failure
  (network / auth / proxy timeout) is a hard error.
- If a remote suite exits 0 but stderr is non-empty, surface stderr
  in your round notes regardless.

## Stage-specific overrides

### Stage 2 — `op-status` needs remote git truth

The default rsync template above uses `--exclude='.git'` (gates only
need source code on the remote, and `.git` adds tens of MB). **Stage 2
breaks this rule**: `bin/harness run op-status` reads
`ops_worktree/<op>/.git` (the gitlink file) and the matching
`.git/worktrees/<op>` metadata to decide if each operator is in
`merged` / `failed` / `not_started` / `inconsistent`. Without git on
the remote, the status reduces to "everything is not_started" and the
entire M2 verdict layer breaks.

Use this stage2-specific sync (note **no** `.git` exclusion):

```bash
bin/harness sync push --stage2
```

`.git/objects/` in packfile form is usually under a few hundred MB; a
single rsync sync is cheap enough. Do not skip `ops_worktree/` to save
bandwidth — `op-status` reads `.git/worktrees/<op>` to tell
"worktree still present" apart from "worktree clean-deleted after a
successful merge".

### Stage 1 — Single-GPU suites still go remote

Bitwise gates must run on the same machine as the reference — a
different framework wheel (or any other environmental delta versus the
ref host) can produce 1-ULP drift even on a local GPU of the same SKU.
Always go remote, regardless of how many GPUs the suite uses.

### Subagent fanout (stage2 M2.1)

`cursor agent CLI`'s `agent -p --workspace ops_worktree/<op>` still
runs the subagent process **locally** (the loop fans out from the
local main agent). Each subagent, in turn, owns its own
`bin/harness sync push → ssh run → ssh tail` cycle against the remote —
same contract as the main agent. The `flock + git merge` step on the
PASS path operates on the local git tree; merge completion is then
synced back to remote on the next `bin/harness sync push` to trigger the
subsequent regression suites.

## Wall-clock budgets

Per-suite wall-clock budgets live in
`harness/config/eval/dense_training.toml` under
`[evals.<suite>].timeout_s` and are enforced by `bin/harness run` itself.
Query the active budget for a suite with `bin/harness budget <suite>`
(prints a bare integer seconds; `--json` for the structured form).
Do **not** hardcode any timeout number in prompt text, shell commands,
or Python code — see the RED LINE block at the top of this file.

## Recovery when the remote drops

If the remote drops (idle timeout, host restart, network / proxy
outage) **do not silently fall back to local** — local fallback is
unsafe for bitwise gates (see "Always go remote") and is forbidden.

The recovery action depends on `[remote].kind`:

- **`ssh`** (this overlay, static host): the harness does **not**
  manage the host's lifecycle. You cannot re-provision it. Re-try the
  ssh call once in case the drop was a transient proxy timeout; if it
  still fails, **surface a blocker** naming the host and the failing
  command so the user can restart / re-allocate it.
- **`devspace`**: a dedicated lifecycle overlay
  (`remote-execution-devspace.md`) is injected with a `cctl
  devspace create` recovery procedure — follow it instead of surfacing a
  blocker.

## Devspace storage discipline

Do **not** write run artifacts — `.artifacts/runs/<suite>-*` payloads,
checkpoints, per-step loss / grad_norm dumps, ref-trace caches, agent
logs, captured profiles, anything large — into a devspace's ephemeral
/ scratch filesystem (typical mount points: `/tmp`, `/run`, `/dev/shm`,
container overlay layers, anything not on the persistent home volume).
Those volumes are size-capped at a few GB and **a single full
`long-train` or `op-long` run will fill them and crash the
devspace**, taking the agent loop down with it.

Pin every write under the user's persistent home volume — on the
current devspaces this is `/user/<username>`,
which is the only mount that (a) survives a devspace restart and
(b) has the multi-hundred-GB capacity required for ref + ours
trajectories + sliding-window kept runs. When in doubt about which
mount a path resolves to, `df -h <path>` on the remote and confirm
it lands on the home volume, not a tmpfs / overlayfs.

**Do not crash the devspace with reckless writes.** Hard rules:

* Never write to `/tmp`, `/run`, `/dev/shm`, `/`, `/root`, `/var`,
  or any container overlay path. Use only `/user/<username>/...`.
* Never `cat`, `tee`, redirect, or `dd` arbitrarily large blobs
  (>10 MB) anywhere on the remote — pipe through the harness's
  `.artifacts/runs/` layout or pull back to local with `scp`.
* Never duplicate a multi-GB checkpoint / ref-trajectory directory
  on the remote (use the existing one in `.artifacts/`; do not
  `cp -r` it to a side path "for safety").
* Before any large remote write, `df -h /user/<username>` and
  abort if free space < 50 GB; surface the blocker to the user
  rather than risk an out-of-space crash that kills the devspace
  and the agent loop with it.
* Forbidden: log spamming (`set -x` in a loop, `strace -f`,
  per-step ASCII grad dumps). If you genuinely need verbose
  instrumentation, gate it behind a `[debug] enabled = true`
  flag and write to a single bounded-size rotated file.
