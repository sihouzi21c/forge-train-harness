# Job Overlay (injected when `[remote].kind = "job"`)

> This overlay is appended **after** `remote-execution.md` (and the
> devspace lifecycle overlay) when the loop runs in **`kind = "job"`**
> (per-suite ephemeral cctl pytorchjobs off a 0-GPU gateway devspace). It
> does not replace the transport procedure — sync, in-place log
> inspection, and pull-back are unchanged. It only changes **where GPU
> suites execute**.

## What is different in this mode

The per-loop devspace `@@REMOTE_SSH_HOST@@` was claimed with **0 GPUs**.
It is a **filesystem gateway only**: the rsync destination, the host you
ssh into to read artifacts in place, and the persistent home volume that
holds the run outputs. **It has no GPU — you cannot run a GPU suite on
it.**

Every GPU suite instead runs as a **short-lived `cctl pytorchjob`** of
`[remote].nodes` pods (default 1), each pod requesting
`[remote].gpu_count` GPUs — so the DP world is `nodes × gpu_count`. It
executes on the **same shared user filesystem** where `bin/harness sync
push` already landed the code, and exits the moment the suite finishes —
returning the GPUs to the cluster. This is automated for you by
`tools/remote_run.sh`; you do not craft `cctl` commands by hand.

At `nodes = 1` this is a single pod, identical to the old single-node
behaviour. At `nodes > 1` the PyTorchJob operator injects the standard
rendezvous env (`RANK` = node rank, `WORLD_SIZE` = node count,
`MASTER_ADDR` / `MASTER_PORT`, `GPUS_PER_NODE`) into every pod; the suite
launcher reads it straight from the environment (`train_ours_al.sh`'s
`torchrun --nnodes/--node_rank`, the RL ray entry) — you never set any of
it by hand.

## RED LINE — GPU suites go through `tools/remote_run.sh`, never a bare ssh-run

> - **MANDATORY**: Launch every GPU suite (`forward-align`,
>   `backward-align`, `multistep-1gpu`, `multistep`, `perf-bitwise`,
>   `resume-gate-20`, `long-train`, `loss-gate-200`, `op-inventory`,
>   `op-long`, …) with:
>   ```bash
>   tools/remote_run.sh <suite> [extra args...]
>   ```
>   In kind=job this submits the cctl pytorchjob **once** and polls it in the
>   foreground, dumps its logs, and propagates its exit code. If the
>   foreground poll is cut short by the command timeout, continue it with
>   `tools/remote_run.sh --poll <suite>` (see the next RED LINE). The SSOT
>   wall-clock budget is the job's **execution** window, counted from the
>   moment it reaches `running` — the preceding Queued time (waiting for
>   the cluster to schedule GPUs) is **unbounded** and never counts against
>   the budget, so a busy cluster never kills a job that would run in time.
>   `tools/remote_run.sh` logs how long each job queued before running.
> - **FORBIDDEN**: `ssh @@REMOTE_SSH_HOST@@ '... bin/harness run <GPU-suite>'`.
>   The gateway devspace has **no GPU**; a bare ssh-run there fails (or
>   silently runs CPU-only) and wastes a round.
> - **MANDATORY (still local / CPU)**: `guard`, `anti-proxy`, `unit`,
>   `pytest`, `ruff`, `git`, code editing, and commits stay **local**,
>   exactly as in the base overlay.
> - **VIOLATION**: ssh-running a GPU suite on the 0-GPU gateway = discard
>   the round and re-run via `tools/remote_run.sh`.

## RED LINE — a foreground poll that times out is NOT a gate failure; re-poll it

A GPU job runs on the cluster, not in your process. `tools/remote_run.sh
<suite>` submits the job **once** and then polls it in the **foreground**,
blocking this turn until the job reaches a terminal phase. A long suite
(`resume-gate-20`, `long-train`, …) or a long GPU queue can outlast the
~10-minute foreground command timeout, so the poll command may be **killed
with "Command timed out"**. **This is normal — it does NOT mean the gate
failed.** The cluster job keeps running; only your local poll was cut short.

> - **MANDATORY**: When a foreground `tools/remote_run.sh <suite>` is killed
>   by the command timeout while the job is still queued/running, **continue
>   monitoring the SAME job** with:
>   ```bash
>   tools/remote_run.sh --poll <suite>
>   ```
>   `--poll` re-connects to the job you already submitted and blocks for
>   another window. Repeat it until you get an actual PASS/FAIL verdict.
>   **Wait it out in this round — do not conclude the round with a job still
>   in flight.** Run the poll in the FOREGROUND and re-poll.
> - **FORBIDDEN**: yielding the round to wait for an async notification.
>   **There is NO external wake in a one-shot round** — nothing re-invokes
>   you when the job finishes. `run_in_background` (blocked by a PreToolUse
>   hook), `Monitor`, `ScheduleWakeup`, and `CronCreate` are all disabled at
>   the CLI layer for exactly this reason: they promise a "notify / re-invoke
>   me on completion" event that will never fire in this headless `-p` round,
>   so trusting one silently orphans your in-flight job and lets review run
>   before the verdict is in. Do NOT reach for them and
>   do NOT end your turn on "the monitor will notify me" / "I scheduled a
>   fallback; it will re-invoke me" — the ONLY way to learn a job's verdict
>   is to `--poll` it to terminal within this same turn.
> - **FORBIDDEN**: treating a foreground-poll timeout as a gate failure — do
>   NOT change code, do NOT resubmit, do NOT give up. Only a real gate
>   verdict, or a `Failed`/`Killed` cctl phase, is a failure.
> - **FORBIDDEN**: re-running the **submit** form `tools/remote_run.sh
>   <suite>` while a job for this suite+commit is already in flight — it will
>   refuse (one job per suite+commit). Use `--poll` to monitor. To verify
>   *new* code, commit it first: a new commit starts a fresh job and stops
>   the stale one.
> - **VIOLATION**: concluding the round with a job still in flight, or
>   abandoning a queued job because a foreground poll timed out.

## Sync, logs, artifacts — unchanged

- **Sync** (Step 1): `bin/harness sync push` still rsyncs the workspace +
  active config to `@@REMOTE_WORKDIR@@` on the gateway devspace. The GPU
  job reads the code from that same shared volume, so **you must
  `sync push` before every `tools/remote_run.sh`**, exactly as before.
- **Live progress**: the suite writes its logs to
  `@@REMOTE_WORKDIR@@/.artifacts/runs/<run>/*.log` on the shared volume.
  Tail them in place via the gateway:
  ```bash
  ssh @@REMOTE_SSH_HOST@@ 'tail -200 @@REMOTE_WORKDIR@@/.artifacts/runs/<run>/<file>.log'
  ```
  (The cctl pytorchjob's pod stdout — all pods, master + any workers — is
  also dumped by `tools/remote_run.sh` when the job ends; the shared-volume
  log is the authoritative record.)
- **MFU**: the job runs `tools/mfu_record.py` on the GPU node and writes
  `mfu_history.jsonl` to the shared volume; the local poller backfills the
  live MFU badge from it. No action needed.

## `bin/harness doctor` probes the GPU stack

`bin/harness doctor` inspects the CUDA / NCCL stack. The **gateway
devspace has no GPU**, so do **not** ssh-run `doctor` on it — route it
through `tools/remote_run.sh doctor` so it lands on a GPU job, or read the
image's baked CUDA stack from a regular GPU suite's run log.

## Recovery

The gateway devspace can still drop (host restart, network outage),
though as a 0-GPU box it is **not** subject to GPU idle-deadline reclaim.
Follow the devspace lifecycle overlay's `cctl devspace create` +
`lease rebind` recovery for the gateway. Separately, if a `cctl pytorchjob`
fails to schedule (no GPU quota in the pool), that is a hard blocker —
surface it with the pool name; do **not** silently fall back to local.
