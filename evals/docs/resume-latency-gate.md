# Resume-Latency Gate — design intent

A second, efficiency-only gate for resume (续训), complementing the
existing `resume-gate-20` correctness gate. This note records *what the
gate means and why it exists*, not how it is wired up.

## The problem it guards against

The resume milestone asks the agent to implement checkpoint save + resume. The existing
`resume-gate-20` only checks **numerical correctness** — that the
post-resume `loss` / `grad_norm` is bitwise-identical to uninterrupted
training.

A correct-but-dumb implementation can pass that gate: to restore the
dataloader position, the agent re-instantiates a fresh (stateless) HF
streaming dataloader and **replays the prefix on CPU** — re-streaming and
re-tokenizing every already-consumed batch from the start of the stream,
throwing the results away just to advance the cursor. Because the data
stream is deterministic, the replayed batches are bitwise-identical, so
the correctness gate is satisfied. But the GPU sits **idle** for the whole
replay. In production this is unacceptable: replay cost grows linearly
with how far into training you resume (resuming at step 1M would idle the
GPUs for a very long time while the CPU re-reads the prefix).

Correctness-based gating can never catch this — the replay *is* correct.
The gate must measure a quantity orthogonal to numerics: the **cost of the
fast-forward**.

## What the gate measures

The real production scenario is two separate processes: train for a while
and checkpoint, the job dies, a fresh process resumes from the
checkpoint. The gate reproduces exactly this — **two real, sequential
torch runs** (100 steps each, same shape as the long-train gate, no hash
capture):

- **Run 1**: train from init for 100 steps, save the checkpoint, exit.
- **Run 2**: resume from that checkpoint, continue for 100 more steps.

The essential signal is **how long Run 2 takes to produce its first real
training step**. A training step cannot run until the dataloader is at the
correct position, so the time-to-first-step strictly contains the
fast-forward cost. Run 1 serves as a built-in control: it cold-starts the
same engine the same way, so subtracting its own time-to-first-step
cancels the irreducible startup baseline (CUDA/NCCL init, model build,
first-step compute) and leaves essentially just the resume overhead.

    Δ = (Run 2 time-to-first-step) − (Run 1 time-to-first-step)

For a correct O(1) resume, Δ is small — only the extra checkpoint state
read (optimizer moments + fp32 master) plus noise. For the CPU-replay
implementation, Δ blows up with the resume depth. The gate passes when:

    Δ ≤ budget

Currently `budget` is a fixed **20 s**. (It could later become a formula
that scales with model size, parallelism, rank count, and I/O bandwidth —
the legitimate overhead is dominated by extra checkpoint bytes / read
bandwidth — but a fixed value is enough while the resume reference stays at
this scale.)

## Why this is more fundamental than GPU-utilization sampling

GPU utilization (e.g. via nvidia-smi) is a sampled, indirect proxy for the
same thing, and forces an arbitrary definition of "back to normal." The
event we actually care about — *training has resumed* — is observable
directly and deterministically as the first training step appearing. The
latency to that event is the production-relevant quantity; GPU idle is
just its shadow.

## Relationship to `resume-gate-20`

The two gates are complementary and both required for resume:

- `resume-gate-20` — proves the resume is **numerically correct** (bitwise
  lossless round-trip).
- resume-latency gate — proves the resume is **efficient** (no GPU-idle
  CPU replay of the data prefix).

An implementation that games one fails the other: a fast-but-wrong resume
fails the bitwise gate; a correct-but-replaying resume fails the latency
gate.
