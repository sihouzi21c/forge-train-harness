# alignment — Single-step single-GPU forward/backward bitwise alignment

> **Prereq**: none. **Postreq**: alignment.forward PASS + alignment.backward PASS → proceed to bitwise-singlecard.
>
> **Architectural duality**: All Stage 1 gates converge into two forms — (1) **alignment**: ref-vs-ours subprocess + single-step tensor dump comparison; (2) **bitwise-singlecard onwards**: ref-vs-ours subprocess multi-step trajectory comparison.
>
> **Execution form**: dispatcher launches two completely independent subprocesses. The ref side bashes the agent-generated alignment capture bridge pointed to by `[defaults].ref_capture_script` — the bridge installs `evals.harness_hook.install` into the customer training stack (contract documented in `evals/harness_hook/recipes/README.md`); after a single-step run it dumps `{ref_dump_dir}/{ref_capture_basename}` (containing `rank<r>.mb0.fwd.<fqn>#<call>` / `rank<r>.mb0.bwd.<fqn>#<call>` activations and `rank<r>.grad.<fqn>.postallreduce` gradients, native framework FQN; single-GPU alignment is `r=0`, one microbatch `mb0`. The `#<call>` suffix is the per-forward call index — a shared module re-used within one forward, e.g. an `output` head on the main + MTP paths, records BOTH fires as `#0` / `#1`. Ours must emit the same `rank<r>.mb0.` prefixes and `#<call>` suffix, not a bare `<fqn>` — see recipes README § key axes). The ours side runs `evals/scripts/eval_capture_align.py`, which pushes the full path `{ours_capture_dir}/{candidate_capture_basename}` (computed by dispatcher) to `training_engine_tensor.train_loop.run_training_loop` via `HARNESS_CAPTURE_OUTPUT_FILE`; the engine takes the capture short-circuit path and writes to that path. dispatcher uses `evals._capture_diff.diff_capture_dicts` to take the intersection by key prefix (`fwd.` or `grad.`) and compare by `max_abs_diff == 0` (vocab pad auto-trim). When `[defaults].ref_capture_script` is empty, dispatcher fail-fasts and points the agent to the recipes README.

## Goal

The in-house executor can complete a full forward + backward of the target dense model — geometry (`num_layers` / `hidden_size` / `num_attention_heads` / `head_dim` / `padded_vocab_size` etc.) is defined in `@@FORGE_CONFIG_DIR@@/model.toml`, never assumed from this prompt; on the torch backend additionally with muP width+depth scaling and an Eagle MTP head per `mtp_num_layers` — with the final logits and all parameter grads **bitwise identical** to the ref's forward / backward results. This stage does not involve `optimizer.step`.

## Alignment basis

Real forward / backward execution result of ref training (obtained by running the L0 ref script).

## Path

1. **Read the ref source** (`ref/reference/`). Identify every custom
   `torch.autograd.Function` subclass and any extension autograd op
   (e.g. `FlashAttnFunc`, `_LinearFn`, `_EmbeddingFn`, `_RMSNormFn`)
   that the ref forward / backward dispatches. For each one, study
   the body — op order, cast timing, accumulation order — because
   these break bitwise even when the underlying aten op is identical.
2. Use the same initial weights and input data as the ref.

## Acceptance criteria

The final logits of the full forward are **bitwise identical (`max_abs_diff == 0`)** to the ref forward output; all parameter grads are **bitwise identical (`max_abs_diff == 0`)** to the ref backward output.

## Gate

**bitwise match (`max_abs_diff == 0`), no fallback tolerance, no 1e-7 or any fp32/bf16 precision error is allowed.**

harness suites (**both must pass**):

- `bin/harness run forward-align` — forward bitwise (alignment.forward)
- `bin/harness run backward-align` — backward bitwise (alignment.backward)

## Constraints

1. Producing only a JSON plan or scheduling sketch is not allowed; there must be real kernel launches.
2. The same computational path and precision strategy as the ref must be used to guarantee bitwise match (precision specification: see `constraint.md` §FP32 precision specification).
3. The backward order must be statically determined at compile time and not depend on autograd.
4. On gate failure (esp. backward-align), follow constraint.md §"Bitwise alignment debugging discipline" before editing.

## Numerical op-order discipline

Floating-point arithmetic is non-associative — when re-implementing any
numerically critical path (manual backward, reductions, normalization, loss),
match ref's exact op order and rounding sequence, not just its mathematical
result. Algebraically-equivalent formulas with different op orders, cast
timings, or accumulation orders can produce bit-different output and break
the bitwise gate.

If a hand-written backward misses the bitwise gate (esp. for C++-decomposed
aten ops like `rms_norm`), capture the real per-op sequence with a
`__torch_dispatch__` mode wrapping one forward+backward call, then replay
it verbatim — algebraic closed-form formulas frequently diverge byte-wise
at bf16/fp32 mix even when mathematically equivalent.
