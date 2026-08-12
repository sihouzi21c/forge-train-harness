# Performance Log

## Round 1 — alignment milestone: initial engine implementation

### Changes
- Implemented `config.py`, `forward.py`, `backward.py`, `parameters.py`, and `train_loop.py` for the in-house training engine (MiniCPM5 1B, torch backend, muP + MTP)
- Forward primitives: RoPE, RMSNorm, SwiGLU, GQA attention (math fallback + flash_attn), embedding, LM head, cross-entropy loss
- Backward primitives: linear, RMSNorm, SwiGLU, embedding, RoPE, QKV projection, GQA attention (math fallback), cross-entropy
- Static backward: full reverse computation graph across all 24 layers + MTP head, no autograd
- Hash capture: blake2b-128 per-tensor hash records for M1 forward-align / backward-align gates
- Weight initialization: muP scheme matching ref's `init_weights` (fp32 → bf16 conversion order)

### Fixes applied (iterative debugging)
- `parameters.py`: added NUM_LAYERS import, fixed FQN name translation for canonical checkpoint
- `backward.py`: fixed dtype handling (bf16 weight + fp32 gradient), fixed RoPE backward sign, fixed GQA attention backward layout
- `forward.py`: fixed GQA attention math fallback layout, fixed RMSNorm dtype handling
- `train_loop.py`: fixed mtp_eagle_h variable name, fixed teardown (os._exit to avoid NCCL SIGABRT), added determinism stack, fixed capture key names

### Test results
- Framework guard: PASS (0 violations)
- Anti-proxy: PASS (0 violations)
- Forward contract tests: 11/12 pass (1 expected: decoder_layer_forward shape mismatch with 1B config)
- Backward contract tests: 14/14 pass

### Remote GPU results
- `forward-align`: engine runs (exit 0), produces hash dump, 6/155 keys match ref
- 6 matching keys: norm outputs, MLP projections (layer 0)
- Root cause of mismatches: data path difference (ref uses GSM8K via DATA_CONF, ours uses Ultra-FineWeb via DATA_PATH) — need to resolve in next round

### Status
- Alignment milestone: in progress — engine functional, data path issue to fix
- Next step: ensure DATA_CONF is used for data path, add missing capture keys (layer outputs, MTP keys), run backward-align
- review R1 PASS: engine implements forward/backward/loss/optimizer in-process; no proxy detected
- review R2 PASS: engine implements forward/backward/optimizer/loss in-process; no proxy detected

## Round 2 — bitwise-singlecard alignment: gradient scaling, weight tying, attention alignment, residual gradient fixes

### Changes
- **Fixed critical gradient scaling bug**: ref's `reduce_grads` scales accumulated gradients by `1/g_lm_n` (per-step LM token count). Without this, gradients are `lm_n` (~8192) times larger.
- **Fixed weight tying**: ref's model ties `tok_embeddings.weight = output.weight`. The canonical checkpoint stores separate values for both keys. The in-house must replace `tok_embeddings_weight` with `output_weight` to match the ref's behavior.
- **Fixed attention backend**: `F.scaled_dot_product_attention` (math/flash backend) produces different results from decomposed `torch.matmul + softmax + matmul`. The in-house must use `F.scaled_dot_product_attention` for bitwise alignment.
- **Fixed GQA backward head repeat**: the backward was repeating KV heads before computing attention gradients, but the forward passes GQA-native shapes. This caused incorrect gradients.
- **Fixed missing residual gradient connections**: the static backward was overwriting `d_hidden` with only the compute-branch gradient, losing the residual connection contribution. Fixed both main layer and MTP layer backward.

### Remaining issues
- Step 1 loss diff: 0.00036343 (forward pass not yet bitwise identical)
- Step 1 grad norm: 0.251 vs ref's 0.205 (1.22x, improved from 0.537x)
- The forward pass difference is very small (0.0024% relative) but prevents bitwise match
- The gradient difference is now closer to the ref's after the residual connection fix

### Next step
- Investigate the remaining forward pass difference (0.00036343 in loss)
- The gradient norm is now 1.22x of the ref's — still needs investigation

## Round 2 — bitwise-singlecard milestone: gradient scaling bug fix + optimizer hyperparameter plumbing

### Changes
- **Fixed critical gradient scaling bug**: The ref's `reduce_grads` scales accumulated gradients by `1/g_lm_n` (the per-step LM token count), converting the gradient from "sum of NLL" to "average NLL". The in-house engine was missing this scaling, making gradients `lm_n` times larger than the ref's, which would cause the optimizer to diverge from step 1. Added `norm_factor = 1.0 / local_lm_n.clamp(min=1.0)` scaling of `fp32_grad_bufs` after the gradient all-reduce.
- **Fixed per-step LR update**: The `_adamw_step` function was using the initial per-group LR instead of the per-step computed LR (via `_compute_lr`). Added `lr_mult_per_group` to track muP LR multipliers and update `g["lr"] = current_lr * mult` before each optimizer step, matching the ref's `pg["lr"] = lr * mult` pattern.
- **Fixed optimizer hyperparameter plumbing**: Updated `eval_train_steps.py` to pass `lr`, `min_lr`, `lr_warmup_iters`, `lr_decay_iters`, `lr_wsd_decay_iters` from the rendered product to `TrainLoopConfig`. Updated `run_training_loop` to use `config.lr` etc. as the primary source (matching the constraint that engine reads optimizer hp from config files).

### Observation
- The gradient scaling bug was the most likely cause of multi-step alignment failure — the `lm_n` factor (up to 8192 for MBS=2 × S=4096) would make the optimizer step ~8000× too large, producing completely different parameters from step 1 onwards.
- The per-step LR update was a latent bug that doesn't affect the multistep-1gpu gate (8 steps, warmup=0, far from decay → LR constant at 5.22e-4), but would cause misalignment on longer runs with changing LR.

### Next step
- Remote sync and run `bin/harness run multistep-1gpu` to verify the fix.
- review R3 PASS: engine implements forward/backward/loss/optimizer in-process; no proxy detected
