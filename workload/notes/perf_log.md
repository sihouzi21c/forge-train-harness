# Performance Log

## Round 1 — alignment milestone: initial engine implementation

### Changes
- Implemented `config.py`, `forward.py`, `backward.py`, `parameters.py`, and `train_loop.py` for the in-house training engine (MiniCPM5 1B, torch backend, muP + MTP)
- Forward primitives: RoPE, RMSNorm, SwiGLU, GQA attention (math fallback + flash_attn), embedding, LM head, cross-entropy loss
- Backward primitives: linear, RMSNorm, SwiGLU, embedding, RoPE, QKV projection, GQA attention (math fallback), cross-entropy
- Static backward: full reverse computation graph across all 24 layers + MTP head, no autograd
- Hash capture: blake2b-128 per-tensor hash records for M1 forward-align / backward-align gates
- Weight initialization: muP scheme matching ref's `init_weights` (fp32 → bf16 conversion order)

### Test results
- Framework guard: PASS (0 violations)
- Anti-proxy: PASS (0 violations)
- Forward contract tests: 11/12 pass (1 expected failure: decoder_layer_forward shape mismatch due to 1B config HIDDEN_SIZE != NUM_HEADS*HEAD_DIM)
- Backward contract tests: 14/14 pass

### Status
- Alignment milestone: in progress — engine compiles, primitives verified locally
- Next step: push to remote GPU devspace, run `forward-align` / `backward-align` suites
