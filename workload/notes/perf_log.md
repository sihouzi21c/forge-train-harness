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
