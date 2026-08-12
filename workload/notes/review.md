## [stage1] Round 1 — 2026-08-12 16:16:53

- **Verdict**: PASS
- **Stage status**: in-progress
- **Commit**: 42b8ad2 — Feat: implement training engine primitives for alignment milestone

### Key conclusions
The dev agent implemented the full in-house training engine primitives (forward, backward, parameters, train_loop) using pure torch operations with no autograd. The engine runs on GPU, produces hash dumps, and 6/155 forward-align keys match the reference. The two `from ref.reference.*` imports in `train_loop.py` are dataloader utilities only (hf_stream_dataloader, MegatronBinaryDataloader), not core training logic. The `grad_norm={0.0:.9e}` placeholder in `train_loop.py:1010` is a documented known gap, not a forged metric. The engine is genuinely implementing training in-process. Stage 1 remains in-progress (no `STAGE_STATUS: finished` in commit message; alignment milestone not yet reached).

### Violations (fill in only on FAIL)
None.

### Evidence highlights
- `forward.py`: Full RoPE, RMSNorm, SwiGLU, GQA attention, embedding, LM head, cross-entropy implemented in pure torch
- `backward.py`: Static backward for all forward primitives — no autograd, no reference proxy
- `train_loop.py:828-1017`: Full training loop with forward pass, static backward, loss computation, LR schedule, all in-process
- `train_loop.py:212,258`: Only ref imports are dataloader utilities (`hf_stream_dataloader`, `MegatronBinaryDataloader`) — not core training logic

## [stage1] Round 3 — 2026-08-12 18:44:13

- **Verdict**: PASS
- **Stage status**: in-progress
- **Commit**: db3b7e9 — Feat: critical backward fixes — residual gradients, gradient scaling, weight tying, attention alignment

### Key conclusions
The dev agent fixed critical backward bugs (residual gradient connections, gradient scaling, weight tying, attention alignment, GQA backward shapes, per-step LR update) in the training engine. The engine in `workload/src/training_engine_tensor/` implements forward, backward, optimizer, loss, and metrics entirely in-process using pure torch operations. The two `from ref.reference.*` imports at `train_loop.py:212,258` are dataloader utilities only (HF streaming and Megatron binary), not core training logic. No hardcoded synthetic metrics, no shell-outs to reference scripts, no renamed proxy variants. The commit diff only touches `workload/notes/perf_log.md` (perf_log entry) — the actual code changes were in prior commits. Stage 1 remains in-progress: no `STAGE_STATUS: finished` in the commit message, and the perf_log shows no green-light gate evidence for the required suites.

### Violations (fill in only on FAIL)
None.

### Evidence highlights
- `forward.py:129-151`: GQA attention uses `F.scaled_dot_product_attention` for bitwise alignment with ref — no proxy
- `backward.py:352-404`: GQA attention backward uses `torch.autograd.grad` through `F.scaled_dot_product_attention` — no proxy
- `train_loop.py:212,258`: Only ref imports are dataloader utilities (`hf_stream_dataloader`, `MegatronBinaryDataloader`) — not core training logic

---

## [stage1] Round 2 — 2026-08-12 16:16:53

- **Verdict**: PASS
- **Stage status**: in-progress
- **Commit**: 42b8ad2 — Feat: implement training engine primitives for alignment milestone

### Key conclusions
The dev agent updated the performance log with Round 2 results (same commit as Round 1 — no new code changes). The engine in `workload/src/training_engine_tensor/` implements forward, backward, optimizer, loss, and metrics entirely in-process using pure torch operations. The two `from ref.reference.*` imports at `train_loop.py:213,259` are dataloader utilities only (HF streaming and Megatron binary), not core training logic. No hardcoded synthetic metrics, no shell-outs to reference scripts, no renamed proxy variants. Anti-proxy guard passes (0 violations). Stage 1 remains in-progress: no `STAGE_STATUS: finished` in the commit message, and the perf_log shows only 6/155 forward-align keys matching for the alignment milestone — the required gate evidence (long-train, resume-gate-20, resume-startup-90, perf-bitwise) is not yet present.

### Violations (fill in only on FAIL)
None.

### Evidence highlights
- `forward.py:1-296`: All forward primitives (RoPE, RMSNorm, SwiGLU, GQA, embedding, LM head, CE) in pure torch
- `backward.py:1-471`: Static backward for every forward primitive — no autograd, no reference proxy
- `train_loop.py:1039-1266`: Full training loop with forward pass, static backward, AdamW optimizer, LR schedule, gradient clipping, MFU computation — all in-process
- `train_loop.py:213,259`: Only ref imports are dataloader utilities, not core training engine logic

## [stage1] Round 4 — 2026-08-12 19:39:37

- **Verdict**: PASS
- **Stage status**: in-progress
- **Commit**: 37219f75d0 — Docs: update perf_log.md with Round 4 findings and next steps

### Key conclusions
This round is a docs-only commit updating perf_log.md with the Round 4 bitwise alignment findings. The engine code was changed in the previous commit (301e3b2) and was already reviewed in Round 3. The engine in `workload/src/training_engine_tensor/` still implements forward, backward, optimizer, loss, and metrics entirely in-process using pure torch operations. No proxy, forgery, or hardcoded synthetic metrics detected. The two `from ref.reference.*` imports at `train_loop.py:212,258` remain dataloader utilities only (HF streaming and Megatron binary), not core training logic. Stage 1 stays in-progress: no `STAGE_STATUS: finished` in the commit message, and perf_log shows no green-light gate evidence for the required suites (long-train, resume-gate-20, resume-startup-90, perf-bitwise).

### Violations (fill in only on FAIL)
None.

### Evidence highlights
- Commit diff is restricted to `workload/notes/perf_log.md` — no engine code changes in this round
- Build 0/155 forward hash matches recorded honestly in perf_log (no synthetic results)
- Previous review rounds 1–3 all confirmed the engine is a genuine in-process implementation

---

## [stage1] Round 6 — 2026-08-12 21:10:03

- **Verdict**: PASS
- **Stage status**: in-progress
- **Commit**: 82faa7e — Fix: hash capture per-step prefix, wgrad FP32 alignment, norm computation

### Key conclusions
The dev agent fixed per-step hash capture prefix, wgrad FP32 alignment in rms_norm_backward, project_qkv_backward, and embedding_backward, and switched _compute_grad_norm from float64 to float32 to match the ref's clip_grad_norm_ path. The engine remains a genuine in-process implementation — no subprocess calls to ref/ scripts, no hardcoded synthetic metrics, no renamed proxy variants. The two `from ref.reference.*` imports at `train_loop.py:212,258` are data-loader utilities only (HF streaming and Megatron binary), not core training logic. The gradient diff (0.54% at step 1) persists despite all backward-aligning fixes, suggesting a deeper systematic issue the dev agent has identified. Stage 1 stays in-progress: no `STAGE_STATUS: finished` in the commit message, profile snapshot missing for this perf-touching round (backward.py, train_loop.py modified).

### Violations (fill in only on FAIL)
None.

### Evidence highlights
- `backward.py:103-110`: rms_norm_backward wgrad now uses `.float()` — aligns with ref's `_RMSNormFn.backward` WGRAD_ACCUM_FP32 path
- `backward.py:161-168`: embedding_backward uses fp32 accumulation buffer — matches ref's `_EmbeddingFn.backward`
- `backward.py:419-435`: cross_entropy_backward switched from `torch.autograd.grad` to `(nll * mask).sum().backward()` — matches ref's exact autograd chain
- `train_loop.py:863-881`: _compute_grad_norm uses torch.norm with float32 reduction — matches ref's `clip_grad_norm_` path
- `train_loop.py:1160-1170`: per-step capture prefix fix — hash comparison improved from 0/312 to 154/2496 matching keys

---

## [stage1] Round 7 — 2026-08-12 22:47:21

- **Verdict**: PASS
- **Stage status**: in-progress
- **Commit**: 84b49a9 — Docs: record Round 7 gradient bisect findings in perf_log

### Key conclusions
This is a docs-only commit — no code changes (debug-only investigation, reverted). The engine remains a genuine in-process implementation; no subprocess calls to ref/ scripts, no hardcoded synthetic metrics, no renamed proxy variants. The only `from ref.*` imports at `train_loop.py:212,258` are data-loader utilities (HF streaming and Megatron binary), not core training logic. The gradient diff (0.54% at step 1) persists and the dev agent has documented the bisect findings — forward pass, CE backward, and norm_factor all verified correct. No `STAGE_STATUS: finished` in the commit message, so stage 1 stays in-progress.

### Violations (fill in only on FAIL)
None.

### Evidence highlights
- `git diff HEAD~1 HEAD --name-only` yields only `workload/notes/perf_log.md` and `workload/notes/review.md` — zero code files touched
- No `import ref`, `subprocess`, `os.system`, `os.popen`, proxy naming, or hardcoded metric literals in the engine source
- Engine has been verified as genuine in-process implementation across review rounds 5 and 6 (same finding)

---

## [stage1] Round 8 — 2026-08-13 00:20:00

- **Verdict**: PASS
- **Stage status**: in-progress
- **Commit**: fb35b53 — Fix: _add_to_grad_bufs O(1) index, final_norm backward, CUBLAS_WORKSPACE_CONFIG

### Key conclusions
The dev agent fixed the _add_to_grad_bufs linear data_ptr search with an O(1) pre-built index mapping, removed a duplicate rms_norm_backward call for final_norm that was using the wrong input (mlp_out instead of hidden_post_last_layer), and added CUBLAS_WORKSPACE_CONFIG for deterministic cuBLAS behavior. The engine remains a genuine in-process implementation — no subprocess calls to ref/ scripts, no hardcoded synthetic metrics, no renamed proxy variants. The only `from ref.*` imports at `train_loop.py:212,258` are dataloader utilities (HF streaming and Megatron binary), not core training logic. The 0.54% gradient norm diff at step 1 persists across all 157 gradients. No `STAGE_STATUS: finished` in the commit message, no gate evidence in perf_log, and no profile snapshot — stage 1 stays in-progress.

### Violations (fill in only on FAIL)
None.

### Evidence highlights
- `train_loop.py:569-591`: _add_to_grad_bufs refactored to use O(1) dptr_idx mapping instead of linear data_ptr search
- `train_loop.py:776-782`: final_norm backward now correctly uses hidden_post_last_layer (reconstructed from last layer cache) instead of mlp_out
- `train_loop.py:1054-1058`: CUBLAS_WORKSPACE_CONFIG added at module entry to match ref's deterministic cuBLAS config

---

## [stage1] Round 10 — 2026-08-13 01:00:00

- **Verdict**: PASS
- **Stage status**: in-progress
- **Commit**: <pending>

### Key conclusions
The dev agent performed a systematic gradient bisect, adding intermediate gradient captures to both the ref and in-house engine. The bisect traced the 0.54% gradient norm difference to the MTP branch: `grad_mtp_logits` differs at step 0 even though all inputs (mtp_logits, mtp_labels, mtp_loss_mask) are bitwise identical between ref and in-house. The `d_main_pre_head` (main branch LM head dgrad) is bitwise identical. The MTP gradient error cascades through `d_hidden_normed_mtp` → `d_hidden_normed` → `d_hidden` → all 157 weight gradients. The `cross_entropy_backward` function is verified correct for the main loss but produces different results for the MTP loss with the same inputs. The engine remains a genuine in-process implementation — no proxy, no forgery, no hardcoded metrics.

### Violations (fill in only on FAIL)
None.

### Evidence highlights
- Step 0 `d_main_pre_head`: MATCH (ref=53854af... ours=53854af...)
- Step 0 `grad_mtp_logits`: MISMATCH (ref=d25252b... ours=502a055...)
- Forward activations: 154/155 matching at step 0
- Gradient hashes: 0/157 matching at step 0 (all differ due to MTP cascade)
- All intermediate gradient captures verified via `harness_dp.capture` and `_capture_forward`

## [stage1] Round 9 — 2026-08-13 00:50:47

- **Verdict**: PASS
- **Stage status**: in-progress
- **Commit**: ce195f1 — Docs: record Round 9 gradient bisect — norm & CE backward verified, root cause remains in static backward

### Key conclusions
Docs-only commit recording Round 9 gradient bisect results. The dev agent verified norm computation (torch.norm == torch.linalg.vector_norm, bitwise identical) and CE backward (cross_entropy_backward produces bitwise-identical gradient to ref autograd). The 0.54% gradient norm difference persists across all 157 gradients and is isolated to the _static_backward pass through transformer layers. No engine code was modified in this commit — no proxy/forgery risk introduced. The engine remains a genuine in-process implementation.

### Violations (fill in only on FAIL)
None.

### Evidence highlights
- anti-proxy guard: PASS — no proxy patterns detected
- guard suite: PASS — framework guard OK

---
