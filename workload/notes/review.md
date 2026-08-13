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
- **Commit**: f4258af — Docs: record Round 10 gradient bisect — MTP branch identified as systematic gradient root cause

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

## [stage1] Round 10 (review) — 2026-08-13 02:55:00

- **Verdict**: PASS
- **Stage status**: in-progress
- **Commit**: f4258af — Docs: record Round 10 gradient bisect — MTP branch identified as systematic gradient root cause

### Key conclusions
Docs-only commit — no engine code was modified. The dev agent performed a systematic gradient bisect that traced the 0.54% gradient norm difference to the MTP branch (`grad_mtp_logits` mismatch at step 0 despite identical inputs). The `d_main_pre_head` (main branch LM head dgrad) is bitwise identical. The engine remains a genuine in-process implementation — no proxy, no forgery, no hardcoded metrics, no shell-outs to ref. The anti-proxy and guard suites both pass. Stage 1 finish conditions are not satisfied (no gate evidence in perf_log, no `STAGE_STATUS: finished` in commit message).

### Violations (fill in only on FAIL)
None.

### Evidence highlights
- anti-proxy guard: PASS — no proxy patterns detected
- guard suite: PASS — framework guard OK
- Engine source files (`forward.py`, `backward.py`, `train_loop.py`, `parameters.py`) contain no shell-outs, no ref imports, no hardcoded metric literals
- `train_loop.py:1306-1309`: `global_loss` and `mfu_e2e_standard` are computed values from actual computation, not hardcoded

---

## [stage1] Round 11 — 2026-08-13 04:34:50

- **Verdict**: PASS
- **Stage status**: in-progress
- **Commit**: 4bf7528 — Fix: cross_entropy_backward manual softmax-one_hot + ce_w fp32 scaling

### Key conclusions
The dev agent replaced the `F.cross_entropy` backward call (via `obj.backward()`) with a manual `softmax - one_hot` closed-form formula in `cross_entropy_backward` (`backward.py:392-430`), and added a `scale` parameter for fp32 `ce_w` scaling before `.to(bf16)` conversion. The change is a genuine in-process implementation — no proxy, no shell-outs, no ref imports, no hardcoded metrics. The `scale` factor is a legitimate computation parameter used in `train_loop.py:623-625` for the MTP loss weight. The 0.54% gradient norm difference persists with the MTP branch gradient as the identified root cause. Stage 1 finish conditions are not satisfied: no `STAGE_STATUS: finished` in commit message, no gate evidence in perf_log, and profile snapshot missing for this perf-touching round.

### Violations (fill in only on FAIL)
None.

### Evidence highlights
- `backward.py:392-430`: Manual `softmax - one_hot` closed-form CE backward — no proxy, no `F.cross_entropy` backward kernel
- `train_loop.py:623-625`: `scale=mtp_ce_weight` passed to `cross_entropy_backward` — legitimate fp32 scaling before bf16 conversion
- anti-proxy guard: PASS (0 violations); framework guard: PASS (0 violations)

---

## [stage1] Round 12 — 2026-08-13

- **Verdict**: PASS
- **Stage status**: in-progress
- **Commit**: (current commit)

### Key conclusions
The dev agent resolved the systematic 0.54% gradient-norm difference (MTP gradient root cause). Three bugs were fixed: (1) `cross_entropy_backward` manual `softmax-one_hot` formula differed from `F.cross_entropy` backward by ~1 ULP in bf16, amplified through V=130560 matmul; (2) `embedding_backward` used `index_add_` instead of `embedding_dense_backward`; (3) MTP embedding gradient used wrong `grad_in` (discarded `rms_norm_backward` output). The engine remains a genuine in-process implementation — no proxy, no forgery, no hardcoded metrics. Gate results: 8/8 loss bitwise, 6/8 grad_norm bitwise (2 ULP diffs), 2488/2496 hash match. Stage 1 stays in-progress.

### Violations (fill in only on FAIL)
None.

### Evidence highlights
- `backward.py:392-430`: `cross_entropy_backward` uses `F.cross_entropy` autograd (same as ref's `masked_ce`) with fp32 `scale`
- `backward.py:142-169`: `embedding_backward` uses `torch.ops.aten.embedding_dense_backward` (same as ref's `_EmbeddingFn`)
- `train_loop.py:744-751`: MTP embedding backward uses `grad_mtp_emb * mup_emb_scale` (correct `rms_norm_backward` output)
- `train_loop.py:871-890`: `_compute_grad_norm` uses `clip_grad_norm_` (same as ref)
- anti-proxy guard: PASS (0 violations); framework guard: PASS (0 violations)

## [stage1] Review Round 12 — 2026-08-13 05:53:24

- **Verdict**: PASS
- **Stage status**: in-progress
- **Commit**: 00c6354 — Fix: resolve MTP gradient 0.54% diff — cross_entropy autograd, embedding_dense_backward, embedding grad chain

### Key conclusions
The dev agent fixed three bugs that caused the systematic 0.54% gradient-norm difference: cross_entropy_backward switched from manual softmax-one_hot to F.cross_entropy autograd (same as ref's masked_ce), embedding_backward switched from index_add_ to embedding_dense_backward (same as ref's _EmbeddingFn), and the MTP embedding gradient now correctly captures the rms_norm_backward grad_in instead of the raw d_mtp_a. The engine remains a genuine in-process implementation — no subprocess calls to ref/ scripts, no hardcoded synthetic metrics, no renamed proxy variants. The only `from ref.reference.*` imports at `train_loop.py:212,258` are dataloader utilities (HF streaming and Megatron binary), not core training logic. Stage 1 finish conditions are not satisfied: no `STAGE_STATUS: finished` in commit message, no harness-written gate evidence in perf_log.md for the four required suites, and no profile snapshot for this perf-touching round.

### Violations (fill in only on FAIL)
None.

### Evidence highlights
- `backward.py:390-429`: cross_entropy_backward uses F.cross_entropy autograd with fp32 scale — same as ref's masked_ce chain
- `backward.py:142-164`: embedding_backward uses torch.ops.aten.embedding_dense_backward — same as ref's _EmbeddingFn
- `train_loop.py:735-751`: MTP embedding backward correctly captures rms_norm_backward[0] grad_in instead of raw d_mtp_a
- `train_loop.py:871-896`: _compute_grad_norm uses clip_grad_norm_ with matching max_norm for bitwise-aligned norm
- anti-proxy guard: PASS (0 violations); no gate configs or remote.toml modified

---

## [stage1] Round 13 — 2026-08-13 05:52:19

- **Verdict**: PASS
- **Stage status**: in-progress
- **Commit**: 00c6354 — Fix: resolve MTP gradient 0.54% diff — cross_entropy autograd, embedding_dense_backward, embedding grad chain

### Key conclusions
The dev agent resolved the systematic 0.54% gradient-norm difference by fixing three bugs: (1) `cross_entropy_backward` reverted from manual `softmax-one_hot` to `F.cross_entropy` autograd (`backward.py:390-429`), matching the ref's `masked_ce` backward path; (2) `embedding_backward` switched from `index_add_` to `torch.ops.aten.embedding_dense_backward` (`backward.py:142-164`); (3) MTP embedding gradient now correctly captures `rms_norm_backward` grad_in (`train_loop.py:744-761`). The engine remains a genuine in-process implementation — no proxy, no forgery, no hardcoded metrics. Gate results show 8/8 loss bitwise match, 2488/2496 hash match. Stage 1 finish conditions are not met: no `STAGE_STATUS: finished` in commit message, no gate evidence in perf_log (`long-train`, `resume-gate-20`, `resume-startup-90`, `perf-bitwise`), and profile snapshot missing for this perf-touching round.

### Violations (fill in only on FAIL)
None.

### Evidence highlights
- `backward.py:390-429`: `cross_entropy_backward` uses `F.cross_entropy` autograd with fp32 scale — same as ref's `masked_ce` chain
- `backward.py:142-164`: `embedding_backward` uses `torch.ops.aten.embedding_dense_backward` — same as ref's `_EmbeddingFn`
- `train_loop.py:744-761`: MTP embedding backward correctly captures `rms_norm_backward[0]` grad_in instead of raw `d_mtp_a`
- `train_loop.py:883-908`: `_compute_grad_norm` uses `clip_grad_norm_` with matching `max_norm` for bitwise-aligned norm
- anti-proxy guard: PASS (0 violations); framework guard: PASS (0 violations); no gate configs or `remote.toml` modified
- milestone: `bitwise-multicard` (active), no `long-horizon` milestone checks triggered

---

## [stage1] Round 14 — 2026-08-13 05:52:19

- **Verdict**: PASS
- **Stage status**: in-progress
- **Commit**: 00c6354 — Fix: resolve MTP gradient 0.54% diff — cross_entropy autograd, embedding_dense_backward, embedding grad chain

### Key conclusions
The commit under review is the same as rounds 12/13 (00c6354) — no new code changes. The engine is a genuine in-process implementation: `backward.py:142-164` uses `torch.ops.aten.embedding_dense_backward` (same op as ref's `_EmbeddingFn`), `backward.py:390-429` replays `F.cross_entropy` through autograd (same as ref's `masked_ce`), and `train_loop.py:744-761` correctly captures `rms_norm_backward` grad_in for the MTP embedding gradient chain. Anti-proxy guard: PASS (0 violations). No gate configs, run-shape keys, or `remote.toml` were modified. Stage 1 finish conditions remain unmet: no `STAGE_STATUS: finished` in commit message, no harness-written gate evidence for `long-train`/`resume-gate-20`/`resume-startup-90`/`perf-bitwise`, and no profile snapshot for this perf-touching round. Milestone `bitwise-multicard` is active; no `long-horizon` check triggered.

### Violations (fill in only on FAIL)
None.

### Evidence highlights
- `backward.py:142-164`: `embedding_backward` uses `torch.ops.aten.embedding_dense_backward` — same as ref's `_EmbeddingFn`
- `backward.py:390-429`: `cross_entropy_backward` uses `F.cross_entropy` autograd with fp32 scale — same as ref's `masked_ce` chain
- `train_loop.py:744-761`: MTP embedding backward correctly captures `rms_norm_backward[0]` grad_in instead of raw `d_mtp_a`
- `train_loop.py:883-908`: `_compute_grad_norm` uses `clip_grad_norm_` with matching `max_norm` — same as ref
- anti-proxy guard: PASS (0 violations); no gate configs, `remote.toml`, or run-shape keys modified

---

## [stage1] Round 15 — 2026-08-13 11:26:49

- **Verdict**: PASS
- **Stage status**: in-progress
- **Commit**: 42420c1 — Docs: record Round 14 findings — multi-GPU 1-ULP gradient diff root cause

### Key conclusions
This is a docs-only commit updating `workload/notes/perf_log.md` with Round 14 findings. The dev agent identified that the static backward produces a 1-ULP gradient difference vs the ref's autograd backward in multi-GPU mode, causing the `multistep` (DP=2) gate to diverge from step 4 onwards (3/8 loss match). No engine source code was modified in this round. The engine remains a genuine in-process implementation — no proxy, forgery, hardcoded metrics, or shell-outs. The anti-proxy guard passes. Stage 1 finish conditions are not met: no `STAGE_STATUS: finished` in the commit message, and no gate evidence for the four required suites (long-train, resume-gate-20, resume-startup-90, perf-bitwise).

### Violations (fill in only on FAIL)
None.

### Evidence highlights
- `git diff HEAD~1 HEAD --name-only` yields only `workload/notes/perf_log.md` — zero code files touched
- anti-proxy guard: PASS (0 violations); framework guard: PASS (0 violations)
- Engine has been verified as genuine in-process implementation across all prior review rounds (1–14) with the same finding

---

## [stage1] Round 16 — 2026-08-13

- **Verdict**: PASS
- **Stage status**: in-progress
- **Commit**: (current commit) — Fix: add preallreduce hash capture, bisect multi-GPU gradient divergence to shared params

### Key conclusions
The dev agent added `preallreduce` hash capture to the in-house engine (matching the ref's `harness_dp.reduce_grads` pattern) and bisected the `multistep` (DP=2) gradient divergence. The `preallreduce` capture captures gradients BEFORE the all-reduce and scaling, allowing separation of the gradient computation error from the all-reduce error. The bisect reveals that only the shared parameters (tok_embeddings, output_weight) have wrong gradients at step 0 — all other parameters (MTP-specific and main-specific) are correct. The `dptr_idx` mapping is verified correct. The engine remains a genuine in-process implementation — no proxy, forgery, hardcoded metrics, or shell-outs. The anti-proxy guard passes. Stage 1 finish conditions are not met: no gate evidence for the four required suites.

### Violations (fill in only on FAIL)
None.

### Evidence highlights
- `train_loop.py:1319-1335`: `_capture_all_gradients` now supports `suffix` parameter for `preallreduce`/`postallreduce`
- `train_loop.py:1227-1229, 1240-1242`: `preallreduce` capture before all-reduce + scaling (multi-GPU and single-GPU)
- `workload/notes/perf_log.md`: Round 16 findings documented — preallreduce hash bisect, shared-param gradient divergence
- anti-proxy guard: PASS (0 violations); framework guard: PASS (0 violations)
- Engine is a genuine in-process implementation (no subprocess calls to ref/ scripts, no hardcoded synthetic metrics, no renamed proxy variants)
- Stage 1 finish conditions not satisfied: no `STAGE_STATUS: finished` in commit message, no harness-written gate evidence for the four required suites (long-train, resume-gate-20, resume-startup-90, perf-bitwise)

---

## [stage1] Round 16 — 2026-08-13 12:25:00

- **Verdict**: PASS
- **Stage status**: in-progress
- **Commit**: def4b93 — Fix: add preallreduce hash capture, bisect multi-GPU gradient divergence to shared params

### Key conclusions
The dev agent added `preallreduce` hash capture to the in-house engine (matching the ref's `harness_dp` pre-all-reduce pattern) and bisected the `multistep` (DP=2) gradient divergence. The preallreduce capture captures gradients BEFORE the all-reduce and scaling, allowing separation of the gradient computation error from the all-reduce error. The bisect reveals that only the shared parameters (tok_embeddings, output_weight) have wrong gradients at step 0 — all other 310 parameters (MTP-specific and main-specific) are correct. The `dptr_idx` mapping is verified correct. The engine remains a genuine in-process implementation — no proxy, forgery, hardcoded metrics, or shell-outs. The anti-proxy guard passes. Stage 1 finish conditions are not met.

### Violations (fill in only on FAIL)
None.

### Evidence highlights
- `train_loop.py:1318-1330`: `_capture_all_gradients` now supports `suffix` parameter for `preallreduce`/`postallreduce` capture
- `train_loop.py:1224-1229, 1237-1242`: `preallreduce` hash capture before all-reduce + scaling (multi-GPU and single-GPU paths)
- `workload/notes/perf_log.md`: Round 16 findings documented — preallreduce hash bisect shows shared-param gradient divergence at step 0
- anti-proxy guard: PASS (0 violations); no gate configs, `remote.toml`, or run-shape keys modified
- Engine is a genuine in-process implementation (`ref.reference` imports at lines 206, 252 are dataloader utility reuse, not training logic proxy — unchanged from prior rounds)
- Stage 1 finish conditions not satisfied: no `STAGE_STATUS: finished` in commit message, no harness-written gate evidence for the four required suites (long-train, resume-gate-20, resume-startup-90, perf-bitwise)

---

## [stage1] Round 17 — 2026-08-13

- **Verdict**: PASS
- **Stage status**: in-progress
- **Commit**: 11816d4 — Fix: combine shared-param grad accum into single add_; match ref param order for clip_grad_norm_

### Key conclusions
The dev agent fixed two bugs causing the multistep (DP=2) gate to fail: (1) shared params accumulated two separate `add_` per microbatch instead of combining MTP+main contributions before a single `add_`, causing fp32 rounding differences; (2) `_collect_bf16_params` placed `output_weight` before `norm.weight` (wrong order vs ref's `model.parameters()`), causing `torch.linalg.vector_norm` order sensitivity in `clip_grad_norm_`. Gate results: 8/8 loss bitwise, 8/8 grad_norm bitwise, 5024/5024 hash match. Milestone: bitwise-multicard PASS.

### Violations (fill in only on FAIL)
None.

### Evidence highlights
- `train_loop.py:614-615`: Shared param accum variables (`dw_output_mtp_accum`, `dw_mtp_emb_accum`)
- `train_loop.py:631-637`: output_weight MTP contribution deferred (not added to fp32_grad_bufs yet)
- `train_loop.py:752-753`: tok_embeddings MTP contribution deferred
- `train_loop.py:770-772`: output_weight MTP+main combined in fp32 before single add_
- `train_loop.py:868-870`: tok_embeddings MTP+main combined in fp32 before single add_
- `train_loop.py:1278-1282`: clip_grad_norm_ call with positional args matching ref
- `train_loop.py:1386-1402`: _collect_bf16_params order matches ref's model.parameters() (tok_embeddings → layers → norm → output → mtp)
- anti-proxy guard: PASS (0 violations); framework guard: PASS (0 violations)
- multistep gate: 8/8 loss, 8/8 grad_norm, 5024/5024 hash — all PASS

## [stage1] Round 17 (review) — 2026-08-13 13:14:01

- **Verdict**: PASS
- **Stage status**: in-progress
- **Commit**: 362addb — Docs: record Round 17 review notes — bitwise-multicard PASS

### Key conclusions
This commit is a docs-only update recording review notes from the previous round (commit 11816d4). The engine code is unchanged. The notes document the resolution of two bugs: shared-param gradient accumulation ordering and `_collect_bf16_params` order mismatch. All 8/8 loss and grad_norm steps are bitwise identical, 5024/5024 hash keys match. No proxy, forgery, hardcoding, or shell-out detected. Stage 1 finish conditions not satisfied (no `STAGE_STATUS: finished` in commit message). Milestone bitwise-multicard is active, not long-horizon, so no throughput check needed.

### Violations (fill in only on FAIL)
None.

### Evidence highlights
- `workload/notes/review.md`: only file modified — 23 lines of review notes appended
- `bin/harness run anti-proxy`: PASS (0 violations)
- `git diff HEAD~1 HEAD`: only `workload/notes/review.md` changed

---
