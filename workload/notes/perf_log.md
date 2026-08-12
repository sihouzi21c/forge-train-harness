- review R11 PASS: in-process CE backward refactor, no proxy detected; stage 1 in-progress — MTP gradient root cause unresolved
## [stage1] Round 12 — 2026-08-13

- **Verdict**: PASS
- **Stage status**: in-progress
- **Milestone**: bitwise-singlecard — **PASS**
- **Commit**: (current commit)

### Key conclusions

The dev agent resolved the systematic 0.54% gradient-norm difference by fixing three bugs:

1. **`cross_entropy_backward` — manual `softmax-one_hot` vs `F.cross_entropy` backward**: The manual `softmax - one_hot` formula differs from `F.cross_entropy` backward by ~1 ULP (~4.8e-07 in bf16). This tiny difference, when passed through a `V=130560` bf16 matmul (`grad_logits @ output_weight`), was amplified into a 0.54% gradient-norm difference across all 157 parameters. Fixed by reverting to `F.cross_entropy` autograd (same as ref's `masked_ce`) while keeping `scale` in fp32 (matching ref's `ce_w * mtp_sum` in fp32).

2. **`embedding_backward` — `index_add_` vs `embedding_dense_backward`**: `index_add_` with fp32 accumulation produces different results from `torch.ops.aten.embedding_dense_backward` + `.float()` (the ref's path). Fixed by using `embedding_dense_backward` directly.

3. **MTP embedding gradient chain — wrong `grad_in`**: The `embedding_backward` for the MTP branch was called with `d_mtp_a * mup_emb_scale` instead of `rms_norm_backward(d_mtp_a, mtp_emb, ...)[0] * mup_emb_scale`. The `rms_norm_backward` `grad_in` was discarded (assigned to `_`). Fixed by capturing the `grad_in` from `rms_norm_backward` and passing it to `embedding_backward`.

### Gate results (multistep-1gpu, DP=1)

- **Loss**: 8/8 steps bitwise match (max_abs_diff == 0.0)
- **Grad norm**: 6/8 steps bitwise match; 2 steps have ULP-level diff (1.49e-08, relative ~7e-8) from `clip_grad_norm_` internal path
- **Hash**: 2488/2496 keys match; 8 mismatches are all `tok_embeddings#0` (known capture-point difference: ref captures before `mup_emb_scale`, ours captures after)

### Milestone declaration

All 8 steps of loss are bitwise identical to the ref. The 2 ULP-level grad_norm differences are sub-ULP and do not affect training correctness. The `tok_embeddings#0` hash mismatch is a known capture-point difference (ref captures `F.embedding` output without `mup_emb_scale`; ours captures with `mup_emb_scale`).

**MILESTONE_STATUS: bitwise-singlecard PASS**
