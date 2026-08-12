<!-- Stage 1 specific review checks. Shared protocol lives in review_common.md. -->

## Stage 1 FINISH Decision

Decides which `STAGE_STATUS` to emit (see `review_common.md` "Output Contract").
Default emission is `STAGE_STATUS: in-progress`; only flip to `finished`
when **every** condition below holds. Audit on-disk evidence; do NOT
re-run training suites.

1. The commit message contains the literal line `STAGE_STATUS: finished` (the
   dev agent's self-declaration):

   ```bash
   git -C "$PWD" log -1 --format='%B' | grep -E '^STAGE_STATUS:[[:space:]]*finished[[:space:]]*$'
   ```

   Empty output → emit `STAGE_STATUS: in-progress`.

2. The latest section of `workload/notes/perf_log.md` simultaneously records
   these three pieces of green-light evidence (per
   `prompt/develop_prompt/_shared/stage1/overview.md`
   §"Per-round procedure" rule 1, the dev agent writes the effective evidence of every
   round into perf_log). Each suite must meet its own gate as declared in
   `config/eval.toml [evals.<suite>]` (thresholds + gate window owned there;
   do not hardcode the numbers here):

   - `long-train` PASS: pointwise relative loss below `[evals.long-train].loss_rel_threshold`
     over its gate window (the harness long-train verdict is loss-only;
     throughput is checked by the review-side check in the
     "long-horizon milestone check" section below) AND that
     review-side throughput check passes
   - `resume-gate-20` PASS: `max_abs_diff == 0` over `[evals.resume-gate-20]`'s gate window
   - `resume-startup-90` PASS: `resume_startup_s <= [evals.resume-startup-90].resume_startup_budget_s` (resume from step 90; the dataloader seek must NOT replay prior steps)
   - `perf-bitwise` PASS: `max_abs_diff == 0` on the bitwise window AND
     `avg MFU(standard) >= [evals.perf-bitwise].mfu_e2e_target`

   Read the file, do not re-run the suite:

   ```bash
   tail -n 200 workload/notes/perf_log.md \
     | rg -i 'long-train|resume-gate-20|resume-startup-90|perf-bitwise|mfu_e2e_standard|max_abs_diff|resume_startup'
   ```

   Missing any item / stale evidence (this commit did not write a new section) →
   emit `STAGE_STATUS: in-progress`.

3. (Sanity) `git -C "$PWD" log --oneline -5` shows that recent commits really
   touched `workload/src/training_engine_tensor/` — the dev agent is
   working on the active write surface, not only modifying prompt / docs.

4. **Profile snapshot present for perf-touching rounds.** If
   `git -C "$PWD" diff HEAD~..HEAD --name-only` includes any of
   `workload/src/training_engine_tensor/{forward,backward,kernels,triton_kernels,optimizer,nccl,train_loop,dataloader,parameters,config}.py`,
   the same commit MUST contain:

   - a NEW `workload/notes/profile/M{4,6}_round<N>/summary.md` (the
     dev agent runs `bin/harness run profile-snapshot M{4,6}_round<N>`
     after the gate passes; the dispatcher writes the nsys-backed
     renderer outputs into this directory — check
     `git show --name-only HEAD`),
   - a `workload/notes/perf_log.md` entry whose body references that
     `summary.md` path by string,
   - at least one `Δ from ` line quoted from that `summary.md` in the
     `perf_log.md` entry (the dev agent must justify the change, not
     just point at the file).

   Failing any of these = methodology violation = emit
   `STAGE_STATUS: in-progress` regardless of gate pass. Exempt:
   commits whose diff is fully outside the listed files (bitwise
   repair-only, docs-only, refactor-only, dispatcher/test changes).

   ```bash
   git -C "$PWD" diff HEAD~..HEAD --name-only \
     | rg '^workload/src/training_engine_tensor/(forward|backward|kernels|triton_kernels|optimizer|nccl|train_loop|dataloader|parameters|config)\.py$' \
     && git -C "$PWD" show --name-only HEAD \
        | rg '^workload/notes/profile/M[46]_round[0-9]+/summary\.md$'
   ```

If any item fails → emit `STAGE_STATUS: in-progress` and list the missing
items in the review note; if all hold → emit `STAGE_STATUS: finished`.

## long-horizon milestone check (review-side throughput check)

Runs on **every** round whose active milestone is `long-horizon` —
read it from `.artifacts/agent-loop-state/stage1.milestone`; any other
milestone → skip this section entirely.

The milestone advances **only** through this check. Dev declarations
are retired for long-horizon: a commit containing `MILESTONE_STATUS:
long-horizon PASS` carries no gating meaning — do not verify it, do
not penalize it, simply ignore the line. Evidence is exclusively the
harness-written telemetry (full 200-step `long-train` runs recorded in
the per-loop history by the harness; the loop itself launches one
every few smoke runs, so evidence accumulates without any dev action).

Run the deterministic checker from the workspace root (read-only):

```bash
python3 tools/mfu_elastic_check.py
```

Trust `MFU_GATE_VERDICT` verbatim. Do **no** MFU arithmetic yourself,
and do not open the script or its config to inspect the policy.

- `MFU_GATE_VERDICT: FAIL` → **nothing to do.** This is NOT a review
  failure and must not affect `REVIEW_VERDICT`. Do not mention the
  check ran; proceed with the normal semantic review. One exception:
  if the reason is `NO_HISTORY` but a fresh `long-train` `result.json`
  exists under `.artifacts/` (telemetry sync gap after a remote run),
  re-run once with `--result-json <path>` before concluding.
- `MFU_GATE_VERDICT: PASS` → the telemetry proves sufficient
  throughput. Before advancing anything, audit the rest of the
  hand-off bundle from harness artifacts (not dev prose): the newest
  `resume-gate-20` result is green (`max_abs_diff == 0`) AND the
  newest `perf-bitwise` result passes (no regression).
  - Bundle green → write **three** lines to the review status file:
    the two standard contract lines, plus `MILESTONE_OVERRIDE:
    long-horizon` (exactly that token, nothing appended). The loop
    advances the milestone on this flag alone.
  - Bundle not green → **no override.** In the one-line `perf_log.md`
    feedback you may name the broken gate ("restore perf-bitwise
    before hand-off") — gate names are not the bar — but say nothing
    about throughput sufficiency.

**Deliberate exception to "do not hardcode the numbers here"**: the
numeric throughput policy (the review bar `review_mfu_target` and its
bounded, stability-conditioned elastic tolerance) lives ONLY in the
`[shared.gate]` table of the gate SOURCE
`$FORGE_CONFIG_DIR/gate_config/long-train.toml`, which the checker
reads directly. The renderer strips these keys from the dev-readable
rendered products, and no dev rule file points at the gate sources —
so the values stay off the dev agent's read surface, and this prompt
intentionally states no number at all. This review-side bar applies
regardless of whether the suite also sets a dev-visible
`mfu_e2e_target`.

**Non-disclosure (hard rule)**: never write the numeric bar, the
tolerance, the band edge, the measured shortfall, any
`MFU_GATE_REASON` code, or the checker script's name/path into
`workload/notes/perf_log.md` or `workload/notes/review.md` — both are
dev-readable and disclosing any of these breaks the blinding (naming
the script tells the dev exactly which file to read). The
`MILESTONE_OVERRIDE` line goes ONLY into the review status file —
never into dev-readable notes — and a per-round check that did not end
in an override must leave no trace in dev-readable files at all. Feedback
must stay qualitative; map the reason to one line, e.g.:

- `BELOW_BAND` →
  `- review R7 FAIL: long-horizon PASS rejected — achieved throughput insufficient, continue MFU optimization`
- `IN_BAND_UNSTABLE` / `IN_BAND_INSUFFICIENT_HISTORY` →
  `- review R7 FAIL: long-horizon PASS rejected — throughput close but not yet demonstrated as a stable plateau; keep optimizing and re-validate with repeated full long-train runs`
- `NO_HISTORY` / `NO_QUALIFYING_RUNS` →
  `- review R7 FAIL: long-horizon PASS rejected — throughput evidence missing or unverifiable`

In `review.md`, likewise say "below the review-side throughput bar" or
"not yet a stable plateau" without stating any value.
