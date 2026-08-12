<!-- Stage 2 specific review checks. Shared protocol lives in review_common.md. -->

## Stage 2 FINISH Decision

Decides which `STAGE_STATUS` to emit (see `review_common.md` "Output Contract").
Default emission is `STAGE_STATUS: in-progress`; only flip to `finished`
when **every** condition below holds. Audit on-disk evidence; do NOT
re-run training suites.

Run op-status to pull the current op table (read-only, does not consume GPU):

```bash
bin/harness run op-status --json | jq '.targets[0].payload.metrics'
```

1. The commit message contains the literal line `STAGE_STATUS: finished` (the
   dev agent's self-declaration, see the finish-protocol table in
   `prompt/develop_prompt/_shared/stage2.md`
   §"Stage 2 Endgame Signal"):

   ```bash
   git -C "$PWD" log -1 --format='%B' | grep -E '^STAGE_STATUS:[[:space:]]*finished[[:space:]]*$'
   ```

   Empty output → emit `STAGE_STATUS: in-progress`.

2. `summary_counts` shows: `failed == 0` AND `not_started == 0` AND
   `inconsistent == 0` AND `merged > 0`. Any other distribution (in
   particular `failed > 0`) → emit `STAGE_STATUS: in-progress`, and list
   the non-merged ops in the review note. **Do not** flip to `finished` while
   there are still `failed` ops.

3. The `attention` op's status == `merged` (priority=1, a mandatory-merge item):

   ```bash
   bin/harness run op-status --json \
     | jq -r '.targets[0].payload.metrics.operators[] | select(.name=="attention") | .status'
   ```

   Not `merged` → emit `STAGE_STATUS: in-progress`.

4. (Sanity) All `workload/ops/*/register.toml` files have `default !=
   "baseline"` — the dispatcher is really routing to the optimized version:

   ```bash
   for f in workload/ops/*/register.toml; do
     grep -E '^default[[:space:]]*=' "$f" | grep -E '"baseline"' \
       && echo "BAD: $f still default=baseline"
   done
   ```

   If there is any output → emit `STAGE_STATUS: in-progress` (a mutual defense
   with op-status's `inconsistent` case).

If any item fails → emit `STAGE_STATUS: in-progress` and list the missing
items in the review note; if all hold → emit `STAGE_STATUS: finished`.

---

The Stage 2 per-op safety checks (`PROMPT.md` verbatim embedding, whether
`test_op.py` is a stub, consistency of the merged op's `register.toml`,
per-op branch path isolation) are already enforced as hard gates in the
`bin/harness run op-inventory` / `bin/harness run op-status` tools; review does not
duplicate that lint. If you have semantic doubts about the `notes.md` /
`test_op.py` of a specific op, follow the "semantic review" flow in
`review_common.md` — read the file and write it into the review note — but do
not promote it into an extra condition of the FINISH decision; the FINISH
SSOT is the four items above only.
