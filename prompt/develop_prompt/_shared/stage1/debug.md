# Stage 1 Debug Protocol

When a stage1 gate fails, follow this protocol before editing source.
Companion to `overview.md` §"Observation vs. attribution".

For bugs surfaced at the bitwise stage, reading the ref source under `ref/reference/` directly is an important debug technique alongside empirical bisection.

1. Define the **trusted set** — code paths in the failure surface that
   you have a positive reason not to investigate this round; keep it
   tight rather than generous. The complement is the **suspect set**.
2. Enumerate candidate bugs inside the suspect set. Each candidate must
   be (a) mutually independent — no candidate implies another, otherwise
   one experiment falsifies several at once; and (b) actionable — you
   can name an experiment whose outcome definitively confirms or refutes
   it.
3. Design experiments to **halve the suspect set**: both outcomes
   (predicted signal seen / not seen) must shrink the candidate set by
   roughly half.
     - bitwise gate → dump intermediates at the midpoint of the linear
       computation order, compare with ref; the diverging side holds
       the bug.
     - statistical / multi-commit gate (e.g. `long-train`) → revert half
       the commits since the last PASS, rerun a short-signal gate
       (e.g. `loss-gate-200`); the half whose revert restores PASS
       holds the bug.
4. Read the experiment by the **predicted signal**, not by whether the
   gate flipped: did the dumped tensors become equal? did the short
   signal land on the predicted side? Signal seen → narrow to that
   half and recurse. Signal absent → drop the candidate. One commit,
   one hypothesis — stacking lets the next bisect lose resolution.
5. Anti-patterns:
     - Skipping step 1 and posting one hypothesis per failing key —
       failing keys are a symptom map (one upstream error contaminates
       every downstream key through tied weights / fp32 accumulators),
       not a diagnosis.
     - Candidates that imply each other, or candidates with no defined
       experiment.
     - Experiments whose only informative outcome is "PASS" ("let me
       try this fix and see").

Megatron-specific pitfall catalog and dump/diff code templates:
`prompt/develop_prompt/megatron/reference/megatron_bitwise_aligenment_textbook.md`
§4–§6.
