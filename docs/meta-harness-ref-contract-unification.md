# Meta-harness ↔ harness ref-contract unification

Status: design proposal (investigation complete, no code changed yet)

## Problem

The meta loop produces a DP×TP torch ref bundle; the forge loop (harness)
later drives that same bundle as its `ref` axis. Today the bundle's
`run.sh` works under the **meta** gate but breaks under the **harness**
gate, because the two sides use **two different, incompatible run
conventions** for the exact same script.

| | how config is selected | where topology comes from | comparison baseline |
|---|---|---|---|
| **meta gate** (`meta_harness/scripts/_shared/loss_drift_gate.py`) | passes `FORGE_GATE=drift` → bundle reads its own bundled `ref/config/drift.toml` | `TP_SIZE` / `WORLD_SIZE` **env** (set by the gate) | single-card oracle |
| **harness** (`bin/harness run <gate>`) | passes `FORGE_REF_CONFIG_DIR` + `FORGE_GATE=<gate>` → ref reads the harness-**rendered** `ref/config/<gate>.toml` | `tensor_parallel_size` read from the **config** | ours |

`drift.toml` is a meta-only, agent-authored artifact: it has no committed
source and lives only under `.artifacts/` (produced at parameterize/dp/tp,
then copied into the bundle at harness_configs by
`assemble_harness_configs._materialize_ref`). It exists solely to satisfy
the meta gate's `FORGE_GATE=drift` + env-topology convention.

### Evidence (code-verified)

- Harness hands over the *directory* + *gate name*, and delegates the
  path `join` to the ref script (bash):
  - `evals/_common.py:757` `merged_extra_env["FORGE_REF_CONFIG_DIR"] = repo_root/"ref"/"config"`
  - `tools/ref_script_runner.py:159` `env["FORGE_GATE"] = gate_name`
  - dispatcher never constructs a `<gate>.toml` path itself.
- The canonical harness ref does the join correctly and honors the dir:
  - `ref/reference/run_minicpm4_8b_dptp.sh:37`
    `_GATE_PRODUCT="${FORGE_REF_CONFIG_DIR:-$SCRIPT_DIR/../config}/$FORGE_GATE.toml"`
  - reads `tensor_parallel_size` from that config (TP is a config SSOT).
- The **meta bundle's** `run.sh` violates both points — it hardcodes
  `$SCRIPT_DIR/ref/config/${FORGE_GATE}.toml` (ignoring
  `FORGE_REF_CONFIG_DIR`) and reads `TP_SIZE`/`WORLD_SIZE` from env only.
  So `bin/harness run dptp` → looks for the bundle's own `dptp.toml`
  (only `drift.toml` ships) → `exit 1`; even if found, topology
  defaults to 1×1.
- The meta gate's judgment (`ref_side._ref_ok_from_status`) only checks
  {status file exists, not timed_out, returncode==0, dump_present}. No
  geometry check → a `run.sh` that *always* reads `drift.toml` and never
  reads the per-gate config would still pass, as long as it exits 0 and
  dumps. The current bundle is caught only by the accidental `exit 1`.

The harness-rendered `ref/config/dptp.toml` and the bundle's `drift.toml`
are **field-for-field the same schema** (`[cli]`+`[env]`, full model
geometry, muP, MTP, optimizer). Only the *values* differ (dptp:
`num_layers=4, world_size=4, tensor_parallel_size=2, gate_bitwise=true,
hash_capture_level=2`). So `train.py --config` can already consume the
harness-rendered toml as-is.

## Design goal

Collapse the two conventions into **one config source + one run
contract**, so a bundle that is correct for the meta gate is, by
construction, correct for the harness gate.

### Pillar 1 — delete the bundle-private `drift.toml`

Make the harness-rendered `ref/config/<gate>.toml` the **sole** physical
config source. The bundle ships `train.py` + a thin wrapper and **no
config of its own**.

Payoff: the "always read drift" cheat path becomes structurally
impossible — there is no drift to read; a wrapper that ignores
`FORGE_REF_CONFIG_DIR` has nothing to read and fails everywhere. No
"geometry echo verification" bolt-on is needed.

### Pillar 2 — the bundle wrapper honors the canonical contract

The bundle's `run.sh` mirrors `ref/reference/run_minicpm4_8b_dptp.sh`
verbatim:
- config path: `${FORGE_REF_CONFIG_DIR:-$SCRIPT_DIR/ref/config}/$FORGE_GATE.toml`
- topology: read `tensor_parallel_size` (and `world_size`) from that
  config, not from `TP_SIZE`/`WORLD_SIZE` env.

Because `run.sh` is agent-authored per meta loop, the durable fix is to
ship a copy-verbatim reference wrapper (adapted to the meta side's
tomllib-direct topology read, honoring `no-projection-tool`) and have the
parameterize/dp/tp/harness_configs SKILLs point the agent at it instead
of re-deriving.

### Pillar 3 — one enforcement point: the harness_configs gate

**Decision (2026-07-03).** We do NOT force parameterize/dp/tp onto
`bin/harness`. Those three meta milestones keep their own runner
(`loss_drift_gate.py`), because their comparison is genuinely different
(candidate-ref vs single-card **oracle** — ref-vs-ref) and the harness
dispatcher only knows ref-vs-**ours**; and because at dp/tp the forge
harness is not provisioned (siblings; meta workspace copies only
`meta_harness/`). Dragging the oracle comparison into the harness core to
satisfy an early meta stage is more invasive than the problem warrants.

Instead, **all enforcement concentrates at `harness_configs`** — the last
meta milestone, the hand-off point where the bundle *becomes* a real
harness ref. Two hard requirements:

1. The `harness_configs` gate MUST itself invoke `bin/harness run <gate>`
   (not trust an agent-produced provenance file).
2. That run MUST consume the forge loop's **rendered**
   `ref/config/<gate>.toml` (from `render_gate_configs`), never a
   bundle-private config.

If the bundle passes `harness_configs`, it is by construction correct for
the downstream forge loop — which drives it the exact same way.

## harness_configs gate — judgment criteria (判别条件)

### What runs today, and the two holes

- `config_gate.py` does two things: `validate()` (structural) +
  `check_provenance()` (reads the agent-produced
  `ref_side_provenance.json`).
- The thing that actually shells out `bin/harness run <gate>` is
  `meta_harness/scripts/_shared/ref_side.py:103` — but it runs *before*
  the gate, on the devspace, driven by the dev agent, and only leaves
  `ref_side_provenance.json` (`{bundle_sha, gates, all_ref_ok}`,
  `assemble_harness_configs.py:136`). The gate merely trusts that file.

Two holes:

- **Fabricable.** The provenance is agent-written; the gate never
  re-runs. `check_provenance` only verifies `bundle_sha` matches the
  config and `all_ref_ok==true` for all ref-side gates
  (`assemble_harness_configs.py:152-185`) — it does not prove a run
  happened. A stub `all_ref_ok:true` passes.
- **Geometry/config-consumption blind.** Even for a real run, the only
  ref-side signal is `ref_capture_status.json`
  `{returncode, timed_out, dump_present}` (`_common.py:950-969`;
  `gate_common.py:78-80`). There is NO runtime check that the ref read the
  rendered `ref/config/<gate>.toml` or ran at its `tensor_parallel_size`
  (confirmed: `tensor_parallel_size` lives only in the renderer,
  `render_gate_configs.py:64,173-192`; the dispatcher's runtime vocabulary
  is `world_size` only). An always-read-drift, always-1×1 ref that exits 0
  and dumps would pass.

### The change: move execution into the gate

Fold `ref_side.py`'s execution into the gate and drop the
`ref_side_provenance.json` trust hop. The `harness_configs` gate itself,
on the devspace:

1. Provisions the forge workspace from the authored config dir
   (`agent-loop.sh LOOP_PROVISION_ONLY=1`), which runs
   `render_gate_configs` → produces `ref/config/<gate>.toml` for every
   gate. Render failure → FAIL. *(This makes the rendered config the sole
   physical source — Pillar 1 — so "ship your own drift" is structurally
   dead.)*
2. For each **ref-side gate** (has a `[ref]` section: `forward-align`,
   `backward-align`, `multistep-1gpu`, `multistep`, `perf-bitwise`,
   `long-train`, `long-train-smoke`, `loss-gate-200`; ours-only
   `production-train`/`resume-*` are skipped), runs the full
   `bin/harness run <gate>` and judges from the **ref-side** artifacts it
   leaves under `.artifacts/runs/<gate>-*/`. The overall `result.json`
   verdict is ignored (see below); no `--ref-only` flag is added.

### Per-gate pass conditions

A ref-side gate PASSES iff **all** of:

1. **Rendered config exists** — `ref/config/<gate>.toml` was produced by
   `render_gate_configs`.
2. **Ref phase sound + capture matches the declared level** — this
   replaces the naive `dump_present==true` check, which is meaningless on
   the trajectory path (there `dump_present = bool(ref_run.succeeded)` =
   `rc==0 && !timed_out`, `gate_common.py:79` — it stats no file; only the
   alignment path really stats the `.pt`, `_common.py:1103` /
   `capture_artifacts.py:73`). Instead:
   - `ref_capture_status.json` exists AND `returncode==0` AND
     `timed_out==false` (missing file = fail-closed).
   - Read `hash_capture_level` from the rendered `ref/config/<gate>.toml`
     and **stat the artifact that level implies** (all paths relative to
     the run's `<artifact_dir>` = `.artifacts/.../runs/<gate>-<stamp>-<uuid>/`).
     The "persistent dump" is the end-of-run JSON that the persistent-mode
     `CaptureSession` writes across a multi-step trajectory
     (`persistent=True`, `harness_hook/__init__.py:356`); the alignment
     gates instead use single-step mode and write a `.pt`:
     - **level 0** (`long-train`, `long-train-smoke`, `loss-gate-200`):
       no hash dump by design — do not use dump as evidence; defer to
       condition 5.
     - **level 1** (`perf-bitwise` — the only ref-side level-1 gate;
       `resume-gate-20` is ours-only, no `[ref]`): the persistent dump
       `<artifact_dir>/ref_hash_dump.json` (dispatcher.py:1014) exists,
       non-empty, carries `loss.*` + `grad.*` keys
       (`harness_hook/__init__.py:401`).
     - **level 2, bitwise/trajectory gates** (`multistep-1gpu`,
       `multistep`, and the DP×TP `dptp`): the persistent dump
       `<artifact_dir>/ref_hash_dump.json` + its `.graph.json` sibling
       (`_dump.py:202`) exist and carry module `fwd.<fqn>` / `bwd.<fqn>`
       keys (registered only at level ≥2, `harness_hook/__init__.py:368`).
     - **level 2, alignment gates** (`forward-align`, `backward-align`):
       the single-step dump `<artifact_dir>/ref_dump__<suite>/ref_capture.pt`
       (basename `ref_capture.pt`, `dense_training.toml:80`) + its
       `.graph.json` sibling exist. NOTE: despite the `.pt` name this file
       is JSON hash records (`{hash, shape, dtype}`), written by the same
       `dump_capture_files` (`_dump.py:199`) as the bitwise path — the
       `.pt` extension + the `capture_artifacts.py` docstring ("pickled
       tensor dict") are a stale misnomer. No gate-comparison artifact
       holds raw tensors; every capture is a hash. (The only real
       `torch.save` `.pt` is the `install_canonical_state_dump` bootstrap
       weight file — a deterministic-init INPUT the comparator never
       touches, `__init__.py:444-495`.) This is the one path where the
       existing `dump_present = artifacts.tensor_dump_exists` already stats
       the real file (`_common.py:1103`, `capture_artifacts.py:73`).

   This doubles as a **capture-level consumption proof**: a ref that read
   a level-0 config (e.g. the bundle's own `drift.toml`) produces NO
   middle-tensor dump, so it cannot satisfy a level-2 gate — catching the
   `drift`(0)-vs-`dptp`(2) mismatch structurally, not by luck.
3. **Config-consumption proof (geometry axis)** *(closes the "did it read
   the rendered config" hole)* — for **every** field present in BOTH the
   ref's emitted `gate_metadata.json` AND the rendered
   `ref/config/<gate>.toml`, the values must be equal (intersection, not a
   fixed list — add as many as both sides carry). Any mismatch → FAIL;
   missing `gate_metadata.json` → FAIL. muP / MTP keys are compared only
   when a given ref actually emits them (not every ref has them), so the
   assertion never demands a field the ref legitimately lacks.
4. **Topology proof** *(closes the geometry blind spot)* —
   `gate_metadata.json` `tensor_parallel_size` (× dp) equals the rendered
   config's `tensor_parallel_size` (× `world_size`). Mismatch → FAIL.
   Feasible because the canonical DP×TP ref already emits
   `gate_metadata.json` with `tensor_parallel_size`
   (`run_minicpm4_8b_dptp.sh:147-150`), which the bundle inherits once its
   wrapper mirrors the canonical one (Pillar 2).
5. **Trajectory non-degenerate** — the parsed `[LOSS]` trajectory has the
   config's `num_steps` steps, all losses finite and not constant (all-0 /
   NaN / flat → FAIL). Reuse `resolve_ref_trajectory`'s window validation
   / `failure` field (`gate_common.py:29`).

### Deliberately NOT a criterion

- `result.json` `status=="passed"`. At authoring time there is no trainee
  engine, so the ours side is a stub and the overall verdict is an
  expected `failed` (`_common.py:955-959`). The gate runs the full
  `bin/harness run <gate>`, lets the ours stub fail, and keys purely off
  the ref-side artifacts above — exactly as `ref_side.py:103` already does
  today (subprocess `bin/harness run`, assert on `ref_capture_status.json`
  because the overall exit code is contaminated). No `--ref-only` is
  needed; the only change is moving that execution into the gate and
  adding conditions 2–5.

## Required harness-side additions

- **Universal `gate_metadata.json` emission** — today only `op-long`
  consumes it and not every ref script is confirmed to emit it. Every
  ref-side gate's ref must emit topology + geometry for conditions 3–4.
- **Config-vs-metadata + level-vs-dump assertion** (conditions 2–4) — new
  logic in the gate: parse the rendered config, stat the level-appropriate
  dump, diff the metadata intersection.

## Phasing

1. Contract first: bundle wrapper mirrors canonical `run.sh`
   (honors `FORGE_REF_CONFIG_DIR`, topology from config); delete the
   bundle-private `drift.toml`; point the meta SKILLs at a
   copy-verbatim reference wrapper.
2. Ensure universal `gate_metadata.json` emission across ref-side gates.
3. Rewrite the `harness_configs` gate to execute (provision → render →
   full `bin/harness run <gate>` per ref-side gate → the five per-gate
   conditions, keyed off ref-side artifacts) and retire
   `ref_side_provenance.json`.

## Prerequisites to verify before implementation

- Every ref-side gate's ref script emits `gate_metadata.json` with the
  geometry + `tensor_parallel_size` conditions 3–4 assert on (canonical
  DP×TP ref does; others unconfirmed).
- ~~The exact on-disk path of the persistent hash dump~~ — CONFIRMED:
  bitwise/trajectory gates write `<artifact_dir>/ref_hash_dump.json`
  (+ `.graph.json`), dispatcher.py:1014; alignment gates write
  `<artifact_dir>/ref_dump__<suite>/ref_capture.pt` (+ `.graph.json`),
  `ref_capture.pt` per `dense_training.toml:80`.
- `harness_configs` runs on the devspace with GPUs (today `config_gate.py`
  is "no GPU"; folding execution in makes it a GPU gate co-located with
  where `ref_side.py` runs now).
- `LOOP_PROVISION_ONLY=1` provisioning + `render_gate_configs` is callable
  from within the gate at `harness_configs` (agent-loop.sh path).
