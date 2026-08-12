# Coding Guidelines

## Scope

Active development surface for the agent loop is the training engine
implementation in `workload/src/training_engine_tensor/` plus the harness
control plane (`harness/`, `evals/`, `tools/`, `prompt/`, `config/eval.toml`,
`harness/tests/`). Read-only / out-of-scope: `ref/` (baseline ground truth),
checkpoint and baseline data, and `$MEGATRON_ROOT` (read as alignment reference).

## TDD Loop

- Add or update a focused test before changing behavior.
- Prefer lightweight local validation: `python -m unittest discover -s harness/tests -v`,
  `python -m harness.cli info --json`, and `python -m harness.cli run guard --json`.

## Configuration And Env

- Treat `config/eval.toml` plus `bin/harness info` as the source of truth for
  suite parameters, runtime knobs, and the harness↔workload contract.
- The L0 ref script (`ref/reference/${ref_script}`, set by `config/ref.toml`) is the
  only authoritative source for training-shape values (`HARNESS_GATE` preset,
  step counts, batch sizes, world size). `config/eval.toml` must not own
  training shape; it only describes how the harness invokes a suite.
- Environment variables that cross process boundaries must be declared and
  injected through the centralized runtime env helpers, not ad hoc in scripts.

## Development Red Lines (absolute zero tolerance)

These apply to every change the dev agent and review agent make. They mirror
the project-wide RED LINE rules in the global meta-rules; they are restated
here so they remain visible inside the stage prompt regardless of how the
outer harness composes its system prompts.

### 1. Single source of truth (SSOT)

- Every piece of knowledge — data schema, business rules, configuration
  values, behavior constraints — must have only one authoritative source.
- Shared registries and configuration files must have only one loader /
  writer pair; before adding a new source, the old source must first be
  deprecated or removed.
- Copying the same knowledge into multiple scripts, skills, rules, configs,
  or temporary logic for long-term coexistence is forbidden.
- Parsing the same authoritative source in multiple locations and forming
  implicit forks is forbidden.

### 2. Unidirectional dependency (DAG)

- All module, subsystem, and layer dependencies must form a Directed Acyclic
  Graph; higher layers only depend on lower layers; lower layers must not
  depend back on higher layers.
- Before adding a new dependency, you must first confirm that the target
  component is already at a more bottom-level or more fundamental position.
- Lower layers must not import, reference, or callback upper layers;
  bypassing the DAG via dynamic dispatch, callback, event, or temporary
  bridge is forbidden.
- Any circular dependency, upward dependency, or cross-layer reverse coupling
  is considered a violation.

### 3. Fail Fast

- Errors must be exposed immediately and the current path must be aborted;
  they must not be silently swallowed or disguised as a normal flow.
- Each class of error can only have one authoritative handling point; the
  boundary layer may translate exceptions into structured failure results,
  but must not fake success.
- "Empty state" and "bad state" must be distinguished; an existing bad state
  must immediately raise an error pointing to the authoritative file, path,
  or data source.
- Multi-layer try-catch wrapping the same error, masking data or protocol
  errors with retries, mapping invariant failures to `None`, `waiting`,
  `unknown` and other fake-normal values are all forbidden.
- Relying only on logging and continuing execution is forbidden; if the
  error should block, it must actually block.
