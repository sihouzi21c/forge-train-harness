"""Static contracts pinning the *shape* of suite ↔ dispatcher communication.

Two related contracts share this file because they both protect the same
seam — the tuple of strings that must remain in sync across (a)
``config/eval.toml`` ``env_inputs`` declarations, (b) the dispatcher
``extras`` it injects into subprocess env, (c) the gate scripts under
``evals/scripts/`` that read those env vars, and (d) the
``RUNNER_METRICS`` advertised in ``harness.run_schema``:

* :class:`TestSuiteEnvContract` (Issue #15) — every required env var read
  by a suite script must be declared in that suite's ``env_inputs`` (or
  the workload-level ``[env]`` table, or come from a small implicit
  allowlist set by the harness — RANK/LOCAL_RANK/WORLD_SIZE injected
  per-child by ``evals/scripts/launch_dp.py`` for DP suites, or by the
  single-card ``ensure_*_single_card_env`` helper in
  ``evals/_common.py``).
* :class:`TestRunnerMetricsContract` (Issue #16) — every metric key
  declared in ``run_schema.RUNNER_METRICS[runner_kind]`` must actually
  appear as a string literal somewhere in the corresponding ``_run_*``
  body inside ``evals/dispatcher_stage2.py`` (op-*) or the verdict
  module / ``harness/app.py`` handler.  This is intentionally a *static*
  string-presence check (not a behavioural mock) because the alternative
  — fully mocking every dispatcher path end-to-end — would couple the
  test to internal helpers and turn it into a refactor blocker.
  String-presence is weaker than runtime equality, but strong enough to
  catch the most common drift mode: someone deletes/renames a metric in
  the dispatcher and forgets to update RUNNER_METRICS (or vice-versa).

Both tests parse the source files directly so the contract holds even
when the GPU host the tests run on is offline; they are part of the
local pre-flight suite.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Env vars provided by the *harness* runtime (not by any suite cfg) that
# scripts may legitimately read without declaring them under env_inputs.
# Keep this allowlist minimal and explicit:
#
#   * ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE`` — set per-child by
#     ``evals/scripts/launch_dp.py`` for DP suites, or by the
#     ``ensure_*_single_card_env`` helper in ``evals/_common.py`` so
#     single-card scripts can initialize a process group without
#     standing up a torchrun launcher.
#   * ``DUMP_DIR`` — set by ``tools/ref_script_runner.py`` for ref runs
#     and by gate scripts for ours-side dumps; the dispatcher passes it
#     through ``base_env``.
#   * ``HOOK_OUTPUT_FILE`` — capture hook contract, set by the
#     dispatcher (``evals/_common.run_ref_capture``) on the ref-side
#     bridge process so :func:`evals.harness_hook.install` writes the
#     tensor dump + .graph.json at that exact path.
_HARNESS_INJECTED_ENV: frozenset[str] = frozenset(
    {
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "DUMP_DIR",
        "HOOK_OUTPUT_FILE",
    }
)


def _required_env_reads(script_path: Path) -> set[str]:
    """Return the set of env vars the script reads *without* a default.

    Patterns recognised:

    * ``os.environ["FOO"]`` (subscript access — always required).
    * ``os.environ.get("FOO")`` *with no second argument* (semantically
      identical to subscript — caller will treat ``None`` as missing).
    * ``_require_env("FOO")`` — the genuine env reader used by the
      product-less op-long runner (``op_long_ours.py``); raises on missing.

    ``_require("FOO")`` is deliberately NOT recognised: in the product-backed
    runners it aliases ``_gate_entry.GateInputs.require`` (``_GATE.require``),
    which reads the gate's value from the rendered ours PRODUCT — not from the
    environment ("no env transport, the product is the sole source", see
    ``_gate_entry.py``). Those keys belong in the product ``[cli]``/``[env]``,
    not in ``env_inputs``; the transport contract is checked by
    ``test_ours_env_inputs_transport`` / ``test_gate_transport_parity``.

    Reads that supply a default (``os.environ.get("FOO", "/tmp")``) are
    *not* required because the script tolerates the var being unset, so
    the env_inputs contract does not need to declare them.

    A lightweight AST walk is used to keep the test free of regex
    false-positives (e.g. inside string literals or comments).
    """
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(script_path))
    required: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            value = node.value
            slice_node = node.slice
            if (
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id == "os"
                and value.attr == "environ"
                and isinstance(slice_node, ast.Constant)
                and isinstance(slice_node.value, str)
            ):
                required.add(slice_node.value)
        elif isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Attribute)
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "os"
                and func.value.attr == "environ"
                and func.attr == "get"
                and len(node.args) == 1
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ) or (
                isinstance(func, ast.Name)
                and func.id == "_require_env"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                required.add(node.args[0].value)
    return required


def _load_workload() -> dict:
    from harness._compat import tomllib

    return tomllib.loads(
        (REPO_ROOT / "config" / "eval" / "dense_training" / "dense_training.toml").read_text(
            encoding="utf-8"
        )
    )


class TestSuiteEnvContract(unittest.TestCase):
    """env_inputs ⊇ (script-required env reads) − harness-injected."""

    def test_every_script_required_env_is_declared(self) -> None:
        wc = _load_workload()
        env_keys = {str(k).upper() for k in wc.get("env", {})}
        for suite_name, cfg in wc.get("evals", {}).items():
            script = cfg.get("script")
            if not script:
                continue
            script_path = REPO_ROOT / script
            if not script_path.exists():
                continue
            if script_path.suffix != ".py":
                # Bash drivers (e.g. train_ours_al.sh for wsd-sft) get their
                # env via the handler's process extra, not env_inputs — the
                # AST contract only covers Python entrypoints.
                continue
            required = _required_env_reads(script_path)
            declared = {str(x).upper() for x in cfg.get("env_inputs", [])}
            allowed = declared | env_keys | _HARNESS_INJECTED_ENV
            missing = sorted(required - allowed)
            with self.subTest(suite=suite_name, script=script):
                self.assertFalse(
                    missing,
                    f"Suite [evals.{suite_name}] script {script!r} reads env "
                    f"{missing!r} but they are not declared in "
                    f"env_inputs nor in the workload [env] table. "
                    "Either add them to env_inputs (and have the dispatcher "
                    "inject them via cfg/extras) or — if the harness sets "
                    "them implicitly — extend "
                    "_HARNESS_INJECTED_ENV in this test with a comment "
                    "explaining who sets the value.",
                )


class TestRunnerMetricsContract(unittest.TestCase):
    """Every RUNNER_METRICS key must appear in the matching judge body.

    Stage1 runner_kinds are judged by ``evals/verdicts/<module>.py`` (the
    thin-dispatcher migration); their pooled source is the verdict module
    text plus the shared ``_shared.py`` + ``_kernels.py`` helpers. Stage2
    op-* and the local
    guard/unit/anti-proxy kinds still map to handler functions parsed out
    of ``evals/dispatcher.py`` / ``harness/app.py``. The check is
    string-literal presence; ``world_size`` is matched both as
    ``"world_size"`` and as a key inside ``f"world_size={n}"`` etc.
    """

    # Stage1 runner_kind → verdict module basename under evals/verdicts/.
    _RUNNER_VERDICTS: dict[str, str] = {
        "forward-align": "align",
        "backward-align": "align",
        "stage1-bitwise-trajectory": "bitwise",
        "long-train": "long_train",
        "loss-gate": "loss_gate",
        "resume-gate": "resume",
        "resume-startup": "resume_startup",
        "production-train": "production",
        "profile-snapshot": "profile",
        "wsd-sft": "wsd_sft",
    }

    # Non-scripted runner_kind → handler function name(s) pooled from
    # dispatcher_stage2.py (op-*) / app.py (local suites).
    _RUNNER_HANDLERS: dict[str, tuple[str, ...]] = {
        "op-long": ("_run_op_long",),
        "op-status": ("_run_op_status",),
        "op-inventory": (
            "_run_op_inventory",
            "_run_stage2_validation",
            "_stage2_validation_result",
        ),
        "guard": ("_run_guard_suite",),
        "unit": ("_run_unit_suite",),
        "anti-proxy": ("_run_anti_proxy_suite",),
    }

    def setUp(self) -> None:
        # Pool the stage2 op-* handlers (``evals.dispatcher_stage2``) and the
        # local-suite handlers in ``harness.app`` (guard / unit emit metrics
        # there, not in the dispatcher) so the source-presence check finds both.
        self._fn_source: dict[str, str] = {}
        for source_path in (
            REPO_ROOT / "evals" / "dispatcher_stage2.py",
            REPO_ROOT / "harness" / "app.py",
        ):
            text = source_path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=source_path.name)
            lines = text.splitlines()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = node.lineno - 1
                    end = (node.end_lineno or node.lineno) - 1
                    self._fn_source.setdefault(node.name, "\n".join(lines[start : end + 1]))
        self._shared_src = "\n".join(
            (REPO_ROOT / "evals" / "verdicts" / name).read_text(encoding="utf-8")
            for name in ("_shared.py", "_kernels.py")
        )

    def _pooled_source(self, runner_kind: str) -> tuple[str, str]:
        """Return (pooled source, human-readable origin) for a runner_kind."""
        module = self._RUNNER_VERDICTS.get(runner_kind)
        if module is not None:
            path = REPO_ROOT / "evals" / "verdicts" / f"{module}.py"
            return (
                path.read_text(encoding="utf-8") + "\n" + self._shared_src,
                f"evals/verdicts/{module}.py (+_shared.py, _kernels.py)",
            )
        handlers = self._RUNNER_HANDLERS.get(runner_kind)
        self.assertIsNotNone(
            handlers,
            f"runner_kind {runner_kind!r} has no entry in _RUNNER_VERDICTS "
            "nor _RUNNER_HANDLERS — extend the test mapping when adding a "
            "new runner_kind.",
        )
        return (
            "\n".join(self._fn_source.get(name, "") for name in handlers),
            repr(handlers),
        )

    def test_runner_metrics_all_declared_in_dispatcher(self) -> None:
        from harness.run_schema import RUNNER_METRICS

        for runner_kind, metric_keys in RUNNER_METRICS.items():
            with self.subTest(runner_kind=runner_kind):
                pooled_src, origin = self._pooled_source(runner_kind)
                if not metric_keys:
                    continue
                self.assertTrue(
                    pooled_src,
                    f"No judge source found for runner_kind {runner_kind!r} ({origin}).",
                )
                for key in metric_keys:
                    quoted = f'"{key}"'
                    self.assertIn(
                        quoted,
                        pooled_src,
                        f"RUNNER_METRICS[{runner_kind!r}] declares "
                        f"{key!r} but no string literal {quoted} appears "
                        f"in {origin}. Either remove the metric from "
                        "RUNNER_METRICS or emit it from the judge.",
                    )

    def test_every_workload_runner_kind_has_metrics_entry(self) -> None:
        from harness.run_schema import RUNNER_METRICS

        wc = _load_workload()
        runner_kinds: set[str] = set()
        for table in ("evals", "local_suites"):
            for cfg in wc.get(table, {}).values():
                rk = cfg.get("runner_kind")
                if isinstance(rk, str):
                    runner_kinds.add(rk)
        missing = sorted(runner_kinds - set(RUNNER_METRICS))
        self.assertFalse(
            missing,
            f"Suites declare runner_kind(s) {missing!r} but "
            "harness.run_schema.RUNNER_METRICS has no entry for them. "
            "Add a (possibly empty) list there so the metric contract is "
            "explicit.",
        )


if __name__ == "__main__":
    unittest.main()
