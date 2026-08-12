"""Behavioral tests for the thin dispatcher and its verdict modules.

Stage1 gate judgment moved out of the dispatcher handlers into
``evals/verdicts/<module>.py`` (thin-dispatcher migration): the generic
scripted executor (``dispatcher._run_scripted_suite``) runs the ref/ours sh
sides and writes ``<side>/status.json`` + logs into the run dir; the verdict
module judges those on-disk artifacts. Tests here therefore exercise the
verdicts directly against on-disk run_dir fixtures (status.json / ref loss
dump / [LOSS] logs / hash dumps / a rendered ref product for shape), plus the
dispatcher surfaces that remain in-process: ``run_suite`` routing and the
stage2 ``op-*`` handlers.

Retired with the legacy handlers (subprocess env threading is now owned by
``evals/scripts/run_ours.sh`` + ``runtime_env.py`` and verified on the GPU
devspace, not by subprocess mocks here):

* ``test_bitwise_trajectory_uses_declared_env_inputs``
* ``test_long_train_uses_ref_metadata_for_runtime_shape_env``
* ``test_long_train_ours_env_overrides_ref_mbs_while_keeping_gbs``
* ``test_resume_gate_composes_run_config_without_injecting_shape``
  (``gate_product.compose_run_config`` was deleted; the resume sh reads the
  product directly)
* ``TestM1CaptureDiffTimeout.test_timeout_argument_falls_back_to_default``
  (default-timeout forwarding is generic-executor mechanics now:
  ``_run_scripted_suite`` resolves ``config_runtime.suite_timeout_s`` and
  records the timeout in ``status.json``)
"""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from evals import dispatcher, dispatcher_stage2
from evals.verdicts import align, bitwise, long_train, loss_gate

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ── run-dir fixture helpers (the generic executor's on-disk contract) ──


def _write_side_status(
    side_dir: Path,
    *,
    returncode: int = 0,
    elapsed_s: float = 1.0,
    timed_out: bool = False,
    timeout_s: int | None = None,
) -> None:
    side_dir.mkdir(parents=True, exist_ok=True)
    (side_dir / "status.json").write_text(
        json.dumps(
            {
                "returncode": returncode,
                "elapsed_s": elapsed_s,
                "timed_out": timed_out,
                "timeout_s": timeout_s,
                "cached": False,
            }
        ),
        encoding="utf-8",
    )


def _write_ref_product(repo_root: Path, gate: str, cli: dict) -> None:
    """Write a minimal rendered ref product carrying only the gate shape.

    Verdict thresholds stay in the suite cfg: ``overlay_product_verdict``
    only overrides keys the product actually declares.
    """
    path = repo_root / "ref" / "config" / f"{gate}.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["[cli]"]
    for key, value in cli.items():
        if isinstance(value, list):
            lines.append(f"{key} = [{', '.join(str(v) for v in value)}]")
        elif isinstance(value, str):
            lines.append(f'{key} = "{value}"')
        else:
            lines.append(f"{key} = {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ref_loss_dump(run_dir: Path, lines: str) -> None:
    dump_dir = run_dir / "ref" / "dump"
    dump_dir.mkdir(parents=True, exist_ok=True)
    (dump_dir / "ref_loss.txt").write_text(lines, encoding="utf-8")


def _write_ours_log(run_dir: Path, text: str) -> None:
    ours_dir = run_dir / "ours"
    ours_dir.mkdir(parents=True, exist_ok=True)
    (ours_dir / "ours.log").write_text(text, encoding="utf-8")


def _run_bitwise_verdict(
    *,
    suite: str = "multistep-1gpu",
    shape: dict | None = None,
    cfg: dict | None = None,
    ref_lines: str,
    ours_lines: str,
    ours_returncode: int = 0,
) -> dict:
    """Drive ``evals.verdicts.bitwise`` against an on-disk run_dir fixture."""
    shape = shape or {"world_size": 1, "num_steps": 8, "gate_window": [0, 8]}
    cfg = cfg or {
        "milestone": "bitwise-singlecard",
        "loss_abs_threshold": 0,
        "grad_norm_abs_threshold": 0,
    }
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        run_dir = repo_root / ".artifacts" / "runs" / "test"
        _write_ref_product(repo_root, suite, shape)
        _write_side_status(run_dir / "ref", returncode=0)
        _write_ref_loss_dump(run_dir, ref_lines)
        _write_side_status(run_dir / "ours", returncode=ours_returncode)
        _write_ours_log(run_dir, ours_lines)
        return bitwise.run(
            suite_key=suite,
            run_dir=run_dir,
            repo_root=repo_root,
            workload_config={"evals": {suite: cfg}},
        )


# Minimal real hash-record dump the align verdict can load (same JSON shape
# ``_dump.dump_capture_files`` writes: ``{hash, shape, dtype}`` per key).
_FWD_RECORDS = {"fwd.layer0": {"hash": "0" * 32, "shape": [4], "dtype": "float32"}}


def _run_align_verdict(
    *,
    suite: str = "forward-align",
    milestone: str = "alignment.forward",
    ref_returncode: int = 0,
    ours_returncode: int = 0,
    ours_timed_out: bool = False,
    ours_timeout_s: int | None = None,
    ref_records: dict | None = _FWD_RECORDS,
    ours_records: dict | None = _FWD_RECORDS,
    ours_log: str = "stdout",
) -> dict:
    """Drive ``evals.verdicts.align`` against an on-disk run_dir fixture.

    ``ref_records`` / ``ours_records`` = None means the side never produced
    its hash dump (crash / timeout before flush).
    """
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        run_dir = repo_root / ".artifacts" / "runs" / "test"
        _write_ref_product(repo_root, suite, {"world_size": 1})
        ref_dir = run_dir / "ref"
        ours_dir = run_dir / "ours"
        _write_side_status(ref_dir, returncode=ref_returncode)
        if ref_records is not None:
            (ref_dir / "ref_hash_dump.json").write_text(json.dumps(ref_records), encoding="utf-8")
        _write_side_status(
            ours_dir,
            returncode=ours_returncode,
            timed_out=ours_timed_out,
            timeout_s=ours_timeout_s,
        )
        if ours_records is not None:
            (ours_dir / "ours_hash_dump.json").write_text(
                json.dumps(ours_records), encoding="utf-8"
            )
        (ours_dir / "ours.log").write_text(ours_log, encoding="utf-8")
        wc = {"evals": {suite: {"milestone": milestone, "gate_atol": 0}}}
        return align.run(suite_key=suite, run_dir=run_dir, repo_root=repo_root, workload_config=wc)


class TestBitwiseGradNormGate(unittest.TestCase):
    """The bitwise milestones must gate on ``grad_norm`` bitwise, not only ``loss``.

    ``stage1/bitwise-multicard.md`` §Gate requires per-step bitwise alignment
    on both ``global_loss`` and ``grad_norm`` (``max_abs_diff == 0``). The
    ``evals.verdicts.bitwise`` verdict therefore compares both fields; a
    grad-norm divergence with bit-identical loss must fail the gate, and a
    ref trajectory missing its grad-norm baseline must fail fast rather than
    silently grading on loss alone.
    """

    _SHAPE = {"world_size": 2, "num_steps": 4, "gate_window": [0, 4]}
    _CFG = {
        "milestone": "bitwise-multicard",
        "loss_abs_threshold": 0,
        "grad_norm_abs_threshold": 0,
    }

    def _run(self, *, ref_lines: str, ours_lines: str) -> dict:
        return _run_bitwise_verdict(
            suite="multistep",
            shape=self._SHAPE,
            cfg=dict(self._CFG),
            ref_lines=ref_lines,
            ours_lines=ours_lines,
        )

    def test_grad_norm_divergence_fails_even_with_bitwise_loss(self) -> None:
        # Loss is bit-identical to the baseline at every step, but the
        # ours-side grad_norm drifts at step 2. The gate must FAIL.
        ref_lines = (
            "\n".join(f"[LOSS] step={s} global_loss=1.0 grad_norm=2.0 time_s=1.0" for s in range(4))
            + "\n"
        )
        ours_lines = (
            "\n".join(
                f"[LOSS] step={s} global_loss=1.0 "
                f"grad_norm={'2.0' if s != 2 else '2.0000001'} time_s=1.0"
                for s in range(4)
            )
            + "\n"
        )
        result = self._run(ref_lines=ref_lines, ours_lines=ours_lines)
        self.assertEqual(result["status"], "failed")

    def test_grad_norm_bitwise_match_passes(self) -> None:
        lines = (
            "\n".join(f"[LOSS] step={s} global_loss=1.0 grad_norm=2.0 time_s=1.0" for s in range(4))
            + "\n"
        )
        result = self._run(ref_lines=lines, ours_lines=lines)
        self.assertEqual(result["status"], "passed")

    def test_missing_grad_baseline_fails_fast(self) -> None:
        # Ref loss dump carries loss-only lines (no grad_norm field — e.g.
        # the ref fell back to a producer that emits no grad_norm). The gate
        # must NOT silently grade on loss alone — it must fail.
        ref_lines = "\n".join(f"[LOSS] step={s} global_loss=1.0" for s in range(4)) + "\n"
        ours_lines = (
            "\n".join(f"[LOSS] step={s} global_loss=1.0 grad_norm=2.0 time_s=1.0" for s in range(4))
            + "\n"
        )
        result = self._run(ref_lines=ref_lines, ours_lines=ours_lines)
        self.assertEqual(result["status"], "failed")


class TestRunSuiteRouting(unittest.TestCase):
    """``run_suite`` routes on the request suite's cfg.

    A non-empty ``ours_runner`` selects the generic scripted executor (every
    stage1 gate); otherwise ``runner_kind`` must name a stage2 handler in
    ``dispatcher.RUNNER_KINDS``. Routing keys come from the request's suite
    entry, never from the suite name itself.
    """

    @staticmethod
    def _result(suite: str) -> dict:
        return {
            "schema_version": 1,
            "status": "passed",
            "suite": suite,
            "summary": "alias routed",
            "metrics": {},
            "details": {},
        }

    @staticmethod
    def _request(suite: str, cfg: object) -> dict:
        return {
            "suite": suite,
            "args": [],
            "workload_config": {"env": {}, "evals": {suite: cfg}},
        }

    def test_ours_runner_routes_to_scripted_executor(self) -> None:
        cfg = {
            "runner_kind": "forward-align",
            "ours_runner": "run_ours",
            "verdict": "align",
            "milestone": "Alias",
        }
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            with mock.patch(
                "evals.dispatcher._run_scripted_suite_kind",
                return_value=self._result("forward-alias"),
            ) as scripted:
                result = dispatcher.run_suite(
                    self._request("forward-alias", cfg),
                    repo_root,
                    repo_root / ".artifacts" / "runs" / "test",
                )
        scripted.assert_called_once()
        # Runner signature: (workload_config, repo_root, artifact_dir, suite, args).
        self.assertEqual(scripted.call_args.args[3], "forward-alias")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["suite"], "forward-alias")
        self.assertEqual(result["schema_version"], 1)

    def test_ours_runner_without_verdict_fails_fast(self) -> None:
        cfg = {"ours_runner": "run_ours"}
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "ours_runner but no verdict"):
                dispatcher.run_suite(
                    self._request("broken", cfg), repo_root, repo_root / ".artifacts"
                )

    def test_runner_kind_routes_stage2_alias(self) -> None:
        # Stage2 op-* suites still dispatch through RUNNER_KINDS by the
        # cfg's runner_kind, with the alias suite name passed through.
        cfg = {"runner_kind": "op-status"}
        fake = mock.Mock(return_value=self._result("status-alias"))
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            with mock.patch.dict(dispatcher.RUNNER_KINDS, {"op-status": fake}):
                result = dispatcher.run_suite(
                    self._request("status-alias", cfg),
                    repo_root,
                    repo_root / ".artifacts" / "runs" / "test",
                )
        fake.assert_called_once()
        self.assertEqual(fake.call_args.args[3], "status-alias")
        self.assertEqual(result["suite"], "status-alias")

    def test_missing_runner_kind_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "does not declare runner_kind"):
                dispatcher.run_suite(self._request("bare", {}), repo_root, repo_root / ".artifacts")

    def test_unknown_runner_kind_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "Unsupported runner_kind"):
                dispatcher.run_suite(
                    self._request("bogus", {"runner_kind": "no-such-kind"}),
                    repo_root,
                    repo_root / ".artifacts",
                )

    def test_unknown_suite_fails_fast(self) -> None:
        request = {"suite": "ghost", "args": [], "workload_config": {"evals": {}}}
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "Unsupported suite"):
                dispatcher.run_suite(request, repo_root, repo_root / ".artifacts")


class TestDispatcherEnvContract(unittest.TestCase):
    """Stage2 ``op-long`` keeps the legacy env_inputs subprocess contract.

    The stage1 env-threading tests that used to live here were retired with
    the handlers (see module docstring); op-long is refactor-exempt until
    step 7 and still builds its subprocess env in-process.
    """

    def test_op_long_env_is_declared_and_built_once(self) -> None:
        cfg = {
            "env_inputs": [
                "CHECKPOINT_ROOT",
                "MEGA" + "TRON_ROOT",
                "DATA_PATH",
                "MICRO_BATCH_SIZE",
                "NUM_STEPS",
                "GLOBAL_BATCH_SIZE",
                "NUM_PROCS",
                "MASTER_ADDR",
                "MASTER_PORT",
                "OP_NAMES",
                # Issue #7 lift — declared in dense_training.toml so the
                # extras injection in _run_op_long satisfies the
                # build_suite_env undeclared-extras fail-fast.
                "SEED",
                "SEQ_LENGTH",
                "GRAD_ACCUM_STEPS",
            ],
            "world_size": 2,
            "num_steps": 3,
            "global_batch_size": 6,
            "micro_batch_size": 1,
            "checkpoint_root": "/ckpt",
            "mega" + "tron_root": "/mg",
            "data_path": "/data/gsm8k_megatron/gsm8k_train_text_document",
            "gate_window_start": 0,
            "gate_window_end": 1,
            "rel_diff_threshold": 0.01,
        }
        workload_config = {
            "env": {},
            "runtime": {"distributed": {"master_addr": "localhost", "master_port": "29500"}},
            "stage2": {
                "checkpoint_root": "/stage2-ckpt",
                "mega" + "tron_root": "/stage2-mg",
                "data_path": "/stage2/gsm8k_megatron/gsm8k_train_text_document",
                "micro_batch_size": 99,
            },
            "evals": {"op-long": cfg},
        }
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            script = repo_root / "evals" / "scripts" / "op_long_ours.py"
            script.parent.mkdir(parents=True)
            script.write_text("print('mocked')\n", encoding="utf-8")
            # build_suite_env hard-requires [ref] tokenizer assets
            # (config_runtime.resolve_assets). A pre-populated dir makes
            # _ensure_tokenizer a no-op — no HF download in tests.
            tok_dir = repo_root / "tokenizer"
            tok_dir.mkdir()
            (tok_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            workload_config["ref"] = {
                "forge_tokenizer_dir": str(tok_dir),
                "tokenizer": "dummy/tokenizer",
            }

            with (
                mock.patch(
                    "evals.gate_common.run_via_ref_script",
                    return_value={
                        "ref_run": SimpleNamespace(
                            succeeded=True,
                            elapsed_s=1.0,
                            timed_out=False,
                            returncode=0,
                            stdout_path=repo_root / "ref.log",
                            # Issue #7 lift: op-long now mirrors
                            # long-train and pulls SEED / SEQ_LENGTH /
                            # GRAD_ACCUM_STEPS out of gate metadata so
                            # the ours-side trajectory is shape-locked
                            # to the ref trajectory.
                            metadata={
                                "num_steps": 3,
                                "world_size": 2,
                                "global_batch_size": 6,
                                "micro_batch_size": 1,
                                "seed": 1234,
                                "seq_length": 8,
                                "grad_accum_steps": 3,
                                "gate_window_start": 0,
                                "gate_window_end": 1,
                            },
                        ),
                        "loss_by_step": {0: 1.0},
                        "dump_dir": repo_root / "dump",
                    },
                ),
                mock.patch(
                    "evals.dispatcher_stage2.run_streaming_subprocess",
                    return_value=(
                        0,
                        "[LOSS] step=0 global_loss=1.0 grad_norm=1.0 time_s=1.0\n",
                    ),
                ) as run_mock,
            ):
                dispatcher_stage2._run_op_long(
                    workload_config,
                    repo_root,
                    repo_root / ".artifacts" / "runs" / "test",
                    op_names=["attention"],
                )

        env = run_mock.call_args.kwargs["env"]
        self.assertEqual(env["OP_NAMES"], "attention")
        self.assertEqual(env["NUM_STEPS"], "3")
        self.assertEqual(env["GLOBAL_BATCH_SIZE"], "6")
        self.assertEqual(env["MICRO_BATCH_SIZE"], "1")
        self.assertEqual(env["SEED"], "1234")
        self.assertEqual(env["SEQ_LENGTH"], "8")
        # grad_accum is recomputed from the batch shape (GBS 6 / (MBS 1 * WS 2)),
        # not lifted from ref metadata — see _suite_ours_batch_shape.
        self.assertEqual(env["GRAD_ACCUM_STEPS"], "3")
        self.assertEqual(env["CHECKPOINT_ROOT"], "/stage2-ckpt")
        self.assertNotIn("UNEXPECTED_KNOB", env)


class TestOpStatus(unittest.TestCase):
    """``op-status`` derives state from register.toml + git truth.

    Worktree/branch existence is the failed/in-progress signal; merged is
    identified by ``register.toml.default != "baseline"`` plus the absence
    of the per-op worktree/branch (i.e. cleanup ran).
    """

    def _write_registry(self, ops_dir: Path, names: list[str]) -> None:
        lines = []
        for idx, name in enumerate(names):
            lines.extend(
                [
                    f"[operators.{name}]",
                    f'name = "{name}"',
                    'category = "kernel"',
                    f"priority = {10 * (idx + 1)}",
                    "",
                ]
            )
        (ops_dir / "_registry.toml").write_text("\n".join(lines), encoding="utf-8")

    def _write_register(
        self, op_dir: Path, *, default: str = "baseline", available: list[str] | None = None
    ) -> None:
        op_dir.mkdir(parents=True, exist_ok=True)
        avail = available or ["baseline"]
        body = [
            f'env_var = "OP_{op_dir.name.upper()}"',
            f'default = "{default}"',
            "available = [" + ", ".join(f'"{v}"' for v in avail) + "]",
            "",
        ]
        (op_dir / "register.toml").write_text("\n".join(body), encoding="utf-8")

    def test_merged_when_default_advanced_and_no_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            ops_dir = repo_root / "workload" / "ops"
            ops_dir.mkdir(parents=True)
            self._write_registry(ops_dir, ["attention"])
            self._write_register(ops_dir / "attention", default="v1", available=["baseline", "v1"])

            with mock.patch.object(dispatcher_stage2, "_git_branch_exists", return_value=False):
                result = dispatcher_stage2._run_op_status({}, repo_root, repo_root / ".artifacts")

        [entry] = result["metrics"]["operators"]
        self.assertEqual(entry["status"], "merged")
        self.assertEqual(entry["default"], "v1")
        self.assertEqual(result["metrics"]["summary_counts"]["merged"], 1)
        self.assertEqual(result["metrics"]["summary_counts"]["failed"], 0)

    def test_failed_when_worktree_and_branch_remain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            ops_dir = repo_root / "workload" / "ops"
            ops_dir.mkdir(parents=True)
            self._write_registry(ops_dir, ["rope"])
            self._write_register(ops_dir / "rope", default="baseline")
            wt_path = repo_root / "ops_worktree" / "rope"
            wt_path.mkdir(parents=True)

            with mock.patch.object(dispatcher_stage2, "_git_branch_exists", return_value=True):
                result = dispatcher_stage2._run_op_status({}, repo_root, repo_root / ".artifacts")

        [entry] = result["metrics"]["operators"]
        self.assertEqual(entry["status"], "failed")
        self.assertEqual(entry["worktree"], "ops_worktree/rope")
        self.assertEqual(entry["branch"], "stage2/op/rope")
        self.assertEqual(result["metrics"]["summary_counts"]["failed"], 1)

    def test_not_started_when_default_baseline_and_no_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            ops_dir = repo_root / "workload" / "ops"
            ops_dir.mkdir(parents=True)
            self._write_registry(ops_dir, ["adam"])
            self._write_register(ops_dir / "adam", default="baseline")

            with mock.patch.object(dispatcher_stage2, "_git_branch_exists", return_value=False):
                result = dispatcher_stage2._run_op_status({}, repo_root, repo_root / ".artifacts")

        [entry] = result["metrics"]["operators"]
        self.assertEqual(entry["status"], "not_started")
        self.assertEqual(result["metrics"]["summary_counts"]["not_started"], 1)

    def test_summary_counts_all_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            ops_dir = repo_root / "workload" / "ops"
            ops_dir.mkdir(parents=True)
            self._write_registry(ops_dir, ["attention", "rope", "adam"])
            self._write_register(ops_dir / "attention", default="v1", available=["baseline", "v1"])
            self._write_register(ops_dir / "rope", default="baseline")
            self._write_register(ops_dir / "adam", default="baseline")
            (repo_root / "ops_worktree" / "rope").mkdir(parents=True)

            def _branch_exists(_repo_root: Path, branch: str) -> bool:
                return branch == "stage2/op/rope"

            with mock.patch.object(
                dispatcher_stage2, "_git_branch_exists", side_effect=_branch_exists
            ):
                result = dispatcher_stage2._run_op_status({}, repo_root, repo_root / ".artifacts")

        counts = result["metrics"]["summary_counts"]
        self.assertEqual(counts["merged"], 1)
        self.assertEqual(counts["failed"], 1)
        self.assertEqual(counts["not_started"], 1)
        self.assertIn("merged=1", result["summary"])
        self.assertIn("failed=1", result["summary"])

    def test_registry_missing_returns_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "workload" / "ops").mkdir(parents=True)
            result = dispatcher_stage2._run_op_status({}, repo_root, repo_root / ".artifacts")
        self.assertEqual(result["status"], "failed")
        self.assertIn("registry missing", result["summary"])

    # ── Suite-level status contract ────────────────────────────────
    # ``op-status`` suite-level ``status`` mirrors the failure-bearing
    # terminal counts so an upstream that reads the suite-level field
    # cannot green-light Stage 2 with ``failed`` / ``inconsistent`` ops
    # on the board. Lifecycle states (``not_started`` / ``in_progress``)
    # intentionally do NOT trip the gate — a fresh bitwise-singlecard round legitimately
    # starts in those states.

    def test_suite_status_passed_when_only_lifecycle_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            ops_dir = repo_root / "workload" / "ops"
            ops_dir.mkdir(parents=True)
            self._write_registry(ops_dir, ["adam", "rope"])
            self._write_register(ops_dir / "adam", default="baseline")
            self._write_register(ops_dir / "rope", default="baseline")
            with mock.patch.object(dispatcher_stage2, "_git_branch_exists", return_value=False):
                result = dispatcher_stage2._run_op_status({}, repo_root, repo_root / ".artifacts")
        counts = result["metrics"]["summary_counts"]
        self.assertEqual(counts["failed"], 0)
        self.assertEqual(counts["inconsistent"], 0)
        self.assertEqual(result["status"], "passed")

    def test_suite_status_passed_when_all_merged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            ops_dir = repo_root / "workload" / "ops"
            ops_dir.mkdir(parents=True)
            self._write_registry(ops_dir, ["attention"])
            self._write_register(ops_dir / "attention", default="v1", available=["baseline", "v1"])
            with mock.patch.object(dispatcher_stage2, "_git_branch_exists", return_value=False):
                result = dispatcher_stage2._run_op_status({}, repo_root, repo_root / ".artifacts")
        self.assertEqual(result["metrics"]["summary_counts"]["merged"], 1)
        self.assertEqual(result["status"], "passed")

    def test_suite_status_failed_when_any_op_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            ops_dir = repo_root / "workload" / "ops"
            ops_dir.mkdir(parents=True)
            self._write_registry(ops_dir, ["attention", "rope"])
            self._write_register(ops_dir / "attention", default="v1", available=["baseline", "v1"])
            self._write_register(ops_dir / "rope", default="baseline")
            (repo_root / "ops_worktree" / "rope").mkdir(parents=True)

            def _branch_exists(_repo_root: Path, branch: str) -> bool:
                return branch == "stage2/op/rope"

            with mock.patch.object(
                dispatcher_stage2, "_git_branch_exists", side_effect=_branch_exists
            ):
                result = dispatcher_stage2._run_op_status({}, repo_root, repo_root / ".artifacts")
        counts = result["metrics"]["summary_counts"]
        self.assertEqual(counts["merged"], 1)
        self.assertEqual(counts["failed"], 1)
        self.assertEqual(result["status"], "failed")

    def test_suite_status_failed_when_any_op_inconsistent(self) -> None:
        # ``inconsistent`` falls out of ``_status_entry_for_op``'s state
        # machine when register.toml has already advanced past
        # ``baseline`` but exactly *one* of (worktree, branch) is still
        # around — i.e. cleanup is half-done, an off-protocol state that
        # needs human / review intervention. It must trip the gate just
        # like a clean ``failed``. (Both worktree+branch present → that
        # path codes as ``failed`` instead, also gate-tripping; tested
        # above.)
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            ops_dir = repo_root / "workload" / "ops"
            ops_dir.mkdir(parents=True)
            self._write_registry(ops_dir, ["attention"])
            self._write_register(
                ops_dir / "attention",
                default="v1",
                available=["baseline", "v1"],
            )
            (repo_root / "ops_worktree" / "attention").mkdir(parents=True)
            with mock.patch.object(dispatcher_stage2, "_git_branch_exists", return_value=False):
                result = dispatcher_stage2._run_op_status({}, repo_root, repo_root / ".artifacts")
        [entry] = result["metrics"]["operators"]
        self.assertEqual(entry["status"], "inconsistent")
        self.assertEqual(result["metrics"]["summary_counts"]["inconsistent"], 1)
        self.assertEqual(result["status"], "failed")


class TestLossParser(unittest.TestCase):
    def test_duplicate_loss_step_fails_fast(self) -> None:
        from evals._common import parse_loss_lines_to_dict

        output = "\n".join(
            [
                "[LOSS] step=1 global_loss=2.0 grad_norm=3.0 time_s=4.0",
                "[LOSS] step=1 global_loss=5.0 grad_norm=6.0 time_s=7.0",
            ]
        )
        with self.assertRaises(ValueError):
            parse_loss_lines_to_dict(output)

    def test_parse_loss_lines_rejects_non_finite_loss(self) -> None:
        from evals._common import parse_loss_lines

        output = "[LOSS] step=1 global_loss=nan grad_norm=3.0 time_s=4.0"
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            parse_loss_lines(output)

    def test_parse_loss_lines_to_dict_rejects_infinite_loss(self) -> None:
        from evals._common import parse_loss_lines_to_dict

        for value in ("inf", "-inf"):
            with self.subTest(value=value):
                output = f"[LOSS] step=1 global_loss={value} grad_norm=3.0 time_s=4.0"
                with self.assertRaisesRegex(ValueError, "Non-finite"):
                    parse_loss_lines_to_dict(output)

    def test_window_loss_diff_metrics(self) -> None:
        from evals._common import missing_window_steps, window_loss_diff_metrics

        metrics = window_loss_diff_metrics(
            {0: 10.0, 1: 20.0},
            {0: 11.0, 1: 18.0},
            range(0, 2),
        )
        self.assertEqual(metrics["compared_steps"], 2)
        self.assertAlmostEqual(metrics["mean_rel_diff"], 0.1)
        self.assertAlmostEqual(metrics["max_rel_diff"], 0.1)
        self.assertAlmostEqual(metrics["signed_mean"], -0.5)
        # signed_mean_rel = mean(signed) / mean(|baseline|)
        # = (1 + -2)/2  /  (10 + 20)/2  =  -0.5 / 15  =  -1/30
        self.assertAlmostEqual(metrics["signed_mean_rel"], -1.0 / 30.0)
        self.assertEqual(len(metrics["step_diffs"]), 2)
        self.assertEqual(missing_window_steps({0: 1.0, 2: 3.0}, range(0, 3)), [1])

    def test_window_loss_diff_metrics_rejects_zero_baseline(self) -> None:
        from evals._common import window_loss_diff_metrics

        with self.assertRaisesRegex(ValueError, "baseline loss is 0"):
            window_loss_diff_metrics({0: 0.0}, {0: 1.0}, range(0, 1))


class TestGateEnvFailFast(unittest.TestCase):
    """loss-gate refuses to judge without its declared threshold.

    Ported from the legacy dispatcher fail-fast (which raised on a missing
    DATA_PATH env input): the verdict reads ``max_avg_relative_loss_diff``
    before anything else, so a registry entry that lost the key raises at
    the runner boundary instead of silently passing.

    ``test_resume_gate_composes_run_config_without_injecting_shape`` was
    retired: ``gate_product.compose_run_config`` was deleted with the
    handler and the resume sh reads the rendered product directly.
    """

    def test_loss_gate_requires_max_avg_relative_loss_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            run_dir = repo_root / ".artifacts" / "runs" / "test"
            run_dir.mkdir(parents=True)
            wc = {"evals": {"loss-gate-200": {"milestone": "loss-guard"}}}
            with self.assertRaises(KeyError):
                loss_gate.run(
                    suite_key="loss-gate-200",
                    run_dir=run_dir,
                    repo_root=repo_root,
                    workload_config=wc,
                )

    def test_loss_gate_with_threshold_proceeds_to_shape_check(self) -> None:
        # Control: with the threshold declared, the same fixture gets past
        # the fail-fast and fails structurally on the missing ref product.
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            run_dir = repo_root / ".artifacts" / "runs" / "test"
            run_dir.mkdir(parents=True)
            wc = {
                "evals": {
                    "loss-gate-200": {
                        "milestone": "loss-guard",
                        "max_avg_relative_loss_diff": 0.01,
                    }
                }
            }
            result = loss_gate.run(
                suite_key="loss-gate-200",
                run_dir=run_dir,
                repo_root=repo_root,
                workload_config=wc,
            )
        self.assertEqual(result["status"], "failed")
        self.assertIn("gate shape unavailable", result["summary"])


class TestM1CaptureDiffTimeout(unittest.TestCase):
    """The align verdict must convert ours-side timeouts into failed runs.

    Timeout *mechanics* (forwarding ``timeout_s`` to the subprocess) moved
    into the generic scripted executor, which records the outcome in
    ``ours/status.json``; the verdict's contract is that a recorded
    ``timed_out`` reads as ``status=failed`` with a summary naming the
    timeout, never as a hang or a silent pass.
    """

    def test_timeout_returns_failed_with_timeout_summary(self) -> None:
        result = _run_align_verdict(
            ours_returncode=-1,
            ours_timed_out=True,
            ours_timeout_s=11,
            ours_records=None,  # timed-out candidate never writes its dump
            ours_log="partial stdout before hang",
        )
        self.assertEqual(result["status"], "failed")
        self.assertIn("timed out after 11s", result["summary"])
        self.assertEqual(result["metrics"]["checks_total"], 0)
        self.assertEqual(result["metrics"]["bitwise_match"], 0)
        self.assertIn("partial stdout", result["details"]["output_tail"])


class TestCleanExitVerdictContract(unittest.TestCase):
    """A gate PASSes iff the run exits 0 AND its artifact is valid.

    This pins the post-``os._exit`` contract: every gate subprocess must
    return normally with ``returncode == 0`` and produce a correct
    artifact. A crash during interpreter shutdown (the classic
    NCCL/flash_attn/pyarrow destructor-order race, surfacing as SIGABRT
    ``returncode == -6``) is a FAIL even when the gate-visible artifact
    was already flushed — the engine must tear down cleanly instead of
    bypassing the finalizer with ``os._exit(0)``.

    Two enforcement points are pinned:

    * **M1** (``evals.verdicts.align``): the ref capture precondition is
      "dump present AND returncode == 0"; the candidate side flows its
      returncode into ``capture_gate_outcome``.
    * **M2/M3/M4** (``evals.verdicts.bitwise``): ``script_succeeded``
      gates on ``returncode == 0``; regression guard so a future
      reintroduction of ``os._exit`` cannot silently mask a crashed run
      behind a perfect loss trajectory.
    """

    def test_m1_clean_exit_with_valid_dumps_passes(self) -> None:
        result = _run_align_verdict()
        self.assertEqual(result["status"], "passed")

    def test_m1_ref_crash_with_valid_dump_fails(self) -> None:
        # SIGABRT during ref interpreter shutdown: dump already flushed,
        # losses correct, but the process aborted. Under the clean-exit
        # contract this is a FAIL — the ref must return 0, not os._exit.
        result = _run_align_verdict(ref_returncode=-6)
        self.assertEqual(result["status"], "failed")

    def test_m1_candidate_crash_with_valid_dump_fails(self) -> None:
        result = _run_align_verdict(ours_returncode=-6)
        self.assertEqual(result["status"], "failed")

    def test_m1_clean_exit_but_candidate_dump_missing_fails(self) -> None:
        result = _run_align_verdict(ours_records=None)
        self.assertEqual(result["status"], "failed")

    def test_backward_milestone_gates_bwd_and_is_ref_authoritative(self) -> None:
        # The comparison family is selected by the milestone VALUE, not the
        # gate name: alignment.backward diffs grad.* AND bwd.* with the ref
        # authoritative, so a candidate omitting a bwd.* key the ref
        # captured must FAIL even though every overlapping key matches.
        grad = {"hash": "a" * 32, "shape": [4], "dtype": "float32"}
        bwd = {"hash": "b" * 32, "shape": [4], "dtype": "float32"}
        result = _run_align_verdict(
            suite="backward-align",
            milestone="alignment.backward",
            ref_records={"grad.w": grad, "bwd.x": bwd},
            ours_records={"grad.w": grad},
        )
        self.assertEqual(result["status"], "failed")
        # Same dumps under alignment.forward compare fwd.* only → no
        # overlapping in-family keys → anti-silent-pass failure, not a pass.
        result_fwd = _run_align_verdict(
            suite="forward-align",
            milestone="alignment.forward",
            ref_records={"grad.w": grad, "bwd.x": bwd},
            ours_records={"grad.w": grad},
        )
        self.assertEqual(result_fwd["status"], "failed")
        self.assertIn("no overlapping", result_fwd["summary"])

    def _run_bitwise(self, *, returncode: int) -> dict:
        lines = (
            "\n".join(f"[LOSS] step={s} global_loss=1.0 grad_norm=1.0 time_s=1.0" for s in range(8))
            + "\n"
        )
        return _run_bitwise_verdict(ref_lines=lines, ours_lines=lines, ours_returncode=returncode)

    def test_bitwise_clean_exit_with_matching_trajectory_passes(self) -> None:
        result = self._run_bitwise(returncode=0)
        self.assertEqual(result["status"], "passed")

    def test_bitwise_crash_with_matching_trajectory_fails(self) -> None:
        # Perfect per-step loss match but the process aborted (-6). A
        # crashed run must not pass on artifact correctness alone.
        result = self._run_bitwise(returncode=-6)
        self.assertEqual(result["status"], "failed")


class TestThresholdDirectionsAreConsistent(unittest.TestCase):
    """Cross-suite invariant on the relative-loss gate boundary.

    The verbal contract is "≤ <pct>% relative loss diff". All three
    cross-suite implementations encode it as a *strict* less-than: a
    trajectory landing exactly on the threshold must FAIL the gate.
    Pin every site as a static source-presence assertion so a refactor
    cannot silently swap one direction to ``<=`` and produce a PASS on
    ``op-long`` that the sister suites would FAIL on the same number.

    Sites: ``op-long`` still lives in ``dispatcher_stage2._run_op_long``
    (stage2 is thin-dispatcher exempt); the ``long-train`` / ``loss-gate``
    judgments moved to their verdict modules under ``evals/verdicts/``.
    """

    _EVALS = _REPO_ROOT / "evals"
    _DISPATCHER_STAGE2 = _EVALS / "dispatcher_stage2.py"
    _VERDICTS = _EVALS / "verdicts"

    # (source file, function to slice or None for whole file, required regex)
    _ALLOWED_PATTERNS: tuple[tuple[Path, str | None, str], ...] = (
        (_DISPATCHER_STAGE2, "_run_op_long", r"\bmean_rel\s*<\s*rel_threshold\b"),
        (
            _VERDICTS / "long_train.py",
            None,
            r"\bpointwise_mean_rel\s*<\s*loss_rel_threshold\b",
        ),
        (
            _VERDICTS / "loss_gate.py",
            None,
            r"\bavg_rel_diff\s*<\s*max_avg_rel\b",
        ),
    )

    def _function_body(self, source: str, fn_name: str) -> str:
        # Helper functions are all top-level ``def``s. Slice from
        # ``def fn_name(`` to the next top-level ``def `` / ``class ``
        # (no leading whitespace) — sufficient for a static substring check.
        start_marker = f"\ndef {fn_name}("
        idx = source.find(start_marker)
        self.assertGreater(idx, 0, f"function {fn_name} not found")
        next_def = re.search(r"\n(?:def |class )", source[idx + 1 :])
        if next_def is None:
            return source[idx:]
        return source[idx : idx + 1 + next_def.start()]

    def test_relative_loss_gates_use_strict_less_than(self) -> None:
        for path, fn_name, pat in self._ALLOWED_PATTERNS:
            with self.subTest(source=path.name, fn=fn_name):
                text = path.read_text(encoding="utf-8")
                body = text if fn_name is None else self._function_body(text, fn_name)
                self.assertRegex(
                    body,
                    pat,
                    f"{path.name}:{fn_name or '<module>'} does not encode the "
                    "relative-loss gate as a strict ``<`` comparison. The "
                    "cross-suite invariant is that all three sites use ``<`` "
                    "so a value exactly on the threshold FAILs uniformly.",
                )

    def test_no_relative_loss_gate_uses_le(self) -> None:
        # Defence in depth: explicitly forbid ``<=`` against any of the
        # three threshold names anywhere in the dispatcher OR the verdict
        # modules. Catches a future refactor that introduces a new wrapper
        # using ``<=``.
        forbidden_pairs = (
            (r"mean_rel\s*<=\s*rel_threshold", "op-long"),
            (r"pointwise_mean_rel\s*<=\s*loss_rel_threshold", "long-train"),
            (r"avg_rel_diff\s*<=\s*max_avg_rel", "loss-gate"),
        )
        sources = [self._DISPATCHER_STAGE2, *sorted(self._VERDICTS.glob("*.py"))]
        for src_path in sources:
            text = src_path.read_text(encoding="utf-8")
            for pat, label in forbidden_pairs:
                with self.subTest(source=src_path.name, suite=label):
                    self.assertNotRegex(
                        text,
                        pat,
                        f"{label} relative-loss gate uses ``<=`` in "
                        f"{src_path.name}; the cross-suite invariant pinned "
                        "by this test requires a strict ``<``.",
                    )


class TestMfuUnitConversion(unittest.TestCase):
    """Wire-format vs gate-target unit invariant for ``mfu_e2e_standard``.

    Single unit invariant: ``mfu_e2e_standard`` on every ``[LOSS]`` line
    is **already in 0–100 percentage scale** (``flops_per_step /
    (step_time * peak_total) * 100``). The ref script
    (``train_pure_mup_mtp.py``) and the ours-side engine both emit it
    that way; the gate products declare ``mfu_e2e_target`` on the same
    0–100 scale (e.g. ``14.5`` for ``perf-bitwise``, ``45.0`` for
    ``long-train``). The verdicts therefore compare and report the wire
    value directly with **no re-scaling** — multiplying by 100 a second
    time inflated a real ~17% MFU into ``1742%`` (see
    ``test_perf_bitwise_does_not_rescale_percent``).
    """

    def _run_long_train_with_mfu_steps(self, loss_lines: str, *, mfu_target: float | None) -> dict:
        """Drive ``evals.verdicts.long_train`` against canned artifacts.

        ``mfu_target=None`` omits the key entirely — the "no dev-side MFU
        floor" configuration (long-train gates on loss only)."""
        cfg: dict = {"loss_rel_threshold": 0.01, "warmup_steps": 0}
        if mfu_target is not None:
            cfg["mfu_e2e_target"] = mfu_target
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            run_dir = repo_root / ".artifacts" / "runs" / "test"
            _write_ref_product(
                repo_root,
                "long-train",
                {"world_size": 8, "num_steps": 1, "gate_window": [1, 2]},
            )
            _write_side_status(run_dir / "ref", returncode=0)
            _write_ref_loss_dump(
                run_dir, "[LOSS] step=1 global_loss=1.0 grad_norm=0.1 time_s=1.0\n"
            )
            _write_side_status(run_dir / "ours", returncode=0)
            _write_ours_log(run_dir, loss_lines)
            return long_train.run(
                suite_key="long-train",
                run_dir=run_dir,
                repo_root=repo_root,
                workload_config={"evals": {"long-train": cfg}},
            )

    def test_long_train_reports_mfu_percent_without_rescaling(self) -> None:
        # Wire value 46.0 (already a percent = 46% MFU) MUST clear a 45%
        # target and be reported as-is. The verdict must NOT re-scale
        # by 100 (which would yield 4600%).
        result = self._run_long_train_with_mfu_steps(
            "[LOSS] step=1 global_loss=1.0 grad_norm=0.1 time_s=0.1 mfu_e2e_standard=46.0\n",
            mfu_target=45.0,
        )
        self.assertEqual(result["status"], "passed", msg=result.get("summary"))
        self.assertAlmostEqual(
            result["metrics"]["avg_mfu_e2e_standard"],
            46.0,
            places=5,
            msg=f"avg_mfu_e2e_standard re-scaled, not raw percent: {result['metrics']}",
        )

    def test_long_train_fails_when_mfu_below_target(self) -> None:
        # 18.0 percent (18% MFU) MUST fail the 45% target.
        result = self._run_long_train_with_mfu_steps(
            "[LOSS] step=1 global_loss=1.0 grad_norm=0.1 time_s=0.1 mfu_e2e_standard=18.0\n",
            mfu_target=45.0,
        )
        self.assertEqual(result["status"], "failed", msg=result.get("summary"))
        self.assertAlmostEqual(
            result["metrics"]["avg_mfu_e2e_standard"],
            18.0,
            places=5,
            msg=f"avg_mfu_e2e_standard not raw percent: {result['metrics']}",
        )

    def test_long_train_without_mfu_target_gates_on_loss_only(self) -> None:
        # No ``mfu_e2e_target`` in the suite cfg (the long-horizon
        # "review-side MFU floor" configuration): a low MFU must NOT fail
        # the gate, the summary must report the measured MFU with no
        # floor comparison, and mfu_target / mfu_pass stay None so the
        # dev agent cannot read a bar out of the gate output.
        result = self._run_long_train_with_mfu_steps(
            "[LOSS] step=1 global_loss=1.0 grad_norm=0.1 time_s=0.1 mfu_e2e_standard=18.0\n",
            mfu_target=None,
        )
        self.assertEqual(result["status"], "passed", msg=result.get("summary"))
        self.assertAlmostEqual(result["metrics"]["avg_mfu_e2e_standard"], 18.0, places=5)
        self.assertIsNone(result["metrics"]["mfu_target"])
        self.assertIsNone(result["metrics"]["mfu_pass"])
        self.assertIn("MFU 18.0%", result["summary"])
        self.assertNotIn("≥", result["summary"].split("MFU")[-1])
        self.assertNotIn("<", result["summary"].split("MFU")[-1])

    def test_perf_bitwise_does_not_rescale_percent(self) -> None:
        # Regression for the 1742% bug: the engine emits a steady-state
        # ~17.4% MFU already in percent scale; the perf-bitwise gate
        # (bitwise verdict, Gate 2) must report ~17.4%, not 17.4 * 100.
        result = _run_bitwise_verdict(
            suite="perf-bitwise",
            shape={"world_size": 8, "num_steps": 1, "gate_window": [1, 2]},
            cfg={
                "milestone": "bitwise-perf",
                "loss_abs_threshold": 0,
                "grad_norm_abs_threshold": 0,
                "mfu_e2e_target": 14.5,
                "warmup_steps": 0,
            },
            ref_lines="[LOSS] step=1 global_loss=1.0 grad_norm=0.1 time_s=1.0\n",
            ours_lines="[LOSS] step=1 global_loss=1.0 grad_norm=0.1 time_s=3.7 mfu_e2e_standard=17.4\n",
        )
        self.assertEqual(result["status"], "passed", msg=result.get("summary"))
        self.assertAlmostEqual(
            result["metrics"]["avg_mfu_e2e_standard"],
            17.4,
            places=5,
            msg=f"avg_mfu_e2e_standard re-scaled into >100%: {result['metrics']}",
        )


if __name__ == "__main__":
    unittest.main()
