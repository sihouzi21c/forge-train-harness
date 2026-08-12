"""Verify the framework guard boundary and its own hygiene.

The ``test_no_violations`` full-repo scan is intentionally excluded because
it scans workload/ sources that are outside the harness optimization scope.
The remaining tests lock the *guard's own* invariants: no orphan allowlist
entries, no deleted-but-referenced tools, and (D15) Python files inside
``DOCUMENTATION_ALLOWLIST_DIRS`` still get scanned for banned imports.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from harness import framework_guard  # noqa: E402


class TestAllowlistHygiene(unittest.TestCase):
    def test_harness_guard_cli_passes_current_repo(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "harness.cli", "run", "guard", "--json"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_harness_guard_suite_runs_path_guard(self) -> None:
        source = (REPO_ROOT / "harness" / "app.py").read_text(encoding="utf-8")
        self.assertIn("--with-path-guard", source)

    def test_tools_dir_only_holds_ref_as_gate_helpers(self) -> None:
        # ``tools/`` is the canonical home of the ref-as-gate runner
        # (see README.md, "SSOT: Ref Script as Gate"). Anything
        # outside that contract is a structural violation — gates /
        # suites belong in ``evals/``, framework code in ``harness/``,
        # the standard capture hook in ``evals/harness_hook/``.
        tools_dir = REPO_ROOT / "tools"
        if not tools_dir.exists():
            self.skipTest("tools/ not present in this repo")
        allowed = {
            "__init__.py",
            "ref_script_runner.py",
            "stage2_op_router.py",
            "agent_loop_config.py",
            "cursor_max_mode.py",
            "stage2_config.py",
            "profile_step.py",
            "capture_profile_batch.py",
            # Deterministic profile-snapshot renderer for perf-bitwise / long-train
            # iteration suites. Stage-1 profile contract: launch_dp
            # wraps rank 0 with ``nsys profile`` when
            # ``FORGE_NSYS_RANK0_OUTPUT`` is set; this module shells
            # out to ``nsys stats`` and renders summary + JSON.
            "profile_render.py",
            # Unified-agent-log spawn helpers: agent-loop.sh shells out
            # to these so every dev/review/stage2-subagent registers as
            # a Session under web-agents/<id>/ and the wrapper itself
            # has a typed loop_event NDJSON channel.
            "spawn_managed_agent.py",
            "loop_wrapper_init.py",
            "loop_wrapper_event.py",
            "codex_config.py",
            # MFU SSOT writer: called by evals.runner after every gate
            # run to persist the per-loop history + cross-loop best
            # that the web Loop Progress badge hydrates from.
            "mfu_record.py",
            # Ref-as-gate helper: one-shot canonical_state_fp32 bootstrap
            # via the ref bridge so the bitwise gates share an init.
            "bootstrap_canonical.py",
            # Ref-as-gate helper: bitwise self-consistency check for the
            # streaming HF dataloader.
            "hf_data_bitwise_check.py",
            # Agent-loop control: exclusive cctl devspace lease per loop.
            "lease.py",
            # Agent-loop control: background production corpus prefetch into the
            # workspace-relative forge_data_dir.
            "prefetch_data.py",
            # Gate-config rendering: the render_gate_configs.py SSOT renderer.
            "render_gate_configs.py",
            # Remote-execution control plane (on-demand cctl GPU jobs +
            # devspace gateway): submit/poll a GPU job, share cctl plumbing,
            # resolve the remote workspace path.
            "gpu_job.py",
            "cctl_common.py",
            "remote_workspace.py",
            # MFU SSOT backfill: replays gate run events into mfu_history.
            "mfu_backfill_events.py",
            # Agent-loop control: trajectory fork ("轨迹续跑") — builds a
            # new loop instance dir anchored at a passed milestone.
            "fork_loop.py",
            # Agent-loop control: offline per-loop report renderer over
            # the wrapper's loop_event NDJSON stream.
            "loop_report.py",
            # Review-side deterministic checker for the long-horizon
            # throughput bar (elastic mfu_e2e_target relaxation).
            "mfu_elastic_check.py",
            # Thin-dispatcher step 1: rendered gate product → shell-env
            # projector consumed by side sh scripts.
            "product_env.py",
            # Deploy-source resolver: maps [ref].backend to the engine /
            # vendored-submodule roots the launchers consume.
            "resolve_deploy.py",
            # config/train/*.toml recipe → shell-env projector for the
            # remote PyTorchJob launch scripts.
            "train_recipe_to_env.py",
        }
        actual = {p.name for p in tools_dir.iterdir() if p.suffix == ".py"}
        unexpected = actual - allowed
        self.assertEqual(
            unexpected,
            set(),
            f"tools/ contains unexpected modules: {sorted(unexpected)}. "
            "Only ref-as-gate helpers, Stage 2 orchestration helpers, "
            "agent-loop control helpers, and agent-writable profiling tools belong here.",
        )

    def test_allowlist_has_no_orphans(self) -> None:
        # D16: every file-path allowlist entry must still exist on disk,
        # otherwise the guard silently weakens over time.
        missing = [rel for rel in framework_guard.ALLOWLIST if not (REPO_ROOT / rel).exists()]
        self.assertEqual(missing, [], f"Orphan ALLOWLIST entries: {missing}")

    def test_tools_package_resolves_to_repo_local_helpers(self) -> None:
        tools_pkg = importlib.import_module("tools")
        self.assertEqual(Path(tools_pkg.__file__).resolve(), REPO_ROOT / "tools" / "__init__.py")

    def test_path_guard_doc_matches_per_op_only_policy(self) -> None:
        source = (REPO_ROOT / "harness" / "framework_guard.py").read_text(encoding="utf-8")
        self.assertIn("operator's own", source)
        self.assertIn("directory", source)
        self.assertNotIn("tools/stage2_op_router.py", source)


class TestIgnoredDirPrune(unittest.TestCase):
    """The full-repo walk must skip build/cache/VCS artefact roots.

    ``iter_candidate_files`` uses ``os.walk`` with in-place
    ``dirnames`` mutation to prune ``.git`` / ``__pycache__`` /
    ``.artifacts`` / ``.pytest_cache`` / ``.ruff_cache`` / ``.venv``
    at the directory level; a plain ``Path.rglob("*")`` would descend
    into them on every invocation and balloon the walk by orders of
    magnitude on a working tree with caches populated.
    """

    def test_walk_text_candidates_prunes_ignored_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Source files that must be visible.
            (root / "src").mkdir()
            visible_py = root / "src" / "visible.py"
            visible_py.write_text("# ok\n", encoding="utf-8")
            visible_md = root / "README.md"
            visible_md.write_text("# ok\n", encoding="utf-8")

            # Ignored directories — each gets a Python file that would
            # trip the banned-keyword scan if it were ever read.
            for ignored in framework_guard.IGNORED_DIR_NAMES:
                d = root / ignored
                d.mkdir()
                (d / "probe.py").write_text("import megatron.core\n", encoding="utf-8")

            # Also assert nesting: a nested ignored dir is pruned even
            # when buried under regular source.
            nested = root / "src" / "__pycache__"
            nested.mkdir()
            (nested / "probe.py").write_text("import deepspeed\n", encoding="utf-8")

            files = sorted(framework_guard._walk_text_candidates(root))

        rel = {p.relative_to(root.resolve()).as_posix() for p in files}
        self.assertEqual(rel, {"src/visible.py", "README.md"})

    def test_full_repo_scan_never_walks_artifacts_or_caches(self) -> None:
        # Sanity-check the whole-repo scan against the live tree: no
        # candidate may live under a pruned directory, regardless of
        # extension. (``__pycache__`` is the most likely offender if
        # someone later swaps the walker back to ``rglob``.)
        candidates = framework_guard.iter_candidate_files([])
        offenders = [
            str(p)
            for p in candidates
            if any(part in framework_guard.IGNORED_DIR_NAMES for part in p.parts)
        ]
        self.assertEqual(offenders, [])

    def test_candidate_scan_excludes_versioned_submodules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vendored = root / "harness" / "third_party" / "megatron" / "v15"
            vendored.mkdir(parents=True)
            (vendored / "README.md").write_text("Megatron-LM\n", encoding="utf-8")
            visible = root / "harness" / "README.md"
            visible.write_text("# ok\n", encoding="utf-8")

            files = sorted(framework_guard._walk_text_candidates(root))

        rel = {p.relative_to(root.resolve()).as_posix() for p in files}
        self.assertEqual(rel, {"harness/README.md"})

    def test_symlink_into_ignored_tree_is_not_surfaced(self) -> None:
        # A workspace lays out ``config/<gate>.toml`` as symlinks whose
        # targets live under ``.artifacts`` (the forge_train/<id> layout).
        # ``_walk_text_candidates`` resolves each candidate, so a symlink
        # whose target is inside a pruned tree must NOT be surfaced —
        # otherwise the banned-keyword scan reads a file it is meant to
        # skip and the full-repo offender check false-positives.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".artifacts").mkdir()
            target = root / ".artifacts" / "real.toml"
            target.write_text("x = 1\n", encoding="utf-8")
            (root / "config").mkdir()
            (root / "config" / "link.toml").symlink_to(target)

            files = sorted(framework_guard._walk_text_candidates(root))

        offenders = [
            str(p)
            for p in files
            if any(part in framework_guard.IGNORED_DIR_NAMES for part in p.parts)
        ]
        self.assertEqual(offenders, [])


class TestRepoWriteSurfaceGuard(unittest.TestCase):
    def _path_violations(
        self, branch: str, staged: list[str]
    ) -> list[framework_guard.GuardViolation]:
        calls = [
            SimpleNamespace(returncode=0, stdout=f"{branch}\n"),
            SimpleNamespace(returncode=0, stdout="\n".join(staged) + "\n"),
        ]
        with mock.patch("subprocess.run", side_effect=calls):
            return framework_guard.check_path_isolation()

    def test_non_op_branch_rejects_ref_baseline_changes(self) -> None:
        violations = self._path_violations(
            "zz/one_harness_run_all",
            ["ref/reference/train_minicpm4_0.5b_gsm8k.sh"],
        )
        self.assertEqual(len(violations), 1)
        self.assertIn("read-only path", violations[0].reason)

    def test_non_op_branch_allows_training_engine_and_control_plane(self) -> None:
        violations = self._path_violations(
            "zz/one_harness_run_all",
            [
                "workload/src/training_engine_tensor/forward.py",
                "workload/src/training_engine_tensor/train_loop.py",
                "workload/ops/attention/kernel.py",
                "workload/notes/perf_log.md",
                "harness/app.py",
                "evals/dispatcher.py",
                "tools/stage2_op_router.py",
                "prompt/develop_prompt/megatron/stage1/overview.md",
                "prompt/develop_prompt/torch/stage1/overview.md",
                "config/eval/dense_training/dense_training.toml",
                "harness/tests/test_framework_guard.py",
                "tools/stage2_config.py",
                "web/server.py",
                "web/static/index.html",
                ".gitmodules",
            ],
        )
        self.assertEqual(violations, [])


class TestPythonNotAllowlistedByDir(unittest.TestCase):
    """D15: dir-level allowlist must not silently permit banned Python imports."""

    def test_banned_import_in_doc_allowlisted_dir_is_flagged(self) -> None:
        # Drop a probe inside ``ref/`` (a DOCUMENTATION_ALLOWLIST_DIRS entry)
        # and confirm the guard still flags banned Python imports there.
        probe = REPO_ROOT / "ref" / "_guard_probe_tmp.py"
        # Build the banned keyword at runtime so this test file itself does
        # not trigger a guard violation when the repo is scanned.
        banned = "meg" + "atron.core"
        probe.write_text(f"import {banned}  # probe\n", encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "harness.framework_guard"],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(
                result.returncode,
                0,
                "guard accepted a banned Python import inside a doc-allowlisted dir",
            )
            self.assertIn("_guard_probe_tmp.py", result.stderr)
        finally:
            probe.unlink(missing_ok=True)


class TestAutogradEntryPointBan(unittest.TestCase):
    """Stage 1 forbids relying on the autograd engine for backward.

    ``torch.autograd`` already covers direct submodule imports, but
    the autograd engine is reachable via two Tensor-method paths that
    do NOT mention the submodule name:

      1. ``tensor.backward()``                       — kicks off backward
      2. ``requires_grad=True`` / ``requires_grad_(True)`` — opts into
         the autograd graph at tensor construction

    The keyword scanner must flag these literal tokens so the
    static-graph requirement is enforced at the framework_guard layer
    (otherwise the rule is only visible in prose and the agent can
    silently call ``.backward()`` to short-circuit the entire spec).
    """

    def test_dot_backward_call_is_flagged(self) -> None:
        reason = framework_guard.match_banned_reference("loss.backward()")
        self.assertIsNotNone(reason)
        self.assertIn(".backward(", reason)
        self.assertIn("autograd entry point", reason)

    def test_dot_backward_with_args_is_flagged(self) -> None:
        reason = framework_guard.match_banned_reference("obj.backward(retain_graph=True)")
        self.assertIsNotNone(reason)
        self.assertIn(".backward(", reason)

    def test_requires_grad_true_kwarg_is_flagged(self) -> None:
        reason = framework_guard.match_banned_reference(
            "t = torch.empty(shape, requires_grad=True)"
        )
        self.assertIsNotNone(reason)
        self.assertIn("requires_grad=True", reason)

    def test_requires_grad_underscore_setter_is_flagged(self) -> None:
        reason = framework_guard.match_banned_reference("t.requires_grad_(True)")
        self.assertIsNotNone(reason)
        self.assertIn("requires_grad_(True)", reason)

    def test_requires_grad_false_is_not_flagged(self) -> None:
        # Explicit opt-out is legitimate (some debug paths need to
        # temporarily detach without going through the autograd graph).
        self.assertIsNone(
            framework_guard.match_banned_reference("t = torch.empty(shape, requires_grad=False)")
        )

    def test_backwards_typo_is_not_flagged(self) -> None:
        # ``.backwards`` is not an autograd entry; only ``.backward(``
        # with the open-paren should match.
        self.assertIsNone(framework_guard.match_banned_reference("self.go_backwards()"))

    def test_word_backward_in_comment_is_not_flagged(self) -> None:
        # We don't ban the bare word "backward" (the spec talks about
        # it constantly) — only the call form ``.backward(``.
        self.assertIsNone(
            framework_guard.match_banned_reference(
                "# Compute backward pass via aten.linear_backward()"
            )
        )

    def test_aten_op_backward_is_not_flagged(self) -> None:
        # The blessed escape hatch: calling the ATen ``*_backward`` ops
        # directly does NOT trigger the autograd engine and is the
        # intended path for the static-graph backward.
        self.assertIsNone(
            framework_guard.match_banned_reference(
                "grad_x = torch.ops.aten.linear_backward(go, x, w, [True, True, False])"
            )
        )

    def test_full_scan_flags_autograd_entry_in_probe_file(self) -> None:
        probe = REPO_ROOT / "harness" / "_autograd_guard_probe_tmp.py"
        # ``harness/`` is not a documentation-allowlist dir, so the
        # probe goes through the full keyword scan path.
        probe.write_text(
            "import torch\nt = torch.empty(4, requires_grad=True)\n(t * 2).sum().backward()\n",
            encoding="utf-8",
        )
        try:
            result = subprocess.run(
                [sys.executable, "-m", "harness.framework_guard"],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(
                result.returncode,
                0,
                "guard accepted autograd entry points",
            )
            self.assertIn("_autograd_guard_probe_tmp.py", result.stderr)
            self.assertIn(".backward(", result.stderr)
            self.assertIn("requires_grad=True", result.stderr)
        finally:
            probe.unlink(missing_ok=True)


class TestDotBackwardBannedUnconditionally(unittest.TestCase):
    """``.backward(`` is banned UNCONDITIONALLY — there is no
    "this file defines its own ``def backward``" carve-out. Such an
    exemption is trivially exploitable: an agent could run a real
    ``loss.backward()`` autograd pass and silence the scanner just by
    parking a ``def backward(self, …)`` method in the same file. The
    engine's hand-coded backward must therefore avoid the ``.backward(``
    literal (e.g. name it ``def bwd(self, …)``).
    """

    def test_match_bans_dot_backward_even_with_user_def_in_file(self) -> None:
        # The presence of a ``def backward`` in the file is irrelevant —
        # ``.backward(`` is always flagged.
        reason = framework_guard.match_banned_reference("grad_in = self.backward(grad_out)")
        self.assertIsNotNone(reason)
        self.assertIn("autograd entry point", reason)

    def test_match_still_bans_requires_grad(self) -> None:
        for snippet in (
            "x = torch.empty(4, requires_grad=True)",
            "x.requires_grad_(True)",
        ):
            self.assertIsNotNone(
                framework_guard.match_banned_reference(snippet),
                snippet,
            )

    def test_match_bans_dot_backward(self) -> None:
        reason = framework_guard.match_banned_reference("layer.backward(grad_out)")
        self.assertIsNotNone(reason)
        self.assertIn("autograd entry point", reason)

    def test_full_scan_flags_dot_backward_in_user_backward_file(self) -> None:
        # The exploit the carve-out enabled: a real autograd call hidden in
        # a file that also defines ``def backward`` MUST still be flagged.
        with tempfile.TemporaryDirectory() as td:
            probe = Path(td) / "engine_layer.py"
            probe.write_text(
                "import torch\n"
                "class L:\n"
                "    def backward(self, grad_out):\n"
                "        return grad_out\n"
                "def driver(layer, grad):\n"
                "    return layer.backward(grad)\n",
                encoding="utf-8",
            )
            violations = framework_guard.scan_file(probe)
            self.assertTrue(
                any(".backward(" in v.reason for v in violations),
                f"`.backward(` must be flagged even with a user `def backward`: {violations}",
            )

    def test_full_scan_still_flags_requires_grad_in_user_backward_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            probe = Path(td) / "engine_layer.py"
            probe.write_text(
                "import torch\n"
                "class L:\n"
                "    def backward(self, grad_out):\n"
                "        return grad_out\n"
                "x = torch.empty(4, requires_grad=True)\n",
                encoding="utf-8",
            )
            violations = framework_guard.scan_file(probe)
            self.assertTrue(
                any("requires_grad=True" in v.reason for v in violations),
                f"requires_grad=True must still be flagged: {violations}",
            )

    def test_full_scan_flags_dot_backward_without_user_def(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            probe = Path(td) / "naive.py"
            probe.write_text(
                "import torch\nloss = torch.tensor(1.0)\nloss.backward()\n",
                encoding="utf-8",
            )
            violations = framework_guard.scan_file(probe)
            self.assertTrue(
                any(".backward(" in v.reason for v in violations),
                f"`.backward(` must be flagged: {violations}",
            )


class TestFullyBannedModuleReferenceMatching(unittest.TestCase):
    """``megatron`` / ``deepspeed`` are banned as *framework imports*, not
    as substrings. The data-format identifiers ``megatron_binary`` /
    ``MegatronBinaryDataloader`` / ``_is_megatron_binary`` name the
    Megatron *.bin/.idx data layout the harness legitimately reads — they
    are not the Megatron-LM training framework, and flagging them forced
    every dev agent to fight false positives in the dataloader path.
    """

    def test_megatron_import_is_flagged(self) -> None:
        reason = framework_guard.match_banned_reference("import megatron.core")
        self.assertIsNotNone(reason)
        self.assertIn("megatron", reason)

    def test_from_megatron_import_is_flagged(self) -> None:
        reason = framework_guard.match_banned_reference("from megatron import core")
        self.assertIsNotNone(reason)

    def test_megatron_attribute_access_is_flagged(self) -> None:
        self.assertIsNotNone(framework_guard.match_banned_reference("x = megatron.core.foo()"))

    def test_bare_deepspeed_import_is_flagged(self) -> None:
        self.assertIsNotNone(framework_guard.match_banned_reference("import deepspeed"))

    def test_megatron_binary_dataclass_name_is_not_flagged(self) -> None:
        self.assertIsNone(framework_guard.match_banned_reference("class MegatronBinaryDataloader:"))

    def test_megatron_binary_loader_kind_string_is_not_flagged(self) -> None:
        self.assertIsNone(
            framework_guard.match_banned_reference(
                'return build_dataloader(loader="megatron_binary")'
            )
        )

    def test_megatron_binary_helper_name_is_not_flagged(self) -> None:
        self.assertIsNone(
            framework_guard.match_banned_reference("def _is_megatron_binary(path_prefix):")
        )


class TestPythonCommentDocstringExemption(unittest.TestCase):
    """Banned tokens that appear *only* inside Python ``#`` comments or
    docstrings (module / class / function) must NOT trigger the guard.

    Tokens that appear in real code — including ordinary string
    literals such as ``importlib.import_module("torch.nn")`` — MUST
    still be flagged. The exemption is a documentation-noise filter,
    not a blanket "anything in a string is fine" rule.
    """

    def _scan_source(self, source: str) -> list[framework_guard.GuardViolation]:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
            fh.write(source)
            tmp = Path(fh.name)
        try:
            return framework_guard.scan_file(tmp)
        finally:
            tmp.unlink(missing_ok=True)

    # ---- Positive exemptions (must NOT be flagged) ----

    def test_module_docstring_with_banned_token_is_not_flagged(self) -> None:
        source = '"""This module is the bare-tensor equivalent of torch.nn.Linear."""\n'
        self.assertEqual(self._scan_source(source), [])

    def test_function_docstring_with_banned_token_is_not_flagged(self) -> None:
        source = (
            "def f(x):\n"
            '    """Equivalent to F.silu(x); torch.nn.functional.silu is banned."""\n'
            "    return x\n"
        )
        self.assertEqual(self._scan_source(source), [])

    def test_class_docstring_with_banned_token_is_not_flagged(self) -> None:
        source = (
            "class C:\n"
            '    """Stores weights matching the convention nn.Linear.weight uses."""\n'
            "    pass\n"
        )
        self.assertEqual(self._scan_source(source), [])

    def test_full_line_hash_comment_with_banned_token_is_not_flagged(self) -> None:
        source = "# using torch.nn under the hood\nx = 1\n"
        self.assertEqual(self._scan_source(source), [])

    def test_inline_hash_comment_with_banned_token_is_not_flagged(self) -> None:
        source = "x = 1  # equivalent to .backward() but static\n"
        self.assertEqual(self._scan_source(source), [])

    def test_multiline_docstring_with_banned_token_is_not_flagged(self) -> None:
        source = (
            "def f():\n"
            '    """First line.\n'
            "\n"
            "    Spans multiple lines and mentions torch.nn here,\n"
            "    plus megatron and deepspeed for context.\n"
            '    """\n'
            "    return 1\n"
        )
        self.assertEqual(self._scan_source(source), [])

    def test_docstring_mentioning_dot_backward_is_not_flagged(self) -> None:
        source = (
            'def f():\n    """Computes gradients without calling .backward()."""\n    return 1\n'
        )
        self.assertEqual(self._scan_source(source), [])

    # ---- Negative cases (regression guards — MUST still be flagged) ----

    def test_real_torch_nn_import_is_still_flagged(self) -> None:
        source = "import torch.nn\n"
        violations = self._scan_source(source)
        self.assertEqual(len(violations), 1)
        self.assertIn("torch.nn", violations[0].reason)

    def test_string_literal_dynamic_import_is_still_flagged(self) -> None:
        # Dynamic import via string is a real risk — must not be
        # silenced by the comment/docstring exemption.
        source = 'import importlib\nm = importlib.import_module("torch.nn")\n'
        violations = self._scan_source(source)
        self.assertEqual(len(violations), 1)
        self.assertIn("torch.nn", violations[0].reason)

    def test_code_before_inline_comment_is_still_flagged(self) -> None:
        source = "import torch.optim  # legit? no, it is banned\n"
        violations = self._scan_source(source)
        self.assertEqual(len(violations), 1)
        self.assertIn("torch.optim", violations[0].reason)

    def test_backward_call_in_code_is_flagged_even_if_docstring_mentions_it(
        self,
    ) -> None:
        source = (
            'def step(loss):\n    """Avoids .backward() in normal paths."""\n    loss.backward()\n'
        )
        violations = self._scan_source(source)
        self.assertEqual(len(violations), 1)
        self.assertIn(".backward(", violations[0].reason)
        self.assertEqual(violations[0].line_number, 3)

    def test_unparseable_python_falls_back_to_raw_scan(self) -> None:
        # Broken syntax must NOT silently disable the guard — the
        # fallback is raw line scanning (fail-open).
        source = "def broken(\nimport megatron.core\n"
        violations = self._scan_source(source)
        # The import line should still be flagged via the fallback.
        self.assertTrue(
            any("megatron" in v.reason for v in violations),
            f"fail-open fallback missed banned token: {violations}",
        )

    def test_violation_line_text_is_original_not_redacted(self) -> None:
        # When a real code-line violation lands next to an inline
        # comment, the reported ``line`` should be the original (so
        # the user sees the full context), not the blanked version.
        source = "import torch.nn  # explanatory note\n"
        violations = self._scan_source(source)
        self.assertEqual(len(violations), 1)
        self.assertIn("# explanatory note", violations[0].line)

    # ---- End-to-end via the CLI entrypoint ----

    def test_cli_passes_when_only_docstring_mentions_banned_token(self) -> None:
        probe = REPO_ROOT / "harness" / "_doc_only_probe_tmp.py"
        probe.write_text(
            '"""Bare-tensor analogue of torch.nn.Linear; replaces F.silu."""\nx = 1\n',
            encoding="utf-8",
        )
        try:
            # Scope the scan to the probe file so pre-existing repo-wide
            # baseline violations do not pollute this assertion.
            result = subprocess.run(
                [sys.executable, "-m", "harness.framework_guard", str(probe)],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                "guard flagged docstring-only banned-token mention: " + result.stderr,
            )
        finally:
            probe.unlink(missing_ok=True)

    def test_cli_fails_when_string_literal_mentions_banned_token(self) -> None:
        probe = REPO_ROOT / "harness" / "_string_literal_probe_tmp.py"
        probe.write_text(
            'import importlib\nm = importlib.import_module("torch.nn")\n',
            encoding="utf-8",
        )
        try:
            result = subprocess.run(
                [sys.executable, "-m", "harness.framework_guard", str(probe)],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(
                result.returncode,
                0,
                "guard missed dynamic import via string literal",
            )
            self.assertIn("_string_literal_probe_tmp.py", result.stderr)
        finally:
            probe.unlink(missing_ok=True)

    # ---- Non-Python files are not affected by this change ----

    def test_shell_file_hash_comment_with_banned_token_is_still_flagged(
        self,
    ) -> None:
        # The exemption is Python-only — .sh / .md / .toml scanning is
        # unchanged on purpose (no AST to lean on).
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, encoding="utf-8") as fh:
            fh.write("# mentions torch.nn here\necho ok\n")
            tmp = Path(fh.name)
        try:
            violations = framework_guard.scan_file(tmp)
        finally:
            tmp.unlink(missing_ok=True)
        self.assertEqual(len(violations), 1)
        self.assertIn("torch.nn", violations[0].reason)


class TestLongRunningGatesDelegate(unittest.TestCase):
    def test_long_running_gates_delegate_to_training_engine_tensor(self) -> None:
        """SSOT: long-running gate scripts must be thin wrappers that
        only construct ``TrainLoopConfig`` and call
        ``run_training_loop`` — dataloader / forward / optimizer /
        checkpoint logic belongs inside ``training_engine_tensor``,
        not in the harness gate layer.

        The set below covers every ours-side subprocess runner:

        * ``eval_capture_align.py`` — single-step tensor-capture diff
          (forward-align / backward-align) on real Megatron ``GPTDataset``;
        * ``eval_train_steps.py``   — bitwise trajectory
          (single-card and DP=8) on real Megatron ``GPTDataset``;
        * ``eval_long_train.py``    — long-horizon main gate;
        * ``eval_resume_train.py``  — resume gate;
        * ``op_long_ours.py``       — op-long auxiliary trajectory.
        """
        scripts_dir = REPO_ROOT / "evals" / "scripts"
        if not scripts_dir.exists():
            self.skipTest("evals/scripts/ not present in this repo")
        thin_wrappers = (
            "eval_capture_align.py",
            "eval_long_train.py",
            "eval_resume_train.py",
            "eval_train_steps.py",
            "op_long_ours.py",
        )
        required_import = (
            "from training_engine_tensor.train_loop import TrainLoopConfig, run_training_loop"
        )
        forbidden_substrings = (
            "DataLoader(",
            "GPTDataset(",
            "BlendedMegatronDatasetBuilder(",
        )
        for name in thin_wrappers:
            path = scripts_dir / name
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            self.assertIn(
                required_import,
                text,
                f"{name} must import the SSOT entry point "
                "(training_engine_tensor.train_loop) — gates are thin "
                "wrappers, not training implementations.",
            )
            self.assertIn(
                "run_training_loop(",
                text,
                f"{name} must delegate execution to run_training_loop().",
            )
            for token in forbidden_substrings:
                self.assertNotIn(
                    token,
                    text,
                    f"{name} re-implements training-side primitives "
                    f"({token!r}); this violates the gate-vs-engine "
                    "SSOT boundary.",
                )

    def test_evals_scripts_dir_has_no_legacy_dataloader_helpers(self) -> None:
        scripts_dir = REPO_ROOT / "evals" / "scripts"
        if not scripts_dir.exists():
            self.skipTest("evals/scripts/ not present in this repo")
        forbidden_helpers = (
            "_build_real_dataloader",
            "_build_mg_dataloader",
            "_build_our_dataloader",
            "parse_data_path_from_conf",
            "next_nonempty_batch",
            "build_gpt_data_iterator",
        )
        offenders: list[tuple[str, str]] = []
        for path in scripts_dir.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for helper in forbidden_helpers:
                if helper in text:
                    offenders.append((path.name, helper))
        self.assertEqual(
            offenders,
            [],
            "evals/scripts/ still defines or imports legacy dataloader "
            "helpers; after the SSOT refactor, long-running gates must "
            "delegate dataloader/training to "
            "training_engine_tensor.train_loop.run_training_loop "
            f"(see prompt/develop_prompt/{{megatron,torch}}/stage1/). "
            f"Offenders: {offenders}",
        )


if __name__ == "__main__":
    unittest.main()
