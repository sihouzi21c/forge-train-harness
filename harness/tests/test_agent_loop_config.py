from __future__ import annotations

import os
import shutil
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from tools import agent_loop_config

_AXES = ("ref", "data", "remote", "agent", "eval", "model", "optim")
# Tests assert against the claudecode template's wording — pin it explicitly
# rather than glob-picking the first toml in config/agent/, which would
# otherwise vary by filename sort and break the assertion.
_AGENT_TEMPLATE = agent_loop_config.REPO_ROOT / "config" / "agent" / "claudecode-default.toml"


class _AgentTomlFixture:
    """Stage a complete per-loop config dir under FORGE_CONFIG_DIR.

    Replaces the prior fixture that wrote to ``<repo>/config/agent.toml``
    (the top-level mutable shared file removed by the per-loop config
    refactor). Each test class gets its own tempdir; ``load_workload_config``
    reads through ``FORGE_CONFIG_DIR`` so this is a drop-in.
    """

    _tmpdir: tempfile.TemporaryDirectory | None = None
    _saved_env: str | None = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()  # type: ignore[misc]
        cls._tmpdir = tempfile.TemporaryDirectory()
        cfg_dir = Path(cls._tmpdir.name)
        for axis in _AXES:
            axis_dir = agent_loop_config.REPO_ROOT / "config" / axis
            if axis == "agent":
                src = _AGENT_TEMPLATE
            else:
                tomls = sorted(axis_dir.glob("*.toml")) if axis_dir.is_dir() else []
                if not tomls:
                    continue
                src = tomls[0]
            shutil.copy2(src, cfg_dir / f"{axis}.toml")
        cls._saved_env = os.environ.get("FORGE_CONFIG_DIR")
        os.environ["FORGE_CONFIG_DIR"] = str(cfg_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._saved_env is None:
            os.environ.pop("FORGE_CONFIG_DIR", None)
        else:
            os.environ["FORGE_CONFIG_DIR"] = cls._saved_env
        if cls._tmpdir is not None:
            cls._tmpdir.cleanup()
            cls._tmpdir = None
        super().tearDownClass()  # type: ignore[misc]


class TestAgentLoopConfig(_AgentTomlFixture, unittest.TestCase):
    def test_loads_agent_config(self) -> None:
        cfg = agent_loop_config.load_agent_config()
        self.assertEqual(cfg["runs_per_stage"], 0)
        self.assertEqual(cfg["poll_seconds"], 5)
        self.assertEqual(cfg["agent_round_tries"], 8)
        self.assertEqual(cfg["retry_base_sleep"], 4)
        self.assertIn("Transient agent CLI / network disconnect", cfg["retry_continue_prompt"])
        self.assertEqual(cfg["max_consecutive_review_fails"], 3)
        self.assertEqual(cfg["state_dir"], ".artifacts/agent-loop-state")
        self.assertIn("model", cfg)
        self.assertIn("effort", cfg)

    def test_codex_default_agent_template_matches_fixed_cli_defaults(self) -> None:
        path = agent_loop_config.REPO_ROOT / "config" / "agent" / "codex-default.toml"
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        cfg = data["agent"]

        self.assertEqual(cfg["backend"], "codex")
        self.assertEqual(cfg["model"], "")
        self.assertEqual(cfg["effort"], "")
        self.assertEqual(cfg["runs_per_stage"], 0)
        self.assertEqual(cfg["poll_seconds"], 5)
        self.assertEqual(cfg["agent_round_tries"], 8)
        self.assertEqual(cfg["retry_base_sleep"], 4)
        self.assertIn("Transient agent CLI / network disconnect", cfg["retry_continue_prompt"])
        self.assertEqual(cfg["state_dir"], ".artifacts/agent-loop-state")
        self.assertEqual(cfg["max_consecutive_review_fails"], 3)

    def test_agent_loop_config_uses_harness_workload_loader(self) -> None:
        text = (agent_loop_config.REPO_ROOT / "tools" / "agent_loop_config.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("config_runtime.load_workload_config", text)
        self.assertNotIn("tomllib.loads", text)

    def test_shell_exports_include_agent_loop_knobs(self) -> None:
        exports = agent_loop_config.shell_exports({})
        for key in (
            "MODEL",
            "REVIEW_MODEL",
            "RUNS_PER_STAGE",
            "POLL_SECONDS",
            "PRODUCTION_POLL_SECONDS",
            "CURSOR_AGENT_ROUND_TRIES",
            "CURSOR_AGENT_RETRY_BASE_SLEEP",
            "CURSOR_AGENT_RETRY_CONTINUE_PROMPT",
            "AGENT_LOOP_STATE_DIR",
            "MAX_CONSECUTIVE_REVIEW_FAILS",
            "MAX_MODE",
            "AGENT_BACKEND",
            "ROUND_TIMEOUT_S",
        ):
            self.assertIn(f"export {key}=", exports)
        self.assertIn("export RUNS_PER_STAGE='0'", exports)
        self.assertIn("export MAX_CONSECUTIVE_REVIEW_FAILS='3'", exports)
        self.assertIn("export ROUND_TIMEOUT_S='18000'", exports)
        self.assertRegex(exports, r"export AGENT_BACKEND='(cursor-cli|claude-code|codex)'")

    def test_shell_exports_proxy_knobs_empty_by_default(self) -> None:
        # The default profile sets no proxy => both knobs emit empty, so
        # agent-loop.sh leaves ANTHROPIC_* unset and the global OAuth
        # default is untouched.
        exports = agent_loop_config.shell_exports({})
        self.assertIn("export AGENT_BASE_URL=''", exports)
        self.assertIn("export AGENT_API_KEY=''", exports)

    def test_shell_exports_emit_configured_proxy_base_url_and_key(self) -> None:
        fake = {
            "backend": "claude-code",
            "base_url": "https://llm-center.ali.modelbest.cn/llm",
            "api_key": "sk-test-key",
        }
        with mock.patch.object(agent_loop_config, "load_agent_config", return_value=fake):
            exports = agent_loop_config.shell_exports({})
        self.assertIn(
            "export AGENT_BASE_URL='https://llm-center.ali.modelbest.cn/llm'",
            exports,
        )
        self.assertIn("export AGENT_API_KEY='sk-test-key'", exports)

    def test_agent_loop_sh_exports_proxy_base_url_for_claude_code(self) -> None:
        # The override must be gated on claude-code AND a non-empty
        # AGENT_BASE_URL so it never fires for the official-OAuth default.
        text = agent_loop_config.REPO_ROOT.joinpath("agent-loop.sh").read_text(encoding="utf-8")
        self.assertIn(
            '[[ "$AGENT_BACKEND" == claude-code && -n "$AGENT_BASE_URL" ]] '
            '&& export ANTHROPIC_BASE_URL="$AGENT_BASE_URL"',
            text,
        )
        self.assertIn('CLI_API_KEY="$AGENT_API_KEY"', text)

    def test_llmcenter_template_has_base_url_and_no_committed_secret(self) -> None:
        tmpl = agent_loop_config.REPO_ROOT / "config" / "agent" / "claudecode-llmcenter.toml"
        cfg = tomllib.loads(tmpl.read_text(encoding="utf-8"))["agent"]
        self.assertEqual(cfg["backend"], "claude-code")
        self.assertEqual(cfg["model"], "GLM_5tjygf[1m]")
        self.assertEqual(cfg["base_url"], "https://llm-center.ali.modelbest.cn/llm")
        # Never ship a real key in a committed template.
        self.assertEqual(cfg["api_key"], "")

    def test_agent_loop_helper_derives_stage_prompt_files(self) -> None:
        self.assertEqual(agent_loop_config.agent_loop_stages(), ["stage1", "stage2"])
        self.assertEqual(
            [path.name for path in agent_loop_config.stage_rule_files("stage2")],
            ["stage2.md"],
        )
        self.assertEqual(agent_loop_config.stage_review_template("stage1").name, "review_stage1.md")
        self.assertEqual(agent_loop_config.coding_guidelines_file().name, "coding-guidelines.md")

    def test_stage_rule_files_default_loads_every_milestone(self) -> None:
        names = [p.name for p in agent_loop_config.stage_rule_files("stage1")]
        self.assertEqual(
            names,
            [
                "overview.md",
                "constraint.md",
                "debug.md",
                "alignment.md",
                "bitwise-singlecard.md",
                "bitwise-multicard.md",
                "bitwise-perf.md",
                "resume.md",
                "long-horizon.md",
                "production.md",
            ],
        )

    def test_stage_rule_files_filters_to_active_milestone(self) -> None:
        names = [p.name for p in agent_loop_config.stage_rule_files("stage1", "bitwise-multicard")]
        self.assertEqual(
            names, ["overview.md", "constraint.md", "debug.md", "bitwise-multicard.md"]
        )

    def test_stage_rule_files_filters_each_milestone_in_isolation(self) -> None:
        for milestone in agent_loop_config.stage_milestones("stage1"):
            with self.subTest(milestone=milestone):
                names = [p.name for p in agent_loop_config.stage_rule_files("stage1", milestone)]
                self.assertEqual(
                    names,
                    ["overview.md", "constraint.md", "debug.md", f"{milestone}.md"],
                )

    def test_stage_rule_files_rejects_invalid_milestone(self) -> None:
        for bogus in ("M1", "M8", "bitwise", "", "foo"):
            with self.subTest(milestone=bogus), self.assertRaises(ValueError):
                agent_loop_config.stage_rule_files("stage1", bogus)

    def test_stage_rule_files_rejects_milestone_for_stage_without_manifest(self) -> None:
        with self.assertRaises(ValueError):
            agent_loop_config.stage_rule_files("stage2", "alignment")

    def test_stage_milestones_returns_linear_progression(self) -> None:
        self.assertEqual(
            agent_loop_config.stage_milestones("stage1"),
            [
                "alignment",
                "bitwise-singlecard",
                "bitwise-multicard",
                "bitwise-perf",
                "resume",
                "long-horizon",
                "production",
            ],
        )
        self.assertEqual(agent_loop_config.stage_milestones("stage2"), [])

    def test_agent_loop_does_not_hardcode_stage_mapping(self) -> None:
        text = agent_loop_config.REPO_ROOT.joinpath("agent-loop.sh").read_text(encoding="utf-8")
        self.assertNotIn('STAGES=("stage1" "stage2")', text)
        self.assertNotIn('STAGES=("stage1")', text)
        self.assertNotIn('STAGES=("stage2")', text)
        self.assertIn('agent_loop_config.py" stages', text)
        self.assertIn('agent_loop_config.py" stage-rule-files', text)

    def test_production_milestone_gates_the_slow_poll(self) -> None:
        # The multi-day production long-train is the only milestone the dev
        # agent babysits with a sparse poll. The gate keys on
        # FORGE_SLOW_POLL_MILESTONE (the production-train suite's milestone,
        # resolved from eval.toml) — NOT a hardcoded milestone name — so a
        # rename of the production milestone cannot silently disable it. The
        # retired positional "M7" label already broke it once (the name-based
        # state machine never emits it, so the throttle was dead).
        text = agent_loop_config.REPO_ROOT.joinpath("agent-loop.sh").read_text(
            encoding="utf-8",
        )
        self.assertIn('"$current_milestone" == "$FORGE_SLOW_POLL_MILESTONE"', text)
        self.assertIn("slow-poll-milestone stage1", text)
        self.assertIn("export FORGE_SLOW_POLL_MILESTONE", text)
        self.assertIn("PRODUCTION_POLL_SECONDS", text)
        self.assertNotIn('"$current_milestone" == "M7"', text)
        # Not re-hardcoded to a literal milestone name either.
        self.assertNotIn('"$current_milestone" == "production"', text)
        self.assertNotIn("M7_POLL_SECONDS", text)

    def test_slow_poll_milestone_derives_from_production_train_suite(self) -> None:
        evals = {
            "perf-bitwise": {
                "runner_kind": "perf-bitwise",
                "milestone": "bitwise-perf",
                "stage": "stage1",
            },
            "production-train": {
                "runner_kind": "production-train",
                "milestone": "production",
                "stage": "stage1",
            },
        }
        self.assertEqual(
            agent_loop_config._slow_poll_milestone_from_evals(evals),
            "production",
        )

    def test_slow_poll_milestone_empty_without_production_train(self) -> None:
        # e.g. the 8b suite, which ends at bitwise-dptp and ships no
        # production long-train -> the slow-poll throttle never engages.
        evals = {
            "bitwise-dptp": {
                "runner_kind": "bitwise-dptp",
                "milestone": "bitwise-dptp",
                "stage": "stage1",
            },
        }
        self.assertEqual(
            agent_loop_config._slow_poll_milestone_from_evals(evals),
            "",
        )

    def test_agent_loop_does_not_hardcode_runtime_knob_fallbacks(self) -> None:
        text = agent_loop_config.REPO_ROOT.joinpath("agent-loop.sh").read_text(
            encoding="utf-8",
        )
        self.assertIn("agent_loop_config.py", text)
        forbidden = (
            'RUNS_PER_STAGE="${RUNS_PER_STAGE:-6}"',
            'RUNS_PER_STAGE="${RUNS_PER_STAGE:-0}"',
            'POLL_SECONDS="${POLL_SECONDS:-5}"',
            'CURSOR_AGENT_ROUND_TRIES="${CURSOR_AGENT_ROUND_TRIES:-8}"',
            'CURSOR_AGENT_RETRY_BASE_SLEEP="${CURSOR_AGENT_RETRY_BASE_SLEEP:-4}"',
            'CURSOR_AGENT_RETRY_CONTINUE_PROMPT="${CURSOR_AGENT_RETRY_CONTINUE_PROMPT:-',
            'MAX_CONSECUTIVE_REVIEW_FAILS="${MAX_CONSECUTIVE_REVIEW_FAILS:-3}"',
            'AGENT_LOOP_STATE_DIR="${AGENT_LOOP_STATE_DIR:-',
            'MAX_MODE="${MAX_MODE:-',
        )
        for snippet in forbidden:
            with self.subTest(snippet=snippet):
                self.assertNotIn(snippet, text)

    def test_agent_loop_supports_cursor_claude_and_codex_backends(self) -> None:
        text = agent_loop_config.REPO_ROOT.joinpath("agent-loop.sh").read_text(encoding="utf-8")

        self.assertIn("--backend", text)
        self.assertIn("AGENT_BACKEND", text)
        self.assertIn("cursor-cli", text)
        self.assertIn("claude-code", text)
        self.assertIn("codex", text)
        self.assertIn("ANTHROPIC_API_KEY", text)
        self.assertIn("OPENAI_API_KEY", text)
        # After the unified-agent-log refactor, every spawn flows through
        # _spawn_child_agent (which calls harness/tools/spawn_managed_agent.py)
        # so dev/review/subagent all land as Session rows under web-agents/.
        self.assertIn("_spawn_child_agent loop_dev_round", text)
        self.assertIn("_spawn_child_agent loop_review", text)
        self.assertIn("spawn_managed_agent.py", text)

    def test_agent_loop_separates_workspace_root_from_source_helpers(self) -> None:
        text = agent_loop_config.REPO_ROOT.joinpath("agent-loop.sh").read_text(encoding="utf-8")

        self.assertIn("FORGE_SOURCE_ROOT", text)
        self.assertIn('export FORGE_REPO_ROOT="$_bootstrap_workspace"', text)
        self.assertIn("FORGE_SOURCE_ROOT/harness/tools/loop_wrapper_init.py", text)
        self.assertIn("FORGE_SOURCE_ROOT/harness/tools/loop_wrapper_event.py", text)
        self.assertIn("FORGE_SOURCE_ROOT/harness/tools/spawn_managed_agent.py", text)
        self.assertIn("sys.path.insert(0, '$FORGE_SOURCE_ROOT')", text)


class TestCheckBackendAvailable(unittest.TestCase):
    """Tests for the unified CLI-based auth probe."""

    @staticmethod
    def _fake_runner(
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
    ):
        def _run(cmd: list[str]) -> tuple[int, str, str]:
            return returncode, stdout, stderr

        return _run

    def test_check_backend_delegates_to_cli_status_subcommand(self) -> None:
        for cli, key in (
            ("cursor", "isAuthenticated"),
            ("claude", "loggedIn"),
            ("codex", None),
        ):
            with self.subTest(cli=cli):
                runner = self._fake_runner(
                    returncode=0,
                    stdout=f'{{"{key}": true}}' if key else "Logged in",
                )
                with mock.patch.dict(os.environ, {}, clear=True):
                    ok, message = agent_loop_config.check_backend_available(
                        cli,
                        which=lambda _cmd: "/usr/local/bin/x",
                        run_status=runner,
                    )
                self.assertTrue(ok, message)

    def test_check_backend_uses_correct_status_command(self) -> None:
        captured: dict[str, list[str]] = {}

        def runner(cmd: list[str]) -> tuple[int, str, str]:
            captured["cmd"] = cmd
            return 0, '{"isAuthenticated": true, "loggedIn": true}', ""

        for cli, expected_cmd in (
            ("cursor", ["agent", "status", "--format", "json"]),
            ("claude", ["claude", "auth", "status", "--json"]),
            ("codex", ["codex", "login", "status"]),
        ):
            with self.subTest(cli=cli):
                captured.clear()
                ok, _ = agent_loop_config.check_backend_available(
                    cli,
                    which=lambda _cmd: "/usr/local/bin/x",
                    run_status=runner,
                )
                self.assertTrue(ok)
                self.assertEqual(captured["cmd"], expected_cmd)

    def test_check_backend_fails_when_binary_missing(self) -> None:
        for cli, expected_install_hint in (
            ("cursor", "agent"),
            ("claude", "claude"),
            ("codex", "codex"),
        ):
            with self.subTest(cli=cli):
                ok, message = agent_loop_config.check_backend_available(
                    cli,
                    which=lambda _cmd: None,
                    run_status=self._fake_runner(returncode=0, stdout="{}"),
                )
                self.assertFalse(ok)
                self.assertIn(expected_install_hint, message)

    def test_check_backend_fails_when_status_reports_not_authed(self) -> None:
        for cli, key, login_hint, env_var in (
            ("cursor", "isAuthenticated", "agent login", "CURSOR_API_KEY"),
            ("claude", "loggedIn", "claude /login", "ANTHROPIC_API_KEY"),
            ("codex", None, "codex login", "OPENAI_API_KEY"),
        ):
            with self.subTest(cli=cli):
                runner = self._fake_runner(
                    returncode=1,
                    stdout=f'{{"{key}": false}}' if key else "Not logged in",
                )
                if cli == "codex":
                    with (
                        tempfile.TemporaryDirectory() as tmp,
                        mock.patch.dict(os.environ, {"CODEX_HOME": tmp}, clear=True),
                    ):
                        ok, message = agent_loop_config.check_backend_available(
                            cli,
                            which=lambda _cmd: "/usr/local/bin/x",
                            run_status=runner,
                        )
                else:
                    with mock.patch.dict(os.environ, {}, clear=True):
                        ok, message = agent_loop_config.check_backend_available(
                            cli,
                            which=lambda _cmd: "/usr/local/bin/x",
                            run_status=runner,
                        )
                self.assertFalse(ok)
                self.assertIn(login_hint, message)
                self.assertIn(env_var, message)

    def test_codex_backend_accepts_active_provider_env_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.toml").write_text(
                "\n".join(
                    [
                        'model_provider = "lm_center"',
                        "",
                        "[model_providers.lm_center]",
                        'env_key = "LLM_CENTER_API_KEY"',
                    ]
                ),
                encoding="utf-8",
            )
            runner = self._fake_runner(returncode=1, stdout="Not logged in")
            with mock.patch.dict(
                os.environ,
                {"CODEX_HOME": tmp, "LLM_CENTER_API_KEY": "env-good"},  # pragma: allowlist secret
                clear=True,
            ):
                ok, message = agent_loop_config.check_backend_available(
                    "codex",
                    which=lambda _cmd: "/usr/local/bin/codex",
                    run_status=runner,
                )
        self.assertTrue(ok, message)

    def test_check_backend_fails_when_status_command_crashes(self) -> None:
        runner = self._fake_runner(
            returncode=1,
            stdout="",
            stderr="Your macOS login keychain is locked.",
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            ok, message = agent_loop_config.check_backend_available(
                "cursor",
                which=lambda _cmd: "/usr/local/bin/agent",
                run_status=runner,
            )
        self.assertFalse(ok)
        self.assertIn("keychain", message)

    def test_check_backend_rejects_unknown_cli(self) -> None:
        with self.assertRaises(ValueError):
            agent_loop_config.check_backend_available(
                "openai",
                which=lambda _cmd: "/usr/local/bin/x",
                run_status=self._fake_runner(returncode=0, stdout="{}"),
            )


class TestAgentLoopShStatus(_AgentTomlFixture, unittest.TestCase):
    """Tests for the agent-loop.sh status subcommand."""

    def test_agent_loop_sh_invokes_backend_availability_check(self) -> None:
        text = agent_loop_config.REPO_ROOT.joinpath("agent-loop.sh").read_text(encoding="utf-8")
        self.assertIn('agent_loop_config.py" check-backend', text)

    def test_agent_loop_sh_supports_status_subcommand(self) -> None:
        text = agent_loop_config.REPO_ROOT.joinpath("agent-loop.sh").read_text(encoding="utf-8")
        self.assertRegex(text, r'SUBCOMMAND\s*=\s*""')
        self.assertIn('"$SUBCOMMAND" == "status"', text)

    def test_agent_loop_sh_status_short_circuits_before_loop_setup(self) -> None:
        text = agent_loop_config.REPO_ROOT.joinpath("agent-loop.sh").read_text(encoding="utf-8")
        status_idx = text.index('"$SUBCOMMAND" == "status"')
        log_dir_idx = text.index('LOG_DIR="$WORKSPACE/.artifacts/agent-logs/')
        self.assertLess(
            status_idx,
            log_dir_idx,
            "status subcommand must short-circuit before LOG_DIR setup",
        )

    def test_agent_loop_sh_status_skips_full_shell_exports(self) -> None:
        text = agent_loop_config.REPO_ROOT.joinpath("agent-loop.sh").read_text(encoding="utf-8")
        status_idx = text.index('"$SUBCOMMAND" == "status"')
        shell_eval_idx = text.index('agent_loop_config.py" shell)')
        self.assertLess(
            status_idx,
            shell_eval_idx,
            "status subcommand must short-circuit before shell_exports eval",
        )

    def test_active_cli_subcommand_returns_resolved_cli(self) -> None:
        cfg = agent_loop_config.load_agent_config()
        self.assertEqual(
            agent_loop_config.active_cli(),
            agent_loop_config._BACKEND_TO_CLI.get(str(cfg.get("backend", "cursor-cli")), "cursor"),
        )

    def test_agent_loop_sh_help_documents_status_subcommand(self) -> None:
        text = agent_loop_config.REPO_ROOT.joinpath("agent-loop.sh").read_text(encoding="utf-8")
        self.assertRegex(text, r"status\b[^\n]*backend auth")


class TestAgentLoopStateMachine(_AgentTomlFixture, unittest.TestCase):
    """Lightweight schema/contract tests for the goal-driven loop knobs."""

    def test_state_file_paths_are_per_stage_under_state_dir(self) -> None:
        sd = agent_loop_config.state_dir()
        self.assertTrue(sd.is_absolute())
        self.assertEqual(sd.name, "agent-loop-state")
        s1 = agent_loop_config.state_file_for("stage1")
        s2 = agent_loop_config.state_file_for("stage2")
        self.assertEqual(s1.parent, sd)
        self.assertEqual(s2.parent, sd)
        self.assertEqual(s1.name, "stage1.status")
        self.assertEqual(s2.name, "stage2.status")

    def test_state_file_for_rejects_unknown_stage(self) -> None:
        with self.assertRaises(ValueError):
            agent_loop_config.state_file_for("stage3")
        with self.assertRaises(ValueError):
            agent_loop_config.state_file_for("")

    def test_valid_stage_statuses_match_loop_contract(self) -> None:
        self.assertEqual(
            tuple(agent_loop_config.VALID_STAGE_STATUSES),
            ("pending", "in-progress", "finished"),
        )

    def test_agent_loop_sh_implements_finish_state_machine(self) -> None:
        text = agent_loop_config.REPO_ROOT.joinpath("agent-loop.sh").read_text(encoding="utf-8")
        for needle in (
            "stage_state_file()",
            "read_stage_status()",
            "write_stage_status()",
            "parse_review_stage_status()",
            "STAGE_STATUS:",
            "RESET_STATE",
            "MAX_CONSECUTIVE_REVIEW_FAILS",
            "consecutive_fails",
            "STAGE_STATUS=finished",
            # After the unified-agent-log refactor, _spawn_child_agent is
            # the single spawn surface (replacing the old run_agent_cmd
            # / run_backend_agent helpers).
            "_spawn_child_agent",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, text)

    def test_agent_loop_sh_downgrades_finished_on_review_fail(self) -> None:
        """A FAIL'd review must never persist STAGE_STATUS=finished.

        review_common.md requires the review agent to emit
        STAGE_STATUS: in-progress when REVIEW_VERDICT is FAIL, but the
        loop must enforce the same invariant as defence-in-depth in case
        a buggy / non-compliant review agent emits the wrong line.
        """
        text = agent_loop_config.REPO_ROOT.joinpath("agent-loop.sh").read_text(encoding="utf-8")
        self.assertIn("review_passed=0", text)
        self.assertRegex(
            text,
            r'review_passed == 0[^\n]*&&[^\n]*stage_status[^\n]*== "finished"',
        )
        self.assertIn('stage_status="in-progress"', text)

    def test_agent_loop_sh_state_machine_handles_all_termination_paths(self) -> None:
        """Spot-check the four termination paths of run_agent_stage."""
        text = agent_loop_config.REPO_ROOT.joinpath("agent-loop.sh").read_text(encoding="utf-8")
        self.assertRegex(
            text, r'\[\[ "\$stage_status" == "finished" \]\];\s*then[\s\S]{0,200}return 0'
        )
        self.assertIn("RUNS_PER_STAGE > 0 && round > RUNS_PER_STAGE", text)
        self.assertIn(
            "MAX_CONSECUTIVE_REVIEW_FAILS > 0 && consecutive_fails >= MAX_CONSECUTIVE_REVIEW_FAILS",
            text,
        )
        self.assertRegex(text, r'initial_status[\s\S]{0,200}== "finished"[\s\S]{0,200}continue')

    def test_remote_mfu_poller_rsync_is_timeout_guarded(self) -> None:
        """The background MFU puller's rsync must carry an I/O watchdog.

        A bare ``rsync -aqz`` over ssh can hang indefinitely when the
        connection is established but the remote sender stalls (e.g. a
        half-open Teleport tunnel after a devspace reclaim). The poller
        loop is serial — ``while true; do sleep 30; rsync ...; done`` —
        so one wedged rsync freezes every subsequent poll, and the MFU
        badge never updates for the rest of the run. ``rsync --timeout=N``
        aborts on N seconds of I/O silence without needing an external
        ``timeout`` binary (absent on macOS), so the loop can recover on
        the next iteration. Both the periodic poller and the EXIT-trap
        final pull must be guarded.
        """
        text = agent_loop_config.REPO_ROOT.joinpath("agent-loop.sh").read_text(encoding="utf-8")
        invocations = [ln for ln in text.splitlines() if ln.strip().startswith("rsync -aqz")]
        self.assertEqual(
            len(invocations),
            2,
            f"expected exactly 2 MFU-poller rsync invocations, found {len(invocations)}: {invocations}",
        )
        for ln in invocations:
            with self.subTest(line=ln.strip()):
                self.assertRegex(
                    ln,
                    r"--timeout=\d+",
                    "MFU-poller rsync must carry an I/O timeout watchdog so a "
                    "stalled remote sender cannot freeze the poll loop forever",
                )


class TestRemoteKind(unittest.TestCase):
    """``[remote].kind`` selects the execution protocol: ``local`` (off),
    ``ssh`` (transport only), or ``devspace`` (transport + lifecycle lease).
    ``shell_exports`` must surface ``REMOTE_KIND`` and derive
    ``REMOTE_ENABLED`` as ``kind != "local"``.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        cfg_dir = Path(self._tmp.name)
        for axis in _AXES:
            if axis == "remote":
                continue
            axis_dir = agent_loop_config.REPO_ROOT / "config" / axis
            if axis == "agent":
                src = _AGENT_TEMPLATE
            elif axis == "eval":
                # eval has no flat top-level template anymore — seed from the
                # directory-form registry.
                src = axis_dir / "dense_training" / "dense_training.toml"
            else:
                src = sorted(axis_dir.glob("*.toml"))[0]
            shutil.copy2(src, cfg_dir / f"{axis}.toml")
        self._cfg_dir = cfg_dir
        self._saved = os.environ.get("FORGE_CONFIG_DIR")
        os.environ["FORGE_CONFIG_DIR"] = str(cfg_dir)

    def tearDown(self) -> None:
        if self._saved is None:
            os.environ.pop("FORGE_CONFIG_DIR", None)
        else:
            os.environ["FORGE_CONFIG_DIR"] = self._saved
        self._tmp.cleanup()

    def _write_remote(self, body: str) -> None:
        (self._cfg_dir / "remote.toml").write_text(body, encoding="utf-8")

    def test_local_kind_disables_remote(self) -> None:
        self._write_remote('[remote]\nkind = "local"\n')
        exports = agent_loop_config.shell_exports({})
        self.assertIn("export REMOTE_KIND='local'", exports)
        self.assertIn("export REMOTE_ENABLED='false'", exports)

    def test_ssh_kind_enables_transport(self) -> None:
        self._write_remote('[remote]\nkind = "ssh"\nhostname = "gpu-box"\n')
        exports = agent_loop_config.shell_exports({})
        self.assertIn("export REMOTE_KIND='ssh'", exports)
        self.assertIn("export REMOTE_ENABLED='true'", exports)
        self.assertIn("export REMOTE_SSH_HOST='gpu-box'", exports)

    def test_devspace_kind_enables_transport(self) -> None:
        self._write_remote('[remote]\nkind = "devspace"\nhostname = "ds-411859"\n')
        exports = agent_loop_config.shell_exports({})
        self.assertIn("export REMOTE_KIND='devspace'", exports)
        self.assertIn("export REMOTE_ENABLED='true'", exports)

    def test_missing_table_defaults_to_local(self) -> None:
        self._write_remote("")
        exports = agent_loop_config.shell_exports({})
        self.assertIn("export REMOTE_KIND='local'", exports)
        self.assertIn("export REMOTE_ENABLED='false'", exports)


class TestRemoteConfigDir(unittest.TestCase):
    """The remote ``harness run`` must read config from an EXPLICIT
    ``FORGE_CONFIG_DIR``, not the implicit ``repo_root()/config`` cwd
    fallback. ``sync push`` lands the active ``*.toml`` in
    ``<remote_workdir>/config``; ``shell_exports`` surfaces that path as
    ``REMOTE_CONFIG_DIR`` so agent-loop.sh can substitute it into the
    remote-execution overlay's ``export FORGE_CONFIG_DIR=...`` line.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        cfg_dir = Path(self._tmp.name)
        for axis in _AXES:
            if axis == "remote":
                continue
            axis_dir = agent_loop_config.REPO_ROOT / "config" / axis
            if axis == "agent":
                src = _AGENT_TEMPLATE
            elif axis == "eval":
                # eval has no flat top-level template anymore — seed from the
                # directory-form registry.
                src = axis_dir / "dense_training" / "dense_training.toml"
            else:
                src = sorted(axis_dir.glob("*.toml"))[0]
            shutil.copy2(src, cfg_dir / f"{axis}.toml")
        self._cfg_dir = cfg_dir
        self._saved_cfg = os.environ.get("FORGE_CONFIG_DIR")
        self._saved_loop = os.environ.get("LOOP_ID")
        self._saved_web = os.environ.get("LOOP_WEB_ID")
        os.environ["FORGE_CONFIG_DIR"] = str(cfg_dir)
        os.environ["LOOP_ID"] = "loop-xyz"
        os.environ.pop("LOOP_WEB_ID", None)

    def tearDown(self) -> None:
        for key, saved in (
            ("FORGE_CONFIG_DIR", self._saved_cfg),
            ("LOOP_ID", self._saved_loop),
            ("LOOP_WEB_ID", self._saved_web),
        ):
            if saved is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = saved
        self._tmp.cleanup()

    def _write_remote(self, body: str) -> None:
        (self._cfg_dir / "remote.toml").write_text(body, encoding="utf-8")

    def test_shell_exports_emits_remote_config_dir(self) -> None:
        self._write_remote('[remote]\nkind = "ssh"\nhostname = "gpu-box"\nworkspace = "/work"\n')
        exports = agent_loop_config.shell_exports({})
        # REMOTE_CONFIG_DIR is the <remote_workdir>/config sibling that
        # sync push lands the active *.toml into.
        self.assertIn(
            "export REMOTE_CONFIG_DIR='/work/.forge_train/loop-xyz/config'",
            exports,
        )

    def test_agent_loop_sh_substitutes_remote_config_dir_placeholder(self) -> None:
        text = agent_loop_config.REPO_ROOT.joinpath("agent-loop.sh").read_text(encoding="utf-8")
        # Every overlay sed pass (remote + devspace + job) must
        # substitute the placeholder so each overlay can export an
        # explicit FORGE_CONFIG_DIR on the remote.
        self.assertEqual(
            text.count("s|@@REMOTE_CONFIG_DIR@@|${REMOTE_CONFIG_DIR}|g"),
            3,
        )

    def test_remote_execution_overlay_exports_forge_config_dir(self) -> None:
        overlay = agent_loop_config.REPO_ROOT.joinpath(
            "prompt", "develop_prompt", "remote-execution.md"
        ).read_text(encoding="utf-8")
        # The remote run must export FORGE_CONFIG_DIR explicitly rather
        # than relying on config_runtime._user_config_dir()'s cwd fallback.
        self.assertIn("export FORGE_CONFIG_DIR=@@REMOTE_CONFIG_DIR@@", overlay)


class TestRoundTimeout(_AgentTomlFixture, unittest.TestCase):
    """Per-round wall-clock cap (default 5h) enforced by the harness.

    A round that blows its ``round_timeout_s`` budget must NOT silently
    keep a half-finished commit: spawn_managed_agent.py SIGTERM->SIGKILLs
    the agent's process group and exits 124, and agent-loop.sh rolls the
    workspace back to the round's pre-spawn HEAD before continuing. This
    locks the contract in code so a refactor can't drop the kill, the
    distinct exit code, or the rollback.
    """

    def test_default_round_timeout_is_five_hours(self) -> None:
        cfg = {**agent_loop_config._AGENT_DEFAULTS, **agent_loop_config.load_agent_config()}
        self.assertEqual(cfg["round_timeout_s"], 18000)

    def test_shell_exports_surface_round_timeout(self) -> None:
        exports = agent_loop_config.shell_exports({})
        self.assertIn("export ROUND_TIMEOUT_S='18000'", exports)

    def test_spawn_helper_accepts_timeout_and_exits_124(self) -> None:
        src = agent_loop_config.REPO_ROOT.joinpath("tools", "spawn_managed_agent.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("--timeout-s", src)
        self.assertIn("asyncio.wait_for", src)
        self.assertIn("kill_process_group", src)
        self.assertIn("return 124", src)

    def test_agent_loop_sh_passes_round_budget_to_dev_spawn(self) -> None:
        text = agent_loop_config.REPO_ROOT.joinpath("agent-loop.sh").read_text(encoding="utf-8")
        # The dev round must hand its remaining budget to the spawn helper.
        self.assertIn("ROUND_TIMEOUT_S", text)
        self.assertIn("--timeout-s", text)

    def test_agent_loop_sh_rolls_back_on_round_timeout(self) -> None:
        text = agent_loop_config.REPO_ROOT.joinpath("agent-loop.sh").read_text(encoding="utf-8")
        # 124 is the timeout sentinel; it must break out of the attempt
        # loop (not be treated as a retryable transient) and trigger the
        # rollback to the round's baseline HEAD.
        self.assertIn("agent_ec == 124", text)
        self.assertIn("round_timed_out", text)
        self.assertIn("round_timeout", text)
        self.assertRegex(text, r'git -C "\$WORKSPACE" reset --hard "\$round_base_sha"')


if __name__ == "__main__":
    unittest.main()
