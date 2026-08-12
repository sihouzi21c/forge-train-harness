"""Agent prompts should include the shared coding guidelines entry."""

from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestPromptGuidelines(unittest.TestCase):
    def test_coding_guidelines_exist(self) -> None:
        path = REPO_ROOT / "prompt" / "review_prompt" / "coding-guidelines.md"
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        self.assertIn("Scope", text)
        self.assertIn("workload/src/training_engine_tensor/", text)
        self.assertIn("ref/", text)

    def test_review_common_exists(self) -> None:
        path = REPO_ROOT / "prompt" / "review_prompt" / "review_common.md"
        self.assertTrue(path.exists())
        # Post-rewrite anchor: the common prompt is the semantic-review
        # SSOT; mechanical pattern lint moved to dev-side
        # ``harness run anti-proxy``. The old "Mechanical Audit
        # Protocol" section was removed deliberately and must not
        # silently come back.
        text = path.read_text(encoding="utf-8")
        self.assertIn("Semantic review", text)
        self.assertIn("anti-proxy", text)
        self.assertNotIn("Mechanical Audit Protocol", text)

    def test_agent_loop_loads_coding_guidelines(self) -> None:
        text = (REPO_ROOT / "agent-loop.sh").read_text(encoding="utf-8")
        self.assertIn("coding-guidelines.md", text)

    def test_project_guide_documents_stage_specific_rule_loading(self) -> None:
        text = (REPO_ROOT / "prompt" / "project-guide.md").read_text(encoding="utf-8")
        self.assertIn("stage-specific", text)
        self.assertIn("prompt/develop_prompt/", text)
        self.assertNotIn("MUST** read **all** files under `prompt/develop_prompt/`", text)

    def test_stage2_subagent_playbook_carries_per_round_protocol(self) -> None:
        # The playbook is the single file ``cat``-ed into every subagent
        # prompt at M2.1 dispatch time and is the SSOT for: (a) the
        # 3a/3b/3c per-round decision tree, (b) the notes.md
        # self-described state anchors that subagents use to rebuild
        # cross-round state, and (c) the merge wrapper / failure
        # retention shell choreography that runs on the 3b PASS path.
        #
        # The playbook now lives at prompt/develop_prompt/_shared/ as a
        # single backend-neutral union markdown (matches the same
        # consolidation that stage1 already adopted).
        playbook = (
            REPO_ROOT / "prompt" / "develop_prompt" / "_shared" / "stage2-subagent-playbook.md"
        )
        self.assertTrue(
            playbook.exists(),
            f"shared playbook missing: {playbook}",
        )
        playbook_text = playbook.read_text(encoding="utf-8")
        for anchor in ("3a", "3b", "3c"):
            self.assertIn(anchor, playbook_text)
        for marker in (
            "OP_LONG: FAIL",
            "STILL_ITERATING",
            "READY_FOR_OP_LONG",
            "SAFETY_NET_TRIGGERED",
        ):
            self.assertIn(marker, playbook_text)
        # The merge wrapper + failure retention live in the playbook
        # alone (stage2.md only describes the main agent's dispatch
        # role).
        for anchor in (
            "stage2-main.lock",
            "register.toml",
            "git worktree remove --force",
        ):
            self.assertIn(anchor, playbook_text)
        # SSOT guards: the row-format / SHA-selection helpers from the
        # earlier explore-then-elect model must not regress back into
        # the playbook.
        self.assertNotIn("ROUND_REPORT round=<N>", playbook_text)
        self.assertNotIn("tools/round_report.py", playbook_text)
        stage2 = (REPO_ROOT / "prompt" / "develop_prompt" / "_shared" / "stage2.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("stage2-subagent-playbook.md", stage2)
        # Per-backend split files must not regress.
        for backend in ("megatron", "torch"):
            for stale in (
                "stage2.md",
                "stage2-subagent-playbook.md",
                "stage2-evidence-contract.md",
            ):
                stale_path = REPO_ROOT / "prompt" / "develop_prompt" / backend / stale
                self.assertFalse(
                    stale_path.exists(),
                    f"per-backend stage2 file should be consolidated into _shared/: {stale_path}",
                )

    def test_stage2_rules_document_dispatcher_wiring(self) -> None:
        text = (REPO_ROOT / "prompt" / "develop_prompt" / "_shared" / "stage2.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("workload/src/training_engine_tensor/", text)
        self.assertIn("get_op_version", text)
        # The dispatcher-wiring section freezes the relevant source
        # files to the subagent (originally written as 对 subagent 冻结
        # in the legacy zh prose; the consolidated shared markdown is
        # English and asserts this literally instead).
        self.assertIn("frozen", text)
        self.assertIn("subagent", text)

    def test_agent_loop_does_not_hardcode_suite_stage_lists(self) -> None:
        text = (REPO_ROOT / "agent-loop.sh").read_text(encoding="utf-8")
        # agent-loop.sh must surface the suite list via `harness info`
        # rather than embedding a literal `<stage>: <suite>` cheatsheet,
        # which would silently drift from config/eval.toml. Probe
        # one sentinel suite per current stage to keep the backstop
        # honest as the suite catalog evolves.
        self.assertIn("harness info", text)
        self.assertNotIn("stage1: forward-align", text)
        self.assertNotIn("stage1: resume-gate-20", text)
        self.assertNotIn("stage1: long-train", text)
        self.assertNotIn("stage2: op-inventory", text)

    def test_long_horizon_prompt_carries_optimization_ordering(self) -> None:
        # long-horizon used to present the optimization techniques as a flat
        # "mix freely" menu with no recommended order, which let agents
        # burn rounds on low-ROI tuning (e.g. recompute-layer fiddling)
        # while a profile already pointed at launch-bound fusion work.
        # The prompt now carries a default-prior ordering
        # (measure-enable → fuse → overlap → cuda-graph) that profile
        # evidence is explicitly allowed to override.
        text = (
            REPO_ROOT / "prompt" / "develop_prompt" / "_shared" / "stage1" / "long-horizon.md"
        ).read_text(encoding="utf-8")
        self.assertIn("Optimization ordering", text)
        # The four phase anchors (phrased uniquely to the ordering
        # section) must all be present and appear in order.
        positions = [
            text.index(anchor)
            for anchor in (
                "Phase 1 — measure-enable",
                "Phase 2 — fuse",
                "Phase 3 — overlap",
                "Phase 4 — capture",
            )
        ]
        self.assertEqual(positions, sorted(positions))
        # Ordering is a default prior, not a hard sequence: profile
        # evidence may override it (keeps profile-first authority).
        self.assertIn("profile may override", text)

    def test_op_prompts_do_not_keep_template_placeholders(self) -> None:
        """Per-op PROMPT.md must not retain ``<X_FROM_Y>`` placeholders.

        Stage 2 uses a single unified PROMPT.md template (M1.5.1 of
        stage2.md). Main agents write real numbers straight into
        PROMPT.md when scouting an operator. This test catches
        regressions whenever any ``workload/ops/<name>/PROMPT.md`` is
        present.
        """
        ops_dir = REPO_ROOT / "workload" / "ops"
        prompt_paths = sorted(ops_dir.glob("*/PROMPT.md")) if ops_dir.exists() else []
        if not prompt_paths:
            self.skipTest("no per-operator PROMPT.md present in this checkout")
        for path in prompt_paths:
            with self.subTest(path=path.relative_to(REPO_ROOT).as_posix()):
                text = path.read_text(encoding="utf-8")
                self.assertNotRegex(
                    text,
                    r"<[A-Z0-9_]+_FROM_[A-Z0-9_]+>",
                    "Stage 2 unified PROMPT template requires real numbers, "
                    "not <…_FROM_…> placeholders (see stage2.md M1.5.1).",
                )
                self.assertNotIn(
                    "5% 容忍",
                    text,
                    "Stage 2 contract does not use a 5%-tolerance gate.",
                )
                self.assertNotIn(
                    "mfu_e2e_standard=",
                    text,
                    "Stage 2 subagent benchmark uses a single `mfu_e2e` "
                    "single-card reference; do not reintroduce a "
                    "`mfu_e2e_standard=` twin knob.",
                )

    def test_bitwise_alignment_debugging_discipline(self) -> None:
        # Regression guard for the loop 5ed64c7b64bc post-mortem: a dev
        # round burned ~45 min chasing an RMSNorm saved-r 1-ULP diff that
        # turned out to be a downstream symptom of a RoPE-backward bug.
        # The fix-it-by-argument anti-pattern is countered by three
        # standing-rule anchors that must not silently regress.
        stage1 = REPO_ROOT / "prompt" / "develop_prompt" / "_shared" / "stage1"

        # A: the backend-agnostic discipline lives in constraint.md.
        constraint = (stage1 / "constraint.md").read_text(encoding="utf-8")
        self.assertIn("Bitwise alignment debugging discipline", constraint)
        self.assertIn("first divergence", constraint)
        self.assertIn("symptom map, not a diagnosis", constraint)

        # B: alignment.md points at it (only the alignment round loads alignment.md).
        alignment = (stage1 / "alignment.md").read_text(encoding="utf-8")
        self.assertIn("Bitwise alignment debugging discipline", alignment)

        # C: attribution must be earned by a fix — applies to both
        # perf_log.md entries and commit messages (overview.md loads
        # every round).
        overview = (stage1 / "overview.md").read_text(encoding="utf-8")
        self.assertIn("attribution must be earned by a fix", overview)

    def test_stage1_debug_md_loaded_every_round(self) -> None:
        # The stage1 debug protocol lives at _shared/stage1/debug.md and
        # is injected into every dev-round prompt via the ALWAYS
        # manifest, so the agent has the diagnostic protocol on hand
        # whenever a stage1 gate fails — regardless of which milestone
        # is currently active. Complements the constraint.md stub
        # (kept for backward compatibility) and overview.md's
        # "attribution earned by fix" rule.
        from tools.agent_loop_config import _STAGE_DIR_MANIFEST_ALWAYS

        self.assertIn("debug.md", _STAGE_DIR_MANIFEST_ALWAYS["stage1"])

        path = REPO_ROOT / "prompt" / "develop_prompt" / "_shared" / "stage1" / "debug.md"
        self.assertTrue(path.exists(), f"missing: {path}")
        text = path.read_text(encoding="utf-8")
        # Anchor phrases that capture the protocol's load-bearing ideas.
        for anchor in (
            "trusted set",
            "suspect set",
            "halve the suspect set",
            "predicted signal",
            "symptom map",
        ):
            self.assertIn(anchor, text)

    def test_op_baseline_contract_is_not_todo(self) -> None:
        """Every ``workload/ops/<name>/BASELINE.md`` must be filled with
        real numbers (no ``TODO`` stubs). Stage 2 only optimises
        FlashAttention + the various GEMM call-sites; check whichever of
        those scaffolds the current checkout has produced.
        """
        ops_dir = REPO_ROOT / "workload" / "ops"
        baseline_paths = sorted(ops_dir.glob("*/BASELINE.md")) if ops_dir.exists() else []
        if not baseline_paths:
            self.skipTest("no workload/ops/<op>/BASELINE.md present in this checkout")
        for path in baseline_paths:
            with self.subTest(path=path.relative_to(REPO_ROOT).as_posix()):
                self.assertNotIn("TODO", path.read_text(encoding="utf-8"))

    def test_no_workflow_keyword_in_standing_rule_surfaces(self) -> None:
        """Claude Code's Workflow tool fires a system-reminder whenever the
        literal substring ``workflow`` / ``workflows`` appears in the
        conversation. Every dev / review agent in this loop is a Claude
        Code instance, so a stray occurrence in a file the agent loads
        (CLAUDE.md, overview.md, remote-execution.md, etc.) misfires that
        reminder and biases the agent toward "this is a multi-phase
        orchestration task" mental model — even after the Workflow tool
        crashes on launch, the phased framing persists and shows up as
        round-1 shipping a "first slice" instead of closing the milestone.

        This regression guard pins the rename: no standing rule surface
        the dev or review agent loads may contain the literal token.
        Use alternatives like ``procedure`` / ``flow`` / ``path`` for
        prose, ``Per-round procedure`` for the overview section heading,
        and ``Bisection Decision Tree`` for the megatron textbook
        diagnostic chapter.
        """
        repo = REPO_ROOT.parent  # REPO_ROOT is .../harness; one level up is the project root

        surfaces = [
            # CLAUDE.md / AGENTS.md / .github/copilot-instructions.md and
            # .cursor/rules/bootstrap-guide.mdc are auto-generated from
            # .rules/bootstrap-guide.md (scripts/sync_project_guide.py).
            # We check the SSOT plus every generated mirror so a manual
            # edit cannot silently re-introduce the keyword on one mirror.
            repo / ".rules" / "bootstrap-guide.md",
            repo / "CLAUDE.md",
            repo / "AGENTS.md",
            repo / ".github" / "copilot-instructions.md",
            repo / ".cursor" / "rules" / "bootstrap-guide.mdc",
            # Dev prompt files: overview.md is in the ALWAYS manifest, the
            # per-milestone <name>.md / debug.md / constraint.md are loaded each round.
            REPO_ROOT / "prompt" / "develop_prompt" / "_shared" / "stage1" / "overview.md",
            REPO_ROOT / "prompt" / "develop_prompt" / "_shared" / "stage1" / "constraint.md",
            REPO_ROOT / "prompt" / "develop_prompt" / "_shared" / "stage1" / "debug.md",
            # Remote-execution overlay (ssh / devspace runs) plus the
            # megatron-only bitwise textbook (loaded when backend=megatron).
            REPO_ROOT / "prompt" / "develop_prompt" / "remote-execution.md",
            REPO_ROOT
            / "prompt"
            / "develop_prompt"
            / "megatron"
            / "reference"
            / "megatron_bitwise_aligenment_textbook.md",
            # Review agent prompts.
            REPO_ROOT / "prompt" / "review_prompt" / "review_common.md",
            REPO_ROOT / "prompt" / "review_prompt" / "review_stage1.md",
        ]

        offenders: list[str] = []
        for path in surfaces:
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            if "workflow" in text.lower():
                # report line numbers so the fixer knows exactly where
                lines = [
                    f"  {path.relative_to(repo)}:{i + 1}: {line.rstrip()}"
                    for i, line in enumerate(text.splitlines())
                    if "workflow" in line.lower()
                ]
                offenders.extend(lines)
        self.assertFalse(
            offenders,
            "Found 'workflow' / 'workflows' in a standing rule surface "
            "the dev or review agent loads. This triggers Claude Code's "
            "Workflow-tool keyword reminder and biases the agent toward "
            "phased orchestration. Rename to 'procedure' / 'flow' / 'path' "
            "or use a context-specific synonym. Offenders:\n" + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
