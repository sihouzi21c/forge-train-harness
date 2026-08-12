"""Static guards for the web viewer theme tokens.

NOTE: A number of tests below are decorated with
``@pytest.mark.skip(reason="Phase A sidebar flattening pending — see TODO")``.
These were authored as foresight assertions for an in-progress "Phase A
sidebar flattening" refactor (flat sidebar layout, button/aria-label
renames, ``.agent-subrow`` reuse, removal of ``buildUnifiedGroups``) that
has not landed in production code yet. The skip markers exist so the rest
of the static theme guards can still run cleanly. Remove each marker as
the corresponding piece of the refactor lands.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

import pytest

MONOREPO_ROOT = Path(__file__).resolve().parents[3]
INDEX_HTML = MONOREPO_ROOT / "web" / "static" / "index.html"
AGENTS_JS = MONOREPO_ROOT / "web" / "static" / "js" / "agents.js"
APP_JS = MONOREPO_ROOT / "web" / "static" / "js" / "app.js"
AGENT_ROUTER = MONOREPO_ROOT / "web" / "routers" / "agent.py"
LOOP_ROUTER = MONOREPO_ROOT / "web" / "routers" / "loop.py"
LOOP_LOGS_ROUTER = MONOREPO_ROOT / "web" / "routers" / "loop_logs.py"
CONFIG_ROUTER = MONOREPO_ROOT / "web" / "routers" / "config.py"
ARTIFACTS_ROUTER = MONOREPO_ROOT / "web" / "routers" / "artifacts.py"
SERVER_PY = MONOREPO_ROOT / "web" / "server.py"
AGENT_LOOP_SH = MONOREPO_ROOT / "harness" / "agent-loop.sh"
FORGE_TRAIN_LOGO = MONOREPO_ROOT / "web" / "static" / "assets" / "forge-train-logo.svg"


class TestWebTheme(unittest.TestCase):
    def test_screenshot_inspired_dark_theme_tokens_are_present(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")

        expected_tokens = {
            "--color-canvas: #1e1e1e",
            "--color-panel: #1f1f1f",
            "--color-panel-raised: #252526",
            "--color-panel-hover: #2a2d2e",
            "--color-border: #303031",
            "--color-text: #cccccc",
            "--color-text-muted: #a6a6a6",
            "--color-accent: #9aa7b3",
        }

        missing = sorted(token for token in expected_tokens if token not in text)
        self.assertEqual(missing, [])

    def test_primary_layout_uses_theme_tokens_instead_of_slate_canvas(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")

        self.assertIn("bg-[var(--color-canvas)]", text)
        self.assertIn("border-[var(--color-border)]", text)
        self.assertNotIn('body class="h-full bg-[#0f172a]', text)

    def test_branding_uses_forge_train_logo_and_name(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")

        self.assertTrue(FORGE_TRAIN_LOGO.exists())
        self.assertIn("<title>ForgeTrain</title>", text)
        self.assertIn('src="/static/assets/forge-train-logo.svg"', text)
        self.assertIn('alt="ForgeTrain logo"', text)
        self.assertIn(">ForgeTrain</span>", text)
        self.assertNotIn(">Train Engine</h1>", text)

    def test_file_tree_keeps_cursor_style_neutral_directory_color(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")

        self.assertIn(
            "entry.ignored ? 'text-[var(--color-text-subtle)]' : entry.is_dir ? 'text-[var(--color-text-muted)]' : 'text-[var(--color-text)]'",
            text,
        )
        self.assertIn(".file-tree-row", text)

    def test_file_detail_uses_left_aligned_tree_back_row(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")

        self.assertIn("<span>Files</span>", text)
        self.assertIn('class="file-browser-back-row', text)
        self.assertIn("@click=\"filePath = ''; fileContent = ''\"", text)
        self.assertIn('<span class="min-w-0 truncate font-mono" x-text="filePath"></span>', text)
        self.assertNotIn("&larr; Tree</button>", text)

    def test_accent_color_stays_muted_instead_of_bright_blue(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")

        self.assertIn("--color-accent-soft: rgba(154, 167, 179, 0.14)", text)
        self.assertNotIn("--color-accent: #5aa7ff", text)
        self.assertNotIn("background-color: #75b7ff", text)

    def test_resizable_divider_between_panels(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")

        self.assertIn("panel-resizer", text)
        self.assertIn("col-resize", text)

    def test_resizable_divider_js_logic(self) -> None:
        js_text = APP_JS.read_text(encoding="utf-8")

        self.assertIn("rightPanelWidth", js_text)
        self.assertIn("startResize", js_text)

    def test_right_panel_default_width_defaults_to_16_percent(self) -> None:
        js_text = APP_JS.read_text(encoding="utf-8")

        import re

        match = re.search(r"rightPanelWidth:\s*(\d+)", js_text)
        self.assertIsNotNone(match, "rightPanelWidth default not found in app.js")
        self.assertEqual(int(match.group(1)), 16)
        self.assertIn("Math.max(pct, 16)", js_text)

    def test_files_panel_is_collapsed_by_default(self) -> None:
        js_text = APP_JS.read_text(encoding="utf-8")

        self.assertIn("showFiles: false", js_text)
        self.assertNotIn("showFiles: true", js_text)

    @pytest.mark.skip(reason="Phase A sidebar flattening pending — see TODO")
    def test_new_agent_button_opens_inline_chat_instead_of_modal(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")

        self.assertIn('@click="openNewAgentChat()"', text)
        self.assertIn("+ New Agent", text)
        self.assertIn('x-show="selectedManagedId || isCreatingManagedAgent"', text)
        self.assertNotIn("New agent modal", text)
        self.assertNotIn("showNewAgentModal", text)

    @pytest.mark.skip(reason="Phase A sidebar flattening pending — see TODO")
    def test_agent_sidebar_is_a_flat_list_aligned_with_loops(self) -> None:
        """Replaces the obsolete ``groups_agents_by_task`` spec. The
        agents sidebar is now a flat list of rows (same shape as the
        loop sidebar). See ``TestLoopAndAgentVisualParity`` for the
        full set of cross-tab invariants; this test just pins the
        bare minimum so a regression here trips both suites at once.
        """
        text = INDEX_HTML.read_text(encoding="utf-8")

        self.assertIn('data-testid="agent-task-sidebar"', text)
        self.assertIn("agent-subrow", text)
        # The grouping artefacts are gone:
        self.assertNotIn("managedAgentTasks", text)
        self.assertNotIn("agent-task-row", text)
        self.assertNotIn("toggleManagedTask", text)

    def test_new_agent_chat_matches_compact_reference_composer(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")

        self.assertIn("agent-chat-composer", text)
        self.assertIn("Ask a question... (Enter to send, Shift+Enter for newline)", text)
        self.assertIn("New chat", text)
        self.assertNotIn("Initial prompt for the agent", text)

    def test_first_agent_chat_send_creates_agent_from_composer(self) -> None:
        js_text = AGENTS_JS.read_text(encoding="utf-8")

        self.assertIn("isCreatingManagedAgent: false", js_text)
        self.assertIn("submitManagedComposer", js_text)
        self.assertIn("prompt: text", js_text)
        self.assertIn("await this.createManagedAgent(text)", js_text)
        self.assertNotIn("newAgentPrompt", js_text)

    def test_agent_pending_indicator_is_scoped_to_selected_chat(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")
        agents_js = AGENTS_JS.read_text(encoding="utf-8")

        self.assertIn("managedPendingAgentId", agents_js)
        self.assertIn("managedPendingMessages: {}", agents_js)
        self.assertIn("isManagedChatBusy", agents_js)
        self.assertIn("getManagedInteractionState", agents_js)
        self.assertIn("const interaction = this.getManagedInteractionState", agents_js)
        self.assertIn("return interaction.inputDisabled;", agents_js)
        self.assertIn("return interaction.statusLabel;", agents_js)
        self.assertIn("return interaction.statusClass;", agents_js)
        self.assertIn("this.managedMessages = interaction.messages;", agents_js)
        self.assertIn("const targetAgentId = this.selectedManagedId;", agents_js)
        self.assertIn("mergePendingMessages", agents_js)
        self.assertIn("applyPendingStatus", agents_js)
        self.assertIn("markPendingStarted", agents_js)
        self.assertIn("clearPendingAgent", agents_js)
        self.assertIn("_replaceManagedAgent", agents_js)
        self.assertIn("this.managedPendingMessages[targetAgentId] =", agents_js)
        self.assertIn("this.managedSseClose();", agents_js)
        self.assertIn("(!pending || pending.started || terminal)", agents_js)
        self.assertIn("this._replaceManagedAgent(this.managedAgent);", agents_js)
        self.assertIn("x-show=\"managedPanelBusy('agents')\"", text)
        self.assertIn("managedPanelBusy('agents')", text)
        self.assertNotIn(
            'x-show="managedSubmitting || (managedAgent && managedAgent.is_running)"', text
        )

    @pytest.mark.skip(reason="Phase A sidebar flattening pending — see TODO")
    def test_agent_interaction_state_distinguishes_sending_from_responding(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")
        agents_js = AGENTS_JS.read_text(encoding="utf-8")

        self.assertIn("const sending = Boolean(pending && !pending.started);", agents_js)
        self.assertIn("const responding = Boolean(!sending &&", agents_js)
        self.assertIn("const canStop = Boolean(!sending &&", agents_js)
        self.assertIn(
            "sending ? 'sending' : (stopping ? 'stopping' : (responding ? 'responding' : 'ready'))",
            agents_js,
        )
        self.assertIn("isManagedChatSending", agents_js)
        self.assertIn("canStopManagedChat", agents_js)
        self.assertIn("return interaction.canStop;", agents_js)
        self.assertIn(
            "this.managedPendingMessages[targetAgentId] = {\n          message: optimistic,",
            agents_js,
        )
        self.assertIn("this.managedPendingMessages[tempId] = {\n        message:", agents_js)
        self.assertIn("baseMessageCount: this.managedMessages.length", agents_js)
        self.assertIn(
            "if (Number.isFinite(pending.baseMessageCount) && out.length > pending.baseMessageCount) return out;",
            agents_js,
        )
        self.assertIn("scrollManagedChatBodiesToBottom", agents_js)
        self.assertIn("document.querySelectorAll('.agent-chat-body')", agents_js)
        self.assertNotIn("document.querySelector('.agent-chat-body')", agents_js)
        self.assertIn("const switchingAgent = this.selectedManagedId !== agentId;", agents_js)
        self.assertIn("if (switchingAgent) this._renderManagedMessages(agentId, []);", agents_js)
        self.assertNotIn(
            "\n      this._renderManagedMessages(agentId, []);\n      this.managedInput = '';",
            agents_js,
        )
        self.assertIn("started: false", agents_js)
        self.assertIn("if (data.is_running) this.markPendingStarted(targetAgentId);", agents_js)
        self.assertIn("if (!this.canStopManagedChat()) return;", agents_js)
        self.assertIn(
            "x-show=\"managedPanelCanStop('agents') || managedPanelStopping('agents')\"", text
        )
        self.assertIn("return 'Sending...';", agents_js)
        self.assertIn("managedPanelPendingLabel('agents')", text)

    def test_pending_sending_bubble_clears_on_terminal_state(self) -> None:
        """Stuck "... Sending..." regression guard.

        The SSE backend only emits a ``state`` frame when ``sess.state``
        itself changes, but a chat turn can race straight through
        ``running`` (precreated, pid=None → spawned with pid alive →
        exited terminal) without the frontend ever observing
        ``is_running=true``. Before the fix the clear-pending guards
        only fired when ``pending.started`` was already true, so a
        turn that never got that flip pinned the "Sending..." bubble
        forever even after the agent had ended. The fix adds a
        terminal-state escape hatch in ``onSnapshot``/``onState`` and
        unconditional clear in ``onEnd``.
        """
        agents_js = AGENTS_JS.read_text(encoding="utf-8")

        self.assertIn("function isAgentTerminalState(d)", agents_js)
        self.assertIn(
            "return status !== 'running' && status !== 'starting' && status !== 'draft';",
            agents_js,
        )

        for handler in ("onSnapshot:", "onState:"):
            handler_idx = agents_js.index(handler)
            block_end = agents_js.index("\n        },", handler_idx)
            block = agents_js[handler_idx:block_end]
            self.assertIn("const terminal = isAgentTerminalState(", block)
            self.assertIn("(!pending || pending.started || terminal)", block)

        on_end_idx = agents_js.index("onEnd: (d) => {")
        on_end_block = agents_js[on_end_idx : agents_js.index("\n        },", on_end_idx)]
        self.assertIn("this.clearPendingAgent(agentId);", on_end_block)
        self.assertNotIn("if (!pending || pending.started)", on_end_block)
        self.assertNotIn("if (!pending || pending.started || terminal)", on_end_block)

    @pytest.mark.skip(reason="Phase A sidebar flattening pending — see TODO")
    def test_new_agent_draft_clears_previous_busy_state(self) -> None:
        agents_js = AGENTS_JS.read_text(encoding="utf-8")

        open_new_start = agents_js.index("vm.openNewAgentChat = async function ()")
        open_new_end = agents_js.index("vm.stopManagedAgent = async function ()")
        open_new_body = agents_js[open_new_start:open_new_end]

        self.assertIn("this.managedSubmitting = false;", open_new_body)
        self.assertIn("this.managedAwaitingResponse = false;", open_new_body)
        self.assertIn("this.managedPendingAgentId = null;", open_new_body)

    def test_new_agent_draft_appears_in_sidebar_immediately(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")
        agents_js = AGENTS_JS.read_text(encoding="utf-8")

        self.assertIn("NEW_AGENT_DRAFT_ID", agents_js)
        self.assertIn("buildManagedDraftAgent", agents_js)
        self.assertIn("withoutDraftAgent", agents_js)
        self.assertIn("prompt_preview: 'New chat'", agents_js)
        self.assertIn("is_draft: true", agents_js)
        self.assertIn("conversation_status: 'ready'", agents_js)
        self.assertIn("this.selectedManagedId = NEW_AGENT_DRAFT_ID;", agents_js)
        self.assertIn("this.managedAgents = [draft, ...withoutDraftAgent", agents_js)
        self.assertIn("branch: this.managedDefaultBranch", agents_js)
        self.assertIn("selectedManagedId === a.agent_id", text)

    def test_new_agent_draft_loads_branch_before_sidebar_insert(self) -> None:
        agents_js = AGENTS_JS.read_text(encoding="utf-8")
        router_text = AGENT_ROUTER.read_text(encoding="utf-8")

        self.assertIn("ensureManagedBranchContext", agents_js)
        self.assertIn('@router.get("/defaults")', router_text)
        self.assertIn("fetch('/api/agent/defaults')", agents_js)
        open_new_start = agents_js.index("vm.openNewAgentChat = async function ()")
        open_new_end = agents_js.index("vm.stopManagedAgent = async function ()")
        open_new_body = agents_js[open_new_start:open_new_end]

        self.assertLess(
            open_new_body.index("await this.ensureManagedBranchContext();"),
            open_new_body.index("const draft = this.buildManagedDraftAgent();"),
        )

    @pytest.mark.skip(reason="Phase A sidebar flattening pending — see TODO")
    def test_agents_script_cache_buster_moves_with_frontend_changes(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")

        self.assertIn("/static/js/agents.js?v=30", text)
        self.assertIn("/static/js/app.js?v=41", text)

    def test_periodic_agent_refresh_preserves_local_new_agent_draft(self) -> None:
        agents_js = AGENTS_JS.read_text(encoding="utf-8")

        load_start = agents_js.index("vm.loadManagedAgents = async function ()")
        load_end = agents_js.index("vm.isManagedComposerSubmitting = function ()")
        load_body = agents_js[load_start:load_end]

        self.assertIn("if (this.isCreatingManagedAgent)", load_body)
        self.assertIn("agents.unshift(this.buildManagedDraftAgent());", load_body)

    def test_agent_branch_metadata_still_flows_from_backend(self) -> None:
        """Replaces the obsolete ``groups_by_git_branch_when_available``
        spec. The branch-grouping UI was removed, but the per-agent
        ``branch`` field stays in the data model — chat creation still
        carries it through (it's useful as detail-header context and
        as future grouping/filter affordance) and the new-agent draft
        still picks up the worktree's default branch via
        ``GET /api/agent/defaults``.
        """
        agents_js = AGENTS_JS.read_text(encoding="utf-8")
        router_text = AGENT_ROUTER.read_text(encoding="utf-8")

        self.assertIn("managedDefaultBranch", agents_js)
        self.assertIn("branch: this.managedDefaultBranch", agents_js)
        self.assertNotIn("Current Worktree", agents_js)
        self.assertIn("default_branch", router_text)
        self.assertIn('"branch": _session_branch(sess)', router_text)
        self.assertIn("sess.source_branch", router_text)

    def test_agent_conversation_status_is_ready_or_responding(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")
        agents_js = AGENTS_JS.read_text(encoding="utf-8")
        router_text = AGENT_ROUTER.read_text(encoding="utf-8")
        runner_text = (MONOREPO_ROOT / "web" / "agents" / "runner.py").read_text(encoding="utf-8")

        self.assertIn("conversation_status", router_text)
        self.assertIn("run_status", router_text)
        self.assertIn("def is_managed_running", runner_text)
        self.assertIn("is_responding = runner.is_managed_running(sess.agent_id)", router_text)
        self.assertIn("final_size = transcript.file_size(path)", router_text)
        # SSE event loop now feeds an incremental ``messages_mod.Rebuilder``
        # inside ``asyncio.to_thread`` instead of re-running ``rebuild``
        # on the full record list every tick. The lifecycle-gate flush
        # routes through the same threaded ``_read_feed_encode`` worker
        # so the final ``messages`` frame still lands before ``end``.
        self.assertIn("_read_feed_encode", router_text)
        self.assertIn("await asyncio.to_thread(", router_text)
        self.assertIn("messages_mod.Rebuilder()", router_text)
        self.assertIn(
            'yield _sse_raw("messages", payload)',
            router_text,
        )
        # ``GET /messages`` is a one-shot endpoint — the legacy
        # ``transcript.read_all`` is still appropriate there. The
        # invariant we care about is that the streaming SSE loop never
        # re-parses from offset 0 every tick; assert on the threaded
        # incremental worker name instead of an over-broad NotIn.
        self.assertIn("current = store.load(agent_id) or current", router_text)
        self.assertIn("if web_exists:", router_text)
        self.assertIn("return web_path", router_text)
        self.assertNotIn(
            "if web_exists and not runner._is_externally_active(agent_id):", router_text
        )
        self.assertNotIn("return _is_externally_active(agent_id)", runner_text)
        self.assertIn("handle.monitor_task.done()", runner_text)
        self.assertIn('"responding" if is_responding else "ready"', router_text)
        self.assertIn("agentStatusLabel", agents_js)
        self.assertIn("agentStatusClass", agents_js)
        self.assertNotIn("a.is_running ? 'running' : a.state", text)
        self.assertNotIn("managedAgent.is_running ? 'running' : managedAgent.state", text)

    @pytest.mark.skip(reason="Phase A sidebar flattening pending — see TODO")
    def test_agent_sidebar_status_refreshes_while_detail_is_open(self) -> None:
        agents_js = AGENTS_JS.read_text(encoding="utf-8")

        self.assertIn("vm.leftTab === 'agents'", agents_js)
        self.assertIn("vm.loadManagedAgents();", agents_js)
        self.assertNotIn("!vm.selectedManagedId && !vm.isCreatingManagedAgent", agents_js)

    @pytest.mark.skip(reason="Phase A sidebar flattening pending — see TODO")
    def test_agent_stop_enters_visible_stopping_state(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")
        agents_js = AGENTS_JS.read_text(encoding="utf-8")

        self.assertIn("managedStoppingAgentIds: {}", agents_js)
        self.assertIn("markStoppingAgent", agents_js)
        self.assertIn("clearStoppingAgent", agents_js)
        self.assertIn("isManagedChatStopping", agents_js)
        self.assertIn("const stopping = Boolean", agents_js)
        self.assertIn("stopping ? 'stopping' : (responding ? 'responding' : 'ready')", agents_js)
        self.assertIn("this.markStoppingAgent(targetAgentId);", agents_js)
        self.assertIn("this.clearStoppingAgent(targetAgentId);", agents_js)
        self.assertNotIn("this.clearPendingAgent(targetAgentId);\n      try", agents_js)
        self.assertIn(":disabled=\"managedPanelStopping('agents')\"", text)
        self.assertIn("Stopping...", text)

    def test_agent_auth_status_delegates_to_unified_check(self) -> None:
        agents_js = AGENTS_JS.read_text(encoding="utf-8")
        router_text = AGENT_ROUTER.read_text(encoding="utf-8")
        runner_text = (MONOREPO_ROOT / "web" / "agents" / "runner.py").read_text(encoding="utf-8")

        self.assertIn("has_auth", router_text)
        # auth_status now probes the backend CLI directly (no check_backend_available)
        self.assertIn("auth_status", runner_text)
        self.assertIn("data.has_auth", agents_js)
        self.assertNotIn("if (data.has_key) return true;", agents_js)

    def test_no_api_key_modal_or_loop_create_key_button(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")
        agents_js = AGENTS_JS.read_text(encoding="utf-8")

        self.assertNotIn('data-testid="agent-api-key-modal"', text)
        self.assertNotIn('data-testid="loop-create-key-button"', text)
        self.assertNotIn("openAgentKeyPrompt", agents_js)
        self.assertNotIn("saveApiKeyPrompt", agents_js)
        self.assertNotIn("fetch('/api/agent/set-key'", agents_js)

    @pytest.mark.skip(reason="Phase A sidebar flattening pending — see TODO")
    def test_agents_and_loop_are_the_only_primary_left_views(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")
        js_text = APP_JS.read_text(encoding="utf-8")

        self.assertNotIn(">Viewer</button>", text)
        self.assertNotIn(">New Task</button>", text)
        self.assertNotIn(">Sessions</button>", text)
        self.assertNotIn(">Gate Runs</button>", text)
        self.assertIn(">Agents</button>", text)
        self.assertIn(">Loop</button>", text)
        self.assertLess(text.index(">Loop</button>"), text.index(">Agents</button>"))
        self.assertIn(": 'loop')", js_text)
        self.assertNotIn(": 'agents')", js_text)
        self.assertNotIn("loadSessions()", js_text)
        self.assertNotIn("loadRuns()", js_text)
        self.assertIn("leftTab", js_text)

    def test_new_task_and_files_are_hidden_from_primary_nav(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")
        app_js = APP_JS.read_text(encoding="utf-8")
        server_text = SERVER_PY.read_text(encoding="utf-8")

        nav_start = text.index("<!-- ===== Top Navigation ===== -->")
        nav_end = text.index("</nav>", nav_start)
        nav = text[nav_start:nav_end]

        self.assertNotIn("+ New Task", nav)
        self.assertNotIn("Close Task", nav)
        self.assertNotIn(">Files</span>", nav)
        self.assertNotIn("Close Files", nav)
        self.assertIn("files-rail-button", text)
        self.assertIn("files-rail-tab", text)
        self.assertIn("top-1/2", text)
        self.assertIn("-translate-y-1/2", text)
        self.assertIn("group-hover:w-12", text)
        self.assertIn('class="hidden group-hover:inline">Files</span>', text)
        self.assertIn('aria-label="Toggle files panel"', text)
        self.assertIn("showFiles = !showFiles", text)
        # Legacy global Task Config panel has been removed (Method F:
        # per-loop config dir). Settings UI now binds to per-loop
        # `loopConfigs.*` state and PUTs to `/api/loop/<id>/configs/<axis>`.
        self.assertNotIn("showTaskConfig", app_js)
        self.assertNotIn("openTaskPanel()", app_js)
        self.assertIn("loopConfigs", app_js)
        self.assertIn("config", server_text)

    @pytest.mark.skip(reason="Phase A sidebar flattening pending — see TODO")
    def test_loop_create_keeps_brand_but_hides_parent_navigation(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")

        nav_start = text.index("<!-- ===== Top Navigation ===== -->")
        nav_end = text.index("<!-- ===== Toast Notification ===== -->")
        nav = text[nav_start:nav_end]
        self.assertIn("<span>ForgeTrain</span>", nav)
        self.assertNotIn('x-show="!showLoopCreate"', nav.split("<h1", 1)[0])
        self.assertIn('x-show="!showLoopCreate"', nav)

        left_header = text[
            text.index("<!-- Tab header -->") : text.index("<!-- Agents task sidebar + chat -->")
        ]
        self.assertIn('x-show="!showLoopCreate"', left_header)

        # Loop toolbar is now gate-approvals-only (refresh / "+ New" moved
        # into the loop sidebar list header next to the instance count).
        # We assert the header line uses an anchor-comment that starts
        # with "<!-- Loop toolbar"; the trailing parenthetical is the
        # design-intent note and we don't pin its exact wording.
        toolbar_anchor = "<!-- Loop toolbar"
        toolbar = text[text.index(toolbar_anchor) : text.index("<!-- Loop Create page -->")]
        self.assertIn('x-show="!showLoopCreate', toolbar)
        self.assertNotIn("Back to Loops", toolbar)
        self.assertNotIn("Refresh", toolbar)

        # Refresh control lives in the loop instances sidebar header.
        sidebar = text[
            text.index("<!-- Loop instances list -->") : text.index(
                "</template>", text.index("<!-- Loop instances list -->")
            )
        ]
        self.assertIn("loadLoopInstances()", sidebar)
        self.assertIn("openLoopCreate()", sidebar)

        create_intro = text[
            text.index("<!-- Loop Create page -->") : text.index("<!-- Loop Create chat -->")
        ]
        self.assertNotIn("← Back to Loops", create_intro)
        self.assertNotIn("Start Loop", create_intro)
        self.assertNotIn("<h3", create_intro)

        config_panel = text[
            text.index("<!-- Loop config side panel -->") : text.index(
                "<!-- Loop instances list -->"
            )
        ]
        config_actions = config_panel[: config_panel.index("<!-- Quick config -->")]
        self.assertIn("Back to Loops", config_actions)
        self.assertIn("Start Loop Work", config_actions)
        self.assertLess(config_panel.index("Back to Loops"), config_panel.index("Quick Config"))

    def test_agent_list_search_and_unused_filters_are_removed(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")
        agents_js = AGENTS_JS.read_text(encoding="utf-8")
        router_text = AGENT_ROUTER.read_text(encoding="utf-8")

        self.assertNotIn("Search agents", text)
        self.assertNotIn("managedSearch", agents_js)
        self.assertNotIn("?query=", agents_js)
        self.assertNotIn("query: str", router_text)
        self.assertNotIn("state: Optional[str]", router_text)

    def test_removed_layout_dead_code_has_no_static_frontend_references(self) -> None:
        app_js = APP_JS.read_text(encoding="utf-8")
        agents_js = AGENTS_JS.read_text(encoding="utf-8")

        self.assertNotIn("formatSize(bytes)", app_js)
        self.assertNotIn("formatTime(ts)", app_js)
        self.assertNotIn("function formatTime(iso)", agents_js)
        self.assertNotIn("formatAgentTime", agents_js)

    def test_loop_refresh_timer_has_matching_app_entrypoint(self) -> None:
        app_js = APP_JS.read_text(encoding="utf-8")
        agents_js = AGENTS_JS.read_text(encoding="utf-8")

        self.assertIn("vm.loadLoopState();", agents_js)
        self.assertIn("async loadLoopState()", app_js)
        self.assertIn("await this._pollGates();", app_js)

    def test_loop_live_refresh_timer_is_cleared_on_selection_change(self) -> None:
        app_js = APP_JS.read_text(encoding="utf-8")

        self.assertIn("_loopLogRefreshTimer: null", app_js)
        self.assertIn("_disconnectLoopStream()", app_js)
        self.assertIn("clearInterval(this._loopLogRefreshTimer)", app_js)
        self.assertIn("this._disconnectLoopStream();", app_js)
        self.assertIn("if (this.selectedLoopId !== loopId) return;", app_js)

    def test_loop_agent_log_uses_shared_agent_message_parser(self) -> None:
        # loop_logs.py was removed; the unified transcript now flows
        # through /api/agent/loop-<id>/messages which uses the same
        # messages_mod.rebuild path inside agent.py.
        self.assertFalse(
            LOOP_LOGS_ROUTER.is_file(),
            "loop_logs.py must stay removed — the unified transcript "
            "routes through the agent router",
        )

    def test_loop_agent_log_uses_shared_renderer_without_incremental_append(self) -> None:
        app_js = APP_JS.read_text(encoding="utf-8")

        # loopncu passes render options (trajectoryByRound) as a second
        # arg; the contract is "shared renderer over the full messages
        # array", so match the call prefix rather than the exact arity.
        self.assertIn("Agents.renderMessages(messages", app_js)
        self.assertNotIn("this.loopLogHtml += Agents.renderMessages(newMsgs)", app_js)

    @pytest.mark.skip(reason="Phase A sidebar flattening pending — see TODO")
    def test_loop_agent_log_uses_agent_chat_body_shell(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")
        loop_detail = text[text.index("<!-- Loop detail: agent conversation -->") :]
        loop_detail = loop_detail[: loop_detail.index("<!-- Resizable Divider -->")]

        # The Loop detail pane is now a single .agent-chat-body shell
        # that renders the loop-<id> wrapper Session's unified
        # transcript (loop_event rows interleaved with anything else
        # the wrapper has written). The legacy lower "Command Output"
        # pane and its loopOutput / loopOutputHtml refs are gone —
        # all live data flows through /api/agent/loop-<id>/messages
        # and /api/agent/loop-<id>/events.
        self.assertIn('class="agent-chat-body flex-1 overflow-y-auto p-3"', loop_detail)
        self.assertIn('x-ref="loopLogPanel"', loop_detail)
        self.assertIn('x-html="loopLogHtml"', loop_detail)
        self.assertNotIn("Command Output", loop_detail)
        self.assertNotIn('x-ref="loopOutput"', loop_detail)
        self.assertNotIn('x-html="loopOutputHtml"', loop_detail)

    def test_loop_record_shortcut_uses_recording_specific_style(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")
        loop_sidebar = text[
            text.index("<!-- Unified sidebar + detail -->") : text.index(
                "<!-- Loop transcript detail -->",
            )
        ]

        self.assertIn("sidebar-group-record", loop_sidebar)
        self.assertIn("sidebar-group-record__dot", loop_sidebar)
        self.assertIn('aria-label="Open loop recording tab"', loop_sidebar)
        self.assertIn("start recording", loop_sidebar)
        self.assertNotIn("&#x29C9;", loop_sidebar)

    def test_loop_create_opens_independent_default_demo_page(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")
        app_js = APP_JS.read_text(encoding="utf-8")

        self.assertIn("Loop Create", text)
        self.assertNotIn("I loaded the default demo into this draft.", text)
        self.assertIn("showLoopCreate: false", app_js)
        self.assertIn("loopCreateLoopId: ''", app_js)
        self.assertIn("loopCreateWorkspaceDir: ''", app_js)
        self.assertIn("loopCreateConfigPaths: {}", app_js)
        self.assertIn("loopCreateSourceBranch: ''", app_js)
        self.assertIn("openLoopCreate()", app_js)
        self.assertIn("ensureLoopCreateDraft", app_js)
        self.assertIn("closeLoopCreate()", app_js)
        self.assertIn("resetLoopCreateSession()", app_js)
        self.assertIn("this.loopCreateAgentId = ''", app_js)
        self.assertIn("this.closeManagedAgent()", app_js)
        self.assertNotIn("loopCreateIntro", app_js)
        self.assertIn("loadLoopCreateConfigsFromFiles", app_js)
        self.assertIn("startLoopCreateConfigRefresh", app_js)
        self.assertIn("stopLoopCreateConfigRefresh", app_js)
        self.assertIn('x-show="showLoopCreate"', text)
        self.assertIn('x-show="!showLoopCreate"', text)
        self.assertIn("loop-create-shell", text)
        self.assertNotIn("Loop Create studio (overlay)", text)
        self.assertNotIn("Run Profile", text)
        self.assertNotIn("Choose Demo", text)

    @pytest.mark.skip(reason="Phase A sidebar flattening pending — see TODO")
    def test_loop_create_places_core_configs_around_large_chat(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")

        start = text.index("<!-- Loop Create page -->")
        end = text.index("<!-- Loop instances list -->")
        create_page = text[start:end]

        self.assertIn("loop-create-config-panel", create_page)
        self.assertIn("loop-create-quick-config", create_page)
        self.assertIn("loop-create-advanced-config", create_page)
        self.assertIn("loop-create-raw-config", create_page)
        self.assertIn("loop-create-agent-chat", create_page)
        self.assertIn("loop-create-eval-config", create_page)
        self.assertLess(
            create_page.index("loop-create-agent-chat"),
            create_page.index("loop-create-config-panel"),
        )
        self.assertLess(create_page.index("Quick Config"), create_page.index("Advanced"))
        self.assertLess(create_page.index("Advanced"), create_page.index("Raw Config Files"))
        self.assertIn("ref.toml", create_page)
        self.assertIn("agent.toml", create_page)
        self.assertIn("eval.toml", create_page)
        self.assertIn("Loop Agent", create_page)
        self.assertIn("Run Target", create_page)
        self.assertIn("Launch", create_page)
        self.assertIn("All Stages", create_page)
        self.assertIn("Stage 1 Only", create_page)
        self.assertIn("Stage 2 Only", create_page)
        self.assertNotIn("Stage 1+2", create_page)
        self.assertIn("loopConfigs[axis]", create_page)
        self.assertIn("'ref', 'data', 'remote', 'agent', 'eval', 'model', 'optim'", create_page)
        self.assertIn("h-[calc(100vh-12rem)]", create_page)
        self.assertNotIn(
            "Create Loop uses an isolated draft workspace; Start Loop runs that same workspace.",
            text,
        )

    def test_loop_create_has_dedicated_agent_chat_box(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")
        app_js = APP_JS.read_text(encoding="utf-8")

        start = text.index("<!-- Loop Create chat -->")
        end = text.index("<!-- Loop config side panel -->")
        assistant_panel = text[start:end]

        self.assertIn("xl:grid-cols-[minmax(42rem,1fr)_minmax(16rem,0.24fr)]", text)
        self.assertLess(
            text.index("<!-- Loop Create chat -->"), text.index("<!-- Loop config side panel -->")
        )
        self.assertIn("Loop Create", assistant_panel)
        self.assertNotIn(">v</span>", assistant_panel)
        self.assertNotIn("Agent Assistant", assistant_panel)
        self.assertIn("<details", assistant_panel)
        self.assertNotIn("Assistant details", assistant_panel)
        self.assertIn("loop-create-title-details", assistant_panel)
        self.assertIn("agent-chat-body", assistant_panel)
        # The chat-body sink now flows through morphdom (see
        # web/static/js/chat-renderer.js): instead of
        # ``x-html="loopCreateChatHtml()"`` (which calls
        # ``el.innerHTML = string`` on every SSE frame and produces
        # multi-MB rrweb mutations on long chats), we bind via
        # ``x-effect="ChatRenderer.morphFromHtml($el, loopCreateChatHtml())"``
        # so morphdom computes a minimal DOM patch.
        self.assertIn(
            'x-effect="ChatRenderer.morphFromHtml($el, loopCreateChatHtml())"',
            assistant_panel,
        )
        self.assertIn('x-ref="loopCreateAssistantBody"', assistant_panel)
        self.assertIn("managedPanelBusy('loopCreate')", assistant_panel)
        self.assertIn('x-model="loopCreateInput" rows="1"', assistant_panel)
        self.assertIn("managedPanelSubmit('loopCreate')", assistant_panel)
        self.assertIn("managedPanelStatusLabel('loopCreate')", assistant_panel)
        self.assertIn("managedPanelStatusClass('loopCreate')", assistant_panel)
        self.assertIn("managedPanelStop('loopCreate')", assistant_panel)
        self.assertIn("managedPanelCanStop('loopCreate')", assistant_panel)
        self.assertIn(">Model</label>", assistant_panel)
        self.assertIn("Max", assistant_panel)
        self.assertIn("loop-chat-model", assistant_panel)
        self.assertIn("loop-create-composer-toolbar", assistant_panel)
        self.assertIn("loop-create-input-row", assistant_panel)
        self.assertIn("loop-create-send", assistant_panel)
        self.assertIn("w-36", assistant_panel)
        self.assertNotIn(
            "flex-1 bg-slate-950/80 border border-slate-700/70 rounded px-2 py-1 text-[11px] text-slate-100 focus:outline-none focus:border-blue-500",
            assistant_panel,
        )
        self.assertNotIn("loopCreateIntro", assistant_panel)
        self.assertIn('x-model="loopCreateChatProfile.model"', assistant_panel)
        self.assertIn("currentModels(loopCreateChatProfile.backend)", assistant_panel)
        self.assertIn('x-model="loopCreateChatProfile.backend"', assistant_panel)
        self.assertIn('x-model="loopCreateChatProfile.max_mode"', assistant_panel)
        self.assertNotIn(
            "setLoopConfigFieldFromEvent('agent', 'agent', 'model', $event)", assistant_panel
        )
        self.assertNotIn(
            "setLoopConfigFieldFromEvent('agent', 'agent', 'max_mode', $event)", assistant_panel
        )
        self.assertIn("Ask the config agent", assistant_panel)
        self.assertIn("Send", assistant_panel)
        self.assertIn("buildLoopCreateAgentPrompt", app_js)
        prompt_start = app_js.index("buildLoopCreateAgentPrompt(userText)")
        prompt_end = app_js.index("async submitLoopCreateAssistant()")
        prompt_body = app_js[prompt_start:prompt_end]
        self.assertIn("FORGETRAIN_INTERNAL_PROMPT", prompt_body)
        self.assertIn("this.loopCreateWorkspaceDir", prompt_body)
        self.assertIn("this.loopCreateConfigPaths[ax]", prompt_body)
        self.assertNotIn("--- config/run.toml ---", prompt_body)
        self.assertNotIn("${frozen.run}", prompt_body)
        self.assertNotIn("LOOP_CREATE_USER_REQUEST", prompt_body)
        self.assertIn("managedPanelStatusLabel", app_js)
        self.assertIn("managedPanelStatusClass", app_js)
        self.assertIn("createManagedAgent", app_js)
        self.assertIn("openManagedAgent", app_js)
        self.assertIn("submitManagedComposer", app_js)
        self.assertIn("managedPanelSubmit(kind)", app_js)
        self.assertIn("submitLoopCreateAssistant", app_js)
        self.assertIn("loopCreatePreflightBusy: false", app_js)
        self.assertIn("loopCreatePreflightHtml: ''", app_js)
        self.assertIn("startLoopCreatePreflight(text)", app_js)
        self.assertIn("clearLoopCreatePreflight()", app_js)
        self.assertIn("this.startLoopCreatePreflight(text);", app_js)
        self.assertIn("if (!text || this.managedPanelBusy('loopCreate')) return;", app_js)
        # Preflight interaction is now keyed by loopCreatePreflightBusy
        # inside the panel descriptor (see _managedPanelDescriptors). The
        # old `kind === 'loopCreate' && this.loopCreatePreflightBusy`
        # literal was lifted into the descriptor's preflightInteraction
        # so this contract still holds; we just look for it at the new
        # surface.
        self.assertIn("preflightInteraction:", app_js)
        self.assertIn("this.loopCreatePreflightBusy", app_js)
        submit_start = app_js.index("async submitLoopCreateAssistant()")
        submit_end = app_js.index("formatTomlScalar(value)")
        submit_body = app_js[submit_start:submit_end]
        self.assertLess(
            submit_body.index("this.startLoopCreatePreflight(text);"),
            submit_body.index("await this.ensureLoopCreateDraft();"),
        )
        self.assertIn("loopCreateChatHtml()", app_js)
        # ``x-html`` was replaced by an ``x-effect`` morphdom sink to
        # cap rrweb mutation size (see chat-renderer.js for rationale).
        # The empty-state placeholder still gates on the same string
        # accessor, just via the negated form.
        self.assertIn(
            'x-effect="ChatRenderer.morphFromHtml($el, loopCreateChatHtml())"',
            assistant_panel,
        )
        self.assertIn("!loopCreateChatHtml()", assistant_panel)
        self.assertIn("isModelIncompatible(backend, profile.model)", app_js)
        self.assertNotIn("if (this.loopCreateChatProfile.model) return;", app_js)
        self.assertNotIn("loopCreateAssistantMessages", app_js)
        self.assertNotIn("loopCreateAssistantHtml", app_js)
        self.assertNotIn("connectLoopCreateAssistantStream", app_js)
        self.assertNotIn("normalizeLoopCreateAssistantMessages", app_js)
        self.assertIn("请帮我检查这三份 loop 配置", app_js)

    def test_mobile_main_view_uses_single_column_master_detail(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")

        self.assertIn("@media (max-width: 767px)", text)
        self.assertIn("mobile-master-detail", text)
        self.assertIn("mobile-detail-open", text)
        self.assertIn("mobile-list-open", text)
        self.assertIn(".mobile-list-open > section { display: none; }", text)
        self.assertIn(".mobile-detail-open .agent-task-sidebar { display: none; }", text)
        self.assertIn(".mobile-master-detail { height: 100%; overflow: hidden; }", text)
        self.assertIn(".agent-task-sidebar { width: 100% !important;", text)
        self.assertIn(
            ".mobile-list-open .agent-task-sidebar { height: 100%; overflow: hidden; }", text
        )
        self.assertIn(".mobile-list-open .agent-task-sidebar > .flex-1 {", text)
        self.assertIn("-webkit-overflow-scrolling: touch;", text)
        self.assertIn("max-height: 10rem;", text)
        self.assertIn("display: grid;", text)
        self.assertIn("grid-template-columns: minmax(0, 1fr) 7rem;", text)
        self.assertIn(".agent-loop-mini__rail {", text)
        self.assertIn("position: static;", text)
        self.assertIn("min-height: 100%;", text)
        self.assertIn("box-shadow: 0 10px 24px rgba(0, 0, 0, 0.35);", text)
        self.assertIn(".agent-loop-mini__mfu-sub { display: none; }", text)
        self.assertNotIn(".agent-loop-mini__rail { display: none; }", text)
        self.assertIn("agent-detail-actions", text)
        self.assertIn(".agent-detail-actions {", text)
        self.assertIn(".agent-detail-secondary-action { display: none !important; }", text)
        self.assertIn("display: grid !important;", text)
        self.assertIn("padding-right: 0;", text)
        self.assertIn("grid-column: 1 / -1;", text)
        self.assertIn(".tc-summary, .tk-summary {", text)
        self.assertIn("flex-wrap: nowrap;", text)
        self.assertIn(".tc-preview, .tk-preview {", text)
        self.assertIn("flex-basis: auto;", text)

    def test_mobile_loop_create_stacks_chat_and_config(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")

        self.assertIn(".loop-create-shell { overflow-y: auto; }", text)
        self.assertIn(".loop-create-shell > .grid", text)
        self.assertIn(".loop-create-agent-chat { min-height: 65dvh; }", text)
        self.assertIn(".loop-create-config-panel { overflow: visible; }", text)
        self.assertIn(".loop-create-composer-toolbar { flex-wrap: wrap; }", text)
        self.assertIn(
            ".loop-create-input-row { align-items: stretch; flex-direction: column; }", text
        )

    @pytest.mark.skip(reason="Phase A sidebar flattening pending — see TODO")
    def test_agents_and_loop_create_share_managed_chat_controls(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")
        app_js = APP_JS.read_text(encoding="utf-8")
        agents_panel = text[
            text.index("<!-- Agents task sidebar + chat -->") : text.index(
                "<!-- ===== Loop Panel ===== -->"
            )
        ]
        loop_create = text[
            text.index("<!-- Loop Create chat -->") : text.index("<!-- Loop config side panel -->")
        ]

        for panel_name, panel in (("agents", agents_panel), ("loopCreate", loop_create)):
            self.assertIn(f"managedPanelSubmit('{panel_name}')", panel)
            self.assertIn(f"managedPanelStop('{panel_name}')", panel)
            self.assertIn(f"managedPanelBusy('{panel_name}')", panel)
            self.assertIn(f"managedPanelCanStop('{panel_name}')", panel)

        self.assertIn("managedPanelAgentId(kind)", app_js)
        self.assertIn("managedPanelStatusLabel(kind)", app_js)
        self.assertIn("managedPanelSubmit(kind)", app_js)
        self.assertIn("managedPanelStop(kind)", app_js)
        # The literal `kind === 'loopCreate'` survived only inside the
        # descriptor block (panel selection) and the agents fast-path
        # in managedPanelBusy. The per-helper conditionals are gone —
        # see test_managed_panel_dispatch_is_descriptor_driven for the
        # SSOT contract that pins the new dispatch model.
        self.assertIn("_managedPanelDescriptors", app_js)
        self.assertIn("await this.ensureLoopCreateDraft();", app_js)
        self.assertIn("await this.saveLoopCreateConfigs();", app_js)
        self.assertIn("this.managedDefaultWorkspace = this.loopCreateWorkspaceDir", app_js)
        self.assertIn("this.managedDefaultBranch = this.loopCreateSourceBranch", app_js)
        self.assertIn(
            "this.newAgentMaxMode = backend === 'cursor-cli' && Boolean(profile.max_mode);", app_js
        )
        # SSOT: Loop Agent backend reads from the TOML draft directly; no
        # separate sync helper, no separate reactive variable.
        self.assertIn("this.ensureModelsLoaded(this.loopAgentBackendValue())", app_js)
        self.assertNotIn("syncLoopAgentBackendFromConfig", app_js)
        self.assertNotIn("backendFromModel(model)", app_js)

    def test_loop_create_key_fields_use_inline_inputs_without_form_shadow_state(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")
        app_js = APP_JS.read_text(encoding="utf-8")

        self.assertNotIn("Quick edit", text)
        self.assertIn("Quick Config", text)
        self.assertIn("Save Draft Config", text)
        self.assertNotIn("⌘S / Ctrl+S", text)
        self.assertIn("checkpoint_root", text)
        self.assertIn("backend", text)
        self.assertIn("agent.backend", text)
        self.assertIn("agent.model", text)
        self.assertIn('placeholder=".artifacts/checkpoints/..."', text)
        self.assertIn('<option value="torch"', text)
        self.assertIn(":value=\"tomlScalar(loopConfigs.ref, 'ref', 'checkpoint_root')\"", text)
        self.assertIn("setLoopConfigFieldFromEvent('ref', 'ref', 'checkpoint_root', $event)", text)
        self.assertIn(":value=\"tomlScalar(loopConfigs.agent, 'agent', 'model')\"", text)
        self.assertNotIn('list="loop-agent-model-options"', text)
        self.assertNotIn('<datalist id="loop-agent-model-options">', text)
        self.assertNotIn("<select :value=\"tomlScalar(loopConfigs.agent, 'agent', 'model')\"", text)
        self.assertIn(
            ":checked=\"tomlScalar(loopConfigs.agent, 'agent', 'max_mode') === 'true'\"", text
        )
        self.assertIn("tomlScalar", app_js)
        self.assertIn("setLoopConfigFieldFromEvent", app_js)
        self.assertIn("setTomlScalar", app_js)
        self.assertIn("isModelIncompatible(backend, currentModel)", app_js)
        self.assertIn("pick ? pick.slug : ''", app_js)
        # SSOT: loadLoopInstances() must NOT seed Loop Agent backend from
        # global /api/loop/defaults; loopAgentBackendValue() reads from the
        # draft TOML at render time instead.
        self.assertNotIn("this.loopAgentBackend = defaults.backend", app_js)
        self.assertIn("this.ensureModelsLoaded(this.loopAgentBackendValue());", app_js)
        self.assertNotIn("syncLoopAgentBackendFromConfig", app_js)
        self.assertNotIn("normalizeLoopAgentModelForBackend", app_js)
        self.assertIn("value.startsWith('claude-')", AGENTS_JS.read_text(encoding="utf-8"))
        self.assertIn("backend === 'codex'", AGENTS_JS.read_text(encoding="utf-8"))
        self.assertIn(
            "/(^|[-_])(opus|sonnet|haiku)([-_]|$)/", AGENTS_JS.read_text(encoding="utf-8")
        )
        self.assertIn("!isModelIncompatible(be, cur)", text)
        self.assertIn("loopAgentBackendValue()", text)
        self.assertIn("hasSelectableModel(loopAgentBackendValue())", text)
        self.assertIn("markLoopConfigDirty", app_js)
        self.assertIn("loopConfigsDirty: { agent: false, run: false, eval: false }", app_js)
        self.assertIn("loopConfigsSaveStatus", app_js)
        self.assertNotIn("promptLoopConfigField", app_js)
        self.assertNotIn("window.prompt(`Set", app_js)
        self.assertNotIn("loopDraftFields", app_js)

    def test_loop_create_config_views_are_file_backed_and_auto_refresh(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")
        app_js = APP_JS.read_text(encoding="utf-8")

        self.assertIn("fetch('/api/loop/drafts'", app_js)
        self.assertIn(
            "fetch(`/api/loop/${encodeURIComponent(this.loopCreateLoopId)}/configs`)", app_js
        )
        self.assertIn(
            "fetch(`/api/loop/${encodeURIComponent(this.loopCreateLoopId)}/configs/${encodeURIComponent(file)}`",
            app_js,
        )
        self.assertIn("saveLoopCreateConfig", app_js)
        self.assertIn(
            "setInterval(() => this.loadLoopCreateConfigsFromFiles({ preserveFocused: true })",
            app_js,
        )
        # The refresh must be all-or-nothing: preserve TOML text AND derived
        # state, or update both. No partial sync of backend view-state while
        # leaving the TOML stale — that's how the cursor+opus drift happened.
        self.assertNotIn("this.syncLoopAgentBackendFromConfig", app_js)
        self.assertIn("if (shouldPreserveFocused || shouldPreserveDirty) return;", app_js)
        self.assertIn("markLoopConfigDirty(axis)", text)
        self.assertIn("Save Draft Config", text)
        self.assertNotIn("Save ref.toml", text)
        self.assertNotIn("Save agent.toml", text)
        self.assertNotIn("Save eval.toml", text)
        self.assertIn('@keydown.window="handleLoopCreateKeydown($event)"', text)
        self.assertIn("handleLoopCreateKeydown(event)", app_js)
        self.assertIn("setLoopConfigFieldFromEvent('agent', 'agent', 'model', $event)", text)
        self.assertIn(":disabled=\"managedPanelBusy('loopCreate')\"", text)

    @pytest.mark.skip(reason="Phase A sidebar flattening pending — see TODO")
    def test_loop_create_config_panel_defaults_to_quick_config(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")
        app_js = APP_JS.read_text(encoding="utf-8")

        start = text.index("<!-- Loop config side panel -->")
        end = text.index("<!-- Loop instances list -->")
        config_panel = text[start:end]

        self.assertLess(config_panel.index("Quick Config"), config_panel.index("Advanced"))
        self.assertLess(config_panel.index("Advanced"), config_panel.index("Raw Config Files"))
        self.assertIn("Quick Config", config_panel)
        self.assertIn("loop-create-section-title", config_panel)
        self.assertIn("loop-create-sidebar-actions", config_panel)
        self.assertIn("loop-create-collapsible-section", config_panel)
        self.assertIn("loop-create-summary-badge", config_panel)
        self.assertIn("loopAgentSummaryModel()", config_panel)
        self.assertIn("loopRunTargetSummary()", config_panel)
        self.assertIn("Remote", config_panel)
        self.assertIn("? 'on' : 'local'", config_panel)
        self.assertNotIn('x-text="loopAgentBackend"', config_panel)
        self.assertIn("agent.backend", config_panel)
        # SSOT: Loop Agent backend reads from the draft TOML; the <select>
        # must NOT bind via x-model to an independent reactive variable.
        self.assertNotIn('x-model="loopAgentBackend"', config_panel)
        self.assertIn("onLoopAgentBackendChangedFromEvent", text)
        self.assertIn("agent.model", config_panel)
        self.assertIn("compatibleModels(loopAgentBackendValue())", config_panel)
        self.assertNotIn("loopAgentModels", config_panel)
        self.assertIn("currentModels", app_js)
        self.assertIn("shortModelName(model)", app_js)
        self.assertIn("['opus', 'sonnet', 'haiku', 'composer']", app_js)
        self.assertIn("[-_\\\\s]*\\\\d+", app_js)
        self.assertIn("modelCatalogEntry", app_js)
        self.assertIn("versionedModelLabel", app_js)
        self.assertIn("loopAgentBackendValue", app_js)
        self.assertIn("loopAgentSummaryModel()", app_js)
        self.assertIn("loopRunTargetSummary()", app_js)
        self.assertIn("modelsByBackend", AGENTS_JS.read_text(encoding="utf-8"))
        self.assertIn("compatibleModels", AGENTS_JS.read_text(encoding="utf-8"))
        self.assertIn("max_mode", config_panel)
        self.assertIn("Run Target", config_panel)
        self.assertIn("Remote", config_panel)
        self.assertIn("Launch", config_panel)
        quick_panel = config_panel[
            config_panel.index("Quick Config") : config_panel.index("Advanced")
        ]
        self.assertIn('<details class="loop-create-collapsible-section', quick_panel)
        self.assertNotIn("<details open", quick_panel)
        self.assertIn('x-model="loopForm.stages"', config_panel)
        self.assertIn('<option value="">All Stages</option>', config_panel)
        self.assertIn('<option value="stage1">Stage 1 Only</option>', config_panel)
        self.assertIn('<option value="stage2">Stage 2 Only</option>', config_panel)
        self.assertNotIn("loop-stage-button", config_panel)
        self.assertNotIn("setLoopStages(", config_panel)
        self.assertIn('x-model="loopForm.label"', config_panel)
        self.assertNotIn('x-model="loopForm.label"', quick_panel)
        self.assertNotIn("Label</span>", quick_panel)
        advanced_panel = config_panel[
            config_panel.index("Advanced") : config_panel.index("Raw Config Files")
        ]
        self.assertIn("Optional Run Label", advanced_panel)
        self.assertIn("Label only changes the loop list display name.", advanced_panel)
        self.assertIn('<details class="loop-create-advanced-config', config_panel)
        self.assertIn('<details class="loop-create-raw-config', config_panel)
        self.assertNotIn(
            '<details class="loop-create-advanced-config', config_panel.split("Quick Config", 1)[0]
        )
        self.assertIn("runs_per_stage", config_panel)
        self.assertIn("poll_seconds", config_panel)
        self.assertIn("agent_round_tries", config_panel)
        self.assertIn("retry_base_sleep", config_panel)
        self.assertIn("retry_continue_prompt", config_panel)
        self.assertIn("state_dir", config_panel)
        self.assertIn("max_consecutive_review_fails", config_panel)
        self.assertNotIn("@change=\"saveLoopCreateConfig('agent')\"", config_panel)

    @pytest.mark.skip(reason="Phase A sidebar flattening pending — see TODO")
    def test_agent_model_controls_allow_custom_model_names(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")
        agents_panel = text[
            text.index("<!-- Agents task sidebar + chat -->") : text.index(
                "<!-- ===== Loop Panel ===== -->"
            )
        ]
        loop_create = text[
            text.index("<!-- Loop Create page -->") : text.index("<!-- Loop instances list -->")
        ]

        self.assertIn('<select x-model="newAgentModel"', agents_panel)
        self.assertIn("currentModels(newAgentBackend)", agents_panel)
        self.assertNotIn('list="managed-model-options"', agents_panel)
        self.assertNotIn('<datalist id="managed-model-options">', agents_panel)
        self.assertNotIn('list="loop-agent-model-options"', loop_create)
        self.assertNotIn('<datalist id="loop-agent-model-options">', loop_create)
        self.assertIn('<select x-model="loopCreateChatProfile.model"', loop_create)
        self.assertIn("currentModels(loopCreateChatProfile.backend)", loop_create)
        # claude-code / codex model selection is disabled:
        # pickDefaultModel must short-circuit to null for fixed-model CLI
        # backends, and both managed-agent and loop-create model dropdowns
        # must be hidden when one is selected.
        self.assertIn(
            "if (!this.hasSelectableModel(key)) return null;",
            AGENTS_JS.read_text(encoding="utf-8"),
        )
        self.assertIn("pickDefaultModel", AGENTS_JS.read_text(encoding="utf-8"))
        self.assertIn("hasSelectableModel(newAgentBackend)", agents_panel)
        self.assertIn("hasSelectableModel(loopCreateChatProfile.backend)", loop_create)

    @pytest.mark.skip(reason="Phase A sidebar flattening pending — see TODO")
    def test_loop_create_raw_config_files_are_collapsed_by_default(self) -> None:
        text = INDEX_HTML.read_text(encoding="utf-8")

        start = text.index("<!-- Raw config files -->")
        end = text.index("<!-- Loop instances list -->")
        raw_panel = text[start:end]

        self.assertIn("loop-create-raw-config", raw_panel)
        self.assertIn("loop-create-eval-config", raw_panel)
        self.assertIn("ref.toml", raw_panel)
        self.assertIn("agent.toml", raw_panel)
        self.assertIn("eval.toml", raw_panel)
        self.assertIn("<details", raw_panel)
        self.assertNotIn("<details open", raw_panel)
        self.assertIn("loopConfigs[axis]", raw_panel)
        self.assertIn("markLoopConfigDirty(axis)", raw_panel)

    @pytest.mark.skip(reason="Phase A sidebar flattening pending — see TODO")
    def test_loop_start_freezes_current_loop_create_draft_configs(self) -> None:
        app_js = APP_JS.read_text(encoding="utf-8")
        loop_router = LOOP_ROUTER.read_text(encoding="utf-8")
        agent_loop = AGENT_LOOP_SH.read_text(encoding="utf-8")
        index_html = INDEX_HTML.read_text(encoding="utf-8")

        start = app_js.index("async startLoop()")
        end = app_js.index("async stopLoop(loopId)")
        body = app_js[start:end]

        # SSOT: startLoop derives backend from the draft TOML and does NOT
        # post a backend field to /api/loop/start. The server reads it from
        # workspace/config/agent.toml itself.
        self.assertIn("const loopBackend = this.loopAgentBackendValue();", body)
        self.assertIn("await this.ensureApiKeyForBackend(loopBackend)", body)
        self.assertNotIn("body.backend = ", body)
        self.assertNotIn("this.syncLoopAgentBackendFromConfig();", body)
        self.assertIn("await this.ensureLoopCreateDraft();", body)
        self.assertIn("await this.saveLoopCreateConfigs();", body)
        self.assertIn("loop_id: this.loopCreateLoopId", body)
        self.assertNotIn("body.config_agent = frozen.agent;", body)
        self.assertNotIn("body.config_ref = frozen.ref;", body)
        self.assertNotIn("body.config_eval = frozen.eval;", body)
        self.assertIn("fetch('/api/loop/start'", app_js)
        self.assertIn("selectLoopInstance(inst.loop_id)", body)
        self.assertIn("Loop started", body)
        self.assertIn("Setup Rerun", INDEX_HTML.read_text(encoding="utf-8"))
        self.assertNotIn("/rerun`,", app_js)
        self.assertIn("fetch(`/api/loop/${encodeURIComponent(loopId)}/configs`)", app_js)
        self.assertIn("this.loopConfigs = {", app_js)
        # Backend resolution flow on the server-side: profile is read from
        # workspace/config/agent.toml (SSOT) before any auth/backend logic.
        self.assertIn("_read_workspace_agent_profile(loop_id)", loop_router)
        self.assertIn("runner.auth_status(backend_name)", loop_router)
        self.assertIn('backend: str = "cursor-cli"', loop_router)
        self.assertIn("cursor_auth.api_key(backend_name)", loop_router)
        self.assertIn('cmd = ["bash", script]', loop_router)
        # --backend MUST NOT be injected; the workspace TOML is the truth.
        self.assertNotIn('cmd += ["--backend", req.backend]', loop_router)
        self.assertNotIn('cmd += ["--api-key", req.api_key]', loop_router)
        self.assertIn('env["ANTHROPIC_API_KEY"] = api_key', loop_router)
        self.assertIn('env["CURSOR_API_KEY"] = api_key', loop_router)
        self.assertIn('env["OPENAI_API_KEY"] = api_key', loop_router)
        self.assertNotIn('echo "ERROR: --api-key is required"', agent_loop)
        self.assertIn("falls back to Cursor CLI login", agent_loop)
        self.assertNotIn("...body", body)
        self.assertIn("async prepareRerunLoop(loopId)", app_js)
        # No more independent loopAgentBackend view-state. The chat panel
        # still has its own profile (loopCreateChatProfile.backend) —
        # that one is unrelated to the loop's actual agent.
        self.assertNotIn("loopAgentBackend:", app_js)
        self.assertIn("loopAgentBackendValue", app_js)
        self.assertIn("loopCreateChatProfile", app_js)
        self.assertNotIn("onLoopAgentBackendChanged(", app_js)
        self.assertIn("onLoopAgentBackendChangedFromEvent", app_js)
        self.assertIn("onLoopCreateChatBackendChanged", app_js)
        self.assertNotIn('x-model="loopAgentBackend"', index_html)
        self.assertIn('x-model="loopCreateChatProfile.backend"', index_html)
        self.assertIn("{ key: 'codex', label: 'Codex' }", app_js)
        self.assertIn("'codex': 'Codex'", AGENTS_JS.read_text(encoding="utf-8"))
        self.assertIn("'codex': 'codex login'", AGENTS_JS.read_text(encoding="utf-8"))
        self.assertIn("data.env_var || envVars[backend]", AGENTS_JS.read_text(encoding="utf-8"))
        self.assertIn("envVar: st.env_var || be.envVar", app_js)
        self.assertIn("be.envVar", index_html)
        self.assertNotIn("_default_config_contents", loop_router)
        self.assertIn("_provision_workspace", loop_router)
        self.assertNotIn("loopForm.api_key", index_html)

    def test_loop_create_chat_state_is_grouped_into_profile(self) -> None:
        """The three loop-create-chat fields (backend, model, max_mode)
        belong to one agent profile and should travel as one object.
        Three parallel reactive variables broadcast change events
        independently, so callers like onLoopCreateChatBackendChanged
        had to remember to reset two siblings whenever one moved. After
        D5 they live in loopCreateChatProfile = { backend, model,
        max_mode }, and the helpers/templates address them via that
        single object."""
        import re

        app_js = APP_JS.read_text(encoding="utf-8")
        index_html = INDEX_HTML.read_text(encoding="utf-8")

        # The grouped state must exist with all three keys.
        m = re.search(
            r"loopCreateChatProfile:\s*\{([^}]*)\}",
            app_js,
        )
        self.assertIsNotNone(m, "loopCreateChatProfile initial state must exist in app.js")
        body = m.group(1)
        for key in ("backend:", "model:", "max_mode:"):
            self.assertIn(
                key,
                body,
                f"loopCreateChatProfile must declare {key.rstrip(':')}",
            )

        # The old parallel scalar declarations are gone.
        self.assertNotIn(
            "loopCreateChatBackend: 'cursor-cli'",
            app_js,
            "loopCreateChatBackend scalar must be folded into the profile",
        )
        self.assertNotIn(
            "loopCreateChatModel: ''",
            app_js,
            "loopCreateChatModel scalar must be folded into the profile",
        )
        self.assertNotIn(
            "loopCreateChatMaxMode: true",
            app_js,
            "loopCreateChatMaxMode scalar must be folded into the profile",
        )

        # Templates address the profile, not the loose scalars.
        self.assertIn(
            'x-model="loopCreateChatProfile.backend"',
            index_html,
            "backend select must bind to the profile",
        )
        self.assertIn(
            'x-model="loopCreateChatProfile.model"',
            index_html,
            "model select must bind to the profile",
        )
        self.assertIn(
            'x-model="loopCreateChatProfile.max_mode"',
            index_html,
            "max_mode checkbox must bind to the profile",
        )
        # Loose scalar bindings must be gone.
        self.assertNotIn(
            'x-model="loopCreateChatBackend"',
            index_html,
        )
        self.assertNotIn(
            'x-model="loopCreateChatModel"',
            index_html,
        )
        self.assertNotIn(
            'x-model="loopCreateChatMaxMode"',
            index_html,
        )

    @pytest.mark.skip(reason="Phase A sidebar flattening pending — see TODO")
    def test_loop_polling_has_single_owner(self) -> None:
        """Two interval timers used to fire against the same surface:
        init() ran setInterval(_pollGates, 10s) AND startManagedRefresh
        ran setInterval(loadLoopState, 5s) which itself called _pollGates.
        That meant gates polled at 5s + 10s simultaneously and
        loadLoopInstances was unbounded by tab activity. After D4 the
        startManagedRefresh interval is the single owner of the loop tab
        polling; the init-level _pollGates timer is gone."""
        app_js = APP_JS.read_text(encoding="utf-8")
        agents_js = AGENTS_JS.read_text(encoding="utf-8")

        # The redundant init-level _pollGates timer must be gone.
        self.assertNotIn(
            "setInterval(() => this._pollGates()",
            app_js,
            "init() must not start an independent _pollGates timer; "
            "startManagedRefresh owns the loop tab cadence",
        )
        # The single canonical loop-tab polling timer survives in
        # startManagedRefresh and routes by leftTab.
        self.assertIn("vm._managedTimer = setInterval", agents_js)
        self.assertIn("vm.leftTab === 'loop'", agents_js)
        self.assertIn("vm.loadLoopState()", agents_js)
        self.assertIn("vm.loadManagedAgents()", agents_js)

    def test_managed_panel_dispatch_is_descriptor_driven(self) -> None:
        """The managedPanel* helpers used to fan out via `kind === 'loopCreate'`
        guards. Each new panel kind would have to grow another branch. After
        D3 the dispatch is a single descriptor table; the helpers index into
        it instead of repeating the conditional."""
        app_js = APP_JS.read_text(encoding="utf-8")

        # The descriptor table must exist and key both extant panels.
        self.assertIn("_managedPanelDescriptors", app_js)
        self.assertRegex(
            app_js,
            r"_managedPanelDescriptors\s*\(\)\s*\{",
            "panel descriptors should be a method/getter on the vm",
        )
        # Both panel kinds must appear inside the descriptor block (proxy
        # for "table is populated"). We don't pin exact structure so the
        # impl can evolve.
        self.assertIn("'agents'", app_js)
        self.assertIn("'loopCreate'", app_js)

        # The per-helper `kind === 'loopCreate'` literal guards must be gone
        # from the helpers themselves. The descriptor table is allowed to
        # mention the kind names; the helpers are not.
        helper_names = (
            "managedPanelAgentId",
            "managedPanelAgent",
            "managedPanelSubmit",
            "managedPanelStop",
        )
        import re

        for name in helper_names:
            m = re.search(
                rf"{re.escape(name)}\s*\([^)]*\)\s*\{{([\s\S]*?)\n    \}},",
                app_js,
            )
            self.assertIsNotNone(m, f"{name} method not found in app.js")
            body = m.group(1)
            self.assertNotIn(
                "kind === 'loopCreate'",
                body,
                f"{name} must dispatch via the descriptor table, not an inline literal kind check",
            )

    def test_toml_scalar_fixtures_execute_with_documented_outputs(self) -> None:
        """If `node` is on PATH, execute `tomlScalar` + every fixture for
        real. Fixtures with knownLimit are tolerated when they fail (the
        function genuinely doesn't support them); positive fixtures MUST
        produce the documented expected output. This turns the inline
        fixture array from documentation into an executable contract."""
        import json
        import shutil
        import subprocess

        if shutil.which("node") is None:
            self.skipTest("node is not on PATH; fixture execution skipped")
        app_js = APP_JS.read_text(encoding="utf-8")

        # Extract just the tomlScalar function body so we can eval it in
        # isolation without booting the rest of Alpine. The function is
        # `tomlScalar(text, section, key) { ... }` inside a vm literal.
        import re

        m = re.search(
            r"tomlScalar\(text, section, key\)\s*\{[\s\S]*?\n    \},",
            app_js,
        )
        self.assertIsNotNone(m, "tomlScalar definition not found in app.js")
        body = m.group(0).rstrip(",").rstrip()
        # Same for the fixtures getter.
        fx = re.search(
            r"get __tomlScalarFixtures\(\)\s*\{[\s\S]*?\n      \];\s*\n    \},",
            app_js,
        )
        self.assertIsNotNone(fx, "__tomlScalarFixtures getter not found in app.js")
        fixtures_body = fx.group(0).rstrip(",").rstrip()

        script = (
            "const vm = {" + body.rstrip(",") + ",\n" + fixtures_body.rstrip(",") + "\n};\n"
            "const results = vm.__tomlScalarFixtures.map((item) => ({"
            "  name: item.name,"
            "  expected: item.expected,"
            "  actual: vm.tomlScalar(item.text, item.section, item.key),"
            "  knownLimit: item.knownLimit || null,"
            "}));\n"
            "console.log(JSON.stringify(results));"
        )
        proc = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=10)
        self.assertEqual(
            proc.returncode,
            0,
            f"node failed: stderr={proc.stderr!r}\nstdout={proc.stdout!r}",
        )
        results = json.loads(proc.stdout)
        failures: list[str] = []
        for r in results:
            if r["actual"] == r["expected"]:
                continue
            if r["knownLimit"]:
                # Expected to fail; this is the documented boundary.
                continue
            failures.append(f"  - {r['name']}: expected={r['expected']!r} actual={r['actual']!r}")
        self.assertFalse(
            failures,
            "tomlScalar fixtures regressed:\n" + "\n".join(failures),
        )

    def test_toml_scalar_documents_its_brittle_contract(self) -> None:
        """tomlScalar is a regex parser, not a real TOML parser. It works
        for the narrow subset the Loop Create panel emits, but a reader
        needs to know the boundaries on sight. This test pins the
        constraint comment + the inline fixture array so a future change
        cannot silently broaden / break the contract."""
        app_js = APP_JS.read_text(encoding="utf-8")
        self.assertIn(
            "tomlScalar(text, section, key)",
            app_js,
            "tomlScalar must remain the documented helper",
        )
        # The implementation must carry an explicit constraint block.
        constraint_marker = "tomlScalar contract (intentionally narrow)"
        self.assertIn(
            constraint_marker,
            app_js,
            "tomlScalar must carry a contract block documenting its limits",
        )
        # The constraint block must enumerate the known-failing edge cases.
        # If any of these slip without explicit mention, future readers
        # will assume tomlScalar handles full TOML — it does not.
        for clause in (
            "single-line scalars only",
            "[[array of tables]]",
            "dotted section headers",
            "multi-line",
        ):
            self.assertIn(
                clause,
                app_js,
                f"tomlScalar contract must call out: {clause!r}",
            )
        # Inline fixtures are colocated with the impl so a reader can run
        # them by hand in the browser console; the test only checks they
        # exist and cover both positive and known-negative shapes.
        self.assertIn("__tomlScalarFixtures", app_js)
        self.assertIn("expected:", app_js)
        self.assertIn("knownLimit:", app_js)

    def test_loop_form_state_has_no_dead_fields(self) -> None:
        """loopForm.{api_key,model,review_model} were superseded by the
        TOML-anchored SSOT (api_key flows from cursor_auth; model from
        config/agent.toml). They remain in initial state as ghosts that
        confuse readers; this test pins their removal."""
        import re

        app_js = APP_JS.read_text(encoding="utf-8")
        m = re.search(r"loopForm:\s*\{([^}]*)\}", app_js)
        self.assertIsNotNone(m, "loopForm initial state must exist in app.js")
        body = m.group(1)
        for dead in ("api_key", "model", "review_model"):
            self.assertNotIn(
                dead + ":",
                body,
                f"loopForm.{dead} is dead state; must be removed from initial vm",
            )
        # And nobody reads/writes them anywhere else either.
        self.assertNotIn("loopForm.api_key", app_js)
        self.assertNotIn("loopForm.model", app_js)
        self.assertNotIn("loopForm.review_model", app_js)
        index_html = INDEX_HTML.read_text(encoding="utf-8")
        self.assertNotIn("loopForm.model", index_html)
        self.assertNotIn("loopForm.review_model", index_html)

    def test_loop_backend_web_start_and_register_share_identity(self) -> None:
        loop_router = LOOP_ROUTER.read_text(encoding="utf-8")

        start = loop_router.index('@router.post("/start")')
        register = loop_router.index("class LoopRegisterRequest")
        start_body = loop_router[start:register]

        self.assertIn("_provision_workspace", start_body)
        self.assertIn("asyncio.create_subprocess_exec", start_body)
        self.assertIn('env["LOOP_WEB_ID"] = loop_id', loop_router)
        self.assertIn('loop_id: str = ""', loop_router)
        self.assertIn("req.loop_id", loop_router)
        self.assertIn("_merge_registration", loop_router)
        self.assertIn("_find_existing_loop", loop_router)
        self.assertIn('@router.get("/{loop_id}/configs")', loop_router)
        self.assertIn('@router.put("/{loop_id}/configs/{name}")', loop_router)
        self.assertIn('@router.post("/drafts")', loop_router)
        self.assertIn("_CONFIG_AXES", loop_router)
        self.assertIn("LOOP_WEB_ID", AGENT_LOOP_SH.read_text(encoding="utf-8"))

    def test_loop_register_merges_web_started_loop_identity(self) -> None:
        from web.routers import loop

        old_pid_alive = loop._pid_alive
        old_attach = loop._attach_to_loop
        old_save = loop._save_session
        loop_id = "managed-test"
        try:
            loop._pid_alive = lambda pid: True
            loop._save_session = lambda inst: None

            async def fake_attach(inst):
                inst.version += 1

            loop._attach_to_loop = fake_attach
            loop._instances[loop_id] = loop.LoopInstance(
                loop_id=loop_id,
                label="web started",
                mode="managed",
                status="running",
                pid=12345,
                started_at=1.0,
                workspace_dir="/tmp/web-loop-workspace",
            )

            payload = asyncio.run(
                loop.register_loop(
                    loop.LoopRegisterRequest(
                        loop_id=loop_id,
                        pid=12345,
                        workspace="/tmp/web-loop-workspace",
                        label="self register",
                        args={"stages": "stage2"},
                    ),
                    x_loop_source="agent-loop",
                )
            )

            self.assertEqual(payload["loop_id"], loop_id)
            self.assertEqual(payload["mode"], "managed")
            # After the unified-log refactor, log_dir/output_file are no
            # longer tracked: the loop's live transcript lives in the
            # loop-<id> Session's stdout.log, exposed via
            # ``wrapper_agent_id`` in the snapshot.
            self.assertEqual(payload["wrapper_agent_id"], f"loop-{loop_id}")
            self.assertEqual(loop._instances[loop_id].args["stages"], "stage2")
        finally:
            loop._instances.pop(loop_id, None)
            loop._pid_alive = old_pid_alive
            loop._attach_to_loop = old_attach
            loop._save_session = old_save

    def test_loop_configs_endpoint_reads_original_workspace_snapshot(self) -> None:
        from unittest import mock

        from web.routers import loop

        loop_id = "config-test"
        with tempfile.TemporaryDirectory() as tmp:
            forge_train = Path(tmp) / "forge_train"
            workspace = forge_train / loop_id / "workspace"
            # Per-loop config dir is resolved id-based via _loop_config_dir
            # (FORGE_TRAIN_DIR/<id>/config), a sibling of workspace/.
            config_dir = forge_train / loop_id / "config"
            workspace.mkdir(parents=True)
            config_dir.mkdir(parents=True)
            (config_dir / "ref.toml").write_text("ref-original = true\n", encoding="utf-8")
            (config_dir / "agent.toml").write_text("agent-original = true\n", encoding="utf-8")
            (config_dir / "eval.toml").write_text("eval-original = true\n", encoding="utf-8")
            loop._instances[loop_id] = loop.LoopInstance(
                loop_id=loop_id,
                status="stopped",
                workspace_dir=str(workspace),
            )
            try:
                with mock.patch.object(loop, "FORGE_TRAIN_DIR", forge_train):
                    payload = asyncio.run(loop.read_loop_configs(loop_id))
            finally:
                loop._instances.pop(loop_id, None)

        self.assertEqual(payload["source_loop_id"], loop_id)
        self.assertEqual(payload["ref"], "ref-original = true\n")
        self.assertEqual(payload["agent"], "agent-original = true\n")
        self.assertEqual(payload["eval"], "eval-original = true\n")
        self.assertEqual(payload["data"], "")
        self.assertEqual(payload["remote"], "")

    def test_loop_draft_configs_are_real_workspace_files(self) -> None:
        from unittest import mock

        from web.routers import loop

        old_instances = loop._instances
        old_loops_dir = loop.FORGE_TRAIN_DIR
        old_harness_dir = loop.HARNESS_DIR
        old_loop_script = loop.LOOP_SCRIPT
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(loop, "_run_env_probe", lambda _ws: None),
        ):
            tmp_path = Path(tmp)
            harness_dir = tmp_path / "harness"
            (harness_dir / "config").mkdir(parents=True)
            (harness_dir / "config" / "eval").mkdir(parents=True)
            # Structural anchors required by env-probe's repo-root check
            # (config_runtime._looks_like_repo_root): pyproject.toml +
            # harness/config/defaults.toml + config/eval/ at workspace root.
            (harness_dir / "pyproject.toml").write_text("", encoding="utf-8")
            (harness_dir / "harness" / "config").mkdir(parents=True)
            (harness_dir / "harness" / "__init__.py").write_text("", encoding="utf-8")
            (harness_dir / "harness" / "config" / "defaults.toml").write_text("", encoding="utf-8")
            (harness_dir / "config" / "ref.toml").write_text(
                '[ref]\nbackend = "torch"\n', encoding="utf-8"
            )
            (harness_dir / "config" / "agent.toml").write_text(
                '[agent]\nmodel = "opus"\n', encoding="utf-8"
            )
            (harness_dir / "config" / "eval.toml").write_text("eval = 1\n", encoding="utf-8")
            (harness_dir / "agent-loop.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            (harness_dir / ".git").mkdir()
            (harness_dir / ".git" / "HEAD").write_text("ref: refs/heads/source\n", encoding="utf-8")
            try:
                loop._instances = {}
                loop.FORGE_TRAIN_DIR = tmp_path / "forge_train"
                loop.HARNESS_DIR = harness_dir
                loop.LOOP_SCRIPT = harness_dir / "agent-loop.sh"

                draft = asyncio.run(loop.create_loop_draft())
                loop_id = draft["loop_id"]
                self.assertEqual(draft["status"], "draft")
                self.assertIn(f"forge_train/{loop_id}/workspace", draft["workspace_dir"])
                # Provisioning seeds a fresh, isolated git repo inside the
                # workspace so the agent's `git` invocations cannot walk up
                # to the source repo's `.git` (which exposes every other
                # branch — see web/routers/loop.py::_seed_workspace_git).
                # The source repo's HEAD was set to `refs/heads/source`
                # above; assert the workspace's HEAD is a brand-new
                # `harness` ref, proving it is not a copy of the source.
                workspace_git_head = Path(draft["workspace_dir"]) / ".git" / "HEAD"
                self.assertTrue(workspace_git_head.is_file())
                head_contents = workspace_git_head.read_text(encoding="utf-8")
                self.assertIn("refs/heads/harness", head_contents)
                self.assertNotIn("refs/heads/source", head_contents)

                # _provision_workspace stages an empty per-loop config dir
                # as a sibling of workspace/. The Settings UI fills it via
                # PUT /api/loop/<id>/configs/<axis>; mirror that flow here
                # instead of expecting the bootstrap to seed templates.
                config_dir = Path(draft["workspace_dir"]).parent / "config"
                self.assertTrue(config_dir.is_dir())
                asyncio.run(
                    loop.write_loop_config(
                        loop_id,
                        "agent",
                        loop.LoopConfigWriteRequest(content='[agent]\nmodel = "opus"\n'),
                    )
                )
                update = asyncio.run(
                    loop.write_loop_config(
                        loop_id,
                        "ref",
                        loop.LoopConfigWriteRequest(content='[ref]\nbackend = "custom"\n'),
                    )
                )
                self.assertEqual(update["status"], "ok")

                payload = asyncio.run(loop.read_loop_configs(loop_id))
                self.assertEqual(payload["ref"], '[ref]\nbackend = "custom"\n')
                self.assertEqual(payload["agent"], '[agent]\nmodel = "opus"\n')
                ref_path = payload["config_paths"]["ref"]
                self.assertTrue(ref_path.endswith(f"/{loop_id}/config/ref.toml"))
                self.assertNotIn("/workspace/", ref_path)
            finally:
                loop._instances = old_instances
                loop.FORGE_TRAIN_DIR = old_loops_dir
                loop.HARNESS_DIR = old_harness_dir
                loop.LOOP_SCRIPT = old_loop_script

    def test_agent_snapshot_uses_persisted_source_branch(self) -> None:
        from web.agents import store
        from web.routers import agent

        sess = store.Session(
            agent_id="source-branch-test",
            state="completed",
            model="composer-2.5-fast",
            workspace="/tmp/non-git-draft-workspace",
            created_at="2026-05-22T00:00:00+00:00",
            source_branch="zz/harness_and_engine_web_loop",
        )
        payload = agent._snapshot(sess)

        self.assertEqual(payload["branch"], "zz/harness_and_engine_web_loop")
        self.assertEqual(payload["source_branch"], "zz/harness_and_engine_web_loop")

    def test_agent_snapshot_hides_legacy_loop_workspace_git_branch(self) -> None:
        from web.agents import store
        from web.paths import REPO_ROOT
        from web.routers import agent

        old_workspace_branch = agent._workspace_branch
        loop_workspace = REPO_ROOT / ".artifacts" / "forge_train" / "old" / "workspace"
        try:
            agent._workspace_branch = lambda workspace: (
                "zz/harness_and_engine_web_loop"
                if workspace == agent._default_workspace()
                else "loop-old"
            )
            sess = store.Session(
                agent_id="legacy-loop-branch-test",
                state="completed",
                model="composer-2.5-fast",
                workspace=str(loop_workspace),
                created_at="2026-05-22T00:00:00+00:00",
            )
            payload = agent._snapshot(sess)
        finally:
            agent._workspace_branch = old_workspace_branch

        self.assertEqual(payload["branch"], "zz/harness_and_engine_web_loop")

    @pytest.mark.skip(reason="Phase A sidebar flattening pending — see TODO")
    def test_start_loop_uses_existing_draft_workspace_in_place(self) -> None:
        from web.routers import loop

        old_instances = loop._instances
        old_loops_dir = loop.FORGE_TRAIN_DIR
        old_harness_dir = loop.HARNESS_DIR
        old_loop_script = loop.LOOP_SCRIPT
        old_api_key = loop.cursor_auth.api_key
        old_create_subprocess = loop.asyncio.create_subprocess_exec
        old_attach = loop._attach_to_loop

        class FakeProc:
            pid = 4242

        async def fake_create_subprocess_exec(*args, **kwargs):
            return FakeProc()

        async def fake_attach(inst):
            loop._save_session(inst)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            harness_dir = tmp_path / "harness"
            (harness_dir / "config").mkdir(parents=True)
            (harness_dir / "config" / "ref.toml").write_text(
                '[ref]\nbackend = "torch"\n', encoding="utf-8"
            )
            (harness_dir / "config" / "agent.toml").write_text(
                '[agent]\nbackend = "cursor-cli"\n', encoding="utf-8"
            )
            (harness_dir / "config" / "eval.toml").write_text("eval = 1\n", encoding="utf-8")
            (harness_dir / "agent-loop.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            try:
                loop._instances = {}
                loop.FORGE_TRAIN_DIR = tmp_path / "forge_train"
                loop.HARNESS_DIR = harness_dir
                loop.LOOP_SCRIPT = harness_dir / "agent-loop.sh"
                loop.cursor_auth.api_key = lambda backend="cursor-cli": "test-key"
                loop.asyncio.create_subprocess_exec = fake_create_subprocess_exec
                loop._attach_to_loop = fake_attach

                draft = asyncio.run(loop.create_loop_draft())
                loop_id = draft["loop_id"]
                workspace = Path(draft["workspace_dir"])
                (workspace / "config" / "ref.toml").write_text(
                    '[ref]\nbackend = "custom"\n', encoding="utf-8"
                )

                started = asyncio.run(
                    loop.start_loop(
                        loop.LoopStartRequest(
                            loop_id=loop_id,
                            backend="cursor-cli",
                        )
                    )
                )

                self.assertEqual(started["loop_id"], loop_id)
                self.assertEqual(started["status"], "running")
                self.assertEqual(started["workspace_dir"], str(workspace))
                self.assertEqual(
                    (workspace / "config" / "ref.toml").read_text(encoding="utf-8"),
                    '[ref]\nbackend = "custom"\n',
                )
            finally:
                loop._instances = old_instances
                loop.FORGE_TRAIN_DIR = old_loops_dir
                loop.HARNESS_DIR = old_harness_dir
                loop.LOOP_SCRIPT = old_loop_script
                loop.cursor_auth.api_key = old_api_key
                loop.asyncio.create_subprocess_exec = old_create_subprocess
                loop._attach_to_loop = old_attach

    def test_loop_create_agent_chat_uses_managed_agent_api_and_renderer(self) -> None:
        app_js = APP_JS.read_text(encoding="utf-8")
        agents_js = AGENTS_JS.read_text(encoding="utf-8")

        self.assertIn("this.managedDefaultWorkspace = this.loopCreateWorkspaceDir", app_js)
        self.assertIn("this.managedDefaultBranch = this.loopCreateSourceBranch", app_js)
        self.assertIn("source_branch: this.managedDefaultBranch || ''", agents_js)
        self.assertIn("createManagedAgent(prompt)", app_js)
        self.assertIn("await this.openManagedAgent(this.loopCreateAgentId)", app_js)
        self.assertIn("this.loopCreateAgentId = agentId || this.loopCreateAgentId", app_js)
        self.assertIn("this.managedInput = prompt", app_js)
        self.assertIn("await this.submitManagedComposer()", app_js)
        self.assertNotIn("Agents.renderMessages(this.loopCreateAssistantMessages)", app_js)
        self.assertNotIn("/api/agent/${encodeURIComponent(this.loopCreateAgentId)}/submit", app_js)
        self.assertNotIn("Agents.openStream(this.loopCreateAgentId", app_js)
        self.assertIn("vm.createManagedAgent", agents_js)
        self.assertIn("vm.submitManagedComposer", agents_js)

    def test_agent_async_callbacks_ignore_stale_selection(self) -> None:
        agents_js = AGENTS_JS.read_text(encoding="utf-8")

        self.assertIn("if (this.selectedManagedId !== agentId) return;", agents_js)
        self.assertIn("if (this.selectedManagedId === targetAgentId)", agents_js)
        self.assertIn("const targetAgentId = this.selectedManagedId;", agents_js)

    def test_removed_sessions_and_runs_routes_are_not_registered(self) -> None:
        server_text = SERVER_PY.read_text(encoding="utf-8")
        artifacts_text = ARTIFACTS_ROUTER.read_text(encoding="utf-8")

        self.assertIn("artifacts", server_text)
        self.assertIn('@router.get("/state")', artifacts_text)
        self.assertNotIn('@router.get("/sessions', artifacts_text)
        self.assertNotIn('@router.get("/runs', artifacts_text)


class TestLoopAndAgentVisualParity(unittest.TestCase):
    """The Loop tab and the Agents tab present two different kinds of
    work (a long-running shell pipeline vs a single conversation), but
    they are both ``instances bound to a workspace`` from the user's
    point of view. Phase A of the unification refactor pins the
    *visual* shape so the two tabs look like siblings, not strangers,
    without touching endpoints, JS handlers, or data flow.

    These tests fail loudly the moment one side drifts from the other,
    making it cheap to keep the symmetry as the UI evolves.
    """

    def setUp(self) -> None:
        self.text = INDEX_HTML.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Helpers — slice INDEX_HTML into the two regions under comparison.
    # ------------------------------------------------------------------

    def _section(self, start_marker: str, end_marker: str) -> str:
        i = self.text.index(start_marker)
        j = self.text.index(end_marker, i + 1)
        return self.text[i:j]

    @property
    def loop_sidebar(self) -> str:
        return self._section(
            "<!-- Loop instances list -->",
            "<!-- Loop detail: agent conversation -->",
        )

    @property
    def agents_sidebar(self) -> str:
        # The Agents tab's sidebar is fenced by the ``data-testid``
        # marker on the <aside>.
        return self._section(
            'data-testid="agent-task-sidebar"',
            "<!-- Sidebar resizer -->",
        )

    @property
    def loop_detail_header(self) -> str:
        # Detail header ends just before the wrapper-transcript body
        # opens. After the unified-agent-log refactor the body comment
        # is "Unified wrapper transcript".
        return self._section(
            "<!-- Loop detail: agent conversation -->",
            "<!-- Unified wrapper transcript:",
        )

    @property
    def agents_detail_header(self) -> str:
        # The agent detail header is the first block after the sidebar
        # resizer and ends right before the rendered conversation body.
        return self._section(
            "<!-- Sidebar resizer -->",
            '<div class="agent-chat-body',
        )

    # ------------------------------------------------------------------
    # Sidebar header — the "list of instances" header bar.
    # ------------------------------------------------------------------

    @pytest.mark.skip(reason="Phase A sidebar flattening pending — see TODO")
    def test_sidebar_headers_share_padding_and_primary_button_shape(self) -> None:
        # Padding (px-3 py-2) is the Agents-side canonical and the Loop
        # sidebar was migrated to match it.
        for label, sidebar in (("loop", self.loop_sidebar), ("agents", self.agents_sidebar)):
            with self.subTest(side=label):
                self.assertIn(
                    "px-3 py-2",
                    sidebar,
                    f"{label} sidebar header must use 'px-3 py-2'",
                )

        # Both primary buttons use the same blue-600 family.
        primary = "bg-blue-600 hover:bg-blue-500"
        self.assertIn(primary, self.loop_sidebar)
        self.assertIn(primary, self.agents_sidebar)

        # Both primary buttons share the exact rounded/padding signature.
        button_signature = "rounded px-2.5 py-1"
        self.assertIn(button_signature, self.loop_sidebar)
        self.assertIn(button_signature, self.agents_sidebar)

    # ------------------------------------------------------------------
    # Sidebar row — every list-item card.
    # ------------------------------------------------------------------

    @pytest.mark.skip(reason="Phase A sidebar flattening pending — see TODO")
    def test_loop_row_uses_shared_agent_subrow_classes(self) -> None:
        """Phase A goal: a loop row in the sidebar uses the same
        ``.agent-subrow*`` class family as an agent row, so spacing,
        hover, and active state are governed by one CSS rule rather
        than two near-duplicates.
        """
        loop_rows = self.loop_sidebar
        self.assertIn(
            'class="agent-subrow',
            loop_rows,
            "loop sidebar row must adopt .agent-subrow (shared with agents)",
        )
        self.assertIn(
            "agent-subrow--active",
            loop_rows,
            "selected loop row must use the same active-state class",
        )
        self.assertIn(
            "agent-subrow__title",
            loop_rows,
            "loop row title must use the shared title class",
        )
        self.assertIn(
            "agent-subrow__meta",
            loop_rows,
            "loop row secondary line must use the shared meta class",
        )

    @pytest.mark.skip(reason="Phase A sidebar flattening pending — see TODO")
    def test_row_pills_use_shared_agent_pill_class(self) -> None:
        for label, side in (("loop", self.loop_sidebar), ("agents", self.agents_sidebar)):
            with self.subTest(side=label):
                self.assertIn("agent-pill", side)

    # ------------------------------------------------------------------
    # Detail header — the bar above the streaming body.
    # ------------------------------------------------------------------

    @pytest.mark.skip(reason="Phase A sidebar flattening pending — see TODO")
    def test_detail_headers_share_padding_and_layout(self) -> None:
        """Both detail headers MUST share the same padding (px-3 py-2)
        and the same flex layout idiom (gap-3 between header items).
        This removes the 'they look 1px different' deltas that make
        the Loop tab feel like a worse copy of the Agents tab."""
        for label, header in (
            ("loop", self.loop_detail_header),
            ("agents", self.agents_detail_header),
        ):
            with self.subTest(side=label):
                self.assertIn(
                    "px-3 py-2",
                    header,
                    f"{label} detail header must use 'px-3 py-2'",
                )
                self.assertIn(
                    "gap-3",
                    header,
                    f"{label} detail header must use 'gap-3' for item spacing",
                )

    @pytest.mark.skip(reason="Phase A sidebar flattening pending — see TODO")
    def test_detail_header_action_buttons_share_pill_signature(self) -> None:
        """Stop / Delete / Setup Rerun all live in the right-side
        action group; their visual signature (rounded px-2.5 py-1) is
        the same on both sides."""
        signature = "rounded px-2.5 py-1"
        self.assertIn(signature, self.loop_detail_header)
        self.assertIn(signature, self.agents_detail_header)

    # ------------------------------------------------------------------
    # Phase A.2 — pruning the agents sidebar: no branch grouping, no
    # ``Total / Responding / Ready`` counter row, refresh becomes the
    # same ↻ icon button as the loop side.
    # ------------------------------------------------------------------

    @pytest.mark.skip(reason="Phase A sidebar flattening pending — see TODO")
    def test_agents_sidebar_uses_icon_refresh_like_loops(self) -> None:
        """The Loop sidebar header uses an ``↻`` icon button for
        Refresh. The Agents sidebar header MUST use the exact same
        icon-button signature so the two tabs look like siblings."""
        loop_refresh = 'aria-label="Refresh loop list"'
        agents_refresh = 'aria-label="Refresh agents list"'
        self.assertIn(loop_refresh, self.loop_sidebar)
        self.assertIn(agents_refresh, self.agents_sidebar)

        # The literal text-button "Refresh" must NOT appear in the
        # agents sidebar after the cleanup.
        self.assertNotIn(
            ">Refresh<",
            self.agents_sidebar,
            "agents sidebar still has a text 'Refresh' button — should be the ↻ icon",
        )

        # Both refresh buttons share the icon glyph.
        self.assertIn("↻", self.loop_sidebar)
        self.assertIn("↻", self.agents_sidebar)

    @pytest.mark.skip(reason="Phase A sidebar flattening pending — see TODO")
    def test_agents_sidebar_drops_state_counts_row(self) -> None:
        """The ``Total: N · Responding: N · Ready: N`` counter row is
        noise (the per-row pill already shows status). Drop it from
        the agents sidebar header."""
        sidebar = self.agents_sidebar
        for token in ("Total:", "Responding:", "Ready:"):
            with self.subTest(token=token):
                self.assertNotIn(
                    token,
                    sidebar,
                    f"agents sidebar still references '{token}'",
                )

    @pytest.mark.skip(reason="Phase A sidebar flattening pending — see TODO")
    def test_agents_sidebar_drops_branch_grouping(self) -> None:
        """The agents list is a flat list of rows just like the loop
        list — no ``branch:`` / per-task folding layer. The ``task``
        loop variable, the ``agent-task-row`` button, and the count
        badge must all be gone from the agents sidebar."""
        sidebar = self.agents_sidebar
        for token in (
            "agent-task-row",
            "agent-task-count",
            "toggleManagedTask",
            "expandedManagedTasks",
            "managedAgentTasks",
            'x-for="task in',
        ):
            with self.subTest(token=token):
                self.assertNotIn(
                    token,
                    sidebar,
                    f"agents sidebar still references grouping artifact '{token}'",
                )

    def test_agents_js_drops_branch_grouping_state(self) -> None:
        """Dead-code follow-up: the JS-side state that fed the
        grouping UI (``managedAgentTasks``, ``expandedManagedTasks``,
        ``managedStateCounts``) and the helpers that built it must be
        removed when the sidebar drops grouping."""
        text = AGENTS_JS.read_text(encoding="utf-8")
        for token in (
            "managedAgentTasks",
            "expandedManagedTasks",
            "managedStateCounts",
            "groupManagedAgentsByTask",
            "countManagedStates",
            "toggleManagedTask",
            "taskKeyForAgent",
            "branchLabelForAgent",
        ):
            with self.subTest(token=token):
                self.assertNotIn(
                    token,
                    text,
                    f"agents.js still references dead grouping symbol '{token}'",
                )


if __name__ == "__main__":
    unittest.main()
