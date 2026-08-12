"""Loop backend is sourced from the per-loop ``config/agent.toml`` (SSOT),
not from the start-loop request body or any view-state in the UI.

These tests pin the invariant that broke the live UI:
the Loop Agent panel was showing ``backend=cursor-cli`` while the
per-loop ``config/agent.toml`` had ``backend="claude-code"``,
because the front-end kept ``loopAgentBackend`` as an independent
reactive variable and the back-end then forwarded it via ``--backend``.

The fix re-anchors both sides on the TOML:

* Front-end deletes ``loopAgentBackend`` and reads ``loopAgentBackendValue()``
  straight from ``loopConfigs.agent``.
* Back-end resolves the backend by parsing the loop's per-loop
  ``config/agent.toml`` (sibling of ``workspace/``) and ignores
  ``LoopStartRequest.backend``.
* ``_build_loop_cmd`` no longer injects ``--backend``;
  ``agent-loop.sh`` already reads ``[agent].backend`` from the TOML
  via ``FORGE_CONFIG_DIR``.
* ``_default_config_contents`` prefers the user's
  per-loop ``config/agent.toml`` over a hard-coded template ordering.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException

MONOREPO_ROOT = Path(__file__).resolve().parents[3]
APP_JS = MONOREPO_ROOT / "web" / "static" / "js" / "app.js"
INDEX_HTML = MONOREPO_ROOT / "web" / "static" / "index.html"


def _agent_toml(backend: str, model: str = "opus", max_mode: bool = False) -> str:
    return (
        f'[agent]\nbackend = "{backend}"\nmodel = "{model}"\nmax_mode = {str(max_mode).lower()}\n'
    )


class TestBuildLoopCmdDoesNotInjectBackend(unittest.TestCase):
    """``--backend`` MUST NOT appear in the agent-loop CLI invocation.

    The TOML inside the draft workspace is the only backend authority;
    a CLI flag would silently override it from a view-state that has
    no business being the truth source.
    """

    def test_build_loop_cmd_omits_backend_flag(self) -> None:
        from web.routers.loop import LoopStartRequest, _build_loop_cmd

        req = LoopStartRequest(backend="cursor-cli", model="", stages="")
        cmd = _build_loop_cmd(req, script_path="/tmp/agent-loop.sh")
        self.assertNotIn("--backend", cmd, f"--backend leaked into cmd: {cmd}")

    def test_build_loop_cmd_omits_backend_flag_even_when_request_disagrees_with_toml(self) -> None:
        from web.routers.loop import LoopStartRequest, _build_loop_cmd

        req = LoopStartRequest(backend="claude-code", model="", stages="stage1")
        cmd = _build_loop_cmd(req, script_path="/tmp/agent-loop.sh")
        self.assertNotIn("--backend", cmd)
        self.assertNotIn("claude-code", cmd)
        self.assertNotIn("cursor-cli", cmd)


class TestStartLoopReadsBackendFromWorkspaceToml(unittest.TestCase):
    """``POST /api/loop/start`` resolves backend from the draft workspace TOML.

    The request body's ``backend`` field is irrelevant for backend
    selection. The API-key resolution path, the env injection, and
    any subprocess spawning all derive from the on-disk TOML.
    """

    def _run_start_loop(
        self, agent_toml_content: str, *, request_backend: str = "cursor-cli"
    ) -> tuple[dict, dict | None]:
        """Drive ``start_loop`` against a temp workspace and capture the env.

        Returns ``(snapshot, captured_env)`` where ``captured_env`` is the
        env dict passed to ``asyncio.create_subprocess_exec`` (or ``None``
        if no subprocess was spawned, e.g. on a 4xx).
        """
        from web.routers import loop as loop_router

        captured: dict[str, dict] = {}

        class _FakeProc:
            def __init__(self) -> None:
                self.pid = 12345
                self.returncode = None
                self.stdout = None
                self.stderr = None

            async def wait(self) -> int:
                return 0

        async def _fake_exec(*cmd, **kwargs):
            captured["cmd"] = list(cmd)
            captured["env"] = dict(kwargs.get("env") or {})
            return _FakeProc()

        async def _noop_attach(inst):
            return None

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            loops_dir = tmp_path / "forge_train"
            loops_dir.mkdir()
            instance_dir = loops_dir / "abcdef123456"
            workspace = instance_dir / "workspace"
            workspace.mkdir(parents=True)
            # Per-loop config dir is a SIBLING of workspace/, not inside it.
            cfg_dir = instance_dir / "config"
            cfg_dir.mkdir(parents=True)
            (cfg_dir / "agent.toml").write_text(agent_toml_content, encoding="utf-8")
            (workspace / "agent-loop.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

            inst = loop_router.LoopInstance(
                loop_id="abcdef123456",
                mode="managed",
                status="draft",
                workspace_dir=str(workspace),
            )
            with (
                mock.patch.object(loop_router, "FORGE_TRAIN_DIR", loops_dir),
                mock.patch.dict(loop_router._instances, {"abcdef123456": inst}, clear=True),
                mock.patch.object(loop_router.asyncio, "create_subprocess_exec", _fake_exec),
                mock.patch.object(loop_router, "_attach_to_loop", _noop_attach),
                mock.patch.object(loop_router.auth, "api_key", lambda _backend: "test-key"),
            ):
                req = loop_router.LoopStartRequest(backend=request_backend, loop_id="abcdef123456")
                snapshot = asyncio.run(loop_router.start_loop(req))
        return snapshot, captured.get("env")

    def test_workspace_toml_with_claude_backend_overrides_cursor_request(self) -> None:
        snapshot, env = self._run_start_loop(
            _agent_toml("claude-code"), request_backend="cursor-cli"
        )
        self.assertEqual(snapshot["status"], "running")
        assert env is not None
        # Backend resolved to claude-code -> only ANTHROPIC_API_KEY should
        # be set to the just-resolved key. Whatever CURSOR_API_KEY happens
        # to be inherited from the outer test env is incidental.
        self.assertEqual(env.get("ANTHROPIC_API_KEY"), "test-key")
        self.assertEqual(snapshot["args"]["backend"], "claude-code")

    def test_workspace_toml_with_cursor_backend_overrides_claude_request(self) -> None:
        snapshot, env = self._run_start_loop(
            _agent_toml("cursor-cli", model="gpt-5.5-high"),
            request_backend="claude-code",
        )
        self.assertEqual(snapshot["status"], "running")
        assert env is not None
        self.assertEqual(env.get("CURSOR_API_KEY"), "test-key")
        self.assertEqual(snapshot["args"]["backend"], "cursor-cli")

    def test_workspace_toml_with_codex_backend_sets_openai_key(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict("os.environ", {"CODEX_HOME": tmp}, clear=False),
        ):
            snapshot, env = self._run_start_loop(
                _agent_toml("codex", model="o3"),
                request_backend="cursor-cli",
            )
        self.assertEqual(snapshot["status"], "running")
        assert env is not None
        self.assertEqual(env.get("OPENAI_API_KEY"), "test-key")
        self.assertEqual(snapshot["args"]["backend"], "codex")

    def test_workspace_toml_without_backend_yields_409(self) -> None:
        with self.assertRaises(HTTPException) as cm:
            self._run_start_loop('[agent]\nmodel = "opus"\n')
        self.assertEqual(cm.exception.status_code, 409)
        self.assertIn("backend", cm.exception.detail.lower())


class TestProvisionWorkspaceLeavesPerLoopConfigEmpty(unittest.TestCase):
    """``_provision_workspace`` MUST NOT pre-fill the per-loop config dir.

    Under the per-loop config isolation refactor, the per-loop config
    dir (``.artifacts/forge_train/<id>/config/``, sibling of
    ``workspace/``) is owned by the Settings UI / new-looptask skill,
    not by the workspace bootstrap. ``_provision_workspace`` only
    needs to *create* the empty dir so the Settings UI has a stable
    write target; any auto-seeding from ``harness/config/*.toml`` is a
    foot-gun that re-opens the shared-mutable-state hole that caused
    the R12 contention incident.
    """

    def test_provision_workspace_creates_empty_per_loop_config_dir(self) -> None:
        from web.routers import loop as loop_router

        with tempfile.TemporaryDirectory() as tmp:
            fake_harness = Path(tmp) / "harness"
            (fake_harness / "config" / "agent").mkdir(parents=True)
            (fake_harness / "config" / "eval").mkdir(parents=True)
            (fake_harness / "pyproject.toml").write_text("", encoding="utf-8")
            (fake_harness / "harness" / "config").mkdir(parents=True)
            (fake_harness / "harness" / "__init__.py").write_text("", encoding="utf-8")
            (fake_harness / "harness" / "config" / "defaults.toml").write_text("", encoding="utf-8")
            (fake_harness / "config" / "agent" / "claudecode-default.toml").write_text(
                _agent_toml("claude-code", model=""), encoding="utf-8"
            )
            (fake_harness / "config" / "eval" / "dense_training").mkdir(parents=True)
            (
                fake_harness / "config" / "eval" / "dense_training" / "dense_training.toml"
            ).write_text('[suite]\nname = "dense"\n', encoding="utf-8")

            forge_dir = Path(tmp) / "forge_train"
            forge_dir.mkdir()
            with (
                mock.patch.object(loop_router, "HARNESS_DIR", fake_harness),
                mock.patch.object(loop_router, "FORGE_TRAIN_DIR", forge_dir),
                mock.patch.object(loop_router, "_run_env_probe", lambda _ws: None),
            ):
                workspace = loop_router._provision_workspace("abc123def456")
                cfg_dir = workspace.parent / "config"

                self.assertTrue(
                    cfg_dir.is_dir(),
                    "per-loop config dir must exist post-provision",
                )
                self.assertEqual(
                    list(cfg_dir.iterdir()),
                    [],
                    "per-loop config dir must be empty; Settings UI / skill fills it",
                )


class TestFrontendLoopAgentBackendReadsFromToml(unittest.TestCase):
    """The front-end MUST NOT carry an independent ``loopAgentBackend``
    reactive variable. The Loop Agent panel's backend select MUST read
    from ``loopConfigs.agent`` (the TOML draft) on every render.
    """

    def setUp(self) -> None:
        self.app_js = APP_JS.read_text(encoding="utf-8")
        self.index_html = INDEX_HTML.read_text(encoding="utf-8")

    def test_app_js_does_not_declare_loop_agent_backend_state(self) -> None:
        self.assertNotIn(
            "loopAgentBackend: ''",
            self.app_js,
            "loopAgentBackend reactive declaration must be removed; "
            "backend should derive from loopConfigs.agent (TOML SSOT).",
        )

    def test_app_js_exposes_loop_agent_backend_value_getter(self) -> None:
        self.assertIn(
            "loopAgentBackendValue",
            self.app_js,
            "loopAgentBackendValue() helper must exist to read backend "
            "from loopConfigs.agent directly.",
        )

    def test_app_js_drops_sync_helpers_that_create_dual_truth(self) -> None:
        forbidden = ["syncLoopAgentBackendFromConfig", "backendFromModel"]
        present = [name for name in forbidden if name in self.app_js]
        self.assertEqual(present, [], f"Stale dual-truth helpers still in app.js: {present}")

    def test_load_loop_instances_does_not_write_loop_agent_backend(self) -> None:
        self.assertNotIn(
            "this.loopAgentBackend = defaults.backend",
            self.app_js,
            "loadLoopInstances must not overwrite the Loop Agent backend "
            "with /api/loop/defaults; that is the bug being fixed.",
        )

    def test_index_html_backend_select_does_not_bind_loop_agent_backend_xmodel(self) -> None:
        self.assertNotIn(
            'x-model="loopAgentBackend"',
            self.index_html,
            "Loop Agent backend <select> must bind via :selected on TOML, "
            "not via x-model on a separate reactive variable.",
        )

    def test_start_loop_body_does_not_send_backend(self) -> None:
        # The body posted to /api/loop/start no longer needs ``backend``;
        # the server reads it from the workspace TOML.
        idx = self.app_js.find("async startLoop(")
        self.assertGreater(idx, -1, "startLoop() must exist in app.js")
        body_end = self.app_js.find("loadLoopInstances()", idx)
        self.assertGreater(body_end, idx)
        slice_ = self.app_js[idx:body_end]
        self.assertNotIn(
            "body.backend = ",
            slice_,
            "startLoop() must not send `backend` in the request body; "
            "backend is sourced from the per-loop config/agent.toml on the server.",
        )


if __name__ == "__main__":
    unittest.main()
