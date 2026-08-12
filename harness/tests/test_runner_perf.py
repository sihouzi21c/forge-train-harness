"""Performance contracts for ``web.agents.runner``.

The web server pays a real subprocess cost for every call to
``list_models`` (``agent --list-models``) and ``auth_status``
(``claude -p ...`` for claude-code, which actually runs an LLM
inference). Both are invoked on UI interactions that the user
considers cheap (opening a panel, switching backend, sending a
message), so they need to be:

* **list_models**: cached per (backend, api_key) for a TTL, with an
  explicit invalidation hook so the persisted-keys flow can refresh.
* **auth_status (claude-code)**: a credential-file / version probe,
  not an LLM round-trip. The probe must remain fast even when
  the Anthropic endpoint is slow or unreachable.

These tests pin those contracts.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


def _fake_proc(stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
    class _P:
        def __init__(self) -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr
            self._waited = False

        async def communicate(self):
            return stdout, stderr

        async def wait(self):
            self._waited = True
            return returncode

        def kill(self):  # pragma: no cover - only used in timeout paths
            self.returncode = -9

    return _P()


class TestListModelsTtlCache(unittest.TestCase):
    """``list_models`` for cursor-cli forks ``agent --list-models``;
    cache the parsed list per (backend, api_key) for a TTL so repeated
    UI dropdown opens don't fork a child process every time."""

    def setUp(self) -> None:
        from web.agents import runner

        # Fresh cache for each test.
        if hasattr(runner, "_LIST_MODELS_CACHE"):
            runner._LIST_MODELS_CACHE.clear()

    def test_repeated_calls_reuse_cache_within_ttl(self) -> None:
        from web.agents import runner

        call_count = {"n": 0}
        stdout = b"gpt-5.5-high - Cursor GPT 5.5 (high)\ncomposer-large - Composer L\n"

        async def fake_exec(*cmd, **kwargs):
            call_count["n"] += 1
            return _fake_proc(stdout=stdout)

        with mock.patch.object(runner.asyncio, "create_subprocess_exec", fake_exec):
            first = asyncio.run(runner.list_models(api_key="k1", backend="cursor-cli"))
            second = asyncio.run(runner.list_models(api_key="k1", backend="cursor-cli"))

        self.assertEqual(call_count["n"], 1, "second call must hit the cache")
        self.assertEqual(first, second)
        self.assertEqual(first[0]["slug"], "gpt-5.5-high")
        self.assertEqual(first[1]["slug"], "composer-large")

    def test_cache_keyed_by_api_key(self) -> None:
        """Different keys → different cache entries (different user, different list)."""
        from web.agents import runner

        outputs = {"k1": b"a - alpha\n", "k2": b"b - beta\n"}

        async def fake_exec(*cmd, **kwargs):
            key = kwargs.get("env", {}).get("CURSOR_API_KEY") or ""
            return _fake_proc(stdout=outputs.get(key, b""))

        with mock.patch.object(runner.asyncio, "create_subprocess_exec", fake_exec):
            a = asyncio.run(runner.list_models(api_key="k1", backend="cursor-cli"))
            b = asyncio.run(runner.list_models(api_key="k2", backend="cursor-cli"))
            a2 = asyncio.run(runner.list_models(api_key="k1", backend="cursor-cli"))

        self.assertEqual(a[0]["slug"], "a")
        self.assertEqual(b[0]["slug"], "b")
        self.assertEqual(a, a2, "cache must keep the per-key entry")

    def test_cache_expires_after_ttl(self) -> None:
        from web.agents import runner

        call_count = {"n": 0}

        async def fake_exec(*cmd, **kwargs):
            call_count["n"] += 1
            return _fake_proc(stdout=b"x - 1\n")

        with (
            mock.patch.object(runner.asyncio, "create_subprocess_exec", fake_exec),
            mock.patch.object(runner, "_LIST_MODELS_TTL_SECONDS", 0.01),
        ):
            asyncio.run(runner.list_models(api_key="k", backend="cursor-cli"))
            time.sleep(0.05)
            asyncio.run(runner.list_models(api_key="k", backend="cursor-cli"))

        self.assertEqual(call_count["n"], 2, "expired entry must re-fetch")

    def test_failure_is_not_cached(self) -> None:
        """A RuntimeError from a bad CLI run must not poison the cache."""
        from web.agents import runner

        attempts = {"n": 0}

        async def fake_exec(*cmd, **kwargs):
            attempts["n"] += 1
            if attempts["n"] == 1:
                return _fake_proc(stderr=b"boom", returncode=1)
            return _fake_proc(stdout=b"ok - O\n")

        with mock.patch.object(runner.asyncio, "create_subprocess_exec", fake_exec):
            with self.assertRaises(RuntimeError):
                asyncio.run(runner.list_models(api_key="k", backend="cursor-cli"))
            models = asyncio.run(runner.list_models(api_key="k", backend="cursor-cli"))

        self.assertEqual(models[0]["slug"], "ok")
        self.assertEqual(attempts["n"], 2)

    def test_claude_backend_remains_inline_constant_no_subprocess(self) -> None:
        """claude-code's model catalogue is a hard-coded constant — it
        must not fork ``agent --list-models`` and it must not consume
        cache entries (no subprocess to amortize). The catalogue itself
        is now empty: model selection is disabled because the Claude
        Code CLI's own default is the only supported choice."""
        from web.agents import runner

        async def fake_exec(*cmd, **kwargs):  # pragma: no cover - asserted unreached
            raise AssertionError("claude-code must not fork to list models")

        with mock.patch.object(runner.asyncio, "create_subprocess_exec", fake_exec):
            models = asyncio.run(runner.list_models(api_key="k", backend="claude-code"))

        self.assertEqual(models, [])


class TestClaudeAuthStatusIsCheap(unittest.TestCase):
    """``auth_status('claude-code')`` MUST NOT spawn ``claude -p`` with a
    prompt — that runs a real LLM inference, costing 5 s+ per UI action.
    A credential file presence check is the contracted cheap probe."""

    def test_claude_auth_status_does_not_invoke_llm_inference(self) -> None:
        from web.agents import runner

        async def fake_exec(*cmd, **kwargs):
            # If any subprocess is spawned, the prompt argument must not
            # look like a chat user message.
            self.assertNotIn(
                "Reply only: ok",
                cmd,
                f"auth_status must not run a chat-style inference: {cmd!r}",
            )
            # Even allow lightweight `claude --version` or `claude /status`;
            # what we forbid is the heavy `-p <prompt>` pattern.
            if "-p" in cmd:
                idx = cmd.index("-p")
                # The arg right after -p must be a flag or absent, not a prompt.
                following = cmd[idx + 1 :]
                self.assertFalse(
                    any(not str(a).startswith("-") for a in following),
                    f"auth_status spawned `claude -p` with a prompt: {cmd!r}",
                )
            return _fake_proc(stdout=b"", returncode=0)

        with (
            mock.patch.object(runner.asyncio, "create_subprocess_exec", fake_exec),
            mock.patch.object(
                runner.backends.get_backend("claude-code").__class__,
                "available",
                lambda self: True,
            ),
        ):
            asyncio.run(runner.auth_status("claude-code"))

    def test_claude_auth_status_returns_authenticated_when_credentials_present(self) -> None:
        """When the CLI's own status probe reports loggedIn, the function
        returns authenticated without making an LLM call."""
        from web.agents import runner

        async def fake_exec(*cmd, **kwargs):  # pragma: no cover - asserted unreached
            raise AssertionError(f"unexpected subprocess: {cmd!r}")

        def fake_run(*_args, **_kwargs):
            class _R:
                returncode = 0
                stdout = '{"loggedIn": true, "email": "test@example.com"}'
                stderr = ""

            return _R()

        with (
            mock.patch.object(runner.asyncio, "create_subprocess_exec", fake_exec),
            mock.patch("shutil.which", lambda _b: "/usr/local/bin/claude"),
            mock.patch("subprocess.run", fake_run),
        ):
            status = asyncio.run(runner.auth_status("claude-code"))

        self.assertTrue(status["is_authenticated"])
        self.assertEqual(status["auth_error"], "")

    def test_claude_auth_status_returns_not_authenticated_when_no_credentials(self) -> None:
        """When the CLI's own status probe reports loggedIn=false,
        return not-authenticated (without forking an LLM call)."""
        from web.agents import runner

        async def fake_exec(*cmd, **kwargs):  # pragma: no cover - asserted unreached
            raise AssertionError(f"unexpected subprocess: {cmd!r}")

        def fake_run(*_args, **_kwargs):
            class _R:
                returncode = 0
                stdout = '{"loggedIn": false}'
                stderr = ""

            return _R()

        with (
            tempfile.TemporaryDirectory() as fake_home,
            mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "", "HOME": fake_home}, clear=False),
            mock.patch.object(runner.asyncio, "create_subprocess_exec", fake_exec),
            mock.patch("shutil.which", lambda _b: "/usr/local/bin/claude"),
            mock.patch("subprocess.run", fake_run),
        ):
            # Make sure no credential file exists under the fake HOME.
            self.assertFalse((Path(fake_home) / ".claude" / "credentials.json").exists())
            status = asyncio.run(runner.auth_status("claude-code"))

        self.assertFalse(status["is_authenticated"])

    def test_claude_auth_status_returns_not_available_when_binary_missing(self) -> None:
        """If the claude binary is absent, return a clear error and DON'T fork."""
        from web.agents import runner

        async def fake_exec(*cmd, **kwargs):  # pragma: no cover - asserted unreached
            raise AssertionError(f"unexpected subprocess: {cmd!r}")

        with (
            mock.patch.object(runner.asyncio, "create_subprocess_exec", fake_exec),
            mock.patch("shutil.which", lambda _b: None),
        ):
            status = asyncio.run(runner.auth_status("claude-code"))

        self.assertFalse(status["is_authenticated"])
        self.assertIn("not found", status["auth_error"].lower())


if __name__ == "__main__":
    unittest.main()
