"""Contract tests for ``harness prefetch`` — the workspace-native corpus
prefetch subcommand.

``agent-loop.sh`` triggers the production long-train corpus prefetch by invoking
``bin/harness prefetch`` (local) or ``ssh … 'cd $WORKDIR && bin/harness
prefetch'`` (remote), exactly like ``harness run`` — so the download
always runs *in the workspace that executes it* and the bytes land local
to that host. These tests pin the seam between the CLI surface and
``tools.prefetch_data.main``:

1. The CLI exposes a ``prefetch`` subcommand that dispatches to
   ``app.prefetch_command`` and returns its exit code verbatim (no render
   payload — same bypass pattern as ``env-probe`` / ``echo-config``).
2. ``app.prefetch_command`` forwards the workspace ``repo_root`` and the
   ``--curl`` fetcher to ``tools.prefetch_data.main``.
3. ``app.prefetch_command`` neutralises the egress proxy and pins an
   ``HF_ENDPOINT`` default before the download — the proxy/endpoint
   handling that used to live inline in ``agent-loop.sh`` now belongs to
   the subcommand so both the local and remote call sites stay trivial.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["FORGE_REPO_ROOT"] = str(REPO_ROOT)

from harness import app, cli, config_runtime  # noqa: E402


class TestPrefetchCli(unittest.TestCase):
    def test_parser_exposes_prefetch_subcommand(self) -> None:
        parser = cli._build_parser()
        sub = next(action for action in parser._actions if action.dest == "command")
        self.assertIn("prefetch", sub.choices)

    def test_parser_accepts_bare_prefetch(self) -> None:
        parser = cli._build_parser()
        parsed = parser.parse_args(["prefetch"])
        self.assertEqual(parsed.command, "prefetch")

    def test_cli_main_routes_to_prefetch_command_and_returns_exit_code(self) -> None:
        with mock.patch.object(app, "prefetch_command", return_value=0) as m:
            rc = cli.main(["prefetch"])
        self.assertEqual(rc, 0)
        m.assert_called_once_with()

    def test_cli_main_propagates_nonzero_exit_code(self) -> None:
        with mock.patch.object(app, "prefetch_command", return_value=1):
            rc = cli.main(["prefetch"])
        self.assertEqual(rc, 1)


class TestPrefetchCommand(unittest.TestCase):
    def test_forwards_repo_root_and_curl_to_prefetch_main(self) -> None:
        from tools import prefetch_data

        captured: dict[str, list[str] | None] = {}

        def fake_main(argv=None):
            captured["argv"] = argv
            return 0

        with (
            mock.patch.object(config_runtime, "repo_root", return_value=Path("/ws")),
            mock.patch.object(prefetch_data, "main", side_effect=fake_main),
        ):
            rc = app.prefetch_command()

        self.assertEqual(rc, 0)
        self.assertEqual(captured["argv"], ["--repo-root", "/ws", "--curl"])

    def test_propagates_prefetch_main_exit_code(self) -> None:
        from tools import prefetch_data

        with (
            mock.patch.object(config_runtime, "repo_root", return_value=Path("/ws")),
            mock.patch.object(prefetch_data, "main", return_value=1),
        ):
            self.assertEqual(app.prefetch_command(), 1)

    def test_unsets_proxy_and_pins_hf_endpoint_before_download(self) -> None:
        from tools import prefetch_data

        seen_env: dict[str, str | None] = {}

        def fake_main(argv=None):
            # Capture the environment as observed *inside* the download.
            for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
                seen_env[k] = os.environ.get(k)
            seen_env["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT")
            return 0

        proxy_env = {
            "http_proxy": "http://proxy:911",
            "https_proxy": "http://proxy:911",
            "HTTP_PROXY": "http://proxy:911",
            "HTTPS_PROXY": "http://proxy:911",
        }
        with (
            mock.patch.object(config_runtime, "repo_root", return_value=Path("/ws")),
            mock.patch.object(prefetch_data, "main", side_effect=fake_main),
            mock.patch.dict(os.environ, proxy_env, clear=False),
        ):
            # Ensure HF_ENDPOINT is unset so the default is what we observe.
            os.environ.pop("HF_ENDPOINT", None)
            app.prefetch_command()

        for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            self.assertIsNone(seen_env[k], f"{k} should be unset during prefetch")
        self.assertEqual(seen_env["HF_ENDPOINT"], "https://hf-mirror.com")

    def test_honours_preexisting_hf_endpoint(self) -> None:
        from tools import prefetch_data

        seen: dict[str, str | None] = {}

        def fake_main(argv=None):
            seen["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT")
            return 0

        with (
            mock.patch.object(config_runtime, "repo_root", return_value=Path("/ws")),
            mock.patch.object(prefetch_data, "main", side_effect=fake_main),
            mock.patch.dict(os.environ, {"HF_ENDPOINT": "https://my-mirror"}, clear=False),
        ):
            app.prefetch_command()

        self.assertEqual(seen["HF_ENDPOINT"], "https://my-mirror")

    def test_restores_proxy_env_after_download(self) -> None:
        # The command must not leak its proxy mutation into the parent
        # process (the CLI dispatches in-process). After it returns, the
        # original proxy vars must be intact.
        from tools import prefetch_data

        with (
            mock.patch.object(config_runtime, "repo_root", return_value=Path("/ws")),
            mock.patch.object(prefetch_data, "main", return_value=0),
            mock.patch.dict(os.environ, {"http_proxy": "http://proxy:911"}, clear=False),
        ):
            app.prefetch_command()
            self.assertEqual(os.environ.get("http_proxy"), "http://proxy:911")


if __name__ == "__main__":
    unittest.main()
