"""Tests for harness.resources — declared external-resource manifest."""

from __future__ import annotations

import hashlib
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from harness import resources


def _write_manifest(workspace: Path, body: str) -> None:
    # Workspace top level (repo_root), sibling of config/ / evals/ / workload/.
    target = workspace / "external_resources.toml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _WorkspaceCtx:
    """Construct a workspace dir and bind resources.repo_root() to it."""

    def __init__(self) -> None:
        self._tmp = TemporaryDirectory()
        self.path = Path(self._tmp.name).resolve()
        # ``harness.resources`` calls ``config_runtime.repo_root()``;
        # the simplest way to redirect it without poking the
        # functools.cache is to monkeypatch the symbol resources.py
        # imported at top.
        self._patcher = patch.object(resources, "repo_root", return_value=self.path)
        self._patcher.start()

    def cleanup(self) -> None:
        self._patcher.stop()
        self._tmp.cleanup()


class TestManifestLoader(unittest.TestCase):
    def test_manifest_path_is_workspace_top_level(self) -> None:
        # Regression: manifest lives at <repo_root>/external_resources.toml,
        # NOT <repo_root>/harness/external_resources.toml (inner package).
        ws = _WorkspaceCtx()
        self.addCleanup(ws.cleanup)
        self.assertEqual(resources.manifest_path(ws.path), ws.path / "external_resources.toml")

    def test_missing_manifest_is_empty(self) -> None:
        ws = _WorkspaceCtx()
        self.addCleanup(ws.cleanup)
        self.assertEqual(resources.load_manifest(ws.path), [])

    def test_empty_manifest_is_no_op(self) -> None:
        ws = _WorkspaceCtx()
        self.addCleanup(ws.cleanup)
        _write_manifest(ws.path, "[meta]\nversion = 1\n")
        self.assertEqual(resources.load_manifest(ws.path), [])
        self.assertEqual(resources.provision(ws.path), [])

    def test_version_must_be_one(self) -> None:
        ws = _WorkspaceCtx()
        self.addCleanup(ws.cleanup)
        _write_manifest(ws.path, "[meta]\nversion = 99\n")
        with self.assertRaises(ValueError) as ctx:
            resources.load_manifest(ws.path)
        self.assertIn("version", str(ctx.exception))

    def test_entry_missing_keys(self) -> None:
        ws = _WorkspaceCtx()
        self.addCleanup(ws.cleanup)
        _write_manifest(
            ws.path,
            '[meta]\nversion = 1\n\n[[entry]]\nname = "x"\nkind = "file"\n',
        )
        with self.assertRaises(ValueError):
            resources.load_manifest(ws.path)


class TestProvision(unittest.TestCase):
    def test_file_resource_local_source_symlinks(self) -> None:
        ws = _WorkspaceCtx()
        self.addCleanup(ws.cleanup)
        # Real upstream file
        upstream = ws.path / "upstream" / "tokenizer.model"
        upstream.parent.mkdir(parents=True)
        upstream.write_bytes(b"tokenizer-bytes")
        sha = _sha256(b"tokenizer-bytes")
        _write_manifest(
            ws.path,
            (
                "[meta]\nversion = 1\n\n"
                "[[entry]]\n"
                'name = "tokenizer.test"\n'
                'kind = "file"\n'
                f'sha256 = "{sha}"\n'
                'canonical_relpath = ".resources/tokenizer/test/tokenizer.model"\n'
                f'sources = ["file://{upstream}"]\n'
            ),
        )
        provisioned = resources.provision(ws.path)
        self.assertEqual([r.name for r in provisioned], ["tokenizer.test"])
        canonical = ws.path / ".resources" / "tokenizer" / "test" / "tokenizer.model"
        self.assertTrue(canonical.is_symlink())
        self.assertEqual(canonical.resolve(), upstream.resolve())

    def test_sha_mismatch_fails_with_diagnostic(self) -> None:
        ws = _WorkspaceCtx()
        self.addCleanup(ws.cleanup)
        upstream = ws.path / "upstream" / "tokenizer.model"
        upstream.parent.mkdir(parents=True)
        upstream.write_bytes(b"actual-bytes")
        _write_manifest(
            ws.path,
            (
                "[meta]\nversion = 1\n\n"
                "[[entry]]\n"
                'name = "tokenizer.test"\n'
                'kind = "file"\n'
                f'sha256 = "{"0" * 64}"\n'
                'canonical_relpath = ".resources/tokenizer/test/tokenizer.model"\n'
                f'sources = ["file://{upstream}"]\n'
            ),
        )
        with self.assertRaises(resources.ResourceProvisionError) as ctx:
            resources.provision(ws.path)
        self.assertIn("sha256 mismatch", str(ctx.exception))
        self.assertIn("tokenizer.test", str(ctx.exception))

    def test_file_resource_without_sha_symlinks_first_existing(self) -> None:
        # Content-addressing opted out (no sha256): first existing source of
        # the right kind wins, no hash comparison.
        ws = _WorkspaceCtx()
        self.addCleanup(ws.cleanup)
        missing = ws.path / "upstream" / "gone.model"
        upstream = ws.path / "upstream" / "tokenizer.model"
        upstream.parent.mkdir(parents=True)
        upstream.write_bytes(b"whatever-bytes")
        _write_manifest(
            ws.path,
            (
                "[meta]\nversion = 1\n\n"
                "[[entry]]\n"
                'name = "tokenizer.nohash"\n'
                'kind = "file"\n'
                'canonical_relpath = ".resources/tokenizer/nohash/tokenizer.model"\n'
                f'sources = ["file://{missing}", "file://{upstream}"]\n'
            ),
        )
        provisioned = resources.provision(ws.path)
        self.assertEqual([r.name for r in provisioned], ["tokenizer.nohash"])
        self.assertEqual([r.sha256 for r in provisioned], [""])
        canonical = ws.path / ".resources" / "tokenizer" / "nohash" / "tokenizer.model"
        self.assertTrue(canonical.is_symlink())
        self.assertEqual(canonical.resolve(), upstream.resolve())

    def test_tree_resource_without_sha_checks_kind(self) -> None:
        # No sha256 tree entry: a directory source passes, a file source is
        # rejected for wrong kind.
        ws = _WorkspaceCtx()
        self.addCleanup(ws.cleanup)
        as_file = ws.path / "upstream" / "not_a_dir"
        as_file.parent.mkdir(parents=True)
        as_file.write_bytes(b"x")
        as_dir = ws.path / "upstream" / "tok_tree"
        as_dir.mkdir()
        (as_dir / "tokenizer.json").write_bytes(b"{}")
        _write_manifest(
            ws.path,
            (
                "[meta]\nversion = 1\n\n"
                "[[entry]]\n"
                'name = "tokenizer.tree"\n'
                'kind = "tree"\n'
                'canonical_relpath = ".resources/tokenizer/tree"\n'
                f'sources = ["file://{as_file}", "file://{as_dir}"]\n'
            ),
        )
        provisioned = resources.provision(ws.path)
        self.assertEqual([r.name for r in provisioned], ["tokenizer.tree"])
        canonical = ws.path / ".resources" / "tokenizer" / "tree"
        self.assertTrue(canonical.is_symlink())
        self.assertEqual(canonical.resolve(), as_dir.resolve())

    def test_entry_without_sha_still_requires_other_keys(self) -> None:
        ws = _WorkspaceCtx()
        self.addCleanup(ws.cleanup)
        _write_manifest(
            ws.path,
            '[meta]\nversion = 1\n\n[[entry]]\nname = "x"\nkind = "file"\n',
        )
        with self.assertRaises(ValueError) as ctx:
            resources.load_manifest(ws.path)
        # Missing canonical_relpath / sources still fails; missing sha256 does not.
        self.assertNotIn("sha256", str(ctx.exception))

    def test_network_source_skipped_by_default(self) -> None:
        ws = _WorkspaceCtx()
        self.addCleanup(ws.cleanup)
        _write_manifest(
            ws.path,
            (
                "[meta]\nversion = 1\n\n"
                "[[entry]]\n"
                'name = "tokenizer.hf"\n'
                'kind = "file"\n'
                f'sha256 = "{"0" * 64}"\n'
                'canonical_relpath = ".resources/tokenizer/hf/tokenizer.model"\n'
                'sources = ["hf://openbmb/MiniCPM4-0.5B#tokenizer.model"]\n'
            ),
        )
        # Ensure the network gate is off (provision should refuse).
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FORGE_ALLOW_NETWORK_RESOURCES", None)
            with self.assertRaises(resources.ResourceProvisionError) as ctx:
                resources.provision(ws.path)
        self.assertIn("network fetch disabled", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
