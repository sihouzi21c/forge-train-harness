"""Per-loop config is write-once: editable while ``draft``, frozen after launch.

This pins the second half of the R12 contention fix: even with the
per-loop config dir in place, the foot-gun would return if a user (or
the UI) could PUT a new ``[remote].hostname`` into ``config/remote.toml``
after ``agent-loop.sh`` already sed-substituted the old value into its
prompts. The server-side 409 mirrors the filesystem ``chmod -R a-w``
the wrapper applies post-launch.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException


class TestPerLoopConfigImmutableAfterLaunch(unittest.TestCase):
    def _make_instance(self, tmp_path: Path, loop_id: str, status: str):
        from web.routers import loop

        forge_train = tmp_path / "forge_train"
        workspace = forge_train / loop_id / "workspace"
        workspace.mkdir(parents=True)
        # Per-loop config dir is resolved id-based via _loop_config_dir
        # (FORGE_TRAIN_DIR/<id>/config), a sibling of workspace/.
        cfg_dir = forge_train / loop_id / "config"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "remote.toml").write_text(
            '[remote]\nkind = "local"\nhostname = ""\n',
            encoding="utf-8",
        )
        inst = loop.LoopInstance(
            loop_id=loop_id,
            mode="managed",
            status=status,
            workspace_dir=str(workspace),
        )
        loop._instances[loop_id] = inst
        return loop, inst

    def test_put_succeeds_while_draft(self) -> None:
        loop_id = "draft-edit-test"
        with tempfile.TemporaryDirectory() as tmp:
            loop, _ = self._make_instance(Path(tmp), loop_id, status="draft")
            try:
                with mock.patch.object(loop, "FORGE_TRAIN_DIR", Path(tmp) / "forge_train"):
                    result = asyncio.run(
                        loop.write_loop_config(
                            loop_id,
                            "remote",
                            loop.LoopConfigWriteRequest(
                                content='[remote]\nkind = "devspace"\nhostname = "ds-X"\n'
                            ),
                        )
                    )
                self.assertEqual(result["status"], "ok")
                cfg_dir = Path(tmp) / "forge_train" / loop_id / "config"
                self.assertIn(
                    'hostname = "ds-X"',
                    (cfg_dir / "remote.toml").read_text(encoding="utf-8"),
                )
            finally:
                loop._instances.pop(loop_id, None)

    def test_put_returns_409_after_launch(self) -> None:
        loop_id = "running-frozen-test"
        with tempfile.TemporaryDirectory() as tmp:
            loop, _ = self._make_instance(Path(tmp), loop_id, status="running")
            try:
                with self.assertRaises(HTTPException) as cm:
                    asyncio.run(
                        loop.write_loop_config(
                            loop_id,
                            "remote",
                            loop.LoopConfigWriteRequest(content='[remote]\nhostname = "ds-Y"\n'),
                        )
                    )
                self.assertEqual(cm.exception.status_code, 409)
                self.assertIn("frozen", cm.exception.detail.lower())
                cfg_dir = Path(tmp) / "forge_train" / loop_id / "config"
                # File was NOT overwritten.
                self.assertNotIn(
                    "ds-Y",
                    (cfg_dir / "remote.toml").read_text(encoding="utf-8"),
                )
            finally:
                loop._instances.pop(loop_id, None)

    def test_put_returns_409_for_stopped_loop(self) -> None:
        """A stopped loop's config is the historical record of what ran;
        it must not be retroactively edited."""
        loop_id = "stopped-frozen-test"
        with tempfile.TemporaryDirectory() as tmp:
            loop, _ = self._make_instance(Path(tmp), loop_id, status="stopped")
            try:
                with self.assertRaises(HTTPException) as cm:
                    asyncio.run(
                        loop.write_loop_config(
                            loop_id,
                            "remote",
                            loop.LoopConfigWriteRequest(content="[remote]\n"),
                        )
                    )
                self.assertEqual(cm.exception.status_code, 409)
            finally:
                loop._instances.pop(loop_id, None)


if __name__ == "__main__":
    unittest.main()
