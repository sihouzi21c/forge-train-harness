"""Per-loop directory layout SSOT.

``loop_layout`` is the single authoritative source for the on-disk
topology of a forge-train loop: where its frozen per-loop config dir
and its isolated workspace live under ``.artifacts/forge_train/<id>/``.

Every consumer — the ``new-looptask`` skill, ``agent-loop.sh``, and the
``web`` layer — must resolve these paths THROUGH this module rather than
re-deriving ``<forge_train_dir>/<id>/config`` by hand. The functions are
pure (no env reads, no cwd) so they stay trivially testable and reusable
from any layer above ``harness``.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from harness import loop_layout


class TestLoopLayout(unittest.TestCase):
    def test_config_dir_is_id_config_under_forge_train(self) -> None:
        got = loop_layout.loop_config_dir(Path("/t/forge_train"), "abc123")
        self.assertEqual(got, Path("/t/forge_train/abc123/config"))

    def test_workspace_dir_is_id_workspace_under_forge_train(self) -> None:
        got = loop_layout.loop_workspace_dir(Path("/t/forge_train"), "abc123")
        self.assertEqual(got, Path("/t/forge_train/abc123/workspace"))

    def test_config_dir_is_sibling_of_workspace(self) -> None:
        # The remote read-path and the web Settings UI both rely on the
        # config dir being a SIBLING of workspace/ (config survives a
        # workspace re-provision). Pin the relationship here so neither
        # path can drift independently.
        ft = Path("/t/forge_train")
        cfg = loop_layout.loop_config_dir(ft, "abc123")
        ws = loop_layout.loop_workspace_dir(ft, "abc123")
        self.assertEqual(cfg.parent, ws.parent)

    def test_accepts_str_forge_train_dir(self) -> None:
        # Callers cross the bash/Python boundary passing a plain string;
        # the function must coerce rather than require a Path.
        got = loop_layout.loop_config_dir("/t/forge_train", "abc123")
        self.assertEqual(got, Path("/t/forge_train/abc123/config"))


if __name__ == "__main__":
    unittest.main()
