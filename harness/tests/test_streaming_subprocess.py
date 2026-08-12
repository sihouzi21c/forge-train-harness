"""Timeout behavior for suite streaming subprocesses."""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

from evals._common import run_streaming_subprocess


class TestStreamingSubprocessTimeout(unittest.TestCase):
    def test_timeout_fires_after_partial_output_goes_quiet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_file = Path(tmp) / "out.log"
            started = time.monotonic()
            returncode, output = run_streaming_subprocess(
                [
                    sys.executable,
                    "-c",
                    "import sys,time; print('ready', flush=True); time.sleep(3)",
                ],
                cwd=Path(tmp),
                env=os.environ.copy(),
                timeout=1,
                out_file_path=out_file,
            )
            elapsed = time.monotonic() - started

        self.assertEqual(returncode, -1)
        self.assertLess(elapsed, 2.5)
        self.assertIn("ready", output)

    def test_timeout_fires_even_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_file = Path(tmp) / "out.log"
            started = time.monotonic()
            returncode, output = run_streaming_subprocess(
                [sys.executable, "-c", "import time; time.sleep(3)"],
                cwd=Path(tmp),
                env=os.environ.copy(),
                timeout=1,
                out_file_path=out_file,
            )
            elapsed = time.monotonic() - started

        self.assertEqual(returncode, -1)
        self.assertLess(elapsed, 2.5)
        self.assertEqual(output, "")


if __name__ == "__main__":
    unittest.main()
