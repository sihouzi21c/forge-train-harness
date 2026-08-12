"""End-to-end test: agent-loop.sh boots a loop-wrapper Session entry.

Drives the real ``harness/agent-loop.sh`` script in ``LOOP_REGISTER_ONLY=1``
mode (short-circuits before any backend CLI spawn) and asserts that
the new wrapper-side machinery emits a complete loop-wrapper Session
on disk:

* ``$FORGE_AGENTS_DIR/loop-<id>/session.json`` exists with
  ``backend=loop-wrapper``, ``kind=loop_wrapper``, terminal state
  (``completed``) after the EXIT trap fires.
* ``stdout.log`` carries (in order) a ``system.init`` seed line and a
  terminal ``loop_event`` of subtype ``loop_exit`` so the frontend
  chat renderer has a typed, well-bounded transcript even for the
  no-op short-circuit path.

Together with ``test_spawn_managed_agent_cli`` (helpers in isolation)
this pins the contract agent-loop.sh shells out to.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
AGENT_LOOP_SH = REPO_ROOT / "harness" / "agent-loop.sh"


class TestAgentLoopWrapperEventsE2E(unittest.TestCase):
    def test_register_only_run_writes_wrapper_session_and_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            forge_train = tmp_path / "forge_train"
            agents_dir = tmp_path / "web-agents"
            env = {
                **os.environ,
                "LOOP_REGISTER_ONLY": "1",
                "LOOP_WEB_ID": "e2etest1234",
                "FORGE_TRAIN_DIR": str(forge_train),
                "FORGE_AGENTS_DIR": str(agents_dir),
                # Suppress the curl dashboard hint (non-fatal, but we
                # don't want it touching real local ports in CI).
                "LOOP_WEB_PORT": "1",
            }
            # Run the real script. It auto-provisions a workspace under
            # forge_train/, re-execs there, runs the wrapper init +
            # registration + EXIT trap, and exits.
            result = subprocess.run(
                [
                    "bash",
                    str(AGENT_LOOP_SH),
                    "--stages",
                    "stage1",
                    "--backend",
                    "cursor-cli",
                    "--model",
                    "test-model",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"agent-loop.sh failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )

            wrapper_dir = agents_dir / "loop-e2etest1234"
            self.assertTrue(
                (wrapper_dir / "session.json").is_file(),
                f"missing session.json under {wrapper_dir}; agents_dir contents: "
                f"{list(agents_dir.iterdir()) if agents_dir.is_dir() else 'absent'}",
            )
            data = json.loads((wrapper_dir / "session.json").read_text(encoding="utf-8"))
            self.assertEqual(data["agent_id"], "loop-e2etest1234")
            self.assertEqual(data["backend"], "loop-wrapper")
            self.assertEqual(data["kind"], "loop_wrapper")
            self.assertEqual(data["loop_id"], "e2etest1234")
            self.assertEqual(
                data["state"], "completed", f"EXIT trap should mark wrapper completed; got {data}"
            )
            self.assertEqual(data["exit_code"], 0)

            stdout_log = wrapper_dir / "stdout.log"
            lines = [
                json.loads(line)
                for line in stdout_log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreaterEqual(
                len(lines), 2, f"expected at least system.init + loop_exit; got {lines}"
            )
            self.assertEqual(lines[0]["type"], "system")
            self.assertEqual(lines[0]["subtype"], "init")
            self.assertEqual(lines[0]["session_id"], "loop-e2etest1234")

            tail = lines[-1]
            self.assertEqual(tail["type"], "loop_event")
            self.assertEqual(tail["subtype"], "loop_exit")
            self.assertEqual(tail["state"], "completed")
            self.assertEqual(tail["exit_code"], 0)


if __name__ == "__main__":
    unittest.main()
