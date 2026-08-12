"""Contract tests for :func:`tools.ref_script_runner.run_ref_script`.

The runner is a pure bash-exec primitive:

* No training-shape knobs (steps, world_size, batch sizes, seed) in the
  signature — those live in the L0 ref script and are addressed via
  ``FORGE_GATE`` presets.
* No hook wiring — the dispatcher's M1 path bashes an
  agent-generated bridge (see
  ``evals/harness_hook/recipes/README.md``) that itself calls
  :func:`evals.harness_hook.install`; the runner just bashes the
  path it is given.
* Accepts ``ref_script_path`` (absolute) — basename resolution for
  the L0 ref script lives in
  ``evals/_common.py::_resolve_customer_ref_script``.
"""

from __future__ import annotations

import contextlib
import inspect
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


def _pid_alive(pid: int) -> bool:
    """True only if ``pid`` is a live process.

    A killpg'd grandchild whose original parent (the bash launcher) was
    SIGKILL'd first becomes a zombie (``Z``/defunct) reparented to PID 1
    until something reaps it — in a container PID 1 may not reap promptly.
    ``os.kill(pid, 0)`` succeeds for a zombie (the PID slot still exists),
    so it would falsely report the worker as alive. A zombie holds no GPU
    or RSS, so for this contract it counts as dead: detect it via
    ``/proc/<pid>/stat`` (Linux) and exclude state ``Z``.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
            state = fh.read().rsplit(")", 1)[1].split()[0]
        return state != "Z"
    except (FileNotFoundError, ProcessLookupError, IndexError):
        # No procfs (e.g. macOS): os.kill already proved it is live and
        # non-zombie there since BSD reaps orphans via launchd promptly.
        return True


class _FakeProc:
    """Stand-in for ``subprocess.Popen`` in env/cmd-construction tests.

    Carries a PID that no process group owns, so the runner's best-effort
    ``os.killpg`` teardown resolves to ``ProcessLookupError`` (a no-op)
    rather than signalling a real group.
    """

    pid = 2_000_000_000

    def wait(self, timeout: object = None) -> int:
        del timeout
        return 0


class TestRefScriptRunnerSSOT(unittest.TestCase):
    def test_runner_api_does_not_accept_training_shape_knobs(self) -> None:
        from tools.ref_script_runner import run_ref_script

        params = inspect.signature(run_ref_script).parameters
        for forbidden in (
            "num_steps",
            "world_size",
            "micro_batch_size",
            "global_batch_size",
            "data_path",
            "seed",
            # No hook wiring in the runner — the agent-generated
            # bridge owns that, see evals/harness_hook/recipes/README.md.
            "hook_module",
            "ref_script_name",
        ):
            self.assertNotIn(forbidden, params)
        self.assertIn("gate_name", params)
        self.assertIn("ref_script_path", params)

    def test_runner_module_has_no_hardcoded_ref_script_constant(self) -> None:
        """tools.ref_script_runner must not own the ref-script filename SSOT.

        The basename lives in ``config/ref.toml [ref].ref_script``
        and is resolved via ``harness.config_runtime.ref_script``;
        a hardcoded module-level constant would be a second source of truth.
        """
        import tools.ref_script_runner as runner_mod

        self.assertFalse(
            hasattr(runner_mod, "REF_SCRIPT_NAME"),
            "tools.ref_script_runner.REF_SCRIPT_NAME re-introduces a "
            "second SSOT for the ref-script basename.",
        )

    def test_ref_script_requires_ref_toml(self) -> None:
        """ref_script is deployment-level and must come from ref.toml.

        When ref.toml is absent and no env override is set,
        ref_script() must raise ValueError (fail-fast).
        """
        from harness import config_runtime

        bare_config = {"ref": {}}
        with self.assertRaises(ValueError):
            config_runtime.ref_script(bare_config)

    def test_workload_toml_exposes_ref_capture_script_knob(self) -> None:
        """``[ref].ref_capture_script`` is the agent-owned M1 bridge knob.

        Empty value is the legitimate check-in default: the agent
        generates the bridge during M1 development and points the
        knob at it. ``run_ref_capture`` translates an empty value
        into a fail-fast result that points at
        ``evals/harness_hook/recipes/README.md``.
        """
        from harness import config_runtime

        _, workload_config = config_runtime.load_workload_config(
            None,
            include_user_config=False,
        )
        value = config_runtime.ref_capture_script(workload_config)
        self.assertIsInstance(value, str)

    def test_runner_selects_gate_without_training_overrides(self) -> None:
        from tools.ref_script_runner import run_ref_script

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            script = repo / "fake_ref.sh"
            script.write_text(
                "#!/usr/bin/env bash\nexit 0\n",
                encoding="utf-8",
            )
            dump_dir = repo / "dump"

            captured_env: dict[str, str] = {}

            def fake_popen(*args: object, **kwargs: object) -> _FakeProc:
                del args
                captured_env.update(kwargs["env"])  # type: ignore[arg-type]
                dump_dir.mkdir(parents=True, exist_ok=True)
                (dump_dir / "gate_metadata.json").write_text(
                    '{"gate": "loss-gate-200", "num_steps": 1000}\n',
                    encoding="utf-8",
                )
                return _FakeProc()

            with mock.patch("subprocess.Popen", side_effect=fake_popen):
                result = run_ref_script(
                    repo,
                    gate_name="loss-gate-200",
                    dump_dir=dump_dir,
                    ref_script_path=script,
                )

        self.assertTrue(result.succeeded)
        self.assertEqual(captured_env["FORGE_GATE"], "loss-gate-200")
        self.assertEqual(captured_env["LOCAL_MODE"], "1")
        self.assertEqual(result.metadata["gate"], "loss-gate-200")
        for forbidden in (
            "NUM_STEPS_OVERRIDE",
            "WORLD_SIZE",
            "NPROC_PER_NODE",
            "MICRO_BATCH_SIZE_OVERRIDE",
            "GLOBAL_BATCH_SIZE_OVERRIDE",
            "DATA_PATH_OVERRIDE",
            "SEED",
        ):
            self.assertNotIn(forbidden, captured_env)

    def test_runner_invokes_bash_on_the_given_absolute_path(self) -> None:
        from tools.ref_script_runner import run_ref_script

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            alt_path = repo / "any_dir" / "train_foo.sh"
            alt_path.parent.mkdir(parents=True)
            alt_path.write_text(
                "#!/usr/bin/env bash\nexit 0\n",
                encoding="utf-8",
            )
            dump_dir = repo / "dump"

            invoked_cmds: list[list[str]] = []

            def fake_popen(*args: object, **kwargs: object) -> _FakeProc:
                cmd = args[0] if args else kwargs.get("args")
                invoked_cmds.append(list(cmd))  # type: ignore[arg-type]
                dump_dir.mkdir(parents=True, exist_ok=True)
                return _FakeProc()

            with mock.patch("subprocess.Popen", side_effect=fake_popen):
                run_ref_script(
                    repo,
                    gate_name="forward-align",
                    dump_dir=dump_dir,
                    ref_script_path=alt_path,
                )

        self.assertEqual(len(invoked_cmds), 1)
        self.assertEqual(invoked_cmds[0][0], "bash")
        self.assertEqual(invoked_cmds[0][1], str(alt_path.resolve()))

    def test_runner_rejects_missing_ref_script_path(self) -> None:
        from tools.ref_script_runner import run_ref_script

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            with self.assertRaisesRegex(FileNotFoundError, "Ref script not found"):
                run_ref_script(
                    repo,
                    gate_name="forward-align",
                    dump_dir=repo / "dump",
                    ref_script_path=repo / "does_not_exist.sh",
                )


class TestRefScriptTimeoutKillsProcessGroup(unittest.TestCase):
    """The real ref script bashes ``torchrun``, which forks rank-worker
    grandchildren; with ``--persistent`` a worker holds the GPU until it
    is explicitly reaped. ``subprocess.run(timeout=…)`` only kills the
    direct child (the ``bash``/``torchrun`` launcher) on ``TimeoutExpired``,
    leaving the grandchild alive and pinning ~all of GPU memory — the next
    gate then OOMs. ``run_ref_script`` MUST launch the script in its own
    process group and SIGKILL the whole group on timeout so no grandchild
    survives. Mirrors ``test_transport_timeout_killpg`` for the ref path.
    """

    def test_grandchild_dies_after_timeout(self) -> None:
        from tools.ref_script_runner import run_ref_script

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            dump_dir = repo / "dump"
            marker = repo / "grandchild.pid"
            # Stand-in for the torchrun rank worker: a backgrounded sleep
            # that records its PID, while the parent (the bash launcher)
            # itself blocks long enough to outlive the timeout. Without
            # killpg the backgrounded grandchild survives the launcher kill.
            script = repo / "fake_ref_with_worker.sh"
            script.write_text(
                f"#!/usr/bin/env bash\nsleep 60 &\necho $! > {marker}\nsleep 60\n",
                encoding="utf-8",
            )

            # Wait long enough that the launcher has reached the second
            # ``sleep`` (so the grandchild PID is on disk) but is still
            # blocked when the timeout fires.
            started = time.monotonic()
            result = run_ref_script(
                repo,
                gate_name="perf-bitwise",
                dump_dir=dump_dir,
                ref_script_path=script,
                timeout_s=3,
            )
            elapsed = time.monotonic() - started

            # Read the grandchild PID while the temp dir still exists.
            self.assertTrue(marker.exists(), "fixture never recorded grandchild PID")
            grandchild_pid = int(marker.read_text().strip())

        self.assertTrue(result.timed_out)
        self.assertEqual(result.returncode, -1)
        self.assertLess(elapsed, 30.0)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if not _pid_alive(grandchild_pid):
                break
            time.sleep(0.1)
        if _pid_alive(grandchild_pid):
            # Best-effort cleanup so a failing run does not leak the sleeper.
            with contextlib.suppress(ProcessLookupError):
                os.kill(grandchild_pid, 9)
        self.assertFalse(
            _pid_alive(grandchild_pid),
            f"grandchild {grandchild_pid} survived the timeout — the ref "
            "subprocess is not reaped as a process group (the persistent "
            "torchrun worker would keep holding the GPU and OOM the next gate)",
        )


if __name__ == "__main__":
    unittest.main()
