"""Unit tests for ``tools/gpu_job.py`` — the kind=job GPU-job runner.

The module submits one harness suite as an ephemeral ``cctl pytorchjob``
(the single submission path — ``nodes = 1`` by default, multi-node when
``[remote].nodes > 1``). These tests exercise:

* the create-argv builder (GPU count, nodes, entry command, env exports),
* the ``--dry-run`` field dump (no cctl needed),
* the submit→poll→exit-code lifecycle through a fake ``cctl`` (CCTL_BIN),
  for the succeeded / failed / wall-clock-overrun paths.

``cctl`` is mocked via the ``CCTL_BIN`` env-var override that
``cctl_common`` honours — no real CLI, no network.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools import gpu_job  # noqa: E402

_REMOTE = {
    "kind": "job",
    "project": "loopharness",
    "cluster": "paratera_shandong",
    "resource_pool": "faxin",
    "image": "infra/forge-train:tag",
    "gpu_count": 2,
    "gpu_model": "h100",
    "priority": "NORMAL",
    "billing_account_id": "N00007",
    "workspace": "",
    "hostname": "ds-555",
}


class BuildCreateArgsTest(unittest.TestCase):
    def test_requests_gpu_count_and_image(self) -> None:
        args = gpu_job.build_create_args(
            _REMOTE, suite="forward-align", workdir="$HOME/.forge_train/L1", loop_id="L1", extra=[]
        )
        # pytorchjob create with -o json; nodes defaults to 1 (single-node)
        self.assertEqual(args[:4], ["pytorchjob", "create", "-o", "json"])
        self.assertIn("--gpu", args)
        self.assertEqual(args[args.index("--gpu") + 1], "2")
        self.assertEqual(args[args.index("--nodes") + 1], "1")
        self.assertNotIn("--init-wait-seconds", args)
        self.assertEqual(args[args.index("--gpu-model") + 1], "h100")
        self.assertEqual(args[args.index("--image") + 1], "infra/forge-train:tag")
        self.assertEqual(args[args.index("--billing-account-id") + 1], "N00007")

    def test_entry_runs_suite_in_workdir_with_env(self) -> None:
        args = gpu_job.build_create_args(
            _REMOTE, suite="multistep", workdir="$HOME/.forge_train/L1", loop_id="L1", extra=[]
        )
        entry = args[args.index("--entry") + 1]
        self.assertIn("cd $HOME/.forge_train/L1", entry)
        # Invoked as ``python3 -m harness.cli`` (shim target), never bare
        # ``harness`` — the job pod has no <workdir>/bin on PATH.
        self.assertIn("python3 -m harness.cli run multistep", entry)
        self.assertNotIn("exec harness run", entry)
        self.assertIn("LOOP_ID=L1", entry)
        self.assertIn("FORGE_CONFIG_DIR=$HOME/.forge_train/L1/config", entry)
        self.assertIn("FORGE_TRAIN_DIR=$HOME/.forge_train/L1/.artifacts/forge_train", entry)
        self.assertIn("PYTHONPATH=$HOME/.forge_train/L1", entry)
        # persistent compile caches mirror remote_run.sh
        self.assertIn("TRITON_CACHE_DIR", entry)

    def test_extra_args_appended_to_suite(self) -> None:
        args = gpu_job.build_create_args(
            _REMOTE,
            suite="op-long",
            workdir="/w",
            loop_id="L1",
            extra=["attention", "--timeout", "12"],
        )
        entry = args[args.index("--entry") + 1]
        self.assertIn("python3 -m harness.cli run op-long attention --timeout 12", entry)

    def test_missing_required_field_raises(self) -> None:
        broken = {**_REMOTE, "image": ""}
        with self.assertRaises(gpu_job.GpuJobError):
            gpu_job.build_create_args(broken, suite="x", workdir="/w", loop_id="L1", extra=[])

    def test_empty_billing_omits_flag(self) -> None:
        args = gpu_job.build_create_args(
            {**_REMOTE, "billing_account_id": ""}, suite="x", workdir="/w", loop_id="L1", extra=[]
        )
        self.assertNotIn("--billing-account-id", args)

    def test_multi_node_passes_nodes_and_init_wait(self) -> None:
        args = gpu_job.build_create_args(
            {**_REMOTE, "nodes": 2}, suite="x", workdir="/w", loop_id="L1", extra=[]
        )
        self.assertEqual(args[args.index("--nodes") + 1], "2")
        # Workers may be scheduled long before the master pod; the init-wait
        # covers that window (only sent for a real multi-node topology).
        self.assertIn("--init-wait-seconds", args)
        # --gpu stays per-node: nodes=2 x gpu_count=2 => world_size 4.
        self.assertEqual(args[args.index("--gpu") + 1], "2")

    def test_invalid_nodes_fails_fast(self) -> None:
        for bad in (0, -3, "sixteen"):
            with self.assertRaises(gpu_job.GpuJobError):
                gpu_job.build_create_args(
                    {**_REMOTE, "nodes": bad}, suite="x", workdir="/w", loop_id="L1", extra=[]
                )


def _write_remote_toml(tmp: Path) -> Path:
    cfg = tmp / "config"
    cfg.mkdir(exist_ok=True)
    (cfg / "remote.toml").write_text(
        "[remote]\n"
        'kind = "job"\n'
        'project = "loopharness"\n'
        'cluster = "paratera_shandong"\n'
        'resource_pool = "faxin"\n'
        'image = "infra/forge-train:tag"\n'
        "gpu_count = 2\n"
        'gpu_model = "h100"\n'
        'priority = "NORMAL"\n'
        'billing_account_id = ""\n'
        'workspace = "/user/lishangzhan"\n'
        'hostname = "ds-555"\n',
        encoding="utf-8",
    )
    return cfg


class CanonicalJobTest(unittest.TestCase):
    """kind=job canonical-state preflight: the 2-GPU bridge cannot run on
    the 0-GPU gateway, so it is submitted as an ephemeral cctl job that runs
    ``evals.canonical_preflight`` (same module the ssh/devspace path runs on
    its GPU host) — reusing the identical create/submit/poll machinery."""

    def test_entry_runs_canonical_preflight_module(self) -> None:
        args = gpu_job.build_canonical_create_args(
            _REMOTE, workdir="$HOME/.forge_train/L1", loop_id="L1"
        )
        self.assertEqual(args[:4], ["pytorchjob", "create", "-o", "json"])
        self.assertEqual(args[args.index("--gpu") + 1], "2")
        entry = args[args.index("--entry") + 1]
        # Runs the preflight module directly, NOT ``harness run <suite>``.
        self.assertIn("python3 -m evals.canonical_preflight", entry)
        self.assertNotIn("harness.cli run", entry)
        self.assertIn("cd $HOME/.forge_train/L1", entry)
        self.assertIn("FORGE_CONFIG_DIR=$HOME/.forge_train/L1/config", entry)
        self.assertIn("LOOP_ID=L1", entry)

    def test_dry_run_canonical_dumps_preflight_command(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            cfg = _write_remote_toml(Path(raw))
            env_keep = {
                k: os.environ.get(k) for k in ("FORGE_CONFIG_DIR", "LOOP_ID", "LOOP_WEB_ID")
            }
            try:
                os.environ["FORGE_CONFIG_DIR"] = str(cfg)
                os.environ["LOOP_ID"] = "L1"
                os.environ.pop("LOOP_WEB_ID", None)
                import contextlib
                import io

                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = gpu_job.main(["--canonical", "--dry-run", "--outer", "1800"])
                out = buf.getvalue()
            finally:
                for k, v in env_keep.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            self.assertEqual(rc, 0)
            fields = dict(line.split("=", 1) for line in out.strip().splitlines() if "=" in line)
            self.assertEqual(fields["mode"], "job")
            self.assertEqual(fields["suite"], "canonical-preflight")
            self.assertIn("evals.canonical_preflight", fields["create_cmd"])
            self.assertIn("--gpu 2", fields["create_cmd"])

    def test_canonical_lifecycle_succeeded(self) -> None:
        # Submit→poll→exit-code lifecycle for the canonical job through a
        # fake cctl; the handle is keyed under the reserved suite name.
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            cfg = tmp / "config"
            cfg.mkdir()
            _git_workspace(tmp / "workspace")
            keep = {k: os.environ.get(k) for k in ("FORGE_CONFIG_DIR", "CCTL_BIN")}
            try:
                os.environ["FORGE_CONFIG_DIR"] = str(cfg)
                os.environ["CCTL_BIN"] = str(_write_fake_cctl(tmp, status="succeeded"))
                import contextlib
                import io

                entry = gpu_job._build_canonical_entry(workdir="/w", loop_id="L1")
                with (
                    contextlib.redirect_stderr(io.StringIO()),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    rc = gpu_job.run_suite(
                        _REMOTE,
                        suite=gpu_job._CANONICAL_SUITE,
                        workdir="/w",
                        loop_id="L1",
                        extra=[],
                        outer=600,
                        entry=entry,
                        poll_interval_s=0,
                    )
                self.assertEqual(rc, 0)
                self.assertTrue((tmp / "gpu_jobs" / "canonical-preflight.json").is_file())
            finally:
                for k, v in keep.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v


class DryRunTest(unittest.TestCase):
    def test_dry_run_dumps_create_command(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            cfg = _write_remote_toml(Path(raw))
            env_keep = {
                k: os.environ.get(k) for k in ("FORGE_CONFIG_DIR", "LOOP_ID", "LOOP_WEB_ID")
            }
            try:
                os.environ["FORGE_CONFIG_DIR"] = str(cfg)
                os.environ["LOOP_ID"] = "L1"
                os.environ.pop("LOOP_WEB_ID", None)
                import contextlib
                import io

                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = gpu_job.main(
                        ["--dry-run", "forward-align", "--outer", "780", "--budget", "600"]
                    )
                out = buf.getvalue()
            finally:
                for k, v in env_keep.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            self.assertEqual(rc, 0)
            fields = dict(line.split("=", 1) for line in out.strip().splitlines() if "=" in line)
            self.assertEqual(fields["mode"], "job")
            self.assertEqual(fields["gpu"], "2")
            self.assertEqual(fields["nodes"], "1")
            self.assertEqual(fields["outer"], "780")
            self.assertEqual(fields["workdir"], "/user/lishangzhan/.forge_train/L1")
            self.assertIn("pytorchjob create", fields["create_cmd"])
            self.assertIn("--gpu 2", fields["create_cmd"])

    def test_empty_workspace_fails_fast(self) -> None:
        # workspace = "" would resolve to the pod-local /root, which the
        # ephemeral GPU job cannot see — main must reject it (rc 2), never
        # silently fall back to "$HOME".
        with tempfile.TemporaryDirectory() as raw:
            cfg = _write_remote_toml(Path(raw))
            (cfg / "remote.toml").write_text(
                (cfg / "remote.toml")
                .read_text(encoding="utf-8")
                .replace('workspace = "/user/lishangzhan"', 'workspace = ""'),
                encoding="utf-8",
            )
            env_keep = {
                k: os.environ.get(k) for k in ("FORGE_CONFIG_DIR", "LOOP_ID", "LOOP_WEB_ID")
            }
            try:
                os.environ["FORGE_CONFIG_DIR"] = str(cfg)
                os.environ["LOOP_ID"] = "L1"
                os.environ.pop("LOOP_WEB_ID", None)
                rc = gpu_job.main(["--dry-run", "forward-align", "--outer", "780"])
            finally:
                for k, v in env_keep.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            self.assertEqual(rc, 2)


def _write_fake_cctl(tmp: Path, *, status: str) -> Path:
    """Fake cctl: ``job create`` returns id 555; ``job get`` returns
    *status*; ``job logs`` / ``job stop`` succeed."""
    bin_path = tmp / "fake-cctl"
    script = (
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "pytorchjob" ] && [ "$2" = "create" ]; then\n'
        '  printf \'{"id": 555, "name": "tasks/555", "status": "queued"}\\n\'\n'
        'elif [ "$1" = "pytorchjob" ] && [ "$2" = "get" ]; then\n'
        f'  printf \'{{"id": 555, "status": "{status}"}}\\n\'\n'
        'elif [ "$1" = "pytorchjob" ] && [ "$2" = "logs" ]; then\n'
        "  printf 'job-log-line\\n'\n"
        "fi\n"
        "exit 0\n"
    )
    bin_path.write_text(script, encoding="utf-8")
    bin_path.chmod(bin_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bin_path


class RunSuiteLifecycleTest(unittest.TestCase):
    def _run(self, *, status: str, outer: int) -> int:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            old = os.environ.get("CCTL_BIN")
            try:
                os.environ["CCTL_BIN"] = str(_write_fake_cctl(tmp, status=status))
                return gpu_job.run_suite(
                    _REMOTE,
                    suite="forward-align",
                    workdir="/w",
                    loop_id="L1",
                    extra=[],
                    outer=outer,
                )
            finally:
                if old is None:
                    os.environ.pop("CCTL_BIN", None)
                else:
                    os.environ["CCTL_BIN"] = old

    def test_succeeded_returns_zero(self) -> None:
        self.assertEqual(self._run(status="succeeded", outer=600), 0)

    def test_failed_returns_nonzero(self) -> None:
        self.assertEqual(self._run(status="failed", outer=600), 1)

    def test_overrun_stops_and_returns_timeout_rc(self) -> None:
        # outer=0 => deadline already passed after the first status read
        # (which reports "running"), so the job is stopped and the
        # timeout exit code is returned.
        self.assertEqual(self._run(status="running", outer=0), gpu_job._TIMEOUT_RC)


def _write_seq_cctl(tmp: Path, *, statuses: list[str]) -> Path:
    """Fake cctl whose ``job get`` walks *statuses* one per call (clamping
    at the last entry), so a single test can drive a queued→running→terminal
    transition. ``job create`` returns id 555; ``logs``/``stop`` succeed."""
    bin_path = tmp / "fake-cctl-seq"
    cnt = tmp / "seq.cnt"
    arr = " ".join(statuses)
    script = (
        "#!/usr/bin/env bash\n"
        f'CNT="{cnt}"\n'
        f"STATUSES=({arr})\n"
        'if [ "$1" = "pytorchjob" ] && [ "$2" = "create" ]; then\n'
        '  printf \'{"id": 555, "name": "tasks/555", "status": "queued"}\\n\'\n'
        'elif [ "$1" = "pytorchjob" ] && [ "$2" = "get" ]; then\n'
        '  n=$(cat "$CNT" 2>/dev/null || echo 0)\n'
        "  idx=$n; last=$(( ${#STATUSES[@]} - 1 ))\n"
        '  [ "$idx" -gt "$last" ] && idx=$last\n'
        '  printf \'{"id": 555, "status": "%s"}\\n\' "${STATUSES[$idx]}"\n'
        '  echo $(( n + 1 )) > "$CNT"\n'
        'elif [ "$1" = "pytorchjob" ] && [ "$2" = "logs" ]; then\n'
        "  printf 'job-log-line\\n'\n"
        "fi\n"
        "exit 0\n"
    )
    bin_path.write_text(script, encoding="utf-8")
    bin_path.chmod(bin_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bin_path


class TimeoutFromRunningTest(unittest.TestCase):
    """The exec wall-clock budget is anchored at the first ``running`` poll,
    NOT at submission — so an arbitrarily long Queued phase never counts
    against (or trips) the budget."""

    def _run(self, *, statuses: list[str], outer: int) -> tuple[int, str]:
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            old = os.environ.get("CCTL_BIN")
            try:
                os.environ["CCTL_BIN"] = str(_write_seq_cctl(tmp, statuses=statuses))
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    rc = gpu_job.run_suite(
                        _REMOTE,
                        suite="forward-align",
                        workdir="/w",
                        loop_id="L1",
                        extra=[],
                        outer=outer,
                        poll_interval_s=0,
                    )
                return rc, err.getvalue()
            finally:
                if old is None:
                    os.environ.pop("CCTL_BIN", None)
                else:
                    os.environ["CCTL_BIN"] = old

    def test_long_queue_is_not_killed_by_exec_budget(self) -> None:
        # outer=0 with a queued prefix: the OLD (submit-anchored) logic
        # would stop the job on the first poll; the new logic must let a
        # queued job run to its eventual success regardless of exec budget.
        rc, _ = self._run(statuses=["queued", "queued", "queued", "succeeded"], outer=0)
        self.assertEqual(rc, 0)

    def test_exec_budget_counts_from_running(self) -> None:
        # Job queues a while, then runs forever. The budget is anchored at
        # the running transition (not submission), so it is stopped only
        # after running — and the running-transition log proves the queue
        # phase did not trip the kill.
        rc, err = self._run(statuses=["queued", "queued", "running"], outer=0)
        self.assertEqual(rc, gpu_job._TIMEOUT_RC)
        self.assertIn("running", err.lower())

    def test_queue_duration_is_recorded(self) -> None:
        # The queued→running transition logs how long the job waited, for
        # later scheduling-latency analysis.
        _, err = self._run(statuses=["queued", "running", "succeeded"], outer=600)
        self.assertRegex(err, r"queued")


_GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
}


def _git_workspace(ws: Path) -> str:
    """Init a git repo at *ws* with one commit; return its HEAD sha."""
    ws.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **_GIT_ENV}
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True, env=env)
    (ws / "f.txt").write_text("1", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=ws, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "c1"], cwd=ws, check=True, env=env)
    return subprocess.run(
        ["git", "-C", str(ws), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def _advance_commit(ws: Path) -> str:
    env = {**os.environ, **_GIT_ENV}
    (ws / "f.txt").write_text("2", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=ws, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "c2"], cwd=ws, check=True, env=env)
    return subprocess.run(
        ["git", "-C", str(ws), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def _write_logging_cctl(tmp: Path, *, get_status: str = "succeeded") -> tuple[Path, Path]:
    """Fake cctl that appends every invocation's argv to a call-log file so
    a test can assert whether ``job create`` / ``job stop`` happened. ``job
    get`` reports *get_status*; ``create`` returns id 555."""
    calllog = tmp / "calls.log"
    bin_path = tmp / "fake-cctl-log"
    script = (
        "#!/usr/bin/env bash\n"
        f'echo "$@" >> "{calllog}"\n'
        'if [ "$1" = "pytorchjob" ] && [ "$2" = "create" ]; then\n'
        '  printf \'{"id": 555, "name": "tasks/555", "status": "queued"}\\n\'\n'
        'elif [ "$1" = "pytorchjob" ] && [ "$2" = "get" ]; then\n'
        f'  printf \'{{"id": 0, "status": "{get_status}"}}\\n\'\n'
        'elif [ "$1" = "pytorchjob" ] && [ "$2" = "logs" ]; then\n'
        "  printf 'log\\n'\n"
        "fi\n"
        "exit 0\n"
    )
    bin_path.write_text(script, encoding="utf-8")
    bin_path.chmod(bin_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bin_path, calllog


def _run_with_loop(
    cfg: Path, cctl_bin: Path, *, suite: str = "forward-align", outer: int = 600
) -> int:
    import contextlib
    import io

    keep = {k: os.environ.get(k) for k in ("FORGE_CONFIG_DIR", "LOOP_ID", "CCTL_BIN")}
    try:
        os.environ["FORGE_CONFIG_DIR"] = str(cfg)
        os.environ["LOOP_ID"] = "L1"
        os.environ["CCTL_BIN"] = str(cctl_bin)
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            return gpu_job.run_suite(
                _REMOTE,
                suite=suite,
                workdir="/w",
                loop_id="L1",
                extra=[],
                outer=outer,
                poll_interval_s=0,
            )
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _poll_with_loop(
    cfg: Path, cctl_bin: Path, *, suite: str = "forward-align", outer: int = 600
) -> int:
    import contextlib
    import io

    keep = {k: os.environ.get(k) for k in ("FORGE_CONFIG_DIR", "LOOP_ID", "CCTL_BIN")}
    try:
        os.environ["FORGE_CONFIG_DIR"] = str(cfg)
        os.environ["LOOP_ID"] = "L1"
        os.environ["CCTL_BIN"] = str(cctl_bin)
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            return gpu_job.poll_suite(suite=suite, outer=outer, poll_interval_s=0)
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class SubmitAndPollTest(unittest.TestCase):
    """Submit is one-shot (refuses a duplicate while a job is in flight, no
    silent re-attach); ``poll_suite`` (--poll) is the explicit, separate
    monitor that continues an in-flight job to terminal. The per-loop handle
    (``<loop_dir>/gpu_jobs/<suite>.json``) ties them together."""

    def _setup(self, tmp: Path) -> tuple[Path, Path, str]:
        cfg = tmp / "config"
        cfg.mkdir()
        head = _git_workspace(tmp / "workspace")
        return cfg, tmp / "gpu_jobs", head

    def _state(self, state_dir: Path, suite: str) -> dict:
        return json.loads((state_dir / f"{suite}.json").read_text(encoding="utf-8"))

    def test_submit_refuses_when_inflight_same_commit(self) -> None:
        # A non-consumed handle for this exact commit => submit must NOT
        # create a duplicate, must NOT silently re-attach/poll, and returns
        # _INFLIGHT_RC pointing the caller at --poll.
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            cfg, sd, head = self._setup(tmp)
            sd.mkdir()
            (sd / "forward-align.json").write_text(
                json.dumps(
                    {
                        "suite": "forward-align",
                        "job_id": "999",
                        "commit_sha": head,
                        "submitted_at": 1.0,
                        "running_since": None,
                        "status": "submitted",
                    }
                ),
                encoding="utf-8",
            )
            cctl_bin, calllog = _write_logging_cctl(tmp, get_status="succeeded")
            rc = _run_with_loop(cfg, cctl_bin)
            self.assertEqual(rc, gpu_job._INFLIGHT_RC)
            calls = calllog.read_text(encoding="utf-8") if calllog.exists() else ""
            self.assertNotIn("pytorchjob create", calls)  # no duplicate submit
            self.assertNotIn("pytorchjob get tasks/999", calls)  # no silent re-attach/poll
            self.assertEqual(self._state(sd, "forward-align")["status"], "submitted")

    def test_poll_monitors_existing_job_to_terminal(self) -> None:
        # --poll reads the handle, polls the in-flight job to terminal, and
        # marks it consumed — never submits.
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            cfg, sd, head = self._setup(tmp)
            sd.mkdir()
            (sd / "forward-align.json").write_text(
                json.dumps(
                    {
                        "suite": "forward-align",
                        "job_id": "999",
                        "commit_sha": head,
                        "submitted_at": 1.0,
                        "running_since": None,
                        "status": "submitted",
                    }
                ),
                encoding="utf-8",
            )
            cctl_bin, calllog = _write_logging_cctl(tmp, get_status="succeeded")
            rc = _poll_with_loop(cfg, cctl_bin)
            self.assertEqual(rc, 0)
            calls = calllog.read_text(encoding="utf-8")
            self.assertNotIn("pytorchjob create", calls)  # poll never submits
            self.assertIn("pytorchjob get tasks/999", calls)  # polled the recorded job
            self.assertEqual(self._state(sd, "forward-align")["status"], "consumed")

    def test_poll_with_no_inflight_job_returns_no_job_rc(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            cfg, _sd, _head = self._setup(tmp)  # no handle written
            cctl_bin, calllog = _write_logging_cctl(tmp, get_status="succeeded")
            rc = _poll_with_loop(cfg, cctl_bin)
            self.assertEqual(rc, gpu_job._NO_JOB_RC)
            self.assertNotIn("job", calllog.read_text(encoding="utf-8") if calllog.exists() else "")

    def test_commit_change_submits_fresh_and_stops_stale(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            cfg, sd, head1 = self._setup(tmp)
            head2 = _advance_commit(tmp / "workspace")
            self.assertNotEqual(head1, head2)
            sd.mkdir()
            (sd / "forward-align.json").write_text(
                json.dumps(
                    {
                        "suite": "forward-align",
                        "job_id": "999",
                        "commit_sha": head1,
                        "submitted_at": 1.0,
                        "running_since": 2.0,
                        "status": "running",
                    }
                ),
                encoding="utf-8",
            )
            cctl_bin, calllog = _write_logging_cctl(tmp, get_status="succeeded")
            rc = _run_with_loop(cfg, cctl_bin)
            self.assertEqual(rc, 0)
            calls = calllog.read_text(encoding="utf-8")
            self.assertIn("pytorchjob stop 999", calls)  # stale job stopped
            self.assertIn("pytorchjob create", calls)  # fresh submit
            rec = self._state(sd, "forward-align")
            self.assertEqual(rec["job_id"], "555")
            self.assertEqual(rec["commit_sha"], head2)

    def test_consumed_record_triggers_fresh_submit(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            cfg, sd, head = self._setup(tmp)
            sd.mkdir()
            (sd / "forward-align.json").write_text(
                json.dumps(
                    {
                        "suite": "forward-align",
                        "job_id": "999",
                        "commit_sha": head,
                        "submitted_at": 1.0,
                        "running_since": 2.0,
                        "status": "consumed",
                    }
                ),
                encoding="utf-8",
            )
            cctl_bin, calllog = _write_logging_cctl(tmp, get_status="succeeded")
            rc = _run_with_loop(cfg, cctl_bin)
            self.assertEqual(rc, 0)
            calls = calllog.read_text(encoding="utf-8")
            self.assertIn("pytorchjob create", calls)  # consumed => do not re-attach
            self.assertNotIn("pytorchjob get tasks/999", calls)


if __name__ == "__main__":
    unittest.main()
