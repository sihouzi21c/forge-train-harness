"""Multi-process launcher for data-parallel training.

Sets RANK, LOCAL_RANK, WORLD_SIZE environment variables for each
spawned process. Does not use any framework-level process group APIs.

Optional rank-0 nsys profiling
------------------------------

When the env var ``FORGE_NSYS_RANK0_OUTPUT`` is set, rank 0's Python
process is wrapped with::

    nsys profile -t cuda,osrt --output=<path> --force-overwrite=true \\
        python <child args>

Other ranks are unaffected. The resulting ``<path>.nsys-rep`` is
post-processed by the ``profile-snapshot`` dispatcher runner via
``tools.profile_render.render``. This is the only profiling hook on
the training path — the engine itself stays profile-agnostic.

The env contract (one variable, no NVTX, no per-step
record_function) is the SSOT for "how to capture a perf snapshot":
the dispatcher sets ``FORGE_NSYS_RANK0_OUTPUT`` for the
``profile-snapshot`` suite and leaves it unset everywhere else.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time


def _rank0_nsys_cmd(child_args: list[str]) -> list[str] | None:
    """Return the nsys-wrapped argv for rank 0, or None when not profiling."""
    output = os.environ.get("FORGE_NSYS_RANK0_OUTPUT")
    if not output:
        return None
    nsys = shutil.which("nsys")
    if nsys is None:
        raise RuntimeError(
            "FORGE_NSYS_RANK0_OUTPUT is set but 'nsys' was not found on PATH. "
            "Install NVIDIA Nsight Systems on the GPU host."
        )
    return [
        nsys,
        "profile",
        "-t",
        "cuda,osrt",
        "--cuda-graph-trace=node",
        "--output=" + output,
        "--force-overwrite=true",
        sys.executable,
        *child_args,
    ]


def _wait_with_fast_fail(
    processes: list[subprocess.Popen],
    *,
    poll_interval_s: float = 0.5,
    terminate_grace_s: float = 30.0,
) -> list[int]:
    """Block until every process exits; if one exits non-zero, immediately
    terminate the rest.

    Without this, a single rank-0 crash at startup keeps the surviving
    rank blocked for torch.distributed's default 600 s rendezvous
    timeout (the ``DistStoreError: Timed out after 601 seconds waiting
    for clients. 1/2 clients joined`` failure observed across 8.5 % of
    dev rounds in the 2026-05/06 sample — ~3.7 h cumulative wall-clock).
    Polling every ``poll_interval_s`` lets the launcher react in
    seconds instead of minutes; agents who copy
    ``dist.init_process_group(backend="nccl")`` from the reference (no
    ``timeout=`` kwarg) get the same benefit without having to
    remember the kwarg.

    SIGTERM is delivered first with a ``terminate_grace_s`` window so
    the child can flush its traceback, tear down CUDA contexts, and
    release NCCL handles before SIGKILL takes over. 30 s is comfortably
    above empirical Python+CUDA shutdown costs while still bounding
    the worst-case wait at ~30.5 s — versus 600 s without fast-fail.
    """
    while not all(p.poll() is not None for p in processes):
        failed = next(
            (p for p in processes if p.poll() is not None and p.returncode != 0),
            None,
        )
        if failed is not None:
            for other in processes:
                if other.poll() is None:
                    other.terminate()
                    try:
                        other.wait(timeout=terminate_grace_s)
                    except subprocess.TimeoutExpired:
                        other.kill()
                        other.wait()
            break
        time.sleep(poll_interval_s)
    # ``poll()`` populates ``returncode``; any process still running
    # at loop exit means every peer was already done — wait() it to
    # populate its returncode too.
    for p in processes:
        if p.returncode is None:
            p.wait()
    return [p.returncode for p in processes]


def main() -> int:
    world_size = int(os.environ.get("NUM_PROCS", "2"))
    if world_size < 1:
        print("NUM_PROCS must be >= 1", file=sys.stderr)
        return 1

    child_args = sys.argv[1:]
    if not child_args:
        print(
            "Usage: NUM_PROCS=N python evals/scripts/launch_dp.py [PYTHON_ARGS...]",
            file=sys.stderr,
        )
        return 1

    rank0_wrapped = _rank0_nsys_cmd(child_args)

    processes = []
    for rank in range(world_size):
        env = os.environ.copy()
        env["RANK"] = str(rank)
        env["LOCAL_RANK"] = str(rank)
        env["WORLD_SIZE"] = str(world_size)
        if rank == 0 and rank0_wrapped is not None:
            cmd = rank0_wrapped
        else:
            cmd = [sys.executable, *child_args]
        proc = subprocess.Popen(cmd, env=env)
        processes.append(proc)

    exit_codes = _wait_with_fast_fail(processes)
    failed = [(r, c) for r, c in enumerate(exit_codes) if c != 0]
    if failed:
        for rank, code in failed:
            print(f"Rank {rank} exited with code {code}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
