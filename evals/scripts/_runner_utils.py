"""Shared utilities for ``evals/scripts/eval_*.py`` gate runners.

The runners themselves should stay thin (env → ``TrainLoopConfig`` → engine
call); anything that runs **around** the engine call — process termination,
pre-flight environment checks, log conventions — belongs here so the same
behaviour applies uniformly across every Stage-1 gate.

Imported as a same-directory module (``from _runner_utils import ...``);
``evals/scripts/`` has no ``__init__.py`` and the scripts are launched as
plain Python files, so the script directory is on ``sys.path[0]`` and the
import resolves without package gymnastics.
"""

from __future__ import annotations

import os
import subprocess


def abort_if_gpu_dirty() -> None:
    """Fail fast at runner startup if the target GPU(s) have leftover
    compute processes from prior killed runs.

    Rationale — see ``02ad357194dc`` R4/R5/R6 + ``bc2ea8320086`` R3 notes:
    when a gate run is killed externally (transport timeout, OOM mid-step,
    manual abort) torchrun does not always propagate SIGTERM to every rank
    worker, so orphan workers can keep tens of GiB of CUDA memory live on
    the device. The next gate run then either OOMs at allocation time
    (loud, recoverable) or silently records a contaminated MFU (quiet,
    misleading). This pre-flight makes the symptom loud and actionable
    *before* any CUDA work begins.

    Only rank 0 of the current torchrun job runs the check — every rank
    sees the same physical GPUs and we want a single, deterministic
    diagnostic instead of an N-way race on ``nvidia-smi`` output.

    Silently a no-op when ``nvidia-smi`` is unavailable (developer Mac,
    CPU-only CI). The gate would not be functional in those environments
    anyway, so skipping the check is safe — actual GPU-required suites
    fail downstream with a clearer message.

    Skipped when ``FORGE_NSYS_RANK0_OUTPUT`` is set (``nsys profile``
    wrapper creates a CUDA context that shows up as a compute app, causing
    a false positive — the ``profile-snapshot`` suite is the only caller
    that sets this env var).
    """
    if int(os.environ.get("RANK", "0")) != 0:
        return
    if os.environ.get("FORGE_NSYS_RANK0_OUTPUT"):
        return
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return
    if not out:
        return
    raise RuntimeError(
        "GPU has leftover compute processes before this gate started:\n"
        f"{out}\n"
        "\n"
        "Likely cause: a prior gate run was killed (transport timeout, OOM, "
        "manual abort) and torchrun did not propagate SIGTERM to every rank "
        "worker, so orphan worker(s) are still holding CUDA memory on the "
        "device. To recover on the remote:\n"
        "  pkill -f 'evals/scripts/eval_'\n"
        "  pkill -f torchrun || true\n"
        "  nvidia-smi   # confirm GPUs are clean\n"
        "then re-run the gate. If processes survive `pkill`, use `kill -9` on "
        "their PIDs."
    )
