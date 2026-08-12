"""resume-startup gate runner.

Reproduces a real resume from ``RESUME_SAVE_STEP`` on the engine's own
streaming dataloader, times the GPU-idle seek window (build loader -> advance
to the saved stream position -> first resumed micro-batch), runs
``RESUME_POST_STEPS`` resumed steps, and emits a single machine-readable

    [RESUME_STARTUP] seconds=<float>

line. The dispatcher (``_run_resume_startup``) owns the verdict: it parses that
marker and compares against the ``resume_startup_budget_s`` threshold it reads
from the frozen registry (``config/eval.toml``) — the budget is an ours-only
verdict threshold and is deliberately NOT carried in the ours product, so this
runner never sees it and always exits cleanly once the marker is emitted.

Every gate-shape scalar (``RESUME_SAVE_STEP`` / ``RESUME_POST_STEPS`` /
``GRAD_ACCUM_STEPS`` / ``MICRO_BATCH_SIZE`` / ``SEQ_LENGTH`` / ``WORLD_SIZE`` /
``TENSOR_PARALLEL_SIZE`` / ``SEED``) is read from the single ``--config
<product>`` the dispatcher passes, via ``_gate_entry.GateInputs``. Nothing but
that one path and the launch_dp rendezvous identity crosses the process
boundary.

The seek (``Dataloader.advance``) is the real engine code path; the canonical
model weights are not needed to time it (the replay cost is pure
tokenize+stream), so this runner stays cheap and weight-free. The GPU is only
touched to mirror "enters training"; it stays GPU-optional so the gate also
runs on a CPU box, where the replay cost is identical.
"""

from __future__ import annotations

import os
import sys
import time

from _gate_entry import gate_inputs

from evals.resume_startup_gate import (
    emit_resume_startup_marker,
    measure_resume_startup,
)

_GATE = gate_inputs()


def main() -> int:
    import torch
    from training_engine_tensor.dataloader import build_dataloader

    # The gated quantity is the dataloader seek (CPU tokenize+stream); the GPU
    # is only touched to mirror "enters training". Stay GPU-optional so the
    # gate runs on a CPU box too — the replay cost is identical.
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    save_step = int(_GATE.require("RESUME_SAVE_STEP"))
    post_steps = int(_GATE.require("RESUME_POST_STEPS"))
    grad_accum = int(_GATE.require("GRAD_ACCUM_STEPS"))
    mbs = int(_GATE.require("MICRO_BATCH_SIZE"))
    seq = int(_GATE.require("SEQ_LENGTH"))
    world_size = int(_GATE.require("WORLD_SIZE"))
    # The ours engine has no tensor parallelism; the dataloader shards by the
    # data-parallel size only. With ref-side TP the gate's WORLD_SIZE is the
    # total GPU count, so the dataloader DP = WORLD_SIZE / TENSOR_PARALLEL_SIZE
    # (TP defaults to 1, i.e. dp == world_size, for the no-TP variants).
    tp_size = int(_GATE.get("TENSOR_PARALLEL_SIZE", "1"))
    dp_world_size = world_size // tp_size
    seed = int(_GATE.get("SEED", "1234"))
    dp_rank = int(_GATE.get("DP_RANK", "0"))
    data_path = _GATE.require("DATA_PATH")

    # Bring the CUDA context up before timing so the seek window measures the
    # dataloader, not one-off device init (constant across resume positions).
    if use_cuda:
        torch.cuda.init()
        torch.cuda.synchronize()

    consumed = save_step * grad_accum

    # One loader: the timing seam captures the instance it builds so the same
    # already-advanced loader serves the post-resume steps and gets an ordered
    # teardown — a SIGABRT from lingering pyarrow/stream threads at interpreter
    # exit is treated as a gate failure, so the runner must exit cleanly.
    built: dict[str, object] = {}

    def build_loader():
        dl = build_dataloader(
            data_path=data_path,
            dp_rank=dp_rank,
            world_size=dp_world_size,
            micro_batch_size=mbs,
            seq_length=seq,
            seed=seed,
        )
        built["dl"] = dl
        return dl

    loader = None
    try:
        startup_s = measure_resume_startup(build_loader, consumed, clock=time.perf_counter)
        emit_resume_startup_marker(
            startup_s,
            rank=int(os.environ.get("RANK", "0")),
            emit=lambda line: print(line, flush=True),
        )

        # Honour "run POST_STEPS steps": continue the resumed loader past the
        # seek (measure already fetched the first micro-batch of step 0).
        loader = built["dl"]
        for step in range(post_steps):
            pulls = grad_accum - 1 if step == 0 else grad_accum
            for _mb in range(pulls):
                next(loader)["tokens"].to(device, non_blocking=True)
        if use_cuda:
            torch.cuda.synchronize()
    finally:
        if loader is not None:
            loader.close()
        if use_cuda:
            torch.cuda.empty_cache()

    # Verdict is dispatcher-owned: it parses the [RESUME_STARTUP] marker above
    # and compares against the frozen ``resume_startup_budget_s`` (registry, not
    # product). The runner just needs a clean exit — a SIGABRT (returncode=-6)
    # from lingering pyarrow/stream threads at interpreter exit is itself a gate
    # failure, so ordered teardown happened in the finally block above.
    return 0


if __name__ == "__main__":
    sys.exit(main())
