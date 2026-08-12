"""Self-developed training engine package — agent-loop write surface.

The agent loop owns this directory: ``backward``, ``forward``,
``optimizer``, ``parameters``, ``nccl``, ``kernels``,
``triton_kernels``, ``dataloader``, etc. are implemented under
``workload/src/training_engine_tensor/`` and are the SSOT for the
ours-side training stack (matching the L0 ref script
SSOT on the Megatron side). The authoritative submodule list lives in
:mod:`training_engine_tensor.train_loop` — keep this docstring in sync
with that module's docstring whenever the surface changes.

The only contract this package exposes to the harness layer is
:mod:`training_engine_tensor.train_loop` — see that module's docstring
for the stable signature and stdout grammar that all long-running
gates depend on.
"""

from training_engine_tensor.train_loop import TrainLoopConfig, run_training_loop

__all__ = ["TrainLoopConfig", "run_training_loop"]
