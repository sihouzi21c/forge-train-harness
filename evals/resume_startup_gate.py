"""Resume startup-time gate (M5).

A correct resume seeks the dataloader to the checkpoint position in O(1) —
restoring a saved stream cursor, not re-streaming the consumed data. The
replay-by-discard loader (``Dataloader.advance`` pulling and discarding
``consumed`` micro-batches) makes the resume startup grow linearly with the
checkpoint step: the GPU sits idle re-tokenizing every prior step before
training continues.

This gate bounds that GPU-idle resume window. Resuming from ``save_step`` and
fetching the first training micro-batch must complete within ``budget_s``
wall-clock. It fails the linear-replay loader and passes an O(1) cursor seek.

SSOT: the pass/fail rule lives only in :func:`evaluate_resume_startup`; the
budget value lives only in ``[evals.<suite>].resume_startup_budget_s``; the
one timing path lives only in :func:`measure_resume_startup` (shared by the
real engine runner and the unit tests through an injected loader factory).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

# The engine runner emits exactly one such line; the dispatcher parses it.
_LINE_RE = re.compile(r"^\[RESUME_STARTUP\]\s+seconds=([0-9]+\.?[0-9]*)\s*$", re.M)


@dataclass(frozen=True)
class ResumeStartupVerdict:
    """Outcome of the resume startup-time gate."""

    passed: bool
    startup_s: float
    budget_s: float
    save_step: int
    consumed_microbatches: int
    summary: str


def evaluate_resume_startup(
    startup_s: float,
    budget_s: float,
    *,
    save_step: int = 0,
    consumed_microbatches: int = 0,
) -> ResumeStartupVerdict:
    """Pure pass/fail decision for the resume startup gate.

    ``budget_s <= 0`` disables the gate (always passes), mirroring the
    ``mfu_e2e_target`` / ``loss_rel_threshold`` "0 = off" convention. A
    negative ``startup_s`` is a measurement bug rather than a slow resume, so
    it fails fast.
    """
    if startup_s < 0:
        raise ValueError(f"startup_s must be >= 0, got {startup_s!r}")
    if budget_s <= 0:
        return ResumeStartupVerdict(
            passed=True,
            startup_s=startup_s,
            budget_s=budget_s,
            save_step=save_step,
            consumed_microbatches=consumed_microbatches,
            summary=f"resume-startup gate disabled (budget_s={budget_s})",
        )
    passed = startup_s <= budget_s
    verb = "within" if passed else "exceeds"
    return ResumeStartupVerdict(
        passed=passed,
        startup_s=startup_s,
        budget_s=budget_s,
        save_step=save_step,
        consumed_microbatches=consumed_microbatches,
        summary=(
            f"resume from step {save_step}: startup {startup_s:.2f}s {verb} "
            f"budget {budget_s:.2f}s "
            f"(seek replayed {consumed_microbatches} micro-batches)"
        ),
    )


class ResumableLoader(Protocol):
    """Duck-typed contract ``measure_resume_startup`` drives: the engine
    ``build_dataloader`` product and the unit-test fakes both provide
    ``advance`` (cursor seek / replay) and iteration."""

    def advance(self, count: int) -> None: ...

    def __next__(self) -> object: ...


def measure_resume_startup(
    build_loader: Callable[[], ResumableLoader],
    consumed: int,
    *,
    clock: Callable[[], float],
) -> float:
    """Time the resume seek window: build the loader, advance it to the saved
    stream position, and fetch the first resumed micro-batch.

    ``build_loader`` is injected so the real engine ``build_dataloader`` and a
    fake loader (unit tests) exercise this one timing path; ``clock`` is
    injected for deterministic tests. The returned wall-clock is the quantity
    the gate bounds — the GPU-idle stretch before training can resume.
    """
    if consumed < 0:
        raise ValueError(f"consumed must be >= 0, got {consumed}")
    t0 = clock()
    loader = build_loader()
    loader.advance(consumed)  # O(1) for a cursor seek; O(consumed) for replay
    next(loader)
    return clock() - t0


def emit_resume_startup_marker(
    startup_s: float, *, rank: int, emit: Callable[[str], None] = print
) -> None:
    """Emit the ``[RESUME_STARTUP]`` marker — DP rank 0 only.

    launch_dp fans the runner out into ``world_size`` processes whose stdout
    all land in the one ``ours.log``; :func:`parse_resume_startup_seconds`
    requires EXACTLY one marker line, so every rank emitting would fail the
    gate structurally (found N, expected 1). All ranks still measure — only
    the emission is rank-guarded.
    """
    if rank != 0:
        return
    emit(f"[RESUME_STARTUP] seconds={startup_s:.6f}")


def parse_resume_startup_seconds(output: str) -> float:
    """Extract the single ``[RESUME_STARTUP] seconds=<f>`` value from runner
    stdout.

    Fail fast on zero or multiple matches — an absent or ambiguous marker
    means the runner did not emit a trustworthy measurement.
    """
    matches = _LINE_RE.findall(output)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one [RESUME_STARTUP] line, found {len(matches)}")
    return float(matches[0])
