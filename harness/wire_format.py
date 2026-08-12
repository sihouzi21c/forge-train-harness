"""SSOT for the harness stdout wire format and trajectory parsers.

Every gate script the harness drives speaks a tiny line-oriented
protocol on stdout (and on the ``LOSS_DUMP_FILE`` written by the
customer training stack when a ``[LOSS]`` line is emitted per step).
This module is the **single source of truth** for those line formats
and their parsers — anything else that wants to emit or parse them
must route through here.

Recognised lines
----------------
``[LOSS] step=<int> global_loss=<float> grad_norm=<float> time_s=<float> [mfu_e2e_standard=<float>]``
    Per-step trajectory line for loss gates. Variants with tag
    ``LOSS_REF`` / ``LOSS_RES`` are emitted by the resume gate so
    reference vs resumed trajectories can be parsed side by side from
    the same combined stdout.

Float precision (canonical spec, gate-decisive fields)
------------------------------------------------------
Gate-decisive numeric fields — ``global_loss`` and ``grad_norm`` —
MUST be emitted with at least 9 significant decimal digits so any
fp32 value uniquely round-trips through ``float(str)``. The canonical
format constant is :data:`LOSS_FLOAT_FORMAT` (``.9e``). Lower precision
(e.g. ``%.6e`` or ``%.10f`` on small magnitudes) silently collapses
sub-print fp32 ULP drift into bitwise-equal strings and lets it
escape the per-step ``max_abs_diff == 0`` gate.

The non-gate fields ``time_s`` and ``mfu_e2e_standard`` retain their
existing precision (``%.6f``); they are not bitwise-compared.
``mfu_e2e_standard`` is emitted already in **0–100 percentage scale**
(``flops_per_step / (step_time * peak_total) * 100``); every consumer
(dispatcher gate, profile snapshot, mfu_record) reads it as-is and never
re-scales — the gate threshold ``mfu_e2e_target`` is on the same scale.

Both the ref entry (``ref/reference/train_pure_mup_mtp.py``) and the
self-developed engine (``train_engine/.../train_loop.py``) restate this
constant inline — they are standalone modules that must not import
from ``harness`` — and reference this docstring for the canonical
spec. ``test_wire_format`` pins the agreement.

``iteration  <step>/<total> | … | lm loss: <float> | …`` (Megatron stdout)
    Fallback parser for the case where the L0 ref script does not
    install a per-step ``[LOSS]`` emitter and only Megatron's native
    ``--log-throughput`` output is available. Step + loss are
    extracted; everything else is ignored.

Strict vs tolerant variants
---------------------------
Trajectory parsers come in two flavours, deliberately:

* ``parse_loss_lines`` / ``parse_loss_lines_to_dict`` are **strict** —
  non-finite numeric fields raise ``ValueError``. They are used on
  ours-side stdout where any NaN / inf means a real gate failure and
  must not be silently dropped.

* ``parse_loss_dump_file`` is **tolerant** — NaN entries are silently
  skipped. Used on the ref-side loss dump where Megatron may
  legitimately write a ``nan`` placeholder for the first few steps
  before the loss dict is fully populated; treating those as gate
  failures would poison every long-horizon / op-long / loss-gate run.

* ``parse_loss_dump_with_grad`` recovers **both** gate-decisive fields
  (``global_loss`` + ``grad_norm``) from the ref-side loss dump for the
  Stage 1 bitwise gates (bitwise-singlecard/bitwise-multicard/bitwise-perf), which assert per-step bitwise
  alignment on grad_norm as well as loss. Same per-field NaN tolerance
  as ``parse_loss_dump_file``.

This module is a leaf inside ``harness/`` (stdlib only) so both
``evals/`` and ``tools/`` can depend on it without inducing a cycle.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "LOSS_FLOAT_FORMAT",
    "LOSS_LINE_PATTERN",
    "MEGATRON_STDOUT_PATTERN",
    "parse_key_values",
    "parse_loss_dump_file",
    "parse_loss_dump_with_grad",
    "parse_loss_lines",
    "parse_loss_lines_to_dict",
    "parse_stdout_loss",
]


LOSS_FLOAT_FORMAT = ".9e"
"""Canonical f-string format spec for gate-decisive floats on ``[LOSS]`` lines.

Pinned at ``.9e`` because fp32 needs ≥9 significant decimal digits to round-trip
uniquely through ``float(str)``. Emitters that cannot import this module
(the L0 ref entry and the self-developed engine, both standalone) restate
the literal ``.9e`` inline; see the module docstring "Float precision" section.
"""


_NUMBER = r"-inf|inf|nan|[\d.eE+\-]+"
_MFU_PATTERN = re.compile(r"mfu_e2e_standard=([\d.eE+\-]+|nan|-?inf)", re.IGNORECASE)

LOSS_LINE_PATTERN = re.compile(
    rf"\[(?P<tag>[A-Z_]+)\]\s+step=(?P<step>\d+)\s+global_loss=(?P<loss>{_NUMBER})\s+"
    rf"grad_norm=(?P<grad>{_NUMBER})\s+time_s=(?P<time>{_NUMBER})",
    re.IGNORECASE,
)

# Simplified subset used when reading the ref-side loss-dump file.
# Tolerates ``nan`` placeholders (the producer writes them when a
# loss field is absent in a step's loss_dict); strict callers should
# prefer ``parse_loss_lines`` which raises on non-finite values.
_LOSS_DUMP_SUBSET = re.compile(r"\[LOSS\]\s+step=(\d+)\s+global_loss=([\d.eE+\-]+|nan)")

# Loss + grad_norm subset for the Stage 1 bitwise gates (bitwise-singlecard/bitwise-multicard/bitwise-perf),
# which assert per-step bitwise alignment on **both** gate-decisive
# fields. Same ``nan`` tolerance as ``_LOSS_DUMP_SUBSET`` on each
# field independently (a step with a ``nan`` grad carries no usable
# bitwise grad baseline and is dropped by ``parse_loss_dump_with_grad``).
_LOSS_DUMP_GRAD_SUBSET = re.compile(
    r"\[LOSS\]\s+step=(\d+)\s+global_loss=([\d.eE+\-]+|nan)\s+grad_norm=([\d.eE+\-]+|nan)"
)

MEGATRON_STDOUT_PATTERN = re.compile(
    r"iteration\s+(\d+)\s*/\s*\d+.*?lm\s+loss:\s*([\d.eE+\-]+)",
    re.IGNORECASE,
)


def parse_key_values(line: str) -> dict[str, str]:
    """Parse space-separated ``key=value`` tokens from a single line."""
    tokens: dict[str, str] = {}
    for token in line.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        tokens[key] = value
    return tokens


def parse_loss_lines(output: str, tag: str = "LOSS") -> list[dict[str, Any]]:
    """Parse strict ``[<tag>] step=N global_loss=X grad_norm=Y time_s=Z [mfu_e2e_standard=W]`` lines.

    Wire format (canonical spec — any consumer must match this):

        ``[LOSS] step=<int> global_loss=<float> grad_norm=<float> time_s=<float> [mfu_e2e_standard=<float>]``

    Variant tags: ``LOSS_REF``, ``LOSS_RES`` (resume gate).

    Non-finite numeric values raise ``ValueError`` — callers on ours-side
    stdout must not silently drop NaN/inf. For the tolerant loss-dump
    flavour, see :func:`parse_loss_dump_file`.
    """
    tag_pattern = re.compile(
        rf"\[{re.escape(tag)}\]\s+step=(\d+)\s+global_loss=({_NUMBER})\s+"
        rf"grad_norm=({_NUMBER})\s+time_s=({_NUMBER})",
        re.IGNORECASE,
    )
    results: list[dict[str, Any]] = []
    for line in output.split("\n"):
        m = tag_pattern.search(line)
        if not m:
            continue
        global_loss = float(m.group(2))
        grad_norm = float(m.group(3))
        time_s = float(m.group(4))
        for field, value in (
            ("global_loss", global_loss),
            ("grad_norm", grad_norm),
            ("time_s", time_s),
        ):
            if not math.isfinite(value):
                raise ValueError(f"Non-finite [{tag}] {field} at step {m.group(1)}: {value}")
        entry: dict[str, Any] = {
            "step": int(m.group(1)),
            "global_loss": global_loss,
            "grad_norm": grad_norm,
            "time_s": time_s,
        }
        mm = _MFU_PATTERN.search(line)
        if mm:
            entry["mfu_e2e_standard"] = float(mm.group(1))
        results.append(entry)
    return results


def parse_loss_lines_to_dict(output: str) -> dict[int, float]:
    """Parse strict ``[LOSS] step=N global_loss=X …`` lines into ``{step: loss}``.

    Duplicate steps and non-finite values are protocol errors.
    """
    out: dict[int, float] = {}
    for entry in parse_loss_lines(output):
        step = int(entry["step"])
        if step in out:
            raise ValueError(f"Duplicate [LOSS] step: {step}")
        out[step] = float(entry["global_loss"])
    return out


def parse_loss_dump_file(loss_file: Path) -> dict[int, float]:
    """Parse the ref-side ``LOSS_DUMP_FILE`` into ``{step: loss}``.

    Tolerates ``nan`` placeholders (the producer writes them when a
    loss field is absent in a step's loss_dict). Steps with NaN loss are
    omitted from the returned mapping rather than poisoning downstream
    math; the caller can treat missing steps as a gate failure.

    See :func:`parse_loss_lines` for the strict variant used on
    ours-side stdout.
    """
    if not loss_file.exists():
        return {}
    out: dict[int, float] = {}
    for line in loss_file.read_text(errors="replace").splitlines():
        m = _LOSS_DUMP_SUBSET.search(line)
        if not m:
            continue
        try:
            step = int(m.group(1))
            loss = float(m.group(2))
        except ValueError:
            continue
        if loss != loss:  # NaN
            continue
        out[step] = loss
    return out


def parse_loss_dump_with_grad(loss_file: Path) -> dict[int, dict[str, float]]:
    """Parse the ref-side ``LOSS_DUMP_FILE`` into ``{step: {global_loss, grad_norm}}``.

    The Stage 1 bitwise gates (bitwise-singlecard / bitwise-multicard / bitwise-perf) require per-step bitwise
    alignment on **both** ``global_loss`` and ``grad_norm`` (see
    ``prompt/develop_prompt/_shared/stage1/bitwise-multicard.md`` §Gate). This is the
    SSOT that recovers the grad-norm baseline the loss-only
    :func:`parse_loss_dump_file` deliberately drops.

    A step whose ``global_loss`` **or** ``grad_norm`` is ``nan`` is
    omitted entirely — it carries no usable bitwise baseline, mirroring
    the tolerant NaN policy of :func:`parse_loss_dump_file`. The caller
    treats missing window steps as a gate failure.
    """
    if not loss_file.exists():
        return {}
    out: dict[int, dict[str, float]] = {}
    for line in loss_file.read_text(errors="replace").splitlines():
        m = _LOSS_DUMP_GRAD_SUBSET.search(line)
        if not m:
            continue
        try:
            step = int(m.group(1))
            loss = float(m.group(2))
            grad = float(m.group(3))
        except ValueError:
            continue
        if loss != loss or grad != grad:  # NaN in either field
            continue
        out[step] = {"global_loss": loss, "grad_norm": grad}
    return out


def parse_stdout_loss(stdout_path: Path) -> dict[int, float]:
    """Fallback parser for ref-script stdout (Megatron ``--log-throughput``).

    Megatron emits lines such as ``iteration  100/ 1000 | … | lm loss:
    1.234E+00 | …``; we extract the step + loss pair so dispatchers can
    fall back to stdout when the ref script did not write a loss
    file (e.g. when the customer training stack has no per-step
    ``[LOSS]`` emitter wired up).
    """
    out: dict[int, float] = {}
    if not stdout_path.exists():
        return out
    for line in stdout_path.read_text(errors="replace").splitlines():
        m = MEGATRON_STDOUT_PATTERN.search(line)
        if not m:
            continue
        try:
            step = int(m.group(1))
            loss = float(m.group(2))
        except ValueError:
            continue
        out[step] = loss
    return out
