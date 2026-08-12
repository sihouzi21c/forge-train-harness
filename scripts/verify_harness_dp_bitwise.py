#!/usr/bin/env python3
"""Bitwise regression check for the ``harness_dp`` torch-ref refactor (Step 1).

The refactor moves the M1–M5 capture call sites *out of* the runtime
source-injection bridge (``ref/bridges/interposer.py``) and *into* the torch
refs themselves (``train_pure_mup_mtp.py`` / ``train_qwen3_dense.py`` →
``evals.harness_dp``). The acceptance bar is **pure refactor**: every ref-side
capture artifact must be byte-identical before and after.

This script does the *comparison* half. The GPU runs that produce the artifacts
happen on a leased devspace (the torch refs import ``flash_attn``, a CUDA-only
wheel that never imports on Mac), driven by the existing eval harness:

  1. On the pre-refactor commit, capture every gate of every torch suite and
     archive each gate's dump under  ``<baseline>/<suite>/<gate>/``.
  2. On the refactor branch, repeat into ``<after>/<suite>/<gate>/``.
  3. Run this script:  ``verify_harness_dp_bitwise.py --baseline B --after A``.

Each ``<suite>/<gate>/`` dir is expected to hold the capture dump produced by
``evals.harness_hook`` (a JSON dict ``key -> {hash, shape, dtype}``; basename
from ``[ref].ref_capture_basename``, default ``ref_capture.pt``), its sibling
``<dump>.graph.json`` (per-module fwd/bwd execution order), and optionally a
``loss.txt`` holding the ``[LOSS] …`` trajectory lines.

Per gate the comparison is three-way:
  * hash dict — identical key set AND identical per-key ``hash``;
  * ``.graph.json`` — byte-for-byte;
  * ``loss.txt`` — line-for-line (only when present on both sides).

Any missing/extra key, differing hash, differing graph, or differing loss line
is a FAIL. Exit code 0 iff every compared gate is byte-identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_SUITES = ("dense_training", "dense_training_1b", "dense_training_qwen3")
DEFAULT_CAPTURE_BASENAME = "ref_capture.pt"
LOSS_BASENAME = "loss.txt"


class GateResult:
    def __init__(self, suite: str, gate: str) -> None:
        self.suite = suite
        self.gate = gate
        self.errors: list[str] = []
        self.checked: list[str] = []

    @property
    def ok(self) -> bool:
        return not self.errors

    def fail(self, msg: str) -> None:
        self.errors.append(msg)


def _load_hash_dict(path: Path) -> dict[str, dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(raw).__name__}")
    return raw


def _compare_hash_dicts(res: GateResult, base: Path, after: Path) -> None:
    b = _load_hash_dict(base)
    a = _load_hash_dict(after)
    bk, ak = set(b), set(a)

    missing = sorted(bk - ak)
    extra = sorted(ak - bk)
    if missing:
        res.fail(
            f"hash keys MISSING in after ({len(missing)}): {missing[:8]}"
            + (" …" if len(missing) > 8 else "")
        )
    if extra:
        res.fail(
            f"hash keys EXTRA in after ({len(extra)}): {extra[:8]}"
            + (" …" if len(extra) > 8 else "")
        )

    mismatched = []
    for key in sorted(bk & ak):
        bh = b[key].get("hash") if isinstance(b[key], dict) else None
        ah = a[key].get("hash") if isinstance(a[key], dict) else None
        if bh != ah:
            mismatched.append((key, bh, ah))
    for key, bh, ah in mismatched[:8]:
        res.fail(f"hash MISMATCH '{key}': baseline={bh} after={ah}")
    if len(mismatched) > 8:
        res.fail(f"… and {len(mismatched) - 8} more hash mismatches")

    if not (missing or extra or mismatched):
        res.checked.append(f"hash ({len(bk)} keys)")


def _compare_bytes(res: GateResult, label: str, base: Path, after: Path) -> None:
    bb = base.read_bytes()
    ab = after.read_bytes()
    if bb != ab:
        res.fail(f"{label} BYTES differ ({len(bb)} vs {len(ab)} bytes)")
    else:
        res.checked.append(label)


def _compare_loss(res: GateResult, base: Path, after: Path) -> None:
    bl = base.read_text(encoding="utf-8").splitlines()
    al = after.read_text(encoding="utf-8").splitlines()
    if len(bl) != len(al):
        res.fail(f"{LOSS_BASENAME} line count differs ({len(bl)} vs {len(al)})")
    n = min(len(bl), len(al))
    diffs = [(i, bl[i], al[i]) for i in range(n) if bl[i] != al[i]]
    for i, b_line, a_line in diffs[:5]:
        res.fail(f"{LOSS_BASENAME} line {i + 1} differs:\n    - {b_line}\n    + {a_line}")
    if len(diffs) > 5:
        res.fail(f"… and {len(diffs) - 5} more {LOSS_BASENAME} line diffs")
    if not diffs and len(bl) == len(al):
        res.checked.append(f"{LOSS_BASENAME} ({len(bl)} lines)")


def compare_gate(
    suite: str, gate: str, base_dir: Path, after_dir: Path, capture_basename: str
) -> GateResult:
    res = GateResult(suite, gate)

    base_dump = base_dir / capture_basename
    after_dump = after_dir / capture_basename
    if not base_dump.is_file():
        res.fail(f"baseline dump missing: {base_dump}")
    if not after_dump.is_file():
        res.fail(f"after dump missing: {after_dump}")
    if base_dump.is_file() and after_dump.is_file():
        try:
            _compare_hash_dicts(res, base_dump, after_dump)
        except (ValueError, json.JSONDecodeError) as exc:
            res.fail(f"hash dump parse error: {exc}")

    base_graph = base_dir / f"{capture_basename}.graph.json"
    after_graph = after_dir / f"{capture_basename}.graph.json"
    if base_graph.is_file() and after_graph.is_file():
        _compare_bytes(res, ".graph.json", base_graph, after_graph)
    elif base_graph.is_file() != after_graph.is_file():
        res.fail(
            f".graph.json present on only one side "
            f"(baseline={base_graph.is_file()} after={after_graph.is_file()})"
        )

    base_loss = base_dir / LOSS_BASENAME
    after_loss = after_dir / LOSS_BASENAME
    if base_loss.is_file() and after_loss.is_file():
        _compare_loss(res, base_loss, after_loss)
    elif base_loss.is_file() != after_loss.is_file():
        res.fail(
            f"{LOSS_BASENAME} present on only one side "
            f"(baseline={base_loss.is_file()} after={after_loss.is_file()})"
        )

    return res


def discover_gates(base: Path, after: Path, suites: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for suite in suites:
        bsuite = base / suite
        if not bsuite.is_dir():
            continue
        for gate_dir in sorted(p for p in bsuite.iterdir() if p.is_dir()):
            pairs.append((suite, gate_dir.name))
    # Surface after-only suites/gates so a stray extra capture is not silently ignored.
    for suite in suites:
        asuite = after / suite
        if not asuite.is_dir():
            continue
        for gate_dir in sorted(p for p in asuite.iterdir() if p.is_dir()):
            if (suite, gate_dir.name) not in pairs:
                pairs.append((suite, gate_dir.name))
    return pairs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="dir of pre-refactor captures: <baseline>/<suite>/<gate>/",
    )
    ap.add_argument(
        "--after",
        type=Path,
        required=True,
        help="dir of post-refactor captures: <after>/<suite>/<gate>/",
    )
    ap.add_argument(
        "--suites",
        nargs="*",
        default=list(DEFAULT_SUITES),
        help=f"suites to compare (default: {' '.join(DEFAULT_SUITES)})",
    )
    ap.add_argument(
        "--gates",
        nargs="*",
        default=None,
        help="restrict to these gate names (default: all discovered)",
    )
    ap.add_argument(
        "--capture-basename",
        default=DEFAULT_CAPTURE_BASENAME,
        help=f"capture dump filename (default: {DEFAULT_CAPTURE_BASENAME})",
    )
    args = ap.parse_args(argv)

    if not args.baseline.is_dir():
        print(f"ERROR: --baseline not a directory: {args.baseline}", file=sys.stderr)
        return 2
    if not args.after.is_dir():
        print(f"ERROR: --after not a directory: {args.after}", file=sys.stderr)
        return 2

    pairs = discover_gates(args.baseline, args.after, args.suites)
    if args.gates:
        wanted = set(args.gates)
        pairs = [(s, g) for (s, g) in pairs if g in wanted]
    if not pairs:
        print("ERROR: no <suite>/<gate>/ dirs found to compare", file=sys.stderr)
        return 2

    results: list[GateResult] = []
    for suite, gate in pairs:
        results.append(
            compare_gate(
                suite,
                gate,
                args.baseline / suite / gate,
                args.after / suite / gate,
                args.capture_basename,
            )
        )

    n_pass = sum(1 for r in results if r.ok)
    n_fail = len(results) - n_pass
    for r in results:
        tag = "PASS" if r.ok else "FAIL"
        detail = ", ".join(r.checked) if r.ok else f"{len(r.errors)} error(s)"
        print(f"[{tag}] {r.suite}/{r.gate}  ({detail})")
        for err in r.errors:
            for line in err.splitlines():
                print(f"        {line}")

    print(f"\n{'=' * 60}")
    print(f"bitwise verify: {n_pass} PASS / {n_fail} FAIL  ({len(results)} gates)")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
