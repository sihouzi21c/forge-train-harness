"""Review-side deterministic checker for the long-horizon throughput bar.

Consumed by the review agent (see ``prompt/review_prompt/
review_stage1.md``) and — for the elastic relaxation of a dev-visible
``mfu_e2e_target`` — by ``evals.dispatcher._run_long_train``.

Policy source (nothing hardcoded here): the ``[shared.gate]`` table of
the long-train gate SOURCE toml,
``$FORGE_CONFIG_DIR/gate_config/long-train.toml`` (the frozen per-loop
config dir). These keys are stripped by ``tools/render_gate_configs.py``
(``REVIEW_ONLY_KEYS``) so they never reach the dev-readable rendered
products — the long-horizon blinding depends on that strip:

    review_mfu_target     — review-side bar T (required for the CLI)
    mfu_elastic_tolerance — band width τ below the bar     (default 2.0)
    mfu_elastic_rounds    — window N of qualifying runs    (default 3)
    mfu_elastic_eps       — max spread across the window   (default 0.5)

Decision, against a bar ``T``:

* **Strict pass** — newest avg MFU >= T.
* **Elastic pass** — the last N qualifying runs (including the newest)
  are all >= T - τ AND their spread is <= eps pp (a real plateau, not
  a lucky sample or a still-climbing curve).
* Otherwise **FAIL**.

Evidence source: the per-loop ``mfu_history.jsonl`` telemetry written
by :mod:`tools.mfu_record` (only full ``long-train`` runs whose loss
verdict did not fail qualify).

CLI output is deliberately number-free so the review agent cannot
mechanically paste a threshold, a measurement, or a shortfall into
dev-readable notes:

    MFU_GATE_VERDICT: PASS | FAIL
    MFU_GATE_REASON: STRICT_PASS | ELASTIC_PASS | BELOW_BAND |
                     IN_BAND_UNSTABLE | IN_BAND_INSUFFICIENT_HISTORY |
                     NO_QUALIFYING_RUNS | NO_HISTORY
    MFU_GATE_SAMPLES: <count of qualifying history rows>

Exit code: 0 = PASS, 1 = FAIL, 2 = usage / IO / policy error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from pathlib import Path

__all__ = [
    "ElasticPolicy",
    "evaluate",
    "load_policy",
    "load_samples",
    "main",
]

# Defaults for the elastic knobs when the gate source omits them; the
# bar itself (review_mfu_target / mfu_e2e_target) is never defaulted.
_DEFAULT_TOLERANCE = 2.0
_DEFAULT_ROUNDS = 3
_DEFAULT_EPS = 0.5
_DEFAULT_FULL_EVERY_SMOKES = 3

# Only full long-train gate runs are throughput evidence — never the
# smoke variant and never loss-gate-200 (which shares the milestone tag).
_SUITE_KEY = "long-train"
# The smoke suite feeds ONLY the --screen decision ("is a full run worth
# its GPU cost now?"), never the verdict itself.
_SMOKE_SUITE_KEY = "long-train-smoke"

_GATE_SOURCE_RELPATH = Path("gate_config") / "long-train.toml"


class ElasticPolicy:
    """Numeric policy loaded from the long-train gate source toml."""

    def __init__(
        self,
        review_target: float | None,
        tolerance: float,
        rounds: int,
        eps: float,
        full_every_smokes: int = _DEFAULT_FULL_EVERY_SMOKES,
    ) -> None:
        self.review_target = review_target
        self.tolerance = tolerance
        self.rounds = rounds
        self.eps = eps
        self.full_every_smokes = full_every_smokes


def _gate_source_path(explicit: Path | None = None) -> Path | None:
    """Resolve the long-train gate SOURCE toml (not the rendered product).

    Order: explicit path → $FORGE_CONFIG_DIR/gate_config/long-train.toml
    → ./config/gate_config/long-train.toml (bare checkouts / tests).
    """
    if explicit is not None:
        return explicit
    cfg_dir = os.environ.get("FORGE_CONFIG_DIR")
    if cfg_dir:
        return Path(cfg_dir) / _GATE_SOURCE_RELPATH
    fallback = Path.cwd() / "config" / _GATE_SOURCE_RELPATH
    if fallback.is_file():
        return fallback
    return None


def load_policy(gate_source: Path | None = None) -> ElasticPolicy | None:
    """Load the policy from the gate source's ``[shared.gate]`` table.

    Returns ``None`` when the source file cannot be found/parsed.
    ``review_target`` is ``None`` when the key is absent — callers that
    need the review bar must treat that as a configuration error.
    """
    path = _gate_source_path(gate_source)
    if path is None or not path.is_file():
        return None
    try:
        with path.open("rb") as fh:
            src = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    gate = (src.get("shared") or {}).get("gate") or {}

    def _num(key: str, default: float | None) -> float | None:
        val = gate.get(key, default)
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            return default
        return float(val)

    review_target = _num("review_mfu_target", None)
    # Non-None defaults make these non-None; the assert narrows for mypy.
    tolerance = _num("mfu_elastic_tolerance", _DEFAULT_TOLERANCE)
    assert tolerance is not None
    rounds_val = gate.get("mfu_elastic_rounds", _DEFAULT_ROUNDS)
    rounds = (
        rounds_val
        if isinstance(rounds_val, int) and not isinstance(rounds_val, bool)
        else _DEFAULT_ROUNDS
    )
    eps = _num("mfu_elastic_eps", _DEFAULT_EPS)
    assert eps is not None
    every_val = gate.get("mfu_full_every_smokes", _DEFAULT_FULL_EVERY_SMOKES)
    full_every = (
        every_val
        if isinstance(every_val, int) and not isinstance(every_val, bool)
        else _DEFAULT_FULL_EVERY_SMOKES
    )
    return ElasticPolicy(
        review_target,
        float(tolerance),
        max(1, rounds),
        float(eps),
        max(1, full_every),
    )


def _load_history(path: Path) -> list[dict] | None:
    """Parse the append-only jsonl, skipping malformed lines. Returns
    ``None`` when the file does not exist (distinct from empty)."""
    if not path.is_file():
        return None
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _qualifying_pairs(rows: list[dict], suite: str) -> list[tuple[float, float]]:
    """(ts, avg_mfu) for one suite's usable measurements, ts-ascending.
    Loss-failed runs are not throughput evidence."""
    picked: list[tuple[float, float]] = []
    for row in rows:
        if row.get("suite") != suite:
            continue
        avg = row.get("avg_mfu")
        if not isinstance(avg, (int, float)) or isinstance(avg, bool):
            continue
        if row.get("precision_pass") is False:
            continue
        ts = row.get("ts")
        ts_f = float(ts) if isinstance(ts, (int, float)) else 0.0
        picked.append((ts_f, float(avg)))
    picked.sort(key=lambda p: p[0])
    return picked


def _qualifying(rows: list[dict]) -> list[float]:
    return [avg for _, avg in _qualifying_pairs(rows, _SUITE_KEY)]


def load_samples(history_path: Path) -> tuple[list[float], bool]:
    """Qualifying ts-ascending samples + whether the file existed."""
    rows = _load_history(history_path)
    return _qualifying(rows or []), rows is not None


def default_history_path(repo_root: Path) -> Path | None:
    """Per-loop mfu_history.jsonl for a workspace root (same layout
    fallback as tools/mfu_record)."""
    env = os.environ
    train_dir = env.get("FORGE_TRAIN_DIR")
    loop_id = env.get("LOOP_ID") or env.get("FORGE_TRAIN_LOOP_ID")
    if train_dir and loop_id:
        return Path(train_dir) / loop_id / "mfu_history.jsonl"
    if repo_root.name == "workspace":
        return repo_root.parent / "mfu_history.jsonl"
    return None


def evaluate(
    samples: list[float],
    bar: float,
    policy: ElasticPolicy,
    *,
    had_history_file: bool = True,
) -> tuple[bool, str]:
    """Pure decision. ``samples`` is ts-ascending qualifying avg MFU;
    the newest sample is the run under judgment."""
    if not samples:
        return False, "NO_QUALIFYING_RUNS" if had_history_file else "NO_HISTORY"
    newest = samples[-1]
    band = bar - policy.tolerance
    if newest >= bar:
        return True, "STRICT_PASS"
    if newest < band:
        return False, "BELOW_BAND"
    if len(samples) < policy.rounds:
        return False, "IN_BAND_INSUFFICIENT_HISTORY"
    tail = samples[-policy.rounds :]
    if min(tail) < band:
        return False, "IN_BAND_INSUFFICIENT_HISTORY"
    if max(tail) - min(tail) > policy.eps:
        return False, "IN_BAND_UNSTABLE"
    return True, "ELASTIC_PASS"


def screen(rows: list[dict], policy: ElasticPolicy) -> bool:
    """Cadence trigger: launch a full long-train every
    ``full_every_smokes`` smoke runs.

    True when at least ``full_every_smokes`` qualifying smoke runs have
    accumulated since the newest qualifying full run (all smokes count
    when no full run exists yet). Unconditional cadence — no threshold
    against the bar — so the trigger timing carries zero information
    about the (blinded) review policy, and the elastic window fills
    steadily whenever the dev iterates on smokes.
    """
    smokes = _qualifying_pairs(rows, _SMOKE_SUITE_KEY)
    if not smokes:
        return False
    fulls = _qualifying_pairs(rows, _SUITE_KEY)
    newest_full_ts = fulls[-1][0] if fulls else float("-inf")
    smokes_since_full = sum(1 for ts, _avg in smokes if ts > newest_full_ts)
    return smokes_since_full >= policy.full_every_smokes


def _result_json_mfu(path: Path) -> float | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    metrics = data.get("metrics") if isinstance(data, dict) else None
    if not isinstance(metrics, dict):
        return None
    avg = metrics.get("avg_mfu_e2e_standard")
    if isinstance(avg, (int, float)) and not isinstance(avg, bool):
        return float(avg)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Review-side long-horizon throughput check "
            "(policy from the gate source toml, evidence from the "
            "per-loop mfu_history.jsonl telemetry)."
        )
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=None,
        help=(
            "Path to mfu_history.jsonl. Default: resolved from "
            "$FORGE_TRAIN_DIR/$LOOP_ID, else ../mfu_history.jsonl when "
            "cwd is a per-loop workspace."
        ),
    )
    parser.add_argument(
        "--gate-config",
        type=Path,
        default=None,
        help=(
            "Path to the long-train gate SOURCE toml. Default: "
            "$FORGE_CONFIG_DIR/gate_config/long-train.toml."
        ),
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=None,
        help=(
            "Optional harness result.json whose "
            "metrics.avg_mfu_e2e_standard is appended as the newest "
            "sample (covers a telemetry sync gap after a remote run)."
        ),
    )
    parser.add_argument(
        "--screen",
        action="store_true",
        help=(
            "Cadence mode: trigger a full long-train once "
            "mfu_full_every_smokes smoke runs have accumulated since "
            "the newest full run. Prints MFU_SCREEN: TRIGGER|NO_TRIGGER; "
            "exit 0 = trigger, 1 = no trigger, 2 = error."
        ),
    )
    args = parser.parse_args(argv)

    policy = load_policy(args.gate_config)
    if policy is None:
        print(
            "mfu_elastic_check: cannot load the long-train gate source "
            "(set FORGE_CONFIG_DIR or pass --gate-config)",
            file=sys.stderr,
        )
        return 2
    if not args.screen and policy.review_target is None:
        print(
            "mfu_elastic_check: gate source declares no review_mfu_target",
            file=sys.stderr,
        )
        return 2

    history_path = args.history
    if history_path is None:
        history_path = default_history_path(Path.cwd())
    if history_path is None:
        print(
            "mfu_elastic_check: cannot resolve mfu_history.jsonl (pass --history)",
            file=sys.stderr,
        )
        return 2

    if args.screen:
        rows = _load_history(history_path)
        trigger = screen(rows or [], policy)
        print(f"MFU_SCREEN: {'TRIGGER' if trigger else 'NO_TRIGGER'}")
        return 0 if trigger else 1

    samples, had_file = load_samples(history_path)

    if args.result_json is not None:
        extra = _result_json_mfu(args.result_json)
        if extra is None:
            print(
                f"mfu_elastic_check: unreadable --result-json {args.result_json}",
                file=sys.stderr,
            )
            return 2
        samples.append(extra)

    # Non-screen mode already returned 2 above when review_target is None.
    assert policy.review_target is not None
    passed, reason = evaluate(
        samples,
        policy.review_target,
        policy,
        had_history_file=had_file,
    )
    print(f"MFU_GATE_VERDICT: {'PASS' if passed else 'FAIL'}")
    print(f"MFU_GATE_REASON: {reason}")
    print(f"MFU_GATE_SAMPLES: {len(samples)}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
