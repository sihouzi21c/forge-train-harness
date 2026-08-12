#!/usr/bin/env python3
"""loop_report.py — offline trajectory / timing / MFU / error report for one ForgeTrain loop.

Reads only on-disk artifacts (no ssh, no GPU, safe on a dead or live loop):

  .artifacts/web-agents/loop-<id>/stdout.log      typed loop_event stream (SSOT for rounds/milestones)
  .artifacts/web-agents/<agent_id>/stdout.log     per-agent stream-json transcripts (--deep)
  .artifacts/forge_train/<id>/config/*.toml       resolved per-loop axis configs
  .artifacts/forge_train/<id>/mfu_history.jsonl   every gate run that produced an MFU sample
  .artifacts/forge_train/<id>/workspace/          git history (commit-per-round trajectory)

Usage:
  python3 harness/tools/loop_report.py <loop_id> [--root <repo_or_.artifacts>] \
      [--deep] [--top 3] [--json out.json] [--md out.md]

  --deep  additionally walks every dev/review agent transcript to attribute
          wall-clock (model-vs-tool share, longest tool calls) and to harvest
          agent-level errors (API 429/5xx, is_error results, CLI retries).
          Costs one linear pass per transcript; everything else is instant.

Sections emitted: loop header & liveness · stage-1 gate configuration
(per-gate shape GBS/MBS, hash-capture level & per-attempt hash records,
ref-cache policy, MFU bar) · stage/milestone table · per-round table
(status, duration, agents, review verdict, commits, errors) · detailed
alignment & bitwise trajectory (round-by-round commits, MFU samples,
gate-run evidence under --deep) · MFU trajectory with per-suite deltas ·
error digest (loop exit cause, review FAILs, agent API errors, wrapper
retry storms) · optional deep time buckets.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # < 3.11
    tomllib = None

TS = "%m-%d %H:%M"


def fmt(ts: float | None) -> str:
    return datetime.fromtimestamp(ts).strftime(TS) if ts else "?"


def fdur(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    seconds = int(seconds)
    h, m = divmod(seconds // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{seconds % 60:02d}s"


def read_jsonl(path: Path):
    if not path.exists():
        return
    with open(path, errors="replace") as fh:
        for line in fh:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ── loop_event stream ────────────────────────────────────────────────────────


def load_loop_events(log: Path) -> list[dict]:
    events = []
    for d in read_jsonl(log):
        if d.get("type") in ("loop_event", "system"):
            events.append(d)
    return events


def build_rounds(events: list[dict], end_ts: float) -> list[dict]:
    """One record per round: window, milestone, agents, verdict, errors."""
    rounds: list[dict] = []
    cur: dict | None = None
    for d in events:
        st, ts = d.get("subtype"), d.get("ts")
        if st == "round_start":
            if cur is not None and cur.get("end") is None:
                cur["end"] = ts
            cur = {
                "round": d.get("round"),
                "milestone": d.get("milestone"),
                "stage": d.get("stage", "stage1"),
                "start": ts,
                "end": None,
                "dev_agents": [],
                "review_agents": [],
                "review": None,
                "advanced_to": None,
                "errors": [],
                "infos": [],
            }
            rounds.append(cur)
        elif cur is None:
            continue
        elif st == "spawn_child":
            kind = d.get("kind", "")
            key = "review_agents" if "review" in kind else "dev_agents"
            if d.get("agent_id") not in cur[key]:  # CLI retries respawn the same id
                cur[key].append(d.get("agent_id"))
        elif st == "review_verdict":
            cur["review"] = d.get("verdict")
        elif st == "milestone_advanced":
            cur["advanced_to"] = d.get("to")
        elif st == "info":
            text = str(d.get("text", ""))
            cur["infos"].append(text)
            low = text.lower()
            if any(k in low for k in ("error", "exited with", "failed", "fatal", "timeout")):
                cur["errors"].append(text[:300])
        elif st == "retry_decision":
            cur.setdefault("retries", 0)
            cur["retries"] = cur.get("retries", 0) + 1
        elif st == "loop_exit":
            cur["end"] = cur.get("end") or ts
    if rounds and rounds[-1]["end"] is None:
        rounds[-1]["end"] = end_ts
    for r in rounds:
        r["dur"] = (r["end"] - r["start"]) if r.get("end") and r.get("start") else None
    return rounds


def build_milestones(rounds: list[dict], events: list[dict], end_ts: float) -> list[dict]:
    """Ordered milestone windows with round counts, review FAILs, status."""
    adv = {}  # from-milestone -> ts of advancement
    for d in events:
        if d.get("subtype") == "milestone_advanced":
            adv[d["from"]] = d["ts"]
    exited = loop_exit_info(events) is not None
    out: list[dict] = []
    for r in rounds:
        ms = r["milestone"]
        if not out or out[-1]["milestone"] != ms:
            out.append(
                {
                    "milestone": ms,
                    "stage": r["stage"],
                    "start": r["start"],
                    "rounds": 0,
                    "review_fails": 0,
                    "errors": 0,
                }
            )
        m = out[-1]
        m["rounds"] += 1
        m["review_fails"] += 1 if r["review"] == "FAIL" else 0
        m["errors"] += len(r["errors"])
        m["last_round_end"] = r["end"]
    for m in out:
        if m["milestone"] in adv:
            m["end"], m["status"] = adv[m["milestone"]], "PASSED"
        elif exited:
            m["end"], m["status"] = m["last_round_end"], "KILLED (loop exit)"
        else:
            m["end"], m["status"] = end_ts, "RUNNING"
        m["dur"] = m["end"] - m["start"]
    return out


def loop_exit_info(events: list[dict]) -> dict | None:
    """Exit record + the closest preceding cause line, if the loop ended.

    Only the LAST loop_exit counts, and only when no timestamped event
    follows it: a wrapper relaunch appends a fresh init + event stream to
    the same log, which voids the earlier exit (seen on aborted launches).
    """
    exit_ev = None
    for d in events:
        if d.get("subtype") == "loop_exit":
            exit_ev = d
        elif exit_ev is not None and d.get("ts"):
            exit_ev = None  # relaunched after that exit — not terminal
    if exit_ev is None:
        return None
    cause = ""
    for d in events:
        if d.get("subtype") == "info" and "exited with" in str(d.get("text", "")):
            cause = str(d["text"])[:400]  # keep the last one before exit
    return {"ts": exit_ev.get("ts"), "cause": cause}


# ── configs / MFU / git ─────────────────────────────────────────────────────

# Stage-1 gate suites in milestone order (suite name -> milestone). The
# detailed-trajectory section and the gate-config table both key off this.
STAGE1_GATES = [
    ("forward-align", "alignment.forward"),
    ("backward-align", "alignment.backward"),
    ("multistep-1gpu", "bitwise-singlecard"),
    ("multistep", "bitwise-multicard"),
    ("perf-bitwise", "bitwise-perf"),
]

# What each hash_capture_level actually hashes (evals/harness_hook docstring).
HASH_LEVEL_DESC = {
    0: "off",
    1: "L1: loss + grads (pre/post allreduce)",
    2: "L2: loss + grads + per-module fwd/bwd",
}


def load_gate_configs(cfg_dir: Path) -> list[dict]:
    """One row per stage-1 gate from the launch-frozen gate_config/<suite>.toml.

    Derives grad_accum = GBS / (MBS x WS) and the per-attempt microbatch
    hash-record count (steps x grad_accum x WS, per side). Ref caching is a
    code-level policy, not a TOML knob: capture gates (hash_capture_level>0)
    always run the ref live (evals/_common.py use_cache rule).
    """
    rows: list[dict] = []
    if tomllib is None:
        return rows
    for suite, milestone in STAGE1_GATES:
        p = cfg_dir / "gate_config" / f"{suite}.toml"
        if not p.exists():
            continue
        try:
            with open(p, "rb") as fh:
                d = tomllib.load(fh)
        except Exception:  # nosec B112 — skip unparsable gate_config; report stays best-effort
            continue
        shared = d.get("shared", {})
        gate = shared.get("gate", {})
        ws = shared.get("world_size_override")
        mbs = shared.get("micro_batch_size_override")
        gbs = shared.get("global_batch_size_override")
        steps = shared.get("num_steps_override")
        accum = (
            (gbs // (mbs * ws))
            if all(isinstance(x, int) and x > 0 for x in (gbs, mbs, ws))
            else None
        )
        level = gate.get("hash_capture_level", 0)
        rows.append(
            {
                "suite": suite,
                "milestone": milestone,
                "world_size": ws,
                "mbs": mbs,
                "gbs": gbs,
                "steps": steps,
                "grad_accum": accum,
                "gate_window": shared.get("gate_window"),
                "hash_level": level,
                # microbatches hashed per attempt, per side (ref and ours each)
                "hash_records": (steps * accum * ws) if accum and isinstance(steps, int) else None,
                "ref_cache_live": level > 0,  # capture gates: ref runs live every attempt
                "mfu_bar": gate.get("mfu_e2e_target") or None,
                "warmup_steps": gate.get("warmup_steps"),
            }
        )
    return rows


def attach_mfu_samples(rounds: list[dict], mfu: list[dict]) -> None:
    """Attach each MFU sample to the round whose window contains it."""
    for r in rounds:
        if r.get("start") is None:
            continue
        end = r.get("end") or float("inf")
        r["mfu_samples"] = [d for d in mfu if r["start"] <= (d.get("ts") or 0) < end + 120]


def load_config(cfg_dir: Path) -> dict:
    out = {}
    if tomllib is None:
        return out

    def grab(axis, table, keys):
        p = cfg_dir / f"{axis}.toml"
        if not p.exists():
            return
        try:
            with open(p, "rb") as fh:
                d = tomllib.load(fh)
        except Exception:
            return
        sect = d.get(table, d)
        for k in keys:
            if k in sect:
                out[f"{axis}.{k}"] = sect[k]

    grab("agent", "agent", ("backend", "model", "effort"))
    grab("model", "model", ("name",))
    grab("remote", "remote", ("kind", "hostname", "workspace", "gpu_count", "gpu_model"))
    return out


def load_mfu(path: Path) -> list[dict]:
    rows = [d for d in read_jsonl(path)]
    last_by_suite: dict[str, float] = {}
    for d in rows:
        suite = d.get("suite", "?")
        prev = last_by_suite.get(suite)
        d["delta"] = (d["avg_mfu"] - prev) if prev is not None else None
        last_by_suite[suite] = d["avg_mfu"]
    return rows


def git_commits(workspace: Path) -> list[dict]:
    if not (workspace / ".git").exists():
        return []
    try:
        raw = subprocess.run(
            ["git", "-C", str(workspace), "log", "--reverse", "--format=%at%x09%h%x09%s"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except Exception:
        return []
    out = []
    for line in raw.splitlines():
        try:
            ts, sha, subj = line.split("\t", 2)
            out.append({"ts": float(ts), "sha": sha, "subject": subj})
        except ValueError:
            continue
    return out


def attach_commits(rounds: list[dict], commits: list[dict]) -> None:
    for r in rounds:
        if r.get("start") is None:
            continue
        end = r.get("end") or float("inf")
        r["commits"] = [c for c in commits if r["start"] <= c["ts"] < end + 120]


# ── deep transcript scan ─────────────────────────────────────────────────────

GATE_SUITE_NAMES = [s for s, _ in STAGE1_GATES]
# ``bin/harness run multistep`` — also matches inside ssh-wrapped one-liners.
_RUN_RE = re.compile(r"\brun\s+(" + "|".join(re.escape(s) for s in GATE_SUITE_NAMES) + r")\b")
_VERDICT_RE = re.compile(r"\b(PASS(?:ED)?|FAIL(?:ED)?|TIMED?\s?_?OUT)\b", re.IGNORECASE)


def _norm_verdict(tok: str) -> str:
    tok = tok.upper()
    if tok.startswith("PASS"):
        return "PASS"
    if tok.startswith("FAIL"):
        return "FAIL"
    return "TIMEOUT"


def _harvest_gate_lines(text: str, ts: float, out: list[dict]) -> None:
    """Scrape gate-verdict evidence from one tool_result payload.

    A line counts only if it names a stage-1 gate suite AND carries a
    PASS/FAIL/TIMEOUT token; grep-pattern lines (``PASS|FAIL``) are skipped.
    """
    for line in text.splitlines():
        if "PASS|FAIL" in line or len(line) > 500:
            continue
        for suite in GATE_SUITE_NAMES:
            if suite in line:
                m = _VERDICT_RE.search(line)
                if m:
                    out.append(
                        {
                            "ts": ts,
                            "suite": suite,
                            "kind": "verdict",
                            "verdict": _norm_verdict(m.group(1)),
                        }
                    )
                break  # longest-first not needed: one suite per line is enough


def scan_transcript(path: Path, top: int) -> dict:
    """One linear pass: wall, tool-time, longest calls, gate evidence, errors."""
    pending: dict[str, tuple[float, str]] = {}
    calls: list[tuple[float, str, float]] = []
    errors: list[str] = []
    gate_events: list[dict] = []
    first = last = None
    for d in read_jsonl(path):
        ts = d.get("_ts")
        if ts is None:
            continue
        first = first if first is not None else ts
        last = ts
        t = d.get("type")
        if t == "assistant":
            for c in d.get("message", {}).get("content", []):
                if isinstance(c, dict) and c.get("type") == "tool_use":
                    cmd = c.get("name", "?")
                    if cmd == "Bash":
                        raw = str(c.get("input", {}).get("command", ""))
                        for m in _RUN_RE.finditer(raw):
                            gate_events.append({"ts": ts, "suite": m.group(1), "kind": "run"})
                        cmd += " " + raw[:100].replace("\n", " ")
                    pending[c["id"]] = (ts, cmd)
        elif t == "user":
            content = d.get("message", {}).get("content")
            if isinstance(content, list):
                for c in content:
                    if not isinstance(c, dict) or c.get("type") != "tool_result":
                        continue
                    if c.get("tool_use_id") in pending:
                        s, cmd = pending.pop(c["tool_use_id"])
                        calls.append((ts - s, cmd, s))
                    body = c.get("content")
                    if isinstance(body, list):
                        body = "\n".join(x.get("text", "") for x in body if isinstance(x, dict))
                    if isinstance(body, str) and body:
                        _harvest_gate_lines(body, ts, gate_events)
        elif t == "result" and (d.get("is_error") or d.get("api_error_status")):
            status = d.get("api_error_status", "")
            errors.append(f"result is_error api_status={status}: {str(d.get('result', ''))[:200]}")
    calls.sort(key=lambda x: -x[0])
    # collapse consecutive duplicate evidence (monitor loops re-read the same log)
    deduped: list[dict] = []
    for e in gate_events:
        prev = deduped[-1] if deduped else None
        if (
            prev
            and prev["suite"] == e["suite"]
            and prev["kind"] == e["kind"]
            and prev.get("verdict") == e.get("verdict")
        ):
            continue
        deduped.append(e)
    return {
        "wall_s": (last - first) if first is not None else 0,
        "n_calls": len(calls),
        "tool_s": sum(c[0] for c in calls),
        "longest": [
            {"dur_s": round(dur, 1), "at": fmt(s), "cmd": cmd} for dur, cmd, s in calls[:top]
        ],
        "unresolved_calls": len(pending),
        "errors": errors,
        "gate_events": deduped,
    }


# ── rendering ────────────────────────────────────────────────────────────────


def dedupe(msgs: list[str]) -> list[str]:
    """Collapse identical repeated messages into 'msg (×N)'."""
    out: list[str] = []
    counts: dict[str, int] = {}
    for m in msgs:
        if m not in counts:
            out.append(m)
        counts[m] = counts.get(m, 0) + 1
    return [m + (f"  (x{counts[m]})" if counts[m] > 1 else "") for m in out]


def md_escape(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def _is_align_or_bitwise(milestone: str | None) -> bool:
    ms = str(milestone or "")
    return ms.startswith("alignment") or ms.startswith("bitwise")


def render_detail_trajectory(report: dict, p) -> None:
    """Round-by-round narrative for the alignment and bitwise milestones:
    full commit subjects, MFU samples vs the bar, review verdicts, and (with
    --deep) gate-run evidence scraped from the dev-agent transcripts."""
    rounds = [r for r in report["rounds"] if _is_align_or_bitwise(r.get("milestone"))]
    if not rounds:
        return
    p("## Detailed trajectory — alignment & bitwise milestones")
    p()
    if not any("deep" in r for r in report["rounds"]):
        p("*(run with `--deep` to add per-attempt gate-run evidence from the dev transcripts)*")
        p()
    ms_status = {m["milestone"]: m for m in report["milestones"]}
    last_ms = None
    for r in rounds:
        ms = r["milestone"]
        if ms != last_ms:
            if last_ms is not None:
                p()
            m = ms_status.get(ms, {})
            p(
                f"### {ms} — {m.get('rounds', '?')} round(s) · {fdur(m.get('dur'))} · {m.get('status', '?')}"
            )
            p()
            last_ms = ms
        verdict = r["review"] or "—"
        head = f"- **R{r['round']}** {fmt(r['start'])} · {fdur(r['dur'])} · review {verdict}"
        if r.get("retries"):
            head += f" · ⚠ {r['retries']} CLI retries"
        if r.get("advanced_to"):
            head += f" · **milestone → {r['advanced_to']}**"
        p(head)
        for c in r.get("commits", []):
            p(f"  - commit `{c['sha']}` {md_escape(c['subject'][:160])}")
        for s in r.get("mfu_samples", []):
            gate = "PASS" if s.get("mfu_pass") else "fail"
            p(
                f"  - MFU sample `{s.get('suite', '?')}`: {s['avg_mfu']:.2f}% vs bar {s.get('mfu_target', '?')}% → {gate}"
            )
        evs = (r.get("deep") or {}).get("gate_events", [])
        if evs:
            bits = []
            for e in evs[:12]:
                if e["kind"] == "run":
                    bits.append(f"{fmt(e['ts'])} `{e['suite']}` launched")
                else:
                    bits.append(f"{fmt(e['ts'])} `{e['suite']}` → {e['verdict']}")
            more = f" … (+{len(evs) - 12} more)" if len(evs) > 12 else ""
            p(f"  - gate evidence: {' · '.join(bits)}{more}")
        for e in dedupe(r.get("errors", []))[:2]:
            p(f"  - ⚠ {md_escape(e[:160])}")
        if not r.get("commits") and not r.get("mfu_samples") and not evs:
            p("  - (no commit / gate evidence — measurement or monitor round)")
    p()


def render(report: dict, out) -> None:
    def p(*a):
        print(*a, file=out)

    hdr = report["header"]
    cfg = hdr.get("config", {})

    p(f"# ForgeTrain Loop Report — `{hdr['loop_id']}`")
    p()
    status = "RUNNING" if hdr["alive"] else "EXITED"
    p(
        f"Generated {hdr['generated']}. Loop **{status}** — launched {hdr['launched']}, "
        f"total wall **{fdur(hdr['wall_s'])}**"
        + (f", last event {hdr['last_event']}." if hdr["alive"] else ".")
    )
    p()
    agent = " / ".join(
        str(cfg[k]) for k in ("agent.backend", "agent.model", "agent.effort") if cfg.get(k)
    )
    remote = " · ".join(
        str(cfg[k]) for k in ("remote.kind", "remote.hostname", "remote.workspace") if cfg.get(k)
    )
    gpus = f"{cfg.get('remote.gpu_count', '?')}x{cfg.get('remote.gpu_model', '?')}"
    p("| | |")
    p("|---|---|")
    if agent:
        p(f"| Agent | {md_escape(agent)} |")
    if cfg.get("model.name"):
        p(f"| Trained model | `{cfg['model.name']}` |")
    if remote:
        p(f"| Remote | {md_escape(remote)} ({gpus}) |")
    p()

    if report.get("gate_configs"):
        p("## Stage-1 gate configuration (launch-frozen)")
        p()
        p("Shapes from `config/gate_config/<suite>.toml`; hash records = microbatches")
        p("hashed per attempt **per side** (steps x grad-accum x world-size). Capture")
        p("gates (hash level > 0) never reuse a cached ref — the ref runs live on")
        p("every attempt by policy.")
        p()
        p(
            "| Milestone | Suite | WS | MBS | GBS | grad-accum | Steps | Gate window | Hash capture | Hash records/attempt | Ref caching | MFU bar |"
        )
        p("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for g in report["gate_configs"]:
            win = f"[{g['gate_window'][0]}, {g['gate_window'][1]}]" if g.get("gate_window") else "?"
            lvl = HASH_LEVEL_DESC.get(g["hash_level"], f"L{g['hash_level']}")
            recs = f"{g['hash_records']} µbatch/side" if g.get("hash_records") else "—"
            cache = (
                "OFF — ref live every attempt"
                if g["ref_cache_live"]
                else "eligible (`FORGE_REF_CACHE`)"
            )
            bar = "—"
            if g.get("mfu_bar"):
                bar = f"**≥{g['mfu_bar']:g}%**" + (
                    f" (warmup {g['warmup_steps']})" if g.get("warmup_steps") else ""
                )
            p(
                f"| {g['milestone']} | `{g['suite']}` | {g['world_size']} | {g['mbs']} | {g['gbs']} "
                f"| {g['grad_accum']} | {g['steps']} | {win} | {lvl} | {recs} | {cache} | {bar} |"
            )
        p()

    if report.get("exit"):
        p("## Terminal event")
        p()
        p(f"The loop **exited at {fmt(report['exit']['ts'])}**. Wrapper's last cause line:")
        p()
        p("```")
        p(report["exit"]["cause"] or "(no cause line captured)")
        p("```")
        p()

    p("## Milestone timeline")
    p()
    total = sum(m["dur"] for m in report["milestones"]) or 1
    p(
        "| Milestone | Start | End | Duration | Rounds | Review FAILs | Errors | % of wall | Status |"
    )
    p("|---|---|---|---|---|---|---|---|---|")
    for m in report["milestones"]:
        bold = "**" if m["dur"] == max(x["dur"] for x in report["milestones"]) else ""
        p(
            f"| {m['milestone']} | {fmt(m['start'])} | {fmt(m['end'])} | {bold}{fdur(m['dur'])}{bold} "
            f"| {m['rounds']} | {m['review_fails'] or ''} | {m['errors'] or ''} "
            f"| {100 * m['dur'] / total:.1f}% | {m['status']} |"
        )
    p()

    p("## Per-round trajectory")
    p()
    has_deep = any("deep" in r for r in report["rounds"])
    cols = (
        "| R | Milestone | Start | Duration | Review |"
        + (" Tool-time |" if has_deep else "")
        + " Work |"
    )
    p(cols)
    p("|---|---|---|---|---|" + ("---|" if has_deep else "") + "---|")
    for r in report["rounds"]:
        review = r["review"] or (
            "running" if report["header"]["alive"] and r is report["rounds"][-1] else "—"
        )
        if r["review"] == "FAIL":
            review = "**FAIL**"
        work = []
        for c in r.get("commits", []):
            work.append(md_escape(c["subject"][:120]))
        if r.get("advanced_to"):
            work.append(f"**→ {r['advanced_to']}**")
        if r.get("retries"):
            work.append(f"⚠ {r['retries']} CLI retries")
        if r.get("errors"):
            work.append(f"⚠ {md_escape(r['errors'][0][:110])}")
        if not work:
            work.append("(no commit — measurement/monitor round)")
        tool_cell = ""
        if has_deep:
            d = r.get("deep")
            tool_cell = (
                f" {fdur(d['tool_s'])} ({100 * d['tool_s'] / d['wall_s']:.0f}%) |"
                if d and d["wall_s"]
                else "  |"
            )
        p(
            f"| {r['round']} | {r['milestone']} | {fmt(r['start'])} | {fdur(r['dur'])} "
            f"| {review} |{tool_cell} {'<br>'.join(work)} |"
        )
    p()

    render_detail_trajectory(report, p)

    if has_deep:
        p("### Deep timing detail (longest tool calls per round)")
        p()
        for r in report["rounds"]:
            d = r.get("deep")
            if not d:
                continue
            share = 100 * d["tool_s"] / d["wall_s"] if d["wall_s"] else 0
            p(
                f"- **R{r['round']}** ({r['milestone']}): wall {fdur(d['wall_s'])}, "
                f"{d['n_calls']} tool calls, tool-time {fdur(d['tool_s'])} ({share:.0f}%)"
                + (f", unresolved={d['unresolved_calls']}" if d["unresolved_calls"] else "")
            )
            for c in d["longest"]:
                p(f"  - {c['dur_s'] / 60:.1f} min @ {c['at']} — `{md_escape(c['cmd'][:110])}`")
            for e in dedupe(d["errors"]):
                p(f"  - ⚠ {md_escape(e[:180])}")
        p()

    p("## MFU trajectory")
    p()
    if not report["mfu"]:
        p("(no `mfu_history.jsonl` samples yet)")
    else:
        p("| Time | Suite | MFU | Δ vs prev | Target | Gate | Milestone |")
        p("|---|---|---|---|---|---|---|")
        for d in report["mfu"]:
            delta = f"{d['delta']:+.2f}" if d.get("delta") is not None else "—"
            gate = {True: "**PASS**", False: "fail", None: "—"}.get(d.get("mfu_pass"), "—")
            p(
                f"| {fmt(d['ts'])} | {d.get('suite', '?')} | {d['avg_mfu']:.2f}% | {delta} "
                f"| {d.get('mfu_target', '?')} | {gate} | {d.get('milestone', '')} |"
            )
        p()
        by_suite: dict[str, list] = {}
        for d in report["mfu"]:
            by_suite.setdefault(d.get("suite", "?"), []).append(d["avg_mfu"])
        for suite, vals in by_suite.items():
            arrow = (
                " → ".join(f"{v:.2f}" for v in vals[:1] + vals[-1:])
                if len(vals) > 1
                else f"{vals[0]:.2f}"
            )
            p(
                f"- `{suite}`: {len(vals)} runs, {arrow}%"
                + (f" (peak {max(vals):.2f}%)" if len(vals) > 2 else "")
            )
    p()

    p("## Error digest")
    p()
    lines = []
    if report.get("exit"):
        lines.append(
            f"**loop_exit @ {fmt(report['exit']['ts'])}** — {md_escape(report['exit']['cause'][:220])}"
        )
    for r in report["rounds"]:
        if r["review"] == "FAIL":
            lines.append(f"R{r['round']} ({r['milestone']}): **review verdict FAIL**")
        for e in dedupe(r.get("errors", [])):
            lines.append(f"R{r['round']} ({r['milestone']}): {md_escape(e[:200])}")
        for e in dedupe((r.get("deep", {}) or {}).get("errors", [])):
            lines.append(f"R{r['round']} agent: {md_escape(e[:200])}")
    if not lines:
        p("None recorded.")
    for ln in lines:
        p(f"- {ln}")


# ── main ─────────────────────────────────────────────────────────────────────


def build_report(loop_id: str, root: Path, deep: bool, top: int) -> dict:
    agents_dir = root / "web-agents"
    loop_dir = root / "forge_train" / loop_id
    # SSOT: the wrapper Session id convention lives in web.paths. Walk
    # upward so both the repo checkout (harness/tools/) and a per-loop
    # workspace copy (workspace/tools/) resolve the same repo root.
    repo_root = next(
        p for p in Path(__file__).resolve().parents if (p / "web" / "paths.py").is_file()
    )
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from web.paths import wrapper_agent_id

    log = agents_dir / wrapper_agent_id(loop_id) / "stdout.log"
    if not log.exists():
        sys.exit(f"error: no loop event log at {log}")

    events = load_loop_events(log)
    now = time.time()
    exit_rec = loop_exit_info(events)
    last_ts = max((d.get("ts") or 0) for d in events if d.get("ts")) if events else now
    end_ts = exit_rec["ts"] if exit_rec else now
    first_ts = next((d.get("ts") for d in events if d.get("ts")), None)

    rounds = build_rounds(events, end_ts)
    milestones = build_milestones(rounds, events, end_ts)
    attach_commits(rounds, git_commits(loop_dir / "workspace"))
    mfu = load_mfu(loop_dir / "mfu_history.jsonl")
    attach_mfu_samples(rounds, mfu)

    if deep:
        for r in rounds:
            for aid in r["dev_agents"]:
                t = agents_dir / str(aid) / "stdout.log"
                if t.exists():
                    scan = scan_transcript(t, top)
                    if "deep" in r:  # merge multi-spawn (CLI-retry) rounds
                        r["deep"]["n_calls"] += scan["n_calls"]
                        r["deep"]["tool_s"] += scan["tool_s"]
                        r["deep"]["errors"] += scan["errors"]
                        r["deep"]["gate_events"] += scan["gate_events"]
                    else:
                        r["deep"] = scan
            if "deep" in r:
                r["deep"]["gate_events"].sort(key=lambda e: e["ts"])

    return {
        "header": {
            "loop_id": loop_id,
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "alive": exit_rec is None,
            "last_event": fmt(last_ts),
            "launched": fmt(first_ts),
            "wall_s": (end_ts - first_ts) if first_ts else 0,
            "config": load_config(loop_dir / "config"),
        },
        "exit": exit_rec,
        "gate_configs": load_gate_configs(loop_dir / "config"),
        "milestones": milestones,
        "rounds": rounds,
        "mfu": mfu,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("loop_id")
    ap.add_argument(
        "--root", default=".artifacts", help="artifacts root or repo root (default .artifacts)"
    )
    ap.add_argument(
        "--deep", action="store_true", help="walk agent transcripts for time buckets + agent errors"
    )
    ap.add_argument("--top", type=int, default=3, help="longest tool calls per round in --deep")
    ap.add_argument("--json", metavar="PATH", help="also dump the raw report as JSON")
    ap.add_argument("--md", metavar="PATH", help="also write the text report to a file")
    args = ap.parse_args()

    root = Path(args.root)
    if not (root / "web-agents").exists() and (root / ".artifacts" / "web-agents").exists():
        root = root / ".artifacts"  # accept a repo root too

    report = build_report(args.loop_id, root, args.deep, args.top)
    render(report, sys.stdout)
    if args.md:
        with open(args.md, "w") as fh:
            render(report, fh)
        print(f"\n[written] {args.md}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=1, default=str)
        print(f"[written] {args.json}")


if __name__ == "__main__":
    main()
