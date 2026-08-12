from __future__ import annotations

import json
from typing import Any

__all__ = ["render"]


def render(payload: dict[str, Any], report: str) -> str:
    if report == "json":
        return json.dumps(payload, indent=2, sort_keys=True, default=str)
    return _render_text(payload)


def _render_text(payload: dict[str, Any]) -> str:
    command = payload["command"]

    if command == "budget":
        # Bare integer — consumed by ``tools/remote_run.sh`` via
        # ``$(harness budget <suite>)``. Anything richer would break
        # the wrapper's arithmetic on the captured stdout.
        return str(payload["payload"]["timeout_s"])

    if command == "info":
        workload = payload["payload"]["workload"]
        lines = [
            f"Workload: {workload['display_name']} ({workload['id']})",
            f"Suites: {', '.join(workload['supported_suites'])}",
            f"Requires CUDA: {'yes' if workload['requires_cuda'] else 'no'}",
        ]
        return "\n".join(lines)

    if command == "doctor":
        target_results = payload.get("targets", [])
        lines = []
        for tr in target_results:
            target_label = tr.get("target") or "local"
            dp = tr["payload"]
            header = f"[{target_label}] {dp['status']} — {dp.get('summary', '')}"
            lines.append(header)
            for check in dp.get("checks", []):
                detail = check.get("detail", "")
                suffix = f" — {detail}" if detail else ""
                lines.append(f"  [{check['status']}] {check['name']}{suffix}")
        return "\n".join(lines)

    if command == "run":
        target_results = payload.get("targets", [])
        lines = []
        for tr in target_results:
            target_label = tr.get("target") or "local"
            rp = tr["payload"]
            lines.append(f"[{target_label}] {rp['status']} — {rp.get('summary', '')}")
            lines.append(f"  Suite: {rp.get('suite', 'unknown')}")
            metrics = rp.get("metrics", {})
            if metrics:
                for k, v in metrics.items():
                    lines.append(f"  {k}: {v}")
            artifact_dir = tr.get("artifact_dir")
            if artifact_dir:
                lines.append(f"  Artifacts: {artifact_dir}")
            # Optional agent-readable Markdown rendering attached by a
            # suite to its run output. Any suite that sets
            # ``details.inline_markdown`` to a non-empty string gets it
            # surfaced verbatim under "inline inventory". Size cap is
            # the suite's responsibility.
            inline_md = (rp.get("details") or {}).get("inline_markdown")
            if isinstance(inline_md, str) and inline_md.strip():
                lines.append("")
                lines.append("───── inline inventory ─────")
                lines.append(inline_md.rstrip())
        return "\n".join(lines)

    if command == "sync":
        p = payload["payload"]
        return (
            f"[sync] {payload['status']} — {p['action']} → "
            f"{p['host']}:{p['dest']} (stage={p['stage']})"
        )

    raise ValueError(f"Unsupported command for rendering: {command}")
