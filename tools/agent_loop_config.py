from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from harness import config_runtime
from tools import stage2_config
from tools.codex_config import active_provider_env_key

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

REPO_ROOT = config_runtime.repo_root()

# Stage rule templates live at ``prompt/develop_prompt/_shared/<stage>/*.md``
# and are returned verbatim — no placeholder rendering. Per-backend
# overrides at ``prompt/develop_prompt/<backend>/<stage>/<name>`` are
# returned verbatim when the shared file is absent.
_SHARED_PROMPT_DIR = "_shared"

AGENT_CLIS: frozenset[str] = frozenset({"cursor", "claude", "codex"})

# Per-CLI availability descriptor. Each backend exposes a native auth
# status subcommand whose JSON output is the canonical source of truth
# for "is the user logged in?" — we shell out to it rather than
# re-implementing keychain / login-config / env-var resolution ourselves.
_BACKEND_PROBES: dict[str, dict[str, Any]] = {
    "cursor": {
        "binary": "agent",
        "env_var": "CURSOR_API_KEY",
        "status_cmd": ["agent", "status", "--format", "json"],
        "status_field": "isAuthenticated",
        "login_hint": "agent login",
    },
    "claude": {
        "binary": "claude",
        "env_var": "ANTHROPIC_API_KEY",
        "status_cmd": ["claude", "auth", "status", "--json"],
        "status_field": "loggedIn",
        "login_hint": "claude /login",
    },
    "codex": {
        "binary": "codex",
        "env_var": "OPENAI_API_KEY",
        "status_cmd": ["codex", "login", "status"],
        "status_field": "",
        "login_hint": "codex login",
    },
}

# Map web backend names (cursor-cli, claude-code, codex) to CLI short names.
_BACKEND_TO_CLI: dict[str, str] = {
    "cursor-cli": "cursor",
    "claude-code": "claude",
    "codex": "codex",
}
_CLI_TO_BACKEND: dict[str, str] = {v: k for k, v in _BACKEND_TO_CLI.items()}


def load_agent_config() -> dict[str, Any]:
    """Load the ``[agent]`` table from config/agent.toml (merged into workload config)."""
    _, data = config_runtime.load_workload_config(None)
    cfg = data.get("agent", {})
    if not isinstance(cfg, dict):
        raise ValueError("[agent] must be a table (config/agent.toml)")
    return cfg


def load_remote_config() -> dict[str, Any]:
    """Load the optional ``[remote]`` table for remote execution.

    ``[remote]`` lives in ``config/remote.toml`` (per-user, gitignored; a
    commented template ships in ``config/remote/*.toml``). The
    ``[remote].kind`` field selects the protocol:

    * ``local`` (or table absent) — remote execution is **off**;
      ``agent-loop.sh`` runs every command locally.
    * ``ssh`` — transport only: rsync / ssh to a static host. No
      devspace lifecycle.
    * ``devspace`` — transport **plus** ``cctl`` lease lifecycle
      (``claim``/``rebind``/``release``); the wrapper creates a per-loop
      exclusive devspace from the configured spec
      (project/cluster/resource_pool/image/gpu_count/gpu_model/priority)
      that holds ``gpu_count`` GPUs for the whole loop.
    * ``job`` — same ``cctl`` lifecycle as ``devspace`` but the box is
      claimed with 0 GPUs (a filesystem gateway) and each GPU suite runs
      as a per-suite ephemeral ``cctl`` GPU job.

    Returns an empty dict when the table is missing, so callers can treat
    "remote not configured" and "kind = local" uniformly.
    """
    _, data = config_runtime.load_workload_config(None)
    cfg = data.get("remote", {})
    if not isinstance(cfg, dict):
        raise ValueError("[remote] must be a table (config/remote.toml)")
    return cfg


# Backward-compatible alias used by tests.
load_agent_loop_automation = load_agent_config


def agent_loop_stages() -> list[str]:
    _, data = config_runtime.load_workload_config(None)
    grouped = config_runtime.suites_by_stage(data)
    return sorted(stage for stage in grouped if stage != "local")


# Files always included for the stage (no milestone filtering).
_STAGE_DIR_MANIFEST_ALWAYS: dict[str, list[str]] = {
    "stage1": ["overview.md", "constraint.md", "debug.md"],
}


def stage_milestones(stage: str) -> list[str]:
    """Linear milestone progression for *stage*, e.g.
    ``["alignment", "bitwise-singlecard", ..., "production"]``.

    Sole SSOT: the active ``eval.toml`` ``[<stage>].milestone_order`` list
    (see milestone_rename_design.md §5). Returns an empty list for stages
    with no ``milestone_order`` key (currently every stage other than
    stage1). The terminal milestone hands off to ``STAGE_STATUS: finished``;
    the per-milestone prompt file is ``<name>.md`` (see _stage_milestone_files).
    """
    _, data = config_runtime.load_workload_config(None)
    order = data.get(stage, {}).get("milestone_order")
    if not isinstance(order, list):
        return []
    return [str(name) for name in order]


def _slow_poll_milestone_from_evals(evals: Mapping[str, object], stage: str = "stage1") -> str:
    """Milestone of the multi-day production long-train suite
    (``runner_kind == "production-train"``) within *evals*, or ``""`` if there
    is none. Pure (no config IO) so it is unit-testable without a full
    workload config."""
    for suite in evals.values():
        if (
            isinstance(suite, dict)
            and suite.get("runner_kind") == "production-train"
            and suite.get("stage", stage) == stage
        ):
            return str(suite.get("milestone") or "")
    return ""


def slow_poll_milestone(stage: str = "stage1") -> str:
    """The milestone agent-loop.sh's slow-poll throttle fires on — the
    multi-day production long-train (the suite whose ``runner_kind`` is
    ``production-train``). Returns ``""`` when the active config ships no such
    suite (e.g. the 8b suite, which ends at ``bitwise-dptp``).

    Exported as ``FORGE_SLOW_POLL_MILESTONE`` so the throttle keys on the
    eval.toml SSOT, not a hardcoded milestone name — renaming the production
    milestone cannot silently disable the throttle (the M7 -> ``production``
    rename already broke it once)."""
    _, data = config_runtime.load_workload_config(None)
    return _slow_poll_milestone_from_evals(data.get("evals") or {}, stage)


def _stage_milestone_files(stage: str) -> list[str]:
    """Per-milestone prompt files for *stage*, derived ``<name>.md`` from the
    milestone order. Only the currently-active milestone's file is injected
    each round; the others are omitted from the standing rule files."""
    return [f"{name}.md" for name in stage_milestones(stage)]


def stage_rule_files(stage: str, milestone: str | None = None) -> list[Path]:
    """Stage rule files cat'd into the dev/review prompt by agent-loop.sh.

    When *milestone* is provided and the stage has a milestone manifest,
    only the matching per-milestone ``<name>.md`` is included alongside the always-loaded
    files. When *milestone* is ``None`` every milestone file is included
    (backward-compatible behavior for stage2 and any pre-milestone-aware
    caller).

    Resolution order (first existing wins):

    1. **Shared template** at ``prompt/develop_prompt/_shared/<stage>/<name>``
       (directory form, e.g. stage1) — returned verbatim. No placeholder
       rendering; the markdown is backend-neutral, with backend-specific
       facts inlined as union prose.
    2. **Backend-specific override** at
       ``prompt/develop_prompt/<backend>/<stage>/<name>`` (directory
       form): returned verbatim. Allows per-backend files that resist
       sharing.
    3. **Shared single-file form** at
       ``prompt/develop_prompt/_shared/<stage>.md`` (e.g. stage2) —
       returned verbatim.
    4. **Legacy backend single-file form** at
       ``prompt/develop_prompt/<backend>/<stage>.md``.

    ``backend = "megatron"`` is the default.
    """
    if stage not in agent_loop_stages():
        raise ValueError(f"unknown stage: {stage}")
    _, data = config_runtime.load_workload_config(None)
    backend = str(data.get("ref", {}).get("backend", "megatron"))
    backend_root = REPO_ROOT / "prompt" / "develop_prompt" / backend
    stage_dir = backend_root / stage
    stage_md = backend_root / f"{stage}.md"
    shared_root = REPO_ROOT / "prompt" / "develop_prompt" / _SHARED_PROMPT_DIR
    shared_stage_dir = shared_root / stage
    shared_stage_md = shared_root / f"{stage}.md"

    always = _STAGE_DIR_MANIFEST_ALWAYS.get(stage)
    milestone_names = stage_milestones(stage)
    milestones = _stage_milestone_files(stage) or None

    if milestone is not None:
        if milestones is None:
            raise ValueError(
                f"stage {stage!r} has no milestone manifest; do not pass a milestone argument"
            )
        if milestone not in milestone_names:
            raise ValueError(
                f"milestone {milestone!r} not in manifest for {stage!r}: {milestone_names}"
            )
        target = f"{milestone}.md"
        manifest = [*(always or []), target]
    elif always is not None or milestones is not None:
        manifest = [*(always or []), *(milestones or [])]
    else:
        manifest = None

    if manifest is not None and (shared_stage_dir.is_dir() or stage_dir.is_dir()):
        files: list[Path] = []
        missing: list[str] = []
        for name in manifest:
            shared_src = shared_stage_dir / name
            override_src = stage_dir / name
            if shared_src.exists():
                files.append(shared_src)
            elif override_src.exists():
                files.append(override_src)
            else:
                missing.append(name)
        if missing:
            raise FileNotFoundError(
                f"Stage rule files missing for stage={stage!r}, "
                f"backend={backend!r}: "
                + ", ".join(missing)
                + f" (looked under {shared_stage_dir} and {stage_dir})"
            )
        return files

    if shared_stage_md.exists():
        return [shared_stage_md]

    if stage_md.exists():
        return [stage_md]

    raise FileNotFoundError(
        f"Stage rules not found for stage={stage!r}, "
        f"backend={backend!r}: expected one of "
        f"{shared_stage_md}, {shared_stage_dir}, {stage_md}, {stage_dir}.\n"
        f"Either the backend is misconfigured or the rule files were "
        f"removed."
    )


def stage_review_template(stage: str) -> Path:
    if stage not in agent_loop_stages():
        raise ValueError(f"unknown stage: {stage}")
    return REPO_ROOT / "prompt" / "review_prompt" / f"review_{stage}.md"


def coding_guidelines_file() -> Path:
    return REPO_ROOT / "prompt" / "review_prompt" / "coding-guidelines.md"


def common_review_template() -> Path:
    return REPO_ROOT / "prompt" / "review_prompt" / "review_common.md"


def state_dir() -> Path:
    """Absolute path of the per-stage agent-loop state directory.

    SSOT comes from ``[agent].state_dir`` in config/agent.toml, interpreted
    relative to the repo root.  The directory is created on demand by
    the agent loop.
    """
    cfg = load_agent_config()
    raw = str(cfg.get("state_dir", _AGENT_DEFAULTS["state_dir"]))
    return REPO_ROOT / raw


def state_file_for(stage: str) -> Path:
    """Path to the status marker for *stage* (e.g. `<state_dir>/stage1.status`)."""
    if stage not in agent_loop_stages():
        raise ValueError(f"unknown stage: {stage}")
    return state_dir() / f"{stage}.status"


VALID_STAGE_STATUSES = ("pending", "in-progress", "finished")


def check_backend_available(
    cli: str,
    *,
    which: Callable[[str], str | None] | None = None,
    run_status: Callable[[list[str]], tuple[int, str, str]] | None = None,
) -> tuple[bool, str]:
    """Probe whether the agent CLI is installed and authenticated.

    Returns ``(ok, message)``. ``message`` is an actionable diagnostic
    when ``ok`` is False: it states what is missing and the exact login
    command (or env var) the user should run to resolve it.

    Delegates the auth decision to the backend CLI's own status
    subcommand (``agent status --format json`` /
    ``claude auth status --json``) so we stay in sync with whatever
    resolution order the CLI itself uses (keychain -> login config ->
    env var) rather than re-implementing it.

    ``which`` and ``run_status`` are injection points for tests; callers
    normally let them default to :func:`shutil.which` and a small
    :mod:`subprocess` wrapper.
    """
    if cli not in AGENT_CLIS:
        raise ValueError(f"unknown agent cli: {cli!r}")
    which_fn = which or shutil.which
    run_fn = run_status or _default_run_status
    probe = _BACKEND_PROBES[cli]
    binary: str = probe["binary"]
    if which_fn(binary) is None:
        return (
            False,
            f"{cli} backend unavailable: '{binary}' not found on PATH. "
            f"Install the {cli} CLI and retry.",
        )
    cmd: list[str] = probe["status_cmd"]
    field: str = probe["status_field"]
    env_var: str = probe["env_var"]
    if cli == "codex":
        try:
            env_var = active_provider_env_key()
        except ValueError as exc:
            return False, str(exc)
    login_hint: str = probe["login_hint"]
    returncode, stdout, stderr = run_fn(cmd)
    parsed: Any = None
    if stdout.strip():
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            parsed = None
    if field and isinstance(parsed, dict) and parsed.get(field) is True:
        email = ""
        user_info = parsed.get("userInfo")
        if isinstance(user_info, dict):
            email = user_info.get("email", "")
        return True, json.dumps({"ok": True, "email": email})
    status_text = f"{stdout}\n{stderr}".lower()
    if not field and returncode == 0 and "not logged" not in status_text:
        return True, json.dumps({"ok": True, "email": ""})
    if cli == "codex" and os.environ.get(env_var):
        return True, json.dumps({"ok": True, "email": ""})
    if not field and "not logged" in status_text:
        return (
            False,
            f"{cli} backend unauthenticated: '{' '.join(cmd)}' reports not logged in. "
            f"Run '{login_hint}' to log in, or export ${env_var}=<key>.",
        )
    if not stdout.strip() or parsed is None:
        detail = stderr.strip() or stdout.strip() or f"exit code {returncode}"
        return (
            False,
            f"{cli} backend unavailable: '{' '.join(cmd)}' failed: {detail}",
        )
    return (
        False,
        f"{cli} backend unauthenticated: '{' '.join(cmd)}' reports {field}=false. "
        f"Run '{login_hint}' to log in, or export ${env_var}=<key>.",
    )


def _default_run_status(cmd: list[str]) -> tuple[int, str, str]:
    """Run *cmd* and return ``(returncode, stdout, stderr)``."""
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def active_cli() -> str:
    """Resolve the active CLI without triggering full config validation.

    Used by ``agent-loop.sh status`` so a pure auth probe works even on
    a freshly-cloned repo with no ``config/agent.toml``.
    """
    try:
        cfg = load_agent_config()
    except (FileNotFoundError, ValueError):
        return "cursor"
    raw = str(cfg.get("backend", "cursor-cli"))
    return _BACKEND_TO_CLI.get(raw, "cursor")


_AGENT_DEFAULTS = {
    "backend": "cursor-cli",
    "model": "claude-sonnet-4",
    "runs_per_stage": 0,
    "poll_seconds": 5,
    "production_poll_seconds": 1800,
    "agent_round_tries": 8,
    "retry_base_sleep": 4,
    "retry_continue_prompt": (
        "Transient Cursor API/network disconnect. Continue exactly where you "
        "left off; do not restart the whole iteration from scratch."
    ),
    "state_dir": ".artifacts/agent-loop-state",
    "max_consecutive_review_fails": 3,
    "round_timeout_s": 18000,
    "max_mode": False,
    "effort": "",
    # claude-code thinking-stream knob ("summarized" / "omitted" / "").
    # Opus 4.8 emits NO chain-of-thought into stream-json transcripts
    # unless the request asks for `display: summarized`, so profiles
    # that pin an Opus 4.8 id (directly or via an llm-center alias)
    # set "summarized" to keep loop logs debuggable. Empty omits the
    # flag entirely (CLI default; Fable 5 emits thinking regardless).
    "thinking_display": "",
    # Per-command wall-clock cap for the claude-code backend. Exported as
    # FORGE_CMD_TIMEOUT and consumed by web/agents/timeout_shell/zsh (the
    # fake-$SHELL shim installed by backends.py): every Bash command claude
    # runs is wrapped in `timeout <cmd_timeout>`, group-killing runaway
    # background poll-loops that otherwise keep `claude -p` alive past its
    # turn result. Accepts any coreutils `timeout` DURATION ("40m", "1h",
    # "3600s"). Empty string disables the cap (shim execs the real shell
    # unchanged). claude-code only — cursor/codex leave SHELL untouched.
    "cmd_timeout": "40m",
    # Optional claude-code proxy knobs. Empty `base_url` => the Claude Code
    # CLI uses its own logged-in claude.ai OAuth subscription (the global
    # default stays untouched). A non-empty value is exported as
    # ANTHROPIC_BASE_URL *inside agent-loop.sh's process only* (never a
    # shell profile or ~/.claude.json), so the proxy override is confined
    # to the one loop. `api_key`, when set, fills ANTHROPIC_API_KEY the
    # same process-local way (only if --api-key was not passed on the CLI).
    "base_url": "",
    "api_key": "",
}


def shell_exports(env: Mapping[str, str] | None = None) -> str:
    cfg = {**_AGENT_DEFAULTS, **load_agent_config()}
    backend = str(cfg["backend"])
    if backend not in ("cursor-cli", "claude-code", "codex"):
        raise ValueError("[agent].backend must be one of: cursor-cli, claude-code, codex")
    model = cfg["model"]
    values = {
        "AGENT_BACKEND": backend,
        "CODEX_API_KEY_ENV": active_provider_env_key(),
        "MODEL": str(model),
        "REVIEW_MODEL": "",
        "RUNS_PER_STAGE": str(cfg["runs_per_stage"]),
        "POLL_SECONDS": str(cfg["poll_seconds"]),
        "PRODUCTION_POLL_SECONDS": str(cfg["production_poll_seconds"]),
        "CURSOR_AGENT_ROUND_TRIES": str(cfg["agent_round_tries"]),
        "CURSOR_AGENT_RETRY_BASE_SLEEP": str(cfg["retry_base_sleep"]),
        "CURSOR_AGENT_RETRY_CONTINUE_PROMPT": str(cfg["retry_continue_prompt"]),
        "AGENT_LOOP_STATE_DIR": str(state_dir()),
        "MAX_CONSECUTIVE_REVIEW_FAILS": str(cfg["max_consecutive_review_fails"]),
        "ROUND_TIMEOUT_S": str(cfg["round_timeout_s"]),
        "MAX_MODE": "true" if cfg.get("max_mode", False) else "false",
        "EFFORT": str(cfg.get("effort", "") or ""),
        "THINKING_DISPLAY": str(cfg.get("thinking_display", "") or ""),
        # Per-command timeout cap for the claude-code fake-$SHELL shim.
        # Empty => no cap. Consumed by web/agents/timeout_shell/zsh.
        "FORGE_CMD_TIMEOUT": str(cfg.get("cmd_timeout", "") or ""),
        # Process-local claude-code proxy override; agent-loop.sh exports
        # these as ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY only within its
        # own process tree. Empty => no override (global OAuth default).
        "AGENT_BASE_URL": str(cfg.get("base_url", "") or ""),
        "AGENT_API_KEY": str(cfg.get("api_key", "") or ""),
    }

    # Optional [remote] plugin: when enabled, agent-loop.sh injects the
    # remote-execution prompt overlay into the dev/review prompt and
    # the values below feed the @@KEY@@ placeholders in
    # prompt/develop_prompt/remote-execution.md.
    #
    # REMOTE_WORKDIR is composed as ``<workspace>/.forge_train/<loop_id>``
    # so sibling loops on the same devspace each get an isolated
    # checkout. An empty ``[remote].workspace`` defaults to the literal
    # string ``$HOME`` (expanded by the remote shell at command time);
    # the loop id is read from ``LOOP_ID`` or ``LOOP_WEB_ID`` (the
    # latter is exported by the bootstrap block before the in-workspace
    # exec, while ``LOOP_ID`` is only set further down agent-loop.sh —
    # we honor both so the per-loop suffix is present regardless of
    # whether ``shell_exports`` runs pre- or post-exec).
    remote = load_remote_config()
    kind = str(remote.get("kind", "local") or "local")
    workspace_root = str(remote.get("workspace", "") or "$HOME")
    loop_id = os.environ.get("LOOP_ID") or os.environ.get("LOOP_WEB_ID") or ""
    if loop_id:
        remote_workdir = f"{workspace_root}/.forge_train/{loop_id}"
    else:
        remote_workdir = f"{workspace_root}/.forge_train"
    values.update(
        {
            "REMOTE_KIND": kind,
            "REMOTE_ENABLED": "true" if kind != "local" else "false",
            "REMOTE_SSH_HOST": str(remote.get("hostname", "")),
            "REMOTE_WORKDIR": remote_workdir,
            "REMOTE_LOOP_ID": loop_id,
            # The remote `harness run` must read config from an explicit
            # FORGE_CONFIG_DIR rather than config_runtime._user_config_dir()'s
            # cwd fallback. `sync push` lands the active *.toml in
            # <remote_workdir>/config; the remote-execution overlay exports
            # this value so both sides resolve the same SSOT config dir.
            "REMOTE_CONFIG_DIR": f"{remote_workdir}/config",
        }
    )

    return "\n".join(
        f"export {key}={stage2_config.shell_quote(value)}" for key, value in values.items()
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if args == ["shell"]:
            print(shell_exports())
            return 0
        if args == ["stages"]:
            print("\n".join(agent_loop_stages()))
            return 0
        if len(args) == 2 and args[0] == "stage-rule-files":
            print("\n".join(str(path) for path in stage_rule_files(args[1])))
            return 0
        if len(args) == 3 and args[0] == "stage-rule-files":
            print("\n".join(str(path) for path in stage_rule_files(args[1], args[2])))
            return 0
        if len(args) == 2 and args[0] == "stage-milestones":
            print("\n".join(stage_milestones(args[1])))
            return 0
        if len(args) == 2 and args[0] == "slow-poll-milestone":
            print(slow_poll_milestone(args[1]))
            return 0
        if len(args) == 2 and args[0] == "review-template":
            print(stage_review_template(args[1]))
            return 0
        if args == ["coding-guidelines"]:
            print(coding_guidelines_file())
            return 0
        if args == ["common-review-template"]:
            print(common_review_template())
            return 0
        if len(args) == 2 and args[0] == "check-backend":
            ok, message = check_backend_available(args[1])
            stream = sys.stdout if ok else sys.stderr
            print(message, file=stream)
            return 0 if ok else 1
        if args == ["active-cli"]:
            print(active_cli())
            return 0
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        "Usage: agent_loop_config.py {shell|stages|stage-rule-files <stage> [milestone]|"
        "stage-milestones <stage>|slow-poll-milestone <stage>|"
        "review-template <stage>|coding-guidelines|common-review-template|"
        "check-backend <cursor|claude|codex>|active-cli}",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
