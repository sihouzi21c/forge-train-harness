from __future__ import annotations

import argparse
import sys

from harness import app, presentation, remote_sync

__all__ = ["main"]


def _add_path_overrides(parser: argparse.ArgumentParser) -> None:
    """Register shared ``--megatron-root`` … CLI flags on *parser*."""
    grp = parser.add_argument_group(
        "path overrides",
        "Override hardcoded paths from config/eval.toml [defaults]. "
        "Environment variables (FORGE_MEGATRON_ROOT, …) are also honoured; "
        "CLI flags take highest precedence.",
    )
    grp.add_argument(
        "--megatron-root",
        default=None,
        metavar="DIR",
        help="Megatron-LM source root (env: FORGE_MEGATRON_ROOT)",
    )
    grp.add_argument(
        "--checkpoint-root",
        default=None,
        metavar="DIR",
        help="Checkpoint / weight root (env: FORGE_CHECKPOINT_ROOT)",
    )
    grp.add_argument(
        "--data-path",
        default=None,
        metavar="PATH",
        help=(
            "Megatron --data-path prefix (HuggingFace dataset preprocessed "
            "into Megatron binary format; env: FORGE_DATA_PATH)"
        ),
    )
    grp.add_argument(
        "--tokenizer-model",
        default=None,
        metavar="PATH",
        help=(
            "SentencePiece tokenizer.model used by the L0 ref script's "
            "--tokenizer-model arg (env: FORGE_TOKENIZER_MODEL)"
        ),
    )
    grp.add_argument(
        "--ref-script",
        default=None,
        metavar="FILE",
        help=("L0 ref-script basename under ref/reference/ (env: FORGE_REF_SCRIPT)"),
    )


def _collect_path_overrides(args: argparse.Namespace) -> dict[str, str] | None:
    """Harvest CLI path overrides into a dict understood by config_runtime."""
    mapping = {
        "megatron_root": getattr(args, "megatron_root", None),
        "checkpoint_root": getattr(args, "checkpoint_root", None),
        "data_path": getattr(args, "data_path", None),
        "tokenizer_model": getattr(args, "tokenizer_model", None),
        "ref_script": getattr(args, "ref_script", None),
    }
    filtered = {k: v for k, v in mapping.items() if v is not None}
    return filtered or None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harness")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a test suite")
    run_p.add_argument("suite", help="Suite to run; use `harness info` to list supported suites")
    run_p.add_argument(
        "suite_args",
        nargs="*",
        help="Positional args forwarded to the suite (for example: op-long <name>)",
    )
    run_p.add_argument("--json", dest="json_output", action="store_true", help="Output as JSON")
    run_p.add_argument("--gpu", default=None, help="GPU device id (CUDA_VISIBLE_DEVICES)")
    run_p.add_argument("--config", default=None, help="Workload config path")
    run_p.add_argument(
        "--timeout",
        type=int,
        default=None,
        metavar="SECS",
        help=(
            "Override the SSOT per-suite budget (seconds). "
            "Use ONLY after deep analysis confirms the dispatcher / runner "
            "is healthy and the toml budget is genuinely too small. "
            "See `harness budget <suite>` for the SSOT value, and the "
            "RED LINE in remote-execution.md before reaching for this."
        ),
    )
    _add_path_overrides(run_p)

    doctor_p = sub.add_parser("doctor", help="Check local environment")
    doctor_p.add_argument("--json", dest="json_output", action="store_true", help="Output as JSON")
    doctor_p.add_argument("--gpu", default=None, help="GPU device id")
    doctor_p.add_argument("--config", default=None, help="Workload config path")
    _add_path_overrides(doctor_p)

    info_p = sub.add_parser("info", help="Show workload metadata")
    info_p.add_argument("--json", dest="json_output", action="store_true", help="Output as JSON")
    info_p.add_argument("--config", default=None, help="Workload config path")
    _add_path_overrides(info_p)

    budget_p = sub.add_parser(
        "budget",
        help=(
            "Print the SSOT wall-clock budget (seconds) for a suite. "
            "Read by tools/remote_run.sh and by humans before deciding "
            "if a `--timeout` override is even worth considering."
        ),
    )
    budget_p.add_argument(
        "suite",
        help="Suite to query; same names as `harness run`/`harness info`",
    )
    budget_p.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Emit {suite, timeout_s, source} JSON instead of a bare integer",
    )
    budget_p.add_argument("--config", default=None, help="Workload config path")
    _add_path_overrides(budget_p)

    sub.add_parser(
        "env-probe",
        help=(
            "Run workspace-contract invariants and exit 0 / 2. "
            "Wired into agent-loop.sh and web/_provision_workspace so a "
            "misconfigured workspace fails fast (e.g. global editable "
            "install shadowing the local harness) instead of burning "
            "agent runtime on opaque infra debugging. See "
            "harness/workspace_contract.py for the invariant list."
        ),
    )

    echo_p = sub.add_parser(
        "echo-config",
        help=(
            "Emit the canonical workload-config bytes to stdout and the "
            "sha256 + resolved path to stderr. Used by the "
            "subprocess_config_isomorphic workspace-contract invariant "
            "to catch PYTHONPATH / cwd / env divergences between the "
            "in-process loader and a child Python process — any drift "
            "would silently corrupt every downstream run."
        ),
    )
    echo_p.add_argument("--config", default=None, help="Workload config path")
    _add_path_overrides(echo_p)

    resources_p = sub.add_parser(
        "resources",
        help=(
            "Manage declared external resources (tokenizer, checkpoint, "
            "megatron source tree). The manifest at "
            "harness/external_resources.toml is the SSOT; provisioning "
            "symlinks verified sources to <workspace>/.resources/..."
        ),
    )
    resources_sub = resources_p.add_subparsers(dest="resources_action", required=True)
    resources_sub.add_parser(
        "provision",
        help=(
            "Provision every manifest entry into <workspace>/.resources/. "
            "Hermetic by default; set FORGE_ALLOW_NETWORK_RESOURCES=1 to "
            "permit hf:// / git+ssh:// fallback sources."
        ),
    )

    sub.add_parser(
        "prefetch",
        help=(
            "Stage the production long-train corpus into this host's data dir. "
            "Workspace-native and host-local: run it where training runs "
            "(locally, or via ssh on the remote execution host) so the "
            "corpus bytes land next to the trainer — symmetric with "
            "`harness run`. agent-loop.sh backgrounds this at loop start "
            "when [data].prefetch_target_gb > 0. See "
            "harness/app.py::prefetch_command for the proxy/HF_ENDPOINT "
            "handling."
        ),
    )

    sync_p = sub.add_parser(
        "sync",
        help="Synchronize workspace to/from the remote execution host",
    )
    sync_sub = sync_p.add_subparsers(dest="sync_action", required=True)
    push_p = sync_sub.add_parser(
        "push",
        help=(
            "rsync local workspace → remote. Cleans stale __pycache__ "
            "and validates the remote directory after sync."
        ),
    )
    push_p.add_argument(
        "--stage2",
        action="store_true",
        help="Stage-2 mode: include .git in the sync (needed for op-status)",
    )
    push_p.add_argument(
        "--verbose",
        action="store_true",
        help="Show rsync file list (passes -v to rsync)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    path_overrides = _collect_path_overrides(args)

    try:
        if args.command == "info":
            payload = app.info_command(
                config_path=args.config,
                json_output=args.json_output,
                path_overrides=path_overrides,
            )
        elif args.command == "doctor":
            payload = app.doctor_command(
                config_path=args.config,
                json_output=args.json_output,
                gpu=args.gpu,
                path_overrides=path_overrides,
            )
        elif args.command == "run":
            payload = app.run_command(
                suite=args.suite,
                suite_args=args.suite_args,
                config_path=args.config,
                json_output=args.json_output,
                gpu=args.gpu,
                path_overrides=path_overrides,
                override_timeout_s=args.timeout,
            )
        elif args.command == "budget":
            payload = app.budget_command(
                suite=args.suite,
                config_path=args.config,
                json_output=args.json_output,
                path_overrides=path_overrides,
            )
        elif args.command == "sync":
            stage = "stage2" if getattr(args, "stage2", False) else "stage1"
            verbose = getattr(args, "verbose", False)
            payload = remote_sync.sync_push(stage=stage, verbose=verbose)
        elif args.command == "env-probe":
            # Dedicated exit-code surface (2 = contract violation) so
            # agent-loop.sh / web can distinguish infra failure from a
            # regular CLI usage error (1). app.env_probe_command bypasses
            # the trailing presentation.render path because env-probe has
            # no suite payload — just a pass/fail with stderr diagnostics.
            return app.env_probe_command()
        elif args.command == "echo-config":
            # Same bypass pattern as env-probe: stdout is the canonical
            # bytes, not a render payload. Exit 1 on load failure (no
            # dedicated code — the byte stream is the contract).
            return app.echo_config_command(
                config_path=args.config,
                path_overrides=path_overrides,
            )
        elif args.command == "prefetch":
            # Same dedicated exit-code bypass as env-probe / echo-config:
            # the corpus download has no render payload — return
            # tools.prefetch_data's exit code verbatim.
            return app.prefetch_command()
        elif args.command == "resources":
            # Same dedicated exit-code pattern as env-probe (2 = provision
            # failure). resources_action is enforced by argparse
            # (required=True) so only "provision" reaches here today.
            if args.resources_action == "provision":
                return app.resources_provision_command()
            raise ValueError(f"Unsupported resources action: {args.resources_action}")
        else:
            raise ValueError(f"Unsupported command: {args.command}")
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    print(presentation.render(payload, payload["report"]))
    status = payload.get("status", "failed")
    return 0 if status in ("passed", "ready") else 1


if __name__ == "__main__":
    sys.exit(main())
