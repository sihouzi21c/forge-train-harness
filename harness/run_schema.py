from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

__all__ = [
    "RESULT_BEGIN",
    "RESULT_END",
    "RUNNER_METRICS",
    "make_failed_result",
    "make_run_result",
    "schema_metadata",
    "validate_run_request",
    "validate_run_result",
    "validate_safe_relative_path",
]

RESULT_BEGIN = "---HARNESS-RESULT-BEGIN---"
RESULT_END = "---HARNESS-RESULT-END---"

RUN_SCHEMA_VERSION = 1
_RUN_REQUEST_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "suite",
        "args",
        "report",
        "target",
        "gpu_count",
        "artifact_relpath",
        "requested_at",
        "requested_from",
        "workload_config",
        "workload_config_path",
        "workload_config_sha",
    }
)
_RUN_RESULT_REQUIRED_KEYS = frozenset(
    {"schema_version", "status", "suite", "summary", "metrics", "details"}
)
_RUN_RESULT_STATUSES = frozenset({"passed", "failed"})


# Authoritative metric-name contract per ``runner_kind``.
#
# This is the *minimal* set of keys the corresponding dispatcher MUST
# include in ``result["metrics"]``.  Extra keys are allowed (the schema
# is lower-bound, not exact), but missing keys break ``harness info``
# contract reporting and downstream agent-loop parsing.
#
# Verified end-to-end by ``harness/tests/test_runner_metrics_contract.py``
# (added together with this contract).  When you add a metric in the
# dispatcher, add it here too — the contract test will fail otherwise.
RUNNER_METRICS: dict[str, list[str]] = {
    "forward-align": ["checks_total", "bitwise_match", "hash_capture_level"],
    "backward-align": ["checks_total", "bitwise_match", "hash_capture_level"],
    # ``stage1-bitwise-trajectory`` drives every multi-step bitwise gate
    # in Stage 1 (``multistep-1gpu``, ``multistep``,
    # ``perf-bitwise``).  All three suites speak the same ref-vs-ours
    # subprocess protocol through ``_run_bitwise_trajectory`` in
    # evals/dispatcher.py — the per-step bitwise / gate-window metrics
    # below are emitted on every run; ``avg_mfu_e2e_standard`` /
    # ``mfu_target`` / ``mfu_pass`` are populated whenever
    # ``cfg.mfu_e2e_target`` > 0 (currently only ``perf-bitwise``) and
    # ``None`` otherwise so the contract shape is stable across the
    # family. ``hash_*`` metrics are populated whenever
    # ``cfg.hash_capture_level`` > 0 (those three gates set it to 2/2/1 in the
    # current TOML) and ``None`` otherwise; Phase-4 of the trajectory
    # driver ANDs the hash diff into the gate verdict.
    "stage1-bitwise-trajectory": [
        "checks_total",
        "bitwise_match",
        "gate_pass",
        "gate_atol",
        "grad_checks_total",
        "grad_bitwise_match",
        "grad_gate_pass",
        "grad_gate_atol",
        "world_size",
        "num_steps",
        "steps_observed",
        "ref_steps_observed",
        "ref_elapsed_s",
        "gate_window",
        "avg_mfu_e2e_standard",
        "mfu_target",
        "mfu_pass",
        "correctness_pass",
        "hash_capture_level",
        "hash_pass",
        "hash_checks_total",
        "hash_bitwise_match",
    ],
    "long-train": [
        "world_size",
        "num_steps",
        "steps_observed",
        "ref_steps_observed",
        "ref_elapsed_s",
        "gate_window",
        "pointwise_mean_rel",
        "max_rel_diff",
        "signed_mean",
        "signed_mean_rel",
        "compared_steps",
        "loss_rel_threshold",
        "loss_pass",
        "avg_mfu_e2e_standard",
        "mfu_target",
        "mfu_pass",
    ],
    "op-long": [
        "op_names",
        "world_size",
        "num_steps",
        "gate_window",
        "compared_steps",
        "mean_rel_diff",
        "max_rel_diff",
        "signed_mean",
        "signed_mean_rel",
        "rel_threshold",
        "ref_elapsed_s",
        "passed",
    ],
    "op-status": ["operators", "summary_counts"],
    "op-inventory": ["operators_registered", "validation_errors", "registry_path"],
    "guard": ["violations"],
    "unit": [],
    "anti-proxy": ["violations"],
    "resume-gate": [
        "world_size",
        "num_steps",
        "ref_steps_observed",
        "res_steps_observed",
        "gate_window",
        "save_step",
        "max_abs_diff_loss",
        "max_abs_diff_grad_norm",
        "bitwise_pass",
        "passed",
        "hash_capture_level",
        "hash_pass",
        "hash_checks_total",
        "hash_bitwise_match",
    ],
    # ours-only resume startup-time gate: bound the GPU-idle dataloader seek
    # on resume. Verdict is dispatcher-owned (parses ``[RESUME_STARTUP]`` and
    # compares against the frozen ``resume_startup_budget_s``); the metric set
    # is the measured seek, the budget it was checked against, the replayed
    # micro-batch count, and the run shape.
    "resume-startup": [
        "resume_startup_s",
        "budget_s",
        "consumed_microbatches",
        "world_size",
    ],
    "loss-gate": [
        "world_size",
        "num_steps",
        "steps_observed",
        "ref_steps_observed",
        "ref_elapsed_s",
        "gate_window",
        "avg_relative_loss_diff",
        "max_relative_loss_diff",
        "signed_mean",
        "signed_mean_rel",
        "compared_steps",
        "gate_threshold",
        "loss_pass",
        "avg_mfu_e2e_standard",
        "passed",
    ],
    "profile-snapshot": [
        "world_size",
        "num_steps",
        "profiled_steps",
        "step_time_ms",
        "mfu_e2e_standard",
        "out_dir",
        "prev_dir",
    ],
    # ours-only production long-train (Scheme B: dispatcher walks
    # ``num_steps`` in 4 equal segments, each saving a resume checkpoint).
    # Checkpoint-only verdict — NO ref comparison, NO loss/MFU threshold —
    # so the metric set is the run shape plus the count of checkpoints
    # that landed on disk.
    "production-train": [
        "world_size",
        "num_steps",
        "save_interval",
        "saved_checkpoints",
        "resumed_from_step",
        "passed",
    ],
    # WSD-SFT 3-phase switch-correctness / crash-resume gate (wsd-sft-70 on
    # the resume milestone; production-resume-70 on the production milestone).
    # Verdict = per-phase per-step loss compare vs the clean-baseline ours run
    # (bitwise at gate_atol=0) + structural [PHASE] switch assertions.
    "wsd-sft": [
        "world_size",
        "stable_steps",
        "decay_steps",
        "sft_steps",
        "gate_atol",
        "max_abs_diff_loss",
        "tolerance_pass",
        "structural_pass",
        "missing_steps",
        "passed",
    ],
}


def schema_metadata() -> dict[str, Any]:
    """Return the machine-readable run request/result artifact contract."""
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "request_required_keys": sorted(_RUN_REQUEST_REQUIRED_KEYS),
        "result_required_keys": sorted(_RUN_RESULT_REQUIRED_KEYS),
        "result_statuses": sorted(_RUN_RESULT_STATUSES),
        "result_markers": {"begin": RESULT_BEGIN, "end": RESULT_END},
        "artifact_files": ["request.json", "result.json"],
    }


def validate_safe_relative_path(value: str, label: str) -> str:
    """Validate and normalise a safe relative POSIX path (SSOT for all callers).

    A safe relative path is non-empty, not absolute, and contains no
    ``..`` segments. Returns the POSIX-normalised string on success.
    Raises :class:`ValueError` otherwise.

    This helper lives in :mod:`harness.run_schema` (a base-layer leaf
    that owns the schema vocabulary) so the validator can be shared by
    :func:`validate_run_request` and by higher layers
    (:mod:`harness.config_runtime`) without a circular dependency.
    """
    if not value:
        raise ValueError(f"{label} must be a safe relative path (got empty string)")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label} must be a safe relative path")
    return candidate.as_posix()


def validate_run_request(payload: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(_RUN_REQUEST_REQUIRED_KEYS - set(payload))
    if missing:
        raise ValueError(f"Malformed run request missing keys: {missing}")
    if payload["schema_version"] != RUN_SCHEMA_VERSION:
        raise ValueError(
            f"Malformed run request has unsupported schema_version: {payload['schema_version']!r}"
        )
    if not isinstance(payload["run_id"], str) or not payload["run_id"]:
        raise ValueError("Malformed run request has invalid run_id")
    if not isinstance(payload["suite"], str) or not payload["suite"]:
        raise ValueError("Malformed run request has invalid suite")
    if not isinstance(payload["args"], list) or not all(
        isinstance(item, str) for item in payload["args"]
    ):
        raise ValueError("Malformed run request has invalid args")
    if payload["report"] not in {"text", "json"}:
        raise ValueError("Malformed run request has invalid report")
    if payload["target"] != "local":
        raise ValueError("Malformed run request has invalid target")
    gpu_count = payload["gpu_count"]
    if gpu_count is not None and (not isinstance(gpu_count, int) or gpu_count <= 0):
        raise ValueError("Malformed run request has invalid gpu_count")
    if not isinstance(payload["requested_at"], str) or not payload["requested_at"]:
        raise ValueError("Malformed run request has invalid requested_at")
    if not isinstance(payload["requested_from"], str) or not payload["requested_from"]:
        raise ValueError("Malformed run request has invalid requested_from")
    if not isinstance(payload["artifact_relpath"], str) or not payload["artifact_relpath"]:
        raise ValueError("Malformed run request has invalid artifact_relpath")
    try:
        validate_safe_relative_path(payload["artifact_relpath"], "artifact_relpath")
    except ValueError as exc:
        raise ValueError("Malformed run request has invalid artifact_relpath") from exc
    if not isinstance(payload["workload_config"], dict):
        raise ValueError("Malformed run request has invalid workload_config")
    if not isinstance(payload["workload_config_path"], str) or not payload["workload_config_path"]:
        raise ValueError("Malformed run request has invalid workload_config_path")
    try:
        validate_safe_relative_path(payload["workload_config_path"], "workload_config_path")
    except ValueError as exc:
        raise ValueError("Malformed run request has invalid workload_config_path") from exc
    sha = payload["workload_config_sha"]
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("Malformed run request has invalid workload_config_sha")
    return payload


def make_run_result(
    *,
    status: str,
    suite: str,
    summary: str,
    metrics: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return validate_run_result(
        {
            "schema_version": RUN_SCHEMA_VERSION,
            "status": status,
            "suite": suite,
            "summary": summary,
            "metrics": dict(metrics or {}),
            "details": dict(details or {}),
        }
    )


def make_failed_result(
    *,
    suite: str,
    summary: str,
    metrics: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return make_run_result(
        status="failed",
        suite=suite,
        summary=summary,
        metrics=metrics,
        details=details,
    )


def validate_run_result(payload: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(_RUN_RESULT_REQUIRED_KEYS - set(payload))
    if missing:
        raise ValueError(f"Malformed suite result missing keys: {missing}")
    normalized = dict(payload)
    schema_version = payload["schema_version"]
    if schema_version != RUN_SCHEMA_VERSION:
        raise ValueError(
            f"Malformed suite result has unsupported schema_version: {schema_version!r}"
        )
    status = payload["status"]
    if status not in _RUN_RESULT_STATUSES:
        raise ValueError(f"Malformed suite result has invalid status: {status!r}")
    if not isinstance(payload["suite"], str) or not payload["suite"]:
        raise ValueError("Malformed suite result has invalid suite")
    if not isinstance(payload["summary"], str):
        raise ValueError("Malformed suite result has invalid summary")
    if not isinstance(payload["metrics"], dict):
        raise ValueError("Malformed suite result has invalid metrics")
    if not isinstance(payload["details"], dict):
        raise ValueError("Malformed suite result has invalid details")
    return normalized
