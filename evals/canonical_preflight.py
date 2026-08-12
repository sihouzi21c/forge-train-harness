"""Idempotent canonical-state bootstrap, run once before the first dev round.

The agent-loop wrapper (``agent-loop.sh``) invokes this after the per-loop
config is frozen and before the first stage, so a dev/review agent never has
to hand-run ``tools/bootstrap_canonical.py`` — the missing-DATA_CONF footgun
that cost alignment loops ~30 min of an agent bare-ssh'ing and rewriting env
setup by hand.

Both ``forge_init_ones`` schemes are ensured::

    <ref.checkpoint_root>/ones/canonical_state_fp32.pt   (--init-ones 1)
    <ref.checkpoint_root>/no1/canonical_state_fp32.pt     (--init-ones 0)

Idempotent BY FINGERPRINT: a canonical is left untouched only when its
sidecar (``canonical_state_fp32.pt.fingerprint.json``) matches the hash of
the CURRENT rendered ref product that would produce it — so a resumed or
re-launched loop pays the (multi-minute) bootstrap cost at most once per
deployment, while a canonical left behind by a previous loop with a
different model axis (the stale MiniCPM-structure canonical that poisoned a
Qwen3 loop's bitwise gates) is detected and regenerated instead of trusted.
No-op for non-torch ref backends, matching
``bootstrap_canonical.py`` which only supports ``backend=torch``.
Fail-fast: a bootstrap subprocess error aborts the preflight (and, via
the wrapper, the loop) — the stage1 bitwise gates cannot pass without
these canonicals, so surfacing the failure here saves the agent from a
doomed run.

The ref-run env (DATA_CONF / FORGE_* model+optim projections / resolved
tokenizer+data asset dirs) is assembled by reusing the dispatcher's own
``_ref_script_path_env`` so the bootstrap subprocess sees exactly the
environment a ref gate would. This is why the block lives in ``evals``
(which already depends on ``tools``) and not in ``tools`` (which must not
depend on ``evals`` — that would close a dependency cycle).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from evals._common import _ref_script_path_env
from harness import config_runtime
from tools import bootstrap_canonical

# forge_init_ones schemes to materialize, in documented order:
# ones = production, no1 = anti-cheat bitwise gates.
_INIT_ONES_SCHEMES: tuple[int, ...] = (1, 0)


def _repo_root() -> Path:
    # evals/canonical_preflight.py → evals/ → harness/ (the import root).
    return Path(__file__).resolve().parent.parent


def missing_init_ones(
    workload,
    *,
    output_root: str | None = None,
    ref_config_dir: Path | None = None,
) -> list[int]:
    """Return the ``forge_init_ones`` schemes whose canonical must be (re)built.

    With ``ref_config_dir`` (the live preflight path) a scheme is also
    reported when its canonical is STALE — the fingerprint sidecar is absent
    or no longer matches the current rendered ref product (see
    ``bootstrap_canonical.canonical_is_current``). Without it the probe is
    existence-only (unit-test hook; no rendered products required).
    """
    if ref_config_dir is None:
        return [
            n
            for n in _INIT_ONES_SCHEMES
            if not bootstrap_canonical.canonical_path(workload, n, output_root=output_root).exists()
        ]
    return [
        n
        for n in _INIT_ONES_SCHEMES
        if not bootstrap_canonical.canonical_is_current(
            workload, n, ref_config_dir=ref_config_dir, output_root=output_root
        )
    ]


def build_bootstrap_env(repo_root: Path, workload) -> dict[str, str]:
    """Process env plus the ref-run PATH/DATA env a canonical dump needs.

    Reuses the dispatcher's ``_ref_script_path_env`` (the SSOT for ref-run
    env assembly) so DATA_CONF and the resolved tokenizer/data asset dirs
    are present — exactly the gap that forced agents to hand-add DATA_CONF
    before bootstrapping. Model/optim structure does NOT travel via env:
    ``bootstrap_canonical.py`` picks a FORGE_GATE and the ref launcher
    sources every launcher param from that gate's rendered ref product —
    the fingerprint sidecar pins the canonical to those product bytes.
    """
    env = dict(os.environ)
    env.update(_ref_script_path_env(repo_root, {}, workload_config=workload))
    return env


def run(argv: list[str] | None = None) -> int:
    repo_root = _repo_root()
    _path, workload = config_runtime.load_workload_config()
    backend = str(workload.get("ref", {}).get("backend", ""))
    if backend != "torch":
        print(
            f"[canonical_preflight] ref backend={backend!r} is not 'torch'; "
            "canonical bootstrap skipped.",
            flush=True,
        )
        return 0

    missing = missing_init_ones(workload, ref_config_dir=repo_root / "ref" / "config")
    if not missing:
        print(
            "[canonical_preflight] both forge_init_ones canonicals present "
            "and fingerprint-current; nothing to do.",
            flush=True,
        )
        return 0

    env = build_bootstrap_env(repo_root, workload)
    script = repo_root / "tools" / "bootstrap_canonical.py"
    for init_ones in missing:
        target = bootstrap_canonical.canonical_path(workload, init_ones)
        print(
            f"[canonical_preflight] generating {target} (--init-ones {init_ones})",
            flush=True,
        )
        result = subprocess.run(
            [sys.executable, str(script), "--init-ones", str(init_ones)],
            cwd=repo_root,
            env=env,
        )
        if result.returncode != 0:
            raise SystemExit(
                f"[canonical_preflight] bootstrap_canonical --init-ones "
                f"{init_ones} failed (rc={result.returncode}); stage1 bitwise "
                f"gates cannot pass without {target}. Fix the ref bridge, then "
                "re-launch."
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
