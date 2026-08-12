"""Background corpus prefetch for the production long-train.

It trains at the production global batch (e.g. GBS=1280 × 4096 tok/step ×
10k steps ≈ 52B tokens). The handful of parquet parts baked into the image
(``[data].download_files``) is nowhere near enough — the streaming
dataloader would wrap and repeat, polluting the long-train signal. This
tool stages a much larger slice (``[data].prefetch_target_gb`` GB) of the
``[data].dataset`` repo into ``forge_data_dir/{en,zh}/`` so the data_conf
glob (``en/*.parquet`` / ``zh/*.parquet``) picks it up automatically — no
config change to the dataloader.

It runs detached, in the background, kicked off by agent-loop.sh right after
config freeze, while the pre-production dev loop runs on the baked slice. The production-train runner
blocks on the ``.prefetch_done`` sentinel before its first segment.

The HF-hub download in ``main`` is network-bound and not unit-tested; the
part-enumeration, en/zh interleave, byte accounting, and sentinel I/O are
pure and covered by ``tests/test_prefetch_data.py``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

# Sentinel file written into forge_data_dir once the target is reached (or
# on a fatal error). The production-train runner's readiness gate reads it.
PREFETCH_SENTINEL = ".prefetch_done"

# ``ultrafineweb-en-part-0007-of-2048.parquet`` → (index=7, total=2048,
# width=4). The seed entry in [data].download_files declares one part; we
# derive the full series from its ``-part-<N>-of-<M>`` shape so no network
# repo-listing call is needed.
_PART_RE = re.compile(r"-part-(?P<idx>\d+)-of-(?P<total>\d+)\.parquet$")


def enumerate_parts_from_seed(seed_local: str, seed_repo: str) -> list[tuple[str, str]]:
    """Expand one ``download_files`` seed entry into the full part series.

    ``seed_local`` is the local-relative path (e.g.
    ``en/ultrafineweb-en-part-0001-of-2048.parquet``); ``seed_repo`` is the
    matching path inside the HF dataset repo. Returns ``[(local_rel,
    repo_path), ...]`` for every part 1..total, preserving the seed's
    zero-padding width and surrounding name.
    """
    m = _PART_RE.search(seed_local)
    if not m:
        raise ValueError(f"prefetch: seed {seed_local!r} has no -part-N-of-M suffix")
    total = int(m.group("total"))
    width = len(m.group("idx"))

    def _expand(template: str, idx: int) -> str:
        # Replace only the ``-part-<N>-`` index, keep ``-of-<total>`` intact.
        return re.sub(
            r"(-part-)\d+(-of-\d+\.parquet)$",
            lambda mm: f"{mm.group(1)}{idx:0{width}d}{mm.group(2)}",
            template,
        )

    return [(_expand(seed_local, i), _expand(seed_repo, i)) for i in range(1, total + 1)]


def interleave(
    en_parts: list[tuple[str, str]],
    zh_parts: list[tuple[str, str]],
    en_fraction: float,
) -> list[tuple[str, str]]:
    """Round-robin en/zh parts at roughly ``en_fraction`` English.

    Deterministic: for each output slot, take from en when the running
    English share has dropped below ``en_fraction``, else from zh; exhausted
    sources are skipped. The result is a single download order so the
    target-GB cutoff lands on a representative en/zh mix instead of all-en
    then all-zh.
    """
    if not 0.0 <= en_fraction <= 1.0:
        raise ValueError(f"en_fraction must be in [0,1], got {en_fraction}")
    en, zh = list(en_parts), list(zh_parts)
    ei = zi = 0
    out: list[tuple[str, str]] = []
    while ei < len(en) or zi < len(zh):
        taken = len(out)
        # Take en when the English count is still below its target share of
        # the sequence-so-far-plus-this-slot. ``ei < en_fraction*(taken+1)``
        # is exact at the 0.0 (all-zh-first) and 1.0 (all-en-first) bounds,
        # and Bresenham-spreads the minority source for any fraction between.
        want_en = ei < en_fraction * (taken + 1)
        if want_en and ei < len(en):
            out.append(en[ei])
            ei += 1
        elif zi < len(zh):
            out.append(zh[zi])
            zi += 1
        elif ei < len(en):
            out.append(en[ei])
            ei += 1
    return out


def sentinel_path(forge_data_dir: Path) -> Path:
    return Path(forge_data_dir) / PREFETCH_SENTINEL


def read_prefetch_status(forge_data_dir: Path) -> dict | None:
    """Return the sentinel payload, or None if prefetch hasn't finished."""
    p = sentinel_path(forge_data_dir)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def write_prefetch_status(forge_data_dir: Path, status: dict) -> None:
    p = sentinel_path(forge_data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
    tmp.replace(p)


def wait_for_prefetch(
    forge_data_dir: Path,
    *,
    timeout_s: float,
    poll_s: float = 30.0,
    sleep=time.sleep,
    monotonic=time.monotonic,
) -> dict:
    """Block until the prefetch sentinel reports ``status == "ok"``.

    Fail-fast (RuntimeError) when the sentinel reports an error or when
    ``timeout_s`` elapses with no/incomplete sentinel — the production-train runner must not train on
    half-staged data. ``sleep`` / ``monotonic`` are injectable for tests.
    """
    deadline = monotonic() + timeout_s
    while True:
        status = read_prefetch_status(forge_data_dir)
        if status is not None:
            if status.get("status") == "ok":
                return status
            raise RuntimeError(
                f"prefetch reported status={status.get('status')!r}: "
                f"{status.get('error', '(no detail)')} (sentinel: "
                f"{sentinel_path(forge_data_dir)})"
            )
        if monotonic() >= deadline:
            raise RuntimeError(
                f"prefetch did not finish within {timeout_s}s — no '{PREFETCH_SENTINEL}' "
                f"under {forge_data_dir}. Start tools/prefetch_data.py or lower "
                "[data].prefetch_target_gb."
            )
        sleep(poll_s)


def _data_axis(repo_root: Path) -> dict:
    from harness import config_runtime

    _, wc = config_runtime.load_workload_config(None)
    data = wc.get("data") or {}
    if not isinstance(data, dict):
        raise SystemExit("prefetch: [data] axis missing/invalid")
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Background ultra-fineweb prefetch for the production long-train."
    )
    ap.add_argument("--repo-root", default=".", help="harness repo root")
    ap.add_argument(
        "--target-gb", type=float, default=None, help="override [data].prefetch_target_gb"
    )
    ap.add_argument(
        "--en-fraction",
        type=float,
        default=None,
        help="override [data].prefetch_en_fraction (default 0.9)",
    )
    ap.add_argument(
        "--curl",
        action="store_true",
        help="fetch with curl via HF_ENDPOINT (for whitelist-proxy "
        "clusters where the hf_hub client stalls on the Xet CDN)",
    )
    args = ap.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    data = _data_axis(repo_root)

    target_gb = (
        args.target_gb
        if args.target_gb is not None
        else float(data.get("prefetch_target_gb", 0) or 0)
    )
    if target_gb <= 0:
        print("prefetch: [data].prefetch_target_gb not set (>0); nothing to do.")
        return 0
    en_fraction = (
        args.en_fraction
        if args.en_fraction is not None
        else float(data.get("prefetch_en_fraction", 0.9))
    )

    repo = (data.get("dataset") or "").strip()
    _fdd = (data.get("forge_data_dir") or "").strip()
    files = data.get("download_files") or {}
    if not repo or not _fdd or not files:
        raise SystemExit("prefetch: [data] needs dataset + forge_data_dir + download_files")
    # Relative forge_data_dir resolves against the workspace (--repo-root),
    # NOT cwd, so the corpus lands under the workspace tree. Kept in
    # lock-step with harness.config_runtime.resolve_forge_data_dir on the
    # reader side; this layer must not import harness (closed DAG).
    _fdd_path = Path(_fdd)
    forge_data_dir = _fdd_path if _fdd_path.is_absolute() else (repo_root / _fdd_path).resolve()

    # Build the full en/zh series from the seed entries.
    en_seed = next(((k, v) for k, v in files.items() if k.startswith("en/")), None)
    zh_seed = next(((k, v) for k, v in files.items() if k.startswith("zh/")), None)
    en_parts = enumerate_parts_from_seed(*en_seed) if en_seed else []
    zh_parts = enumerate_parts_from_seed(*zh_seed) if zh_seed else []
    order = interleave(en_parts, zh_parts, en_fraction)

    target_bytes = int(target_gb * (1024**3))
    print(
        f"prefetch: target={target_gb}GB en_fraction={en_fraction} "
        f"en_parts={len(en_parts)} zh_parts={len(zh_parts)} -> {forge_data_dir}"
    )

    fetcher = _curl_fetch(repo) if args.curl else None
    try:
        total, staged = download_until(repo, forge_data_dir, order, target_bytes, fetch=fetcher)
    except Exception as exc:
        write_prefetch_status(
            forge_data_dir,
            {
                "status": "error",
                "error": repr(exc),
            },
        )
        print(f"prefetch: FAILED — {exc}", file=sys.stderr)
        return 1

    write_prefetch_status(
        forge_data_dir,
        {
            "status": "ok",
            "target_gb": target_gb,
            "bytes": total,
            "en_fraction": en_fraction,
            "staged_new": staged,
        },
    )
    print(f"prefetch: DONE — {total / 1024**3:.1f}GB staged under {forge_data_dir}")
    return 0


def _hf_fetch(repo: str):
    """Default fetcher: copy an HF-hub dataset file to a local destination."""
    import shutil

    from huggingface_hub import hf_hub_download

    def fetch(repo_path: str, dest: Path) -> None:
        cached = hf_hub_download(  # nosec B615 — corpus revision pinned by the data config
            repo_id=repo, repo_type="dataset", filename=repo_path
        )
        shutil.copyfile(cached, dest)

    return fetch


# How many times to re-invoke ``curl -C -`` (resuming) for a single part
# before giving up — each attempt continues from the last byte, so this
# bounds total drops absorbed, not wasted full re-downloads.
_CURL_RESUME_ATTEMPTS = 40


def _curl_endpoint() -> str:
    import os

    return os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")


def _curl_fetch(repo: str, *, endpoint: str | None = None):
    """Fetcher that streams each part with ``curl`` instead of the hf_hub client.

    Needed on clusters whose egress goes through a *whitelist* proxy that
    allows the HF mirror (metadata) but blocks the Xet/LFS CDN where the
    bytes live (e.g. shandong: hf-mirror.com is whitelisted, but the parquet
    redirects to cas-bridge.xethub.hf.co which the proxy drops). Running the
    prefetch with http_proxy/https_proxy unset reaches both directly, and
    curl follows the resolve→CDN redirect reliably where the hf_hub httpx /
    Xet client stalls. ``HF_ENDPOINT`` selects the mirror.
    """
    import subprocess

    base = endpoint or _curl_endpoint()

    def fetch(repo_path: str, dest: Path) -> None:
        url = f"{base}/datasets/{repo}/resolve/main/{repo_path}"
        # Multi-GB parts (~1.3GB each) drop mid-transfer on this CDN
        # (curl rc=18 partial) far more often than curl's own --retry can
        # absorb. Loop ``curl -C -`` so each attempt RESUMES the .part from
        # where the last drop left off (the CDN supports range requests), so
        # a flaky link still converges. Download to .part, then atomically
        # rename, so a failed transfer never leaves a truncated file that
        # download_until would mistake for a complete part.
        tmp = Path(str(dest) + ".part")
        rc = -1
        for _ in range(_CURL_RESUME_ATTEMPTS):
            rc = subprocess.run(
                [
                    "curl",
                    "-fsSL",
                    "-C",
                    "-",
                    "--retry",
                    "3",
                    "--retry-all-errors",
                    "--retry-delay",
                    "3",
                    "--max-time",
                    "3600",
                    "-o",
                    str(tmp),
                    url,
                ]
            ).returncode
            if rc == 0 and tmp.is_file():
                tmp.replace(dest)
                return
        raise RuntimeError(f"curl rc={rc} for {url} after {_CURL_RESUME_ATTEMPTS} resume attempts")

    return fetch


def download_until(
    repo: str,
    forge_data_dir: Path,
    order: list[tuple[str, str]],
    target_bytes: int,
    *,
    fetch=None,
    on_progress=None,
) -> tuple[int, int]:
    """Stage parts from ``order`` into ``forge_data_dir`` until ``target_bytes``.

    Skips already-present files (their bytes still count toward the target,
    so a resumed prefetch stops at the same place). ``fetch(repo_path, dest)``
    is the per-file copy — defaults to HF-hub, injectable for tests. Returns
    ``(total_bytes_on_disk, newly_staged_count)``.
    """
    forge_data_dir = Path(forge_data_dir)
    if fetch is None:
        fetch = _hf_fetch(repo)
    total = 0
    staged = 0
    for local_rel, repo_path in order:
        if total >= target_bytes:
            break
        target = forge_data_dir / local_rel
        if target.is_file():
            total += target.stat().st_size
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        fetch(repo_path, target)
        total += target.stat().st_size
        staged += 1
        if on_progress is not None:
            on_progress(staged, total)
    return total, staged


if __name__ == "__main__":
    raise SystemExit(main())
