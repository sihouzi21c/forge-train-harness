"""External-resource manifest: declared, content-addressed dependencies.

The manifest replaces the implicit "agent discovers tokenizer / checkpoint /
megatron paths at run time" pattern that burned ~12 min per loop in
26737772. A workspace declares every external artifact it needs in
``harness/external_resources.toml`` (file, git-checkout, tree); the
:func:`provision` step walks each entry's declared source list (local
fast paths first, network fallbacks gated behind
``FORGE_ALLOW_NETWORK_RESOURCES``) and symlinks the first verifying source
to a canonical relative path under ``<workspace>/.resources/``. The
canonical relpath is the only path :mod:`harness.config_runtime`
resolvers read at runtime — no auto-download, no git clone.

Content-addressing is *opt-in*: an entry may declare ``sha256`` (file
digest, ``git rev-parse HEAD`` for git-checkouts, or sorted
(relpath, sha256) tree hash for trees) to make drift detectable — a
swapped artifact then fails ``provision`` instead of silently feeding
the training loop wrong bytes. Omitting ``sha256`` (or setting it to
the empty string) opts out: ``provision`` takes the first *existing*
source of the right kind (file vs directory) and symlinks it, with no
content check. This keeps the manifest authorable offline for
artifacts whose truth bytes live on a remote box.

Manifest schema (``harness/external_resources.toml``)::

    [meta]
    version = 1

    [[entry]]
    name = "tokenizer.minicpm4_0_5b"
    kind = "file"  # or "git-checkout" or "tree"
    sha256 = "<sha256-hex>"  # optional; "" / omitted = existence+kind check only
    canonical_relpath = ".resources/tokenizer/minicpm4-0.5b/tokenizer.model"
    sources = [
      "file:///abs/path/to/tokenizer.model",
      "file://${HOME}/.forge_train/.artifacts/.../tokenizer.model",
      "hf://openbmb/MiniCPM4-0.5B#tokenizer.model",  # network fallback
    ]

An empty manifest (``version = 1`` with no ``[[entry]]`` tables) is a
valid opt-out: ``provision`` is a no-op and the workspace contract's
:func:`assert_provisioned` check passes trivially. Per-deployment
artifacts are populated after the user runs ``harness resources verify``
on the box that owns the truth bytes.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess  # nosec B404 — invoked only for `git rev-parse`
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness._compat import tomllib
from harness.config_runtime import repo_root

__all__ = [
    "Resource",
    "ResourceProvisionError",
    "load_manifest",
    "manifest_path",
    "provision",
    "verify",
]

_MANIFEST_FILENAME = "external_resources.toml"
_NETWORK_GATE_ENV = "FORGE_ALLOW_NETWORK_RESOURCES"
_NETWORK_SCHEMES = ("hf://", "git+ssh://", "git+https://", "https://", "http://")


class ResourceProvisionError(RuntimeError):
    """A manifest entry could not be provisioned from any declared source."""


@dataclass(frozen=True, slots=True)
class Resource:
    """A single manifest entry."""

    name: str
    kind: str
    sha256: str
    canonical_relpath: str
    sources: tuple[str, ...]


def manifest_path(workspace_root: Path | None = None) -> Path:
    """Return the absolute path of the manifest for *workspace_root*.

    The manifest is a workspace-level config file living at the workspace
    (``repo_root()``) top level — a sibling of ``config/`` / ``evals/`` /
    ``workload/`` — NOT inside the inner ``harness/`` Python package. (An
    earlier version prepended ``harness/`` and so never found the committed
    top-level file, silently making every ``load_manifest`` return ``[]``.)
    """
    root = workspace_root if workspace_root is not None else repo_root()
    return (root / _MANIFEST_FILENAME).resolve()


def load_manifest(workspace_root: Path | None = None) -> list[Resource]:
    """Parse the manifest and return its entries.

    A missing manifest file returns an empty list — the workspace simply
    hasn't declared any external resources yet. A malformed manifest
    raises ``ValueError`` with the offending field.
    """
    path = manifest_path(workspace_root)
    if not path.exists():
        return []
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: manifest must be a TOML table")
    version = raw.get("meta", {}).get("version")
    if version != 1:
        raise ValueError(f"{path}: [meta].version must be 1 (got {version!r})")
    entries = raw.get("entry", [])
    if not isinstance(entries, list):
        raise ValueError(f"{path}: [[entry]] must be an array of tables")
    return [_parse_entry(path, item) for item in entries]


def _parse_entry(manifest: Path, raw: Any) -> Resource:
    if not isinstance(raw, dict):
        raise ValueError(f"{manifest}: each [[entry]] must be a table")
    for key in ("name", "kind", "canonical_relpath", "sources"):
        if key not in raw:
            raise ValueError(f"{manifest}: [[entry]] missing required key '{key}'")
    if raw["kind"] not in {"file", "git-checkout", "tree"}:
        raise ValueError(
            f"{manifest}: [[entry]] '{raw['name']}'.kind must be "
            f"'file' | 'git-checkout' | 'tree' (got {raw['kind']!r})"
        )
    if not isinstance(raw["sources"], list) or not raw["sources"]:
        raise ValueError(f"{manifest}: [[entry]] '{raw['name']}'.sources must be a non-empty list")
    return Resource(
        name=str(raw["name"]),
        kind=str(raw["kind"]),
        sha256=str(raw.get("sha256", "")),
        canonical_relpath=str(raw["canonical_relpath"]),
        sources=tuple(str(s) for s in raw["sources"]),
    )


def verify(resource: Resource, target: Path) -> tuple[bool, str]:
    """Verify *target* against the resource declaration.

    With a declared ``sha256`` the target's content hash must match. With
    ``sha256`` empty (content-addressing opted out) only existence and
    kind (file vs directory) are checked. Returns ``(ok, diagnostic)``;
    ``diagnostic`` is empty on success and a one-line human-readable
    explanation otherwise.
    """
    if not target.exists():
        return False, f"path does not exist: {target}"
    if not resource.sha256:
        return _verify_kind(resource, target)
    if resource.kind == "file":
        return _verify_file(resource, target)
    if resource.kind == "git-checkout":
        return _verify_git(resource, target)
    if resource.kind == "tree":
        return _verify_tree(resource, target)
    return False, f"unknown kind {resource.kind!r}"


def _verify_kind(resource: Resource, target: Path) -> tuple[bool, str]:
    """Existence-passed; check only that the target's kind is as declared."""
    if resource.kind == "file":
        if not target.is_file():
            return False, f"expected a file, got {target}"
        return True, ""
    if resource.kind in {"git-checkout", "tree"}:
        if not target.is_dir():
            return False, f"expected a directory, got {target}"
        return True, ""
    return False, f"unknown kind {resource.kind!r}"


def _verify_file(resource: Resource, target: Path) -> tuple[bool, str]:
    if not target.is_file():
        return False, f"expected a file, got {target}"
    digest = _sha256_file(target)
    if digest != resource.sha256:
        return False, f"sha256 mismatch: got {digest} expected {resource.sha256}"
    return True, ""


def _verify_git(resource: Resource, target: Path) -> tuple[bool, str]:
    if not target.is_dir():
        return False, f"expected a directory, got {target}"
    completed = subprocess.run(  # nosec B603 B607 — invoked with literal argv
        ["git", "-C", str(target), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return False, f"git rev-parse failed: {completed.stderr.strip()}"
    head = completed.stdout.strip()
    if head != resource.sha256:
        return False, f"HEAD sha mismatch: got {head} expected {resource.sha256}"
    return True, ""


def _verify_tree(resource: Resource, target: Path) -> tuple[bool, str]:
    if not target.is_dir():
        return False, f"expected a directory, got {target}"
    digest = _sha256_tree(target)
    if digest != resource.sha256:
        return False, f"tree sha mismatch: got {digest} expected {resource.sha256}"
    return True, ""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_tree(root: Path) -> str:
    """Stable tree hash: sorted ``(relpath, sha256)`` over every regular file."""
    rows: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        rows.append(f"{rel}:{_sha256_file(path)}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def provision(workspace_root: Path | None = None) -> list[Resource]:
    """Provision every manifest entry into the workspace.

    For each entry, the first source that passes :func:`verify` is
    symlinked (or copied for cross-volume git checkouts) to
    ``<workspace>/<canonical_relpath>``. Raises
    :exc:`ResourceProvisionError` if no source verifies — the error
    message enumerates every failed source and its diagnostic so the
    operator knows exactly what to fix.

    Network-scheme sources (``hf://``, ``git+ssh://``, …) are skipped
    unless ``FORGE_ALLOW_NETWORK_RESOURCES=1`` is set. This preserves
    the harness's offline-by-default posture: a workspace cannot
    silently pull bytes from the public internet without the operator
    flipping that flag on.
    """
    root = workspace_root if workspace_root is not None else repo_root()
    manifest = load_manifest(root)
    if not manifest:
        return []
    network_allowed = os.environ.get(_NETWORK_GATE_ENV) == "1"
    for resource in manifest:
        _provision_one(root, resource, network_allowed=network_allowed)
    return manifest


def _provision_one(workspace_root: Path, resource: Resource, *, network_allowed: bool) -> None:
    canonical = workspace_root / resource.canonical_relpath
    if canonical.exists():
        ok, diag = verify(resource, canonical)
        if ok:
            return
        # Existing target is corrupt; re-provision so a swapped/stale
        # artifact cannot poison the next loop silently.
        if canonical.is_symlink() or canonical.is_file():
            canonical.unlink()
        else:
            shutil.rmtree(canonical)

    failures: list[str] = []
    for source in resource.sources:
        if any(source.startswith(scheme) for scheme in _NETWORK_SCHEMES) and not network_allowed:
            failures.append(
                f"{source}: skipped (network fetch disabled; set {_NETWORK_GATE_ENV}=1 to enable)"
            )
            continue
        if not source.startswith("file://"):
            failures.append(f"{source}: unsupported scheme")
            continue
        source_path = Path(os.path.expandvars(source[len("file://") :]))
        if not source_path.exists():
            failures.append(f"{source}: path not found")
            continue
        ok, diag = verify(resource, source_path)
        if not ok:
            failures.append(f"{source}: {diag}")
            continue
        canonical.parent.mkdir(parents=True, exist_ok=True)
        if canonical.is_symlink() or canonical.exists():
            canonical.unlink()
        canonical.symlink_to(source_path.resolve())
        return

    raise ResourceProvisionError(
        f"resource '{resource.name}' could not be provisioned. "
        f"Tried {len(resource.sources)} source(s):\n  " + "\n  ".join(failures)
    )
