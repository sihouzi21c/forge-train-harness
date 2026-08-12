#!/usr/bin/env bash
# remote_bootstrap.sh — idempotent devspace env setup for the agent loop.
#
# Loop e180edc7 rotated through 5 fresh devspaces and each rotation
# made the agent re-derive the same incantations: clear the corp
# HTTP_PROXY (pip can't reach the campus mirror through it), point
# pip at the Tsinghua mirror, install harness with
# --no-build-isolation (PEP 517's fresh build venv routinely fails
# because torch + setuptools are only present in the devspace's
# system Python), and hand-copy modelbest_sdk into site-packages
# (the SDK has no pypi release). This script codifies that recipe
# and stamps ~/.forge_train/env_ready so a second invocation is a
# near-zero-cost no-op.
#
# Usage:
#   bash remote_bootstrap.sh <harness_path>      # do the work
#   bash remote_bootstrap.sh --check             # exit 0 if ready, 1 otherwise
#
# Caller (agent-loop.sh) is expected to run --check first and skip
# the full bootstrap when it returns 0.
set -euo pipefail

# Bump on backwards-incompatible bootstrap changes so older markers
# stop satisfying --check and the next agent re-runs the full path.
BOOTSTRAP_VERSION=1

MARKER="$HOME/.forge_train/env_ready"
MODELBEST_SDK_SRC="/user/lizhen/tmp/modelbest_sdk_031_unzip"
PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

_check_ready() {
  [ -f "$MARKER" ] && grep -q "^version=${BOOTSTRAP_VERSION}$" "$MARKER"
}

if [ "${1:-}" = "--check" ]; then
  _check_ready && exit 0 || exit 1
fi

HARNESS_PATH="${1:?usage: remote_bootstrap.sh <harness_path> | --check}"

if _check_ready; then
  echo "[remote-bootstrap] already ready (version=${BOOTSTRAP_VERSION}); skipping"
  exit 0
fi

echo "[remote-bootstrap] starting bootstrap (version=${BOOTSTRAP_VERSION})"

# 1. Devspaces inherit HTTP_PROXY from the corp env; pip then can't
#    reach the campus mirror. Clear it for this shell only.
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

# 2. modelbest_sdk has no pypi release. The canonical unzipped tree
#    lives at the path below; copying it into the active venv's
#    site-packages is what makes ``import modelbest_sdk`` resolve.
if [ ! -d "$MODELBEST_SDK_SRC" ]; then
  echo "[remote-bootstrap] FATAL: missing $MODELBEST_SDK_SRC on this devspace" >&2
  exit 2
fi

SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
if [ ! -d "$SITE_PACKAGES/modelbest_sdk" ]; then
  cp -r "$MODELBEST_SDK_SRC" "$SITE_PACKAGES/modelbest_sdk"
fi

# 3. Install harness editable. --no-build-isolation reuses the
#    devspace's pre-installed torch / setuptools instead of fetching
#    them into a fresh build venv (which routinely fails behind the
#    mirror with torch's massive download).
pip install --no-build-isolation --index-url "$PIP_INDEX_URL" -e "$HARNESS_PATH"

# 4. Stamp the marker last so a partial bootstrap (interrupted, OOM,
#    network failure) leaves _check_ready returning false and the
#    next invocation re-runs.
mkdir -p "$(dirname "$MARKER")"
{
  echo "version=${BOOTSTRAP_VERSION}"
  echo "stamped_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "harness_path=${HARNESS_PATH}"
} > "$MARKER"

echo "[remote-bootstrap] done (marker stamped at $MARKER)"
