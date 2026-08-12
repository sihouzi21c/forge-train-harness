#!/usr/bin/env python3
# ============================================================================
# _pins.py — extract version pins from a release Dockerfile (the SSOT).
#
# Phase 2 torch-ref scripts call this INSTEAD of sourcing a versions.*.env
# manifest. The open-source release image is the single source of truth for
# every package pin; there are no derived env files:
#
#     ngc2501   -> repo-root  Dockerfile                  (FROM nvcr.io/...:25.01)
#
# We parse two things out of the Dockerfile:
#
#   1. the Layer-8 `EXACT = {...}` dict  -> the exact pip pins the build itself
#      asserts (numpy, sentencepiece, flash_attn, triton, ...). This dict is
#      already the build's own pin authority, so reusing it here means zero
#      duplication: the verify gate checks exactly what the image installs.
#
#   2. the torch self-check lines        -> TORCH_PIN + TORCH_PIN_MODE + CUDA_PIN
#      (ngc2501 only floors torch `>= 2.3` + CUDA `12.` because the digest-pinned
#      NGC base owns the exact heavy-stack labels).
#
# Output: bash-sourceable `KEY="VALUE"` lines for the names verify_env_torch.sh
# / gate_sweep_torch.sh consume. A package the verify gate checks but the
# Dockerfile does NOT pin (base-image-owned: flash_attn/triton,
# transformer_engine) is emitted with an EMPTY value, which the shell side
# treats as "present is enough" (no version compare) rather than falling back to
# a stale inline default.
#
# Usage:
#   python3 _pins.py --image ngc2501 [--env-dir DIR]
#
#   --env-dir   directory holding this script (harness/env). Dockerfiles are
#               resolved relative to it so the extractor is layout-independent
#               (works after `harness sync push` lands the tree on a remote).
#               Default: the directory of this file.
#
# Exit codes: 0 = pins printed, 2 = usage / Dockerfile-not-found.
# ============================================================================
import argparse
import ast
import os
import re
import sys

# image key -> Dockerfile location, as a list of candidate paths relative to
# the harness/env directory (first existing wins).
DOCKERFILE_CANDIDATES = {
    "ngc2501": ["../Dockerfile", "../../Dockerfile"],
}

# pip distribution name (as written in the Dockerfile EXACT dict) -> the shell
# variable name verify_env_torch.sh / gate_sweep_torch.sh read. Only packages
# the verify gate actually version-checks are mapped; the rest of the EXACT
# dict is ignored here (still build-enforced inside the image).
NAME_TO_KEY = {
    "numpy": "NUMPY_PIN",
    "triton": "TRITON_PIN",
    "flash_attn": "FLASH_ATTN_PIN",
    "sentencepiece": "SENTENCEPIECE_PIN",
    "pandas": "PANDAS_PIN",
    "pyarrow": "PYARROW_PIN",
    "nvidia-cutlass-dsl": "NVIDIA_CUTLASS_DSL_PIN",
    "cuda-bindings": "CUDA_BINDINGS_PIN",
    "modelbest-sdk": "MODELBEST_SDK_PIN",
    "transformer_engine": "TRANSFORMER_ENGINE_PIN",
}

# every pin KEY the shell consumes, so we always emit it (empty if unpinned) and
# the shell never silently inherits a stale inline default.
EMITTED_KEYS = [
    *dict.fromkeys(NAME_TO_KEY.values()),
    "TORCH_PIN",
    "TORCH_PIN_MODE",
    "CUDA_PIN",
]


def resolve_dockerfile(image, env_dir):
    for rel in DOCKERFILE_CANDIDATES[image]:
        p = os.path.normpath(os.path.join(env_dir, rel))
        if os.path.isfile(p):
            return p
    return None


def extract_exact(text):
    """The Layer-8 `EXACT = { ... }` dict. Values are flat string literals
    (no nested braces), so a non-greedy brace span + literal_eval is safe."""
    m = re.search(r"EXACT\s*=\s*(\{[^}]*\})", text, re.S)
    if not m:
        return {}
    try:
        d = ast.literal_eval(m.group(1))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def extract_torch(text):
    """Return (torch_pin, torch_mode, cuda_prefix) from the Dockerfile's torch
    self-check. Two shapes are recognized:
        exact/base : torch_base != "2.6.0"            -> ("2.6.0", "base")
        floor      : (tmaj, tmin) < (2, 3)            -> ("2.3",   "floor")
    CUDA prefix comes from the torch.version.cuda startswith("12...") check."""
    pin, mode = "", ""
    m = re.search(r'torch_base\s*!=\s*"([0-9][0-9A-Za-z.\-+_]*)"', text)
    if m:
        pin, mode = m.group(1), "base"
    else:
        m = re.search(r"<\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)
        if m:
            pin, mode = f"{m.group(1)}.{m.group(2)}", "floor"

    cuda = ""
    mc = re.search(r'startswith\("(12[0-9.]*)"\)', text)
    if mc:
        cuda = mc.group(1).rstrip(".")
    return pin, mode, cuda


def sh_quote(v):
    # values are version strings (no quotes/backslashes); double-quote and
    # escape defensively anyway.
    escaped = v.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, choices=sorted(DOCKERFILE_CANDIDATES))
    ap.add_argument("--env-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    df = resolve_dockerfile(args.image, args.env_dir)
    if df is None:
        sys.stderr.write(
            f"error: Dockerfile for image '{args.image}' not found under "
            f"{args.env_dir} (tried {DOCKERFILE_CANDIDATES[args.image]})\n"
        )
        return 2

    with open(df, encoding="utf-8") as fh:
        text = fh.read()

    exact = extract_exact(text)
    torch_pin, torch_mode, cuda_pin = extract_torch(text)

    values = {k: "" for k in EMITTED_KEYS}
    for name, key in NAME_TO_KEY.items():
        if name in exact:
            values[key] = str(exact[name])
    values["TORCH_PIN"] = torch_pin
    values["TORCH_PIN_MODE"] = torch_mode
    values["CUDA_PIN"] = cuda_pin

    print(f"# pins parsed from {df} (image={args.image}) — SSOT, do not hand-edit")
    print(f"IMAGE_KEY={sh_quote(args.image)}")
    print(f"FORGE_DOCKERFILE={sh_quote(df)}")
    for k in EMITTED_KEYS:
        print(f"{k}={sh_quote(values[k])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
