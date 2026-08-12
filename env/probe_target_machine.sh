#!/usr/bin/env bash
# ============================================================================
# probe_target_machine.sh — Phase 2 Script 1 (extended).
#
# Decides whether the machine(s) involved in a MiniCPM4-0.5B autoloop run
# CAN host the reproduction. Two execution layouts, one script:
#
#   Mode 1 (Local+Local)  : agent + training on the same Linux H100 box
#       → probe_target_machine.sh --role both
#
#   Mode 2 (Local+SSH)    : agent on a dev box, training on a remote H100 box
#       → probe_target_machine.sh --role dev --ssh <alias> \
#                                 --remote-workdir <dir>
#         (on remote, via rsync push)
#         probe_target_machine.sh --role exec
#
# Scope: ONLY hard machine constraints that the agent cannot fix from
# user-space (driver, hardware, kernel, network). Everything pip-installable
# (torch, flash_attn, modelbest_sdk, ...) is verified by Script 2
# (verify_env_torch.sh), not here.
#
# Host-capability floors (VRAM minimum, GPU compute_cap, per-image driver
# floors) are inline constants below — host gates, not image package pins, so
# there is no versions.*.env to source.
#
# Exit codes:
#   0 — installable        (every blocking check passed)
#   1 — blocked            (at least one non-auto-installable check failed)
#   2 — needs_install      (only A1a/A1b auto-installable failures)
#   3 — usage error
# ============================================================================

set -o pipefail

# ---------------------------------------------------------------------------
# Host-capability floors are inline (below): GPU compute_cap, VRAM, and the
# per-image driver floors are host gates, not image package pins, so they do not
# live in the release Dockerfiles. There are no versions.*.env files — package
# pins are owned by the two Dockerfiles and checked by verify_env_torch.sh.
# ---------------------------------------------------------------------------
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MIN_GPU_VRAM_MIB="${GPU_VRAM_MIB_MIN:-80000}"
MIN_GLIBC_PROBE="2.28"   # Glibc floor for manylinux_2_28 PyTorch wheels.
REQUIRED_COMPUTE_CAP="${GPU_COMPUTE_CAP_PIN:-9.0}"

# ---------------------------------------------------------------------------
# Single release image: forge_train (CUDA 12.8, NGC-25.01 based). The 12.4
# image was retired — there is exactly one image now.
#
# The 570.26 driver floor is the CUDA 12.8 paired/min driver (NGC 25.01:
# "requires Driver release 570 or later"). It is treated as ADVISORY, not
# blocking: a host whose driver is below it is WARNED ("does not meet
# reproduction conditions") but still allowed to try the image at its own
# risk. The hardware floors (Hopper sm_90, 80GB VRAM, >= 2 GPUs) stay HARD
# gates. The advisory/hard split lives in Check 4 + the IMGCAP block below.
#
# Parallel indexed arrays (not assoc) keep this bash-3.2-safe, matching the
# CHK_* style; with one image they each hold a single element.
# ---------------------------------------------------------------------------
IMG_NAMES=(ngc2501)
IMG_FLOOR=(570.26)
IMG_CMAX=(12.8)
IMG_CAP=()                          # 1|0 (single image; by index)
IMG_REASON=()                       # human reason (by index)
RECOMMENDED_IMAGE=""
HOST_IMAGE_CLASS="unknown"

# The driver floor is the single image's CUDA 12.8 floor — but it is ADVISORY
# (Check 4 records a warn, never a hard fail) so a sub-12.8 driver does not
# block the verdict.
MIN_DRIVER="${IMG_FLOOR[0]}"

# Disk thresholds depend on role.
DISK_DEV_GB=10
DISK_EXEC_GB=50

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
ROLE="both"
MODE="minimal"
WORKDIR="$PWD"
SSH_ALIAS=""
REMOTE_WORKDIR=""
JSON=0

usage() {
    cat >&2 <<EOF
Usage: $(basename "$0") --role {dev|exec|both} [options]

  --role dev|exec|both    which side of the loop this host plays
                          dev  = agent host (drives loop, runs ssh/rsync)
                          exec = training host (H100 hardware lives here)
                          both = single machine plays both (Local+Local)
  --mode minimal|full     2x H100 (default) | 8x H100  (exec/both only)
  --workdir DIR           Directory to probe (default: \$PWD)
  --ssh ALIAS             SSH alias to probe (dev role; enables A2/A3)
  --remote-workdir DIR    Remote workdir to test under --ssh (default:
                          ~/web_harness)
  --json                  Emit machine-readable JSON (no colors)

Exit codes:
  0 = installable
  1 = blocked
  2 = needs_install   (only auto-installable failures — see auto_install_hints)
  3 = usage error
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --role)
            ROLE="$2"; shift 2
            case "$ROLE" in
                dev|exec|both) ;;
                *) echo "error: --role must be dev|exec|both" >&2; exit 3 ;;
            esac
            ;;
        --mode)
            MODE="$2"; shift 2
            if [[ "$MODE" != "minimal" && "$MODE" != "full" ]]; then
                echo "error: --mode must be 'minimal' or 'full'" >&2
                exit 3
            fi
            ;;
        --workdir)        WORKDIR="$2";        shift 2 ;;
        --ssh)            SSH_ALIAS="$2";      shift 2 ;;
        --remote-workdir) REMOTE_WORKDIR="$2"; shift 2 ;;
        --json)           JSON=1; shift ;;
        -h|--help)        usage; exit 0 ;;
        *) echo "error: unknown argument '$1'" >&2; usage; exit 3 ;;
    esac
done

# Validate role/ssh combinations.
if [[ -n "$SSH_ALIAS" && "$ROLE" == "exec" ]]; then
    echo "error: --ssh is meaningful only with --role dev (or unused with both)" >&2
    exit 3
fi
if [[ -n "$SSH_ALIAS" && "$ROLE" == "both" ]]; then
    echo "error: --ssh + --role both is contradictory (both = single machine)" >&2
    exit 3
fi
if [[ -n "$SSH_ALIAS" && -z "$REMOTE_WORKDIR" ]]; then
    # shellcheck disable=SC2088  # literal ~ is expanded by the REMOTE shell
    REMOTE_WORKDIR="~/web_harness"
fi

# Mode → GPU count threshold (exec/both only). Reproduction needs >= 2 H100
# (multistep runs DP=2), so the minimal floor is 2 (not 1); full keeps the 8x floor.
if [[ "$MODE" == "full" ]]; then
    REQUIRED_GPUS=8
else
    REQUIRED_GPUS=2
fi

# Resolve WORKDIR absolute path if it exists; preserve raw if not.
WORKDIR="$(cd "$WORKDIR" 2>/dev/null && pwd || echo "$WORKDIR")"

# Role booleans.
ROLE_DEV=0
ROLE_EXEC=0
case "$ROLE" in
    dev)  ROLE_DEV=1 ;;
    exec) ROLE_EXEC=1 ;;
    both) ROLE_DEV=1; ROLE_EXEC=1 ;;
esac

# Disk threshold: exec/both uses 50GB; pure dev uses 10GB.
if (( ROLE_EXEC )); then
    MIN_DISK_GB=$DISK_EXEC_GB
else
    MIN_DISK_GB=$DISK_DEV_GB
fi

# ---------------------------------------------------------------------------
# Colors (disabled for --json or non-tty)
# ---------------------------------------------------------------------------
if [[ $JSON -eq 0 && -t 1 ]]; then
    C_OK=$'\033[32m'; C_FAIL=$'\033[31m'; C_WARN=$'\033[33m'
    C_DIM=$'\033[2m'; C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'
else
    C_OK=""; C_FAIL=""; C_WARN=""; C_DIM=""; C_RESET=""; C_BOLD=""
fi

# ---------------------------------------------------------------------------
# Probe cleanup
# ---------------------------------------------------------------------------
PROBE_PATHS=()
cleanup() {
    local p
    for p in "${PROBE_PATHS[@]}"; do
        [[ -n "$p" ]] && rm -f "$p" 2>/dev/null
    done
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Helpers (unchanged from prior round)
# ---------------------------------------------------------------------------

# ver_ge A B  ->  exit 0 if A >= B (segment-wise numeric compare on '.')
ver_ge() {
    local a="$1" b="$2"
    local -a A B
    IFS='.' read -ra A <<<"$a"
    IFS='.' read -ra B <<<"$b"
    local n=${#A[@]}; (( ${#B[@]} > n )) && n=${#B[@]}
    local i ai bi
    for ((i=0; i<n; i++)); do
        ai=${A[i]:-0}; bi=${B[i]:-0}
        ai=${ai%%[!0-9]*}; bi=${bi%%[!0-9]*}
        ai=${ai:-0}; bi=${bi:-0}
        if (( 10#$ai > 10#$bi )); then return 0; fi
        if (( 10#$ai < 10#$bi )); then return 1; fi
    done
    return 0
}

json_escape() {
    local s="$1"
    s=${s//\\/\\\\}
    s=${s//\"/\\\"}
    s=${s//$'\n'/\\n}
    s=${s//$'\r'/\\r}
    s=${s//$'\t'/\\t}
    printf '%s' "$s"
}

# Parallel arrays of check results.
CHK_ID=()
CHK_NAME=()
CHK_STATUS=()      # pass | fail | warn
CHK_DETAIL=()
CHK_HINT=()
CHK_AUTOINSTALL=() # 1 if this failure is auto-installable by the agent

record() {
    # record ID NAME STATUS DETAIL [HINT] [AUTO_INSTALL]
    CHK_ID+=("$1")
    CHK_NAME+=("$2")
    CHK_STATUS+=("$3")
    CHK_DETAIL+=("$4")
    CHK_HINT+=("${5:-}")
    CHK_AUTOINSTALL+=("${6:-0}")
}

probe_url() {
    local url="$1"
    command -v curl >/dev/null 2>&1 || return 2
    curl -sSfL --max-time 5 -o /dev/null -I "$url" 2>/dev/null
}

probe_write_exec() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then echo "missing"; return; fi
    local touch_f="${dir}/.forge_probe_touch.$$"
    local exec_f="${dir}/.forge_probe_exec.$$"
    PROBE_PATHS+=("$touch_f" "$exec_f")
    if ! ( : > "$touch_f" ) 2>/dev/null; then echo "ro"; return; fi
    rm -f "$touch_f" 2>/dev/null
    cat > "$exec_f" <<'__EOF__' 2>/dev/null
#!/bin/sh
exit 0
__EOF__
    if [[ ! -s "$exec_f" ]]; then echo "ro"; return; fi
    chmod +x "$exec_f" 2>/dev/null || { echo "noexec"; return; }
    if "$exec_f" 2>/dev/null; then
        rm -f "$exec_f" 2>/dev/null
        echo "ok"
    else
        rm -f "$exec_f" 2>/dev/null
        echo "noexec"
    fi
}

# probe_ssh ALIAS -> 0 if ssh succeeds non-interactively within
# SSH_PROBE_TIMEOUT seconds (default 15; teleport-proxied devspaces
# routinely take 5–10s on the first connection, so 5s is too tight).
probe_ssh() {
    local alias="$1"
    ssh -o BatchMode=yes \
        -o ConnectTimeout="${SSH_PROBE_TIMEOUT:-15}" \
        -o StrictHostKeyChecking=accept-new \
        -o ServerAliveInterval=0 \
        "$alias" 'exit 0' 2>/dev/null
}

# probe_remote_workdir ALIAS DIR -> echoes "ok" | "missing_perm" | "ssh_fail"
probe_remote_workdir() {
    local alias="$1" dir="$2"
    # ssh runs a shell expansion so ~ resolves on the remote.
    local out
    out=$(ssh -o BatchMode=yes -o ConnectTimeout="${SSH_PROBE_TIMEOUT:-15}" \
            -o StrictHostKeyChecking=accept-new \
            "$alias" "mkdir -p $dir 2>/dev/null && test -w $dir && echo OK" 2>/dev/null)
    if [[ "$out" == "OK" ]]; then
        echo "ok"
    elif [[ -z "$out" ]]; then
        echo "ssh_fail"
    else
        echo "missing_perm"
    fi
}

# ===========================================================================
# Checks
# ===========================================================================

# ---------------------------------------------------------------------------
# EXEC-only: GPU + driver checks (1, 2, 3, 4, 7)
# ---------------------------------------------------------------------------
if (( ROLE_EXEC )); then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        record 1 gpu_model    fail "nvidia-smi not found in PATH" \
            "Install NVIDIA driver (admin required) or run on a GPU host"
        record 2 gpu_vram     fail "nvidia-smi not found" ""
        record 3 gpu_count    fail "nvidia-smi not found" ""
        record 7 gpu_access   fail "nvidia-smi not found" ""
    else
        if ! nvidia-smi -q >/dev/null 2>&1; then
            record 7 gpu_access fail "nvidia-smi present but failed to query GPUs" \
                "Driver/device problem — likely needs admin attention"
            record 1 gpu_model  fail "cannot enumerate GPUs (nvidia-smi -q failed)" ""
            record 2 gpu_vram   fail "cannot enumerate GPUs" ""
            record 3 gpu_count  fail "cannot enumerate GPUs" ""
        else
            record 7 gpu_access pass "nvidia-smi -q succeeded" ""

            gpu_lines="$(nvidia-smi --query-gpu=name,compute_cap,memory.total \
                             --format=csv,noheader 2>/dev/null || true)"

            if [[ -z "$gpu_lines" ]]; then
                record 1 gpu_model fail "no GPUs reported by nvidia-smi" ""
                record 2 gpu_vram  fail "no GPUs reported" ""
                record 3 gpu_count fail "0 GPUs detected" \
                    "Run on a host with H100 GPUs"
            else
                n_gpus=0; bad_model=""; bad_vram=""
                sample_name=""; sample_cc=""; min_vram=999999
                while IFS= read -r line; do
                    [[ -z "$line" ]] && continue
                    n_gpus=$((n_gpus + 1))
                    name="${line%%,*}"
                    rest="${line#*, }"
                    cc="${rest%%,*}"
                    vram_str="${rest#*, }"
                    vram_mib="${vram_str%% *}"
                    vram_mib="${vram_mib//[!0-9]/}"
                    [[ -z "$vram_mib" ]] && vram_mib=0
                    sample_name="$name"; sample_cc="$cc"
                    (( vram_mib < min_vram )) && min_vram=$vram_mib
                    [[ "$cc" != "$REQUIRED_COMPUTE_CAP" ]] && bad_model="$name (cc=$cc)"
                    (( vram_mib < MIN_GPU_VRAM_MIB )) && bad_vram="$name has ${vram_mib} MiB"
                done <<<"$gpu_lines"

                if [[ -z "$bad_model" ]]; then
                    record 1 gpu_model pass \
                        "${n_gpus}x ${sample_name} (compute_cap=${sample_cc})" ""
                else
                    record 1 gpu_model fail \
                        "non-H100 GPU detected: ${bad_model} (need compute_cap=${REQUIRED_COMPUTE_CAP})" \
                        "Run on an H100 host (sm_90); other archs are not supported"
                fi

                if [[ -z "$bad_vram" ]]; then
                    record 2 gpu_vram pass \
                        "all GPUs >= ${MIN_GPU_VRAM_MIB} MiB (min seen: ${min_vram} MiB)" ""
                else
                    record 2 gpu_vram fail \
                        "${bad_vram} < ${MIN_GPU_VRAM_MIB} MiB required" \
                        "Use 80GB H100 cards (HBM3); 40GB variants are insufficient"
                fi

                if (( n_gpus >= REQUIRED_GPUS )); then
                    record 3 gpu_count pass \
                        "${n_gpus} GPU(s) detected (need >= ${REQUIRED_GPUS} for mode=${MODE})" ""
                else
                    record 3 gpu_count fail \
                        "${n_gpus} GPU(s) < ${REQUIRED_GPUS} required for mode=${MODE}" \
                        "Reproduction needs >= 2 H100 (multistep runs DP=2); move to a 2x/8x H100 host"
                fi
            fi
        fi
    fi

    # Check 4 — driver version
    if command -v nvidia-smi >/dev/null 2>&1; then
        drv_line="$(nvidia-smi 2>/dev/null | grep -m1 -E 'Driver Version: *[0-9]')"
        drv_ver="$(printf '%s' "$drv_line" | sed -n 's/.*Driver Version: *\([0-9][0-9.]*\).*/\1/p')"
        if [[ -z "$drv_ver" ]]; then
            record 4 nvidia_driver fail \
                "could not parse driver version from nvidia-smi" \
                "Check nvidia-smi output manually"
        elif ver_ge "$drv_ver" "$MIN_DRIVER"; then
            record 4 nvidia_driver pass "Driver ${drv_ver} >= ${MIN_DRIVER} (CUDA 12.8 OK)" ""
        else
            # ADVISORY, not blocking: warn so the verdict still allows the user
            # to try the image, but make the reproduction caveat explicit.
            record 4 nvidia_driver warn \
                "Driver ${drv_ver} < ${MIN_DRIVER} — does not meet CUDA 12.8 reproduction conditions; the forge_train image may still run but numerical reproducibility is not guaranteed" \
                "For guaranteed reproduction, upgrade the NVIDIA driver to >= ${MIN_DRIVER} (requires admin)"
        fi
    else
        record 4 nvidia_driver fail "nvidia-smi not found" \
            "Install NVIDIA driver (admin required)"
    fi

    # Check 5 — os_arch (exec must be Linux x86_64; dev can be anything ssh works on)
    os_kernel="$(uname -s 2>/dev/null || echo unknown)"
    os_arch="$(uname -m 2>/dev/null || echo unknown)"
    if [[ "$os_kernel" == "Linux" && "$os_arch" == "x86_64" ]]; then
        record 5 os_arch pass "Linux x86_64" ""
    else
        record 5 os_arch fail \
            "${os_kernel} ${os_arch} — only Linux x86_64 is supported for the exec side" \
            "Run on a Linux x86_64 host (no Windows/macOS/arm64 support in v1)"
    fi

    # Check 6 — glibc
    if command -v ldd >/dev/null 2>&1; then
        glibc_line="$(ldd --version 2>&1 | head -1)"
        glibc_ver="$(printf '%s' "$glibc_line" | awk '{print $NF}')"
        if [[ -z "$glibc_ver" || ! "$glibc_ver" =~ ^[0-9]+\.[0-9]+ ]]; then
            record 6 glibc_version warn \
                "could not parse glibc version from: ${glibc_line}" \
                "Verify glibc >= ${MIN_GLIBC_PROBE} manually (ldd --version)"
        elif ver_ge "$glibc_ver" "$MIN_GLIBC_PROBE"; then
            record 6 glibc_version pass "glibc ${glibc_ver} >= ${MIN_GLIBC_PROBE}" ""
        else
            record 6 glibc_version fail \
                "glibc ${glibc_ver} < ${MIN_GLIBC_PROBE} (PyTorch manylinux_2_28 wheels won't load)" \
                "Use a newer base OS (Ubuntu 20.04+/RHEL 8+); glibc cannot be upgraded in user-space"
        fi
    else
        record 6 glibc_version fail "ldd not found" \
            "Install glibc tooling or run on a standard Linux distribution"
    fi

    # ---------------------------------------------------------------------
    # IMGCAP — single release-image readiness (exec only). Hardware floors
    # (Hopper sm_90, 80GB VRAM, >= 2 GPUs) are HARD and already recorded by
    # checks 1/2/3 above — they alone drive the verdict. Here we only
    # summarize image readiness and surface the ADVISORY driver caveat: a
    # sub-12.8 driver does NOT block (records a warn), it just means the
    # image may run but exact reproducibility is not guaranteed.
    # ---------------------------------------------------------------------
    _hw_cc="${sample_cc:-}"
    _hw_vram="${min_vram:-0}"
    _hw_drv="${drv_ver:-}"
    _hopper_ok=0; [[ "$_hw_cc" == "$REQUIRED_COMPUTE_CAP" ]] && _hopper_ok=1
    _vram_ok=0;   (( _hw_vram >= MIN_GPU_VRAM_MIB )) && _vram_ok=1
    _floor="${IMG_FLOOR[0]}"
    _cmax="${IMG_CMAX[0]}"

    if [[ -z "$_hw_drv" ]] || (( ! _hopper_ok )) || (( ! _vram_ok )); then
        # Hardware (or driver presence) is the blocker — already a hard fail
        # in checks 1/2/3. Mark the image unrunnable for reporting only; do
        # NOT record a second verdict-blocking check here.
        IMG_CAP[0]=0
        HOST_IMAGE_CLASS="blocked"
        if [[ -z "$_hw_drv" ]]; then
            IMG_REASON[0]="no NVIDIA driver detected (nvidia-smi missing/failed)"
        elif (( ! _hopper_ok )); then
            IMG_REASON[0]="GPU is not Hopper sm_90 (compute_cap=${_hw_cc:-unknown}); the image requires sm_90"
        else
            IMG_REASON[0]="GPU VRAM ${_hw_vram} MiB < ${MIN_GPU_VRAM_MIB} MiB; the image requires 80GB H100"
        fi
    else
        # Hardware OK → the host can run the image. The driver only gates
        # exact reproducibility, and is advisory.
        IMG_CAP[0]=1
        RECOMMENDED_IMAGE="${IMG_NAMES[0]}"
        if ver_ge "$_hw_drv" "$_floor"; then
            HOST_IMAGE_CLASS="ready"
            IMG_REASON[0]="driver ${_hw_drv} >= ${_floor} → CUDA ${_cmax} reproduction-ready"
            record IMGCAP images_capable pass \
                "host is reproduction-ready for ${IMG_NAMES[0]} (driver ${_hw_drv} >= ${_floor})" ""
        else
            HOST_IMAGE_CLASS="driver-advisory"
            IMG_REASON[0]="driver ${_hw_drv} < ${_floor} → CUDA ${_cmax} reproduction not guaranteed (advisory); image may still run"
            record IMGCAP images_capable warn \
                "hardware OK, but driver ${_hw_drv} < ${_floor}: does not meet CUDA 12.8 reproduction conditions — you may still try the ${IMG_NAMES[0]} image, reproducibility not guaranteed" \
                "Upgrade NVIDIA driver to >= ${_floor} for guaranteed reproduction (requires admin)"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# DEV+EXEC: workdir + tmp + disk
# ---------------------------------------------------------------------------
wd_result="$(probe_write_exec "$WORKDIR")"
case "$wd_result" in
    ok)
        record 8 workdir_writable pass "$WORKDIR is writable" ""
        record 9 workdir_exec     pass "$WORKDIR allows execve of created files" ""
        ;;
    ro)
        record 8 workdir_writable fail "$WORKDIR is read-only" \
            "Choose a writable --workdir or fix filesystem permissions"
        record 9 workdir_exec     fail "$WORKDIR is read-only (cannot test exec)" ""
        ;;
    noexec)
        record 8 workdir_writable pass "$WORKDIR is writable" ""
        record 9 workdir_exec     fail "$WORKDIR is mounted noexec (created files cannot execute)" \
            "Pick a --workdir on a filesystem without the noexec mount flag"
        ;;
    missing)
        record 8 workdir_writable fail "$WORKDIR does not exist" \
            "Pass --workdir pointing at an existing directory"
        record 9 workdir_exec     fail "$WORKDIR does not exist" ""
        ;;
esac

tmp_result="$(probe_write_exec "/tmp")"
case "$tmp_result" in
    ok)     record 10 tmp_writable_exec pass "/tmp is writable and exec-allowed" "" ;;
    ro)     record 10 tmp_writable_exec warn "/tmp is not writable" \
                "Set TMPDIR=${WORKDIR}/tmp before running the agent" ;;
    noexec) record 10 tmp_writable_exec warn "/tmp is mounted noexec" \
                "Set TMPDIR=${WORKDIR}/tmp before running the agent (some build steps execute from \$TMPDIR)" ;;
    missing) record 10 tmp_writable_exec warn "/tmp does not exist" \
                "Set TMPDIR=${WORKDIR}/tmp before running the agent" ;;
esac

free_gb=""
if [[ -d "$WORKDIR" ]]; then
    if df -BG --output=avail "$WORKDIR" >/dev/null 2>&1; then
        free_gb="$(df -BG --output=avail "$WORKDIR" 2>/dev/null \
                   | awk 'NR==2 {gsub("G",""); print $1}')"
    else
        free_gb="$(df -g "$WORKDIR" 2>/dev/null \
                   | awk 'NR==2 {print $4}')"
    fi
fi
if [[ -z "$free_gb" || ! "$free_gb" =~ ^[0-9]+$ ]]; then
    record 11 disk_space fail "could not determine free disk space at $WORKDIR" \
        "Verify $WORKDIR exists and df is available"
elif (( free_gb >= MIN_DISK_GB )); then
    record 11 disk_space pass "${free_gb} GB free at $WORKDIR (>= ${MIN_DISK_GB} GB for role=${ROLE})" ""
else
    record 11 disk_space fail \
        "${free_gb} GB free at $WORKDIR < ${MIN_DISK_GB} GB required for role=${ROLE}" \
        "Free up space or pass --workdir on a larger filesystem"
fi

# ---------------------------------------------------------------------------
# Checks 12, 13, 14 — network split (github, pypi, hf)
# ---------------------------------------------------------------------------
gh_ok=0; gh_url=""
for u in https://github.com https://api.github.com; do
    if probe_url "$u"; then gh_ok=1; gh_url="$u"; break; fi
done
if (( gh_ok )); then
    record 12 network_github pass "reachable: $gh_url" ""
else
    # github is NOT load-bearing on every install path. The in-place
    # path needs only a PyPI mirror + an HF endpoint. Mark as WARN so the
    # verdict doesn't block in-place runs; the agent must escalate to FAIL
    # only if install.method=docker (image build may need github) OR if the
    # dev box itself needs the Cursor Agent CLI installer.
    record 12 network_github warn "github.com unreachable" \
        "Only blocks install.method=docker (image build) and dev-box Cursor Agent CLI install. The in-place path does not need github."
fi

pypi_ok=0; pypi_url=""
for u in https://pypi.org/simple/ \
         https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/ \
         https://mirrors.aliyun.com/pypi/simple/; do
    if probe_url "$u"; then pypi_ok=1; pypi_url="$u"; break; fi
done
if (( pypi_ok )); then
    record 13 network_pypi pass "reachable: $pypi_url" ""
else
    record 13 network_pypi fail "no PyPI mirror reachable" \
        "Need one of pypi.org / mirrors.tuna.tsinghua.edu.cn / mirrors.aliyun.com — fix DNS/proxy"
fi

if (( ROLE_EXEC )); then
    hf_ok=0; hf_url=""
    for u in https://huggingface.co https://hf-mirror.com; do
        if probe_url "$u"; then hf_ok=1; hf_url="$u"; break; fi
    done
    if (( hf_ok )); then
        record 14 network_hf pass "reachable: $hf_url" ""
    else
        record 14 network_hf fail "no HuggingFace endpoint reachable" \
            "Need huggingface.co or hf-mirror.com (tokenizer + gsm8k data) — fix DNS/proxy or set HF_ENDPOINT=https://hf-mirror.com"
    fi
fi

# ---------------------------------------------------------------------------
# EXEC-only: install-method feasibility probes (E1 docker / E2 nvcr.io /
# E3 NGC-base detection).
#
# These DO NOT block the verdict on their own — they feed the
# `feasible_install_methods` aggregator below, which is what the agent
# reads to recommend a Step-2 playbook.
# ---------------------------------------------------------------------------
if (( ROLE_EXEC )); then
    # E1 — docker daemon + GPU runtime on exec (lets us "docker run --gpus all"
    # the harness image here). Cybertron devspaces ARE containers and do
    # not expose a docker socket, so this WARN-fails on most Cybertron
    # exec hosts — that's expected and triggers the "use cctl image swap
    # instead" path below.
    if command -v docker >/dev/null 2>&1; then
        if docker info >/dev/null 2>&1; then
            if docker info 2>/dev/null | grep -qi "nvidia"; then
                record E1 exec_docker pass "docker daemon up + nvidia runtime registered" ""
            else
                record E1 exec_docker warn "docker daemon up but nvidia runtime not detected" \
                    "Install nvidia-container-toolkit and restart docker; required for 'docker run --gpus all'."
            fi
        else
            record E1 exec_docker warn "docker CLI present but daemon not reachable" \
                "Start dockerd, or accept that this exec host can't 'docker run' the harness image (use cctl image swap instead)."
        fi
    else
        record E1 exec_docker warn "docker not available on exec" \
            "Expected on Cybertron devspaces (they ARE containers, no docker-in-docker). Deploy via 'cctl devspace copy --image <forge-train>' instead."
    fi

    # E2 — nvcr.io reachable for image pulls. The forge_train (ngc2501) release
    # image (repo-root Dockerfile, FROM nvcr.io/nvidia/pytorch:25.01-py3) pulls
    # its base from nvcr.io, so a "build the image" path on this host needs this.
    nvcr_token_url="https://nvcr.io/proxy_auth?scope=repository:nvidia/pytorch:pull"
    if curl -fsS --max-time 8 -o /dev/null "$nvcr_token_url" 2>/dev/null; then
        record E2 exec_nvcr_reach pass "nvcr.io anonymous token endpoint reachable" ""
    else
        record E2 exec_nvcr_reach warn "nvcr.io not reachable from exec" \
            "Required only if you BUILD the harness image on this host. If you build elsewhere (dev box or build server) and push to a reachable registry, this is moot."
    fi

    # E3 — is the exec host ALREADY an NGC PyTorch 25.06 container?
    # This is the in-place install method's hard precondition: the NGC
    # alpha torch wheel cannot be pip-installed from any public mirror,
    # so in-place can only work when the host already has it in
    # /usr/local/lib/python3.12/dist-packages.
    ngc_base_evidence=""
    if [[ -f /opt/pytorch/NVREADME.md ]]; then
        if grep -qE "PyTorch.*25\.06" /opt/pytorch/NVREADME.md 2>/dev/null; then
            ngc_base_evidence="/opt/pytorch/NVREADME.md confirms NGC PyTorch 25.06"
        elif grep -qE "PyTorch" /opt/pytorch/NVREADME.md 2>/dev/null; then
            ngc_ver="$(grep -oE 'PyTorch [0-9]+\.[0-9]+' /opt/pytorch/NVREADME.md 2>/dev/null | head -1)"
            ngc_base_evidence="NGC PyTorch present but ${ngc_ver:-unknown version} (not 25.06)"
        fi
    fi
    if [[ -z "$ngc_base_evidence" ]] && command -v python3 >/dev/null 2>&1; then
        tv="$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null)"
        if [[ "$tv" == "2.8.0a0+5228986c39.nv25.06" ]]; then
            ngc_base_evidence="torch.__version__ matches NGC 25.06 alpha exactly"
        elif [[ -n "$tv" ]]; then
            ngc_base_evidence="torch present but $tv (not the NGC 25.06 alpha)"
        fi
    fi
    if [[ "$ngc_base_evidence" == *"matches"* ]] || [[ "$ngc_base_evidence" == *"confirms"* ]]; then
        record E3 exec_ngc_base pass "$ngc_base_evidence" ""
    elif [[ -n "$ngc_base_evidence" ]]; then
        record E3 exec_ngc_base warn "$ngc_base_evidence" \
            "in-place install requires NGC PyTorch 25.06 base. Use docker install method instead."
    else
        record E3 exec_ngc_base warn "no NGC PyTorch base detected on exec host" \
            "in-place install requires an NGC PyTorch 25.06 container. Use docker install method instead."
    fi
fi

# ---------------------------------------------------------------------------
# DEV-only: A1a ssh client, A1b rsync client, A1c tmux
# Failures of A1a / A1b are AUTO-INSTALLABLE.
# ---------------------------------------------------------------------------
if (( ROLE_DEV )); then
    if command -v ssh >/dev/null 2>&1; then
        record A1a dev_ssh_client pass "$(ssh -V 2>&1 | head -1)" "" 0
    else
        record A1a dev_ssh_client fail "ssh not found in PATH" \
            "Install via 'micromamba install -c conda-forge openssh' (preferred) or 'apt-get install -y openssh-client' (root)" \
            1
    fi

    if command -v rsync >/dev/null 2>&1; then
        record A1b dev_rsync_client pass "$(rsync --version 2>/dev/null | head -1)" "" 0
    else
        record A1b dev_rsync_client fail "rsync not found in PATH" \
            "Install via 'micromamba install -c conda-forge rsync' (preferred) or 'apt-get install -y rsync' (root)" \
            1
    fi

    if command -v tmux >/dev/null 2>&1; then
        record A1c dev_tmux pass "$(tmux -V)" ""
    else
        record A1c dev_tmux warn "tmux not found" \
            "Recommended (agent-loop runs in detached tmux). Install via 'micromamba install -c conda-forge tmux' or use --no-tmux launch fallback."
    fi

    # D1 — docker client on dev (lets the dev box BUILD images locally).
    # Failure is not blocking on its own — for Cybertron-style deployment,
    # the image can be built on a remote build host and the dev box only
    # needs cctl to swap the exec devspace's image.
    if command -v docker >/dev/null 2>&1; then
        if docker version --format '{{.Server.Version}}' >/dev/null 2>&1; then
            record D1 dev_docker pass "$(docker version --format '{{.Client.Version}}' 2>/dev/null) (daemon: $(docker version --format '{{.Server.Version}}' 2>/dev/null))" ""
        else
            record D1 dev_docker warn "docker CLI present but daemon not reachable" \
                "Start Docker Desktop or 'systemctl start docker'; required if you want to BUILD images locally."
        fi
    else
        record D1 dev_docker warn "docker not found on dev" \
            "Optional. Install Docker Desktop to build the harness image locally. Skip if you let a build host / CI do the build."
    fi

    # D2 — cctl CLI on dev (Cybertron-style deployment: image swap via
    # `cctl devspace copy --image`, no docker run needed on exec).
    if command -v cctl >/dev/null 2>&1; then
        record D2 dev_cctl pass "$(cctl --version 2>/dev/null | head -1 || echo "cctl present")" ""
    else
        record D2 dev_cctl warn "cctl not found on dev" \
            "Required only if you deploy via a Cybertron-managed devspace (swap image via 'cctl devspace copy'). Optional for bare-host setups."
    fi
fi

# ---------------------------------------------------------------------------
# DEV-only: A2 ssh_reachable + A3 ssh_remote_workdir (only when --ssh given)
# ---------------------------------------------------------------------------
if (( ROLE_DEV )) && [[ -n "$SSH_ALIAS" ]]; then
    if probe_ssh "$SSH_ALIAS"; then
        record A2 ssh_reachable pass "ssh $SSH_ALIAS 'exit 0' succeeded (BatchMode, ${SSH_PROBE_TIMEOUT:-15}s timeout)" ""

        rwd_result="$(probe_remote_workdir "$SSH_ALIAS" "$REMOTE_WORKDIR")"
        case "$rwd_result" in
            ok)
                record A3 ssh_remote_workdir pass \
                    "remote $REMOTE_WORKDIR writable on $SSH_ALIAS" ""
                ;;
            missing_perm)
                record A3 ssh_remote_workdir fail \
                    "$REMOTE_WORKDIR not writable on $SSH_ALIAS" \
                    "Pick an absolute, writable --remote-workdir or fix permissions on the remote"
                ;;
            ssh_fail)
                record A3 ssh_remote_workdir fail \
                    "ssh succeeded earlier but workdir probe got no output (transient or shell config issue)" \
                    "Verify the remote shell prints OK from 'mkdir -p <dir> && test -w <dir> && echo OK'"
                ;;
        esac
    else
        record A2 ssh_reachable fail \
            "ssh $SSH_ALIAS not reachable (BatchMode, ${SSH_PROBE_TIMEOUT:-15}s timeout)" \
            "Check ~/.ssh/config alias spelling, key path, ProxyCommand (teleport devspaces need 'tsh status' OK)"
        record A3 ssh_remote_workdir fail "skipped — ssh unreachable" \
            "Resolve A2 first"
    fi
fi

# ===========================================================================
# Verdict
# ===========================================================================
hard_fails=()
install_fails=()
for i in "${!CHK_ID[@]}"; do
    if [[ "${CHK_STATUS[$i]}" == "fail" ]]; then
        if [[ "${CHK_AUTOINSTALL[$i]}" == "1" ]]; then
            install_fails+=("${CHK_NAME[$i]}: ${CHK_HINT[$i]}")
        else
            hard_fails+=("${CHK_NAME[$i]}: ${CHK_DETAIL[$i]}")
        fi
    fi
done

if (( ${#hard_fails[@]} == 0 && ${#install_fails[@]} == 0 )); then
    VERDICT="installable"; EXIT=0
elif (( ${#hard_fails[@]} == 0 )); then
    VERDICT="needs_install"; EXIT=2
else
    VERDICT="blocked"; EXIT=1
fi

# Image classification only runs on the exec side; a pure-dev probe has no GPU
# to classify, so report it as not-applicable rather than "neither".
if (( ! ROLE_EXEC )); then
    HOST_IMAGE_CLASS="n/a (dev-only probe)"
fi

# ===========================================================================
# Install-method feasibility aggregator.
#
# Two install methods are supported by the Phase-2 SKILL Playbooks:
#   * docker     — build the forge_train release image's Dockerfile (ngc2501 =
#                  repo-root Dockerfile FROM nvcr.io/.../25.01-py3), run it
#                  (or have cctl swap the exec image to it). Recommended path.
#   * in-place   — provision a conda env on top of an EXISTING release-image
#                  container, reusing system site-packages for the heavy stack.
#                  Experimental; preconditions are strict.
#
# We compute feasibility purely from probe results so the agent (and the
# user) get a deterministic "what's available here" answer.
# ===========================================================================
status_for() {
    local id="$1" i
    for i in "${!CHK_ID[@]}"; do
        if [[ "${CHK_ID[$i]}" == "$id" ]]; then
            echo "${CHK_STATUS[$i]}"; return
        fi
    done
    echo "absent"
}

# docker is feasible when EITHER:
#   (a) exec can docker-run directly (E1 pass), OR
#   (b) dev can build + cctl can swap the exec image (D1 pass + D2 pass)
# In both cases we also need nvcr.io reachable from wherever the BUILD
# happens. We only probe nvcr from exec (E2) — for (b) the agent should
# check dev-side nvcr separately if it actually builds on dev; we err
# toward "feasible" because Mac/CI build hosts usually have open net.
docker_feasible=0
docker_reason=""
e1="$(status_for E1)"; d1="$(status_for D1)"; d2="$(status_for D2)"
if [[ "$e1" == "pass" ]]; then
    docker_feasible=1
    docker_reason="exec can docker run (E1 pass)"
elif [[ "$d1" == "pass" && "$d2" == "pass" ]]; then
    docker_feasible=1
    docker_reason="dev builds + cctl image swap on exec (D1/D2 pass)"
elif [[ "$d2" == "pass" ]]; then
    # cctl available but no local docker — assume a build-server can do it.
    docker_feasible=1
    docker_reason="cctl available (D2 pass); image build needs a separate build host"
fi

# in-place is feasible only when exec is an NGC 25.06 base (E3 pass).
inplace_feasible=0
inplace_reason=""
if [[ "$(status_for E3)" == "pass" ]]; then
    inplace_feasible=1
    inplace_reason="exec is NGC 25.06 base (E3 pass)"
fi

# Build the feasible_install_methods list. Docker comes first because it's
# the validated default.
feasible_methods=()
(( docker_feasible )) && feasible_methods+=("docker")
(( inplace_feasible )) && feasible_methods+=("in-place")

# Recommended method = first feasible (docker if available, else in-place,
# else empty string meaning "neither, Phase 2 blocked").
recommended_method=""
if (( ${#feasible_methods[@]} > 0 )); then
    recommended_method="${feasible_methods[0]}"
fi


# ===========================================================================
# Output
# ===========================================================================
if (( JSON )); then
    printf '{\n'
    printf '  "verdict": "%s",\n' "$VERDICT"
    printf '  "role": "%s",\n' "$ROLE"
    printf '  "mode": "%s",\n' "$MODE"
    printf '  "workdir": "%s",\n' "$(json_escape "$WORKDIR")"
    if [[ -n "$SSH_ALIAS" ]]; then
        printf '  "ssh_alias": "%s",\n' "$(json_escape "$SSH_ALIAS")"
    else
        printf '  "ssh_alias": null,\n'
    fi
    if [[ -n "$REMOTE_WORKDIR" ]]; then
        printf '  "remote_workdir": "%s",\n' "$(json_escape "$REMOTE_WORKDIR")"
    else
        printf '  "remote_workdir": null,\n'
    fi

    # Install-method feasibility — the agent reads this to decide which
    # Step-2 playbook to offer the user.
    printf '  "feasible_install_methods": ['
    if (( ${#feasible_methods[@]} == 0 )); then
        printf '],\n'
    else
        for i in "${!feasible_methods[@]}"; do
            sep=", "; (( i == ${#feasible_methods[@]} - 1 )) && sep=""
            printf '"%s"%s' "${feasible_methods[$i]}" "$sep"
        done
        printf '],\n'
    fi
    if [[ -n "$recommended_method" ]]; then
        printf '  "recommended_install_method": "%s",\n' "$recommended_method"
    else
        printf '  "recommended_install_method": null,\n'
    fi
    printf '  "install_method_reasons": {\n'
    printf '    "docker":   {"feasible": %s, "reason": "%s"},\n' \
        "$([[ $docker_feasible == 1 ]] && echo true || echo false)" \
        "$(json_escape "${docker_reason:-no docker / cctl path available}")"
    printf '    "in-place": {"feasible": %s, "reason": "%s"}\n' \
        "$([[ $inplace_feasible == 1 ]] && echo true || echo false)" \
        "$(json_escape "${inplace_reason:-exec is not an NGC 25.06 base container}")"
    printf '  },\n'

    # Release-image classification (the Step-1 image-selection authority).
    printf '  "capable_images": ['
    _cap_arr=()
    for k in "${!IMG_NAMES[@]}"; do
        [[ "${IMG_CAP[$k]:-0}" == "1" ]] && _cap_arr+=("${IMG_NAMES[$k]}")
    done
    if (( ${#_cap_arr[@]} == 0 )); then
        printf '],\n'
    else
        for i in "${!_cap_arr[@]}"; do
            sep=", "; (( i == ${#_cap_arr[@]} - 1 )) && sep=""
            printf '"%s"%s' "${_cap_arr[$i]}" "$sep"
        done
        printf '],\n'
    fi
    if [[ -n "$RECOMMENDED_IMAGE" ]]; then
        printf '  "recommended_image": "%s",\n' "$RECOMMENDED_IMAGE"
    else
        printf '  "recommended_image": null,\n'
    fi
    printf '  "host_image_class": "%s",\n' "$(json_escape "$HOST_IMAGE_CLASS")"
    printf '  "image_reasons": {\n'
    for k in "${!IMG_NAMES[@]}"; do
        sep=","; (( k == ${#IMG_NAMES[@]} - 1 )) && sep=""
        printf '    "%s": {"capable": %s, "reason": "%s", "driver_floor": "%s", "cuda_max": "%s"}%s\n' \
            "${IMG_NAMES[$k]}" \
            "$([[ "${IMG_CAP[$k]:-0}" == "1" ]] && echo true || echo false)" \
            "$(json_escape "${IMG_REASON[$k]:-not evaluated (dev-only probe)}")" \
            "${IMG_FLOOR[$k]}" \
            "${IMG_CMAX[$k]}" \
            "$sep"
    done
    printf '  },\n'

    printf '  "checks": [\n'
    n=${#CHK_ID[@]}
    for i in "${!CHK_ID[@]}"; do
        hint_field="null"
        [[ -n "${CHK_HINT[$i]}" ]] && hint_field="\"$(json_escape "${CHK_HINT[$i]}")\""
        ai="${CHK_AUTOINSTALL[$i]:-0}"
        ai_field="false"
        [[ "$ai" == "1" ]] && ai_field="true"
        sep=","; (( i == n - 1 )) && sep=""
        printf '    {"id": "%s", "name": "%s", "status": "%s", "detail": "%s", "fix_hint": %s, "auto_install": %s}%s\n' \
            "$(json_escape "${CHK_ID[$i]}")" \
            "$(json_escape "${CHK_NAME[$i]}")" \
            "${CHK_STATUS[$i]}" \
            "$(json_escape "${CHK_DETAIL[$i]}")" \
            "$hint_field" \
            "$ai_field" \
            "$sep"
    done
    printf '  ],\n'

    printf '  "blocking_reasons": ['
    nb=${#hard_fails[@]}
    if (( nb == 0 )); then
        printf ']'
    else
        printf '\n'
        for i in "${!hard_fails[@]}"; do
            sep=","; (( i == nb - 1 )) && sep=""
            printf '    "%s"%s\n' "$(json_escape "${hard_fails[$i]}")" "$sep"
        done
        printf '  ]'
    fi
    printf ',\n  "auto_install_hints": ['
    nh=${#install_fails[@]}
    if (( nh == 0 )); then
        printf ']\n'
    else
        printf '\n'
        for i in "${!install_fails[@]}"; do
            sep=","; (( i == nh - 1 )) && sep=""
            printf '    "%s"%s\n' "$(json_escape "${install_fails[$i]}")" "$sep"
        done
        printf '  ]\n'
    fi
    printf '}\n'
else
    printf '%sMachine readiness probe%s  (role=%s, mode=%s, workdir=%s' \
        "$C_BOLD" "$C_RESET" "$ROLE" "$MODE" "$WORKDIR"
    [[ -n "$SSH_ALIAS" ]] && printf ', ssh=%s' "$SSH_ALIAS"
    printf ')\n'
    printf '%s----------------------------------------------------------------%s\n' \
        "$C_DIM" "$C_RESET"
    for i in "${!CHK_ID[@]}"; do
        case "${CHK_STATUS[$i]}" in
            pass) mark="${C_OK}[OK]${C_RESET}  " ;;
            fail) mark="${C_FAIL}[FAIL]${C_RESET}" ;;
            warn) mark="${C_WARN}[WARN]${C_RESET}" ;;
            *)    mark="[??]  " ;;
        esac
        printf '%s %-22s %s\n' "$mark" "${CHK_NAME[$i]}" "${CHK_DETAIL[$i]}"
        if [[ -n "${CHK_HINT[$i]}" && "${CHK_STATUS[$i]}" != "pass" ]]; then
            printf '       %s-> %s%s\n' "$C_DIM" "${CHK_HINT[$i]}" "$C_RESET"
        fi
    done
    printf '%s----------------------------------------------------------------%s\n' \
        "$C_DIM" "$C_RESET"

    # Release-image capability report (exec only) — the Step-1 image picker.
    if (( ROLE_EXEC )); then
        printf '%sRelease-image capability:%s  (host class: %s)\n' \
            "$C_BOLD" "$C_RESET" "$HOST_IMAGE_CLASS"
        for k in "${!IMG_NAMES[@]}"; do
            if [[ "${IMG_CAP[$k]:-0}" == "1" ]]; then
                printf '  %s[YES]%s %-9s — %s\n' "$C_OK" "$C_RESET" "${IMG_NAMES[$k]}" "${IMG_REASON[$k]}"
            else
                printf '  %s[NO]%s  %-9s — %s\n' "$C_FAIL" "$C_RESET" "${IMG_NAMES[$k]}" "${IMG_REASON[$k]:-not evaluated}"
            fi
        done
        if [[ -n "$RECOMMENDED_IMAGE" ]]; then
            printf '  %sRecommended image:%s %s\n' "$C_BOLD" "$C_RESET" "$RECOMMENDED_IMAGE"
        else
            printf '  %sRecommended image:%s (none — no release image can run here)\n' \
                "$C_FAIL$C_BOLD" "$C_RESET"
        fi
        printf '%s----------------------------------------------------------------%s\n' \
            "$C_DIM" "$C_RESET"
    fi

    # Install-method feasibility report (read by the agent + shown to user).
    printf '%sInstall-method feasibility:%s\n' "$C_BOLD" "$C_RESET"
    if (( docker_feasible )); then
        printf '  %s[YES]%s docker     — %s\n' "$C_OK" "$C_RESET" "$docker_reason"
    else
        printf '  %s[NO]%s  docker     — %s\n' "$C_FAIL" "$C_RESET" \
            "${docker_reason:-no docker on exec, no cctl+image-swap path on dev}"
    fi
    if (( inplace_feasible )); then
        printf '  %s[YES]%s in-place   — %s  %s(experimental)%s\n' \
            "$C_OK" "$C_RESET" "$inplace_reason" "$C_DIM" "$C_RESET"
    else
        printf '  %s[NO]%s  in-place   — %s  %s(experimental)%s\n' \
            "$C_FAIL" "$C_RESET" "${inplace_reason:-exec is not an NGC 25.06 base container}" "$C_DIM" "$C_RESET"
    fi
    if [[ -n "$recommended_method" ]]; then
        printf '  %sRecommended:%s %s\n' "$C_BOLD" "$C_RESET" "$recommended_method"
    else
        printf '  %sRecommended:%s (none — Phase 2 cannot start without a feasible method)\n' \
            "$C_FAIL$C_BOLD" "$C_RESET"
    fi
    printf '%s----------------------------------------------------------------%s\n' \
        "$C_DIM" "$C_RESET"

    case "$VERDICT" in
        installable)
            printf '%sVERDICT: installable%s — agent can proceed.\n' \
                "$C_OK$C_BOLD" "$C_RESET"
            ;;
        needs_install)
            printf '%sVERDICT: needs_install%s — auto-installable failures only:\n' \
                "$C_WARN$C_BOLD" "$C_RESET"
            for r in "${install_fails[@]}"; do
                printf '  %s* %s%s\n' "$C_WARN" "$r" "$C_RESET"
            done
            ;;
        blocked)
            printf '%sVERDICT: blocked%s\n' "$C_FAIL$C_BOLD" "$C_RESET"
            for r in "${hard_fails[@]}"; do
                printf '  %s* %s%s\n' "$C_FAIL" "$r" "$C_RESET"
            done
            ;;
    esac
fi

exit $EXIT
