#!/bin/bash
# Ours-side WSD-SFT CRASH-RESUME wrapper — GATE ONLY (production-resume-70).
#
# Wraps train_ours_al.sh (the 3-phase driver) to prove crash-resume is bitwise:
# it launches the driver N+1 times and SIGKILLs it N times (once mid stable, once
# mid decay, once mid sft), each kill leaving the phase's latest versioned
# step_<abs>/ ckpt on disk. On every relaunch the driver's has_ckpt branch
# full-resumes the crashed phase from that ckpt (weights + optimizer + step +
# cursor). The stitched per-step [LOSS] trajectory the dispatcher parses must
# equal the never-crashed clean-baseline ours run bitwise, per phase per step.
#
# Kill is LOG-DRIVEN (deterministic, tied to training progress — NOT wall-clock):
# we watch the driver's merged stdout for the engine's `[PHASE] name=<x>` banner
# and `[LOSS] step=<n>` lines, and kill exactly when (phase, emitted-step) hits
# the next crash target. save_interval (SAVE_INTERVAL, injected by the dispatcher)
# guarantees a versioned ckpt already landed before each crash point.
#
# Continuous crash numbering (1..stable+decay+sft) → (phase, emitted step):
#   c <= STABLE_ITERS                    -> stable, emit=c
#   c <= STABLE_ITERS+DECAY_ITERS        -> decay,  emit=c            (start_step=STABLE_ITERS)
#   else                                 -> sft,    emit=c-(STABLE+DECAY)  (sft resets to 0)
#
# Env in (dispatcher injects; the rest is inherited and passed through to the
# driver unchanged): CRASH_STEPS="12,33,54", STABLE_ITERS, DECAY_ITERS, SAVE_ROOT,
# SAVE_INTERVAL, plus the full gate env set consumed by train_ours_al.sh.
#
# Production does NOT use this wrapper — it invokes train_ours_al.sh directly
# (real save_interval + operator-driven restart), so this file is inert there.

set -uo pipefail   # NOT -e: killed children return non-zero; we handle rc by hand.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AL_SCRIPT="$SCRIPT_DIR/train_ours_al.sh"
[[ -f "$AL_SCRIPT" ]] || { echo "ERROR: missing $AL_SCRIPT" >&2; exit 1; }

SAVE_ROOT="${SAVE_ROOT:?SAVE_ROOT must be set (ours scratch root; the wrapper cleans + resumes under it)}"
STABLE_ITERS="${STABLE_ITERS:?STABLE_ITERS must be set}"
DECAY_ITERS="${DECAY_ITERS:?DECAY_ITERS must be set}"

# Clean start ONCE: drop any stale per-phase ckpts so attempt 1 is a true first
# launch (a leftover step_<N>/ would trip the driver's has_ckpt restart branch).
# Scoped to the driver's own subdirs — never the whole SAVE_ROOT.
rm -rf "$SAVE_ROOT/stable" "$SAVE_ROOT/decay" "$SAVE_ROOT/sft"
LOG_DIR="$SAVE_ROOT/_wrapper_logs"
mkdir -p "$LOG_DIR"

# map_crash <continuous> -> sets G_PHASE, G_EMIT
G_PHASE=""; G_EMIT=""
map_crash() {
    local c="$1"
    if   (( c <= STABLE_ITERS ));               then G_PHASE=stable; G_EMIT=$c
    elif (( c <= STABLE_ITERS + DECAY_ITERS ));  then G_PHASE=decay;  G_EMIT=$c
    else                                              G_PHASE=sft;    G_EMIT=$(( c - STABLE_ITERS - DECAY_ITERS ))
    fi
}

# run_attempt <target|""> <logfile>. target="phase:emit" kills on hit; ""=run to
# completion and propagate the driver's exit code. Streams the driver's merged
# stdout/stderr to OUR stdout so the dispatcher parses the full [PHASE]/[LOSS]
# trajectory.
run_attempt() {
    local target="$1" logf="$2"
    local tphase="" temit=""
    [[ -n "$target" ]] && { tphase="${target%%:*}"; temit="${target##*:}"; }
    : > "$logf"

    setsid bash "$AL_SCRIPT" >"$logf" 2>&1 &
    local pgid=$!            # setsid makes the child a session/group leader: PGID == its PID
    local cur_phase="" triggered=0

    # process substitution keeps the while-loop in THIS shell (cur_phase / triggered
    # persist; pgid in scope). tail --pid exits when the driver dies → loop ends.
    while IFS= read -r line; do
        printf '%s\n' "$line"
        [[ -n "$target" && "$triggered" == 0 ]] || continue
        if [[ "$line" =~ \[PHASE\]\ name=([a-z_]+) ]]; then
            cur_phase="${BASH_REMATCH[1]}"
        elif [[ "$line" =~ \[LOSS\]\ step=([0-9]+) ]]; then
            if [[ "$cur_phase" == "$tphase" && "${BASH_REMATCH[1]}" == "$temit" ]]; then
                echo "[WRAPPER] CRASH trigger: phase=$cur_phase emit=${BASH_REMATCH[1]} -> SIGKILL -$pgid" >&2
                kill -9 -"$pgid" 2>/dev/null || true
                triggered=1
            fi
        fi
    done < <(tail -n +1 -F --pid="$pgid" "$logf" 2>/dev/null)

    wait "$pgid" 2>/dev/null; local rc=$?
    if [[ -n "$target" ]]; then
        [[ "$triggered" == 1 ]] || echo "[WRAPPER] WARNING: target $target never hit (mapping wrong or phase skipped)" >&2
        # reap any GPU workers orphaned by the group kill so the next attempt's
        # torchrun can re-acquire the devices (gate runs serially → safe).
        pkill -9 -f "train_ours_phase.py" 2>/dev/null || true
        sleep 2
        return 0
    fi
    return "$rc"
}

IFS=',' read -ra CRASH_ARR <<< "${CRASH_STEPS:-}"
# drop empty tokens (e.g. trailing comma / unset)
CLEAN=(); for c in "${CRASH_ARR[@]:-}"; do [[ "$c" =~ ^[0-9]+$ ]] && CLEAN+=("$c"); done
CRASH_ARR=("${CLEAN[@]:-}")
[[ "${CRASH_ARR[*]:-}" == "" ]] && CRASH_ARR=()

n=${#CRASH_ARR[@]}
echo "[WRAPPER] crash schedule (continuous steps): ${CRASH_ARR[*]:-<none>} ; ${n} kills, $((n+1)) launches; SAVE_INTERVAL=${SAVE_INTERVAL:-<unset>}" >&2

for (( i=0; i<n; i++ )); do
    map_crash "${CRASH_ARR[i]}"
    echo "[WRAPPER] === attempt $((i+1))/$((n+1)): crash at continuous=${CRASH_ARR[i]} -> phase=$G_PHASE emit=$G_EMIT ===" >&2
    run_attempt "$G_PHASE:$G_EMIT" "$LOG_DIR/attempt_$((i+1)).log"
done

echo "[WRAPPER] === attempt $((n+1))/$((n+1)): final, run to completion ===" >&2
run_attempt "" "$LOG_DIR/attempt_$((n+1)).log"
final_rc=$?
echo "[WRAPPER] pipeline done rc=$final_rc" >&2
exit "$final_rc"
