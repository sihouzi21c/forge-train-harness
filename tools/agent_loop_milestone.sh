#!/usr/bin/env bash
# Single source of truth for Stage 1 milestone state I/O used by
# agent-loop.sh's run_agent_stage loop.
#
# Stage 1 progresses through a linear, NAMED milestone sequence (e.g.
# alignment -> bitwise-singlecard -> ... -> production). The sequence
# itself is the sole SSOT in the active eval.toml [stage1].milestone_order
# (see milestone_rename_design.md §5); agent-loop.sh resolves it once and
# exports it as a space-separated list in:
#
#     FORGE_MILESTONE_ORDER="alignment bitwise-singlecard ... production"
#
# The active milestone is persisted in <state_dir>/<stage>.milestone (one
# line, a milestone NAME). The dev agent self-declares progress by writing
# a literal line into its commit message:
#
#     MILESTONE_STATUS: <name> PASS
#
# After every round, agent-loop.sh scans recent commits for this line and
# advances the persisted milestone to the SUCCESSOR (in the order) of the
# furthest-along declared milestone -- i.e. "<name> PASS" means "<name>'s
# gate has passed, the next active milestone is the one after it". Stage 2
# has no milestone manifest; the per-stage wrappers in agent-loop.sh skip
# the advance path for stages without one.
#
# Helpers exposed:
#   parse_commit_milestone_pass            read commit messages from stdin;
#                                          print the furthest-along milestone
#                                          name declared, or nothing
#   validate_milestone <value>             return 0 iff value is in the order
#   read_stage_milestone_file <path>       print stored value or the first
#                                          milestone; warn on garbage to stderr
#   write_stage_milestone_file <path> <m>  validate then write
#   next_milestone_after <high> <current>  echo next active milestone (capped
#                                          at the last), empty if no advance
#
# Ordering SSOT is injected via FORGE_MILESTONE_ORDER -- the helpers stay
# python-free and git-free so the unit test can drive each helper in
# isolation by setting that env var. agent-loop.sh wraps these with
# stage-aware paths and emits the milestone_advanced loop event.

set -uo pipefail

_milestone_order() {
  # Echo the configured milestone order, one name per line. Fail fast when
  # FORGE_MILESTONE_ORDER is unset/empty -- the progression cannot be
  # resolved without the SSOT and silently defaulting would mis-order a run.
  local order="${FORGE_MILESTONE_ORDER:-}"
  if [[ -z "${order// }" ]]; then
    printf 'ERROR: FORGE_MILESTONE_ORDER unset; cannot resolve milestone progression\n' >&2
    return 1
  fi
  # Intentional word-splitting: the env var is a space-separated list.
  # shellcheck disable=SC2086
  printf '%s\n' $order
}

_milestone_index() {
  # Print the 0-based index of $1 in the order; return 1 (no output) if absent.
  local target="${1:-}" i=0 name
  while IFS= read -r name; do
    if [[ "$name" == "$target" ]]; then
      printf '%d' "$i"
      return 0
    fi
    i=$((i + 1))
  done < <(_milestone_order)
  return 1
}

validate_milestone() {
  _milestone_index "${1:-}" >/dev/null 2>&1
}

read_stage_milestone_file() {
  local f="${1:?read_stage_milestone_file: path required}"
  local first
  first=$(_milestone_order | head -n1) || return 1
  if [[ ! -s "$f" ]]; then
    printf '%s' "$first"
    return 0
  fi
  local raw
  raw=$(head -n1 "$f" | tr -d '[:space:]')
  if validate_milestone "$raw"; then
    printf '%s' "$raw"
    return 0
  fi
  printf 'WARN: corrupt milestone file %s contents %q; resetting to %s\n' \
    "$f" "$raw" "$first" >&2
  printf '%s' "$first" > "$f"
  printf '%s' "$first"
}

write_stage_milestone_file() {
  local f="${1:?write_stage_milestone_file: path required}"
  local value="${2:?write_stage_milestone_file: milestone required}"
  if ! validate_milestone "$value"; then
    printf 'ERROR: invalid milestone %q; expected one of: %s\n' \
      "$value" "$(_milestone_order | tr '\n' ' ')" >&2
    return 1
  fi
  # Atomic write: stage into a sibling tmp file, then rename. A SIGKILL
  # mid-write would otherwise leave the file truncated to zero bytes,
  # which read_stage_milestone_file would self-heal to the first milestone
  # -- silently regressing a loop that was already further along. With
  # tmp+rename the file either holds the previous value (rename never ran)
  # or the new one (rename ran atomically); it can never be torn.
  local tmp="$f.tmp.$$"
  if ! printf '%s\n' "$value" > "$tmp"; then
    rm -f "$tmp" 2>/dev/null || true
    return 1
  fi
  mv -f "$tmp" "$f"
}

next_milestone_after() {
  # Given the furthest-along milestone declared PASS in recent commits and
  # the currently-persisted active milestone, echo the next active milestone
  # (the successor in the order), capped at the last entry. Echoes nothing
  # when no advance is warranted (current already at or past that successor).
  #
  # Semantics: "MILESTONE_STATUS: <name> PASS" means the agent has finished
  # <name>, so the next active milestone is the one after it. The cap at the
  # last entry keeps a terminal PASS (or a malformed one) from escaping the
  # documented linear progression.
  local highest="${1:-}" current="${2:-}"
  local high_idx cur_idx last_idx next_idx count
  high_idx=$(_milestone_index "$highest" 2>/dev/null) || return 0
  count=$(_milestone_order | grep -c .) || return 0
  last_idx=$((count - 1))
  next_idx=$((high_idx + 1))
  if (( next_idx > last_idx )); then next_idx=$last_idx; fi
  if cur_idx=$(_milestone_index "$current" 2>/dev/null); then :; else cur_idx=-1; fi
  if (( next_idx <= cur_idx )); then
    return 0
  fi
  # No trailing newline (matches the caller's `next=$(...)` contract and the
  # unit-test exact-match assertions).
  printf '%s' "$(_milestone_order | sed -n "$((next_idx + 1))p")"
}

parse_commit_milestone_pass() {
  # Reads commit messages from stdin (e.g. `git log --format=%B`). Prints
  # the milestone name declared via "MILESTONE_STATUS: <name> PASS" that is
  # FURTHEST along the order, or empty string if none. Names not in the
  # order are ignored. Strict on the marker shape so accidental prose in a
  # commit body cannot advance the milestone.
  #
  # The pipeline tolerates a no-match grep (exits 1 under pipefail) via the
  # `|| true` below: "no matching marker" is the contracted-empty case
  # (every Stage 1 round before the dev agent first declares PASS hits it),
  # not an error.
  local declared
  declared=$(grep -oE '^MILESTONE_STATUS:[[:space:]]+[A-Za-z0-9._-]+[[:space:]]+PASS[[:space:]]*$' 2>/dev/null \
    | sed -E 's/^MILESTONE_STATUS:[[:space:]]+([A-Za-z0-9._-]+)[[:space:]]+PASS[[:space:]]*$/\1/' \
    || true)
  local best="" best_idx=-1 name idx
  while IFS= read -r name; do
    [[ -z "$name" ]] && continue
    idx=$(_milestone_index "$name" 2>/dev/null) || continue
    if (( idx > best_idx )); then
      best_idx=$idx
      best="$name"
    fi
  done <<< "$declared"
  [[ -n "$best" ]] && printf '%s' "$best" || true
}

# Direct-invocation dispatcher so the unit test can drive each helper
# without sourcing bash functions.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  cmd="${1:-}"
  shift || true
  case "$cmd" in
    parse) parse_commit_milestone_pass ;;
    validate) validate_milestone "${1:-}" ;;
    read) read_stage_milestone_file "${1:-}" ;;
    write) write_stage_milestone_file "${1:-}" "${2:-}" ;;
    next-after) next_milestone_after "${1:-}" "${2:-}" ;;
    *)
      printf 'Usage: %s {parse|validate <m>|read <path>|write <path> <m>|next-after <high> <current>}\n' \
        "$0" >&2
      exit 2
      ;;
  esac
fi
