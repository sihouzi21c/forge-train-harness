#!/usr/bin/env bash
# Devspace lease lifecycle for agent-loop.sh (sourced).
#
# This is the SINGLE place a loop claims its exclusive cctl devspace,
# rewrites [remote].hostname to the derived ds-<id>, and freezes the
# per-loop config dir. It is invoked from the shared post-bootstrap path
# so BOTH the CLI bootstrap flow and the web launch flow run it exactly
# once. Previously this logic lived only inside the
# `FORGE_TRAIN_PROVISIONED != 1` bootstrap block; the web flow sets that
# sentinel, so web-launched devspace loops skipped it entirely — no GPU
# lease was claimed (silent leak + anti-collision defeated) and no
# keepalive SSH stanza was synthesized (a black-holed devspace then hung
# the loop forever).
#
# Sourced — not executed — so it can mutate the loop's environment if
# needed and reuse $PYTHON / $LOOP_ID. The shared lease registry root is
# owned by tools.lease (anchored to FORGE_SOURCE_ROOT); this helper does
# not pass --repo-root.

# _devspace_claim_and_freeze <config_dir>
#
# Writability is the launch/resume discriminator:
#   * a WRITABLE config dir is a first launch — claim a devspace (when
#     kind="devspace"), rewrite the hostname, then freeze the dir.
#   * a READ-ONLY (already-frozen) config dir is a resume — no-op, so a
#     resumed loop never double-books a second devspace.
# ssh / local kinds claim nothing; they only get the freeze.
#
# Requires globals: PYTHON. Must be called with cwd on a tree where
# `python -m tools.lease` resolves (the provisioned workspace).
_devspace_claim_and_freeze() {
  local cfg_dir="$1"
  [[ -n "$cfg_dir" && -d "$cfg_dir" ]] || return 0
  # Already frozen → resume; re-claiming would double-book a GPU and the
  # chmod is a no-op anyway.
  [[ -w "$cfg_dir" ]] || return 0

  if [[ -f "$cfg_dir/remote.toml" ]] \
     && grep -qE '^kind[[:space:]]*=[[:space:]]*"(devspace|job)"' "$cfg_dir/remote.toml"; then
    local _project _cluster _pool _image _gpu _gpu_model _priority _billing _new_host
    local _kind _claim_gpu _ws _ws_err
    _ds_field() { awk -F'"' "/^$1[[:space:]]*=/{print \$2; exit}" "$cfg_dir/remote.toml"; }
    _project="$(_ds_field project)"
    _cluster="$(_ds_field cluster)"
    _pool="$(_ds_field resource_pool)"
    _image="$(_ds_field image)"
    _gpu="$(awk -F'=' '/^gpu_count[[:space:]]*=/{gsub(/[^0-9]/,"",$2);print $2; exit}' "$cfg_dir/remote.toml")"
    _gpu_model="$(_ds_field gpu_model)"
    _priority="$(_ds_field priority)"
    _billing="$(_ds_field billing_account_id)"
    _kind="$(_ds_field kind)"
    _ws="$(_ds_field workspace)"
    unset -f _ds_field
    if [[ -z "$_project" || -z "$_cluster" || -z "$_pool" || -z "$_image" \
          || -z "$_gpu" || -z "$_gpu_model" ]]; then
      echo "ERROR: remote.toml kind=\"$_kind\" missing required spec field (project/cluster/resource_pool/image/gpu_count/gpu_model)" >&2
      return 2
    fi
    # Only kind="job" MUST sync to / run from a shared persistent cluster
    # path (e.g. /user/<username>): its per-suite GPU job runs on a
    # DIFFERENT pod than the 0-GPU gateway and can only see code synced to
    # that shared root — an empty workspace (→ pod-local /root) is invisible
    # to the job, so fail fast (via the SSOT validator) before the
    # minutes-long cctl devspace create. kind="devspace" runs suites on the
    # held box's OWN filesystem, so it does NOT need a shared root; an empty
    # workspace defaults to the remote $HOME and is fine.
    if [[ "$_kind" == "job" ]] \
       && ! _ws_err="$("$PYTHON" -m tools.remote_workspace --validate "$_ws" 2>&1)"; then
      echo "ERROR: remote.toml kind=\"$_kind\": $_ws_err" >&2
      return 2
    fi
    # In kind="job" the devspace is a 0-GPU filesystem gateway; the
    # gpu_count GPUs are requested per-suite as ephemeral cctl
    # pytorchjobs (see tools/gpu_job.py). gpu_count stays in remote.toml
    # unchanged so the job submitter reads it; only THIS claim's --gpu is
    # forced to 0.
    if [[ "$_kind" == "job" ]]; then
      _claim_gpu=0
    else
      _claim_gpu="$_gpu"
    fi
    echo "[$(date '+%F %T')] claiming devspace lease (create project=$_project gpu=$_claim_gpu kind=$_kind loop=${LOOP_ID:-?})" >&2
    _new_host="$( \
      "$PYTHON" -m tools.lease claim devspace \
        --loop-id "${LOOP_ID:?LOOP_ID required for lease claim}" \
        --project "$_project" --cluster "$_cluster" --resource-pool "$_pool" \
        --image "$_image" --gpu "$_claim_gpu" --gpu-model "$_gpu_model" \
        --priority "${_priority:-NORMAL}" \
        ${_billing:+--billing-account-id "$_billing"} \
    )" || {
      echo "ERROR: lease claim failed; see stderr above" >&2
      return 2
    }
    # Rewrite the per-loop hostname BEFORE the freeze below.
    sed -i.bak -E "s|^hostname[[:space:]]*=.*|hostname = \"$_new_host\"|" "$cfg_dir/remote.toml"
    rm -f "$cfg_dir/remote.toml.bak"
    echo "[$(date '+%F %T')] lease ready: $_new_host" >&2
    unset _project _cluster _pool _image _gpu _gpu_model _priority _billing _new_host
    unset _kind _claim_gpu _ws _ws_err
  fi

  # Mirror the frozen axis configs into <workspace>/config/ so the agent
  # cats the same source of truth the harness CLI reads.
  #
  # Background: provision (web/routers/loop.py:_provision_workspace and
  # the CLI bootstrap block in agent-loop.sh) explicitly skips the 7 axis
  # toml basenames during shutil.copytree, leaving <workspace>/config/
  # populated only with committed templates. The active per-axis values
  # written by the UI / wrapper live in <cfg_dir> (= <loop>/config/) and
  # are reached by the harness CLI via FORGE_CONFIG_DIR. An agent inside
  # the workspace, however, naturally cats config/<axis>.toml relative to
  # its cwd and would see file-not-found without help. Symlinking each
  # active file into the workspace makes "cat config/data.toml" return
  # the same content the CLI reads — no second truth, no divergence.
  #
  # Done BEFORE the chmod -R a-w below so the link targets land while the
  # dir is still writable. After the freeze, the targets become read-only
  # and the symlinks transparently inherit that semantics (writes to the
  # link fail at the target). Skipped when the workspace config dir is
  # absent (e.g. tests that exercise the lease step in isolation).
  local _ws_cfg_dir
  _ws_cfg_dir="$(dirname "$cfg_dir")/workspace/config"
  if [[ -d "$_ws_cfg_dir" ]]; then
    local _axis _active _ws_link
    for _axis in ref data remote agent eval model optim; do
      _active="$cfg_dir/$_axis.toml"
      _ws_link="$_ws_cfg_dir/$_axis.toml"
      # Always drop any pre-existing entry (regular file from a pre-fix
      # workspace, a dangling symlink from an earlier resume, or a stale
      # template the provision filter didn't catch on an older build).
      rm -f "$_ws_link"
      [[ -f "$_active" ]] && ln -s "$_active" "$_ws_link"
    done
    unset _axis _active _ws_link
  fi
  unset _ws_cfg_dir

  # Seed the active per-gate single sources into the per-loop config home,
  # then render them into the workspace's ref/ours product dirs and freeze
  # the ref side.
  #
  # The single sources belong in <cfg_dir>/<gate_config_dir>/ — a flat
  # sibling of the seven axis files (ref.toml/model.toml/…), so a loop has
  # ONE config home, not a second one buried under <workspace>/config. The
  # committed templates ride into <workspace>/config/eval/<suite>/<gcd>/ via
  # the provision copytree; here we copy that suite's gate dir into the
  # config home (resolved from <cfg_dir>/eval.toml [suite].name +
  # gate_config_dir). The renderer then reads <cfg_dir>/<gcd>/ and writes
  # both products. --skip-if-absent keeps this a no-op for legacy suites
  # (flat registry, no [suite]/gate_config) so coexistence stays additive.
  #
  # Done BEFORE the cfg-dir freeze (so the seeded sources land while the dir
  # is writable, and are frozen alongside the axes): ref/config is frozen
  # read-only (frozen truth the L0 ref scripts read) while workload/src/config
  # stays writable for the agent. cwd is the workspace (agent-loop.sh cd's
  # there first), so `python -m tools.render_gate_configs` resolves.
  local _ws
  _ws="$(dirname "$cfg_dir")/workspace"
  if [[ -d "$_ws" ]]; then
    if [[ -f "$cfg_dir/eval.toml" ]]; then
      local _suite_name _gcd _src_gates
      read -r _suite_name _gcd < <("$PYTHON" - "$cfg_dir/eval.toml" <<'PY'
import sys, tomllib
try:
    suite = tomllib.load(open(sys.argv[1], "rb")).get("suite", {})
    print(suite.get("name", ""), suite.get("gate_config_dir", "gate_config"))
except Exception:
    print("", "")
PY
)
      if [[ -n "$_suite_name" ]]; then
        # Prefer a meta_harness-staged NESTED gate source at
        # <cfg_dir>/eval/<suite>/<gcd>/ (carries the per-loop autotune-merged
        # topology). Fall back to the workspace's harness/config copy for the
        # human new-looptask flow, which has no nested staging.
        _src_gates="$cfg_dir/eval/$_suite_name/$_gcd"
        [[ -d "$_src_gates" ]] || _src_gates="$_ws/config/eval/$_suite_name/$_gcd"
        if [[ -d "$_src_gates" ]]; then
          mkdir -p "$cfg_dir/$_gcd"
          cp "$_src_gates"/*.toml "$cfg_dir/$_gcd/" 2>/dev/null || true
        fi
      fi
      unset _suite_name _gcd _src_gates
    fi
    ( cd "$_ws" && "$PYTHON" -m tools.render_gate_configs \
        --workspace "$_ws" --config-dir "$cfg_dir" --skip-if-absent ) || {
      echo "ERROR: gate-config render failed for $cfg_dir" >&2
      return 2
    }
    # Freeze-fill the deployment <runtime> sentinels (MASTER_PORT /
    # CHECKPOINT_ROOT / MEGATRON_ROOT) with machine-INDEPENDENT values so no
    # frozen product ships a "<runtime>" placeholder. Runs on the LAUNCH box
    # after render, before the ref/config freeze; machine-dependent resolution
    # (repo_root lift, physical storage) stays on the exec side. Guarded on
    # products existing so legacy suites (no gate products) stay a no-op.
    if ls "$_ws"/ref/config/*.toml "$_ws"/workload/src/config/*.toml >/dev/null 2>&1; then
      ( cd "$_ws" && "$PYTHON" -m tools.resolve_deploy \
          --workspace "$_ws" --config-dir "$cfg_dir" \
          --loop-id "${LOOP_ID:?LOOP_ID required for deploy fill}" ) || {
        echo "ERROR: deploy fill (resolve_deploy) failed for $cfg_dir" >&2
        return 2
      }
    fi
    [[ -d "$_ws/ref/config" ]] && { chmod -R a-w "$_ws/ref/config" || true; }
  fi
  unset _ws

  # Freeze the per-loop config dir so nothing downstream (agent edits,
  # stray web PUTs, stage-2 fan-out) can mutate the active config. Mirrors
  # the server-side 409 the web router returns once status != "draft".
  chmod -R a-w "$cfg_dir" || true
}
