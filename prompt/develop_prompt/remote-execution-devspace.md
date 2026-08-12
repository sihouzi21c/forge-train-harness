# Devspace Lifecycle Overlay (injected only when `[remote].kind = "devspace"`)

> Appended after `remote-execution.md` when `config/remote.toml`
> `[remote].kind = "devspace"`. The transport contract (sync / launch /
> log retrieval) is in that shared overlay; this file adds the
> **lifecycle** the harness manages for an auto-provisioned `cctl`
> devspace: how to recover when the leased devspace drops mid-run, and
> how to keep the lease registry pointed at the live machine so the
> wrapper stops the right one on exit.
>
> `@@KEY@@` placeholders are substituted at inject time exactly as in
> the shared overlay.

## Recovery when the devspace drops

If the devspace drops (idle timeout, host restart, quota reclaim,
spot preemption, out-of-inventory) **you must re-create it yourself
before surfacing any blocker**. Local fallback is unsafe for bitwise
gates and is forbidden.

Procedure (use `cctl <subcmd> --help` for exact flags):

1. Read the devspace spec from the per-loop config
   (`$FORGE_CONFIG_DIR/remote.toml`): `project`, `cluster`,
   `resource_pool`, `image`, `gpu_count`, `gpu_model`, `priority`.
   The dead alias is `[remote].hostname` (`ds-<OLD_ID>`); its id is
   the trailing segment. The dir is frozen (`chmod -R a-w`) for
   safety; the recovery below temporarily lifts that to rewrite the
   hostname.
2. `cctl devspace create -o json` with the spec flags
   (`--project`/`--cluster`/`--resource-pool`/`--image`/`--gpu`/
   `--gpu-model`/`--priority`) to provision a fresh task; capture the
   server-assigned `id` (`<NEW_ID>`) from the JSON.
3. Poll `cctl devspace get tasks/<NEW_ID>` until `Running`, then
   `tsh ls --format json` until a node's `spec.hostname` ends in
   `-<NEW_ID>` — that is the reachable Teleport node. Add a matching
   `Host ds-<NEW_ID>` stanza to `~/.ssh/config` (HostName
   `<node>.teleport.cybertron.modelbest.co`, `User root`, the fixed
   Teleport `ProxyCommand`).
4. Bootstrap SSH key + git `safe.directory` / user.name + HF env
   on the new host (same recipe the initial bootstrap used).
5. Rsync the local worktree across; re-bootstrap
   `config/eval.toml` from `config/eval/dense_training.toml`.
6. `chmod u+w "$FORGE_CONFIG_DIR/remote.toml"`, update
   `[remote].hostname` to the new alias `ds-<NEW_ID>`, then
   `chmod a-w "$FORGE_CONFIG_DIR/remote.toml"` to restore the
   freeze.
7. **Re-point the lease** at the recovered host so the wrapper's
   exit-time `cctl devspace stop` targets the live machine instead
   of the dead original (otherwise the recovered copy leaks and
   keeps burning quota):

   ```bash
   "$PYTHON" -m tools.lease rebind devspace \
     --loop-id @@REMOTE_LOOP_ID@@ --host ds-<NEW_ID>
   ```

   The `.artifacts/lease/` registry is **not** frozen, so this call
   works without lifting any `chmod`. `rebind` only re-points the
   registry + reverse-lookup file; it does not stop the old machine
   (it already dropped).
8. Resume by re-running the gate that triggered the drop — do not
   skip ahead.

Only surface a blocker if `cctl` itself fails (auth / quota /
cluster down). Going straight to "remote is gone, please
re-provision" without attempting `cctl devspace create` is a
procedural failure — every prior multi-round out-of-inventory
streak in `perf_log.md` was recoverable by this path.
