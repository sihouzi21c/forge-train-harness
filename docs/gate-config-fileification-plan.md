# 门禁配置文件化 —— 代码修改计划

配套文档：设计动机与三时刻机制见 `gate-config-fileification-design.md`；逐参数落位见
`gate-parameters-reference.md`。本文件按"探索每个已知门禁"的结果，落到**具体文件 / 函数 /
每个门禁**的改动清单，并标出与设计相悖、需要额外处理的细节。

> 阅读顺序：§0 统一原则(方向定稿)→ §1 冲突总览(C1–C10)→ §2 跨门禁公共改动
> → §3 逐 runner → §4 特例详解 → §5 迁移顺序。同一改动点在 C 表只给结论，
> 详细动作在 §2/§3/§4，不重复展开。

---

## 0. 统一原则（方向定稿，2026-07-03 校正）

早期草案在多处写"输出路径由**引擎按约定派生**（`artifact_root`+gate）"。**这条已废弃**，
改为下面的统一原则——更简单、更贴"ours 只拿一个配置文件"的核心要求，且不需要把编排
逻辑塞进引擎：

1. **ours 唯一输入面 = 一个 `--config <product>`**。除产物路径外，harness 不向 ours
   子进程传任何 env / CLI 参数。ours 无法独立运行、无法硬编码外部路径。

2. **一切 ours 运行时需要的值，都由 harness 提前写进配置文件**，按 **gate + 第几次运行**
   区分：
   - **freeze 时（启动机）** 写机器无关的静态值：shape、`[env]` 静态项、
     `CHECKPOINT_ROOT`（含按 gate 区分的 `ones/no1` 子目录，**相对路径**）、
     `.resources/...` 相对路径。
   - **dispatch 时（执行机）** 按 **(gate, run-iteration)** 组一份 **run-config**
     （= 冻结产物 ⊕ 本次运行专属值：`capture_output_file` / `save_path` / `start_step` /
     `resume_from` / `scratch_dir`），落在 `<artifact_dir>/` 下，`--config <run-config>`
     传给 ours。单次运行的门禁，run-config 基本 = 冻结产物（顶多叠一个 capture 路径）；
     多段门禁（production 4 段、resume 存/恢复 2 次）每次运行各得一份 run-config。

3. **ours 从配置读所有值，读逻辑由 dev-agent 实现**。dev-agent 没把某个键读进引擎、
   或读错，就会与 ref 不一致 → 门禁挂。这就是"改不对必过不了门禁"。

4. **agent 可访问面限制在 loop workspace 内**。所有路径一律 **相对 `repo_root`（= workspace
   目录）**，靠 `cwd=repo_root` 解析。**已验证**：`launch_dp.py:135` 的子进程
   `subprocess.Popen(cmd, env=env)` 不传 `cwd=`，继承父进程 cwd；dispatcher 以
   `cwd=repo_root` 起 launch_dp → 每个多卡 ours 子进程都继承 `cwd=repo_root`，相对路径全程正确。
   （`repo_root` = `config_runtime.repo_root()`，本地 `.artifacts/forge_train/<id>/workspace/`，
   远程 `<用户填的 workspace>/.forge_train/<id>/`，运行时由 `FORGE_REPO_ROOT`/cwd 现算，
   **不进配置文件**。）

5. **repo 外的依赖一律经 manifest → `.resources/` 软链**。tokenizer / megatron 源码 /
   训练数据 / canonical checkpoint 都通过 `external_resources.toml` 声明，`harness resources
   provision` 在执行机把它们软链到 `<workspace>/.resources/...`；产物里只存相对 `.resources/...`。
   manifest 是**唯一允许写主机绝对路径的边界文件**（`sources = [file://<abs>]`），归
   **meta_harness 侧**管、materialize 进每个 loop 工作区，**dev-agent 不碰**。

> 三个关键事实（探索确认）：
> - Megatron resolver（`config_runtime.py:279 _resolve_megatron`）已支持级联：本地路径 →
>   `<workspace>/.resources/megatron/*` → vendored submodule，返回运行时 `.resolve()` 绝对值。
>   **迁移不用改 resolver**，只需清空 `[ref].megatron` 让它落到 `.resources` 分支 + manifest 填条目。
> - **manifest 当前是死的**：`resources.manifest_path()` 找 `<repo_root>/harness/external_resources.toml`
>   （内层），实际提交的文件在 `<repo_root>/external_resources.toml`（外层），`exists=False` →
>   `load_manifest()` 恒返回 `[]`。**启用前必须修这个内/外层路径 bug**（docstring 写的是
>   `harness/external_resources.toml`，倾向去掉 `manifest_path` 里多余的 `/harness/`）。
> - 冻结不动 ours 产物：lease 只 `chmod -R a-w` `ref/config` 与 config dir，`workload/src/config`
>   保持可写，因此 dispatch 侧组 run-config、freeze 侧写 ours 静态值都可行。

---

## 0.1 范围

- **收编**（stage1，走 render → 产物）：`forward-align` `backward-align` `multistep-1gpu`
  `multistep` `perf-bitwise` `resume-gate-20` `long-train` `long-train-smoke` `loss-gate-200`
  `production-train`。
- **特例**（stage1 但当前无 gate_config 产物）：`resume-startup-90`（inline shape）、
  `profile-snapshot`（`mirror_gate` 镜像，无自有 shape）。见 §4。
- **不收编**（stage2，注册表已声明"本次不收编"）：`op-inventory` `op-long` `op-status`。
  它们仍用 legacy `[ref_env]`/`[ours_env]` inline，本期不动。

三套件 `dense_training` / `dense_training_1b` / `dense_training_8b`（+ `_qwen3`）门禁集合相同，
改动对四套件对称生效（8b 多一个 `dptp.toml`，同 stage1 处理）。

---

## 1. 探索发现：与设计相悖 / 需额外处理的点

| # | 现状 | 与设计的冲突 | 处理（按 §0 定稿） |
|---|---|---|---|
| C1 | `dense_training.toml [runtime.distributed] master_port = "auto"` → 渲染 `<runtime>` → dispatch 时 `_common._allocate_free_port` | 设计要 freeze 填实端口 | freeze 时把 `auto` 解析成确定端口（独占 devspace + 门禁串行；`config_runtime.py:1028` 已支持），写进产物 `[env].MASTER_PORT`。（已实现：`resolve_deploy.derive_master_port`） |
| C2 | `[evals.<gate>].env_inputs` 驱动 `build_suite_env` 在 dispatch 提升键进子进程 env | ours 只收 `--config`，env 提升取消 | `env_inputs` 语义改为"**freeze 必须填实进产物的键清单**"；ours 侧运行时不再提升任何 env |
| C3 | `resume-startup-90` shape 写在注册表 inline，无 gate_config，不渲染产物 | 设计要"每 gate 一份产物" | 建 `gate_config/resume-startup-90.toml` + render 产物；执行输入迁 gate_config，判定阈值迁 eval.toml（§4.1） |
| C4 | `profile-snapshot` 无 gate_config，用 `mirror_gate` + inline `num_steps`/`warmup_steps` | 无自有产物 | freeze 时按 `mirror_gate` 拷被镜像 gate 产物再叠加 inline 项；非 commit gate，优先级低（§4.2） |
| C5 | `_run_production_train` 在 dispatch 把 10000 步切 4 段，逐段注入 `NUM_STEPS/START_STEP/FORGE_SAVE_PATH/FORGE_RESUME_FROM`，靠 `_latest_production_checkpoint` 扫盘 | 早期草案想搬进引擎；**已否决** | **dispatcher 保留分段编排**，但改为：每段组一份 run-config（写 `num_steps/start_step/save_path/resume_from`）→ `--config` 传入；**引擎只读 config 跑一段**，不自扫盘、不自分段。转输从 env 注入改为配置文件（§4.3） |
| C6 | `_run_resume_gate` 注入 shape（产物已有 → 双灌）+ `RESUME_SCRATCH_DIR`（dispatch 派生） | shape 双灌；scratch dispatch 派生 | 删 shape 双灌（从产物读）；`RESUME_SCRATCH_DIR` 由 **dispatcher 写进 run-config**，ours 从 config 读（保留"落挂载卷、不进 `.artifacts`/`/tmp`"约束，`dispatcher.py:3104`） |
| C7 | `CHECKPOINT_ROOT` 的 `ones/no1` 重定向由 `_common.forge_init_ones_checkpoint_root` 在 dispatch 算 | dispatch 派生 ours 路径 | **freeze 时**按 gate 把 `ones/no1` 拼进产物的**相对** `CHECKPOINT_ROOT`；ours 直接读。**删掉消费端派生**（含已提交 Step4 的 `_gate_entry.checkpoint_root()` lift/派生，见 §5 回退） |
| C8 | align gate：`run_candidate_capture` 注 `FORGE_CAPTURE_OUTPUT_FILE` + `--hash-*` CLI | 路径/CLI 直传 | `hash_capture_level` 判定语义 → 从冻结 eval.toml 读；`persistent` 从产物；**capture 输出路径由 dispatcher 写进 run-config**，ours 从 config 读，dispatcher 用同一 run-config 读回 |
| C9 | 判定阈值 `_overlay_product_verdict(..., side="ref")` 从 ref 产物读回阈值 | 设计改为阈值统一进 eval.toml | **删 `_overlay_product_verdict`**：判定阈值全迁注册表 `[evals.<gate>]`（冻结 eval.toml），dispatcher 直接读 `workload_config["evals"][suite_key]` |
| C10 | ours 产物不冻结（freeze 只 chmod `ref/config` 与 config dir） | 曾担心 agent 篡改 deployment 值 | 保持不冻结：正因不冻结，dispatch 才能组 run-config、freeze 才能写 ours 静态值；agent 改坏 → 与 ref 不一致 → 门禁挂（符合"改不对过不了"） |
| **C11** | 外部依赖（tokenizer/megatron/data/ckpt）全绝对路径直连；manifest 空且路径 bug（§0 注） | 违背"agent 只访问 loop 内 + 相对路径" | 迁 `.resources/` 软链：配置字段绝对→相对，manifest 填条目并归 meta_harness 管，修 `manifest_path` bug |

---

## 2. 跨门禁公共改动（做一次，全门禁受益）

### 2.1 freeze：`tools/resolve_deploy.py`（已落地，需扩展）+ `tools/agent_loop_lease.sh`
freeze 跑在**启动机**、门禁跑在 **devspace**——只有机器无关的值能在 freeze 填。已实现
`resolve_deploy` 填 `MASTER_PORT`（C1）/ `CHECKPOINT_ROOT`（相对约定）/ `MEGATRON_ROOT`。
**本次扩展**：

- `CHECKPOINT_ROOT` 填成**按 gate 区分的相对含 `ones/no1`** 值（C7）：freeze 读产物
  `[cli].forge_init_ones`，写 `.artifacts/checkpoints/<backend>/{ones,no1}`（无该字段的 meta
  gate 写裸根）。→ 消费端不再派生。
- 相对约定路径的落盘：`agent_loop_lease.sh` 在执行侧 freeze 前挂 `harness resources provision`
  建 `<workspace>/.resources/*` 软链；ckpt / megatron / tokenizer / data 的相对路径由此可解析
  到真实挂载（workspace 盘小/临时，大文件经软链落挂载卷）。
- **暂缓**：`resolve_assets` 资产下载（往执行机缓存落数据的副作用，harness 基本没用过）。
- 冻结顺序不变（填完再 chmod `ref/config` 与 config dir；`workload/src/config` 不冻结，C10）。

### 2.2 ours 读取器：`evals/scripts/_gate_entry.py`
- `GateInputs` **只从 `--config` 指向的 run-config 文件读**（已落地 argv 解析 + product-first）。
- **回退 Step4 的 `checkpoint_root()`**：删掉 `_repo_root()` lift + `ones/no1` 派生，
  `CHECKPOINT_ROOT` 改为直接 `require("CHECKPOINT_ROOT")` 读产物已填好的相对值（cwd 解析绝对）。
- `_RUNTIME_ENV_KEYS` 收缩到 `RANK/LOCAL_RANK/WORLD_SIZE`（已落地）。**新增**从 config 读
  `START_STEP`/`FORGE_SAVE_PATH`/`FORGE_RESUME_FROM`/`FORGE_CAPTURE_OUTPUT_FILE`/`RESUME_SCRATCH_DIR`
  ——这些现在由 dispatcher 写进 run-config，不再从 env 读。
- legacy `FORGE_GATE`+`FORGE_OURS_CONFIG_DIR` 回退：最终步删除。

### 2.3 dispatcher：`evals/dispatcher.py`
- **组 run-config**：新增一个 helper（如 `_compose_run_config(frozen_product, artifact_dir, overrides) → Path`），
  把冻结产物 ⊕ 本次运行专属值写成 `<artifact_dir>/run_config.toml`，`--config` 指它。
  各 runner 的 `extra_env` 注入 → 改为写进 run-config 的 `[env]`/`[cli]`。
- `suite_process_env` 瘦身：不再注 `FORGE_GATE`/`FORGE_OURS_CONFIG_DIR`/`CHECKPOINT_ROOT` 重定向
  （已删）；distributed rendezvous（`MASTER_*`/`NUM_PROCS`）仍按需注（rendezvous 是运行时身份，
  非 gate 参数）。
- 读回：capture/hash/ckpt 路径 dispatcher 从它**写进 run-config 的同一值**读回（不猜约定）。
- **删 `_overlay_product_verdict`**（C9）+ `_gate_hash_capture_level` 改**只读** eval.toml。
- **删 `_latest_production_checkpoint`**？→ 否，保留（dispatcher 仍负责扫最新 ckpt 决定下一段
  `resume_from`，写进 run-config；引擎不扫盘，C5）。

### 2.4 ref 投影：`tools/gate_product_to_shell.py` + `evals/_common.py`
- freeze 后 ref 产物 `[env]` 无 `<runtime>`；emitter 对三个部署 key（`_DEPLOYMENT_ENV_KEYS`）
  **恒跳过**（已落地），其余 `[env]` 全导出。
- `_common._ref_script_path_env` 注入的 `MEGATRON_ROOT`/`DATA_PATH`/`DATA_CONF`/`SAVE_PATH`/
  `TENSORBOARD_DIR`/`FORGE_TOKENIZER_DIR`/`FORGE_DATA_DIR`/`INIT_ONES` → 逐步改为 freeze 填进
  ref 产物 + `.resources` 相对；**MEGATRON_ROOT 绝对注入删除**，ref 侧读相对 `.resources/megatron`。
- `FORGE_REF_CONFIG_DIR`/`FORGE_GATE`/`FORGE_DATA_TOML` 指针最终删除。

### 2.5 外部资源 manifest：`external_resources.toml` + `harness/resources.py`（C11）

> **决定（2026-07-03，已定稿）：本节除 `manifest_path` 修复 + sha256 可选外，
> 其余 manifest 化 + meta_harness materialize 一律 WON'T-DO（判定为过度设计）。**
> 查证:`harness.resources`/`provision` 建好且有单测,但**是死代码** —— 唯一调用方是手敲
> `harness resources provision` CLI,agent-loop.sh / workspace bootstrap / web / lease 都不接。
> 而它要治的东西早被别的机制解决:
> - tokenizer 随 ref bundle 走(`meta_harness/workload/input/<model>_hf_singlecard/tokenizer/`
>   自带真 SentencePiece `tokenizer.model` → copytree 进 `ref/reference/<tag>/tokenizer/`;
>   `run.sh` 用 `FORGE_TOKENIZER_DIR=${FORGE_TOKENIZER_DIR:-$SCRIPT_DIR/tokenizer}`)。
> - megatron:`_resolve_megatron` 已级联 local → `.resources/megatron/*` → submodule。
> - checkpoint:`_derive_checkpoint_root` 约定 `.artifacts/checkpoints/<backend>`;freeze 已写相对
>   `CHECKPOINT_ROOT`(含 ones/no1)。
> - data 语料 `/opt/forge-data/ultra_fineweb` 是大 scratch 挂载盘,**本就该绝对**,不该塞进每 loop 的 `.resources`。
> - 当年"烧 12 分钟"的 git-clone-Megatron-on-demand 早删了。
>
> tokenizer 字段**也不改**:harness `config/ref/*.toml` 的 `forge_tokenizer_dir=/opt/forge-data/tokenizer`
> 只被"直连 CLI"用,那里 devspace 路径本就对;meta 路径自带 bundle 不用它;`resolve_assets` 硬要求该字段
> 非空,置空反而搞坏直连路径。已落地的仅:sha256 可选(commit `2778673c3` step 5.8a),让 manifest
> 可离线 author —— 万一将来真要用,注意它得先被**接进 loop bootstrap**。
>
> 以下为原始计划,保留作背景,不再执行:

- ~~**修 `manifest_path` 内/外层 bug**（§0 注）：让它指向真实提交位置。~~（已做,step 5.1 `99bbff439`）
- ~~**配置字段绝对→相对**：`[ref].forge_tokenizer_dir`、`[data].data_path`/`conf_path`、
  `[ref].megatron`（清空走级联）、`CHECKPOINT_ROOT` → 全部 `.resources/...`。~~（WON'T-DO,见上）
- ~~**填 manifest 条目**（清单见下）。~~（WON'T-DO）
- ~~**归属 meta_harness**：manifest 作为 authored 产物纳入 `meta_harness/config`，由
  `meta_harness/scripts/_shared/assemble_harness_configs.py` 的 validate/materialize 覆盖，
  materialize 进每个 loop 工作区的正确位置。dev-agent 不编辑。~~（WON'T-DO）

**需要进 manifest 的内容：**

| 依赖 | 现状（绝对） | canonical_relpath | kind | 触发条件 |
|------|-------------|-------------------|------|---------|
| tokenizer | `[ref].forge_tokenizer_dir=/opt/forge-data/tokenizer[/…]` | `.resources/tokenizer/<model>/` | `tree` | 按 model 轴（0.5b/1b/qwen3/8b） |
| canonical ckpt | `<ckpt>/{ones,no1}/canonical_state_fp32.pt` | `.resources/checkpoints/<backend>/{ones,no1}/canonical_state_fp32.pt` | `file`×2 | bitwise/align 门禁 |
| Megatron 树 | `_ref_script_path_env` 注入绝对 | `.resources/megatron/<ver>/` | `git-checkout`/`tree` | 仅 megatron backend |
| 训练数据 | `[data]` DATA_PATH/DATA_CONF 绝对 | `.resources/data/<dataset>/` | `tree` | 按 data 轴 |

---

## 3. 逐 runner_kind 改动

`RUNNER_KINDS`（`dispatcher.py:401`）逐项。"删注入"= 删该 runner 的 `extra_env`；
"写 run-config"= 该值由 dispatcher 写进 `<artifact_dir>/run_config.toml`，ours 从 config 读。

| runner_kind | 门禁 | 当前 dispatch 注入 | 目标（§0 定稿） |
|---|---|---|---|
| `forward-align` / `backward-align` | forward/backward-align | `FORGE_CAPTURE_OUTPUT_FILE` + `--hash-*` | capture 路径写 run-config；`hash_capture_level` 从冻结 eval.toml 读；dispatcher 读回（C8） |
| `stage1-bitwise-trajectory` | multistep-1gpu / multistep / perf-bitwise | 仅 `suite_process_env` | `--config` 启动；MFU 门由产物 `mfu_e2e_target` 驱动（perf-bitwise=10） |
| `resume-gate` | resume-gate-20 | shape 双灌 + `RESUME_SCRATCH_DIR` | 删双灌；`RESUME_SCRATCH_DIR` 写 run-config（C6） |
| `resume-startup` | resume-startup-90 | 读 inline shape via env_inputs | 建 gate_config + 产物（C3），再按 resume-gate 同法 |
| `long-train` | long-train / long-train-smoke | 仅 `suite_process_env` | `--config` 启动 |
| `loss-gate` | loss-gate-200 | 同上 | 同上；判定用产物 `max_avg_relative_loss_diff` |
| `production-train` | production-train | 分段 4×`NUM_STEPS`/`START_STEP`/`FORGE_SAVE_PATH`/`FORGE_RESUME_FROM` | dispatcher 每段组 run-config（含 resume_from，由扫盘决定）；引擎只读 config 跑一段（C5） |
| `profile-snapshot` | profile-snapshot | `FORGE_NSYS_RANK0_OUTPUT` + mirror shape | mirror 产物（C4）；nsys 路径写 run-config |
| `op-inventory`/`op-long`/`op-status` | stage2 | legacy inline | **不动**（范围外） |

---

## 4. 特例门禁详解

### 4.1 `resume-startup-90`（C3）
现状：shape inline 在 `dense_training.toml [evals.resume-startup-90]`（`world_size=2`
`micro_batch_size=4` `seq_length=4096` `grad_accum_steps=10` `resume_save_step=90`
`resume_post_steps=10` `resume_startup_budget_s=18.0`），无 gate_config。
- **改**：新建 `config/eval/<suite>/gate_config/resume-startup-90.toml`，把**执行输入**迁入
  `[shared]`/`[shared.gate]`；render 自然产出产物。之后同 `resume-gate` 处理（scratch 写 run-config）。
- `resume_startup_budget_s` 是 ours-only **判定阈值** → 放注册表 `[evals.resume-startup-90]`
  （冻结 eval.toml），dispatcher 从 eval.toml 读，不进产物。

### 4.2 `profile-snapshot`（C4）
现状：`mirror_gate = { "bitwise-perf"="perf-bitwise", "long-horizon"="long-train" }` +
inline `num_steps=12` `warmup_steps=5`，无 gate_config。非 commit gate（诊断）。
- **改**：freeze 时按 `mirror_gate` 把被镜像 gate 已填实的产物拷成 `profile-snapshot` 产物，
  再叠加 inline 诊断项。nsys 输出路径写 run-config。优先级最低。

### 4.3 `production-train`（C5，转输方式改，编排仍在 dispatcher）
- ours-only，无 `[ref]`。**引擎不新增分段/扫盘逻辑**——只读 run-config 的
  `num_steps`（本段步数）/`start_step`/`save_path`/`resume_from` 跑一段、按需周期存 ckpt。
- dispatcher `_run_production_train` 保留段循环：每段
  ① 扫 `<save_root>` 取最新 ckpt（`_latest_production_checkpoint`，保留）；
  ② 组 run-config 写该段 `start_step`/`num_steps`/`save_path`/`resume_from`；
  ③ `launcher script --config <run_config>` 跑；超时 kill 后重跑同段。
- `save_root` 走相对 `.resources`/挂载卷，不落 workspace 盘。
- 数据就绪门 `_await_prefetch_if_configured` 保留。

---

## 5. 迁移顺序（建议）

**先做最小闭环（checkpoint 一条端到端）：**
1. ~~**修 `manifest_path` 内/外层 bug**（§0 注 / §2.5）——单测覆盖。~~（已做,step 5.1 `99bbff439`）
2. **回退 Step4 的 `_gate_entry.checkpoint_root()` 派生**（§2.2）：删 lift + `ones/no1`，改直接读产物。
3. **freeze 按 gate 把 `ones/no1` 写进相对 `CHECKPOINT_ROOT`**（`resolve_deploy`，§2.1 / C7）。
4. `multistep-1gpu` 端到端验证（DP=1，最小形状）：ours 读到含 `ones/no1` 的相对 ckpt，cwd 解析正确。

**再铺开：**
5. **run-config 组装 helper**（§2.3）+ align/resume/profile 的 capture/scratch/nsys 值写 run-config（C6/C8/C4）。
6. **production 段循环转 run-config**（C5，§4.3）。
7. **resume-startup 建 gate_config**（C3，§4.1）。
8. ~~**manifest 化 tokenizer / megatron / data**（C11，§2.5）+ meta_harness materialize（§2.5 归属）。~~
   **WON'T-DO(过度设计,见 §2.5 决定框)**;仅保留 sha256 可选(step 5.8a `2778673c3`)。
9. **ref 折 env**（§2.4）：对称、风险最低。
10. **判定阈值迁 eval.toml + 删 `_overlay_product_verdict`**（C9）；stage2 不动。

**风险**：`RESULT_BEGIN/END`、`[LOSS]` stdout 判定线不动 → 判定零风险；回归集中在
"ours 能否从 run-config + `.resources` 拿全"。每步先单 gate 验证再铺开。

~~**meta_harness 侧联动**（与 §2.5 / 步骤 8 同批）：`external_resources.toml` 纳入
`meta_harness/config`，`assemble_harness_configs` 的 validate 增加 manifest 校验、materialize
把它写进 loop 工作区正确位置。此改动跨 harness / meta_harness 两侧，需一起提交。~~
（**WON'T-DO**,见 §2.5 决定框:manifest 化整体取消,无跨仓联动。）
