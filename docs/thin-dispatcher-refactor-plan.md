# 门禁运行薄化重构计划（thin dispatcher）

状态：§10 步骤 1–7 全部完成（1–5 真机验证于 devspace 566168；6/7 均经
HEAD 基线全量单测比对——0 新失败、净修复 10 项）· 2026-07-15
范围：stage1 全部门禁。**stage2（op-inventory / op-long / op-status）明确豁免，本次不动。**

步骤 7 收尾结论（stage2 抽离 + 注册表同步，本机全量 unittest 基线 diff 0 新失败）：
- 代码面：op-* handler 及其专属 helper 逐字移入 `evals/dispatcher_stage2.py`
  （810 行,`RUNNER_KINDS` 归其所有,dispatcher.py 再导出保持
  `dispatcher.RUNNER_KINDS` 地址不变且 `is` 同一）;dispatcher.py 1145→~345 行,
  只剩通用三步执行器。`_gate_hash_capture_level` 留在 dispatcher.py
  （`_run_scripted_ref` 缓存策略消费）;`_require_cfg` 死代码删除。
- 注册表面:dense_training_1b / _8b / _qwen3 三套件 stage1 全部 gate 补齐
  `needs_ref`/`ours_runner`/`verdict`(+profile-snapshot `requires_label`),
  值与 0.5B 目标态逐 gate 矩阵校验零偏差(8B 的 dptp 对齐 multistep 的 bitwise
  路由;8B 无 profile-snapshot/op-*)。stage1 `env_inputs` 全删,op-long 的保留。
- 测试面:test_dispatcher_behavior(handler 调用/patch 目标/阈值方向扫描)、
  test_gate_transport_parity、test_runner_metrics_contract(源池加
  dispatcher_stage2.py)、test_suites_shape(resolve_ref_trajectory 消费方断言
  重定位)重定位到 dispatcher_stage2;test_layer_dag 登记新模块并收紧
  dispatcher 边;3 处 docstring 引用(gate_shape/op_long_ours/ref_script_runner)
  同步。dispatcher.py 文档串补 `config/eval.toml` 出处(套件清单 SSOT 断言)。

步骤 6 收尾结论（本机全量 unittest，与删除前 HEAD 做 git-worktree 基线 diff）：
- 已删：全部 stage1 handler（~2500 行）、`compose_run_config` 及调用点、
  `_common` env 叠层族、`GateInputs` 的 FORGE_GATE/FORGE_OURS_CONFIG_DIR 兜底
  （改为强制 `--config <product.toml>`）+ env 兜底收紧到 `_ENV_FALLBACK_KEYS`
  白名单、stage1 注册表 `env_inputs` 键；ref 缓存已确认即 §7.3 新 key（无需改）。
- §8 删除清单两处**有意保留**（与原表不一致，记录如下）：
  1. `script`/`launcher` 键保留——`runtime_env.py _ours_overrides` 消费它们导出
     FORGE_OURS_ENTRY/FORGE_OURS_LAUNCHER，是 ours sh 的入口路由源；
  2. stage1 的 `runner_kind` 键保留——不再参与路由（`ours_runner` 优先），但
     `app.py _suite_contracts` 以它为 `RUNNER_METRICS` 合同键（run_schema 在
     "不动"清单内，不重键）。仅 `env_inputs` 按表删除（现为 stage2 专属键）。
- 附带发现（易踩坑）：runtime_env.py 的 FORGE_GATE/FORGE_OURS_CONFIG_DIR 导出
  **不是** GateInputs 遗留兜底——引擎 `workload runtime_config._product_cli`
  靠这对指针定位产物读 `[cli]` 超参，新路上 runtime_env 是唯一导出方，必须保留。
- 测试面：9 个测试文件重定位到 verdicts/* 与 run_ours*.sh 契约；align 接线静态
  断言改扫 `verdicts/align.py` 的 milestone 分支；prefetch 解析断言改打
  `production_ckpt.prefetch_wait`。

步骤 5 验证结论（真机，假引擎等价矩阵）：
- production-train：fresh / all-skip resume 两态 legacy-vs-new 归一化 diff 均 EQUIVALENT；
  crash-resume 杀段测试通过（杀第 3 段 → rc=143 + checkpoint MISSING 判失败，重跑自动
  从 step_20 续，segments.jsonl 只含补跑段）。
- profile-snapshot：新路真 nsys 端到端通过；legacy 路对照发现**存量缺陷**（见 §11 表末行），
  临时补丁放行后归一化 diff EQUIVALENT。
- 回归：multistep 重跑与 step-2 基线 diff 仅剩 ref 侧耗时波动，env-first 白名单无漂移。

---

## 1. 目标与原则

现状：`evals/dispatcher.py`（3353 行）为每个 runner_kind 各写一个 handler，内含
复杂的参数决议（shape 读取、env 叠层、per-run config 合成、判分阈值 overlay），
增删门禁必须改 dispatcher。

目标：dispatcher 不再理解任何具体参数，退化为通用三步执行器：

```
① 调 ref 侧一个 sh   → 往 run_dir 落产物
② 调 ours 侧一个 sh  → 往 run_dir 落产物
③ 调 harness 侧一个判分脚本 → 读两侧产物 + 阈值 → pass/fail
```

设计原则：

1. **两侧 sh 各自直读渲染产物**（`ref/config/<gate>.toml` /
   `workload/src/config/<gate>.toml`）获得全部参数数值。
2. **sh 内禁止按 gate 名分支**。行为差异一律由产物里的参数值驱动
   （如 `hash_capture_level`、`nsys_profile`、`ref_script`、`ours_entry`）。
3. ours 侧允许多个 sh，但按**运行形态**拆（单次 launch / 分段循环），
   不按 gate 拆。
4. 两侧 sh 只负责「跑 + 落盘」，完全不知道比较逻辑；判分独立成第三步。
5. 增删一个门禁 = 改注册表 + 加/删 `gate_config/<gate>.toml`
   （判分方式新颖时再加一个判分脚本），dispatcher 零改动。

---

## 2. 目标架构

```
harness transport
  └─ python3 -m evals.runner run <request.json> <repo_root>     (不变)
       └─ dispatcher.run_suite()          ← 通用执行器，~300 行
            run_dir = artifact_dir                    # 即现有 artifact_dir
            ① [needs_ref 时，经 ref 缓存]
               bash ref/run_gate.sh <gate> <run_dir>/ref
            ② bash evals/scripts/<ours_runner>.sh <gate> <run_dir>/ours
            ③ python3 -m evals.verdicts.<verdict> <gate> <run_dir>
                  → 写 <run_dir>/verdict.json
            组装 result.json（RESULT_BEGIN/END、run_schema 不变）
```

路由三键写在注册表每个 `[evals.<gate>]` 段里，dispatcher 只做机械转发：

```toml
[evals.multistep-1gpu]
needs_ref   = true
ours_runner = "run_ours"            # → evals/scripts/run_ours.sh
verdict     = "bitwise"             # → evals/verdicts/bitwise.py
timeout_s   = 600                   # 保留，超时仍归 dispatcher
```

（`runner_kind` / `script` / `launcher` / `env_inputs` 键随迁移完成删除，
见 §8 删除清单。）

---

## 3. run_dir 落盘公约

每次门禁运行的唯一工作目录 = 现有 `artifact_dir`。两侧产物落位是**固定公约**，
路径拼法写死在脚本里，没有任何一方需要"传"路径：

```
<run_dir>/
  ref/
    ref.log               # ref stdout（含 [LOSS] 行）
    ref_hash_dump.json    # 仅 hash_capture_level>0
    ref_capture.pt        # 仅 align 采集
    save/  tb/            # SAVE_PATH / TENSORBOARD_DIR
  ours/
    ours.log              # ours stdout（含 [LOSS]/[LOSS_REF]/[LOSS_RES]/[RESUME_STARTUP] 行）
    capture.json          # 仅 align 采集（FORGE_CAPTURE_OUTPUT_FILE）
    ours_hash_dump*.json  # 仅 hash_capture_level>0（resume 产 .ref.json/.res.json 两份）
    profile.nsys-rep      # 仅 nsys_profile=1
    run_config_step_*.log # 仅 production（每段一份训练日志）
  verdict.json            # 判分脚本输出
  result.json             # dispatcher 组装（schema 不变）
```

每侧**落什么由参数控制**（hash_capture_level>0 → 多落 hash dump；
nsys_profile=1 → 多落 nsys-rep），落在哪由公约控制。

resume scratch 特例：不能进 run_dir（devspace overlay 50GiB 配额，
见现 dispatcher.py:3123-3132 注释），维持
`<workspace>/tmp/resume_scratch_<basename run_dir>`，由 ours sh 按公式拼。
production 稳定 checkpoint 根（跨 run 存活以支持 crash-resume）同理不进
run_dir，从产物键读。

---

## 4. 新增文件

### 4.1 `tools/product_env.py` — 产物 → export 行（唯一 TOML 读取点）

```
用法：eval "$(python3 tools/product_env.py <product.toml>)"
```

**完全通用：规则只基于值的形态，不含任何键名注册表。**
配置里新增任意键自动透传，永不需要改本脚本：

1. `[env]` 段：逐键原样 `export KEY="value"`。
2. `[cli]` 段：逐键升格 `export UPPER(key)="value"`。
   升格后与 `[env]` 同名冲突时 `[env]` 优先并 stderr 告警。
3. 标量转换（按类型）：bool → `"1"`/`"0"`；int/float → str；
   list → 空格拼接（`gate_window=[0,10]` → `GATE_WINDOW="0 10"`）；
   全部 `shlex.quote` 防注入。
4. 值哨兵（按值判断，不按键名）：
   - `"<runtime>"` → 不输出，该键留给 os.environ 兜底；
   - `"<auto-port>"` → 现场分配空闲端口后输出（取代现
     `master_port="auto"` 的 dispatcher 运行时填实——渲染器改写
     `<auto-port>` 哨兵，任何键想要此语义写同一哨兵即可）。
5. 键名升格后不是合法 shell 标识符 → fail-fast 报错。

两侧 sh 共用。bash 侧永远不解析 TOML。gate_window 不做键名特判拆分：
sh 不消费它，判分脚本经 `load_gate_product()` 直读产物。

### 4.2 `ref/run_gate.sh <gate> <run_dir>` — ref 侧唯一 sh

```bash
#!/usr/bin/env bash
set -euo pipefail
GATE=$1; RUN_DIR=$2; mkdir -p "$RUN_DIR"
eval "$(python3 tools/product_env.py "ref/config/$GATE.toml")"

# per-run 路径：固定公约，无条件设置
export DUMP_DIR="$RUN_DIR"
export SAVE_PATH="$RUN_DIR/save"
export TENSORBOARD_DIR="$RUN_DIR/tb"

# 值驱动分支 1：采集级别决定走 L0 还是 bridge
if [ "${HASH_CAPTURE_LEVEL:-0}" -gt 0 ]; then
    bash "$REF_CAPTURE_SCRIPT" \
        --hash-capture-level "$HASH_CAPTURE_LEVEL" \
        --hash-output "$RUN_DIR/ref_hash_dump.json" \
        ${REF_PERSISTENT:+--persistent} \
        |& tee "$RUN_DIR/ref.log"
else
    # 值驱动分支 2：跑哪个 L0 脚本由产物 ref_script 决定
    bash "ref/reference/$REF_SCRIPT" |& tee "$RUN_DIR/ref.log"
fi
```

取代现 `_common.run_via_ref_script` 的五层 env 叠层
（L0 preset ← path shim ← 产物指针 ← INIT_ONES ← extra_env）：
产物 `[env]` 已含 MEGATRON_ROOT/DATA_PATH/INIT_ONES 等（B 类 freeze 回填），
per-run 路径由公约拼出，指针 env（FORGE_REF_CONFIG_DIR 等）不再需要——
脚本自己就在读产物。

### 4.3 `evals/scripts/run_ours.sh <gate> <run_dir>` — ours 通用 sh

覆盖除 production-train 外全部 stage1 gate（bitwise×3、long-train、loss-gate、
forward/backward-align、resume-gate、resume-startup、profile-snapshot）。
放 `evals/scripts/`（harness 属地）而非 `workload/`，防 loop agent 篡改判分链路。

```bash
#!/usr/bin/env bash
set -euo pipefail
GATE=$1; RUN_DIR=$2; mkdir -p "$RUN_DIR"
PRODUCT="workload/src/config/$GATE.toml"
eval "$(python3 tools/product_env.py "$PRODUCT")"

# per-run 路径：固定公约，无条件 export（不消费的引擎不读，无害）
export RESUME_SCRATCH_DIR="$PWD/tmp/resume_scratch_$(basename "$RUN_DIR")"
export FORGE_CAPTURE_OUTPUT_FILE="$RUN_DIR/capture.json"

# 值驱动分支 1：nsys 包裹（launch_dp 见到该 env 即包 rank0，不能无条件设）
if [ "${NSYS_PROFILE:-0}" = "1" ]; then
    export FORGE_NSYS_RANK0_OUTPUT="$RUN_DIR/profile"
fi

# 值驱动分支 2：hash 采集三参
HASH_ARGS=()
if [ "${HASH_CAPTURE_LEVEL:-0}" -gt 0 ]; then
    HASH_ARGS=(--hash-capture-level "$HASH_CAPTURE_LEVEL"
               --hash-output "$RUN_DIR/ours_hash_dump" --persistent)
fi

# 入口脚本由产物 OURS_ENTRY 决定（原注册表 script 键，渲染进产物）
python3 evals/scripts/launch_dp.py "evals/scripts/$OURS_ENTRY" \
    --config "$PRODUCT" "${HASH_ARGS[@]}" \
    |& tee "$RUN_DIR/ours.log"
```

关键事实（已验证）：resume-gate 的 save→resume 两阶段在 `eval_resume_train.py`
**进程内部**完成（单次 launch），resume-startup 同理——两者都走此通用脚本，
不需要专用 sh。

### 4.4 `evals/scripts/run_ours_production.sh <gate> <run_dir>` — 分段循环 sh

唯一多次 launch 的运行形态。吸收现 `_run_production_train` 的编排：

```
eval product_env（得 NUM_STEPS 总步数、PRODUCTION_SAVE_SEGMENTS、
                  PRODUCTION_CKPT_ROOT 等）
resumed_from = 调 python helper 扫描 ckpt 根下最新完整 step_<N>   # crash-resume
for 每段 [start, end)：
    end <= resumed_from → skip
    NUM_STEPS/START_STEP/FORGE_SAVE_PATH/FORGE_RESUME_FROM 以 env 覆盖
    launch_dp $OURS_ENTRY --config $PRODUCT |& tee $RUN_DIR/seg_<end>.log
    段后校验 checkpoint 完整（同一 helper），不完整即 exit 非零
```

配套小 helper `evals/scripts/production_ckpt.py`
（`latest`/`check <dir>` 两个子命令，平移 `_latest_production_checkpoint` +
`_production_checkpoint_complete`）。prefetch 门控
（`_await_prefetch_if_configured`）一并移入此 helper 或脚本头部。

### 4.5 `evals/verdicts/` — 判分脚本（第三步，harness 属地）

**定位：读 run_dir 两侧产物 + 产物 TOML 里的阈值 → 判 pass/fail。**
与两侧 sh 完全解耦；按**比较方式**（而非 gate）拆分，避免单文件巨型分支。
统一入口约定：

```
python3 -m evals.verdicts.<name> <gate> <run_dir>
  输入：$RUN_DIR/ref|ours/ 下的公约产物 + load_gate_product() 读阈值
  输出：$RUN_DIR/verdict.json
        { "passed": bool, "summary": str, "metrics": {...}, "details": {...} }
  退出码：0=判分完成（无论 pass/fail），非 0=判分自身出错
```

| 判分脚本 | 吸收现 dispatcher 逻辑 | 读的产物 | 服务的 gate |
|---|---|---|---|
| `bitwise.py` | `_run_bitwise_trajectory` 的逐位/atol 比较、gate_window 校验、MFU floor | 两侧 loss 行、hash dump | multistep-1gpu / multistep / perf-bitwise |
| `align.py` | `_run_align_capture_diff` 张量 diff | ref_capture.pt vs capture.json | forward-align / backward-align |
| `long_train.py` | `window_loss_diff_metrics` 四桶漂移 | 两侧 loss 行 | long-train / long-train-smoke |
| `loss_gate.py` | loss_rel/abs、grad_norm 阈值 | 两侧 loss 行 | loss-gate-200 |
| `resume.py` | `[LOSS_REF]` vs `[LOSS_RES]` 对比、hash .ref/.res diff | ours.log、双 hash dump | resume-gate-20 |
| `resume_startup.py` | `[RESUME_STARTUP] seconds=` 阈值 | ours.log | resume-startup-90 |
| `production.py` | 段完整性 + 全部 step_<N> checkpoint 校验 | seg 日志 + ckpt 根 | production-train |
| `profile.py` | `tools.profile_render.render` 后处理 + 校验 | profile.nsys-rep | profile-snapshot |

阈值来源：**判分脚本自己 `load_gate_product()` 读产物 `[cli]`**
（gate_bitwise / gate_atol / mfu_e2e_target / warmup_steps /
loss_rel_threshold / loss_abs_threshold / grad_norm_abs_threshold /
max_avg_relative_loss_diff / gate_window）。dispatcher 的
`_overlay_product_verdict` 及 `_PRODUCT_VERDICT_KEYS` 随之删除。
loss 行解析继续用 `harness.wire_format.parse_loss_lines`（改从文件读）。
失败分类 `classify_ref_failure` 移到判分/执行器共用模块。

---

## 5. 参数传递契约修订（设计 §0 措辞更新）

原：「ours 的唯一输入是 `--config` 产物；env 是禁区」。
新：

1. **静态参数**（shape/阈值/部署值）：从冻结产物读，`--config` 原样指向
   `workload/src/config/<gate>.toml`。**冻结产物即最终输入，永不复制改写。**
2. **per-run 路径**：由 sh 按 run_dir 公约拼出，经 **env 白名单**传入：
   `RESUME_SCRATCH_DIR / FORGE_CAPTURE_OUTPUT_FILE / FORGE_NSYS_RANK0_OUTPUT /
   FORGE_SAVE_PATH / FORGE_RESUME_FROM / NUM_STEPS / START_STEP`
   （后四个仅 production 段循环覆盖）+ C 类
   `RANK / LOCAL_RANK / WORLD_SIZE`（launch_dp 注入，不变）。
3. **`compose_run_config` 与临时 per-run TOML 全部删除。**
4. `_gate_entry.GateInputs` 收紧：env 兜底只允许白名单键；
   删 `FORGE_GATE` / `FORGE_OURS_CONFIG_DIR` 遗留兜底。

## 6. 渲染器 / 注册表配套改动

`tools/render_gate_configs.py` 新增渲染进产物的键：

| 新产物键 | 来源 | 消费者 |
|---|---|---|
| `OURS_ENTRY`（ours [env]） | 注册表 `script` 键（去掉路径前缀取文件名） | run_ours*.sh |
| `REF_SCRIPT`（ref [env]） | gate_config 既有 `ref_script` | run_gate.sh |
| `REF_CAPTURE_SCRIPT`（ref [env]） | 注册表 `[ref].ref_capture_script` | run_gate.sh |
| `HASH_CAPTURE_LEVEL`（双侧 [env]） | gate_config `[shared.gate].hash_capture_level`（已有，确认落 [env]） | 两侧 sh |
| `NSYS_PROFILE`（ours [env]） | gate_config 新键（仅 profile-snapshot=1） | run_ours.sh |
| `PRODUCTION_SAVE_SEGMENTS` / `PRODUCTION_CKPT_ROOT`（ours [env]） | 现 dispatcher 常量 `_PRODUCTION_SAVE_SEGMENTS=4` + 路径公式，迁到 gate_config | run_ours_production.sh |

注册表每个 `[evals.<gate>]` 段：新增 `needs_ref` / `ours_runner` / `verdict`
三键；迁移完成后删除 `runner_kind` / `script` / `launcher` / `env_inputs`。
`[runtime.distributed].master_port="auto"` 渲染改写 `<auto-port>` 值哨兵，
由 product_env.py 现场分配（见 §4.1 规则 4）。

profile-snapshot 的 mirror_gate 机制（`_resolve_profile_shape`）：改由渲染器
在渲染期解析——profile 产物直接渲染成镜像 gate 的 shape，运行期不再有
mirror 逻辑。`_MILESTONE_PREFIX`（`xxx_roundN` 归并）属于渲染/注册表层，
一并前移。

## 7. dispatcher 薄化后的保留职责（~300 行）

1. `run_suite`：读注册表路由三键 → 通用三步执行 → 组装 result.json。
2. `run_streaming_subprocess`：超时/killpg/日志双写（不变，sh 及其
   launch_dp 子孙同 session，SIGKILL 整组；**脚本规范：sh 内禁用
   setsid/nohup**）。
3. **ref 缓存**：留在执行器，包在「调 ref sh」外面。key 重定义为
   `SHA-256(gate, run_gate.sh 字节, ref 产物文件字节, torch+git 指纹)`
   ——不再维护易失 env 排除表（易失路径全在 run_dir 公约里，天然不进 key）。
   命中时回放缓存的 `ref/` 产物目录到 run_dir。
   仍仅 `hash_capture_level==0` 生效；`FORGE_REF_CACHE=0` 关闭。
4. 边界异常翻译、result.json 落盘、MFU 遥测（runner.py 不变）。
5. stage2 三个 handler 原样保留（挪到 `evals/dispatcher_stage2.py`，
   路由表显式标注豁免）。

## 8. 删除清单

| 删除对象 | 位置 |
|---|---|
| 全部 stage1 handler（`_run_align_capture_diff`/`_run_bitwise_trajectory`/`_run_long_train`/`_run_loss_gate`/`_run_resume_gate`/`_run_resume_startup`/`_run_production_train`/`_run_profile_snapshot`）| dispatcher.py，~2500 行 |
| `RUNNER_KINDS` 表 | dispatcher.py:396 |
| `_overlay_product_verdict` / `_PRODUCT_VERDICT_KEYS` / `_maybe_gate_shape` / `_ref_metadata_int` / `_suite_ours_batch_shape` / `_gate_hash_capture_level` | dispatcher.py |
| 6 处硬编码 ours 产物路径、死变量读取（1044-1047/1446-1447/2587-2588）、遗留 env shape 注入（1453-1467 等） | dispatcher.py |
| `compose_run_config` 及全部调用点 | gate_product.py / dispatcher.py:3004,3134 / _common.py:1200 |
| `run_via_ref_script` env 叠层、`_ref_script_path_env`、`run_ref_capture`、`run_candidate_capture`、`build_suite_env`/`env_inputs` 提升、`distributed_env`/`_resolve_master_port` | _common.py（`run_streaming_subprocess`/缓存/`classify_ref_failure`/`window_loss_diff_metrics` 保留并迁移归属） |
| `gate_common.resolve_ref_trajectory` / `RefTrajectory` | 职责拆入 ref sh（跑）+ verdicts（校验窗口/分类失败） |
| `GateInputs` 的 `FORGE_GATE`/`FORGE_OURS_CONFIG_DIR` 兜底；env 兜底收紧到白名单 | _gate_entry.py |
| `gate_shape.py` 的 dispatcher 侧消费 | 模块保留供 verdicts 用 |
| 注册表 `runner_kind`/`script`/`launcher`/`env_inputs` 键 | dense_training*.toml ×4 套件。**步骤 6 实施修订**：仅 `env_inputs` 从 stage1 删除；`script`/`launcher` 保留（runtime_env.py 入口路由源），`runner_kind` 保留（RUNNER_METRICS 合同键，见文首步骤 6 结论） |

**不动**：渲染器四层结构、产物 `[cli]/[env]` 格式、`gate_product.py` 读取器、
`[LOSS]` 行格式、RESULT_BEGIN/END、run_schema、launch_dp.py、
各 `eval_*.py` 引擎入口、L0 ref 脚本、harness transport 侧全部。

## 9. 逐 gate 映射

| gate | needs_ref | ours_runner | verdict | 备注 |
|---|---|---|---|---|
| forward-align / backward-align | ✓ | run_ours | align | hash bridge 由 HASH_CAPTURE_LEVEL 驱动 |
| multistep-1gpu / multistep / perf-bitwise | ✓ | run_ours | bitwise | perf 的 MFU floor 是 bitwise.py 内读 mfu_e2e_target 的值驱动行为 |
| long-train / long-train-smoke | ✓ | run_ours | long_train | 遗留 env shape 注入随 handler 删除 |
| loss-gate-200 | ✓ | run_ours | loss_gate | RESUME_SAVE_STEP 引擎自读产物 |
| resume-gate-20 | ✗* | run_ours | resume | 引擎进程内两阶段；*若现实现需 ref 轨迹则 needs_ref=✓，迁移时按现 handler 核对 |
| resume-startup-90 | ✗ | run_ours | resume_startup | |
| production-train | ✗ | run_ours_production | production | crash-resume 在 sh+helper |
| profile-snapshot | ✗ | run_ours | profile | NSYS_PROFILE=1 驱动；shape 渲染期镜像 |
| op-* | — | — | — | 豁免，走 dispatcher_stage2.py 旧路 |

## 10. 迁移步骤（每步独立可验证）

1. **基建**：`product_env.py` + `ref/run_gate.sh` + `run_ours.sh` +
   `verdicts/bitwise.py` + dispatcher 通用执行器（与 `RUNNER_KINDS` 并存，
   注册表有路由三键的 gate 走新路，否则走旧路）。
   只给 `multistep-1gpu` 挂新路由，跑通并与旧路结果比对一致。
2. 铺开同构 gate：multistep / perf-bitwise / long-train(-smoke) /
   loss-gate-200（+`long_train.py`/`loss_gate.py`）。
3. align 两个 gate（+`align.py`，验证 bridge 值驱动分支）。
4. resume-gate-20 / resume-startup-90（+`resume.py`/`resume_startup.py`，
   验证 scratch 公约与 hash .ref/.res）。
5. production-train（`run_ours_production.sh` + helper + `production.py`，
   重点验证 crash-resume：杀段后重跑能续）+ profile-snapshot
   （渲染器 mirror 前移 + `profile.py`）。
6. **收尾删除**：旧 handler / `compose_run_config` / `_common` 叠层 /
   `GateInputs` 兜底 / 注册表旧键；ref 缓存切新 key。
7. stage2 挪 `dispatcher_stage2.py`，路由表标豁免；4 套件注册表
   （0.5B/1B/8B/qwen3）同步路由三键。

## 11. 风险与决策记录

| 风险/决策 | 结论 |
|---|---|
| ref 缓存全量失效（重 key） | 接受；切换日 loop 会重跑一轮 ref。注意脏 scratch 类 auto-resume 陷阱勿叠加（见 wsd-sft-70 前科） |
| ours sh 的完整性边界 | 必须放 `evals/scripts/`（harness 属地）。`workload/` 是 loop agent 可写区，放那里等于开放判分链路篡改面 |
| per-run 值走 env 白名单（放弃"全走 config"） | 已拍板。§0 措辞按 §5 修订 |
| 超时归属 | 仍归 dispatcher killpg；sh 禁 setsid/nohup |
| production crash-resume | 下沉 sh+helper（方案 A）；稳定 ckpt 根不进 run_dir |
| stage2 | 豁免，原样保留 |
| `[LOSS]` stdout→文件 | 行格式不变，仅由 sh `tee` 落盘、判分从文件读；引擎与 ref 脚本零改动 |
| 4 个套件注册表 | dense_training / _1b / _8b / _qwen3 都要加路由三键（第 7 步统一做） |
| legacy profile-snapshot 存量缺陷（步骤 5 对照时发现） | `harness/app.py` `_workload_config_snapshot` 把发给 runner 的 `[evals]` 裁成只剩当前套件，而旧 `_resolve_profile_shape` 要求镜像 gate 也在 `evals` 里 → CLI/runner 路径必然抛 "not a declared eval suite"；旧单测直接 load 全量 eval.toml 调内部函数所以从未暴露。新路免疫（镜像解析前移渲染期）。旧 handler 第 6 步整体删除，不单独修 |
