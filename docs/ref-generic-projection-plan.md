# ref 侧通用投影 + 双侧公共运行库 —— 重构计划

状态：待评审（写于 2026-07-16，thin-dispatcher 重构收尾之后）

## 目标（两条，来自评审讨论）

1. **删掉 ref 侧参数注册表**（`tools/gate_product_to_shell.py` 的
   `_CLI_TO_SHELL_VAR` / `_CLI_NOT_LAUNCHER_VARS`）：ref 侧与 ours 侧一致，
   "toml 里有什么，消费端就收到什么"，投影零 key 知识。
2. **抽取 ref/ours 公共运行库**：两个 gate runner（`ref/run_gate.sh` /
   `evals/scripts/run_ours.sh`）的重叠逻辑收编为一份，改运行逻辑不再改两遍。

非目标：不改渲染器（`render_gate_configs.py` 的叠加/切分逻辑不动）、不改
verdict 判分、不改 ours 侧引擎契约（`--config` 产物仍是唯一正式输入）。

## 核心认识

注册表的唯一职能是**变量名翻译**：产物 key（`micro_batch_size`）→ 各 L0
启动器历史级联里优先级最高的名字（`MICRO_BATCH_SIZE_OVERRIDE` /
`FORGE_NUM_LAYERS` / bare `LR`）。而 `tools/product_env.py`（ours 侧投影）
的输出名就是 **key 大写**。所以"删注册表"的实质是：**把 ref 侧全部消费点
改名成 key 大写的规范名**，两侧读同一套名字，翻译层就消失了。

## 改造后的预期形态（验收时对照）

骨架不变（渲染 → 冻结 → dispatcher 三步 → verdict），变的是"产物→进程"
的翻译层：从两套机制收敛为一套机制、两处薄壳。

```
harness run <gate>
  │
  ▼ dispatcher（三步不变）
  ├─ ① ref 步：缓存 key = SHA256(suite + 显式文件清单字节 + ref产物 + 环境指纹)
  │      清单 = run_gate.sh + gate_runner_common.sh + product_env.py
  │           + runtime_env.py + L0启动器 + capture桥（若配）
  │      命中→复用三件套；未中→bash ref/run_gate.sh
  ├─ ② ours 步：bash evals/scripts/run_ours.sh
  └─ ③ verdict：进程内 python，完全不变
```

两个 runner 同构：

```
              run_gate.sh (ref)           run_ours.sh (ours)
source        gate_runner_common.sh  ←同一份→  gate_runner_common.sh
产物选择      resolve_product ref         resolve_product ours（含@milestone）
环境注入      eval product_env 全量  ←同一行→  eval product_env 全量
              eval runtime_env ref        eval runtime_env ours（后eval赢，压部署键）
hash采集      setup_hash_capture    ←同一函数→ setup_hash_capture
侧专属        DUMP_DIR/LOSS_DUMP 约定     FORGE_GATE重推导/nsys/RESUME_SCRATCH
exec          bash L0启动器或capture桥    python launcher entry --config 产物
```

L0 启动器/模型侧：`source gate_product_to_shell.py` 那行删除；消费名一律
"产物 key 大写"；`${VAR:?}` / `os.environ["X"]` fail-fast，无静默默认值；
`grad_accum_steps`/`seq_length` 不再重算，直接吃渲染器派生值。

心智模型一句话：**渲染器是唯一算数的地方；产物是唯一参数载体；
product_env.py 是唯一投影器（零 key 知识）；两个 runner 是同一公共库的两
个薄壳，只在"产物在哪、exec 什么"上不同；verdict 不变。**

加新参数的路径：改 gate_config →（若需派生）改渲染器 → 消费端直接读大写
名。不再有注册表一站。

明确不变的边界：判分两级结构（verdict 硬门禁 + review prompt 软门禁）、
信任边界（ref 产物冻结、公共库在 evals/scripts/ agent 不可写、判分阈值只
在 registry）、ours 引擎契约（--config 唯一正式输入，env 兜底）。

接受的代价：判分阈值类 key（MFU_E2E_TARGET 等）以 env 噪音形式进 ref 训练
进程（无消费者、无害）；"孤儿 key 无人消费"从投影期报错变为静默忽略（与
ours 对称，靠静态一致性测试兜部分底）。

## 现状盘点（已核实）

### 消费面

- **5 个 L0 启动器** source 注册表投影：
  `run_qwen3_dense.sh`、`run_16gpu_1000step_pure_mup_mtp.sh`、
  `run_minicpm4_8b_dptp.sh`、`train_minicpm4_0.5b_fineweb_modelbestsdk.sh`、
  `train_minicpm4_0.5b_gsm8k.sh`
  （`run_minicpm4_8b_hf_singlecard.sh` 不 source，单独确认其参数来源）
- **python 侧 import 期读 `FORGE_*` env 且带静默默认值**：
  `model_qwen3.py:57-68`（如 `FORGE_NUM_LAYERS` 缺失默默用 28）、
  `model_pure_mup_mtp.py`、`model_minicpm4_8b_tp.py`（各自核对）
- **引用注册表的其他代码**：`evals/_common.py`、
  `evals/scripts/runtime_env.py`（`FORGE_GATE_SHELL_TOOL` 导出，:54）、
  `tools/bootstrap_canonical.py`
- **7 个测试文件**：test_suites_shape / test_minicpm4_8b_ref / test_layer_dag /
  test_framework_guard / test_ours_env_inputs_transport / test_qwen3_ref /
  test_gate_transport_parity

### 缓存 key 现状（dispatcher.py:113-127）

```
key = SHA256(suite + run_gate.sh 字节 + ref 产物字节 + 环境指纹)
环境指纹 = torch 版本 + git HEAD sha（_common.py:585）
```

**已核实的坑**：loop workspace 是删掉 `.git` 的拷贝 → 指纹的 git 半边是
`"no-git"` → **在 workspace 里改启动器 / 投影工具字节，缓存不会失效**。
这是 pre-existing 缺口，本次必须一并补：key 的文件集要显式覆盖 ref 执行路
径上的全部脚本字节，不能赖 git sha。

### runtime_env.py ref 模式缺口（:39-80，已核实）

`_ref_overrides` 导出 MEGATRON_ROOT（经 `_ref_script_path_env`）、
FORGE_REF_CONFIG_DIR、FORGE_DATA_TOML、INIT_ONES、启动器/桥路径、
FORGE_REF_EXTRA_ARGS 等，但 **不导出 MASTER_ADDR / MASTER_PORT**（ours 侧
在 :117-118 导出）。ref 侧的 MASTER_PORT 目前靠 dispatcher 进程注入 +
产物投影时被 `_DEPLOYMENT_ENV_KEYS` 跳过。切到通用投影后，freeze-fill 过
的产物里 MASTER_PORT 是具体值、会被 export 出来盖掉进程注入 → **必须给
`_ref_overrides` 补 MASTER_ADDR/MASTER_PORT 解析**（镜像 ours 写法），靠
"product_env 先、runtime_env 后、后者赢"的既有次序压住产物值。

## 改动步骤

### Step 1：抽公共运行库（纯搬移，零行为变化，单独提交）

新建 `evals/scripts/gate_runner_common.sh`（harness 管辖区，agent 不可写），
两个 runner 共 source：

```
resolve_product SIDE GATE [LABEL]
    # ref:  ref/config/$GATE.toml
    # ours: workload/src/config/$GATE.toml + label→@milestone 变体选择
    # 共同的存在性检查与报错格式
project_env PRODUCT SIDE GATE
    # eval product_env.py（Step 2 起 ref 也走这行）
    # eval runtime_env.py SIDE GATE     ← 次序固定：runtime 后 eval、赢
setup_hash_capture PRODUCT RUN_DIR SIDE
    # HASH_CAPTURE_LEVEL 读取、HASH_ARGS 拼装、
    # NUM_STEPS>1 → --persistent、dump 文件路径约定
    #（ref: ref_hash_dump.json + HOOK_OUTPUT_FILE 兼容位；
    #  ours: ours_hash_dump.json + FORGE_CAPTURE_OUTPUT_FILE）
```

run_gate.sh 保留：DUMP_DIR/LOSS_DUMP_FILE 约定、capture 桥二选一 exec。
run_ours.sh 保留：FORGE_GATE 重推导（有测试钉着）、nsys wire、
RESUME_SCRATCH_DIR、exec python launcher。

**同 Step 落缓存 key 扩容**（dispatcher.py `_scripted_ref_cache_dir`）：
文件集从 `run_gate.sh` 扩成显式清单——
`run_gate.sh + gate_runner_common.sh + tools/product_env.py +
evals/scripts/runtime_env.py + 解析后的 L0 启动器文件 + capture 桥（若配）`。
一次性作废现存 ref 缓存，接受。

验收：全量套件基线 diff 零新增；两个 runner 的现有行为测试全绿。

### Step 2：ref 侧切通用投影（核心步，按启动器分小步）

1. `run_gate.sh` 的环境注入改成与 ours 相同的两行（走 `project_env`）：
   product_env 全量 export → runtime_env ref 覆盖。
2. `_ref_overrides` 补 MASTER_ADDR/MASTER_PORT（见上）。
   **实施偏差**：`FORGE_GATE_SHELL_TOOL` 导出的删除推迟到 Step 4——
   megatron 捕获分支（bridge.sh）跑的是启动器的 mktemp 副本，其
   `$SCRIPT_DIR/../../tools/` 相对回退定位不到 tools/，绝对路径导出必须
   活到所有还在 source 注册表的启动器改完为止；qwen3 改完后该导出对
   qwen3 已无消费者，双设无害。
3. 逐个启动器改消费名（**一个启动器一个提交，各自过逐位验证**）：

   | 旧名 | 新名（= 产物 key 大写） |
   |---|---|
   | `NUM_STEPS_OVERRIDE` | `NUM_STEPS` |
   | `MICRO_BATCH_SIZE_OVERRIDE` | `MICRO_BATCH_SIZE` |
   | `GLOBAL_BATCH_SIZE_OVERRIDE` | `GLOBAL_BATCH_SIZE` |
   | `FORGE_NUM_LAYERS` 等 10 个几何 | `NUM_LAYERS` 等 |
   | `FORGE_INIT_METHOD_STD` | `INIT_METHOD_STD` |
   | `INIT_ONES` | `FORGE_INIT_ONES`（key=forge_init_ones） |
   | `FORGE_MUP_*` / `FORGE_MTP_*` | `MUP_*` / `MTP_*` |
   | `TP_SIZE`（来自 tensor_parallel_size） | `TENSOR_PARALLEL_SIZE` |

   顺序：`run_qwen3_dense.sh` 打头（当前 loop 在用，逐位验证最方便）→
   pure_mup_mtp → 8b_dptp → fineweb → gsm8k。
4. python 侧 model/train .py 同步改名，且**删掉几何键的静默默认值，改
   fail-fast**（`os.environ["NUM_LAYERS"]`，缺失即 KeyError）——这是删注册
   表后"参数缺失必响"保障的一半（另一半是启动器的 `${VAR:?}`）。
5. **实施补充**：`tools/bootstrap_canonical.py` 是隐藏的启动器直连调用方
   （经 bridge.sh 绕过 run_gate.sh），启动器改名后它必须自带投影——
   product_env 新增 `export_map()`（dict 版投影，规则与 export 行一致、
   零 key 知识），bootstrap 在解析出 canonical gate 后
   `env.update(export_map(ref/config/<gate>.toml))`，其自身的显式覆盖
   （WORLD_SIZE/MASTER_PORT/INIT_ONES/CANONICAL_STATE_OUTPUT_FILE）在
   投影之后赋值，保持覆盖胜出。
6. **实施补充（pure_mup_mtp 轮发现）**：第二个隐藏直连调用方是 stage2 的
   `run_via_ref_script`（evals/_common.py，经 gate_common.resolve_ref_trajectory
   → tools/ref_script_runner.py），它原来只设 FORGE_REF_CONFIG_DIR /
   FORGE_GATE_SHELL_TOOL 靠启动器自 source。改法同 bootstrap：
   merged_extra_env 最底层 `update(export_map(ref/config/<suite>.toml))`
   （产物不存在的 meta gate 跳过），紧跟 runtime_env 同款的
   MASTER_ADDR/MASTER_PORT 解析压制 freeze-fill 具体值；其后的路径垫片 /
   指针 / INIT_ONES / caller extra_env 层次不动、照旧胜出。另有三个直调
   启动器的外围冒烟脚本一并自带投影+改名：env/gate_sweep_torch.sh、
   tools/run_ref_milestones.sh（override 在投影后 export，恢复"调用方
   覆盖 shape"的本意——旧注册表 source 其实会 clobber 调用方 env）；
   env/verify_env_torch.sh 只 `bash -n` 不实跑，无需改。
7. **实施补充（8b_dptp 轮发现）**：`TP_SIZE`（旧注册表对 tensor_parallel_size
   的注册名）改为通用大写名 `TENSOR_PARALLEL_SIZE`，且它是**可选键**——单卡
   gate 的产物不携带，启动器 `${TENSOR_PARALLEL_SIZE:-1}` 回退到 TP=1
   （dense 代理路径），dptp gate 才设 2；这是通用投影下"可选拓扑键"的先例
   （与必填 shape/几何键的 `:?` 探针区分）。`export FORGE_TP_SIZE` 整体删除：
   train py 的 argparse default 改读 TENSOR_PARALLEL_SIZE，且启动器恒传
   `--tensor-parallel-size` CLI（CLI 胜出），环境名只剩兜底。几何裸名
   （NUM_LAYERS 等）自此与 0.5B ref 模块共享，test 内 import 8B model 前
   须 save/restore 这批 env，防跨用例污染。逐位验证备注：2 卡 devspace
   跑不了 world=4 的 dptp，双侧 cfg 同改 world_size_override 4→2
   （TP=2/DP=1/accum=8）后对比——覆盖 TP=2 路径，公平成立。

8. **实施补充（fineweb 轮发现）**：megatron fineweb ref 轴在 HEAD 上从未
   端到端跑通过，存在四个与本重构无关的既有 blocker（两树相同，未在本轮
   修复，单独立项）：
   ① `config/ref/megatron_minicpm4_0.5b.toml` 钉的 `cpm_core_r0.15.0`
   分支没有任何 mup/eagle argparse 参数，而启动器无条件传
   `--mup-*`/`--eagle-*` 五参；② 正确谱系 `cpm_core_r0.8.0_mtp` 有这五参
   （但靠启动器从不传的 `--use-mup` 门控 mup 生效——fidelity 注意点），
   却缺 `--padded-vocab-size`/`--ckpt-format`——没有分支能原生吃下启动器
   参数集；③ bridge.sh 的 megatron sed 锚点找 `pretrain_gpt.py`，该启动器
   入口是 `pretrain_minicpm.py`，6 个 capture 门拒绝 patch；④ 入口本身也
   错：MTP 常开时 batch 为 9 元组而 `pretrain_minicpm.py` 只解 6 元，
   正确入口是 `pretrain_mtp.py`。
   逐位验证因此以"同 shim 环境下 before/after 全等"达成：
   (a) fake-torchrun 截获 9/9 个 gate 的最终 argv+env——argv 逐字节一致
   （唯一差异为产物本身的 master_port），env 差异恰好等于被删注册器导出的
   16 个 FORGE_* 几何中间变量（megatron 不消费）；
   (b) 真跑 long-train-smoke（r0.8.0_mtp + 三个披露补丁的共用 shim：
   pkg_resources import 修复、补 --padded-vocab-size/--ckpt-format 两
   argparse、入口 cp pretrain_mtp.py→pretrain_minicpm.py；数据用生产
   modelbest sstable 轴），before/after 20/20 迭代 lm loss、
   eagle_ce_loss、grad norm 打印逐位一致。
   另：megatron fork 不支持 LOSS_DUMP_FILE（torch ref 独有），megatron
   门的数值证据口径 = 训练日志逐迭代 loss 行；TOKENIZER_MODEL 不经产物
   投影（运行时只出 FORGE_TOKENIZER_DIR），裸跑须自行 export——既有
   HEAD 行为，两树一致。

9. **实施补充（gsm8k 轮发现）**：gsm8k ref 轴（`train_minicpm4_0.5b_gsm8k.sh`）
   同样在 HEAD 上从未端到端跑通过，累计七个与本重构无关的既有 blocker
   （两树相同，未在本轮修复，单独立项）：
   ① 没有任何 config/ref/*.toml 引用该启动器（megatron_minicpm4_0.5b.toml
   的 ref_script 指 fineweb），gsm8k 只能手改 ref_script 选中；
   ② r0.8.0_mtp 分支 `pretrain_gpt.py` 是死代码——fork 的 GPT model 恒返
   dict，loss_func 里 `output_tensor.float()` 直接 AttributeError；
   ③ r0.8.0_mtp `utils.py` 在 eagle≠0 时无条件读 `data["mtp_tokens"]`，
   但标准 GPTDataset 从不产 mtp 键（只有 SDK EagleBatchPacker 产），而
   gsm8k 启动器不传 `--use-modelbest-sdk` → KeyError；
   ④ `pretrain_mtp.py` 也救不了：forward_step 无条件解 9 元组（eagle=0 时
   batch 只有 5 键）且 model() 位置参数强制要 mtp_tokens/mtp_labels；
   ⑤ 钉住的 r0.15.0 缺全部 mup/eagle argparse 参数而启动器无条件传；
   ⑥ r0.15.0 fork 的 `get_batch_on_this_tp_rank` 删掉了非 SDK 的取数分支
   （`data = next(...)` 只在 `if args.use_modelbest_sdk:` 里），普通
   GPTDataset 路径 `data` 未赋值 → UnboundLocalError；
   ⑦ r0.15.0 fork 自家 `utils.py` 给 batch 无条件塞第 6 键 `dataset_id`
   （log_task_loss_interval=-1 时为 None 也塞），而自家 `pretrain_gpt.py`
   forward_step 仍按上游 5 元组解包 → ValueError。
   逐位验证以"同 shim 环境下 before/after 全等"达成：
   (a) fake-torchrun 截获 9/9 gate 最终 argv+env——argv 逐字节一致（唯一
   差异为产物本身的 master_port），env 差异恰好等于被删注册器导出的同一批
   16 个 FORGE_* 几何中间变量；
   (b) 真跑 long-train-smoke（r0.15.0 + 三个披露补丁的共用 shim：
   arguments.py 加 5 个 no-op mup/eagle argparse（解析即弃，产物保持
   mtp_num_layers=1 原值不妥协）、utils.py 补非 SDK 取数 else 分支、
   pretrain_gpt.py 解包改 `*_` 兜多余键；数据经启动器自带的
   DATA_PATH_OVERRIDE 钩子指向预制 gsm8k 二进制），before/after 20/20
   迭代 lm loss、grad norm、loss scale、consumed samples 字段级
   120/120 逐位一致。r0.15.0 无 eagle 实现，日志无 eagle_ce_loss 行，
   数值口径 = lm loss 单头。

10. **实施补充（Step 3 落地记录）**：3.1/3.2/3.3/3.5 已在 Step 2 各启动器
    轮内顺带完成；本轮落地 3.4 与 3.6。
    **3.4**（grad_accum_steps/seq_length 直吃产物）：五个启动器统一删
    `GRAD_ACCUM_STEPS=$((GBS / (MBS·DP)))` 重算与
    `SEQ_LENGTH_ARG=$MAX_POSITION_EMBEDDINGS` 推导，改
    `: "${GRAD_ACCUM_STEPS:?}"` / `: "${SEQ_LENGTH:?}"` fail-fast 探针
    （渲染器 render_gate_configs.py 已派生并在 GBS 不可整除时 RenderError，
    产物是唯一权威）。旧重算仅喂 gate_metadata.json 与 echo 横幅，训练进程
    自行推导 accum，`--seq-length` 值 = max_position_embeddings = 4096 →
    argv 逐字节不变。验证：产物 9/9 相同；fake-torchrun 9/9 argv+env+
    gate_metadata 全等；真跑 long-train-smoke before/after 20/20 迭代
    字段级 120/120 逐位一致（GATE_RC=0）。
    **3.6**（INIT_ONES 双通道收敛）：9/9 gate ref 产物均携带
    forge_init_ones（前置确认）；裸 `INIT_ONES` 在启动器侧早已零消费者
    （pure_mup_mtp 只读 FORGE_INIT_ONES），故删除均为死通道：
    ① `evals/scripts/runtime_env.py` `_ref_overrides` 的 INIT_ONES 导出；
    ② `evals/_common.py` `run_via_ref_script` 的 merged_extra_env 注入、
    `forge_init_ones_ref_env`、以及随之归零调用的
    `forge_init_ones_value`/`forge_init_ones_resolved`（整块删除）；
    ③ `tools/bootstrap_canonical.py` 的 `env["INIT_ONES"]`（保留
    FORGE_INIT_ONES 显式覆盖——方案迭代需盖过 freeze 回填的产物值）。
    验证：fake-torchrun 9/9 对比 Step3.4 树，唯一 env 增量 = INIT_ONES
    键消失（argv 逐字节一致）；全量单测基线 140 条零新增。

11. **实施补充（Step 4 落地记录）**：删 `tools/gate_product_to_shell.py`
    （注册表本体，此时已零 shell 消费者——Step 2 后没有任何启动器
    source 它）。随删更新：
    ① `evals/dispatcher.py` 缓存 key 文件清单去掉注册表条目（缓存 key
    随之改变，现存 ref 缓存一次性作废，计划预期内）；
    ② `FORGE_GATE_SHELL_TOOL` 指针导出（runtime_env.py `_ref_overrides`
    与 `_common.run_via_ref_script`）删除；
    ③ gate_runner_common.sh / _common.py / bootstrap_canonical.py 各处
    "注册表退役中"过渡注释清成终态描述；
    ④ 测试：`StageBRefTransportTest` 从注册表输出断言改写为
    `product_env.export_map` 通用名完整性断言（14 个必备键含
    GRAD_ACCUM_STEPS/SEQ_LENGTH/FORGE_INIT_ONES）；新增
    `RefConsumerNameConsistencyTest`——五个启动器全部 `${VAR:?}` fail-fast
    探针名必须 ∈ upper(_CLI_ORDER) ∪ {DATA_CONF,DATA_PATH,MEGATRON_ROOT,
    TOKENIZER_MODEL}（运行时部署键），钉死"不再长翻译层"；
    gate_runner_common 函数级测试与缓存 key 文件集测试已存在于
    test_gate_runner_common.py（动态遍历清单，删表自适应）；
    layer_dag / framework_guard / ours_env_inputs 各删一条注册表引用。
    验证：删表前跑一次性静态投影等价脚本（不入库）——9 gates 旧链
    （旧名→新名映射后）vs product_env 共有键逐值相等、单侧键全部落入
    预期类别（旧链显式排除键→新链无消费者噪音、GATE_WINDOW 拆分→合并、
    NPROC_PER_NODE 已挪 megatron 启动器、部署键旧跳过新导出但被
    runtime_env 压制），0 fail；fake-torchrun 9/9 对比 Step3 树唯一
    env 增量 = FORGE_GATE_SHELL_TOOL 指针消失（argv 逐字节一致）；
    全量单测基线 140 条零新增，新测试单跑通过。

1. `NPROC_PER_NODE`（megatron 从 WORLD_SIZE 播种）→ 挪进 megatron 启动器。
2. `gate_window`（list → `GATE_WINDOW="11 26"`）→ 消费方
   `read -r GATE_WINDOW_START GATE_WINDOW_END <<< "$GATE_WINDOW"`。
3. 部署键防冲（MASTER_PORT/MEGATRON_ROOT/CHECKPOINT_ROOT）→ Step 2.2 的
   runtime_env 次序压制解决；CHECKPOINT_ROOT 无 ref 消费者，export 无害。
4. `grad_accum_steps` / `seq_length`：启动器原本自己重算/推导 → 改成直接
   吃产物值、删重算（渲染器已派生，双算是漂移源；此改动本身要逐位验证）。
5. export/bare 区分消失（全部 export）：判分阈值（`MFU_E2E_TARGET` 等）会
   进 ref 训练进程 env——无消费者、无害；预期内的 env 噪音，review 时知晓。
6. `INIT_ONES` 双通道收敛：启动器改读 FORGE_INIT_ONES 后，确认全部 gate 的
   ref 产物均携带 forge_init_ones（50eea46 修过渲染），然后删
   `_ref_overrides` 的 INIT_ONES 导出，只留产物一个来源。

### Step 4：删注册表 + 收尾

- 删 `tools/gate_product_to_shell.py` 与其单测；
- 更新 7 个引用测试 + `evals/_common.py` / `bootstrap_canonical.py` 引用；
- 新增测试：
  - gate_runner_common.sh 的函数级测试（resolve/项目/hash 三函数）；
  - "ref 消费名 = 产物 key 大写"的静态一致性测试（grep 启动器消费名 ∈
    产物可导出名集合，防再长出翻译层）；
  - 缓存 key 文件集测试（改任一清单内文件字节 → key 变）。

## 保障替换对照（删表不裸删）

| 注册表原保障 | 替代机制 |
|---|---|
| 启动器要的 key 产物里缺 → 投影期 raise | 消费端 `${VAR:?}` + model .py 删默认值 fail-fast（推迟到运行期，仍响） |
| 产物新 key 到不了启动器 → raise | **有意放弃**（与 ours 对称：未知 env 被忽略）；静态一致性测试兜部分底 |
| 部署键不被产物值 clobber | runtime_env 后 eval 压制 + `_ref_overrides` 补 MASTER_* |

## 验证（量尺改造，逐位是硬标准）

1. **ref 轨迹逐位对比**（每个启动器改完各做一次）：needs_ref gate 的 ref
   侧改造前后各跑一遍，`dump/ref_loss.txt` 逐字节相同；hash-capture gate
   （forward-align / perf-bitwise）`ref_hash_dump.json` 相同。参考
   93177e6 artifact 的既有流程。
2. **静态投影等价对比**（删表前的一次性脚本，不入库）：对每个 gate 的 ref
   产物，旧投影"变量集"（按旧名→新名映射后）与新链 `product_env` 输出逐值
   相等；差异必须逐条解释（预期差异 = 判分阈值等新增噪音变量）。
3. 全量套件基线 diff（当前基线 140 条既有失败，零新增）。
4. ruff 全绿。

## 风险

- **最大风险**：改名手滑导致 ref 轨迹静默漂移（如 seq_length 从推导改直吃
  时值不等价）→ 所有 gate 的 pass/fail 含义变化，且单测测不出。对策即上面
  的逐位验证 + 每启动器独立提交（可单独回滚）。
- 缓存 key 文件集扩容 → 现存 ref 缓存全部一次性作废（预期内，改造后首轮
  loop 的 ref 会重跑一遍）。
- `run_minicpm4_8b_hf_singlecard.sh` 不走注册表投影，参数来源待确认，可能
  在 Step 2 清单之外（单列核实项）。**已核实（Step 4 收尾）**：它是独立
  oracle（非门禁），参数来源 = env 默认值（FORGE_DATA_DIR/TOKENIZER_DIR/
  DATA_CONF/FORGE_DATA_TOML）+ `--` 透传 CLI，模型形状取自 HF Hub 官方
  config——从不消费注册表也不消费 gate 产物，删表零影响，无需改动。
- 正在跑的 loop f6b9c438e05f 的 workspace 是隔离副本，不受影响；本重构对
  下一个新 loop 生效。

## 提交切分

1. Step 1（公共库 + 缓存 key 扩容），零行为变化
2. Step 2.1-2.2（run_gate.sh 切投影 + runtime_env 补 MASTER_*）+ qwen3
   启动器改名 + 逐位验证
3. 其余 4 个启动器 ×（各一提交，各逐位验证）
4. Step 3 特例安置（部分随 2/3 顺带完成，剩余单提）
5. Step 4 删表 + 测试收尾
