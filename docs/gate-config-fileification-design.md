# 门禁配置文件化设计方案

> 配套文档:逐参数落位见 `gate-parameters-reference.md`(参数字典);逐文件/逐门禁的改动
> 清单与迁移顺序见 `gate-config-fileification-plan.md`(代码修改计划)。本文件只讲**为什么这么
> 设计**和**核心机制**,不重复罗列参数表与迁移步骤。

## 0. 名词:什么是"哨兵(sentinel)"

指产物 `[env]` 段里写成字面量 `<runtime>` 的键(`tools/render_gate_configs.py:58` 定义
`RUNTIME = "<runtime>"`)。渲染(render)发生在拿到 devspace 之前,部署路径 / 端口这些值
此时**还不知道**,于是先填 `<runtime>` 占位,等真正要用时再替换成真值。

今天这个替换发生在 **dispatch 时刻**:

- ours:`_gate_entry.GateInputs.get()` 遇到 `<runtime>` 就 `return None` 落回 `os.environ`;
- ref:`gate_product_to_shell.py` 的 emitter 遇到 `<runtime>` 直接 `continue` 跳过,改由
dispatcher 注入的环境变量顶上。

**本方案的核心一句话**:把"替换 `<runtime>`"这一步从 dispatch 时刻**提前到 freeze 时刻**。
替换完,配置文件里就没有任何占位符,dispatcher 也就没有任何东西需要注入了。

---

## 1. 目标(用户意图)

1. **配置文件化**:ref / ours 两侧的配置参数全部进文件。
2. **ours 边界收成一根**:跑门禁时 harness 传给 ours 子进程的,**只能有一个 config 文件路径**,
  不再直接传任何 env / CLI 参数。目的:ours 引擎脱离 harness 跑不起来、也没法收硬编码路径,
   dev-agent 只能写"从 config 读一切"的规矩引擎。
3. **ref 配置全冻结**:ref 侧配置文件对 agent 只读、完全冻结。
4. **dispatcher 变薄**:dispatch 代码里**不得硬编码任何 ref/ours 参数**,也不得注入。
  否则出问题时 dev-agent 会想去改 dispatcher,破坏隔离。

> 说明:曾讨论过加一条 lint(禁止引擎读 `os.environ` / 禁止 `--config` 外的 argparse)来硬堵
> "偷读 env / 收硬编码路径"。用户认为暂不合理,**本期不做**。约束靠"harness 只传 config 路径 +
> 判定阈值在 agent 改不到的冻结侧"这两条结构性手段达成,而非静态扫描。

---

## 2. 核心机制:三个时刻分离

问题根源是"部署值解析"被放在了 dispatch。把它挪到更早的 freeze,dispatch 就无参可注入。


| 时刻           | 执行者                                                                    | 职责                                                        | 产出               |
| ------------ | ---------------------------------------------------------------------- | --------------------------------------------------------- | ---------------- |
| **render**   | `tools/render_gate_configs.py`(已有,不改语义)                                | 产物 + `<runtime>` 哨兵(部署值此时确实未知)                            | 带哨兵的产物           |
| **freeze**   | `tools/agent_loop_lease.sh:_devspace_claim_and_freeze()`(**已有函数,扩展它**) | 见下                                                        | **零哨兵、全具体**的冻结配置 |
| **dispatch** | `evals/dispatcher.py`(**变薄**)                                          | 读配置取拓扑 + 判定阈值 → `launch --config <path>` → 按派生约定读回产物 → 判定 | verdict          |


### 2.1 "freeze" 具体是哪一步

**不是新造的步骤**——就是 loop 启动时 `agent-loop.sh:537` 调用的
`tools/agent_loop_lease.sh:_devspace_claim_and_freeze()`,即 CLAUDE.md 里"launch 后 config dir
被 `chmod -R a-w` 冻结 + 改 hostname"那一刻。它在一次 loop 生命周期里**只跑一次**,现有顺序:

1. **claim** 独占 devspace lease(`agent_loop_lease.sh:31`)
2. **rewrite hostname** → `ds-<id>`(`:68-69`,`sed` 改 `remote.toml` 的 `hostname`)
3. **render** gate 产物到 workspace 的 `ref/config` / `workload/src/config`(`:156`)
4. **chmod -R a-w** 冻结 `ref/config`(`:161`)和整个 per-loop config dir(`:168`)

走到这一步时 devspace **已经 lease 好**——第 2 步就已经在填一个部署值(hostname)了。既然
hostname 此刻已知,同一台 devspace 上固定的其它部署值(`checkpoint_root` / `megatron_root` /
`data_path` / `master_addr` / `master_port` / 资产路径)**也全部已知**。

**本方案对 freeze 的唯一改动**:在第 3 步 render 之后、第 4 步 chmod 之前,**新增一步"把产物
`[env]` 里的 `<runtime>` 替换成真值"**,替换完再 chmod。本质上是把现在只对 `remote.toml` 一个
hostname 做的 `sed` rewrite,扩展成对部署值做替换。

**但有跨机边界**:freeze 跑在启动机、门禁跑在 devspace(`kind="local"` 才同机),所以只有
**机器无关**的值(`MASTER_ADDR=localhost`、`MASTER_PORT` 固定端口)能在 freeze 直接填;
`repo_root()` 派生的绝对路径(`CHECKPOINT_ROOT`/`MEGATRON_ROOT`)改填**相对约定路径**、执行侧
按各自 `repo_root` 拼绝对(+ 绝对存储字段 + lease 软链接);`DATA_PATH` 解析与资产下载**暂缓**
(下载是执行机副作用、且实际基本没用)。逐项落位与理由见 `gate-config-fileification-plan.md §2.1`。

"解析部署值"这件事**只在 freeze 发生一次**。dispatch 侧不再出现 `<runtime>` 解析、
不再出现任何 `extra_env`、不再有 `_latest_production_checkpoint` 之类的形状/编排计算。

---

## 3. 参数归属:四类

搜全 ours/ref 两侧当前所有参数来源后,每个参数归到下列四类之一:

- **freeze 填实**:部署固定值(路径 / 端口 / host / 资产),freeze 时把 `<runtime>` 替换成真值
  写进冻结文件。
- **引擎派生**:由 `artifact_root` + gate 名 + 磁盘状态派生,ours 引擎自己算,harness 不传(见 §4)。
- **保留(豁免)**:torch rendezvous(`RANK` / `LOCAL_RANK`),冻结 launcher / torchrun 设,
  非可调参数(见 §6)。
- **文件已有 / 删除**:本就在产物 `[cli]` / `[env]` 里,或收编后不再需要的指针 env。

> 逐参数的"当前怎么来 → 新归属"完整落位表(ours 侧、ref 侧,每参数带作用 / 取值 / 归属)见
> **`gate-parameters-reference.md`**——本设计不再重复罗列,避免两处漂移。

---

## 4. 引擎派生规则(ours 侧,写在引擎里)

ours 引擎入口固定 `train(--config <path>)`,从 config 读一切;下列**输出/续训**由引擎自派生,
harness 不传(这本就是"一个真训练引擎该会的事",也正是 agent 要实现对的地方):

1. **输出路径派生**:以 config 中的 `artifact_root`(freeze 填实)+ gate 名为根,按固定约定拼
  `save/`、`capture.bin`、`nsys.rep`、`resume_scratch/`。dispatcher 用**同一约定**从相同位置
   读回,因此无需传路径。
2. **resume 自恢复**:启动扫 `save_root`,有 checkpoint 就从最新续、无则从头。把
  `_latest_production_checkpoint` 的扫盘逻辑从 dispatcher 搬进引擎。resume 门禁的两次进程共享
   同一冻结 config,run2 靠磁盘上有 ckpt 自动续——dispatcher 只"跑、杀、再跑",不传 `START_STEP`。
3. **步数派生**:`target_steps` 在 config,`done_steps` 从 ckpt 读,续到 target;production
  的分段不再由 dispatcher 计算。

---

## 5. 防放水:判定阈值统一进 eval.toml

"agent 改不对就一定过不了门禁"的前提是**判定阈值必须在 agent 改不到的地方**——绝不能从
ours 可改的 `workload/src/config/<gate>.toml` 读,否则 agent 把 `gate_atol` 设成 1e9 就过了。

判定阈值统一放注册表 `config/eval/<suite>/<suite>.toml`(= 每 loop 拷成 `eval.toml`,随 config
dir 一起 `chmod a-w` 冻结)的 `[evals.<gate>]`。dispatcher 直接读它本就加载的
`workload_config["evals"][suite_key]`,**不从任何产物读回阈值**(不再有 overlay 一层)。

- 阈值不渲染进产物、不从产物读。eval.toml 冻结、agent 不可写 → 连无 ref 对照的 ours-only 门禁
  (production / resume-startup)也有权威阈值,agent 改不动。
- ref-vs-ours 门禁与 ours-only 门禁的判定量来源**统一**,dispatcher 少一层 overlay。

**两类务必分清**(只有"阈值"进 eval.toml,执行输入仍走 gate_config → 产物):

| 类别 | 归属 | 键 |
|---|---|---|
| **判定阈值**(训练不需要,仅 dispatcher 判定) | **eval.toml `[evals.<gate>]`** | `gate_bitwise` `gate_atol` `hash_capture_level` `mfu_e2e_target` `warmup_steps` `loss_abs_threshold` `grad_norm_abs_threshold` `loss_rel_threshold` `max_avg_relative_loss_diff` `resume_startup_budget_s` |
| **执行输入**(引擎/ref 跑起来真需要) | gate_config → 产物 | shape(world_size/num_steps/mbs/gbs/seq/seed)、`forge_init_ones`、`resume_save_step`、`deterministic`、optim override、`@unset` |

**`hash_capture_level` 是判定语义,不是执行输入**:它决定 bitwise 比对覆盖哪些张量
(`0` 不比 / `1` loss+grad / `2` 再加每模块 `fwd.<fqn>`+`bwd.<fqn>`),错的 level 会**静默削弱
比对**(`dispatcher.py:500-502`);现状本就从冻结侧读(注册表 inline → ref 产物),agent 改不到。
故归 eval.toml。**执行侧固定**:ours 引擎恒按最高 level(2)全采,dispatcher 按 eval.toml 里的
level 决定实际比对哪些张量——agent 改不动比对范围,ours 多采无害。

**待定**:`gate_window` 双重用途——判定划窗(→ eval.toml)+ ref 采集可能要它决定 emit 哪些步
(现投影为 `GATE_WINDOW_START/END`)。二选一:ref 全量 emit、dispatcher 划窗;或 window 作为
执行输入仍渲进 ref 产物(与 eval.toml 权威值同源,冗余但无害)。落地时定。

---

## 6. 唯一豁免:torch rendezvous

`RANK` / `LOCAL_RANK`(以及 torchrun 自身吃的 `MASTER_*`)由**冻结的**
`evals/scripts/launch_dp.py` 读 config 里的 DP / 端口后 exec torchrun 设置。从"harness →
agent 引擎的**可调面**"看仍只有 `--config`;rendezvous 是框架基建、由 harness 冻结侧设,
不进 agent 认知面。这是唯一无法写进共享文件的东西(每个 rank 的 `RANK` 不同)。

---

## 7. dispatcher 变薄后的样子(伪代码)

```python
cfg = load_frozen_config(gate)                 # 读文件,无 <runtime> 解析
thr = load_gate_thresholds(gate)               # 从冻结 eval.toml [evals.<gate>] 读判定量
run(f"{cfg.launcher} {cfg.engine} --config {cfg.path}")   # 零 extra_env
out = read_back(cfg.artifact_root, gate)       # 按派生约定读回 loss/hash/ckpt
verdict = judge(out, ref_out, thr)             # 只有这里做比较
```

无 `extra_env`、无 `_latest_production_checkpoint`、无 `<runtime>` 解析、无任何 ref/ours 形状
参数——出问题时 agent 也没有"想改 dispatcher"的入口,全去改自己的引擎与 ours 配置。

---

## 8. 迁移顺序与风险

见 `gate-config-fileification-plan.md §5`——逐步骤(freeze 填实 → ours 引擎自足 → 削 dispatcher
→ ref 折 env → 判定阈值迁 eval.toml)与风险面收敛在计划文档里统一维护,此处不再重复。

一句话风险结论:`RESULT_BEGIN/END`、`[LOSS]` 的 **stdout 判定线不动**,只动"入参侧",判定逻辑
零风险;回归集中在"引擎能否从 config + 派生拿全所需"。
