# Loop 轨迹续跑(fork 式)设计方案

> 状态:v4,已按三轮评审意见定稿待确认。本文档同时包含设计与实施计划。
>
> 已定决策:分叉新 loop(复制 workspace,不用 git worktree)、只做
> harness 层、**锚点粒度 = 里程碑边界**(任意轮方案被否,见 §3.4)、
> round 接锚点计数、锚点解析复用现有 `milestone_advanced` 事件与 git
> 历史,**harness 主循环与事件流零改动**。

## 1. 目标(用户意图)

一个 harness loop 顺序推进多个里程碑(milestone),每个里程碑要跑很多
round。今天如果发现第 3 个里程碑跑歪了(坏实现被误判通过、门禁配置错、
agent 走进死胡同),唯一的选择是 `--reset-state` 从第 1 个里程碑重头跑,
前两个里程碑攒下的全部工作作废。

目标:**从某个已通过里程碑结束时刻的状态直接续跑**——即"loop 轨迹
续跑"。

## 2. 现状分析:哪些已经有,缺什么

### 2.1 已经有的

| 能力 | 位置 | 说明 |
|------|------|------|
| round 级续跑 | `harness/agent-loop.sh:662,716` | 同 `--loop-id` 重启,从 `<state_dir>/<stage>.{round,milestone,status}` 状态文件继续 |
| milestone 推进机制 | `harness/agent-loop.sh:862` `advance_stage_milestone` | dev agent 在 commit message 声明 `MILESTONE_STATUS: <name> PASS`,每轮结束扫 git log 推进 `<stage>.milestone` |
| **锚点数据已齐全** | `harness/agent-loop.sh:934-936` | `milestone_advanced` 事件已携带 `stage`/`round`/`from`/`to`/`commit`(被采信的 PASS 声明 commit SHA);git 历史亦可独立回溯声明 commit。**里程碑锚点不需要任何新增记录** |
| 持久化事件流 | `.artifacts/web-agents/loop-<id>/stdout.log` | wrapper 的 NDJSON `loop_event`,上述事件在此落盘 |
| workspace 全量持久化 | `.artifacts/forge_train/<loop_id>/workspace/` | 代码 + git 历史 |
| 远端目录按 loop 隔离 | `harness/tools/agent_loop_config.py:507`, `harness/tools/remote_workspace.py:71` | 远端 workdir = `<workspace>/.forge_train/<loop_id>`,不同 loop 互不污染 |
| dev agent 每轮无状态 | `harness/agent-loop.sh:1489` | 每 round 全新 spawn(`--resume` 仅用于同轮内重试);跨轮记忆的唯一载体是 workspace git 历史 + 状态文件 |

**最后一条是本方案可行性的基石**:agent 没有跨轮对话记忆需要恢复,
"续跑"只需要把 **文件系统状态** 恢复到目标时刻即可。

### 2.2 缺什么

只缺一样:**没有入口能表达"回到 milestone N 结束时"**。现有 resume 是
"从当前状态继续"——如果 milestone 3 已经把 workspace 写坏,继续跑
带着污染;`--reset-state` 则整个清零。锚点数据本身已经齐全(2.1 表
第三行),缺的只是消费它的工具。

## 3. 关键设计决策(已定)

### 3.1 分叉新 loop,而非原地回滚

续跑 = **fork 出一个新 loop_id**,其初始状态构造为"源 loop 在目标
里程碑结束时刻的快照";源 loop 完整只读保留。

- **源轨迹零破坏**:出问题的轨迹(git 历史、事件流、agent transcript)
  原样保留,可事后对比分析——这本身就是训练轨迹数据。
- **远端天然干净**:远端 workdir 按 loop_id 隔离(见 2.1 表),新
  loop_id 拿到全新远端目录。出问题里程碑期间产生的脏 scratch / 半截
  ckpt 不可能被续跑捡到(历史上踩过"脏 scratch 触发 auto-resume 零步"
  的坑,原地回滚必须额外处理,分叉自动消解)。
- **事件流 / web 回放无分叉歧义**:每个 loop_id 的 round 序列保持
  线性,不需要在回放里表达"时间线回卷"。

### 3.2 fork 的 workspace 用复制,不用 git worktree

- worktree 与源仓库**共享 .git**(对象库、分支、tag 全局共享),而
  dev agent 在 loop 里会跑任意 git 命令(reset/clean/tag/branch),
  两条时间线互踩 refs 的风险真实存在,"源轨迹零破坏"目标被打破;
- worktree 的 `.git` 是指针文件,源 loop 目录一旦归档/清理,fork 挂;
- 收益小:workspace 是 `harness/` 的一份拷贝(几十 MB 量级),复制
  本来就便宜。将来嫌大可换 `git clone --local`(对象 hardlink、refs
  独立),同样不共享 refs。

### 3.3 只做 harness 层

meta_harness 的里程碑循环(`meta_harness/agent-loop.sh:762`)不做。
meta loop 短、带人工 gate,重跑代价小;真实痛点在 harness dev loop。

### 3.4 锚点粒度 = 里程碑边界(任意轮方案被否)

只支持"回到 milestone N 结束时"。曾考虑支持任意轮,被否,理由:

1. **里程碑边界必定是一个干净的 commit**:推进的采信对象就是携带
   `MILESTONE_STATUS: <name> PASS` 的 commit,门禁验的是提交了的
   代码,锚点语义干净;
2. **轮边界在语义上就是坏锚点**:round 结束时工作区可能有未提交、
   未暂存的脏修改,下一轮是**带着这些脏文件**继续跑的——即
   "round N 结束的 HEAD" ≠ "round N 结束的完整状态",reset 到该
   HEAD 会静默丢掉这部分,续出来的轨迹和原轨迹从此说不清;
3. 砍掉任意轮后,锚点数据全部现成(§2.1),harness 零改动。

### 3.5 round 接锚点计数

fork 到 milestone N(其推进发生在 round R),新 loop 的 `.round` 写
R,续跑第一轮是 R+1。烧掉的轮数不计入——round 反映新时间线的真实
进度;返工成本从源 loop 的轨迹和 fork 溯源字段里查。

### 3.6 fork 之后走正常启动路径,主循环零改动

fork 的本质是**构造一个状态文件恰好等于目标时刻的新 loop 目录**,
然后用现有的 `--loop-id <new_id>` 启动,现有 resume 契约自动接管。
锚点数据现成、主循环不动,**全部新逻辑收敛在一个独立的 fork 工具里**
(§4.1),外加启动段一小截 `loop_forked` 事件(§4.2)。

## 4. 机制设计

### 4.1 fork 工具:`harness/tools/fork_loop.py`

CLI:

```
python3 harness/tools/fork_loop.py \
    --src-loop-id <id> \
    --at-milestone <name>          # 锚点:该里程碑被采信通过的声明 commit
    [--at-sha <sha>]               # 显式锚点,覆盖自动解析(兜底)
    [--new-loop-id <id>]           # 缺省自动生成
    [--dry-run]                    # 只打印将执行的动作
```

步骤(全部在本机 `.artifacts/forge_train/` 下操作):

1. **前置检查**
   - 源 loop 存在且**没有活跃进程在跑**(检查 session.json 状态 /
     wrapper 存活);在跑则拒绝,提示先停。
   - `--at-milestone` 必须属于 stage1 的 milestone_order 且在源 loop
     中已通过(stage2 是算子 fan-out,无 milestone manifest;源 loop
     已进 stage2 时,仍可 fork 回 stage1 的里程碑锚点)。

2. **解析锚点**(两个互相印证的来源):
   - **首选事件流**:读
     `.artifacts/web-agents/loop-<src>/stdout.log` 中目标里程碑的
     `milestone_advanced` 事件,一次拿到全部三个值:
     锚点 SHA(`commit`)、`.milestone` 应写入值(`to`)、
     `.round` 应写入值(`round`)。
   - **git 回溯兜底**(事件流缺失/损坏时):
     `git log --grep '^MILESTONE_STATUS: <name> PASS'` 找声明
     commit;`.milestone` 由 milestone_order 计算后继。此路径拿不到
     round(写 0,续跑轮号从 1 起,仅影响展示不影响正确性,打印
     警告);review-gated 里程碑可能有被 review 否决的历史声明,
     无事件流交叉验证时歧义,报错要求 `--at-sha`。
   - `--at-sha`:最高优先级,校验该 commit 存在且其 message 携带
     目标里程碑的 PASS 声明。

3. **搭建新 loop 目录** `.artifacts/forge_train/<new_id>/`:
   - `workspace/` ← 复制源 workspace(含 `.git`),然后
     `git reset --hard <锚点SHA>` + `git clean -fd`(未跟踪脏文件
     一并清掉,锚点状态 = 纯提交状态,呼应 §3.4 的语义)。
   - `config/` ← 复制源 config 各 axis TOML。源目录是
     `chmod -R a-w` 冻结的,复制后先恢复写权限;若
     `[remote].kind = devspace`(hostname 是 lease 派生的),清掉
     hostname 交给启动路径重新 claim lease;然后重新冻结。
   - `agent-loop-state/`:
     - `<stage1>.milestone` ← 锚点解析的 `to` 值(下一个里程碑)
     - `<stage1>.status` ← `in-progress`
     - `<stage1>.round` ← 锚点解析的 `round` 值(续跑第一轮 = R+1)
     - 不写 stage2 的任何状态文件(保持"未开始"语义)。
   - 溯源写独立文件 `forked_from.json`(与 session.json 同级):

     ```json
     {
       "loop_id": "<src>",
       "milestone": "<name>",
       "anchor_sha": "<sha>",
       "anchor_source": "event | git-fallback | explicit-sha",
       "forked_at": "<iso8601>"
     }
     ```

     不放进 session.json:该文件由 `_publish_session_json`
     (agent-loop.sh:1714)在每次启动时**全量重写**,写进去会被抹掉。

4. **打印后续操作指引**:输出 `agent-loop.sh --loop-id <new_id> ...`
   启动命令(fork 工具本身不启动 loop,启动权留给用户/web)。

### 4.2 可观测性

- 新 loop 启动时,wrapper 在事件流(`loop-<new_id>` 的 stdout.log)
  emit 一条 `loop_forked` 事件:`src_loop_id`、`milestone`、
  `anchor_sha`。实现:agent-loop.sh 在首次 `_publish_session_json
  running` 之后读 `forked_from.json`,存在即 emit;同目录哨兵文件
  `.loop_forked_emitted` 保证 fork loop 自身后续续跑重启不重复 emit。
- web Loop tab 读到该事件可渲染 "forked from \<src\> @ \<milestone\>"
  角标(前端渲染属二期,事件先落数据)。
- (可选,锦上添花)`advance_stage_milestone` 推进时打轻量 tag
  `ms/<stage>/<name>` 指向声明 commit,纯粹方便人手 `git describe`;
  机器解析不依赖它,可与主体解耦单独提交或不做。

## 5. 边界与风险

| 场景 | 处理 |
|------|------|
| 源 loop 还在跑 | fork 前置检查拒绝,要求先停(避免读到半写状态) |
| 目标里程碑未通过 / 属 stage2 | 报错;允许从已进 stage2 的源 loop fork 回 stage1 里程碑 |
| 事件流缺失/损坏 | git log 回溯兜底;round 不可得则写 0 并警告 |
| review-gated 里程碑有被否决的历史声明 | 事件流的 `commit` 字段是采信真值;仅在走 git 兜底路径时有歧义,报错要求 `--at-sha` |
| 锚点 commit 之后、推进采信之前 dev 又打了提交 | 锚点=声明 commit,同轮更晚的提交被裁掉——符合"里程碑结束时刻"语义(那些提交属于下一段工作) |
| 未跟踪脏文件 | fork 时 `reset --hard` + `clean -fd` 一并清掉 |
| 新 loop 远端目录 | 不需要处理:loop_id 隔离,首次 sync push 推送回滚后的 workspace 到全新目录 |
| devspace lease | fork 不继承源 loop 的 lease;新 loop 走正常 claim 路径拿自己的 devspace |
| round 计数 | 接锚点 round 递增,不回拨、不清零——单一 loop 内事件流单调 |
| config 漂移 | fork 复制的是**源 loop 冻结时刻**的 config,不是 `harness/config/` 模板的当前值——续跑语义要求环境与源一致 |

## 6. 实施计划(按提交顺序)

harness 主循环零改动,工作量集中在 fork 工具。每步独立可提交、可单测。
状态:1-3 已实现(`tools/fork_loop.py` + `harness/tests/test_fork_loop.py`
10 个用例 + agent-loop.sh 的 `_emit_loop_forked_once`);4 未做;5 二期。

1. **锚点解析库**(`fork_loop.py` 解析部分)
   - 事件流查询 `milestone_advanced`、git log 回溯兜底、`--at-sha`
     校验、review-gated 歧义检测。
   - 单测:构造带/不带事件流、带被否决声明的 fixture。
2. **fork 目录构建**(`fork_loop.py` 主体)
   - workspace 复制 + reset + clean、config 复制/解冻/重冻、状态
     文件写入、session.json 溯源字段、前置检查、`--dry-run`。
   - 集成测:fork 后用 `--loop-id <new_id>` 干跑一轮,断言从锚点
     round+1、锚点后继 milestone 起跑。
3. **`loop_forked` 事件**
   - agent-loop.sh 启动段读 `forked_from` 并 emit。
4. **(可选)里程碑 tag**:`advance_stage_milestone` 推进时打
   `ms/<stage>/<name>`,纯人手便利,独立小提交。
5. **(二期,另行评审)** web:`POST /api/loop/<id>/fork` + Loop tab
   角标 + fork 按钮。

## 7. 评审决策记录

| 问题 | 决策 |
|------|------|
| 原地回滚 vs 分叉新 loop | 分叉新 loop |
| fork workspace:复制 vs git worktree | 复制(worktree 共享 .git/refs,agent 会跑任意 git 命令,隔离性不可接受) |
| 作用范围 | 只做 harness 层 |
| round 计数起点 | 接锚点 round(烧掉的轮不计入) |
| 锚点粒度 | **只做里程碑边界**。曾定"任意轮",复议后否:轮结束时可能有未提交/未暂存的脏修改,轮边界 HEAD ≠ 轮完整状态,锚点语义不干净;里程碑必定落在干净 commit 上 |
| 轮锚点记录机制(已随任意轮一并作废) | v2 boundary-commit、v3 事件流带 SHA 均不再需要;`milestone_advanced` 事件现成字段已覆盖里程碑锚点全部所需 |
