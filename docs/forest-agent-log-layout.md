# 设计：把扁平的 agent 日志目录改造成森林结构

状态：草案（待评审）
作者：harness
日期：2026-06-29

---

## 1. 背景与动机

一个 loop 在运行中会延伸出多个 agent（1 个 wrapper + N 个 dev round /
review 子 agent）。逻辑上这是一棵树：

```
loop-<id> (wrapper)
├── web-... (dev round)
├── web-... (review)
└── ...
```

但磁盘上当前是**扁平**的——wrapper 和所有子 agent 都是 `AGENTS_DIR/`
下的兄弟目录，父子关系只靠 `session.json` 里的 `loop_id` /
`parent_agent_id` 字段维系：

```
.artifacts/web-agents/
├── loop-98179bd97008/      # wrapper
├── web-1be11b44271f/       # dev round（与 wrapper 平级）
├── web-1243bb0c7ab8/       # review（与 wrapper 平级）
└── ...
```

问题：浏览 / 归档 / "推轨迹" 时，一个 loop 的若干 agent 散落在同一层，
无法一眼归组；删除整个 loop 要逐个删；导出一个 loop 的完整轨迹要先按字段
筛选再收集。

目标：让 `AGENTS_DIR` 成为一个**森林**——同一个 loop 的所有 agent 归到该
loop 的树根目录下；独立 chat（无 loop）作为单节点树留在顶层。**且 loop 内部
agent 本就有序（按 stage→round→spawn 顺序），要在目录层面直接体现这个顺序**，
而不是只靠 session.json 字段事后排序。

```
.artifacts/web-agents/
├── loop-98179bd97008/                       ← 树根 = wrapper 自己
│   ├── session.json  stdout.log  recording/
│   └── agents/
│       └── stage1/                          ← 按 stage 分组
│           ├── r001/                        ← 按 round 分组（有序）
│           │   ├── loop-98179bd97008_stage1_r001_01_web-1be11b/  ← dev（序号 01）
│           │   └── loop-98179bd97008_stage1_r001_02_web-1243bb/  ← review（序号 02）
│           ├── r002/ ...
│           └── r003/ ...
├── web-3f9a.../                             ← 独立 chat = 单节点树
└── loop-<其它loop>/ ...
```

> round 有序性见 §3.3。`stage` / `round` / `seq` 三段都零填充，目录字典序
> 即 spawn 时序。

---

## 2. 现状与核心约束

### 2.1 布局 SSOT 只有一个函数

整个目录拓扑只由一个纯函数决定：

```python
# web/agents/store.py:112
def session_dir(agent_id: str) -> Path:
    return AGENTS_DIR / agent_id
```

`stdout_file` / `stderr_file` / `session_file` 全部从它派生。改布局的爆破
半径因此被收敛到这一处。

### 2.2 决定性约束：调用方只有 `agent_id`，没有 `loop_id`

绝大多数路径解析的入口只持有 `agent_id`：

- HTTP 路由 `/api/agent/<agent_id>/messages` → `agent.py:454`
  `store.stdout_file(agent_id)`
- resume 路径 / pump / `store.load(agent_id)` / `store.save(session)`

要把子 agent 物理嵌套，`session_dir(agent_id)` 必须**只凭 agent_id 反推出
它属于哪个 loop**。若引入"agent_id→loop_id 索引文件"或"glob 扫描"，会破坏
当前纯函数 / O(1) / 无 I/O / 单一真源的性质（也违反 SSOT 与 fail-fast 规约）。

**解法**：把层级编码进 agent_id 本身——给子 agent 一个自描述、不含斜杠、
URL 安全的复合 id（完整形态见 §3.1）。这样 `session_dir` 仍是纯函数。

**分隔符约定**（评审决策点）。id 由若干"块"拼成，块之间 + 有序标签内部需要分隔符。

- **我们引入的分隔一律用 `_`**：块边界（root / 有序标签 / 后缀）与有序标签内部
  （stage / round / seq）统一 `_`。解析靠"第一个 `_` 是主边界，其余按位置切"。
- **不用 `-` 当分隔**：`-` 已被既有原子 id 形态占用——wrapper 是 `loop-<loop_id>`、
  后缀是 `web-<uuid>`（现有 `wrapper_agent_id` / `new_agent_id` 产物）。这两块内部的
  `-` 是**沿用、不改**（改它们会波及 `wrapper_agent_id` 的大量调用点，纯为美观折腾）；
  把它们当**不可拆的名字**看待。除此之外我们不再引入任何 `-`。
- **不用 `~`**：tilde 只在词首展开、我们的字符永远在 token 中间，技术安全；但 `~`
  强联想 home 魔法，是"看着危险"的字符，不适合会被 grep / copy / 塞进 rsync·URL
  的标识符。
- **不用 `.`**：有扩展名 / 隐藏文件联想，部分路径处理按 `.` 切，次优。
- **约束**：loop_id 与 stage 名都不得含 `_` 或 `-`（loop_id 为 hex、stage 为 `stage1`，
  均满足；`--loop-id` 须校验为 `[a-z0-9]`）。前端哨兵 `_draft_new_agent` 以 `_` 开头但
  永不落盘，`session_dir` 仍以 `not root` 兜底（见 §3.2）。

净结果：除两个原子名 `loop-<id>` / `web-<uuid>` 内部沿用的 `-`，**我们设计的所有分隔
都是 `_`**。

### 2.3 前端不依赖磁盘布局（已逐处实测）

结论：**改 agent_id 格式 / 物理布局不影响前端渲染。** 证据：

- **分类一律靠 `kind` 字段，无人按 id 前缀判定**：后端 `agent.py:291` /
  `internal_replay.py:58` / `spawn.py:684`，前端 `app.js:299/304/756`
  （`a.loop_id === loop.loop_id && a.kind !== 'loop_wrapper'`），回放工具
  `driver.js / camera.js / timeline.js` 全按 `kind`。全库唯一的 `loop-`
  前缀判断是 `backends.py:72` `self.name == "loop-wrapper"`——判的是 **backend
  名**，不是 agent_id 前缀。这点关键：子 agent 的复合 id 也以 `loop-` 开头，
  若有谁 `startsWith('loop-')` 就会误伤——经查没有。
- **agent_id 只作不透明 key + URL 参数**：进 URL 一律
  `fetch('/api/agent/${encodeURIComponent(agentId)}/...')`；进 DOM 是
  `data-agent-id="${escapeHtml(...)}"` 再读回；其余是等值比较。`~` / `_`
  都是 URL unreserved + HTML 属性安全字符。

### 2.4 读侧与 recording 不受影响

- `web/agents/messages.py` / `transcript.py` 接收的是**已解析的 Path**
  （由 `agent.py:_transcript_source` 先算好再传入），与布局无关。
- `recording.py` / `internal_replay.py` 用的都是 wrapper（树根）路径
  （`session_dir(wrapper_agent_id(loop_id))`），嵌套后树根位置不变，零改动。
- `runner.ide_transcript_path` 解析的是 `~/.cursor/...`，与 `AGENTS_DIR` 无关。

### 2.5 "推轨迹" 是手动 rsync

把 loop 轨迹推到山东集群（如 `/user/heqingfeng/traja/98179bd9/`）是用户手动
rsync `.artifacts/forge_train/<id>/` 加上收集 web-agents，仓库内无专用工具。
本设计只覆盖**运行时日志存储布局**；存储变成森林后，推轨迹直接整树 copy 即可，
无需额外工具。（顺带提醒：归档目录名应使用完整 loop_id `98179bd97008`，
而非现状的 8 位前缀 `98179bd9`，以免对不上。该截断在导出环节，属另一议题。）

---

## 3. 设计

### 3.1 id 约定

子 agent 的 id 把 **stage + round + 序号**编进一个有序标签
`<stage>-r<RRR>-<SS>`（`RRR` / `SS` 零填充，保证字典序 == 时序）：

| 角色 | agent_id | 目录 |
|------|----------|------|
| wrapper（树根） | `loop-<loop_id>` | `AGENTS_DIR/loop-<loop_id>/` |
| 子 agent | `loop-<loop_id>_<stage>_r<RRR>_<SS>_web-<uuid>` | `AGENTS_DIR/loop-<loop_id>/agents/<stage>/r<RRR>/<full-id>/` |
| 独立 chat | `web-<uuid>` | `AGENTS_DIR/web-<uuid>/`（单节点树，留顶层） |

例：`loop-98179bd97008_stage1_r003_02_web-1243bb0c7ab8`
→ `loop-98179bd97008/agents/stage1/r003/loop-98179bd97008_stage1_r003_02_web-1243bb0c7ab8/`

分隔符 `_`（块间 + 有序标签内统一）、序号宽度都定义在 `web/paths.py` 一处，禁止散落
复制。`loop-`/`web-` 内的 `-` 是沿用既有原子 id 形态，不计入本设计的分隔约定。

### 3.2 路径解析（仍是纯函数 / O(1) / 无 I/O）

```python
# web/agents/store.py
def session_dir(agent_id: str) -> Path:
    root, sep, rest = agent_id.partition(AGENT_ID_SEP)   # 第一个 "_" = 主边界
    if not sep or not root:                              # wrapper / 独立 chat / 哨兵
        return AGENTS_DIR / agent_id                     # not root 防 `_`-前缀哨兵
    stage, round_dir, _seq, _suffix = rest.split(AGENT_ID_SEP, 3)  # stage1,r003,02,web-<uuid>
    return AGENTS_DIR / root / "agents" / stage / round_dir / agent_id
```

叶子目录名 == 完整 agent_id，保证 id ↔ path 一一映射，无歧义。解析只做字符串
切分，不读盘。`not root` 那一支是防御：前端草稿哨兵 `_draft_new_agent` 以 `_`
开头（`partition` 后 root 为空），虽然它被前端过滤、永不落盘到 `session_dir`，
仍兜底当扁平处理而非拼出 `AGENTS_DIR//agents/...` 的坏路径。

### 3.3 round 有序性从哪来（已验证可行）

`agent-loop.sh` 在 spawn 那一刻就持有 `stage` / `round` / 序号，可直接透传：

- **dev**：`run_agent_stage` 主循环里 `round` 在 scope（`agent-loop.sh:1316/1318`），
  序号恒为 `01`。
- **review**：`run_review_agent` 收 `round`（`:1161`）并在同 round 内 spawn
  （`:1170`），序号 `02`。
- **dev 重试 = resume 同一 agent_id**（`:1318` 传 `$dev_agent_id`），**不新建
  目录**，故一个 round 通常只产生 2 个叶子（dev/review），序号天然有序。
- round 是 **per-stage** 的（`stage_round_file` per stage），多 stage 时 round
  号会跨 stage 重复，**所以 stage 必须进 id**（否则 `stage1/r001` 与
  `stage2/r001` 撞目录）。
- 序号当前由调用点按 kind 直接给（dev=1 / review=2）。未来若一个 round 派生多个
  `loop_subagent`，再在 shell 里引入 per-round 自增计数器即可，id 格式不变。
- **resume 不重新编码**：resume 时 agent_id 已存在，不走 `new_agent_id`，路径稳定。

### 3.4 与旧扁平归档的天然共存

旧归档里子 agent 的 id 是**扁平** `web-<uuid>`（hex，不含 `_`），新
`session_dir` 对其走 `AGENTS_DIR/web-<uuid>` 分支，仍能读出；`list_sessions`
也会在顶层找到它们。因此**无需写迁移代码**，旧归档可继续读，只是不物理嵌套。
嵌套只对**新建** loop 生效（符合项目 "无向后兼容" 规约）。

---

## 4. 改动清单（逐文件）

> 仅运行时 7 处生产代码 + 测试。读侧、recording、前端零改动。

### 4.1 `web/paths.py`
- 新增 `AGENT_ID_SEP = "_"`、序号宽度常量（round 3 位 / seq 2 位）。
- 新增 `child_agent_id(loop_id, stage, round_no, seq, suffix) -> str`：返回
  `f"{wrapper_agent_id(loop_id)}_{stage}_r{round_no:03d}_{seq:02d}_{suffix}"`。
  （参数名用 `round_no`，避免遮蔽内建 `round()`。）
- `wrapper_agent_id` 不变（它就是树根）。
- 导出到 `__all__`。

### 4.2 `web/agents/store.py`
- `session_dir`：改为 §3.2 的有序标签解析（stage / round 两级中间目录）。
- `new_agent_id(loop_id=None, stage=None, round_no=None, seq=None) -> str`：四者俱全
  时走 `child_agent_id(...)`，否则返回扁平 `web-<uuid>`（独立 chat）。
- `list_sessions`：多层遍历——顶层 + 每个 `loop-*/agents/<stage>/r<RRR>/` 叶子。
  实现上对顶层每个 `loop-*` 递归 `agents/` 子树即可（深度固定为 stage/round 两级）。
- `delete`：基于 `session_dir(agent_id)` rmtree。删 wrapper 树根时整棵子树
  （含 `agents/`）一并删除——**"删一个 loop = 删整棵树" 成为天然能力**。

### 4.3 `web/agents/spawn.py`
- `spawn_session`：新增 `stage` / `round` / `seq` 形参，
  `agent_id = resume_agent_id or store.new_agent_id(loop_id, stage, round, seq)`。
  这些字段仅用于新建子 agent 时构造 id；resume 时已有 id，忽略。
- `precreate_web_session`（`spawn.py:526`）：保持
  `new_agent_id()`（web chat 无 loop / round，留扁平）。
- 其余 `session_dir/stdout_file/...` 调用全部经由 SSOT，无需改。

### 4.3a `harness/tools/spawn_managed_agent.py`（透传 round 链路）
- `_parse_args`：新增 `--stage` / `--round`（int）/ `--seq`（int），仅对
  loop 子 agent 必填。
- `_run`：把三者传入 `spawn.spawn_session(...)`。
- 可选增强：`append_loop_event("spawn_child", {..., "round": round, "seq": seq})`，
  让 wrapper 的 stdout.log 里 spawn_child 卡片也带 round。

### 4.3b `harness/agent-loop.sh`（在 spawn 点提供 stage/round/seq）
- `_spawn_child_agent`：新增形参 `stage` / `round` / `seq`，拼进 `spawn_args`
  （`--stage/--round/--seq`）。
- dev 调用点（`:1316/1318`）：传 `"$stage" "$round" 1`。
- review 调用点（`run_review_agent`，`:1170`）：传 `"$stage" "$round" 2`。
- resume 分支（`:1318` 传 `$dev_agent_id`）：round 段已固化在原 id 里，
  spawn_managed 走 resume，不再用 `--round` 重新编码（无害，仍可传）。

### 4.4 `web/routers/loop.py`
- `_draft_last_active`（`loop.py:374`，草稿回收器）：**保持顶层 `iterdir` +
  裸 `json.loads` 不变**。修正初版设计——这里**不**改用 `store.list_sessions()`：
  (1) 草稿是启动前状态，指向其 instance_dir 的只有扁平的 "Loop Create" chat 会话；
  嵌套的 dev/review 子 agent 仅在启动后（status≠draft，此函数不被调用）才出现，
  故顶层迭代对草稿是完备的；(2) `store.load` 的惰性僵尸 reconcile 会改写
  session.json、抬高其 mtime，使陈旧草稿显得"活跃"而回收不掉——必须避免。仅加注释
  说明该不变量。
- `loop.py:1618` `agent_store.delete(wrapper_agent_id)`：天然变成整树删除
  （`delete` 经 `session_dir` rmtree 树根），逻辑不变、语义更对。无需改。

### 4.5 其余 `session_dir`/`wrapper_agent_id` 调用方
- `recording.py` / `internal_replay.py` / `spawn_managed_agent.py:151`
  （spawn_errors.log 写到 wrapper 树根）：用的都是 wrapper 树根，**零改动**。
- `agent.py:454`、resume：仅持 agent_id，经 `session_dir` 自动解析，零改动。

### 4.6 测试改动（项目强制 TDD：先改/加失败测试，再实现）
- `test_unified_agent_log_foundation.py`：
  - 新增/调整：子 agent 用有序复合 id（`loop-X_stage1_r003_02_web-Y`）落到
    `loop-X/agents/stage1/r003/<full-id>/`；断言 `session_dir` 解析出
    stage / round 两级中间目录。
  - 新增：`new_agent_id(loop, stage, round, seq)` 产出零填充、字典序==时序的 id；
    无 round 参数时回退扁平。
  - 保留：扁平 chat（`web-x` / `web-legacy`）仍落顶层并可 `load`。
  - 新增：`list_sessions` 能同时枚举顶层与嵌套子 agent。
  - 新增：`delete(wrapper)` 删除整棵子树。
- `test_loop_disk_registry.py` / `test_loop_wrapper_messages_api.py` /
  `test_loop_wrapper_events_e2e.py` / `test_spawn_managed_agent_cli.py` /
  `test_recording_api.py` / `test_loop_active_endpoint.py` / `test_loop_drafts.py`：
  逐个核对其中对 `AGENTS_DIR/<id>` 的路径断言，凡涉及子 agent 的改为复合
  id + 嵌套路径；涉及 wrapper / chat 的不变。
- `test_web_layer_dag.py:807`（`new_agent_id` 签名引用）：同步新签名。

---

## 5. 不受影响（明确列出以降低评审负担）

- 前端 `agents.js`（用字段建树，agent_id 仍是合法 URL token）。
- `messages.py` / `transcript.py`（接收已解析 Path）。
- `recording.py` / `internal_replay.py`（用 wrapper 树根）。
- `runner.ide_transcript_path`（`~/.cursor/...`，与 AGENTS_DIR 无关）。
- `.artifacts/forge_train/<id>/` 注册记录布局（与 web-agents 正交）。
- 旧扁平归档（id 不含 `_`，照旧可读）。

---

## 6. 行动计划（分阶段、可独立验证）

**Phase 0 — 锁定契约（先写测试，红）**
1. 在 `test_unified_agent_log_foundation.py` 写出 §4.6 的新断言：
   - `session_dir("loop-X_stage1_r003_02_web-Y") ==`
     `AGENTS_DIR/"loop-X"/"agents"/"stage1"/"r003"/"loop-X_stage1_r003_02_web-Y"`
   - `session_dir("loop-X") == AGENTS_DIR/"loop-X"`
   - `session_dir("web-Z") == AGENTS_DIR/"web-Z"`
   - `new_agent_id("X","stage1",3,2)` 含有序标签且零填充；`new_agent_id()` 不含。
2. 跑测试确认红。

**Phase 1 — 路径层（最小实现，转绿）**
3. `web/paths.py`：加 `AGENT_ID_SEP` + 宽度常量 + `child_agent_id`。
4. `web/agents/store.py`：改 `session_dir` / `new_agent_id` /
   `list_sessions` / `delete`。
5. 跑 Phase 0 测试转绿；跑 `store` 相关全量回归。

**Phase 2 — 接入 spawn + round 透传（端到端）**
6. `web/agents/spawn.py`：`spawn_session` 加 `stage/round/seq` 形参并传入
   `new_agent_id`。
7. `harness/tools/spawn_managed_agent.py`：加 `--stage/--round/--seq` 并透传。
8. `harness/agent-loop.sh`：`_spawn_child_agent` 加三参；dev 传 `(stage,round,1)`、
   review 传 `(stage,round,2)`。
9. 修 `test_spawn_managed_agent_cli.py` / `test_loop_wrapper_*` 的路径断言。
10. 端到端：起一个最小 loop（或用现有 e2e 测试），确认磁盘出现
    `loop-<id>/agents/stage1/r001/loop-<id>_stage1_r001_01_web-*/` 嵌套结构，
    且 `/api/agent/<复合id>/messages` 能读到内容。

**Phase 3 — 收尾**
11. `web/routers/loop.py:374` 交联改用 `list_sessions()`；验证
    `loop.py:1618` 整树删除。
12. 全量 `pytest`；人工核对 web Loop tab 树视图正常。
13. 更新 `harness/CLAUDE.md` 的 "Workspace isolation" 段落，描述新嵌套布局
    （stage/round 分组 + 有序子 agent id）。

**验证口径**
- 单测：上述各 test 文件全绿。
- 端到端：新 loop 产出嵌套目录；前端树、消息、recording、删除均正常。
- 回归：旧扁平归档仍可 `load` / `list_sessions`。

---

## 7. 风险与回滚

| 风险 | 评估 | 缓解 |
|------|------|------|
| agent_id 格式变更波及面 | 低——只新增分隔符，wrapper/chat 不变；前端用字段不解析 id | 全量测试覆盖；`_` 无 shell/glob/扩展名语义 |
| 复合 id 偏长（含 loop_id + stage-round-seq） | 低——纯机器路径，换叶子名==id 的可调试性 | 如需可让叶子名去掉 `loop-` 前缀，但会破坏 id↔path 1:1，不推荐 |
| round 透传链路（shell→py） | 中——新增 3 个参数贯穿 3 个文件 | per-stage round 语义不变；resume 不重新编码 |
| 测试改动量 | 中——约 8 个测试文件涉路径断言 | 多为机械替换；Phase 化推进 |
| 旧归档读取 | 低——扁平 id 不含 `_`，自动走扁平分支 | 已在 §3.4 验证逻辑 |
| 回滚 | 低——SSOT 集中在 `session_dir` 一处 | 还原该函数 + `new_agent_id` 即回到扁平 |

回滚成本极小：核心逻辑集中在 `store.session_dir` / `new_agent_id` 两个函数，
还原即恢复扁平布局（新建 loop 重新扁平，已嵌套的旧目录因 id 仍带 `_`
需手动摊平或忽略）。
