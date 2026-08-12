# data 轴 SSOT + 单向依赖:全仓库理顺 data_loader / data_path / tokenizer

> **状态:已实现(A1–A4 + E)。** tokenizer 经确认保留在 ref 轴(已合规,不改)。
> meta 侧零改动(`ref_bundle_run.sh` 本就 D1)。data_path 已文件化,不改。
> 校验:`test_gate_transport_parity`(18)、`test_gate_family_invariants`(10)、
> `test_ours_env_inputs_transport`、`test_data_config_schema`、
> `TestResolveRef` 全绿;全仓库无 `os.environ[DATA_LOADER]` / `export DATA_LOADER`,
> 无 env_inputs 含 `DATA_LOADER`。(test_harness_core 的 `_resolve_tokenizer` /
> 缺 model.toml / pytest 三处失败为**既有**问题,与本次无关。)

## 一条总原则

> **值只存一处(SSOT),依赖单向流动;环境变量只传"指针(路径)",绝不传"值"。**
> render 产物 `<gate>.toml` 只承载 model/optim/ref 的投影,**零 data 轴的值**。
> 每个消费者从指针指向的**文件**里读值。数据永远是 `data.toml → 消费者`,没有回写、没有第二副本。

判据:`FORGE_DATA_TOML=/…/data.toml`(路径)= 合法指针;`DATA_LOADER=hf`(值)= 违规,必须消灭。

## 三条子轴的目标态

| 子轴 | 唯一值源 (SSOT) | 指针 (env) | 值的传输 | 消费者 |
|---|---|---|---|---|
| **data_loader** | `data.toml [data].data_loader` | `FORGE_DATA_TOML` | 文件 (`tomllib`) | ref/sisters `train*.py`(`--data-config`)、ours(`runtime_config.data_loader()`)、meta bundle(已然) |
| **data_path** | `data.toml [data].conf_path` 指向的 conf 脚本(生成加权 shard 串) | `DATA_CONF`(→conf)/`DATA_PATH_FILE`(→已落盘的值) | 文件 (`--data-path-file`) | 所有 `train*.py`(早已文件化) |
| **tokenizer** | `ref.toml [ref].forge_tokenizer_dir` + `tokenizer`(repo id) | `FORGE_TOKENIZER_DIR`(目录路径) | 目录(磁盘上的 tokenizer 文件) | `hf_stream_dataloader.py`(已然) |

**data_path 与 tokenizer 已合规**(文件/目录指针 + 单源),本次基本不动。
**唯一没理顺的是 data_loader**:主链 D1 已迁一半,sisters 仍走 env,且 renderer/config_runtime 还残留旧的"提升进 env"副本。

## 改动清单

### A. data_loader —— 收尾整条迁移

**A0. 主链(0.5b MTP + 1b)—— D1 已完成,仅校验**
- `train_pure_mup_mtp.py`:`--data-config` + `_data_loader_from_config`(275-292) ✅
- `runtime_config.data_loader()`:经 `FORGE_DATA_TOML` 直读 ✅
- `render_gate_configs.py`:产物不再有 `DATA_LOADER` ✅
- `dense_training_1b/*.toml`:env_inputs 已无 `DATA_LOADER` ✅
- `_common.py:suite_process_env`:ours 子进程注入 `FORGE_DATA_TOML`(250) ✅

**A1. sisters —— 从 env 读改成 `--data-config` 文件读**(核心新增)
把 `train_pure_mup_mtp.py:275` 那个自包含 helper 复制进各 sister(refs 要求自包含,不跨目录 import,允许小段复制):
- `ref/reference/train_qwen3_dense.py`:加 `--data-config` + helper;替换 `loader=os.environ.get("DATA_LOADER","")`(193)。
- `ref/reference/train_minicpm4_8b_tp.py`:同上(227)。
- `ref/reference/train_minicpm4_8b_hf_singlecard.py`:同上(237)。

各自 launcher 透传 `--data-config "$FORGE_DATA_TOML"`(照 `ref_bundle_run.sh` 约定,shell 不抽值):
- `run_qwen3_dense.sh`:PY_ARGS 加 `--data-config`(219 `--data-path-file` 旁)。
- `run_minicpm4_8b_dptp.sh`:同(232)。
- `run_minicpm4_8b_hf_singlecard.sh`:加 `--data-config`;**删** `export DATA_LOADER`(34)与 DATA_CONF→DATA_LOADER 耦合注释(29)。DATA_CONF 此后只为 DATA_PATH 服务。

**A2. 干掉 configs 里的 DATA_LOADER env 声明**
- `config/eval/dense_training/dense_training.toml`:env_inputs 删 `DATA_LOADER`(261、315)。
- `config/eval/dense_training_qwen3/dense_training_qwen3.toml`:env_inputs 删 `DATA_LOADER`(318)。
- (`dense_training_1b` 已删。)

**A3. renderer —— 确保产物零 data 值**
- `ENV_INFRA` 已无 `DATA_LOADER` ✅。
- **删** `_build_baseline` 里对字面 `data_path` 的投影(`render_gate_configs.py:153-154
  if "data_path" in datab: resolved["data_path"] = …`),使**任何** data.toml 字段都不再落进产物。
  (harness 用 conf_path、无字面 data_path,该分支对 harness 是死代码;删掉是贯彻"别把 data.toml
  渲染进产物"。已核:`gate_product_to_shell` 跳过 infra `[env]`,ours 只读 `[cli]`,无人消费产物里的
  data_path,删除安全。)

**A4. config_runtime.py —— 拆掉旧的"conf 导出→提升进 env/ref"轨**
- 保留 238-247 的 `FORGE_DATA_TOML` 直读轨(好轨)。
- **删** data_loader 的提升:`_resolve_data_env_from_conf`(335 注释 + 363-366 从 conf stdout 切出
  `data_loader` 并写入 `out`)只保留 `data_path`;`_resolve_ref`(449-450 `if "data_loader" in
  resolved_env: ref["data_loader"]=…`)整段删。迁移后无下游消费者。

### B. data_path —— 已合规,不改代码
`data.toml.conf_path` 指向 conf 脚本(做 `FORGE_DATA_DIR` 插值 + 加权,是**真·计算源**,不宜塞成
data.toml 里的静态标量)→ run.sh 落盘 → `--data-path-file` → train.py。全程文件化、单向。
DATA_PATH-as-env 只是 launcher 内部 conf→文件的一跳,非值源,保留。

### C. tokenizer —— 已合规,不改代码
`ref.toml [ref].forge_tokenizer_dir` 单源 → `FORGE_TOKENIZER_DIR`(目录指针)→ `resolve_assets`
缺失即下载 → `hf_stream_dataloader` 读目录。指针经 env、值在目录文件,合规。
**待你拍板**:tokenizer 现挂在 **ref 轴**(非 data 轴)。它与 data 相邻但也与 model vocab 绑定,
搬轴会牵动 megatron 路径,属额外风险。**建议维持 ref 轴**(已单源、已合规);若你想归到 data 轴再单开一项。

### D. meta —— 零改动
`ref_bundle_run.sh` 已经 `FORGE_DATA_TOML` + `--data-config`,注释明确"data 值在独立 data.toml,
不在 gate config"。它本就是 D1。无需 SKILL steer——之前"约束 meta 读产物"的方向已废弃。

### E. 测试
- `test_gate_transport_parity.py`:断言 baseline/产物含 `data_loader` → 改为断言**不含**。
- `test_harness_core.py`:`[env]` 里 `DATA_LOADER` 相关断言。
- `test_ours_env_inputs_transport.py`:若断言 ours 经 env_inputs 拿 DATA_LOADER,改为经 `FORGE_DATA_TOML`。
- `test_data_config_schema.py` / `test_hf_stream_dataloader.py`:同步 data_loader 传输语义。
- 可补:一条"sisters 读 `--data-config`、仓库无 `os.environ[...DATA_LOADER...]`、无 `export DATA_LOADER`、
  无 env_inputs 含 DATA_LOADER"的守卫测试。

## 验证
1. 渲染一个 gate:产物 `[cli]` 无 `data_loader`,`[env]` 无 `DATA_LOADER`/`data_path`。
2. `grep -rn "environ.*DATA_LOADER\|export DATA_LOADER" ref/ workload/ evals/ tools/` → 空。
3. `grep -rn "DATA_LOADER" config/eval/*/*.toml` → 空(env_inputs 全清)。
4. ref + 三个 sister:launcher 传 `--data-config`,python 从 data.toml 读到 loader 并正确路由。
5. ours:`runtime_config.data_loader()` 读到值。
6. `pytest test_gate_transport_parity test_harness_core test_ours_env_inputs_transport test_data_config_schema test_hf_stream_dataloader`。
7. 经真实 per-loop worktree 验证(memory: test-via-loop-worktree),非裸 checkout。

## 一图流(目标态)
```
config/data.toml ──(FORGE_DATA_TOML 指针)──▶ ref/sisters train*.py (--data-config, tomllib)
      │  [data].data_loader ─────────────────▶ ours runtime_config.data_loader()
      │                       ─────────────────▶ meta ref_bundle_run.sh (--data-config)
      │  [data].conf_path ──▶ conf.sh ──▶ DATA_PATH_FILE ──(--data-path-file)──▶ train*.py
      ▼
render_gate_configs.py ──▶ <gate>.toml：只含 model/optim/ref 投影,零 data 值

ref.toml [ref].forge_tokenizer_dir ──(FORGE_TOKENIZER_DIR 目录指针)──▶ hf_stream_dataloader
```
```
```
