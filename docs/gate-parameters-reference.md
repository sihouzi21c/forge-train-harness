# 门禁参数手册（MiniCPM4 0.5B / dense_training）

> 本文件是**参数落位的唯一权威**（每个参数在哪个文件、作用、取值、文件化后归属）。设计动机与
> 三时刻机制见 `gate-config-fileification-design.md`；逐文件改动清单与迁移顺序见
> `gate-config-fileification-plan.md`。

覆盖一次 `harness run <gate>` 从注册表到产物到运行时用到的**所有**参数，逐个讲作用和取值。
按"来源层"组织：注册表全局 → 每门禁路由 → gate_config 单一源 → 轴基线（model/optim/ref/data）。

> 四层关系：注册表 `dense_training.toml`（身份+路由+全局）→ `gate_config/<gate>.toml`（单一源，
> per-gate 可调）→ 轴基线（`model`/`optim`/`ref`/`data`，提供 override 的底座）→
> `render_gate_configs.py` 渲染成两份产物 `ref/config/<gate>.toml` + `workload/src/config/<gate>.toml`。
> `<gate>` 值：三套件（0.5b / 1b / 8b）门禁名相同。

---

## 1. 注册表全局段（`config/eval/dense_training/dense_training.toml`，渲染器直接读）

### `[family]` — new-looptask 选单耦合
| 参数 | 作用 | 取值 |
|---|---|---|
| `label` | 人读套件名 | 字符串，如 `"MiniCPM4 0.5B"` |
| `model` | 关联 model 轴模板名 | `minicpm4_0.5b` / `minicpm4_8b` / `minicpm5_1b` / `qwen3_0.6b` |
| `ref` | 关联 ref 轴模板名 | `torch_minicpm4_0.5b` 等 |
| `optim` | 关联 optim 轴模板名 | `default` 等 |

### `[suite]`
| 参数 | 作用 | 取值 |
|---|---|---|
| `name` | 套件标识 | `dense_training` |
| `gate_config_dir` | 单一源子目录名 | `gate_config` |

### `[defaults]`
| 参数 | 作用 | 取值 |
|---|---|---|
| `seed` | 全局随机种子默认（渲染基线最低层；gate 可 `seed_override`） | 整数，默认 `1234` |

### `[runtime.distributed]` — torch rendezvous
| 参数 | 作用 | 取值 |
|---|---|---|
| `master_addr` | 进程组主机 | `localhost`（本地）/ lease 派生 host（远端 freeze 时改写） |
| `master_port` | 进程组端口 | `"auto"`（现分配空闲端口）或固定数字串。**文件化后建议固定** |

### `[env]` — 全局 env 清单（传输分类器：出现在此的键 → 产物 `[env]` 大写下发；否则 → `[cli]`）
| 参数 | 作用 | 取值 |
|---|---|---|
| `PYTORCH_CUDA_ALLOC_CONF` | CUDA 分配器策略 | `expandable_segments:True` |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | TransformerEngine 是否允许非确定算法 | `0`（确定）/ `1`；long-horizon ours 侧 `@unset` |
| `NVTE_FUSED_ATTN` | 启用 TE 融合注意力 | `1` / `0` |
| `CUBLAS_WORKSPACE_CONFIG` | cuBLAS 确定性 workspace | `:4096:8`；long-horizon ours 侧 `@unset` |
| `CUDA_DEVICE_MAX_CONNECTIONS` | CUDA 设备连接数（确定性/顺序） | `1` |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` | NCCL 心跳超时 | 秒，`1800` |

### `[ref]` — 共享采集线缆
| 参数 | 作用 | 取值 |
|---|---|---|
| `ref_capture_script` | hash 采集 bridge 脚本 | `ref/bridges/bridge.sh` |
| `ref_capture_basename` | ref 侧张量 dump 文件名 | `ref_capture.pt` |
| `candidate_capture_basename` | ours 侧张量 dump 文件名 | `candidate_capture.pt` |

### `[workload]` — 套件身份
| 参数 | 作用 | 取值 |
|---|---|---|
| `id` / `display_name` / `description` | 标识与描述 | 字符串 |
| `requires_cuda` | 是否需 GPU | `true` / `false` |

### `[stage1].milestone_order` — 进度排序唯一源
| 参数 | 作用 | 取值 |
|---|---|---|
| `milestone_order` | 里程碑顺序（进度引擎只读这里） | `["alignment","bitwise-singlecard","bitwise-multicard","bitwise-perf","resume","long-horizon","production"]` |

### `[automation.stage2]`
| 参数 | 作用 | 取值 |
|---|---|---|
| `max_concurrent` | stage2 subagent 扇出并发上限 | 整数，`6` |
| `max_op_long_failures` | op-long 允许失败数 | 整数，`3` |

---

## 2. 每门禁路由（`[evals.<gate>]`，只装身份+分发，无可调参数）

所有 stage1 门禁共有字段：

| 参数 | 作用 | 取值 |
|---|---|---|
| `stage` | 阶段归属 | `stage1` / `stage2` / `local` |
| `runner_kind` | 分发到 `RUNNER_KINDS` 的处理器 | 见下表 |
| `milestone` | 归属里程碑（进度显示） | `alignment.forward` / `bitwise-singlecard` / `long-horizon` / `production` … |
| `script` | ours 入口脚本 | `evals/scripts/eval_*.py` |
| `launcher` | DP 启动器 | `evals/scripts/launch_dp.py` |
| `env_inputs` | **（文件化前）** dispatch 提升进子进程的 env 键；文件化后改为 freeze 填实清单 | 键名列表 |
| `timeout_s` | 该 gate wall-clock 上限（秒） | 见各 gate |
| `ref_timeout_s` | ref 侧独立 wall 上限（仅 perf-bitwise） | 秒 |
| `use_real_dataloader` | 是否用真实数据流（bitwise 系列） | `true` |
| `mirror_gate` | 镜像其它 gate 形状（仅 profile-snapshot） | `{ "bitwise-perf"="perf-bitwise", ... }` |

各门禁 `runner_kind` / `timeout_s` 一览：

| 门禁 | runner_kind | milestone | timeout_s | 备注 |
|---|---|---|---|---|
| `forward-align` | forward-align | alignment.forward | 600 | 单步 DP=1 forward 张量对齐 |
| `backward-align` | backward-align | alignment.backward | 600 | 单步 grad 对齐 |
| `multistep-1gpu` | stage1-bitwise-trajectory | bitwise-singlecard | 600 | DP=1 多步 bitwise |
| `multistep` | stage1-bitwise-trajectory | bitwise-multicard | 600 | DP=2 多步 bitwise |
| `perf-bitwise` | stage1-bitwise-trajectory | bitwise-perf | 720 (+ref 720) | bitwise + MFU≥10% |
| `resume-gate-20` | resume-gate | resume | 1800 | checkpoint save→load 确定性 |
| `resume-startup-90` | resume-startup | resume | 600 | 续训启动时间门（ours-only，inline shape） |
| `long-train` | long-train | long-horizon | 3600 | loss_rel≤2.5%（loss-only 判定；MFU 仅测量上报，吞吐由 review 侧审） |
| `long-train-smoke` | long-train | long-horizon-smoke | 600 | 同 long-train，20 步快反馈 |
| `loss-gate-200` | loss-gate | long-horizon | 2600 | loss-only，无 MFU 门 |
| `production-train` | production-train | production | 86400/段 | ours-only 生产长训，无 ref |
| `profile-snapshot` | profile-snapshot | —（诊断） | 900 | 非 commit gate，mirror 形状 |

### 注册表 inline shape（仅 `resume-startup-90` / `profile-snapshot`，当前未渲染产物）
> 文件化后（设计 §4）：`resume-startup-90` 的**执行输入**（`world_size`/`micro_batch_size`/
> `seq_length`/`grad_accum_steps`/`resume_save_step`/`resume_post_steps`）迁进新建的
> `gate_config/resume-startup-90.toml`→产物；**判定阈值** `resume_startup_budget_s` 迁进
> 注册表 `[evals.resume-startup-90]`（冻结 eval.toml）。`profile-snapshot` 仍用 `mirror_gate`。

| 参数 | 作用 | 取值（resume-startup-90 / profile-snapshot） | 文件化后归属 |
|---|---|---|---|
| `world_size` | DP 进程数 | 2 / —（mirror） | gate_config→产物 |
| `micro_batch_size` | 单卡 micro-batch | 4 / —（mirror） | gate_config→产物 |
| `seq_length` | 序列长度 | 4096 / —（mirror） | gate_config→产物 |
| `grad_accum_steps` | 梯度累积步 | 10 / —（mirror） | gate_config→产物 |
| `resume_save_step` | 存 ckpt 的步 | 90 / — | gate_config→产物 |
| `resume_post_steps` | 续训后再跑步数 | 10 / — | gate_config→产物 |
| `resume_startup_budget_s` | 续训启动耗时上限（**判定阈值**，0=关） | 18.0 / — | **eval.toml** |
| `num_steps` | 诊断步数 | — / 12 | mirror + inline |
| `warmup_steps` | 诊断 warmup | — / 5 | mirror + inline |

---

## 3. gate_config 单一源（`gate_config/<gate>.toml`，per-gate 可调，渲染进产物）

`_override` 后缀 = 覆盖对应基线裸键；`@unset` = 从全局 `[env]` 抹掉该键（这侧不 export）。

### `[shared]` — 形状 override（渲染进两侧产物 `[cli]`）
| 参数 | 作用 | 取值 |
|---|---|---|
| `world_size_override` | DP 进程数 | 1（单卡）/ 2（多卡）/ … |
| `num_steps_override` | 训练步数 | 1（align）/ 8 / 20 / 50 / 200 / 10000（production） |
| `micro_batch_size_override` | 单卡 micro-batch | 4（多数）/ 10（production；long-horizon ours 侧覆盖） |
| `global_batch_size_override` | 全局 batch | 4（单卡 align）/ 80（下采样）/ 1280（production 全批） |
| `gate_window` | 判定窗口 `[start, end)`（哪些步纳入 loss/hash 比较） | `[1,9]` / `[21,51]` / `[101,201]` / `[0,1]`（单步）/ `[10,20]` |
| `resume_save_step` | resume gate 存 ckpt 的步 | 10（resume-gate-20） |
| `seed_override` | 覆盖全局 seed | 整数（production=1234 显式留存） |
| `seq_length_override` | 覆盖 model 轴 seq_length | 4096（production） |

grad_accum 不在文件里：渲染器按 `GBS/(MBS·DP)` 派生（如 production 1280/(10·2)=64）。

### `[shared.gate]` — init + 采集（渲染进产物 `[cli]`）
> **判定阈值已迁出**：按设计 §5，下表中标 **→eval.toml** 的**判定阈值**统一放注册表
> `[evals.<gate>]`（= 冻结 `eval.toml`），dispatcher 从那里读、**不从产物读回**，`gate_config`
> 不再承载它们。留在 `[shared.gate]`→产物的只有**执行输入/采集项**（引擎/ref 真需要）。

| 参数 | 作用 | 取值 | 归属 |
|---|---|---|---|
| `gate_bitwise` | 是否 bitwise 逐位比较 | `true`（align/bitwise 系列）/ 缺省 false | **→eval.toml** |
| `gate_atol` | bitwise 绝对容差 | `0`（严格逐位）；bitwise 套件恒 0 | **→eval.toml** |
| `mfu_e2e_target` | 端到端 MFU 门（%）；`0`/缺省 = 关（long-train 系列不设 dev 侧 MFU 门，MFU 仅测量上报，吞吐达标与否由 review 侧判定） | `0`（关）/ `10.0`（perf-bitwise） | **→eval.toml** |
| `warmup_steps` | MFU 测量前 warmup 步 | `5` / `20`；仅带 MFU 门的 gate | **→eval.toml** |
| `loss_abs_threshold` | loss 绝对差阈 | `0`（bitwise 不看绝对差） | **→eval.toml** |
| `grad_norm_abs_threshold` | grad-norm 绝对差阈 | `0` | **→eval.toml** |
| `loss_rel_threshold` | loss 相对差阈（long-horizon 主判据） | `0.025`（2.5%） | **→eval.toml** |
| `max_avg_relative_loss_diff` | 窗口平均相对 loss 差上限（loss-gate 判据） | `0.025` | **→eval.toml** |
| `hash_capture_level` | bitwise 比对覆盖哪些张量（`0` 不比 / `1` loss+grad / `2` 再加每模块 `fwd.<fqn>`+`bwd.<fqn>`）；错的 level 会静默削弱比对 | `0` / `1` / `2`（align/bitwise 用 2） | **→eval.toml**（判定语义；ours 引擎恒按 level 2 全采，dispatcher 按 eval.toml 的 level 比对） |
| `forge_init_ones` | 权重初始化模式（0=正常随机/muP；1=全 1，用于长程可比） | `0`（align/bitwise/resume）/ `1`（long-horizon/production） | 产物（执行输入） |

> `gate_window` 双重用途待定（详见设计 §5）：判定划窗算判定阈值，但 ref 采集可能也需它 → 落地时定。

### `[optim]` — optim override（叠加在 optim 轴基线上）
| 参数 | 作用 | 取值 |
|---|---|---|
| `lr_warmup_iters_override` | 覆盖 warmup 迭代数 | `0`（所有 gate：门禁不做长 warmup） |

### `[ref]` / `[ours]` — 分侧 override
| 参数 | 作用 | 取值 |
|---|---|---|
| `deterministic` | 确定性开关（ref→`--no-deterministic`；ours→`DETERMINISTIC=0/1`） | `true`（align/bitwise/resume）/ `false`（long-horizon） |
| `micro_batch_size_override` | ours 侧单独覆盖 MBS | `10`（long-horizon/loss-gate ours 侧，ref 仍 4） |
| `cublas_workspace_config` | 抹掉全局确定性 env（仅 ours long-horizon） | `"@unset"` |
| `nvte_allow_nondeterministic_algo` | 同上 | `"@unset"` |

---

## 4. 轴基线参数（gate_config 未 override 时的底座值）

### 4.1 model 轴（`config/model/minicpm4_0.5b.toml [model]`）→ 渲染成 `FORGE_*` / 产物
| 参数 | 作用 | 取值（0.5B） |
|---|---|---|
| `name` | 模型标识 | `minicpm4_0.5b` |
| `num_layers` | 层数 | 24 |
| `hidden_size` | 隐藏维 | 1024 |
| `ffn_hidden_size` | FFN 维 | 4096 |
| `num_attention_heads` | 注意力头数 | 16 |
| `num_query_groups` | GQA 组数 | 2 |
| `head_dim` | 每头维（独立旋钮） | 64 |
| `seq_length` | 序列长度 | 4096 |
| `max_position_embeddings` | 最大位置 | 4096 |
| `padded_vocab_size` | 词表（pad 后） | 73448 |
| `rotary_base` | RoPE base | 10000 |
| `norm_epsilon` | LayerNorm eps | 1e-6 |
| `init_method_std` | 初始化 std | 0.1 |
| `mup_base_hidden_size` | muP 基准宽度 | 256 |
| `mup_emb_scale` | muP embedding 缩放 | 12.0 |
| `mup_depth_scale` | muP 深度缩放 | 1.4 |
| `mtp_num_layers` | MTP/Eagle 层数 | 1 |
| `mtp_loss_weight` | MTP loss 权重 | 0.3 |

### 4.2 optim 轴（`config/optim/default.toml [optim]`）
| 参数 | 作用 | 取值 |
|---|---|---|
| `lr` | 峰值学习率 | 3.0e-4 |
| `min_lr` | WSD 衰减下限 | 0.0 |
| `lr_warmup_iters` | warmup 迭代（gate 恒 override 为 0） | 2000 |
| `lr_decay_iters` | 衰减总迭代 | 250000 |
| `lr_wsd_decay_iters` | WSD anneal 迭代（0=保持峰值到 decay） | 0 |
| `weight_decay` | 权重衰减 | 0.1 |
| `adam_beta1` / `adam_beta2` | AdamW 动量 | 0.9 / 0.95 |
| `clip_grad` | 梯度裁剪 | 1.0 |

### 4.3 ref 轴（`config/ref/torch_minicpm4_0.5b.toml [ref]`）
| 参数 | 作用 | 取值 |
|---|---|---|
| `backend` | 参考后端 | `torch` / `megatron` |
| `ref_script` | ref launcher 脚本名（非硬编码，dispatcher 从此读） | `run_16gpu_1000step_pure_mup_mtp.sh`（torch） |
| `megatron` / `megatron_branch` | Megatron 源与分支 | git URL / `cpm_core_r0.15.0` |
| `tokenizer` | tokenizer 仓库 | `openbmb/MiniCPM4-0.5B` |
| `forge_tokenizer_dir` | 本地 tokenizer 目录 | `/opt/forge-data/tokenizer` |
| `checkpoint_root` | ckpt 根（缺省自动派生 `.artifacts/checkpoints/<backend>/`） | 路径或缺省 |

### 4.4 data 轴（`config/data/ultra_fineweb.toml [data]`）
| 参数 | 作用 | 取值 |
|---|---|---|
| `conf_path` | 数据 conf 脚本 | `ref/reference/ultra_fineweb_data_conf.sh` |
| `data_loader` | 数据加载器种类（原 DATA_LOADER） | `hf` |
| `dataset` | HF 数据集 | `openbmb/Ultra-FineWeb` |
| `forge_data_dir` | 本地数据目录（workspace 相对） | `.artifacts/forge-data/ultra_fineweb` |
| `baked_data_dir` | 镜像内烘焙数据目录（拷贝种子） | `/opt/forge-data/ultra_fineweb` |
| `prefetch_target_gb` | production 后台预取语料量（GB，0/缺省=关） | 10（校验）/ ~100（全量生产） |
| `prefetch_en_fraction` | 预取英文占比 | 0.9 |
| `prefetch_wait_timeout_s` | production 等预取 sentinel 上限 | 默认 6h |
| `[data.download_files]` | 本地相对路径 → HF 仓库内路径映射 | parquet 分片映射 |

---

## 5. 运行时值（不在文件里，来源见设计文档）

| 值 | 来源 | 文件化后归属 |
|---|---|---|
| `RANK` / `LOCAL_RANK` | torchrun 每进程设 | **保留**（唯一豁免） |
| `MASTER_ADDR` / `MASTER_PORT` | dispatch/launcher rendezvous | freeze 填实 |
| `CHECKPOINT_ROOT` / `MEGATRON_ROOT` / `DATA_PATH` | 部署路径（`<runtime>`→env） | freeze 填实 |
| `START_STEP` / `FORGE_RESUME_FROM` | dispatch 扫盘（`_latest_production_checkpoint`） | 引擎派生（resume-if-present） |
| `FORGE_SAVE_PATH` / `RESUME_SCRATCH_DIR` / `FORGE_CAPTURE_OUTPUT_FILE` / `FORGE_NSYS_RANK0_OUTPUT` | dispatch 按 artifact 目录给 | 引擎按 `artifact_root`+gate 派生 |

（各值的作用与文件化归属详见 `gate-config-fileification-design.md` §3、`gate-config-fileification-plan.md` §1。）
