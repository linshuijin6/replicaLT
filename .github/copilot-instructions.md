# Copilot 项目指令：replicaLT（MRI -> PET + Plasma）

## 0. 作用范围（高优先级）
本仓库提问时，默认只把下列代码视为项目核心：

1. `adapter_v2/train.py`：预训练图文对齐（CoCoOpTAUModel）
2. `plasma_train.py`：MRI 生成 PET（FDG/AV45/TAU 三种示踪剂分别训练，不建模示踪剂间关系）

回答实现细节时，只追踪这两条主链及其 **实际 import 到的模块**，不要扩展到未被 import 的实验脚本。

---

## 1. 项目一句话总结
这是一个“先对齐、后生成”的两阶段体系：

- 阶段 A（`adapter_v2/train.py`）：用 TAU 图像 token、MRI token、诊断文本和 plasma 数值做多路对齐训练，得到可复用的 plasma 语义嵌入能力。
- 阶段 B（`plasma_train.py`）：把阶段 A 预计算的 `plasma_emb` 作为条件 token，联合模态文本 token，驱动 3D 扩散/rectified-flow 风格的 MRI->PET 生成。

---

## 2. 阶段 A：对齐训练（adapter_v2/train.py）

### 2.1 数据与缓存
关键依赖：
- `adapter_v2/dataset.py`
- `adapter_v2/precompute_cache.py`
- `adapter_v2/config.yaml`

训练前逻辑：
- 自动检查/补齐 TAU 与 MRI 的 vision cache（`.vision.pt` / `.mri_vision.pt`）。
- 样本由 `TAUPlasmaDataset` 提供。
- 采样使用 `SubjectBatchSampler`，按 subject 组织 batch，降低伪负样本风险。

### 2.2 模型
关键依赖：
- `adapter_v2/models.py`

核心类：`CoCoOpTAUModel`
- 图像分支：`tau_tokens` 与 `mri_tokens` 经注意力池化得到表征。
- Context 分支：`mri` 表征进入 `ContextNet` 生成动态 context token。
- 文本分支：
  - class prompt + context -> class embedding
  - plasma prompt + context -> plasma embedding（按 plasma 权重汇聚）
- 三路 embedding 最终映射到 512 维 CLIP 空间。

### 2.3 损失与验证
关键依赖：
- `adapter_v2/losses.py`

训练损失由 `compute_total_loss` 组成：
- `L_img_class`
- `L_img_plasma`
- `L_class_plasma`
- `L_reg`

验证指标以分类可分性与对齐诊断为主：
- 线性探针（Balanced Acc / Macro-F1）
- context 注入分类（Balanced Acc / Macro-F1）
- plasma shuffle 反事实分析（within/cross score drop 与 margin drop）

### 2.4 产物
- 运行目录：`adapter_v2/runs/<time_pid>/`
- 关键 checkpoint：`ckpt/best.pt`
- 可复现元数据：`meta/repro_manifest.json` 等

---

## 3. 阶段 B：生成训练（plasma_train.py）

### 3.1 条件设计（关键）
`plasma_train.py` 使用固定两 token 条件：
- Token 0：预计算 `plasma_emb`（来自阶段 A）
- Token 1：BiomedCLIP 编码的“示踪剂模态优化文本”

条件张量形状保持 `(B, 2, 512)`，因此无需改动扩散模型主架构。

### 3.2 三示踪剂策略
- FDG / AV45 / TAU 被视作三种独立目标模态。
- 每步从当前样本可用模态中随机选一个训练（不是联合建模示踪剂关系）。

### 3.3 训练范式（rectified-flow 风格）
训练目标是速度场回归：
- 构造：`x_t = t * PET + (1 - t) * MRI`
- 目标：`v = PET - MRI`
- 损失：`MSE(v_pred, v)`

推理/验证阶段以迭代积分形式从 MRI 逐步更新到 PET 近似。

### 3.4 关键依赖
- `DistributedDiffusionModelUNet`（多 GPU 模型并行）
- `BiomedCLIP`（仅用于模态文本特征编码）
- `PersistentDataset`（MONAI 缓存）
- 预计算 `plasma_emb` 缓存目录

---

## 4. 回答问题时的检索优先级
当用户在本仓库提问，请优先按以下顺序定位：

1. 训练对齐问题 -> `adapter_v2/train.py`
2. 对齐数据处理 -> `adapter_v2/dataset.py`
3. 对齐模型结构 -> `adapter_v2/models.py`
4. 对齐损失定义 -> `adapter_v2/losses.py`
5. 缓存生成逻辑 -> `adapter_v2/precompute_cache.py`
6. 生成训练问题 -> `plasma_train.py`

若问题超出以上文件，先明确“该结论不在核心链路中”，再决定是否扩展。

---

## 5. 回答风格约束（本仓库）
- 默认中文。
- 优先给出“基于核心代码的确定性结论”，避免泛化到未 import 的脚本。
- 解释模型时区分两阶段，不要把 `adapter_v2/train.py` 与 `plasma_train.py` 混成单阶段端到端训练。
- 讨论三示踪剂时，明确“独立目标，不建模示踪剂间关系”。
- 讨论 plasma 时，明确“在生成阶段作为条件 token 引导，而非额外监督标签”。

---

## 6. 代码变更与文档同步（强制）

当核心链路代码发生重要修改时，必须同步更新本文件（`.github/copilot-instructions.md`）。

### 6.1 需要同步更新文档的“重要修改”
满足以下任一项即视为重要修改：
- 训练目标、损失项或损失权重语义变化（新增/删除/重命名 loss，训练范式改变）。
- 条件注入方式变化（context token 数量、含义、shape，plasma 注入位置变化）。
- 数据流变化（输入字段、缓存格式、样本筛选、split 逻辑、关键路径变化）。
- 模型结构变化（核心模块替换、关键分支新增/删除、关键超参数默认值变化）。
- 训练/验证流程变化（指标口径、推理/验证流程、checkpoint 关键产物路径变化）。
- 核心文件范围变化（新增进入主链的 import 依赖，或主链文件迁移）。

### 6.2 同步更新要求
- 代码 PR/提交中必须包含对应 md 更新；禁止“代码已变更但文档仍描述旧逻辑”。
- 若存在冲突（代码与文档不一致），以最新核心代码为准，并在同一修改中修正文档。
- 文档更新应最小化且可追踪：只改受影响条目，不重写无关部分。

### 6.3 提交前检查清单（Copilot 默认执行）
- 检查 `adapter_v2/train.py`、`plasma_train.py` 及其核心 import 依赖是否有重要变更。
- 若有，逐项核对本文件以下章节是否仍准确：
  - 第 1 节（项目一句话总结）
  - 第 2 节（对齐训练）
  - 第 3 节（生成训练）
  - 第 4 节（检索优先级）
  - 第 6 节（本节本身）
- 若未更新文档，视为任务未完成，需先补齐 md 再结束。

---

## 8. Sync Logger（强制完成门槛）

### 8.1 每轮必做
- 只要本轮有回复输出，就必须在结束前**追加**一条记录到 `sync/chat-log.md`。
- 记录内容只保留：用户问题摘要、最终结论摘要、关键修改结果；不记录推理过程与工具细节。

### 8.2 有文件修改时必做
- 只要本轮发生创建 / 编辑 / 删除文件，就必须在结束前**追加**一条记录到 `sync/changes-log.md`。
- `sync/chat-log.md` 与 `sync/changes-log.md` 自身更新，不再反向记录进 changes-log。

### 8.3 长任务防跳过
- 多文件、长命令、长链路任务中，**不要把日志步骤拖到最后碰运气**。
- 在主要代码修改完成后、最终总结前，就应先完成本地日志落盘。
- 若本轮应写的日志尚未完成，则**不得输出最终答复，不得宣称任务完成**。

### 8.4 Notion 同步
- 若本轮有代码修改，则在本地日志完成后，检查 Notion 工具是否可用；可用则必须尝试同步，不可用则静默跳过。
- Notion 失败不影响主任务，但不允许因此跳过本地日志；若失败，只在答复末尾简短提示。

### 8.5 续聊恢复
- 当用户要求“继续之前的话题”等续聊行为时，先读取 `sync/chat-log.md` 最近 5–10 条记录，再继续回答。

### 8.6 规范来源
- 用户级规则主副本：`/home/data/linshuijin/.config/sync-logger.instructions.md`
- 本仓库内自动读取依赖本文件与仓库根 `AGENTS.md`；修改规则时优先保持三者一致。

---

## 9. 常见提问快速模板（供 Copilot 复用）
- “对齐阶段如何计算总损失与各项权重？” -> 看 `adapter_v2/losses.py` + `adapter_v2/train.py`
- “plasma 在生成阶段具体注入到哪里？” -> 看 `plasma_train.py` 中 index_transform 与 context 构造
- “为什么说是 rectified-flow 风格？” -> 看 `plasma_train.py` 中 `x_t` 与 `v = PET - MRI` 的训练目标定义
- “三种 PET 如何训练？” -> 看 `plasma_train.py` 中可用模态随机选择逻辑
