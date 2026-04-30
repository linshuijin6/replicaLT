# Antigravity 专属规则：replicaLT（MRI -> PET + Plasma）

## 0. 环境与执行规则（最高优先级 / Global Settings）
在此项目中，**所有程序的运行必须默认在 `xiaochou` 环境中执行**：
- **自动切换**：在执行任何 Python 代码或 Shell 脚本时，请通过执行类似 `conda run -n xiaochou [command]` 或 `source ~/.bashrc && conda activate xiaochou` 的方式来执行代码。
- **禁止询问**：为了避免每次修改代码后耗时调整环境，请不要反复向用户询问。默认就是 `xiaochou`，并保证命令直接用该环境运行！

---

## 1. 作用范围（高优先级）
本仓库提问时，默认只把下列代码视为项目核心：

1. `adapter_v2/train.py`：预训练图文对齐（CoCoOpTAUModel）
2. `plasma_train.py`：MRI 生成 PET（FDG/AV45/TAU 三种示踪剂分别训练，不建模示踪剂间关系）
   - `plasma_train_codex.py` 是其三 token 条件变体入口，保持模型架构不变。

回答实现细节时，只追踪这两条主链及其 **实际 import 到的模块**，不要扩展到未被 import 的实验脚本。

---

## 2. 项目一句话总结
这是一个“先对齐、后生成”的两阶段体系：

- 阶段 A（`adapter_v2/train.py`）：用 TAU 图像 token、MRI token、诊断文本和 plasma 数值做多路对齐训练，得到可复用的 plasma 语义嵌入能力。
- 阶段 B（`plasma_train.py` / `plasma_train_codex.py`）：把阶段 A 预计算的 `plasma_emb` 作为条件 token，联合模态文本 token，驱动 3D 扩散/rectified-flow 风格的 MRI->PET 生成；三 token 变体会额外注入临床量表文本 token。

---

## 3. 阶段 A：对齐训练（adapter_v2/train.py）

### 3.1 数据与缓存
关键依赖：
- `adapter_v2/dataset.py`
- `adapter_v2/precompute_cache.py`
- `adapter_v2/config.yaml`

训练前逻辑：
- 自动检查/补齐 TAU 与 MRI 的 vision cache（`.vision.pt` / `.mri_vision.pt`）。
- 样本由 `TAUPlasmaDataset` 提供。
- 采样使用 `SubjectBatchSampler`，按 subject 组织 batch，降低伪负样本风险。

### 3.2 模型
关键依赖：
- `adapter_v2/models.py`

核心类：`CoCoOpTAUModel`
- 图像分支：`tau_tokens` 与 `mri_tokens` 经注意力池化得到表征。
- Context 分支：`mri` 表征进入 `ContextNet` 生成动态 context token。
- 文本分支：
  - class prompt + context -> class embedding
  - plasma prompt + context -> plasma embedding（按 plasma 权重汇聚）
- 三路 embedding 最终映射到 512 维 CLIP 空间。

### 3.3 损失与验证
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

### 3.4 产物
- 运行目录：`adapter_v2/runs/<time_pid>/`
- 关键 checkpoint：`ckpt/best.pt`
- 可复现元数据：`meta/repro_manifest.json` 等

---

## 4. 阶段 B：生成训练（plasma_train.py）

### 4.1 条件设计（关键）
`plasma_train.py` 使用固定两 token 条件：
- Token 0：预计算 `plasma_emb`（来自阶段 A）
- Token 1：BiomedCLIP 编码的“示踪剂模态优化文本”

条件张量形状保持 `(B, 2, 512)`，因此无需改动扩散模型主架构。

`plasma_train_codex.py` 使用三 token 条件变体：
- Token 0：预计算 `plasma_emb`（来自阶段 A）
- Token 1：BiomedCLIP 编码的 `old_descr` 临床量表文本（缺失时回退 `description`）
- Token 2：BiomedCLIP 编码的“示踪剂模态优化文本”

条件张量形状为 `(B, 3, 512)`，仍保持 `cross_attention_dim=512` 和原扩散模型架构不变。

### 4.2 三示踪剂策略
- FDG / AV45 / TAU 被视作三种独立目标模态。
- 每步从当前样本可用模态中随机选一个训练（不是联合建模示踪剂关系）。

### 4.3 训练范式（rectified-flow 风格）
训练目标是速度场回归：
- 构造：`x_t = t * PET + (1 - t) * MRI`
- 目标：`v = PET - MRI`
- 损失：`MSE(v_pred, v)`

推理/验证阶段以迭代积分形式从 MRI 逐步更新到 PET 近似。

### 4.4 关键依赖
- `DistributedDiffusionModelUNet`（多 GPU 模型并行）
- `BiomedCLIP`（用于模态文本特征编码；三 token 变体还用于临床量表文本编码）
- `PersistentDataset`（MONAI 缓存）
- 预计算 `plasma_emb` 缓存目录

---

## 5. 回答问题时的检索优先级
当用户在本仓库提问，请优先按以下顺序定位：

1. 训练对齐问题 -> `adapter_v2/train.py`
2. 对齐数据处理 -> `adapter_v2/dataset.py`
3. 对齐模型结构 -> `adapter_v2/models.py`
4. 对齐损失定义 -> `adapter_v2/losses.py`
5. 缓存生成逻辑 -> `adapter_v2/precompute_cache.py`
6. 生成训练问题 -> `plasma_train.py`；若问题涉及三 token 临床文本条件 -> `plasma_train_codex.py`

若问题超出以上文件，先明确“该结论不在核心链路中”，再决定是否扩展。

---

## 6. 回答风格约束（本仓库）
- 默认中文。
- **强制约束**：遇到执行命令需求，无须问询，直接选用 `xiaochou` 环境执行。
- 优先给出“基于核心代码的确定性结论”，避免泛化到未 import 的脚本。
- 解释模型时区分两阶段，不要把 `adapter_v2/train.py` 与 `plasma_train.py` 混成单阶段端到端训练。
- 讨论三示踪剂时，明确“独立目标，不建模示踪剂间关系”。
- 讨论 plasma 时，明确“在生成阶段作为条件 token 引导，而非额外监督标签”。

---

## 7. 代码变更与文档同步（强制）

当核心链路代码发生重要修改时，必须同步更新 `.github/copilot-instructions.md` 与本文件（`.agent/rules.md`）。

### 7.1 需要同步更新文档的“重要修改”
满足以下任一项即视为重要修改：
- 训练目标、损失项或损失权重语义变化（新增/删除/重命名 loss，训练范式改变）。
- 条件注入方式变化（context token 数量、含义、shape，plasma 注入位置变化）。
- 数据流变化（输入字段、缓存格式、样本筛选、split 逻辑、关键路径变化）。
- 模型结构变化（核心模块替换、关键分支新增/删除、关键超参数默认值变化）。
- 训练/验证流程变化（指标口径、推理/验证流程、checkpoint 关键产物路径变化）。
- 核心文件范围变化（新增进入主链的 import 依赖，或主链文件迁移）。

### 7.2 同步更新要求
- 代码 PR/提交中必须包含对应 md 更新；禁止“代码已变更但文档仍描述旧逻辑”。
- 若存在冲突（代码与文档不一致），以最新核心代码为准，并在同一修改中修正文档。
- 文档更新应最小化且可追踪：只改受影响条目，不重写无关部分。

### 7.3 提交前检查清单（Antigravity 默认执行）
- 检查 `adapter_v2/train.py`、`plasma_train.py`、`plasma_train_codex.py` 及其核心 import 依赖是否有重要变更。
- 若有，逐项核对本文件及 `.github/copilot-instructions.md` 中的对应章节是否仍准确。
- 若未更新文档，视为任务未完成，需先补齐 md 再结束。

---

## 8. 常见提问快速模板
- “对齐阶段如何计算总损失与各项权重？” -> 看 `adapter_v2/losses.py` + `adapter_v2/train.py`
- “plasma 在生成阶段具体注入到哪里？” -> 看 `plasma_train.py` 中 index_transform 与 context 构造；三 token 变体看 `plasma_train_codex.py`
- “为什么说是 rectified-flow 风格？” -> 看 `plasma_train.py` 中 `x_t` 与 `v = PET - MRI` 的训练目标定义
- “三种 PET 如何训练？” -> 看 `plasma_train.py` 中可用模态随机选择逻辑
