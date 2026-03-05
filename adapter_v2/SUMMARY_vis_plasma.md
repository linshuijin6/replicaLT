# Plasma 对齐可视化 - 实施总结

## 📋 已完成的工作

### 1. 核心脚本

**文件**: `adapter_v2/vis_plasma_alignment.py` (704 行)

**功能**:
- ✅ 加载训练好的 checkpoint（包含 context net 和 projection）
- ✅ Forward 提取 plasma_emb（512 维，经过模型完整处理）
- ✅ 对 plasma_emb 进行 UMAP 降维（2D）
- ✅ 生成 1×3 并排对比图：
  - 子图1: 真实 pT217_F 着色
  - 子图2: within-class shuffle 后的 pT217_F 着色（反事实对照）
  - 子图3: 从 plasma_emb 线性回归预测的 pT217_F + Spearman ρ
- ✅ 为 CN/MCI/AD 三个类别分别生成 train 和 val 图（共 6 组）
- ✅ 保存元信息 JSON（样本数、ρ、UMAP 参数、shuffle 成功标志）

**关键特性**:
- 复用 `train.py` 的 `fixed_split.json` 格式（确保划分一致）
- 复用 `TAUPlasmaDataset` 和 `CoCoOpTAUModel`（无需重复实现）
- 支持大小写兼容的 plasma key 解析（通过 config 的 alias 机制）
- 自动处理缺失样本和小样本类别
- 固定 UMAP 随机种子确保可复现

### 2. 完整文档

**文件**: `adapter_v2/README_vis_plasma.md` (333 行)

**内容**:
- ✅ 脚本概念与设计思路
- ✅ 依赖安装指南（umap-learn, scipy, scikit-learn, matplotlib）
- ✅ 完整参数说明（15 个参数）
- ✅ 输出文件格式详解（1x3.png + meta.json）
- ✅ 典型工作流（训练 → 可视化 → 分析）
- ✅ 常见问题（FAQ）与解决方案
- ✅ 技术细节（UMAP 参数、within-class shuffle、线性回归）

### 3. 快速启动脚本

**文件**: `adapter_v2/run_vis_plasma.sh` (可执行)

**功能**:
- ✅ 自动检查依赖（Python、UMAP）
- ✅ 验证 checkpoint 是否存在
- ✅ 一键运行可视化
- ✅ 自动总结输出（列出所有生成的文件和 Spearman ρ）

---

## 🚀 快速开始

### 步骤 1: 安装依赖

```bash
pip install umap-learn scipy scikit-learn matplotlib
```

### 步骤 2: 确保已有训练好的模型

运行 `train.py` 生成 checkpoint 和 `fixed_split.json`:

```bash
cd /home/ssddata/linshuijin/replicaLT/adapter_v2

python train.py \
  --config config.yaml \
  --val_split_json fixed_split.json \
  --epochs 200
```

输出:
- `fixed_split.json` - 固定的 train/val 划分
- `runs/{timestamp}_{pid}/ckpt/best.pt` - 最佳模型

### 步骤 3: 生成可视化

**方法 A: 使用快速启动脚本**

```bash
# 1. 编辑脚本，修改 CKPT_PATH
vim run_vis_plasma.sh
# 将 CKPT_PATH="runs/YOUR_RUN_NAME/ckpt/best.pt" 
# 改为实际路径，如: CKPT_PATH="runs/0302_12345/ckpt/best.pt"

# 2. 运行
bash run_vis_plasma.sh
```

**方法 B: 直接运行 Python 脚本**

```bash
python vis_plasma_alignment.py \
  --config config.yaml \
  --ckpt runs/0302_12345/ckpt/best.pt \
  --val_split_json fixed_split.json \
  --plasma_key pT217_F \
  --output_dir vis_plasma_output \
  --seed 42
```

### 步骤 4: 查看结果

```bash
# 查看目录结构
tree vis_plasma_output

# 查看所有 Spearman ρ
find vis_plasma_output -name "meta.json" -exec jq -r '"\(.class_name) \(.split): ρ=\(.spearman_rho)"' {} \;

# 查看图片（Linux）
eog vis_plasma_output/*/train/1x3.png
eog vis_plasma_output/*/val/1x3.png
```

**输出示例**:
```
vis_plasma_output/
├── CN/
│   ├── train/
│   │   ├── 1x3.png
│   │   └── meta.json
│   └── val/
│       ├── 1x3.png
│       └── meta.json
├── MCI/
│   ├── train/
│   └── val/
└── AD/
    ├── train/
    └── val/
```

---

## 📊 结果解读

### Spearman ρ 解释

| ρ 范围 | 含义 |
|--------|------|
| **ρ > 0.5** | plasma_emb **强编码** pT217_F（理想情况） |
| **0.3 < ρ < 0.5** | **中等编码**（可接受，但可调优） |
| **ρ < 0.3** | **编码弱或失败**（需检查训练） |

### 三子图对比原则

✅ **成功编码的特征**:
1. 子图1（真实值）：颜色呈现**有规律的渐变**（如从蓝到黄）
2. 子图2（shuffle）：颜色呈现**随机分布**，无明显模式
3. 子图3（预测值）：与子图1 **模式相似**，且 ρ > 0.5

❌ **编码失败的特征**:
- 三个子图看起来几乎相同（都是随机噪声）
- ρ 接近 0 或为负
- 子图1 和子图3 完全不同

### 案例分析

**案例 1: 理想情况（ρ=0.72）**
```
Real pT217_F:      Shuffled pT217_F:  Predicted pT217_F (ρ=0.72):
  蓝→绿→黄渐变        随机色块分布          蓝→绿→黄渐变
```
✅ **结论**: plasma_emb 成功编码了 pT217_F，与影像特征对齐良好

**案例 2: 失败情况（ρ=0.12）**
```
Real pT217_F:      Shuffled pT217_F:  Predicted pT217_F (ρ=0.12):
  杂乱色块            杂乱色块             杂乱色块
```
❌ **结论**: plasma_emb 未学到 pT217_F 信息，需调整训练策略

---

## 🔧 故障排除

### 问题 1: 报错 "UMAP not found"

```bash
pip install umap-learn
```

### 问题 2: 报错 "plasma_key 'xxx' 不在 selected_keys 中"

**原因**: `config.yaml` 中未启用该 plasma key

**解决**: 编辑 `config.yaml`:
```yaml
plasma:
  selected_keys: ["pT217_F"]  # 确保包含目标 key
```

### 问题 3: 某个类别没有输出

**原因**: 该类别样本不足（< 5 个有效 plasma 样本）

**检查**: 查看控制台警告信息:
```
[WARN] No samples for CN in train split, skipping.
```

### 问题 4: ρ 很低（接近 0）

**可能原因**:
1. Plasma 分支未被利用（`lambda_img_plasma` 太小）
2. 训练未收敛
3. 数据噪声

**排查步骤**:
```bash
# 1. 查看训练日志中的 plasma 相关损失
grep "L_ip" runs/*/tensorboard_log.txt

# 2. 查看 validation 日志中的 plasma_margin_drop
grep "plasma_margin_drop" runs/*/validation_log.txt

# 3. 对比 train 和 val 的 ρ
find vis_plasma_output -name "meta.json" -exec jq '.split, .spearman_rho' {} \;
```

**解决方案**:
- 增大 `config.yaml` 中的 `lambda_img_plasma`（如从 1.0 → 2.0）
- 延长训练 epochs
- 检查数据质量（pT217_F 是否有缺失或异常值）

---

## 📁 文件清单

```
adapter_v2/
├── vis_plasma_alignment.py      # 主脚本（704 行）
├── README_vis_plasma.md         # 完整文档（333 行）
├── run_vis_plasma.sh            # 快速启动脚本（可执行）
└── SUMMARY_vis_plasma.md        # 本总结文件
```

---

## 🎯 核心决策回顾

根据用户需求和交互问答，本实施遵循以下决策：

1. **Embedding 来源**: plasma_emb（模型 forward 输出的 512 维）
   - ✅ 包含 context net 调制
   - ✅ 包含 projection head 处理
   - ✅ 反映模型真实学习的对齐空间

2. **降维方法**: 仅支持 UMAP
   - ✅ 明确报错提示安装依赖
   - ✅ 固定参数确保可复现
   - ❌ 不支持 t-SNE（避免混淆）

3. **类别覆盖**: 自动为 CN/MCI/AD 全部生成
   - ✅ 每个类别输出 train + val（共 6 组）
   - ✅ 自动跳过样本不足的类别
   - ❌ 不设置默认类别（避免误读）

4. **颜色编码**: pT217_F 原始值 / shuffle 后 / 线性预测值
   - ✅ 子图1: 真实值
   - ✅ 子图2: within-class shuffle（反事实）
   - ✅ 子图3: 预测值 + Spearman ρ

5. **不触碰训练逻辑**:
   - ✅ 完全独立的事后分析脚本
   - ✅ 不修改 `train.py`、`models.py`、`dataset.py`
   - ✅ 不使用 `*_v2.py` 结尾的文件

---

## 📖 相关文档

- **脚本详细文档**: `README_vis_plasma.md`
- **训练脚本**: `train.py`
- **模型定义**: `models.py`（不使用 `models_v2.py`）
- **数据集**: `dataset.py`（不使用 `dataset_v2.py`）

---

## ✅ 验证清单

- [x] 脚本语法正确（通过 `python -m py_compile`）
- [x] 所有依赖明确标注（UMAP/scipy/sklearn/matplotlib）
- [x] 参数默认值合理（seed=42, umap_n_neighbors=15 等）
- [x] 错误处理完善（缺失 checkpoint、plasma_key 不存在等）
- [x] 输出格式标准（JSON + PNG）
- [x] 文档完整（README + SUMMARY + 快速启动脚本）
- [x] 可复现性保证（固定随机种子、记录所有参数）
- [x] 与现有工具链兼容（复用 `fixed_split.json`、`config.yaml`）

---

## 🔮 后续扩展建议

1. **支持其他 plasma keys**:
   ```bash
   python vis_plasma_alignment.py --plasma_key NfL_Q
   ```

2. **交互式可视化**:
   - 使用 Plotly 生成可交互的 HTML
   - 鼠标悬停显示 subject_id

3. **批量比较**:
   - 对比不同 checkpoint 的 ρ 变化
   - 绘制 ρ-epoch 曲线

4. **统计检验**:
   - 对 ρ 进行显著性检验（bootstrap p-value）
   - 对比 train vs val 的 ρ 差异是否显著

---

## 📞 联系与反馈

如有问题或建议，请：
1. 查看 `README_vis_plasma.md` 常见问题部分
2. 检查训练日志和可视化输出的 meta.json
3. 联系项目维护者

---

**实施日期**: 2026-03-02  
**版本**: v1.0  
**状态**: ✅ 已完成并测试语法

---

**Happy Visualizing! 🎨📊**
