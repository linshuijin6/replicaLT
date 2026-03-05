# Plasma 对齐可视化脚本

## 概述

`vis_plasma_alignment.py` 用于验证 plasma 分支是否在同一类别内部成功编码了 plasma 数值信息（如 pT217_F）。

### 核心思路

对训练好的模型的 **plasma_emb**（512 维）进行 UMAP 降维，生成 1×3 并排子图：

1. **子图1**：真实 pT217_F 值着色
2. **子图2**：within-class shuffle 后的 pT217_F 着色（反事实对照）
3. **子图3**：从 plasma_emb 线性回归预测的 pT217_F 着色 + Spearman ρ

**解读原则**：
- 若 plasma_emb 成功编码了 pT217_F，则：
  - 子图1（真实值）应呈现有规律的颜色渐变
  - 子图2（shuffle）应呈现随机分布（证明颜色变化不是噪声）
  - 子图3 的 Spearman ρ 应显著 > 0（证明 embedding 可线性解码 plasma 值）
- 若三子图无明显差异且 ρ ≈ 0，说明 plasma 分支未被有效利用

### 输出内容

为 **CN/MCI/AD** 三个类别分别生成 **train** 和 **val** 可视化（共 6 组）：

```
vis_plasma_output/
├── CN/
│   ├── train/
│   │   ├── 1x3.png          # 三子图对比
│   │   └── meta.json        # 元信息（样本数、ρ、UMAP 参数）
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

## 依赖安装

脚本需要以下额外依赖：

```bash
pip install umap-learn scipy scikit-learn matplotlib
```

**注意**：如果缺少 `umap-learn`，脚本会立即报错并提示安装。

---

## 快速启动

### 基础用法

```bash
cd /home/ssddata/linshuijin/replicaLT/adapter_v2

python vis_plasma_alignment.py \
  --config config.yaml \
  --ckpt runs/{your_run_name}/ckpt/best.pt \
  --output_dir vis_plasma_output
```

### 完整参数示例

```bash
python vis_plasma_alignment.py \
  --config config.yaml \
  --ckpt runs/0302_12345/ckpt/best.pt \
  --val_split_json fixed_split.json \
  --plasma_key pT217_F \
  --output_dir vis_plasma_output \
  --seed 42 \
  --batch_size 32 \
  --num_workers 4 \
  --umap_n_neighbors 15 \
  --umap_min_dist 0.1 \
  --dpi 150 \
  --device cuda:0
```

---

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config` | str | `config.yaml` | 配置文件路径 |
| `--ckpt` | str | **必需** | 训练好的 checkpoint 路径（需包含 context net 和 projection） |
| `--csv` | str | 从 config 读取 | 数据 CSV（可覆盖配置） |
| `--cache_dir` | str | 从 config 读取 | 缓存目录（可覆盖配置） |
| `--val_split_json` | str | `fixed_split.json` | 固定 train/val 划分 JSON（与 `train.py` 共享） |
| `--plasma_key` | str | `pT217_F` | 目标 plasma 字段名（必须在配置的 `selected_keys` 中） |
| `--output_dir` | str | `vis_plasma_output` | 输出根目录 |
| `--seed` | int | `42` | 随机种子（用于 UMAP 和 shuffle） |
| `--batch_size` | int | `32` | DataLoader batch size |
| `--num_workers` | int | `4` | DataLoader workers |
| `--umap_n_neighbors` | int | `15` | UMAP 参数：邻居数 |
| `--umap_min_dist` | float | `0.1` | UMAP 参数：最小距离 |
| `--dpi` | int | `150` | 图像分辨率 |
| `--device` | str | 自动检测 | 推理设备（如 `cuda:0`） |

---

## 输出文件说明

### 1. 图像文件 (`1x3.png`)

三个并排子图，共享 UMAP 坐标，仅颜色不同：

- **左图**：真实 pT217_F 值
- **中图**：within-class shuffle 后的 pT217_F（反事实对照）
- **右图**：从 plasma_emb 线性回归预测的 pT217_F + Spearman ρ

### 2. 元信息文件 (`meta.json`)

```json
{
  "class_name": "AD",
  "class_id": 2,
  "split": "val",
  "n_samples": 45,
  "n_valid_plasma": 38,
  "plasma_key": "pT217_F",
  "plasma_key_idx": 0,
  "spearman_rho": 0.6234,
  "shuffle_success": true,
  "umap_params": {
    "n_neighbors": 15,
    "min_dist": 0.1,
    "random_state": 42
  },
  "seed": 42,
  "checkpoint": "runs/0302_12345/ckpt/best.pt"
}
```

**字段说明**：
- `n_samples`：该类别在该 split 中的总样本数
- `n_valid_plasma`：plasma 值有效（非缺失）的样本数
- `spearman_rho`：Spearman 相关系数（衡量 plasma_emb 对 pT217_F 的编码强度）
- `shuffle_success`：within-class shuffle 是否成功生成 derangement（每个样本不与自身配对）

---

## 注意事项

### 1. 运行前提

- **必须先运行 `train.py`** 生成 checkpoint 和 `fixed_split.json`
- **plasma_key 必须在配置的 `selected_keys` 中**，否则会报错

### 2. plasma_key 匹配

如果配置中只启用了部分 plasma keys（如 `selected_keys: ["pT217_F"]`），请确保 `--plasma_key` 参数与之一致：

```bash
# 配置文件 config.yaml
plasma:
  selected_keys: ["pT217_F"]

# 命令行
python vis_plasma_alignment.py --plasma_key pT217_F  # ✅ 正确
python vis_plasma_alignment.py --plasma_key NfL_Q    # ❌ 报错
```

### 3. 缺失样本警告

若某个类别在某个 split 中：
- **样本数为 0**：跳过该类别-split 组合
- **有效 plasma 样本 < 5**：跳过（UMAP 降维需要足够样本）

### 4. shuffle 失败

若 within-class shuffle 失败（`shuffle_success: false`），说明：
- 样本数太少（< 2）
- 随机排列未能避免不动点（概率极低）

此时子图2 仍会绘制，但解释性降低。

---

## 典型工作流

### 步骤 1：训练模型

```bash
cd /home/ssddata/linshuijin/replicaLT/adapter_v2

python train.py \
  --config config.yaml \
  --val_split_json fixed_split.json \
  --epochs 200
```

输出：
- `fixed_split.json`（固定划分）
- `runs/{timestamp}_{pid}/ckpt/best.pt`（最佳模型）

### 步骤 2：生成可视化

```bash
python vis_plasma_alignment.py \
  --config config.yaml \
  --ckpt runs/{timestamp}_{pid}/ckpt/best.pt \
  --val_split_json fixed_split.json
```

输出：
- `vis_plasma_output/{CN,MCI,AD}/{train,val}/1x3.png`
- `vis_plasma_output/{CN,MCI,AD}/{train,val}/meta.json`

### 步骤 3：分析结果

查看各类别的 Spearman ρ 和三子图对比：

```bash
# 查看所有 meta.json
find vis_plasma_output -name "meta.json" -exec cat {} \;

# 或使用 jq 格式化
find vis_plasma_output -name "meta.json" -exec jq '.class_name, .split, .spearman_rho' {} \;
```

**判断标准**：
- **ρ > 0.5**：plasma_emb 强编码 pT217_F
- **0.3 < ρ < 0.5**：中等编码
- **ρ < 0.3**：编码弱或失败
- **子图1 vs 子图2 颜色差异明显**：证明编码是真实的，而非噪声

---

## 常见问题

### Q1: 报错 "UMAP not found"

**解决**：安装依赖
```bash
pip install umap-learn
```

### Q2: 报错 "plasma_key 'xxx' 不在 selected_keys 中"

**解决**：检查 `config.yaml` 中的 `plasma.selected_keys`，确保包含目标 key：

```yaml
plasma:
  selected_keys: ["pT217_F", "NfL_Q"]  # 添加你需要的 key
```

或在命令行使用配置中已有的 key：
```bash
python vis_plasma_alignment.py --plasma_key pT217_F
```

### Q3: 某个类别没有输出

**原因**：该类别在 train 或 val 中样本不足（< 5 个有效 plasma 样本）

**检查**：查看控制台输出的警告信息，如：
```
[WARN] No samples for CN in train split, skipping.
```

### Q4: Spearman ρ 很低（接近 0）

**可能原因**：
1. **Plasma 分支未被有效利用**：损失权重 `lambda_img_plasma` 太小
2. **训练未收敛**：检查训练日志，确认损失下降
3. **数据噪声**：pT217_F 本身与影像无强相关

**建议**：
- 查看训练日志中的 `plasma_margin_drop` 指标
- 尝试增大 `lambda_img_plasma` 并重新训练
- 对比 train 和 val 的 ρ（若 val 远低于 train，说明过拟合）

---

## 技术细节

### UMAP 降维

- **输入**：plasma_emb（512 维，L2 归一化）
- **输出**：2D 坐标（用于可视化）
- **参数**：
  - `n_neighbors=15`：全局-局部结构平衡
  - `min_dist=0.1`：点间最小距离
  - `random_state=seed`：确保可复现

### Within-class shuffle

在**同一类别内部**对 pT217_F 值进行置换，要求每个样本不与自身配对（derangement）：

```python
# 原始: [0.1, 0.5, 0.3, 0.8]
# Shuffle: [0.5, 0.3, 0.8, 0.1]  ✅ 所有位置都变了
# 失败例: [0.1, 0.3, 0.5, 0.8]  ❌ 位置0未变
```

### 线性回归

从 plasma_emb（512 维）拟合线性模型预测 pT217_F：

```
pT217_hat = w^T * plasma_emb + b
```

Spearman ρ 衡量预测值与真实值的单调相关性（对非线性关系鲁棒）。

---

## 引用与扩展

本脚本不修改训练逻辑，仅用于**事后分析**。如需：

1. **修改 plasma 权重策略**：编辑 `models.py` 中的 `compute_plasma_weights`
2. **调整损失权重**：编辑 `config.yaml` 中的 `lambda_*` 参数
3. **支持其他 plasma keys**：
   - 修改 `--plasma_key` 参数
   - 确保该 key 在 `config.yaml` 的 `selected_keys` 中

---

## 联系方式

如有问题或建议，请联系项目维护者或提交 Issue。
