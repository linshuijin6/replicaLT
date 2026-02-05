# MRI → TAU-PET Generation Baseline

## 概述

这是一个 MRI(T1) → TAU-PET(AV1451) 生成的 baseline 实现。

### 特性

- **模型**: 3D Residual U-Net（4 层级，残差输出模式）
- **损失**: L1 + 3D SSIM（加权组合）
- **训练**: AdamW 优化器，cosine 学习率调度，混合精度训练
- **评估**: 分层指标（按 PET 厂商 / 质量等级 / 诊断）

## 文件结构

```
baseline/
├── __init__.py       # 模块入口
├── config.py         # 配置管理
├── dataset.py        # 数据集和数据加载器
├── model.py          # 3D Residual U-Net 模型
├── losses.py         # L1 + 3D SSIM 损失函数
├── train.py          # 训练脚本
├── evaluate.py       # 评估脚本
├── visualize.py      # 可视化模块
└── README.md         # 本文件
```

## 快速开始

### 1. 训练

```bash
cd /home/ssddata/linshuijin/replicaLT

# 使用默认配置训练
python -m baseline.train

# 自定义参数
python -m baseline.train --epochs 100 --batch_size 2 --lr 1e-4
```

### 2. 评估

```bash
# 评估最佳模型
python -m baseline.evaluate runs/<run_dir>/checkpoints/best_model.pth

# 保存预测结果
python -m baseline.evaluate runs/<run_dir>/checkpoints/best_model.pth --save_predictions
```

### 3. 可视化

```bash
# 可视化前 10 个样本
python -m baseline.visualize runs/<run_dir>/checkpoints/best_model.pth --n_samples 10

# 分层可视化（每层 2 个样本）
python -m baseline.visualize runs/<run_dir>/checkpoints/best_model.pth --stratified

# 保存 NIfTI 文件
python -m baseline.visualize runs/<run_dir>/checkpoints/best_model.pth --save_nifti
```

## 配置说明

### 数据配置 (DataConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `tau_dir` | `/mnt/.../TAU` | TAU-PET 数据目录 |
| `mri_dir` | `/mnt/.../MRI` | MRI 数据目录 |
| `pairs_csv` | `...pairs_180d_dx_plasma_90d_matched_with_demog.csv` | 配对 CSV |
| `qc_csv` | `...tau_qc_results.csv` | QC 结果 CSV |
| `pet_info_csv` | `...3_PET_ADNI3_4_with_Plasma_PET_Images_04Feb2026.csv` | PET 信息 CSV |
| `input_shape` | `(182, 218, 182)` | 原始体积尺寸 |
| `crop_shape` | `(160, 192, 160)` | 中心裁剪后尺寸 |
| `train_ratio` | `0.8` | 训练集比例 |
| `val_ratio` | `0.1` | 验证集比例 |
| `test_ratio` | `0.1` | 测试集比例 |

### 模型配置 (ModelConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `in_channels` | `1` | 输入通道数 |
| `out_channels` | `1` | 输出通道数 |
| `base_features` | `32` | 基础特征数 |
| `n_levels` | `4` | U-Net 层级数 |
| `residual_output` | `True` | 残差输出模式 |

### 损失配置 (LossConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `l1_weight` | `1.0` | L1 损失权重 |
| `ssim_weight` | `0.1` | SSIM 损失权重 |
| `ssim_window_size` | `7` | SSIM 窗口大小 |

### 训练配置 (TrainConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | `1` | 批大小 |
| `epochs` | `200` | 训练轮数 |
| `learning_rate` | `1e-4` | 学习率 |
| `weight_decay` | `1e-4` | 权重衰减 |
| `accumulation_steps` | `4` | 梯度累积步数 |
| `use_amp` | `True` | 混合精度训练 |
| `warmup_epochs` | `5` | 预热轮数 |
| `early_stopping_patience` | `20` | 早停耐心值 |

## 输出目录结构

```
runs/<YYYYMMDD_HHMMSS>_<PID>/
├── config.json           # 训练配置
├── train.log             # 训练日志
├── events.out.tfevents.* # TensorBoard 事件
├── checkpoints/
│   ├── best_model.pth    # 最佳模型
│   ├── last_model.pth    # 最后模型
│   └── epoch_*.pth       # 定期保存的模型
├── visualizations/       # 可视化图像
│   ├── *.png             # 对比图
│   └── nifti/            # NIfTI 文件
├── predictions/          # 预测结果
├── metrics_test.csv      # 测试集详细指标
├── metrics_stratified.json # 分层统计指标
└── evaluation_report.txt  # 评估报告
```

## 评估指标

- **MAE** (Mean Absolute Error): 平均绝对误差
- **PSNR** (Peak Signal-to-Noise Ratio): 峰值信噪比
- **SSIM** (Structural Similarity Index): 结构相似性

### 分层分析

- 按 **PET 厂商** (Siemens / GE Medical Systems / Philips Medical Systems)
- 按 **质量等级** (Clean / Medium / Noisy)
- 按 **诊断** (CN / MCI / Dementia)

## 数据权重

根据 QC 结果为每个样本分配训练权重：

| 质量等级 | 权重 |
|----------|------|
| Clean    | 1.0  |
| Medium   | 0.7  |
| Noisy    | 0.3-0.4 |

## 依赖

- Python >= 3.8
- PyTorch >= 1.10
- nibabel
- numpy
- pandas
- matplotlib
- tensorboard
- tqdm

## 作者

MRI-to-TAU-PET Baseline v1.0.0
