# MRI → TAU PET 四方法对比实验报告

## 实验概述

| 项目 | 说明 |
|------|------|
| 对比方法 | PASTA, Legacy, Plasma (Ours), FiCD |
| 公共测试集 | 43 subjects |
| 评估指标 | SSIM, PSNR, MAE, MSE, NCC |
| 统计检验 | Wilcoxon signed-rank test |

### 方法简述

| 方法 | 模型类型 | 输出分辨率 | 条件注入 |
|------|----------|-----------|---------|
| PASTA | 2.5D DDIM-100 扩散 | 96×112×96 (1.5mm) | Slice-level conditioning |
| Legacy | Rectified-flow 1-step | 160×192×160 (1mm) | BiomedCLIP text token |
| Plasma (Ours) | Rectified-flow 1-step | 160×192×160 (1mm) | Plasma embedding + text token |
| FiCD | DDPM concat-conditioning | 160×180×160 | MRI concat + text embedding |

## 定量结果

### 整体指标 (Mean ± Std)

| 方法 | N | SSIM ↑ | PSNR ↑ | MAE ↓ | MSE ↓ | NCC ↑ |
|------|---|--------|--------|-------|-------|-------|
| PASTA | 43 | 0.7941±0.0983 | 23.35±3.64 | 0.0437±0.0263 | 0.007143±0.009966 | 0.9256±0.2071 |
| Legacy | 43 | 0.9021±0.0291 | 28.33±3.50 | 0.0203±0.0096 | 0.002009±0.001689 | 0.9830±0.0071 |
| Plasma (Ours) | 43 | 0.9016±0.0278 | 28.26±3.36 | 0.0204±0.0096 | 0.001999±0.001630 | 0.9833±0.0066 |
| FiCD | 43 | 0.0059±0.0008 | 4.30±0.16 | 0.5580±0.0188 | 0.311735±0.021044 | N/A |

### 最优方法

- **SSIM** ↑: **Legacy**
- **PSNR** ↑: **Legacy**
- **NCC** ↑: **Plasma (Ours)**
- **MAE** ↓: **Legacy**
- **MSE** ↓: **Plasma (Ours)**

## 统计检验 (Wilcoxon signed-rank test)

| 指标 | 对比 | N | Statistic | p-value | 显著性 |
|------|------|---|-----------|---------|--------|
| SSIM | Plasma (Ours) vs PASTA | 43 | 0.0 | 2.27e-13 | *** |
| PSNR | Plasma (Ours) vs PASTA | 43 | 2.0 | 6.82e-13 | *** |
| MAE | Plasma (Ours) vs PASTA | 43 | 0.0 | 2.27e-13 | *** |
| SSIM | Plasma (Ours) vs Legacy | 43 | 469.0 | 9.67e-01 | ns |
| PSNR | Plasma (Ours) vs Legacy | 43 | 468.0 | 9.57e-01 | ns |
| MAE | Plasma (Ours) vs Legacy | 43 | 466.0 | 9.38e-01 | ns |
| SSIM | Plasma (Ours) vs FiCD | 43 | 0.0 | 2.27e-13 | *** |
| PSNR | Plasma (Ours) vs FiCD | 43 | 0.0 | 2.27e-13 | *** |
| MAE | Plasma (Ours) vs FiCD | 43 | 0.0 | 2.27e-13 | *** |

## 注意事项

1. **分辨率差异**: PASTA 在 96×112×96 (1.5mm) 下评估，其余方法在 ~160³ (1mm) 下评估。各方法使用自身分辨率下的配对 GT 计算指标，因此指标间存在分辨率偏差。
2. **FiCD 训练不足**: FiCD 仅进行了 1 epoch smoke test 训练，指标预期较差。
3. **NCC**: FiCD 的 NCC 未计算（缺少同分辨率 GT NIfTI）。
4. **Plasma (Ours)** 使用预训练 plasma embedding 作为条件 token，是本项目的核心创新方法。
