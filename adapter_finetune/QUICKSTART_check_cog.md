# 配准文件检查工具 - 快速使用指南

## 工具说明

本工具包含两个主要脚本：
1. **check_cog_files.py** - 检查配准文件完成度
2. **analyze_cog_results.py** - 分析检查结果并生成报告

## 快速开始

### 1. 检查配准文件

```bash
python check_cog_files.py \
  --csv pairs_180d_dx.csv \
  --cog_root /mnt/nfsdata/nfsdata/ADNI/ADNI0103/Coregistration \
  --logs_root /mnt/nfsdata/nfsdata/ADNI/ADNI0103/logs_pipeline0106_parallel
```

**输出文件：**
- `pairs_180d_dx_mri_missing.csv` - MRI 问题详情
- `pairs_180d_dx_pet_missing.csv` - PET 问题详情
- `pairs_180d_dx_cog_summary.json` - 汇总统计

### 2. 分析检查结果

```bash
python analyze_cog_results.py --output analysis_report.json
```

**输出：**终端显示详细分析报告，并保存到 JSON 文件

## 实际案例结果

### 数据集：pairs_180d_dx.csv (3360行)

**完成度：**
- ✓ MRI:  98.5% (3310/3360) - 50个部分完成
- ✓ FDG:  100% (419/419) - 全部完整
- ✓ AV45: 98.1% (1340/1366) - 26个完全缺失
- ✓ TAU:  98.6% (2062/2091) - 29个完全缺失

**主要问题：**
1. MRI: 50个FAST失败(脑组织分割问题)
2. PET: 52个无日志记录(可能未处理)

## 关键特性

✨ **智能判断**
- 以实际文件存在为准（不仅依赖日志）
- 即使日志显示成功，文件不存在也会标记为失败

✨ **详细分析**
- 提取日志中的具体失败原因
- 支持分区日志（part_0, part_1...）
- 区分失败类型（FAST, SYNTHSTRIP, REORIENT等）

✨ **多维度统计**
- 按状态统计（完整/部分/缺失）
- 按模态统计（FDG/AV45/TAU）
- 按失败原因统计

## 输出字段说明

### log_status（日志状态）
- `OK`: 日志显示成功
- `FAIL`: 日志显示失败
- `SKIP`: 日志显示跳过
- `NOT_FOUND`: 无日志记录

### status（综合状态）
文件存在情况 + 日志状态的组合描述

例如：
- "文件部分存在(1/4) [日志:失败]"
- "文件完全缺失 [日志:无记录]"

### reason（失败原因）
从日志提取的具体错误，如：
- `FAST::Command failed` - 脑组织分割失败
- `SYNTHSTRIP::Command failed` - 脑提取失败
- `REORIENT::Command failed` - 方向调整失败

## 常见问题处理

### Q: 文件显示部分存在(1/4)是什么意思？
A: 表示应有4个文件，但只找到1个。通常是处理中断或失败导致。

### Q: 为什么有些PET显示"无日志记录"？
A: 可能原因：
1. 该PET未包含在处理批次中
2. 对应的MRI处理失败，导致PET无法处理
3. 在其他日志目录中（检查其他logs文件夹）

### Q: FAST失败如何处理？
A: FAST是FSL的脑组织分割工具，失败通常是因为：
1. 图像质量问题（对比度低、伪影）
2. 图像格式/方向问题
3. 需要手动检查这些图像

## 更多信息

详细文档请参阅：[README_check_cog_files.md](README_check_cog_files.md)
