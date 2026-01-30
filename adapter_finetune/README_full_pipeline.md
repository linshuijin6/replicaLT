# ADNI 数据处理完整流程

## 概述

`run_full_pipeline.sh` 整合了 MRI-PET 配对、诊断信息添加、血浆生物标志物扩展的完整数据处理流程。

## 处理步骤

```
┌─────────────────────────────────────────────────────────────────────┐
│  Step 1: MRI-PET 配对                                                │
│  输入: MRI0114.me.csv + PET0114.me.csv                              │
│  输出: pairs_{MAX_DAYS}d.csv                                        │
├─────────────────────────────────────────────────────────────────────┤
│  Step 2: 添加诊断信息                                                │
│  输入: pairs + DXSUM_25Dec2025.csv                                  │
│  输出: pairs_{MAX_DAYS}d_dx.csv                                     │
├─────────────────────────────────────────────────────────────────────┤
│  Step 3: 添加血浆信息                                                │
│  输入: pairs_dx + UPENN + C2N                                       │
│  输出: pairs_{MAX_DAYS}d_dx_plasma_{PLASMA_DAYS}d.csv               │
└─────────────────────────────────────────────────────────────────────┘
```

## 文件结构

```
adapter_finetune/
├── ADNI_csv/                    # ADNI 原始数据（输入）
│   ├── MRI0114.me.csv
│   ├── PET0114.me.csv
│   ├── DXSUM_25Dec2025.csv
│   ├── UPENN_PLASMA_*.csv
│   └── C2N_PRECIVITYAD2_*.csv
│
├── gen_csv/                     # 生成的数据（输出）
│   ├── pairs_180d.csv
│   ├── pairs_180d_dx.csv
│   ├── pairs_180d_dx_plasma_90d.csv
│   └── pipeline_YYYYMMDD_HHMMSS.log  # 处理日志
│
└── run_full_pipeline.sh         # 主脚本
```

## 使用方法

### 基本运行

```bash
cd adapter_finetune
./run_full_pipeline.sh
```

### 自定义参数

编辑 `run_full_pipeline.sh` 中的配置参数：

```bash
# MRI-PET 匹配时间窗口（天）
MAX_DAYS=180

# 诊断查找时间窗口（天）
DX_MAX_DAYS=180

# 血浆数据匹配时间窗口（天）
PLASMA_MAX_DAYS=90

# MRI phases 筛选
MRI_PHASES="ADNI3 ADNI4"
```

## 日志说明

每次运行会在 `gen_csv/` 目录生成带时间戳的日志文件：

```
pipeline_20260127_143052.log
```

日志包含：
- 参数配置
- 每步处理的进度和结果
- 详细的统计报告（PET匹配、诊断分布、血浆数据匹配）
- 数据质量检查

## 输出文件命名规则

| 文件名 | 说明 |
|--------|------|
| `pairs_180d.csv` | 180天窗口的MRI-PET配对 |
| `pairs_180d_dx.csv` | 添加诊断信息后 |
| `pairs_180d_dx_plasma_90d.csv` | 添加90天窗口血浆数据后 |

## 依赖

- Python 3.x
- pandas
- conda 环境: `xc`
