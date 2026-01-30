# 扩展 pairs_90d.csv 添加血浆信息

## 功能说明

本脚本用于将血浆生物标志物数据（UPENN 和 C2N）匹配并扩展到 `pairs_90d.csv` 文件中。

## 扩展字段

脚本会添加以下字段：

| 字段名 | 说明 | 来源 |
|--------|------|------|
| `pT217_F` | pTau217 (pg/mL) | UPENN: pT217_F<br>C2N: pT217_C2N |
| `AB42_F` | Aβ42 (pg/mL) | UPENN: AB42_F<br>C2N: AB42_C2N |
| `AB40_F` | Aβ40 (pg/mL) | UPENN: AB40_F<br>C2N: AB40_C2N |
| `AB42_AB40_F` | Aβ42/Aβ40 比值 | UPENN: AB42_AB40_F<br>C2N: AB42_AB40_C2N |
| `pT217_AB42_F` | pTau217/Aβ42 比值 | UPENN: pT217_AB42_F<br>C2N: pT217_npT217_C2N |
| `NfL_Q` | NfL (pg/mL) | UPENN: NfL_Q<br>C2N: 无 |
| `GFAP_Q` | GFAP (pg/mL) | UPENN: GFAP_Q<br>C2N: 无 |
| `plasma_source` | 数据来源 | "UPENN" 或 "C2N" |
| `plasma_date` | 血浆检测日期 | 实际匹配的检测日期 |
| `plasma_date_diff` | 日期差异（天） | pairs 中的 EXAMDATE 与血浆检测日期的差异 |

## 匹配规则

1. **优先级**：优先匹配 UPENN 数据，未匹配的再匹配 C2N 数据
2. **匹配键**：通过 `PTID` 和 `EXAMDATE` 进行匹配
3. **时间窗口**：通过 `--max-days` 参数控制允许的最大日期差异
4. **最佳匹配**：当有多个匹配时，选择日期差异最小的记录

## 使用方法

### 基本用法

```bash
conda run -n xc python adapter_finetune/extend_pairs_with_plasma.py
```

这将使用默认参数：
- `--pairs`: `adapter_finetune/adapter_finetune/pairs_90d.csv`
- `--upenn`: `adapter_finetune/UPENN_PLASMA_FUJIREBIO_QUANTERIX_21Dec2025.csv`
- `--c2n`: `adapter_finetune/C2N_PRECIVITYAD2_PLASMA_29Dec2025.csv`
- `--output`: `adapter_finetune/pairs_90d_with_plasma.csv`
- `--max-days`: `30`

### 自定义参数

```bash
conda run -n xc python adapter_finetune/extend_pairs_with_plasma.py \
    --pairs <pairs文件路径> \
    --upenn <UPENN文件路径> \
    --c2n <C2N文件路径> \
    --output <输出文件路径> \
    --max-days <最大日期差异天数>
```

### 示例

#### 示例 1: 使用 30 天时间窗口（默认）

```bash
conda run -n xc python adapter_finetune/extend_pairs_with_plasma.py \
    --max-days 30 \
    --output adapter_finetune/pairs_90d_with_plasma_30d.csv
```

#### 示例 2: 使用更严格的 14 天时间窗口

```bash
conda run -n xc python adapter_finetune/extend_pairs_with_plasma.py \
    --max-days 14 \
    --output adapter_finetune/pairs_90d_with_plasma_14d.csv
```

#### 示例 3: 使用更宽松的 60 天时间窗口

```bash
conda run -n xc python adapter_finetune/extend_pairs_with_plasma.py \
    --max-days 60 \
    --output adapter_finetune/pairs_90d_with_plasma_60d.csv
```

### 批量运行示例

运行预定义的示例（14、30、60 天）：

```bash
bash adapter_finetune/run_extend_plasma_examples.sh
```

## 输出统计

运行完成后，脚本会输出以下统计信息：

- 总记录数
- 成功匹配记录数及百分比
- UPENN 匹配数及百分比
- C2N 匹配数及百分比
- 未匹配记录数及百分比
- 日期差异统计（平均、中位数、最小、最大）

### 示例输出

```
================================================================================
匹配统计:
================================================================================
总记录数: 3360
成功匹配: 1008 (30.00%)
  - UPENN: 805 (23.96%)
  - C2N: 203 (6.04%)
未匹配: 2352 (70.00%)

日期差异统计 (天):
  - 平均: 10.57
  - 中位数: 8.00
  - 最小: 0
  - 最大: 30
```

## 输出文件

输出文件为 CSV 格式，包含原始 `pairs_90d.csv` 的所有列，外加新增的血浆数据列。

### 文件示例

```csv
PTID,EXAMDATE,id_mri,id_fdg,id_av45,id_av1451,pT217_F,AB42_F,AB40_F,AB42_AB40_F,pT217_AB42_F,NfL_Q,GFAP_Q,plasma_source,plasma_date,plasma_date_diff
002_S_0413,2025-02-19,11128519,,11142890.0,11142891.0,0.244,35.93,420.81,0.0854,0.00679,24.8,156.6,UPENN,2025-02-19,0.0
002_S_5178,2025-01-07,11065829,,11069825.0,11069826.0,0.278,22.08,321.06,0.0688,0.0126,25.5,166.5,UPENN,2025-01-07,0.0
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--pairs` | str | `adapter_finetune/adapter_finetune/pairs_90d.csv` | pairs_90d.csv 文件路径 |
| `--upenn` | str | `adapter_finetune/UPENN_PLASMA_FUJIREBIO_QUANTERIX_21Dec2025.csv` | UPENN 血浆数据文件路径 |
| `--c2n` | str | `adapter_finetune/C2N_PRECIVITYAD2_PLASMA_29Dec2025.csv` | C2N 血浆数据文件路径 |
| `--output` | str | `adapter_finetune/pairs_90d_with_plasma.csv` | 输出文件路径 |
| `--max-days` | int | `30` | 最大允许的日期差异（天） |

## 注意事项

1. **数据来源差异**：
   - UPENN 数据包含 NfL_Q 和 GFAP_Q
   - C2N 数据不包含 NfL_Q 和 GFAP_Q（这些字段会是 NaN）

2. **日期匹配**：
   - 只匹配日期差异在 `max-days` 以内的记录
   - 当有多个匹配时，选择日期最接近的记录

3. **优先级**：
   - UPENN 数据优先于 C2N 数据
   - 已匹配的记录不会再被 C2N 数据覆盖

4. **缺失值**：
   - 未匹配的记录，血浆相关字段为空（NaN 或空字符串）

## 依赖

- Python 3.x
- pandas
- numpy
- Conda 环境：`xc`

## 文件位置

- 脚本：`adapter_finetune/extend_pairs_with_plasma.py`
- 示例脚本：`adapter_finetune/run_extend_plasma_examples.sh`
- 文档：`adapter_finetune/README_extend_plasma.md`

## 常见问题

### Q: 为什么某些记录没有匹配到血浆数据？

A: 可能的原因：
1. 该 PTID 在 UPENN/C2N 数据中不存在
2. 该 PTID 的血浆检测日期与 pairs 中的 EXAMDATE 相差超过 `max-days`
3. 该 PTID 的血浆数据日期无效

### Q: 如何调整匹配的严格程度？

A: 通过 `--max-days` 参数：
- 更小的值（如 7、14）：更严格，只匹配时间非常接近的记录
- 更大的值（如 60、90）：更宽松，匹配更多记录但时间差异可能较大

### Q: C2N 数据中的某些字段为何是 NaN？

A: C2N 数据不包含 NfL_Q 和 GFAP_Q 字段，因此这些字段会显示为 NaN。

## 版本历史

- v1.0 (2026-01-14): 初始版本
  - 支持 UPENN 和 C2N 数据匹配
  - 支持自定义时间窗口
  - 详细的匹配统计输出
