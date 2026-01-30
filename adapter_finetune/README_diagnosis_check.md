# MRI-PET 配对诊断一致性检查功能

## 功能说明

在 MRI-PET 配对脚本中添加了诊断一致性检查功能。当启用此功能时，脚本会：

1. 从 DXSUM CSV 文件加载诊断信息
2. 为每个 MRI 和 PET 扫描查找对应的诊断
3. 只保留诊断一致的配对
4. 排除诊断不一致的配对

## 诊断代码

根据 ADNI 数据标准：
- **1** = 正常 (Normal/CN)
- **2** = 轻度认知障碍 (MCI)
- **3** = 阿尔茨海默病 (Dementia/AD)

## 使用方法

### 基本用法（不检查诊断）

```bash
conda run -n xc python adapter_finetune/match_mri_pet_pairs.py \
    --mri adapter_finetune/data_csv/MRI0114.me.csv \
    --pet adapter_finetune/data_csv/PET0114.me.csv \
    --max-days 90 \
    --output adapter_finetune/pairs_90d.csv
```

### 启用诊断检查

添加 `--check-diagnosis` 参数：

```bash
conda run -n xc python adapter_finetune/match_mri_pet_pairs.py \
    --mri adapter_finetune/data_csv/MRI0114.me.csv \
    --pet adapter_finetune/data_csv/PET0114.me.csv \
    --max-days 90 \
    --check-diagnosis \
    --dxsum adapter_finetune/DXSUM_25Dec2025.csv \
    --output adapter_finetune/pairs_90d_dx.csv
```

### 自定义 DXSUM 文件路径

```bash
conda run -n xc python adapter_finetune/match_mri_pet_pairs.py \
    --mri MRI.csv \
    --pet PET.csv \
    --max-days 90 \
    --check-diagnosis \
    --dxsum /path/to/custom/DXSUM.csv \
    --output pairs_with_dx.csv
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--check-diagnosis` | flag | False | 启用诊断一致性检查 |
| `--dxsum` | Path | `adapter_finetune/DXSUM_25Dec2025.csv` | DXSUM 文件路径 |
| `--diagnosis-max-days` | int | 180 | 诊断查找的最大时间窗口（天） |

## 匹配逻辑

### 1. 诊断查找

对于每个 image_id：
- 使用该扫描的 PTID 和 image_date
- 在 DXSUM 中查找匹配的诊断记录
- 如果找不到完全匹配的日期，查找 ±max_days 天内最接近的诊断
- **回退机制**：如果在 max_days 范围内仍找不到诊断，则尝试获取 examdate 前后的两个诊断记录，如果这两个诊断相同，则使用该诊断

### 2. 诊断比较

对于每个 MRI-PET 配对：
- 获取 MRI 的诊断
- 获取 PET 的诊断（FDG、AV45、AV1451）
- 只保留诊断代码完全相同的配对
- 如果任一扫描找不到诊断，则不进行过滤

### 3. 过滤规则

- **MRI 诊断 = PET 诊断** → 保留配对
- **MRI 诊断 ≠ PET 诊断** → 排除配对
- **任一诊断缺失** → 保留配对（不过滤）

## 示例结果

### 不检查诊断（90天窗口）

```
Total MRI records processed: 3360
Total paired records: 3360

Matched counts by tracer:
  FDG:      367 (10.9%)
  AV45:    1251 (37.2%)
  AV1451:  1867 (55.6%)

Records with at least one PET match: 2282 (67.9%)
```

### 检查诊断后（90天窗口）

```
Total MRI records processed: 3360
Total paired records: 3360

Matched counts by tracer:
  FDG:      367 (10.9%)
  AV45:    1244 (37.0%)
  AV1451:  1858 (55.3%)

Records with at least one PET match: 2273 (67.6%)
```

### 差异分析

- **FDG 被过滤**: 0 条
- **AV45 被过滤**: 7 条
- **AV1451 被过滤**: 9 条
- **总配对被过滤**: 9 条

这表明大约 0.4% 的配对由于诊断不一致被排除。

## 批量运行示例

使用更新后的 `run_matching_examples.sh` 脚本：

```bash
bash adapter_finetune/run_matching_examples.sh
```

该脚本会生成：

**不检查诊断：**
- `adapter_finetune/pairs_180d.csv`
- `adapter_finetune/pairs_90d.csv`
- `adapter_finetune/pairs_365d.csv`

**检查诊断一致性：**
- `adapter_finetune/pairs_90d_dx.csv`
- `adapter_finetune/pairs_180d_dx.csv`

## 注意事项

1. **诊断时间窗口**：
   - 诊断匹配使用 `--diagnosis-max-days` 指定的时间窗口（默认 180 天）
   - 这与 MRI-PET 匹配的时间窗口（由 `--max-days` 指定）是独立的

2. **诊断回退机制**：
   - 如果在 diagnosis-max-days 范围内找不到诊断
   - 会尝试获取 examdate 之前和之后最近的两个诊断
   - 仅当这两个诊断相同时，才使用该诊断
   - 这可以覆盖更多有效记录，同时保证诊断的准确性

3. **缺失诊断**：
   - 如果 DXSUM 中找不到某次扫描的诊断（包括回退方案也失败），该配对不会被过滤
   - 这确保了诊断数据不完整时不会丢失有效配对

3. **多次 PET 扫描**：
   - 同一 MRI 可能匹配多个 PET 示踪剂（FDG、AV45、AV1451）
   - 每种示踪剂独立检查诊断一致性
   - 可能出现某些示踪剂被过滤而其他保留的情况

4. **诊断变化**：
   - 如果受试者在 MRI 和 PET 之间的时间段内诊断发生变化（如从 MCI 进展到 AD），该配对会被排除
   - 这有助于确保数据的一致性，但也可能排除一些临床上有意义的转换期数据

## 实现细节

### 诊断查找函数

```python
def find_diagnosis_fallback(
    ptid: str, 
    exam_date: pd.Timestamp, 
    dxsum: pd.DataFrame
) -> Optional[int]:
    \"\"\"
    回退方案：当在 max_days 范围内找不到诊断时，
    尝试获取 examdate 前后的两个诊断。
    如果这两个诊断相同，则返回该诊断。
    
    返回: 诊断代码 (1, 2, 3) 或 None
    \"\"\"


def find_diagnosis(
    ptid: str, 
    exam_date: pd.Timestamp, 
    dxsum: pd.DataFrame, 
    max_days: int = 90,
    use_fallback: bool = True
) -> Optional[int]:
    \"\"\"
    根据 PTID 和 EXAMDATE 查找诊断。
    
    1. 首先在 max_days 范围内查找最接近的记录
    2. 如果找不到且 use_fallback=True，则使用回退方案
    
    返回: 诊断代码 (1, 2, 3) 或 None
    \"\"\"
```

### 过滤逻辑

```python
# 检查诊断一致性
if check_diagnosis and dxsum is not None:
    mri_dx = find_diagnosis(subject_id, mri_date, dxsum)
    
    # 对每种 PET 示踪剂检查诊断
    if fdg_id and mri_dx and fdg_dx:
        if mri_dx != fdg_dx:
            fdg_id = None  # 排除诊断不一致的配对
    
    # 同样处理 AV45 和 AV1451...
```

## 常见问题

### Q: 为什么有些配对被过滤了？

A: 配对被过滤的原因：
1. MRI 和 PET 的诊断代码不同
2. 受试者在两次扫描之间诊断发生了变化
3. 诊断记录的日期与扫描日期不完全匹配，但在 ±30 天内找到了不同的诊断

### Q: 如何查看哪些配对被过滤了？

A: 比较有无诊断检查的两个输出文件：

```python
import pandas as pd

without_dx = pd.read_csv('pairs_90d.csv')
with_dx = pd.read_csv('pairs_90d_dx.csv')

# 找出被过滤的配对
for tracer in ['id_fdg', 'id_av45', 'id_av1451']:
    filtered = without_dx[tracer].notna() & with_dx[tracer].isna()
    print(f"{tracer} 被过滤的记录:")
    print(without_dx[filtered][['PTID', 'EXAMDATE', tracer]])
```

### Q: 诊断检查会影响性能吗？

A: 性能影响很小：
- 诊断查找使用索引优化
- 额外的处理时间通常 < 10%
- 对于 3360 条 MRI 记录，总处理时间仍在秒级

### Q: 可以调整诊断匹配的时间窗口吗？

A: 可以使用 `--diagnosis-max-days` 参数来调整诊断匹配的时间窗口：

```bash
python match_mri_pet_pairs.py \
    --mri MRI.csv --pet PET.csv \
    --max-days 180 \
    --diagnosis-max-days 90 \
    --check-diagnosis \
    --output pairs.csv
```

如果在指定窗口内找不到诊断，会自动尝试回退方案（使用 examdate 前后的诊断）。

## 版本历史

- v1.2 (2026-01-27): 增加诊断回退机制和参数化配置
  - 新增 `--diagnosis-max-days` 参数
  - 新增 `find_diagnosis_fallback()` 函数
  - 当在 max_days 范围内找不到诊断时，尝试使用 examdate 前后的诊断
  - 仅当前后诊断相同时才使用回退诊断
  
- v1.1 (2026-01-14): 添加诊断一致性检查功能
  - 新增 `--check-diagnosis` 参数
  - 新增 `--dxsum` 参数
  - 添加 `find_diagnosis()` 函数
  - 更新匹配逻辑以支持诊断过滤
  
- v1.0: 初始版本（仅时间窗口匹配）

## 相关文件

- 主脚本：`adapter_finetune/match_mri_pet_pairs.py`
- 示例脚本：`adapter_finetune/run_matching_examples.sh`
- DXSUM 数据：`adapter_finetune/DXSUM_25Dec2025.csv`
