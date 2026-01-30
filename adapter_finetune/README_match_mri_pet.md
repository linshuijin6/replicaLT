# MRI-PET Matching Pipeline

## 功能说明

该脚本用于将 MRI 扫描与 PET 扫描进行配对匹配。对于每条 MRI 记录，脚本会在指定的时间窗口内查找对应的 PET 扫描（FDG、AV45、AV1451/TAU）。

## 匹配原则

1. **subject_id 精确匹配**：MRI 和 PET 必须属于同一受试者
2. **时间窗口限制**：PET 扫描日期与 MRI 扫描日期的差值不超过 `max_days` 天
3. **多匹配选择策略**：如果某个 PET 类型有多个匹配，选择日期最新的（最晚的）那一个

## 使用方法

### 基本用法

```bash
python match_mri_pet_pairs.py \
    --mri <MRI_CSV文件路径> \
    --pet <PET_CSV文件路径> \
    --max-days <最大天数> \
    --output <输出文件路径>
```

### 示例

```bash
# 使用180天时间窗口进行匹配
python adapter_finetune/match_mri_pet_pairs.py \
    --mri adapter_finetune/MRIT1.0114.csv \
    --pet adapter_finetune/PET.0114.csv \
    --max-days 180 \
    --output adapter_finetune/mri_pet_pairs_180d.csv

# 使用90天时间窗口进行匹配
python adapter_finetune/match_mri_pet_pairs.py \
    --mri adapter_finetune/MRIT1.0114.csv \
    --pet adapter_finetune/PET.0114.csv \
    --max-days 90 \
    --output adapter_finetune/mri_pet_pairs_90d.csv
```

## 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--mri` | 是 | - | MRI CSV 文件路径（必须包含列：subject_id, image_id, image_date） |
| `--pet` | 是 | - | PET CSV 文件路径（必须包含列：subject_id, image_id, image_date, radiopharmaceutical） |
| `--output` | 否 | adapter_finetune/mri_pet_pairs_matched.csv | 输出配对结果的 CSV 文件路径 |
| `--max-days` | 否 | 180 | MRI 和 PET 扫描之间允许的最大时间差（天） |
| `--mri-phases` | 否 | ADNI3 ADNI4 | 要包含的 MRI 协议阶段（可多个） |
| `--series-type` | 否 | T1w | 要过滤的 MRI 序列类型 |

## 输入文件要求

### MRI CSV 文件

必需列：
- `subject_id`: 受试者 ID
- `image_id`: MRI 图像 ID
- `image_date`: MRI 扫描日期（格式：YYYY/MM/DD）

可选列（用于过滤）：
- `mri_protocol_phase`: MRI 协议阶段（如 ADNI3, ADNI4）
- `series_type`: 序列类型（如 T1w）
- `series_description`: 序列描述（用于识别重复扫描）

### PET CSV 文件

必需列：
- `subject_id`: 受试者 ID
- `image_id`: PET 图像 ID
- `image_date`: PET 扫描日期（格式：YYYY/MM/DD）
- `radiopharmaceutical`: 示踪剂类型（18F-FDG, 18F-AV45, 18F-AV1451）

## 输出文件格式

输出 CSV 文件包含以下列：

| 列名 | 说明 |
|------|------|
| `PTID` | 受试者 ID（与 subject_id 相同） |
| `EXAMDATE` | MRI 扫描日期 |
| `id_mri` | MRI 图像 ID |
| `id_fdg` | FDG-PET 图像 ID（如果匹配到） |
| `id_av45` | AV45-PET 图像 ID（如果匹配到） |
| `id_av1451` | AV1451/TAU-PET 图像 ID（如果匹配到） |

注：如果某个 PET 类型没有匹配到，对应列为空（NaN）。

## 输出示例

```csv
PTID,EXAMDATE,id_mri,id_fdg,id_av45,id_av1451
127_S_1427,2017/9/6,901027.0,,1601016,1600803
127_S_1427,2018/10/12,1059960.0,,1601016,
127_S_1427,2019/9/30,1234305.0,,1600884,1601018
067_S_0056,2019/1/10,1116451.0,,1598956,1598985
067_S_0056,2017/11/28,1189749.0,,1598956,1598898
```

## 运行统计

脚本运行后会输出以下统计信息：

1. **数据加载统计**：显示原始记录数和过滤后记录数
2. **匹配统计**：
   - 每种 PET 类型的匹配数量和百分比
   - 至少有一个 PET 匹配的记录数
   - 没有任何 PET 匹配的记录数

## 注意事项

1. **日期格式**：输入文件中的日期必须是可解析的格式（推荐 YYYY/MM/DD）
2. **重复扫描**：脚本会自动识别并移除标记为 "repeat" 的重复 MRI 扫描
3. **示踪剂类型**：只支持 18F-FDG、18F-AV45、18F-AV1451 三种示踪剂
4. **性能**：处理大量数据时可能需要较长时间，脚本会显示进度信息

## 故障排除

### 常见问题

1. **文件未找到**
   - 检查输入文件路径是否正确
   - 确保文件具有读取权限

2. **列名不匹配**
   - 确认输入文件包含所有必需列
   - 检查列名拼写是否正确

3. **日期解析错误**
   - 检查日期格式是否一致
   - 查看警告信息中提到的具体行

4. **内存不足**
   - 对于超大文件，可能需要增加系统内存
   - 考虑分批处理数据

## 代码逻辑验证

### 匹配逻辑

```python
# 对于每条 MRI 记录：
for mri_record in mri_data:
    # 1. 找到同一受试者的所有 PET 记录
    pet_records = filter(pet_data, subject_id == mri_record.subject_id)
    
    # 2. 按示踪剂类型分组
    fdg_records = filter(pet_records, radiopharmaceutical == "18F-FDG")
    av45_records = filter(pet_records, radiopharmaceutical == "18F-AV45")
    av1451_records = filter(pet_records, radiopharmaceutical == "18F-AV1451")
    
    # 3. 对于每种示踪剂：
    for tracer_records in [fdg_records, av45_records, av1451_records]:
        # 3.1 计算每条 PET 与 MRI 的天数差
        # 3.2 过滤掉超过 max_days 的记录
        # 3.3 按天数差排序（升序），然后按日期排序（降序，最新的在前）
        # 3.4 选择第一条记录（天数差最小且日期最新）
```

### 测试用例

运行测试：
```bash
# 测试1：使用提供的示例数据
python adapter_finetune/match_mri_pet_pairs.py \
    --mri adapter_finetune/MRIT1.0114.csv \
    --pet adapter_finetune/PET.0114.csv \
    --max-days 180 \
    --output adapter_finetune/test_output.csv

# 验证输出文件
head adapter_finetune/test_output.csv
wc -l adapter_finetune/test_output.csv
```

## 更新日志

- **v2.0 (2026-01-14)**
  - 添加命令行参数支持
  - 修复多匹配时的选择逻辑（优先选择最新的）
  - 统一输出列名为 PTID, EXAMDATE, id_mri, id_fdg, id_av45, id_av1451
  - 改进进度显示和统计信息
  - 添加详细的帮助文档

- **v1.0**
  - 初始版本
