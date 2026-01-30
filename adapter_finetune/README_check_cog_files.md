# check_cog_files.py 使用说明

## 功能描述

检查配准(Coregistration)文件的完成度，根据 CSV 表格检查生成数据路径下的文件是否完整，并可选地从日志文件中分析缺失原因。

## 文件命名规则

- **MRI 文件**: `<PTID>__I<id_mri>.nii.gz` 或 `<PTID>__<id_mri>.nii.gz`
- **PET 文件**: `<PTID>__I<id_mri>__I<id_pet>.nii.gz` 或其他组合

## 检查的文件类型

### MRI (每个 id_mri 需要 4 个文件)
1. `MRI/<PTID>__I<id_mri>.nii.gz` - MRI 主文件
2. `MRI_MASK/<PTID>__I<id_mri>_mask.nii.gz` - 脑掩膜
3. `MRI_XFM/<PTID>__I<id_mri>_mri2mni.mat` - MRI到MNI变换矩阵
4. `MRI_NATIVE_RSTD/<PTID>__I<id_mri>_rstd.nii.gz` - 重定向的原始MRI

### PET (每个 id_pet 需要 4 个文件)
1. `PET_MNI/<modality>/<PTID>__I<id_mri>__I<id_pet>.nii.gz` - PET brain
2. `PET_MNI/<modality>/<PTID>__I<id_mri>__I<id_pet>_full.nii.gz` - PET full
3. `PET_MNI/<modality>/<PTID>__I<id_mri>__I<id_pet>_pet2mni.mat` - PET到MNI变换
4. `PET_MNI/<modality>/<PTID>__I<id_mri>__I<id_pet>_pet2mri.mat` - PET到MRI变换

其中 modality 包括：FDG, AV45, TAU

## 使用方法

### 基本用法（不检查日志）

```bash
python check_cog_files.py \
  --csv pairs_180d_dx.csv \
  --cog_root /mnt/nfsdata/nfsdata/ADNI/ADNI0103/Coregistration
```

### 完整用法（包含日志分析）

```bash
python check_cog_files.py \
  --csv /path/to/pairs_180d_dx.csv \
  --cog_root /mnt/nfsdata/nfsdata/ADNI/ADNI0103/Coregistration \
  --logs_root /mnt/nfsdata/nfsdata/ADNI/ADNI0103/logs_pipeline0106_parallel
```

## 参数说明

- `--csv`: 输入的 CSV 文件路径（必需）
  - 需要包含的列：PTID, id_mri
  - 可选列：id_fdg, id_av45, id_av1451
  
- `--cog_root`: 配准生成数据存放路径（必需）
  - 例如：`/mnt/nfsdata/nfsdata/ADNI/ADNI0103/Coregistration`
  
- `--logs_root`: 生成过程日志文件夹路径（可选）
  - 例如：`/mnt/nfsdata/nfsdata/ADNI/ADNI0103/logs_pipeline0106_parallel`
  - 支持直接日志文件和分区日志（part_0, part_1等）

## 输出文件

运行后会在 CSV 文件所在目录生成以下文件：

1. **`<csv_name>_mri_missing.csv`** - MRI 缺失/部分完成的文件详情
   - 列：PTID, id_mri, found, total, missing_files, log_status, status, reason
   - log_status: OK/FAIL/SKIP/NOT_FOUND（日志中的状态）
   - status: 综合状态描述（文件存在情况+日志状态）
   - reason: 从日志中提取的具体失败原因
   
2. **`<csv_name>_pet_missing.csv`** - PET 缺失/部分完成的文件详情
   - 列：PTID, id_mri, modality, id_pet, found, total, missing_files, log_status, status, reason
   
3. **`<csv_name>_cog_summary.json`** - 汇总统计报告
   - 包含 MRI, FDG, AV45, TAU 各类文件的完整度统计
   
4. **`<csv_name>_analysis.json`** - 问题分析报告（可选生成）
   - MRI/PET 失败原因统计
   - 按类型分类的问题分布

## 输出示例

### 终端输出

```
================================================================================
检查配准文件完成度
CSV 文件: pairs_180d_dx.csv
配准路径: /mnt/nfsdata/nfsdata/ADNI/ADNI0103/Coregistration
日志路径: /mnt/nfsdata/nfsdata/ADNI/ADNI0103/logs_pipeline0106_parallel
================================================================================

开始检查 3360 行数据...

进度: 3360/3360 (100.0%)

检查完成！

================================================================================
统计结果
================================================================================
MRI    - 总数: 3360, 完整: 3310 (98.5%), 部分:   50, 缺失:    0
FDG    - 总数:  419, 完整:  419 (100.0%), 部分:    0, 缺失:    0
AV45   - 总数: 1366, 完整: 1340 (98.1%), 部分:    0, 缺失:   26
TAU    - 总数: 2091, 完整: 2062 (98.6%), 部分:    0, 缺失:   29
================================================================================
```

### JSON 汇总报告

```json
{
  "total_rows": 3360,
  "mri": {
    "complete": 3310,
    "partial": 50,
    "missing": 0,
    "total": 3360
  },
  "fdg": {
    "complete": 419,
    "partial": 0,
    "missing": 0,
    "total": 419
  },
  "av45": {
    "complete": 1340,
    "partial": 0,
    "missing": 26,
    "total": 1366
  },
  "tau": {
    "complete": 2062,
    "partial": 0,
    "missing": 29,
    "total": 2091
  }
}
```

## 状态说明

- **完整(complete)**: 所有应有文件都存在（4/4）
- **部分(partial)**: 部分文件存在但不完整（1-3/4）
- **缺失(missing)**: 完全没有文件（0/4）

**注意**：状态判断以实际文件是否存在为准，不依赖日志状态。

## 日志分析

如果提供了 `--logs_root` 参数，脚本会：
1. 读取 `pipeline_mri.csv` 和 `pipeline_pet.csv` 日志文件
2. 支持分区日志结构（如 part_0, part_1, part_2...）
3. 查找所有相关记录并分析状态（OK/FAIL/SKIP）
4. 提取失败原因并添加到输出的 CSV 文件中
5. **重要**：以实际文件是否存在为准，而非日志状态
   - 即使日志显示成功，但文件不存在，仍会标记为失败
   - 即使日志显示失败，但文件存在，仍会标记为成功

## 状态字段说明

### log_status（日志状态）
- **OK**: 日志中显示处理成功
- **FAIL**: 日志中显示处理失败
- **SKIP**: 日志中显示跳过处理
- **NOT_FOUND**: 日志中无相关记录

### status（综合状态）
综合文件存在情况和日志状态的描述，例如：
- "文件部分存在(1/4) [日志:失败]"
- "文件完全缺失 [日志:成功但文件缺失]"
- "文件完全缺失 [日志:无记录]"

### reason（失败原因）
从日志中提取的具体错误信息，常见原因包括：
- **FAST::Command failed**: FSL FAST 工具失败（脑组织分割）
- **SYNTHSTRIP::Command failed**: FreeSurfer SynthStrip 失败（脑提取）
- **REORIENT::Command failed**: FSL 方向调整失败
- **FLIRT::Command failed**: FSL 配准工具失败

## 注意事项

1. 脚本会递归搜索配准路径下的所有子文件夹
2. 支持文件名中带 "I" 前缀和不带 "I" 前缀的两种格式
3. CSV 中 image_id 为空值的会自动跳过
4. 日志文件不存在或无法读取时不会报错，只是显示"未找到日志信息"
5. 大型数据集检查可能需要几分钟时间，请耐心等待

## 依赖要求

```bash
pandas>=1.0.0
```

## 分析结果

运行检查后，可以使用 `analyze_cog_results.py` 生成详细分析报告：

```bash
python analyze_cog_results.py --output analysis_report.json
```

该脚本会自动查找检查结果文件并生成：
- 完成度统计
- 失败原因分类
- 问题分布分析

示例输出：
```
================================================================================
配准完成度分析报告
================================================================================

总行数: 3360

各类文件完成情况:
  MRI   : 3310/3360 ( 98.5%) - 部分: 50 缺失:  0
  FDG   :  419/ 419 (100.0%) - 部分:  0 缺失:  0
  AV45  : 1340/1366 ( 98.1%) - 部分:  0 缺失: 26
  TAU   : 2062/2091 ( 98.6%) - 部分:  0 缺失: 29

MRI 问题分析
- 总问题数: 50
- 主要失败原因: FAST失败(脑组织分割) 49例
```

## 作者

根据 `check_nifti_files.py` 改编，用于检查配准文件完成度。
