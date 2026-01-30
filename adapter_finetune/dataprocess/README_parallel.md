# 并行Pipeline处理脚本使用说明

## 功能说明

`run_pipeline_parallel.sh` 脚本用于加速 `pipeline_reuse_mri_pet.py` 的处理速度，通过以下方式实现：

1. 将输入CSV文件分割成多个部分
2. 为每个部分创建独立的日志和临时目录
3. 使用nohup并行运行所有部分
4. 记录每个部分的处理日志

## 快速开始

### 1. 配置参数

编辑脚本中的配置参数（在脚本顶部）：

```bash
# 输入CSV路径
INPUT_CSV="/home/ssddata/linshuijin/replicaLT/adapter_finetune/pairs0106_filtered.csv"

# 分割成多少个部分（根据CPU核心数调整，建议4-8）
NUM_PARTS=4

# Conda环境
CONDA_ENV="xc"
```

### 2. 启动并行处理

```bash
cd /home/ssddata/linshuijin/replicaLT/adapter_finetune/dataprocess
bash run_pipeline_parallel.sh
```

脚本会自动：
- 分割CSV文件
- 为每个部分创建独立目录
- 启动所有任务（后台运行）
- 显示监控命令

## 监控和管理

### 查看任务状态

```bash
bash run_pipeline_parallel.sh status
```

### 查看实时日志

查看某个part的nohup日志（所有输出）：
```bash
tail -f /mnt/nfsdata/nfsdata/ADNI/ADNI0103/nohup_logs_pipeline0106/part_0.log
```

查看某个part的处理日志（MRI/PET处理详情）：
```bash
# MRI处理日志
tail -f /mnt/nfsdata/nfsdata/ADNI/ADNI0103/logs_pipeline0106_parallel/part_0/pipeline_mri.csv

# PET处理日志
tail -f /mnt/nfsdata/nfsdata/ADNI/ADNI0103/logs_pipeline0106_parallel/part_0/pipeline_pet.csv
```

### 停止所有任务

```bash
bash run_pipeline_parallel.sh stop
```

### 查看所有日志文件

```bash
ls -lh /mnt/nfsdata/nfsdata/ADNI/ADNI0103/nohup_logs_pipeline0106/*.log
```

## 目录结构

运行后会生成以下目录结构：

```
/mnt/nfsdata/nfsdata/ADNI/ADNI0103/
├── nohup_logs_pipeline0106/          # nohup日志目录
│   ├── part_0.log                    # part 0的完整输出日志
│   ├── part_1.log
│   ├── part_0.pid                    # 进程ID文件
│   ├── part_1.pid
│   ├── run_info.txt                  # 运行信息
│   └── csv_parts/                    # 分割后的CSV文件
│       ├── part_0.csv
│       ├── part_1.csv
│       └── ...
├── logs_pipeline0106_parallel/       # 处理日志目录
│   ├── part_0/
│   │   ├── pipeline_mri.csv          # MRI处理记录
│   │   └── pipeline_pet.csv          # PET处理记录
│   ├── part_1/
│   └── ...
├── tmp_pipeline0106_parallel/        # 临时文件目录
│   ├── part_0/
│   ├── part_1/
│   └── ...
└── Coregistration/                   # 最终输出目录（所有part共享）
    ├── MRI/
    ├── MRI_MASK/
    ├── MRI_XFM/
    ├── MRI_NATIVE_RSTD/
    └── PET_MNI/
```

## 注意事项

1. **分割数量**：根据服务器CPU核心数和内存设置 `NUM_PARTS`，建议4-8个
2. **资源占用**：每个part会独立运行FSL工具，注意总体资源占用
3. **日志监控**：定期检查日志确保各部分正常运行
4. **断点续跑**：脚本支持断点续跑，已处理的文件会自动跳过
5. **环境变量**：确保在xc conda环境中有所需的所有依赖（FSL、freesurfer等）

## 性能优化建议

1. 如果服务器有8核，设置 `NUM_PARTS=4` 或 `NUM_PARTS=6`
2. 如果某些part完成较早，可以手动停止并重新分割未完成的数据
3. 检查 `OMP_NUM_THREADS=1` 设置，避免单个进程占用过多CPU

## 故障排查

### 任务没有启动

检查：
- Conda环境是否正确：`conda env list`
- Python脚本路径是否正确
- CSV文件是否存在

### 任务运行中断

检查nohup日志找到错误原因：
```bash
grep -i "error\|fail\|exception" /mnt/nfsdata/nfsdata/ADNI/ADNI0103/nohup_logs_pipeline0106/part_*.log
```

### 进程僵尸

清理僵尸进程：
```bash
bash run_pipeline_parallel.sh stop
# 然后手动检查并kill残留进程
ps aux | grep pipeline_reuse_mri_pet
```

## 示例：完整运行流程

```bash
# 1. 进入目录
cd /home/ssddata/linshuijin/replicaLT/adapter_finetune/dataprocess

# 2. 启动并行处理
bash run_pipeline_parallel.sh

# 3. 查看状态
bash run_pipeline_parallel.sh status

# 4. 实时监控第一个part
tail -f /mnt/nfsdata/nfsdata/ADNI/ADNI0103/nohup_logs_pipeline0106/part_0.log

# 5. 等待所有完成后，检查处理结果
wc -l /mnt/nfsdata/nfsdata/ADNI/ADNI0103/logs_pipeline0106_parallel/part_*/pipeline_mri.csv
wc -l /mnt/nfsdata/nfsdata/ADNI/ADNI0103/logs_pipeline0106_parallel/part_*/pipeline_pet.csv
```

## 合并结果

所有part的输出会写入同一个 `Coregistration` 目录，无需手动合并。

要统计总体处理情况：

```bash
# 合并所有MRI日志
cat /mnt/nfsdata/nfsdata/ADNI/ADNI0103/logs_pipeline0106_parallel/part_*/pipeline_mri.csv \
  | grep -v "^subject_id" | sort | uniq > merged_mri.csv

# 合并所有PET日志
cat /mnt/nfsdata/nfsdata/ADNI/ADNI0103/logs_pipeline0106_parallel/part_*/pipeline_pet.csv \
  | grep -v "^subject_id" | sort | uniq > merged_pet.csv

# 统计成功/失败数量
grep ",OK$" merged_mri.csv | wc -l
grep ",FAIL$" merged_mri.csv | wc -l
```
