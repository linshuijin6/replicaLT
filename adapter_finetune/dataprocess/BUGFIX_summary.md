# 并行脚本问题修复总结

## 问题描述

用户运行 `run_pipeline_parallel.sh` 时，只创建了CSV分割文件，但没有生成任何进程或日志文件。

## 问题根因

**主要问题：`set -e` + `return` 非零值导致脚本意外退出**

1. 脚本开头使用了 `set -e`，这会在任何命令返回非零状态码时立即退出
2. `split_csv()` 函数使用了 `return ${part_num}`（非零值）返回分割的部分数
3. 当 `split_csv()` 返回非零值时，`set -e` 导致整个脚本立即退出
4. 因此后续的启动进程代码没有执行

## 解决方案

### 1. 移除 `set -e`
```bash
# 修改前
set -e

# 修改后
# 不使用 set -e，避免在启动后台任务时因为返回值导致退出
# set -e
```

### 2. 修复 return 语句
```bash
# 修改前
return ${part_num}

# 修改后
# 返回0表示成功，避免set -e导致退出
return 0
```

### 3. 添加调试输出和进程验证
```bash
echo "[DEBUG] 执行命令: conda run -n ${CONDA_ENV} python ${PYTHON_SCRIPT} ..."

# 启动后等待并验证进程
sleep 1
if ps -p ${pid} > /dev/null 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] part_${part_num} 进程确认运行中"
else
    echo "[错误] part_${part_num} 进程启动失败，请查看日志: ${nohup_log}"
fi
```

## 验证结果

修复后的脚本成功：
- ✅ 分割CSV成4个部分（每部分840行，共3360行）
- ✅ 启动4个并行进程（PID: 945832, 946100, 946376, 946650）
- ✅ 创建所有nohup日志文件
- ✅ 创建所有处理日志目录和CSV文件
- ✅ 进程正常执行MRI和PET处理任务

## 使用建议

1. **启动脚本**：
   ```bash
   cd /home/ssddata/linshuijin/replicaLT/adapter_finetune/dataprocess
   bash run_pipeline_parallel.sh
   ```

2. **查看状态**：
   ```bash
   bash run_pipeline_parallel.sh status
   ```

3. **监控日志**：
   ```bash
   # nohup完整日志
   tail -f /mnt/nfsdata/nfsdata/ADNI/ADNI0103/nohup_logs_pipeline0106/part_0.log
   
   # MRI处理日志
   tail -f /mnt/nfsdata/nfsdata/ADNI/ADNI0103/logs_pipeline0106_parallel/part_0/pipeline_mri.csv
   
   # PET处理日志
   tail -f /mnt/nfsdata/nfsdata/ADNI/ADNI0103/logs_pipeline0106_parallel/part_0/pipeline_pet.csv
   ```

4. **停止所有任务**（如需要）：
   ```bash
   bash run_pipeline_parallel.sh stop
   ```

## 教训

1. 在编写包含后台任务和函数返回值的shell脚本时，避免使用 `set -e`
2. 如果必须使用 `set -e`，确保所有函数都返回0表示成功
3. 对于启动后台任务的脚本，应添加进程验证机制
4. 添加充足的调试输出帮助排查问题
