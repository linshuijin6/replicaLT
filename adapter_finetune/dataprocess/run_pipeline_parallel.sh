#!/bin/bash
# 并行运行pipeline_reuse_mri_pet.py的脚本
# 将CSV分割成多个部分并行处理

# 不使用 set -e，避免在启动后台任务时因为返回值导致退出
# set -e

# ==================== 配置参数 ====================
# 输入CSV路径
INPUT_CSV="/home/ssddata/linshuijin/replicaLT/adapter_finetune/adapter_finetune/pairs_180d_dx.csv"

# 分割成多少个部分（根据CPU核心数调整）
NUM_PARTS=6

# 固定参数
NIFTI_ROOT="/mnt/nfsdata/nfsdata/ADNI/ADNI0103/NIFTI"
SOURCE_MRI_ROOT="/mnt/nfsdata/nfsdata/LorenzoT/processed"
TARGET_ROOT="/mnt/nfsdata/nfsdata/ADNI/ADNI0103/Coregistration"

# 基础日志和临时目录（每个part会有独立子目录）
BASE_LOGS_ROOT="/mnt/nfsdata/nfsdata/ADNI/ADNI0103/logs_pipeline0106_parallel"
BASE_TMP_ROOT="/mnt/nfsdata/nfsdata/ADNI/ADNI0103/tmp_pipeline0106_parallel"

# nohup日志目录
NOHUP_LOG_DIR="/mnt/nfsdata/nfsdata/ADNI/ADNI0103/nohup_logs_pipeline0106"

# Python脚本路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/pipeline_reuse_mri_pet.py"

# Conda环境
CONDA_ENV="xc"

# ==================== 函数定义 ====================

# 分割CSV文件
split_csv() {
    local input_csv="$1"
    local num_parts="$2"
    local output_dir="$3"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始分割CSV: ${input_csv}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 分割成 ${num_parts} 个部分"
    
    mkdir -p "${output_dir}"
    
    # 读取CSV总行数（不包括表头）
    local total_lines=$(tail -n +2 "${input_csv}" | wc -l)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] CSV总数据行数: ${total_lines}"
    
    # 计算每个部分的行数
    local lines_per_part=$(( (total_lines + num_parts - 1) / num_parts ))
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 每个部分约 ${lines_per_part} 行"
    
    # 提取表头
    head -n 1 "${input_csv}" > "${output_dir}/header.txt"
    
    # 分割数据部分
    tail -n +2 "${input_csv}" | split -l ${lines_per_part} -d -a 3 - "${output_dir}/part_"
    
    # 为每个部分添加表头
    local part_num=0
    for part_file in "${output_dir}"/part_*; do
        if [[ -f "$part_file" && ! "$part_file" =~ \.csv$ ]]; then
            local csv_file="${output_dir}/part_${part_num}.csv"
            cat "${output_dir}/header.txt" "$part_file" > "$csv_file"
            rm "$part_file"
            local lines=$(tail -n +2 "$csv_file" | wc -l)
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 创建 part_${part_num}.csv (${lines} 行)"
            part_num=$((part_num + 1))
        fi
    done
    
    rm "${output_dir}/header.txt"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] CSV分割完成，共 ${part_num} 个文件"
    # 返回0表示成功，避免set -e导致退出
    return 0
}

# 启动单个处理任务
start_part() {
    local part_num="$1"
    local csv_file="$2"
    
    local logs_root="${BASE_LOGS_ROOT}/part_${part_num}"
    local tmp_root="${BASE_TMP_ROOT}/part_${part_num}"
    local nohup_log="${NOHUP_LOG_DIR}/part_${part_num}.log"
    
    mkdir -p "${logs_root}"
    mkdir -p "${tmp_root}"
    mkdir -p "$(dirname "${nohup_log}")"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动 part_${part_num}..."
    echo "  - CSV: ${csv_file}"
    echo "  - 日志: ${logs_root}"
    echo "  - nohup日志: ${nohup_log}"
    echo "  - Python脚本: ${PYTHON_SCRIPT}"
    echo "  - Conda环境: ${CONDA_ENV}"
    
    # 使用conda run在xc环境中运行
    echo "[DEBUG] 执行命令: conda run -n ${CONDA_ENV} python ${PYTHON_SCRIPT} --pairs_csv ${csv_file} ..."
    nohup conda run -n ${CONDA_ENV} python "${PYTHON_SCRIPT}" \
        --pairs_csv "${csv_file}" \
        --nifti_root "${NIFTI_ROOT}" \
        --source_mri_root "${SOURCE_MRI_ROOT}" \
        --target_root "${TARGET_ROOT}" \
        --logs_root "${logs_root}" \
        --tmp_root "${tmp_root}" \
        > "${nohup_log}" 2>&1 &
    
    local pid=$!
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] part_${part_num} 已启动，PID: ${pid}"
    echo "${pid}" > "${NOHUP_LOG_DIR}/part_${part_num}.pid"
    
    # 等待一下确认进程启动
    sleep 1
    if ps -p ${pid} > /dev/null 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] part_${part_num} 进程确认运行中"
    else
        echo "[错误] part_${part_num} 进程启动失败，请查看日志: ${nohup_log}"
    fi
}

# 检查所有任务状态
check_status() {
    echo ""
    echo "==================== 任务状态 ===================="
    for pid_file in "${NOHUP_LOG_DIR}"/part_*.pid; do
        if [[ -f "$pid_file" ]]; then
            local part_name=$(basename "$pid_file" .pid)
            local pid=$(cat "$pid_file")
            if ps -p ${pid} > /dev/null 2>&1; then
                echo "[运行中] ${part_name} (PID: ${pid})"
            else
                echo "[已完成] ${part_name} (PID: ${pid})"
            fi
        fi
    done
    echo "=================================================="
    echo ""
}

# ==================== 主流程 ====================

echo "=========================================="
echo "  并行处理 Pipeline"
echo "=========================================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "输入CSV: ${INPUT_CSV}"
echo "分割数量: ${NUM_PARTS}"
echo "Python脚本: ${PYTHON_SCRIPT}"
echo "Conda环境: ${CONDA_ENV}"
echo "=========================================="
echo ""

# 检查输入文件
if [[ ! -f "${INPUT_CSV}" ]]; then
    echo "错误: 输入CSV文件不存在: ${INPUT_CSV}"
    exit 1
fi

# 检查Python脚本
if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "错误: Python脚本不存在: ${PYTHON_SCRIPT}"
    exit 1
fi

# 创建工作目录
WORK_DIR="${NOHUP_LOG_DIR}/csv_parts"
mkdir -p "${WORK_DIR}"
mkdir -p "${NOHUP_LOG_DIR}"
mkdir -p "${BASE_LOGS_ROOT}"
mkdir -p "${BASE_TMP_ROOT}"

# 分割CSV
split_csv "${INPUT_CSV}" "${NUM_PARTS}" "${WORK_DIR}"

# 启动所有任务
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始启动所有任务..."
echo ""

part_num=0
for csv_file in "${WORK_DIR}"/part_*.csv; do
    if [[ -f "$csv_file" ]]; then
        start_part ${part_num} "${csv_file}"
        part_num=$((part_num + 1))
        sleep 2  # 避免同时启动导致资源竞争
    fi
done

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 所有任务已启动！"
echo ""

# 显示初始状态
check_status

# 提供监控命令提示
echo "=========================================="
echo "  监控命令"
echo "=========================================="
echo "查看所有nohup日志:"
echo "  ls -lh ${NOHUP_LOG_DIR}/*.log"
echo ""
echo "实时查看某个part的日志:"
echo "  tail -f ${NOHUP_LOG_DIR}/part_0.log"
echo ""
echo "查看所有任务状态:"
echo "  bash $0 status"
echo ""
echo "停止所有任务:"
echo "  bash $0 stop"
echo "=========================================="

# 保存运行信息
cat > "${NOHUP_LOG_DIR}/run_info.txt" <<EOF
启动时间: $(date '+%Y-%m-%d %H:%M:%S')
输入CSV: ${INPUT_CSV}
分割数量: ${NUM_PARTS}
日志目录: ${BASE_LOGS_ROOT}
临时目录: ${BASE_TMP_ROOT}
nohup日志: ${NOHUP_LOG_DIR}
EOF

# ==================== 额外命令 ====================

# 如果有参数，执行相应命令
if [[ "$1" == "status" ]]; then
    check_status
    exit 0
fi

if [[ "$1" == "stop" ]]; then
    echo "停止所有任务..."
    for pid_file in "${NOHUP_LOG_DIR}"/part_*.pid; do
        if [[ -f "$pid_file" ]]; then
            pid=$(cat "$pid_file")
            if ps -p ${pid} > /dev/null 2>&1; then
                echo "停止 PID: ${pid}"
                kill ${pid}
            fi
        fi
    done
    echo "所有任务已停止"
    exit 0
fi
