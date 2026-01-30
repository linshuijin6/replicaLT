#!/bin/bash
# 多线程运行 pipeline_reuse_mri_pet.py（仅CPU模式）
# 用法: bash run_pipeline_multiproc.sh [线程数，默认8]

set -e

# ============ 配置 ============
NUM_WORKERS=${1:-8}                   # 线程数，默认8
CSV_IN="/mnt/nfsdata/nfsdata/lsj.14/replicaLT/adapter_finetune/data_csv/pairs_180d_dx_loren.csv"
SPLIT_DIR="/mnt/nfsdata/nfsdata/lsj.14/replicaLT/adapter_finetune/data_csv/splits"
LOG_BASE="/mnt/nfsdata/nfsdata/lsj.14/ADNI_CSF/logs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_SCRIPT="${SCRIPT_DIR}/pipeline_reuse_mri_pet.py"

# Conda 环境
CONDA_ENV="xiaochou"
CONDA_BASE="/mnt/ssd/linshuijin/miniconda3"

# ============ 准备工作 ============
echo "[INFO] 线程数: ${NUM_WORKERS}"
echo "[INFO] 源CSV: ${CSV_IN}"
echo "[INFO] 分割目录: ${SPLIT_DIR}"
echo "[INFO] 日志目录: ${LOG_BASE}"

mkdir -p "${SPLIT_DIR}"
mkdir -p "${LOG_BASE}"

# 提取表头
HEADER=$(head -1 "${CSV_IN}")
TOTAL_LINES=$(tail -n +2 "${CSV_IN}" | wc -l)
LINES_PER_WORKER=$(( (TOTAL_LINES + NUM_WORKERS - 1) / NUM_WORKERS ))

echo "[INFO] 总数据行: ${TOTAL_LINES}, 每线程约: ${LINES_PER_WORKER}"

# 分割CSV（跳过表头）
tail -n +2 "${CSV_IN}" | split -l ${LINES_PER_WORKER} -d -a 2 - "${SPLIT_DIR}/part_"

# 给每个分割文件添加表头
for part in "${SPLIT_DIR}"/part_*; do
    tmpf="${part}.tmp"
    echo "${HEADER}" > "${tmpf}"
    cat "${part}" >> "${tmpf}"
    mv "${tmpf}" "${part}.csv"
    rm -f "${part}"
done

# 列出分割文件
PARTS=($(ls "${SPLIT_DIR}"/part_*.csv | sort))
echo "[INFO] 分割完成，共 ${#PARTS[@]} 个文件:"
for p in "${PARTS[@]}"; do
    echo "       - ${p} ($(wc -l < "${p}") 行)"
done

# ============ 启动多进程 ============
echo ""
echo "[INFO] 启动 ${#PARTS[@]} 个后台进程..."
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

for i in "${!PARTS[@]}"; do
    PART_CSV="${PARTS[$i]}"
    PART_NAME=$(basename "${PART_CSV}" .csv)
    NOHUP_LOG="${LOG_BASE}/nohup_${PART_NAME}_${TIMESTAMP}.log"
    LOGS_SUB="${LOG_BASE}/${PART_NAME}"
    mkdir -p "${LOGS_SUB}"

    CMD="source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && python ${PIPELINE_SCRIPT} \
        --pairs_csv ${PART_CSV} \
        --nifti_root /mnt/nfsdata/nfsdata/ADNI/ADNI0103/NIFTI \
        --target_root /mnt/nfsdata/nfsdata/lsj.14/ADNI_CSF \
        --logs_root ${LOGS_SUB} \
        --tmp_root /mnt/nfsdata/nfsdata/lsj.14/ADNI_CSF/tmp_${PART_NAME} \
        --n4_threads 2"

    echo "[${i}] nohup bash -c '${CMD}' > ${NOHUP_LOG} 2>&1 &"
    nohup bash -c "${CMD}" > "${NOHUP_LOG}" 2>&1 &
    
    echo "    PID: $!"
done

echo ""
echo "[DONE] 所有进程已在后台启动。"
echo "       查看日志: tail -f ${LOG_BASE}/nohup_part_*.log"
echo "       查看进程: ps aux | grep pipeline_reuse_mri_pet"
