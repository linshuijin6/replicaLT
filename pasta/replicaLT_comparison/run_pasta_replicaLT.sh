#!/bin/bash
# run_pasta_replicaLT.sh
# ========================
# 在 xiaochou 环境中运行 PASTA 训练（replicaLT 对比实验）
#
# 用法:
#   cd /mnt/nfsdata/nfsdata/lsj.14/PASTA
#   bash replicaLT_comparison/run_pasta_replicaLT.sh          # 默认使用 GPU 0
#   bash replicaLT_comparison/run_pasta_replicaLT.sh 2        # 使用 GPU 2
#   bash replicaLT_comparison/run_pasta_replicaLT.sh 0,1      # 使用 GPU 0 和 1
#
# 步骤:
#   1. 数据转换（NIfTI → HDF5）
#   2. 训练
#   3. 评估（synthesis 模式生成合成 PET）

set -e

# ============ CUDA 设备设置 ============
# 第一个位置参数为 CUDA 编号，默认为 0
CUDA_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="$CUDA_ID"
echo "🖥️  使用 GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

PASTA_ROOT="/home/ssddata/linshuijin/PASTA"
CONFIG="replicaLT_comparison/pasta_replicaLT.yaml"
DATA_DIR="data"
RESULTS_DIR="replicaLT_comparison/results"

cd "$PASTA_ROOT"

# ============ 1. 数据转换 ============
if [ ! -f "${DATA_DIR}/train.h5" ]; then
    echo "========================================"
    echo "[Step 1] 转换 NIfTI → HDF5..."
    echo "========================================"
    conda run --no-capture-output -n xiaochou python replicaLT_comparison/convert_nifti_to_h5.py
else
    echo "✅ HDF5 数据已存在，跳过转换"
fi

# ============ 2. 训练 ============
DATE_PREFIX=$(date +"%Y-%m-%d")
export RESULTS_DIR="replicaLT_comparison/results/${DATE_PREFIX}_$$"

echo "========================================"
echo "[Step 2] 开始 PASTA 训练..."
echo "========================================"
mkdir -p "$RESULTS_DIR"
cp "$CONFIG" "$RESULTS_DIR/"
conda run --no-capture-output -n xiaochou python replicaLT_comparison/train_pasta_replicaLT.py 2>&1 | tee "${RESULTS_DIR}/train.log"

# Python renames the folder to use its own PID; find the actual folder
ACTUAL_DIR=$(ls -dt ${RESULTS_DIR%_*}_* 2>/dev/null | head -1)
echo "========================================"
echo "✅ 训练完成！结果保存在: ${ACTUAL_DIR:-$RESULTS_DIR}"
echo "========================================"
