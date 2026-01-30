#!/bin/bash
# 
# MRI-PET 配对脚本使用示例
# 

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 激活 conda 环境
echo "激活 conda 环境 xc..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate xc

# 设置路径
MRI_FILE="/home/ssddata/linshuijin/replicaLT/adapter_finetune/data_csv/MRI0114.me.csv"
PET_FILE="/home/ssddata/linshuijin/replicaLT/adapter_finetune/data_csv/PET0114.me.csv"
DXSUM_FILE="/home/ssddata/linshuijin/replicaLT/adapter_finetune/DXSUM_25Dec2025.csv"

echo "========================================================================"
echo "MRI-PET 配对脚本"
echo "========================================================================"
echo ""

# 示例1: 180天窗口（不检查诊断）
echo "示例1: 使用 180 天时间窗口（不检查诊断）"
python adapter_finetune/match_mri_pet_pairs.py \
    --mri "$MRI_FILE" \
    --pet "$PET_FILE" \
    --max-days 180 \
    --output adapter_finetune/pairs_180d.csv

echo ""
echo "------------------------------------------------------------------------"
echo ""

# 示例2: 90天窗口（不检查诊断）
echo "示例2: 使用 90 天时间窗口（不检查诊断）"
python adapter_finetune/match_mri_pet_pairs.py \
    --mri "$MRI_FILE" \
    --pet "$PET_FILE" \
    --max-days 90 \
    --output adapter_finetune/pairs_90d.csv

echo ""
echo "------------------------------------------------------------------------"
echo ""

# 示例3: 365天窗口（不检查诊断）
echo "示例3: 使用 365 天时间窗口（不检查诊断）"
python adapter_finetune/match_mri_pet_pairs.py \
    --mri "$MRI_FILE" \
    --pet "$PET_FILE" \
    --max-days 365 \
    --output adapter_finetune/pairs_365d.csv

echo ""
echo "------------------------------------------------------------------------"
echo ""

# 示例4: 90天窗口 + 诊断检查
echo "示例4: 使用 90 天时间窗口 + 诊断一致性检查"
python adapter_finetune/match_mri_pet_pairs.py \
    --mri "$MRI_FILE" \
    --pet "$PET_FILE" \
    --max-days 90 \
    --check-diagnosis \
    --dxsum "$DXSUM_FILE" \
    --output adapter_finetune/pairs_90d_dx.csv

echo ""
echo "------------------------------------------------------------------------"
echo ""

# 示例5: 180天窗口 + 诊断检查
echo "示例5: 使用 180 天时间窗口 + 诊断一致性检查"
python adapter_finetune/match_mri_pet_pairs.py \
    --mri "$MRI_FILE" \
    --pet "$PET_FILE" \
    --max-days 180 \
    --check-diagnosis \
    --dxsum "$DXSUM_FILE" \
    --output adapter_finetune/pairs_180d_dx.csv

echo ""
echo "========================================================================"
echo "完成！生成的文件："
echo "  不检查诊断："
echo "    - adapter_finetune/pairs_180d.csv"
echo "    - adapter_finetune/pairs_90d.csv"
echo "    - adapter_finetune/pairs_365d.csv"
echo "  检查诊断一致性："
echo "    - adapter_finetune/pairs_90d_dx.csv"
echo "    - adapter_finetune/pairs_180d_dx.csv"
echo "========================================================================"
