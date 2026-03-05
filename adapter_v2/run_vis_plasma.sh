#!/bin/bash
# 
# Plasma 对齐可视化 - 快速启动示例
# 
# 用法：
#   1. 确保已训练模型并生成 checkpoint
#   2. 修改下面的 CKPT_PATH 为实际路径
#   3. 运行此脚本: bash run_vis_plasma.sh
#

set -e  # 遇到错误立即退出

# ============================================================================
# 配置参数
# ============================================================================

# 必需：训练好的 checkpoint 路径
CKPT_PATH="/home/ssddata/linshuijin/replicaLT/adapter_v2/runs/02.28_2125545/ckpt/best.pt"

# 可选：其他参数
CONFIG="config.yaml"
VAL_SPLIT_JSON="fixed_split.json"
PLASMA_KEY="pT217_F"
OUTPUT_DIR="vis_plasma_output"
SEED=42
BATCH_SIZE=32
NUM_WORKERS=4
UMAP_N_NEIGHBORS=15
UMAP_MIN_DIST=0.1
DPI=150
# 颜色归一化方式: rank(百分位排名,默认) | linear(线性) | percentile_clip(百分位截断)
COLOR_NORM="rank"

# ============================================================================
# 检查依赖
# ============================================================================

echo "Checking dependencies..."

# 检查 Python 环境
if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

# 检查 UMAP
if ! python -c "import umap" 2>/dev/null; then
    echo "Warning: umap-learn not found"
    echo "Installing umap-learn..."
    pip install umap-learn
fi

# 检查 checkpoint 是否存在
if [ ! -f "$CKPT_PATH" ]; then
    echo "Error: Checkpoint not found: $CKPT_PATH"
    echo "Please update CKPT_PATH in this script."
    exit 1
fi

# ============================================================================
# 运行可视化
# ============================================================================

echo ""
echo "========================================================================"
echo "Running Plasma Alignment Visualization"
echo "========================================================================"
echo "Checkpoint: $CKPT_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "========================================================================"
echo ""

python vis_plasma_alignment.py \
    --config "$CONFIG" \
    --ckpt "$CKPT_PATH" \
    --val_split_json "$VAL_SPLIT_JSON" \
    --plasma_key "$PLASMA_KEY" \
    --output_dir "$OUTPUT_DIR" \
    --seed $SEED \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --umap_n_neighbors $UMAP_N_NEIGHBORS \
    --umap_min_dist $UMAP_MIN_DIST \
    --dpi $DPI \
    --color_norm "$COLOR_NORM"

# ============================================================================
# 总结输出
# ============================================================================

echo ""
echo "========================================================================"
echo "Visualization completed!"
echo "========================================================================"
echo ""
echo "Output files:"
find "$OUTPUT_DIR" -name "1x3.png" -o -name "meta.json" | sort
echo ""
echo "Quick check Spearman rho:"
find "$OUTPUT_DIR" -name "meta.json" -exec jq -r '"\(.class_name) \(.split): ρ=\(.spearman_rho)"' {} \;
echo ""
echo "View images with:"
echo "  eog $OUTPUT_DIR/*/train/1x3.png  # Linux"
echo "  open $OUTPUT_DIR/*/train/1x3.png  # macOS"
echo ""
