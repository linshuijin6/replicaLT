#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/home/ssddata/linshuijin/replicaLT"
RUNS_DIR="${BASE_DIR}/baseline/runs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

GPU_ID="${CUDA_VISIBLE_DEVICES:-4}"

BASELINE_DIR="${RUNS_DIR}/exp_baseline_${TIMESTAMP}"
# BASELINE_CKPT="${BASELINE_DIR}/checkpoints/best_model.pth"
BASELINE_CKPT="/home/ssddata/linshuijin/replicaLT/baseline/runs/exp_baseline_20260207_142331/checkpoints/best_model.pth"

# ── 通用训练+评估函数 ──
run_exp () {
  local name="$1"
  local mode="$2"
  local pretrained="${3:-}"   # 可选: 预训练 backbone 路径
  local out_dir="${RUNS_DIR}/exp_${name}_${TIMESTAMP}"

  local extra_args=()
  if [[ -n "${pretrained}" ]]; then
    extra_args+=(--pretrained_backbone "${pretrained}")
  fi

  echo "[${name}] Training on GPU ${GPU_ID}..."
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m baseline.train \
    --output_dir "${out_dir}" \
    --condition_mode "${mode}" \
    --cuda_visible_devices "${GPU_ID}" \
    "${extra_args[@]+${extra_args[@]}}"

  echo "[${name}] Evaluating on GPU ${GPU_ID}..."
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m baseline.evaluate \
    "${out_dir}/checkpoints/best_model.pth" \
    --output_dir "${out_dir}" \
    --condition_mode "${mode}"
}

# ── 阶段 1: 训练纯 baseline (mode=none) ──
# run_exp "baseline" "none"

# ── 阶段 2: 用 baseline 预训练 backbone 初始化条件模型 ──
run_exp "clinical" "clinical"
run_exp "plasma"   "plasma"
run_exp "both"     "both"

# ── 阶段 2: 用 baseline 预训练 backbone 初始化条件模型 ──
# run_exp "clinical" "clinical" "${BASELINE_CKPT}"
# run_exp "plasma"   "plasma"   "${BASELINE_CKPT}"
# run_exp "both"     "both"     "${BASELINE_CKPT}"

echo "All experiments finished."