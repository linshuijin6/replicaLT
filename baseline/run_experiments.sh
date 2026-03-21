#!/usr/bin/env bash
set -euo pipefail

# 基于脚本位置自动定位仓库根目录，避免因执行目录不同导致模块导入失败
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNS_DIR="${BASE_DIR}/baseline/runs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# GPU_ID="${CUDA_VISIBLE_DEVICES:-4}"
GPU_ID="4"

# 目标 PET 列表（空格分隔），可在命令行覆盖：
# PET_TARGETS="tau fdg av45" bash baseline/run_experiments.sh
PET_TARGETS_STR="${PET_TARGETS:-tau fdg av45}"
read -r -a PET_TARGETS_ARR <<< "${PET_TARGETS_STR}"

# 条件模式列表（空格分隔），可覆盖：
# CONDITION_MODES="none clinical plasma both" bash baseline/run_experiments.sh
CONDITION_MODES_STR="${CONDITION_MODES:-none clinical plasma both}"
read -r -a CONDITION_MODES_ARR <<< "${CONDITION_MODES_STR}"

# ── 训练超参数（均可通过命令行环境变量覆盖）──
# 用法示例: EPOCHS=100 BATCH_SIZE=2 LEARNING_RATE=5e-5 bash baseline/run_experiments.sh
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
EARLY_STOPPING="${EARLY_STOPPING:-30}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-4}"
VAL_FREQ="${VAL_FREQ:-5}"

# 复用核心方法现有划分（优先 train/val JSON，fixed_split 作为回退）
# SPLIT_SUBJECTS_JSON="${SPLIT_SUBJECTS_JSON:-${BASE_DIR}/fixed_split.json}"
SPLIT_TRAIN_JSON="${SPLIT_TRAIN_JSON:-${BASE_DIR}/train_data_with_description.json}"
SPLIT_VAL_JSON="${SPLIT_VAL_JSON:-${BASE_DIR}/val_data_with_description.json}"

# echo ${SPLIT_TRAIN_JSON}

# 强制在仓库根目录执行，保证 `python -m baseline.*` 可用
cd "${BASE_DIR}"

# ── 通用训练+评估函数 ──
run_exp () {
  local pet="$1"
  local mode="$2"
  local pretrained="${3:-}"   # 可选: 预训练 backbone 路径
  local out_dir="${RUNS_DIR}/exp_${pet}_${mode}_${TIMESTAMP}"

  local extra_args=()
  if [[ -n "${pretrained}" ]]; then
    extra_args+=(--pretrained_backbone "${pretrained}")
  fi

  echo "[${pet}/${mode}] Training on GPU ${GPU_ID}..."
  echo "  Hyperparams: epochs=${EPOCHS} bs=${BATCH_SIZE} lr=${LEARNING_RATE} early_stop=${EARLY_STOPPING} accum=${ACCUMULATION_STEPS} val_freq=${VAL_FREQ}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m baseline.train \
    --output_dir "${out_dir}" \
    --target_pet "${pet}" \
    --condition_mode "${mode}" \
    --cuda_visible_devices "${GPU_ID}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LEARNING_RATE}" \
    --early_stopping "${EARLY_STOPPING}" \
    --accumulation_steps "${ACCUMULATION_STEPS}" \
    --val_freq "${VAL_FREQ}" \
    --split_train_json "${SPLIT_TRAIN_JSON}" \
    --split_val_json "${SPLIT_VAL_JSON}" \
    --split_fallback_test_from_val \
    "${extra_args[@]+${extra_args[@]}}"

  echo "[${pet}/${mode}] Evaluating on GPU ${GPU_ID}..."
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m baseline.evaluate \
    "${out_dir}/checkpoints/best_model.pth" \
    --output_dir "${out_dir}" \
    --target_pet "${pet}" \
    --condition_mode "${mode}" \
    --cuda_visible_devices "${GPU_ID}" \
    --split_train_json "${SPLIT_TRAIN_JSON}" \
    --split_val_json "${SPLIT_VAL_JSON}" \
    --split_fallback_test_from_val
}

echo "PET targets: ${PET_TARGETS_STR}"
echo "Condition modes: ${CONDITION_MODES_STR}"
for pet in "${PET_TARGETS_ARR[@]}"; do
  for mode in "${CONDITION_MODES_ARR[@]}"; do
    run_exp "${pet}" "${mode}"
  done
done

echo "All experiments finished."
