#!/bin/bash
# ==============================================================================
# ADNI MRI-PET 完整数据处理流程
# ==============================================================================
# 流程概述:
#   1. Match: 使用时间窗口匹配 MRI 和 PET 数据
#   2. Diagnosis: 添加诊断信息 (DXSUM)
#   3. Plasma: 扩展血浆生物标志物信息 (UPENN & C2N)
#
# 输入数据 (ADNI_csv/):
#   - MRI0114.me.csv          : MRI 扫描记录
#   - PET0114.me.csv          : PET 扫描记录
#   - DXSUM_25Dec2025.csv     : 诊断信息
#   - UPENN_PLASMA_*.csv      : UPENN 血浆数据
#   - C2N_PRECIVITYAD2_*.csv  : C2N 血浆数据
#
# 输出数据 (gen_csv/):
#   - pairs_{MAX_DAYS}d.csv                    : MRI-PET 配对结果
#   - pairs_{MAX_DAYS}d_dx.csv                 : 添加诊断后的配对
#   - pairs_{MAX_DAYS}d_dx_plasma_{PLASMA_DAYS}d.csv : 添加血浆信息后的最终数据
#   - pipeline_YYYYMMDD_HHMMSS.log             : 处理日志
# ==============================================================================

set -e

# ------------------------------------------------------------------------------
# 配置参数（可根据需要修改）
# ------------------------------------------------------------------------------
# MRI-PET 匹配时间窗口（天）
MAX_DAYS=180

# 诊断查找时间窗口（天）
DX_MAX_DAYS=180

# 血浆数据匹配时间窗口（天）
PLASMA_MAX_DAYS=90

# MRI phases 筛选
MRI_PHASES="ADNI3 ADNI4"

# ------------------------------------------------------------------------------
# 路径配置
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 输入目录
ADNI_CSV_DIR="$SCRIPT_DIR/ADNI_csv"

# 输出目录
GEN_CSV_DIR="$SCRIPT_DIR/gen_csv"

# 输入文件
MRI_FILE="$ADNI_CSV_DIR/MRI0114.me.csv"
PET_FILE="$ADNI_CSV_DIR/PET0114.me.csv"
DXSUM_FILE="$ADNI_CSV_DIR/DXSUM_25Dec2025.csv"
UPENN_FILE="$ADNI_CSV_DIR/UPENN_PLASMA_FUJIREBIO_QUANTERIX_21Dec2025.csv"
C2N_FILE="$ADNI_CSV_DIR/C2N_PRECIVITYAD2_PLASMA_29Dec2025.csv"

# 输出文件名（根据参数动态生成）
PAIRS_FILE="$GEN_CSV_DIR/pairs_${MAX_DAYS}d.csv"
PAIRS_DX_FILE="$GEN_CSV_DIR/pairs_${MAX_DAYS}d_dx.csv"
PAIRS_FINAL_FILE="$GEN_CSV_DIR/pairs_${MAX_DAYS}d_dx_plasma_${PLASMA_MAX_DAYS}d.csv"

# 日志文件（使用时间戳命名）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$GEN_CSV_DIR/pipeline_${TIMESTAMP}.log"

# ------------------------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------------------------

# 日志函数：同时输出到控制台和日志文件
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg" | tee -a "$LOG_FILE"
}

log_section() {
    echo "" | tee -a "$LOG_FILE"
    echo "========================================================================" | tee -a "$LOG_FILE"
    log "$1"
    echo "========================================================================" | tee -a "$LOG_FILE"
}

log_subsection() {
    echo "" | tee -a "$LOG_FILE"
    echo "------------------------------------------------------------------------" | tee -a "$LOG_FILE"
    log "$1"
    echo "------------------------------------------------------------------------" | tee -a "$LOG_FILE"
}

check_file() {
    if [ ! -f "$1" ]; then
        log "错误: 文件不存在 - $1"
        exit 1
    fi
    log "  ✓ 文件存在: $(basename "$1")"
}

# ------------------------------------------------------------------------------
# 初始化
# ------------------------------------------------------------------------------

# 确保输出目录存在
mkdir -p "$GEN_CSV_DIR"

# 初始化日志文件
echo "" > "$LOG_FILE"
log_section "ADNI MRI-PET 数据处理流程 开始"
log "日志文件: $LOG_FILE"
log "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

# 记录参数配置
log_subsection "参数配置"
log "  MRI-PET 匹配窗口:    ${MAX_DAYS} 天"
log "  诊断查找窗口:        ${DX_MAX_DAYS} 天"
log "  血浆匹配窗口:        ${PLASMA_MAX_DAYS} 天"
log "  MRI Phases:          ${MRI_PHASES}"

# 激活 conda 环境
log_subsection "环境准备"
log "激活 conda 环境 xc..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate xc
log "  ✓ conda 环境已激活: $(conda info --envs | grep '*' | awk '{print $1}')"

# 检查输入文件
log_subsection "检查输入文件"
check_file "$MRI_FILE"
check_file "$PET_FILE"
check_file "$DXSUM_FILE"
check_file "$UPENN_FILE"
check_file "$C2N_FILE"

# ------------------------------------------------------------------------------
# 步骤 1: MRI-PET 配对
# ------------------------------------------------------------------------------
log_section "步骤 1/3: MRI-PET 配对"
log "参数: max-days=${MAX_DAYS}, mri-phases=${MRI_PHASES}"
log "输出: $PAIRS_FILE"

python match_mri_pet_pairs.py \
    --mri "$MRI_FILE" \
    --pet "$PET_FILE" \
    --max-days "$MAX_DAYS" \
    --mri-phases $MRI_PHASES \
    --output "$PAIRS_FILE" 2>&1 | tee -a "$LOG_FILE"

if [ -f "$PAIRS_FILE" ]; then
    PAIRS_COUNT=$(wc -l < "$PAIRS_FILE")
    PAIRS_COUNT=$((PAIRS_COUNT - 1))  # 减去表头
    log "  ✓ 配对完成: 共 $PAIRS_COUNT 条记录"
else
    log "  ✗ 配对失败: 输出文件未生成"
    exit 1
fi

# ------------------------------------------------------------------------------
# 步骤 2: 添加诊断信息
# ------------------------------------------------------------------------------
log_section "步骤 2/3: 添加诊断信息 (DXSUM)"
log "参数: max-days=${MAX_DAYS}, diagnosis-max-days=${DX_MAX_DAYS}, check-diagnosis=True"
log "输出: $PAIRS_DX_FILE"

python match_mri_pet_pairs.py \
    --mri "$MRI_FILE" \
    --pet "$PET_FILE" \
    --max-days "$MAX_DAYS" \
    --mri-phases $MRI_PHASES \
    --check-diagnosis \
    --dxsum "$DXSUM_FILE" \
    --diagnosis-max-days "$DX_MAX_DAYS" \
    --output "$PAIRS_DX_FILE" 2>&1 | tee -a "$LOG_FILE"

if [ -f "$PAIRS_DX_FILE" ]; then
    DX_COUNT=$(wc -l < "$PAIRS_DX_FILE")
    DX_COUNT=$((DX_COUNT - 1))
    log "  ✓ 诊断添加完成: 共 $DX_COUNT 条记录"
else
    log "  ✗ 诊断添加失败: 输出文件未生成"
    exit 1
fi

# 生成诊断统计报告
log_subsection "诊断统计报告"
python - <<PYCODE 2>&1 | tee -a "$LOG_FILE"
import pandas as pd

df = pd.read_csv('$PAIRS_DX_FILE')

print(f"总记录数: {len(df)}")

# PET 匹配统计
print("\n--- PET 匹配统计 ---")
for pet_type in ['id_fdg', 'id_av45', 'id_av1451']:
    count = df[pet_type].notna().sum()
    pct = 100 * count / len(df) if len(df) > 0 else 0
    print(f"  {pet_type:12s}: {count:5d} ({pct:.1f}%)")

has_any_pet = df[['id_fdg', 'id_av45', 'id_av1451']].notna().any(axis=1).sum()
print(f"  至少一个 PET: {has_any_pet:5d} ({100*has_any_pet/len(df):.1f}%)")

# DIAGNOSIS 统计
print("\n--- DIAGNOSIS 统计 ---")
dx_found = df['DIAGNOSIS'].notna().sum()
dx_missing = df['DIAGNOSIS'].isna().sum()
print(f"  有诊断信息:   {dx_found:5d} ({100*dx_found/len(df):.1f}%)")
print(f"  缺失诊断信息: {dx_missing:5d} ({100*dx_missing/len(df):.1f}%)")

# 按诊断类别统计
print("\n--- 按诊断类别分布 ---")
dx_labels = {1: 'CN (正常)', 2: 'MCI (轻度认知障碍)', 3: 'AD (阿尔茨海默病)'}
dx_counts = df['DIAGNOSIS'].value_counts().sort_index()
for dx, count in dx_counts.items():
    label = dx_labels.get(int(dx), f'未知({int(dx)})')
    print(f"  {label}: {count:5d} ({100*count/len(df):.1f}%)")
PYCODE

# ------------------------------------------------------------------------------
# 步骤 3: 添加血浆信息
# ------------------------------------------------------------------------------
log_section "步骤 3/3: 添加血浆生物标志物信息"
log "参数: max-days=${PLASMA_MAX_DAYS}"
log "数据源: UPENN + C2N"
log "输出: $PAIRS_FINAL_FILE"

python extend_pairs_with_plasma.py \
    --pairs "$PAIRS_DX_FILE" \
    --upenn "$UPENN_FILE" \
    --c2n "$C2N_FILE" \
    --output "$PAIRS_FINAL_FILE" \
    --max-days "$PLASMA_MAX_DAYS" 2>&1 | tee -a "$LOG_FILE"

if [ -f "$PAIRS_FINAL_FILE" ]; then
    FINAL_COUNT=$(wc -l < "$PAIRS_FINAL_FILE")
    FINAL_COUNT=$((FINAL_COUNT - 1))
    log "  ✓ 血浆信息添加完成: 共 $FINAL_COUNT 条记录"
else
    log "  ✗ 血浆信息添加失败: 输出文件未生成"
    exit 1
fi

# ------------------------------------------------------------------------------
# 最终统计报告
# ------------------------------------------------------------------------------
log_section "最终统计报告"

python - <<PYCODE 2>&1 | tee -a "$LOG_FILE"
import pandas as pd
import numpy as np

df = pd.read_csv('$PAIRS_FINAL_FILE')

print(f"=== 最终数据集概览 ===")
print(f"总记录数: {len(df)}")
print(f"唯一受试者数: {df['PTID'].nunique()}")
print(f"列数: {len(df.columns)}")
print(f"列名: {', '.join(df.columns.tolist())}")

# PET 统计
print("\n--- PET 扫描统计 ---")
pet_cols = ['id_fdg', 'id_av45', 'id_av1451']
for col in pet_cols:
    if col in df.columns:
        count = df[col].notna().sum()
        print(f"  {col:12s}: {count:5d}")

# 诊断统计
print("\n--- 诊断分布 ---")
if 'DIAGNOSIS' in df.columns:
    dx_labels = {1: 'CN', 2: 'MCI', 3: 'AD'}
    for dx, label in dx_labels.items():
        count = (df['DIAGNOSIS'] == dx).sum()
        pct = 100 * count / len(df) if len(df) > 0 else 0
        print(f"  {label:4s}: {count:5d} ({pct:.1f}%)")
    missing = df['DIAGNOSIS'].isna().sum()
    print(f"  未知: {missing:5d} ({100*missing/len(df):.1f}%)")

# 血浆数据统计
print("\n--- 血浆数据统计 ---")
# UPENN 数据列
upenn_cols = [c for c in df.columns if 'UPENN' in c.upper() or c.startswith('upenn_')]
if upenn_cols:
    print(f"  UPENN 列数: {len(upenn_cols)}")
    # 检查是否有匹配
    upenn_match_col = [c for c in df.columns if 'upenn' in c.lower() and 'match' in c.lower()]
    if upenn_match_col:
        upenn_matched = df[upenn_match_col[0]].notna().sum()
        print(f"  UPENN 匹配记录: {upenn_matched}")

# C2N 数据列
c2n_cols = [c for c in df.columns if 'C2N' in c.upper() or c.startswith('c2n_')]
if c2n_cols:
    print(f"  C2N 列数: {len(c2n_cols)}")
    c2n_match_col = [c for c in df.columns if 'c2n' in c.lower() and 'match' in c.lower()]
    if c2n_match_col:
        c2n_matched = df[c2n_match_col[0]].notna().sum()
        print(f"  C2N 匹配记录: {c2n_matched}")

# 检查常见血浆列
plasma_indicator_cols = ['has_upenn', 'has_c2n', 'upenn_examdate', 'c2n_examdate']
for col in plasma_indicator_cols:
    if col in df.columns:
        count = df[col].notna().sum() if df[col].dtype != bool else df[col].sum()
        print(f"  {col}: {count}")

print("\n=== 数据质量检查 ===")
# 检查缺失值
missing_pct = df.isnull().sum() / len(df) * 100
high_missing = missing_pct[missing_pct > 50].sort_values(ascending=False)
if len(high_missing) > 0:
    print(f"缺失率 >50% 的列: {len(high_missing)} 个")
    for col, pct in high_missing.head(5).items():
        print(f"  {col}: {pct:.1f}%")
else:
    print("所有列缺失率均 <= 50%")
PYCODE

# ------------------------------------------------------------------------------
# 完成
# ------------------------------------------------------------------------------
log_section "处理完成"
log "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
log ""
log "生成的文件:"
log "  1. $PAIRS_FILE"
log "  2. $PAIRS_DX_FILE"
log "  3. $PAIRS_FINAL_FILE"
log ""
log "日志文件: $LOG_FILE"

echo ""
echo "========================================================================"
echo "  ADNI 数据处理流程完成！"
echo "  最终输出: $PAIRS_FINAL_FILE"
echo "  日志文件: $LOG_FILE"
echo "========================================================================"
