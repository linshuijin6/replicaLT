#!/bin/bash
# 
# MRI-PET 配对脚本（启用诊断一致性检查）
# 使用 180 天 MRI-PET 匹配窗口 + 180 天诊断查找窗口
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 激活 conda 环境
echo "激活 conda 环境 xc..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate xc

# 设置路径
MRI_FILE="/home/ssddata/linshuijin/replicaLT/adapter_finetune/ADNI_csv/MRI0114.me.csv"
PET_FILE="/home/ssddata/linshuijin/replicaLT/adapter_finetune/ADNI_csv/PET0114.me.csv"
DXSUM_FILE="/home/ssddata/linshuijin/replicaLT/adapter_finetune/ADNI_csv/DXSUM_25Dec2025.csv"

echo "========================================================================"
echo "MRI-PET 配对（启用诊断一致性检查）"
echo "========================================================================"
echo ""

# 运行配对脚本
echo "参数: max-days=180, diagnosis-max-days=180, check-diagnosis=True"
python match_mri_pet_pairs.py \
    --mri "$MRI_FILE" \
    --pet "$PET_FILE" \
    --max-days 180 \
    --diagnosis-max-days 180 \
    --check-diagnosis \
    --dxsum "$DXSUM_FILE" \
    --output gen_csv/pairs_180d_dx.csv

echo ""
echo "------------------------------------------------------------------------"
echo ""

# 生成详细报告
echo "生成详细报告..."
python - <<'PYCODE'
import pandas as pd

df = pd.read_csv('gen_csv/pairs_180d_dx.csv')

print("\n" + "=" * 70)
print("详细统计报告")
print("=" * 70)
print(f"\n总记录数: {len(df)}")

# PET 匹配统计
print("\n--- PET 匹配统计 ---")
print(f"  FDG:    {df['id_fdg'].notna().sum():5d} ({100*df['id_fdg'].notna().sum()/len(df):.1f}%)")
print(f"  AV45:   {df['id_av45'].notna().sum():5d} ({100*df['id_av45'].notna().sum()/len(df):.1f}%)")
print(f"  AV1451: {df['id_av1451'].notna().sum():5d} ({100*df['id_av1451'].notna().sum()/len(df):.1f}%)")

has_any_pet = df[['id_fdg', 'id_av45', 'id_av1451']].notna().any(axis=1).sum()
print(f"  至少一个 PET: {has_any_pet:5d} ({100*has_any_pet/len(df):.1f}%)")
print(f"  无 PET 匹配:  {len(df) - has_any_pet:5d} ({100*(len(df)-has_any_pet)/len(df):.1f}%)")

# DIAGNOSIS 统计
print("\n--- DIAGNOSIS 统计 ---")
diagnosis_found = df['DIAGNOSIS'].notna().sum()
diagnosis_missing = df['DIAGNOSIS'].isna().sum()
print(f"  有诊断信息:   {diagnosis_found:5d} ({100*diagnosis_found/len(df):.1f}%)")
print(f"  缺失诊断信息: {diagnosis_missing:5d} ({100*diagnosis_missing/len(df):.1f}%)")

# 按诊断类别统计
print("\n--- 按诊断类别统计 ---")
dx_counts = df['DIAGNOSIS'].value_counts().sort_index()
dx_labels = {1: 'CN (正常)', 2: 'MCI (轻度认知障碍)', 3: 'AD (阿尔茨海默病)'}
for dx, count in dx_counts.items():
    label = dx_labels.get(int(dx), f'未知({int(dx)})')
    print(f"  {label}: {count:5d} ({100*count/len(df):.1f}%)")

# 显示缺失诊断的前10条记录
if diagnosis_missing > 0:
    print(f"\n--- 缺失诊断的记录示例（前10条）---")
    missing_dx = df[df['DIAGNOSIS'].isna()][['PTID', 'EXAMDATE', 'id_mri']].head(10)
    print(missing_dx.to_string(index=False))

print("\n" + "=" * 70)
PYCODE

echo ""
echo "========================================================================"
echo "完成！生成的文件: gen_csv/pairs_180d_dx.csv"
echo "========================================================================"
