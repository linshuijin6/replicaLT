"""
TAU PET 厂商分布分析
分析 pairs 表格中的 TAU 扫描厂商分布，并统计 diagnosis 情况
"""

import pandas as pd
import numpy as np
from collections import Counter

# ========== 配置路径 ==========
PAIRS_CSV = '/home/ssddata/linshuijin/replicaLT/adapter_finetune/gen_csv/pairs_180d_dx_plasma_90d_matched_with_demog.csv'
PET_CSV = '/home/ssddata/linshuijin/replicaLT/analysis/3_PET_ADNI3_4_with_Plasma_PET_Images_04Feb2026.csv'

def main():
    print("=" * 70)
    print("TAU PET 厂商分布分析报告")
    print("=" * 70)
    
    # 读取数据
    print("\n[1] 读取数据...")
    pairs_df = pd.read_csv(PAIRS_CSV)
    pet_df = pd.read_csv(PET_CSV)
    
    print(f"  pairs 表格行数: {len(pairs_df)}")
    print(f"  3_PET 表格行数: {len(pet_df)}")
    
    # 筛选有 TAU ID 的记录
    pairs_with_tau = pairs_df[pairs_df['ID_AV1451'].notna()].copy()
    print(f"  pairs 中有 TAU ID 的记录: {len(pairs_with_tau)}")
    
    # 将 ID_AV1451 转换为整数（忽略格式差异）
    pairs_with_tau['ID_AV1451_int'] = pairs_with_tau['ID_AV1451'].astype(int)
    
    # 将 pet_df 的 image_id 也转换为整数
    pet_df['image_id_int'] = pet_df['image_id'].astype(int)
    
    # 筛选 AV1451 (TAU) 相关的 PET 扫描
    tau_pet = pet_df[pet_df['pet_radiopharm'].str.contains('AV1451', na=False, case=False)].copy()
    print(f"  3_PET 中 AV1451 (TAU) 扫描数: {len(tau_pet)}")
    
    # 进行匹配
    print("\n[2] 匹配 TAU 厂商信息...")
    merged = pairs_with_tau.merge(
        tau_pet[['image_id_int', 'pet_mfr', 'pet_mfr_model', 'pet_radiopharm']],
        left_on='ID_AV1451_int',
        right_on='image_id_int',
        how='left'
    )
    
    matched = merged[merged['pet_mfr'].notna()]
    unmatched = merged[merged['pet_mfr'].isna()]
    
    print(f"  成功匹配: {len(matched)}")
    print(f"  未匹配: {len(unmatched)}")
    
    # ========== 分析结果 ==========
    print("\n" + "=" * 70)
    print("分析结果")
    print("=" * 70)
    
    # 1. 厂商分布
    print("\n" + "-" * 50)
    print("【1】TAU PET 厂商分布")
    print("-" * 50)
    vendor_counts = matched['pet_mfr'].value_counts()
    total_matched = len(matched)
    
    print(f"\n{'厂商':<35} {'数量':>8} {'占比':>10}")
    print("-" * 55)
    for vendor, count in vendor_counts.items():
        pct = count / total_matched * 100
        print(f"{vendor:<35} {count:>8} {pct:>9.1f}%")
    print("-" * 55)
    print(f"{'总计':<35} {total_matched:>8} {'100.0%':>10}")
    
    # 2. 厂商-型号分布
    print("\n" + "-" * 50)
    print("【2】TAU PET 厂商-型号详细分布")
    print("-" * 50)
    vendor_model_counts = matched.groupby(['pet_mfr', 'pet_mfr_model']).size().reset_index(name='count')
    vendor_model_counts = vendor_model_counts.sort_values(['pet_mfr', 'count'], ascending=[True, False])
    
    current_vendor = None
    for _, row in vendor_model_counts.iterrows():
        if row['pet_mfr'] != current_vendor:
            current_vendor = row['pet_mfr']
            print(f"\n  {current_vendor}:")
        pct = row['count'] / total_matched * 100
        print(f"    - {row['pet_mfr_model']}: {row['count']} ({pct:.1f}%)")
    
    # 3. Diagnosis 整体分布
    print("\n" + "-" * 50)
    print("【3】Diagnosis 整体分布")
    print("-" * 50)
    diagnosis_counts = matched['diagnosis'].value_counts()
    
    print(f"\n{'诊断':<15} {'数量':>8} {'占比':>10}")
    print("-" * 35)
    for diag, count in diagnosis_counts.items():
        pct = count / total_matched * 100
        print(f"{diag:<15} {count:>8} {pct:>9.1f}%")
    print("-" * 35)
    print(f"{'总计':<15} {total_matched:>8} {'100.0%':>10}")
    
    # 4. 按厂商分组的 Diagnosis 分布
    print("\n" + "-" * 50)
    print("【4】按厂商分组的 Diagnosis 分布")
    print("-" * 50)
    
    # 创建交叉表
    cross_tab = pd.crosstab(matched['pet_mfr'], matched['diagnosis'], margins=True, margins_name='总计')
    
    # 按行计算百分比
    cross_tab_pct = pd.crosstab(matched['pet_mfr'], matched['diagnosis'], normalize='index') * 100
    
    print("\n绝对数量:")
    print(cross_tab.to_string())
    
    print("\n按厂商的百分比分布:")
    print(cross_tab_pct.round(1).to_string())
    
    # 5. 详细的厂商-诊断统计
    print("\n" + "-" * 50)
    print("【5】厂商-诊断详细统计")
    print("-" * 50)
    
    for vendor in vendor_counts.index:
        vendor_data = matched[matched['pet_mfr'] == vendor]
        vendor_total = len(vendor_data)
        vendor_diag = vendor_data['diagnosis'].value_counts()
        
        print(f"\n  {vendor} (n={vendor_total}):")
        for diag, count in vendor_diag.items():
            pct = count / vendor_total * 100
            print(f"    - {diag}: {count} ({pct:.1f}%)")
    
    # 6. 摘要统计
    print("\n" + "=" * 70)
    print("摘要统计")
    print("=" * 70)
    
    print(f"""
总样本数: {len(pairs_with_tau)}
成功匹配厂商信息: {len(matched)} ({len(matched)/len(pairs_with_tau)*100:.1f}%)
未匹配: {len(unmatched)} ({len(unmatched)/len(pairs_with_tau)*100:.1f}%)

厂商数量: {len(vendor_counts)}
主要厂商:
  - {vendor_counts.index[0]}: {vendor_counts.iloc[0]} ({vendor_counts.iloc[0]/total_matched*100:.1f}%)
  - {vendor_counts.index[1]}: {vendor_counts.iloc[1]} ({vendor_counts.iloc[1]/total_matched*100:.1f}%)
  - {vendor_counts.index[2] if len(vendor_counts) > 2 else 'N/A'}: {vendor_counts.iloc[2] if len(vendor_counts) > 2 else 0} ({vendor_counts.iloc[2]/total_matched*100:.1f}% if len(vendor_counts) > 2 else 0)

诊断分布:
""")
    for diag, count in diagnosis_counts.items():
        print(f"  - {diag}: {count} ({count/total_matched*100:.1f}%)")
    
    # 保存详细结果
    output_csv = '/home/ssddata/linshuijin/replicaLT/analysis/tau_vendor_diagnosis_analysis.csv'
    matched[['PTID', 'EXAMDATE', 'ID_AV1451', 'diagnosis', 'pet_mfr', 'pet_mfr_model']].to_csv(output_csv, index=False)
    print(f"\n详细结果已保存至: {output_csv}")
    
    print("\n" + "=" * 70)
    print("分析完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
