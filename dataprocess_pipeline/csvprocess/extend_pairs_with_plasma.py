#!/usr/bin/env python3
"""
扩展 pairs_90d.csv 文件，添加血浆信息
通过 PTID 和 EXAMDATE 匹配 UPENN 和 C2N 数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import os


def parse_date(date_str):
    """解析日期字符串"""
    try:
        return pd.to_datetime(date_str)
    except:
        return pd.NaT


def match_plasma_data(pairs_df, plasma_df, source_name, max_days=30):
    """
    匹配血浆数据到 pairs 数据
    
    参数:
        pairs_df: pairs 数据框
        plasma_df: 血浆数据框 (UPENN 或 C2N)
        source_name: 数据源名称 ('UPENN' 或 'C2N')
        max_days: 最大允许的日期差异（天）
    
    返回:
        匹配结果的数据框
    """
    print(f"\n匹配 {source_name} 数据...")
    print(f"  - pairs 记录数: {len(pairs_df)}")
    print(f"  - {source_name} 记录数: {len(plasma_df)}")
    print(f"  - 最大日期差异: {max_days} 天")
    
    # 转换日期列
    pairs_df['EXAMDATE_dt'] = pairs_df['EXAMDATE'].apply(parse_date)
    plasma_df['EXAMDATE_dt'] = plasma_df['EXAMDATE'].apply(parse_date)
    
    # 删除日期解析失败的行
    pairs_df = pairs_df[pairs_df['EXAMDATE_dt'].notna()].copy()
    plasma_df = plasma_df[plasma_df['EXAMDATE_dt'].notna()].copy()
    
    print(f"  - 有效日期的 pairs 记录: {len(pairs_df)}")
    print(f"  - 有效日期的 {source_name} 记录: {len(plasma_df)}")
    
    # 结果列表
    matches = []
    
    # 对每个 PTID 进行匹配
    for ptid in pairs_df['PTID'].unique():
        pairs_ptid = pairs_df[pairs_df['PTID'] == ptid]
        plasma_ptid = plasma_df[plasma_df['PTID'] == ptid].copy()  # 显式创建副本避免警告
        
        if len(plasma_ptid) == 0:
            continue
        
        # 对该 PTID 的每条 pairs 记录
        for idx, pair_row in pairs_ptid.iterrows():
            pair_date = pair_row['EXAMDATE_dt']
            
            # 计算日期差异
            plasma_ptid['date_diff'] = abs((plasma_ptid['EXAMDATE_dt'] - pair_date).dt.days)
            
            # 找到日期差异最小且在 max_days 内的记录
            valid_plasma = plasma_ptid[plasma_ptid['date_diff'] <= max_days]
            
            if len(valid_plasma) > 0:
                # 选择日期差异最小的记录
                best_match = valid_plasma.loc[valid_plasma['date_diff'].idxmin()]
                matches.append({
                    'pairs_idx': idx,
                    'plasma_idx': best_match.name,
                    'date_diff': best_match['date_diff'],
                    'source': source_name
                })
    
    print(f"  - 成功匹配: {len(matches)} 条")
    return matches


def extend_pairs_with_plasma(pairs_file, upenn_file, c2n_file, output_file, max_days=30):
    """
    主函数：扩展 pairs 文件添加血浆信息
    
    参数:
        pairs_file: pairs_90d.csv 文件路径
        upenn_file: UPENN 血浆数据文件路径
        c2n_file: C2N 血浆数据文件路径
        output_file: 输出文件路径
        max_days: 最大允许的日期差异（天）
    """
    print("="*80)
    print("扩展 pairs_90d.csv 添加血浆信息")
    print("="*80)
    
    # 读取数据
    print("\n读取文件...")
    pairs_df = pd.read_csv(pairs_file)
    upenn_df = pd.read_csv(upenn_file)
    c2n_df = pd.read_csv(c2n_file)
    
    print(f"  - pairs_90d.csv: {len(pairs_df)} 行")
    print(f"  - UPENN: {len(upenn_df)} 行")
    print(f"  - C2N: {len(c2n_df)} 行")
    
    # 初始化新列
    new_columns = {
        'pT217_F': np.nan,
        'AB42_F': np.nan,
        'AB40_F': np.nan,
        'AB42_AB40_F': np.nan,
        'pT217_AB42_F': np.nan,
        'NfL_Q': np.nan,
        'GFAP_Q': np.nan,
        'plasma_source': '',
        'plasma_date': '',
        'plasma_date_diff': np.nan
    }
    
    for col, default_val in new_columns.items():
        pairs_df[col] = default_val
    
    # 匹配 UPENN 数据
    upenn_matches = match_plasma_data(
        pairs_df.copy(), 
        upenn_df, 
        'UPENN', 
        max_days
    )
    
    # 填充 UPENN 数据
    print("\n填充 UPENN 数据...")
    for match in upenn_matches:
        pairs_idx = match['pairs_idx']
        plasma_idx = match['plasma_idx']
        
        upenn_row = upenn_df.loc[plasma_idx]
        
        pairs_df.loc[pairs_idx, 'pT217_F'] = upenn_row.get('pT217_F', np.nan)
        pairs_df.loc[pairs_idx, 'AB42_F'] = upenn_row.get('AB42_F', np.nan)
        pairs_df.loc[pairs_idx, 'AB40_F'] = upenn_row.get('AB40_F', np.nan)
        pairs_df.loc[pairs_idx, 'AB42_AB40_F'] = upenn_row.get('AB42_AB40_F', np.nan)
        pairs_df.loc[pairs_idx, 'pT217_AB42_F'] = upenn_row.get('pT217_AB42_F', np.nan)
        pairs_df.loc[pairs_idx, 'NfL_Q'] = upenn_row.get('NfL_Q', np.nan)
        pairs_df.loc[pairs_idx, 'GFAP_Q'] = upenn_row.get('GFAP_Q', np.nan)
        pairs_df.loc[pairs_idx, 'plasma_source'] = 'UPENN'
        pairs_df.loc[pairs_idx, 'plasma_date'] = upenn_row.get('EXAMDATE', '')
        pairs_df.loc[pairs_idx, 'plasma_date_diff'] = match['date_diff']
    
    # 匹配 C2N 数据（仅匹配尚未匹配的记录）
    unmatched_pairs = pairs_df[pairs_df['plasma_source'] == ''].copy()
    c2n_matches = match_plasma_data(
        unmatched_pairs, 
        c2n_df, 
        'C2N', 
        max_days
    )
    
    # 填充 C2N 数据
    print("\n填充 C2N 数据...")
    for match in c2n_matches:
        pairs_idx = match['pairs_idx']
        plasma_idx = match['plasma_idx']
        
        c2n_row = c2n_df.loc[plasma_idx]
        
        # C2N 使用不同的列名
        pairs_df.loc[pairs_idx, 'pT217_F'] = c2n_row.get('pT217_C2N', np.nan)
        pairs_df.loc[pairs_idx, 'AB42_F'] = c2n_row.get('AB42_C2N', np.nan)
        pairs_df.loc[pairs_idx, 'AB40_F'] = c2n_row.get('AB40_C2N', np.nan)
        pairs_df.loc[pairs_idx, 'AB42_AB40_F'] = c2n_row.get('AB42_AB40_C2N', np.nan)
        pairs_df.loc[pairs_idx, 'pT217_AB42_F'] = c2n_row.get('pT217_npT217_C2N', np.nan)  # 使用 pT217_npT217_C2N
        pairs_df.loc[pairs_idx, 'NfL_Q'] = np.nan  # C2N 没有 NfL_Q
        pairs_df.loc[pairs_idx, 'GFAP_Q'] = np.nan  # C2N 没有 GFAP_Q
        pairs_df.loc[pairs_idx, 'plasma_source'] = 'C2N'
        pairs_df.loc[pairs_idx, 'plasma_date'] = c2n_row.get('EXAMDATE', '')
        pairs_df.loc[pairs_idx, 'plasma_date_diff'] = match['date_diff']
    
    # 删除临时列
    if 'EXAMDATE_dt' in pairs_df.columns:
        pairs_df = pairs_df.drop('EXAMDATE_dt', axis=1)
    
    # 统计信息（在类型转换前进行，以便使用数值类型计算）
    print("\n" + "="*80)
    print("匹配统计:")
    print("="*80)
    total_matched = len(pairs_df[pairs_df['plasma_source'] != ''])
    upenn_matched = len(pairs_df[pairs_df['plasma_source'] == 'UPENN'])
    c2n_matched = len(pairs_df[pairs_df['plasma_source'] == 'C2N'])
    unmatched = len(pairs_df[pairs_df['plasma_source'] == ''])
    
    print(f"总记录数: {len(pairs_df)}")
    print(f"成功匹配: {total_matched} ({total_matched/len(pairs_df)*100:.2f}%)")
    print(f"  - UPENN: {upenn_matched} ({upenn_matched/len(pairs_df)*100:.2f}%)")
    print(f"  - C2N: {c2n_matched} ({c2n_matched/len(pairs_df)*100:.2f}%)")
    print(f"未匹配: {unmatched} ({unmatched/len(pairs_df)*100:.2f}%)")
    
    # 日期差异统计（使用原始数值类型）
    matched_df = pairs_df[pairs_df['plasma_source'] != '']
    if len(matched_df) > 0 and 'plasma_date_diff' in matched_df.columns:
        date_diff_numeric = pd.to_numeric(matched_df['plasma_date_diff'], errors='coerce')
        if date_diff_numeric.notna().any():
            print(f"\n日期差异统计 (天):")
            print(f"  - 平均: {date_diff_numeric.mean():.2f}")
            print(f"  - 中位数: {date_diff_numeric.median():.2f}")
            print(f"  - 最小: {date_diff_numeric.min():.0f}")
            print(f"  - 最大: {date_diff_numeric.max():.0f}")
    
    # 在保存前处理字段类型
    # id_x 字段和 DIAGNOSIS 字段应为字符串类型，避免被转换为 float
    id_columns = ['id_mri', 'id_fdg', 'id_av45', 'id_av1451']
    for col in id_columns:
        if col in pairs_df.columns:
            # 将非空值转为字符串（整数形式），空值保持为空字符串
            pairs_df[col] = pairs_df[col].apply(
                lambda x: str(int(float(x))) if pd.notna(x) and str(x).strip() != '' else ''
            )
    
    # DIAGNOSIS 转换为字符串（整数形式）
    if 'DIAGNOSIS' in pairs_df.columns:
        pairs_df['DIAGNOSIS'] = pairs_df['DIAGNOSIS'].apply(
            lambda x: str(int(float(x))) if pd.notna(x) and str(x).strip() != '' else ''
        )
    
    # plasma_date_diff 转换为整数字符串（天数）
    if 'plasma_date_diff' in pairs_df.columns:
        pairs_df['plasma_date_diff'] = pairs_df['plasma_date_diff'].apply(
            lambda x: str(int(x)) if pd.notna(x) else ''
        )
    
    # 保存结果
    print(f"\n保存结果到: {output_file}")
    pairs_df.to_csv(output_file, index=False)
    
    # 生成仅包含成功匹配的记录的CSV
    matched_only_df = pairs_df[pairs_df['plasma_source'] != ''].copy()
    
    # 生成匹配成功的文件名（在原文件名基础上添加 _matched）
    output_dir = os.path.dirname(output_file)
    output_basename = os.path.basename(output_file)
    output_name, output_ext = os.path.splitext(output_basename)
    matched_output_file = os.path.join(output_dir, f"{output_name}_matched{output_ext}")
    
    print(f"保存仅匹配成功的记录到: {matched_output_file}")
    print(f"  - 匹配成功记录数: {len(matched_only_df)}")
    matched_only_df.to_csv(matched_output_file, index=False)
    
    print("\n完成！")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='扩展 pairs_90d.csv 文件，添加血浆信息'
    )
    parser.add_argument(
        '--pairs',
        default='adapter_finetune/adapter_finetune/pairs_90d.csv',
        help='pairs_90d.csv 文件路径'
    )
    parser.add_argument(
        '--upenn',
        default='adapter_finetune/UPENN_PLASMA_FUJIREBIO_QUANTERIX_21Dec2025.csv',
        help='UPENN 血浆数据文件路径'
    )
    parser.add_argument(
        '--c2n',
        default='adapter_finetune/C2N_PRECIVITYAD2_PLASMA_29Dec2025.csv',
        help='C2N 血浆数据文件路径'
    )
    parser.add_argument(
        '--output',
        default='adapter_finetune/pairs_90d_with_plasma.csv',
        help='输出文件路径'
    )
    parser.add_argument(
        '--max-days',
        type=int,
        default=90,
        help='最大允许的日期差异（天），默认 90 天'
    )
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    for file_path in [args.pairs, args.upenn, args.c2n]:
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在: {file_path}")
            return
    
    # 执行扩展
    extend_pairs_with_plasma(
        pairs_file=args.pairs,
        upenn_file=args.upenn,
        c2n_file=args.c2n,
        output_file=args.output,
        max_days=args.max_days
    )


if __name__ == '__main__':
    main()
