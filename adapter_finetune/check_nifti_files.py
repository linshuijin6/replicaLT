#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整检查 NIFTI 目录下的 MRI 和 PET 文件
目录结构：/mnt/nfsdata/nfsdata/ADNI/ADNI0103/NIFTI/{subject_id}/{series_description}/{timestamp}/I{image_id}/xxx.nii.gz
"""

import os
import pandas as pd
from pathlib import Path
from collections import defaultdict
import json

# 设置路径
NIFTI_BASE = "/mnt/nfsdata/nfsdata/ADNI/ADNI0103/NIFTI"
MRI_CSV = "/home/ssddata/linshuijin/replicaLT/adapter_finetune/MRIT1.0114.csv"
PET_CSV = "/home/ssddata/linshuijin/replicaLT/adapter_finetune/PET.0114.csv"

def find_nifti_file(base_dir, subject_id, image_id):
    """
    在 subject_id 目录下查找对应的 image_id 的 nii.gz 文件
    返回：(found, file_path)
    """
    # 检查输入是否有效
    if pd.isna(subject_id) or pd.isna(image_id):
        return False, None
    
    subject_dir = os.path.join(base_dir, str(subject_id))
    if not os.path.exists(subject_dir):
        return False, None
    
    # 查找 I{image_id} 目录
    try:
        image_dir_name = f"I{int(image_id)}"
    except (ValueError, TypeError):
        return False, None
    
    # 遍历 subject 目录的所有子目录
    for root, dirs, files in os.walk(subject_dir):
        if image_dir_name in dirs:
            image_dir_path = os.path.join(root, image_dir_name)
            # 在该目录下查找 .nii.gz 文件
            try:
                for f in os.listdir(image_dir_path):
                    if f.endswith('.nii.gz'):
                        return True, os.path.join(image_dir_path, f)
            except Exception as e:
                print(f"  警告: 无法读取目录 {image_dir_path}: {e}")
                continue
    
    return False, None

def check_files(csv_path, file_type):
    """
    检查 CSV 文件中所有 image_id 对应的 nii.gz 文件
    """
    print(f"\n{'='*80}")
    print(f"检查 {file_type} 文件: {csv_path}")
    print(f"{'='*80}\n")
    
    # 读取 CSV，指定 low_memory=False 以避免警告
    df = pd.read_csv(csv_path, low_memory=False)
    
    # 删除 image_id 或 subject_id 为空的行
    df = df.dropna(subset=['image_id', 'subject_id'])
    
    total_count = len(df)
    
    print(f"CSV 文件总行数（有效）: {total_count}")
    print(f"唯一 image_id 数量: {df['image_id'].nunique()}")
    print(f"唯一 subject_id 数量: {df['subject_id'].nunique()}")
    
    # 获取唯一的 image_id，因为 CSV 中可能有重复
    unique_images = df.drop_duplicates(subset=['image_id'])[['image_id', 'subject_id']]
    print(f"唯一的 image_id-subject_id 组合: {len(unique_images)}")
    
    # 检查每个文件
    found_list = []
    missing_list = []
    
    for idx, row in unique_images.iterrows():
        image_id = row['image_id']
        subject_id = row['subject_id']
        
        found, file_path = find_nifti_file(NIFTI_BASE, subject_id, image_id)
        
        if found:
            found_list.append({
                'image_id': image_id,
                'subject_id': subject_id,
                'file_path': file_path
            })
        else:
            missing_list.append({
                'image_id': image_id,
                'subject_id': subject_id
            })
        
        # 进度显示
        if (len(found_list) + len(missing_list)) % 100 == 0:
            checked = len(found_list) + len(missing_list)
            print(f"进度: {checked}/{len(unique_images)} ({checked/len(unique_images)*100:.1f}%)")
    
    # 统计结果
    found_count = len(found_list)
    missing_count = len(missing_list)
    checked_count = found_count + missing_count
    
    print(f"\n{'='*80}")
    print(f"检查结果统计 - {file_type}")
    print(f"{'='*80}")
    print(f"检查的唯一 image_id: {checked_count}")
    print(f"找到: {found_count} ({found_count/checked_count*100:.2f}%)")
    print(f"缺失: {missing_count} ({missing_count/checked_count*100:.2f}%)")
    
    # 保存结果
    if missing_count > 0:
        missing_df = pd.DataFrame(missing_list)
        missing_file = csv_path.replace('.csv', '_missing.csv')
        missing_df.to_csv(missing_file, index=False)
        print(f"\n缺失文件列表已保存到: {missing_file}")
        
        # 显示前 20 条缺失记录
        print(f"\n前 20 条缺失记录:")
        print(missing_df.head(20).to_string())
    else:
        print(f"\n✓ 所有 {file_type} 文件都存在！")
    
    # 保存找到的文件列表
    if found_count > 0:
        found_df = pd.DataFrame(found_list)
        found_file = csv_path.replace('.csv', '_found.csv')
        found_df.to_csv(found_file, index=False)
        print(f"\n找到的文件列表已保存到: {found_file}")
    
    return {
        'total': checked_count,
        'found': found_count,
        'missing': missing_count,
        'missing_list': missing_list
    }

def main():
    print(f"\n开始检查 NIFTI 文件...")
    print(f"基础目录: {NIFTI_BASE}")
    
    # 检查基础目录是否存在
    if not os.path.exists(NIFTI_BASE):
        print(f"错误: 基础目录不存在: {NIFTI_BASE}")
        return
    
    # 检查 MRI 文件
    mri_results = check_files(MRI_CSV, "MRI")
    
    # 检查 PET 文件
    pet_results = check_files(PET_CSV, "PET")
    
    # 总体统计
    print(f"\n{'='*80}")
    print(f"总体统计")
    print(f"{'='*80}")
    print(f"MRI - 检查: {mri_results['total']}, 找到: {mri_results['found']}, 缺失: {mri_results['missing']}")
    print(f"PET - 检查: {pet_results['total']}, 找到: {pet_results['found']}, 缺失: {pet_results['missing']}")
    print(f"{'='*80}")
    
    # 保存汇总报告
    summary = {
        'mri': {
            'checked': mri_results['total'],
            'found': mri_results['found'],
            'missing': mri_results['missing'],
            'found_rate': f"{mri_results['found']/mri_results['total']*100:.2f}%"
        },
        'pet': {
            'checked': pet_results['total'],
            'found': pet_results['found'],
            'missing': pet_results['missing'],
            'found_rate': f"{pet_results['found']/pet_results['total']*100:.2f}%"
        },
        'total': {
            'checked': mri_results['total'] + pet_results['total'],
            'found': mri_results['found'] + pet_results['found'],
            'missing': mri_results['missing'] + pet_results['missing']
        }
    }
    
    summary_file = '/home/ssddata/linshuijin/replicaLT/adapter_finetune/nifti_check_summary.json'
    with open(summary_file, 'w') as f:
        # 不包含 missing_list，只保留统计信息
        json.dump(summary, f, indent=2)
    
    print(f"\n汇总报告已保存到: {summary_file}")

if __name__ == '__main__':
    main()
