#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从备份目录查找并复制缺失的 NIFTI 文件
"""

import os
import pandas as pd
import shutil
from pathlib import Path

# 路径配置
SOURCE_MRI_DIR = "/mnt/nfsdata/nfsdata/LorenzoT/ADNI/raw/MRI/MRI_Dataset"
TARGET_NIFTI_DIR = "/mnt/nfsdata/nfsdata/ADNI/ADNI0103/NIFTI"
MRI_MISSING_CSV = "/home/ssddata/linshuijin/replicaLT/adapter_finetune/MRIT1.0114_missing.csv"
PET_MISSING_CSV = "/home/ssddata/linshuijin/replicaLT/adapter_finetune/PET.0114_missing.csv"

def find_image_dir_in_source(source_base, subject_id, image_id):
    """
    在源目录中查找 image_id 对应的目录
    """
    subject_dir = os.path.join(source_base, str(subject_id))
    if not os.path.exists(subject_dir):
        return None
    
    image_dir_name = f"I{int(image_id)}"
    
    # 遍历查找
    for root, dirs, files in os.walk(subject_dir):
        if image_dir_name in dirs:
            return os.path.join(root, image_dir_name)
    
    return None

def copy_directory_preserve_structure(src_dir, target_base):
    """
    复制目录并保持相对结构
    src_dir: 源目录完整路径，例如 /source/002_S_0413/Accelerated_Sagittal_MPRAGE/2017-06-21_13_23_38.0/I863056
    target_base: 目标基础目录
    
    保持从 subject_id 开始的结构
    """
    # 提取从 subject_id 开始的相对路径
    parts = Path(src_dir).parts
    
    # 找到 subject_id 的位置（格式为 XXX_S_XXXX）
    subject_idx = None
    for i, part in enumerate(parts):
        if '_S_' in part:
            subject_idx = i
            break
    
    if subject_idx is None:
        print(f"  错误: 无法识别 subject_id，路径: {src_dir}")
        return False
    
    # 构建相对路径
    relative_parts = parts[subject_idx:]
    target_path = os.path.join(target_base, *relative_parts)
    
    # 如果目标已存在，跳过
    if os.path.exists(target_path):
        print(f"  目标已存在，跳过: {target_path}")
        return True
    
    # 创建父目录
    target_parent = os.path.dirname(target_path)
    os.makedirs(target_parent, exist_ok=True)
    
    # 复制整个目录
    try:
        shutil.copytree(src_dir, target_path)
        print(f"  ✓ 已复制: {src_dir}")
        print(f"      到: {target_path}")
        return True
    except Exception as e:
        print(f"  ✗ 复制失败: {e}")
        return False

def process_missing_files(missing_csv, source_dir, file_type):
    """
    处理缺失文件的复制
    """
    print(f"\n{'='*80}")
    print(f"处理缺失的 {file_type} 文件")
    print(f"{'='*80}\n")
    
    # 读取缺失文件列表
    df = pd.read_csv(missing_csv)
    total = len(df)
    
    print(f"需要查找的文件数: {total}")
    
    found_count = 0
    copied_count = 0
    not_found_count = 0
    
    for idx, row in df.iterrows():
        image_id = row['image_id']
        subject_id = row['subject_id']
        
        # 处理可能的浮点数
        try:
            image_id = int(image_id)
        except (ValueError, TypeError):
            print(f"\n{idx+1}/{total} - 跳过无效的 image_id: {image_id}")
            not_found_count += 1
            continue
        
        print(f"\n{idx+1}/{total} - 查找 image_id={image_id}, subject_id={subject_id}")
        
        # 在源目录中查找
        src_dir = find_image_dir_in_source(source_dir, subject_id, image_id)
        
        if src_dir:
            found_count += 1
            print(f"  找到源目录: {src_dir}")
            
            # 复制到目标目录
            if copy_directory_preserve_structure(src_dir, TARGET_NIFTI_DIR):
                copied_count += 1
        else:
            not_found_count += 1
            print(f"  ✗ 在源目录中未找到")
    
    # 统计
    print(f"\n{'='*80}")
    print(f"{file_type} 处理统计")
    print(f"{'='*80}")
    print(f"需要查找: {total}")
    print(f"找到: {found_count}")
    print(f"成功复制: {copied_count}")
    print(f"未找到: {not_found_count}")
    print(f"{'='*80}\n")
    
    return {
        'total': total,
        'found': found_count,
        'copied': copied_count,
        'not_found': not_found_count
    }

def main():
    print("\n开始处理缺失的 NIFTI 文件...")
    print(f"源目录: {SOURCE_MRI_DIR}")
    print(f"目标目录: {TARGET_NIFTI_DIR}")
    
    # 检查目录是否存在
    if not os.path.exists(SOURCE_MRI_DIR):
        print(f"错误: 源目录不存在: {SOURCE_MRI_DIR}")
        return
    
    if not os.path.exists(TARGET_NIFTI_DIR):
        print(f"错误: 目标目录不存在: {TARGET_NIFTI_DIR}")
        return
    
    # 处理 MRI 缺失文件
    if os.path.exists(MRI_MISSING_CSV):
        mri_stats = process_missing_files(MRI_MISSING_CSV, SOURCE_MRI_DIR, "MRI")
    else:
        print(f"警告: MRI 缺失文件列表不存在: {MRI_MISSING_CSV}")
        mri_stats = None
    
    # 注意：PET 文件通常在不同的目录，这里主要处理 MRI
    # 如果 PET 也在同一个源目录，可以取消注释以下代码
    # if os.path.exists(PET_MISSING_CSV):
    #     pet_stats = process_missing_files(PET_MISSING_CSV, SOURCE_MRI_DIR, "PET")
    
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    if mri_stats:
        print(f"MRI: 需要{mri_stats['total']}个，找到{mri_stats['found']}个，复制{mri_stats['copied']}个")
    print("="*80)
    print("\n完成！")

if __name__ == '__main__':
    main()
