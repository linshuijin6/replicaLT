#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将刚复制的 DICOM 文件转换为 NIfTI 格式
使用 dcm2niix 工具进行转换
"""

import os
import subprocess
import pandas as pd
from pathlib import Path

# 路径配置
NIFTI_BASE = "/mnt/nfsdata/nfsdata/ADNI/ADNI0103/NIFTI"
MISSING_CSV = "/home/ssddata/linshuijin/replicaLT/adapter_finetune/MRIT1.0114_missing.csv"

def find_image_dir(base_dir, subject_id, image_id):
    """查找 I{image_id} 目录"""
    subject_dir = os.path.join(base_dir, str(subject_id))
    if not os.path.exists(subject_dir):
        return None
    
    image_dir_name = f"I{int(image_id)}"
    
    for root, dirs, files in os.walk(subject_dir):
        if image_dir_name in dirs:
            return os.path.join(root, image_dir_name)
    
    return None

def has_dcm_files(directory):
    """检查目录中是否有 dcm 文件"""
    try:
        for f in os.listdir(directory):
            if f.endswith('.dcm'):
                return True
    except Exception:
        pass
    return False

def convert_dcm_to_nifti(dcm_dir):
    """
    使用 dcm2niix 将 DICOM 转换为 NIfTI
    dcm2niix 参数：
    -z y : 压缩为 .nii.gz
    -f : 输出文件名格式
    -o : 输出目录
    -s y : 单文件模式
    -b n : 不生成 BIDS JSON
    """
    try:
        # 获取目录的父目录作为输出目录
        output_dir = dcm_dir
        
        # 构建命令
        # 使用 %s (series description) 和 %i (series number) 作为文件名
        cmd = [
            'dcm2niix',
            '-z', 'y',           # 压缩输出
            '-f', '%s_%i',       # 文件名: series_description_series_number
            '-o', output_dir,    # 输出到相同目录
            '-s', 'y',           # 单文件模式
            '-b', 'n',           # 不生成 JSON
            '-v', '0',           # 静默模式
            dcm_dir
        ]
        
        print(f"  转换命令: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            # 检查是否生成了 nii.gz 文件
            nifti_files = [f for f in os.listdir(output_dir) if f.endswith('.nii.gz')]
            if nifti_files:
                print(f"  ✓ 转换成功，生成文件: {', '.join(nifti_files)}")
                
                # 删除 dcm 文件
                dcm_count = 0
                for f in os.listdir(dcm_dir):
                    if f.endswith('.dcm'):
                        os.remove(os.path.join(dcm_dir, f))
                        dcm_count += 1
                print(f"  ✓ 已删除 {dcm_count} 个 dcm 文件")
                
                return True, nifti_files[0]
            else:
                print(f"  ✗ 转换失败：未生成 nii.gz 文件")
                print(f"  stdout: {result.stdout}")
                print(f"  stderr: {result.stderr}")
                return False, None
        else:
            print(f"  ✗ dcm2niix 执行失败 (返回码: {result.returncode})")
            print(f"  stdout: {result.stdout}")
            print(f"  stderr: {result.stderr}")
            return False, None
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ 转换超时（60秒）")
        return False, None
    except Exception as e:
        print(f"  ✗ 转换出错: {e}")
        return False, None

def main():
    print("开始转换 DICOM 文件为 NIfTI 格式...")
    print(f"目标目录: {NIFTI_BASE}")
    
    # 检查 dcm2niix 是否可用
    try:
        subprocess.run(['dcm2niix', '-h'], capture_output=True, check=True)
        print("✓ dcm2niix 工具可用\n")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ 错误: 未找到 dcm2niix 工具")
        print("请安装: conda install -c conda-forge dcm2niix")
        return
    
    # 读取缺失文件列表
    df = pd.read_csv(MISSING_CSV)
    df = df.dropna(subset=['image_id', 'subject_id'])
    
    print(f"需要处理的 image_id 数量: {len(df)}\n")
    
    success_count = 0
    failed_count = 0
    no_dcm_count = 0
    not_found_count = 0
    
    for idx, row in df.iterrows():
        image_id = int(row['image_id'])
        subject_id = row['subject_id']
        
        print(f"[{idx+1}/{len(df)}] 处理 {subject_id}/I{image_id}")
        
        # 查找目录
        image_dir = find_image_dir(NIFTI_BASE, subject_id, image_id)
        
        if not image_dir:
            print(f"  ✗ 未找到目录")
            not_found_count += 1
            continue
        
        print(f"  目录: {image_dir}")
        
        # 检查是否有 dcm 文件
        if not has_dcm_files(image_dir):
            print(f"  - 目录中没有 dcm 文件，跳过")
            no_dcm_count += 1
            continue
        
        # 转换
        success, nifti_file = convert_dcm_to_nifti(image_dir)
        
        if success:
            success_count += 1
        else:
            failed_count += 1
        
        print()
    
    # 统计
    print("="*80)
    print("转换完成！")
    print("="*80)
    print(f"总计: {len(df)}")
    print(f"转换成功: {success_count}")
    print(f"转换失败: {failed_count}")
    print(f"无 dcm 文件: {no_dcm_count}")
    print(f"目录未找到: {not_found_count}")

if __name__ == "__main__":
    main()
