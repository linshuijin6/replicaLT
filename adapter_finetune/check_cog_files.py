#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查配准(Coregistration)文件的完成度
根据 CSV 表格（如 pairs_180d_dx.csv）检查生成数据路径下的文件

文件命名规则：
- MRI: <PTID>__<id_mri>.nii.gz
- PET: <PTID>__<id_mri>__<id_pet>.nii.gz

目录结构：
- MRI 相关文件在 MRI/, MRI_MASK/, MRI_XFM/, MRI_NATIVE_RSTD/ 下
- PET 文件在 PET_MNI/FDG/, PET_MNI/AV45/, PET_MNI/TAU/ 下
"""

import argparse
import json
import os
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def norm_img_id(val) -> str:
    """标准化 image_id"""
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", ""):
        return ""
    # 移除可能的 'I' 前缀
    if s.upper().startswith("I"):
        s = s[1:]
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s


def find_file_recursive(root: Path, pattern: str) -> Optional[Path]:
    """递归查找文件"""
    try:
        for p in root.rglob(pattern):
            if p.is_file():
                return p
    except Exception:
        return None
    return None


def check_mri_files(ptid: str, id_mri: str, cog_root: Path) -> Dict[str, any]:
    """
    检查 MRI 相关文件（4个文件）
    返回：{
        'found': int,  # 找到的文件数量
        'total': int,  # 应有的文件数量
        'missing': List[str],  # 缺失的文件类型
        'files': Dict[str, Optional[Path]]  # 找到的文件路径
    }
    """
    # 尝试两种命名格式：带 I 前缀和不带 I 前缀
    base_name_with_i = f"{ptid}__I{id_mri}"
    base_name_without_i = f"{ptid}__{id_mri}"
    
    expected_files = {
        "mri": ("MRI", [f"{base_name_with_i}.nii.gz", f"{base_name_without_i}.nii.gz"]),
        "mask": ("MRI_MASK", [f"{base_name_with_i}_mask.nii.gz", f"{base_name_without_i}_mask.nii.gz"]),
        "xfm": ("MRI_XFM", [f"{base_name_with_i}_mri2mni.mat", f"{base_name_without_i}_mri2mni.mat"]),
        "rstd": ("MRI_NATIVE_RSTD", [f"{base_name_with_i}_rstd.nii.gz", f"{base_name_without_i}_rstd.nii.gz"]),
    }
    
    found_files = {}
    missing = []
    
    for key, (subdir, filenames) in expected_files.items():
        search_root = cog_root / subdir
        if not search_root.exists():
            missing.append(key)
            found_files[key] = None
            continue
        
        file_path = None
        for filename in filenames:
            file_path = find_file_recursive(search_root, filename)
            if file_path:
                break
        
        if file_path:
            found_files[key] = file_path
        else:
            missing.append(key)
            found_files[key] = None
    
    return {
        "found": len(found_files) - len(missing),
        "total": len(expected_files),
        "missing": missing,
        "files": found_files,
    }


def check_pet_file(
    ptid: str, id_mri: str, modality: str, id_pet: str, cog_root: Path
) -> Dict[str, any]:
    """
    检查单个 PET 文件（4个文件）
    返回：{
        'found': int,
        'total': int,
        'missing': List[str],
        'files': Dict[str, Optional[Path]]
    }
    """
    # 尝试两种命名格式：带 I 前缀和不带 I 前缀
    base_with_i = f"{ptid}__I{id_mri}__I{id_pet}"
    base_without_i = f"{ptid}__{id_mri}__{id_pet}"
    base_mri_i_pet_no = f"{ptid}__I{id_mri}__{id_pet}"
    base_mri_no_pet_i = f"{ptid}__{id_mri}__I{id_pet}"
    
    pet_root = cog_root / "PET_MNI" / modality
    
    expected_files = {
        "brain": [
            f"{base_with_i}.nii.gz",
            f"{base_without_i}.nii.gz",
            f"{base_mri_i_pet_no}.nii.gz",
            f"{base_mri_no_pet_i}.nii.gz",
        ],
        "full": [
            f"{base_with_i}_full.nii.gz",
            f"{base_without_i}_full.nii.gz",
            f"{base_mri_i_pet_no}_full.nii.gz",
            f"{base_mri_no_pet_i}_full.nii.gz",
        ],
        "pet2mni": [
            f"{base_with_i}_pet2mni.mat",
            f"{base_without_i}_pet2mni.mat",
            f"{base_mri_i_pet_no}_pet2mni.mat",
            f"{base_mri_no_pet_i}_pet2mni.mat",
        ],
        "pet2mri": [
            f"{base_with_i}_pet2mri.mat",
            f"{base_without_i}_pet2mri.mat",
            f"{base_mri_i_pet_no}_pet2mri.mat",
            f"{base_mri_no_pet_i}_pet2mri.mat",
        ],
    }
    
    found_files = {}
    missing = []
    
    if not pet_root.exists():
        return {
            "found": 0,
            "total": len(expected_files),
            "missing": list(expected_files.keys()),
            "files": {k: None for k in expected_files.keys()},
        }
    
    for key, filenames in expected_files.items():
        file_path = None
        for filename in filenames:
            file_path = find_file_recursive(pet_root, filename)
            if file_path:
                break
        
        if file_path:
            found_files[key] = file_path
        else:
            missing.append(key)
            found_files[key] = None
    
    return {
        "found": len(found_files) - len(missing),
        "total": len(expected_files),
        "missing": missing,
        "files": found_files,
    }


def parse_log_files(
    logs_root: Optional[Path], ptid: str, id_mri: str, modality: Optional[str] = None, id_pet: Optional[str] = None
) -> Tuple[List[str], str]:
    """
    从 log 文件中查找处理状态和失败原因
    支持直接日志和分区日志（part_0, part_1等）
    返回：(失败原因列表, 最终状态)
    状态：OK, FAIL, SKIP, NOT_FOUND
    """
    if not logs_root or not logs_root.exists():
        return [], "NOT_FOUND"
    
    reasons = []
    all_statuses = []
    
    # 标准化 ID（确保带 I 前缀）
    norm_id_mri = f"I{id_mri}" if not str(id_mri).startswith("I") else str(id_mri)
    norm_id_pet = f"I{id_pet}" if id_pet and not str(id_pet).startswith("I") else str(id_pet) if id_pet else None
    
    # 收集所有可能的日志文件
    log_files = []
    # 检查直接日志
    if modality:
        direct_log = logs_root / "pipeline_pet.csv"
        if direct_log.exists():
            log_files.append(("pet", direct_log))
    else:
        direct_log = logs_root / "pipeline_mri.csv"
        if direct_log.exists():
            log_files.append(("mri", direct_log))
    
    # 检查分区日志
    for part_dir in sorted(logs_root.glob("part_*")):
        if modality:
            part_log = part_dir / "pipeline_pet.csv"
            if part_log.exists():
                log_files.append(("pet", part_log))
        else:
            part_log = part_dir / "pipeline_mri.csv"
            if part_log.exists():
                log_files.append(("mri", part_log))
    
    # 解析日志文件
    for log_type, log_path in log_files:
        try:
            df = pd.read_csv(log_path)
            
            if log_type == "mri":
                # MRI 日志
                mask = (df["subject_id"] == ptid) & (df["id_mri"] == norm_id_mri)
                matched = df[mask]
            else:
                # PET 日志
                if not norm_id_pet:
                    continue
                mask = (
                    (df["subject_id"] == ptid)
                    & (df["id_mri"] == norm_id_mri)
                    & (df["modality"] == modality)
                    & (df["id_pet"] == norm_id_pet)
                )
                matched = df[mask]
            
            if len(matched) > 0:
                for _, row in matched.iterrows():
                    status = row.get("status", "")
                    detail = row.get("detail", "")
                    action = row.get("action", "") if "action" in row else ""
                    
                    all_statuses.append(status)
                    
                    # 收集失败和跳过的原因
                    if status in ["FAIL", "SKIP"]:
                        if action:
                            reasons.append(f"{action}: {detail}")
                        else:
                            reasons.append(detail)
                    
        except Exception:
            pass
    
    # 确定最终状态
    if not all_statuses:
        final_status = "NOT_FOUND"
    elif "FAIL" in all_statuses:
        final_status = "FAIL"
    elif all(s == "OK" for s in all_statuses):
        final_status = "OK"
    elif "SKIP" in all_statuses and "OK" in all_statuses:
        final_status = "OK"  # 有成功记录就算成功
    elif "SKIP" in all_statuses:
        final_status = "SKIP"
    else:
        final_status = all_statuses[-1] if all_statuses else "NOT_FOUND"
    
    # 去重原因
    reasons = list(dict.fromkeys(reasons))
    
    return reasons, final_status


def check_csv_file(
    csv_path: Path, cog_root: Path, logs_root: Optional[Path] = None
) -> Dict:
    """检查整个 CSV 文件中的配准完成度"""
    print(f"\n{'='*80}")
    print(f"检查配准文件完成度")
    print(f"CSV 文件: {csv_path}")
    print(f"配准路径: {cog_root}")
    if logs_root:
        print(f"日志路径: {logs_root}")
    print(f"{'='*80}\n")
    
    # 读取 CSV
    df = pd.read_csv(csv_path)
    required_cols = {"PTID", "id_mri"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV 缺少必需列: {required_cols}")
    
    # 统计数据
    stats = {
        "total_rows": len(df),
        "mri": {"complete": 0, "partial": 0, "missing": 0, "total": 0},
        "fdg": {"complete": 0, "partial": 0, "missing": 0, "total": 0},
        "av45": {"complete": 0, "partial": 0, "missing": 0, "total": 0},
        "tau": {"complete": 0, "partial": 0, "missing": 0, "total": 0},
    }
    
    # 详细记录
    mri_details = []
    pet_details = []
    
    # PET 模态映射
    modality_map = {
        "id_fdg": ("FDG", "fdg"),
        "id_av45": ("AV45", "av45"),
        "id_av1451": ("TAU", "tau"),
    }
    
    print(f"开始检查 {len(df)} 行数据...\n")
    
    for idx, row in df.iterrows():
        ptid = str(row["PTID"]).strip()
        id_mri = norm_img_id(row["id_mri"])
        
        if not ptid or not id_mri:
            continue
        
        # 检查 MRI 文件
        mri_result = check_mri_files(ptid, id_mri, cog_root)
        stats["mri"]["total"] += 1
        
        # 获取日志状态
        reasons, log_status = parse_log_files(logs_root, ptid, id_mri) if logs_root else ([], "NOT_FOUND")
        
        # 判断实际状态：以文件是否存在为准
        if mri_result["found"] == mri_result["total"]:
            stats["mri"]["complete"] += 1
            # 文件完整但不记录
        elif mri_result["found"] > 0:
            stats["mri"]["partial"] += 1
            # 记录部分完成的 MRI
            status_str = f"文件部分存在({mri_result['found']}/{mri_result['total']})"
            if log_status == "FAIL":
                status_str += " [日志:失败]"
            elif log_status == "OK":
                status_str += " [日志:成功但文件不全]"
            elif log_status == "SKIP":
                status_str += " [日志:跳过]"
            
            mri_details.append(
                {
                    "PTID": ptid,
                    "id_mri": id_mri,
                    "found": mri_result["found"],
                    "total": mri_result["total"],
                    "missing_files": ",".join(mri_result["missing"]),
                    "log_status": log_status,
                    "status": status_str,
                    "reason": "; ".join(reasons) if reasons else "无具体失败原因",
                }
            )
        else:
            stats["mri"]["missing"] += 1
            # 记录完全缺失的 MRI
            status_str = "文件完全缺失"
            if log_status == "FAIL":
                status_str += " [日志:失败]"
            elif log_status == "OK":
                status_str += " [日志:成功但文件缺失]"
            elif log_status == "SKIP":
                status_str += " [日志:跳过]"
            elif log_status == "NOT_FOUND":
                status_str += " [日志:无记录]"
            
            mri_details.append(
                {
                    "PTID": ptid,
                    "id_mri": id_mri,
                    "found": mri_result["found"],
                    "total": mri_result["total"],
                    "missing_files": ",".join(mri_result["missing"]),
                    "log_status": log_status,
                    "status": status_str,
                    "reason": "; ".join(reasons) if reasons else "无具体失败原因",
                }
            )
        
        # 检查 PET 文件
        for col_name, (modality_name, stat_key) in modality_map.items():
            if col_name not in df.columns:
                continue
            
            id_pet = norm_img_id(row[col_name])
            if not id_pet:
                continue
            
            pet_result = check_pet_file(ptid, id_mri, modality_name, id_pet, cog_root)
            stats[stat_key]["total"] += 1
            
            # 获取日志状态
            reasons, log_status = (
                parse_log_files(logs_root, ptid, id_mri, modality_name, id_pet)
                if logs_root
                else ([], "NOT_FOUND")
            )
            
            # 判断实际状态：以文件是否存在为准
            if pet_result["found"] == pet_result["total"]:
                stats[stat_key]["complete"] += 1
                # 文件完整但不记录
            elif pet_result["found"] > 0:
                stats[stat_key]["partial"] += 1
                # 记录部分完成的 PET
                status_str = f"文件部分存在({pet_result['found']}/{pet_result['total']})"
                if log_status == "FAIL":
                    status_str += " [日志:失败]"
                elif log_status == "OK":
                    status_str += " [日志:成功但文件不全]"
                elif log_status == "SKIP":
                    status_str += " [日志:跳过]"
                
                pet_details.append(
                    {
                        "PTID": ptid,
                        "id_mri": id_mri,
                        "modality": modality_name,
                        "id_pet": id_pet,
                        "found": pet_result["found"],
                        "total": pet_result["total"],
                        "missing_files": ",".join(pet_result["missing"]),
                        "log_status": log_status,
                        "status": status_str,
                        "reason": "; ".join(reasons) if reasons else "无具体失败原因",
                    }
                )
            else:
                stats[stat_key]["missing"] += 1
                # 记录完全缺失的 PET
                status_str = "文件完全缺失"
                if log_status == "FAIL":
                    status_str += " [日志:失败]"
                elif log_status == "OK":
                    status_str += " [日志:成功但文件缺失]"
                elif log_status == "SKIP":
                    status_str += " [日志:跳过]"
                elif log_status == "NOT_FOUND":
                    status_str += " [日志:无记录]"
                
                pet_details.append(
                    {
                        "PTID": ptid,
                        "id_mri": id_mri,
                        "modality": modality_name,
                        "id_pet": id_pet,
                        "found": pet_result["found"],
                        "total": pet_result["total"],
                        "missing_files": ",".join(pet_result["missing"]),
                        "log_status": log_status,
                        "status": status_str,
                        "reason": "; ".join(reasons) if reasons else "无具体失败原因",
                    }
                )
        
        # 进度显示
        if (idx + 1) % 50 == 0:
            print(
                f"进度: {idx + 1}/{len(df)} ({(idx + 1) / len(df) * 100:.1f}%)", flush=True
            )
    
    print(f"\n检查完成！\n")
    
    # 打印统计结果
    print(f"{'='*80}")
    print(f"统计结果")
    print(f"{'='*80}")
    for key in ["mri", "fdg", "av45", "tau"]:
        data = stats[key]
        if data["total"] > 0:
            complete_rate = data["complete"] / data["total"] * 100
            print(
                f"{key.upper():6s} - 总数: {data['total']:4d}, "
                f"完整: {data['complete']:4d} ({complete_rate:.1f}%), "
                f"部分: {data['partial']:4d}, "
                f"缺失: {data['missing']:4d}"
            )
    print(f"{'='*80}\n")
    
    # 保存缺失文件详细信息
    output_dir = csv_path.parent
    if mri_details:
        mri_df = pd.DataFrame(mri_details)
        mri_missing_file = output_dir / f"{csv_path.stem}_mri_missing.csv"
        mri_df.to_csv(mri_missing_file, index=False)
        print(f"MRI 缺失文件详情已保存: {mri_missing_file}")
        print(f"  缺失数量: {len(mri_details)}")
        print(f"  前 10 条:")
        print(mri_df.head(10).to_string(index=False))
        print()
    
    if pet_details:
        pet_df = pd.DataFrame(pet_details)
        pet_missing_file = output_dir / f"{csv_path.stem}_pet_missing.csv"
        pet_df.to_csv(pet_missing_file, index=False)
        print(f"PET 缺失文件详情已保存: {pet_missing_file}")
        print(f"  缺失数量: {len(pet_details)}")
        print(f"  前 10 条:")
        print(pet_df.head(10).to_string(index=False))
        print()
    
    # 保存汇总报告
    summary_file = output_dir / f"{csv_path.stem}_cog_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"汇总报告已保存: {summary_file}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="检查配准文件的完成度",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s --csv pairs_180d_dx.csv --cog_root /mnt/nfsdata/nfsdata/ADNI/ADNI0103/Coregistration
  
  %(prog)s --csv pairs_180d_dx.csv \\
           --cog_root /mnt/nfsdata/nfsdata/ADNI/ADNI0103/Coregistration \\
           --logs_root /mnt/nfsdata/nfsdata/ADNI/ADNI0103/logs_pipeline0106_parallel
        """,
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="输入的 CSV 文件路径（如 pairs_180d_dx.csv）",
    )
    parser.add_argument(
        "--cog_root",
        type=Path,
        required=True,
        help="配准生成数据存放路径（如 /mnt/.../Coregistration）",
    )
    parser.add_argument(
        "--logs_root",
        type=Path,
        default=None,
        help="生成过程日志文件夹路径（可选，用于分析缺失原因）",
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not args.csv.exists():
        print(f"错误: CSV 文件不存在: {args.csv}")
        return 1
    
    if not args.cog_root.exists():
        print(f"错误: 配准路径不存在: {args.cog_root}")
        return 1
    
    if args.logs_root and not args.logs_root.exists():
        print(f"警告: 日志路径不存在: {args.logs_root}")
        args.logs_root = None
    
    # 执行检查
    try:
        check_csv_file(args.csv, args.cog_root, args.logs_root)
        print("\n检查完成！")
        return 0
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
