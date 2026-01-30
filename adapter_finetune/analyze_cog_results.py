#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析配准检查结果，生成可读性报告
"""

import argparse
import json
import pandas as pd
from collections import Counter
from pathlib import Path


def analyze_mri_issues(mri_df: pd.DataFrame) -> dict:
    """分析 MRI 问题"""
    if len(mri_df) == 0:
        return {"总问题数": 0}
    
    analysis = {
        "总问题数": len(mri_df),
        "按日志状态统计": mri_df["log_status"].value_counts().to_dict(),
        "主要失败原因": {},
        "问题分布": {},
    }
    
    # 分析失败原因
    reasons = []
    for r in mri_df["reason"].dropna():
        if "FAST::Command failed" in r:
            reasons.append("FAST失败(脑组织分割)")
        elif "SYNTHSTRIP::Command failed" in r:
            reasons.append("SYNTHSTRIP失败(脑提取)")
        elif "REORIENT::Command failed" in r:
            reasons.append("REORIENT失败(方向调整)")
        elif "FLIRT::Command failed" in r:
            reasons.append("FLIRT失败(配准)")
        else:
            reasons.append("其他")
    
    reason_counts = Counter(reasons)
    analysis["主要失败原因"] = dict(reason_counts.most_common())
    
    # 按 PTID 统计问题分布
    ptid_counts = mri_df["PTID"].value_counts()
    analysis["问题最多的受试者"] = ptid_counts.head(10).to_dict()
    
    return analysis


def analyze_pet_issues(pet_df: pd.DataFrame) -> dict:
    """分析 PET 问题"""
    if len(pet_df) == 0:
        return {"总问题数": 0}
    
    analysis = {
        "总问题数": len(pet_df),
        "按日志状态统计": pet_df["log_status"].value_counts().to_dict(),
        "按模态统计": pet_df["modality"].value_counts().to_dict(),
        "主要失败原因": {},
    }
    
    # 分析失败案例
    failed = pet_df[pet_df["log_status"] == "FAIL"]
    if len(failed) > 0:
        reasons = []
        for r in failed["reason"].dropna():
            if "REORIENT::Command failed" in r:
                reasons.append("REORIENT失败(方向调整)")
            elif "FLIRT::Command failed" in r:
                reasons.append("FLIRT失败(配准)")
            elif "CONCAT_XFM::Command failed" in r:
                reasons.append("变换矩阵拼接失败")
            elif "APPLY_XFM::Command failed" in r:
                reasons.append("变换应用失败")
            else:
                reasons.append("其他")
        
        reason_counts = Counter(reasons)
        analysis["主要失败原因"] = dict(reason_counts.most_common())
    
    # 无日志记录的案例
    no_log = pet_df[pet_df["log_status"] == "NOT_FOUND"]
    if len(no_log) > 0:
        analysis["无日志记录数"] = len(no_log)
        analysis["无日志记录按模态"] = no_log["modality"].value_counts().to_dict()
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description="分析配准检查结果")
    parser.add_argument(
        "--mri_csv",
        type=Path,
        help="MRI 缺失文件 CSV",
    )
    parser.add_argument(
        "--pet_csv",
        type=Path,
        help="PET 缺失文件 CSV",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help="汇总统计 JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="输出分析报告 JSON（可选）",
    )
    
    args = parser.parse_args()
    
    # 如果没有指定文件，尝试查找默认文件
    if not args.mri_csv:
        candidates = list(Path(".").glob("*_mri_missing.csv"))
        if candidates:
            args.mri_csv = candidates[0]
    
    if not args.pet_csv:
        candidates = list(Path(".").glob("*_pet_missing.csv"))
        if candidates:
            args.pet_csv = candidates[0]
    
    if not args.summary:
        candidates = list(Path(".").glob("*_cog_summary.json"))
        if candidates:
            args.summary = candidates[0]
    
    report = {}
    
    # 读取汇总统计
    if args.summary and args.summary.exists():
        with open(args.summary, encoding="utf-8") as f:
            summary = json.load(f)
        report["汇总统计"] = summary
        
        print("=" * 80)
        print("配准完成度分析报告")
        print("=" * 80)
        print(f"\n总行数: {summary.get('total_rows', 0)}")
        print("\n各类文件完成情况:")
        for key in ["mri", "fdg", "av45", "tau"]:
            if key in summary:
                data = summary[key]
                total = data["total"]
                complete = data["complete"]
                if total > 0:
                    rate = complete / total * 100
                    print(
                        f"  {key.upper():6s}: {complete:4d}/{total:4d} ({rate:5.1f}%) "
                        f"- 部分:{data['partial']:3d} 缺失:{data['missing']:3d}"
                    )
    
    # 分析 MRI 问题
    if args.mri_csv and args.mri_csv.exists():
        mri_df = pd.read_csv(args.mri_csv)
        mri_analysis = analyze_mri_issues(mri_df)
        report["MRI问题分析"] = mri_analysis
        
        print("\n" + "=" * 80)
        print("MRI 问题分析")
        print("=" * 80)
        print(f"总问题数: {mri_analysis['总问题数']}")
        
        if "按日志状态统计" in mri_analysis:
            print("\n日志状态分布:")
            for status, count in mri_analysis["按日志状态统计"].items():
                print(f"  {status:12s}: {count:4d}")
        
        if "主要失败原因" in mri_analysis and mri_analysis["主要失败原因"]:
            print("\n主要失败原因:")
            for reason, count in mri_analysis["主要失败原因"].items():
                print(f"  {reason:30s}: {count:4d}")
        
        if "问题最多的受试者" in mri_analysis:
            print("\n问题最多的受试者（Top 5）:")
            for ptid, count in list(mri_analysis["问题最多的受试者"].items())[:5]:
                print(f"  {ptid:12s}: {count:4d} 个MRI")
    
    # 分析 PET 问题
    if args.pet_csv and args.pet_csv.exists():
        pet_df = pd.read_csv(args.pet_csv)
        pet_analysis = analyze_pet_issues(pet_df)
        report["PET问题分析"] = pet_analysis
        
        print("\n" + "=" * 80)
        print("PET 问题分析")
        print("=" * 80)
        print(f"总问题数: {pet_analysis['总问题数']}")
        
        if "按日志状态统计" in pet_analysis:
            print("\n日志状态分布:")
            for status, count in pet_analysis["按日志状态统计"].items():
                print(f"  {status:12s}: {count:4d}")
        
        if "按模态统计" in pet_analysis:
            print("\n按模态分布:")
            for modality, count in pet_analysis["按模态统计"].items():
                print(f"  {modality:12s}: {count:4d}")
        
        if "主要失败原因" in pet_analysis and pet_analysis["主要失败原因"]:
            print("\n主要失败原因:")
            for reason, count in pet_analysis["主要失败原因"].items():
                print(f"  {reason:30s}: {count:4d}")
        
        if "无日志记录数" in pet_analysis:
            print(f"\n无日志记录: {pet_analysis['无日志记录数']} 个")
            if "无日志记录按模态" in pet_analysis:
                for modality, count in pet_analysis["无日志记录按模态"].items():
                    print(f"  {modality:12s}: {count:4d}")
    
    # 保存报告
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n分析报告已保存到: {args.output}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
