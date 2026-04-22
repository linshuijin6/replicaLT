#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Match MRI to PET exams (FDG, AV45, AV1451/TAU) within a specified time window.

Usage:
    python match_mri_pet_pairs.py --mri MRIT1.0114.csv --pet PET.0114.csv --max-days 180 --output pairs.csv
"""

from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match MRI to PET exams (FDG, AV45, AV1451) within a specified time window."
    )
    parser.add_argument(
        "--mri",
        type=Path,
        required=True,
        help="Path to MRI CSV file (expects columns: subject_id, image_id, image_date).",
    )
    parser.add_argument(
        "--pet",
        type=Path,
        required=True,
        help="Path to PET CSV file (expects columns: subject_id, image_id, image_date, radiopharmaceutical).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("adapter_finetune/mri_pet_pairs_matched.csv"),
        help="Output CSV file path for the matched pairs.",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=180,
        help="Maximum allowed time difference in days between MRI and PET scans (default: 180).",
    )
    parser.add_argument(
        "--mri-phases",
        nargs="+",
        default=["ADNI3", "ADNI4"],
        help="MRI protocol phases to include (default: ADNI3 ADNI4).",
    )
    parser.add_argument(
        "--series-type",
        default="T1w",
        help="MRI series type to filter (default: T1w).",
    )
    parser.add_argument(
        "--check-diagnosis",
        action="store_true",
        help="Check diagnosis consistency between MRI and PET. Exclude pairs with different diagnoses.",
    )
    parser.add_argument(
        "--dxsum",
        type=Path,
        default=Path("/home/ssddata/linshuijin/replicaLT/adapter_finetune/DXSUM_25Dec2025.csv"),
        help="Path to DXSUM CSV file for diagnosis lookup (default: adapter_finetune/DXSUM_25Dec2025.csv).",
    )
    parser.add_argument(
        "--diagnosis-max-days",
        type=int,
        default=180,
        help="Maximum allowed time difference in days for diagnosis lookup (default: 180).",
    )
    return parser.parse_args()


def load_diagnosis_data(dxsum_path: Path) -> pd.DataFrame:
    """
    加载 DXSUM 数据并创建诊断查找表。
    
    返回包含 PTID, EXAMDATE, DIAGNOSIS 的数据框。
    """
    print(f"\nLoading diagnosis data from {dxsum_path}...")
    dxsum = pd.read_csv(dxsum_path)
    print(f"  Original DXSUM records: {len(dxsum)}")
    
    # 只保留需要的列
    required_cols = ["PTID", "EXAMDATE", "DIAGNOSIS"]
    dxsum = dxsum[required_cols].copy()
    
    # 转换日期
    dxsum["EXAMDATE_dt"] = pd.to_datetime(dxsum["EXAMDATE"], errors="coerce")
    dxsum = dxsum.dropna(subset=["PTID", "EXAMDATE_dt", "DIAGNOSIS"])
    print(f"  Valid DXSUM records: {len(dxsum)}")
    
    return dxsum


def find_diagnosis_fallback(
    ptid: str, exam_date: pd.Timestamp, dxsum: pd.DataFrame
) -> Optional[int]:
    """
    回退方案：当在 max_days 范围内找不到诊断时，尝试获取 examdate 前后的两个诊断。
    如果这两个诊断相同，则返回该诊断。
    
    参数:
        ptid: 受试者 ID
        exam_date: 检查日期
        dxsum: DXSUM 数据框
    
    返回:
        诊断代码 (1=正常, 2=MCI, 3=AD) 或 None
    """
    if pd.isna(exam_date):
        return None
    
    # 获取该受试者的所有诊断记录
    subject_dx = dxsum[dxsum["PTID"] == ptid].copy()
    if subject_dx.empty:
        return None
    
    # 分离 exam_date 之前和之后的记录
    before = subject_dx[subject_dx["EXAMDATE_dt"] < exam_date]
    after = subject_dx[subject_dx["EXAMDATE_dt"] > exam_date]
    
    # 获取最近的前一次和后一次诊断
    dx_before = None
    dx_after = None
    
    if not before.empty:
        # 获取 exam_date 之前最近的诊断
        closest_before = before.loc[before["EXAMDATE_dt"].idxmax()]
        dx_before = int(closest_before["DIAGNOSIS"])
    
    if not after.empty:
        # 获取 exam_date 之后最近的诊断
        closest_after = after.loc[after["EXAMDATE_dt"].idxmin()]
        dx_after = int(closest_after["DIAGNOSIS"])
    
    # 如果前后诊断都存在且相同，返回该诊断
    if dx_before is not None and dx_after is not None and dx_before == dx_after:
        return dx_before
    
    return None


def find_diagnosis(
    ptid: str, exam_date: pd.Timestamp, dxsum: pd.DataFrame, max_days: int = 90,
    use_fallback: bool = True
) -> Optional[int]:
    """
    根据 PTID 和 EXAMDATE 查找诊断。
    
    如果找不到完全匹配的日期，会在 max_days 范围内查找最接近的记录。
    如果仍找不到，且 use_fallback=True，则尝试回退方案：
    获取 examdate 前后的两个诊断，如果相同则使用该诊断。
    
    参数:
        ptid: 受试者 ID
        exam_date: 检查日期
        dxsum: DXSUM 数据框
        max_days: 最大允许的日期差异（天），默认 90 天
        use_fallback: 是否使用回退方案，默认 True
    
    返回:
        诊断代码 (1=正常, 2=MCI, 3=AD) 或 None
    """
    if pd.isna(exam_date):
        return None
    
    # 获取该受试者的所有诊断记录
    subject_dx = dxsum[dxsum["PTID"] == ptid]
    if subject_dx.empty:
        return None
    
    # 计算日期差异
    subject_dx = subject_dx.copy()
    subject_dx["date_diff"] = (subject_dx["EXAMDATE_dt"] - exam_date).abs().dt.days
    
    # 查找最接近的记录
    closest = subject_dx[subject_dx["date_diff"] <= max_days]
    if closest.empty:
        # 如果在 max_days 内找不到，尝试回退方案
        if use_fallback:
            return find_diagnosis_fallback(ptid, exam_date, dxsum)
        return None
    
    # 返回最接近的诊断
    best_match = closest.loc[closest["date_diff"].idxmin()]
    return int(best_match["DIAGNOSIS"])


def _pick_closest(
    pet_rows: pd.DataFrame, mri_date: pd.Timestamp, max_days: int
) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    从给定的 PET 行中选择与 MRI 日期最接近的一条记录。
    
    策略：
    1. 计算每条 PET 记录与 MRI 日期的天数差（绝对值）
    2. 过滤掉超过 max_days 的记录
    3. 如果有多条满足条件的记录，选择最新的（日期最晚的）
    4. 如果日期相同，选择 image_id 最小的
    
    返回: (image_id, image_date, delta_days)
    """
    if pd.isna(mri_date) or pet_rows.empty:
        return None, None, None

    pet_rows = pet_rows.copy()
    pet_rows["delta_days"] = (pet_rows["image_date_dt"] - mri_date).abs().dt.days
    pet_rows = pet_rows[pet_rows["delta_days"] <= max_days]
    
    if pet_rows.empty:
        return None, None, None

    # 按照天数差、日期降序（最新的在前）排序
    # 这样可以在天数差相同时，优先选择最新的 PET 扫描
    row = pet_rows.sort_values(["delta_days", "image_date_dt"], ascending=[True, False]).iloc[0]
    return str(row["image_id"]), row["image_date"], int(row["delta_days"])


def _remove_repeat_when_duplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    移除同一受试者在同一日期进行的重复扫描（标记为 repeat 的记录）。
    """
    if "series_description" not in df.columns:
        return df
        
    df = df.copy()
    group_sizes = df.groupby(["subject_id", "image_date"]).size()
    dup_keys = set(group_sizes[group_sizes > 1].index)
    
    if not dup_keys:
        return df

    df["_dup"] = df.apply(lambda r: (r["subject_id"], r["image_date"]) in dup_keys, axis=1)
    df["_repeat"] = df["series_description"].str.contains("repeat", case=False, na=False)
    cleaned = df[~(df["_dup"] & df["_repeat"])].drop(columns=["_dup", "_repeat"])
    return cleaned


def load_filtered(
    mri_path: Path, pet_path: Path, mri_phases: list, series_type: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载并过滤 MRI 和 PET 数据。
    
    MRI 过滤：
    - 只保留指定的 mri_protocol_phase（如果列存在）
    - 只保留指定的 series_type（如果列存在）
    - 移除重复扫描中标记为 "repeat" 的记录
    
    PET 过滤：
    - 只保留 FDG、AV45、AV1451 三种示踪剂
    """
    PET_TRACERS = {"18F-FDG", "18F-AV45", "18F-AV1451"}
    
    print(f"Loading MRI data from {mri_path}...")
    mri = pd.read_csv(mri_path)
    print(f"  Original MRI records: {len(mri)}")

    # 过滤 MRI：检查列是否存在
    if "mri_protocol_phase" in mri.columns:
        mri = mri[mri["mri_protocol_phase"].isin(mri_phases)]
        print(f"  After phase filter: {len(mri)}")
    
    if "series_type" in mri.columns:
        mri = mri[mri["series_type"] == series_type]
        print(f"  After series type filter: {len(mri)}")
    
    mri = _remove_repeat_when_duplicate(mri)
    print(f"  After removing repeats: {len(mri)}")
    
    mri["image_date_dt"] = pd.to_datetime(mri["image_date"], errors="coerce")
    mri = mri.dropna(subset=["subject_id", "image_id", "image_date_dt"])
    print(f"  After removing invalid dates: {len(mri)}")

    # 过滤 PET
    print(f"\nLoading PET data from {pet_path}...")
    pet = pd.read_csv(pet_path)
    print(f"  Original PET records: {len(pet)}")
    
    pet = pet[pet["radiopharmaceutical"].isin(PET_TRACERS)].copy()
    print(f"  After tracer filter: {len(pet)}")
    
    pet["image_date_dt"] = pd.to_datetime(pet["image_date"], errors="coerce")
    pet = pet.dropna(subset=["subject_id", "image_id", "image_date_dt", "radiopharmaceutical"])
    print(f"  After removing invalid dates: {len(pet)}")
    
    return mri, pet


def match_records(
    mri: pd.DataFrame, 
    pet: pd.DataFrame, 
    max_days: int,
    check_diagnosis: bool = False,
    dxsum: Optional[pd.DataFrame] = None,
    diagnosis_max_days: int = 180
) -> pd.DataFrame:
    """
    为每条 MRI 记录匹配对应的 PET 记录（FDG、AV45、AV1451）。
    
    匹配原则：
    1. subject_id 必须完全相同
    2. image_date 在 max_days 天数范围内
    3. 如果有多条匹配，选择最新的（日期最晚的）
    4. (可选) 检查诊断一致性，排除诊断不同的配对
    
    参数：
        diagnosis_max_days: 诊断查找的最大日期差（天）
    
    输出列：PTID, EXAMDATE, DIAGNOSIS, id_mri, id_fdg, id_av45, id_av1451
    """
    print(f"\nMatching MRI to PET records (max_days={max_days})...")
    
    records = []
    for idx, row in mri.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(mri)} MRI records...", end="\r")
            
        subject_id = row["subject_id"]
        mri_date = row["image_date_dt"]
        mri_date_str = row["image_date"]

        # 获取该受试者的所有 PET 记录
        sub_pet = pet[pet["subject_id"] == subject_id]
        
        # 为每种示踪剂找到最接近的记录
        fdg_id, fdg_date, fdg_delay = _pick_closest(
            sub_pet[sub_pet["radiopharmaceutical"] == "18F-FDG"], mri_date, max_days
        )
        av45_id, av45_date, av45_delay = _pick_closest(
            sub_pet[sub_pet["radiopharmaceutical"] == "18F-AV45"], mri_date, max_days
        )
        av1451_id, av1451_date, av1451_delay = _pick_closest(
            sub_pet[sub_pet["radiopharmaceutical"] == "18F-AV1451"], mri_date, max_days
        )

        # 计算最小延迟天数（用于参考）
        delays = [d for d in [fdg_delay, av45_delay, av1451_delay] if d is not None]
        min_delay = min(delays) if delays else None

        # 获取 MRI 的诊断（始终获取，用于输出）
        mri_dx = None
        if dxsum is not None:
            mri_dx = find_diagnosis(subject_id, mri_date, dxsum, max_days=diagnosis_max_days)

        # 如果需要检查诊断一致性
        if check_diagnosis and dxsum is not None and mri_dx is not None:
            # 获取 PET 的诊断（针对每种示踪剂）
            fdg_dx = None
            av45_dx = None
            av1451_dx = None
            
            if fdg_date:
                fdg_date_dt = pd.to_datetime(fdg_date)
                fdg_dx = find_diagnosis(subject_id, fdg_date_dt, dxsum, max_days=diagnosis_max_days)
            if av45_date:
                av45_date_dt = pd.to_datetime(av45_date)
                av45_dx = find_diagnosis(subject_id, av45_date_dt, dxsum, max_days=diagnosis_max_days)
            if av1451_date:
                av1451_date_dt = pd.to_datetime(av1451_date)
                av1451_dx = find_diagnosis(subject_id, av1451_date_dt, dxsum, max_days=diagnosis_max_days)
            
            # 检查诊断一致性，如果不一致则设置为 None
            if fdg_id and fdg_dx is not None:
                if mri_dx != fdg_dx:
                    fdg_id = None
                    fdg_date = None
                    fdg_delay = None
            
            if av45_id and av45_dx is not None:
                if mri_dx != av45_dx:
                    av45_id = None
                    av45_date = None
                    av45_delay = None
            
            if av1451_id and av1451_dx is not None:
                if mri_dx != av1451_dx:
                    av1451_id = None
                    av1451_date = None
                    av1451_delay = None
            
            # 重新计算最小延迟
            delays = [d for d in [fdg_delay, av45_delay, av1451_delay] if d is not None]
            min_delay = min(delays) if delays else None

        records.append(
            {
                "PTID": subject_id,  # 使用 PTID 作为列名
                "EXAMDATE": mri_date_str,  # 使用 EXAMDATE 作为列名（MRI 日期）
                "DIAGNOSIS": mri_dx,  # MRI 检查日期对应的诊断
                "id_mri": str(row["image_id"]),
                "id_fdg": fdg_id,
                "id_av45": av45_id,
                "id_av1451": av1451_id,
                "date_fdg": fdg_date,
                "date_av45": av45_date,
                "date_av1451": av1451_date,
                "min_delay_days": min_delay,
            }
        )

    print(f"  Processed {len(mri)}/{len(mri)} MRI records.    ")
    
    result = pd.DataFrame.from_records(records)
    result = result.sort_values(["PTID", "EXAMDATE", "id_mri"])
    return result


def main() -> None:
    args = parse_args()
    
    print("=" * 70)
    print("MRI-PET Matching Pipeline")
    print("=" * 70)
    print(f"MRI file: {args.mri}")
    print(f"PET file: {args.pet}")
    print(f"Max days: {args.max_days}")
    print(f"MRI phases: {args.mri_phases}")
    print(f"Series type: {args.series_type}")
    print(f"Check diagnosis: {args.check_diagnosis}")
    print(f"DXSUM file: {args.dxsum}")
    print(f"Diagnosis max days: {args.diagnosis_max_days}")
    print(f"Output: {args.output}")
    print("=" * 70)
    
    # 加载并过滤数据
    mri, pet = load_filtered(args.mri, args.pet, args.mri_phases, args.series_type)
    
    # 加载诊断数据（始终加载，用于输出 DIAGNOSIS 字段）
    dxsum = None
    if args.dxsum.exists():
        dxsum = load_diagnosis_data(args.dxsum)
    else:
        print(f"\n警告: DXSUM 文件不存在: {args.dxsum}，DIAGNOSIS 字段将为空")
    
    # 匹配记录
    matches = match_records(mri, pet, args.max_days, args.check_diagnosis, dxsum, args.diagnosis_max_days)
    
    # 保存结果
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # 只保留核心列到最终输出
    output_columns = ["PTID", "EXAMDATE", "DIAGNOSIS", "id_mri", "id_fdg", "id_av45", "id_av1451"]
    output_df = matches[output_columns].copy()
    
    # 将 id_x 和 DIAGNOSIS 字段转换为字符串类型，避免被 pandas 转换为 float
    id_columns = ["id_mri", "id_fdg", "id_av45", "id_av1451"]
    for col in id_columns:
        # 将非空值转为字符串，空值保持为空
        output_df[col] = output_df[col].apply(lambda x: str(int(float(x))) if pd.notna(x) else '')
    
    # DIAGNOSIS 转换为字符串（整数形式）
    output_df["DIAGNOSIS"] = output_df["DIAGNOSIS"].apply(lambda x: str(int(x)) if pd.notna(x) else '')
    
    output_df.to_csv(args.output, index=False)
    
    # 打印统计信息
    print("\n" + "=" * 70)
    print("Matching Results Summary")
    print("=" * 70)
    print(f"Total MRI records processed: {len(mri)}")
    print(f"Total paired records: {len(matches)}")
    
    matched_counts = matches[["id_fdg", "id_av45", "id_av1451"]].notna().sum()
    print("\nMatched counts by tracer:")
    print(f"  FDG:    {matched_counts['id_fdg']:5d} ({100*matched_counts['id_fdg']/len(matches):.1f}%)")
    print(f"  AV45:   {matched_counts['id_av45']:5d} ({100*matched_counts['id_av45']/len(matches):.1f}%)")
    print(f"  AV1451: {matched_counts['id_av1451']:5d} ({100*matched_counts['id_av1451']/len(matches):.1f}%)")
    
    # 统计至少有一个 PET 匹配的记录
    has_any_pet = matches[["id_fdg", "id_av45", "id_av1451"]].notna().any(axis=1).sum()
    print(f"\nRecords with at least one PET match: {has_any_pet} ({100*has_any_pet/len(matches):.1f}%)")
    print(f"Records with no PET match: {len(matches) - has_any_pet} ({100*(len(matches)-has_any_pet)/len(matches):.1f}%)")
    
    # 统计 DIAGNOSIS 缺失数目
    diagnosis_missing = matches["DIAGNOSIS"].isna().sum()
    diagnosis_found = matches["DIAGNOSIS"].notna().sum()
    print(f"\nDiagnosis statistics:")
    print(f"  With diagnosis:    {diagnosis_found:5d} ({100*diagnosis_found/len(matches):.1f}%)")
    print(f"  Missing diagnosis: {diagnosis_missing:5d} ({100*diagnosis_missing/len(matches):.1f}%)")
    
    print(f"\n✓ Output saved to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
