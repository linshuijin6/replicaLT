from __future__ import annotations

from pathlib import Path

import pandas as pd

MRI_CSV = Path("adapter_finetune/MRI_Sub_0102.csv")
PET_CSV = Path("adapter_finetune/PET_Sub_0102.csv")
PAIRS_CSV = Path("adapter_finetune/mri_pet_pairs_matched.csv")

OUT_DIR = Path("adapter_finetune/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MRI_PHASES = {"ADNI3", "ADNI4"}
PET_TRACERS = {"18F-FDG", "18F-AV45", "18F-AV1451"}
MAX_DAYS = 180


def _series_is_repeat(series_desc: pd.Series) -> pd.Series:
    return series_desc.astype(str).str.contains("repeat", case=False, na=False)


def summarize_delay(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce")
    s = s.dropna()
    if s.empty:
        return {"n": 0}

    q = s.quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).to_dict()
    return {
        "n": int(s.shape[0]),
        "min": float(q.get(0.0)),
        "p10": float(q.get(0.1)),
        "p25": float(q.get(0.25)),
        "median": float(q.get(0.5)),
        "p75": float(q.get(0.75)),
        "p90": float(q.get(0.9)),
        "max": float(q.get(1.0)),
        "mean": float(s.mean()),
    }


def main() -> None:
    mri_all = pd.read_csv(MRI_CSV)
    pet_all = pd.read_csv(PET_CSV)
    pairs = pd.read_csv(PAIRS_CSV)

    # ---------------- MRI filtering breakdown ----------------
    mri_all["_is_phase_ok"] = mri_all["mri_protocol_phase"].isin(MRI_PHASES)
    mri_all["_is_t1w"] = mri_all["series_type"] == "T1w"
    mri_phase_t1 = mri_all[mri_all["_is_phase_ok"] & mri_all["_is_t1w"]].copy()

    # duplicates by subject_id + image_date
    group_sizes = mri_phase_t1.groupby(["subject_id", "image_date"]).size()
    dup_keys = set(group_sizes[group_sizes > 1].index)

    mri_phase_t1["_dup_key"] = list(zip(mri_phase_t1["subject_id"], mri_phase_t1["image_date"]))
    mri_phase_t1["_is_dup"] = mri_phase_t1["_dup_key"].isin(dup_keys)
    mri_phase_t1["_is_repeat"] = _series_is_repeat(mri_phase_t1["series_description"])

    mri_repeat_removed = mri_phase_t1[mri_phase_t1["_is_dup"] & mri_phase_t1["_is_repeat"]].copy()
    mri_filtered = mri_phase_t1[~(mri_phase_t1["_is_dup"] & mri_phase_t1["_is_repeat"])].copy()

    # Entries filtered out at each stage
    mri_out_not_phase = mri_all[~mri_all["_is_phase_ok"]].copy()
    mri_out_not_t1w = mri_all[mri_all["_is_phase_ok"] & ~mri_all["_is_t1w"]].copy()

    # After repeat removal: remaining duplicates (kept by spec)
    group_sizes_after = mri_filtered.groupby(["subject_id", "image_date"]).size()
    remaining_dup_groups = group_sizes_after[group_sizes_after > 1]

    # Save filtered-out MRI rows
    mri_out_not_phase.to_csv(OUT_DIR / "mri_filtered_out_not_adni3or4.csv", index=False)
    mri_out_not_t1w.to_csv(OUT_DIR / "mri_filtered_out_not_t1w.csv", index=False)
    mri_repeat_removed.to_csv(OUT_DIR / "mri_removed_repeat_when_duplicate.csv", index=False)
    # Save final filtered MRI
    mri_filtered.drop(columns=["_dup_key", "_is_dup", "_is_repeat"], errors="ignore").to_csv(
        OUT_DIR / "mri_after_filters.csv", index=False
    )

    # ---------------- PET filtering breakdown ----------------
    pet_all["_is_tracer_ok"] = pet_all["radiopharmaceutical"].isin(PET_TRACERS)
    pet_filtered = pet_all[pet_all["_is_tracer_ok"]].copy()
    pet_out = pet_all[~pet_all["_is_tracer_ok"]].copy()

    pet_filtered.to_csv(OUT_DIR / "pet_after_tracer_filter.csv", index=False)
    pet_out.to_csv(OUT_DIR / "pet_filtered_out_other_tracers.csv", index=False)

    # counts per tracer
    tracer_counts_all = pet_all["radiopharmaceutical"].value_counts(dropna=False)
    tracer_counts_kept = pet_filtered["radiopharmaceutical"].value_counts(dropna=False)

    # ---------------- Pair CSV stats ----------------
    total_pairs = int(pairs.shape[0])
    has_any_pet = pairs[["image_id_fdg", "image_id_av45", "image_id_av1451"]].notna().any(axis=1)
    has_fdg = pairs["image_id_fdg"].notna()
    has_av45 = pairs["image_id_av45"].notna()
    has_av1451 = pairs["image_id_av1451"].notna()

    delay_stats_all = summarize_delay(pairs["delay"])

    # Delay buckets
    delay = pd.to_numeric(pairs["delay"], errors="coerce")
    bins = [-0.1, 0, 7, 30, 90, 180]
    labels = ["0", "1-7", "8-30", "31-90", "91-180"]
    delay_bucket = pd.cut(delay, bins=bins, labels=labels)
    bucket_counts = delay_bucket.value_counts(dropna=False).sort_index()

    # ---------------- Matching sanity check vs 180 days ----------------
    # Ensure that chosen PET dates are within 180 days when present.
    pairs_dt = pairs.copy()
    pairs_dt["date_mri_dt"] = pd.to_datetime(pairs_dt["date_mri"], errors="coerce")
    for col in ["date_fdg", "date_av45", "date_av1451"]:
        pairs_dt[col + "_dt"] = pd.to_datetime(pairs_dt[col], errors="coerce")

    def within_180(mri_dt: pd.Series, pet_dt: pd.Series) -> pd.Series:
        d = (pet_dt - mri_dt).abs().dt.days
        return d.le(MAX_DAYS)

    sanity_fdg = within_180(pairs_dt["date_mri_dt"], pairs_dt["date_fdg_dt"]) | pairs_dt["date_fdg_dt"].isna()
    sanity_av45 = within_180(pairs_dt["date_mri_dt"], pairs_dt["date_av45_dt"]) | pairs_dt["date_av45_dt"].isna()
    sanity_av1451 = within_180(pairs_dt["date_mri_dt"], pairs_dt["date_av1451_dt"]) | pairs_dt["date_av1451_dt"].isna()
    sanity_ok = sanity_fdg & sanity_av45 & sanity_av1451

    # ---------------- Print report ----------------
    print("=== MRI 过滤统计 ===")
    print(f"MRI 总行数: {len(mri_all)}")
    print(f"- phase 不在 ADNI3/ADNI4 被剔除: {len(mri_out_not_phase)}")
    print(f"- 在 ADNI3/ADNI4 但 series_type != T1w 被剔除: {len(mri_out_not_t1w)}")
    print(f"- ADNI3/4 且 T1w 暂存行数: {len(mri_phase_t1)}")
    print(f"- 其中 (subject_id, image_date) 重复组数: {len(dup_keys)}")
    print(f"- 重复组内含 Repeat 条目被删除行数: {len(mri_repeat_removed)}")
    print(f"- 最终 MRI 行数(用于配对): {len(mri_filtered)}")
    if not remaining_dup_groups.empty:
        print(f"- 最终仍存在重复 (subject_id, image_date) 组数(按规则保留): {len(remaining_dup_groups)}")
        print("  (这些组里没有包含 REPEAT，或 REPEAT 已被删后仍多于1条)")

    print("\n=== PET 过滤统计 ===")
    print(f"PET 总行数: {len(pet_all)}")
    print(f"- radiopharmaceutical 不在 {sorted(PET_TRACERS)} 被剔除: {len(pet_out)}")
    print(f"- 最终 PET 行数(用于配对): {len(pet_filtered)}")
    print("- PET radiopharmaceutical 分布(全部 top 10):")
    print(tracer_counts_all.head(10))
    print("- PET radiopharmaceutical 分布(保留):")
    print(tracer_counts_kept)

    print("\n=== 配组结果统计 (按最终CSV) ===")
    print(f"配组表行数(MRI条目数): {total_pairs}")
    print(f"- 任意一种PET命中: {int(has_any_pet.sum())} ({has_any_pet.mean()*100:.2f}%)")
    print(f"- FDG 命中: {int(has_fdg.sum())} ({has_fdg.mean()*100:.2f}%)")
    print(f"- AV45 命中: {int(has_av45.sum())} ({has_av45.mean()*100:.2f}%)")
    print(f"- AV1451 命中: {int(has_av1451.sum())} ({has_av1451.mean()*100:.2f}%)")
    print(f"- 完全无PET匹配: {int((~has_any_pet).sum())} ({((~has_any_pet).mean()*100):.2f}%)")

    print("\nDelay(最小匹配天数差) 统计(仅对有匹配者):")
    print(delay_stats_all)
    print("Delay 分桶计数(含 NaN):")
    print(bucket_counts)

    print("\n=== 180天窗口 sanity check ===")
    bad = int((~sanity_ok).sum())
    print(f"存在超出180天的已填配对? {bad} 行")
    if bad > 0:
        pairs_dt.loc[~sanity_ok].to_csv(OUT_DIR / "pairs_sanity_bad_over_180_days.csv", index=False)
        print(f"已导出异常行: {OUT_DIR / 'pairs_sanity_bad_over_180_days.csv'}")

    # Save summary tables
    summary = {
        "mri_total": len(mri_all),
        "mri_out_not_phase": len(mri_out_not_phase),
        "mri_out_not_t1w": len(mri_out_not_t1w),
        "mri_phase_t1": len(mri_phase_t1),
        "mri_dup_groups": len(dup_keys),
        "mri_repeat_removed": len(mri_repeat_removed),
        "mri_final": len(mri_filtered),
        "pet_total": len(pet_all),
        "pet_out_other_tracers": len(pet_out),
        "pet_final": len(pet_filtered),
        "pairs_rows": total_pairs,
        "pairs_has_any_pet": int(has_any_pet.sum()),
        "pairs_has_fdg": int(has_fdg.sum()),
        "pairs_has_av45": int(has_av45.sum()),
        "pairs_has_av1451": int(has_av1451.sum()),
        "pairs_no_pet": int((~has_any_pet).sum()),
    }
    pd.Series(summary).to_csv(OUT_DIR / "summary_counts.csv")


if __name__ == "__main__":
    main()
