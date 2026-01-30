#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

PAIRS = Path("/mnt/nfsdata/nfsdata/lsj.14/replicaLT/adapter_finetune/data_csv/pairs0106_filtered.csv")
UPENN = Path("/mnt/nfsdata/nfsdata/lsj.14/replicaLT/adapter_finetune/data_csv/PLASMA_UPENN_ADNI3_4.0103.csv")


def main():
    pairs = pd.read_csv(PAIRS)
    pairs.columns = [c.lower() for c in pairs.columns]
    pairs["examdate"] = pd.to_datetime(pairs["examdate"], errors="coerce")

    up = pd.read_csv(UPENN, low_memory=False)
    up.columns = [c.lower() for c in up.columns]
    up["examdate"] = pd.to_datetime(up["examdate"], errors="coerce")

    # merge on ptid
    merged = pairs.merge(up[["ptid", "examdate"]], on="ptid", how="left", suffixes=("_pair", "_plasma"))
    merged["diff_days"] = (merged["examdate_plasma"] - merged["examdate_pair"]).abs().dt.days

    total = len(merged)
    have_plasma = merged["examdate_plasma"].notna().sum()
    within_90 = (merged["diff_days"] <= 90).sum()
    within_180 = (merged["diff_days"] <= 180).sum()

    print(f"Total pairs rows: {total}")
    print(f"Rows with any UPENN record: {have_plasma}")
    print(f"Within 90 days: {within_90}")
    print(f"Within 180 days: {within_180}")

    # Show a few with large gaps
    worst = merged.sort_values("diff_days", ascending=False).head(10)
    print("\nTop 10 largest gaps:")
    print(worst[["ptid", "examdate_pair", "examdate_plasma", "diff_days"]].to_string(index=False))


if __name__ == "__main__":
    main()
