#!/usr/bin/env python3
"""
Phase E: Robustness analysis.

Produces diagnosis- and vendor-stratified reconstruction summaries from the base
per-subject comparison metrics, plus vendor-level Kruskal-Wallis tests.
"""

import pandas as pd
from scipy import stats

from comparison_common import EXTENDED_OUT_DIR, load_subject_metadata, load_vendor_info


METRICS = ["ssim", "psnr", "mae"]


def run_robustness() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_subject = pd.read_csv(EXTENDED_OUT_DIR.parent / "per_subject_metrics.csv")
    meta = load_subject_metadata()[["name", "diagnosis"]].rename(columns={"name": "subject"})
    vendor = load_vendor_info().rename(columns={"name": "subject"})

    merged = per_subject.merge(meta, on="subject", how="left").merge(vendor[["subject", "vendor"]], on="subject", how="left")

    by_diag = (
        merged.groupby(["method", "diagnosis"], dropna=False)
        .agg(
            n=("subject", "count"),
            ssim_mean=("ssim", "mean"), ssim_std=("ssim", "std"),
            psnr_mean=("psnr", "mean"), psnr_std=("psnr", "std"),
            mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        )
        .reset_index()
        .rename(columns={"diagnosis": "stratum"})
    )

    by_vendor = (
        merged.dropna(subset=["vendor"])
        .groupby(["method", "vendor"], dropna=False)
        .agg(
            n=("subject", "count"),
            ssim_mean=("ssim", "mean"), ssim_std=("ssim", "std"),
            psnr_mean=("psnr", "mean"), psnr_std=("psnr", "std"),
            mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        )
        .reset_index()
    )

    kw_rows = []
    for method, sub_df in merged.dropna(subset=["vendor"]).groupby("method"):
        for metric in METRICS:
            groups = [g[metric].dropna().values for _, g in sub_df.groupby("vendor") if len(g[metric].dropna()) > 0]
            if len(groups) >= 2:
                stat, p = stats.kruskal(*groups)
            else:
                stat, p = float("nan"), float("nan")
            kw_rows.append({"method": method, "metric": metric, "kruskal_stat": stat, "p_value": p})
    kw_df = pd.DataFrame(kw_rows)

    EXTENDED_OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_diag.to_csv(EXTENDED_OUT_DIR / "robustness_by_diagnosis.csv", index=False)
    by_vendor.to_csv(EXTENDED_OUT_DIR / "robustness_by_vendor.csv", index=False)
    kw_df.to_csv(EXTENDED_OUT_DIR / "robustness_kruskal_wallis.csv", index=False)
    print(f"Saved robustness outputs to {EXTENDED_OUT_DIR}")
    return by_diag, by_vendor, kw_df


if __name__ == "__main__":
    run_robustness()
