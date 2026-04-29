#!/usr/bin/env python3
"""
Phase F: Cognitive / biomarker correlation analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from comparison_common import (
    EXTENDED_OUT_DIR,
    METHOD_ORDER,
    bootstrap_ci,
    build_roi_masks,
    extract_roi_values,
    get_method_label,
    load_aligned_volumes,
    load_atlas,
    load_subject_metadata,
    load_val43_names,
)

ROI_NAME = "Temporal_metaROI"


def _safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan, mask
    res = stats.spearmanr(x[mask], y[mask])
    return float(res.statistic), float(res.pvalue), mask


def _bootstrap_spearman(x, y, n_boot=2000):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(42)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(len(x), size=len(x), replace=True)
        vals.append(stats.spearmanr(x[idx], y[idx]).statistic)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _plot_scatter(df, out_path):
    clinical_vars = ["MMSE", "ADAS13"]
    methods = df[df["clinical_var"].isin(clinical_vars)]["feature_source"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(methods), len(clinical_vars), figsize=(10, 3.2 * len(methods)))
    if len(methods) == 1:
        axes = np.array([axes])
    for i, method in enumerate(methods):
        for j, var in enumerate(clinical_vars):
            ax = axes[i, j]
            sub = df[(df["feature_source"] == method) & (df["clinical_var"] == var)]
            if sub.empty or "x_values" not in sub.columns:
                ax.axis("off")
                continue
            x = np.asarray(sub.iloc[0]["x_values"], dtype=float)
            y = np.asarray(sub.iloc[0]["y_values"], dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            ax.scatter(x[mask], y[mask], s=18, alpha=0.75)
            ax.set_title(f"{method} vs {var}")
            ax.set_xlabel(ROI_NAME)
            ax.set_ylabel(var)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_correlation(method_ids=None, subjects=None):
    if method_ids is None:
        method_ids = METHOD_ORDER
    if subjects is None:
        subjects = load_val43_names()

    atlas = load_atlas()
    roi_masks = build_roi_masks(atlas)
    meta = load_subject_metadata().set_index("name")
    meta = meta.loc[[s for s in subjects if s in meta.index]].copy()
    volumes = load_aligned_volumes(subjects=meta.index.tolist(), method_ids=method_ids)

    source_vectors = {
        "Real PET": np.array([extract_roi_values(volumes[method_ids[0]][sid]["gt"], roi_masks)[ROI_NAME] for sid in meta.index], dtype=float)
    }
    for method_id in method_ids:
        source_vectors[get_method_label(method_id)] = np.array([
            extract_roi_values(volumes[method_id][sid]["pred"], roi_masks)[ROI_NAME] for sid in meta.index
        ], dtype=float)

    rows = []
    plot_rows = []
    for feature_source, uptake in source_vectors.items():
        source_id = "gt" if feature_source == "Real PET" else next((m for m in method_ids if get_method_label(m) == feature_source), feature_source)
        for clinical_var in ["MMSE", "ADAS13"]:
            y = meta[clinical_var].to_numpy(dtype=float)
            rho, p, mask = _safe_spearman(uptake, y)
            lo, hi = _bootstrap_spearman(uptake, y)
            rows.append({
                "feature_source": feature_source,
                "source_id": source_id,
                "clinical_var": clinical_var,
                "roi": ROI_NAME,
                "spearman_rho": rho,
                "rho_lo": lo,
                "rho_hi": hi,
                "p_value": p,
                "n": int(mask.sum()),
            })
            plot_rows.append({
                "feature_source": feature_source,
                "clinical_var": clinical_var,
                "x_values": uptake.tolist(),
                "y_values": y.tolist(),
            })

        apoe = meta["APOE4"].to_numpy(dtype=float)
        carriers = uptake[apoe >= 1]
        non_carriers = uptake[apoe == 0]
        if len(carriers) and len(non_carriers):
            pval = stats.mannwhitneyu(carriers, non_carriers, alternative="two-sided").pvalue
            rows.append({
                "feature_source": feature_source,
                "source_id": source_id,
                "clinical_var": "APOE4_stratification",
                "roi": ROI_NAME,
                "p_value": float(pval),
                "n": int(np.isfinite(apoe).sum()),
                "carrier_mean": float(np.nanmean(carriers)),
                "noncarrier_mean": float(np.nanmean(non_carriers)),
                "n_carrier": int(len(carriers)),
                "n_noncarrier": int(len(non_carriers)),
            })

    out_df = pd.DataFrame(rows)
    EXTENDED_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = EXTENDED_OUT_DIR / "correlation_analysis.csv"
    out_df.to_csv(out_csv, index=False)
    _plot_scatter(pd.DataFrame(plot_rows), EXTENDED_OUT_DIR / "correlation_scatter_plots.png")
    print(f"Saved correlation results to {out_csv}")
    return out_df


if __name__ == "__main__":
    run_correlation()
