#!/usr/bin/env python3
"""
Phase B: Clinical fidelity analysis for MRI -> TAU PET comparison.

Computes regional normalized uptake consistency metrics across the 43-subject
comparison cohort using aligned synthetic/real PET volumes and atlas-defined ROIs.
"""

from pathlib import Path

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
    load_val43_names,
)


DEFAULT_ROIS = [
    "Global_cortical",
    "Temporal_metaROI",
    "Braak_I_II",
    "Braak_III_IV",
    "Braak_V_VI",
    "Entorhinal_proxy",
    "Hippocampus",
    "Amygdala",
]


def _safe_corr(func, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.allclose(x[mask], x[mask][0]) or np.allclose(y[mask], y[mask][0]):
        return np.nan, np.nan
    try:
        res = func(x[mask], y[mask])
        if hasattr(res, "statistic"):
            return float(res.statistic), float(res.pvalue)
        return float(res[0]), float(res[1])
    except Exception:
        return np.nan, np.nan


def _loa_stats(pred_vals, gt_vals):
    pred_vals = np.asarray(pred_vals, dtype=float)
    gt_vals = np.asarray(gt_vals, dtype=float)
    diff = pred_vals - gt_vals
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return np.nan, np.nan, np.nan
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1)) if diff.size > 1 else 0.0
    return bias, bias - 1.96 * sd, bias + 1.96 * sd


def _roi_table(volumes, roi_masks, rois):
    rows = []
    gt_cache = {}
    for method_id, by_subject in volumes.items():
        for sid, entry in by_subject.items():
            pred_roi = extract_roi_values(entry["pred"], roi_masks)
            if sid not in gt_cache:
                gt_cache[sid] = extract_roi_values(entry["gt"], roi_masks)
            gt_roi = gt_cache[sid]
            for roi in rois:
                rows.append({
                    "method_id": method_id,
                    "method": get_method_label(method_id),
                    "subject": sid,
                    "roi": roi,
                    "pred": pred_roi.get(roi, np.nan),
                    "gt": gt_roi.get(roi, np.nan),
                })
    return pd.DataFrame(rows)


def run_clinical_fidelity(method_ids=None, subjects=None, rois=None) -> pd.DataFrame:
    if method_ids is None:
        method_ids = METHOD_ORDER
    if subjects is None:
        subjects = load_val43_names()
    if rois is None:
        rois = DEFAULT_ROIS

    atlas = load_atlas()
    roi_masks = build_roi_masks(atlas)
    volumes = load_aligned_volumes(subjects=subjects, method_ids=method_ids)
    roi_df = _roi_table(volumes, roi_masks, rois)

    rows = []
    for (method_id, method, roi), sub_df in roi_df.groupby(["method_id", "method", "roi"]):
        pred = sub_df["pred"].to_numpy(dtype=float)
        gt = sub_df["gt"].to_numpy(dtype=float)
        mae_subject = np.abs(pred - gt)
        mae, mae_lo, mae_hi = bootstrap_ci(mae_subject)
        pearson_r, pearson_p = _safe_corr(stats.pearsonr, pred, gt)
        spearman_r, spearman_p = _safe_corr(stats.spearmanr, pred, gt)
        bias, loa_lo, loa_hi = _loa_stats(pred, gt)
        rows.append({
            "method": method,
            "method_id": method_id,
            "roi": roi,
            "n_subjects": int(np.isfinite(mae_subject).sum()),
            "MAE": mae,
            "MAE_lo": mae_lo,
            "MAE_hi": mae_hi,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "bias": bias,
            "loa_lo": loa_lo,
            "loa_hi": loa_hi,
        })

    out_df = pd.DataFrame(rows)
    EXTENDED_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXTENDED_OUT_DIR / "clinical_fidelity.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved clinical fidelity results to {out_path}")
    return out_df


if __name__ == "__main__":
    run_clinical_fidelity()
