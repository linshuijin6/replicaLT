#!/usr/bin/env python3
"""
Phase C: Tau staging / positivity analysis.

Fits ROI-specific positivity thresholds on real training PET using a simple
2-component Gaussian mixture over regional normalized uptake, then evaluates
real-vs-synthetic agreement on the 43-subject comparison cohort.
"""

import json
from pathlib import Path

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.mixture import GaussianMixture

from comparison_common import (
    EXTENDED_OUT_DIR,
    METHOD_ORDER,
    REF_SHAPE,
    align_volume,
    bootstrap_ci,
    build_roi_masks,
    extract_roi_values,
    get_method_label,
    load_aligned_volumes,
    load_atlas,
    load_train_data,
    load_val43_names,
)

STAGE_ORDER = ["Negative", "Braak_I_II", "Braak_III_IV", "Braak_V_VI"]
STAGE_ROIS = ["Braak_I_II", "Braak_III_IV", "Braak_V_VI"]


def _load_train_roi_table(roi_masks):
    rows = []
    for entry in load_train_data():
        tau_path = entry.get("tau")
        sid = entry.get("name")
        diagnosis = entry.get("diagnosis")
        if not tau_path or not Path(tau_path).exists():
            continue
        try:
            arr = nib.load(tau_path).get_fdata().astype(np.float32)
            arr = align_volume(arr, "plasma", REF_SHAPE)
            roi_vals = extract_roi_values(arr, roi_masks)
            row = {"name": sid, "diagnosis": diagnosis}
            for roi in STAGE_ROIS:
                row[roi] = roi_vals.get(roi, np.nan)
            rows.append(row)
        except Exception as exc:
            print(f"[Warning] skip training subject {sid}: {exc}")
    return pd.DataFrame(rows)


def _fit_threshold(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 10:
        return np.nan
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(vals.reshape(-1, 1))
    means = gmm.means_.ravel()
    covs = gmm.covariances_.reshape(-1)
    weights = gmm.weights_.ravel()
    lo, hi = np.min(means - 4 * np.sqrt(covs)), np.max(means + 4 * np.sqrt(covs))
    grid = np.linspace(lo, hi, 4000)
    pdf0 = weights[0] * np.exp(-(grid - means[0]) ** 2 / (2 * covs[0])) / np.sqrt(2 * np.pi * covs[0])
    pdf1 = weights[1] * np.exp(-(grid - means[1]) ** 2 / (2 * covs[1])) / np.sqrt(2 * np.pi * covs[1])
    idx = np.argmin(np.abs(pdf0 - pdf1))
    return float(grid[idx])


def fit_thresholds(train_roi_df):
    thresholds = {}
    for roi in STAGE_ROIS:
        thresholds[roi] = _fit_threshold(train_roi_df[roi].values)
    return thresholds


def assign_stage(roi_values, thresholds):
    positive = [roi for roi in STAGE_ROIS if roi_values.get(roi, np.nan) >= thresholds.get(roi, np.inf)]
    if not positive:
        return "Negative"
    return positive[-1]


def _bootstrap_acc(values):
    return bootstrap_ci(np.asarray(values, dtype=float), statistic=np.mean)


def _plot_confusions(method_confusions, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    for ax, (method, cm) in zip(axes, method_confusions.items()):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(method)
        ax.set_xticks(range(len(STAGE_ORDER)))
        ax.set_yticks(range(len(STAGE_ORDER)))
        ax.set_xticklabels(STAGE_ORDER, rotation=30, ha="right")
        ax.set_yticklabels(STAGE_ORDER)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_staging(method_ids=None, subjects=None):
    if method_ids is None:
        method_ids = METHOD_ORDER
    if subjects is None:
        subjects = load_val43_names()

    atlas = load_atlas()
    roi_masks = build_roi_masks(atlas)

    train_roi_df = _load_train_roi_table(roi_masks)
    thresholds = fit_thresholds(train_roi_df)

    volumes = load_aligned_volumes(subjects=subjects, method_ids=method_ids)
    gt_by_subject = {sid: extract_roi_values(volumes[method_ids[0]][sid]["gt"], roi_masks) for sid in subjects}
    gt_stage = {sid: assign_stage(v, thresholds) for sid, v in gt_by_subject.items()}

    rows = []
    method_confusions = {}
    for method_id in method_ids:
        pred_stage = {
            sid: assign_stage(extract_roi_values(volumes[method_id][sid]["pred"], roi_masks), thresholds)
            for sid in subjects
        }
        y_true = [gt_stage[sid] for sid in subjects]
        y_pred = [pred_stage[sid] for sid in subjects]
        stage_correct = np.array([a == b for a, b in zip(y_true, y_pred)], dtype=float)
        tau_true = np.array([x != "Negative" for x in y_true], dtype=float)
        tau_pred = np.array([x != "Negative" for x in y_pred], dtype=float)
        tau_correct = (tau_true == tau_pred).astype(float)
        _, tau_lo, tau_hi = _bootstrap_acc(tau_correct)
        _, st_lo, st_hi = _bootstrap_acc(stage_correct)
        cm = confusion_matrix(y_true, y_pred, labels=STAGE_ORDER)
        method_confusions[get_method_label(method_id)] = cm
        row = {
            "method": get_method_label(method_id),
            "method_id": method_id,
            "n_subjects": len(subjects),
            "tau_positivity_accuracy": float(tau_correct.mean()),
            "tau_pos_acc_lo": tau_lo,
            "tau_pos_acc_hi": tau_hi,
            "stage_agreement": float(stage_correct.mean()),
            "stage_agree_lo": st_lo,
            "stage_agree_hi": st_hi,
            "weighted_kappa": float(cohen_kappa_score(y_true, y_pred, labels=STAGE_ORDER, weights="quadratic")),
            "threshold_source": f"GMM on {len(train_roi_df)} training subjects",
        }
        for idx, stage in enumerate(STAGE_ORDER):
            row[f"gt_{stage}"] = int(cm[idx, :].sum())
            row[f"pred_{stage}"] = int(cm[:, idx].sum())
            row[f"correct_{stage}"] = int(cm[idx, idx])
        rows.append(row)

    EXTENDED_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = EXTENDED_OUT_DIR / "staging_positivity.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    _plot_confusions(method_confusions, EXTENDED_OUT_DIR / "staging_confusion_matrices.png")
    with open(EXTENDED_OUT_DIR / "staging_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"Saved staging results to {out_csv}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run_staging()
