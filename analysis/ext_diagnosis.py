#!/usr/bin/env python3
"""
Phase D: Downstream diagnosis classification.

Trains lightweight classifiers on real-PET ROI features from the training cohort,
and evaluates real/synthetic/MRI feature sources on the 43-subject comparison set
using leave-one-out cross-validation.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
    load_train_data,
    load_val43_subjects,
    load_val43_names,
    load_atlas,
)

FEATURE_ROIS = [
    "Global_cortical",
    "Temporal_metaROI",
    "Braak_I_II",
    "Braak_III_IV",
    "Braak_V_VI",
    "Entorhinal_proxy",
    "Hippocampus",
    "Amygdala",
]
LABELS = ["CN", "MCI", "AD"]


def _volume_to_features(volume, roi_masks):
    values = extract_roi_values(volume, roi_masks)
    return [values.get(roi, np.nan) for roi in FEATURE_ROIS]


def _load_train_feature_table(roi_masks):
    rows = []
    for entry in load_train_data():
        tau_path = entry.get("tau")
        mri_path = entry.get("mri")
        diagnosis = entry.get("diagnosis")
        if diagnosis not in LABELS or not tau_path or not Path(tau_path).exists():
            continue
        try:
            tau = align_volume(nib.load(tau_path).get_fdata().astype(np.float32), "plasma", REF_SHAPE)
            tau_feat = _volume_to_features(tau, roi_masks)
            row = {"name": entry.get("name"), "diagnosis": diagnosis}
            for roi, val in zip(FEATURE_ROIS, tau_feat):
                row[f"tau_{roi}"] = val
            if mri_path and Path(mri_path).exists():
                mri = align_volume(nib.load(mri_path).get_fdata().astype(np.float32), "plasma", REF_SHAPE)
                mri_feat = _volume_to_features(mri, roi_masks)
                for roi, val in zip(FEATURE_ROIS, mri_feat):
                    row[f"mri_{roi}"] = val
            rows.append(row)
        except Exception as exc:
            print(f"[Warning] skip training subject {entry.get('name')}: {exc}")
    return pd.DataFrame(rows)


def _evaluate_loocv(X, y):
    if len(X) == 0:
        return None
    loo = LeaveOneOut()
    y_true, y_pred, y_prob = [], [], []
    for train_idx, test_idx in loo.split(X):
        clf = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, multi_class="multinomial", class_weight="balanced")),
        ])
        clf.fit(X[train_idx], y[train_idx])
        y_true.append(y[test_idx][0])
        y_pred.append(clf.predict(X[test_idx])[0])
        y_prob.append(clf.predict_proba(X[test_idx])[0])
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc = np.nan
    return {
        "balanced_accuracy": float(bacc),
        "macro_f1": float(f1),
        "auc_ovr": float(auc),
        "per_sample_acc": (y_true == y_pred).astype(float),
    }


def _ci_from_point(values, point):
    _, lo, hi = bootstrap_ci(values)
    return point, lo, hi


def run_diagnosis(method_ids=None, subjects=None):
    if method_ids is None:
        method_ids = METHOD_ORDER
    if subjects is None:
        subjects = load_val43_names()

    atlas = load_atlas()
    roi_masks = build_roi_masks(atlas)
    test_meta = pd.DataFrame(load_val43_subjects())[["name", "diagnosis", "mri"]].drop_duplicates("name")
    test_meta = test_meta[test_meta["name"].isin(subjects)].copy()
    test_meta = test_meta.sort_values("name")

    le = LabelEncoder()
    le.fit(LABELS)
    y = le.transform(test_meta["diagnosis"].values)

    volumes = load_aligned_volumes(subjects=test_meta["name"].tolist(), method_ids=method_ids)
    gt_features = np.array([_volume_to_features(volumes[method_ids[0]][sid]["gt"], roi_masks) for sid in test_meta["name"]])

    mri_features = []
    for _, row in test_meta.iterrows():
        arr = nib.load(row["mri"]).get_fdata().astype(np.float32)
        arr = align_volume(arr, "plasma", REF_SHAPE)
        mri_features.append(_volume_to_features(arr, roi_masks))
    mri_features = np.asarray(mri_features, dtype=float)

    rows = []
    sources = [("Real PET (upper bound)", "gt", gt_features)]
    for method_id in method_ids:
        feat = np.asarray([_volume_to_features(volumes[method_id][sid]["pred"], roi_masks) for sid in test_meta["name"]], dtype=float)
        sources.append((get_method_label(method_id), method_id, feat))
    sources.append(("MRI-only (lower bound)", "mri", mri_features))

    for label, source_id, X in sources:
        metrics = _evaluate_loocv(X, y)
        if metrics is None:
            rows.append({"feature_source": label, "source_id": source_id, "n_subjects": 0})
            continue
        bacc, bacc_lo, bacc_hi = _ci_from_point(metrics["per_sample_acc"], metrics["balanced_accuracy"])
        _, f1_lo, f1_hi = bootstrap_ci(metrics["per_sample_acc"], statistic=np.mean)
        _, auc_lo, auc_hi = bootstrap_ci(metrics["per_sample_acc"], statistic=np.mean)
        rows.append({
            "feature_source": label,
            "source_id": source_id,
            "n_subjects": len(y),
            "balanced_accuracy": bacc,
            "macro_f1": metrics["macro_f1"],
            "auc_ovr": metrics["auc_ovr"],
            "bal_acc_lo": bacc_lo,
            "bal_acc_hi": bacc_hi,
            "f1_lo": f1_lo,
            "f1_hi": f1_hi,
            "auc_lo": auc_lo,
            "auc_hi": auc_hi,
            "n_CN": int((test_meta["diagnosis"] == "CN").sum()),
            "n_MCI": int((test_meta["diagnosis"] == "MCI").sum()),
            "n_AD": int((test_meta["diagnosis"] == "AD").sum()),
        })

    out_df = pd.DataFrame(rows)
    EXTENDED_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXTENDED_OUT_DIR / "diagnosis_classification.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved diagnosis results to {out_path}")
    return out_df


if __name__ == "__main__":
    run_diagnosis()
