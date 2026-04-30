#!/usr/bin/env python3
"""Compute real AV45/FBP composite SUVR and A-beta positivity.

This validation follows the Lan 2025 / ADNI-style amyloid PET summary:
target uptake is measured from bilateral frontal, cingulate, parietal,
and temporal cortical regions; reference uptake is measured from the
whole cerebellum; A-beta positivity is defined by composite SUVR > 1.11.
"""

from __future__ import annotations

import argparse
import csv
import ssl
import tarfile
import urllib.request
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    import nibabel as nib
    from nilearn import datasets, image
except Exception as exc:  # pragma: no cover - exercised by CLI users.
    raise SystemExit(
        "Missing neuroimaging dependencies. Run with an environment that has "
        "nibabel and nilearn installed, e.g.:\n"
        "  conda run -n xiaochou python validation/E1/compute_real_av45_suvr.py\n"
        f"Original import error: {type(exc).__name__}: {exc}"
    ) from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc as sklearn_auc
    from sklearn.metrics import roc_auc_score, roc_curve
except Exception as exc:  # pragma: no cover - exercised by CLI users.
    raise SystemExit(
        "Missing plotting/statistics dependencies. Please install matplotlib "
        f"and scikit-learn. Original import error: {type(exc).__name__}: {exc}"
    ) from exc


AMYLOID_THRESHOLD = 1.11
EXPECTED_SHAPE = (182, 218, 182)
AAL3_URL = "https://www.gin.cnrs.fr/wp-content/uploads/AAL3v2_for_SPM12.tar.gz"

# Harvard-Oxford cortical labels from nilearn. We intentionally exclude medial
# temporal / tau-metaROI labels such as parahippocampal and fusiform regions.
TARGET_ROI_LABELS: Dict[str, List[str]] = {
    "frontal": [
        "Frontal Pole",
        "Superior Frontal Gyrus",
        "Middle Frontal Gyrus",
        "Inferior Frontal Gyrus, pars triangularis",
        "Inferior Frontal Gyrus, pars opercularis",
        "Frontal Medial Cortex",
        "Frontal Orbital Cortex",
    ],
    "cingulate": [
        "Cingulate Gyrus, anterior division",
        "Cingulate Gyrus, posterior division",
    ],
    "parietal": [
        "Superior Parietal Lobule",
        "Supramarginal Gyrus, anterior division",
        "Supramarginal Gyrus, posterior division",
        "Angular Gyrus",
        "Precuneous Cortex",
    ],
    "temporal": [
        "Superior Temporal Gyrus, anterior division",
        "Superior Temporal Gyrus, posterior division",
        "Middle Temporal Gyrus, anterior division",
        "Middle Temporal Gyrus, posterior division",
        "Middle Temporal Gyrus, temporooccipital part",
        "Inferior Temporal Gyrus, anterior division",
        "Inferior Temporal Gyrus, posterior division",
        "Inferior Temporal Gyrus, temporooccipital part",
        "Temporal Pole",
    ],
}


@dataclass
class RoiWeights:
    composite: np.ndarray
    categories: Dict[str, np.ndarray]
    cerebellum: np.ndarray
    labels_used: Dict[str, object]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    default_out = Path(__file__).resolve().parent / "results"
    default_cache = Path(__file__).resolve().parent / "nilearn_data"
    parser = argparse.ArgumentParser(
        description="Compute Lan 2025-style real AV45/FBP composite SUVR."
    )
    parser.add_argument(
        "--data-json",
        type=Path,
        default=root / "train_data_with_description.json",
        help="Input JSON with mri/fdg/av45/tau paths.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_out,
        help="Directory for CSV, JSON, and figures.",
    )
    parser.add_argument(
        "--nilearn-data-dir",
        type=Path,
        default=default_cache,
        help="Fixed cache directory for nilearn atlas downloads.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=AMYLOID_THRESHOLD,
        help="FBP/AV45 composite SUVR threshold for A-beta positivity.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for smoke tests. 0 means all real AV45 samples.",
    )
    return parser.parse_args()


def load_real_av45_rows(data_json: Path, limit: int = 0) -> List[dict]:
    with data_json.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected list JSON, got {type(rows).__name__}: {data_json}")

    real_rows = []
    for row in rows:
        av45 = str(row.get("av45") or "")
        if not av45 or "zero/AV45" in av45:
            continue
        real_rows.append(row)
        if limit > 0 and len(real_rows) >= limit:
            break
    return real_rows


def _as_image(maps):
    if hasattr(maps, "shape") and hasattr(maps, "affine"):
        return maps
    return nib.load(str(maps))


def _labels_for_4d_prob_atlas(labels: List[str], n_maps: int) -> List[str]:
    if len(labels) == n_maps:
        return labels
    if len(labels) == n_maps + 1 and str(labels[0]).lower() == "background":
        return labels[1:]
    raise ValueError(
        f"Cannot align {len(labels)} labels with {n_maps} probabilistic atlas maps"
    )


def _probability_maps_by_label(atlas) -> Tuple[nib.Nifti1Image, Dict[str, int]]:
    atlas_img = _as_image(atlas.maps)
    if len(atlas_img.shape) != 4:
        raise ValueError(f"Expected 4D probabilistic atlas, got {atlas_img.shape}")
    labels = _labels_for_4d_prob_atlas(list(atlas.labels), atlas_img.shape[3])
    return atlas_img, {label: idx for idx, label in enumerate(labels)}


def _resample_img_to_ref(
    source_img: nib.Nifti1Image,
    ref_img: nib.Nifti1Image,
    interpolation: str,
) -> nib.Nifti1Image:
    if source_img.shape[:3] == ref_img.shape[:3] and np.allclose(
        source_img.affine, ref_img.affine, atol=1e-3
    ):
        return source_img
    return image.resample_to_img(
        source_img,
        ref_img,
        interpolation=interpolation,
        force_resample=True,
        copy_header=True,
    )


def _load_prob_roi_weights(
    ref_img: nib.Nifti1Image,
    data_dir: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    atlas = datasets.fetch_atlas_harvard_oxford(
        "cort-prob-1mm",
        data_dir=str(data_dir),
        verbose=1,
    )
    atlas_img, label_to_map = _probability_maps_by_label(atlas)
    atlas_img = _resample_img_to_ref(atlas_img, ref_img, interpolation="continuous")
    atlas_data = np.asarray(atlas_img.get_fdata(dtype=np.float32), dtype=np.float32)

    weights: Dict[str, np.ndarray] = {}
    labels_used: Dict[str, List[str]] = {}
    for category, labels in TARGET_ROI_LABELS.items():
        missing = [label for label in labels if label not in label_to_map]
        if missing:
            raise ValueError(f"Missing Harvard-Oxford labels for {category}: {missing}")
        selected = [label_to_map[label] for label in labels]
        # Nilearn probability maps are percentages in [0, 100].
        category_weight = np.sum(atlas_data[..., selected], axis=3) / 100.0
        category_weight = np.clip(category_weight, 0.0, None)
        if float(np.sum(category_weight)) <= 0:
            raise ValueError(f"Empty target ROI category after atlas load: {category}")
        weights[category] = category_weight
        labels_used[category] = labels
    return weights, labels_used


def _aal_label_table(atlas) -> pd.DataFrame:
    if hasattr(atlas, "lut") and atlas.lut is not None:
        lut = atlas.lut.copy()
        if "index" in lut.columns and "name" in lut.columns:
            return lut[["index", "name"]]
    return pd.DataFrame({"index": atlas.indices, "name": atlas.labels})


def _download_aal3_unverified(data_dir: Path) -> Tuple[Path, pd.DataFrame, str]:
    """Fallback for hosts whose Python cert store cannot verify gin.cnrs.fr."""
    manual_dir = data_dir / "aal3_manual"
    extract_dir = manual_dir / "AAL3"
    atlas_path = extract_dir / "AAL3v1_1mm.nii.gz"
    labels_path = extract_dir / "AAL3v1_1mm.nii.txt"
    if not atlas_path.exists() or not labels_path.exists():
        manual_dir.mkdir(parents=True, exist_ok=True)
        archive_path = manual_dir / "AAL3v2_for_SPM12.tar.gz"
        if not archive_path.exists():
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(AAL3_URL, context=context, timeout=180) as resp:
                archive_path.write_bytes(resp.read())
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(manual_dir)
    rows = []
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            rows.append({"index": idx, "name": parts[1]})
    if not rows:
        raise ValueError(f"No labels parsed from {labels_path}")
    return atlas_path, pd.DataFrame(rows), "AAL3 manual SSL fallback"


def _load_aal3_img_and_lut(data_dir: Path) -> Tuple[nib.Nifti1Image, pd.DataFrame, str]:
    manual_atlas = data_dir / "aal3_manual" / "AAL3" / "AAL3v1_1mm.nii.gz"
    manual_labels = data_dir / "aal3_manual" / "AAL3" / "AAL3v1_1mm.nii.txt"
    if manual_atlas.exists() and manual_labels.exists():
        atlas_path, lut, source = _download_aal3_unverified(data_dir)
        return nib.load(str(atlas_path)), lut, source

    try:
        atlas = datasets.fetch_atlas_aal(
            version="3v2",
            data_dir=str(data_dir),
            verbose=1,
        )
        return _as_image(atlas.maps), _aal_label_table(atlas), "nilearn fetch_atlas_aal 3v2"
    except Exception as exc:  # noqa: BLE001 - fallback keeps the validation runnable.
        print(f"[AAL] nilearn fetch_atlas_aal failed, using manual fallback: {exc}")
        atlas_path, lut, source = _download_aal3_unverified(data_dir)
        return nib.load(str(atlas_path)), lut, source


def _load_cerebellum_mask(
    ref_img: nib.Nifti1Image,
    data_dir: Path,
) -> Tuple[np.ndarray, List[dict], str]:
    aal_img, lut, source = _load_aal3_img_and_lut(data_dir)
    aal_img = _resample_img_to_ref(aal_img, ref_img, interpolation="nearest")
    aal_data = np.rint(np.asarray(aal_img.get_fdata(dtype=np.float32))).astype(np.int32)
    selected_rows = []
    for _, row in lut.iterrows():
        name = str(row["name"])
        if "cerebel" not in name.lower() and "vermis" not in name.lower():
            continue
        selected_rows.append({"index": int(row["index"]), "name": name})

    if not selected_rows:
        raise ValueError("No cerebellum/vermis labels found in AAL3v2 atlas")

    label_values = [row["index"] for row in selected_rows]
    mask = np.isin(aal_data, label_values)
    if int(mask.sum()) == 0:
        raise ValueError("AAL3v2 cerebellum mask is empty after resampling")
    return mask, selected_rows, source


def build_roi_weights(ref_pet_path: Path, data_dir: Path) -> RoiWeights:
    ref_img = nib.load(str(ref_pet_path))
    category_weights, target_labels = _load_prob_roi_weights(ref_img, data_dir)
    composite = np.zeros(ref_img.shape[:3], dtype=np.float32)
    for weight in category_weights.values():
        composite += weight
    composite = np.clip(composite, 0.0, 1.0)
    if float(np.sum(composite)) <= 0:
        raise ValueError("Composite target ROI weight map is empty")

    cerebellum_mask, cerebellum_labels, cerebellum_source = _load_cerebellum_mask(
        ref_img,
        data_dir,
    )
    return RoiWeights(
        composite=composite,
        categories=category_weights,
        cerebellum=cerebellum_mask.astype(bool),
        labels_used={
            "target_atlas": "Harvard-Oxford cortical probabilistic atlas, cort-prob-1mm",
            "target_roi_definition": TARGET_ROI_LABELS,
            "target_labels_used": target_labels,
            "reference_atlas": cerebellum_source,
            "reference_labels_used": cerebellum_labels,
            "nilearn_data_dir": str(data_dir),
            "reference_pet_grid": {
                "shape": list(ref_img.shape[:3]),
                "affine": np.asarray(ref_img.affine).round(6).tolist(),
            },
        },
    )


def weighted_mean(volume: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(volume) & np.isfinite(weights) & (weights > 0)
    denom = float(np.sum(weights[valid]))
    if denom <= 0:
        return math.nan
    return float(np.sum(volume[valid] * weights[valid]) / denom)


def binary_mean(volume: np.ndarray, mask: np.ndarray) -> float:
    valid = np.isfinite(volume) & mask
    if int(valid.sum()) == 0:
        return math.nan
    return float(np.mean(volume[valid]))


def load_pet_volume(path: Path) -> Tuple[np.ndarray, List[int], int]:
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    input_shape = list(data.shape)
    n_frames = 1
    if data.ndim == 4:
        n_frames = int(data.shape[3])
        if n_frames <= 0:
            raise ValueError(f"invalid 4D PET frame count: {data.shape}")
        data = np.nanmean(data, axis=3)
    if data.ndim != 3:
        raise ValueError(f"expected 3D PET or 4D PET, got shape {tuple(input_shape)}")
    return data, input_shape, n_frames


def compute_rows(
    rows: Iterable[dict],
    weights: RoiWeights,
    threshold: float,
) -> Tuple[List[dict], List[dict]]:
    results = []
    failures = []
    for row in rows:
        name = str(row.get("name", ""))
        examdate = str(row.get("examdate", ""))
        av45_path = Path(str(row.get("av45", "")))
        try:
            if not av45_path.exists():
                raise FileNotFoundError(f"missing AV45 file: {av45_path}")
            volume, input_shape, n_frames = load_pet_volume(av45_path)
            if volume.shape != weights.composite.shape:
                raise ValueError(
                    f"PET shape {volume.shape} does not match ROI shape "
                    f"{weights.composite.shape}"
                )
            if not np.isfinite(volume).any() or float(np.nanmax(np.abs(volume))) == 0:
                raise ValueError("PET volume is empty or all zero")

            target_mean = weighted_mean(volume, weights.composite)
            cerebellum_mean = binary_mean(volume, weights.cerebellum)
            if not np.isfinite(target_mean):
                raise ValueError("target_mean is not finite")
            if not np.isfinite(cerebellum_mean) or cerebellum_mean <= 0:
                raise ValueError(f"invalid cerebellum_mean: {cerebellum_mean}")
            suvr = target_mean / cerebellum_mean
            if not np.isfinite(suvr):
                raise ValueError("SUVR is not finite")

            out = {
                "name": name,
                "examdate": examdate,
                "diagnosis": row.get("diagnosis", ""),
                "av45_path": str(av45_path),
                "target_mean": target_mean,
                "cerebellum_mean": cerebellum_mean,
                "suvr": suvr,
                "amyloid_label": int(suvr > threshold),
                "input_shape": "x".join(map(str, input_shape)),
                "n_frames": n_frames,
            }
            for category, category_weight in weights.categories.items():
                out[f"{category}_mean"] = weighted_mean(volume, category_weight)
            results.append(out)
        except Exception as exc:  # noqa: BLE001 - keep per-sample failures.
            failures.append(
                {
                    "name": name,
                    "examdate": examdate,
                    "diagnosis": row.get("diagnosis", ""),
                    "av45_path": str(av45_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return results, failures


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "examdate",
        "diagnosis",
        "av45_path",
        "target_mean",
        "cerebellum_mean",
        "suvr",
        "amyloid_label",
        "input_shape",
        "n_frames",
        "frontal_mean",
        "cingulate_mean",
        "parietal_mean",
        "temporal_mean",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def compute_roc_payload(results: List[dict], threshold: float) -> Tuple[dict, pd.DataFrame]:
    scores = np.asarray([row["suvr"] for row in results], dtype=float)
    labels = np.asarray([row["amyloid_label"] for row in results], dtype=int)
    payload = {
        "threshold": threshold,
        "n_total": int(len(results)),
        "n_abeta_positive": int(labels.sum()) if len(labels) else 0,
        "n_abeta_negative": int((labels == 0).sum()) if len(labels) else 0,
        "auc": None,
        "note": (
            "AUC uses real AV45 SUVR to predict labels defined from the same "
            "real AV45 SUVR threshold; interpret as sanity check / upper bound."
        ),
    }
    if len(np.unique(labels)) < 2:
        payload["error"] = "ROC/AUC unavailable because only one A-beta class exists"
        return payload, pd.DataFrame(columns=["fpr", "tpr", "threshold"])

    fpr, tpr, roc_thresholds = roc_curve(labels, scores)
    payload["auc"] = float(roc_auc_score(labels, scores))
    payload["auc_trapezoid"] = float(sklearn_auc(fpr, tpr))
    roc_df = pd.DataFrame(
        {
            "fpr": fpr,
            "tpr": tpr,
            "threshold": roc_thresholds,
        }
    )
    return payload, roc_df


def plot_outputs(
    results: List[dict],
    roc_df: pd.DataFrame,
    out_dir: Path,
    threshold: float,
) -> None:
    scores = np.asarray([row["suvr"] for row in results], dtype=float)
    labels = np.asarray([row["amyloid_label"] for row in results], dtype=int)

    plt.figure(figsize=(7, 5))
    if not roc_df.empty:
        plt.plot(roc_df["fpr"], roc_df["tpr"], marker="o", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="0.5")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Real AV45 SUVR ROC")
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    if len(scores):
        plt.hist(scores[labels == 0], bins=30, alpha=0.65, label="Aβ-")
        plt.hist(scores[labels == 1], bins=30, alpha=0.65, label="Aβ+")
    plt.axvline(threshold, linestyle="--", color="black", label=f"SUVR {threshold:g}")
    plt.xlabel("Composite SUVR")
    plt.ylabel("Count")
    plt.title("Real AV45 composite SUVR distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "suvr_histogram.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.nilearn_data_dir.mkdir(parents=True, exist_ok=True)

    real_rows = load_real_av45_rows(args.data_json, args.limit)
    if not real_rows:
        raise SystemExit("No real AV45 rows found after filtering zero/AV45 placeholders")

    ref_pet = Path(str(real_rows[0]["av45"]))
    roi_weights = build_roi_weights(ref_pet, args.nilearn_data_dir)
    results, failures = compute_rows(real_rows, roi_weights, args.threshold)

    write_csv(args.out_dir / "real_av45_suvr.csv", results)
    write_csv(args.out_dir / "failed_samples.csv", failures)
    write_json(args.out_dir / "roi_labels_used.json", roi_weights.labels_used)

    roc_payload, roc_df = compute_roc_payload(results, args.threshold)
    write_json(args.out_dir / "roc_auc.json", roc_payload)
    roc_df.to_csv(args.out_dir / "roc_curve.csv", index=False)
    plot_outputs(results, roc_df, args.out_dir, args.threshold)

    summary = {
        "input_json": str(args.data_json),
        "real_av45_rows": len(real_rows),
        "computed_rows": len(results),
        "failed_rows": len(failures),
        "out_dir": str(args.out_dir),
        "threshold": args.threshold,
        "expected_shape": list(EXPECTED_SHAPE),
    }
    write_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
