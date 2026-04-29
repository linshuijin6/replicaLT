#!/usr/bin/env python3
"""
comparison_common.py — Shared helpers for the extended MRI->TAU PET comparison pipeline.

Extracted from run_comparison.py and extended with:
- Atlas ROI extraction (Harvard-Oxford)
- Subject metadata loading
- Vendor info loading
- Unified volume loading with spatial alignment
- Bootstrap CI utilities
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom as nd_zoom


# ── Project paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]

# ── Method registry (mirrors run_comparison.py) ─────────────────────────────
METHOD_ORDER = ["pasta", "legacy", "plasma", "ficd"]
METHOD_ALIGNMENT = {
    "pasta": "pasta",
    "legacy": "native",
    "plasma": "native",
    "ficd": "ficd",
}
FICD_CROP = (11, 10, 20, 17, 0, 21)
REF_SHAPE = (160, 192, 160)

METHOD_SPECS = {
    "pasta": {
        "label": "PASTA",
        "summary": {
            "model_type": "2.5D DDIM-100 diffusion",
            "resolution": "96x112x96 (1.5mm)",
            "conditioning": "Slice-level conditioning",
        },
        "existing_results": {
            "output_dir": Path("/home/data/linshuijin/replicaLT/pasta/replicaLT_comparison/results/2026-04-12_331111/inference_output"),
        },
    },
    "legacy": {
        "label": "Legacy",
        "summary": {
            "model_type": "Rectified-flow 1-step",
            "resolution": "160x192x160 (1mm)",
            "conditioning": "BiomedCLIP text token",
        },
        "existing_results": {
            "output_dir": Path("/mnt/nfsdata/nfsdata/lsj.14/replicaLT/analysis/comparison_results/legacy"),
        },
    },
    "plasma": {
        "label": "Plasma (Ours)",
        "summary": {
            "model_type": "Rectified-flow 1-step",
            "resolution": "160x192x160 (1mm)",
            "conditioning": "Plasma embedding + text token",
        },
        "existing_results": {
            "output_dir": Path("/mnt/nfsdata/nfsdata/lsj.14/replicaLT/analysis/comparison_results/plasma"),
        },
    },
    "ficd": {
        "label": "FiCD",
        "summary": {
            "model_type": "DDPM concat-conditioning",
            "resolution": "160x180x160",
            "conditioning": "MRI concat + text embedding",
        },
        "existing_results": {
            "run_dir": Path("/home/data/linshuijin/replicaLT/runs/ficd_aligned_tau/260422.1177288/inference"),
        },
    },
}

# ── Data paths ───────────────────────────────────────────────────────────────
COMMON_SUBJECTS_JSON = ROOT / "analysis" / "_common_subjects.json"
VAL_SUBJECTS_JSON = ROOT / "analysis" / "comparison_results" / "val_43_subjects.json"
PLASMA_VAL_JSON = ROOT / "val_data_with_description.json"
TRAIN_DATA_JSON = ROOT / "train_data_with_description.json"
TEST_TABULAR_JSON = ROOT / "pasta" / "replicaLT_comparison" / "data" / "test_tabular.json"
VENDOR_CSV = ROOT / "analysis" / "tau_vendor_diagnosis_analysis.csv"
OUT_DIR = ROOT / "analysis" / "comparison_results"
EXTENDED_OUT_DIR = OUT_DIR / "extended"


# ── Braak ROI definitions (Harvard-Oxford label names) ───────────────────────
# Reference: Chen et al. 2021 (Translational Psychiatry)
# Harvard-Oxford cortical atlas label names (from nilearn)
BRAAK_ROI_DEFINITIONS = {
    "Braak_I_II": [
        "Parahippocampal Gyrus, anterior division",
        "Temporal Fusiform Cortex, posterior division",
    ],
    "Braak_III_IV": [
        "Middle Temporal Gyrus, posterior division",
        "Inferior Temporal Gyrus, posterior division",
        "Temporal Fusiform Cortex, anterior division",
        "Temporal Fusiform Cortex, posterior division",
    ],
    "Braak_V_VI": [
        "Superior Frontal Gyrus",
        "Middle Frontal Gyrus",
        "Inferior Frontal Gyrus, pars triangularis",
        "Inferior Frontal Gyrus, pars opercularis",
        "Superior Parietal Lobule",
        "Precuneous Cortex",
        "Lateral Occipital Cortex, superior division",
        "Lateral Occipital Cortex, inferior division",
        "Cingulate Gyrus, anterior division",
        "Cingulate Gyrus, posterior division",
    ],
    "Temporal_metaROI": [
        "Parahippocampal Gyrus, anterior division",
        "Temporal Fusiform Cortex, anterior division",
        "Temporal Fusiform Cortex, posterior division",
        "Inferior Temporal Gyrus, anterior division",
        "Inferior Temporal Gyrus, posterior division",
        "Inferior Temporal Gyrus, temporooccipital part",
        "Middle Temporal Gyrus, anterior division",
        "Middle Temporal Gyrus, posterior division",
        "Middle Temporal Gyrus, temporooccipital part",
    ],
}

# Subcortical ROIs from Harvard-Oxford subcortical atlas
SUBCORTICAL_ROIS = {
    "Braak_III_IV": ["Left Hippocampus", "Right Hippocampus",
                     "Left Amygdala", "Right Amygdala"],
    "Temporal_metaROI": ["Left Amygdala", "Right Amygdala"],
}


# ── Basic helpers ────────────────────────────────────────────────────────────
def require_exists(path, desc=""):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path}")
    return path


def get_method_label(method_id: str) -> str:
    return METHOD_SPECS[method_id]["label"]


def get_enabled_methods(method_ids: Optional[List[str]] = None) -> List[str]:
    if method_ids is None:
        method_ids = METHOD_ORDER
    return [m for m in METHOD_ORDER if m in method_ids]


def get_method_source(method_id: str) -> dict:
    return METHOD_SPECS[method_id]["existing_results"]


# ── Spatial alignment functions (from run_comparison.py) ─────────────────────
def resample_volume(arr: np.ndarray, tgt_shape: tuple) -> np.ndarray:
    """Resample 3D volume to target shape via trilinear interpolation."""
    if arr.shape == tgt_shape:
        return arr
    factors = [t / s for t, s in zip(tgt_shape, arr.shape)]
    return np.clip(
        nd_zoom(arr.astype(np.float64), factors, order=1).astype(np.float32),
        0, 1,
    )


def uncrop_resample(arr: np.ndarray, tgt_shape: tuple,
                    crop: tuple = FICD_CROP) -> np.ndarray:
    """Reverse FiCD crop and resample to target shape."""
    padded = np.pad(
        arr,
        [(crop[0], crop[1]), (crop[2], crop[3]), (crop[4], crop[5])],
        mode="constant", constant_values=0,
    )
    return resample_volume(padded, tgt_shape)


def crop_or_pad(data: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Center-crop or zero-pad to target shape."""
    result = np.zeros(target_shape, dtype=data.dtype)
    starts_src, ends_src, starts_dst, ends_dst = [], [], [], []
    for s, t in zip(data.shape, target_shape):
        if s >= t:
            start_s = (s - t) // 2
            starts_src.append(start_s); ends_src.append(start_s + t)
            starts_dst.append(0); ends_dst.append(t)
        else:
            start_d = (t - s) // 2
            starts_src.append(0); ends_src.append(s)
            starts_dst.append(start_d); ends_dst.append(start_d + s)
    result[
        starts_dst[0]:ends_dst[0],
        starts_dst[1]:ends_dst[1],
        starts_dst[2]:ends_dst[2],
    ] = data[
        starts_src[0]:ends_src[0],
        starts_src[1]:ends_src[1],
        starts_src[2]:ends_src[2],
    ]
    return result


def pasta_to_ref_space(arr: np.ndarray, tgt_shape: tuple = REF_SHAPE) -> np.ndarray:
    """Convert PASTA 96x112x96 output to 160x192x160 reference space."""
    padded = np.pad(arr, [(12, 13), (16, 17), (12, 13)],
                    mode="constant", constant_values=0)
    mni = nd_zoom(padded.astype(np.float64), 1.5, order=1).astype(np.float32)
    return np.clip(crop_or_pad(mni, tgt_shape), 0, 1)


def align_volume(arr: np.ndarray, method_id: str,
                 tgt_shape: tuple = REF_SHAPE) -> np.ndarray:
    """Align a method's output volume to the reference space."""
    alignment = METHOD_ALIGNMENT[method_id]
    if alignment == "native":
        return resample_volume(arr, tgt_shape)
    elif alignment == "pasta":
        return pasta_to_ref_space(arr, tgt_shape)
    elif alignment == "ficd":
        return uncrop_resample(arr, tgt_shape)
    else:
        raise ValueError(f"Unknown alignment: {alignment}")


# ── Subject & metadata loading ───────────────────────────────────────────────
def load_val43_subjects() -> List[dict]:
    """Load the 43 common test subjects from val_43_subjects.json."""
    with open(VAL_SUBJECTS_JSON) as f:
        return json.load(f)


def load_val43_names() -> List[str]:
    """Return sorted unique subject names from the 43 test set."""
    subjects = load_val43_subjects()
    return sorted(set(d["name"] for d in subjects))


def load_subject_metadata() -> pd.DataFrame:
    """
    Merge val_43_subjects.json + test_tabular.json to get a DataFrame with:
    name, diagnosis, AGE, PTGENDER, PTEDUCAT, MMSE, ADAS13, APOE4
    """
    val43 = load_val43_subjects()
    val43_names = set(d["name"] for d in val43)

    # Build base from val_43
    base = {d["name"]: {"diagnosis": d["diagnosis"]} for d in val43}

    # Enrich with test_tabular.json
    with open(TEST_TABULAR_JSON) as f:
        tabular = json.load(f)

    for entry in tabular:
        name = entry["name"]
        if name in base:
            for key in ["AGE", "PTGENDER", "PTEDUCAT", "MMSE", "ADAS13", "APOE4"]:
                if key in entry:
                    val = entry[key]
                    # Keep first occurrence (earliest examdate)
                    if key not in base[name]:
                        base[name][key] = val

    rows = []
    for name in sorted(base.keys()):
        row = {"name": name}
        row.update(base[name])
        rows.append(row)

    df = pd.DataFrame(rows)
    # Clean up NaN-like values
    for col in ["MMSE", "ADAS13", "APOE4", "AGE", "PTEDUCAT"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_vendor_info() -> pd.DataFrame:
    """
    Load scanner vendor info from tau_vendor_diagnosis_analysis.csv.
    Returns DataFrame with columns: name, vendor, vendor_model.
    Matches on PTID (= subject name).
    """
    vendor_df = pd.read_csv(VENDOR_CSV)
    val43_names = set(load_val43_names())

    rows = []
    for _, row in vendor_df.iterrows():
        ptid = row["PTID"]
        if ptid in val43_names:
            rows.append({
                "name": ptid,
                "vendor": row["pet_mfr"],
                "vendor_model": row["pet_mfr_model"],
            })

    result = pd.DataFrame(rows)
    # Deduplicate: keep first occurrence per subject
    if len(result) > 0:
        result = result.drop_duplicates(subset="name", keep="first")
    return result


# ── NIfTI loading per method ─────────────────────────────────────────────────
def _find_pasta_pair(output_dir: Path, sid: str):
    """Find PASTA syn/gt NIfTI pair for a subject, matching exam date when needed."""
    syn = sorted(output_dir.glob(f"{sid}_*_syn_pet.nii.gz"))
    gt = sorted(output_dir.glob(f"{sid}_*_GT_pet.nii.gz"))
    if len(syn) == 1 and len(gt) == 1:
        return syn[0], gt[0]

    if len(syn) != len(gt) or len(syn) == 0:
        raise FileNotFoundError(
            f"[PASTA] Expected matched syn/gt files for {sid}, got {len(syn)}/{len(gt)}"
        )

    subjects = load_val43_subjects()
    target = next((d for d in subjects if d["name"] == sid), None)
    if target is not None:
        examdate = str(target.get("examdate", "")).replace("-", "")
        diagnosis = str(target.get("diagnosis", ""))
        syn_match = [p for p in syn if examdate in p.name and diagnosis in p.name]
        gt_match = [p for p in gt if examdate in p.name and diagnosis in p.name]
        if len(syn_match) == 1 and len(gt_match) == 1:
            return syn_match[0], gt_match[0]
        syn_match = [p for p in syn if examdate in p.name]
        gt_match = [p for p in gt if examdate in p.name]
        if len(syn_match) == 1 and len(gt_match) == 1:
            return syn_match[0], gt_match[0]

    raise FileNotFoundError(
        f"[PASTA] Expected 1 syn + 1 gt for {sid}, got {len(syn)}/{len(gt)}"
    )


def load_nifti(path) -> np.ndarray:
    """Load a NIfTI file and return float32 array."""
    return nib.load(str(path)).get_fdata().astype(np.float32)


def load_method_volumes(method_id: str, subjects: List[str]) -> Dict[str, dict]:
    """
    Load pred (and gt where available) volumes for a method.
    Returns {sid: {"pred": ndarray, "gt": ndarray_or_None}}.
    Volumes are NOT yet aligned to reference space.
    """
    source = get_method_source(method_id)
    result = {}

    if method_id == "pasta":
        output_dir = Path(source["output_dir"])
        for sid in subjects:
            syn_path, gt_path = _find_pasta_pair(output_dir, sid)
            result[sid] = {
                "pred": np.clip(load_nifti(syn_path), 0, 1),
                "gt": np.clip(load_nifti(gt_path), 0, 1),
            }

    elif method_id in ("legacy", "plasma"):
        nifti_dir = Path(source["output_dir"]) / "nifti"
        for sid in subjects:
            pred_path = nifti_dir / f"{sid}_tau_pred.nii.gz"
            gt_path = nifti_dir / f"{sid}_tau_gt.nii.gz"
            result[sid] = {
                "pred": np.clip(load_nifti(pred_path), 0, 1),
                "gt": np.clip(load_nifti(gt_path), 0, 1),
            }

    elif method_id == "ficd":
        pred_dir = Path(source["run_dir"])
        for sid in subjects:
            raw = load_nifti(pred_dir / f"{sid}.nii.gz")
            result[sid] = {
                "pred": np.clip((raw + 1.0) / 2.0, 0, 1),
                "gt": None,  # FiCD doesn't save GT in prediction dir
            }

    return result


def load_aligned_volumes(
    subjects: List[str],
    method_ids: Optional[List[str]] = None,
) -> Dict[str, Dict[str, dict]]:
    """
    Load and align all method volumes to REF_SHAPE.
    Returns {method_id: {sid: {"pred": ndarray, "gt": ndarray}}}.
    GT is taken from plasma/legacy (native space) when available.
    """
    if method_ids is None:
        method_ids = METHOD_ORDER

    # First load a native-space GT from plasma or legacy
    gt_source = None
    for m in ["plasma", "legacy"]:
        if m in method_ids:
            gt_source = m
            break

    all_volumes = {}
    gt_cache = {}  # sid -> aligned GT

    for method_id in method_ids:
        raw = load_method_volumes(method_id, subjects)
        aligned = {}
        for sid in subjects:
            entry = raw[sid]
            pred = align_volume(entry["pred"], method_id)

            # For GT: use native-space GT from plasma/legacy
            if method_id in ("plasma", "legacy") and entry["gt"] is not None:
                gt = resample_volume(entry["gt"], REF_SHAPE)
                if sid not in gt_cache:
                    gt_cache[sid] = gt
            elif method_id == "pasta" and entry["gt"] is not None:
                gt = pasta_to_ref_space(entry["gt"])
                if sid not in gt_cache:
                    gt_cache[sid] = gt

            aligned[sid] = {"pred": pred, "gt": gt_cache.get(sid)}
        all_volumes[method_id] = aligned

    return all_volumes


# ── Atlas ROI extraction ─────────────────────────────────────────────────────
_atlas_cache = {}


def load_atlas(ref_shape: tuple = REF_SHAPE) -> dict:
    """
    Download Harvard-Oxford cortical + subcortical atlases and resample
    to the reference space. Returns dict with:
    - cortical_data: 3D int array (label indices)
    - cortical_labels: list of label names
    - subcortical_data: 3D int array
    - subcortical_labels: list of label names
    """
    cache_key = ref_shape
    if cache_key in _atlas_cache:
        return _atlas_cache[cache_key]

    from nilearn import datasets, image

    print("[Atlas] Downloading Harvard-Oxford cortical atlas...")
    ho_cort = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-1mm")
    print("[Atlas] Downloading Harvard-Oxford subcortical atlas...")
    ho_sub = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-1mm")

    # Load atlas images
    cort_img = ho_cort.maps if hasattr(ho_cort.maps, "shape") else nib.load(ho_cort.maps)
    sub_img = ho_sub.maps if hasattr(ho_sub.maps, "shape") else nib.load(ho_sub.maps)

    cort_data = np.asarray(cort_img.dataobj, dtype=np.int16)
    sub_data = np.asarray(sub_img.dataobj, dtype=np.int16)

    # Resample to reference shape if needed
    if cort_data.shape != ref_shape:
        print(f"[Atlas] Resampling cortical atlas {cort_data.shape} -> {ref_shape}")
        cort_data = _resample_atlas(cort_data, ref_shape)
    if sub_data.shape != ref_shape:
        print(f"[Atlas] Resampling subcortical atlas {sub_data.shape} -> {ref_shape}")
        sub_data = _resample_atlas(sub_data, ref_shape)

    result = {
        "cortical_data": cort_data,
        "cortical_labels": ho_cort.labels,
        "subcortical_data": sub_data,
        "subcortical_labels": ho_sub.labels,
    }
    _atlas_cache[cache_key] = result
    print(f"[Atlas] Loaded. Cortical labels: {len(ho_cort.labels)}, "
          f"Subcortical labels: {len(ho_sub.labels)}")
    return result


def _resample_atlas(data: np.ndarray, tgt_shape: tuple) -> np.ndarray:
    """Resample atlas (integer labels) using nearest-neighbor interpolation."""
    factors = [t / s for t, s in zip(tgt_shape, data.shape)]
    return nd_zoom(data.astype(np.float32), factors, order=0).astype(np.int16)


def build_roi_masks(atlas: dict) -> Dict[str, np.ndarray]:
    """
    Build binary masks for each ROI from the atlas.
    Returns {roi_name: bool_mask_3d}.
    """
    cort_data = atlas["cortical_data"]
    cort_labels = atlas["cortical_labels"]
    sub_data = atlas["subcortical_data"]
    sub_labels = atlas["subcortical_labels"]

    # Build label-to-index maps
    cort_name2idx = {name: idx for idx, name in enumerate(cort_labels)}
    sub_name2idx = {name: idx for idx, name in enumerate(sub_labels)}

    masks = {}

    # Cortical ROIs
    for roi_name, label_names in BRAAK_ROI_DEFINITIONS.items():
        mask = np.zeros(cort_data.shape, dtype=bool)
        for lname in label_names:
            if lname in cort_name2idx:
                idx = cort_name2idx[lname]
                mask |= (cort_data == idx)
            else:
                print(f"  [Warning] Cortical label '{lname}' not found in atlas")
        masks[roi_name] = mask

    # Add subcortical components
    for roi_name, label_names in SUBCORTICAL_ROIS.items():
        if roi_name not in masks:
            masks[roi_name] = np.zeros(sub_data.shape, dtype=bool)
        for lname in label_names:
            if lname in sub_name2idx:
                idx = sub_name2idx[lname]
                masks[roi_name] |= (sub_data == idx)
            else:
                print(f"  [Warning] Subcortical label '{lname}' not found in atlas")

    # Individual ROIs for detailed analysis
    individual_rois = {
        "Entorhinal_proxy": ["Parahippocampal Gyrus, anterior division"],
        "Hippocampus": [],  # subcortical only
        "Amygdala": [],     # subcortical only
    }
    for roi_name, cort_names in individual_rois.items():
        mask = np.zeros(cort_data.shape, dtype=bool)
        for lname in cort_names:
            if lname in cort_name2idx:
                mask |= (cort_data == cort_name2idx[lname])

    # Hippocampus
    hipp_mask = np.zeros(sub_data.shape, dtype=bool)
    for lname in ["Left Hippocampus", "Right Hippocampus"]:
        if lname in sub_name2idx:
            hipp_mask |= (sub_data == sub_name2idx[lname])
    masks["Hippocampus"] = hipp_mask

    # Amygdala
    amyg_mask = np.zeros(sub_data.shape, dtype=bool)
    for lname in ["Left Amygdala", "Right Amygdala"]:
        if lname in sub_name2idx:
            amyg_mask |= (sub_data == sub_name2idx[lname])
    masks["Amygdala"] = amyg_mask

    # Entorhinal proxy
    ent_mask = np.zeros(cort_data.shape, dtype=bool)
    if "Parahippocampal Gyrus, anterior division" in cort_name2idx:
        ent_mask |= (cort_data == cort_name2idx["Parahippocampal Gyrus, anterior division"])
    masks["Entorhinal_proxy"] = ent_mask

    # Global cortical mask (all non-zero cortical labels)
    masks["Global_cortical"] = cort_data > 0

    # Report mask sizes
    for name, mask in masks.items():
        n_voxels = mask.sum()
        print(f"  [ROI] {name}: {n_voxels} voxels")

    return masks


def extract_roi_values(volume: np.ndarray, roi_masks: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Extract mean intensity within each ROI mask.
    Returns {roi_name: mean_value}.
    """
    result = {}
    for roi_name, mask in roi_masks.items():
        if mask.shape != volume.shape:
            # Resample mask to volume shape
            mask = _resample_atlas(mask.astype(np.int16), volume.shape).astype(bool)
        voxels = volume[mask]
        result[roi_name] = float(np.mean(voxels)) if len(voxels) > 0 else np.nan
    return result


# ── Statistical utilities ────────────────────────────────────────────────────
def bootstrap_ci(values, n_boot: int = 2000, ci: float = 0.95,
                 statistic=np.mean) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval.
    Returns (point_estimate, ci_lower, ci_upper).
    """
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)

    point = float(statistic(values))
    rng = np.random.default_rng(42)
    boot_stats = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_stats.append(statistic(sample))
    boot_stats = np.array(boot_stats)
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_stats, 100 * alpha))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha)))
    return (point, lo, hi)


def load_train_data() -> List[dict]:
    """Load training data entries from train_data_with_description.json."""
    with open(TRAIN_DATA_JSON) as f:
        return json.load(f)
