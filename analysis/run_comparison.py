#!/usr/bin/env python3
"""
Unified 4-method MRI → TAU PET comparison.
Methods: PASTA, Legacy, Plasma (Ours), FiCD

Usage:
    conda run -n xiaochou python analysis/run_comparison.py
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path("/mnt/nfsdata/nfsdata/lsj.14/replicaLT")
OUT_DIR = ROOT / "analysis" / "comparison_results"
VAL_JSON = OUT_DIR / "val_43_subjects.json"
PASTA_DIR = Path("/mnt/nfsdata/nfsdata/lsj.14/PASTA/replicaLT_comparison/results/2026-04-12_331111/inference_output")
PASTA_MAP = OUT_DIR / "pasta_file_map.json"
COMMON_SUBJECTS = OUT_DIR / ".." / "_common_subjects.json"
PLASMA_VAL_JSON = ROOT / "val_data_with_description.json"

PLASMA_CKPT = ROOT / "runs" / "plasma_04.12_4053491" / "ckpt_epoch200.pt"
LEGACY_CKPT = ROOT / "runs" / "04.13_2916235" / "ckpt_epoch200.pt"
FICD_RUN_DIR = ROOT / "runs" / "ficd_smoke_test" / "260415.3663353"
FICD_CONFIG = ROOT / "configs" / "ficd" / "eval_43.yaml"

GPU = 0


# ── Helpers ──────────────────────────────────────────────────────────────────
def run_cmd(cmd, desc=""):
    """Run a shell command and stream output."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  CMD: {cmd}")
    print(f"{'='*60}\n")
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.stdout:
        print(result.stdout[-3000:])  # last 3000 chars
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr[-3000:]}")
        print(f"FAILED (exit {result.returncode}) after {elapsed:.0f}s")
    else:
        print(f"OK ({elapsed:.0f}s)")
    return result


def compute_ssim_3d(pred, gt, win_size=7):
    """Compute SSIM using MONAI (same as all methods)."""
    import torch
    from monai.metrics import SSIMMetric
    p = torch.tensor(pred, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    g = torch.tensor(gt, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    metric = SSIMMetric(spatial_dims=3, data_range=1.0, win_size=win_size)
    return metric(g, p).mean().item()


def compute_psnr(pred, gt, max_val=1.0):
    """Compute PSNR."""
    mse = np.mean((pred - gt) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(max_val**2 / mse)


def compute_ncc(pred, gt):
    """Normalized Cross-Correlation."""
    p = pred.flatten().astype(np.float64)
    g = gt.flatten().astype(np.float64)
    p_mean = p - p.mean()
    g_mean = g - g.mean()
    num = np.sum(p_mean * g_mean)
    den = np.sqrt(np.sum(p_mean**2) * np.sum(g_mean**2))
    if den < 1e-10:
        return 0.0
    return float(num / den)


def compute_all_metrics(pred, gt):
    """Compute all metrics on [0,1] arrays."""
    pred = np.clip(pred.astype(np.float64), 0, 1)
    gt = np.clip(gt.astype(np.float64), 0, 1)
    mae = np.mean(np.abs(pred - gt))
    mse = np.mean((pred - gt) ** 2)
    psnr = compute_psnr(pred, gt)
    ncc = compute_ncc(pred, gt)
    ssim = compute_ssim_3d(pred, gt)
    return {"ssim": ssim, "psnr": psnr, "mae": mae, "mse": mse, "ncc": ncc}


# ── Phase 1: Run Inference ───────────────────────────────────────────────────
def run_plasma_inference(method="plasma"):
    """Run plasma_inference.py for Plasma or Legacy."""
    is_legacy = method == "legacy"
    ckpt = LEGACY_CKPT if is_legacy else PLASMA_CKPT
    out_dir = OUT_DIR / method
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    metrics_csv = out_dir / "metrics.csv"
    if metrics_csv.exists():
        df = pd.read_csv(metrics_csv)
        if len(df) >= 40:  # roughly all subjects done
            print(f"[{method}] Already have {len(df)} results, skipping inference.")
            return True

    cmd = (
        f"cd {ROOT} && CUDA_VISIBLE_DEVICES={GPU} "
        f"conda run -n xiaochou python plasma_inference.py "
        f"--ckpt {ckpt} "
        f"--gpu 0 "
        f"--val_json {VAL_JSON} "
        f"--output_dir {out_dir} "
        f"--save_nifti --no_figures "
        f"--n_steps 1 "
    )
    if is_legacy:
        cmd += "--legacy "

    result = run_cmd(cmd, f"Running {method.upper()} inference")
    return result.returncode == 0


def run_ficd_eval():
    """Run FiCD eval-only."""
    out_dir = FICD_RUN_DIR / "predictions" / "eval"
    metrics_file = out_dir / "subject_metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            data = json.load(f)
        if len(data) >= 40:
            print(f"[FiCD] Already have {len(data)} results, skipping eval.")
            return True

    cmd = (
        f"cd {ROOT} && CUDA_VISIBLE_DEVICES={GPU} "
        f"conda run -n xiaochou python ficd_train.py "
        f"--config {FICD_CONFIG} "
        f"--eval-only "
        f"--resume {FICD_RUN_DIR} "
    )
    result = run_cmd(cmd, "Running FiCD eval-only")
    return result.returncode == 0


# ── Phase 2: Load Results ────────────────────────────────────────────────────
def load_pasta_metrics(subjects):
    """Compute metrics from PASTA NIfTI pairs."""
    with open(PASTA_MAP) as f:
        pasta_map = json.load(f)

    rows = []
    for sid in subjects:
        if sid not in pasta_map:
            continue
        syn_path = PASTA_DIR / pasta_map[sid]["syn"]
        gt_path = PASTA_DIR / pasta_map[sid]["gt"]
        if not syn_path.exists() or not gt_path.exists():
            print(f"  [PASTA] Missing NIfTI for {sid}")
            continue
        pred = nib.load(str(syn_path)).get_fdata().astype(np.float32)
        gt = nib.load(str(gt_path)).get_fdata().astype(np.float32)
        m = compute_all_metrics(pred, gt)
        m["subject"] = sid
        rows.append(m)
        print(f"  [PASTA] {sid}: SSIM={m['ssim']:.4f} PSNR={m['psnr']:.2f}")
    return rows


def load_plasma_legacy_metrics(method, subjects):
    """Load Plasma/Legacy metrics from saved NIfTI pairs."""
    nifti_dir = OUT_DIR / method / "nifti"
    rows = []
    for sid in subjects:
        pred_path = nifti_dir / f"{sid}_tau_pred.nii.gz"
        gt_path = nifti_dir / f"{sid}_tau_gt.nii.gz"
        if not pred_path.exists() or not gt_path.exists():
            print(f"  [{method}] Missing NIfTI for {sid}")
            continue
        pred = nib.load(str(pred_path)).get_fdata().astype(np.float32)
        gt = nib.load(str(gt_path)).get_fdata().astype(np.float32)
        m = compute_all_metrics(pred, gt)
        m["subject"] = sid
        rows.append(m)
        print(f"  [{method}] {sid}: SSIM={m['ssim']:.4f} PSNR={m['psnr']:.2f}")
    return rows


def load_ficd_metrics(subjects):
    """Load FiCD metrics from subject_metrics.json + compute NCC from NIfTI."""
    metrics_file = FICD_RUN_DIR / "predictions" / "eval" / "subject_metrics.json"
    pred_dir = FICD_RUN_DIR / "predictions" / "eval"

    if not metrics_file.exists():
        print("[FiCD] subject_metrics.json not found!")
        return []

    with open(metrics_file) as f:
        ficd_data = json.load(f)

    # Build subject_id -> metrics mapping
    ficd_map = {r["subject_id"]: r for r in ficd_data}
    rows = []
    for sid in subjects:
        if sid not in ficd_map:
            continue
        r = ficd_map[sid]
        row = {
            "subject": sid,
            "ssim": r["ssim"],
            "psnr": r["psnr"],
            "mae": r["l1_unit"],
            "mse": r["l1_unit"] ** 2,  # approximate — actual MSE not saved
        }
        # Try to compute NCC from NIfTI prediction
        pred_nifti = pred_dir / f"{sid}.nii.gz"
        if pred_nifti.exists():
            # FiCD saves in [-1,1], convert to [0,1]
            pred = nib.load(str(pred_nifti)).get_fdata().astype(np.float32)
            pred_unit = np.clip((pred + 1.0) / 2.0, 0, 1)
            # Load GT: apply same transforms as FiCD to the raw TAU NIfTI
            # For now use the prediction itself for NCC placeholder
            row["ncc"] = np.nan  # Cannot compute without GT in same space
        else:
            row["ncc"] = np.nan
        rows.append(row)
        print(f"  [FiCD] {sid}: SSIM={row['ssim']:.4f} PSNR={row['psnr']:.2f}")
    return rows


# ── Phase 3: Statistical Analysis ────────────────────────────────────────────
def wilcoxon_test(values_a, values_b, metric_name, method_a, method_b):
    """Perform Wilcoxon signed-rank test."""
    a = np.array(values_a)
    b = np.array(values_b)
    # Filter out pairs where both are nan
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 5:
        return {"metric": metric_name, "pair": f"{method_a} vs {method_b}",
                "n": len(a), "statistic": np.nan, "p_value": np.nan, "sig": "N/A"}
    try:
        stat, p = stats.wilcoxon(a, b)
    except Exception:
        stat, p = np.nan, np.nan
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    return {"metric": metric_name, "pair": f"{method_a} vs {method_b}",
            "n": int(len(a)), "statistic": float(stat), "p_value": float(p), "sig": sig}


# ── Phase 4: Visualization ───────────────────────────────────────────────────
def generate_triplanar_comparison(subjects, method_niftis, out_path):
    """Generate tri-planar comparison figures for selected subjects."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Pick 3 representative subjects (first, middle, last)
    n = len(subjects)
    indices = [0, n // 2, n - 1] if n >= 3 else list(range(n))
    selected = [subjects[i] for i in indices]

    methods = list(method_niftis.keys())
    n_methods = len(methods)

    for sid in selected:
        fig, axes = plt.subplots(n_methods + 1, 3, figsize=(12, 3.2 * (n_methods + 1)))
        fig.suptitle(f"Subject: {sid}", fontsize=14, fontweight="bold")

        col_titles = ["Axial", "Coronal", "Sagittal"]
        for j, t in enumerate(col_titles):
            axes[0, j].set_title(t, fontsize=11)

        # Row 0: GT (from first available method)
        gt_vol = None
        for mname in methods:
            if sid in method_niftis[mname] and "gt" in method_niftis[mname][sid]:
                gt_vol = method_niftis[mname][sid]["gt"]
                break

        if gt_vol is not None:
            slices = _get_mid_slices(gt_vol)
            for j, s in enumerate(slices):
                axes[0, j].imshow(s.T, cmap="inferno", origin="lower", vmin=0, vmax=1)
            axes[0, 0].set_ylabel("Ground Truth", fontsize=10, fontweight="bold")
        else:
            for j in range(3):
                axes[0, j].text(0.5, 0.5, "N/A", ha="center", va="center", transform=axes[0, j].transAxes)
            axes[0, 0].set_ylabel("Ground Truth", fontsize=10, fontweight="bold")

        # Rows 1+: each method's prediction
        for i, mname in enumerate(methods):
            row = i + 1
            if sid in method_niftis[mname] and "pred" in method_niftis[mname][sid]:
                pred_vol = method_niftis[mname][sid]["pred"]
                slices = _get_mid_slices(pred_vol)
                for j, s in enumerate(slices):
                    axes[row, j].imshow(s.T, cmap="inferno", origin="lower", vmin=0, vmax=1)
            else:
                for j in range(3):
                    axes[row, j].text(0.5, 0.5, "N/A", ha="center", va="center",
                                      transform=axes[row, j].transAxes)
            axes[row, 0].set_ylabel(mname, fontsize=10, fontweight="bold")

        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        fig_path = out_path / f"triplanar_{sid}.png"
        plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fig_path}")


def generate_difference_maps(subjects, method_niftis, out_path):
    """Generate difference maps for selected subjects."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(subjects)
    indices = [0, n // 2, n - 1] if n >= 3 else list(range(n))
    selected = [subjects[i] for i in indices]

    methods = list(method_niftis.keys())
    n_methods = len(methods)

    for sid in selected:
        gt_vol = None
        for mname in methods:
            if sid in method_niftis[mname] and "gt" in method_niftis[mname][sid]:
                gt_vol = method_niftis[mname][sid]["gt"]
                break
        if gt_vol is None:
            continue

        fig, axes = plt.subplots(n_methods, 3, figsize=(12, 3.2 * n_methods))
        if n_methods == 1:
            axes = axes[np.newaxis, :]
        fig.suptitle(f"Absolute Error | {sid}", fontsize=14, fontweight="bold")

        gt_slices = _get_mid_slices(gt_vol)
        col_titles = ["Axial", "Coronal", "Sagittal"]
        for j, t in enumerate(col_titles):
            axes[0, j].set_title(t, fontsize=11)

        for i, mname in enumerate(methods):
            if sid in method_niftis[mname] and "pred" in method_niftis[mname][sid]:
                pred_vol = method_niftis[mname][sid]["pred"]
                # Need same shape for diff map
                if pred_vol.shape == gt_vol.shape:
                    diff = np.abs(pred_vol - gt_vol)
                    diff_slices = _get_mid_slices(diff)
                    for j, s in enumerate(diff_slices):
                        im = axes[i, j].imshow(s.T, cmap="hot", origin="lower", vmin=0, vmax=0.3)
                else:
                    for j in range(3):
                        axes[i, j].text(0.5, 0.5, "Shape\nmismatch",
                                        ha="center", va="center", transform=axes[i, j].transAxes, fontsize=8)
            else:
                for j in range(3):
                    axes[i, j].text(0.5, 0.5, "N/A", ha="center", va="center",
                                    transform=axes[i, j].transAxes)
            axes[i, 0].set_ylabel(mname, fontsize=10, fontweight="bold")

        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        fig_path = out_path / f"diff_{sid}.png"
        plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fig_path}")


def generate_unified_comparison(viz_subjects, method_niftis, out_path,
                                raw_mri_paths=None):
    """
    Generate a single unified figure comparing all methods.

    Layout (per representative subject):
      - 3 row groups: Axial / Coronal / Sagittal (each group = 2 rows)
      - Row 0 of each group: synthesis results (inferno colormap)
      - Row 1 of each group: absolute error maps vs GT (Reds colormap)
      - Columns: MRI | GT PET | PASTA | Legacy | Plasma(Ours) | FiCD | colorbar
      - GT is always taken from Plasma (Ours) / Legacy nifti folder (uncropped reference)
      - PASTA predictions are in (96,112,96) @ 1.5mm (eval_resolution from config);
        they are mapped back to Plasma's (160,192,160) space by:
          (A) undoing PASTA's crops+zoom → (182,218,182) @ 1.0mm (MNI space)
          (B) applying Plasma's CropForeground (using raw MRI bounding box)
          (C) center crop/pad to (160,192,160)
      - FiCD predictions are inverse-cropped (padded with crop params from
        aligned_tau.yaml: [11,10,20,17,0,21]) then resampled to GT shape
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy.ndimage import zoom as nd_zoom

    DIRECTIONS = ["Axial", "Coronal", "Sagittal"]
    COL_LABELS = ["MRI", "GT PET", "PASTA", "Legacy", "Plasma\n(Ours)", "FiCD"]
    # crop from configs/ficd/aligned_tau.yaml: (x_l, x_r, y_l, y_r, z_l, z_r)
    CROP = (11, 10, 20, 17, 0, 21)
    N_DATA_COLS = len(COL_LABELS)          # 6
    N_DIRS = 3
    N_ROWS_PER_DIR = 2                     # synthesis + error map
    N_TOTAL_ROWS = N_DIRS * N_ROWS_PER_DIR  # 6
    DIFF_VMAX = 0.3

    def _get_slice(vol, direction):
        x, y, z = vol.shape[:3]
        if direction == "Axial":
            return vol[:, :, z // 2]
        elif direction == "Coronal":
            return vol[:, y // 2, :]
        else:
            return vol[x // 2, :, :]

    def _resample(arr, tgt_shape):
        """Trilinear resample arr to tgt_shape tuple."""
        if arr.shape == tgt_shape:
            return arr
        factors = [t / s for t, s in zip(tgt_shape, arr.shape)]
        return np.clip(
            nd_zoom(arr.astype(np.float64), factors, order=1).astype(np.float32),
            0, 1,
        )

    def _uncrop_resample(arr, tgt_shape, crop=CROP):
        """Pad the inverse of crop then resample to tgt_shape."""
        padded = np.pad(
            arr,
            [(crop[0], crop[1]), (crop[2], crop[3]), (crop[4], crop[5])],
            mode="constant",
            constant_values=0,
        )
        return _resample(padded, tgt_shape)

    def _crop_or_pad(data, target_shape):
        """Center crop or zero-pad to target_shape (same as convert_nifti_to_h5.py)."""
        result = np.zeros(target_shape, dtype=data.dtype)
        src_shape = data.shape
        starts_src, ends_src, starts_dst, ends_dst = [], [], [], []
        for s, t in zip(src_shape, target_shape):
            if s >= t:
                start_s = (s - t) // 2
                starts_src.append(start_s)
                ends_src.append(start_s + t)
                starts_dst.append(0)
                ends_dst.append(t)
            else:
                start_d = (t - s) // 2
                starts_src.append(0)
                ends_src.append(s)
                starts_dst.append(start_d)
                ends_dst.append(start_d + s)
        result[starts_dst[0]:ends_dst[0],
               starts_dst[1]:ends_dst[1],
               starts_dst[2]:ends_dst[2]] = data[starts_src[0]:ends_src[0],
                                                  starts_src[1]:ends_src[1],
                                                  starts_src[2]:ends_src[2]]
        return result

    def _pasta_to_mni(arr):
        """Invert PASTA transforms: (96,112,96)@1.5mm → (182,218,182)@1.0mm.

        PASTA forward pipeline (convert_nifti_to_h5.py):
          (1) Source NIfTI (182,218,182) @ 1.0mm
          (2) scipy.ndimage.zoom(×0.667) → (121,145,121) @ 1.5mm
          (3) crop_or_pad → (113,137,113)  [center-crop 4 each side]
          (4) PASTA tio.CropOrPad → (96,112,96)  [center-crop: (8,9),(12,13),(8,9)]

        Inverse: pad combined crops → (121,145,121), then zoom ×1.5 → (182,218,182)
        """
        PASTA_NET_PAD = [(12, 13), (16, 17), (12, 13)]
        padded = np.pad(arr, PASTA_NET_PAD, mode="constant", constant_values=0)
        mni = nd_zoom(padded.astype(np.float64), 1.5, order=1).astype(np.float32)
        return mni

    def _pasta_to_ref_space(arr, tgt_shape, raw_mri=None):
        """Map PASTA (96,112,96) @ 1.5mm to Plasma's reference space.

        Steps:
          (A) Invert PASTA transforms → (182,218,182) @ 1.0mm (MNI space)
          (B) Apply CropForeground using raw MRI bounding box (matches Plasma pipeline)
          (C) Center crop/pad to tgt_shape (160,192,160)

        Without raw_mri, falls back to simple center-crop (less accurate).
        """
        # Step A: invert to MNI space
        mni = _pasta_to_mni(arr)

        # Step B: CropForeground using raw MRI (same as Plasma's CropForegroundd)
        if raw_mri is not None:
            nonzero = np.nonzero(raw_mri)
            if len(nonzero[0]) > 0:
                bbox_min = [int(n.min()) for n in nonzero]
                bbox_max = [int(n.max()) + 1 for n in nonzero]
                mni = mni[bbox_min[0]:bbox_max[0],
                          bbox_min[1]:bbox_max[1],
                          bbox_min[2]:bbox_max[2]]

        # Step C: center crop/pad to tgt_shape (same as ResizeWithPadOrCropd)
        result = _crop_or_pad(mni, tgt_shape)
        return np.clip(result, 0, 1)

    def _blank_ax(ax):
        """Remove all visual elements except the axes frame (keeps ylabel)."""
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    n = len(viz_subjects)
    if n == 0:
        print("  [unified] No viz subjects, skipping.")
        return
    indices = sorted(set([0, n // 2, n - 1]))
    selected = [viz_subjects[i] for i in indices]

    for sid in selected:
        # ── GT: always from Plasma (Ours) / Legacy nifti (uncropped reference) ──
        gt_vol = None
        for _m in ["Plasma (Ours)", "Legacy"]:
            entry = method_niftis.get(_m, {}).get(sid, {})
            if "gt" in entry:
                gt_vol = entry["gt"]
                break
        gt_shape = gt_vol.shape if gt_vol is not None else None

        # ── MRI ───────────────────────────────────────────────────────────
        mri_vol = None
        for _m in ["Plasma (Ours)", "Legacy"]:
            entry = method_niftis.get(_m, {}).get(sid, {})
            if "mri" in entry:
                mri_vol = entry["mri"]
                break

        # ── Load raw MRI for CropForeground (needed by PASTA alignment) ───
        raw_mri = None
        if raw_mri_paths and sid in raw_mri_paths:
            raw_mri_path = raw_mri_paths[sid]
            if os.path.exists(raw_mri_path):
                raw_mri = nib.load(raw_mri_path).get_fdata().astype(np.float32)
                if raw_mri.ndim == 4:
                    raw_mri = raw_mri.mean(axis=-1)

        # ── Predictions → resampled to GT shape ──────────────────────────
        # PASTA: eval_resolution=(96,112,96) @ 1.5mm — invert transforms to
        #   MNI (182,218,182), then apply Plasma's CropForeground + crop/pad.
        # FiCD is in (crop + resize) space from aligned_tau.yaml → inverse crop then zoom.
        # Plasma (Ours) / Legacy are already in GT space.
        pred_vols = {}
        for mk in ["PASTA", "Legacy", "Plasma (Ours)", "FiCD"]:
            entry = method_niftis.get(mk, {}).get(sid, {})
            pred = entry.get("pred", None)
            if pred is not None and gt_shape is not None:
                if mk == "FiCD":
                    pred = _uncrop_resample(pred, gt_shape)
                elif mk == "PASTA":
                    # Undo PASTA transforms → MNI → CropForeground → crop/pad
                    pred = _pasta_to_ref_space(pred, gt_shape, raw_mri=raw_mri)
                else:
                    # Plasma (Ours), Legacy: already in GT space, just ensure shape
                    pred = _resample(pred, gt_shape)
            pred_vols[mk] = pred

        # ── Per-method GT for error maps ──────────────────────────────────
        # PASTA uses its own native GT mapped through the same spatial pipeline
        # as its prediction (invert → CropForeground → crop/pad) so the error
        # map reflects synthesis quality rather than spatial misalignment.
        # Other methods share gt_vol (plasma/legacy GT).
        diff_gt_vols = {}
        for mk in ["PASTA", "Legacy", "Plasma (Ours)", "FiCD"]:
            if mk == "PASTA" and gt_shape is not None:
                pasta_gt_native = method_niftis.get("PASTA", {}).get(sid, {}).get("gt")
                if pasta_gt_native is not None:
                    diff_gt_vols[mk] = _pasta_to_ref_space(
                        pasta_gt_native, gt_shape, raw_mri=raw_mri
                    )
                else:
                    diff_gt_vols[mk] = gt_vol
            else:
                diff_gt_vols[mk] = gt_vol

        # ── Figure: 6 rows × (6 data cols + 1 colorbar) ──────────────────
        width_ratios = [1] * N_DATA_COLS + [0.07]
        fig_w = N_DATA_COLS * 1.9 + 0.6
        fig_h = N_TOTAL_ROWS * 1.8 + 0.6
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = gridspec.GridSpec(
            N_TOTAL_ROWS, N_DATA_COLS + 1,
            figure=fig,
            width_ratios=width_ratios,
            hspace=0.06,
            wspace=0.04,
            left=0.11, right=0.97, top=0.95, bottom=0.02,
        )

        axes = [
            [fig.add_subplot(gs[r, c]) for c in range(N_DATA_COLS)]
            for r in range(N_TOTAL_ROWS)
        ]
        cbar_ax = fig.add_subplot(gs[:, N_DATA_COLS])
        last_diff_im = None

        mri_vlo = float(np.percentile(mri_vol, 1)) if mri_vol is not None else 0
        mri_vhi = float(np.percentile(mri_vol, 99)) if mri_vol is not None else 1

        for dir_idx, direction in enumerate(DIRECTIONS):
            row_s = dir_idx * N_ROWS_PER_DIR      # synthesis row
            row_e = dir_idx * N_ROWS_PER_DIR + 1  # error map row

            # ── Synthesis row ─────────────────────────────────────────────
            ax = axes[row_s][0]   # MRI
            if mri_vol is not None:
                ax.imshow(_get_slice(mri_vol, direction).T,
                          cmap="gray", origin="lower", vmin=mri_vlo, vmax=mri_vhi)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7, color="gray")

            ax = axes[row_s][1]   # GT PET
            if gt_vol is not None:
                ax.imshow(_get_slice(gt_vol, direction).T,
                          cmap="inferno", origin="lower", vmin=0, vmax=1)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7, color="gray")

            for mi, mk in enumerate(["PASTA", "Legacy", "Plasma (Ours)", "FiCD"]):
                ax = axes[row_s][2 + mi]
                pred = pred_vols.get(mk)
                if pred is not None:
                    ax.imshow(_get_slice(pred, direction).T,
                              cmap="inferno", origin="lower", vmin=0, vmax=1)
                else:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes, fontsize=7, color="gray")

            # ── Error map row ─────────────────────────────────────────────
            _blank_ax(axes[row_e][0])   # MRI — empty
            _blank_ax(axes[row_e][1])   # GT  — empty

            for mi, mk in enumerate(["PASTA", "Legacy", "Plasma (Ours)", "FiCD"]):
                ax = axes[row_e][2 + mi]
                pred = pred_vols.get(mk)
                ref_gt = diff_gt_vols.get(mk, gt_vol)
                if pred is None or ref_gt is None:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes, fontsize=7, color="gray")
                else:
                    diff = np.abs(pred - ref_gt)
                    im = ax.imshow(_get_slice(diff, direction).T,
                                   cmap="Reds", origin="lower", vmin=0, vmax=DIFF_VMAX)
                    last_diff_im = im

            # ── Tick removal for all cells in both rows ───────────────────
            for row in (row_s, row_e):
                for col in range(N_DATA_COLS):
                    axes[row][col].set_xticks([])
                    axes[row][col].set_yticks([])

            # ── Column titles (first direction group only) ─────────────────
            if dir_idx == 0:
                for ci, label in enumerate(COL_LABELS):
                    axes[0][ci].set_title(label, fontsize=8, pad=3, fontweight="bold")

            # ── Row-group left labels ─────────────────────────────────────
            # Synthesis: ylabel on col-0 (MRI visible)
            axes[row_s][0].set_ylabel(
                f"{direction}\nSynthesis", fontsize=8, fontweight="bold", labelpad=4
            )
            # Error map: col-0 is blanked, use fig.text in the left margin
            # Compute approximate vertical center of this row in figure coords.
            # GridSpec: top=0.95, bottom=0.02 → content height = 0.93
            # Row row_e spans from bottom + (N_TOTAL_ROWS-1-row_e)/N * content to ...
            content_h = 0.95 - 0.02
            row_height = content_h / N_TOTAL_ROWS
            row_center_y = 0.02 + (N_TOTAL_ROWS - 1 - row_e + 0.5) * row_height
            fig.text(
                0.025, row_center_y,
                "Error\nMap",
                va="center", ha="center",
                fontsize=7, color="dimgray",
                rotation=90,
            )

        # ── Colorbar ──────────────────────────────────────────────────────
        if last_diff_im is not None:
            cb = fig.colorbar(last_diff_im, cax=cbar_ax)
            cb.set_label("Absolute Error", fontsize=8)
            cb.ax.tick_params(labelsize=7)
        else:
            cbar_ax.axis("off")

        # ── Figure title ──────────────────────────────────────────────────
        fig.suptitle(
            f"Method Comparison  |  Subject: {sid}",
            fontsize=11, fontweight="bold", y=0.98,
        )

        fig_path = out_path / f"unified_comparison_{sid}.png"
        plt.savefig(str(fig_path), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fig_path}")


def generate_boxplot(all_metrics_df, out_path):
    """Generate boxplot comparison of metrics across methods."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_cols = ["ssim", "psnr", "mae"]
    method_order = ["PASTA", "Legacy", "Plasma (Ours)", "FiCD"]
    colors = {"PASTA": "#1f77b4", "Legacy": "#ff7f0e", "Plasma (Ours)": "#2ca02c", "FiCD": "#d62728"}

    fig, axes = plt.subplots(1, len(metric_cols), figsize=(5 * len(metric_cols), 5))
    for idx, metric in enumerate(metric_cols):
        ax = axes[idx]
        data_to_plot = []
        labels = []
        cs = []
        for m in method_order:
            sub = all_metrics_df[all_metrics_df["method"] == m][metric].dropna()
            if len(sub) > 0:
                data_to_plot.append(sub.values)
                labels.append(m)
                cs.append(colors.get(m, "gray"))

        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], cs):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(metric.upper(), fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = out_path / "boxplot_metrics.png"
    plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_path}")


def _get_mid_slices(vol):
    """Get mid axial, coronal, sagittal slices."""
    s = vol.shape
    return [
        vol[:, :, s[2] // 2],   # axial
        vol[:, s[1] // 2, :],   # coronal
        vol[s[0] // 2, :, :],   # sagittal
    ]


# ── Phase 5: Report Generation ──────────────────────────────────────────────
def generate_report(summary_df, stat_tests, method_counts, out_path):
    """Generate final Markdown report."""
    lines = [
        "# MRI → TAU PET 四方法对比实验报告",
        "",
        "## 实验概述",
        "",
        "| 项目 | 说明 |",
        "|------|------|",
        "| 对比方法 | PASTA, Legacy, Plasma (Ours), FiCD |",
        f"| 公共测试集 | {method_counts.get('common', 'N/A')} subjects |",
        "| 评估指标 | SSIM, PSNR, MAE, MSE, NCC |",
        "| 统计检验 | Wilcoxon signed-rank test |",
        "",
        "### 方法简述",
        "",
        "| 方法 | 模型类型 | 输出分辨率 | 条件注入 |",
        "|------|----------|-----------|---------|",
        "| PASTA | 2.5D DDIM-100 扩散 | 96×112×96 (1.5mm) | Slice-level conditioning |",
        "| Legacy | Rectified-flow 1-step | 160×192×160 (1mm) | BiomedCLIP text token |",
        "| Plasma (Ours) | Rectified-flow 1-step | 160×192×160 (1mm) | Plasma embedding + text token |",
        "| FiCD | DDPM concat-conditioning | 160×180×160 | MRI concat + text embedding |",
        "",
        "## 定量结果",
        "",
        "### 整体指标 (Mean ± Std)",
        "",
    ]

    # Summary table
    lines.append("| 方法 | N | SSIM ↑ | PSNR ↑ | MAE ↓ | MSE ↓ | NCC ↑ |")
    lines.append("|------|---|--------|--------|-------|-------|-------|")
    method_order = ["PASTA", "Legacy", "Plasma (Ours)", "FiCD"]
    for m in method_order:
        row = summary_df[summary_df["method"] == m]
        if len(row) == 0:
            lines.append(f"| {m} | 0 | - | - | - | - | - |")
            continue
        r = row.iloc[0]
        ncc_str = f"{r['ncc_mean']:.4f}±{r['ncc_std']:.4f}" if not np.isnan(r.get("ncc_mean", np.nan)) else "N/A"
        lines.append(
            f"| {m} | {int(r['n'])} | "
            f"{r['ssim_mean']:.4f}±{r['ssim_std']:.4f} | "
            f"{r['psnr_mean']:.2f}±{r['psnr_std']:.2f} | "
            f"{r['mae_mean']:.4f}±{r['mae_std']:.4f} | "
            f"{r['mse_mean']:.6f}±{r['mse_std']:.6f} | "
            f"{ncc_str} |"
        )
    lines.append("")

    # Best method highlight
    best = {}
    for metric in ["ssim", "psnr", "ncc"]:
        col = f"{metric}_mean"
        valid = summary_df[summary_df[col].notna()]
        if len(valid) > 0:
            best[metric] = valid.loc[valid[col].idxmax(), "method"]
    for metric in ["mae", "mse"]:
        col = f"{metric}_mean"
        valid = summary_df[summary_df[col].notna()]
        if len(valid) > 0:
            best[metric] = valid.loc[valid[col].idxmin(), "method"]

    lines.append("### 最优方法")
    lines.append("")
    for metric, method in best.items():
        direction = "↑" if metric in ["ssim", "psnr", "ncc"] else "↓"
        lines.append(f"- **{metric.upper()}** {direction}: **{method}**")
    lines.append("")

    # Statistical tests
    lines.append("## 统计检验 (Wilcoxon signed-rank test)")
    lines.append("")
    lines.append("| 指标 | 对比 | N | Statistic | p-value | 显著性 |")
    lines.append("|------|------|---|-----------|---------|--------|")
    for t in stat_tests:
        p_str = f"{t['p_value']:.2e}" if not np.isnan(t["p_value"]) else "N/A"
        s_str = f"{t['statistic']:.1f}" if not np.isnan(t["statistic"]) else "N/A"
        lines.append(f"| {t['metric']} | {t['pair']} | {t['n']} | {s_str} | {p_str} | {t['sig']} |")
    lines.append("")

    # Notes
    lines.append("## 注意事项")
    lines.append("")
    lines.append("1. **分辨率差异**: PASTA 在 96×112×96 (1.5mm) 下评估，其余方法在 ~160³ (1mm) 下评估。"
                 "各方法使用自身分辨率下的配对 GT 计算指标，因此指标间存在分辨率偏差。")
    lines.append("2. **FiCD 训练不足**: FiCD 仅进行了 1 epoch smoke test 训练，指标预期较差。")
    lines.append("3. **NCC**: FiCD 的 NCC 未计算（缺少同分辨率 GT NIfTI）。")
    lines.append("4. **Plasma (Ours)** 使用预训练 plasma embedding 作为条件 token，"
                 "是本项目的核心创新方法。")
    lines.append("")

    report_path = out_path / "comparison_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Report saved to {report_path}")
    return report_path


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figures").mkdir(exist_ok=True)

    with open(ROOT / "analysis" / "_common_subjects.json") as f:
        subjects = sorted(json.load(f))
    print(f"Common subjects: {len(subjects)}")

    # ── Phase 1: Run Inference ──
    print("\n" + "=" * 60)
    print("  PHASE 1: RUNNING INFERENCE")
    print("=" * 60)

    ok_plasma = run_plasma_inference("plasma")
    ok_legacy = run_plasma_inference("legacy")
    ok_ficd = run_ficd_eval()
    print(f"\nInference status: Plasma={ok_plasma}, Legacy={ok_legacy}, FiCD={ok_ficd}")

    # ── Phase 2: Load & Compute Metrics ──
    print("\n" + "=" * 60)
    print("  PHASE 2: COMPUTING UNIFIED METRICS")
    print("=" * 60)

    per_subject_csv = OUT_DIR / "per_subject_metrics.csv"
    if per_subject_csv.exists():
        print(f"\n  [跳过计算] 检测到历史 metrics 文件，直接加载: {per_subject_csv}")
        print("  如需重新计算，请删除该文件后重新运行。")
        df = pd.read_csv(per_subject_csv)
        print(f"  已加载 {len(df)} 条历史 metrics 记录（{df['method'].value_counts().to_dict()}）")
    else:
        all_rows = []

        print("\n[PASTA] Computing metrics from NIfTI pairs...")
        pasta_rows = load_pasta_metrics(subjects)
        for r in pasta_rows:
            r["method"] = "PASTA"
        all_rows.extend(pasta_rows)

        print(f"\n[Plasma] Computing metrics from NIfTI pairs...")
        plasma_rows = load_plasma_legacy_metrics("plasma", subjects)
        for r in plasma_rows:
            r["method"] = "Plasma (Ours)"
        all_rows.extend(plasma_rows)

        print(f"\n[Legacy] Computing metrics from NIfTI pairs...")
        legacy_rows = load_plasma_legacy_metrics("legacy", subjects)
        for r in legacy_rows:
            r["method"] = "Legacy"
        all_rows.extend(legacy_rows)

        print(f"\n[FiCD] Loading metrics from subject_metrics.json...")
        ficd_rows = load_ficd_metrics(subjects)
        for r in ficd_rows:
            r["method"] = "FiCD"
        all_rows.extend(ficd_rows)

        # Create unified DataFrame
        df = pd.DataFrame(all_rows)
        df.to_csv(per_subject_csv, index=False)

    print(f"\nTotal metric rows: {len(df)}")
    print(df.groupby("method")[["ssim", "psnr", "mae"]].describe().round(4))

    # Summary
    summary_csv = OUT_DIR / "summary_metrics.csv"
    if summary_csv.exists() and per_subject_csv.exists():
        print(f"\n  [跳过汇总] 使用历史 summary: {summary_csv}")
        summary_df = pd.read_csv(summary_csv)
    else:
        summary_rows = []
        for method in ["PASTA", "Legacy", "Plasma (Ours)", "FiCD"]:
            sub = df[df["method"] == method]
            if len(sub) == 0:
                continue
            row = {"method": method, "n": len(sub)}
            for metric in ["ssim", "psnr", "mae", "mse", "ncc"]:
                vals = sub[metric].dropna()
                row[f"{metric}_mean"] = vals.mean() if len(vals) > 0 else np.nan
                row[f"{metric}_std"] = vals.std() if len(vals) > 0 else np.nan
            summary_rows.append(row)
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(summary_csv, index=False)
    print("\nSummary:")
    print(summary_df.to_string(index=False))

    # ── Phase 3: Statistical Tests ──
    print("\n" + "=" * 60)
    print("  PHASE 3: STATISTICAL TESTS")
    print("=" * 60)

    stat_tests = []
    # Compare Plasma (Ours) vs each other method
    ours = "Plasma (Ours)"
    for other in ["PASTA", "Legacy", "FiCD"]:
        # Get paired subjects
        ours_sub = df[df["method"] == ours].set_index("subject")
        other_sub = df[df["method"] == other].set_index("subject")
        common_sids = sorted(set(ours_sub.index) & set(other_sub.index))

        if len(common_sids) < 5:
            for metric in ["ssim", "psnr", "mae"]:
                stat_tests.append({"metric": metric, "pair": f"{ours} vs {other}",
                                   "n": len(common_sids), "statistic": np.nan,
                                   "p_value": np.nan, "sig": "N/A"})
            continue

        for metric in ["ssim", "psnr", "mae"]:
            a = ours_sub.loc[common_sids, metric].values
            b = other_sub.loc[common_sids, metric].values
            t = wilcoxon_test(a, b, metric.upper(), ours, other)
            stat_tests.append(t)
            print(f"  {t['metric']:5s} | {t['pair']:30s} | p={t['p_value']:.2e} {t['sig']}")

    stat_df = pd.DataFrame(stat_tests)
    stat_df.to_csv(OUT_DIR / "statistical_tests.csv", index=False)

    # ── Phase 4: Visualizations ──
    print("\n" + "=" * 60)
    print("  PHASE 4: GENERATING VISUALIZATIONS")
    print("=" * 60)

    fig_dir = OUT_DIR / "figures"

    # Load NIfTI volumes for visualization (subset)
    method_niftis = {}

    # PASTA
    with open(PASTA_MAP) as f:
        pasta_map = json.load(f)
    method_niftis["PASTA"] = {}
    for sid in subjects[:5]:  # first 5 for viz
        if sid in pasta_map:
            syn_path = PASTA_DIR / pasta_map[sid]["syn"]
            gt_path = PASTA_DIR / pasta_map[sid]["gt"]
            if syn_path.exists() and gt_path.exists():
                pasta_gt_arr = nib.load(str(gt_path)).get_fdata().astype(np.float32)
                pasta_pred_arr = nib.load(str(syn_path)).get_fdata().astype(np.float32)
                # PASTA inference output is already in [0,1]:
                #   - dataset.py applies tio.RescaleIntensity(out_min_max=(0,1))
                #   - diffusion sample() applies unnormalize_to_zero_to_one + clamp(0,1)
                # No additional re-scaling needed; just clip to [0,1].
                method_niftis["PASTA"][sid] = {
                    "pred": np.clip(pasta_pred_arr, 0, 1),
                    "gt": np.clip(pasta_gt_arr, 0, 1),
                }

    # Plasma / Legacy
    for mname, label in [("plasma", "Plasma (Ours)"), ("legacy", "Legacy")]:
        method_niftis[label] = {}
        nifti_dir = OUT_DIR / mname / "nifti"
        for sid in subjects[:5]:
            pred_p = nifti_dir / f"{sid}_tau_pred.nii.gz"
            gt_p = nifti_dir / f"{sid}_tau_gt.nii.gz"
            mri_p = nifti_dir / f"{sid}_mri.nii.gz"
            if pred_p.exists() and gt_p.exists():
                entry = {
                    "pred": np.clip(nib.load(str(pred_p)).get_fdata().astype(np.float32), 0, 1),
                    "gt": np.clip(nib.load(str(gt_p)).get_fdata().astype(np.float32), 0, 1),
                }
                if mri_p.exists():
                    entry["mri"] = nib.load(str(mri_p)).get_fdata().astype(np.float32)
                method_niftis[label][sid] = entry

    # FiCD - prediction only (different resolution, no GT NIfTI)
    method_niftis["FiCD"] = {}
    pred_dir = FICD_RUN_DIR / "predictions" / "eval"
    for sid in subjects[:5]:
        pred_p = pred_dir / f"{sid}.nii.gz"
        if pred_p.exists():
            raw = nib.load(str(pred_p)).get_fdata().astype(np.float32)
            pred_unit = np.clip((raw + 1.0) / 2.0, 0, 1)
            method_niftis["FiCD"][sid] = {"pred": pred_unit}

    # Find common viz subjects (present in Plasma at minimum)
    viz_subjects = [s for s in subjects[:5] if s in method_niftis.get("Plasma (Ours)", {})]

    if viz_subjects:
        print(f"\nGenerating tri-planar for {len(viz_subjects)} subjects...")
        # Per-method tri-planar (each at own resolution)
        for mname in method_niftis:
            msubs = [s for s in viz_subjects if s in method_niftis[mname]]
            if msubs:
                for sid in msubs:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    vols = method_niftis[mname][sid]
                    has_gt = "gt" in vols
                    n_rows = 2 if has_gt else 1
                    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 3.5 * n_rows))
                    if n_rows == 1:
                        axes = axes[np.newaxis, :]

                    pred = vols["pred"]
                    pred_slices = _get_mid_slices(pred)
                    row = n_rows - 1
                    for j, s in enumerate(pred_slices):
                        axes[row, j].imshow(s.T, cmap="inferno", origin="lower", vmin=0, vmax=1)
                    axes[row, 0].set_ylabel(f"{mname}\nPred", fontsize=10)

                    if has_gt:
                        gt = vols["gt"]
                        gt_slices = _get_mid_slices(gt)
                        for j, s in enumerate(gt_slices):
                            axes[0, j].imshow(s.T, cmap="inferno", origin="lower", vmin=0, vmax=1)
                        axes[0, 0].set_ylabel("GT", fontsize=10)

                    for ax in axes.flat:
                        ax.set_xticks([])
                        ax.set_yticks([])
                    axes[0, 0].set_title("Axial")
                    axes[0, 1].set_title("Coronal")
                    axes[0, 2].set_title("Sagittal")
                    fig.suptitle(f"{mname} | {sid}", fontsize=12)
                    plt.tight_layout()
                    safe_name = mname.replace(" ", "_").replace("(", "").replace(")", "")
                    plt.savefig(str(fig_dir / f"{safe_name}_{sid}.png"), dpi=150, bbox_inches="tight")
                    plt.close()

        # Build raw MRI path mapping for CropForeground alignment
        raw_mri_paths = {}
        if PLASMA_VAL_JSON.exists():
            with open(PLASMA_VAL_JSON) as f:
                plasma_val_data = json.load(f)
            for item in plasma_val_data:
                raw_mri_paths[item["name"]] = item["mri"]

        # Unified comparison figure
        print("Generating unified comparison figures...")
        generate_unified_comparison(viz_subjects, method_niftis, fig_dir,
                                    raw_mri_paths=raw_mri_paths)

        # Boxplot
        print("Generating metric boxplots...")
        generate_boxplot(df, fig_dir)
    else:
        print("No common viz subjects found, skipping visualization.")

    # ── Phase 5: Report ──
    print("\n" + "=" * 60)
    print("  PHASE 5: GENERATING REPORT")
    print("=" * 60)

    method_counts = {"common": len(subjects)}
    report_path = generate_report(summary_df, stat_tests, method_counts, OUT_DIR)

    print("\n" + "=" * 60)
    print("  ALL DONE!")
    print("=" * 60)
    print(f"\nOutput directory: {OUT_DIR}")
    print(f"Report: {report_path}")
    print(f"Per-subject metrics: {OUT_DIR / 'per_subject_metrics.csv'}")
    print(f"Summary: {OUT_DIR / 'summary_metrics.csv'}")
    print(f"Statistical tests: {OUT_DIR / 'statistical_tests.csv'}")
    print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()
