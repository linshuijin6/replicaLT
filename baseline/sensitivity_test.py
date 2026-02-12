"""
baseline/sensitivity_test.py
===========================
条件敏感性测试：固定 MRI，改变 plasma（pT217）观察输出变化
"""

import os
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import nibabel as nib

import sys
sys.path.insert(0, str(Path(__file__).parent))

from .config import get_default_config
from .dataset import create_dataloaders
from .model import create_model
from .condition import PLASMA_FIELDS


def _save_nifti(volume: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    affine = np.eye(4)
    vol = volume.astype(np.float32, copy=False)
    nib.save(nib.Nifti1Image(vol, affine), path)


def _compute_plasma_norm_stats(train_df, tabular_stats) -> Dict[str, Dict[str, float]]:
    stats = {}
    for field in PLASMA_FIELDS:
        values = []
        for _, row in train_df.iterrows():
            raw = row.get(field)
            try:
                v = float(raw)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(v) or v <= -3.9:
                continue
            source = row.get("plasma_source")
            field_stats = tabular_stats.get_plasma_stats(source).get(field, (0.0, 1.0))
            median, iqr = field_stats
            if field in {"nfl_q", "gfap_q"}:
                v = np.log1p(v)
            normed = (v - median) / (iqr + 1e-8)
            values.append(normed)
        if len(values) == 0:
            stats[field] = {"mean": 0.0, "std": 1.0}
        else:
            stats[field] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return stats


def main():
    parser = argparse.ArgumentParser(description="FiLM 条件敏感性测试")
    parser.add_argument("checkpoint", type=str, help="模型 checkpoint 路径")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--num_samples", type=int, default=5, help="测试样本数量")
    args = parser.parse_args()

    config = get_default_config()
    if config.condition.mode == "none":
        raise ValueError("condition.mode=none，无法进行敏感性测试")

    if args.output_dir:
        config.output_dir = args.output_dir

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    _, _, test_loader, train_df, _, _ = create_dataloaders(config)

    model = create_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    tabular_stats = test_loader.dataset.tabular_stats
    plasma_stats = _compute_plasma_norm_stats(train_df, tabular_stats)
    pt217_idx = PLASMA_FIELDS.index("pt217_f")

    out_dir = os.path.join(config.output_dir, "sensitivity")
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= args.num_samples:
                break
            mri = batch["mri"].to(device)
            tau = batch["tau"].to(device)
            condition = {
                "clinical": batch["clinical"].to(device),
                "plasma": batch["plasma"].to(device),
                "clinical_mask": batch["clinical_mask"].to(device),
                "plasma_mask": batch["plasma_mask"].to(device),
                "sex": batch["sex"].to(device),
                "source": batch["source"].to(device),
            }

            plasma = condition["plasma"].clone()
            std = plasma_stats["pt217_f"]["std"]
            plasma_low = plasma.clone()
            plasma_high = plasma.clone()
            plasma_low[:, pt217_idx] = plasma[:, pt217_idx] - 2.0 * std
            plasma_high[:, pt217_idx] = plasma[:, pt217_idx] + 2.0 * std

            cond_real = dict(condition)
            cond_low = dict(condition)
            cond_high = dict(condition)
            cond_low["plasma"] = plasma_low
            cond_high["plasma"] = plasma_high

            pred_real = model(mri, cond_real)
            pred_low = model(mri, cond_low)
            pred_high = model(mri, cond_high)

            pred_real_np = pred_real[0, 0].cpu().numpy()
            pred_low_np = pred_low[0, 0].cpu().numpy()
            pred_high_np = pred_high[0, 0].cpu().numpy()
            tau_np = tau[0, 0].cpu().numpy()

            base_name = f"sample_{i:03d}"
            _save_nifti(pred_real_np, os.path.join(out_dir, f"{base_name}_real.nii.gz"))
            _save_nifti(pred_low_np, os.path.join(out_dir, f"{base_name}_low.nii.gz"))
            _save_nifti(pred_high_np, os.path.join(out_dir, f"{base_name}_high.nii.gz"))
            _save_nifti(tau_np, os.path.join(out_dir, f"{base_name}_gt.nii.gz"))

            diff_low = np.abs(pred_real_np - pred_low_np)
            diff_high = np.abs(pred_real_np - pred_high_np)
            _save_nifti(diff_low, os.path.join(out_dir, f"{base_name}_diff_low.nii.gz"))
            _save_nifti(diff_high, os.path.join(out_dir, f"{base_name}_diff_high.nii.gz"))

            print(f"[{base_name}] saved.")


if __name__ == "__main__":
    main()
