#!/usr/bin/env python3
"""
Phase A: Deployability metrics collection.

Collects: params, model_size_MB, train_time, inference_time,
          output_resolution, diffusion_steps, GPU_count.
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from comparison_common import (
    ROOT, METHOD_ORDER, METHOD_SPECS, EXTENDED_OUT_DIR, get_method_label,
)


# ── Checkpoint paths ─────────────────────────────────────────────────────────
CHECKPOINT_PATHS = {
    "pasta": ROOT / "pasta" / "replicaLT_comparison" / "results" / "2026-04-12_331111" / "best_val_model.pt",
    "legacy": ROOT / "runs" / "04.13_2916235" / "ckpt_epoch200.pt",
    "plasma": ROOT / "runs" / "plasma_04.12_4053491" / "ckpt_epoch200.pt",
    "ficd": ROOT / "runs" / "ficd_aligned_tau" / "260422.1177288" / "ckpt_epoch20.pt",
}

PASTA_TRAIN_LOG = ROOT / "pasta" / "replicaLT_comparison" / "results" / "2026-04-12_331111" / "train.log"
FICD_TRAIN_LOG = ROOT / "runs" / "ficd_aligned_tau" / "260422.1177288" / "train.log"

SUMMARY_JSONS = {
    "plasma": ROOT / "analysis" / "comparison_results" / "plasma" / "summary.json",
    "legacy": ROOT / "analysis" / "comparison_results" / "legacy" / "summary.json",
}

# ── Known values from logs / configs ─────────────────────────────────────────
KNOWN_VALUES = {
    "pasta": {
        "params": 89_095_710,
        "train_time_str": "71h45m",
        "train_time_hours": 71.75,
        "train_time_source": "train.log parsed",
        "diffusion_steps": 100,
        "output_resolution": "96x112x96 (1.5mm)",
    },
    "legacy": {
        "diffusion_steps": 1,
        "output_resolution": "160x192x160 (1mm)",
    },
    "plasma": {
        "diffusion_steps": 1,
        "output_resolution": "160x192x160 (1mm)",
    },
    "ficd": {
        "diffusion_steps": 1000,
        "output_resolution": "160x180x160",
    },
}


def count_params_from_checkpoint(ckpt_path: Path) -> int:
    """Count total parameters from a PyTorch checkpoint."""
    import torch
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # Try different keys
    for key in ["model_state_dict", "ema_model", "state_dict", "model"]:
        if key in ckpt:
            state_dict = ckpt[key]
            return sum(v.numel() for v in state_dict.values())

    # If checkpoint is itself a state dict
    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return sum(v.numel() for v in ckpt.values())

    raise ValueError(f"Cannot find state_dict in checkpoint: {ckpt_path}")


def get_model_size_mb(ckpt_path: Path) -> float:
    """Get checkpoint file size in MB."""
    return os.path.getsize(ckpt_path) / (1024 * 1024)


def parse_pasta_train_log(log_path: Path) -> dict:
    """Parse PASTA train.log for params and training time."""
    info = {}
    with open(log_path) as f:
        content = f.read()

    # Look for parameter count
    m = re.search(r"(\d[\d,]+)\s*param", content)
    if m:
        info["params"] = int(m.group(1).replace(",", ""))

    # Look for total time
    m = re.search(r"(\d+):(\d+):(\d+)", content[-2000:])
    if m:
        h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
        info["train_time_hours"] = h + mi / 60 + s / 3600

    return info


def parse_ficd_train_time(log_path: Path) -> dict:
    """Estimate FiCD training time from its train.log."""
    info = {}
    if not log_path.exists():
        return info

    import re
    from datetime import datetime

    timestamps = []
    with open(log_path) as f:
        for line in f:
            m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            if m:
                timestamps.append(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))

    if len(timestamps) >= 2:
        delta = timestamps[-1] - timestamps[0]
        info["train_time_hours"] = delta.total_seconds() / 3600
        info["train_time_source"] = "train.log parsed (ongoing)"

    # Check for epoch count
    epochs = []
    with open(log_path) as f:
        for line in f:
            m = re.search(r"Epoch\s+(\d+)", line)
            if m:
                epochs.append(int(m.group(1)))
    if epochs:
        info["epochs_completed"] = max(epochs)

    return info


def collect_inference_times() -> dict:
    """Collect inference times from summary.json files."""
    times = {}
    for method_id, json_path in SUMMARY_JSONS.items():
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            times[method_id] = {
                "inference_total_sec": data.get("elapsed_sec", np.nan),
                "n_subjects": data.get("n_samples", 43),
            }

    # PASTA: estimate from known ~3.59 s/it, 43 subjects
    # Actually PASTA inference is batch-based, use rough estimate
    times["pasta"] = {
        "inference_total_sec": np.nan,  # Not directly available
        "n_subjects": 43,
        "note": "batch inference, timing not separately logged",
    }

    # FiCD: ~2.66 min/subject * 43 = ~114 min = ~6840 sec
    times["ficd"] = {
        "inference_total_sec": 2.66 * 60 * 43,  # ~6862 sec
        "n_subjects": 43,
        "note": "estimated from ~2.66 min/subject (1000 DDPM steps)",
    }

    return times


def run_deployability(method_ids=None) -> pd.DataFrame:
    """
    Collect deployability metrics for all methods.
    Returns DataFrame with one row per method.
    """
    if method_ids is None:
        method_ids = METHOD_ORDER

    inference_times = collect_inference_times()
    rows = []

    for method_id in method_ids:
        label = get_method_label(method_id)
        known = KNOWN_VALUES.get(method_id, {})
        row = {
            "method": label,
            "method_id": method_id,
        }

        # Parameters
        ckpt_path = CHECKPOINT_PATHS.get(method_id)
        if method_id == "pasta":
            row["params"] = known.get("params", 89_095_710)
            row["params_source"] = "train.log"
        elif ckpt_path and ckpt_path.exists():
            try:
                row["params"] = count_params_from_checkpoint(ckpt_path)
                row["params_source"] = "checkpoint"
            except Exception as e:
                print(f"  [Warning] Could not count params for {method_id}: {e}")
                row["params"] = np.nan
                row["params_source"] = "failed"
        else:
            row["params"] = np.nan
            row["params_source"] = "missing"

        # Model size
        if ckpt_path and ckpt_path.exists():
            row["model_size_MB"] = get_model_size_mb(ckpt_path)
        else:
            row["model_size_MB"] = np.nan

        # Training time
        if method_id == "pasta":
            row["train_time_hours"] = known.get("train_time_hours", 71.75)
            row["train_time_source"] = known.get("train_time_source", "train.log parsed")
        elif method_id == "ficd" and FICD_TRAIN_LOG.exists():
            ficd_info = parse_ficd_train_time(FICD_TRAIN_LOG)
            row["train_time_hours"] = ficd_info.get("train_time_hours", np.nan)
            row["train_time_source"] = ficd_info.get("train_time_source", "extrapolated")
        else:
            # Plasma/Legacy: estimate from known training setup
            # Both trained 200 epochs, similar architecture
            row["train_time_hours"] = np.nan
            row["train_time_source"] = "not available"

        # Inference time
        inf = inference_times.get(method_id, {})
        total_sec = inf.get("inference_total_sec", np.nan)
        n_subj = inf.get("n_subjects", 43)
        row["inference_time_total_sec"] = total_sec
        row["inference_time_per_subject_sec"] = total_sec / n_subj if not np.isnan(total_sec) else np.nan
        row["inference_n_subjects"] = n_subj

        # Config-derived
        row["output_resolution"] = known.get("output_resolution",
                                              METHOD_SPECS[method_id]["summary"]["resolution"])
        row["diffusion_steps"] = known.get("diffusion_steps", np.nan)
        row["gpu_count_inference"] = 1

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def main():
    EXTENDED_OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  Phase A: Deployability Metrics")
    print("=" * 60)

    df = run_deployability()

    out_path = EXTENDED_OUT_DIR / "deployability_table.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
