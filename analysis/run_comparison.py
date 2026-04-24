#!/usr/bin/env python3
"""
Unified MRI -> TAU PET comparison.
Methods: PASTA, Legacy, Plasma (Ours), FiCD

Usage:
    conda run -n xiaochou python analysis/run_comparison.py
"""

import json
import os
import subprocess
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats


# ── Config ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]

# RUN_MODE = "rerun_inference"  # "use_existing_results" | "rerun_inference"
RUN_MODE = "use_existing_results"
ENABLED_METHODS = ["pasta", "legacy", "plasma", "ficd"]
GPU = 5
OUT_DIR = ROOT / "analysis" / "comparison_results"
COMMON_SUBJECTS_JSON = ROOT / "analysis" / "_common_subjects.json"
VAL_SUBJECTS_JSON = OUT_DIR / "val_43_subjects.json"
PLASMA_VAL_JSON = ROOT / "val_data_with_description.json"
MAX_VIZ_SUBJECTS = 5
REFERENCE_VIS_SHAPE = (160, 192, 160)

METHOD_SPECS = {
    "pasta": {
        "label": "PASTA",
        "summary": {
            "model_type": "2.5D DDIM-100 diffusion",
            "resolution": "96×112×96 (1.5mm)",
            "conditioning": "Slice-level conditioning",
        },
        "inference_inputs": {
            "script_path": ROOT / "pasta" / "replicaLT_comparison" / "inference_pasta_replicaLT.py",
            "config_path": ROOT / "pasta" / "replicaLT_comparison" / "pasta_replicaLT.yaml",
            "test_data": ROOT / "pasta" / "data" / "test.h5",
            "ckpt": ROOT / "pasta" / "replicaLT_comparison" / "results" / "2026-04-12_331111" / "best_val_model.pt",
            "gpu_id": GPU,
            "batch_size": 32,
            "amp": "fp16",
        },
        "existing_results": {
            "output_dir": Path("/home/data/linshuijin/replicaLT/pasta/replicaLT_comparison/results/2026-04-12_331111/inference_output"),
        },
    },
    "legacy": {
        "label": "Legacy",
        "summary": {
            "model_type": "Rectified-flow 1-step",
            "resolution": "160×192×160 (1mm)",
            "conditioning": "BiomedCLIP text token",
        },
        "inference_inputs": {
            "script_path": ROOT / "plasma_inference.py",
            "ckpt": ROOT / "runs" / "04.13_2916235" / "ckpt_epoch200.pt",
            "val_json": VAL_SUBJECTS_JSON,
            "n_steps": 1,
            "legacy": True,
        },
        "existing_results": {
            "output_dir": Path("/mnt/nfsdata/nfsdata/lsj.14/replicaLT/analysis/comparison_results/legacy"),
        },
    },
    "plasma": {
        "label": "Plasma (Ours)",
        "summary": {
            "model_type": "Rectified-flow 1-step",
            "resolution": "160×192×160 (1mm)",
            "conditioning": "Plasma embedding + text token",
        },
        "inference_inputs": {
            "script_path": ROOT / "plasma_inference.py",
            "ckpt": ROOT / "runs" / "plasma_04.12_4053491" / "ckpt_epoch200.pt",
            "val_json": VAL_SUBJECTS_JSON,
            "n_steps": 1,
            "legacy": False,
        },
        "existing_results": {
            "output_dir": Path("/mnt/nfsdata/nfsdata/lsj.14/replicaLT/analysis/comparison_results/plasma"),
        },
    },
    "ficd": {
        "label": "FiCD",
        "summary": {
            "model_type": "DDPM concat-conditioning",
            "resolution": "160×180×160",
            "conditioning": "MRI concat + text embedding",
        },
        "inference_inputs": {
            "script_path": ROOT / "ficd" / "ficd_train.py",
            "config_path": ROOT / "ficd" / "aligned_tau.yaml",
            "checkpoint": ROOT / "runs" / "ficd_aligned_tau" / "260422.1177288" / "ckpt_epoch20.pt",
            "gpu_id": GPU,
        },
        "existing_results": {
            "run_dir": Path("/home/data/linshuijin/replicaLT/runs/ficd_aligned_tau/260422.1177288/inference"),
        },
    },
}

METHOD_ORDER = ["pasta", "legacy", "plasma", "ficd"]
METHOD_ALIGNMENT = {
    "pasta": "pasta",
    "legacy": "native",
    "plasma": "native",
    "ficd": "ficd",
}
FICD_DEFAULT_CROP = (11, 10, 20, 17, 0, 21)
FICD_DEFAULT_TARGET_SHAPE = (160, 180, 160)


def build_config():
    return {
        "root": ROOT,
        "run_mode": RUN_MODE,
        "enabled_methods": ENABLED_METHODS,
        "gpu": GPU,
        "out_dir": OUT_DIR,
        "common_subjects_json": COMMON_SUBJECTS_JSON,
        "val_subjects_json": VAL_SUBJECTS_JSON,
        "plasma_val_json": PLASMA_VAL_JSON,
        "max_viz_subjects": MAX_VIZ_SUBJECTS,
        "method_specs": METHOD_SPECS,
    }


CONFIG = build_config()


# ── Helpers ──────────────────────────────────────────────────────────────────
def require_exists(path, desc):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path}")
    return path


def output_paths(config):
    out_dir = config["out_dir"]
    return {
        "per_subject_csv": out_dir / "per_subject_metrics.csv",
        "summary_csv": out_dir / "summary_metrics.csv",
        "stats_csv": out_dir / "statistical_tests.csv",
        "report_md": out_dir / "comparison_report.md",
        "figures_dir": out_dir / "figures",
    }


def get_enabled_methods(config):
    enabled = config["enabled_methods"]
    invalid = [m for m in enabled if m not in METHOD_SPECS]
    if invalid:
        raise ValueError(f"Unknown methods in ENABLED_METHODS: {invalid}")
    return [m for m in METHOD_ORDER if m in enabled]


def get_method_label(method_id):
    return METHOD_SPECS[method_id]["label"]


def get_enabled_labels(config):
    return [get_method_label(m) for m in get_enabled_methods(config)]


def method_output_dir(config, method_id):
    return config["out_dir"] / method_id


def get_method_source(config, method_id):
    spec = config["method_specs"][method_id]
    if config["run_mode"] == "rerun_inference":
        if method_id == "ficd":
            return {"run_dir": method_output_dir(config, method_id)}
        return {"output_dir": method_output_dir(config, method_id)}
    return spec["existing_results"]


def resolve_ficd_prediction_dir(run_dir):
    run_dir = Path(run_dir)
    candidates = [
        run_dir,
        run_dir / "inference",
        run_dir / "predictions" / "eval",
        run_dir / "predictions" / "best_model",
    ]
    for path in candidates:
        if path.exists() and (path / "subject_metrics.json").exists():
            return path
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"FiCD prediction dir not found under: {run_dir}")


def run_cmd(cmd, desc=""):
    print(f"\n{'=' * 60}")
    print(f"  {desc}")
    print(f"  CMD: {cmd}")
    print(f"{'=' * 60}\n")
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.stdout:
        print(result.stdout[-4000:])
    if result.returncode != 0:
        if result.stderr:
            print(f"STDERR:\n{result.stderr[-4000:]}")
        raise RuntimeError(f"{desc} failed with exit code {result.returncode} after {elapsed:.0f}s")
    print(f"OK ({elapsed:.0f}s)")
    return result


def compute_ssim_3d(pred, gt, win_size=7):
    import torch
    from monai.metrics import SSIMMetric

    p = torch.tensor(pred, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    g = torch.tensor(gt, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    metric = SSIMMetric(spatial_dims=3, data_range=1.0, win_size=win_size)
    return metric(g, p).mean().item()


def compute_psnr(pred, gt, max_val=1.0):
    mse = np.mean((pred - gt) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(max_val ** 2 / mse)


def compute_ncc(pred, gt):
    p = pred.flatten().astype(np.float64)
    g = gt.flatten().astype(np.float64)
    p_mean = p - p.mean()
    g_mean = g - g.mean()
    num = np.sum(p_mean * g_mean)
    den = np.sqrt(np.sum(p_mean ** 2) * np.sum(g_mean ** 2))
    if den < 1e-10:
        return 0.0
    return float(num / den)


def compute_all_metrics(pred, gt):
    pred = np.clip(pred.astype(np.float64), 0, 1)
    gt = np.clip(gt.astype(np.float64), 0, 1)
    mae = np.mean(np.abs(pred - gt))
    mse = np.mean((pred - gt) ** 2)
    psnr = compute_psnr(pred, gt)
    ncc = compute_ncc(pred, gt)
    ssim = compute_ssim_3d(pred, gt)
    return {"ssim": ssim, "psnr": psnr, "mae": mae, "mse": mse, "ncc": ncc}


def validate_config(config):
    require_exists(config["root"], "workspace root")
    require_exists(config["common_subjects_json"], "common subjects json")
    require_exists(config["plasma_val_json"], "plasma val json")
    enabled_methods = get_enabled_methods(config)
    if not enabled_methods:
        raise ValueError("ENABLED_METHODS is empty")

    for method_id in enabled_methods:
        spec = config["method_specs"][method_id]
        if config["run_mode"] == "rerun_inference":
            inputs = spec["inference_inputs"]
            if method_id == "pasta":
                require_exists(inputs["script_path"], f"{method_id} script")
                require_exists(inputs["config_path"], f"{method_id} config")
                require_exists(inputs["test_data"], f"{method_id} test data")
                require_exists(inputs["ckpt"], f"{method_id} checkpoint")
            elif method_id in {"legacy", "plasma"}:
                require_exists(inputs["script_path"], f"{method_id} script")
                require_exists(inputs["ckpt"], f"{method_id} checkpoint")
                require_exists(inputs["val_json"], f"{method_id} val json")
            elif method_id == "ficd":
                require_exists(inputs["script_path"], f"{method_id} script")
                require_exists(inputs["config_path"], f"{method_id} config")
                require_exists(inputs["checkpoint"], f"{method_id} checkpoint")
        else:
            source = spec["existing_results"]
            if method_id == "ficd":
                require_exists(source["run_dir"], f"{method_id} existing run dir")
                pred_dir = resolve_ficd_prediction_dir(source["run_dir"])
                require_exists(pred_dir / "subject_metrics.json", f"{method_id} subject metrics")
            else:
                require_exists(source["output_dir"], f"{method_id} existing output dir")


def load_subject_records(config):
    with open(config["common_subjects_json"]) as f:
        raw_subjects = json.load(f)

    if not isinstance(raw_subjects, list):
        raise ValueError(f"Expected a JSON list in {config['common_subjects_json']}")
    if not raw_subjects:
        return []

    if all(isinstance(item, str) for item in raw_subjects):
        return [{"name": str(name), "examdate": None} for name in sorted(raw_subjects)]

    if not all(isinstance(item, dict) for item in raw_subjects):
        raise TypeError(
            "common_subjects_json must contain either a list of subject IDs "
            "or a list of subject metadata dicts"
        )

    common_names = {str(item["name"]) for item in raw_subjects if item.get("name")}
    if not common_names:
        raise ValueError(f"No valid subject names found in {config['common_subjects_json']}")

    val_subjects_path = config.get("val_subjects_json")
    if val_subjects_path and Path(val_subjects_path).exists():
        with open(val_subjects_path) as f:
            val_subjects = json.load(f)
        if isinstance(val_subjects, list):
            selected_records = []
            seen_names = set()
            for item in val_subjects:
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                if name and name in common_names:
                    if name in seen_names:
                        raise ValueError(
                            f"Duplicate subject name in {val_subjects_path}: {name}. "
                            "run_comparison.py still keys outputs by subject name, so "
                            "the comparison cohort must be unique by name."
                        )
                    selected_records.append(dict(item))
                    seen_names.add(name)
            if selected_records:
                return selected_records

    records = []
    for item in raw_subjects:
        if item.get("name"):
            records.append(
                {
                    "name": str(item["name"]),
                    "examdate": item.get("examdate"),
                }
            )
    return sorted(records, key=lambda row: row["name"])


def load_subjects(config):
    return [record["name"] for record in load_subject_records(config)]


def load_subject_lookup(config):
    for key in ("val_subjects_json", "common_subjects_json"):
        path = config.get(key)
        if not path or not Path(path).exists():
            continue

        with open(path) as f:
            records = json.load(f)

        if not isinstance(records, list):
            continue

        lookup = {}
        for item in records:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if name and name not in lookup:
                lookup[str(name)] = item
        if lookup:
            return lookup

    return {}


def _same_path(path_a, path_b):
    if not path_a or not path_b:
        return False
    return Path(path_a).resolve(strict=False) == Path(path_b).resolve(strict=False)


def find_ficd_run_root(source):
    source_run_dir = Path(source["run_dir"])
    pred_dir = resolve_ficd_prediction_dir(source_run_dir)
    candidates = [pred_dir, source_run_dir, pred_dir.parent, source_run_dir.parent]
    candidates.extend(pred_dir.parents)
    for path in candidates:
        if (path / "hparams.json").exists() or (path / "samples_val.json").exists():
            return path
    raise FileNotFoundError(f"FiCD run root with hparams/samples_val not found near: {source_run_dir}")


def load_ficd_preprocess_spec(config, source):
    run_root = find_ficd_run_root(source)
    hparams_path = run_root / "hparams.json"
    config_path = config["method_specs"]["ficd"]["inference_inputs"]["config_path"]

    if hparams_path.exists():
        with open(hparams_path) as f:
            payload = json.load(f)
        data_cfg = payload.get("data", {})
        source_path = hparams_path
    else:
        import yaml

        with open(config_path) as f:
            payload = yaml.safe_load(f) or {}
        data_cfg = payload.get("data", {})
        source_path = config_path

    crop = tuple(int(x) for x in data_cfg.get("crop", FICD_DEFAULT_CROP))
    target_shape = tuple(int(x) for x in data_cfg.get("target_shape", FICD_DEFAULT_TARGET_SHAPE))
    if len(crop) != 6:
        raise ValueError(f"FiCD crop must have 6 values in {source_path}, got: {crop}")
    if len(target_shape) != 3:
        raise ValueError(f"FiCD target_shape must have 3 values in {source_path}, got: {target_shape}")
    return {
        "run_root": run_root,
        "source_path": source_path,
        "crop": crop,
        "target_shape": target_shape,
    }


def validate_ficd_subject_records(config, subjects):
    if "ficd" not in get_enabled_methods(config):
        return None

    source = get_method_source(config, "ficd")
    spec = load_ficd_preprocess_spec(config, source)
    samples_path = require_exists(spec["run_root"] / "samples_val.json", "FiCD samples_val.json")
    with open(samples_path) as f:
        samples = json.load(f)
    sample_map = {}
    for sample in samples:
        key = (str(sample.get("subject_id")), str(sample.get("examdate")))
        sample_map.setdefault(key, []).append(sample)

    subject_lookup = load_subject_lookup(config)
    checked = 0
    for sid in subjects:
        record = subject_lookup.get(sid)
        if record is None:
            raise KeyError(f"[FiCD data check] Missing comparison record for {sid}")
        examdate = record.get("examdate")
        if not examdate:
            raise ValueError(f"[FiCD data check] Missing examdate for comparison subject {sid}")

        matches = sample_map.get((str(sid), str(examdate)), [])
        if len(matches) != 1:
            raise ValueError(
                f"[FiCD data check] Expected one FiCD val sample for "
                f"({sid}, {examdate}), got {len(matches)}"
            )
        ficd_sample = matches[0]
        if not _same_path(ficd_sample.get("mri_path"), record.get("mri")):
            raise ValueError(
                f"[FiCD data check] MRI path mismatch for {sid} {examdate}:\n"
                f"  FiCD: {ficd_sample.get('mri_path')}\n"
                f"  Ref:  {record.get('mri')}"
            )
        if not _same_path(ficd_sample.get("pet_path"), record.get("tau")):
            raise ValueError(
                f"[FiCD data check] TAU path mismatch for {sid} {examdate}:\n"
                f"  FiCD: {ficd_sample.get('pet_path')}\n"
                f"  Ref:  {record.get('tau')}"
            )
        checked += 1

    print(
        f"  [FiCD data check] {checked} subject-examdate pairs match "
        f"{samples_path}; crop={spec['crop']} target_shape={spec['target_shape']}"
    )
    return spec


# ── Inference ────────────────────────────────────────────────────────────────
def run_plasma_family_inference(config, method_id):
    spec = config["method_specs"][method_id]["inference_inputs"]
    out_dir = method_output_dir(config, method_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    legacy_flag = "--legacy " if spec["legacy"] else ""
    cmd = (
        f"cd {config['root']} && CUDA_VISIBLE_DEVICES={config['gpu']} "
        f"conda run -n xiaochou python {spec['script_path']} "
        f"--ckpt {spec['ckpt']} "
        f"--gpu 0 "
        f"--val_json {spec['val_json']} "
        f"--output_dir {out_dir} "
        f"--save_nifti --no_figures "
        f"--n_steps {spec['n_steps']} "
        f"{legacy_flag}"
    )
    run_cmd(cmd, f"Running {get_method_label(method_id)} inference")


def run_pasta_inference(config):
    spec = config["method_specs"]["pasta"]["inference_inputs"]
    out_dir = method_output_dir(config, "pasta")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = (
        f"cd {config['root']} && CUDA_VISIBLE_DEVICES={config['gpu']} "
        f"conda run -n xiaochou python {spec['script_path']} "
        f"--config {spec['config_path']} "
        f"--test_data {spec['test_data']} "
        f"--ckpt {spec['ckpt']} "
        f"--output_dir {out_dir} "
        f"--gpu_id {spec['gpu_id']} "
        f"--batch_size {spec['batch_size']} "
        f"--amp {spec['amp']}"
    )
    run_cmd(cmd, "Running PASTA inference")


def run_ficd_inference(config):
    spec = config["method_specs"]["ficd"]["inference_inputs"]
    run_dir = method_output_dir(config, "ficd")
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = (
        f"cd {config['root'] / 'ficd'} && CUDA_VISIBLE_DEVICES={config['gpu']} "
        f"conda run -n xiaochou python {spec['script_path']} "
        f"--config {spec['config_path']} "
        f"--resume {run_dir} "
        f"--checkpoint {spec['checkpoint']} "
        f"--eval-only "
        f"--gpu 0"
    )
    run_cmd(cmd, "Running FiCD eval-only")


def maybe_run_inference(config):
    if config["run_mode"] != "rerun_inference":
        print("\n[Mode] use_existing_results -> skip all inference.")
        return

    print("\n" + "=" * 60)
    print("  PHASE 1: RUNNING INFERENCE")
    print("=" * 60)
    for method_id in get_enabled_methods(config):
        if method_id == "pasta":
            run_pasta_inference(config)
        elif method_id in {"legacy", "plasma"}:
            run_plasma_family_inference(config, method_id)
        elif method_id == "ficd":
            run_ficd_inference(config)


# ── Metrics Loading ──────────────────────────────────────────────────────────
def find_pasta_pair(output_dir, sid, subject_lookup=None):
    output_dir = Path(output_dir)
    syn_candidates = sorted(output_dir.glob(f"{sid}_*_syn_pet.nii.gz"))
    gt_candidates = sorted(output_dir.glob(f"{sid}_*_GT_pet.nii.gz"))
    if len(syn_candidates) == 1 and len(gt_candidates) == 1:
        return syn_candidates[0], gt_candidates[0]

    target = (subject_lookup or {}).get(sid)
    if target is not None:
        examdate = str(target.get("examdate", "") or "").replace("-", "")
        diagnosis = str(target.get("diagnosis", "") or "")

        if examdate:
            syn_match = [p for p in syn_candidates if examdate in p.name]
            gt_match = [p for p in gt_candidates if examdate in p.name]

            if diagnosis:
                syn_diag_match = [p for p in syn_match if diagnosis in p.name]
                gt_diag_match = [p for p in gt_match if diagnosis in p.name]
                if len(syn_diag_match) == 1 and len(gt_diag_match) == 1:
                    return syn_diag_match[0], gt_diag_match[0]

            if len(syn_match) == 1 and len(gt_match) == 1:
                return syn_match[0], gt_match[0]

    raise FileNotFoundError(
        f"[PASTA] Expected one syn and one gt file for {sid}, "
        f"got syn={len(syn_candidates)}, gt={len(gt_candidates)} in {output_dir}"
    )


def load_pasta_metrics(config, subjects, source):
    output_dir = require_exists(source["output_dir"], "PASTA output dir")
    subject_lookup = load_subject_lookup(config)
    rows = []
    for sid in subjects:
        syn_path, gt_path = find_pasta_pair(output_dir, sid, subject_lookup)
        pred = nib.load(str(syn_path)).get_fdata().astype(np.float32)
        gt = nib.load(str(gt_path)).get_fdata().astype(np.float32)
        metrics = compute_all_metrics(pred, gt)
        metrics["subject"] = sid
        rows.append(metrics)
        print(f"  [PASTA] {sid}: SSIM={metrics['ssim']:.4f} PSNR={metrics['psnr']:.2f}")
    return rows


def load_plasma_family_metrics(method_id, subjects, source):
    nifti_dir = require_exists(Path(source["output_dir"]) / "nifti", f"{method_id} nifti dir")
    rows = []
    for sid in subjects:
        pred_path = require_exists(nifti_dir / f"{sid}_tau_pred.nii.gz", f"{method_id} pred nifti for {sid}")
        gt_path = require_exists(nifti_dir / f"{sid}_tau_gt.nii.gz", f"{method_id} gt nifti for {sid}")
        pred = nib.load(str(pred_path)).get_fdata().astype(np.float32)
        gt = nib.load(str(gt_path)).get_fdata().astype(np.float32)
        metrics = compute_all_metrics(pred, gt)
        metrics["subject"] = sid
        rows.append(metrics)
        print(f"  [{get_method_label(method_id)}] {sid}: SSIM={metrics['ssim']:.4f} PSNR={metrics['psnr']:.2f}")
    return rows


def load_ficd_metrics(subjects, source):
    run_dir = require_exists(source["run_dir"], "FiCD run dir")
    pred_dir = resolve_ficd_prediction_dir(run_dir)
    metrics_file = require_exists(pred_dir / "subject_metrics.json", "FiCD subject_metrics.json")
    with open(metrics_file) as f:
        ficd_data = json.load(f)

    ficd_map = {row["subject_id"]: row for row in ficd_data}
    rows = []
    for sid in subjects:
        if sid not in ficd_map:
            raise KeyError(f"[FiCD] Missing metrics row for subject {sid}")
        row_data = ficd_map[sid]
        row = {
            "subject": sid,
            "ssim": row_data["ssim"],
            "psnr": row_data["psnr"],
            "mae": row_data["l1_unit"],
            "mse": row_data["l1_unit"] ** 2,
        }
        pred_nifti = require_exists(pred_dir / f"{sid}.nii.gz", f"FiCD prediction nifti for {sid}")
        _ = pred_nifti
        row["ncc"] = np.nan
        rows.append(row)
        print(f"  [FiCD] {sid}: SSIM={row['ssim']:.4f} PSNR={row['psnr']:.2f}")
    return rows


def compute_metrics_dataframe(config, subjects):
    rows = []
    for method_id in get_enabled_methods(config):
        print(f"\n[{get_method_label(method_id)}] Loading metrics...")
        source = get_method_source(config, method_id)
        if method_id == "pasta":
            method_rows = load_pasta_metrics(config, subjects, source)
        elif method_id in {"legacy", "plasma"}:
            method_rows = load_plasma_family_metrics(method_id, subjects, source)
        else:
            method_rows = load_ficd_metrics(subjects, source)
        for row in method_rows:
            row["method"] = get_method_label(method_id)
        rows.extend(method_rows)

    return pd.DataFrame(rows)


# ── Statistics ───────────────────────────────────────────────────────────────
def wilcoxon_test(values_a, values_b, metric_name, method_a, method_b):
    a = np.array(values_a)
    b = np.array(values_b)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 5:
        return {
            "metric": metric_name,
            "pair": f"{method_a} vs {method_b}",
            "n": len(a),
            "statistic": np.nan,
            "p_value": np.nan,
            "sig": "N/A",
        }
    try:
        stat, p = stats.wilcoxon(a, b)
    except Exception:
        stat, p = np.nan, np.nan
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    return {
        "metric": metric_name,
        "pair": f"{method_a} vs {method_b}",
        "n": int(len(a)),
        "statistic": float(stat),
        "p_value": float(p),
        "sig": sig,
    }


def build_summary(df, enabled_labels):
    summary_rows = []
    for method in enabled_labels:
        sub = df[df["method"] == method]
        if len(sub) == 0:
            continue
        row = {"method": method, "n": len(sub)}
        for metric in ["ssim", "psnr", "mae", "mse", "ncc"]:
            vals = sub[metric].dropna()
            row[f"{metric}_mean"] = vals.mean() if len(vals) > 0 else np.nan
            row[f"{metric}_std"] = vals.std() if len(vals) > 0 else np.nan
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def run_statistical_tests(df, enabled_labels):
    if "Plasma (Ours)" in enabled_labels:
        anchor = "Plasma (Ours)"
        others = [label for label in enabled_labels if label != anchor]
    else:
        anchor = enabled_labels[0]
        others = enabled_labels[1:]

    stat_tests = []
    anchor_df = df[df["method"] == anchor].set_index("subject")
    for other in others:
        other_df = df[df["method"] == other].set_index("subject")
        common_sids = sorted(set(anchor_df.index) & set(other_df.index))
        if len(common_sids) < 5:
            for metric in ["ssim", "psnr", "mae"]:
                stat_tests.append(
                    {
                        "metric": metric.upper(),
                        "pair": f"{anchor} vs {other}",
                        "n": len(common_sids),
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "sig": "N/A",
                    }
                )
            continue

        for metric in ["ssim", "psnr", "mae"]:
            a = anchor_df.loc[common_sids, metric].values
            b = other_df.loc[common_sids, metric].values
            test = wilcoxon_test(a, b, metric.upper(), anchor, other)
            stat_tests.append(test)
            print(f"  {test['metric']:5s} | {test['pair']:30s} | p={test['p_value']:.2e} {test['sig']}")

    return stat_tests


# ── Visualization ────────────────────────────────────────────────────────────
def _get_mid_slices(vol):
    s = vol.shape
    return [
        vol[:, :, s[2] // 2],
        vol[:, s[1] // 2, :],
        vol[s[0] // 2, :, :],
    ]


def load_raw_mri_paths(config):
    subject_lookup = load_subject_lookup(config)
    if subject_lookup:
        return {name: item["mri"] for name, item in subject_lookup.items() if item.get("mri")}

    with open(config["plasma_val_json"]) as f:
        plasma_val_data = json.load(f)
    return {item["name"]: item["mri"] for item in plasma_val_data if item.get("mri")}


def load_pasta_visual_entry(output_dir, sid, subject_lookup=None):
    syn_path, gt_path = find_pasta_pair(output_dir, sid, subject_lookup)
    return {
        "pred": np.clip(nib.load(str(syn_path)).get_fdata().astype(np.float32), 0, 1),
        "gt": np.clip(nib.load(str(gt_path)).get_fdata().astype(np.float32), 0, 1),
    }


def load_plasma_visual_entry(nifti_dir, sid):
    pred_p = require_exists(nifti_dir / f"{sid}_tau_pred.nii.gz", f"prediction nifti for {sid}")
    gt_p = require_exists(nifti_dir / f"{sid}_tau_gt.nii.gz", f"gt nifti for {sid}")
    entry = {
        "pred": np.clip(nib.load(str(pred_p)).get_fdata().astype(np.float32), 0, 1),
        "gt": np.clip(nib.load(str(gt_p)).get_fdata().astype(np.float32), 0, 1),
    }
    mri_p = nifti_dir / f"{sid}_mri.nii.gz"
    if mri_p.exists():
        entry["mri"] = nib.load(str(mri_p)).get_fdata().astype(np.float32)
    return entry


def load_ficd_visual_entry(pred_dir, sid, ficd_preprocess):
    pred_p = require_exists(pred_dir / f"{sid}.nii.gz", f"FiCD prediction nifti for {sid}")
    raw = nib.load(str(pred_p)).get_fdata().astype(np.float32)
    target_shape = tuple(ficd_preprocess["target_shape"])
    if raw.shape != target_shape:
        raise ValueError(f"[FiCD] Expected prediction shape {target_shape} for {sid}, got {raw.shape}")

    raw_min = float(np.nanmin(raw))
    raw_max = float(np.nanmax(raw))
    if raw_min < -1e-3 or raw_max > 1.0 + 1e-3:
        raise ValueError(
            f"[FiCD] Prediction {sid} is not in [0, 1]: min={raw_min:.6f}, max={raw_max:.6f}. "
            "FiCD eval saves pred_unit, so run_comparison.py should not apply (x + 1) / 2."
        )
    return {
        "pred": np.clip(raw, 0, 1),
        "raw_shape": raw.shape,
        "raw_range": (raw_min, raw_max),
    }


def load_method_niftis(config, subjects, ficd_preprocess=None):
    method_niftis = {}
    subject_lookup = load_subject_lookup(config)
    for method_id in get_enabled_methods(config):
        label = get_method_label(method_id)
        source = get_method_source(config, method_id)
        method_niftis[label] = {}
        if method_id == "pasta":
            output_dir = require_exists(source["output_dir"], "PASTA output dir")
            for sid in subjects:
                method_niftis[label][sid] = load_pasta_visual_entry(output_dir, sid, subject_lookup)
        elif method_id in {"legacy", "plasma"}:
            nifti_dir = require_exists(Path(source["output_dir"]) / "nifti", f"{method_id} nifti dir")
            for sid in subjects:
                method_niftis[label][sid] = load_plasma_visual_entry(nifti_dir, sid)
        elif method_id == "ficd":
            if ficd_preprocess is None:
                ficd_preprocess = load_ficd_preprocess_spec(config, source)
            pred_dir = resolve_ficd_prediction_dir(source["run_dir"])
            for sid in subjects:
                method_niftis[label][sid] = load_ficd_visual_entry(pred_dir, sid, ficd_preprocess)

    print("\n  [visual audit] native loaded shapes:")
    for label, subject_dict in method_niftis.items():
        for sid, vols in subject_dict.items():
            pred = vols.get("pred")
            gt = vols.get("gt")
            mri = vols.get("mri")
            msg = f"    {label} | {sid} | pred={None if pred is None else pred.shape}"
            if gt is not None:
                msg += f" gt={gt.shape}"
            if mri is not None:
                msg += f" mri={mri.shape}"
            if label == "FiCD":
                raw_min, raw_max = vols["raw_range"]
                msg += f" raw_range=[{raw_min:.5f}, {raw_max:.5f}]"
            print(msg)
    return method_niftis


def generate_unified_comparison(
    viz_subjects,
    method_niftis,
    out_path,
    method_ids,
    raw_mri_paths=None,
    ficd_preprocess=None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy.ndimage import zoom as nd_zoom

    directions = ["Axial", "Coronal", "Sagittal"]
    method_labels = [get_method_label(m) for m in method_ids]
    col_labels = ["MRI", "GT PET"] + method_labels
    n_data_cols = len(col_labels)
    n_total_rows = 6
    diff_vmax = 0.3

    def _get_slice(vol, direction):
        x, y, z = vol.shape[:3]
        if direction == "Axial":
            return vol[:, :, z // 2]
        if direction == "Coronal":
            return vol[:, y // 2, :]
        return vol[x // 2, :, :]

    def _resample(arr, tgt_shape):
        if arr.shape == tgt_shape:
            return arr
        factors = [t / s for t, s in zip(tgt_shape, arr.shape)]
        return np.clip(nd_zoom(arr.astype(np.float64), factors, order=1).astype(np.float32), 0, 1)

    def _crop_or_pad(data, target_shape):
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

    def _crop_foreground_then_ref_space(arr, tgt_shape, raw_mri):
        if raw_mri is not None:
            nonzero = np.nonzero(raw_mri)
            if len(nonzero[0]) > 0:
                bbox_min = [int(n.min()) for n in nonzero]
                bbox_max = [int(n.max()) + 1 for n in nonzero]
                arr = arr[
                    bbox_min[0]:bbox_max[0],
                    bbox_min[1]:bbox_max[1],
                    bbox_min[2]:bbox_max[2],
                ]
        return np.clip(_crop_or_pad(arr, tgt_shape), 0, 1)

    def _ficd_to_ref_space(arr, tgt_shape, raw_mri, preprocess, sid):
        if preprocess is None:
            raise ValueError("FiCD preprocessing metadata is required for unified comparison.")
        if raw_mri is None:
            raise ValueError(f"[FiCD] Raw MRI is required to align {sid} into reference display space.")

        crop = tuple(preprocess["crop"])
        target_shape = tuple(preprocess["target_shape"])
        if arr.shape != target_shape:
            raise ValueError(f"[FiCD] Expected prediction shape {target_shape} for {sid}, got {arr.shape}")

        raw_shape = tuple(int(x) for x in raw_mri.shape[:3])
        cropped_shape = (
            raw_shape[0] - crop[0] - crop[1],
            raw_shape[1] - crop[2] - crop[3],
            raw_shape[2] - crop[4] - crop[5],
        )
        if any(dim <= 0 for dim in cropped_shape):
            raise ValueError(f"[FiCD] Invalid crop {crop} for raw MRI shape {raw_shape} ({sid})")

        crop_space = _resample(arr, cropped_shape)
        mni_space = np.pad(
            crop_space,
            [(crop[0], crop[1]), (crop[2], crop[3]), (crop[4], crop[5])],
            mode="constant",
            constant_values=0,
        )
        if mni_space.shape != raw_shape:
            raise ValueError(
                f"[FiCD] Uncropped shape mismatch for {sid}: got {mni_space.shape}, expected {raw_shape}"
            )

        aligned = _crop_foreground_then_ref_space(mni_space, tgt_shape, raw_mri)
        print(
            f"  [FiCD align] {sid}: native={arr.shape} crop_space={crop_space.shape} "
            f"mni={mni_space.shape} ref={aligned.shape}"
        )
        return aligned

    def _pasta_to_mni(arr):
        padded = np.pad(arr, [(12, 13), (16, 17), (12, 13)], mode="constant", constant_values=0)
        return nd_zoom(padded.astype(np.float64), 1.5, order=1).astype(np.float32)

    def _pasta_to_ref_space(arr, tgt_shape, raw_mri=None):
        mni = _pasta_to_mni(arr)
        return _crop_foreground_then_ref_space(mni, tgt_shape, raw_mri)

    def _blank_ax(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    if not viz_subjects:
        print("  [unified] No viz subjects, skipping.")
        return

    selected = [viz_subjects[i] for i in sorted(set([0, len(viz_subjects) // 2, len(viz_subjects) - 1]))]

    native_gt_priority = [label for label in ["Plasma (Ours)", "Legacy"] if label in method_labels]
    if not native_gt_priority:
        native_gt_priority = method_labels[:1]

    for sid in selected:
        gt_vol = None
        for label in native_gt_priority:
            entry = method_niftis.get(label, {}).get(sid, {})
            if "gt" in entry:
                gt_vol = entry["gt"]
                break
        gt_shape = gt_vol.shape if gt_vol is not None else None
        if gt_shape is not None and gt_shape != REFERENCE_VIS_SHAPE:
            raise ValueError(
                f"[unified] Expected reference GT shape {REFERENCE_VIS_SHAPE} for {sid}, got {gt_shape}"
            )

        mri_vol = None
        for label in native_gt_priority:
            entry = method_niftis.get(label, {}).get(sid, {})
            if "mri" in entry:
                mri_vol = entry["mri"]
                break

        raw_mri = None
        if raw_mri_paths and sid in raw_mri_paths and os.path.exists(raw_mri_paths[sid]):
            raw_mri = nib.load(raw_mri_paths[sid]).get_fdata().astype(np.float32)
            if raw_mri.ndim == 4:
                raw_mri = raw_mri.mean(axis=-1)

        pred_vols = {}
        diff_gt_vols = {}
        for method_id in method_ids:
            label = get_method_label(method_id)
            entry = method_niftis.get(label, {}).get(sid, {})
            pred = entry.get("pred")
            if pred is not None and gt_shape is not None:
                alignment = METHOD_ALIGNMENT[method_id]
                if alignment == "ficd":
                    pred = _ficd_to_ref_space(pred, gt_shape, raw_mri, ficd_preprocess, sid)
                elif alignment == "pasta":
                    pred = _pasta_to_ref_space(pred, gt_shape, raw_mri=raw_mri)
                else:
                    pred = _resample(pred, gt_shape)
            pred_vols[label] = pred

            if label == "PASTA" and gt_shape is not None:
                pasta_gt_native = entry.get("gt")
                diff_gt_vols[label] = _pasta_to_ref_space(pasta_gt_native, gt_shape, raw_mri=raw_mri)
            else:
                diff_gt_vols[label] = gt_vol

        width_ratios = [1] * n_data_cols + [0.07]
        fig = plt.figure(figsize=(n_data_cols * 1.9 + 0.6, n_total_rows * 1.8 + 0.6))
        gs = gridspec.GridSpec(
            n_total_rows,
            n_data_cols + 1,
            figure=fig,
            width_ratios=width_ratios,
            hspace=0.06,
            wspace=0.04,
            left=0.11,
            right=0.97,
            top=0.95,
            bottom=0.02,
        )
        axes = [[fig.add_subplot(gs[r, c]) for c in range(n_data_cols)] for r in range(n_total_rows)]
        cbar_ax = fig.add_subplot(gs[:, n_data_cols])
        last_diff_im = None

        mri_vlo = float(np.percentile(mri_vol, 1)) if mri_vol is not None else 0
        mri_vhi = float(np.percentile(mri_vol, 99)) if mri_vol is not None else 1

        for dir_idx, direction in enumerate(directions):
            row_s = dir_idx * 2
            row_e = row_s + 1

            ax = axes[row_s][0]
            if mri_vol is not None:
                ax.imshow(_get_slice(mri_vol, direction).T, cmap="gray", origin="lower", vmin=mri_vlo, vmax=mri_vhi)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, fontsize=7, color="gray")

            ax = axes[row_s][1]
            if gt_vol is not None:
                ax.imshow(_get_slice(gt_vol, direction).T, cmap="inferno", origin="lower", vmin=0, vmax=1)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, fontsize=7, color="gray")

            for idx, label in enumerate(method_labels, start=2):
                ax = axes[row_s][idx]
                pred = pred_vols[label]
                if pred is not None:
                    ax.imshow(_get_slice(pred, direction).T, cmap="inferno", origin="lower", vmin=0, vmax=1)
                else:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, fontsize=7, color="gray")

            _blank_ax(axes[row_e][0])
            _blank_ax(axes[row_e][1])
            for idx, label in enumerate(method_labels, start=2):
                ax = axes[row_e][idx]
                pred = pred_vols[label]
                ref_gt = diff_gt_vols[label]
                if pred is None or ref_gt is None:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, fontsize=7, color="gray")
                else:
                    diff = np.abs(pred - ref_gt)
                    last_diff_im = ax.imshow(_get_slice(diff, direction).T, cmap="Reds", origin="lower", vmin=0, vmax=diff_vmax)

            for row in (row_s, row_e):
                for col in range(n_data_cols):
                    axes[row][col].set_xticks([])
                    axes[row][col].set_yticks([])

            if dir_idx == 0:
                for ci, label in enumerate(col_labels):
                    axes[0][ci].set_title(label, fontsize=8, pad=3, fontweight="bold")

            axes[row_s][0].set_ylabel(f"{direction}\nSynthesis", fontsize=8, fontweight="bold", labelpad=4)
            content_h = 0.95 - 0.02
            row_height = content_h / n_total_rows
            row_center_y = 0.02 + (n_total_rows - 1 - row_e + 0.5) * row_height
            fig.text(0.025, row_center_y, "Error\nMap", va="center", ha="center", fontsize=7, color="dimgray", rotation=90)

        if last_diff_im is not None:
            cb = fig.colorbar(last_diff_im, cax=cbar_ax)
            cb.set_label("Absolute Error", fontsize=8)
            cb.ax.tick_params(labelsize=7)
        else:
            cbar_ax.axis("off")

        fig.suptitle(f"Method Comparison  |  Subject: {sid}", fontsize=11, fontweight="bold", y=0.98)
        fig_path = out_path / f"unified_comparison_{sid}.png"
        plt.savefig(str(fig_path), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fig_path}")


def generate_boxplot(all_metrics_df, out_path, method_labels):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_cols = ["ssim", "psnr", "mae"]
    colors = {
        "PASTA": "#1f77b4",
        "Legacy": "#ff7f0e",
        "Plasma (Ours)": "#2ca02c",
        "FiCD": "#d62728",
    }

    fig, axes = plt.subplots(1, len(metric_cols), figsize=(5 * len(metric_cols), 5))
    for idx, metric in enumerate(metric_cols):
        ax = axes[idx]
        data_to_plot, labels, cs = [], [], []
        for label in method_labels:
            sub = all_metrics_df[all_metrics_df["method"] == label][metric].dropna()
            if len(sub) == 0:
                continue
            data_to_plot.append(sub.values)
            labels.append(label)
            cs.append(colors.get(label, "gray"))

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


# ── Report ───────────────────────────────────────────────────────────────────
def generate_report(summary_df, stat_tests, method_counts, out_path, enabled_methods):
    enabled_labels = [get_method_label(m) for m in enabled_methods]
    lines = [
        "# MRI → TAU PET 多方法对比实验报告",
        "",
        "## 实验概述",
        "",
        "| 项目 | 说明 |",
        "|------|------|",
        f"| 对比方法 | {', '.join(enabled_labels)} |",
        f"| 公共测试集 | {method_counts.get('common', 'N/A')} subjects |",
        "| 评估指标 | SSIM, PSNR, MAE, MSE, NCC |",
        "| 统计检验 | Wilcoxon signed-rank test |",
        "",
        "### 方法简述",
        "",
        "| 方法 | 模型类型 | 输出分辨率 | 条件注入 |",
        "|------|----------|-----------|---------|",
    ]
    for method_id in enabled_methods:
        spec = METHOD_SPECS[method_id]
        summary = spec["summary"]
        lines.append(
            f"| {spec['label']} | {summary['model_type']} | "
            f"{summary['resolution']} | {summary['conditioning']} |"
        )
    lines.extend(
        [
            "",
            "## 定量结果",
            "",
            "### 整体指标 (Mean ± Std)",
            "",
            "| 方法 | N | SSIM ↑ | PSNR ↑ | MAE ↓ | MSE ↓ | NCC ↑ |",
            "|------|---|--------|--------|-------|-------|-------|",
        ]
    )

    for label in enabled_labels:
        row = summary_df[summary_df["method"] == label]
        if len(row) == 0:
            lines.append(f"| {label} | 0 | - | - | - | - | - |")
            continue
        record = row.iloc[0]
        ncc_str = f"{record['ncc_mean']:.4f}±{record['ncc_std']:.4f}" if not np.isnan(record.get("ncc_mean", np.nan)) else "N/A"
        lines.append(
            f"| {label} | {int(record['n'])} | "
            f"{record['ssim_mean']:.4f}±{record['ssim_std']:.4f} | "
            f"{record['psnr_mean']:.2f}±{record['psnr_std']:.2f} | "
            f"{record['mae_mean']:.4f}±{record['mae_std']:.4f} | "
            f"{record['mse_mean']:.6f}±{record['mse_std']:.6f} | "
            f"{ncc_str} |"
        )

    lines.extend(["", "### 最优方法", ""])
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
    for metric, method in best.items():
        direction = "↑" if metric in {"ssim", "psnr", "ncc"} else "↓"
        lines.append(f"- **{metric.upper()}** {direction}: **{method}**")

    lines.extend(
        [
            "",
            "## 统计检验 (Wilcoxon signed-rank test)",
            "",
            "| 指标 | 对比 | N | Statistic | p-value | 显著性 |",
            "|------|------|---|-----------|---------|--------|",
        ]
    )
    for test in stat_tests:
        p_str = f"{test['p_value']:.2e}" if not np.isnan(test["p_value"]) else "N/A"
        s_str = f"{test['statistic']:.1f}" if not np.isnan(test["statistic"]) else "N/A"
        lines.append(f"| {test['metric']} | {test['pair']} | {test['n']} | {s_str} | {p_str} | {test['sig']} |")

    lines.extend(
        [
            "",
            "## 注意事项",
            "",
            "1. PASTA 在 96×112×96 (1.5mm) 分辨率下评估，其余方法在约 160³ (1mm) 或相近空间下评估。",
            "2. FiCD 的 NCC 仍未计算，因为当前保存结果缺少可直接对齐的 GT NIfTI。",
            "3. 若启用 `rerun_inference`，所有新生成结果统一保存在当前 `OUT_DIR` 下。",
            "",
        ]
    )

    report_path = out_path / "comparison_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Report saved to {report_path}")
    return report_path


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    config = CONFIG
    validate_config(config)

    config["out_dir"].mkdir(parents=True, exist_ok=True)
    outputs = output_paths(config)
    outputs["figures_dir"].mkdir(parents=True, exist_ok=True)

    subject_records = load_subject_records(config)
    subjects = [record["name"] for record in subject_records]
    enabled_methods = get_enabled_methods(config)
    enabled_labels = get_enabled_labels(config)
    ficd_preprocess = validate_ficd_subject_records(config, subjects)
    print(f"Run mode: {config['run_mode']}")
    print(f"Enabled methods: {enabled_methods}")
    print(f"Common subjects: {len(subjects)}")

    maybe_run_inference(config)

    print("\n" + "=" * 60)
    print("  PHASE 2: COMPUTING UNIFIED METRICS")
    print("=" * 60)
    if config["run_mode"] == "use_existing_results" and outputs["per_subject_csv"].exists():
        print(f"\n  [跳过计算] 检测到历史 metrics 文件，直接加载: {outputs['per_subject_csv']}")
        df = pd.read_csv(outputs["per_subject_csv"])
        df = df[df["method"].isin(enabled_labels)].copy()
    else:
        df = compute_metrics_dataframe(config, subjects)
        df.to_csv(outputs["per_subject_csv"], index=False)

    print(f"\nTotal metric rows: {len(df)}")
    print(df.groupby("method")[["ssim", "psnr", "mae"]].describe().round(4))

    if config["run_mode"] == "use_existing_results" and outputs["summary_csv"].exists() and outputs["per_subject_csv"].exists():
        print(f"\n  [跳过汇总] 使用历史 summary: {outputs['summary_csv']}")
        summary_df = pd.read_csv(outputs["summary_csv"])
        summary_df = summary_df[summary_df["method"].isin(enabled_labels)].copy()
    else:
        summary_df = build_summary(df, enabled_labels)
        summary_df.to_csv(outputs["summary_csv"], index=False)
    print("\nSummary:")
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("  PHASE 3: STATISTICAL TESTS")
    print("=" * 60)
    stat_tests = run_statistical_tests(df, enabled_labels)
    pd.DataFrame(stat_tests).to_csv(outputs["stats_csv"], index=False)

    print("\n" + "=" * 60)
    print("  PHASE 4: GENERATING VISUALIZATIONS")
    print("=" * 60)
    viz_subjects = subjects[: config["max_viz_subjects"]]
    method_niftis = load_method_niftis(config, viz_subjects, ficd_preprocess=ficd_preprocess)

    for label, subject_dict in method_niftis.items():
        for sid, vols in subject_dict.items():
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            has_gt = "gt" in vols
            n_rows = 2 if has_gt else 1
            fig, axes = plt.subplots(n_rows, 3, figsize=(12, 3.5 * n_rows))
            if n_rows == 1:
                axes = axes[np.newaxis, :]

            pred_slices = _get_mid_slices(vols["pred"])
            row = n_rows - 1
            for j, slice_2d in enumerate(pred_slices):
                axes[row, j].imshow(slice_2d.T, cmap="inferno", origin="lower", vmin=0, vmax=1)
            axes[row, 0].set_ylabel(f"{label}\nPred", fontsize=10)

            if has_gt:
                gt_slices = _get_mid_slices(vols["gt"])
                for j, slice_2d in enumerate(gt_slices):
                    axes[0, j].imshow(slice_2d.T, cmap="inferno", origin="lower", vmin=0, vmax=1)
                axes[0, 0].set_ylabel("GT", fontsize=10)

            for ax in axes.flat:
                ax.set_xticks([])
                ax.set_yticks([])
            axes[0, 0].set_title("Axial")
            axes[0, 1].set_title("Coronal")
            axes[0, 2].set_title("Sagittal")
            fig.suptitle(f"{label} | {sid}", fontsize=12)
            plt.tight_layout()
            safe_name = label.replace(" ", "_").replace("(", "").replace(")", "")
            plt.savefig(str(outputs["figures_dir"] / f"{safe_name}_{sid}.png"), dpi=150, bbox_inches="tight")
            plt.close()

    raw_mri_paths = load_raw_mri_paths(config)
    generate_unified_comparison(
        viz_subjects,
        method_niftis,
        outputs["figures_dir"],
        enabled_methods,
        raw_mri_paths=raw_mri_paths,
        ficd_preprocess=ficd_preprocess,
    )
    generate_boxplot(df, outputs["figures_dir"], enabled_labels)

    print("\n" + "=" * 60)
    print("  PHASE 5: GENERATING REPORT")
    print("=" * 60)
    report_path = generate_report(summary_df, stat_tests, {"common": len(subjects)}, config["out_dir"], enabled_methods)

    print("\n" + "=" * 60)
    print("  ALL DONE!")
    print("=" * 60)
    print(f"\nOutput directory: {config['out_dir']}")
    print(f"Report: {report_path}")
    print(f"Per-subject metrics: {outputs['per_subject_csv']}")
    print(f"Summary: {outputs['summary_csv']}")
    print(f"Statistical tests: {outputs['stats_csv']}")
    print(f"Figures: {outputs['figures_dir']}")


if __name__ == "__main__":
    main()
