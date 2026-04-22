#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MRI-only preprocessing for MISSING ADNI data (ADNI0103 structure).
Input CSV must contain: subject_id, id_mri.

ADNI0103 NIfTI Structure:
  root / subject_id / SeriesDesc / Date / ImageID / file.nii.gz

This script adapts the search logic to find raw NIfTIs in this structure.

Pipeline:
1) fslreorient2std
2) fslorient -copysform2qform
3) mri_synthstrip (brain extraction) -> brain image
4) fslmaths brain -bin -> native brain mask
5) N4BiasFieldCorrection (SimpleITK) on *reoriented full-head* (mri_rstd) - faster than FAST
6) FLIRT rigid 6 DOF: (mri_n4_restore) -> MNI152_T1_1mm (full-head)
7) Apply transform to native mask (nearest) -> mask in MNI
8) Final brain-only MRI in MNI

"""

from __future__ import annotations
import argparse
import csv
import os
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple, List, Set

import pandas as pd
import SimpleITK as sitk

# -------------------------
# Utils
# -------------------------
def which_or_die(exe: str) -> str:
    p = shutil.which(exe)
    if not p:
        raise RuntimeError(f"[FATAL] Missing executable in PATH: {exe}")
    return p


def run(cmd, logf: Path, check=True, env=None):
    logf.parent.mkdir(parents=True, exist_ok=True)
    with open(logf, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 120 + "\n")
        f.write("CMD: " + " ".join(map(str, cmd)) + "\n")
        f.write("=" * 120 + "\n")
        f.flush()
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        f.write(p.stdout or "")
        f.write("\n")
    if check and p.returncode != 0:
        tail = "\n".join((p.stdout or "").splitlines()[-120:])
        raise RuntimeError(f"Command failed (code={p.returncode}): {' '.join(map(str, cmd))}\nTAIL:\n{tail}")
    return p


def n4_bias_correction(
    input_path: Path,
    output_path: Path,
    mask_path: Optional[Path] = None,
    shrink_factor: int = 4,
    num_iterations: Tuple[int, ...] = (50, 50, 30, 20),
    convergence_threshold: float = 1e-6,
    num_threads: int = 1,
) -> None:
    """
    Apply N4 bias field correction using SimpleITK.
    Faster than FSL FAST with comparable results.
    
    Args:
        input_path: Input NIfTI file
        output_path: Output bias-corrected NIfTI file
        mask_path: Optional brain mask to focus correction
        shrink_factor: Downsampling factor for speed (default 4)
        num_iterations: Iterations per level (default: 50,50,30,20)
        convergence_threshold: Stopping criterion
        num_threads: Number of threads for ITK
    """
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(num_threads)
    
    img = sitk.ReadImage(str(input_path), sitk.sitkFloat32)
    
    # Optional mask
    mask = None
    if mask_path and mask_path.exists():
        mask = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
        # Ensure mask is binary
        mask = sitk.BinaryThreshold(mask, lowerThreshold=0.5, upperThreshold=1e10, insideValue=1, outsideValue=0)
    else:
        # Create a simple Otsu threshold mask for robustness
        mask = sitk.OtsuThreshold(img, 0, 1, 200)
    
    # Shrink for speed
    img_shrunk = sitk.Shrink(img, [shrink_factor] * img.GetDimension())
    mask_shrunk = sitk.Shrink(mask, [shrink_factor] * mask.GetDimension())
    
    # N4 corrector
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(list(num_iterations))
    corrector.SetConvergenceThreshold(convergence_threshold)
    
    # Run on shrunk image to get bias field
    _ = corrector.Execute(img_shrunk, mask_shrunk)
    
    # Get log bias field and apply to full-res image
    log_bias_field = corrector.GetLogBiasFieldAsImage(img)
    corrected = img / sitk.Exp(log_bias_field)
    
    # Write output
    sitk.WriteImage(corrected, str(output_path))


def norm_img_id(x) -> str:
    s = str(x).strip().strip('"').strip("'")
    if not s or s.lower() in {"nan", "none", "null"}:
        return ""
    if s.startswith("I"):
        return s
    if s.isdigit():
        return "I" + s
    m = re.search(r"(\d+)", s)
    return "I" + m.group(1) if m else ""


def find_nifti_struct(root: Path, subject: str, img_id: str) -> Optional[Path]:
    """
    Find NIfTI in ADNI0103 structure:
    root / <subject> / ... / <img_id> / *.nii.gz
    """
    subj_dir = root / subject
    if not subj_dir.exists():
        return None
    
    # img_id is typically "Ixxxx"
    # Search for directory named exactly img_id
    # Using rglob to find the directory
    # Note: This might be slow if file system is huge, but subject folders aren't too deep
    candidates = list(subj_dir.rglob(img_id))
    
    for d in candidates:
        if d.is_dir() and d.name == img_id:
            # Found the ImageID folder, look for nii inside
            niis = list(d.glob("*.nii.gz")) + list(d.glob("*.nii"))
            if niis:
                return niis[0]
            # Some ADNI downloads allow JSON sidecars but no nii if conversion failed?
            # Or nested? Usually it's directly inside.
    return None


def ensure_csv_headers(done_csv: Path, fail_csv: Path):
    done_csv.parent.mkdir(parents=True, exist_ok=True)
    if not done_csv.exists():
        with open(done_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "subject_id", "id_mri", "raw_path",
                "final_mri", "final_mask", "final_mat",
                "status"
            ])
    if not fail_csv.exists():
        with open(fail_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "subject_id", "id_mri", "stage", "error"
            ])

def load_done_set(done_csv: Path) -> Set[Tuple[str, str]]:
    s = set()
    if done_csv.exists():
        try:
            with open(done_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sub = row.get("subject_id", "").strip()
                    mri = row.get("id_mri", "").strip()
                    if sub and mri:
                        s.add((sub, mri))
        except:
            pass
    return s


# -------------------------
# Single MRI job
# -------------------------
def process_one_mri(
    subject: str,
    mri_id: str,
    raw_path: str,
    out_mri: str,
    out_mask: str,
    out_xfm: str,
    out_rstd: str,
    logs_root: str,
    tmp_root: str,
    mni_ref_full: str,
    overwrite: bool,
    synthstrip_no_csf: bool = False,
    fill_mask_holes: bool = False,
    synthstrip_use_gpu: bool = False,
    synthstrip_cuda_device: Optional[str] = None,
) -> Tuple[str, str, str, str, str, str]:

    raw = Path(raw_path)
    out_mri = Path(out_mri); out_mri.mkdir(parents=True, exist_ok=True)
    out_mask = Path(out_mask); out_mask.mkdir(parents=True, exist_ok=True)
    out_xfm = Path(out_xfm); out_xfm.mkdir(parents=True, exist_ok=True)
    out_rstd = Path(out_rstd); out_rstd.mkdir(parents=True, exist_ok=True)

    logs_root = Path(logs_root); logs_root.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tmp_root); tmp_root.mkdir(parents=True, exist_ok=True)

    mni_ref_full = Path(mni_ref_full)

    final_mri  = out_mri  / f"{subject}__{mri_id}.nii.gz"
    final_mask = out_mask / f"{subject}__{mri_id}_mask.nii.gz"
    final_mat  = out_xfm  / f"{subject}__{mri_id}_mri2mni.mat"
    final_rstd = out_rstd / f"{subject}__{mri_id}_rstd.nii.gz"

    # Skip if already done and not overwrite
    if (not overwrite) and final_mri.exists() and final_mask.exists() and final_mat.exists() and final_rstd.exists():
        return subject, mri_id, str(raw), str(final_mri), str(final_mask), str(final_mat)

    work = tmp_root / subject / f"mri_{mri_id}"
    work.mkdir(parents=True, exist_ok=True)
    logf = logs_root / "per_mri" / subject / f"{mri_id}.log"

    mri_rstd = work / "mri_rstd.nii.gz"
    mri_brain = work / "mri_brain.nii.gz"
    mri_mask = work / "mri_mask.nii.gz"
    mri_mask_raw = work / "mri_mask_raw.nii.gz"

    mri_n4_restore = work / "mri_n4_restore.nii.gz"

    mri_in_mni = work / "mri_in_mni.nii.gz"
    mri2mni_mat = work / "mri_to_mni.mat"
    mask_in_mni = work / "mri_mask_in_mni.nii.gz"
    brainonly = work / "mri_in_mni_brainonly.nii.gz"

    stage = "START"
    try:
        # 1) Reorient
        stage = "REORIENT"
        run(["fslreorient2std", str(raw), str(mri_rstd)], logf)
        run(["fslorient", "-copysform2qform", str(mri_rstd)], logf)

        if overwrite and final_rstd.exists():
            final_rstd.unlink()
        shutil.copyfile(mri_rstd, final_rstd)

        # 2) Skull strip + mask
        stage = "SYNTHSTRIP"
        # Build env with CUDA_VISIBLE_DEVICES for synthstrip subprocess
        synth_env = os.environ.copy()
        # Ensure GPU ordering matches nvidia-smi output
        synth_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if synthstrip_cuda_device is not None:
            synth_env["CUDA_VISIBLE_DEVICES"] = str(synthstrip_cuda_device)
        cmd = ["mri_synthstrip", "-i", str(mri_rstd), "-o", str(mri_brain), "-m", str(mri_mask_raw)]
        if synthstrip_use_gpu:
            cmd.append("-g")
        if synthstrip_no_csf:
            cmd.append("--no-csf")
        p = run(cmd, logf, check=False, env=synth_env)
        if p.returncode != 0 or (not mri_mask_raw.exists()):
            # Backward-compatible fallback for older synthstrip builds without -m
            run(["mri_synthstrip", "-i", str(mri_rstd), "-o", str(mri_brain)] + (["-g"] if synthstrip_use_gpu else []) + (["--no-csf"] if synthstrip_no_csf else []), logf, env=synth_env)
            stage = "MASK_NATIVE_FALLBACK"
            run(["fslmaths", str(mri_brain), "-bin", str(mri_mask)], logf)
        else:
            stage = "MASK_NATIVE"
            # Use the model-produced mask directly; ensure it is binary, optionally fill internal holes.
            if fill_mask_holes:
                run(["fslmaths", str(mri_mask_raw), "-bin", "-fillh", str(mri_mask)], logf)
            else:
                run(["fslmaths", str(mri_mask_raw), "-bin", str(mri_mask)], logf)

        # 3) N4 bias field correction on full-head rstd (faster than FAST)
        stage = "N4_BIAS"
        n4_bias_correction(
            input_path=mri_rstd,
            output_path=mri_n4_restore,
            mask_path=mri_mask,  # Use brain mask to focus correction
            shrink_factor=4,  # Downsample for speed
            num_iterations=(50, 50, 30, 20),  # Multi-resolution iterations
            num_threads=2,  # Use 2 threads per job
        )

        if not mri_n4_restore.exists():
            raise RuntimeError("N4BiasFieldCorrection did not create mri_n4_restore.nii.gz")

        # 4) FLIRT rigid 6DOF to MNI (full-head)
        stage = "FLIRT_6DOF"
        run([
            "flirt",
            "-in", str(mri_n4_restore),
            "-ref", str(mni_ref_full),
            "-out", str(mri_in_mni),
            "-omat", str(mri2mni_mat),
            "-dof", "6",
            "-cost", "normmi",
            "-searchrx", "-30", "30",
            "-searchry", "-30", "30",
            "-searchrz", "-30", "30",
        ], logf)

        # 5) Mask -> MNI
        stage = "MASK_MNI"
        run([
            "flirt",
            "-in", str(mri_mask),
            "-ref", str(mni_ref_full),
            "-out", str(mask_in_mni),
            "-applyxfm", "-init", str(mri2mni_mat),
            "-interp", "nearestneighbour",
        ], logf)

        # 6) Final brain-only
        stage = "FINAL_MASK_APPLY"
        run(["fslmaths", str(mri_in_mni), "-mas", str(mask_in_mni), str(brainonly)], logf)

        # Write finals
        stage = "WRITE_OUTPUTS"
        if overwrite:
            for p in (final_mri, final_mask, final_mat):
                if p.exists():
                    p.unlink()

        shutil.copyfile(brainonly, final_mri)
        shutil.copyfile(mask_in_mni, final_mask)
        shutil.copyfile(mri2mni_mat, final_mat)

        return subject, mri_id, str(raw), str(final_mri), str(final_mask), str(final_mat)

    except Exception as e:
        raise RuntimeError(f"{stage}::{str(e).splitlines()[0]}") from e


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", type=Path, required=True)
    # nifti_root: The root folder containing Subject folders (ADNI0103/NIFTI)
    ap.add_argument("--nifti_root", type=Path, required=True)

    ap.add_argument("--out_root", type=Path, default=Path("/mnt/nfsdata/nfsdata/ADNI/ADNI0103"))
    ap.add_argument("--logs_root", type=Path, default=Path("/mnt/nfsdata/nfsdata/ADNI/ADNI0103/logs_mri"))
    ap.add_argument("--tmp_root", type=Path, default=Path("/mnt/nfsdata/nfsdata/ADNI/ADNI0103/tmp_mri"))

    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=-1)

    ap.add_argument(
        "--synthstrip_no_csf",
        action="store_true",
        help="Pass --no-csf to mri_synthstrip (can create ventricular/CSF holes in mask).",
    )
    ap.add_argument(
        "--no_fill_mask_holes",
        action="store_true",
        help="Disable fslmaths -fillh on the native brain mask.",
    )
    ap.add_argument(
        "--synthstrip_gpu",
        action="store_true",
        help="Use GPU for mri_synthstrip (-g).",
    )

    args = ap.parse_args()

    # Derived paths
    out_mri = args.out_root / "MRI"
    out_mask = args.out_root / "MRI_MASK"
    out_xfm = args.out_root / "MRI_XFM"
    out_rstd = args.out_root / "MRI_NATIVE_RSTD"

    # Avoid oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")

    # Tools check (fast removed - now using SimpleITK N4BiasFieldCorrection)
    for exe in ["fslreorient2std", "fslorient", "mri_synthstrip", "fslmaths", "flirt"]:
        which_or_die(exe)

    fsldir = os.environ.get("FSLDIR", "")
    if not fsldir:
        raise RuntimeError("[FATAL] FSLDIR not set. Source FSL first.")
    mni_ref_full = Path(fsldir) / "data/standard/MNI152_T1_1mm.nii.gz"
    if not mni_ref_full.exists():
        raise RuntimeError(f"[FATAL] Missing MNI ref: {mni_ref_full}")

    df = pd.read_csv(args.pairs_csv)
    if not {"subject_id", "id_mri"}.issubset(df.columns):
        raise RuntimeError(f"[FATAL] CSV must contain subject_id, id_mri.")

    if args.limit > 0:
        df = df.head(args.limit).copy()

    args.logs_root.mkdir(parents=True, exist_ok=True)
    args.tmp_root.mkdir(parents=True, exist_ok=True)
    args.out_root.mkdir(parents=True, exist_ok=True)
    out_mri.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)
    out_xfm.mkdir(parents=True, exist_ok=True)
    out_rstd.mkdir(parents=True, exist_ok=True)

    done_csv = args.logs_root / "done.csv"
    fail_csv = args.logs_root / "fail.csv"
    ensure_csv_headers(done_csv, fail_csv)

    done_set = load_done_set(done_csv)

    jobs: List[Tuple[str, str, str]] = []
    seen: Set[Tuple[str, str]] = set()

    for _, row in df.iterrows():
        subject = str(row["subject_id"]).strip()
        mri_id = norm_img_id(row["id_mri"])
        if not subject or not mri_id:
            continue
        key = (subject, mri_id)
        if key in seen:
            continue
        seen.add(key)

        # Check existing final outputs (if overwrite=False)
        final_mri  = out_mri  / f"{subject}__{mri_id}.nii.gz"
        final_mask = out_mask / f"{subject}__{mri_id}_mask.nii.gz"
        final_mat  = out_xfm  / f"{subject}__{mri_id}_mri2mni.mat"
        final_rstd = out_rstd / f"{subject}__{mri_id}_rstd.nii.gz"
        
        all_exist = final_mri.exists() and final_mask.exists() and final_mat.exists() and final_rstd.exists()

        if (not args.overwrite) and key in done_set and all_exist:
            continue
        
        if (not args.overwrite) and all_exist:
             # Add to done set
            with open(done_csv, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([subject, mri_id, "EXISTING", str(final_mri), str(final_mask), str(final_mat), "OK_EXISTING"])
            done_set.add(key)
            continue

        # Find RAW
        # Use structured search
        raw = find_nifti_struct(args.nifti_root, subject, mri_id)
        if not raw:
            with open(fail_csv, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([subject, mri_id, "FIND_RAW", f"Missing raw NIfTI in {args.nifti_root}/{subject}/**/{mri_id}"])
            continue
        
        jobs.append((subject, mri_id, str(raw)))

    total = len(jobs)
    print(f"[INFO] Unique MRI rows:      {len(seen)}")
    print(f"[INFO] To process now:       {total}")
    print(f"[INFO] Output root:          {args.out_root}")
    
    if total == 0:
        print("[DONE] Nothing to process.")
        return

    n_done = 0
    n_fail = 0

    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {
            ex.submit(
                process_one_mri,
                subject, mri_id, raw_path,
                str(out_mri), str(out_mask), str(out_xfm), str(out_rstd),
                str(args.logs_root), str(args.tmp_root),
                str(mni_ref_full),
                args.overwrite,
                args.synthstrip_no_csf,
                (not args.no_fill_mask_holes),
                args.synthstrip_gpu,
            ): (subject, mri_id, raw_path)
            for subject, mri_id, raw_path in jobs
        }

        for fut in as_completed(futs):
            subject, mri_id, raw_path = futs[fut]
            try:
                res = fut.result()
                # res is tuple
                subject, mri_id, _, final_mri, final_mask, final_mat = res
                
                with open(done_csv, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([subject, mri_id, raw_path, final_mri, final_mask, final_mat, "OK"])
                
                n_done += 1
                if n_done % 10 == 0:
                    print(f"[PROGRESS] {n_done}/{total} done ...")
            
            except Exception as e:
                msg = str(e)
                stage = "PIPELINE"
                if "::" in msg:
                    stage, msg = msg.split("::", 1)
                
                with open(fail_csv, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([subject, mri_id, stage, msg])
                
                n_fail += 1
                print(f"[FAIL] {subject} {mri_id}: {stage} - {msg}")

    print(f"[DONE] Finished. Done={n_done}, Fail={n_fail}")

if __name__ == "__main__":
    main()
