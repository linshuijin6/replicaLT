#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MRI-only preprocessing for MISSING ADNI data (ADNI0103 structure).
Input CSV must contain: subject_id, id_mri.

ADNI0103 NIfTI Structure:
  root / subject_id / SeriesDesc / Date / ImageID / file.nii.gz

This script adapts the search logic to find raw NIfTIs in this structure.

Pipeline (same as standard fast script):
1) fslreorient2std
2) fslorient -copysform2qform
3) mri_synthstrip (brain extraction) -> brain image
4) fslmaths brain -bin -> native brain mask
5) FAST bias-field correction on *reoriented full-head* (mri_rstd)
6) FLIRT rigid 6 DOF: (mri_fast_restore) -> MNI152_T1_1mm (full-head)
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

# -------------------------
# Utils
# -------------------------
def which_or_die(exe: str) -> str:
    p = shutil.which(exe)
    if not p:
        raise RuntimeError(f"[FATAL] Missing executable in PATH: {exe}")
    return p


def run(cmd, logf: Path, check=True):
    logf.parent.mkdir(parents=True, exist_ok=True)
    with open(logf, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 120 + "\n")
        f.write("CMD: " + " ".join(map(str, cmd)) + "\n")
        f.write("=" * 120 + "\n")
        f.flush()
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        f.write(p.stdout or "")
        f.write("\n")
    if check and p.returncode != 0:
        tail = "\n".join((p.stdout or "").splitlines()[-120:])
        raise RuntimeError(f"Command failed (code={p.returncode}): {' '.join(map(str, cmd))}\nTAIL:\n{tail}")


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

    fast_prefix = work / "mri_fast"
    mri_fast_restore = work / "mri_fast_restore.nii.gz"

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
        run(["mri_synthstrip", "-i", str(mri_rstd), "-o", str(mri_brain), "--no-csf"], logf)
        
        # NOTE: synthstrip creates brain image. We create mask by binarizing it.
        stage = "MASK_NATIVE"
        run(["fslmaths", str(mri_brain), "-bin", str(mri_mask)], logf)

        # 3) FAST bias correction on full-head rstd
        stage = "FAST"
        # CLEANUP old fast outputs
        for p in work.glob("mri_fast*"):
            if p.name == "mri_fast_restore.nii.gz": continue # Keep if just created? no rework
            if p.is_file(): p.unlink()

        run(["fast", "-B", "-t", "1", "-o", str(fast_prefix), str(mri_rstd)], logf)

        if not mri_fast_restore.exists():
            raise RuntimeError("FAST did not create mri_fast_restore.nii.gz")

        # 4) FLIRT rigid 6DOF to MNI (full-head)
        stage = "FLIRT_6DOF"
        run([
            "flirt",
            "-in", str(mri_fast_restore),
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

    args = ap.parse_args()

    # Derived paths
    out_mri = args.out_root / "MRI"
    out_mask = args.out_root / "MRI_MASK"
    out_xfm = args.out_root / "MRI_XFM"
    out_rstd = args.out_root / "MRI_NATIVE_RSTD"

    # Avoid oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")

    # Tools check
    for exe in ["fslreorient2std", "fslorient", "mri_synthstrip", "fslmaths", "fast", "flirt"]:
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
