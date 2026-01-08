#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_mri_centric.py — MRI-centric preprocessing:
- Preprocess EACH MRI once (conform -> synthstrip -> mask -> FAST -> MRI->MNI + mask->MNI)
- For each available PET type on that MRI row (FDG / AV45 / AV1451):
    PET 4D->3D (Tmean if 4D)
    PET->MRI (rigid dof=6)
    concat to PET->MNI
    apply PET->MNI
    skull-strip PET in MNI using MRI mask in MNI

Supports 3 CSV schemas:
A) Subject, PET_ID, MRI_ID, Tracer
B) Subject, PET_ImageDataID, MRI_ImageDataID, PET_Description
C) MRI-centric: subject_id, id_mri, id_fdg, id_av45, id_av1451   (PET columns optional/None)

Notes:
- Requires FSL + FreeSurfer in PATH.
- FreeSurfer requires a license: pass --fs_license or set FS_LICENSE.
- Recommended: use --unique_names so outputs don't overwrite across visits.
"""

from __future__ import annotations
import argparse
import csv
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import pandas as pd


# -------------------------
# Utils
# -------------------------
def which_or_die(exe: str) -> str:
    p = shutil.which(exe)
    if not p:
        raise RuntimeError(
            f"[FATAL] Executable not found in PATH: {exe}\n"
            f"Fix: source FreeSurfer + FSL in the SAME shell, then retry."
        )
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
    """Normalize to ADNI-like 'Ixxxx' if possible."""
    s = str(x).strip().strip('"').strip("'")
    if not s or s.lower() in {"nan", "none", "null"}:
        return ""
    if s.startswith("I"):
        return s
    if s.isdigit():
        return "I" + s
    m = re.search(r"(\d+)", s)
    return "I" + m.group(1) if m else s


def is_none_like(x) -> bool:
    s = str(x).strip().strip('"').strip("'").lower()
    return (s == "") or (s in {"none", "nan", "null"})


def tracer_from_description(desc: str) -> str:
    """Infer tracer from PET description."""
    d = (desc or "").upper()
    if "FDG" in d:
        return "FDG"
    if "AV45" in d or "FLORBETAPIR" in d or "AMYVID" in d:
        return "AV45"
    if "AV1451" in d or "FLORTAUCIPIR" in d or "TAU" in d:
        return "AV1451"
    if "PIB" in d:
        return "PIB"
    if "FBB" in d or "FLORBETABEN" in d:
        return "FBB"
    if "NAV4694" in d or "NAV" in d:
        return "NAV4694"
    return "UNK"


def find_nifti_by_id(folder: Path, img_id: str) -> Optional[Path]:
    """Find NIfTI containing img_id in filename (nii/nii.gz)."""
    if not img_id:
        return None
    for pat in (f"{img_id}*.nii.gz", f"{img_id}*.nii"):
        hits = sorted(folder.glob(pat))
        if hits:
            return hits[0]
    for pat in (f"*{img_id}*.nii.gz", f"*{img_id}*.nii"):
        hits = sorted(folder.glob(pat))
        if hits:
            return hits[0]
    return None


def fsl_value(fsldir: str, rel: str) -> Path:
    p = Path(fsldir) / rel
    if not p.exists():
        raise RuntimeError(f"[FATAL] Missing FSL file: {p}")
    return p


def is_4d_nifti(path: Path) -> bool:
    """Lightweight check with fslinfo (no nibabel dependency)."""
    p = subprocess.run(["fslinfo", str(path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        return False
    txt = p.stdout or ""
    m = re.search(r"dim4\s+(\d+)", txt)
    if not m:
        return False
    return int(m.group(1)) > 1


def detect_csv_schema(df: pd.DataFrame) -> Dict[str, Any]:
    cols = set(df.columns)

    # Schema A
    if {"Subject", "PET_ID", "MRI_ID"}.issubset(cols):
        out = {"schema": "A", "subject": "Subject", "pet_id": "PET_ID", "mri_id": "MRI_ID"}
        if "Tracer" in cols:
            out["tracer"] = "Tracer"
        return out

    # Schema B
    if {"Subject", "PET_ImageDataID", "MRI_ImageDataID"}.issubset(cols):
        out = {"schema": "B", "subject": "Subject", "pet_id": "PET_ImageDataID", "mri_id": "MRI_ImageDataID"}
        if "PET_Description" in cols:
            out["pet_desc"] = "PET_Description"
        return out

    # Schema C (MRI-centric)
    if {"subject_id", "id_mri"}.issubset(cols):
        return {
            "schema": "C",
            "subject": "subject_id",
            "mri_id": "id_mri",
            "pet_cols": {  # tracer -> column name
                "FDG": "id_fdg" if "id_fdg" in cols else None,
                "AV45": "id_av45" if "id_av45" in cols else None,
                "AV1451": "id_av1451" if "id_av1451" in cols else None,
            },
        }

    raise RuntimeError(
        "[FATAL] pairs_csv columns not recognized.\n"
        f"Found columns: {list(df.columns)}\n"
        "Expected either:\n"
        "  A) Subject, PET_ID, MRI_ID, Tracer\n"
        "  B) Subject, PET_ImageDataID, MRI_ImageDataID, PET_Description (optional)\n"
        "  C) subject_id, id_mri, id_fdg (opt), id_av45 (opt), id_av1451 (opt)\n"
    )


# -------------------------
# MRI-only pipeline
# -------------------------
def process_mri_once(
    subject: str,
    mri_id: str,
    mri_path: Path,
    out_root: Path,
    tmp_root: Path,
    logs_root: Path,
    mni_ref: Path,
    keep_tmp: bool,
    overwrite: bool,
    unique_names: bool,
) -> Dict[str, Path]:
    """
    Runs MRI steps once and returns paths needed for PET processing.
    Outputs MRI MNI + mask MNI to out_root.
    """
    out_mri_dir = out_root / "MRI"
    out_mask_dir = out_root / "MRI_MASK"
    out_mri_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    if unique_names:
        final_mri = out_mri_dir / f"{subject}__{mri_id}.nii.gz"
        final_mask = out_mask_dir / f"{subject}__{mri_id}_mask.nii.gz"
    else:
        final_mri = out_mri_dir / f"{subject}.nii.gz"
        final_mask = out_mask_dir / f"{subject}_mask.nii.gz"

    # work + log
    work = tmp_root / subject / f"mri_{mri_id}"
    work.mkdir(parents=True, exist_ok=True)
    logf = logs_root / "per_mri" / subject / f"{mri_id}.log"

    mri_conform   = work / "mri_conformed.nii"
    mri_brain     = work / "mri_brain.nii.gz"
    mri_mask      = work / "mri_mask.nii.gz"
    mri_restore   = work / "mri_fast_restore.nii.gz"
    mri_mni       = work / "mri_in_mni.nii.gz"
    mri2mni_mat   = work / "mri_to_mni.mat"
    mri_mask_mni  = work / "mri_mask_in_mni.nii.gz"

    # If already done and not overwrite, skip heavy work
    if (not overwrite) and final_mri.exists() and final_mask.exists():
        # Still need paths for PET processing; but we can reuse final outputs and recompute matrix? (not stored)
        # We will assume we need the matrix for PET->MNI. So if missing, we recompute in tmp.
        if mri2mni_mat.exists() and (tmp_root / subject / f"mri_{mri_id}" / "mri_brain.nii.gz").exists():
            return {
                "final_mri": final_mri,
                "final_mask": final_mask,
                "mri_brain": mri_brain,
                "mri_mask": mri_mask,
                "mri2mni_mat": mri2mni_mat,
                "mri_mask_mni": mri_mask_mni,
                "work": work,
                "logf": logf,
            }

    # 1) MRI conform
    run(["mri_convert", "--conform", str(mri_path), str(mri_conform)], logf)

    # 2) MRI skull strip (SynthStrip CPU only)
    run(["mri_synthstrip", "-i", str(mri_conform), "-o", str(mri_brain), "--no-csf"], logf)

    # 3) MRI mask
    run(["fslmaths", str(mri_brain), "-bin", str(mri_mask)], logf)

    # 4) Bias correction (FAST)
    run(["fast", "-B", "-t", "1", "-o", str(work / "mri_fast"), str(mri_brain)], logf)
    if not mri_restore.exists():
        raise RuntimeError(f"[FATAL] FAST did not create {mri_restore}")

    # 5) MRI -> MNI
    run(["flirt", "-in", str(mri_restore), "-ref", str(mni_ref),
         "-out", str(mri_mni), "-omat", str(mri2mni_mat), "-dof", "12"], logf)

    # 6) MRI mask -> MNI (nearest)
    run(["flirt", "-in", str(mri_mask), "-ref", str(mni_ref),
         "-out", str(mri_mask_mni), "-applyxfm", "-init", str(mri2mni_mat),
         "-interp", "nearestneighbour"], logf)

    # Move finals
    if overwrite and final_mri.exists():
        final_mri.unlink()
    if overwrite and final_mask.exists():
        final_mask.unlink()
    shutil.copyfile(mri_mni, final_mri)
    shutil.copyfile(mri_mask_mni, final_mask)

    if not keep_tmp:
        # keep only what PET needs? If you want aggressive cleanup, you can remove work entirely,
        # but then PET processing later in same run still needs these. We keep work until end of this MRI row.
        pass

    return {
        "final_mri": final_mri,
        "final_mask": final_mask,
        "mri_brain": mri_brain,
        "mri_mask": mri_mask,
        "mri2mni_mat": mri2mni_mat,
        "mri_mask_mni": mri_mask_mni,
        "work": work,
        "logf": logf,
    }


# -------------------------
# PET given MRI
# -------------------------
def process_one_pet_given_mri(
    subject: str,
    tracer: str,
    pet_id: str,
    pet_path: Path,
    mri_brain: Path,
    mni_ref: Path,
    mri2mni_mat: Path,
    mri_mask_mni: Path,
    out_root: Path,
    work_mri: Path,
    logs_root: Path,
    overwrite: bool,
    unique_names: bool,
) -> Path:
    out_pet_dir = out_root / "PET" / tracer
    out_pet_dir.mkdir(parents=True, exist_ok=True)

    if unique_names:
        final_pet = out_pet_dir / f"{subject}__{pet_id}.nii.gz"
    else:
        final_pet = out_pet_dir / f"{subject}.nii.gz"

    if (not overwrite) and final_pet.exists():
        return final_pet

    # per-pet work
    work = work_mri / f"pet_{tracer}_{pet_id}"
    work.mkdir(parents=True, exist_ok=True)
    logf = logs_root / "per_pet" / subject / f"{tracer}__{pet_id}.log"

    pet_3d        = work / "pet_3d.nii.gz"
    pet2mri_mat   = work / "pet_to_mri.mat"
    pet2mni_mat   = work / "pet_to_mni.mat"
    pet_in_mni    = work / "pet_in_mni.nii.gz"
    pet_in_mni_br = work / "pet_in_mni_brain.nii.gz"

    # 7) PET 4D -> 3D
    if is_4d_nifti(pet_path):
        run(["fslmaths", str(pet_path), "-Tmean", str(pet_3d)], logf)
    else:
        shutil.copyfile(pet_path, pet_3d)

    # 8) PET -> MRI (rigid)
    run(["flirt", "-in", str(pet_3d), "-ref", str(mri_brain),
         "-omat", str(pet2mri_mat), "-dof", "6", "-out", str(work / "pet_in_mri.nii.gz")], logf)

    # 9) concat PET->MNI  (MRI->MNI) ∘ (PET->MRI)
    run(["convert_xfm", "-omat", str(pet2mni_mat), "-concat", str(mri2mni_mat), str(pet2mri_mat)], logf)

    # 10) apply PET->MNI
    run(["flirt", "-in", str(pet_3d), "-ref", str(mni_ref),
         "-out", str(pet_in_mni), "-applyxfm", "-init", str(pet2mni_mat),
         "-interp", "trilinear"], logf)

    # 11) skull strip PET in MNI using MRI mask in MNI
    run(["fslmaths", str(pet_in_mni), "-mas", str(mri_mask_mni), str(pet_in_mni_br)], logf)

    # move final
    if overwrite and final_pet.exists():
        final_pet.unlink()
    shutil.copyfile(pet_in_mni_br, final_pet)
    return final_pet


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", type=Path, required=True)

    ap.add_argument("--nifti_mri", type=Path, default=Path("/home/ssddata/user071/pet_project/data/nifti/MRI"))
    ap.add_argument("--nifti_pet", type=Path, default=Path("/home/ssddata/user071/pet_project/data/nifti/PET"))

    ap.add_argument("--out_root", type=Path, default=Path("/home/ssddata/user071/pet_project/data/processed"))
    ap.add_argument("--logs_root", type=Path, default=Path("/home/ssddata/user071/pet_project/data/logs"))
    ap.add_argument("--tmp_root", type=Path, default=Path("/tmp/adni_preproc_tmp"))

    ap.add_argument("--keep_tmp", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--unique_names", action="store_true",
                    help="Save filenames with IDs: Subject__Ixxxx.nii.gz (recommended).")

    ap.add_argument("--fs_license", type=Path, default=None,
                    help="Path to FreeSurfer license.txt (sets FS_LICENSE).")

    args = ap.parse_args()

    if args.fs_license is not None:
        os.environ["FS_LICENSE"] = str(args.fs_license)

    # executables needed
    for exe in ["mri_convert", "mri_synthstrip", "fast", "flirt", "fslmaths", "convert_xfm", "fslinfo"]:
        which_or_die(exe)

    fsldir = os.environ.get("FSLDIR", "")
    if not fsldir:
        raise RuntimeError("[FATAL] FSLDIR not set. Source FSL first.")
    mni_ref = fsl_value(fsldir, "data/standard/MNI152_T1_1mm_brain.nii.gz")

    # Warn about missing FS license early
    if not os.environ.get("FS_LICENSE"):
        default = Path.home() / "freesurfer" / ".license"
        if default.exists():
            os.environ["FS_LICENSE"] = str(default)
        else:
            print("[WARN] FS_LICENSE not set and default ~/freesurfer/.license not found.")
            print("       FreeSurfer tools may fail. Provide --fs_license /path/to/license.txt")

    if not args.pairs_csv.exists():
        raise RuntimeError(f"[FATAL] pairs_csv not found: {args.pairs_csv}")

    df = pd.read_csv(args.pairs_csv)
    schema = detect_csv_schema(df)

    print("[INFO] pairs_csv:", args.pairs_csv)
    print("[INFO] columns:", list(df.columns))
    print("[INFO] detected schema:", schema)

    if args.limit > 0:
        df = df.head(args.limit).copy()

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.logs_root.mkdir(parents=True, exist_ok=True)
    args.tmp_root.mkdir(parents=True, exist_ok=True)

    done_csv = args.logs_root / "done.csv"
    fail_csv = args.logs_root / "failures.csv"

    if not done_csv.exists():
        with open(done_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["Subject", "MRI_ID", "Final_MRI", "PET_ID", "Tracer", "Final_PET"])
    if not fail_csv.exists():
        with open(fail_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["Subject", "MRI_ID", "PET_ID", "Tracer", "Stage", "Error"])

    n_mri_done = 0
    n_pet_done = 0
    n_skip_mri = 0
    n_skip_pet = 0
    n_fail = 0

    for _, row in df.iterrows():
        subject = str(row[schema["subject"]]).strip()

        # ---- MRI id
        mri_raw = row[schema["mri_id"]]
        mri_id = norm_img_id(mri_raw)

        # output names
        if args.unique_names:
            final_mri = args.out_root / "MRI" / f"{subject}__{mri_id}.nii.gz"
            final_mask = args.out_root / "MRI_MASK" / f"{subject}__{mri_id}_mask.nii.gz"
        else:
            final_mri = args.out_root / "MRI" / f"{subject}.nii.gz"
            final_mask = args.out_root / "MRI_MASK" / f"{subject}_mask.nii.gz"

        # find MRI nifti
        mri_path = find_nifti_by_id(args.nifti_mri, mri_id)
        if not mri_path:
            n_fail += 1
            with open(fail_csv, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([subject, mri_id, "", "", "find_mri", f"Missing MRI nifti for {mri_id}"])
            continue

        # ---- MRI preprocess once
        try:
            # If already exists and not overwrite, count skip for MRI
            if (not args.overwrite) and final_mri.exists() and final_mask.exists():
                n_skip_mri += 1
                # We still need internal mats for PET; but if we're skipping MRI, we can't guarantee mats exist.
                # To keep behavior consistent, we still run MRI step if any PET exists in this row.
                # We'll decide based on schema and PET presence:
                need_pets = False
                if schema["schema"] in {"A", "B"}:
                    pet_raw = row[schema["pet_id"]]
                    need_pets = not is_none_like(pet_raw)
                elif schema["schema"] == "C":
                    for tr, col in schema["pet_cols"].items():
                        if col and (col in df.columns) and (not is_none_like(row.get(col, ""))):
                            need_pets = True
                            break
                if not need_pets:
                    # MRI-only row: log done and continue
                    with open(done_csv, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([subject, mri_id, str(final_mri), "", "", ""])
                    continue

            mri_pack = process_mri_once(
                subject=subject,
                mri_id=mri_id,
                mri_path=mri_path,
                out_root=args.out_root,
                tmp_root=args.tmp_root,
                logs_root=args.logs_root,
                mni_ref=mni_ref,
                keep_tmp=True,          # keep until PETs processed; we'll cleanup per MRI row below
                overwrite=args.overwrite,
                unique_names=args.unique_names,
            )
            n_mri_done += 1
        except Exception as e:
            n_fail += 1
            with open(fail_csv, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([subject, mri_id, "", "", "mri_pipeline", str(e).splitlines()[0]])
            continue

        # ---- Decide PET list from schema
        pet_jobs = []  # list of (tracer, pet_id)
        if schema["schema"] in {"A", "B"}:
            pet_raw = row[schema["pet_id"]]
            pet_id = norm_img_id(pet_raw)

            tracer = "UNK"
            if schema["schema"] == "A" and "tracer" in schema and schema["tracer"] in df.columns:
                tracer = str(row.get(schema["tracer"], "UNK")).strip().upper()
            elif schema["schema"] == "B" and "pet_desc" in schema:
                tracer = tracer_from_description(str(row.get(schema["pet_desc"], "")))
            tracer = (tracer or "UNK").upper()

            if pet_id:
                pet_jobs.append((tracer, pet_id))

        elif schema["schema"] == "C":
            for tracer, col in schema["pet_cols"].items():
                if not col:
                    continue
                raw = row.get(col, "")
                if is_none_like(raw):
                    continue
                pet_id = norm_img_id(raw)
                if pet_id:
                    pet_jobs.append((tracer, pet_id))

        # ---- Process PET jobs (0..N)
        any_pet_done = False
        for tracer, pet_id in pet_jobs:
            # find PET nifti
            pet_path = find_nifti_by_id(args.nifti_pet, pet_id)
            if not pet_path:
                n_fail += 1
                with open(fail_csv, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([subject, mri_id, pet_id, tracer, "find_pet", f"Missing PET nifti for {pet_id}"])
                continue

            # skip if exists and not overwrite
            if args.unique_names:
                final_pet = args.out_root / "PET" / tracer / f"{subject}__{pet_id}.nii.gz"
            else:
                final_pet = args.out_root / "PET" / tracer / f"{subject}.nii.gz"

            if (not args.overwrite) and final_pet.exists():
                n_skip_pet += 1
                with open(done_csv, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([subject, mri_id, str(mri_pack["final_mri"]), pet_id, tracer, str(final_pet)])
                continue

            try:
                fpet = process_one_pet_given_mri(
                    subject=subject,
                    tracer=tracer,
                    pet_id=pet_id,
                    pet_path=pet_path,
                    mri_brain=mri_pack["mri_brain"],
                    mni_ref=mni_ref,
                    mri2mni_mat=mri_pack["mri2mni_mat"],
                    mri_mask_mni=mri_pack["mri_mask_mni"],
                    out_root=args.out_root,
                    work_mri=mri_pack["work"],
                    logs_root=args.logs_root,
                    overwrite=args.overwrite,
                    unique_names=args.unique_names,
                )
                any_pet_done = True
                n_pet_done += 1
                with open(done_csv, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([subject, mri_id, str(mri_pack["final_mri"]), pet_id, tracer, str(fpet)])
            except Exception as e:
                n_fail += 1
                with open(fail_csv, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([subject, mri_id, pet_id, tracer, "pet_pipeline", str(e).splitlines()[0]])

        # If MRI had no PET jobs, still log MRI done
        if not pet_jobs:
            with open(done_csv, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([subject, mri_id, str(mri_pack["final_mri"]), "", "", ""])

        # Cleanup per MRI row if requested
        if not args.keep_tmp:
            shutil.rmtree(mri_pack["work"], ignore_errors=True)

    print("\n" + "=" * 80)
    print("[SUMMARY]")
    print(f"MRI done={n_mri_done}  MRI skip={n_skip_mri}")
    print(f"PET done={n_pet_done}  PET skip={n_skip_pet}")
    print(f"fail={n_fail}  total_rows={len(df)}")
    print(f"[OUT]  {args.out_root}")
    print(f"[LOG]  {args.logs_root}")
    print("=" * 80)


if __name__ == "__main__":
    main()
