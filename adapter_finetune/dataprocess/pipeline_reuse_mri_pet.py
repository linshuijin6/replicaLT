#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline to reuse existing MRI outputs (LorenzoT/processed) before running PET preprocessing.
Steps per row of the 180d CSV:
1) Check if MRI outputs already present in target Coregistration; if missing, try to copy from source processed folder.
2) If still missing, run MRI preprocessing (process_one_mri) to generate outputs.
3) For each PET modality present in the row, locate raw PET NIfTI via ADNI0103 structure and run process_one_pet.

This script only orchestrates; it reuses the existing processing functions from:
- preprocess_mri_missing_adni0103.py
- preprocess_pet_from_mri_mni_rigid6dof.py
"""
from __future__ import annotations
import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from report_error import email_on_error

from preprocess_mri_missing_adni0103 import (
    find_nifti_struct,
    norm_img_id as norm_mri_id,
    process_one_mri,
    which_or_die,
)
from preprocess_pet_from_mri_mni_rigid6dof import (
    norm_img_id as norm_pet_id,
    process_one_pet,
)


def ensure_headers(path: Path, header: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)


def append_row(path: Path, row: List[str]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def target_mri_paths(root: Path, subject: str, mri_id: str) -> Dict[str, Path]:
    return {
        "mri": root / "MRI" / f"{subject}__{mri_id}.nii.gz",
        "mask": root / "MRI_MASK" / f"{subject}__{mri_id}_mask.nii.gz",
        "xfm": root / "MRI_XFM" / f"{subject}__{mri_id}_mri2mni.mat",
        "rstd": root / "MRI_NATIVE_RSTD" / f"{subject}__{mri_id}_rstd.nii.gz",
    }


def source_mri_paths(root: Path, subject: str, mri_id: str) -> Dict[str, Path]:
    return {
        "mri": root / "MRI" / f"{subject}__{mri_id}.nii.gz",
        "mask": root / "MRI_MASK" / f"{subject}__{mri_id}_mask.nii.gz",
        "xfm": root / "MRI_XFM" / f"{subject}__{mri_id}_mri2mni.mat",
        "rstd": root / "MRI_NATIVE_RSTD" / f"{subject}__{mri_id}_rstd.nii.gz",
    }
    
def find_first(root: Path, pattern: str) -> Path | None:
    try:
        for p in root.rglob(pattern):
            return p
    except Exception:
        return None
    return None

def locate_source_mri(root: Path, subject: str, mri_id: str) -> Dict[str, Path]:
    base = f"{subject}__{mri_id}"
    candidates = {
        "mri": [f"**/{base}.nii.gz", f"**/{base}.nii"],
        "mask": [f"**/{base}_mask.nii.gz", f"**/{base}_mask.nii"],
        "xfm": [f"**/{base}_mri2mni.mat"],
        "rstd": [f"**/{base}_rstd.nii.gz", f"**/{base}_rstd.nii"],
    }
    out: Dict[str, Path] = {}
    for key, patterns in candidates.items():
        found: Path | None = None
        for pat in patterns:
            found = find_first(root, pat)
            if found:
                break
        if found:
            out[key] = found
    return out


def all_exist(paths: Dict[str, Path]) -> bool:
    return all(p.exists() for p in paths.values())


def copy_mri(src: Dict[str, Path], dst: Dict[str, Path]) -> bool:
    ok = True
    for key, srcp in src.items():
        dstp = dst[key]
        if dstp.exists():
            continue
        if not srcp.exists():
            ok = False
            continue
        dstp.parent.mkdir(parents=True, exist_ok=True)
        dstp.write_bytes(srcp.read_bytes())
    return ok and all_exist(dst)

@email_on_error()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", type=Path, required=True, help="CSV like plasma_mri_pet_matched_180d.csv")
    ap.add_argument("--nifti_root", type=Path, required=True, help="ADNI0103 NIFTI root")
    ap.add_argument("--source_mri_root", type=Path, required=True, help="Existing processed MRI root (LorenzoT)")
    ap.add_argument("--target_root", type=Path, required=True, help="Coregistration root to store MRI/PET outputs")
    ap.add_argument("--logs_root", type=Path, required=True)
    ap.add_argument("--tmp_root", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--overwrite", action="store_true", help="Overwrite PET outputs if present")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")

    for exe in [
        "fslreorient2std",
        "fslorient",
        "mri_synthstrip",
        "fslmaths",
        "fast",
        "flirt",
        "convert_xfm",
    ]:
        which_or_die(exe)

    fsldir = os.environ.get("FSLDIR", "")
    if not fsldir:
        raise RuntimeError("FSLDIR not set")
    mni_ref_full = Path(fsldir) / "data/standard/MNI152_T1_1mm.nii.gz"
    if not mni_ref_full.exists():
        raise RuntimeError(f"Missing MNI ref: {mni_ref_full}")

    target_root = args.target_root
    for sub in ["MRI", "MRI_MASK", "MRI_NATIVE_RSTD", "MRI_XFM", "PET_MNI"]:
        (target_root / sub).mkdir(parents=True, exist_ok=True)
    args.logs_root.mkdir(parents=True, exist_ok=True)
    args.tmp_root.mkdir(parents=True, exist_ok=True)

    mri_log = args.logs_root / "pipeline_mri.csv"
    pet_log = args.logs_root / "pipeline_pet.csv"
    ensure_headers(mri_log, ["subject_id", "id_mri", "action", "detail", "status"])
    ensure_headers(pet_log, ["subject_id", "id_mri", "modality", "id_pet", "raw_path", "status", "detail"])

    df = pd.read_csv(args.pairs_csv)
    required_cols = {"PTID", "id_mri"}
    if not required_cols.issubset(df.columns):
        raise RuntimeError(f"CSV missing columns: {required_cols}")
    if args.limit > 0:
        df = df.head(args.limit).copy()

    modality_cols: List[Tuple[str, str]] = [
        ("FDG", "id_fdg"),
        ("AV45", "id_av45"),
        ("TAU", "id_av1451"),
    ]

    rows: List[Tuple[str, str, pd.Series]] = []
    for _, row in df.iterrows():
        subject = str(row["PTID"]).strip()
        mri_id_raw = row["id_mri"] if "id_mri" in row else ""
        mri_id = norm_mri_id(mri_id_raw)
        if not subject or not mri_id:
            append_row(mri_log, [subject, str(mri_id_raw), "skip", "missing subject or mri id", "SKIP"])
            continue
        rows.append((subject, mri_id, row))

    total_rows = len(rows)
    print(f"[INFO] total rows to consider: {total_rows}", flush=True)

    def process_row(subject: str, mri_id: str, row: pd.Series, phase: str) -> None:
        tgt_paths = target_mri_paths(target_root, subject, mri_id)

        print(f"[{phase.upper()}][MRI] {subject} {mri_id}", flush=True)

        if all_exist(tgt_paths):
            append_row(mri_log, [subject, mri_id, f"{phase}_reuse_target", "already present", "OK"])
        else:
            src_paths = source_mri_paths(args.source_mri_root, subject, mri_id)
            if all_exist(src_paths):
                copied = copy_mri(src_paths, tgt_paths)
                if copied:
                    append_row(mri_log, [subject, mri_id, f"{phase}_copy", "from source_mri_root", "OK"])
                else:
                    append_row(mri_log, [subject, mri_id, f"{phase}_copy", "missing pieces in source_mri_root", "FAIL"])
                    return
            else:
                located = locate_source_mri(args.source_mri_root, subject, mri_id)
                if len(located) == 4:
                    copied = copy_mri(located, tgt_paths)
                    if copied:
                        append_row(mri_log, [subject, mri_id, f"{phase}_copy_rglob", "from source_mri_root", "OK"])
                        print(f"[{phase.upper()}][MRI] located via rglob", flush=True)
                    else:
                        append_row(mri_log, [subject, mri_id, f"{phase}_copy_rglob", "missing pieces in rglob", "FAIL"])
                        return
                elif phase == "phase2":
                    try:
                        raw = find_nifti_struct(args.nifti_root, subject, mri_id)
                        if not raw:
                            append_row(mri_log, [subject, mri_id, "find_raw", "missing raw in nifti_root", "FAIL"])
                            return
                        res = process_one_mri(
                            subject,
                            mri_id,
                            str(raw),
                            str(target_root / "MRI"),
                            str(target_root / "MRI_MASK"),
                            str(target_root / "MRI_XFM"),
                            str(target_root / "MRI_NATIVE_RSTD"),
                            str(args.logs_root),
                            str(args.tmp_root),
                            str(mni_ref_full),
                            args.overwrite,
                        )
                        append_row(mri_log, [subject, mri_id, "run", res[2], "OK"])
                    except Exception as e:  # noqa: BLE001
                        append_row(mri_log, [subject, mri_id, "run", str(e).splitlines()[0], "FAIL"])
                        return
                else:
                    append_row(mri_log, [subject, mri_id, f"{phase}_copy", "missing in source, postpone", "SKIP"])
                    return

        if not all_exist(tgt_paths):
            append_row(mri_log, [subject, mri_id, "verify", "target MRI still missing", "FAIL"])
            return

        for modality, col in modality_cols:
            if col not in df.columns:
                continue
            pet_raw_val = row[col]
            pet_id = norm_pet_id(pet_raw_val)
            if not pet_id:
                continue
            print(f"[{phase.upper()}][PET] {subject} {mri_id} {modality} {pet_id}", flush=True)
            pet_out_root = target_root / "PET_MNI"
            pet_brain = pet_out_root / modality / f"{subject}__{mri_id}__{pet_id}.nii.gz"
            pet_full = pet_out_root / modality / f"{subject}__{mri_id}__{pet_id}_full.nii.gz"
            pet2mni = pet_out_root / modality / f"{subject}__{mri_id}__{pet_id}_pet2mni.mat"
            pet2mri = pet_out_root / modality / f"{subject}__{mri_id}__{pet_id}_pet2mri.mat"
            if (not args.overwrite) and pet_brain.exists() and pet_full.exists() and pet2mni.exists() and pet2mri.exists():
                append_row(pet_log, [subject, mri_id, modality, pet_id, "", "SKIP", "already present"])
                continue

            raw_pet = find_nifti_struct(args.nifti_root, subject, pet_id)
            if not raw_pet:
                append_row(pet_log, [subject, mri_id, modality, pet_id, "", "FAIL", "missing raw PET"])
                continue

            try:
                res_pet = process_one_pet(
                    subject,
                    mri_id,
                    modality,
                    pet_id,
                    str(raw_pet),
                    str(tgt_paths["rstd"]),
                    str(tgt_paths["xfm"]),
                    str(tgt_paths["mask"]),
                    str(pet_out_root),
                    str(args.logs_root),
                    str(args.tmp_root),
                    str(mni_ref_full),
                    args.overwrite,
                )
                append_row(pet_log, [subject, mri_id, modality, res_pet[3], res_pet[4], "OK", ""])
            except Exception as e:  # noqa: BLE001
                append_row(pet_log, [subject, mri_id, modality, pet_id, str(raw_pet), "FAIL", str(e).splitlines()[0]])

    # Phase 1: rows with source MRI available (or already in target)
    for idx, (subject, mri_id, row) in enumerate(rows, start=1):
        src_paths = source_mri_paths(args.source_mri_root, subject, mri_id)
        if all_exist(target_mri_paths(target_root, subject, mri_id)) or all_exist(src_paths):
            print(f"[PHASE1] {idx}/{total_rows} {subject} {mri_id}", flush=True)
            process_row(subject, mri_id, row, "phase1")

    # Phase 2: remaining rows (generate MRI if needed)
    for idx, (subject, mri_id, row) in enumerate(rows, start=1):
        if not all_exist(target_mri_paths(target_root, subject, mri_id)) and not all_exist(source_mri_paths(args.source_mri_root, subject, mri_id)):
            print(f"[PHASE2] {idx}/{total_rows} {subject} {mri_id}", flush=True)
            process_row(subject, mri_id, row, "phase2")

    print("[DONE] pipeline finished")


if __name__ == "__main__":
    main()
