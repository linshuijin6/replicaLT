#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""PET preprocessing using completed MRI-to-MNI results.

This script is designed to mirror the style/logic of
`preprocess_mri_only_rigid6dof_from_csv_fast.py`, but for PET.

Goal (per PET image):
1) fslreorient2std + copysform2qform
2) FLIRT rigid 6DOF PET -> subject MRI (reoriented native, `out_rstd` from MRI pipeline)
3) Concatenate transforms: (PET->MRI) + (MRI->MNI) = (PET->MNI)
4) Apply PET->MNI to get PET in MNI space
5) Skull-strip PET in MNI using subject MRI mask in MNI (from MRI pipeline)

Why this pipeline (academic/common practice):
- PET-to-MRI rigid registration with MI/NMI is standard because PET has low structural detail.
- Reuse MRI-derived brain mask to skull-strip PET is more stable than PET-only skull stripping.
- Compose transforms instead of direct PET->MNI is typically more accurate.

CSV requirements:
- Must contain `subject_id` and `id_mri`.
- PET id columns are inferred from common names, e.g. `id_fdg`, `id_av45`, `id_av1451`.

Expected MRI outputs (produced by MRI script):
- mri_rstd_dir/<subject>__<Imri>_rstd.nii.gz
- mri_xfm_dir/<subject>__<Imri>_mri2mni.mat
- mri_mask_mni_dir/<subject>__<Imri>_mask.nii.gz

Outputs:
- out_pet/<modality>/<subject>__<Imri>__<Ipet>.nii.gz          (PET in MNI, brain-only)
- out_pet/<modality>/<subject>__<Imri>__<Ipet>_full.nii.gz     (PET in MNI, full-head)
- out_pet/<modality>/<subject>__<Imri>__<Ipet>_pet2mri.mat
- out_pet/<modality>/<subject>__<Imri>__<Ipet>_pet2mni.mat

Progress tracking:
- logs_root/done_pet.csv
- logs_root/fail_pet.csv
- logs_root/per_pet/<subject>/<modality>_<Ipet>.log

Notes:
- Requires FSL tools: fslreorient2std, fslorient, flirt, convert_xfm, fslmaths
- Set env to avoid oversubscription:
    export OMP_NUM_THREADS=1
    export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
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
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


# -------------------------
# Utils (mirrors MRI script)
# -------------------------

def which_or_die(exe: str) -> str:
    p = shutil.which(exe)
    if not p:
        raise RuntimeError(f"[FATAL] Missing executable in PATH: {exe}")
    return p


def run(cmd: List[str], logf: Path, check: bool = True) -> None:
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


def find_nifti_by_id(folder: Path, img_id: str) -> Optional[Path]:
    if not img_id:
        return None
    for pat in (f"*{img_id}*.nii.gz", f"*{img_id}*.nii"):
        hits = sorted(folder.glob(pat))
        if hits:
            return hits[0]
    return None


def ensure_csv_headers(done_csv: Path, fail_csv: Path) -> None:
    done_csv.parent.mkdir(parents=True, exist_ok=True)
    if not done_csv.exists():
        with open(done_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    "subject_id",
                    "id_mri",
                    "modality",
                    "id_pet",
                    "raw_path",
                    "final_pet_brain_mni",
                    "final_pet_full_mni",
                    "pet2mri_mat",
                    "pet2mni_mat",
                    "status",
                ]
            )
    if not fail_csv.exists():
        with open(fail_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["subject_id", "id_mri", "modality", "id_pet", "stage", "error"])


def load_done_set(done_csv: Path) -> Set[Tuple[str, str, str, str]]:
    done: Set[Tuple[str, str, str, str]] = set()
    if not done_csv.exists():
        return done
    try:
        with open(done_csv, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                s = (row.get("subject_id") or "").strip()
                mri = (row.get("id_mri") or "").strip()
                mod = (row.get("modality") or "").strip()
                pet = (row.get("id_pet") or "").strip()
                if s and mri and mod and pet:
                    done.add((s, mri, mod, pet))
    except Exception:
        pass
    return done


def pick_first_existing(row: pd.Series, candidates: List[str]) -> str:
    for c in candidates:
        if c in row and pd.notna(row[c]):
            v = str(row[c]).strip()
            if v and v.lower() not in {"nan", "none", "null"}:
                return v
    return ""


# -------------------------
# Single PET job
# -------------------------

def process_one_pet(
    subject: str,
    mri_id: str,
    modality: str,
    pet_id: str,
    raw_pet_path: str,
    mri_rstd_path: str,
    mri2mni_mat_path: str,
    mri_mask_mni_path: str,
    out_root: str,
    logs_root: str,
    tmp_root: str,
    mni_ref_full: str,
    overwrite: bool,
) -> Tuple[str, str, str, str, str, str, str, str, str]:

    raw_pet = Path(raw_pet_path)
    mri_rstd = Path(mri_rstd_path)
    mri2mni_mat = Path(mri2mni_mat_path)
    mri_mask_mni = Path(mri_mask_mni_path)

    out_root = Path(out_root) / modality
    out_root.mkdir(parents=True, exist_ok=True)

    logs_root = Path(logs_root)
    logs_root.mkdir(parents=True, exist_ok=True)

    tmp_root = Path(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    mni_ref_full = Path(mni_ref_full)

    pet_id_norm = norm_img_id(pet_id)
    if not pet_id_norm:
        raise RuntimeError("START::Empty pet_id")

    final_full = out_root / f"{subject}__{mri_id}__{pet_id_norm}_full.nii.gz"
    final_brain = out_root / f"{subject}__{mri_id}__{pet_id_norm}.nii.gz"
    final_pet2mri = out_root / f"{subject}__{mri_id}__{pet_id_norm}_pet2mri.mat"
    final_pet2mni = out_root / f"{subject}__{mri_id}__{pet_id_norm}_pet2mni.mat"

    if (
        (not overwrite)
        and final_full.exists()
        and final_brain.exists()
        and final_pet2mri.exists()
        and final_pet2mni.exists()
    ):
        return (
            subject,
            mri_id,
            modality,
            pet_id_norm,
            str(raw_pet),
            str(final_brain),
            str(final_full),
            str(final_pet2mri),
            str(final_pet2mni),
        )

    work = tmp_root / subject / f"{modality.lower()}_{pet_id_norm}"
    work.mkdir(parents=True, exist_ok=True)
    logf = logs_root / "per_pet" / subject / f"{modality}_{pet_id_norm}.log"

    pet_rstd = work / "pet_rstd.nii.gz"
    pet_in_mri = work / "pet_in_mri.nii.gz"
    pet2mri_mat = work / "pet_to_mri.mat"

    pet_in_mni = work / "pet_in_mni_full.nii.gz"
    pet2mni_mat = work / "pet_to_mni.mat"

    pet_brain = work / "pet_in_mni_brainonly.nii.gz"

    stage = "START"
    try:
        # sanity checks
        stage = "CHECK_INPUTS"
        if not raw_pet.exists():
            raise RuntimeError(f"Missing PET NIfTI: {raw_pet}")
        if not mri_rstd.exists():
            raise RuntimeError(f"Missing MRI rstd: {mri_rstd}")
        if not mri2mni_mat.exists():
            raise RuntimeError(f"Missing MRI->MNI mat: {mri2mni_mat}")
        if not mri_mask_mni.exists():
            raise RuntimeError(f"Missing MRI mask in MNI: {mri_mask_mni}")

        # 1) Reorient PET
        stage = "REORIENT"
        run(["fslreorient2std", str(raw_pet), str(pet_rstd)], logf)
        run(["fslorient", "-copysform2qform", str(pet_rstd)], logf)

        # 2) Rigid PET -> MRI (mutual information)
        stage = "FLIRT_PET2MRI_6DOF"
        run(
            [
                "flirt",
                "-in",
                str(pet_rstd),
                "-ref",
                str(mri_rstd),
                "-out",
                str(pet_in_mri),
                "-omat",
                str(pet2mri_mat),
                "-dof",
                "6",
                "-cost",
                "normmi",
                "-searchrx",
                "-30",
                "30",
                "-searchry",
                "-30",
                "30",
                "-searchrz",
                "-30",
                "30",
            ],
            logf,
        )

        # 3) Compose PET->MNI = (MRI->MNI) o (PET->MRI)
        stage = "CONCAT_XFM"
        run(
            [
                "convert_xfm",
                "-omat",
                str(pet2mni_mat),
                "-concat",
                str(mri2mni_mat),
                str(pet2mri_mat),
            ],
            logf,
        )

        # 4) Apply PET->MNI
        stage = "APPLY_XFM_MNI"
        run(
            [
                "flirt",
                "-in",
                str(pet_rstd),
                "-ref",
                str(mni_ref_full),
                "-out",
                str(pet_in_mni),
                "-applyxfm",
                "-init",
                str(pet2mni_mat),
                "-interp",
                "trilinear",
            ],
            logf,
        )

        # 5) Skull strip in MNI using MRI-derived mask
        stage = "SKULLSTRIP_MNI"
        run(["fslmaths", str(pet_in_mni), "-mas", str(mri_mask_mni), str(pet_brain)], logf)

        # 6) Write outputs
        stage = "WRITE_OUTPUTS"
        if overwrite:
            for p in (final_full, final_brain, final_pet2mri, final_pet2mni):
                if p.exists():
                    p.unlink()

        shutil.copyfile(pet_in_mni, final_full)
        shutil.copyfile(pet_brain, final_brain)
        shutil.copyfile(pet2mri_mat, final_pet2mri)
        shutil.copyfile(pet2mni_mat, final_pet2mni)

        return (
            subject,
            mri_id,
            modality,
            pet_id_norm,
            str(raw_pet),
            str(final_brain),
            str(final_full),
            str(final_pet2mri),
            str(final_pet2mni),
        )

    except Exception as e:
        raise RuntimeError(f"{stage}::{str(e).splitlines()[0]}") from e


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--pairs_csv", type=Path, required=True, help="CSV with subject_id, id_mri and PET ids")

    ap.add_argument("--nifti_fdg", type=Path, default=None, help="Folder containing FDG NIfTI")
    ap.add_argument("--nifti_av45", type=Path, default=None, help="Folder containing AV45 NIfTI")
    ap.add_argument("--nifti_tau", type=Path, default=None, help="Folder containing AV1451(TAU) NIfTI")

    ap.add_argument("--mri_rstd_dir", type=Path, required=True, help="Folder of MRI rstd outputs")
    ap.add_argument("--mri_xfm_dir", type=Path, required=True, help="Folder of MRI->MNI mats")
    ap.add_argument("--mri_mask_mni_dir", type=Path, required=True, help="Folder of MRI masks in MNI")

    ap.add_argument("--out_root", type=Path, required=True, help="Output root folder (subfolders per modality)")
    ap.add_argument("--logs_root", type=Path, required=True)
    ap.add_argument("--tmp_root", type=Path, required=True)

    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=-1)

    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")

    for exe in ["fslreorient2std", "fslorient", "flirt", "convert_xfm", "fslmaths"]:
        which_or_die(exe)

    fsldir = os.environ.get("FSLDIR", "")
    if not fsldir:
        raise RuntimeError("[FATAL] FSLDIR not set. Source FSL first.")
    mni_ref_full = Path(fsldir) / "data/standard/MNI152_T1_1mm.nii.gz"
    if not mni_ref_full.exists():
        raise RuntimeError(f"[FATAL] Missing MNI ref: {mni_ref_full}")

    df = pd.read_csv(args.pairs_csv)
    if not {"subject_id", "id_mri"}.issubset(df.columns):
        raise RuntimeError(f"[FATAL] CSV must contain subject_id, id_mri. Found: {list(df.columns)}")

    if args.limit > 0:
        df = df.head(args.limit).copy()

    args.logs_root.mkdir(parents=True, exist_ok=True)
    args.tmp_root.mkdir(parents=True, exist_ok=True)
    args.out_root.mkdir(parents=True, exist_ok=True)

    done_csv = args.logs_root / "done_pet.csv"
    fail_csv = args.logs_root / "fail_pet.csv"
    ensure_csv_headers(done_csv, fail_csv)
    done_set = load_done_set(done_csv)

    # Candidate PET id columns (flexible)
    pet_col_candidates: Dict[str, List[str]] = {
        "FDG": ["id_fdg", "image_id_fdg", "image_id(18f-fdg)", "image_id(18F-FDG)", "image_id_fdg_pet"],
        "AV45": ["id_av45", "image_id_av45", "image_id(18f-av45)", "image_id(18F-AV45)", "image_id_av45_pet"],
        "TAU": ["id_av1451", "id_tau", "image_id_av1451", "image_id(18f-av1451)", "image_id(18F-AV1451)"],
    }

    nifti_dirs: Dict[str, Optional[Path]] = {
        "FDG": args.nifti_fdg,
        "AV45": args.nifti_av45,
        "TAU": args.nifti_tau,
    }

    # Build jobs
    jobs: List[Tuple[str, str, str, str, str, str, str, str]] = []
    seen: Set[Tuple[str, str, str, str]] = set()

    for _, row in df.iterrows():
        subject = str(row["subject_id"]).strip()
        mri_id = norm_img_id(row["id_mri"])
        if not subject or not mri_id:
            continue

        mri_rstd_path = args.mri_rstd_dir / f"{subject}__{mri_id}_rstd.nii.gz"
        mri2mni_mat_path = args.mri_xfm_dir / f"{subject}__{mri_id}_mri2mni.mat"
        mri_mask_mni_path = args.mri_mask_mni_dir / f"{subject}__{mri_id}_mask.nii.gz"

        for modality, candidates in pet_col_candidates.items():
            nifti_dir = nifti_dirs.get(modality)
            if nifti_dir is None:
                continue

            pet_raw_id = pick_first_existing(row, candidates)
            pet_id = norm_img_id(pet_raw_id)
            if not pet_id:
                continue

            key = (subject, mri_id, modality, pet_id)
            if key in seen:
                continue
            seen.add(key)

            if (not args.overwrite) and (key in done_set):
                continue

            raw_pet = find_nifti_by_id(nifti_dir, pet_id)
            if not raw_pet:
                with open(fail_csv, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([subject, mri_id, modality, pet_id, "FIND_RAW", f"Missing PET NIfTI for {pet_id}"])
                continue

            if (not mri_rstd_path.exists()) or (not mri2mni_mat_path.exists()) or (not mri_mask_mni_path.exists()):
                with open(fail_csv, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([subject, mri_id, modality, pet_id, "CHECK_MRI", "Missing required MRI outputs (rstd/mat/mask)"])
                continue

            jobs.append(
                (
                    subject,
                    mri_id,
                    modality,
                    pet_id,
                    str(raw_pet),
                    str(mri_rstd_path),
                    str(mri2mni_mat_path),
                    str(mri_mask_mni_path),
                )
            )

    total = len(jobs)
    print(f"[INFO] Unique PET rows (dedup): {len(seen)}")
    print(f"[INFO] Already done (skipped):  {len(load_done_set(done_csv))}")
    print(f"[INFO] To process now:          {total}")
    print(f"[INFO] Parallel jobs:           {args.jobs}")
    print(f"[INFO] Overwrite:               {args.overwrite}")
    print(f"[INFO] done_pet.csv:            {done_csv}")
    print(f"[INFO] fail_pet.csv:            {fail_csv}")

    if total == 0:
        print("[DONE] Nothing to do.")
        return

    n_done = 0
    n_fail = 0

    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {
            ex.submit(
                process_one_pet,
                subject,
                mri_id,
                modality,
                pet_id,
                raw_pet,
                mri_rstd,
                mri2mni,
                mri_mask_mni,
                str(args.out_root),
                str(args.logs_root),
                str(args.tmp_root),
                str(mni_ref_full),
                args.overwrite,
            ): (subject, mri_id, modality, pet_id)
            for (
                subject,
                mri_id,
                modality,
                pet_id,
                raw_pet,
                mri_rstd,
                mri2mni,
                mri_mask_mni,
            ) in jobs
        }

        for fut in as_completed(futs):
            subject, mri_id, modality, pet_id = futs[fut]
            try:
                (
                    subject,
                    mri_id,
                    modality,
                    pet_id_norm,
                    raw_path,
                    final_brain,
                    final_full,
                    mat_pet2mri,
                    mat_pet2mni,
                ) = fut.result()

                with open(done_csv, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(
                        [
                            subject,
                            mri_id,
                            modality,
                            pet_id_norm,
                            raw_path,
                            final_brain,
                            final_full,
                            mat_pet2mri,
                            mat_pet2mni,
                            "OK",
                        ]
                    )
                done_set.add((subject, mri_id, modality, pet_id_norm))

                n_done += 1
                if n_done % 25 == 0 or (n_done + n_fail) == total:
                    print(f"[PROGRESS] done={n_done} fail={n_fail} / total={total}")

            except Exception as e:
                msg = str(e)
                if "::" in msg:
                    stage, err = msg.split("::", 1)
                else:
                    stage, err = "PIPELINE", msg.splitlines()[0] if msg else "Unknown error"

                with open(fail_csv, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([subject, mri_id, modality, pet_id, stage, err])

                n_fail += 1
                if (n_done + n_fail) % 25 == 0 or (n_done + n_fail) == total:
                    print(f"[PROGRESS] done={n_done} fail={n_fail} / total={total}")

    print("[DONE] PET preprocessing completed.")
    try:
        done_rows = sum(1 for _ in open(done_csv, "r", encoding="utf-8")) - 1
    except Exception:
        done_rows = -1
    print(f"[SUMMARY] done={n_done} fail={n_fail} (this run)  |  done_pet.csv rows ~ {done_rows}")


if __name__ == "__main__":
    main()
