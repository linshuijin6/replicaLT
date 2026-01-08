import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match PLASMA rows to MRI and PET exams within a day window and emit a combined table."
    )
    parser.add_argument(
        "--plasma",
        type=Path,
        required=True,
        help="Path to PLASMA CSV (expects columns PTID and EXAMDATE).",
    )
    parser.add_argument(
        "--mri",
        type=Path,
        required=True,
        help="Path to MRI CSV (expects columns subject_id, image_date, image_id, series_description).",
    )
    parser.add_argument(
        "--pet",
        type=Path,
        required=True,
        help="Path to PET CSV (expects columns subject_id, image_date, image_id, radiopharmaceutical).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("adapter_finetune/data_csv/plasma_mri_pet_matched_180d.csv"),
        help="Where to write the combined table.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("adapter_finetune/data_csv"),
        help="Directory to store helper reports for unmatched items.",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=180,
        help="Maximum allowed |EXAMDATE - image_date| in days for a match.",
    )
    return parser.parse_args()


def load_plasma(path: Path) -> pd.DataFrame:
    usecols = ["PTID", "EXAMDATE"]
    df = pd.read_csv(path, usecols=usecols, dtype=str)
    df["EXAMDATE_DT"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")
    return df


def load_mri(path: Path) -> pd.DataFrame:
    usecols = ["subject_id", "image_date", "image_id", "series_description"]
    df = pd.read_csv(
        path,
        usecols=usecols,
        dtype={"image_id": str, "subject_id": str},
        low_memory=False,
    )
    df["image_date"] = pd.to_datetime(df["image_date"], errors="coerce")
    df = df.dropna(subset=["subject_id", "image_id", "image_date"], how="any")
    df = df[df["subject_id"].astype(str).str.strip() != ""]
    df = df[df["image_id"].astype(str).str.strip() != ""]
    return df


def load_pet(path: Path) -> pd.DataFrame:
    usecols = ["subject_id", "image_date", "image_id", "radiopharmaceutical"]
    df = pd.read_csv(
        path,
        usecols=usecols,
        dtype={"image_id": str, "subject_id": str},
        low_memory=False,
    )
    df["image_date"] = pd.to_datetime(df["image_date"], errors="coerce")
    df = df.dropna(subset=["subject_id", "image_id", "image_date", "radiopharmaceutical"], how="any")
    df = df[df["subject_id"].astype(str).str.strip() != ""]
    df = df[df["image_id"].astype(str).str.strip() != ""]
    df["rad_type"] = df["radiopharmaceutical"].str.strip().str.upper()
    return df


def select_best(
    df: Optional[pd.DataFrame],
    target_date: pd.Timestamp,
    window_days: int,
    prefer_non_repeat: bool = False,
    repeat_col: Optional[str] = None,
) -> Optional[str]:
    if df is None or df.empty or pd.isna(target_date):
        return None
    candidates = df.copy()
    candidates["days_diff"] = (candidates["image_date"] - target_date).abs().dt.days
    candidates = candidates[candidates["days_diff"] <= window_days]
    if candidates.empty:
        return None

    if prefer_non_repeat and repeat_col:
        non_repeat = candidates[~candidates[repeat_col].str.contains("repeat", case=False, na=False)]
        if not non_repeat.empty:
            candidates = non_repeat

    best = candidates.sort_values(["days_diff", "image_date"]).iloc[0]
    return best["image_id"]


def build_pet_lookup(pet_df: pd.DataFrame) -> dict:
    lookup = {}
    for subject_id, group in pet_df.groupby("subject_id"):
        lookup[subject_id] = {
            "18F-FDG": group[group["rad_type"] == "18F-FDG"],
            "18F-AV45": group[group["rad_type"] == "18F-AV45"],
            "18F-AV1451": group[group["rad_type"] == "18F-AV1451"],
        }
    return lookup


def main() -> int:
    args = parse_args()

    plasma = load_plasma(args.plasma)
    mri = load_mri(args.mri)
    pet = load_pet(args.pet)

    mri_lookup = {sid: g for sid, g in mri.groupby("subject_id")}
    pet_lookup = build_pet_lookup(pet)

    results = []
    for ptid, p_rows in plasma.groupby("PTID"):
        mri_df = mri_lookup.get(ptid)
        pet_dict = pet_lookup.get(ptid, {})

        for _, row in p_rows.iterrows():
            exam_date = row["EXAMDATE_DT"]
            exam_date_str = row["EXAMDATE"]

            result_row = {
                "PTID": ptid,
                "EXAMDATE": exam_date_str,
                "image_id(MRI)": select_best(
                    mri_df,
                    exam_date,
                    window_days=args.window_days,
                    prefer_non_repeat=True,
                    repeat_col="series_description",
                ),
                "image_id(18F-FDG)": select_best(
                    pet_dict.get("18F-FDG"),
                    exam_date,
                    window_days=args.window_days,
                ),
                "image_id(18F-AV45)": select_best(
                    pet_dict.get("18F-AV45"),
                    exam_date,
                    window_days=args.window_days,
                ),
                "image_id(18F-AV1451)": select_best(
                    pet_dict.get("18F-AV1451"),
                    exam_date,
                    window_days=args.window_days,
                ),
            }
            results.append(result_row)

    output_df = pd.DataFrame(results, columns=[
        "PTID",
        "EXAMDATE",
        "image_id(MRI)",
        "image_id(18F-FDG)",
        "image_id(18F-AV45)",
        "image_id(18F-AV1451)",
    ])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)

    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    def _missing(col: pd.Series) -> pd.Series:
        return col.isna() | col.astype(str).str.strip().eq("")

    missing_mri = _missing(output_df["image_id(MRI)"])
    missing_fdg = _missing(output_df["image_id(18F-FDG)"])
    missing_av45 = _missing(output_df["image_id(18F-AV45)"])
    missing_av1451 = _missing(output_df["image_id(18F-AV1451)"])

    plasma_missing_flags = output_df.assign(
        missing_mri=missing_mri,
        missing_fdg=missing_fdg,
        missing_av45=missing_av45,
        missing_av1451=missing_av1451,
    )
    plasma_missing_flags.to_csv(report_dir / "plasma_missing_by_modality.csv", index=False)

    plasma_unmatched_all = plasma_missing_flags[
        missing_mri & missing_fdg & missing_av45 & missing_av1451
    ]
    plasma_unmatched_all.to_csv(report_dir / "plasma_unmatched_all_modalities.csv", index=False)

    used_mri = set(output_df["image_id(MRI)"].dropna())
    used_fdg = set(output_df["image_id(18F-FDG)"].dropna())
    used_av45 = set(output_df["image_id(18F-AV45)"].dropna())
    used_av1451 = set(output_df["image_id(18F-AV1451)"].dropna())

    mri_unmatched = mri[~mri["image_id"].isin(used_mri)]
    mri_unmatched.to_csv(report_dir / "mri_unmatched.csv", index=False)

    used_pet = used_fdg | used_av45 | used_av1451
    pet_unmatched = pet[~pet["image_id"].isin(used_pet)]
    pet_unmatched.to_csv(report_dir / "pet_unmatched.csv", index=False)

    print(f"Wrote {len(output_df)} rows to {args.output}")
    print(
        "Summary: "
        f"plasma missing MRI {missing_mri.sum()}, "
        f"FDG {missing_fdg.sum()}, AV45 {missing_av45.sum()}, AV1451 {missing_av1451.sum()}; "
        f"plasma unmatched all {len(plasma_unmatched_all)}; "
        f"mri unmatched {len(mri_unmatched)}; pet unmatched {len(pet_unmatched)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
