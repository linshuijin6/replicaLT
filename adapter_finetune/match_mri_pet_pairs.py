from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

MRI_CSV = Path("adapter_finetune/MRI_Sub_0102.csv")
PET_CSV = Path("adapter_finetune/PET_Sub_0102.csv")
OUTPUT_CSV = Path("adapter_finetune/mri_pet_pairs_matched.csv")

MRI_PHASES = {"ADNI3", "ADNI4"}
PET_TRACERS = {"18F-FDG", "18F-AV45", "18F-AV1451"}
MAX_DAYS = 180


def _pick_closest(pet_rows: pd.DataFrame, mri_date: pd.Timestamp) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    if pd.isna(mri_date) or pet_rows.empty:
        return None, None, None

    pet_rows = pet_rows.copy()
    pet_rows["delta_days"] = (pet_rows["image_date_dt"] - mri_date).abs().dt.days
    pet_rows = pet_rows[pet_rows["delta_days"] <= MAX_DAYS]
    if pet_rows.empty:
        return None, None, None

    row = pet_rows.sort_values(["delta_days", "image_date_dt", "image_id"]).iloc[0]
    return str(row["image_id"]), row["image_date"], int(row["delta_days"])


def _remove_repeat_when_duplicate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    group_sizes = df.groupby(["subject_id", "image_date"]).size()
    dup_keys = set(group_sizes[group_sizes > 1].index)
    if not dup_keys:
        return df

    df["_dup"] = df.apply(lambda r: (r["subject_id"], r["image_date"]) in dup_keys, axis=1)
    df["_repeat"] = df["series_description"].str.contains("repeat", case=False, na=False)
    cleaned = df[~(df["_dup"] & df["_repeat"])].drop(columns=["_dup", "_repeat"])
    return cleaned


def load_filtered() -> Tuple[pd.DataFrame, pd.DataFrame]:
    mri = pd.read_csv(MRI_CSV)
    pet = pd.read_csv(PET_CSV)

    mri = mri[(mri["mri_protocol_phase"].isin(MRI_PHASES)) & (mri["series_type"] == "T1w")]
    mri = _remove_repeat_when_duplicate(mri)
    mri["image_date_dt"] = pd.to_datetime(mri["image_date"], errors="coerce")

    pet = pet[pet["radiopharmaceutical"].isin(PET_TRACERS)].copy()
    pet["image_date_dt"] = pd.to_datetime(pet["image_date"], errors="coerce")
    return mri, pet


def match_records(mri: pd.DataFrame, pet: pd.DataFrame) -> pd.DataFrame:
    pet_by_tracer: Dict[str, pd.DataFrame] = {
        tracer: pet[pet["radiopharmaceutical"] == tracer].copy() for tracer in PET_TRACERS
    }

    records = []
    for _, row in mri.iterrows():
        subject_id = row["subject_id"]
        mri_date = row["image_date_dt"]

        sub_pet = pet[pet["subject_id"] == subject_id]
        fdg_id, fdg_date, fdg_delay = _pick_closest(sub_pet[sub_pet["radiopharmaceutical"] == "18F-FDG"], mri_date)
        av45_id, av45_date, av45_delay = _pick_closest(sub_pet[sub_pet["radiopharmaceutical"] == "18F-AV45"], mri_date)
        av1451_id, av1451_date, av1451_delay = _pick_closest(sub_pet[sub_pet["radiopharmaceutical"] == "18F-AV1451"], mri_date)

        delays = [d for d in [fdg_delay, av45_delay, av1451_delay] if d is not None]
        delay = min(delays) if delays else None

        records.append(
            {
                "subject_id": subject_id,
                "image_id_mri": row["image_id"],
                "date_mri": row["image_date"],
                "image_id_fdg": fdg_id,
                "date_fdg": fdg_date,
                "image_id_av45": av45_id,
                "date_av45": av45_date,
                "image_id_av1451": av1451_id,
                "date_av1451": av1451_date,
                "delay": delay,
            }
        )

    result = pd.DataFrame.from_records(records)
    result = result.sort_values(["subject_id", "date_mri", "image_id_mri"])
    return result


def main() -> None:
    mri, pet = load_filtered()
    matches = match_records(mri, pet)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(OUTPUT_CSV, index=False)

    print(f"MRI rows after filter: {len(mri)}")
    print(f"PET rows after filter: {len(pet)}")
    matched_counts = (
        matches[["image_id_fdg", "image_id_av45", "image_id_av1451"]]
        .notna()
        .rename(columns=lambda c: c.replace("image_id_", ""))
        .sum()
    )
    print("Matched counts by tracer (non-null image_id):")
    print(matched_counts)
    print(f"Wrote pairs to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
