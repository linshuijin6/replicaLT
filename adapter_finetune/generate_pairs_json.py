import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

MRI_ROOT = Path("/mnt/nfsdata/nfsdata/ADNI/ADNI0103/Coregistration/MRI")
PET_ROOTS: Dict[str, Path] = {
    "FDG": Path("/mnt/nfsdata/nfsdata/ADNI/ADNI0103/Coregistration/PET_MNI/FDG"),
    "AV45": Path("/mnt/nfsdata/nfsdata/ADNI/ADNI0103/Coregistration/PET_MNI/AV45"),
    "AV1451": Path("/mnt/nfsdata/nfsdata/ADNI/ADNI0103/Coregistration/PET_MNI/AV1451"),
}

DEFAULTS = {
    "sex": "U",
    "weight": 0.0,
    "mmse": 0,
    "gdscale": 0,
    "cdr": 0.0,
    "faq": 0,
    "npiq": 0,
}


def build_path(root: Path, subject_id: str, image_id: Optional[str]) -> Optional[str]:
    if image_id is None or str(image_id) == "":
        return None
    return str(root / f"{subject_id}__{image_id}.nii.gz")


def make_description(age: Optional[float], weight: float, mmse: int, gdscale: int, cdr: float, faq: int, npiq: int) -> str:
    age_text = f"{age:.1f}" if age is not None else "unknown-age"
    weight_text = f"{weight:.1f}" if weight is not None else "unknown-weight"
    return (
        f"Subject is a {age_text}-year-old participant with a weight of {weight_text} kg. "
        f"The global Clinical Dementia Rating (CDR) score is {cdr}. "
        f"The Mini-Mental State Examination (MMSE) score is {mmse}. "
        f"The Geriatric Depression Scale (GDS) score is {gdscale}. "
        f"The Functional Activities Questionnaire (FAQ) score is {faq}. "
        f"The Neuropsychiatric Inventory Questionnaire (NPI-Q) Total Score is {npiq}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate JSON pairs from CSV with modality paths.")
    parser.add_argument("--input", type=str, default="/home/ssddata/linshuijin/replicaLT/adapter_finetune/data_csv/pairs0106_filtered.csv", help="Input pairs CSV")
    parser.add_argument("--output", type=str, default="adapter_finetune/pairs_generated.json", help="Output JSON path")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df.columns = [c.lower() for c in df.columns]

    records = []
    for _, row in df.iterrows():
        subject_id = str(row.get("ptid", "")).strip()
        if subject_id == "":
            continue

        age = row.get("examdate")
        try:
            age = float(age)
        except Exception:
            age = None

        sex = DEFAULTS["sex"]
        weight = DEFAULTS["weight"]
        mmse = DEFAULTS["mmse"]
        gdscale = DEFAULTS["gdscale"]
        cdr = DEFAULTS["cdr"]
        faq = DEFAULTS["faq"]
        npiq = DEFAULTS["npiq"]

        id_mri = str(int(row.get("id_mri"))) if row.get("id_mri") ==row.get("id_mri") else None
        id_fdg = str(int(row.get("id_fdg"))) if row.get("id_fdg") == row.get("id_fdg") else None
        id_av45 = str(int(row.get("id_av45"))) if row.get("id_av45") == row.get("id_av45") else None
        id_av1451 = str(int(row.get("id_av1451"))) if row.get("id_av1451") == row.get("id_av1451") else None

        mri_path = build_path(MRI_ROOT, subject_id, id_mri)
        fdg_path = build_path(PET_ROOTS["FDG"], subject_id, id_fdg)
        av45_path = build_path(PET_ROOTS["AV45"], subject_id, id_av45)
        av1451_path = build_path(PET_ROOTS["AV1451"], subject_id, id_av1451)

        desc = make_description(age, weight, mmse, gdscale, cdr, faq, npiq)

        records.append(
            {
                "name": subject_id,
                "research_group": row.get("research_group", "UNKNOWN"),
                "sex": sex,
                "weight": weight,
                "age": age,
                "mri": mri_path,
                "av45": av45_path,
                "av1451": av1451_path,
                "fdg": fdg_path,
                "mmse": mmse,
                "gdscale": gdscale,
                "cdr": cdr,
                "faq": faq,
                "npiq": npiq,
                "description": desc,
                "mri_index": -1,
                "fdg_index": -1,
                "av45_index": -1,
                "av1451_index": -1,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)

    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
