import argparse
from pathlib import Path
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter valid MRI+PET rows and check file existence.")
    p.add_argument("--input", type=Path, required=True, help="Matched CSV produced by match_plasma_mri_pet.py")
    p.add_argument(
        "--output-valid",
        type=Path,
        default=Path("adapter_finetune/data_csv/plasma_mri_pet_valid.csv"),
        help="Where to write rows that have MRI and at least one PET ID.",
    )
    p.add_argument(
        "--output-files",
        type=Path,
        default=Path("adapter_finetune/data_csv/plasma_mri_pet_valid_with_files.csv"),
        help="Where to write rows with file existence annotations.",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/ssddata/user071/pet_project/data/processed"),
        help="Root directory that contains MRI and PET subfolders.",
    )
    return p.parse_args()


def load_matched(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    def _clean_id(v: str) -> str:
        s = str(v).strip()
        if s.lower() in {"", "nan"}:
            return ""
        try:
            f = float(s)
            if f.is_integer():
                return str(int(f))
        except ValueError:
            pass
        return s
    for col in ["PTID", "EXAMDATE", "image_id(MRI)", "image_id(18F-FDG)", "image_id(18F-AV45)", "image_id(18F-AV1451)"]:
        if col in df.columns:
            if col.startswith("image_id"):
                df[col] = df[col].apply(_clean_id)
            else:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace({"nan": ""})
    return df


def has_any_pet(row: pd.Series) -> bool:
    return any(row[c] != "" for c in ["image_id(18F-FDG)", "image_id(18F-AV45)", "image_id(18F-AV1451)"])


def path_for(modality: str, ptid: str, image_id: str, data_root: Path) -> Path:
    sub = {
        "MRI": Path("MRI"),
        "18F-FDG": Path("PET/FDG"),
        "18F-AV45": Path("PET/AV45"),
        "18F-AV1451": Path("PET/AV1451"),
    }[modality]
    return data_root / sub / f"{ptid}__I{image_id}.nii.gz"


def annotate_files(df: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        ptid = row["PTID"]
        mri_id = row["image_id(MRI)"]
        fdg_id = row["image_id(18F-FDG)"]
        av45_id = row["image_id(18F-AV45)"]
        av1451_id = row["image_id(18F-AV1451)"]

        def handle(modality: str, img_id: str):
            if img_id == "":
                return "", False
            p = path_for(modality, ptid, img_id, data_root)
            return str(p), p.exists()

        mri_path, mri_exists = handle("MRI", mri_id)
        fdg_path, fdg_exists = handle("18F-FDG", fdg_id)
        av45_path, av45_exists = handle("18F-AV45", av45_id)
        av1451_path, av1451_exists = handle("18F-AV1451", av1451_id)

        rows.append({
            "PTID": ptid,
            "EXAMDATE": row["EXAMDATE"],
            "image_id(MRI)": mri_id,
            "mri_path": mri_path,
            "mri_exists": mri_exists,
            "image_id(18F-FDG)": fdg_id,
            "fdg_path": fdg_path,
            "fdg_exists": fdg_exists,
            "image_id(18F-AV45)": av45_id,
            "av45_path": av45_path,
            "av45_exists": av45_exists,
            "image_id(18F-AV1451)": av1451_id,
            "av1451_path": av1451_path,
            "av1451_exists": av1451_exists,
        })
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    df = load_matched(args.input)
    mask_valid = (df["image_id(MRI)"].ne("")) & df.apply(has_any_pet, axis=1)
    valid_df = df[mask_valid].copy()

    args.output_valid.parent.mkdir(parents=True, exist_ok=True)
    valid_df.to_csv(args.output_valid, index=False)

    annotated = annotate_files(valid_df, args.data_root)
    annotated.to_csv(args.output_files, index=False)

    missing_summary = {
        "mri_missing_files": (~annotated["mri_exists"]).sum(),
        "fdg_missing_files": (~annotated["fdg_exists"]).sum(),
        "av45_missing_files": (~annotated["av45_exists"]).sum(),
        "av1451_missing_files": (~annotated["av1451_exists"]).sum(),
    }
    print(f"Valid rows (MRI + >=1 PET): {len(valid_df)}")
    print("File existence summary:", missing_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
