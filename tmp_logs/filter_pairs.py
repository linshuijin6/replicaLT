import pandas as pd
from pathlib import Path
src = Path("adapter_finetune/data_csv/pairs0106.csv")
dst = Path("adapter_finetune/data_csv/pairs0106_filtered.csv")
print("src exists", src.exists())
df = pd.read_csv(src)
for col in ["id_mri","id_fdg","id_av45","id_av1451"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()
mask_mri = df["id_mri"].notna() & (df["id_mri"].str.len() > 0) & (df["id_mri"].str.lower() != "nan")
mask_pet = False
for col in ["id_fdg","id_av45","id_av1451"]:
    mask_pet = mask_pet | (df[col].notna() & (df[col].str.len() > 0) & (df[col].str.lower() != "nan"))
filtered = df[mask_mri & mask_pet].copy()
dst.parent.mkdir(parents=True, exist_ok=True)
filtered.to_csv(dst, index=False)
print("saved rows", len(filtered), "to", dst)
