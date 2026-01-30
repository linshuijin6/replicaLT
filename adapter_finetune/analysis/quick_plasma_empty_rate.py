#!/usr/bin/env python3
from pathlib import Path
import sys
THIS = Path(__file__).resolve()
ROOT = THIS.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from adapter_finetune.dataset import PETAdapterDataset
from transformers import AutoProcessor

pairs = Path("/mnt/nfsdata/nfsdata/lsj.14/replicaLT/adapter_finetune/data_csv/pairs0106_filtered.csv")
# 使用已整合的 pairs_withPlasma.csv
upenn = Path("/mnt/nfsdata/nfsdata/lsj.14/replicaLT/adapter_finetune/data_csv/pairs_withPlasma.csv")
mri = Path("/mnt/nfsdata/nfsdata/lsj.14/replicaLT/adapter_finetune/MRI_PET_IDs.csv")
model_path = "/mnt/nfsdata/nfsdata/lsj.14/replicaLT/BiomedCLIP"

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

ds = PETAdapterDataset(
    root_dir=Path("/mnt/nfsdata/nfsdata/ADNI/cached_npy"),
    link_csv=pairs,
    plasma_csv=upenn,
    mri_csv=mri,
    yaml_config=None,
    processor=processor,
    synthetic=False,
    synthetic_count=0,
    cache_root=Path("/mnt/nfsdata/nfsdata/ADNI/cached_npy"),
)

total = len(ds.samples)
empty = sum(1 for s in ds.samples if not s.get("plasma"))
print(f"Samples built: {total}, plasma empty: {empty}")
print("First 5 with empty plasma:")
shown = 0
for s in ds.samples:
    if not s.get("plasma"):
        print(s["ptid"], s.get("examdate"))
        shown += 1
        if shown >= 5:
            break
