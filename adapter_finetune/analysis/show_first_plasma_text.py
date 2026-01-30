#!/usr/bin/env python3
import sys
from pathlib import Path
THIS = Path(__file__).resolve()
ROOT = THIS.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from adapter_finetune.dataset import PETAdapterDataset, build_plasma_text, load_yaml_config
from transformers import AutoProcessor

pairs = Path("/mnt/nfsdata/nfsdata/lsj.14/replicaLT/adapter_finetune/data_csv/pairs0106_filtered.csv")
plasma = Path("/mnt/nfsdata/nfsdata/lsj.14/replicaLT/adapter_finetune/data_csv/pairs_withPlasma.csv")
mri = Path("/mnt/nfsdata/nfsdata/lsj.14/replicaLT/adapter_finetune/MRI_PET_IDs.csv")
model_path = "/mnt/nfsdata/nfsdata/lsj.14/replicaLT/BiomedCLIP"

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

ds = PETAdapterDataset(
    root_dir=Path("/mnt/nfsdata/nfsdata/ADNI/cached_npy"),
    link_csv=pairs,
    plasma_csv=plasma,
    mri_csv=mri,
    yaml_config=None,
    processor=processor,
    synthetic=False,
    synthetic_count=0,
    cache_root=Path("/mnt/nfsdata/nfsdata/ADNI/cached_npy"),
)

cfg = load_yaml_config(None)
plasma_row = ds.samples[0]["plasma"]
text = build_plasma_text(plasma_row, cfg.get("plasma_thresholds", {}))
print("First sample PTID:", ds.samples[0]["ptid"])
print("Plasma text:\n" + text)
