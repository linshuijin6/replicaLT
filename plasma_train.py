"""
plasma_train.py
===============
基于 plasma_emb 引导的 MRI→PET 扩散模型训练脚本。

核心改动（相对于 train.py legacy 模式）：
  - Token 0: 预计算的 plasma_emb (512,) 来自 adapter_v2 CoCoOpTAUModel
  - Token 1: 模态优化文本 (512,) 保留不变
  - context shape 保持 (B, 2, 512)，扩散模型架构零修改

前置依赖：
  1. 运行 precompute_plasma_emb.py 生成 plasma_emb 缓存
  2. 已有 MRI/PET NIfTI 数据和 PersistentDataset 缓存

用法：
  python plasma_train.py
"""
#%%
# ============ GPU 设备锁定（必须在 import torch 之前设置） ============
import os
_TARGET_GPU_IDS = [5]  # 指定使用的 GPU ID
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, _TARGET_GPU_IDS))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import pandas as pd
import random
from report_error import email_on_error
from sklearn.model_selection import train_test_split
from typing import Dict
import nibabel as nib
import numpy as np
from generative.networks.nets.diffusion_model_unet import DistributedDiffusionModelUNet


def get_subject_id(filename):
    """从文件名中提取统一的 subject ID（前三个部分，如 '002_S_0295'）"""
    parts = filename.split('_')
    return f"{parts[0]}_{parts[1]}_{parts[2]}"


def mat_load(filepath):
    if filepath is None:
        return None
    return nib.load(filepath).get_fdata()


# ============ TensorBoard 图像记录辅助函数 ============
def get_3d_slices(volume, normalize=True):
    import torch
    if isinstance(volume, torch.Tensor):
        vol = volume.detach().cpu().numpy()
    else:
        vol = np.array(volume)
    while vol.ndim > 3:
        vol = vol[0]
    h, w, d = vol.shape
    slices = {
        'axial': vol[:, :, d // 2],
        'coronal': vol[:, w // 2, :],
        'sagittal': vol[h // 2, :, :],
    }
    if normalize:
        for key in slices:
            s = slices[key]
            s_min, s_max = s.min(), s.max()
            if s_max - s_min > 1e-8:
                slices[key] = (s - s_min) / (s_max - s_min)
            else:
                slices[key] = np.zeros_like(s)
    return slices


def log_3d_volume_to_tensorboard(writer, tag_prefix, volume, global_step, cmap='gray'):
    import matplotlib.pyplot as plt
    from io import BytesIO
    from PIL import Image
    slices = get_3d_slices(volume)
    for view_name, slice_2d in slices.items():
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(slice_2d, cmap=cmap, vmin=0, vmax=1)
        ax.axis('off')
        ax.set_title(f"{view_name}")
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=100)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        img_array = np.array(img)
        if img_array.ndim == 3:
            img_array = img_array[:, :, :3]
            img_array = img_array.transpose(2, 0, 1)
        writer.add_image(f"{tag_prefix}/{view_name}", img_array, global_step)


def log_comparison_figure(writer, tag, volumes_dict, global_step):
    import matplotlib.pyplot as plt
    from io import BytesIO
    from PIL import Image
    n_vols = len(volumes_dict)
    if n_vols == 0:
        return
    fig, axes = plt.subplots(3, n_vols, figsize=(3 * n_vols, 9))
    if n_vols == 1:
        axes = axes.reshape(3, 1)
    view_names = ['axial', 'coronal', 'sagittal']
    for col_idx, (name, vol) in enumerate(volumes_dict.items()):
        if vol is None:
            for row_idx in range(3):
                axes[row_idx, col_idx].axis('off')
                axes[row_idx, col_idx].set_title(f"{name}\nN/A")
            continue
        slices = get_3d_slices(vol)
        cmap = 'jet' if 'Diff' in name else 'gray'
        for row_idx, view in enumerate(view_names):
            ax = axes[row_idx, col_idx]
            ax.imshow(slices[view], cmap=cmap, vmin=0, vmax=1)
            ax.axis('off')
            if row_idx == 0:
                ax.set_title(name, fontsize=10)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    if img_array.ndim == 3:
        img_array = img_array[:, :, :3]
        img_array = img_array.transpose(2, 0, 1)
    writer.add_image(tag, img_array, global_step)


@email_on_error()
def main():
    import torch
    import logging
    logging.basicConfig(level=logging.INFO)
    import torch.nn as nn
    from liutuo_utils import compare_3d_jet, compare_3d, donkey_noise_like
    from monai.data import PersistentDataset
    from transformers import AutoProcessor, AutoModel
    import json
    from tqdm import tqdm
    from torch.amp import GradScaler, autocast
    import torch.nn.functional as F
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    from pathlib import Path

    # ================================================================
    # 0. 训练参数设置
    # ================================================================
    base_dir = "/mnt/nfsdata/nfsdata/lsj.14/ADNI_CSF"
    # ★ 使用独立的缓存目录，避免与 train.py 缓存冲突（因 index_transform 不同）
    cache_dir = '/mnt/nfsdata/nfsdata/linshuijin/ADNI_CSF_plasma_cache192'
    plasma_csv_path = "adapter_finetune/ADNI_csv/UPENN_PLASMA_FUJIREBIO_QUANTERIX_21Dec2025.csv"

    # ★ plasma_emb 预计算缓存目录（由 precompute_plasma_emb.py 生成）
    plasma_emb_dir = "/mnt/nfsdata/nfsdata/linshuijin/plasma_emb_cache"

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "val"), exist_ok=True)
    print(f"✅ 缓存目录已创建: {cache_dir}")

    device_id = [0]
    clip_sample_min = 0
    clip_sample_max = 1
    alpha = 1.0  # FDG 损失权重
    beta = 1.0   # AV45 损失权重
    gamma = 1.0  # TAU 损失权重
    accumulation_steps = 1

    if torch.cuda.is_available():
        primary_device = device_id[0]
        torch.cuda.set_device(primary_device)
        print(f"✅ 设置主CUDA设备: {primary_device}")

    size_of_dataset = None
    bs = 1
    n_epochs = 200
    val_interval = 10
    checkpoint_dir = './checkpoint'
    logdir = './runs'
    log_every = 1
    image_log_interval = 10
    run_name = None
    last_epoch = False

    # GPU 信息
    if torch.cuda.is_available():
        print(f"\n📊 GPU配置信息:")
        for did in device_id:
            device_name = torch.cuda.get_device_name(did)
            free_mem = torch.cuda.get_device_properties(did).total_memory / 1024**3
            print(f"   GPU {did}: {device_name} - 总显存: {free_mem:.2f} GB")
        print(f"   使用模式: {'双GPU模型并行' if len(device_id) == 2 else '单GPU'}")

    # ============ TensorBoard 日志初始化 ============
    if run_name:
        final_run_name = run_name
    else:
        now = datetime.now()
        final_run_name = f"plasma_{now:%m.%d}_{os.getpid()}"
    run_dir = Path(logdir) / final_run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))
    print(f"\n📝 TensorBoard 日志目录: {run_dir}")

    # 保存超参数
    hparams = {
        "base_dir": base_dir,
        "cache_dir": cache_dir,
        "plasma_emb_dir": plasma_emb_dir,
        "device_id": device_id,
        "accumulation_steps": accumulation_steps,
        "n_epochs": n_epochs,
        "val_interval": val_interval,
        "alpha": alpha, "beta": beta, "gamma": gamma,
        "clip_sample_min": clip_sample_min,
        "clip_sample_max": clip_sample_max,
        "bs": bs,
        "log_every": log_every,
        "image_log_interval": image_log_interval,
        "guidance_mode": "plasma_emb",
    }
    hparam_path = run_dir / "hparams.json"
    with hparam_path.open("w", encoding="utf-8") as f:
        json.dump(hparams, f, ensure_ascii=False, indent=2)
    print(f"✅ 超参数已保存到: {hparam_path}")

    # ================================================================
    # 1. 加载 BiomedCLIP（仅用于编码模态优化文本 → Token 1）
    # ================================================================
    local_model_path = "./BiomedCLIP"
    processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
    bio_model = AutoModel.from_pretrained(local_model_path, trust_remote_code=True)

    primary_gpu = device_id[0]
    device = torch.device(f"cuda:{primary_gpu}" if torch.cuda.is_available() else "cpu")
    print(f"\n📍 将BiomedCLIP模型加载到设备: {device}")
    bio_model.to(device)
    bio_model.eval()
    print(f"✅ BiomedCLIP模型加载成功")

    # ================================================================
    # 2. 预计算模态优化文本特征（Token 1，与 train.py legacy 完全一致）
    # ================================================================
    modality_optimized_texts = {
        "FDG": (
            "FDG PET is a functional brain imaging technique that visualizes the dynamic changes "
            "in glucose metabolism, directly linked to neuronal energy demands and synaptic activity. "
            "It serves as a tool to assess functional connectivity and energy utilization across brain "
            "regions. Areas with decreased metabolic activity, such as those affected by neurodegenerative "
            "diseases, should exhibit reduced signal intensity. High-intensity metabolic hotspots in gray matter "
            "(e.g., the cerebral cortex and basal ganglia) are key markers of neuronal activity. "
        ),
        "AV45": (
            "AV45 PET is a molecular imaging technique that highlights the static distribution of "
            "amyloid-beta plaques, a critical pathological marker of Alzheimer's disease. "
            "This imaging modality provides a spatial map of amyloid deposition in cortical regions "
            "(e.g., the temporal, parietal, and frontal lobes) and can distinguish amyloid-positive areas "
            "from amyloid-negative white matter regions. The primary focus is on identifying amyloid "
            "deposition patterns to assess disease progression and pathological burden."
        ),
        "TAU": (
            "TAU PET is a molecular neuroimaging technique that visualizes the spatial distribution of "
            "aggregated tau protein, which reflects the presence of neurofibrillary tangles associated "
            "with neurodegeneration. Tau PET highlights region-specific tau accumulation, particularly "
            "in medial temporal, parietal, and association cortices, providing a topographical map of "
            "tau pathology that correlates with disease stage, cognitive decline, and neuronal dysfunction."
        ),
    }

    print("\n🔧 Plasma 引导模式: context = [plasma_emb, modality_text_optimized]")
    modality_optimized_features = {}
    for modality, text in modality_optimized_texts.items():
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            feat = bio_model.get_text_features(**inputs)
        modality_optimized_features[modality] = feat  # shape: (1, 512)
        print(f"   {modality} optimized feature: {feat.shape}")
    fdg_feature_optimized = modality_optimized_features["FDG"]
    av45_feature_optimized = modality_optimized_features["AV45"]
    tau_feature_optimized = modality_optimized_features["TAU"]

    # 释放 BiomedCLIP（不再需要）
    del bio_model, processor
    torch.cuda.empty_cache()
    print("✅ BiomedCLIP 已释放（仅用于编码模态文本）")

    # ================================================================
    # 3. 数据准备
    # ================================================================
    mri_dir = os.path.join(base_dir, "MRI")
    av45_dir = os.path.join(base_dir, "PET_MNI", 'AV45')
    fdg_dir = os.path.join(base_dir, "PET_MNI", 'FDG')
    tau_dir = os.path.join(base_dir, "PET_MNI", 'TAU')

    # 复用 train.py legacy 的 JSON 文件（保持训练/验证划分一致）
    train_json_path = "./train_data_with_description.json"
    val_json_path = "./val_data_with_description.json"

    def _filter_pet_files(files):
        keep = []
        for f in files:
            name_lower = f.lower()
            if not name_lower.endswith(".nii.gz"):
                continue
            if "pet2mni" in name_lower or "full" in name_lower:
                continue
            keep.append(f)
        return keep

    mri_files = sorted(os.listdir(mri_dir))
    av45_files = _filter_pet_files(sorted(os.listdir(av45_dir)))
    fdg_files = _filter_pet_files(sorted(os.listdir(fdg_dir)))
    tau_files = _filter_pet_files(sorted(os.listdir(tau_dir))) if os.path.isdir(tau_dir) else []

    mri_dict = {get_subject_id(f): os.path.join(mri_dir, f) for f in mri_files}
    av45_dict = {get_subject_id(f): os.path.join(av45_dir, f) for f in av45_files}
    fdg_dict = {get_subject_id(f): os.path.join(fdg_dir, f) for f in fdg_files}
    tau_dict = {get_subject_id(f): os.path.join(tau_dir, f) for f in tau_files}
    tau_available = len(tau_dict) > 0

    # ================================================================
    # 4. 加载 paired_data 并构建 plasma_emb 索引
    # ================================================================
    # 从已有 JSON 加载，保持训练/验证划分与 train.py 一致
    if os.path.exists(train_json_path):
        with open(train_json_path, "r") as f:
            train_data_json = json.load(f)
        with open(val_json_path, "r") as f:
            val_data_json = json.load(f)
        paired_data = train_data_json + val_data_json
        print(f"\n📂 从 JSON 加载了 {len(paired_data)} 条数据")
    else:
        # 从文件系统构建（与 train.py 逻辑一致）
        paired_data = []
        csv_path = 'filtered_subjects_with_description.csv'
        csv_data = pd.read_csv(csv_path)
        csv_dict = csv_data.set_index("Subject ID")["Description"].to_dict()
        for patient_id, mri_file in mri_dict.items():
            has_fdg = patient_id in fdg_dict
            has_av45 = patient_id in av45_dict
            has_tau = tau_available and patient_id in tau_dict
            if not (has_fdg or has_av45 or has_tau):
                continue
            entry = {
                "name": patient_id,
                "mri": mri_file,
                "av45": av45_dict.get(patient_id),
                "fdg": fdg_dict.get(patient_id),
            }
            if tau_available:
                entry["tau"] = tau_dict.get(patient_id)
            paired_data.append(entry)

    if size_of_dataset:
        paired_data = paired_data[:size_of_dataset]
    print(f"Total matched pairs: {len(paired_data)}")

    # 为每个样本分配索引
    for idx, data in enumerate(paired_data):
        data["fdg_index"] = idx
        data["av45_index"] = idx
        if tau_available:
            data["tau_index"] = idx

    # ================================================================
    # 5. ★ 加载预计算的 plasma_emb（Token 0）
    # ================================================================
    print(f"\n🧬 加载预计算的 plasma_emb: {plasma_emb_dir}")
    plasma_emb_list = []
    missing_plasma = 0
    for item in paired_data:
        ptid = item["name"]
        emb_path = os.path.join(plasma_emb_dir, f"{ptid}_plasma_emb.pt")
        if os.path.exists(emb_path):
            payload = torch.load(emb_path, map_location="cpu")
            plasma_emb_list.append(payload["plasma_emb"])  # (512,)
        else:
            # 无 plasma_emb 的样本：使用零向量（退化为无条件生成）
            plasma_emb_list.append(torch.zeros(512))
            missing_plasma += 1

    all_plasma_embs = torch.stack(plasma_emb_list, dim=0)  # (N, 512)
    print(f"   plasma_emb shape: {all_plasma_embs.shape}")
    print(f"   缺失 plasma_emb 的样本: {missing_plasma}/{len(paired_data)}")
    if missing_plasma > 0:
        print(f"   ⚠️  缺失样本将使用零向量引导（请先运行 precompute_plasma_emb.py）")

    # 验证 plasma_emb 的 L2 范数（应接近 1.0）
    norms = all_plasma_embs.norm(dim=-1)
    valid_norms = norms[norms > 0]
    if len(valid_norms) > 0:
        print(f"   plasma_emb L2 范数: mean={valid_norms.mean():.4f}, std={valid_norms.std():.4f}")

    # ================================================================
    # 6. 定义 index_transform（Token 0: plasma_emb, Token 1: modality_text）
    # ================================================================
    from monai.data import CacheDataset, DataLoader
    import monai.transforms as mt

    class FillMissingPET(mt.MapTransform):
        def __init__(self, keys, ref_key="mri"):
            super().__init__(keys)
            self.ref_key = ref_key
        def __call__(self, data):
            d = dict(data)
            ref = d.get(self.ref_key)
            if ref is None:
                return d
            for key in self.keys:
                if key == self.ref_key:
                    continue
                if key not in d or d[key] is None:
                    d[key] = np.zeros_like(ref)
            return d

    class ReduceTo3D(mt.MapTransform):
        def __init__(self, keys, reduce='mean'):
            super().__init__(keys)
            self.reduce = reduce
        def __call__(self, data):
            d = dict(data)
            for k in self.keys:
                v = d.get(k)
                if isinstance(v, np.ndarray) and v.ndim == 4:
                    if self.reduce == 'mean':
                        d[k] = v.mean(axis=-1)
                    elif self.reduce == 'max':
                        d[k] = v.max(axis=-1)
                    elif self.reduce == 'mid':
                        d[k] = v[..., v.shape[-1] // 2]
                    else:
                        d[k] = v.mean(axis=-1)
                elif hasattr(v, 'ndim') and v.ndim == 4:
                    if self.reduce == 'mean':
                        d[k] = v.mean(dim=-1)
                    elif self.reduce == 'max':
                        d[k] = torch.max(v, dim=-1).values
                    elif self.reduce == 'mid':
                        d[k] = v[..., v.shape[-1] // 2]
                    else:
                        d[k] = v.mean(dim=-1)
            return d

    # ★ 核心改动：Token 0 是 plasma_emb，Token 1 是模态优化文本
    all_plasma_embs_cpu = all_plasma_embs.cpu()  # (N, 512)
    fdg_feature_optimized_cpu = fdg_feature_optimized.cpu()  # (1, 512)
    av45_feature_optimized_cpu = av45_feature_optimized.cpu()
    tau_feature_optimized_cpu = tau_feature_optimized.cpu()

    def fdg_index_transform(x):
        """context = [plasma_emb, FDG_modality_text] → (2, 512)"""
        return torch.cat([all_plasma_embs_cpu[x].unsqueeze(0), fdg_feature_optimized_cpu], dim=0)

    def av45_index_transform(x):
        """context = [plasma_emb, AV45_modality_text] → (2, 512)"""
        return torch.cat([all_plasma_embs_cpu[x].unsqueeze(0), av45_feature_optimized_cpu], dim=0)

    def tau_index_transform(x):
        """context = [plasma_emb, TAU_modality_text] → (2, 512)"""
        return torch.cat([all_plasma_embs_cpu[x].unsqueeze(0), tau_feature_optimized_cpu], dim=0)

    # ================================================================
    # 7. 加载 JSON 数据并构建 DataLoader
    # ================================================================
    with open(train_json_path, "r") as f:
        train_data = json.load(f)
    with open(val_json_path, "r") as f:
        val_data = json.load(f)
    print(f"Loaded {len(train_data)} training, {len(val_data)} validation samples.")

    train_data = [
        {
            "name": item["name"],
            "mri": item["mri"],
            "av45": item["av45"],
            "fdg": item["fdg"],
            "tau": item.get("tau") or (tau_dict.get(item["name"]) if tau_available else None),
            "fdg_index": item.get("fdg_index") if item.get("fdg_index") is not None else idx,
            "av45_index": item.get("av45_index") if item.get("av45_index") is not None else idx,
            "tau_index": (item.get("tau_index") if item.get("tau_index") is not None else idx) if tau_available else None,
        }
        for idx, item in enumerate(train_data)
    ]
    val_data = [
        {
            "name": item["name"],
            "mri": item["mri"],
            "av45": item["av45"],
            "fdg": item["fdg"],
            "tau": item.get("tau") or (tau_dict.get(item["name"]) if tau_available else None),
            "fdg_index": item.get("fdg_index") if item.get("fdg_index") is not None else idx,
            "av45_index": item.get("av45_index") if item.get("av45_index") is not None else idx,
            "tau_index": (item.get("tau_index") if item.get("tau_index") is not None else idx) if tau_available else None,
        }
        for idx, item in enumerate(val_data)
    ]

    if tau_available:
        for item in train_data:
            if item["tau_index"] is None:
                item["tau_index"] = item["fdg_index"]
        for item in val_data:
            if item["tau_index"] is None:
                item["tau_index"] = item["fdg_index"]

    pet_keys = ["mri", "av45", "fdg"] + (["tau"] if tau_available else [])

    train_transforms = mt.Compose([
        mt.Lambdad(keys=pet_keys, func=mat_load),
        ReduceTo3D(keys=pet_keys, reduce='mean'),
        FillMissingPET(keys=pet_keys, ref_key="mri"),
        mt.CropForegroundd(keys=pet_keys, source_key="mri"),
        mt.EnsureChannelFirstd(keys=pet_keys, channel_dim='no_channel'),
        mt.HistogramNormalized(keys=["mri"]),
        mt.ResizeWithPadOrCropd(keys=pet_keys, spatial_size=[160, 192, 160]),
        mt.Spacingd(keys=pet_keys, pixdim=(1.0, 1.0, 1.0)),
        mt.NormalizeIntensityd(keys=pet_keys),
        mt.ScaleIntensityd(keys=pet_keys),
        mt.Lambdad(keys=["fdg_index"], func=fdg_index_transform),
        mt.Lambdad(keys=["av45_index"], func=av45_index_transform),
        mt.Lambdad(keys=["tau_index"], func=tau_index_transform) if tau_available else mt.Identity(),
    ])

    val_transforms = mt.Compose([
        mt.Lambdad(keys=pet_keys, func=mat_load),
        ReduceTo3D(keys=pet_keys, reduce='mean'),
        FillMissingPET(keys=pet_keys, ref_key="mri"),
        mt.CropForegroundd(keys=pet_keys, source_key="mri"),
        mt.EnsureChannelFirstd(keys=pet_keys, channel_dim='no_channel'),
        mt.HistogramNormalized(keys=["mri"]),
        mt.ResizeWithPadOrCropd(keys=pet_keys, spatial_size=[160, 192, 160]),
        mt.Spacingd(keys=pet_keys, pixdim=(1.0, 1.0, 1.0)),
        mt.NormalizeIntensityd(keys=pet_keys),
        mt.ScaleIntensityd(keys=pet_keys),
        mt.Lambdad(keys=["fdg_index"], func=fdg_index_transform),
        mt.Lambdad(keys=["av45_index"], func=av45_index_transform),
        mt.Lambdad(keys=["tau_index"], func=tau_index_transform) if tau_available else mt.Identity(),
    ])

    train_cache_dir = os.path.join(cache_dir, "train")
    val_cache_dir = os.path.join(cache_dir, "val")

    print(f"\n📦 创建训练集 PersistentDataset...")
    print(f"   数据量: {len(train_data)} 样本")
    print(f"   缓存目录: {train_cache_dir}")
    train_ds = PersistentDataset(data=train_data, transform=train_transforms, cache_dir=train_cache_dir)

    cache_files = os.listdir(train_cache_dir)
    print(f"   已有缓存文件数: {len(cache_files)}")
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

    print(f"\n📦 创建验证集 PersistentDataset...")
    print(f"   数据量: {len(val_data)} 样本")
    print(f"   缓存目录: {val_cache_dir}")
    val_ds = PersistentDataset(data=val_data, transform=val_transforms, cache_dir=val_cache_dir)

    cache_files = os.listdir(val_cache_dir)
    print(f"   已有缓存文件数: {len(cache_files)}")
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    # ================================================================
    # 8. 加载扩散模型（与 train.py 完全一致）
    # ================================================================
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from generative.networks.nets import DiffusionModelUNet
    from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
    from generative.inferers import DiffusionInferer

    if len(device_id) == 1:
        model = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            num_channels=(32, 64, 64, 128),
            attention_levels=(False, False, False, True),
            num_res_blocks=1,
            num_head_channels=(0, 0, 0, 128),
            with_conditioning=True,
            cross_attention_dim=512,
            use_flash_attention=True,
        )
        model.to(device)
    elif len(device_id) == 2:
        model = DistributedDiffusionModelUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            num_channels=(32, 64, 64, 128),
            attention_levels=(False, False, False, True),
            num_res_blocks=1,
            num_head_channels=(0, 0, 0, 128),
            with_conditioning=True,
            cross_attention_dim=512,
            use_flash_attention=True,
            device_ids=device_id,
        )
        print(f"✅ 双GPU模式: 模型分布在 GPU {device_id[0]} 和 GPU {device_id[1]}")
    else:
        raise ValueError("Currently only support 1 or 2 GPU(s) for training.")

    use_distributed = len(device_id) == 2

    optimizer = torch.optim.Adam(params=model.parameters(), lr=2.5e-5)
    start_epoch = 0

    if last_epoch:
        checkpoint = torch.load(f'{checkpoint_dir}/first_part_{last_epoch}.pth', map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        val_interval = 5

    scheduler = DDPMScheduler(
        prediction_type="v_prediction", num_train_timesteps=1000,
        schedule="scaled_linear_beta", beta_start=0.0005, beta_end=0.0195,
    )
    scheduler.set_timesteps(num_inference_steps=1000)
    epoch_loss_list = []
    val_epoch_loss_list = []
    scaler = GradScaler(device=device)
    global_step = 0

    # ================================================================
    # 9. 训练循环（与 train.py 完全一致，context 已通过 index_transform 替换）
    # ================================================================
    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, data in progress_bar:
            images = data["mri"].to(device, non_blocking=True)
            seg_fdg = data["fdg"].to(device, non_blocking=True)
            seg_av45 = data["av45"].to(device, non_blocking=True)
            fdg_index = data["fdg_index"].to(device, non_blocking=True)
            av45_index = data["av45_index"].to(device, non_blocking=True)
            if tau_available:
                seg_tau = data["tau"].to(device, non_blocking=True)
                tau_index = data["tau_index"].to(device, non_blocking=True)
            else:
                seg_tau = None
                tau_index = None

            if step % accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            time_embedding = torch.randint(
                0, 1000, (images.shape[0],), device=images.device
            ).long()

            if epoch >= 140:
                time_embedding = torch.tensor([0], device=images.device, dtype=torch.long)

            with torch.no_grad():
                t = (time_embedding.float() / 1000).view(-1, 1, 1, 1, 1)

            has_fdg = not torch.all(seg_fdg == 0)
            has_av45 = not torch.all(seg_av45 == 0)
            has_tau = tau_available and seg_tau is not None and not torch.all(seg_tau == 0)

            total_loss = 0.0

            available_modalities = []
            if has_fdg:
                available_modalities.append('fdg')
            if has_av45:
                available_modalities.append('av45')
            if has_tau:
                available_modalities.append('tau')

            if len(available_modalities) > 0:
                selected_modality = random.choice(available_modalities)

                if selected_modality == 'fdg':
                    with autocast(device_type='cuda', enabled=True):
                        x_t_fdg = t * seg_fdg + (1 - t) * images
                        v_fdg_prediction = model(x=x_t_fdg, timesteps=time_embedding, context=fdg_index)
                        if use_distributed:
                            v_fdg_prediction = v_fdg_prediction.to(device)
                        v_fdg = seg_fdg - images
                        loss_fdg = F.mse_loss(v_fdg.float(), v_fdg_prediction.float())
                        loss_fdg = (alpha * loss_fdg) / accumulation_steps
                    scaler.scale(loss_fdg).backward()
                    total_loss = loss_fdg.item() * accumulation_steps
                    del x_t_fdg, v_fdg_prediction, v_fdg, loss_fdg

                elif selected_modality == 'av45':
                    with autocast(device_type='cuda', enabled=True):
                        x_t_av45 = t * seg_av45 + (1 - t) * images
                        v_av45_prediction = model(x=x_t_av45, timesteps=time_embedding, context=av45_index)
                        if use_distributed:
                            v_av45_prediction = v_av45_prediction.to(device)
                        v_av45 = seg_av45 - images
                        loss_av45 = F.mse_loss(v_av45.float(), v_av45_prediction.float())
                        loss_av45 = (beta * loss_av45) / accumulation_steps
                    scaler.scale(loss_av45).backward()
                    total_loss = loss_av45.item() * accumulation_steps
                    del x_t_av45, v_av45_prediction, v_av45, loss_av45

                elif selected_modality == 'tau':
                    with autocast(device_type='cuda', enabled=True):
                        x_t_tau = t * seg_tau + (1 - t) * images
                        v_tau_prediction = model(x=x_t_tau, timesteps=time_embedding, context=tau_index)
                        if use_distributed:
                            v_tau_prediction = v_tau_prediction.to(device)
                        v_tau = seg_tau - images
                        loss_tau = F.mse_loss(v_tau.float(), v_tau_prediction.float())
                        loss_tau = (gamma * loss_tau) / accumulation_steps
                    scaler.scale(loss_tau).backward()
                    total_loss = loss_tau.item() * accumulation_steps
                    del x_t_tau, v_tau_prediction, v_tau, loss_tau

            if (step + 1) % accumulation_steps == 0 and total_loss > 0:
                scaler.step(optimizer)
                scaler.update()

            epoch_loss += total_loss
            global_step += 1

            if log_every > 0 and global_step % log_every == 0:
                avg_loss = epoch_loss / (step + 1)
                lr = optimizer.param_groups[0].get("lr", 2.5e-5)
                writer.add_scalar("train/loss", avg_loss, global_step)
                writer.add_scalar("train/step_loss", total_loss, global_step)
                writer.add_scalar("optim/lr", lr, global_step)
                writer.add_scalar("train/epoch", epoch, global_step)
                if len(available_modalities) > 0:
                    writer.add_scalar(f"train/modality_{selected_modality}", 1.0, global_step)

            progress_bar.set_postfix({
                "loss": epoch_loss / (step + 1),
                "accum": f"{(step % accumulation_steps) + 1}/{accumulation_steps}"
            })

        epoch_avg_loss = epoch_loss / (step + 1)
        epoch_loss_list.append(epoch_avg_loss)
        writer.add_scalar("train/epoch_loss", epoch_avg_loss, epoch)

        # ============ 验证 ============
        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_epoch_loss = 0

            for step, data_val in enumerate(val_loader):
                images = data_val["mri"].to(device, non_blocking=True)
                seg_fdg = data_val["fdg"].to(device, non_blocking=True)
                seg_av45 = data_val["av45"].to(device, non_blocking=True)
                fdg_index = data_val["fdg_index"].to(device, non_blocking=True)
                av45_index = data_val["av45_index"].to(device, non_blocking=True)
                if tau_available:
                    seg_tau = data_val["tau"].to(device, non_blocking=True)
                    tau_index = data_val["tau_index"].to(device, non_blocking=True)
                else:
                    seg_tau = None
                    tau_index = None

                has_fdg = not torch.all(seg_fdg == 0)
                has_av45 = not torch.all(seg_av45 == 0)
                has_tau = tau_available and seg_tau is not None and not torch.all(seg_tau == 0)

                x_t = images
                N_sample = 1
                N_sample_tensor = torch.tensor(N_sample, dtype=torch.float32, device=device)

                progress_bar_val = [(i / N_sample) for i in range(N_sample)]
                for t in progress_bar_val:
                    with autocast(device_type='cuda', enabled=False):
                        with torch.no_grad():
                            time_embedding = int(t * 1000)

                            if has_fdg:
                                v_fdg_output = model(x=x_t, timesteps=torch.Tensor((time_embedding,)).to(device), context=fdg_index)
                                if use_distributed:
                                    v_fdg_output = v_fdg_output.to(device)
                                x_fdg_t = x_t + (v_fdg_output / N_sample_tensor)
                                x_fdg_t = torch.clamp(x_fdg_t, min=clip_sample_min, max=clip_sample_max)
                            else:
                                x_fdg_t = None

                            if has_av45:
                                v_av45_output = model(x=x_t, timesteps=torch.Tensor((time_embedding,)).to(device), context=av45_index)
                                if use_distributed:
                                    v_av45_output = v_av45_output.to(device)
                                x_av45_t = x_t + (v_av45_output / N_sample_tensor)
                                x_av45_t = torch.clamp(x_av45_t, min=clip_sample_min, max=clip_sample_max)
                            else:
                                x_av45_t = None

                            if has_tau:
                                v_tau_output = model(x=x_t, timesteps=torch.Tensor((time_embedding,)).to(device), context=tau_index)
                                if use_distributed:
                                    v_tau_output = v_tau_output.to(device)
                                x_tau_t = x_t + (v_tau_output / N_sample_tensor)
                                x_tau_t = torch.clamp(x_tau_t, min=clip_sample_min, max=clip_sample_max)
                            else:
                                x_tau_t = None

                val_fdg_loss = torch.tensor(0.0, device=device)
                val_av45_loss = torch.tensor(0.0, device=device)
                val_tau_loss = torch.tensor(0.0, device=device)

                if has_fdg and x_fdg_t is not None:
                    val_fdg_loss = F.mse_loss(x_fdg_t.float(), seg_fdg.float())
                if has_av45 and x_av45_t is not None:
                    val_av45_loss = F.mse_loss(x_av45_t.float(), seg_av45.float())
                if has_tau and x_tau_t is not None:
                    val_tau_loss = F.mse_loss(x_tau_t.float(), seg_tau.float())

                val_loss = alpha * val_fdg_loss + beta * val_av45_loss + gamma * val_tau_loss
                val_epoch_loss += val_loss.item()

            print("Epoch", epoch + 1, "Validation loss", val_epoch_loss / (step + 1))
            val_epoch_loss_list.append(val_epoch_loss / (step + 1))

            val_avg_loss = val_epoch_loss / (step + 1)
            writer.add_scalar("val/loss", val_avg_loss, epoch)

            print('epoch:', epoch + 1)
            print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])

            # 保存 checkpoint
            ckpt_path = run_dir / f"ckpt_epoch{epoch + 1}.pt"
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
            }
            torch.save(checkpoint, str(ckpt_path))
            print(f"✅ Checkpoint saved: {ckpt_path}")

            torch.save(checkpoint, f'{checkpoint_dir}/first_part_{epoch + 1}.pth')
            print('Saved all parameters!\n')

            # 可视化
            current_fdg_img = x_fdg_t.to('cpu') if x_fdg_t is not None else None
            current_av45_img = x_av45_t.to('cpu') if x_av45_t is not None else None
            current_tau_img = x_tau_t.to('cpu') if tau_available and x_tau_t is not None else None
            labels_fdg = seg_fdg.to('cpu')
            labels_av45 = seg_av45.to('cpu')
            labels_tau = seg_tau.to('cpu') if tau_available else None
            mri_cpu = images.to('cpu')

            if current_fdg_img is not None or current_av45_img is not None or current_tau_img is not None:
                gray_imgs = [mri_cpu]
                jet_diffs = []
                if current_fdg_img is not None:
                    gray_imgs.extend([labels_fdg, current_fdg_img])
                    jet_diffs.append(current_fdg_img - labels_fdg)
                if current_av45_img is not None:
                    gray_imgs.extend([labels_av45, current_av45_img])
                    jet_diffs.append(current_av45_img - labels_av45)
                if current_tau_img is not None and labels_tau is not None:
                    gray_imgs.extend([labels_tau, current_tau_img])
                    jet_diffs.append(current_tau_img - labels_tau)
                if len(gray_imgs) > 1:
                    compare_3d(gray_imgs)
                if len(jet_diffs) > 0:
                    compare_3d_jet(jet_diffs)

            # TensorBoard 图像
            if (epoch + 1) % image_log_interval == 0:
                print(f"📷 保存过程图像到 TensorBoard (epoch {epoch + 1})...")
                comparison_volumes = {"MRI": mri_cpu}
                if has_fdg and current_fdg_img is not None:
                    comparison_volumes["FDG_GT"] = labels_fdg
                    comparison_volumes["FDG_Pred"] = current_fdg_img
                    comparison_volumes["FDG_Diff"] = torch.abs(current_fdg_img - labels_fdg)
                if has_av45 and current_av45_img is not None:
                    comparison_volumes["AV45_GT"] = labels_av45
                    comparison_volumes["AV45_Pred"] = current_av45_img
                    comparison_volumes["AV45_Diff"] = torch.abs(current_av45_img - labels_av45)
                if tau_available and has_tau and current_tau_img is not None:
                    comparison_volumes["TAU_GT"] = labels_tau
                    comparison_volumes["TAU_Pred"] = current_tau_img
                    comparison_volumes["TAU_Diff"] = torch.abs(current_tau_img - labels_tau)

                log_comparison_figure(writer, f"val/comparison_epoch{epoch+1}", comparison_volumes, epoch)
                log_3d_volume_to_tensorboard(writer, "val/MRI", mri_cpu, epoch)
                if has_fdg and current_fdg_img is not None:
                    log_3d_volume_to_tensorboard(writer, "val/FDG_GT", labels_fdg, epoch)
                    log_3d_volume_to_tensorboard(writer, "val/FDG_Pred", current_fdg_img, epoch)
                if has_av45 and current_av45_img is not None:
                    log_3d_volume_to_tensorboard(writer, "val/AV45_GT", labels_av45, epoch)
                    log_3d_volume_to_tensorboard(writer, "val/AV45_Pred", current_av45_img, epoch)
                if tau_available and has_tau and current_tau_img is not None:
                    log_3d_volume_to_tensorboard(writer, "val/TAU_GT", labels_tau, epoch)
                    log_3d_volume_to_tensorboard(writer, "val/TAU_Pred", current_tau_img, epoch)
                print(f"✅ 图像已保存到 TensorBoard")

            # SSIM / PSNR 指标
            from generative.metrics import SSIMMetric
            from monai.metrics import MAEMetric, MSEMetric, PSNRMetric
            ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=7)
            psnr_metric = PSNRMetric(1.0)

            if has_fdg and current_fdg_img is not None:
                ssim_fdg_value = ssim_metric(labels_fdg, current_fdg_img)
                psnr_fdg_value = psnr_metric(labels_fdg, current_fdg_img)
                print(f"FDG SSIM: {ssim_fdg_value.mean().item()}")
                print(f"FDG PSNR: {psnr_fdg_value.mean().item()}")
                writer.add_scalar("val/FDG_SSIM", ssim_fdg_value.mean().item(), epoch)
                writer.add_scalar("val/FDG_PSNR", psnr_fdg_value.mean().item(), epoch)

            if has_av45 and current_av45_img is not None:
                ssim_av45_value = ssim_metric(labels_av45, current_av45_img)
                psnr_av45_value = psnr_metric(labels_av45, current_av45_img)
                print(f"AV45 SSIM: {ssim_av45_value.mean().item()}")
                print(f"AV45 PSNR: {psnr_av45_value.mean().item()}")
                writer.add_scalar("val/AV45_SSIM", ssim_av45_value.mean().item(), epoch)
                writer.add_scalar("val/AV45_PSNR", psnr_av45_value.mean().item(), epoch)

            if tau_available and has_tau and current_tau_img is not None and labels_tau is not None:
                ssim_tau_value = ssim_metric(labels_tau, current_tau_img)
                psnr_tau_value = psnr_metric(labels_tau, current_tau_img)
                print(f"TAU SSIM: {ssim_tau_value.mean().item()}")
                print(f"TAU PSNR: {psnr_tau_value.mean().item()}")
                writer.add_scalar("val/TAU_SSIM", ssim_tau_value.mean().item(), epoch)
                writer.add_scalar("val/TAU_PSNR", psnr_tau_value.mean().item(), epoch)

    writer.flush()
    writer.close()
    print(f"\n🎉 训练完成！日志保存在: {run_dir}")


if __name__ == "__main__":
    main()
