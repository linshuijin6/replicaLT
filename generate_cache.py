#!/usr/bin/env python3
"""
专门用于生成数据集缓存的脚本（与 train.py 同步）
使用方法: python generate_cache.py
或: CUDA_VISIBLE_DEVICES=5 python generate_cache.py

注意：此脚本生成的缓存与 train.py 使用的缓存完全兼容
"""
import os
import sys
import json
import torch
import torch.nn as nn
import pandas as pd
import nibabel as nib
import numpy as np
from tqdm import tqdm
from typing import Dict
from monai.data import PersistentDataset
import monai.transforms as mt


def get_subject_id(filename):
    """从文件名中提取统一的 subject ID（前三个部分，如 '002_S_0295'）"""
    parts = filename.split('_')
    return f"{parts[0]}_{parts[1]}_{parts[2]}"


def mat_load(filepath):
    """使用 nibabel 加载 NIfTI 文件，并转换为 NumPy 数组。"""
    if filepath is None:
        return None
    return nib.load(filepath).get_fdata()


class FillMissingPET(mt.MapTransform):
    """为缺失的 PET 模态填充与 MRI 尺寸一致的零数组。"""
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


class RemoveCropForegroundMeta(mt.Transform):
    """移除 CropForegroundd 添加的元数据键，避免 collate 时类型不匹配。"""
    def __call__(self, data):
        d = dict(data)
        # 移除可能导致 collate 问题的元数据键
        keys_to_remove = ['foreground_start_coord', 'foreground_end_coord']
        for k in keys_to_remove:
            d.pop(k, None)
        return d


class ConvertToTensor(mt.Transform):
    """
    将所有 MetaTensor 转换为普通 torch.Tensor，避免 batch_size > 1 时 collate 报错。
    MONAI 的 MetaTensor 在 collate 时会检查 .meta 属性，混合类型会导致错误。
    """
    def __call__(self, data):
        from monai.data import MetaTensor
        d = dict(data)
        for k, v in d.items():
            if isinstance(v, MetaTensor):
                # 转换为普通 Tensor，丢弃 meta 信息
                d[k] = v.as_tensor()
            elif isinstance(v, torch.Tensor):
                # 确保是普通 Tensor
                d[k] = v.clone().detach()
        return d


class ReduceTo3D(mt.MapTransform):
    """将可能的 4D 体积 (H, W, D, T) 沿最后一维聚合为 3D (H, W, D)。"""
    def __init__(self, keys, reduce='mean'):
        super().__init__(keys)
        self.reduce = reduce
        
    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            v = d.get(k)
            if isinstance(v, np.ndarray):
                if v.ndim == 4:
                    if self.reduce == 'mean':
                        d[k] = v.mean(axis=-1)
                    elif self.reduce == 'max':
                        d[k] = v.max(axis=-1)
                    elif self.reduce == 'mid':
                        mid = v.shape[-1] // 2
                        d[k] = v[..., mid]
                    else:
                        d[k] = v.mean(axis=-1)
            elif hasattr(v, 'ndim') and v.ndim == 4:
                # torch.Tensor case
                if self.reduce == 'mean':
                    d[k] = v.mean(dim=-1)
                elif self.reduce == 'max':
                    d[k] = torch.max(v, dim=-1).values
                elif self.reduce == 'mid':
                    mid = v.shape[-1] // 2
                    d[k] = v[..., mid]
                else:
                    d[k] = v.mean(dim=-1)
        return d


class Adapter(nn.Module):
    """Adapter 模块（与 train.py 相同）"""
    def __init__(self, dim: int, hidden: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main():
    print("=" * 80)
    print("数据集缓存生成工具 (与 train.py 同步)")
    print("=" * 80)
    
    # ============ 配置参数（与 train.py 保持一致）============
    base_dir = "/mnt/nfsdata/nfsdata/ADNI/ADNI0103/Coregistration"
    cache_dir = '/mnt/nfsdata/nfsdata/lsj.14/ADNI_cache_v2'  # 与 train.py 相同
    plasma_csv_path = "adapter_finetune/UPENN_PLASMA_FUJIREBIO_QUANTERIX_21Dec2025.csv"
    train_json_path = "adapter_finetune/json/train_data_with_description.json"
    val_json_path = "adapter_finetune/json/val_data_with_description.json"
    csv_path = 'filtered_subjects_with_description.csv'
    
    # BiomedCLIP 和 Adapter 配置
    local_model_path = "./BiomedCLIP"
    adapter_ckpt_path = "/mnt/nfsdata/nfsdata/lsj.14/replicaLT/adapter_finetune/runs/01.19_2945992/ckpt_epoch400.pt"
    adapter_hidden = 512
    adapter_dropout = 0.1
    
    # GPU 配置
    device_id = 5  # 默认使用 GPU 5，可通过环境变量修改
    
    # 确保缓存目录存在
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "val"), exist_ok=True)
    print(f"✅ 缓存目录: {cache_dir}")
    
    # 检查CUDA
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        print(f"✅ CUDA可用，使用 GPU {device_id}")
        print(f"   GPU名称: {torch.cuda.get_device_name(device_id)}")
        device = torch.device(f"cuda:{device_id}")
    else:
        print("⚠️  CUDA不可用，将使用CPU（会很慢）")
        device = torch.device("cpu")
    
    # ============ 加载 BiomedCLIP 模型（与 train.py 相同）============
    print(f"\n📍 加载 BiomedCLIP 模型...")
    from transformers import AutoProcessor, AutoModel
    
    processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
    bio_model = AutoModel.from_pretrained(local_model_path, trust_remote_code=True)
    bio_model.to(device)
    bio_model.eval()
    print(f"✅ BiomedCLIP 模型加载成功")
    
    # 模态公共描述文本（与 train.py 保持一致）
    modality_common_texts = {
        "FDG": (
            "a functional brain imaging technique that visualizes the dynamic changes in glucose metabolism, "
            "directly linked to neuronal energy demands and synaptic activity. Areas with decreased metabolic activity "
            "exhibit reduced signal intensity. High-intensity metabolic hotspots in gray matter are key markers of neuronal activity."
        ),
        "AV45": (
            "a molecular imaging technique that highlights the static distribution of amyloid-beta plaques, "
            "a critical pathological marker of Alzheimer's disease. This imaging modality provides a spatial map of amyloid deposition in cortical regions and "
            "can distinguish amyloid-positive areas from amyloid-negative white matter regions. The primary focus is on identifying amyloid deposition patterns to assess disease progression and pathological burden."
        ),
        "TAU": (
            "a molecular neuroimaging technique that visualizes the spatial distribution of aggregated tau protein, "
            "which reflects the presence of neurofibrillary tangles associated with neurodegeneration. Tau PET highlights region-specific tau accumulation, particularly in medial temporal, parietal, and association cortices, "
            "providing a topographical map of tau pathology that correlates with disease stage, cognitive decline, and neuronal dysfunction. "
            "This modality emphasizes the progression and regional spread of tau pathology rather than metabolic activity or amyloid burden."
        ),
    }
    
    feat_dim = bio_model.config.projection_dim
    modalities_all = ["FDG", "AV45", "TAU"]
    adapters = {m: Adapter(feat_dim, hidden=adapter_hidden, dropout=adapter_dropout).to(device) for m in modalities_all}
    
    # 加载 Adapter 权重
    if adapter_ckpt_path and os.path.exists(adapter_ckpt_path):
        ckpt = torch.load(adapter_ckpt_path, map_location=device, weights_only=False)
        ckpt_adapters = ckpt.get("text_adapters", {})
        for m, adp in adapters.items():
            if m in ckpt_adapters:
                adp.load_state_dict(ckpt_adapters[m])
        print(f"✅ 加载 Adapter 权重: {adapter_ckpt_path}")
    else:
        print(f"⚠️  Adapter 权重未找到: {adapter_ckpt_path}")
    
    for adp in adapters.values():
        adp.eval()
    
    # ============ 加载 Plasma 数据（与 train.py 相同）============
    from adapter_finetune.dataset import build_plasma_text as build_plasma_text_ds, _default_config
    
    def _load_plasma_table(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        if "examdate" in df.columns:
            df["examdate"] = pd.to_datetime(df["examdate"], errors="coerce")
        return df
    
    plasma_table = _load_plasma_table(plasma_csv_path)
    
    def _pick_plasma(ptid: str, examdate: str = None) -> Dict[str, float]:
        if ptid is None:
            return {}
        df = plasma_table
        ptid_col = "ptid"
        exam_col = "examdate"
        if exam_col in df.columns:
            examdate_ts = pd.to_datetime(examdate) if examdate else None
            hit = df[df[ptid_col] == ptid].copy()
            if examdate_ts is not None and not hit.empty:
                hit["_diff_days"] = (hit[exam_col] - examdate_ts).abs().dt.days
                hit = hit[hit["_diff_days"] <= 90].sort_values("_diff_days")
        else:
            hit = df[df[ptid_col] == ptid]
        if hit.empty:
            return {}
        row = hit.iloc[0].to_dict()
        for k in ["ab42_ab40_f", "pt217_ab42_f", "nfl_q", "gfap_q", "ab42_f", "ab40_f", "pt217_f"]:
            if k in row:
                try:
                    row[k] = float(row[k])
                except Exception:
                    pass
        return {k.upper(): v for k, v in row.items()}
    
    def encode_template(modality, ptid: str = None, examdate: str = None):
        common_text = modality_common_texts[modality]
        plasma_config = _default_config()["plasma_thresholds"]
        plasma_row = _pick_plasma(ptid, examdate)
        plasma_text = build_plasma_text_ds(plasma_row, plasma_config)
        text = f"[PLASMA]\n{plasma_text}\n[/PLASMA]\n[SEP]\n{common_text}"
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            base = bio_model.get_text_features(**inputs).squeeze(0)
        return adapters[modality](base).unsqueeze(0)
    
    # ============ 加载数据集 JSON ============
    print(f"\n📂 加载数据集...")
    
    if not os.path.exists(train_json_path) or not os.path.exists(val_json_path):
        print(f"❌ 错误: JSON 文件不存在，请先运行 train.py 生成数据集划分")
        print(f"   train: {train_json_path}")
        print(f"   val: {val_json_path}")
        return 1
    
    with open(train_json_path, "r") as f:
        train_data = json.load(f)
    with open(val_json_path, "r") as f:
        val_data = json.load(f)
    
    print(f"   训练集样本数: {len(train_data)}")
    print(f"   验证集样本数: {len(val_data)}")
    
    # 检查 TAU 是否可用
    tau_available = any(item.get("tau") is not None for item in train_data)
    print(f"   TAU 模态可用: {tau_available}")
    
    # ============ 构建 paired_data 用于生成 text embeddings（与 train.py 相同）============
    print(f"\n📝 构建配对数据...")
    mri_dir = os.path.join(base_dir, "MRI")
    av45_dir = os.path.join(base_dir, "PET_MNI", 'AV45')
    fdg_dir = os.path.join(base_dir, "PET_MNI", 'FDG')
    tau_dir = os.path.join(base_dir, "PET_MNI", 'TAU')
    
    csv_data = pd.read_csv(csv_path)
    csv_dict = csv_data.set_index("Subject ID")["Description"].to_dict()
    
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
    
    # 构建 paired_data（与 train.py 相同逻辑）
    paired_data = []
    for patient_id, mri_file in mri_dict.items():
        has_fdg = patient_id in fdg_dict
        has_av45 = patient_id in av45_dict
        has_tau = tau_available and patient_id in tau_dict
        if not (has_fdg or has_av45 or has_tau):
            continue
        description = csv_dict.get(patient_id, None)
        entry = {
            "name": patient_id,
            "mri": mri_file,
            "av45": av45_dict.get(patient_id),
            "fdg": fdg_dict.get(patient_id),
            "description": description
        }
        if tau_available:
            entry["tau"] = tau_dict.get(patient_id)
        paired_data.append(entry)
    
    for idx, data in enumerate(paired_data):
        data["fdg_index"] = idx
        data["av45_index"] = idx
        if tau_available:
            data["tau_index"] = idx
    
    print(f"   配对数据总数: {len(paired_data)}")
    
    # ============ 生成 Text Embeddings（与 train.py 相同）============
    print(f"\n🔤 生成 Text Embeddings...")
    modal_information = [data.get("description", "NA") or "NA" for data in paired_data]
    
    # 生成 template features
    template_features = {m: [] for m in modalities_all if m != "TAU" or tau_available}
    print(f"   生成模态 template features...")
    for item in tqdm(paired_data, desc="Template Features"):
        ptid = item["name"]
        template_features["FDG"].append(encode_template("FDG", ptid))
        template_features["AV45"].append(encode_template("AV45", ptid))
        if tau_available:
            template_features["TAU"].append(encode_template("TAU", ptid))
    
    fdg_template_features = torch.cat(template_features["FDG"], dim=0).cpu().detach() if template_features["FDG"] else None
    av45_template_features = torch.cat(template_features["AV45"], dim=0).cpu().detach() if template_features["AV45"] else None
    tau_template_features = torch.cat(template_features["TAU"], dim=0).cpu().detach() if tau_available and template_features["TAU"] else None
    
    # 生成 personal description embeddings
    print(f"   生成 personal description embeddings...")
    personal_description = [desc for desc in modal_information]
    text_inputs = processor(
        text=personal_description,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=256,
    ).to(device)
    with torch.no_grad():
        base_feats = bio_model.get_text_features(**text_inputs)
    personal_descri_embed = base_feats.cpu().detach()
    
    print(f"✅ Text Embeddings 生成完成")
    print(f"   personal_descri_embed shape: {personal_descri_embed.shape}")
    print(f"   fdg_template_features shape: {fdg_template_features.shape}")
    
    # ============ 定义 transform 函数（与 train.py 相同）============
    def fdg_index_transform(x):
        text_feat = personal_descri_embed[x].unsqueeze(0)
        tmpl_feat = fdg_template_features[x].unsqueeze(0)
        return torch.cat([text_feat, tmpl_feat], dim=0)

    def av45_index_transform(x):
        text_feat = personal_descri_embed[x].unsqueeze(0)
        tmpl_feat = av45_template_features[x].unsqueeze(0)
        return torch.cat([text_feat, tmpl_feat], dim=0)

    def tau_index_transform(x):
        text_feat = personal_descri_embed[x].unsqueeze(0)
        tmpl_feat = tau_template_features[x].unsqueeze(0)
        return torch.cat([text_feat, tmpl_feat], dim=0)
    
    # ============ 准备数据（与 train.py 相同）============
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
    
    # ============ 构建 transforms（与 train.py 完全相同）============
    print(f"\n🔧 构建数据处理管道...")
    print(f"   处理的模态: {pet_keys}")
    
    base_transforms = mt.Compose([
        mt.Lambdad(keys=pet_keys, func=mat_load),
        ReduceTo3D(keys=pet_keys, reduce='mean'),
        FillMissingPET(keys=pet_keys, ref_key="mri"),
        mt.CropForegroundd(keys=pet_keys, source_key="mri"),
        RemoveCropForegroundMeta(),  # 移除 CropForegroundd 添加的元数据
        mt.EnsureChannelFirstd(keys=pet_keys, channel_dim='no_channel'),
        mt.HistogramNormalized(keys=["mri"]),
        mt.ResizeWithPadOrCropd(keys=pet_keys, spatial_size=[128, 128, 128]),
        mt.Spacingd(keys=pet_keys, pixdim=(1.0, 1.0, 1.0)),
        mt.NormalizeIntensityd(keys=pet_keys),
        mt.ScaleIntensityd(keys=pet_keys),
        mt.Lambdad(keys=["fdg_index"], func=fdg_index_transform),
        mt.Lambdad(keys=["av45_index"], func=av45_index_transform),
        mt.Lambdad(keys=["tau_index"], func=tau_index_transform) if tau_available else mt.Identity(),
        ConvertToTensor(),  # 将所有 MetaTensor 转换为普通 Tensor，避免 batch_size > 1 时 collate 报错
    ])
    
    # ============ 生成训练集缓存 ============
    print(f"\n" + "=" * 80)
    print("📦 生成训练集缓存")
    print("=" * 80)
    train_cache_dir = os.path.join(cache_dir, "train")
    print(f"缓存目录: {train_cache_dir}")
    
    # 检查已有缓存
    existing_cache = os.listdir(train_cache_dir) if os.path.exists(train_cache_dir) else []
    if existing_cache:
        print(f"⚠️  发现 {len(existing_cache)} 个已有缓存文件")
        print("   输入 'y' 清空并重新生成，输入其他跳过已有缓存")
        try:
            response = input("是否清空并重新生成? (y/n): ")
            if response.lower() == 'y':
                print("🗑️  清空旧缓存...")
                for f in existing_cache:
                    os.remove(os.path.join(train_cache_dir, f))
            else:
                print("✅ 保留现有缓存，仅生成缺失的")
        except EOFError:
            print("   非交互模式，保留现有缓存")
    
    try:
        train_ds = PersistentDataset(
            data=train_data, 
            transform=base_transforms, 
            cache_dir=train_cache_dir
        )
        
        print(f"\n⭐ 开始生成训练集缓存（共 {len(train_ds)} 个样本）...")
        for i in tqdm(range(len(train_ds)), desc="训练集缓存"):
            try:
                _ = train_ds[i]
            except Exception as e:
                print(f"\n❌ 样本 {i} ({train_data[i].get('name', 'unknown')}) 缓存失败:")
                print(f"   错误: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        print("✅ 训练集缓存生成完成！")
        cache_files = os.listdir(train_cache_dir)
        print(f"   生成的缓存文件数: {len(cache_files)}")
        if cache_files:
            print(f"   文件示例: {cache_files[0]}")
            # 检查文件大小
            total_size = sum(os.path.getsize(os.path.join(train_cache_dir, f)) for f in cache_files)
            print(f"   总大小: {total_size / 1024**3:.2f} GB")
    
    except Exception as e:
        print(f"\n❌ 训练集缓存生成失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ============ 生成验证集缓存 ============
    print(f"\n" + "=" * 80)
    print("📦 生成验证集缓存")
    print("=" * 80)
    val_cache_dir = os.path.join(cache_dir, "val")
    print(f"缓存目录: {val_cache_dir}")
    
    # 检查已有缓存
    existing_cache = os.listdir(val_cache_dir) if os.path.exists(val_cache_dir) else []
    if existing_cache:
        print(f"⚠️  发现 {len(existing_cache)} 个已有缓存文件")
        try:
            response = input("是否清空并重新生成? (y/n): ")
            if response.lower() == 'y':
                print("🗑️  清空旧缓存...")
                for f in existing_cache:
                    os.remove(os.path.join(val_cache_dir, f))
            else:
                print("✅ 保留现有缓存，仅生成缺失的")
        except EOFError:
            print("   非交互模式，保留现有缓存")
    
    try:
        val_ds = PersistentDataset(
            data=val_data, 
            transform=base_transforms, 
            cache_dir=val_cache_dir
        )
        
        print(f"\n⭐ 开始生成验证集缓存（共 {len(val_ds)} 个样本）...")
        for i in tqdm(range(len(val_ds)), desc="验证集缓存"):
            try:
                _ = val_ds[i]
            except Exception as e:
                print(f"\n❌ 样本 {i} ({val_data[i].get('name', 'unknown')}) 缓存失败:")
                print(f"   错误: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        print("✅ 验证集缓存生成完成！")
        cache_files = os.listdir(val_cache_dir)
        print(f"   生成的缓存文件数: {len(cache_files)}")
        if cache_files:
            print(f"   文件示例: {cache_files[0]}")
            # 检查文件大小
            total_size = sum(os.path.getsize(os.path.join(val_cache_dir, f)) for f in cache_files)
            print(f"   总大小: {total_size / 1024**3:.2f} GB")
    
    except Exception as e:
        print(f"\n❌ 验证集缓存生成失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\n" + "=" * 80)
    print("🎉 所有缓存生成完成！")
    print("=" * 80)
    print(f"缓存位置: {cache_dir}")
    print("现在可以运行训练脚本 train.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
