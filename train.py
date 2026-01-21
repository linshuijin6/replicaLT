#%%
"""
train_refactored.py - 重构版训练脚本

主要改动：
1. 数据读取：直接从JSON读取（由preprocess.py生成），包含index、plasma、description、diagnosis等
2. plasma文本生成：使用build_plasma_text_from_json函数，处理缺失值（-999.0 -> "NA"）
3. BiomedCLIP处理：为每个样本生成 [plasma_text + modality_common_text] 的特征向量
4. has_*判断：基于文件路径是否以 "_zero.nii.gz" 结尾
5. 删除冗余代码：移除CSV检索、_pick_plasma、_load_plasma_table等
"""

import pandas as pd
from report_error import email_on_error
from typing import Dict, List, Optional
import nibabel as nib
import numpy as np
import torch.multiprocessing as mp
from generative.networks.nets.diffusion_model_unet import DistributedDiffusionModelUNet
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import math

# 设置多进程启动方式为'spawn'，避免CUDA在fork进程中初始化问题
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass


# ==================== 工具函数 ====================
def mat_load(filepath):
    """使用 nibabel 加载 NIfTI 文件，并转换为 NumPy 数组。
    如果是4D数据（包含时间维度），自动取第一个时间点。
    """
    if filepath is None:
        return None
    data = nib.load(filepath).get_fdata()
    # 如果是4D数据，取第一个时间点
    if data.ndim == 4:
        data = data[..., 0]
    return data


def is_zero_filled_file(filepath: str) -> bool:
    """判断文件是否为零填充文件（由preprocess.py生成）"""
    if filepath is None:
        return True
    return filepath.endswith("_zero.nii.gz")


# ==================== Plasma 文本生成（参照 adapter_finetune/dataset.py）====================
# 预设缺失值标识（与 preprocess.py 一致）
MISSING_PLASMA = -999.0

# Plasma 阈值配置（与 adapter_finetune/dataset.py 一致）
PLASMA_THRESHOLDS = {
    "AB42_AB40": {
        "negative_above": 0.1053,
        "positive_below": 0.0820,
        "intermediate_range": [0.0821, 0.1052],
    },
    "pT217": {
        "negative_below": 0.128,
        "positive_above": 0.300,
        "intermediate_range": [0.129, 0.299],
    },
    "pT217_AB42": {
        "negative_below": 0.0055,
        "positive_above": 0.0086,
        "intermediate_range": [0.0056, 0.0085],
    },
    # NfL 和 GFAP 暂无阈值定义，显示为 UNKNOWN
}


def map_value_to_state(value: float, rule: Dict) -> str:
    """将数值映射为状态（POSITIVE/NEGATIVE/INTERMEDIATE/UNKNOWN）"""
    if value is None or value == MISSING_PLASMA or (isinstance(value, float) and math.isnan(value)):
        return "UNKNOWN"
    
    if "positive_below" in rule and value <= rule["positive_below"]:
        return "POSITIVE"
    if "positive_above" in rule and value >= rule["positive_above"]:
        return "POSITIVE"
    if "negative_below" in rule and value <= rule["negative_below"]:
        return "NEGATIVE"
    if "negative_above" in rule and value >= rule["negative_above"]:
        return "NEGATIVE"
    if "intermediate_range" in rule:
        lo, hi = rule["intermediate_range"]
        if lo <= value <= hi:
            return "INTERMEDIATE"
    return "UNKNOWN"


def format_biomarker_value(value: Optional[float]) -> str:
    """格式化生物标志物数值"""
    if value is None or value == MISSING_PLASMA or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    if isinstance(value, (int, float)):
        formatted = f"{value:.6f}"
        if "." in formatted:
            formatted = formatted.rstrip("0").rstrip(".")
        return formatted
    return str(value)


def build_plasma_text_from_json(plasma_data: Dict[str, float]) -> str:
    """
    从JSON中的plasma数据生成文本描述。
    处理preprocess.py中设置的缺失值（-999.0）。
    
    Args:
        plasma_data: {"AB42_AB40": 0.08, "pT217": 0.15, ...} 或包含 -999.0 的缺失值
    
    Returns:
        格式化的plasma文本，如：
        "Aβ42/Aβ40 = 0.08 (POSITIVE);
         p-tau217 = 0.15 (INTERMEDIATE);
         ..."
    """
    # 固定顺序和标签映射
    order = [
        ("AB42_AB40", "Aβ42/Aβ40"),
        ("pT217", "p-tau217"),
        ("pT217_AB42", "p-tau217/Aβ42"),
        ("NfL", "NfL"),
        ("GFAP", "GFAP"),
    ]
    
    segments: List[str] = []
    for key, label in order:
        value = plasma_data.get(key, MISSING_PLASMA)
        rule = PLASMA_THRESHOLDS.get(key, {})
        state = map_value_to_state(value, rule)
        val_text = format_biomarker_value(value)
        segments.append(f"{label} = {val_text} ({state});")
    
    return "\n".join(segments)


def build_modality_text(plasma_data: Dict[str, float], description: Optional[str], 
                        diagnosis: Optional[str], modality: str, 
                        modality_common_texts: Dict[str, str]) -> str:
    """
    构建完整的模态文本（用于BiomedCLIP编码）。
    
    格式：
    [PLASMA]
    Aβ42/Aβ40 = 0.08 (POSITIVE);
    p-tau217 = 0.15 (INTERMEDIATE);
    ...
    [/PLASMA]
    [SEP]
    <modality_common_text>
    
    Args:
        plasma_data: plasma生物标志物数据
        description: 个人描述（可选）
        diagnosis: 诊断结果（可选）
        modality: 模态类型 ("FDG", "AV45", "TAU")
        modality_common_texts: 模态通用描述文本字典
    
    Returns:
        完整的模态文本
    """
    plasma_text = build_plasma_text_from_json(plasma_data)
    common_text = modality_common_texts.get(modality, "")
    
    return f"[PLASMA]\n{plasma_text}\n[/PLASMA]\n[SEP]\n{common_text}"


@email_on_error()
def main():
    import torch
    import logging
    logging.basicConfig(level=logging.INFO)
    import torch.nn as nn
    from liutuo_utils import compare_3d_jet, compare_3d
    from monai.data import PersistentDataset
    from transformers import AutoProcessor, AutoModel
    import os
    import json
    from tqdm import tqdm
    from torch.amp import GradScaler, autocast
    import torch.nn.functional as F
    from torch.utils.tensorboard import SummaryWriter

    # ==================== 0. 训练参数配置 ====================
    cache_dir = '/mnt/nfsdata/nfsdata/lsj.14/ADNI_cache_tem'
    adapter_ckpt_path = "/mnt/nfsdata/nfsdata/lsj.14/replicaLT/adapter_finetune/runs/01.19_2945992/ckpt_epoch400.pt"
    
    # JSON 数据路径（由 preprocess.py 生成）
    train_json_path = "./train_data_with_description.json"
    val_json_path = "./val_data_with_description.json"
    
    # 确保缓存目录存在
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "val"), exist_ok=True)
    print(f"✅ 缓存目录已创建: {cache_dir}")
    
    # GPU配置
    device_id = [5]
    clip_sample_min = 0
    clip_sample_max = 1
    
    # 模态损失权重
    alpha = 1.0  # FDG 损失权重
    beta = 1.0   # AV45 损失权重
    gamma = 1.0  # TAU 损失权重
    
    # 训练配置
    batch_size = 1
    num_workers = 0
    accumulation_steps = 2
    size_of_dataset = None
    n_epochs = 200
    val_interval = 10
    checkpoint_dir = './checkpoint'
    logdir = './runs'
    log_every = 1
    image_log_interval = 10
    run_name = None
    
    # Adapter 配置
    adapter_hidden = 512
    adapter_dropout = 0.1
    
    # 是否继续训练
    last_epoch = False
    resume_checkpoint = None
    
    # 设置 CUDA 设备
    if torch.cuda.is_available():
        try:
            primary_device = device_id[0]
            torch.cuda.set_device(primary_device)
            print(f"✅ 设置主CUDA设备: {primary_device}")
        except RuntimeError as e:
            print(f"Warning: Failed to set CUDA device: {e}")
    
    primary_gpu = device_id[0]
    device = torch.device(f"cuda:{primary_gpu}" if torch.cuda.is_available() else "cpu")
    
    # 打印GPU信息
    if torch.cuda.is_available():
        print(f"\n📊 GPU配置信息:")
        for did in device_id:
            device_name = torch.cuda.get_device_name(did)
            free_mem = torch.cuda.get_device_properties(did).total_memory / 1024**3
            print(f"   GPU {did}: {device_name} - 总显存: {free_mem:.2f} GB")

    # ==================== 1. TensorBoard 日志初始化 ====================
    if run_name:
        final_run_name = run_name
    else:
        now = datetime.now()
        final_run_name = f"{now:%m.%d}_{os.getpid()}"
    run_dir = Path(logdir) / final_run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))
    print(f"\n📝 TensorBoard 日志目录: {run_dir}")

    # 保存超参数
    hparams = {
        "cache_dir": cache_dir,
        "device_id": device_id,
        "batch_size": batch_size,
        "accumulation_steps": accumulation_steps,
        "n_epochs": n_epochs,
        "val_interval": val_interval,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "adapter_ckpt_path": adapter_ckpt_path,
        "adapter_hidden": adapter_hidden,
        "adapter_dropout": adapter_dropout,
    }
    hparam_path = run_dir / "hparams.json"
    with hparam_path.open("w", encoding="utf-8") as f:
        json.dump(hparams, f, ensure_ascii=False, indent=2)
    print(f"✅ 超参数已保存到: {hparam_path}")

    # ==================== 2. 加载 BiomedCLIP 模型 ====================
    local_model_path = "./BiomedCLIP"
    processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
    bio_model = AutoModel.from_pretrained(local_model_path, trust_remote_code=True)
    
    print(f"\n📍 将BiomedCLIP模型加载到设备: {device}")
    bio_model.to(device)
    bio_model.eval()
    print(f"✅ BiomedCLIP模型加载成功")

    # ==================== 3. 定义 Adapter 模块 ====================
    class Adapter(nn.Module):
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

    # 模态公共描述文本
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

    # 加载 Adapter 权重（如有）
    if adapter_ckpt_path and os.path.exists(adapter_ckpt_path):
        ckpt = torch.load(adapter_ckpt_path, map_location=device)
        ckpt_adapters = ckpt.get("text_adapters", {})
        for m, adp in adapters.items():
            if m in ckpt_adapters:
                adp.load_state_dict(ckpt_adapters[m])
        print(f"✅ 加载Adapter权重: {adapter_ckpt_path}")
    else:
        print(f"⚠️ Adapter权重未找到，使用初始化权重")

    for adp in adapters.values():
        adp.eval()

    # ==================== 4. 从JSON加载数据 ====================
    print(f"\n📂 加载训练数据: {train_json_path}")
    with open(train_json_path, "r") as f:
        train_data_raw = json.load(f)
    print(f"   训练样本数: {len(train_data_raw)}")
    
    print(f"📂 加载验证数据: {val_json_path}")
    with open(val_json_path, "r") as f:
        val_data_raw = json.load(f)
    print(f"   验证样本数: {len(val_data_raw)}")
    
    if size_of_dataset:
        train_data_raw = train_data_raw[:size_of_dataset]
        val_data_raw = val_data_raw[:min(size_of_dataset // 10, len(val_data_raw))]
    
    # 打印第一个样本结构
    if train_data_raw:
        print(f"\n📋 样本结构示例:")
        sample = train_data_raw[0]
        for key in ["name", "examdate", "mri", "fdg", "av45", "tau", "diagnosis"]:
            if key in sample:
                val = sample[key]
                if isinstance(val, str) and len(val) > 60:
                    val = val[:60] + "..."
                print(f"   {key}: {val}")
        if "plasma" in sample:
            print(f"   plasma: {sample['plasma']}")

    # ==================== 5. 为每个样本生成 BiomedCLIP 特征（批量并行加速）====================
    print(f"\n🔄 生成模态特征向量（批量并行模式）...")
    
    # 合并所有数据用于特征生成
    all_data = train_data_raw + val_data_raw
    
    # 批量编码参数
    ENCODE_BATCH_SIZE = 32  # 根据GPU显存调整
    
    def encode_texts_batch(texts: List[str], batch_size: int = ENCODE_BATCH_SIZE) -> torch.Tensor:
        """批量使用BiomedCLIP编码文本"""
        all_features = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            with torch.no_grad():
                features = bio_model.get_text_features(**inputs)
            all_features.append(features.cpu())
        return torch.cat(all_features, dim=0)
    
    # 预先构建所有文本（避免重复计算）
    print(f"   构建文本列表...")
    fdg_texts = []
    av45_texts = []
    tau_texts = []
    desc_texts = []
    
    for item in all_data:
        plasma_data = item.get("plasma", {})
        description = item.get("description", None)
        diagnosis = item.get("diagnosis", "UNKNOWN")
        
        # 各模态文本
        fdg_texts.append(build_modality_text(plasma_data, description, diagnosis, "FDG", modality_common_texts))
        av45_texts.append(build_modality_text(plasma_data, description, diagnosis, "AV45", modality_common_texts))
        tau_texts.append(build_modality_text(plasma_data, description, diagnosis, "TAU", modality_common_texts))
        
        # 个人描述文本
        desc_texts.append(description if description else "NA")
    
    n_samples = len(all_data)
    print(f"   样本数: {n_samples}, 批量大小: {ENCODE_BATCH_SIZE}")
    
    # 批量编码各模态文本
    print(f"   编码 FDG 文本...")
    fdg_base_features = encode_texts_batch(fdg_texts)
    
    print(f"   编码 AV45 文本...")
    av45_base_features = encode_texts_batch(av45_texts)
    
    print(f"   编码 TAU 文本...")
    tau_base_features = encode_texts_batch(tau_texts)
    
    print(f"   编码 描述 文本...")
    personal_descri_embed = encode_texts_batch(desc_texts)
    
    # 批量通过Adapter（在GPU上处理）
    print(f"   通过 Adapter 网络...")
    with torch.no_grad():
        # 分批处理Adapter以避免显存溢出
        fdg_adapted_list = []
        av45_adapted_list = []
        tau_adapted_list = []
        
        for i in range(0, n_samples, ENCODE_BATCH_SIZE):
            end_idx = min(i + ENCODE_BATCH_SIZE, n_samples)
            
            fdg_batch = fdg_base_features[i:end_idx].to(device)
            av45_batch = av45_base_features[i:end_idx].to(device)
            tau_batch = tau_base_features[i:end_idx].to(device)
            
            fdg_adapted_list.append(adapters["FDG"](fdg_batch).cpu())
            av45_adapted_list.append(adapters["AV45"](av45_batch).cpu())
            tau_adapted_list.append(adapters["TAU"](tau_batch).cpu())
        
        fdg_template_features = torch.cat(fdg_adapted_list, dim=0)
        av45_template_features = torch.cat(av45_adapted_list, dim=0)
        tau_template_features = torch.cat(tau_adapted_list, dim=0)
    
    print(f"✅ 特征生成完成: FDG={fdg_template_features.shape}, AV45={av45_template_features.shape}, TAU={tau_template_features.shape}")

    # ==================== 6. 准备 DataLoader 数据 ====================
    # 为训练/验证数据分配正确的索引
    train_size = len(train_data_raw)
    
    def prepare_data_entry(item: Dict, global_idx: int) -> Dict:
        """准备单个数据条目"""
        return {
            "name": item["name"],
            "mri": item["mri"],
            "fdg": item["fdg"],
            "av45": item["av45"],
            "tau": item["tau"],
            "fdg_index": global_idx,
            "av45_index": global_idx,
            "tau_index": global_idx,
            # 标记是否为零填充文件（用于跳过损失计算）
            "is_fdg_zero": is_zero_filled_file(item["fdg"]),
            "is_av45_zero": is_zero_filled_file(item["av45"]),
            "is_tau_zero": is_zero_filled_file(item["tau"]),
        }
    
    train_data = [prepare_data_entry(item, idx) for idx, item in enumerate(train_data_raw)]
    val_data = [prepare_data_entry(item, train_size + idx) for idx, item in enumerate(val_data_raw)]
    
    print(f"\n📊 数据准备完成:")
    print(f"   训练集: {len(train_data)} 样本")
    print(f"   验证集: {len(val_data)} 样本")
    
    # 统计零填充文件
    zero_stats = {"fdg": 0, "av45": 0, "tau": 0}
    for item in train_data + val_data:
        if item["is_fdg_zero"]:
            zero_stats["fdg"] += 1
        if item["is_av45_zero"]:
            zero_stats["av45"] += 1
        if item["is_tau_zero"]:
            zero_stats["tau"] += 1
    print(f"   零填充文件统计 - FDG: {zero_stats['fdg']}, AV45: {zero_stats['av45']}, TAU: {zero_stats['tau']}")

    # ==================== 7. 数据变换和 DataLoader ====================
    from monai.data import DataLoader, pad_list_data_collate, PersistentDataset
    import monai.transforms as mt

    class RemoveCropForegroundMeta(mt.Transform):
        """移除 CropForegroundd 添加的元数据键"""
        def __call__(self, data):
            d = dict(data)
            keys_to_remove = ['foreground_start_coord', 'foreground_end_coord']
            for k in keys_to_remove:
                d.pop(k, None)
            return d

    class ConvertToTensor(mt.Transform):
        """将所有 MetaTensor 转换为普通 torch.Tensor"""
        def __call__(self, data):
            from monai.data import MetaTensor
            d = dict(data)
            for k, v in d.items():
                if isinstance(v, MetaTensor):
                    d[k] = v.as_tensor()
                elif isinstance(v, torch.Tensor):
                    d[k] = v.clone().detach()
            return d

    # 特征索引转换函数
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

    pet_keys = ["mri", "fdg", "av45", "tau"]

    # 数据增强流程（移除了 FillMissingPET，因为JSON中不存在None路径）
    train_transforms = mt.Compose([
        mt.Lambdad(keys=pet_keys, func=mat_load),
        mt.CropForegroundd(keys=pet_keys, source_key="mri"),
        RemoveCropForegroundMeta(),
        mt.EnsureChannelFirstd(keys=pet_keys, channel_dim='no_channel'),
        mt.HistogramNormalized(keys=["mri"]),
        mt.ResizeWithPadOrCropd(keys=pet_keys, spatial_size=[128, 128, 128]),
        mt.Spacingd(keys=pet_keys, pixdim=(1.0, 1.0, 1.0)),
        mt.NormalizeIntensityd(keys=pet_keys),
        mt.ScaleIntensityd(keys=pet_keys),
        mt.Lambdad(keys=["fdg_index"], func=fdg_index_transform),
        mt.Lambdad(keys=["av45_index"], func=av45_index_transform),
        mt.Lambdad(keys=["tau_index"], func=tau_index_transform),
        ConvertToTensor(),
    ])

    val_transforms = mt.Compose([
        mt.Lambdad(keys=pet_keys, func=mat_load),
        mt.CropForegroundd(keys=pet_keys, source_key="mri"),
        RemoveCropForegroundMeta(),
        mt.EnsureChannelFirstd(keys=pet_keys, channel_dim='no_channel'),
        mt.HistogramNormalized(keys=["mri"]),
        mt.ResizeWithPadOrCropd(keys=pet_keys, spatial_size=[128, 128, 128]),
        mt.Spacingd(keys=pet_keys, pixdim=(1.0, 1.0, 1.0)),
        mt.NormalizeIntensityd(keys=pet_keys),
        mt.ScaleIntensityd(keys=pet_keys),
        mt.Lambdad(keys=["fdg_index"], func=fdg_index_transform),
        mt.Lambdad(keys=["av45_index"], func=av45_index_transform),
        mt.Lambdad(keys=["tau_index"], func=tau_index_transform),
        ConvertToTensor(),
    ])

    # 创建 DataLoader
    train_cache_dir = os.path.join(cache_dir, "train")
    val_cache_dir = os.path.join(cache_dir, "val")
    
    print(f"\n📦 创建训练集 PersistentDataset...")
    train_ds = PersistentDataset(data=train_data, transform=train_transforms, cache_dir=train_cache_dir)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=False, collate_fn=pad_list_data_collate
    )

    print(f"📦 创建验证集 PersistentDataset...")
    val_ds = PersistentDataset(data=val_data, transform=val_transforms, cache_dir=val_cache_dir)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False, collate_fn=pad_list_data_collate
    )

    # ==================== 8. 加载扩散模型 ====================
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from generative.networks.nets import DiffusionModelUNet
    from generative.networks.schedulers import DDPMScheduler

    if len(device_id) == 1:
        model = DiffusionModelUNet(
            spatial_dims=3, in_channels=1, out_channels=1,
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
            spatial_dims=3, in_channels=1, out_channels=1,
            num_channels=(32, 64, 64, 128),
            attention_levels=(False, False, False, True),
            num_res_blocks=1,
            num_head_channels=(0, 0, 0, 128),
            with_conditioning=True,
            cross_attention_dim=512,
            use_flash_attention=True,
            device_ids=device_id
        )
        print(f"✅ 双GPU模式: GPU {device_id[0]} 和 GPU {device_id[1]}")
    else:
        raise ValueError("仅支持 1 或 2 个 GPU")
    
    use_distributed = len(device_id) == 2
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=2.5e-5)
    scaler = GradScaler()
    global_step = 0
    start_epoch = 0

    # 恢复检查点
    if last_epoch and resume_checkpoint:
        print(f"\n📂 恢复训练: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint['model'])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        if "global_step" in checkpoint:
            global_step = checkpoint['global_step']
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"✅ 从 epoch {start_epoch} 继续训练")

    scheduler = DDPMScheduler(
        prediction_type="v_prediction", num_train_timesteps=1000,
        schedule="scaled_linear_beta", beta_start=0.0005, beta_end=0.0195
    )
    scheduler.set_timesteps(num_inference_steps=1000)

    # ==================== 9. 辅助函数 ====================
    def get_3d_slices(volume, normalize=True):
        """从3D体积中提取三个正交切面"""
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

    def log_comparison_figure(writer, tag, volumes_dict, step):
        """将多体积对比图记录到TensorBoard"""
        from io import BytesIO
        from PIL import Image
        
        n_vols = len(volumes_dict)
        if n_vols == 0:
            return
        
        fig, axes = plt.subplots(3, n_vols, figsize=(3 * n_vols, 9))
        if n_vols == 1:
            axes = axes.reshape(3, 1)
        
        for col_idx, (name, vol) in enumerate(volumes_dict.items()):
            if vol is None:
                continue
            slices = get_3d_slices(vol)
            cmap = 'jet' if 'Diff' in name else 'gray'
            for row_idx, view in enumerate(['axial', 'coronal', 'sagittal']):
                ax = axes[row_idx, col_idx]
                ax.imshow(slices[view], cmap=cmap, vmin=0, vmax=1)
                ax.axis('off')
                if row_idx == 0:
                    ax.set_title(name, fontsize=10)
        
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        img_array = np.array(img)[:, :, :3].transpose(2, 0, 1)
        writer.add_image(tag, img_array, step)

    def save_checkpoint(tag: str, epoch: int, step: int) -> None:
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": step,
            "hparams": hparams,
        }
        ckpt_path = run_dir / f"ckpt_{tag}.pt"
        torch.save(ckpt, ckpt_path)
        os.makedirs(checkpoint_dir, exist_ok=True)
        legacy_path = f'{checkpoint_dir}/first_part_{epoch + 1}.pth'
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, legacy_path)
        print(f"✅ Checkpoint saved: {ckpt_path}")

    # ==================== 10. 训练循环 ====================
    epoch_loss_list = []
    val_epoch_loss_list = []
    
    print(f"\n🚀 开始训练 (共 {n_epochs} epochs)...")
    
    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(total=len(train_loader), ncols=70, desc=f"Epoch {epoch}")

        for step, data in enumerate(train_loader):
            images = data["mri"].clone().detach().to(device)
            seg_fdg = data["fdg"].clone().detach().to(device)
            seg_av45 = data["av45"].clone().detach().to(device)
            seg_tau = data["tau"].clone().detach().to(device)
            fdg_index = data["fdg_index"].clone().detach().to(device)
            av45_index = data["av45_index"].clone().detach().to(device)
            tau_index = data["tau_index"].clone().detach().to(device)
            
            # 使用文件路径标记判断是否为有效模态（非零填充）
            is_fdg_zero = data.get("is_fdg_zero", [False])[0] if isinstance(data.get("is_fdg_zero"), list) else data.get("is_fdg_zero", False)
            is_av45_zero = data.get("is_av45_zero", [False])[0] if isinstance(data.get("is_av45_zero"), list) else data.get("is_av45_zero", False)
            is_tau_zero = data.get("is_tau_zero", [False])[0] if isinstance(data.get("is_tau_zero"), list) else data.get("is_tau_zero", False)
            
            has_fdg = not is_fdg_zero
            has_av45 = not is_av45_zero
            has_tau = not is_tau_zero

            if step % accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            time_embedding = torch.randint(0, 1000, (images.shape[0],), device=device).long()
            if epoch >= 140:
                time_embedding = torch.tensor([0], device=device, dtype=torch.long)

            with torch.no_grad():
                t = time_embedding.float() / 1000
                t = t.view(-1, 1, 1, 1, 1)

            total_loss = 0.0
            
            # 收集可用模态
            available_modalities = []
            if has_fdg:
                available_modalities.append('fdg')
            if has_av45:
                available_modalities.append('av45')
            if has_tau:
                available_modalities.append('tau')
            
            # 随机选择一个模态训练
            if len(available_modalities) > 0:
                import random
                selected_modality = random.choice(available_modalities)
                
                # 用于保存训练过程图像的变量
                train_x_t = None
                train_v_pred = None
                train_gt = None
                train_modality_name = None
                
                if selected_modality == 'fdg':
                    with autocast(device_type='cuda', enabled=True):
                        x_t = t * seg_fdg + (1 - t) * images
                        v_pred = model(x=x_t, timesteps=time_embedding, context=fdg_index)
                        if use_distributed:
                            v_pred = v_pred.to(device)
                        v_gt = seg_fdg - images
                        loss = F.mse_loss(v_gt.float(), v_pred.float())
                        loss = (alpha * loss) / accumulation_steps
                    scaler.scale(loss).backward()
                    total_loss = loss.item() * accumulation_steps
                    # 保存用于可视化
                    train_x_t = x_t.detach()
                    train_v_pred = v_pred.detach()
                    train_gt = seg_fdg.detach()
                    train_modality_name = "FDG"
                    
                elif selected_modality == 'av45':
                    with autocast(device_type='cuda', enabled=True):
                        x_t = t * seg_av45 + (1 - t) * images
                        v_pred = model(x=x_t, timesteps=time_embedding, context=av45_index)
                        if use_distributed:
                            v_pred = v_pred.to(device)
                        v_gt = seg_av45 - images
                        loss = F.mse_loss(v_gt.float(), v_pred.float())
                        loss = (beta * loss) / accumulation_steps
                    scaler.scale(loss).backward()
                    total_loss = loss.item() * accumulation_steps
                    # 保存用于可视化
                    train_x_t = x_t.detach()
                    train_v_pred = v_pred.detach()
                    train_gt = seg_av45.detach()
                    train_modality_name = "AV45"
                    
                elif selected_modality == 'tau':
                    with autocast(device_type='cuda', enabled=True):
                        x_t = t * seg_tau + (1 - t) * images
                        v_pred = model(x=x_t, timesteps=time_embedding, context=tau_index)
                        if use_distributed:
                            v_pred = v_pred.to(device)
                        v_gt = seg_tau - images
                        loss = F.mse_loss(v_gt.float(), v_pred.float())
                        loss = (gamma * loss) / accumulation_steps
                    scaler.scale(loss).backward()
                    total_loss = loss.item() * accumulation_steps
                    # 保存用于可视化
                    train_x_t = x_t.detach()
                    train_v_pred = v_pred.detach()
                    train_gt = seg_tau.detach()
                    train_modality_name = "TAU"
                
                torch.cuda.empty_cache()
            
            if (step + 1) % accumulation_steps == 0 and total_loss > 0:
                scaler.step(optimizer)
                scaler.update()

            epoch_loss += total_loss
            global_step += 1

            if log_every > 0 and global_step % log_every == 0:
                writer.add_scalar("train/loss", epoch_loss / (step + 1), global_step)
                writer.add_scalar("train/epoch", epoch, global_step)

            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
            progress_bar.update(1)
            
            # 训练阶段图像保存（直接使用训练过程中的变量，不重新推理）
            if (epoch + 1) % image_log_interval == 0 and step == 0 and train_x_t is not None:
                with torch.no_grad():
                    # 使用训练过程中实际计算的 x_t 和 v_pred
                    x_pred = torch.clamp(train_x_t + train_v_pred, clip_sample_min, clip_sample_max)
                    
                    train_vis_volumes = {
                        "MRI": images.cpu(),
                        "x_t": train_x_t.cpu(),
                        f"{train_modality_name}_GT": train_gt.cpu(),
                        f"{train_modality_name}_Pred": x_pred.cpu(),
                        f"{train_modality_name}_Diff": torch.abs(train_gt - x_pred).cpu(),
                    }
                    
                    log_comparison_figure(writer, f"train/comparison_epoch{epoch+1}", train_vis_volumes, epoch)
            
            torch.cuda.empty_cache()

        progress_bar.close()
        epoch_avg_loss = epoch_loss / max(step + 1, 1)
        epoch_loss_list.append(epoch_avg_loss)
        writer.add_scalar("train/epoch_loss", epoch_avg_loss, epoch)

        # ==================== 验证阶段 ====================
        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_epoch_loss = 0

            for step, data in enumerate(val_loader):
                images = data["mri"].clone().detach().to(device)
                seg_fdg = data["fdg"].clone().detach().to(device)
                seg_av45 = data["av45"].clone().detach().to(device)
                seg_tau = data["tau"].clone().detach().to(device)
                fdg_index = data["fdg_index"].clone().detach().to(device)
                av45_index = data["av45_index"].clone().detach().to(device)
                tau_index = data["tau_index"].clone().detach().to(device)

                is_fdg_zero = data.get("is_fdg_zero", [False])[0] if isinstance(data.get("is_fdg_zero"), list) else data.get("is_fdg_zero", False)
                is_av45_zero = data.get("is_av45_zero", [False])[0] if isinstance(data.get("is_av45_zero"), list) else data.get("is_av45_zero", False)
                is_tau_zero = data.get("is_tau_zero", [False])[0] if isinstance(data.get("is_tau_zero"), list) else data.get("is_tau_zero", False)
                
                has_fdg = not is_fdg_zero
                has_av45 = not is_av45_zero
                has_tau = not is_tau_zero

                x_t = images
                N_sample = 1
                N_sample_tensor = torch.tensor(N_sample, dtype=torch.float32, device=device)

                with torch.no_grad():
                    for t_val in [(i / N_sample) for i in range(N_sample)]:
                        time_emb = int(t_val * 1000)
                        
                        if has_fdg:
                            v_out = model(x=x_t, timesteps=torch.Tensor((time_emb,)).to(device), context=fdg_index)
                            if use_distributed:
                                v_out = v_out.to(device)
                            x_fdg_t = torch.clamp(x_t + v_out / N_sample_tensor, clip_sample_min, clip_sample_max)
                        else:
                            x_fdg_t = None

                        if has_av45:
                            v_out = model(x=x_t, timesteps=torch.Tensor((time_emb,)).to(device), context=av45_index)
                            if use_distributed:
                                v_out = v_out.to(device)
                            x_av45_t = torch.clamp(x_t + v_out / N_sample_tensor, clip_sample_min, clip_sample_max)
                        else:
                            x_av45_t = None

                        if has_tau:
                            v_out = model(x=x_t, timesteps=torch.Tensor((time_emb,)).to(device), context=tau_index)
                            if use_distributed:
                                v_out = v_out.to(device)
                            x_tau_t = torch.clamp(x_t + v_out / N_sample_tensor, clip_sample_min, clip_sample_max)
                        else:
                            x_tau_t = None

                val_loss = 0.0
                if has_fdg and x_fdg_t is not None:
                    val_loss += alpha * F.mse_loss(x_fdg_t.float(), seg_fdg.float())
                if has_av45 and x_av45_t is not None:
                    val_loss += beta * F.mse_loss(x_av45_t.float(), seg_av45.float())
                if has_tau and x_tau_t is not None:
                    val_loss += gamma * F.mse_loss(x_tau_t.float(), seg_tau.float())

                val_epoch_loss += val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss
                torch.cuda.empty_cache()

            val_avg_loss = val_epoch_loss / max(step + 1, 1)
            val_epoch_loss_list.append(val_avg_loss)
            writer.add_scalar("val/loss", val_avg_loss, epoch)
            print(f"Epoch {epoch + 1} Validation loss: {val_avg_loss:.6f}")

            save_checkpoint(f"epoch{epoch + 1}", epoch, global_step)

            # 可视化（包含输入x_t）
            if (epoch + 1) % image_log_interval == 0:
                comparison_volumes = {
                    "MRI(x_t)": images.cpu(),  # 输入MRI即为x_t
                }
                if has_fdg and x_fdg_t is not None:
                    comparison_volumes["FDG_GT"] = seg_fdg.cpu()
                    comparison_volumes["FDG_Pred"] = x_fdg_t.cpu()
                    # 计算差异图
                    comparison_volumes["FDG_Diff"] = torch.abs(seg_fdg - x_fdg_t).cpu()
                if has_av45 and x_av45_t is not None:
                    comparison_volumes["AV45_GT"] = seg_av45.cpu()
                    comparison_volumes["AV45_Pred"] = x_av45_t.cpu()
                    comparison_volumes["AV45_Diff"] = torch.abs(seg_av45 - x_av45_t).cpu()
                if has_tau and x_tau_t is not None:
                    comparison_volumes["TAU_GT"] = seg_tau.cpu()
                    comparison_volumes["TAU_Pred"] = x_tau_t.cpu()
                    comparison_volumes["TAU_Diff"] = torch.abs(seg_tau - x_tau_t).cpu()
                log_comparison_figure(writer, f"val/comparison_epoch{epoch+1}", comparison_volumes, epoch)

            # 计算SSIM和PSNR
            from generative.metrics import SSIMMetric
            from monai.metrics import PSNRMetric
            ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=7)
            psnr_metric = PSNRMetric(1.0)

            if has_fdg and x_fdg_t is not None:
                ssim_val = ssim_metric(seg_fdg.cpu(), x_fdg_t.cpu())
                psnr_val = psnr_metric(seg_fdg.cpu(), x_fdg_t.cpu())
                writer.add_scalar("val/FDG_SSIM", ssim_val.mean().item(), epoch)
                writer.add_scalar("val/FDG_PSNR", psnr_val.mean().item(), epoch)

            if has_av45 and x_av45_t is not None:
                ssim_val = ssim_metric(seg_av45.cpu(), x_av45_t.cpu())
                psnr_val = psnr_metric(seg_av45.cpu(), x_av45_t.cpu())
                writer.add_scalar("val/AV45_SSIM", ssim_val.mean().item(), epoch)
                writer.add_scalar("val/AV45_PSNR", psnr_val.mean().item(), epoch)

            if has_tau and x_tau_t is not None:
                ssim_val = ssim_metric(seg_tau.cpu(), x_tau_t.cpu())
                psnr_val = psnr_metric(seg_tau.cpu(), x_tau_t.cpu())
                writer.add_scalar("val/TAU_SSIM", ssim_val.mean().item(), epoch)
                writer.add_scalar("val/TAU_PSNR", psnr_val.mean().item(), epoch)

    # 保存最终模型
    writer.flush()
    writer.close()
    save_checkpoint("last", epoch if 'epoch' in locals() else 0, global_step)
    print(f"\n🎉 训练完成！日志保存在: {run_dir}")


if __name__ == "__main__":
    main()
