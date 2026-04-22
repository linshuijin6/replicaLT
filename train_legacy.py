# 直接运行一次将自动保存缓存，不可使用precompute_cache.py
#%%
# ============ GPU 设备锁定（必须在 import torch 之前设置） ============
# 防止 CUDA 使用非指定 GPU，避免显存泄漏
import os
_TARGET_GPU_IDS = [5]  # 指定使用的 GPU ID，与下方 device_id 保持一致
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, _TARGET_GPU_IDS))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 确保物理 GPU ID 顺序一致

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


# 自定义加载函数
def mat_load(filepath):
    """
    使用 nibabel 加载 NIfTI 文件，并转换为 NumPy 数组。
    """
    if filepath is None:
        return None
    return nib.load(filepath).get_fdata()


# ============ TensorBoard 图像记录辅助函数 ============
def get_3d_slices(volume, normalize=True):
    """
    从3D体积中提取三个正交切面（axial, coronal, sagittal）用于可视化。
    Args:
        volume: 形状为 (B, C, H, W, D) 或 (C, H, W, D) 或 (H, W, D) 的张量
        normalize: 是否归一化到 [0, 1]
    Returns:
        dict: 包含三个切面的字典
    """
    import torch
    if isinstance(volume, torch.Tensor):
        vol = volume.detach().cpu().numpy()
    else:
        vol = np.array(volume)
    
    # 处理不同维度
    while vol.ndim > 3:
        vol = vol[0]  # 去掉 batch 和 channel 维度
    
    h, w, d = vol.shape
    slices = {
        'axial': vol[:, :, d // 2],      # 轴向切面 (中间)
        'coronal': vol[:, w // 2, :],    # 冠状切面
        'sagittal': vol[h // 2, :, :],   # 矢状切面
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
    """
    将3D体积的三个切面记录到TensorBoard。
    """
    import matplotlib.pyplot as plt
    from io import BytesIO
    from PIL import Image
    
    slices = get_3d_slices(volume)
    
    for view_name, slice_2d in slices.items():
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(slice_2d, cmap=cmap, vmin=0, vmax=1)
        ax.axis('off')
        ax.set_title(f"{view_name}")
        
        # 保存到内存缓冲区
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=100)
        plt.close(fig)
        buf.seek(0)
        
        # 转换为 numpy 数组
        img = Image.open(buf)
        img_array = np.array(img)
        
        # 添加到 TensorBoard (HWC -> CHW)
        if img_array.ndim == 3:
            img_array = img_array[:, :, :3]  # 只取 RGB，去掉 alpha
            img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
        writer.add_image(f"{tag_prefix}/{view_name}", img_array, global_step)


def log_comparison_figure(writer, tag, volumes_dict, global_step):
    """
    创建一个包含多个体积对比的综合图像。
    Args:
        volumes_dict: {"MRI": vol1, "FDG_GT": vol2, "FDG_Pred": vol3, ...}
    """
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
    
    # 保存到内存缓冲区
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close(fig)
    buf.seek(0)
    
    # 转换为 numpy 数组并记录
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
    import os
    import json
    from transformers import AutoTokenizer, AutoModel
    from tqdm import tqdm
    from torch.amp import GradScaler, autocast  # 从 torch.amp 导入 GradScaler
    import torch.nn.functional as F
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    from pathlib import Path

    # 0.训练参数设置
    base_dir = "/mnt/nfsdata/nfsdata/lsj.14/ADNI_CSF"
    # ★ 方案B：只缓存活跃模态 + float16，单文件 76MB → 19MB，总缓存约 19 GB
    cache_dir = '/mnt/linshuijin/ADNI_cache_legacy_fp16'
    plasma_csv_path = "adapter_finetune/ADNI_csv/UPENN_PLASMA_FUJIREBIO_QUANTERIX_21Dec2025.csv"

    # 确保缓存目录存在
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "val"), exist_ok=True)
    print(f"✅ 缓存目录已创建: {cache_dir}")
    
    # 多GPU配置
    # 重要：由于在文件开头设置了 CUDA_VISIBLE_DEVICES，这里的 device_id 是逻辑 ID（从 0 开始）
    # 物理 GPU ID 在文件开头的 _TARGET_GPU_IDS 中指定
    # 例如：_TARGET_GPU_IDS = [2] 表示物理 GPU 2，此时 device_id = [0] 表示逻辑 GPU 0（即物理 GPU 2）
    device_id = [0]  # 不要改这个，逻辑 GPU ID（对应物理 GPU 由 CUDA_VISIBLE_DEVICES 控制）
    # 定义裁剪的最小值和最大值
    clip_sample_min = 0  # 设置合适的最小值
    clip_sample_max = 1   # 设置合适的最大值

    # ============ 模态开关 ============
    # 可在此处选择只训练特定模态（例如：仅 TAU）
    use_fdg = False
    use_av45 = False
    use_tau = True

    # 定义模态权重（根据开关自动置 0）
    alpha = 1.0 if use_fdg else 0.0  # FDG 损失权重
    beta = 1.0 if use_av45 else 0.0  # AV45 损失权重
    gamma = 1.0 if use_tau else 0.0  # TAU 损失权重
    
    # ============ 多GPU优化配置 ============
    accumulation_steps = 1  # 梯度累积步数：双GPU时设为1（无梯度累积）
    # 单GPU 24GB: accumulation_steps=2 → 约10GB显存
    # 双GPU 24GB: accumulation_steps=1 → 约5GB/卡（推荐）
    # 单GPU 16GB: accumulation_steps=4 → 约8GB显存
    # 注意：使用多模态顺序训练时，建议accumulation_steps >= 2
    # 设置 PyTorch 的默认 CUDA 设备
    if torch.cuda.is_available():
        try:
            # 使用device_id列表中的第一个GPU作为主设备
            primary_device = device_id[0]
            torch.cuda.set_device(primary_device)
            print(f"✅ 设置主CUDA设备: {primary_device}")
        except RuntimeError as e:
            print(f"Warning: Failed to set CUDA device {primary_device}: {e}")
            print("Continuing with default CUDA device or CPU...")
    size_of_dataset = None  # 设置为 None 以使用完整数据集，或设置为所需的样本数量
    bs = 1  # batch_size
    n_epochs = 200
    val_interval =10
    checkpoint_dir = './checkpoint'
    logdir = './runs'  # TensorBoard 日志目录
    log_every = 1  # 每隔多少步记录一次日志
    image_log_interval = 10  # 每隔多少个 epoch 保存一次过程图像
    run_name = None  # 自动生成 run_name

    # Adapter 配置
    adapter_ckpt_path = "/home/ssddata/linshuijin/replicaLT/runs/12.31_3568410/ckpt_last.pt"  # 根据需要替换
    adapter_hidden = 512
    adapter_dropout = 0.1

    # ============ 原始方案开关 ============
    # True: 使用原始方案（train_single_gpu.py），不使用adapter，context = old_descr + modality_text_optimized
    # False: 使用adapter方案，context = plasma_text + adapter(text_features)
    use_legacy_mode = True

    # 是否继续上次训练
    last_epoch = False

    # 确认当前默认 CUDA 设备
    if torch.cuda.is_available():
        try:
            print(f"\n📊 GPU配置信息:")
            for did in device_id:
                device_name = torch.cuda.get_device_name(did)
                free_mem = torch.cuda.get_device_properties(did).total_memory / 1024**3
                print(f"   GPU {did}: {device_name} - 总显存: {free_mem:.2f} GB")
            print(f"   使用模式: {'双GPU模型并行' if len(device_id) == 2 else '单GPU'}")
        except RuntimeError as e:
            print(f"❌ Warning: CUDA device check failed: {e}")
            print("将尝试使用CPU或可用设备...")
    else:
        print("⚠️  CUDA不可用，将使用CPU")

    # ============ TensorBoard 日志初始化 ============
    if run_name:
        final_run_name = run_name
    else:
        now = datetime.now()
        final_run_name = f"{now:%m.%d}_{os.getpid()}"
    run_dir = Path(logdir) / final_run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))
    print(f"\n📝 TensorBoard 日志目录: {run_dir}")

    # 保存超参数到 hparams.json
    hparams = {
        "use_fdg": use_fdg,
        "use_av45": use_av45,
        "use_tau": use_tau,
        "base_dir": base_dir,
        "cache_dir": cache_dir,
        "device_id": device_id,
        "accumulation_steps": accumulation_steps,
        "n_epochs": n_epochs,
        "val_interval": val_interval,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "adapter_ckpt_path": adapter_ckpt_path,
        "adapter_hidden": adapter_hidden,
        "adapter_dropout": adapter_dropout,
        "clip_sample_min": clip_sample_min,
        "clip_sample_max": clip_sample_max,
        "bs": bs,
        "log_every": log_every,
        "image_log_interval": image_log_interval,
        "use_legacy_mode": use_legacy_mode,
    }
    hparam_path = run_dir / "hparams.json"
    with hparam_path.open("w", encoding="utf-8") as f:
        json.dump(hparams, f, ensure_ascii=False, indent=2)
    print(f"✅ 超参数已保存到: {hparam_path}")

    # 1. 加载 BiomedCLIP 模型和处理器
    local_model_path = "./BiomedCLIP"
    processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
    bio_model = AutoModel.from_pretrained(local_model_path, trust_remote_code=True)

    # 使用主GPU设备（device_id列表中的第一个）
    primary_gpu = device_id[0]
    device = torch.device(f"cuda:{primary_gpu}" if torch.cuda.is_available() else "cpu")
    print(f"\n📍 将BiomedCLIP模型加载到设备: {device}")
    try:
        bio_model.to(device)
        bio_model.eval()
        print(f"✅ BiomedCLIP模型加载成功")
    except Exception as e:
        print(f"❌ BiomedCLIP模型加载失败: {e}")
        raise

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

    # ============ 原始方案：模态优化文本（参考 train_single_gpu.py）============
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

    # 预计算原始方案的模态优化特征
    if use_legacy_mode:
        print("\n🔧 使用原始方案 (legacy mode): 不使用 adapter，context = old_descr + modality_text_optimized")
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

    # 模态公共描述文本（与 adapter_finetune/dataset.py 保持一致风格）
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

    def build_plasma_text(desc_text):
        text = desc_text if desc_text is not None else "NA"
        return f"[PLASMA]\n{text}\n[/PLASMA]"

    def build_modal_text(desc_text, modality):
        plasma_block = build_plasma_text(desc_text)
        common_text = modality_common_texts[modality]
        return f"{plasma_block}\n[SEP]\n{common_text}"

    feat_dim = bio_model.config.projection_dim
    modalities_all = ["FDG", "AV45", "TAU"]

    # 仅在非原始方案时加载 adapter
    if not use_legacy_mode:
        adapters = {m: Adapter(feat_dim, hidden=adapter_hidden, dropout=adapter_dropout).to(device) for m in modalities_all}

        adapter_args = None
        if adapter_ckpt_path and os.path.exists(adapter_ckpt_path):
            ckpt = torch.load(adapter_ckpt_path, map_location=device)
            ckpt_adapters = ckpt.get("text_adapters", {})
            for m, adp in adapters.items():
                if m in ckpt_adapters:
                    adp.load_state_dict(ckpt_adapters[m])
            adapter_args = ckpt.get("args", {})
            print(f"Loaded adapters from {adapter_ckpt_path}")
        else:
            print(f"Adapter checkpoint not found: {adapter_ckpt_path}. Using freshly initialized adapters.")

        for adp in adapters.values():
            adp.eval()
    else:
        adapters = None
        adapter_args = None

    from dataprocess_pipeline.dataset import build_plasma_text as build_plasma_text_ds, _default_config


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
        if use_legacy_mode:
            # 原始方案不使用此函数，返回 None
            return None
        common_text = modality_common_texts[modality]
        plasma_config = _default_config()["plasma_thresholds"]
        plasma_row = _pick_plasma(ptid, examdate)
        plasma_text = build_plasma_text_ds(plasma_row, plasma_config)
        text = f"[PLASMA]\n{plasma_text}\n[/PLASMA]\n[SEP]\n{common_text}"
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            base = bio_model.get_text_features(**inputs).squeeze(0)
        return adapters[modality](base).unsqueeze(0)

    # cosine similarity between modality templates could be computed here if needed

    # 2. 预处理数据，对患者信息生成Text Embeddings via BiomedCLIP
    mri_dir = os.path.join(base_dir, "MRI")
    av45_dir = os.path.join(base_dir, "PET_MNI", 'AV45')
    fdg_dir = os.path.join(base_dir, "PET_MNI", 'FDG')
    tau_dir = os.path.join(base_dir, "PET_MNI", 'TAU')
    csv_path = 'filtered_subjects_with_description.csv'

    # JSON 文件保存路径
    if use_legacy_mode:
        train_json_path = "./train_data_with_description.json"
        val_json_path = "./val_data_with_description.json"
    else:
        train_json_path = "adapter_finetune/json/train_data_with_description.json"
        val_json_path = "adapter_finetune/json/val_data_with_description.json"

    # 加载 CSV 文件（保持不变）
    csv_data = pd.read_csv(csv_path)
    csv_dict = csv_data.set_index("Subject ID")["Description"].to_dict()

    def _filter_pet_files(files):
        """过滤掉不需要的 PET 文件，仅保留主 nii.gz 文件。"""
        keep = []
        for f in files:
            name_lower = f.lower()
            if not name_lower.endswith(".nii.gz"):
                continue
            if "pet2mni" in name_lower or "full" in name_lower:
                continue
            keep.append(f)
        return keep

    # 获取文件列表（过滤 PET 的冗余文件）
    mri_files = sorted(os.listdir(mri_dir))
    av45_files = _filter_pet_files(sorted(os.listdir(av45_dir)))
    fdg_files = _filter_pet_files(sorted(os.listdir(fdg_dir)))
    tau_files = _filter_pet_files(sorted(os.listdir(tau_dir))) if os.path.isdir(tau_dir) else []



    # 使用统一的 get_subject_id 处理所有文件（保持不变）
    mri_dict = {get_subject_id(f): os.path.join(mri_dir, f) for f in mri_files}
    av45_dict = {get_subject_id(f): os.path.join(av45_dir, f) for f in av45_files}
    fdg_dict = {get_subject_id(f): os.path.join(fdg_dir, f) for f in fdg_files}
    tau_dict = {get_subject_id(f): os.path.join(tau_dir, f) for f in tau_files}

    # 匹配文件并加入描述信息和 Subject ID
    tau_available = len(tau_dict) > 0
    
    # Legacy 模式直接从 JSON 加载 paired_data（包含 old_descr 字段）
    if use_legacy_mode and os.path.exists(train_json_path):
        print(f"\n📂 Legacy 模式: 直接从 JSON 加载数据以获取 old_descr 字段")
        with open(train_json_path, "r") as f:
            train_data_json = json.load(f)
        with open(val_json_path, "r") as f:
            val_data_json = json.load(f)
        paired_data = train_data_json + val_data_json
        print(f"   从 JSON 加载了 {len(paired_data)} 条数据")
        # 检查 old_descr 字段
        sample_old_descr = paired_data[0].get("old_descr", None) if paired_data else None
        print(f"   示例 old_descr: {sample_old_descr[:80] if sample_old_descr else 'None'}...")
    else:
        # 非 Legacy 模式或 JSON 不存在时，从文件系统构建 paired_data
        paired_data = []
        for patient_id, mri_file in mri_dict.items():
            has_fdg = patient_id in fdg_dict
            has_av45 = patient_id in av45_dict
            has_tau = tau_available and patient_id in tau_dict

            # 只要存在任意 PET 即可纳入
            if not (has_fdg or has_av45 or has_tau):
                continue

            description = csv_dict.get(patient_id, None)  # 从 csv_dict 中获取 Description 信息
            entry = {
                "name": patient_id,  # 添加 name 字段
                "mri": mri_file,
                "av45": av45_dict.get(patient_id),
                "fdg": fdg_dict.get(patient_id),
                "description": description  # 加入 Description 信息
            }
            if tau_available:
                entry["tau"] = tau_dict.get(patient_id)
            paired_data.append(entry)
        
    if size_of_dataset:
        paired_data = paired_data[:size_of_dataset]  # 根据需要调整数据集大小
    print(f"Total matched pairs with description: {len(paired_data)}")
    # 在 paired_data 中新增键 fdg_index、av45_index、tau_index
    for idx, data in enumerate(paired_data):
        data["fdg_index"] = idx  # 将样本在 paired_data 中的索引作为 fdg_index
        data["av45_index"] = idx
        if tau_available:
            data["tau_index"] = idx

    # 保证每个被试都有文本（无描述时使用 NA）
    # 原始方案使用 old_descr，adapter 方案使用 description
    if use_legacy_mode:
        modal_information = [data.get("old_descr", "NA") or "NA" for data in paired_data]
        print(f"\n📝 原始方案: 使用 old_descr 字段，共 {len(modal_information)} 条")
    else:
        modal_information = [data.get("description", "NA") or "NA" for data in paired_data]

    modality_text_features = {}

    if use_legacy_mode:
        # ============ 原始方案：直接编码 old_descr，不使用 adapter ============
        print("\n🔄 编码 old_descr 文本特征...")
        text_inputs = processor(
            text=modal_information,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=256,
        ).to(device)
        with torch.no_grad():
            desc_text_features = bio_model.get_text_features(**text_inputs)
        print(f"   desc_text_features: {desc_text_features.shape}, dtype={desc_text_features.dtype}")
        
        # 原始方案不使用 template features
        fdg_template_features = None
        av45_template_features = None
        tau_template_features = None
        fdg_text_features = None
        av45_text_features = None
        tau_text_features = None
    else:
        # ============ Adapter 方案：使用 adapter 和 template features ============
        # 为每个样本生成对应模态的 template feature
        template_features = {m: [] for m in modalities_all if m != "TAU" or tau_available}
        for item in paired_data:
            ptid = item["name"]
            template_features["FDG"].append(encode_template("FDG", ptid))
            template_features["AV45"].append(encode_template("AV45", ptid))
            if tau_available:
                template_features["TAU"].append(encode_template("TAU", ptid))

        fdg_template_features = torch.cat(template_features["FDG"], dim=0) if template_features["FDG"] else None
        av45_template_features = torch.cat(template_features["AV45"], dim=0) if template_features["AV45"] else None
        tau_template_features = torch.cat(template_features["TAU"], dim=0) if tau_available and template_features["TAU"] else None

        for modality in modalities_all:
            texts_for_modality = [desc for desc in modal_information]
            text_inputs = processor(
                text=texts_for_modality,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=256,
            ).to(device)
            with torch.no_grad():
                base_feats = bio_model.get_text_features(**text_inputs)
            modality_text_features[modality] = base_feats
            print(f"{modality} text features: {base_feats.shape}, dtype={base_feats.dtype}")

        fdg_text_features = modality_text_features["FDG"]
        av45_text_features = modality_text_features["AV45"]
        tau_text_features = modality_text_features.get("TAU") if tau_available else None
        desc_text_features = None  # adapter 方案不使用此变量
    # 划分训练集和验证集
    train_data, val_data = train_test_split(paired_data, test_size=int(len(paired_data)*0.1), random_state=42)

    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")

    if not os.path.exists(train_json_path):
        # 保存到 JSON 文件
        with open(train_json_path, "w") as f:
            json.dump(train_data, f, indent=4)
        with open(val_json_path, "w") as f:
            json.dump(val_data, f, indent=4)
        print(f"Saved train data to: {train_json_path}")
        print(f"Saved validation data to: {val_json_path}")
    else:
        print(f"JSON files already exist. Skipping save step.")

    with open(train_json_path, "r") as f:
        train_data = json.load(f)
    print(f"Loaded {len(train_data)} training samples.")
    print("First training sample:", train_data[0])

    # 验证验证集
    with open(val_json_path, "r") as f:
        val_data = json.load(f)
    print(f"Loaded {len(val_data)} validation samples.")
    print("First validation sample:", val_data[0])

    from monai.data import CacheDataset, DataLoader
    import json

    import monai.transforms as mt

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

    class DebugShape(mt.MapTransform):
        """调试用：打印各键的数组形状与类型"""
        def __init__(self, keys):
            super().__init__(keys)
        def __call__(self, data):
            d = dict(data)
            try:
                for k in self.keys:
                    v = d.get(k)
                    if v is None:
                        print(f"[DebugShape] {k}: None")
                    else:
                        if hasattr(v, 'shape'):
                            print(f"[DebugShape] {k}: shape={v.shape}, dtype={getattr(v, 'dtype', type(v))}")
                        else:
                            print(f"[DebugShape] {k}: type={type(v)}")
            except Exception as e:
                print(f"[DebugShape] error: {e}")
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
                    import torch
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

    # 定义 transform 函数
    # 重要：transform 返回 CPU 张量，确保缓存可在不同 GPU 上加载
    if use_legacy_mode:
        # ============ 原始方案：context = desc_text_features[x] + modality_feature_optimized ============
        # 预先将特征移到 CPU，避免缓存包含 GPU 设备信息
        desc_text_features_cpu = desc_text_features.cpu()
        fdg_feature_optimized_cpu = fdg_feature_optimized.cpu()
        av45_feature_optimized_cpu = av45_feature_optimized.cpu()
        tau_feature_optimized_cpu = tau_feature_optimized.cpu()

        def fdg_index_transform(x):
            return torch.cat([desc_text_features_cpu[x].unsqueeze(0), fdg_feature_optimized_cpu], dim=0)

        def av45_index_transform(x):
            return torch.cat([desc_text_features_cpu[x].unsqueeze(0), av45_feature_optimized_cpu], dim=0)

        def tau_index_transform(x):
            return torch.cat([desc_text_features_cpu[x].unsqueeze(0), tau_feature_optimized_cpu], dim=0)
    else:
        # ============ Adapter 方案：context = text_features[x] + template_features[x] ============
        # 预先将特征移到 CPU，避免缓存包含 GPU 设备信息
        fdg_text_features_cpu = fdg_text_features.cpu()
        av45_text_features_cpu = av45_text_features.cpu()
        fdg_template_features_cpu = fdg_template_features.cpu()
        av45_template_features_cpu = av45_template_features.cpu()
        tau_text_features_cpu = tau_text_features.cpu() if tau_text_features is not None else None
        tau_template_features_cpu = tau_template_features.cpu() if tau_template_features is not None else None

        def fdg_index_transform(x):
            text_feat = fdg_text_features_cpu[x].unsqueeze(0)
            tmpl_feat = fdg_template_features_cpu[x].unsqueeze(0)
            return torch.cat([text_feat, tmpl_feat], dim=0)

        def av45_index_transform(x):
            text_feat = av45_text_features_cpu[x].unsqueeze(0)
            tmpl_feat = av45_template_features_cpu[x].unsqueeze(0)
            return torch.cat([text_feat, tmpl_feat], dim=0)

        def tau_index_transform(x):
            text_feat = tau_text_features_cpu[x].unsqueeze(0)
            tmpl_feat = tau_template_features_cpu[x].unsqueeze(0)
            return torch.cat([text_feat, tmpl_feat], dim=0)


    # 3. 加载处理好的数据
    # train_json_path = "./train_data_with_description.json"
    # val_json_path = "./val_data_with_description.json"

    # 加载 JSON 文件
    with open(train_json_path, "r") as f:
        train_data = json.load(f)

    with open(val_json_path, "r") as f:
        val_data = json.load(f)

    # 重新计算每个样本在 paired_data 中的位置
    name_to_idx = {d["name"]: i for i, d in enumerate(paired_data)}

    # 转换数据格式，确保所有索引都有有效值，并根据模态开关进行过滤
    # 注意：description 字段已通过索引转换为文本特征，此处不再需要
    # ★ 方案B：只在 data dict 中包含活跃模态，跳过关闭的模态以节省缓存空间
    def filter_and_format_data(data_list):
        formatted = []
        for item in data_list:
            av45_file = item["av45"] if use_av45 else None
            fdg_file = item["fdg"] if use_fdg else None
            tau_file = (item.get("tau") or (tau_dict.get(item["name"]) if tau_available else None)) if use_tau else None
            
            # 如果该被试没有任何选定的模态，则跳过（避免无用的 MRI 处理）
            if fdg_file is None and av45_file is None and tau_file is None:
                continue

            entry = {
                "name": item["name"],
                "mri": item["mri"],
            }
            idx = name_to_idx[item["name"]]
            if use_fdg:
                entry["fdg"] = fdg_file
                entry["fdg_index"] = idx
            if use_av45:
                entry["av45"] = av45_file
                entry["av45_index"] = idx
            if use_tau and tau_available:
                entry["tau"] = tau_file
                entry["tau_index"] = idx
            formatted.append(entry)
        return formatted

    train_data = filter_and_format_data(train_data)
    val_data = filter_and_format_data(val_data)
    print(f"\n🔍 根据模态开关过滤后，使用数据量: train={len(train_data)}, val={len(val_data)}")

    # 确保缺失的 TAU 索引被填充
    if use_tau and tau_available:
        for item in train_data:
            if item.get("tau_index") is None:
                item["tau_index"] = name_to_idx.get(item["name"], 0)
        for item in val_data:
            if item.get("tau_index") is None:
                item["tau_index"] = name_to_idx.get(item["name"], 0)

    # ★ 方案B：pet_keys 只包含活跃模态
    pet_keys = ["mri"]
    if use_fdg:
        pet_keys.append("fdg")
    if use_av45:
        pet_keys.append("av45")
    if use_tau and tau_available:
        pet_keys.append("tau")

    # ★ 方案B：VolumesToFloat16 将体积转为 float16 缓存，节省约 50% 空间
    class VolumesToFloat16(mt.Transform):
        """将体积 tensor 转为 float16，节省约 50% 缓存空间（最大误差 < 0.025%）。
        放在 PersistentDataset transform pipeline 末尾（ScaleIntensityd 之后）。
        DataLoader 加载后需调用 .float() 转回 float32 再送入模型/损失函数。
        """
        def __call__(self, data):
            d = dict(data)
            for k, v in d.items():
                if isinstance(v, torch.Tensor) and v.numel() > 1000:
                    d[k] = v.half()
            return d

    # 构建 index_transform 列表（动态，基于模态开关）
    index_transforms = []
    if use_fdg:
        index_transforms.append(mt.Lambdad(keys=["fdg_index"], func=fdg_index_transform))
    if use_av45:
        index_transforms.append(mt.Lambdad(keys=["av45_index"], func=av45_index_transform))
    if use_tau and tau_available:
        index_transforms.append(mt.Lambdad(keys=["tau_index"], func=tau_index_transform))

    # 构建数据增强 pipeline
    # ★ 方案B：只处理活跃模态 + float16 缓存
    shared_transforms = [
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
        VolumesToFloat16(),
    ] + index_transforms

    train_transforms = mt.Compose(shared_transforms)
    val_transforms = mt.Compose(shared_transforms)

    # TODO: 保存为 HDF5 格式以加快加载速度
    # 保存 train 和 val
    # save_to_hdf5(train_data, train_transforms, os.path.join(h5_dir, "train.h5"))
    # save_to_hdf5(val_data, val_transforms, os.path.join(h5_dir, "val.h5"))

    # 创建 DataLoader
    # train_ds_h5 = HDF5Dataset(os.path.join(h5_dir, "train.h5"))
    # train_loader_h5 = DataLoader(train_ds_h5, batch_size=1, shuffle=True, num_workers=0)
    #
    # val_ds = HDF5Dataset(os.path.join(h5_dir, "val.h5"))
    # val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # 构建 PersistentDataset（会自动缓存）
    train_cache_dir = os.path.join(cache_dir, "train")
    val_cache_dir = os.path.join(cache_dir, "val")
    
    print(f"\n📦 创建训练集 PersistentDataset...")
    print(f"   数据量: {len(train_data)} 样本")
    print(f"   缓存目录: {train_cache_dir}")
    train_ds = PersistentDataset(data=train_data, transform=train_transforms, cache_dir=train_cache_dir)
    
    # # ============ 暂时注释：缓存已在后台生成 ============
    # print(f"\n⭐ 开始生成训练集缓存（共 {len(train_ds)} 个样本）...")
    # from tqdm import tqdm as tqdm_module
    # for i in tqdm_module(range(len(train_ds)), desc="训练集缓存"):
    #     try:
    #         _ = train_ds[i]
    #     except Exception as e:
    #         print(f"\n❌ 样本 {i} 缓存失败: {e}")
    #         raise
    # print("✅ 训练集缓存生成完成！")
    
    # 检查缓存文件
    cache_files = os.listdir(train_cache_dir)
    print(f"   已有缓存文件数: {len(cache_files)}")
    if len(cache_files) > 0:
        print(f"   缓存文件示例: {cache_files[0]}")
    
    # num_workers > 0 可以在后台并行加载缓存数据，避免 I/O 阻塞 GPU 训练
    # pin_memory=True 可以加速 CPU->GPU 数据传输
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

    print(f"\n📦 创建验证集 PersistentDataset...")
    print(f"   数据量: {len(val_data)} 样本")
    print(f"   缓存目录: {val_cache_dir}")
    val_ds = PersistentDataset(data=val_data, transform=val_transforms, cache_dir=val_cache_dir)
    
    # ============ 暂时注释：缓存已在后台生成 ============
    # print(f"\n⭐ 开始生成验证集缓存（共 {len(val_ds)} 个样本）...")
    # for i in tqdm_module(range(len(val_ds)), desc="验证集缓存"):
    #     try:
    #         _ = val_ds[i]
    #     except Exception as e:
    #         print(f"\n❌ 样本 {i} 缓存失败: {e}")
    #         raise
    # print("✅ 验证集缓存生成完成！")
    
    # 检查缓存文件
    cache_files = os.listdir(val_cache_dir)
    print(f"   已有缓存文件数: {len(cache_files)}")
    if len(cache_files) > 0:
        print(f"   缓存文件示例: {cache_files[0]}")
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from generative.networks.nets import DiffusionModelUNet
    from generative.networks.schedulers import DDPMScheduler,DDIMScheduler
    from generative.inferers import DiffusionInferer

    # 4. 加载网络
    if len(device_id) == 1:
        model= DiffusionModelUNet(
            spatial_dims=3,
            in_channels=1,#1,
            out_channels=1,
            num_channels=(32,64,64,128),
            attention_levels=(False,False,False,True),
            num_res_blocks=1,
            num_head_channels=(0,0,0, 128),
            with_conditioning=True,
            cross_attention_dim=512,
            use_flash_attention=True,
        )
        model.to(device)
    elif len(device_id) ==2:
        model= DistributedDiffusionModelUNet(
            spatial_dims=3,
            in_channels=1,#1,
            out_channels=1,
            num_channels=(32,64,64,128),
            attention_levels=(False,False,False,True),
            num_res_blocks=1,
            num_head_channels=(0,0,0, 128),
            with_conditioning=True,
            cross_attention_dim=512,
            use_flash_attention=True,
            device_ids=device_id
        )
        print(f"✅ 双GPU模式: 模型分布在 GPU {device_id[0]} 和 GPU {device_id[1]}")
    else:
        raise ValueError("Currently only support 1 or 2 GPU(s) for training.")
    
    # 标记是否使用分布式模型（双GPU）
    use_distributed = len(device_id) == 2
    
    optimizer= torch.optim.Adam(params=model.parameters(), lr=2.5e-5)
    start_epoch = 0

    # 加载上次训练的检查点（如果有的话）
    if last_epoch:
        # Load the checkpoint
        # checkpoint_dir = '/home/ssddata/liutuo/checkpoint/mri2pet _two trace_add clip_flow_noHistogramNormalized'
        checkpoint = torch.load(f'{checkpoint_dir}/first_part_{last_epoch}.pth', map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch after the last saved one
        val_interval = 5  # Keep validation interval unchanged

    scheduler = DDPMScheduler(prediction_type="v_prediction", num_train_timesteps=1000,schedule="scaled_linear_beta",
                              beta_start=0.0005, beta_end=0.0195)
    scheduler.set_timesteps(num_inference_steps=1000)
    epoch_loss_list = []
    val_epoch_loss_list = []
    # 显式指定 GradScaler 使用的设备，防止默认使用 cuda:0
    scaler = GradScaler(device=device)
    global_step = 0  # 全局步数计数器（用于 TensorBoard 日志）
    best_val_loss = float('inf')  # ★ 最佳验证损失，用于保存 best_model.pt

    # ★ 提前初始化验证指标，避免内嵌定义
    from generative.metrics import SSIMMetric
    from monai.metrics import MAEMetric, MSEMetric, PSNRMetric
    ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=7)
    psnr_metric = PSNRMetric(1.0)

    # 5. 训练网络
    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, data in progress_bar:
            # ★ 方案B：fp16 缓存加载后 .float() 转回 float32 再送入模型
            images = data["mri"].float().to(device, non_blocking=True)
            seg_fdg = data["fdg"].float().to(device, non_blocking=True) if "fdg" in data else None
            seg_av45 = data["av45"].float().to(device, non_blocking=True) if "av45" in data else None
            fdg_index = data["fdg_index"].to(device, non_blocking=True) if "fdg_index" in data else None
            av45_index = data["av45_index"].to(device, non_blocking=True) if "av45_index" in data else None
            if tau_available and "tau" in data:
                seg_tau = data["tau"].float().to(device, non_blocking=True)
                tau_index = data["tau_index"].to(device, non_blocking=True)
            else:
                seg_tau = None
                tau_index = None

            # ============ 梯度累积：每 accumulation_steps 步才清零梯度 ============
            if step % accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            
            timesteps = torch.randint(0, 1000, (len(images),)).to(device)  # 保留兼容（未使用）

            # Create time_embedding
            time_embedding = torch.randint(
                0, 1000, (images.shape[0],), device=images.device
            ).long()

            if epoch >= 140:
                time_embedding = torch.tensor([0], device=images.device, dtype=torch.long)

            # Create time - 使用 detach() 确保 t 不参与梯度计算
            # 扩展 t 的维度以支持 batch_size > 1 的广播: (B,) -> (B, 1, 1, 1, 1)
            with torch.no_grad():
                t = (time_embedding.float() / 1000).view(-1, 1, 1, 1, 1)

            # 检查 FDG/AV45/TAU 数据是否为二值化（只有 0 和 1）
            has_fdg = seg_fdg is not None and not torch.all(seg_fdg == 0)
            has_av45 = seg_av45 is not None and not torch.all(seg_av45 == 0)
            has_tau = tau_available and seg_tau is not None and not torch.all(seg_tau == 0)

            # 默认损失为 0
            total_loss = 0.0
            
            # 收集所有可用模态
            available_modalities = []
            if has_fdg:
                available_modalities.append('fdg')
            if has_av45:
                available_modalities.append('av45')
            if has_tau:
                available_modalities.append('tau')
            
            # 随机选择一个模态进行训练
            if len(available_modalities) > 0:
                selected_modality = random.choice(available_modalities)
                
                if selected_modality == 'fdg':
                    with autocast(device_type='cuda', enabled=True):
                        x_t_fdg = t * seg_fdg + (1 - t) * images
                        v_fdg_prediction = model(x=x_t_fdg, timesteps=time_embedding, context=fdg_index)
                        # 双GPU模式下，模型输出在device0，确保ground truth也在同一设备
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
                        # 双GPU模式下，模型输出在device0，确保ground truth也在同一设备
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
                        # 双GPU模式下，模型输出在device0，确保ground truth也在同一设备
                        if use_distributed:
                            v_tau_prediction = v_tau_prediction.to(device)
                        v_tau = seg_tau - images
                        loss_tau = F.mse_loss(v_tau.float(), v_tau_prediction.float())
                        loss_tau = (gamma * loss_tau) / accumulation_steps
                    
                    scaler.scale(loss_tau).backward()
                    total_loss = loss_tau.item() * accumulation_steps
                    del x_t_tau, v_tau_prediction, v_tau, loss_tau
                
                # 注意：移除了 torch.cuda.empty_cache()，它会导致 CUDA 同步，严重影响性能
                # 只在显存不足时才考虑启用（可以在每 N 个 step 调用一次）
            
            # ============ 梯度累积：每 accumulation_steps 步才更新参数 ============
            # 只有在至少有一个模态参与训练时才更新优化器
            if (step + 1) % accumulation_steps == 0 and total_loss > 0:
                scaler.step(optimizer)
                scaler.update()

            epoch_loss += total_loss
            global_step += 1

            # ============ TensorBoard 日志记录 ============
            if log_every > 0 and global_step % log_every == 0:
                avg_loss = epoch_loss / (step + 1)
                lr = optimizer.param_groups[0].get("lr", 2.5e-5)
                writer.add_scalar("train/loss", avg_loss, global_step)
                writer.add_scalar("train/step_loss", total_loss, global_step)
                writer.add_scalar("optim/lr", lr, global_step)
                writer.add_scalar("train/epoch", epoch, global_step)
                # 记录当前选择的模态
                if len(available_modalities) > 0:
                    writer.add_scalar(f"train/modality_{selected_modality}", 1.0, global_step)

            progress_bar.set_postfix({
                "loss": epoch_loss / (step + 1),
                "accum": f"{(step % accumulation_steps) + 1}/{accumulation_steps}"
            })

            # 移除了 torch.cuda.empty_cache()，避免每个 batch 都进行 CUDA 同步

        # Epoch 结束后记录平均损失
        epoch_avg_loss = epoch_loss / (step + 1)
        epoch_loss_list.append(epoch_avg_loss)
        writer.add_scalar("train/epoch_loss", epoch_avg_loss, epoch)
        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_epoch_loss = 0
            # ★ 逐 batch 累积 —— 解决原代码仅用最后一个 batch 计算指标的问题
            ssim_tau_list, psnr_tau_list, mae_tau_list = [], [], []

            for step, data_val in enumerate(val_loader):
                # ★ 方案B：fp16 缓存加载后 .float() 转回 float32
                images = data_val["mri"].float().to(device, non_blocking=True)
                seg_fdg = data_val["fdg"].float().to(device, non_blocking=True) if "fdg" in data_val else None
                seg_av45 = data_val["av45"].float().to(device, non_blocking=True) if "av45" in data_val else None
                fdg_index = data_val["fdg_index"].to(device, non_blocking=True) if "fdg_index" in data_val else None
                av45_index = data_val["av45_index"].to(device, non_blocking=True) if "av45_index" in data_val else None
                if tau_available and "tau" in data_val:
                    seg_tau = data_val["tau"].float().to(device, non_blocking=True)
                    tau_index = data_val["tau_index"].to(device, non_blocking=True)
                else:
                    seg_tau = None
                    tau_index = None

                has_fdg = seg_fdg is not None and not torch.all(seg_fdg == 0)
                has_av45 = seg_av45 is not None and not torch.all(seg_av45 == 0)
                has_tau = tau_available and seg_tau is not None and not torch.all(seg_tau == 0)

                x_t = images
                N_sample = 1
                N_sample_tensor = torch.tensor(N_sample, dtype=torch.float32, device=device)

                progress_bar_val = [(i / N_sample) for i in range(N_sample)]
                for t in progress_bar_val:  # go through the noising process
                    with autocast(device_type='cuda', enabled=False):
                        with torch.no_grad():
                            time_embedding = int(t * 1000)

                            # FDG 输出（仅当非二值化时计算）
                            if has_fdg:
                                v_fdg_output = model(x=x_t, timesteps=torch.Tensor((time_embedding,)).to(device), context=fdg_index)
                                # 双 GPU模式下确保输出在正确设备
                                if use_distributed:
                                    v_fdg_output = v_fdg_output.to(device)
                                x_fdg_t = x_t + (v_fdg_output / N_sample_tensor)
                                x_fdg_t = torch.clamp(x_fdg_t, min=clip_sample_min, max=clip_sample_max)
                            else:
                                x_fdg_t = None  # 跳过 FDG 计算

                            # AV45 输出（仅当非二值化时计算）
                            if has_av45:
                                v_av45_output = model(x=x_t, timesteps=torch.Tensor((time_embedding,)).to(device), context=av45_index)
                                # 双 GPU模式下确保输出在正确设备
                                if use_distributed:
                                    v_av45_output = v_av45_output.to(device)
                                x_av45_t = x_t + (v_av45_output / N_sample_tensor)
                                x_av45_t = torch.clamp(x_av45_t, min=clip_sample_min, max=clip_sample_max)
                            else:
                                x_av45_t = None  # 跳过 AV45 计算

                            # TAU 输出（仅当非二值化时计算）
                            if has_tau:
                                v_tau_output = model(x=x_t, timesteps=torch.Tensor((time_embedding,)).to(device), context=tau_index)
                                # 双 GPU模式下确保输出在正确设备
                                if use_distributed:
                                    v_tau_output = v_tau_output.to(device)
                                x_tau_t = x_t + (v_tau_output / N_sample_tensor)
                                x_tau_t = torch.clamp(x_tau_t, min=clip_sample_min, max=clip_sample_max)
                            else:
                                x_tau_t = None

                # 默认损失为 0
                val_fdg_loss = torch.tensor(0.0, device=device)
                val_av45_loss = torch.tensor(0.0, device=device)
                val_tau_loss = torch.tensor(0.0, device=device)

                # 计算 FDG 损失（仅当非二值化时计算）
                if has_fdg and x_fdg_t is not None:
                    val_fdg_loss = F.mse_loss(x_fdg_t.float(), seg_fdg.float())

                # 计算 AV45 损失（仅当非二值化时计算）
                if has_av45 and x_av45_t is not None:
                    val_av45_loss = F.mse_loss(x_av45_t.float(), seg_av45.float())

                if has_tau and x_tau_t is not None:
                    val_tau_loss = F.mse_loss(x_tau_t.float(), seg_tau.float())
                    # ★ 逐 batch 累积 TAU 指标
                    ssim_tau_list.append(ssim_metric(seg_tau.cpu(), x_tau_t.cpu()).mean().item())
                    psnr_tau_list.append(psnr_metric(seg_tau.cpu(), x_tau_t.cpu()).mean().item())
                    mae_tau_list.append(F.l1_loss(x_tau_t.float(), seg_tau.float()).item())

                # 加权总损失
                val_loss = alpha * val_fdg_loss + beta * val_av45_loss + gamma * val_tau_loss

                val_epoch_loss += val_loss.item()
                # 移除了 torch.cuda.empty_cache()，验证阶段不需要频繁清理

            val_avg_loss = val_epoch_loss / (step + 1)
            print(f"Epoch {epoch + 1} | Validation loss {val_avg_loss:.4f}")
            val_epoch_loss_list.append(val_avg_loss)

            # ★ 全验证集聚合指标输出
            if ssim_tau_list:
                tau_ssim_avg = sum(ssim_tau_list) / len(ssim_tau_list)
                tau_psnr_avg = sum(psnr_tau_list) / len(psnr_tau_list)
                tau_mae_avg  = sum(mae_tau_list)  / len(mae_tau_list)
                print(f"  TAU SSIM={tau_ssim_avg:.4f}  PSNR={tau_psnr_avg:.2f}  MAE={tau_mae_avg:.4f}  (n={len(ssim_tau_list)} batches)")
                writer.add_scalar("val/TAU_SSIM", tau_ssim_avg, epoch)
                writer.add_scalar("val/TAU_PSNR", tau_psnr_avg, epoch)
                writer.add_scalar("val/TAU_MAE",  tau_mae_avg,  epoch)

            # 验证集日志记录
            writer.add_scalar("val/loss", val_avg_loss, epoch)
            print('epoch:', epoch + 1)
            print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])

            # ★ Best model 跟踪并保存
            if val_avg_loss < best_val_loss:
                best_val_loss = val_avg_loss
                best_ckpt_path = run_dir / "best_model.pt"
                checkpoint_best = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step,
                    'val_loss': best_val_loss,
                }
                torch.save(checkpoint_best, str(best_ckpt_path))
                print(f"🏆 Best model saved (val_loss={best_val_loss:.4f}): {best_ckpt_path}")
            writer.add_scalar("val/best_loss", best_val_loss, epoch)

            # 保存 checkpoint 到 run_dir
            ckpt_path = run_dir / f"ckpt_epoch{epoch + 1}.pt"
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
            }
            torch.save(checkpoint, str(ckpt_path))
            print(f"✅ Checkpoint saved: {ckpt_path}")

            # 也保存到 checkpoint_dir（保持兼容）
            torch.save(checkpoint, f'{checkpoint_dir}/first_part_{epoch + 1}.pth')
            print('Saved all parameters!\n')

            # 可视化和评估指标
            current_fdg_img = x_fdg_t.to('cpu') if x_fdg_t is not None else None
            current_av45_img = x_av45_t.to('cpu') if x_av45_t is not None else None
            current_tau_img = x_tau_t.to('cpu') if tau_available and x_tau_t is not None else None
            labels_fdg = seg_fdg.to('cpu') if seg_fdg is not None else None
            labels_av45 = seg_av45.to('cpu') if seg_av45 is not None else None
            labels_tau = seg_tau.to('cpu') if tau_available and seg_tau is not None else None
            mri_cpu = images.to('cpu')

            # 如果存在非二值化的 FDG 或 AV45 数据，则进行可视化和评估
            if current_fdg_img is not None or current_av45_img is not None or current_tau_img is not None:
                # 构建可视化列表，过滤掉 None 的图像
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
            
            # ============ 图像记录到 TensorBoard（每 image_log_interval 个 epoch 保存一次）============
            if (epoch + 1) % image_log_interval == 0:
                print(f"📷 保存过程图像到 TensorBoard (epoch {epoch + 1})...")
                
                # 构建对比图像字典
                comparison_volumes = {"MRI": mri_cpu}
                
                if has_fdg and current_fdg_img is not None:
                    comparison_volumes["FDG_GT"] = labels_fdg
                    comparison_volumes["FDG_Pred"] = current_fdg_img
                    # 计算差异图
                    import torch
                    fdg_diff = torch.abs(current_fdg_img - labels_fdg)
                    comparison_volumes["FDG_Diff"] = fdg_diff
                
                if has_av45 and current_av45_img is not None:
                    comparison_volumes["AV45_GT"] = labels_av45
                    comparison_volumes["AV45_Pred"] = current_av45_img
                    av45_diff = torch.abs(current_av45_img - labels_av45)
                    comparison_volumes["AV45_Diff"] = av45_diff
                
                if tau_available and has_tau and current_tau_img is not None:
                    comparison_volumes["TAU_GT"] = labels_tau
                    comparison_volumes["TAU_Pred"] = current_tau_img
                    tau_diff = torch.abs(current_tau_img - labels_tau)
                    comparison_volumes["TAU_Diff"] = tau_diff
                
                # 记录综合对比图
                log_comparison_figure(writer, f"val/comparison_epoch{epoch+1}", comparison_volumes, epoch)
                
                # 单独记录各模态的切片图像
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

    # 训练结束，关闭 writer
    writer.flush()
    writer.close()
    print(f"\n🎉 训练完成！日志保存在: {run_dir}")


if __name__ == "__main__":
    main()
