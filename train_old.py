#%%
import pandas as pd
from report_error import email_on_error
from sklearn.model_selection import train_test_split
from typing import Dict
import nibabel as nib
import numpy as np
import torch.multiprocessing as mp
from generative.networks.nets.diffusion_model_unet import DistributedDiffusionModelUNet
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
# 设置多进程启动方式为'spawn'，避免CUDA在fork进程中初始化问题
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # 已经设置过了


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

    # 0.训练参数设置
    base_dir = "/mnt/nfsdata/nfsdata/ADNI/ADNI0103/Coregistration"
    cache_dir = '/mnt/nfsdata/nfsdata/lsj.14/ADNI_cache_tem'  # 使用新缓存目录避免GPU张量问题
    plasma_csv_path = "adapter_finetune/UPENN_PLASMA_FUJIREBIO_QUANTERIX_21Dec2025.csv"
    adapter_ckpt_path = "/mnt/nfsdata/nfsdata/lsj.14/replicaLT/adapter_finetune/runs/01.19_2945992/ckpt_epoch400.pt"  # 根据需要替换
    train_json_path = "adapter_finetune/json/train_data_with_description.json"
    val_json_path = "adapter_finetune/json/val_data_with_description.json"
    csv_path = '/mnt/nfsdata/nfsdata/lsj.14/replicaLT/adapter_finetune/data_csv/pairs_with_demog.csv'

    # 确保缓存目录存在
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "val"), exist_ok=True)
    print(f"✅ 缓存目录已创建: {cache_dir}")
    
    # 多GPU配置：使用2块GPU进行模型并行
    # 注意：这里的device_id是物理GPU编号（不使用CUDA_VISIBLE_DEVICES）
    # 根据当前空闲显存选择: GPU 3, 5, 6 有24GB空闲
    device_id = [5]  
    # 定义裁剪的最小值和最大值
    clip_sample_min = 0  # 设置合适的最小值
    clip_sample_max = 1   # 设置合适的最大值

    # 定义模态权重
    alpha = 1.0  # FDG 损失权重
    beta = 1.0   # AV45 损失权重
    gamma = 1.0  # TAU 损失权重
    
    # ============ GPU利用率优化配置 ============
    batch_size = 1  # 双GPU模型并行时使用batch_size=1避免OOM
    num_workers = 0  # 暂时使用单进程，避免pickle问题（NFS缓存已足够快）
    prefetch_factor = None  # num_workers=0时不需要
    accumulation_steps = 2  # 梯度累积步数：batch_size=1时设为2模拟batch_size=2
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
    n_epochs = 200
    val_interval =10
    checkpoint_dir = './checkpoint'
    logdir = './runs'  # TensorBoard 日志目录
    log_every = 1  # 每隔多少步记录一次日志
    image_log_interval = 10  # 每隔多少个 epoch 保存一次过程图像（防止占用大量空间）
    run_name = None  # 自动生成 run_name

    # Adapter 配置
    adapter_hidden = 512
    adapter_dropout = 0.1

    # 是否继续上次训练
    last_epoch = False # 设置为 True 并指定 resume_checkpoint 路径以继续训练
    resume_checkpoint = None  # 例如: "runs/01.19_2994993/ckpt_epoch10.pt" 或 "./checkpoint/first_part_10.pth"

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
    # 生成 run 名：优先用户指定；否则使用 日期_进程号，如 01.15_3716967
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
        "base_dir": base_dir,
        "cache_dir": cache_dir,
        "device_id": device_id,
        "batch_size": batch_size,
        "accumulation_steps": accumulation_steps,
        "num_workers": num_workers,
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
        "log_every": log_every,
        "image_log_interval": image_log_interval,
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

    from adapter_finetune.dataset import build_plasma_text as build_plasma_text_ds, _default_config


    def _load_plasma_table(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        if "examdate" in df.columns:
            df["examdate"] = pd.to_datetime(df["examdate"], errors="coerce")
        return df

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

    # cosine similarity between modality templates could be computed here if needed

    plasma_table = _load_plasma_table(plasma_csv_path)

    # 2. 预处理数据，对患者信息生成Text Embeddings via BiomedCLIP
    mri_dir = os.path.join(base_dir, "MRI")
    av45_dir = os.path.join(base_dir, "PET_MNI", 'AV45')
    fdg_dir = os.path.join(base_dir, "PET_MNI", 'FDG')
    tau_dir = os.path.join(base_dir, "PET_MNI", 'TAU')

    # JSON 文件保存路径（保持不变）


    # 加载 CSV 文件（保持不变）
    csv_data = pd.read_csv(csv_path)
    csv_dict = csv_data.set_index("subject_id")["description"].to_dict()

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
    paired_data = []
    tau_available = len(tau_dict) > 0
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
    modal_information = [data.get("description", "NA") or "NA" for data in paired_data]

    # 为每个样本生成对应模态的 template feature
    template_features = {m: [] for m in modalities_all if m != "TAU" or tau_available}
    for item in paired_data:
        ptid = item["name"]
        template_features["FDG"].append(encode_template("FDG", ptid))
        template_features["AV45"].append(encode_template("AV45", ptid))
        if tau_available:
            template_features["TAU"].append(encode_template("TAU", ptid))

    # 将 template features 移到 CPU 以避免 DataLoader collate 时的设备不匹配问题
    fdg_template_features = torch.cat(template_features["FDG"], dim=0).cpu().detach() if template_features["FDG"] else None
    av45_template_features = torch.cat(template_features["AV45"], dim=0).cpu().detach() if template_features["AV45"] else None
    tau_template_features = torch.cat(template_features["TAU"], dim=0).cpu().detach() if tau_available and template_features["TAU"] else None

    # 为每个样本生成个人描述文本的特征向量，各模态通用
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
    


    if not os.path.exists(train_json_path):
        # 划分训练集和验证集
        train_data, val_data = train_test_split(paired_data, test_size=int(len(paired_data)*0.1), random_state=42)

        print(f"Training set size: {len(train_data)}")
        print(f"Validation set size: {len(val_data)}")
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

    from monai.data import CacheDataset, DataLoader, pad_list_data_collate
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


    # 转换数据格式，确保所有索引都有有效值
    # 注意：description 字段已通过索引转换为文本特征，此处不再需要
    train_data = [
        {
            "name": item["name"],
            "mri": item["mri"],
            "av45": item["av45"],
            "fdg": item["fdg"],
            "tau": item['tau'],
            "fdg_index": item.get("fdg_index"),
            "av45_index": item.get("av45_index"),
            "tau_index": item.get("tau_index"),
        }
        for idx, item in enumerate(train_data)
    ]

    val_data = [
        {
            "name": item["name"],
            "mri": item["mri"],
            "av45": item["av45"],
            "fdg": item["fdg"],
            "tau": item['tau'],
            "fdg_index": item.get("fdg_index"),
            "av45_index": item.get("av45_index"),
            "tau_index": item.get("tau_index"),
        }
        for idx, item in enumerate(val_data)
    ]

    pet_keys = ["mri", "av45", "fdg"] + (["tau"] if tau_available else [])

    # 构建数据增强 pipeline
    # 定义训练集数据增强流程
    train_transforms = mt.Compose([
        mt.Lambdad(keys=pet_keys, func=mat_load),  # 加载 NIfTI 文件
        # ReduceTo3D(keys=pet_keys, reduce='mean'),
        # DebugShape(keys=pet_keys),
        FillMissingPET(keys=pet_keys, ref_key="mri"),
        # DebugShape(keys=pet_keys),
        mt.CropForegroundd(keys=pet_keys, source_key="mri"),
        RemoveCropForegroundMeta(),  # 移除 CropForegroundd 添加的元数据，避免 collate 错误
        # DebugShape(keys=pet_keys),
        mt.EnsureChannelFirstd(keys=pet_keys, channel_dim='no_channel'),
        mt.HistogramNormalized(keys=["mri"]),
        mt.ResizeWithPadOrCropd(keys=pet_keys, spatial_size=[128, 128, 128]),  # 减小尺寸节省显存
        mt.Spacingd(keys=pet_keys, pixdim=(1.0, 1.0, 1.0)),
        mt.NormalizeIntensityd(keys=pet_keys),
        mt.ScaleIntensityd(keys=pet_keys),
        mt.Lambdad(keys=["fdg_index"], func=fdg_index_transform),  # 添加 fdg_index 转换
        mt.Lambdad(keys=["av45_index"], func=av45_index_transform),  # 添加 av45_index 转换
        mt.Lambdad(keys=["tau_index"], func=tau_index_transform) if tau_available else mt.Identity(),
        ConvertToTensor(),  # 将所有 MetaTensor 转换为普通 Tensor，避免 batch_size > 1 时 collate 报错
    ])

    # 定义验证集数据增强流程（通常与训练集一致，但不含随机性增强）
    val_transforms = mt.Compose([
        mt.Lambdad(keys=pet_keys, func=mat_load),
        # ReduceTo3D(keys=pet_keys, reduce='mean'),
        # DebugShape(keys=pet_keys),
        FillMissingPET(keys=pet_keys, ref_key="mri"),
        # DebugShape(keys=pet_keys),
        mt.CropForegroundd(keys=pet_keys, source_key="mri"),
        RemoveCropForegroundMeta(),  # 移除 CropForegroundd 添加的元数据，避免 collate 错误
        # DebugShape(keys=pet_keys),
        mt.EnsureChannelFirstd(keys=pet_keys, channel_dim='no_channel'),
        mt.HistogramNormalized(keys=["mri"]),
        mt.ResizeWithPadOrCropd(keys=pet_keys, spatial_size=[128, 128, 128]),  # 减小尺寸节省显存
        mt.Spacingd(keys=pet_keys, pixdim=(1.0, 1.0, 1.0)),
        mt.NormalizeIntensityd(keys=pet_keys),
        mt.ScaleIntensityd(keys=pet_keys),
        mt.Lambdad(keys=["fdg_index"], func=fdg_index_transform),
        mt.Lambdad(keys=["av45_index"], func=av45_index_transform),
        mt.Lambdad(keys=["tau_index"], func=tau_index_transform) if tau_available else mt.Identity(),
        ConvertToTensor(),  # 将所有 MetaTensor 转换为普通 Tensor，避免 batch_size > 1 时 collate 报错
    ])

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
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=False,  # MONAI MetaTensor 不支持 pin_memory
        collate_fn=pad_list_data_collate  # 使用 MONAI 的 collate 函数处理混合类型
    )

    print(f"\n📦 创建验证集 PersistentDataset...")
    print(f"   数据量: {len(val_data)} 样本")
    print(f"   缓存目录: {val_cache_dir}")
    val_ds = PersistentDataset(data=val_data, transform=val_transforms, cache_dir=val_cache_dir)
    
    # # ============ 暂时注释：缓存已在后台生成 ============
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
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=False,  # MONAI MetaTensor 不支持 pin_memory
        collate_fn=pad_list_data_collate  # 使用 MONAI 的 collate 函数处理混合类型
    )

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
    scaler = GradScaler()
    global_step = 0  # 全局步数计数器
    start_epoch = 0

    # 加载上次训练的检查点（如果有的话）
    if last_epoch and resume_checkpoint:
        print(f"\n📂 正在从检查点恢复训练: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        
        # 兼容新旧两种checkpoint格式
        if "model" in checkpoint:
            model.load_state_dict(checkpoint['model'])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        if "global_step" in checkpoint:
            global_step = checkpoint['global_step']
        
        # 获取起始epoch
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"✅ 检查点加载成功！从 epoch {start_epoch} 继续训练，global_step={global_step}")

    scheduler = DDPMScheduler(prediction_type="v_prediction", num_train_timesteps=1000,schedule="scaled_linear_beta",
                              beta_start=0.0005, beta_end=0.0195)
    scheduler.set_timesteps(num_inference_steps=1000)
    epoch_loss_list = []
    val_epoch_loss_list = []

    # ============ 3D 体积切片可视化辅助函数 ============
    def get_3d_slices(volume, normalize=True):
        """
        从3D体积中提取三个正交切面（axial, coronal, sagittal）用于可视化。
        Args:
            volume: 形状为 (B, C, H, W, D) 或 (C, H, W, D) 或 (H, W, D) 的张量
            normalize: 是否归一化到 [0, 1]
        Returns:
            dict: 包含三个切面的字典
        """
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

    # 定义保存checkpoint的函数
    def save_checkpoint(tag: str, epoch: int, global_step: int) -> None:
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "hparams": hparams,
        }
        ckpt_path = run_dir / f"ckpt_{tag}.pt"
        torch.save(ckpt, ckpt_path)
        # 同时保存到 checkpoint_dir 以保持兼容性
        os.makedirs(checkpoint_dir, exist_ok=True)
        legacy_path = f'{checkpoint_dir}/first_part_{epoch + 1}.pth'
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, legacy_path)
        print(f"✅ Checkpoint saved: {ckpt_path} & {legacy_path}")

    # 5. 训练网络
    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(total=len(train_loader), ncols=70, desc=f"Epoch {epoch}")

        for step, data in enumerate(train_loader):
            # 使用 .clone().detach() 确保所有张量完全独立，切断与 PersistentDataset 缓存的计算图连接
            images = data["mri"].clone().detach().to(device)
            seg_fdg = data["fdg"].clone().detach().to(device)  # this is the ground truth segmentation
            seg_av45 = data["av45"].clone().detach().to(device)
            fdg_index = data["fdg_index"].clone().detach().to(device)
            av45_index = data["av45_index"].clone().detach().to(device)
            if tau_available:
                seg_tau = data["tau"].clone().detach().to(device)
                tau_index = data["tau_index"].clone().detach().to(device)
            else:
                seg_tau = None
                tau_index = None

            # ============ 梯度累积：每 accumulation_steps 步才清零梯度 ============
            if step % accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            
            timesteps = torch.randint(0, 1000, (len(images),)).to(device)  # pick a random time step t

            # Create time_embedding
            time_embedding = torch.randint(
                0, 1000, (images.shape[0],), device=images.device
            ).long()

            if epoch >= 140:
                time_embedding = torch.tensor([0], device=images.device, dtype=torch.long)

            # Create time - 使用 detach() 确保 t 不参与梯度计算
            # t 需要 reshape 为 (batch, 1, 1, 1, 1) 以便与 5D 张量正确广播
            with torch.no_grad():
                t = time_embedding.float() / 1000
                t = t.view(-1, 1, 1, 1, 1)  # (batch_size,) -> (batch_size, 1, 1, 1, 1) for broadcasting

            # 检查 FDG/AV45/TAU 数据是否为二值化（只有 0 和 1）
            has_fdg = not torch.all(seg_fdg == 0)  # 如果不是二值化数据，则参与计算
            has_av45 = not torch.all(seg_av45 == 0)  # 如果不是二值化数据，则参与计算
            has_tau = tau_available and seg_tau is not None and not torch.all(seg_tau == 0)

            # 默认损失为 0
            total_loss = 0.0
            
            # ============ 最优显存方案：随机选一个模态训练，避免多模态同时占用显存 ============
            # 优势：大幅降低显存占用（~70%），每步只训练一个模态
            # 原理：每个batch随机选择一个可用模态进行训练，长期来看等效于多模态联合训练
            
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
                import random
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
                
                torch.cuda.empty_cache()
            
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
            progress_bar.update(1)

            torch.cuda.empty_cache()

        progress_bar.close()
        
        # Epoch 结束后记录平均损失
        epoch_avg_loss = epoch_loss / (step + 1)
        epoch_loss_list.append(epoch_avg_loss)
        writer.add_scalar("train/epoch_loss", epoch_avg_loss, epoch)

        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_epoch_loss = 0

            for step, data_val in enumerate(val_loader):
                # 验证阶段的数据加载 - 使用 .clone().detach() 确保张量独立
                images = data_val["mri"].clone().detach().to(device)
                seg_fdg = data_val["fdg"].clone().detach().to(device)
                seg_av45 = data_val["av45"].clone().detach().to(device)
                fdg_index = data_val["fdg_index"].clone().detach().to(device)
                av45_index = data_val["av45_index"].clone().detach().to(device)
                if tau_available:
                    seg_tau = data_val["tau"].clone().detach().to(device)
                    tau_index = data_val["tau_index"].clone().detach().to(device)
                else:
                    seg_tau = None
                    tau_index = None

                # 检查 FDG/AV45/TAU 数据是否为二值化（只有 0 和 1）
                has_fdg = not torch.all(seg_fdg == 0)  # 如果不是二值化数据，则参与计算
                has_av45 = not torch.all(seg_av45 == 0)  # 如果不是二值化数据，则参与计算
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
                                # 双GPU模式下确保输出在正确设备
                                if use_distributed:
                                    v_fdg_output = v_fdg_output.to(device)
                                x_fdg_t = x_t + (v_fdg_output / N_sample_tensor)
                                x_fdg_t = torch.clamp(x_fdg_t, min=clip_sample_min, max=clip_sample_max)
                            else:
                                x_fdg_t = None  # 跳过 FDG 计算

                            # AV45 输出（仅当非二值化时计算）
                            if has_av45:
                                v_av45_output = model(x=x_t, timesteps=torch.Tensor((time_embedding,)).to(device), context=av45_index)
                                # 双GPU模式下确保输出在正确设备
                                if use_distributed:
                                    v_av45_output = v_av45_output.to(device)
                                x_av45_t = x_t + (v_av45_output / N_sample_tensor)
                                x_av45_t = torch.clamp(x_av45_t, min=clip_sample_min, max=clip_sample_max)
                            else:
                                x_av45_t = None  # 跳过 AV45 计算

                            # TAU 输出（仅当非二值化时计算）
                            if has_tau:
                                v_tau_output = model(x=x_t, timesteps=torch.Tensor((time_embedding,)).to(device), context=tau_index)
                                # 双GPU模式下确保输出在正确设备
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

                # 加权总损失
                val_loss = alpha * val_fdg_loss + beta * val_av45_loss + gamma * val_tau_loss

                val_epoch_loss += val_loss.item()
                torch.cuda.empty_cache()

            print("Epoch", epoch + 1, "Validation loss", val_epoch_loss / (step + 1))
            val_epoch_loss_list.append(val_epoch_loss / (step + 1))

            # 验证集日志记录
            val_avg_loss = val_epoch_loss / (step + 1)
            writer.add_scalar("val/loss", val_avg_loss, epoch)

            # Saving model parameters
            print('epoch:', epoch + 1)
            print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])
            save_checkpoint(f"epoch{epoch + 1}", epoch, global_step)

            # 可视化和评估指标
            current_fdg_img = x_fdg_t.to('cpu') if x_fdg_t is not None else None
            current_av45_img = x_av45_t.to('cpu') if x_av45_t is not None else None
            current_tau_img = x_tau_t.to('cpu') if tau_available and x_tau_t is not None else None
            labels_fdg = seg_fdg.to('cpu')
            labels_av45 = seg_av45.to('cpu')
            labels_tau = seg_tau.to('cpu') if tau_available else None
            mri_cpu = images.to('cpu')

            # 如果存在非二值化的 FDG 或 AV45 数据，则进行可视化和评估
            if current_fdg_img is not None or current_av45_img is not None or current_tau_img is not None:
                # 过滤掉 None 值，只传入有效的图像
                valid_images_3d = [img for img in [mri_cpu, labels_fdg, current_fdg_img, labels_av45, current_av45_img] if img is not None]
                valid_diff_images = []
                if current_fdg_img is not None and labels_fdg is not None:
                    valid_diff_images.append(current_fdg_img - labels_fdg)
                if current_av45_img is not None and labels_av45 is not None:
                    valid_diff_images.append(current_av45_img - labels_av45)
                
                if len(valid_images_3d) > 0:
                    compare_3d(valid_images_3d)
                if len(valid_diff_images) > 0:
                    compare_3d_jet(valid_diff_images)
            
            # ============ 图像记录到 TensorBoard（每 image_log_interval 个 epoch 保存一次）============
            if (epoch + 1) % image_log_interval == 0:
                print(f"📷 保存过程图像到 TensorBoard (epoch {epoch + 1})...")
                
                # 构建对比图像字典
                comparison_volumes = {"MRI": mri_cpu}
                
                if has_fdg and current_fdg_img is not None:
                    comparison_volumes["FDG_GT"] = labels_fdg
                    comparison_volumes["FDG_Pred"] = current_fdg_img
                    # 计算差异图
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

            from generative.metrics import SSIMMetric
            from monai.metrics import MAEMetric, MSEMetric, PSNRMetric

            ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=7)
            psnr_metric = PSNRMetric(1.0)

            # 计算 SSIM 和 PSNR（仅当非二值化时计算）
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

    # 训练结束，保存最终 checkpoint 并关闭 writer
    writer.flush()
    writer.close()
    save_checkpoint("last", epoch if 'epoch' in locals() else 0, global_step)
    print(f"\n🎉 训练完成！日志保存在: {run_dir}")


if __name__ == "__main__":
    main()
