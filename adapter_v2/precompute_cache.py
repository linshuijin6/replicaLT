"""
adapter_v2/precompute_cache.py
==============================
TAU 单示踪剂 vision embedding 预缓存生成脚本

功能：
- 从 CSV 读取样本列表
- 使用 BiomedCLIP 提取 vision embedding（cls_token + region_token）
- 以 {subject_id}_{image_id}.vision.pt 命名保存缓存
- 支持断点续跑（已存在的缓存跳过）
- 支持并行处理

使用方式：
    conda run -n xiaochou python precompute_cache.py --config config.yaml
    conda run -n xiaochou python precompute_cache.py --csv /path/to/csv --cache-dir /path/to/cache
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

import yaml
import numpy as np
import pandas as pd

# 延迟导入：重量级依赖（torch/torchvision/nibabel）只在真正需要时导入
# 这样 --stats-only 和 --list-gpus 等命令可以快速执行
torch = None
nib = None
TF = None
InterpolationMode = None
tqdm = None


def _lazy_import_torch():
    """延迟导入 torch 及相关模块"""
    global torch, nib, TF, InterpolationMode, tqdm
    if torch is not None:
        return
    
    print("[导入] 正在加载 PyTorch 和 torchvision...")
    import torch as _torch
    import nibabel as _nib
    from torchvision.transforms import InterpolationMode as _InterpolationMode
    from torchvision.transforms import functional as _TF
    from tqdm import tqdm as _tqdm
    
    torch = _torch
    nib = _nib
    TF = _TF
    InterpolationMode = _InterpolationMode
    tqdm = _tqdm
    print("[导入] PyTorch 加载完成")

# 添加 CLIP-MRI2PET 到 path（延迟导入模型）
CLIP_MRI2PET_ROOT = Path(__file__).resolve().parents[2] / "CLIP-MRI2PET"
CLIP_MRI2PET_SRC = CLIP_MRI2PET_ROOT / "src"


def _setup_clip_mri2pet_path():
    """添加 CLIP-MRI2PET 到 sys.path（延迟调用）"""
    if str(CLIP_MRI2PET_ROOT) not in sys.path:
        sys.path.insert(0, str(CLIP_MRI2PET_ROOT))
    if str(CLIP_MRI2PET_SRC) not in sys.path:
        sys.path.insert(0, str(CLIP_MRI2PET_SRC))


# ============================================================================
# 常量
# ============================================================================

DEFAULT_ADNI_ROOT = "/mnt/nfsdata/nfsdata/lsj.14/ADNI_CSF"
DEFAULT_TAU_SUBDIR = "PET_MNI/TAU"


# ============================================================================
# GPU 工具函数
# ============================================================================

def get_gpu_free_memory() -> Dict[int, int]:
    """
    获取各 GPU 的空闲显存（MiB）
    
    Returns:
        Dict[gpu_id, free_memory_mib]
    """
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return {}
        
        gpu_free = {}
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                parts = line.split(",")
                if len(parts) == 2:
                    gpu_id = int(parts[0].strip())
                    free_mem = int(parts[1].strip())
                    gpu_free[gpu_id] = free_mem
        return gpu_free
    except Exception:
        return {}


def select_best_gpu(min_free_mb: int = 8000) -> Optional[int]:
    """
    自动选择空闲显存最大的 GPU
    
    Args:
        min_free_mb: 最小要求空闲显存（MiB）
        
    Returns:
        最佳 GPU 卡号，如果没有满足条件的返回 None
    """
    gpu_free = get_gpu_free_memory()
    if not gpu_free:
        return None
    
    # 按空闲显存降序排序
    sorted_gpus = sorted(gpu_free.items(), key=lambda x: x[1], reverse=True)
    
    for gpu_id, free_mem in sorted_gpus:
        if free_mem >= min_free_mb:
            return gpu_id
    
    # 没有满足最小要求的，返回空闲最大的
    if sorted_gpus:
        return sorted_gpus[0][0]
    return None


def ensure_cpu_tensor(t: Any) -> Any:
    """
    确保张量在 CPU 上（用于保存缓存时确保设备无关）
    
    Args:
        t: 任意对象，如果是 Tensor 则移到 CPU
        
    Returns:
        CPU 上的张量或原对象
    """
    _lazy_import_torch()
    if isinstance(t, torch.Tensor):
        return t.detach().cpu()
    return t


def save_cache_payload(payload: Dict[str, Any], cache_file: Path) -> None:
    """
    保存缓存，确保所有张量在 CPU 上（设备无关）
    
    Args:
        payload: 包含张量的字典
        cache_file: 保存路径
    """
    _lazy_import_torch()
    # 确保所有张量在 CPU 上
    cpu_payload = {}
    for key, value in payload.items():
        cpu_payload[key] = ensure_cpu_tensor(value)
    
    torch.save(cpu_payload, cache_file)


def load_cache_payload(cache_file: Path, device: str = "cpu") -> Dict[str, Any]:
    """
    加载缓存，支持指定目标设备
    
    Args:
        cache_file: 缓存文件路径
        device: 目标设备（默认 cpu）
        
    Returns:
        包含张量的字典
    """
    _lazy_import_torch()
    payload = torch.load(cache_file, map_location="cpu", weights_only=False)
    
    if device != "cpu":
        # 移动张量到目标设备
        for key, value in payload.items():
            if isinstance(value, torch.Tensor):
                payload[key] = value.to(device)
    
    return payload


def setup_gpu(gpu: Optional[int] = None, auto_select: bool = True, min_free_mb: int = 8000) -> int:
    """
    设置 GPU 环境
    
    Args:
        gpu: 指定的 GPU 卡号（None 表示自动选择）
        auto_select: 当指定 GPU 无效时是否自动选择其他 GPU
        min_free_mb: 自动选择时的最小空闲显存要求
        
    Returns:
        实际使用的 GPU 卡号
    """
    gpu_free = get_gpu_free_memory()
    
    if gpu is not None:
        # 检查指定 GPU 是否可用
        if gpu in gpu_free:
            free_mem = gpu_free[gpu]
            if free_mem >= min_free_mb:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
                print(f"[GPU] 使用指定 GPU {gpu}（空闲 {free_mem} MiB）")
                return gpu
            else:
                print(f"[GPU] 警告: GPU {gpu} 空闲显存不足（{free_mem} MiB < {min_free_mb} MiB）")
                if not auto_select:
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
                    return gpu
        else:
            print(f"[GPU] 警告: GPU {gpu} 不存在")
            if not auto_select:
                raise RuntimeError(f"GPU {gpu} 不存在")
    
    # 自动选择
    best_gpu = select_best_gpu(min_free_mb)
    if best_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
        print(f"[GPU] 自动选择 GPU {best_gpu}（空闲 {gpu_free.get(best_gpu, 0)} MiB）")
        return best_gpu
    
    # 没有可用 GPU
    print("[GPU] 警告: 没有找到可用 GPU，将使用 CPU")
    return -1

# 默认 ROI 配置（Harvard-Oxford Atlas）
DEFAULT_ROI_NAMES = [
    "Frontal Pole",
    "Insular Cortex", 
    "Superior Frontal Gyrus",
    "Middle Frontal Gyrus",
    "Inferior Frontal Gyrus",
    "Precentral Gyrus",
    "Temporal Pole",
    "Superior Temporal Gyrus",
    "Middle Temporal Gyrus",
    "Inferior Temporal Gyrus",
    "Postcentral Gyrus",
    "Superior Parietal Lobule",
    "Angular Gyrus",
    "Lateral Occipital Cortex",
    "Precuneous Cortex",
    "Cingulate Gyrus",
    "Parahippocampal Gyrus",
    "Hippocampus",
    "Amygdala",
    "Thalamus",
]


# ============================================================================
# 工具函数
# ============================================================================

def load_config(config_path: str) -> dict:
    """加载 YAML 配置"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_tau_nifti(
    adni_root: str,
    ptid: str,
    mri_id: str,
    tau_id: str,
    tau_subdir: str = DEFAULT_TAU_SUBDIR,
) -> Optional[Path]:
    """
    根据 PTID、MRI_ID、TAU_ID 查找 TAU NIfTI 文件
    
    文件命名格式: {PTID}__I{MRI_ID}__I{TAU_ID}.nii.gz
    """
    tau_dir = Path(adni_root) / tau_subdir
    
    # 标准命名
    patterns = [
        f"{ptid}__I{mri_id}__I{tau_id}.nii.gz",
        f"{ptid}__I{mri_id}__I{tau_id}_full.nii.gz",
    ]
    
    for pattern in patterns:
        nifti_path = tau_dir / pattern
        if nifti_path.exists():
            return nifti_path
    
    # 尝试通配符匹配（只用 PTID 和 TAU_ID）
    import glob
    wildcard = str(tau_dir / f"{ptid}__*__I{tau_id}.nii.gz")
    matches = glob.glob(wildcard)
    if matches:
        return Path(matches[0])
    
    return None


def build_sample_list(
    csv_path: str,
    adni_root: str,
    tau_subdir: str = DEFAULT_TAU_SUBDIR,
) -> List[Dict[str, Any]]:
    """
    从 CSV 构建样本列表
    
    Returns:
        List[Dict] with keys: ptid, mri_id, tau_id, nifti_path, cache_name
    """
    # 指定 ID 列为字符串，避免被解析为 float 出现小数点问题
    id_columns = ["PTID", "id_mri", "id_fdg", "id_av45", "id_av1451", "plasma_source"]
    dtype_spec = {col: str for col in id_columns}
    df = pd.read_csv(csv_path, dtype=dtype_spec)
    
    # 只保留有 TAU (id_av1451) 的行 - 注意 dtype=str 时 NaN 变为字符串 "nan"
    df = df[(df["id_av1451"].notna()) & (df["id_av1451"] != "nan")].reset_index(drop=True)
    
    samples = []
    missing_count = 0
    
    for _, row in df.iterrows():
        ptid = str(row["PTID"])
        
        # 由于已指定 dtype=str，直接作为字符串使用
        tau_id = str(row["id_av1451"])
        
        # MRI ID
        mri_id = row.get("id_mri")
        if pd.isna(mri_id) or mri_id == "nan":
            mri_id = None
        else:
            mri_id = str(mri_id)
        
        # 查找 NIfTI 文件
        nifti_path = find_tau_nifti(adni_root, ptid, mri_id, tau_id, tau_subdir)
        
        if nifti_path is None:
            missing_count += 1
            continue
        
        # 缓存命名: {subject_id}_{image_id}.vision.pt
        cache_name = f"{ptid}_{tau_id}.vision.pt"
        
        samples.append({
            "ptid": ptid,
            "mri_id": mri_id,
            "tau_id": tau_id,
            "nifti_path": str(nifti_path),
            "cache_name": cache_name,
        })
    
    print(f"[build_sample_list] 总行数: {len(df)}, 找到 NIfTI: {len(samples)}, 缺失: {missing_count}")
    return samples


# ============================================================================
# Vision Encoder
# ============================================================================

class TAUVisionEncoder:
    """TAU PET Vision Embedding 提取器（支持批量并行）"""
    
    def __init__(
        self,
        biomedclip_path: str = None,
        slice_axis: str = "axial",
        slices_per_batch: int = 64,  # 增大默认值以提高 GPU 利用率
        device: str = "cuda",
    ):
        # 延迟导入 torch 和 BiomedCLIP 模块
        _lazy_import_torch()
        _setup_clip_mri2pet_path()
        from clip_mri2pet.models.biomedclip_image import (
            BiomedCLIPImageEncoder,
            load_biomedclip_model,
        )
        
        # 支持 "cuda", "cuda:0", "cpu" 等格式
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[TAUVisionEncoder] 警告: CUDA 不可用，回退到 CPU")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.slice_axis = slice_axis
        self.slices_per_batch = slices_per_batch
        
        # 加载 BiomedCLIP
        print(f"[TAUVisionEncoder] 加载 BiomedCLIP...")
        clip_model, preprocess = load_biomedclip_model()
        
        self.image_encoder = BiomedCLIPImageEncoder(
            clip_model=clip_model,
            preprocess=preprocess,
            slice_axis=slice_axis,
            slices_per_batch=slices_per_batch,
        )
        self.image_encoder = self.image_encoder.to(self.device)
        self.image_encoder.eval()
        
        self.preprocess = preprocess
        print(f"[TAUVisionEncoder] 就绪，device={self.device}, slices_per_batch={slices_per_batch}")
    
    def _load_volume(self, nifti_path: str) -> "torch.Tensor":
        """加载 NIfTI 体积并预处理"""
        nii = nib.load(nifti_path)
        volume = nii.get_fdata().astype(np.float32)
        
        # 归一化到 [0, 1]
        vmin, vmax = volume.min(), volume.max()
        if vmax - vmin > 1e-6:
            volume = (volume - vmin) / (vmax - vmin)
        else:
            volume = np.zeros_like(volume)
        
        return torch.from_numpy(volume)
    
    def _extract_slices(self, volume: "torch.Tensor") -> List["torch.Tensor"]:
        """从体积中提取 2D 切片"""
        axis_map = {"sagittal": 0, "coronal": 1, "axial": 2}
        axis = axis_map.get(self.slice_axis, 2)
        
        slices = []
        n_slices = volume.shape[axis]
        
        for i in range(n_slices):
            if axis == 0:
                slice_2d = volume[i, :, :]
            elif axis == 1:
                slice_2d = volume[:, i, :]
            else:
                slice_2d = volume[:, :, i]
            
            # 跳过空切片
            if slice_2d.max() < 0.01:
                continue
            
            # 转换为 RGB 格式 (3, H, W)
            slice_2d = slice_2d.unsqueeze(0).expand(3, -1, -1)
            
            # 旋转和调整大小
            rotated = torch.rot90(slice_2d, k=1, dims=(1, 2))
            resized = TF.resize(
                rotated,
                224,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            cropped = TF.center_crop(resized, [224, 224])
            slices.append(cropped)
        
        return slices
    
    def encode(self, nifti_path: str) -> Dict[str, "torch.Tensor"]:
        """
        编码单个 TAU PET 体积
        
        Returns:
            Dict with keys: cls_token (512,), region_token (N, 768)
        """
        with torch.no_grad():
            volume = self._load_volume(nifti_path)
            slices = self._extract_slices(volume)
            
            if len(slices) == 0:
                # 返回零向量
                return {
                    "cls_token": torch.zeros(512),
                    "region_token": torch.zeros(196, 768),
                }
            
            # 批量编码切片
            cls_tokens = []
            patch_tokens = []
            
            for start in range(0, len(slices), self.slices_per_batch):
                batch = torch.stack(slices[start:start + self.slices_per_batch], dim=0)
                batch = batch.to(self.device)
                
                cls, tokens = self.image_encoder.visual(batch)
                cls_tokens.append(cls.cpu())
                patch_tokens.append(tokens.cpu())
            
            # 聚合
            cls_all = torch.cat(cls_tokens, dim=0)  # (num_slices, 512)
            tokens_all = torch.cat(patch_tokens, dim=0)  # (num_slices, 196, 768)
            
            # 平均池化
            pooled_cls = cls_all.mean(dim=0)  # (512,)
            pooled_tokens = tokens_all.mean(dim=0)  # (196, 768)
            
            return {
                "cls_token": pooled_cls,
                "region_token": pooled_tokens,
            }


# ============================================================================
# 批量并行缓存生成（单GPU + 多线程预加载）
# ============================================================================

def _preload_sample(sample: Dict[str, Any], encoder: "TAUVisionEncoder") -> Tuple[Dict, List[Any]]:
    """预加载单个样本的切片（在后台线程中执行）"""
    try:
        volume = encoder._load_volume(sample["nifti_path"])
        slices = encoder._extract_slices(volume)
        return sample, slices
    except Exception as e:
        return sample, None


def precompute_caches_parallel(
    csv_path: str,
    cache_dir: str,
    adni_root: str,
    tau_subdir: str = DEFAULT_TAU_SUBDIR,
    device: str = "cuda",
    force: bool = False,
    gpu: Optional[int] = None,  # None 表示自动选择
    num_workers: int = 4,  # 数据预加载线程数
    slices_per_batch: int = 64,  # 每批切片数
    auto_select_gpu: bool = True,  # 自动选择空闲 GPU
    min_gpu_free_mb: int = 8000,  # 最小空闲显存要求
) -> Tuple[int, int, List[str]]:
    """
    使用单 GPU + 多线程预加载并行生成缓存
    
    Args:
        csv_path: 样本 CSV 路径
        cache_dir: 缓存输出目录
        adni_root: ADNI 数据根目录
        tau_subdir: TAU 数据子目录
        device: 计算设备
        force: 是否强制重新计算已存在的缓存
        gpu: GPU 卡号（None 表示自动选择）
        num_workers: 数据预加载线程数
        slices_per_batch: 每批送入 GPU 的切片数
        auto_select_gpu: 当指定 GPU 不可用时是否自动选择其他 GPU
        min_gpu_free_mb: 自动选择时的最小空闲显存要求（MiB）
        
    Returns:
        (success_count, skip_count, failed_list)
    """
    # 延迟导入 torch
    _lazy_import_torch()
    
    # 智能 GPU 选择
    actual_gpu = setup_gpu(gpu, auto_select=auto_select_gpu, min_free_mb=min_gpu_free_mb)
    
    # 创建缓存目录
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # 构建样本列表
    samples = build_sample_list(csv_path, adni_root, tau_subdir)
    
    if len(samples) == 0:
        print("[Error] 没有找到任何有效样本！")
        return 0, 0, []
    
    # 过滤已存在的缓存
    if not force:
        pending_samples = []
        skip_count = 0
        for s in samples:
            if (cache_path / s["cache_name"]).exists():
                skip_count += 1
            else:
                pending_samples.append(s)
        print(f"[precompute_caches_parallel] 跳过已存在: {skip_count}, 待处理: {len(pending_samples)}")
    else:
        pending_samples = samples
        skip_count = 0
    
    if len(pending_samples) == 0:
        print("[完成] 所有缓存已存在")
        return 0, skip_count, []
    
    # 初始化编码器
    encoder = TAUVisionEncoder(device=device, slices_per_batch=slices_per_batch)
    
    success_count = 0
    failed_list = []
    
    # 使用 ThreadPoolExecutor 进行数据预加载
    # 创建预加载队列
    prefetch_queue = queue.Queue(maxsize=num_workers * 2)
    stop_event = threading.Event()
    
    def prefetch_worker():
        """后台预加载线程"""
        for sample in pending_samples:
            if stop_event.is_set():
                break
            try:
                volume = encoder._load_volume(sample["nifti_path"])
                slices = encoder._extract_slices(volume)
                prefetch_queue.put((sample, slices))
            except Exception as e:
                prefetch_queue.put((sample, None))
        # 放入结束标记
        prefetch_queue.put(None)
    
    # 启动预加载线程
    prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
    prefetch_thread.start()
    
    pbar = tqdm(total=len(pending_samples), desc="生成缓存")
    
    while True:
        item = prefetch_queue.get()
        if item is None:
            break
        
        sample, slices = item
        cache_file = cache_path / sample["cache_name"]
        
        if slices is None:
            failed_list.append({"sample": sample, "error": "预加载失败"})
            pbar.update(1)
            continue
        
        try:
            if len(slices) == 0:
                payload = {
                    "cls_token": torch.zeros(512),
                    "region_token": torch.zeros(196, 768),
                }
            else:
                # GPU 编码
                with torch.no_grad():
                    cls_tokens = []
                    patch_tokens = []
                    
                    for start in range(0, len(slices), encoder.slices_per_batch):
                        batch = torch.stack(slices[start:start + encoder.slices_per_batch], dim=0)
                        batch = batch.to(encoder.device)
                        
                        cls, tokens = encoder.image_encoder.visual(batch)
                        cls_tokens.append(cls.cpu())
                        patch_tokens.append(tokens.cpu())
                    
                    cls_all = torch.cat(cls_tokens, dim=0)
                    tokens_all = torch.cat(patch_tokens, dim=0)
                    
                    pooled_cls = cls_all.mean(dim=0)
                    pooled_tokens = tokens_all.mean(dim=0)
                    
                    payload = {
                        "cls_token": pooled_cls,
                        "region_token": pooled_tokens,
                    }
            
            payload["ptid"] = sample["ptid"]
            payload["tau_id"] = sample["tau_id"]
            payload["nifti_path"] = sample["nifti_path"]
            
            # 使用设备无关的保存方式
            save_cache_payload(payload, cache_file)
            success_count += 1
            
        except Exception as e:
            failed_list.append({"sample": sample, "error": str(e)})
            tqdm.write(f"[Error] {sample['cache_name']}: {e}")
        
        pbar.set_postfix({"done": success_count, "fail": len(failed_list)})
        pbar.update(1)
    
    pbar.close()
    stop_event.set()
    
    print(f"\n[完成] 成功: {success_count}, 跳过: {skip_count}, 失败: {len(failed_list)}")
    
    return success_count, skip_count, failed_list


# ============================================================================
# 原始单线程版本（保留兼容）
# ============================================================================

def precompute_caches(
    csv_path: str,
    cache_dir: str,
    adni_root: str,
    tau_subdir: str = DEFAULT_TAU_SUBDIR,
    device: str = "cuda",
    force: bool = False,
    gpu: Optional[int] = None,
    auto_select_gpu: bool = True,
    min_gpu_free_mb: int = 8000,
) -> Tuple[int, int, List[str]]:
    """
    预计算 vision embedding 缓存
    
    Args:
        csv_path: 样本 CSV 路径
        cache_dir: 缓存输出目录
        adni_root: ADNI 数据根目录
        tau_subdir: TAU 数据子目录
        device: 计算设备
        force: 是否强制重新计算已存在的缓存
        gpu: GPU 卡号（None 表示自动选择）
        auto_select_gpu: 当指定 GPU 不可用时是否自动选择其他 GPU
        min_gpu_free_mb: 最小空闲显存要求（MiB）
        
    Returns:
        (success_count, skip_count, failed_list)
    """
    # 延迟导入 torch
    _lazy_import_torch()
    
    # 智能 GPU 选择
    actual_gpu = setup_gpu(gpu, auto_select=auto_select_gpu, min_free_mb=min_gpu_free_mb)
    
    # 创建缓存目录
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # 构建样本列表
    samples = build_sample_list(csv_path, adni_root, tau_subdir)
    
    if len(samples) == 0:
        print("[Error] 没有找到任何有效样本！")
        return 0, 0, []
    
    # 初始化编码器
    encoder = TAUVisionEncoder(device=device)
    
    success_count = 0
    skip_count = 0
    failed_list = []
    
    pbar = tqdm(samples, desc="生成缓存")
    for sample in pbar:
        cache_file = cache_path / sample["cache_name"]
        
        # 检查是否已存在
        if cache_file.exists() and not force:
            skip_count += 1
            pbar.set_postfix({"done": success_count, "skip": skip_count})
            continue
        
        try:
            # 编码
            payload = encoder.encode(sample["nifti_path"])
            payload["ptid"] = sample["ptid"]
            payload["tau_id"] = sample["tau_id"]
            payload["nifti_path"] = sample["nifti_path"]
            
            # 保存（设备无关）
            save_cache_payload(payload, cache_file)
            success_count += 1
            
        except Exception as e:
            failed_list.append({
                "sample": sample,
                "error": str(e),
            })
            tqdm.write(f"[Error] {sample['cache_name']}: {e}")
        
        pbar.set_postfix({"done": success_count, "skip": skip_count, "fail": len(failed_list)})
    
    print(f"\n[完成] 成功: {success_count}, 跳过: {skip_count}, 失败: {len(failed_list)}")
    
    return success_count, skip_count, failed_list


def generate_missing_caches(
    csv_path: str,
    cache_dir: str,
    adni_root: str,
    tau_subdir: str = DEFAULT_TAU_SUBDIR,
    device: str = "cuda",
    gpu: int = None,
) -> Tuple[int, List[str]]:
    """
    仅生成缺失的缓存（用于 train.py 调用）
    
    Returns:
        (generated_count, missing_nifti_list)
    """
    # 延迟导入 torch
    _lazy_import_torch()
    
    # 设置 GPU
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # 构建样本列表
    samples = build_sample_list(csv_path, adni_root, tau_subdir)
    
    # 找出缺失缓存的样本
    missing_samples = []
    missing_nifti = []
    
    id_columns = ["PTID", "id_mri", "id_fdg", "id_av45", "id_av1451", "plasma_source"]
    dtype_spec = {col: str for col in id_columns}
    df = pd.read_csv(csv_path, dtype=dtype_spec)
    df = df[(df["id_av1451"].notna()) & (df["id_av1451"] != "nan")].reset_index(drop=True)
    
    for _, row in df.iterrows():
        ptid = str(row["PTID"])
        tau_id = str(row["id_av1451"])
        cache_name = f"{ptid}_{tau_id}.vision.pt"
        cache_file = cache_path / cache_name
        
        if not cache_file.exists():
            # 在 samples 中查找
            found = None
            for s in samples:
                if s["ptid"] == ptid and s["tau_id"] == tau_id:
                    found = s
                    break
            
            if found:
                missing_samples.append(found)
            else:
                missing_nifti.append(f"{ptid}_{tau_id}")
    
    if len(missing_samples) == 0:
        print(f"[generate_missing_caches] 所有缓存已存在，无需生成")
        if missing_nifti:
            print(f"  注意: {len(missing_nifti)} 个样本缺少 NIfTI 文件")
        return 0, missing_nifti
    
    print(f"[generate_missing_caches] 需要生成 {len(missing_samples)} 个缓存...")
    
    # 初始化编码器
    encoder = TAUVisionEncoder(device=device)
    
    generated = 0
    for sample in tqdm(missing_samples, desc="生成缺失缓存"):
        cache_file = cache_path / sample["cache_name"]
        
        try:
            payload = encoder.encode(sample["nifti_path"])
            payload["ptid"] = sample["ptid"]
            payload["tau_id"] = sample["tau_id"]
            payload["nifti_path"] = sample["nifti_path"]
            # 保存（设备无关）
            save_cache_payload(payload, cache_file)
            generated += 1
        except Exception as e:
            tqdm.write(f"[Error] {sample['cache_name']}: {e}")
    
    print(f"[generate_missing_caches] 完成，生成 {generated} 个缓存")
    return generated, missing_nifti


def get_cache_stats(
    csv_path: str,
    cache_dir: str,
) -> Dict[str, int]:
    """
    获取缓存统计信息
    
    Returns:
        Dict with keys: total, cached, missing, missing_list
        missing_list: List of (ptid, tau_id) tuples for missing caches
    """
    id_columns = ["PTID", "id_mri", "id_fdg", "id_av45", "id_av1451", "plasma_source"]
    dtype_spec = {col: str for col in id_columns}
    df = pd.read_csv(csv_path, dtype=dtype_spec)
    df = df[(df["id_av1451"].notna()) & (df["id_av1451"] != "nan")].reset_index(drop=True)
    
    cache_path = Path(cache_dir)
    
    cached = 0
    missing = 0
    missing_list = []
    
    for _, row in df.iterrows():
        ptid = str(row["PTID"])
        tau_id = str(row["id_av1451"])
        cache_name = f"{ptid}_{tau_id}.vision.pt"
        cache_file = cache_path / cache_name
        
        if cache_file.exists():
            cached += 1
        else:
            missing += 1
            missing_list.append((ptid, tau_id))
    
    return {
        "total": len(df),
        "cached": cached,
        "missing": missing,
        "missing_list": missing_list,
    }


def main():
    parser = argparse.ArgumentParser(description="TAU Vision Embedding 缓存预生成")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--csv", type=str, default=None, help="样本 CSV 路径（覆盖配置）")
    parser.add_argument("--cache-dir", type=str, default=None, help="缓存输出目录（覆盖配置）")
    parser.add_argument("--adni-root", type=str, default=None, help="ADNI 数据根目录（覆盖配置）")
    parser.add_argument("--gpu", type=int, default=None, help="GPU 卡号（默认自动选择空闲最大的）")
    parser.add_argument("--num-workers", type=int, default=4, help="数据预加载线程数")
    parser.add_argument("--slices-per-batch", type=int, default=64, help="每批送入 GPU 的切片数")
    parser.add_argument("--force", action="store_true", help="强制重新生成已存在的缓存")
    parser.add_argument("--stats-only", action="store_true", help="仅显示缓存统计，不生成")
    parser.add_argument("--no-parallel", action="store_true", help="禁用多线程预加载，使用单线程模式")
    parser.add_argument("--no-auto-gpu", action="store_true", help="禁用 GPU 自动选择")
    parser.add_argument("--min-gpu-free", type=int, default=8000, help="最小 GPU 空闲显存要求（MiB）")
    parser.add_argument("--list-gpus", action="store_true", help="列出所有可用 GPU 及显存状态")
    args = parser.parse_args()
    
    # 加载配置
    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    
    if config_path.exists():
        config = load_config(config_path)
    else:
        config = {}
    
    # --list-gpus 模式
    if args.list_gpus:
        gpu_free = get_gpu_free_memory()
        if not gpu_free:
            print("[Error] 无法获取 GPU 信息")
            return
        print("[可用 GPU]")
        for gpu_id, free_mem in sorted(gpu_free.items()):
            status = "✓" if free_mem >= args.min_gpu_free else "✗"
            print(f"  GPU {gpu_id}: {free_mem:>6} MiB 空闲  {status}")
        best = select_best_gpu(args.min_gpu_free)
        if best is not None:
            print(f"\n推荐: GPU {best}")
        return
    
    # 参数优先级：命令行 > 配置文件 > 默认值
    csv_path = args.csv or config.get("data", {}).get("csv_path")
    cache_dir = args.cache_dir or config.get("data", {}).get("cache_dir")
    adni_root = args.adni_root or config.get("data", {}).get("adni_root", DEFAULT_ADNI_ROOT)
    gpu = args.gpu  # None 表示自动选择
    
    if csv_path is None:
        print("[Error] 请指定 --csv 或在 config.yaml 中配置 data.csv_path")
        return
    if cache_dir is None:
        print("[Error] 请指定 --cache-dir 或在 config.yaml 中配置 data.cache_dir")
        return
    
    print(f"[配置]")
    print(f"  CSV: {csv_path}")
    print(f"  Cache Dir: {cache_dir}")
    print(f"  ADNI Root: {adni_root}")
    print(f"  GPU: {gpu if gpu is not None else '自动选择'}")
    print(f"  Auto GPU: {not args.no_auto_gpu}")
    print(f"  Min GPU Free: {args.min_gpu_free} MiB")
    print(f"  Num Workers: {args.num_workers}")
    print(f"  Slices per Batch: {args.slices_per_batch}")
    print(f"  Force: {args.force}")
    print(f"  Parallel: {not args.no_parallel}")
    print()
    
    # 仅统计模式
    if args.stats_only:
        stats = get_cache_stats(csv_path, cache_dir)
        print(f"[缓存统计]")
        print(f"  总样本: {stats['total']}")
        print(f"  已缓存: {stats['cached']}")
        print(f"  缺失: {stats['missing']}")
        return
    
    # 执行预缓存
    start_time = datetime.now()
    
    if args.no_parallel:
        # 使用原始单线程模式
        success, skip, failed = precompute_caches(
            csv_path=csv_path,
            cache_dir=cache_dir,
            adni_root=adni_root,
            device="cuda",
            force=args.force,
            gpu=gpu,
        )
    else:
        # 使用多线程预加载并行模式
        success, skip, failed = precompute_caches_parallel(
            csv_path=csv_path,
            cache_dir=cache_dir,
            adni_root=adni_root,
            device="cuda",
            force=args.force,
            gpu=gpu,
            num_workers=args.num_workers,
            slices_per_batch=args.slices_per_batch,
            auto_select_gpu=not args.no_auto_gpu,
            min_gpu_free_mb=args.min_gpu_free,
        )
    
    elapsed = datetime.now() - start_time
    print(f"\n[耗时] {elapsed}")
    
    # 输出失败列表
    if failed:
        fail_log = script_dir / "precompute_failed.log"
        with open(fail_log, "w") as f:
            for item in failed:
                f.write(f"{item['sample']['cache_name']}: {item['error']}\n")
        print(f"失败列表已保存到: {fail_log}")


if __name__ == "__main__":
    main()
