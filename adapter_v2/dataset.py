"""
adapter_v2/dataset.py
=====================
TAU 单示踪剂 CoCoOp 微调的数据集与采样器实现。

核心设计：
1. 以 subject 为单位采样，避免同 subject 多条记录落入同 batch 造成伪负样本
2. 加载预缓存的 vision embedding（.vision.pt），包含 cls_token 和 region_token
3. 结构化 plasma 处理：z-score 归一化 + 缺失 mask
4. 诊断标签映射
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import random

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader

# 添加 CLIP-MRI2PET 到 path 以复用组件
CLIP_MRI2PET_ROOT = Path(__file__).resolve().parents[2] / "CLIP-MRI2PET"
if str(CLIP_MRI2PET_ROOT) not in sys.path:
    sys.path.insert(0, str(CLIP_MRI2PET_ROOT))

# ============================================================================
# 常量定义
# ============================================================================

# 诊断标签规范化映射
DIAGNOSIS_ALIAS = {
    "CN": "CN",
    "COGNITIVELY NORMAL": "CN",
    "NORMAL": "CN",
    "CONTROL": "CN",
    "MCI": "MCI",
    "EMCI": "MCI",
    "LMCI": "MCI",
    "MILD COGNITIVE IMPAIRMENT": "MCI",
    "AD": "AD",
    "ALZHEIMERS DISEASE": "AD",
    "ALZHEIMER'S DISEASE": "AD",
}


# 数值型 diagnosis 码（可被 config 覆盖）
DEFAULT_DIAGNOSIS_CODE_MAP: Dict[int, str] = {
    1: "CN",
    2: "MCI",
    3: "AD",
}

# 默认 plasma 字段
DEFAULT_PLASMA_KEYS = ["AB42_AB40_F", "pT217_F", "pT217_AB42_F", "NfL_Q", "GFAP_Q"]


# ============================================================================
# 工具函数
# ============================================================================

def normalize_diagnosis(
    raw: Any,
    diagnosis_code_map: Dict[Any, str] | None = None,
    diagnosis_alias: Dict[str, str] | None = None,
) -> Optional[str]:
    """将诊断标签规范化为 CN/MCI/AD。

    支持：
    - 字符串标签（如 CN/MCI/AD、NORMAL、LMCI 等）
    - CSV 中用 1/2/3（或 1.0/2.0/3.0）表示的数值型 diagnosis
    """
    if raw is None or pd.isna(raw):
        return None

    alias = diagnosis_alias or DIAGNOSIS_ALIAS
    code_map = diagnosis_code_map or DEFAULT_DIAGNOSIS_CODE_MAP

    # 1) 先尝试数值 code 映射（兼容 int/float/numpy scalar/字符串数字）
    try:
        if raw in code_map:
            mapped = code_map.get(raw)
            if mapped is not None:
                return str(mapped).strip().upper()
    except Exception:
        pass

    try:
        # numpy 标量 / pandas 数值
        if isinstance(raw, (int, np.integer)):
            mapped = code_map.get(int(raw))
            if mapped is not None:
                return str(mapped).strip().upper()
        if isinstance(raw, (float, np.floating)):
            f = float(raw)
            if np.isfinite(f) and float(f).is_integer():
                mapped = code_map.get(int(f))
                if mapped is not None:
                    return str(mapped).strip().upper()
    except Exception:
        pass

    # 2) 字符串标签 / 字符串数字
    key = str(raw).strip().upper()
    if key in alias:
        return alias.get(key)

    # 3) 尝试把字符串数字转成 code
    try:
        f = float(key)
        if np.isfinite(f) and float(f).is_integer():
            mapped = code_map.get(int(f))
            if mapped is not None:
                return str(mapped).strip().upper()
    except Exception:
        pass

    return None


def compute_plasma_stats(
    df: pd.DataFrame,
    plasma_keys: List[str],
    by_source: bool = True,
    source_col: str = "plasma_source",
    na_value: float = -4.0,
) -> Dict[str, Dict[str, float]]:
    """
    计算 plasma 各字段的归一化统计量（z-score），按 source 分别计算
    
    Args:
        df: 包含 plasma 字段的 DataFrame
        plasma_keys: plasma 字段名列表
        by_source: 是否按 source 分别计算统计量
        source_col: source 列名
        na_value: 表示缺失值的特殊值（如 -4.0），将被排除
        
    Returns:
        若 by_source=True:
            Dict[source, Dict[field_name, {"mean": float, "std": float}]]
        若 by_source=False:
            Dict[field_name, {"mean": float, "std": float}]
    """
    if by_source and source_col in df.columns:
        sources = df[source_col].dropna().unique()
        stats = {}
        for source in sources:
            source_df = df[df[source_col] == source]
            stats[source] = {}
            for key in plasma_keys:
                if key not in df.columns:
                    stats[source][key] = {"mean": 0.0, "std": 1.0}
                    continue
                values = pd.to_numeric(source_df[key], errors="coerce")
                # 排除 NA 值（-4 或负值）
                valid = values[(values.notna()) & (values >= 0) & (values != na_value)]
                if len(valid) == 0:
                    stats[source][key] = {"mean": 0.0, "std": 1.0}
                else:
                    # C2N 的 pT217_AB42_F 是百分比单位，需要除以 100 转换为小数
                    if key == "pT217_AB42_F" and source == "C2N":
                        valid = valid / 100.0
                    mean_val = float(valid.mean())
                    std_val = float(valid.std(ddof=0))
                    if std_val < 1e-8:
                        std_val = 1.0
                    stats[source][key] = {"mean": mean_val, "std": std_val}
        return stats
    else:
        # 全局统计（不按 source 分组）
        stats = {}
        for key in plasma_keys:
            if key not in df.columns:
                stats[key] = {"mean": 0.0, "std": 1.0}
                continue
            values = pd.to_numeric(df[key], errors="coerce")
            valid = values[(values.notna()) & (values >= 0) & (values != na_value)]
            if len(valid) == 0:
                stats[key] = {"mean": 0.0, "std": 1.0}
            else:
                mean_val = float(valid.mean())
                std_val = float(valid.std(ddof=0))
                if std_val < 1e-8:
                    std_val = 1.0
                stats[key] = {"mean": mean_val, "std": std_val}
        return stats


# ============================================================================
# Dataset 实现
# ============================================================================

class TAUPlasmaDataset(Dataset):
    """
    TAU 单示踪剂 CoCoOp 数据集
    
    输出字段（单条样本）：
    - subject_id: str
    - tau_tokens: Tensor (N, Dv)     # patch tokens，N=num_patches, Dv=token_dim
    - tau_cls: Tensor (D_clip,)      # cls token，D_clip=512
    - diagnosis_id: LongTensor ()    # 0=CN, 1=MCI, 2=AD
    - plasma_values: FloatTensor (5,)
    - plasma_mask: BoolTensor (5,)   # 有效为 True，缺失为 False
    """
    
    def __init__(
        self,
        csv_path: str | Path,
        cache_dir: str | Path,
        plasma_keys: List[str] = None,
        class_names: List[str] = None,
        plasma_stats: Dict[str, Dict[str, float]] = None,
        diagnosis_csv: str | Path = None,
        diagnosis_code_map: Dict[Any, str] | None = None,
        subset_indices: List[int] = None,
        skip_cache_set: set = None,
    ):
        """
        Args:
            csv_path: pairs_withPlasma.csv 路径
            cache_dir: .vision.pt 缓存目录
            plasma_keys: plasma 字段名列表
            class_names: 诊断类别名，如 ["CN", "MCI", "AD"]
            plasma_stats: plasma 归一化统计；若 None 则自动计算
            diagnosis_csv: 可选的诊断标签 CSV（若主 CSV 无 diagnosis 字段）
            subset_indices: 子集索引（用于 train/val 划分）
            skip_cache_set: Set of (ptid, tau_id) tuples to skip (missing caches)
        """
        self.csv_path = Path(csv_path)
        self.cache_dir = Path(cache_dir)
        self.plasma_keys = plasma_keys or DEFAULT_PLASMA_KEYS
        self.class_names = class_names or ["CN", "MCI", "AD"]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        # diagnosis 数值映射（优先使用外部传入，其次默认 1/2/3）
        self.diagnosis_code_map = diagnosis_code_map or DEFAULT_DIAGNOSIS_CODE_MAP
        self.skip_cache_set = skip_cache_set or set()
        
        # 加载 CSV - 显式指定 image_id 相关列为字符串，避免被解析为 float
        # 只有 plasma 相关字段是数值，其余字段都应当是字符串
        id_columns = ["PTID", "id_mri", "id_fdg", "id_av45", "id_av1451", "plasma_source"]
        dtype_spec = {col: str for col in id_columns}
        self.df = pd.read_csv(self.csv_path, dtype=dtype_spec)
        
        # 如果有外部诊断 CSV，合并
        if diagnosis_csv is not None:
            diag_df = pd.read_csv(diagnosis_csv)
            # 假设 PTID 列可以 join
            self.df = self.df.merge(diag_df[["PTID", "diagnosis"]], on="PTID", how="left")
        
        # 过滤只有 TAU (id_av1451) 的样本 - 注意 dtype=str 时 NaN 变为字符串 "nan"
        self.df = self.df[
            (self.df["id_av1451"].notna()) & (self.df["id_av1451"] != "nan")
        ].reset_index(drop=True)
        
        # 计算或使用提供的 plasma 统计（按 source 分别计算）
        self.source_col = "plasma_source"
        self.na_value = -4.0  # 表示缺失值的特殊值
        if plasma_stats is None:
            self.plasma_stats = compute_plasma_stats(
                self.df, self.plasma_keys, 
                by_source=True, 
                source_col=self.source_col,
                na_value=self.na_value,
            )
            # 打印统计量供调试
            print(f"[Dataset] Plasma stats by source:")
            for source, source_stats in self.plasma_stats.items():
                print(f"  {source}:")
                for key, s in source_stats.items():
                    print(f"    {key}: mean={s['mean']:.4f}, std={s['std']:.4f}")
        else:
            self.plasma_stats = plasma_stats
        
        # 构建样本列表
        self.samples = self._build_samples()
        
        # 应用子集
        if subset_indices is not None:
            self.samples = [self.samples[i] for i in subset_indices]
        
        # 缓存维度信息（首次加载时填充）
        self.cached_token_dim: Optional[int] = None
        self.cached_cls_dim: Optional[int] = None
        self._prime_dims()
    
    def _build_samples(self) -> List[Dict[str, Any]]:
        """构建样本列表，每行 CSV 对应一个样本"""
        samples = []
        skipped = 0
        for idx, row in self.df.iterrows():
            ptid = str(row["PTID"])
            tau_id = str(row["id_av1451"])
            
            # 跳过缺失缓存的样本
            if (ptid, tau_id) in self.skip_cache_set:
                skipped += 1
                continue
            
            # 诊断标签（尝试多个字段）
            diag_raw = row.get("diagnosis") or row.get("DX") or row.get("research_group") or row.get("DIAGNOSIS")
            diagnosis = normalize_diagnosis(diag_raw, diagnosis_code_map=self.diagnosis_code_map)
            
            # plasma source
            plasma_source = row.get(self.source_col, "UPENN")  # 默认 UPENN
            if pd.isna(plasma_source):
                plasma_source = "UPENN"
            
            # plasma 值和 mask
            plasma_vals = []
            plasma_mask = []
            for key in self.plasma_keys:
                val = row.get(key)
                if val is None or pd.isna(val):
                    plasma_vals.append(0.0)
                    plasma_mask.append(False)
                else:
                    try:
                        fval = float(val)
                        # 检查是否为 NA 值（-4 或负值）
                        if fval < 0 or fval == self.na_value:
                            plasma_vals.append(0.0)
                            plasma_mask.append(False)
                        else:
                            # C2N 的 pT217_AB42_F 是百分比单位，需要除以 100 转换为小数
                            if key == "pT217_AB42_F" and plasma_source == "C2N":
                                fval = fval / 100.0
                            plasma_vals.append(fval)
                            plasma_mask.append(True)
                    except (ValueError, TypeError):
                        plasma_vals.append(0.0)
                        plasma_mask.append(False)
            
            # 缓存文件名：{subject_id}_{image_id}.vision.pt
            # image_id 来自 TAU ID (id_av1451)
            cache_name = f"{ptid}_{tau_id}.vision.pt"
            cache_path = self.cache_dir / cache_name
            
            samples.append({
                "subject_id": ptid,
                "tau_id": tau_id,
                "diagnosis": diagnosis,  # 可能为 None
                "plasma_values": plasma_vals,
                "plasma_mask": plasma_mask,
                "plasma_source": plasma_source,  # 用于选择归一化统计量
                "cache_path": str(cache_path),
            })
        
        return samples
    
    def _prime_dims(self) -> None:
        """预加载一个缓存文件以获取维度信息"""
        for sample in self.samples:
            cache_path = Path(sample["cache_path"])
            if cache_path.exists():
                try:
                    payload = torch.load(str(cache_path), map_location="cpu")
                    cls_token = payload.get("cls_token")
                    region_token = payload.get("region_token")
                    if cls_token is not None:
                        self.cached_cls_dim = int(cls_token.numel())
                    if region_token is not None:
                        self.cached_token_dim = region_token.shape[-1]
                    break
                except Exception:
                    continue
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # ===============================
        # 加载缓存的 vision embedding
        # ===============================
        cache_path = Path(sample["cache_path"])
        
        # 缓存不存在时抛出异常，不使用默认值（防止无效数据参与训练）
        if not cache_path.exists():
            raise FileNotFoundError(
                f"缓存文件不存在: {cache_path}\n"
                f"请先运行 precompute_cache.py 生成缓存，或检查缓存目录配置。"
            )
        
        try:
            payload = torch.load(str(cache_path), map_location="cpu")
            # payload 结构: {"cls_token": (D,), "region_token": (N, D), ...}
            tau_cls = payload.get("cls_token")
            tau_tokens = payload.get("region_token")
            
            if tau_cls is None or tau_tokens is None:
                raise ValueError(f"缓存文件格式错误: {cache_path}")
            
            # 确保是 tensor
            if not isinstance(tau_cls, torch.Tensor):
                tau_cls = torch.as_tensor(tau_cls, dtype=torch.float32)
            if not isinstance(tau_tokens, torch.Tensor):
                tau_tokens = torch.as_tensor(tau_tokens, dtype=torch.float32)
            
            # 展平 cls_token
            tau_cls = tau_cls.view(-1)
            
            # tau_tokens: (N, D)
            if tau_tokens.dim() == 1:
                tau_tokens = tau_tokens.unsqueeze(0)
            elif tau_tokens.dim() > 2:
                tau_tokens = tau_tokens.view(-1, tau_tokens.shape[-1])
                
        except FileNotFoundError:
            raise
        except Exception as e:
            raise RuntimeError(f"加载缓存失败 {cache_path}: {e}")
        
        # ===============================
        # 诊断标签
        # ===============================
        diagnosis = sample["diagnosis"]
        if diagnosis is not None and diagnosis in self.class_to_idx:
            diagnosis_id = self.class_to_idx[diagnosis]
        else:
            # 若无诊断标签，默认为 0（或可选其他处理）
            diagnosis_id = -1  # 标记为无效
        
        # ===============================
        # Plasma 值：z-score 归一化（按 source 分别处理）
        # ===============================
        plasma_raw = sample["plasma_values"]  # List[float]
        plasma_mask = sample["plasma_mask"]   # List[bool]
        plasma_source = sample.get("plasma_source", "UPENN")
        
        # 获取该 source 的统计量
        if isinstance(self.plasma_stats, dict) and plasma_source in self.plasma_stats:
            source_stats = self.plasma_stats[plasma_source]
        else:
            # 兼容旧格式或未知 source
            source_stats = self.plasma_stats
        
        plasma_norm = []
        for i, key in enumerate(self.plasma_keys):
            stats = source_stats.get(key, {"mean": 0.0, "std": 1.0})
            mean_val = stats.get("mean", 0.0)
            std_val = stats.get("std", 1.0)
            if std_val < 1e-8:
                std_val = 1.0
            if plasma_mask[i]:
                norm = (plasma_raw[i] - mean_val) / std_val
            else:
                norm = 0.0  # 缺失值填 0（mask 会屏蔽）
            plasma_norm.append(norm)
        
        return {
            "subject_id": sample["subject_id"],
            # tau_tokens: (N, Dv) - patch tokens 用于 pooling 和 ContextNet
            # 尺寸说明：N=num_patches（如 196 或 ROI 数量），Dv=token_dim（如 768）
            "tau_tokens": tau_tokens.float(),
            # tau_cls: (D_clip,) - cls token，D_clip=512
            "tau_cls": tau_cls.float(),
            # diagnosis_id: () - 诊断类别 ID
            "diagnosis_id": torch.tensor(diagnosis_id, dtype=torch.long),
            # plasma_values: (5,) - z-score 归一化后的 plasma 值（按 source 分别归一化）
            "plasma_values": torch.tensor(plasma_norm, dtype=torch.float32),
            # plasma_mask: (5,) - 有效 mask，True 表示该 plasma 值有效
            "plasma_mask": torch.tensor(plasma_mask, dtype=torch.bool),
        }


# ============================================================================
# Subject-based Sampler
# ============================================================================

class SubjectBatchSampler(Sampler):
    """
    以 subject 为单位采样的 Batch Sampler
    
    确保同一 batch 内每个 subject 只出现一次，
    避免同 subject 多条记录成为伪负样本。
    """
    
    def __init__(
        self,
        dataset: TAUPlasmaDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        
        # 构建 subject -> sample_indices 映射
        self.subject_to_indices: Dict[str, List[int]] = {}
        for idx, sample in enumerate(dataset.samples):
            sid = sample["subject_id"]
            if sid not in self.subject_to_indices:
                self.subject_to_indices[sid] = []
            self.subject_to_indices[sid].append(idx)
        
        self.subjects = list(self.subject_to_indices.keys())
    
    def __iter__(self):
        rng = random.Random(self.seed)
        
        # 每个 subject 随机选一个样本
        subject_indices = []
        for sid in self.subjects:
            indices = self.subject_to_indices[sid]
            # 随机选一个（若该 subject 有多条记录）
            chosen = rng.choice(indices)
            subject_indices.append(chosen)
        
        if self.shuffle:
            rng.shuffle(subject_indices)
        
        # 分 batch
        batch = []
        for idx in subject_indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        
        if batch and not self.drop_last:
            yield batch
    
    def __len__(self) -> int:
        n_subjects = len(self.subjects)
        if self.drop_last:
            return n_subjects // self.batch_size
        return (n_subjects + self.batch_size - 1) // self.batch_size


# ============================================================================
# Collate 函数
# ============================================================================

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    将样本列表整理为 batch tensor
    
    输出 batch 字段（与 models.py forward 接口对齐）：
    - subjects: List[str] (B,)
    - patch_emb: Tensor (B, N, Dv) - patch tokens
    - img_emb: Tensor (B, D_clip) - cls embedding
    - label_idx: LongTensor (B,) - 诊断类别 ID
    - plasma_vals: FloatTensor (B, 5) - z-score plasma
    - plasma_mask: BoolTensor (B, 5) - 有效 mask
    """
    subjects = [s["subject_id"] for s in batch]
    
    # Stack tensors
    patch_emb = torch.stack([s["tau_tokens"] for s in batch], dim=0)
    img_emb = torch.stack([s["tau_cls"] for s in batch], dim=0)
    label_idx = torch.stack([s["diagnosis_id"] for s in batch], dim=0)
    plasma_vals = torch.stack([s["plasma_values"] for s in batch], dim=0)
    plasma_mask = torch.stack([s["plasma_mask"] for s in batch], dim=0)
    
    return {
        "subjects": subjects,
        # patch_emb: (B, N, Dv) - B=batch_size, N=num_patches, Dv=token_dim
        "patch_emb": patch_emb,
        # img_emb: (B, D_clip) - cls embedding，作为 image representation
        "img_emb": img_emb,
        # label_idx: (B,) - 诊断类别 ID
        "label_idx": label_idx,
        # plasma_vals: (B, 5) - z-score 归一化的 plasma
        "plasma_vals": plasma_vals,
        # plasma_mask: (B, 5) - 有效 mask
        "plasma_mask": plasma_mask,
    }


# ============================================================================
# 数据划分与 DataLoader 构建
# ============================================================================

def split_by_subject(
    dataset: TAUPlasmaDataset,
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """
    按 subject 划分训练/验证集，确保同一 subject 不同时出现在两个集合
    
    Returns:
        (train_indices, val_indices)
    """
    # 收集所有 unique subjects
    subject_to_indices: Dict[str, List[int]] = {}
    for idx, sample in enumerate(dataset.samples):
        sid = sample["subject_id"]
        if sid not in subject_to_indices:
            subject_to_indices[sid] = []
        subject_to_indices[sid].append(idx)
    
    subjects = list(subject_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(subjects)
    
    val_count = max(1, int(len(subjects) * val_ratio))
    val_subjects = set(subjects[:val_count])
    
    train_indices = []
    val_indices = []
    for sid, indices in subject_to_indices.items():
        if sid in val_subjects:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)
    
    return train_indices, val_indices


def build_dataloaders(
    csv_path: str | Path,
    cache_dir: str | Path,
    batch_size: int,
    val_ratio: float = 0.15,
    seed: int = 42,
    plasma_keys: List[str] = None,
    class_names: List[str] = None,
    num_workers: int = 0,
    skip_cache_set: set = None,
) -> Tuple[DataLoader, DataLoader, Dict[str, Dict[str, float]]]:
    """
    构建训练和验证 DataLoader
    
    Returns:
        (train_loader, val_loader, plasma_stats)
    """
    # 先加载完整数据集以计算 plasma stats
    full_dataset = TAUPlasmaDataset(
        csv_path=csv_path,
        cache_dir=cache_dir,
        plasma_keys=plasma_keys,
        class_names=class_names,
        skip_cache_set=skip_cache_set,
    )
    plasma_stats = full_dataset.plasma_stats
    
    # 划分
    train_indices, val_indices = split_by_subject(full_dataset, val_ratio, seed)
    
    # 重新构建带子集的数据集
    train_dataset = TAUPlasmaDataset(
        csv_path=csv_path,
        cache_dir=cache_dir,
        plasma_keys=plasma_keys,
        class_names=class_names,
        plasma_stats=plasma_stats,
        subset_indices=train_indices,
        skip_cache_set=skip_cache_set,
    )
    val_dataset = TAUPlasmaDataset(
        csv_path=csv_path,
        cache_dir=cache_dir,
        plasma_keys=plasma_keys,
        class_names=class_names,
        plasma_stats=plasma_stats,
        subset_indices=val_indices,
        skip_cache_set=skip_cache_set,
    )
    
    # 构建 DataLoader
    train_sampler = SubjectBatchSampler(
        train_dataset, batch_size, shuffle=True, drop_last=False, seed=seed
    )
    val_sampler = SubjectBatchSampler(
        val_dataset, batch_size, shuffle=False, drop_last=False, seed=seed
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    print(f"[Dataset] Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    print(f"[Dataset] Plasma stats: {plasma_stats}")
    
    return train_loader, val_loader, plasma_stats
