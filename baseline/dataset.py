"""
baseline/dataset.py
===================
MRI → TAU-PET 配对数据集实现

功能：
1. 分层抽样的 train/val/test 划分
2. 中央裁剪至目标尺寸并归一化
3. 加权采样支持
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from config import Config


# ============================================================================
# 诊断标签规范化
# ============================================================================

DIAGNOSIS_MAP = {
    1: "CN", 1.0: "CN",
    2: "MCI", 2.0: "MCI",
    3: "AD", 3.0: "AD",
    "CN": "CN", "NORMAL": "CN", "COGNITIVELY NORMAL": "CN",
    "MCI": "MCI", "EMCI": "MCI", "LMCI": "MCI",
    "AD": "AD", "Dementia": "AD",
}


def normalize_diagnosis(raw) -> Optional[str]:
    """将诊断标签规范化为 CN/MCI/AD"""
    if pd.isna(raw):
        return None
    # 尝试数值映射
    if isinstance(raw, (int, float)):
        if np.isfinite(raw):
            key = int(raw) if float(raw).is_integer() else raw
            if key in DIAGNOSIS_MAP:
                return DIAGNOSIS_MAP[key]
    # 字符串映射
    if isinstance(raw, str):
        upper = raw.strip().upper()
        for k, v in DIAGNOSIS_MAP.items():
            if isinstance(k, str) and k.upper() == upper:
                return v
    return None


# ============================================================================
# 数据准备
# ============================================================================

def find_nifti_file(
    adni_root: str,
    subdir: str,
    ptid: str,
    mri_id: str,
    target_id: str,
) -> Optional[Path]:
    """
    查找 NIfTI 文件
    
    命名格式: {PTID}__I{MRI_ID}__I{TARGET_ID}.nii.gz
    """
    base_dir = Path(adni_root) / subdir
    
    # 标准命名
    patterns = [
        f"{ptid}__I{mri_id}__I{target_id}.nii.gz",
        f"{ptid}__I{mri_id}__I{target_id}_full.nii.gz",
    ]
    
    for pattern in patterns:
        path = base_dir / pattern
        if path.exists():
            return path
    
    # 通配符匹配
    wildcard = str(base_dir / f"{ptid}__*__I{target_id}.nii.gz")
    matches = glob.glob(wildcard)
    if matches:
        return Path(matches[0])
    
    return None


def find_mri_file(
    adni_root: str,
    mri_subdir: str,
    ptid: str,
    mri_id: str,
) -> Optional[Path]:
    """查找 MRI 文件"""
    base_dir = Path(adni_root) / mri_subdir
    
    patterns = [
        f"{ptid}__I{mri_id}.nii.gz",
        f"{ptid}__I{mri_id}_stripped.nii.gz",
        f"I{mri_id}.nii.gz",
    ]
    
    for pattern in patterns:
        path = base_dir / pattern
        if path.exists():
            return path
    
    # 通配符
    wildcard = str(base_dir / f"*{mri_id}*.nii.gz")
    matches = glob.glob(wildcard)
    if matches:
        return Path(matches[0])
    
    return None


def build_sample_list(config: Config) -> pd.DataFrame:
    """
    从 CSV 构建样本列表，合并 QC 和厂商信息
    
    Returns:
        DataFrame with columns:
        - ptid, mri_id, tau_id
        - mri_path, tau_path
        - diagnosis, quality_class, train_weight
        - pet_mfr (厂商)
    """
    # 读取 pairs 表
    pairs_df = pd.read_csv(config.data.pairs_csv)
    pairs_df.columns = [c.lower() for c in pairs_df.columns]
    
    # 标准化列名
    col_map = {
        "ptid": "ptid",
        "id_mri": "mri_id",
        "id_av1451": "tau_id",
        "diagnosis": "diagnosis",
    }
    
    for old, new in col_map.items():
        if old in pairs_df.columns:
            pairs_df[new] = pairs_df[old]
    
    # 只保留有 TAU 的行
    pairs_df = pairs_df[pairs_df["tau_id"].notna()].copy()
    pairs_df["tau_id"] = pairs_df["tau_id"].astype(int).astype(str)
    pairs_df["mri_id"] = pairs_df["mri_id"].astype(int).astype(str)
    
    print(f"[build_sample_list] pairs 中有 TAU 的记录: {len(pairs_df)}")
    
    # 读取 QC 结果
    qc_df = pd.read_csv(config.data.qc_csv)
    qc_df["id_mri"] = qc_df["id_mri"].astype(str)
    qc_df["id_tau"] = qc_df["id_tau"].apply(lambda x: str(int(float(x))) if pd.notna(x) else None)
    
    # 合并 QC 信息
    pairs_df = pairs_df.merge(
        qc_df[["id_mri", "id_tau", "quality_class", "train_weight"]],
        left_on=["mri_id", "tau_id"],
        right_on=["id_mri", "id_tau"],
        how="left"
    )
    
    # 填充缺失的 QC
    pairs_df["quality_class"] = pairs_df["quality_class"].fillna("Medium")
    pairs_df["train_weight"] = pairs_df["train_weight"].fillna(0.7)
    
    # 读取 PET 厂商信息
    if os.path.exists(config.data.pet_info_csv):
        pet_df = pd.read_csv(config.data.pet_info_csv)
        # 筛选 AV1451
        tau_pet = pet_df[pet_df["pet_radiopharm"].str.contains("AV1451", na=False, case=False)].copy()
        tau_pet["image_id"] = tau_pet["image_id"].astype(str)
        
        pairs_df = pairs_df.merge(
            tau_pet[["image_id", "pet_mfr"]],
            left_on="tau_id",
            right_on="image_id",
            how="left"
        )
        pairs_df["pet_mfr"] = pairs_df["pet_mfr"].fillna("Unknown")
    else:
        pairs_df["pet_mfr"] = "Unknown"
    
    # 规范化诊断
    pairs_df["diagnosis"] = pairs_df["diagnosis"].apply(normalize_diagnosis)
    pairs_df["diagnosis"] = pairs_df["diagnosis"].fillna("Unknown")
    
    # 查找文件路径
    samples = []
    missing_mri = 0
    missing_tau = 0
    
    for _, row in pairs_df.iterrows():
        ptid = str(row["ptid"])
        mri_id = str(row["mri_id"])
        tau_id = str(row["tau_id"])
        
        # 查找 MRI
        mri_path = find_mri_file(
            config.data.adni_root,
            config.data.mri_subdir,
            ptid, mri_id
        )
        
        # 查找 TAU
        tau_path = find_nifti_file(
            config.data.adni_root,
            config.data.tau_subdir,
            ptid, mri_id, tau_id
        )
        
        if mri_path is None:
            missing_mri += 1
            continue
        if tau_path is None:
            missing_tau += 1
            continue
        
        samples.append({
            "ptid": ptid,
            "mri_id": mri_id,
            "tau_id": tau_id,
            "mri_path": str(mri_path),
            "tau_path": str(tau_path),
            "diagnosis": row["diagnosis"],
            "quality_class": row["quality_class"],
            "train_weight": row["train_weight"],
            "pet_mfr": row["pet_mfr"],
        })
    
    print(f"[build_sample_list] 有效样本: {len(samples)}, 缺失 MRI: {missing_mri}, 缺失 TAU: {missing_tau}")
    
    return pd.DataFrame(samples)


def stratified_split(
    df: pd.DataFrame,
    config: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    分层抽样划分 train/val/test
    
    分层变量: pet_mfr, diagnosis, quality_class
    """
    # 创建分层键
    stratify_cols = [c for c in config.data.stratify_by if c in df.columns]
    
    if stratify_cols:
        df["_stratify_key"] = df[stratify_cols].apply(
            lambda x: "_".join(x.astype(str)), axis=1
        )
    else:
        df["_stratify_key"] = "all"
    
    # 确保每个分层至少有 3 个样本
    key_counts = df["_stratify_key"].value_counts()
    rare_keys = key_counts[key_counts < 3].index
    df.loc[df["_stratify_key"].isin(rare_keys), "_stratify_key"] = "rare_combined"
    
    # 第一次划分: train vs (val + test)
    val_test_ratio = config.data.val_ratio + config.data.test_ratio
    train_df, val_test_df = train_test_split(
        df,
        test_size=val_test_ratio,
        stratify=df["_stratify_key"],
        random_state=config.data.seed,
    )
    
    # 第二次划分: val vs test
    test_ratio = config.data.test_ratio / val_test_ratio
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=test_ratio,
        stratify=val_test_df["_stratify_key"],
        random_state=config.data.seed,
    )
    
    # 删除辅助列
    for df_ in [train_df, val_df, test_df]:
        df_.drop(columns=["_stratify_key"], inplace=True, errors="ignore")
    
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def print_split_statistics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """打印并保存划分统计"""
    splits = {"train": train_df, "val": val_df, "test": test_df}
    
    stats_lines = []
    stats_lines.append("=" * 70)
    stats_lines.append("数据划分统计")
    stats_lines.append("=" * 70)
    
    for name, df in splits.items():
        stats_lines.append(f"\n{name.upper()} 集: {len(df)} 样本")
        
        # 诊断分布
        if "diagnosis" in df.columns:
            stats_lines.append(f"  诊断分布: {df['diagnosis'].value_counts().to_dict()}")
        
        # 厂商分布
        if "pet_mfr" in df.columns:
            stats_lines.append(f"  厂商分布: {df['pet_mfr'].value_counts().to_dict()}")
        
        # 质量分布
        if "quality_class" in df.columns:
            stats_lines.append(f"  质量分布: {df['quality_class'].value_counts().to_dict()}")
    
    stats_lines.append("=" * 70)
    
    stats_text = "\n".join(stats_lines)
    print(stats_text)
    
    if save_path:
        with open(save_path, "w") as f:
            f.write(stats_text)
        
        # 保存 CSV
        for name, df in splits.items():
            df.to_csv(save_path.replace(".txt", f"_{name}.csv"), index=False)


# ============================================================================
# 数据处理
# ============================================================================

def center_crop_3d(volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    中央裁剪 3D 体积到目标尺寸
    
    如果原尺寸小于目标，则进行零填充
    """
    current_shape = volume.shape
    result = np.zeros(target_shape, dtype=volume.dtype)
    
    # 计算裁剪或填充的起始位置
    src_starts = []
    src_ends = []
    dst_starts = []
    dst_ends = []
    
    for i in range(3):
        if current_shape[i] >= target_shape[i]:
            # 需要裁剪
            start = (current_shape[i] - target_shape[i]) // 2
            src_starts.append(start)
            src_ends.append(start + target_shape[i])
            dst_starts.append(0)
            dst_ends.append(target_shape[i])
        else:
            # 需要填充
            start = (target_shape[i] - current_shape[i]) // 2
            src_starts.append(0)
            src_ends.append(current_shape[i])
            dst_starts.append(start)
            dst_ends.append(start + current_shape[i])
    
    result[
        dst_starts[0]:dst_ends[0],
        dst_starts[1]:dst_ends[1],
        dst_starts[2]:dst_ends[2],
    ] = volume[
        src_starts[0]:src_ends[0],
        src_starts[1]:src_ends[1],
        src_starts[2]:src_ends[2],
    ]
    
    return result


def normalize_volume(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """归一化到 [0, 1]"""
    v_min = volume.min()
    v_max = volume.max()
    if v_max - v_min < eps:
        return np.zeros_like(volume)
    return (volume - v_min) / (v_max - v_min + eps)


# ============================================================================
# Dataset 类
# ============================================================================

class MRITauDataset(Dataset):
    """
    MRI → TAU-PET 配对数据集
    
    每次返回:
    - mri: float32 tensor, [1, D, H, W]
    - tau: float32 tensor, [1, D, H, W]
    - weight: float32 标量
    - meta: dict (ptid, mri_path, tau_path, diagnosis, quality_class, pet_mfr)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        target_shape: Tuple[int, int, int] = (160, 192, 160),
        return_weight: bool = True,
        augment: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.target_shape = target_shape
        self.return_weight = return_weight
        self.augment = augment
        
        # 验证
        self.bad_samples = []
    
    def __len__(self) -> int:
        return len(self.df)
    
    def _load_and_preprocess(self, path: str) -> np.ndarray:
        """加载并预处理 NIfTI 文件"""
        img = nib.load(path)
        data = img.get_fdata(dtype=np.float32)
        
        # 中央裁剪
        data = center_crop_3d(data, self.target_shape)
        
        # 归一化
        data = normalize_volume(data)
        
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        
        # 加载数据
        try:
            mri = self._load_and_preprocess(row["mri_path"])
            tau = self._load_and_preprocess(row["tau_path"])
        except Exception as e:
            print(f"[WARNING] 加载失败 idx={idx}, ptid={row['ptid']}: {e}")
            self.bad_samples.append(idx)
            # 返回零数据
            mri = np.zeros(self.target_shape, dtype=np.float32)
            tau = np.zeros(self.target_shape, dtype=np.float32)
        
        # 验证 shape
        assert mri.shape == self.target_shape, f"MRI shape mismatch: {mri.shape} != {self.target_shape}"
        assert tau.shape == self.target_shape, f"TAU shape mismatch: {tau.shape} != {self.target_shape}"
        
        # 转为 tensor，添加通道维度
        mri_tensor = torch.from_numpy(mri).unsqueeze(0)  # [1, D, H, W]
        tau_tensor = torch.from_numpy(tau).unsqueeze(0)  # [1, D, H, W]
        
        # 数据增强
        if self.augment:
            mri_tensor, tau_tensor = self._augment(mri_tensor, tau_tensor)
        
        # 构建返回值
        result = {
            "mri": mri_tensor,
            "tau": tau_tensor,
            "meta": {
                "ptid": row["ptid"],
                "mri_id": row["mri_id"],
                "tau_id": row["tau_id"],
                "mri_path": row["mri_path"],
                "tau_path": row["tau_path"],
                "diagnosis": row["diagnosis"],
                "quality_class": row["quality_class"],
                "pet_mfr": row["pet_mfr"],
            }
        }
        
        if self.return_weight:
            result["weight"] = torch.tensor(row["train_weight"], dtype=torch.float32)
        else:
            result["weight"] = torch.tensor(1.0, dtype=torch.float32)
        
        return result
    
    def _augment(
        self,
        mri: torch.Tensor,
        tau: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """简单的数据增强"""
        # 随机翻转
        if torch.rand(1) > 0.5:
            mri = torch.flip(mri, dims=[1])  # D
            tau = torch.flip(tau, dims=[1])
        if torch.rand(1) > 0.5:
            mri = torch.flip(mri, dims=[2])  # H
            tau = torch.flip(tau, dims=[2])
        if torch.rand(1) > 0.5:
            mri = torch.flip(mri, dims=[3])  # W
            tau = torch.flip(tau, dims=[3])
        
        return mri, tau


# ============================================================================
# DataLoader 工厂
# ============================================================================

def create_dataloaders(
    config: Config,
) -> Tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    创建 train/val/test DataLoader
    
    Returns:
        train_loader, val_loader, test_loader, train_df, val_df, test_df
    """
    # 构建样本列表
    sample_df = build_sample_list(config)
    
    # 分层划分
    train_df, val_df, test_df = stratified_split(sample_df, config)
    
    # 打印统计
    stats_path = os.path.join(config.output_dir, "split_statistics.txt")
    print_split_statistics(train_df, val_df, test_df, stats_path)
    
    # 创建 Dataset
    train_dataset = MRITauDataset(
        train_df,
        target_shape=config.data.target_shape,
        return_weight=True,
        augment=True,
    )
    val_dataset = MRITauDataset(
        val_df,
        target_shape=config.data.target_shape,
        return_weight=False,
        augment=False,
    )
    test_dataset = MRITauDataset(
        test_df,
        target_shape=config.data.target_shape,
        return_weight=False,
        augment=False,
    )
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # 测试时逐个处理
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    
    return train_loader, val_loader, test_loader, train_df, val_df, test_df


if __name__ == "__main__":
    # 测试代码
    from config import get_default_config
    
    config = get_default_config()
    
    # 构建样本列表
    sample_df = build_sample_list(config)
    print(f"\n样本总数: {len(sample_df)}")
    print(sample_df.head())
    
    # 分层划分
    train_df, val_df, test_df = stratified_split(sample_df, config)
    print_split_statistics(train_df, val_df, test_df)
