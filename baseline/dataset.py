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
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from .config import Config
from .condition import (
    TabularStats,
    CLINICAL_FIELDS,
    PLASMA_FIELDS,
    LOG1P_FIELDS,
    normalize_source,
    source_to_id,
)


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
    # 处理 None 和 NaN
    if raw is None:
        return None
    try:
        if pd.isna(raw):
            return None
    except (ValueError, TypeError):
        pass
    
    # 尝试数值映射
    if isinstance(raw, (int, float)):
        try:
            if np.isfinite(raw):
                key = int(raw) if float(raw).is_integer() else raw
                if key in DIAGNOSIS_MAP:
                    return DIAGNOSIS_MAP[key]
        except (ValueError, TypeError):
            pass
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
    target_pet = str(getattr(config.data, "target_pet", "tau")).strip().lower()
    if target_pet == "tau":
        target_id_col = "id_av1451"
        target_subdir = config.data.tau_subdir
        radiopharm_keyword = "AV1451"
        use_tau_qc = True
    elif target_pet == "fdg":
        target_id_col = "id_fdg"
        target_subdir = config.data.fdg_subdir
        radiopharm_keyword = "FDG"
        use_tau_qc = False
    elif target_pet == "av45":
        target_id_col = "id_av45"
        target_subdir = config.data.av45_subdir
        radiopharm_keyword = "AV45"
        use_tau_qc = False
    else:
        raise ValueError(f"不支持的 target_pet={target_pet}，仅支持 tau/fdg/av45")

    def _normalize_img_id(x) -> Optional[str]:
        if x is None:
            return None
        try:
            if pd.isna(x):
                return None
        except (ValueError, TypeError):
            pass
        try:
            return str(int(float(x)))
        except (ValueError, TypeError):
            sx = str(x).strip()
            return sx if sx else None

    # 读取 pairs 表
    pairs_df = pd.read_csv(config.data.pairs_csv)
    
    # 处理重复列名：保留第一个，删除后续同名列
    orig_cols = pairs_df.columns.tolist()
    lower_cols = [c.lower() for c in orig_cols]
    seen = {}
    keep_idx = []
    for i, lc in enumerate(lower_cols):
        if lc not in seen:
            seen[lc] = i
            keep_idx.append(i)
    pairs_df = pairs_df.iloc[:, keep_idx]
    pairs_df.columns = [lower_cols[i] for i in keep_idx]
    
    # 标准化列名
    col_map = {
        "ptid": "ptid",
        "id_mri": "mri_id",
        "diagnosis": "diagnosis",
    }
    
    for old, new in col_map.items():
        if old in pairs_df.columns:
            pairs_df[new] = pairs_df[old]
    
    if target_id_col not in pairs_df.columns:
        raise KeyError(f"pairs_csv 缺少目标 PET 列: {target_id_col}")

    # 兼容下游字段命名，统一将当前目标 PET id 写入 tau_id
    pairs_df["mri_id"] = pairs_df["mri_id"].apply(_normalize_img_id)
    pairs_df["tau_id"] = pairs_df[target_id_col].apply(_normalize_img_id)

    # 只保留有目标 PET 的行
    pairs_df = pairs_df[pairs_df["tau_id"].notna()].copy()
    pairs_df = pairs_df[pairs_df["mri_id"].notna()].copy()

    print(f"[build_sample_list] target_pet={target_pet}, pairs 中有目标 PET 的记录: {len(pairs_df)}")

    # 仅 TAU 使用现有 QC 权重；FDG/AV45 默认权重
    if use_tau_qc and os.path.exists(config.data.qc_csv):
        qc_df = pd.read_csv(config.data.qc_csv)
        qc_df["id_mri"] = qc_df["id_mri"].apply(_normalize_img_id)
        qc_df["id_tau"] = qc_df["id_tau"].apply(_normalize_img_id)

        pairs_df = pairs_df.merge(
            qc_df[["id_mri", "id_tau", "quality_class", "train_weight"]],
            left_on=["mri_id", "tau_id"],
            right_on=["id_mri", "id_tau"],
            how="left"
        )
        pairs_df["quality_class"] = pairs_df["quality_class"].fillna("Medium")
        pairs_df["train_weight"] = pairs_df["train_weight"].fillna(0.7)
    else:
        pairs_df["quality_class"] = "Medium"
        pairs_df["train_weight"] = 0.7

    # 读取 PET 厂商信息
    if os.path.exists(config.data.pet_info_csv):
        pet_df = pd.read_csv(config.data.pet_info_csv)
        target_pet_df = pet_df[pet_df["pet_radiopharm"].str.contains(radiopharm_keyword, na=False, case=False)].copy()
        target_pet_df["image_id"] = target_pet_df["image_id"].apply(_normalize_img_id)

        pairs_df = pairs_df.merge(
            target_pet_df[["image_id", "pet_mfr"]],
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
    missing_target = 0
    
    extra_fields = list({
        *config.condition.clinical_fields,
        *config.condition.plasma_fields,
        "sex",
        config.condition.source_col,
    })

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
        
        # 查找目标 PET（字段仍沿用 tau_path/tau_id，以兼容下游代码）
        tau_path = find_nifti_file(
            config.data.adni_root,
            target_subdir,
            ptid, mri_id, tau_id
        )
        
        if mri_path is None:
            missing_mri += 1
            continue
        if tau_path is None:
            missing_target += 1
            continue
        # 跳过 _zero.nii.gz（表示该被试缺少此模态）
        if _is_missing_modal_path(str(mri_path)):
            missing_mri += 1
            continue
        if _is_missing_modal_path(str(tau_path)):
            missing_target += 1
            continue
        
        sample = {
            "ptid": ptid,
            "mri_id": mri_id,
            "tau_id": tau_id,
            "mri_path": str(mri_path),
            "tau_path": str(tau_path),
            "diagnosis": row["diagnosis"],
            "quality_class": row["quality_class"],
            "train_weight": row["train_weight"],
            "pet_mfr": row["pet_mfr"],
        }
        for field in extra_fields:
            if field in row.index:
                sample[field] = row[field]
            else:
                sample[field] = None
        samples.append(sample)
    
    print(f"[build_sample_list] 有效样本: {len(samples)}, 缺失 MRI: {missing_mri}, 缺失 {target_pet.upper()}: {missing_target}")
    
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
    
    # 第二次划分前重新检查分层键
    key_counts_2 = val_test_df["_stratify_key"].value_counts()
    rare_keys_2 = key_counts_2[key_counts_2 < 2].index
    if len(rare_keys_2) > 0:
        val_test_df = val_test_df.copy()
        val_test_df.loc[val_test_df["_stratify_key"].isin(rare_keys_2), "_stratify_key"] = "rare_combined_2"
    
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


def _is_missing_modal_path(path_value: Any) -> bool:
    """判断 JSON 中模态路径是否表示缺失（如 zero.nii.gz / NaN）。"""
    if path_value is None:
        return True
    try:
        if pd.isna(path_value):
            return True
    except (ValueError, TypeError):
        pass

    s = str(path_value).strip().lower()
    if s in {"", "nan", "none", "null"}:
        return True
    s_norm = s.replace("\\", "/")
    base_name = os.path.basename(s_norm)
    if base_name in {"zero.nii.gz", "zero.nii"}:
        return True
    if base_name.endswith("_zero.nii.gz") or base_name.endswith("_zero.nii"):
        return True
    if "/zero/" in s_norm and base_name.endswith(".nii.gz"):
        return True
    return False


def _load_subjects_from_paired_json(json_path: str, target_pet: Optional[str] = None) -> List[str]:
    """从 train/val/test_data_with_description.json 提取 subject 列表。"""
    if not json_path:
        return []
    with open(json_path, "r") as f:
        data = json.load(f)

    target_field = None
    if target_pet is not None:
        target_pet_norm = str(target_pet).strip().lower()
        target_field = {
            "tau": "tau",
            "fdg": "fdg",
            "av45": "av45",
        }.get(target_pet_norm)

    subjects = []
    for item in data:
        # 若指定了目标 PET，则过滤掉该模态缺失（zero.nii.gz 等）的记录
        if target_field is not None:
            if _is_missing_modal_path(item.get(target_field)):
                continue

        sid = item.get("name") or item.get("ptid") or item.get("Subject ID")
        if sid is not None:
            sid_str = str(sid)
            # name 字段常为 "PTID__exam_id"，固定划分按 PTID 对齐
            if "__" in sid_str:
                sid_str = sid_str.split("__", 1)[0]
            subjects.append(sid_str)
    return sorted(set(subjects))


def resolve_external_split_subjects(config: Config) -> Optional[Dict[str, List[str]]]:
    """
    解析外部固定划分。

    优先级:
    1) data.split_train_json / split_val_json / split_test_json: paired json（核心现有划分）
    2) data.split_subjects_json: {train_subjects, val_subjects, test_subjects?}
    """
    split_cfg = config.data

    if split_cfg.split_train_json and split_cfg.split_val_json:
        train_subjects = _load_subjects_from_paired_json(
            split_cfg.split_train_json,
            target_pet=split_cfg.target_pet,
        )
        val_subjects = _load_subjects_from_paired_json(
            split_cfg.split_val_json,
            target_pet=split_cfg.target_pet,
        )
        test_subjects = []
        if split_cfg.split_test_json:
            test_subjects = _load_subjects_from_paired_json(
                split_cfg.split_test_json,
                target_pet=split_cfg.target_pet,
            )
        elif split_cfg.split_fallback_test_from_val:
            test_subjects = list(val_subjects)
        return {
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "test_subjects": sorted(set(test_subjects)),
        }

    if split_cfg.split_subjects_json:
        with open(split_cfg.split_subjects_json, "r") as f:
            payload = json.load(f)
        train_subjects = payload.get("train_subjects", [])
        val_subjects = payload.get("val_subjects", [])
        test_subjects = payload.get("test_subjects", [])
        if (not test_subjects) and split_cfg.split_fallback_test_from_val:
            test_subjects = list(val_subjects)
        return {
            "train_subjects": sorted(set(map(str, train_subjects))),
            "val_subjects": sorted(set(map(str, val_subjects))),
            "test_subjects": sorted(set(map(str, test_subjects))),
        }

    return None


def external_subject_split(
    df: pd.DataFrame,
    split_subjects: Dict[str, List[str]],
    strict: bool = True,
    allow_val_test_overlap: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按外部 subject 列表执行固定划分。"""
    train_set = set(split_subjects.get("train_subjects", []))
    val_set = set(split_subjects.get("val_subjects", []))
    test_set = set(split_subjects.get("test_subjects", []))

    if strict:
        overlap = (train_set & val_set) | (train_set & test_set)
        if not allow_val_test_overlap:
            overlap = overlap | (val_set & test_set)
        if overlap:
            raise ValueError(f"外部划分存在重叠 subjects: {sorted(list(overlap))[:10]}")

    train_df = df[df["ptid"].astype(str).isin(train_set)].copy()
    val_df = df[df["ptid"].astype(str).isin(val_set)].copy()
    test_df = df[df["ptid"].astype(str).isin(test_set)].copy()

    if strict:
        if len(train_df) == 0:
            raise ValueError("外部划分后训练集为空，请检查 split 与样本表的 ptid 对齐。")
        if len(val_df) == 0:
            raise ValueError("外部划分后验证集为空，请检查 split 与样本表的 ptid 对齐。")
        if len(test_df) == 0:
            raise ValueError("外部划分后测试集为空，请提供 test_subjects 或启用 val->test 回退。")

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
        condition_mode: str = "none",
        tabular_stats: Optional[TabularStats] = None,
        source_col: str = "plasma_source",
        clinical_fields: Optional[List[str]] = None,
        plasma_fields: Optional[List[str]] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.target_shape = target_shape
        self.return_weight = return_weight
        self.augment = augment
        self.condition_mode = condition_mode
        self.tabular_stats = tabular_stats
        self.source_col = source_col
        self.clinical_fields = clinical_fields or CLINICAL_FIELDS
        self.plasma_fields = plasma_fields or PLASMA_FIELDS
        
        # 验证
        self.bad_samples = []
    
    def __len__(self) -> int:
        return len(self.df)
    
    def _load_and_preprocess(self, path: str) -> np.ndarray:
        """加载并预处理 NIfTI 文件"""
        img = nib.load(path)
        # Reorient to canonical (RAS+) so axis meaning is consistent across files.
        img = nib.as_closest_canonical(img)
        data = img.get_fdata(dtype=np.float32)
        # Convert from (X, Y, Z) to (D=Z, H=Y, W=X) for model/visualization.
        data = np.transpose(data, (2, 1, 0))
        
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

        if self.condition_mode != "none" and self.tabular_stats is not None:
            cond = self._build_condition(row)
            result.update(cond)
            result["meta"]["plasma_source"] = normalize_source(row.get(self.source_col))
            if "pt217_f" in row.index:
                result["meta"]["pt217_f"] = row.get("pt217_f")
        
        if self.return_weight:
            result["weight"] = torch.tensor(row["train_weight"], dtype=torch.float32)
        else:
            result["weight"] = torch.tensor(1.0, dtype=torch.float32)
        
        return result

    def _safe_float(self, x: Any) -> Optional[float]:
        if x is None:
            return None
        try:
            if pd.isna(x):
                return None
        except (ValueError, TypeError):
            pass
        try:
            return float(x)
        except (ValueError, TypeError):
            return None

    def _is_missing(self, value: Optional[float], plasma: bool = False) -> bool:
        if value is None:
            return True
        if not np.isfinite(value):
            return True
        if plasma and value <= -3.9:
            return True
        return False

    def _normalize_field(
        self,
        value: Optional[float],
        stats: Tuple[float, float],
        log1p: bool = False,
    ) -> Tuple[float, float]:
        if value is None or not np.isfinite(value):
            return 0.0, 1.0
        v = float(value)
        if log1p:
            v = float(np.log1p(v))
        median, iqr = stats
        return (v - median) / (iqr + 1e-8), 0.0

    def _build_condition(self, row: pd.Series) -> Dict[str, torch.Tensor]:
        clinical_vals = []
        clinical_mask = []
        plasma_vals = []
        plasma_mask = []

        plasma_source = row.get(self.source_col)
        plasma_stats = self.tabular_stats.get_plasma_stats(plasma_source)

        for field in self.clinical_fields:
            raw = row.get(field)
            v = self._safe_float(raw)
            if self._is_missing(v, plasma=False):
                clinical_vals.append(0.0)
                clinical_mask.append(1.0)
            else:
                stats = self.tabular_stats.clinical_stats.get(field, (0.0, 1.0))
                normed, mask = self._normalize_field(v, stats, log1p=False)
                clinical_vals.append(normed)
                clinical_mask.append(mask)

        for field in self.plasma_fields:
            raw = row.get(field)
            v = self._safe_float(raw)
            if self._is_missing(v, plasma=True):
                plasma_vals.append(0.0)
                plasma_mask.append(1.0)
            else:
                stats = plasma_stats.get(field, self.tabular_stats.plasma_stats.get("__global__", {}).get(field, (0.0, 1.0)))
                normed, mask = self._normalize_field(v, stats, log1p=field in LOG1P_FIELDS)
                plasma_vals.append(normed)
                plasma_mask.append(mask)

        sex_raw = row.get("sex")
        sex_id = 0
        if isinstance(sex_raw, str):
            sx = sex_raw.strip().lower()
            if sx in {"m", "male", "man"}:
                sex_id = 1
            elif sx in {"f", "female", "woman"}:
                sex_id = 2

        source_id = source_to_id(plasma_source)

        return {
            "clinical": torch.tensor(clinical_vals, dtype=torch.float32),
            "plasma": torch.tensor(plasma_vals, dtype=torch.float32),
            "clinical_mask": torch.tensor(clinical_mask, dtype=torch.float32),
            "plasma_mask": torch.tensor(plasma_mask, dtype=torch.float32),
            "sex": torch.tensor(sex_id, dtype=torch.long),
            "source": torch.tensor(source_id, dtype=torch.long),
        }
    
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
    print(f"[create_dataloaders] target_pet={config.data.target_pet}")
    # 构建样本列表
    sample_df = build_sample_list(config)
    
    # 优先使用外部固定划分（核心方法现有 split），否则走 baseline 内部分层切分
    external_split_subjects_cfg = resolve_external_split_subjects(config)
    if external_split_subjects_cfg is not None:
        print("[create_dataloaders] 使用外部固定划分（来自核心方法 split）")
        allow_val_test_overlap = (
            config.data.split_fallback_test_from_val and not bool(config.data.split_test_json)
        )
        if allow_val_test_overlap:
            print("[create_dataloaders] 启用 val->test 回退，允许 val/test subjects 重合")
        train_df, val_df, test_df = external_subject_split(
            sample_df,
            external_split_subjects_cfg,
            strict=config.data.split_strict,
            allow_val_test_overlap=allow_val_test_overlap,
        )
    else:
        train_df, val_df, test_df = stratified_split(sample_df, config)
    
    # 打印统计
    stats_path = os.path.join(config.output_dir, "split_statistics.txt")
    print_split_statistics(train_df, val_df, test_df, stats_path)
    
    # 条件特征统计（仅使用训练集）
    tabular_stats = None
    if config.condition.mode != "none":
        tabular_stats = TabularStats.from_dataframe(
            train_df,
            clinical_fields=config.condition.clinical_fields,
            plasma_fields=config.condition.plasma_fields,
            source_col=config.condition.source_col,
        )

    # 创建 Dataset
    train_dataset = MRITauDataset(
        train_df,
        target_shape=config.data.target_shape,
        return_weight=True,
        augment=True,
        condition_mode=config.condition.mode,
        tabular_stats=tabular_stats,
        source_col=config.condition.source_col,
        clinical_fields=config.condition.clinical_fields,
        plasma_fields=config.condition.plasma_fields,
    )
    val_dataset = MRITauDataset(
        val_df,
        target_shape=config.data.target_shape,
        return_weight=False,
        augment=False,
        condition_mode=config.condition.mode,
        tabular_stats=tabular_stats,
        source_col=config.condition.source_col,
        clinical_fields=config.condition.clinical_fields,
        plasma_fields=config.condition.plasma_fields,
    )
    test_dataset = MRITauDataset(
        test_df,
        target_shape=config.data.target_shape,
        return_weight=False,
        augment=False,
        condition_mode=config.condition.mode,
        tabular_stats=tabular_stats,
        source_col=config.condition.source_col,
        clinical_fields=config.condition.clinical_fields,
        plasma_fields=config.condition.plasma_fields,
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
