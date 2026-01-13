import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

# 仅使用三种 PET 模态进行训练
MODALITIES: Tuple[str, ...] = ("FDG", "AV45", "TAU")

'''
    FDG:   This is an image of FDG PET, a functional brain imaging technique that visualizes the dynamic changes in glucose metabolism, directly linked to neuronal energy demands and synaptic activity. Areas with decreased metabolic activity exhibit reduced signal intensity. High-intensity metabolic hotspots in gray matter are key markers of neuronal activity."
    AV45: "This is an image of AV45 PET, a molecular imaging technique that highlights the static distribution of amyloid-beta plaques, a critical pathological marker of Alzheimer's disease. This imaging modality provides a spatial map of amyloid deposition in cortical regions and can distinguish amyloid-positive areas from amyloid-negative white matter regions. The primary focus is on identifying amyloid deposition patterns to assess disease progression and pathological burden."
'''

# 映射 CSV 列到模态（MRI 仅用于推导 examdate，不参与训练）
MODALITY_ID_COL = {
    "MRI": "id_mri",
    "FDG": "id_fdg",
    "AV45": "id_av45",
    "TAU": "id_av1451",
}


def _default_config() -> Dict:
    """Fallback config used when no yaml is provided."""
    return {
        "modalities": {
            "FDG": {
                "common_text": "a functional brain imaging technique that visualizes the dynamic changes in glucose metabolism, "
                "directly linked to neuronal energy demands and synaptic activity. Areas with decreased metabolic activity exhibit reduced signal intensity. "
                "High-intensity metabolic hotspots in gray matter are key markers of neuronal activity."
            },
            "AV45": {
                "common_text": " a molecular imaging technique that highlights the static distribution of amyloid-beta plaques, "
                "a critical pathological marker of Alzheimer's disease. This imaging modality provides a spatial map of amyloid deposition in cortical regions and "
                "can distinguish amyloid-positive areas from amyloid-negative white matter regions. The primary focus is on identifying amyloid deposition patterns to assess disease progression and pathological burden."
            },
            "TAU": {
                "common_text": "a molecular neuroimaging technique that visualizes the spatial distribution of aggregated tau protein, "
                "which reflects the presence of neurofibrillary tangles associated with neurodegeneration. "
                "Tau PET highlights region-specific tau accumulation, particularly in medial temporal, parietal, and association cortices, "
                "providing a topographical map of tau pathology that correlates with disease stage, cognitive decline, and neuronal dysfunction. "
                "This modality emphasizes the progression and regional spread of tau pathology rather than metabolic activity or amyloid burden."
            }

        },
        # Plasma规则：仅考虑 p-tau217、p-tau217/Aβ42、Aβ42/Aβ40 三项
        "plasma_thresholds": {
            # p-tau217：Negative <= 0.128; Positive >= 0.300; Intermediate 0.129-0.299
            "PT217_F": {
                "negative_below": 0.128,
                "positive_above": 0.300,
                "intermediate_range": [0.129, 0.299],
            },
            # p-tau217/Aβ42：Negative <= 0.0055; Positive >= 0.0086; Intermediate 0.0056-0.0085
            "PT217_AB42_F": {
                "negative_below": 0.0055,
                "positive_above": 0.0086,
                "intermediate_range": [0.0056, 0.0085],
            },
            # Aβ42/Aβ40：Negative >= 0.1053; Positive <= 0.0820; Intermediate 0.0821-0.1052
            "AB42_AB40_F": {
                "negative_above": 0.1053,
                "positive_below": 0.0820,
                "intermediate_range": [0.0821, 0.1052],
            },
        },
    }


def load_yaml_config(path: Optional[str]) -> Dict:
    if path is None:
        return _default_config()
    yaml_path = Path(path)
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return _default_config()


def map_value_to_state(value: float, rule: Dict) -> str:
    """Map numeric biomarker value to discrete state using rule dictionary."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "UNKNOWN"

    # 优先判断极值正负，再判断中间区间
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
    if "normal_range" in rule:
        lo, hi = rule["normal_range"]
        if lo <= value <= hi:
            return "NORMAL"
    return "UNKNOWN"


def format_biomarker_value(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    if isinstance(value, (int, float)):
        formatted = f"{value:.4f}"
        if "." in formatted:
            formatted = formatted.rstrip("0").rstrip(".")
        return formatted
    return str(value)


def build_plasma_text(row: Dict[str, float], thresholds: Dict[str, Dict]) -> str:
    """Render plasma biomarker states in固定顺序并使用指定关键词，并携带原始数值。"""
    order = [
        ("AB42_AB40_F", "Aβ42/Aβ40"),
        ("PT217_F", "p-tau217"),
        ("PT217_AB42_F", "p-tau217/Aβ42"),
        ("NFL_Q", "NfL"),
        ("GFAP_Q", "GFAP"),
    ]
    segments: List[str] = []
    for key, label in order:
        value = row.get(key)
        rule = thresholds.get(key, {})
        state = map_value_to_state(value, rule)
        val_text = format_biomarker_value(value)
        segments.append(f"{label} = {val_text} ({state});")
    return "\n".join(segments)


def load_cached_feature(path: Path) -> torch.Tensor:
    """加载预缓存的 BiomedCLIP ImageEncoder 输出，并返回 cls_token (dim=512)。"""
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Cached feature格式错误: {path}")
    cls = payload.get("cls_token")
    if cls is None:
        raise ValueError(f"Cached feature缺少 cls_token: {path}")
    cls = torch.as_tensor(cls, dtype=torch.float32)
    if cls.ndim == 1:
        cls = cls.unsqueeze(0)
    return cls  # shape: (1, 512) or (B, 512)


def synthetic_plasma_row(rng: random.Random) -> Dict[str, float]:
    # 随机值覆盖正常与异常范围，方便观察文本输出
    return {
        "AB42_AB40_F": rng.uniform(0.05, 0.12),
        "pT217_AB42_F": rng.uniform(0.004, 0.009),
        "NfL_Q": rng.uniform(10.0, 30.0),
        "GFAP_Q": rng.uniform(80.0, 170.0),
    }


def create_random_feature(dim: int = 512) -> torch.Tensor:
    return torch.randn(1, dim, dtype=torch.float32)


def find_matching_file(modality_dir: Path, subject_token: str) -> Path:
    # 优先匹配缓存特征文件（可能是 .pt/.pth/.npy）
    for pattern in ["*.pt", "*.pth", "*.npy", "*"]:
        candidates = sorted(modality_dir.glob(f"{subject_token}{pattern[1:]}"))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"No cached feature found for {subject_token} in {modality_dir}")


class PETAdapterDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: Path,
        link_csv: Optional[Path],
        plasma_csv: Optional[Path],
        mri_csv: Optional[Path],
        yaml_config: Optional[Path],
        processor,
        synthetic: bool = False,
        synthetic_count: int = 8,
        image_size: int = 224,
        cache_root: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.link_csv = Path(link_csv) if link_csv else None
        self.plasma_csv = Path(plasma_csv) if plasma_csv else None
        self.mri_csv = Path(mri_csv) if mri_csv else None
        self.config = load_yaml_config(str(yaml_config) if yaml_config else None)
        self.processor = processor  # 保留接口但不再使用图像预处理
        self.synthetic = synthetic
        self.image_size = image_size
        self.rng = random.Random(42)
        self.cache_root = Path(cache_root) if cache_root else Path("/mnt/nfsdata/nfsdata/ADNI/cached_npy")
        self.mri_table = self._load_mri_table(self.mri_csv) if self.mri_csv else None

        if synthetic:
            self.samples = self._build_synthetic_samples(synthetic_count)
        else:
            if self.link_csv is None:
                raise ValueError("link_csv is required when synthetic=False")
            # plasma_csv 可为空：若 link_csv 已整合血浆列，则直接使用 link_csv
            self.plasma_table = self._load_plasma_table(self.plasma_csv) if self.plasma_csv else None
            self.samples = self._build_real_samples()

    def _build_synthetic_samples(self, count: int) -> List[Dict]:
        samples = []
        for idx in range(count):
            ptid = f"SYN_{idx:03d}"
            plasma = synthetic_plasma_row(self.rng)
            samples.append({"ptid": ptid, "plasma": plasma, "examdate": None, "ids": {m: f"SYN_{idx:03d}_{m}" for m in MODALITIES}})
        return samples

    def _load_plasma_table(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, low_memory=False)
        # 统一列名小写
        df.columns = [c.lower() for c in df.columns]
        # 统一日期
        if "examdate" in df.columns:
            df["examdate"] = pd.to_datetime(df["examdate"], errors="coerce")

        # 兼容两类来源：
        # 1) 原始 UPENN（列名：ab42_ab40_f, pt217_f, pt217_ab42_f, nfl_q, gfap_q）
        # 2) 已整合的 pairs_withPlasma（列名：ab42_ab40, pt217, pt217_ab42, nfl, gfap）
        has_integrated = all(col in df.columns for col in ["ab42_ab40", "pt217", "pt217_ab42"]) and "ptid" in df.columns
        if has_integrated:
            # 复制一份到标准键，后续统一从标准键读取
            rename_map = {
                "ab42_ab40": "ab42_ab40_f",
                "pt217": "pt217_f",
                "pt217_ab42": "pt217_ab42_f",
                "nfl": "nfl_q",
                "gfap": "gfap_q",
            }
            for src, dst in rename_map.items():
                if src in df.columns and dst not in df.columns:
                    df[dst] = df[src]
        return df

    def _extract_plasma_from_row(self, row: Dict) -> Dict[str, float]:
        """优先从 link_csv 的行内直接读取已整合的血浆列。缺失时返回 {}。"""
        needed = {
            "ab42_ab40_f": "ab42_ab40",
            "pt217_f": "pt217",
            "pt217_ab42_f": "pt217_ab42",
            "nfl_q": "nfl",
            "gfap_q": "gfap",
        }
        have_any = False
        data = {}
        for dst, src in needed.items():
            val = row.get(src)
            if isinstance(val, str) and val.strip() == "":
                val = None
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                try:
                    data[dst] = float(val)
                except Exception:
                    data[dst] = val
                have_any = True
        return {k.upper(): v for k, v in data.items()} if have_any else {}

    def _load_mri_table(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        if "modality" in df.columns:
            df = df[df["modality"].str.lower() == "mri"]
        if "study date" in df.columns:
            df["study date"] = pd.to_datetime(df["study date"], errors="coerce")
        if "image id" in df.columns:
            df["image id"] = df["image id"].astype(str)
        return df

    def _pick_plasma(self, ptid: str, examdate: Optional[str]) -> Dict[str, float]:
        if self.plasma_table is None:
            return {}
        df = self.plasma_table
        ptid_col = "ptid"
        exam_col = "examdate"
        hit = df[df[ptid_col] == ptid].copy()
        if hit.empty:
            return {}

        # 优先精确匹配 examdate（已在 _load_plasma_table 统一为 datetime）
        if exam_col in hit.columns:
            exam_ts = pd.to_datetime(examdate) if examdate else None
            if exam_ts is not None:
                exact = hit[hit[exam_col] == exam_ts]
                if not exact.empty:
                    hit = exact
                else:
                    # 无精确匹配则退化为“最近一条”（不再强制 90 天窗口）
                    hit["_diff_days"] = (hit[exam_col] - exam_ts).abs().dt.days
                    hit = hit.sort_values("_diff_days")

        row = hit.iloc[0]
        data = row.to_dict()
        # 将整合或原始列统一映射为标准键
        numeric_cols = [
            "ab42_ab40_f",
            "pt217_f",
            "pt217_ab42_f",
            "nfl_q",
            "gfap_q",
        ]
        for k in numeric_cols:
            if k in data:
                try:
                    # 将空字符串等转换为缺失
                    val = data[k]
                    if isinstance(val, str) and val.strip() == "":
                        data[k] = float("nan")
                    else:
                        data[k] = float(val)
                except Exception:
                    # 保持原样（可能为 NaN 或无法解析的字符串）
                    pass
        # 统一键为大写，方便后续取值
        data = {k.upper(): v for k, v in data.items()}
        return data

    def _infer_examdate_from_mri(self, id_mri: Optional[str]) -> Optional[str]:
        if not id_mri or self.mri_table is None:
            return None
        id_clean = str(id_mri).lstrip("I")
        df = self.mri_table
        if "image id" not in df.columns or "study date" not in df.columns:
            return None
        match = df[df["image id"] == id_clean]
        if match.empty:
            return None
        date = match.iloc[0]["study date"]
        if pd.isna(date):
            return None
        return pd.to_datetime(date).strftime("%Y-%m-%d")

    def _build_real_samples(self) -> List[Dict]:
        link_df = pd.read_csv(self.link_csv)
        link_df.columns = [c.lower() for c in link_df.columns]

        if "examdate" not in link_df.columns:
            link_df["examdate"] = None
        if "id_mri" in link_df.columns:
            link_df["examdate"] = link_df["examdate"].fillna(
                link_df["id_mri"].apply(self._infer_examdate_from_mri)
            )
        link_df["examdate"] = pd.to_datetime(link_df["examdate"], errors="coerce").dt.strftime("%Y-%m-%d")
        link_df.loc[link_df["examdate"].isin(["NaT", None]), "examdate"] = None

        samples: List[Dict] = []
        for _, row in link_df.iterrows():
            ptid = str(row.get("subject_id", row.iloc[0]))
            examdate = row.get("examdate") if "examdate" in row else None
            # 先尝试直接从 link_csv 行读取血浆值（针对已整合的 pairs_withPlasma）
            plasma_row = self._extract_plasma_from_row(row)
            if not plasma_row:
                plasma_row = self._pick_plasma(ptid, examdate)
            ids: Dict[str, Optional[str]] = {}
            for modality, col in MODALITY_ID_COL.items():
                ids[modality] = None
                if col in row and isinstance(row[col], str) and row[col].strip() != "":
                    ids[modality] = row[col].strip()
            has_pet = any(ids[m] for m in MODALITIES)
            if not has_pet:
                continue
            samples.append({"ptid": ptid, "examdate": examdate, "plasma": plasma_row, "ids": ids})
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _build_text(self, modality: str, plasma: Dict[str, float]) -> str:
        modality_cfg = self.config.get("modalities", {}).get(modality, {})
        common_text = modality_cfg.get("common_text", "")
        plasma_text = build_plasma_text(plasma, self.config.get("plasma_thresholds", {}))
        return f"[PLASMA]\n{plasma_text}\n[/PLASMA]\n[SEP]\n{common_text}"

    def _load_feature(self, modality: str, sample: Dict) -> torch.Tensor:
        if self.synthetic:
            return create_random_feature()

        modality_id = sample["ids"].get(modality)
        if modality_id is None:
            raise FileNotFoundError(f"{modality} id missing for sample {sample['ptid']}")

        # 缓存命名示例：168_S_6591_av1451_I1167581.vision.pt
        suffix_map = {"FDG": "fdg", "AV45": "av45", "TAU": "av1451"}
        suffix = suffix_map.get(modality, modality.lower())
        filename = f"{sample['ptid']}_{suffix}_{modality_id}.vision.pt"
        feat_path = self.cache_root / filename

        if not feat_path.exists():
            raise FileNotFoundError(f"Cached feature not found: {feat_path}")

        return load_cached_feature(feat_path)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        plasma = sample.get("plasma", {})

        texts: Dict[str, str] = {}
        pixel_values: Dict[str, torch.Tensor] = {}
        available: List[str] = []
        for modality in MODALITIES:
            if sample["ids"].get(modality):
                texts[modality] = self._build_text(modality, plasma)  # "{'TAU': 'TAU PET highlights tau aggregation in temporal and parietal cortices.\n[SEP]\n[PLASMA]\np-tau217 = NEGATIVE;\np-tau217/Aβ42 = NEGATIVE;\nAβ42/Aβ40 = POSITIVE;\n[/PLASMA]'}"
                feature = self._load_feature(modality, sample)
                pixel_values[modality] = feature.squeeze(0)
                available.append(modality)

        return {
            "ptid": sample.get("ptid", f"IDX_{idx}"),
            "texts": texts,
            "pixel_values": pixel_values,
            "available": available,
        }


def collate_batch(batch: List[Dict]) -> Dict:
    ptids = [item["ptid"] for item in batch]
    texts: Dict[str, List[str]] = {m: [] for m in MODALITIES}
    pixel_values: Dict[str, List[torch.Tensor]] = {m: [] for m in MODALITIES}
    avail_list: List[List[str]] = []

    for item in batch:
        avail_list.append(item["available"])
        for modality in MODALITIES:
            if modality in item["texts"]:
                texts[modality].append(item["texts"][modality])
                pixel_values[modality].append(item["pixel_values"][modality])

    # 对缺失模态，不堆叠空列表，保持只含有样本的模态
    stacked_pixels: Dict[str, torch.Tensor] = {}
    for m, tensors in pixel_values.items():
        if tensors:
            stacked_pixels[m] = torch.stack(tensors, dim=0)

    return {"ptid": ptids, "texts": texts, "pixel_values": stacked_pixels, "available": avail_list}
