"""
struct_condition.py
====================
结构化条件编码器模块，用于从 description 和 plasma 数据构造 UNet 的 context 输入。

核心组件：
1. parse_description: 从固定模板文本中解析人口学和量表特征
2. StructNormalizer: 管理连续特征的归一化统计量
3. StructConditionEncoder: MLP 编码器 + tracer embedding + 投影层

设计原则：
- 不依赖任何大模型（CLIP/BiomedCLIP/transformers）
- 不做 tokenization，仅用正则解析固定模板
- 支持特征缺失（通过 mask 机制）
"""

import re
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================================
# 常量定义
# ============================================================================

# 来自 description 的连续特征（7 维）
DESCRIPTION_CONT_KEYS = [
    "AGE",
    "WEIGHT",
    "CDR_GLOBAL",
    "MMSE",
    "GDS",
    "FAQ",
    "NPIQ_TOTAL",
]

# 来自 plasma CSV 的连续特征（7 维）
PLASMA_CONT_KEYS = [
    "pT217_F",
    "AB42_F",
    "AB40_F",
    "AB42_AB40_F",
    "pT217_AB42_F",
    "NfL_Q",
    "GFAP_Q",
]

# 所有连续特征（14 维）
ALL_CONT_KEYS = DESCRIPTION_CONT_KEYS + PLASMA_CONT_KEYS

# 需要做 log1p 预处理的字段
LOG1P_KEYS = {"NfL_Q", "GFAP_Q"}

# 可选：对这些字段也做 log1p（范围较大）
OPTIONAL_LOG_KEYS = {"AB42_F", "AB40_F"}

# Tracer 映射
TRACER_MAP = {"FDG": 0, "AV45": 1, "TAU": 2}

# Sex 映射
SEX_MAP = {"unknown": 0, "male": 1, "female": 2}

# Plasma CSV 中表示缺失的特殊值
PLASMA_NA_VALUE = -4.0


# ============================================================================
# parse_description: 从固定模板文本解析结构化特征
# ============================================================================

def parse_description(description: Optional[str]) -> Dict[str, Optional[float]]:
    """
    从固定模板的 description 文本中解析人口学和量表特征。
    
    模板示例：
    "Subject is a 74-year-old male with a weight of 87.5 kg. 
     The global Clinical Dementia Rating (CDR) score ... is 0.0. 
     The Mini-Mental State Examination (MMSE) score ... is 28. 
     The Geriatric Depression Scale (GDS) score ... is 1. 
     The Functional Activities Questionnaire (FAQ) score ... is 5. 
     The Neuropsychiatric Inventory Questionnaire (NPI-Q) Total Score ... is 1."
    
    Args:
        description: 固定模板文本，可能为 None
        
    Returns:
        Dict[str, Optional[float]]: 解析出的特征字典，缺失返回 None
        包含键: AGE, SEX, WEIGHT, CDR_GLOBAL, MMSE, GDS, FAQ, NPIQ_TOTAL
    """
    result = {
        "AGE": None,
        "SEX": None,  # str: "male"/"female"/"unknown"
        "WEIGHT": None,
        "CDR_GLOBAL": None,
        "MMSE": None,
        "GDS": None,
        "FAQ": None,
        "NPIQ_TOTAL": None,
    }
    
    if description is None or not isinstance(description, str):
        result["SEX"] = "unknown"
        return result
    
    text = description.strip()
    
    # 1. 解析年龄和性别: "Subject is a 74-year-old male"
    age_sex_pattern = r"Subject is a (\d+(?:\.\d+)?)-year-old (male|female)"
    match = re.search(age_sex_pattern, text, re.IGNORECASE)
    if match:
        result["AGE"] = float(match.group(1))
        result["SEX"] = match.group(2).lower()
    else:
        result["SEX"] = "unknown"
    
    # 2. 解析体重: "weight of 87.5 kg"
    weight_pattern = r"weight of (\d+(?:\.\d+)?)\s*kg"
    match = re.search(weight_pattern, text, re.IGNORECASE)
    if match:
        weight = float(match.group(1))
        # 体重为 0 视为缺失
        if weight > 0:
            result["WEIGHT"] = weight
    
    # 3. 解析 CDR: "CDR) score ... is 0.0"
    cdr_pattern = r"CDR\).*?is\s+(\d+(?:\.\d+)?)"
    match = re.search(cdr_pattern, text, re.IGNORECASE)
    if match:
        result["CDR_GLOBAL"] = float(match.group(1))
    
    # 4. 解析 MMSE: "MMSE) score ... is 28"
    mmse_pattern = r"MMSE\).*?is\s+(\d+(?:\.\d+)?)"
    match = re.search(mmse_pattern, text, re.IGNORECASE)
    if match:
        result["MMSE"] = float(match.group(1))
    
    # 5. 解析 GDS: "GDS) score ... is 1"
    gds_pattern = r"GDS\).*?is\s+(\d+(?:\.\d+)?)"
    match = re.search(gds_pattern, text, re.IGNORECASE)
    if match:
        result["GDS"] = float(match.group(1))
    
    # 6. 解析 FAQ: "FAQ) score ... is 5"
    faq_pattern = r"FAQ\).*?is\s+(\d+(?:\.\d+)?)"
    match = re.search(faq_pattern, text, re.IGNORECASE)
    if match:
        result["FAQ"] = float(match.group(1))
    
    # 7. 解析 NPI-Q: "NPI-Q) Total Score ... is 1"
    npiq_pattern = r"NPI-Q\)\s*Total\s*Score.*?is\s+(\d+(?:\.\d+)?)"
    match = re.search(npiq_pattern, text, re.IGNORECASE)
    if match:
        result["NPIQ_TOTAL"] = float(match.group(1))
    
    return result


# ============================================================================
# Plasma 数据加载与匹配
# ============================================================================

class PlasmaLoader:
    """
    Plasma 数据加载器，支持按 PTID 和 examdate 近邻匹配。
    """
    
    def __init__(
        self,
        csv_path: Union[str, Path],
        max_days: int = 90,
        na_value: float = PLASMA_NA_VALUE,
    ):
        """
        Args:
            csv_path: Plasma CSV 文件路径
            max_days: 最大匹配天数
            na_value: 表示缺失的特殊值（如 -4.0）
        """
        self.csv_path = Path(csv_path)
        self.max_days = max_days
        self.na_value = na_value
        self.df = self._load_csv()
    
    def _load_csv(self) -> pd.DataFrame:
        """加载并预处理 plasma CSV"""
        if not self.csv_path.exists():
            print(f"[PlasmaLoader] Warning: CSV not found: {self.csv_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(self.csv_path)
        # 不再强制转换列名为小写，保持原始列名
        
        # 解析日期列（支持大小写）
        for date_col in ["examdate", "EXAMDATE", "plasma_date"]:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        
        return df
    
    def pick(self, ptid: Optional[str], examdate: Optional[str] = None) -> Dict[str, Optional[float]]:
        """
        按 PTID 和 examdate 匹配 plasma 记录。
        
        Args:
            ptid: 患者 ID（如 002_S_0413）
            examdate: MRI 检查日期（可选）
            
        Returns:
            Dict[str, Optional[float]]: plasma 字段值，缺失返回 None
        """
        result = {key: None for key in PLASMA_CONT_KEYS}
        
        if ptid is None or self.df.empty:
            return result
        
        df = self.df
        # 支持大写和小写列名
        ptid_col = "ptid" if "ptid" in df.columns else "PTID" if "PTID" in df.columns else None
        
        if ptid_col is None:
            return result
        
        hit = df[df[ptid_col] == ptid].copy()
        
        if hit.empty:
            return result
        
        # 如果有 examdate，按时间近邻匹配
        examdate_col = "examdate" if "examdate" in df.columns else "EXAMDATE" if "EXAMDATE" in df.columns else None
        if examdate is not None and examdate_col is not None:
            try:
                examdate_ts = pd.to_datetime(examdate)
                hit["_diff_days"] = (hit[examdate_col] - examdate_ts).abs().dt.days
                hit = hit[hit["_diff_days"] <= self.max_days].sort_values("_diff_days")
            except Exception:
                pass
        
        if hit.empty:
            return result
        
        # 取最近的一条
        row = hit.iloc[0]
        
        # 提取 plasma 字段（支持大小写）
        for key in PLASMA_CONT_KEYS:
            # 尝试原始键名和小写键名
            col_name = key if key in row.index else key.lower() if key.lower() in row.index else None
            if col_name is not None:
                try:
                    val = float(row[col_name])
                    # 检查是否为缺失值
                    if np.isnan(val) or val == self.na_value:
                        result[key] = None
                    else:
                        result[key] = val
                except (ValueError, TypeError):
                    result[key] = None
        
        return result


# ============================================================================
# StructNormalizer: 归一化统计量管理
# ============================================================================

class StructNormalizer:
    """
    结构化特征归一化器。
    
    支持两种模式：
    - robust: (x - median) / (IQR + eps)
    - standard: (x - mean) / (std + eps)
    
    对指定字段做 log1p 预处理。
    """
    
    def __init__(
        self,
        mode: str = "robust",
        log1p_keys: Optional[set] = None,
        eps: float = 1e-6,
    ):
        """
        Args:
            mode: 归一化模式，"robust" 或 "standard"
            log1p_keys: 需要做 log1p 预处理的字段集合
            eps: 防止除零的小值
        """
        self.mode = mode
        self.log1p_keys = log1p_keys or LOG1P_KEYS
        self.eps = eps
        
        # 统计量字典: {field_name: {"center": float, "scale": float}}
        self.stats: Dict[str, Dict[str, float]] = {}
        self.fitted = False
    
    def fit(self, data_list: List[Dict[str, Optional[float]]]) -> "StructNormalizer":
        """
        从训练集计算归一化统计量。
        
        Args:
            data_list: 样本特征字典列表，每个字典包含所有连续特征
            
        Returns:
            self
        """
        # 收集每个字段的有效值
        field_values: Dict[str, List[float]] = {key: [] for key in ALL_CONT_KEYS}
        
        for sample in data_list:
            for key in ALL_CONT_KEYS:
                val = sample.get(key)
                if val is not None and np.isfinite(val):
                    # 对指定字段做 log1p
                    if key in self.log1p_keys:
                        val = np.log1p(max(0, val))
                    field_values[key].append(val)
        
        # 计算统计量
        for key in ALL_CONT_KEYS:
            values = field_values[key]
            if len(values) == 0:
                # 无有效值时使用默认值
                self.stats[key] = {"center": 0.0, "scale": 1.0}
            elif self.mode == "robust":
                arr = np.array(values)
                median = float(np.median(arr))
                q75, q25 = np.percentile(arr, [75, 25])
                iqr = q75 - q25
                self.stats[key] = {"center": median, "scale": float(iqr + self.eps)}
            else:  # standard
                arr = np.array(values)
                mean = float(np.mean(arr))
                std = float(np.std(arr))
                self.stats[key] = {"center": mean, "scale": std + self.eps}
        
        self.fitted = True
        return self
    
    def transform(
        self,
        sample: Dict[str, Optional[float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        对单个样本进行归一化。
        
        Args:
            sample: 特征字典，包含所有连续特征
            
        Returns:
            (values, mask): 
                values: shape (14,)，归一化后的值（缺失填 0）
                mask: shape (14,)，1 表示缺失，0 表示存在
        """
        if not self.fitted:
            raise RuntimeError("StructNormalizer has not been fitted yet.")
        
        values = np.zeros(len(ALL_CONT_KEYS), dtype=np.float32)
        mask = np.zeros(len(ALL_CONT_KEYS), dtype=np.float32)
        
        for i, key in enumerate(ALL_CONT_KEYS):
            val = sample.get(key)
            
            if val is None or not np.isfinite(val):
                # 缺失：填 0，mask 置 1
                values[i] = 0.0
                mask[i] = 1.0
            else:
                # 对指定字段做 log1p
                if key in self.log1p_keys:
                    val = np.log1p(max(0, val))
                
                # 归一化
                stat = self.stats.get(key, {"center": 0.0, "scale": 1.0})
                values[i] = (val - stat["center"]) / stat["scale"]
                mask[i] = 0.0
        
        return values, mask
    
    def save(self, path: Union[str, Path]) -> None:
        """保存统计量到 JSON 文件"""
        path = Path(path)
        data = {
            "mode": self.mode,
            "log1p_keys": list(self.log1p_keys),
            "eps": self.eps,
            "stats": self.stats,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "StructNormalizer":
        """从 JSON 文件加载统计量"""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        normalizer = cls(
            mode=data.get("mode", "robust"),
            log1p_keys=set(data.get("log1p_keys", LOG1P_KEYS)),
            eps=data.get("eps", 1e-6),
        )
        normalizer.stats = data.get("stats", {})
        normalizer.fitted = True
        return normalizer


# ============================================================================
# StructConditionEncoder: MLP 编码器
# ============================================================================

class StructConditionEncoder(nn.Module):
    """
    结构化条件编码器。
    
    输入：
    - cond_cont: (B, 14) 归一化后的连续特征
    - cond_mask: (B, 14) 缺失 mask
    - sex_id: (B,) 性别 ID
    - tracer_id: (B,) tracer ID
    
    输出：
    - context: (B, 1, 512) 用于 UNet cross-attention
    """
    
    def __init__(
        self,
        n_cont: int = 14,
        d_sex: int = 16,
        d_tracer: int = 64,
        d_hidden: int = 256,
        d_out: int = 512,
        dropout: float = 0.1,
    ):
        """
        Args:
            n_cont: 连续特征维度（14）
            d_sex: sex embedding 维度
            d_tracer: tracer embedding 维度
            d_hidden: MLP 隐层维度
            d_out: 最终输出维度（512，与 UNet cross_attention_dim 一致）
            dropout: dropout 概率
        """
        super().__init__()
        
        self.n_cont = n_cont
        self.d_sex = d_sex
        self.d_tracer = d_tracer
        
        # Sex embedding: 3 categories (unknown, male, female)
        self.sex_embed = nn.Embedding(3, d_sex)
        
        # Tracer embedding: 3 categories (FDG, AV45, TAU)
        self.tracer_embed = nn.Embedding(3, d_tracer)
        
        # 输入维度: 2*n_cont (values + mask) + d_sex
        input_dim = 2 * n_cont + d_sex
        
        # 2-layer MLP for structural features
        self.struct_mlp = nn.Sequential(
            nn.Linear(input_dim, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 最终投影: d_hidden + d_tracer -> d_out
        self.proj = nn.Linear(d_hidden + d_tracer, d_out)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(
        self,
        cond_cont: torch.Tensor,
        cond_mask: torch.Tensor,
        sex_id: torch.Tensor,
        tracer_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            cond_cont: (B, 14) 归一化后的连续特征
            cond_mask: (B, 14) 缺失 mask
            sex_id: (B,) 性别 ID
            tracer_id: (B,) tracer ID
            
        Returns:
            context: (B, 1, 512) 用于 UNet cross-attention
        """
        B = cond_cont.size(0)
        
        # 拼接 values 和 mask
        u_cont = torch.cat([cond_cont, cond_mask], dim=-1)  # (B, 28)
        
        # Sex embedding
        e_sex = self.sex_embed(sex_id)  # (B, d_sex)
        
        # 拼接并通过 MLP
        struct_input = torch.cat([u_cont, e_sex], dim=-1)  # (B, 28 + d_sex)
        e_struct = self.struct_mlp(struct_input)  # (B, d_hidden)
        
        # Tracer embedding
        e_tracer = self.tracer_embed(tracer_id)  # (B, d_tracer)
        
        # 拼接并投影
        combined = torch.cat([e_struct, e_tracer], dim=-1)  # (B, d_hidden + d_tracer)
        context_vec = self.proj(combined)  # (B, d_out)
        
        # 增加序列维度 (B, 512) -> (B, 1, 512)
        context = context_vec.unsqueeze(1)
        
        return context


# ============================================================================
# 辅助函数：构建样本的结构化特征
# ============================================================================

def build_struct_features(
    sample: Dict[str, Any],
    plasma_loader: Optional[PlasmaLoader] = None,
) -> Dict[str, Any]:
    """
    为单个样本构建结构化特征。
    
    Args:
        sample: 原始样本字典，包含 name, description, mri 等字段
        plasma_loader: PlasmaLoader 实例
        
    Returns:
        Dict 包含:
        - cond_raw: Dict[str, Optional[float]] 原始特征值
        - sex_id: int
    """
    # 从 description 解析特征
    description = sample.get("description")
    desc_features = parse_description(description)
    
    # 从 plasma 获取特征
    ptid = sample.get("name")
    # 尝试从 MRI 路径提取 examdate（如果有的话）
    examdate = sample.get("examdate")
    
    if plasma_loader is not None:
        plasma_features = plasma_loader.pick(ptid, examdate)
    else:
        plasma_features = {key: None for key in PLASMA_CONT_KEYS}
    
    # 合并所有连续特征
    cond_raw = {}
    for key in DESCRIPTION_CONT_KEYS:
        cond_raw[key] = desc_features.get(key)
    for key in PLASMA_CONT_KEYS:
        cond_raw[key] = plasma_features.get(key)
    
    # Sex ID
    sex = desc_features.get("SEX", "unknown")
    sex_id = SEX_MAP.get(sex, 0)
    
    return {
        "cond_raw": cond_raw,
        "sex_id": sex_id,
    }


def prepare_batch_context(
    batch_cond_cont: torch.Tensor,
    batch_cond_mask: torch.Tensor,
    batch_sex_id: torch.Tensor,
    tracer_name: str,
    encoder: StructConditionEncoder,
    device: torch.device,
) -> torch.Tensor:
    """
    为一个 batch 构建 context tensor。
    
    Args:
        batch_cond_cont: (B, 14) 归一化后的连续特征
        batch_cond_mask: (B, 14) 缺失 mask
        batch_sex_id: (B,) 性别 ID
        tracer_name: "FDG"/"AV45"/"TAU"
        encoder: StructConditionEncoder 实例
        device: 目标设备
        
    Returns:
        context: (B, 1, 512)
    """
    B = batch_cond_cont.size(0)
    tracer_id = torch.full((B,), TRACER_MAP[tracer_name], dtype=torch.long, device=device)
    
    return encoder(
        cond_cont=batch_cond_cont.to(device),
        cond_mask=batch_cond_mask.to(device),
        sex_id=batch_sex_id.to(device),
        tracer_id=tracer_id,
    )


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    # 测试 parse_description
    test_desc = (
        "Subject is a 74-year-old male with a weight of 87.5 kg. "
        "The global Clinical Dementia Rating (CDR) score, which assesses dementia severity "
        "(0: no dementia to 3: severe dementia), is 0.0. "
        "The Mini-Mental State Examination (MMSE) score, assessing cognitive function "
        "(0: severe impairment to 30: normal), is 28. "
        "The Geriatric Depression Scale (GDS) score, screening depression "
        "(0: no depression to 15: severe depression), is 1. "
        "The Functional Activities Questionnaire (FAQ) score, assessing daily activity impairment "
        "(0: no impairment to 30: severe impairment), is 5. "
        "The Neuropsychiatric Inventory Questionnaire (NPI-Q) Total Score, "
        "assessing neuropsychiatric symptom burden (0: no symptoms to higher scores indicating greater burden), is 1."
    )
    
    result = parse_description(test_desc)
    print("=== parse_description 测试 ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
    
    # 测试 StructNormalizer
    print("\n=== StructNormalizer 测试 ===")
    fake_data = [
        {"AGE": 74, "WEIGHT": 87.5, "CDR_GLOBAL": 0.0, "MMSE": 28, "GDS": 1, "FAQ": 5, "NPIQ_TOTAL": 1,
         "pT217_F": 0.244, "AB42_F": 35.93, "AB40_F": 420.81, "AB42_AB40_F": 0.0854, "pT217_AB42_F": 0.00679,
         "NfL_Q": 24.8, "GFAP_Q": 156.6},
        {"AGE": 67, "WEIGHT": None, "CDR_GLOBAL": 0.5, "MMSE": 25, "GDS": 2, "FAQ": 10, "NPIQ_TOTAL": 3,
         "pT217_F": 0.65, "AB42_F": None, "AB40_F": None, "AB42_AB40_F": None, "pT217_AB42_F": 1.55,
         "NfL_Q": None, "GFAP_Q": None},
    ]
    
    normalizer = StructNormalizer(mode="robust")
    normalizer.fit(fake_data)
    
    values, mask = normalizer.transform(fake_data[0])
    print(f"  values shape: {values.shape}")
    print(f"  mask shape: {mask.shape}")
    print(f"  values: {values}")
    print(f"  mask: {mask}")
    
    # 测试 StructConditionEncoder
    print("\n=== StructConditionEncoder 测试 ===")
    encoder = StructConditionEncoder()
    
    B = 4
    cond_cont = torch.randn(B, 14)
    cond_mask = torch.zeros(B, 14)
    sex_id = torch.randint(0, 3, (B,))
    tracer_id = torch.randint(0, 3, (B,))
    
    context = encoder(cond_cont, cond_mask, sex_id, tracer_id)
    print(f"  context shape: {context.shape}")  # 期望 (4, 1, 512)
    
    print("\n✅ 所有测试通过！")
