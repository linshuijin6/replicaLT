"""
baseline/condition.py
====================
Tabular condition encoding + FiLM layers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


PLASMA_FIELDS = [
    "pt217_f",
    "ab42_f",
    "ab40_f",
    "ab42_ab40_f",
    "pt217_ab42_f",
    "nfl_q",
    "gfap_q",
]

CLINICAL_FIELDS = [
    "age",
    "weight",
    "cdr",
    "mmse",
    "gds",
    "faq",
    "npi-q",
]

LOG1P_FIELDS = {"nfl_q", "gfap_q"}

SOURCE_MAP = {
    "UNKNOWN": 0,
    "UPENN": 1,
    "C2N": 2,
}


def normalize_source(value: Optional[str]) -> str:
    if value is None:
        return "UNKNOWN"
    if isinstance(value, str):
        key = value.strip().upper()
        if key in SOURCE_MAP:
            return key
    return "UNKNOWN"


def source_to_id(value: Optional[str]) -> int:
    return SOURCE_MAP.get(normalize_source(value), 0)


def _safe_float(x) -> Optional[float]:
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


def _is_missing(value: Optional[float], plasma: bool = False) -> bool:
    if value is None:
        return True
    if not np.isfinite(value):
        return True
    if plasma and value <= -3.9:
        return True
    return False


def compute_robust_stats(values: Iterable[float]) -> Tuple[float, float]:
    values = np.asarray(list(values), dtype=np.float32)
    if values.size == 0:
        return 0.0, 1.0
    median = float(np.median(values))
    q75 = float(np.percentile(values, 75))
    q25 = float(np.percentile(values, 25))
    iqr = q75 - q25
    if iqr <= 0:
        iqr = 1.0
    return median, iqr


def _collect_values(
    df: pd.DataFrame,
    field: str,
    plasma: bool,
) -> List[float]:
    vals = []
    if field not in df.columns:
        return vals
    for raw in df[field].tolist():
        v = _safe_float(raw)
        if _is_missing(v, plasma=plasma):
            continue
        if plasma and field in LOG1P_FIELDS:
            v = float(np.log1p(v))
        vals.append(v)
    return vals


@dataclass
class TabularStats:
    clinical_stats: Dict[str, Tuple[float, float]]
    plasma_stats: Dict[str, Dict[str, Tuple[float, float]]]

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        clinical_fields: List[str],
        plasma_fields: List[str],
        source_col: str = "plasma_source",
    ) -> "TabularStats":
        clinical_stats: Dict[str, Tuple[float, float]] = {}
        plasma_stats: Dict[str, Dict[str, Tuple[float, float]]] = {"__global__": {}}

        for field in clinical_fields:
            values = _collect_values(df, field, plasma=False)
            clinical_stats[field] = compute_robust_stats(values)

        for field in plasma_fields:
            values = _collect_values(df, field, plasma=True)
            plasma_stats["__global__"][field] = compute_robust_stats(values)

        if source_col in df.columns:
            for source in df[source_col].fillna("UNKNOWN").unique().tolist():
                norm_source = normalize_source(source)
                if norm_source not in plasma_stats:
                    plasma_stats[norm_source] = {}
                source_df = df[df[source_col].fillna("UNKNOWN").apply(normalize_source) == norm_source]
                for field in plasma_fields:
                    values = _collect_values(source_df, field, plasma=True)
                    plasma_stats[norm_source][field] = compute_robust_stats(values)

        return cls(clinical_stats=clinical_stats, plasma_stats=plasma_stats)

    def get_plasma_stats(self, source: Optional[str]) -> Dict[str, Tuple[float, float]]:
        norm_source = normalize_source(source)
        if norm_source in self.plasma_stats:
            return self.plasma_stats[norm_source]
        return self.plasma_stats.get("__global__", {})


def normalize_value(
    value: Optional[float],
    median: float,
    iqr: float,
    log1p: bool = False,
) -> Tuple[float, float]:
    if value is None or not np.isfinite(value):
        return 0.0, 1.0
    v = float(value)
    if log1p:
        v = float(np.log1p(v))
    return (v - median) / (iqr + 1e-8), 0.0


class TabularEncoder(nn.Module):
    def __init__(
        self,
        clinical_dim: int,
        plasma_dim: int,
        embed_dim: int = 128,
        sex_embed_dim: int = 16,
        source_embed_dim: int = 8,
        hidden_dim: int = 256,
        mode: str = "both",
    ):
        super().__init__()
        self.clinical_dim = clinical_dim
        self.plasma_dim = plasma_dim
        self.mode = mode

        self.sex_emb = nn.Embedding(3, sex_embed_dim)
        self.source_emb = nn.Embedding(len(SOURCE_MAP), source_embed_dim)

        input_dim = (clinical_dim + plasma_dim) * 2 + sex_embed_dim + source_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(
        self,
        clinical: torch.Tensor,
        plasma: torch.Tensor,
        clinical_mask: Optional[torch.Tensor],
        plasma_mask: Optional[torch.Tensor],
        sex: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "clinical":
            plasma = torch.zeros_like(plasma)
            plasma_mask = torch.zeros_like(plasma)
        elif self.mode == "plasma":
            clinical = torch.zeros_like(clinical)
            clinical_mask = torch.zeros_like(clinical)

        if clinical_mask is None:
            clinical_mask = torch.zeros_like(clinical)
        if plasma_mask is None:
            plasma_mask = torch.zeros_like(plasma)

        sex_emb = self.sex_emb(sex)
        source_emb = self.source_emb(source)

        features = torch.cat(
            [clinical, plasma, clinical_mask, plasma_mask, sex_emb, source_emb],
            dim=1,
        )
        return self.mlp(features)


class FiLMLayer(nn.Module):
    def __init__(self, cond_dim: int, num_channels: int, hidden_dim: int = 128):
        super().__init__()
        self.num_channels = num_channels
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_channels * 2),
        )
        self._init_identity()
        self._last_gamma: Optional[torch.Tensor] = None
        self._last_beta: Optional[torch.Tensor] = None

    def _init_identity(self):
        nn.init.zeros_(self.mlp[-1].weight)
        bias = torch.zeros(self.num_channels * 2)
        bias[: self.num_channels] = 1.0
        with torch.no_grad():
            self.mlp[-1].bias.copy_(bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        params = self.mlp(cond)
        gamma, beta = params[:, : self.num_channels], params[:, self.num_channels :]
        self._last_gamma = gamma
        self._last_beta = beta
        gamma = gamma.view(gamma.size(0), gamma.size(1), 1, 1, 1)
        beta = beta.view(beta.size(0), beta.size(1), 1, 1, 1)
        return gamma * x + beta

    def reg_loss(self) -> torch.Tensor:
        if self._last_gamma is None or self._last_beta is None:
            return torch.tensor(0.0, device=self.mlp[0].weight.device)
        gamma = self._last_gamma
        beta = self._last_beta
        return ((gamma - 1.0) ** 2).mean() + (beta ** 2).mean()
