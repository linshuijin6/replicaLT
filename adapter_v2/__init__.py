"""
adapter_v2
==========
TAU 单示踪剂 CoCoOp 对比学习模块

模块组成：
    - dataset.py: TAUPlasmaDataset, SubjectBatchSampler
    - models.py: CoCoOpTAUModel
    - losses.py: 对比损失函数
    - train.py: 训练脚本
    - demo.py: 调试脚本
"""

from .dataset import TAUPlasmaDataset, SubjectBatchSampler, collate_fn, build_dataloaders
from .models import CoCoOpTAUModel, TokenAttentionPool, ProjectionHead
from .losses import compute_total_loss, compute_recall_at_k

__all__ = [
    "TAUPlasmaDataset",
    "SubjectBatchSampler",
    "collate_fn",
    "build_dataloaders",
    "CoCoOpTAUModel",
    "TokenAttentionPool",
    "ProjectionHead",
    "compute_total_loss",
    "compute_recall_at_k",
]
