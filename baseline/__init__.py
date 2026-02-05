"""
baseline - MRI → TAU-PET 生成 Baseline
=======================================

模块:
- config: 配置管理
- dataset: 数据加载
- model: 3D Residual U-Net 模型
- losses: L1 + 3D SSIM 损失函数
- train: 训练脚本
- evaluate: 评估脚本
- visualize: 可视化模块
"""

from .config import Config, DataConfig, ModelConfig, LossConfig, TrainConfig, get_default_config
from .dataset import MRITauDataset, create_dataloaders
from .model import ResidualUNet3D, create_model
from .losses import CombinedLoss, SSIM3DLoss, MetricsCalculator, create_loss

__version__ = "1.0.0"
__all__ = [
    # Config
    "Config",
    "DataConfig", 
    "ModelConfig",
    "LossConfig",
    "TrainConfig",
    "get_default_config",
    # Dataset
    "MRITauDataset",
    "create_dataloaders",
    # Model
    "ResidualUNet3D",
    "create_model",
    # Losses
    "CombinedLoss",
    "SSIM3DLoss",
    "MetricsCalculator",
    "create_loss",
]
