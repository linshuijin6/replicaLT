"""
baseline/config.py
==================
MRI → TAU-PET 生成 baseline 配置模块
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime


@dataclass
class DataConfig:
    """数据相关配置"""
    # 数据路径
    pairs_csv: str = "/home/ssddata/linshuijin/replicaLT/adapter_finetune/gen_csv/pairs_180d_dx_plasma_90d_matched_with_demog.csv"
    qc_csv: str = "/home/ssddata/linshuijin/replicaLT/analysis/tau_qc_results.csv"
    pet_info_csv: str = "/home/ssddata/linshuijin/replicaLT/analysis/3_PET_ADNI3_4_with_Plasma_PET_Images_04Feb2026.csv"
    
    # ADNI 数据根目录
    adni_root: str = "/mnt/nfsdata/nfsdata/lsj.14/ADNI_CSF"
    mri_subdir: str = "MRI"
    tau_subdir: str = "PET_MNI/TAU"
    
    # 目标尺寸 (D, H, W)
    target_shape: tuple = (160, 192, 160)
    
    # 数据划分
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    
    # 分层变量
    stratify_by: List[str] = field(default_factory=lambda: ["pet_mfr", "diagnosis", "quality_class"])
    
    # 数据加载
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    """模型相关配置"""
    # U-Net 配置
    in_channels: int = 1
    out_channels: int = 1
    base_features: int = 32  # 初始通道数
    num_scales: int = 4  # 下采样次数 (4 表示原尺度 + 3 次下采样)
    use_residual: bool = True  # 残差输出模式
    use_residual_output: bool = False  # True: clamp(input + delta)，False: sigmoid(delta)
    norm_type: str = "instance"  # instance, group, batch
    activation: str = "leaky_relu"
    dropout_rate: float = 0.0


@dataclass
class LossConfig:
    """损失函数相关配置"""
    lambda_l1: float = 1.0
    lambda_ssim: float = 0.3
    ssim_window_size: int = 11
    ssim_sigma: float = 1.5
    eps: float = 1e-8


@dataclass
class TrainConfig:
    """训练相关配置"""
    # 基本设置
    epochs: int = 60
    batch_size: int = 4
    accumulation_steps: int = 4  # 梯度累积以模拟更大 batch
    
    # 优化器
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # 学习率调度
    scheduler: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # 混合精度
    use_amp: bool = True

    # 质量权重
    use_quality_weight: bool = False
    
    # 验证
    val_freq: int = 5
    
    # 保存
    save_best: bool = True
    save_last: bool = True
    save_freq: int = 10  # 每 N 个 epoch 保存一次
    
    # 早停
    early_stopping_patience: int = 30


@dataclass
class ConditionConfig:
    """条件注入配置"""
    mode: str = "none"  # none / clinical / plasma / both
    embed_dim: int = 128
    film_hidden_dim: int = 128
    film_reg_lambda: float = 0.01
    freeze_backbone_epochs: int = 5
    joint_lr_factor: float = 0.5
    source_col: str = "plasma_source"

    # 预训练 backbone 权重路径（用 baseline best_model.pth 初始化 backbone）
    pretrained_backbone: Optional[str] = None

    clinical_fields: List[str] = field(default_factory=lambda: [
        "age", "weight", "cdr", "mmse", "gds", "faq", "npi-q",
    ])
    plasma_fields: List[str] = field(default_factory=lambda: [
        "pt217_f", "ab42_f", "ab40_f", "ab42_ab40_f", "pt217_ab42_f", "nfl_q", "gfap_q",
    ])


@dataclass
class Config:
    """主配置类"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    condition: ConditionConfig = field(default_factory=ConditionConfig)
    
    # 输出目录
    output_dir: str = field(default_factory=lambda: os.path.join(
        "/home/ssddata/linshuijin/replicaLT/baseline/runs",
        datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    ))
    
    # 设备
    device: str = "cuda"
    # 指定可见显卡（如 "0" 或 "0,1"，None 表示不限制/使用环境变量）
    cuda_visible_devices: Optional[str] = '4'
    
    def __post_init__(self):
        """确保输出目录存在"""
        # 约束可见显卡（仅在显式指定时覆盖环境变量）
        if self.cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cuda_visible_devices)

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "predictions"), exist_ok=True)


def get_default_config() -> Config:
    """获取默认配置"""
    return Config()
