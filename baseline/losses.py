"""
baseline/losses.py
==================
损失函数模块

包含:
1. L1 损失
2. 3D SSIM 损失
3. 加权组合损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


# ============================================================================
# 3D SSIM 实现
# ============================================================================

def gaussian_kernel_3d(
    kernel_size: int = 11,
    sigma: float = 1.5,
    channels: int = 1,
    device: torch.device = None,
) -> torch.Tensor:
    """
    创建 3D 高斯核
    
    Returns:
        kernel: [channels, 1, K, K, K]
    """
    # 1D 高斯
    x = torch.arange(kernel_size, dtype=torch.float32, device=device)
    x = x - kernel_size // 2
    gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    
    # 3D 高斯（外积）
    kernel_1d = gauss.view(1, 1, -1)  # [1, 1, K]
    kernel_2d = kernel_1d * kernel_1d.transpose(-1, -2)  # [1, K, K]
    kernel_3d = kernel_2d.unsqueeze(-1) * kernel_1d  # [1, K, K, K]
    
    # 扩展到 channels
    kernel_3d = kernel_3d.expand(channels, 1, kernel_size, kernel_size, kernel_size)
    
    return kernel_3d.contiguous()


from torch.cuda.amp import autocast

@autocast(enabled=False)
def ssim_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    计算 3D SSIM
    
    Args:
        pred: [B, C, D, H, W] 预测
        target: [B, C, D, H, W] 目标
        window_size: 高斯窗口大小
        sigma: 高斯标准差
        data_range: 数据范围
        eps: 数值稳定性
        reduction: 归约方式 (none, mean, sum)
    
    Returns:
        ssim: SSIM 值
    """
    B, C, D, H, W = pred.shape
    
    # 强制 float32 计算，避免混合精度 (float16) 导致方差为负 / SSIM > 1
    orig_dtype = pred.dtype
    pred = pred.float()
    target = target.float()

    # 高斯核改为分离卷积（Separable Convolution），理论加速 K^2 倍 (11x11=121倍，实测约40倍)
    x = torch.arange(window_size, dtype=torch.float32, device=pred.device)
    x = x - window_size // 2
    gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel_1d = gauss.view(1, 1, window_size).expand(C, 1, window_size).contiguous()
    
    padding = window_size // 2
    def apply_gaussian(tensor):
        # 分离卷积: D -> H -> W
        out = F.conv3d(tensor, kernel_1d.view(C, 1, window_size, 1, 1), padding=(padding, 0, 0), groups=C)
        out = F.conv3d(out, kernel_1d.view(C, 1, 1, window_size, 1), padding=(0, padding, 0), groups=C)
        out = F.conv3d(out, kernel_1d.view(C, 1, 1, 1, window_size), padding=(0, 0, padding), groups=C)
        return out
        
    # 动态参数
    k1, k2 = 0.01, 0.03
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    
    # 计算均值
    mu_pred = apply_gaussian(pred)
    mu_target = apply_gaussian(target)
    
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target
    
    # 计算方差和协方差
    sigma_pred_sq = apply_gaussian(pred ** 2) - mu_pred_sq
    sigma_target_sq = apply_gaussian(target ** 2) - mu_target_sq
    sigma_pred_target = apply_gaussian(pred * target) - mu_pred_target
    
    # 钳位方差为非负（浮点误差可能导致 E[X^2]-E[X]^2 < 0）
    sigma_pred_sq = torch.clamp(sigma_pred_sq, min=0.0)
    sigma_target_sq = torch.clamp(sigma_target_sq, min=0.0)
    
    # SSIM
    numerator = (2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    
    ssim_map = numerator / (denominator + eps)
    
    # 归约
    if reduction == "none":
        return ssim_map
    elif reduction == "mean":
        return ssim_map.mean()
    elif reduction == "sum":
        return ssim_map.sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


class SSIM3DLoss(nn.Module):
    """
    3D SSIM 损失
    
    loss = 1 - SSIM
    """
    
    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
        reduction: str = "none",  # 返回每个样本的损失
    ):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C, D, H, W]
            target: [B, C, D, H, W]
        
        Returns:
            loss: [B] if reduction="none" else scalar
        """
        ssim_val = ssim_3d(
            pred, target,
            window_size=self.window_size,
            sigma=self.sigma,
            data_range=self.data_range,
            reduction="none",
        )
        
        # 1 - SSIM 作为损失
        loss = 1.0 - ssim_val
        
        if self.reduction == "none":
            # 每个样本的平均损失
            return loss.mean(dim=[1, 2, 3, 4])  # [B]
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


# ============================================================================
# L1 损失（逐样本）
# ============================================================================

class L1LossPerSample(nn.Module):
    """
    L1 损失，返回每个样本的损失
    """
    
    def __init__(self, reduction: str = "none"):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C, D, H, W]
            target: [B, C, D, H, W]
        
        Returns:
            loss: [B] if reduction="none" else scalar
        """
        diff = torch.abs(pred - target)
        
        if self.reduction == "none":
            # 每个样本的平均 L1
            return diff.mean(dim=[1, 2, 3, 4])  # [B]
        elif self.reduction == "mean":
            return diff.mean()
        elif self.reduction == "sum":
            return diff.sum()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


# ============================================================================
# 组合损失
# ============================================================================

class CombinedLoss(nn.Module):
    """
    组合损失: L1 + SSIM
    
    loss_i = weight_i * (lambda_l1 * L1_i + lambda_ssim * (1 - SSIM_i))
    batch_loss = sum(loss_i) / (sum(weight_i) + eps)
    """
    
    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_ssim: float = 0.3,
        ssim_window_size: int = 11,
        ssim_sigma: float = 1.5,
        eps: float = 1e-8,
    ):
        super().__init__()
        
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.eps = eps
        
        self.l1_loss = L1LossPerSample(reduction="none")
        self.ssim_loss = SSIM3DLoss(
            window_size=ssim_window_size,
            sigma=ssim_sigma,
            reduction="none",
        )
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred: [B, C, D, H, W] 预测
            target: [B, C, D, H, W] 目标
            weights: [B] 每个样本的权重，如果为 None 则不加权
        
        Returns:
            loss: 标量损失
            loss_dict: 包含各分量的字典
        """
        B = pred.size(0)
        
        # 计算各分量
        l1_per_sample = self.l1_loss(pred, target)  # [B]
        ssim_per_sample = self.ssim_loss(pred, target)  # [B]
        
        # 组合
        loss_per_sample = self.lambda_l1 * l1_per_sample + self.lambda_ssim * ssim_per_sample  # [B]
        
        # 加权
        if weights is not None:
            weighted_loss = loss_per_sample * weights
            total_loss = weighted_loss.sum() / (weights.sum() + self.eps)
        else:
            total_loss = loss_per_sample.mean()
        
        # 记录
        loss_dict = {
            "loss": total_loss.item(),
            "l1": l1_per_sample.mean().item(),
            "ssim": (1.0 - ssim_per_sample.mean()).item(),  # 返回 SSIM 值而非 loss
            "l1_weighted": (self.lambda_l1 * l1_per_sample).mean().item(),
            "ssim_loss_weighted": (self.lambda_ssim * ssim_per_sample).mean().item(),
        }
        
        return total_loss, loss_dict


def create_loss(config) -> CombinedLoss:
    """根据配置创建损失函数"""
    return CombinedLoss(
        lambda_l1=config.loss.lambda_l1,
        lambda_ssim=config.loss.lambda_ssim,
        ssim_window_size=config.loss.ssim_window_size,
        ssim_sigma=config.loss.ssim_sigma,
        eps=config.loss.eps,
    )


# ============================================================================
# 评估指标
# ============================================================================

def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """计算 MAE"""
    return torch.abs(pred - target).mean().item()


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """计算 PSNR"""
    mse = F.mse_loss(pred, target)
    if mse < 1e-10:
        return 100.0
    psnr = 10 * torch.log10(data_range ** 2 / mse)
    return psnr.item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    """计算 SSIM"""
    ssim_val = ssim_3d(pred, target, window_size=window_size, reduction="mean")
    return ssim_val.item()


class MetricsCalculator:
    """评估指标计算器"""
    
    def __init__(self, window_size: int = 11):
        self.window_size = window_size
    
    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> dict:
        """
        计算所有指标
        
        Args:
            pred: [B, C, D, H, W] or [C, D, H, W]
            target: [B, C, D, H, W] or [C, D, H, W]
        
        Returns:
            metrics: dict with MAE, PSNR, SSIM
        """
        # 评估指标统一用 float32，避免混合精度导致类型不一致
        if pred.dtype != torch.float32:
            pred = pred.float()
        if target.dtype != torch.float32:
            target = target.float()

        if pred.dim() == 4:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        
        return {
            "mae": compute_mae(pred, target),
            "psnr": compute_psnr(pred, target),
            "ssim": compute_ssim(pred, target, self.window_size),
        }


if __name__ == "__main__":
    # 测试
    import sys
    sys.path.insert(0, "/home/ssddata/linshuijin/replicaLT/baseline")
    from config import get_default_config
    
    config = get_default_config()
    
    # 创建测试数据
    B = 2
    pred = torch.rand(B, 1, 64, 64, 64)
    target = torch.rand(B, 1, 64, 64, 64)
    weights = torch.tensor([1.0, 0.5])
    
    # 测试损失
    loss_fn = create_loss(config)
    loss, loss_dict = loss_fn(pred, target, weights)
    
    print("损失函数测试:")
    print(f"  总损失: {loss.item():.4f}")
    print(f"  L1: {loss_dict['l1']:.4f}")
    print(f"  SSIM: {loss_dict['ssim']:.4f}")
    
    # 测试指标
    metrics_calc = MetricsCalculator()
    metrics = metrics_calc.compute(pred, target)
    
    print("\n评估指标测试:")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  PSNR: {metrics['psnr']:.2f} dB")
    print(f"  SSIM: {metrics['ssim']:.4f}")
