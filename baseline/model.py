"""
baseline/model.py
=================
3D Residual U-Net for MRI → TAU-PET Generation

特性:
1. 编码器-解码器结构，4 个尺度（3 次下采样）
2. 残差块：Conv3D → Norm → LeakyReLU × 2
3. Skip connections
4. 残差输出模式: pred = clip(mri + delta, 0, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .condition import FiLMLayer, TabularEncoder


# ============================================================================
# 基础组件
# ============================================================================

def get_norm_layer(norm_type: str, num_features: int, num_groups: int = 8):
    """获取归一化层"""
    if norm_type == "instance":
        return nn.InstanceNorm3d(num_features, affine=True)
    elif norm_type == "batch":
        return nn.BatchNorm3d(num_features)
    elif norm_type == "group":
        return nn.GroupNorm(num_groups, num_features)
    elif norm_type == "layer":
        return nn.GroupNorm(1, num_features)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


def get_activation(activation: str, inplace: bool = True):
    """获取激活函数"""
    if activation == "relu":
        return nn.ReLU(inplace=inplace)
    elif activation == "leaky_relu":
        return nn.LeakyReLU(0.2, inplace=inplace)
    elif activation == "prelu":
        return nn.PReLU()
    elif activation == "elu":
        return nn.ELU(inplace=inplace)
    else:
        raise ValueError(f"Unknown activation: {activation}")


class ConvBlock(nn.Module):
    """
    基础卷积块: Conv3D → Norm → Activation
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm_type: str = "instance",
        activation: str = "leaky_relu",
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            get_norm_layer(norm_type, out_channels),
            get_activation(activation),
        ]
        
        if dropout_rate > 0:
            layers.append(nn.Dropout3d(dropout_rate))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    残差块: 两个 ConvBlock + Skip Connection
    
    如果 in_channels != out_channels，使用 1x1 卷积调整
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "instance",
        activation: str = "leaky_relu",
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        
        self.conv1 = ConvBlock(
            in_channels, out_channels,
            norm_type=norm_type, activation=activation, dropout_rate=dropout_rate
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False),
            get_norm_layer(norm_type, out_channels),
        )
        
        # 残差连接
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, 1, 1, 0, bias=False)
        else:
            self.skip = nn.Identity()
        
        self.activation = get_activation(activation)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        out = self.activation(out)
        return out


class DoubleConv(nn.Module):
    """
    双卷积块（非残差）: ConvBlock × 2
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "instance",
        activation: str = "leaky_relu",
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels, norm_type=norm_type, 
                     activation=activation, dropout_rate=dropout_rate),
            ConvBlock(out_channels, out_channels, norm_type=norm_type,
                     activation=activation, dropout_rate=dropout_rate),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """
    下采样块: MaxPool + ResidualBlock/DoubleConv
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_residual: bool = True,
        norm_type: str = "instance",
        activation: str = "leaky_relu",
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        
        self.pool = nn.MaxPool3d(2)
        
        if use_residual:
            self.conv = ResidualBlock(
                in_channels, out_channels,
                norm_type=norm_type, activation=activation, dropout_rate=dropout_rate
            )
        else:
            self.conv = DoubleConv(
                in_channels, out_channels,
                norm_type=norm_type, activation=activation, dropout_rate=dropout_rate
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    """
    上采样块: TransposedConv/Upsample + Concat Skip + ResidualBlock/DoubleConv
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_residual: bool = True,
        use_transposed_conv: bool = True,
        norm_type: str = "instance",
        activation: str = "leaky_relu",
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        
        # 上采样
        if use_transposed_conv:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, 2, 2)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(in_channels, in_channels // 2, 1, 1, 0),
            )
        
        # 卷积（输入是 up 输出 + skip，所以是 in_channels // 2 + in_channels // 2 = in_channels）
        if use_residual:
            self.conv = ResidualBlock(
                in_channels, out_channels,
                norm_type=norm_type, activation=activation, dropout_rate=dropout_rate
            )
        else:
            self.conv = DoubleConv(
                in_channels, out_channels,
                norm_type=norm_type, activation=activation, dropout_rate=dropout_rate
            )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        
        # 处理尺寸不匹配（如果有的话）
        if x.shape != skip.shape:
            diff_d = skip.size(2) - x.size(2)
            diff_h = skip.size(3) - x.size(3)
            diff_w = skip.size(4) - x.size(4)
            x = F.pad(x, [
                diff_w // 2, diff_w - diff_w // 2,
                diff_h // 2, diff_h - diff_h // 2,
                diff_d // 2, diff_d - diff_d // 2,
            ])
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


# ============================================================================
# 主模型
# ============================================================================

class ResidualUNet3D(nn.Module):
    """
    3D Residual U-Net for MRI → TAU-PET Generation
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        base_features: 初始特征通道数
        num_scales: 尺度数（包括原始尺度）
        use_residual: 是否使用残差块
        use_residual_output: 是否使用残差输出模式 (pred = clip(input + delta, 0, 1))
        norm_type: 归一化类型
        activation: 激活函数
        dropout_rate: Dropout 率
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_features: int = 32,
        num_scales: int = 4,
        use_residual: bool = True,
        use_residual_output: bool = True,
        norm_type: str = "instance",
        activation: str = "leaky_relu",
        dropout_rate: float = 0.0,
        condition_dim: Optional[int] = None,
        film_hidden_dim: int = 128,
        film_reg_lambda: float = 0.0,
        clinical_dim: int = 7,
        plasma_dim: int = 7,
        condition_mode: str = "both",
    ):
        super().__init__()
        
        self.use_residual_output = use_residual_output
        self.num_scales = num_scales
        self.condition_dim = condition_dim
        self.film_reg_lambda = film_reg_lambda
        self.use_film = condition_dim is not None
        self.condition_mode = condition_mode
        
        # 特征通道数: [32, 64, 128, 256] for num_scales=4
        features = [base_features * (2 ** i) for i in range(num_scales)]
        
        # 编码器
        # 初始卷积
        if use_residual:
            self.inc = ResidualBlock(
                in_channels, features[0],
                norm_type=norm_type, activation=activation, dropout_rate=dropout_rate
            )
        else:
            self.inc = DoubleConv(
                in_channels, features[0],
                norm_type=norm_type, activation=activation, dropout_rate=dropout_rate
            )
        
        # 下采样路径
        self.down_blocks = nn.ModuleList()
        for i in range(num_scales - 1):
            self.down_blocks.append(
                DownBlock(
                    features[i], features[i + 1],
                    use_residual=use_residual,
                    norm_type=norm_type, activation=activation, dropout_rate=dropout_rate
                )
            )
        
        # 瓶颈层（bottleneck）
        if use_residual:
            self.bottleneck = ResidualBlock(
                features[-1], features[-1] * 2,
                norm_type=norm_type, activation=activation, dropout_rate=dropout_rate
            )
        else:
            self.bottleneck = DoubleConv(
                features[-1], features[-1] * 2,
                norm_type=norm_type, activation=activation, dropout_rate=dropout_rate
            )
        
        # 解码器
        # 上采样路径
        self.up_blocks = nn.ModuleList()
        reversed_features = features[::-1]  # [256, 128, 64, 32]
        
        # 第一个上采样从 bottleneck 开始
        self.up_blocks.append(
            UpBlock(
                features[-1] * 2, reversed_features[0],
                use_residual=use_residual,
                norm_type=norm_type, activation=activation, dropout_rate=dropout_rate
            )
        )
        
        for i in range(len(reversed_features) - 1):
            self.up_blocks.append(
                UpBlock(
                    reversed_features[i], reversed_features[i + 1],
                    use_residual=use_residual,
                    norm_type=norm_type, activation=activation, dropout_rate=dropout_rate
                )
            )
        
        # 输出层
        self.outc = nn.Conv3d(features[0], out_channels, 1)

        # FiLM 条件注入（bottleneck + 最后一个解码层）
        if self.use_film:
            self.condition_encoder = TabularEncoder(
                clinical_dim=clinical_dim,
                plasma_dim=plasma_dim,
                embed_dim=condition_dim,
                mode=condition_mode,
            )
            self.film_bottleneck = FiLMLayer(condition_dim, features[-1] * 2, hidden_dim=film_hidden_dim)
            self.film_dec_last = FiLMLayer(condition_dim, features[0], hidden_dim=film_hidden_dim)
        else:
            self.condition_encoder = None
            self.film_bottleneck = None
            self.film_dec_last = None
    
    def forward(self, x: torch.Tensor, condition: Optional[object] = None) -> torch.Tensor:
        """
        Args:
            x: [B, 1, D, H, W] MRI 输入
        
        Returns:
            pred: [B, 1, D, H, W] 预测的 TAU-PET
        """
        # 保存输入用于残差输出
        input_x = x
        
        # 编码器
        skips = []
        
        x = self.inc(x)
        skips.append(x)
        
        for down in self.down_blocks:
            x = down(x)
            skips.append(x)
        
        # 瓶颈
        x = self.bottleneck(skips[-1])
        if self.use_film:
            if condition is None:
                raise ValueError("Condition is required when FiLM is enabled.")
            if isinstance(condition, dict):
                condition = self.condition_encoder(
                    condition["clinical"],
                    condition["plasma"],
                    condition.get("clinical_mask"),
                    condition.get("plasma_mask"),
                    condition["sex"],
                    condition["source"],
                )
            x = self.film_bottleneck(x, condition)
        
        # 解码器
        for i, up in enumerate(self.up_blocks):
            skip_idx = len(skips) - 1 - i
            x = up(x, skips[skip_idx])
            if self.use_film and i == len(self.up_blocks) - 1:
                x = self.film_dec_last(x, condition)
        
        # 输出
        delta = self.outc(x)
        
        # 残差输出模式
        if self.use_residual_output:
            pred = torch.clamp(input_x + delta, 0, 1)
        else:
            pred = torch.sigmoid(delta)
        
        return pred

    def get_film_reg_loss(self) -> torch.Tensor:
        if not self.use_film or self.film_reg_lambda <= 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        reg = self.film_bottleneck.reg_loss() + self.film_dec_last.reg_loss()
        return reg * self.film_reg_lambda
    
    def get_num_parameters(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_parameters(self) -> int:
        """获取可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config) -> ResidualUNet3D:
    """
    根据配置创建模型
    """
    model = ResidualUNet3D(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        base_features=config.model.base_features,
        num_scales=config.model.num_scales,
        use_residual=config.model.use_residual,
        use_residual_output=config.model.use_residual_output,
        norm_type=config.model.norm_type,
        activation=config.model.activation,
        dropout_rate=config.model.dropout_rate,
        condition_dim=(config.condition.embed_dim if config.condition.mode != "none" else None),
        film_hidden_dim=config.condition.film_hidden_dim,
        film_reg_lambda=config.condition.film_reg_lambda,
        clinical_dim=len(config.condition.clinical_fields),
        plasma_dim=len(config.condition.plasma_fields),
        condition_mode=config.condition.mode,
    )
    
    return model


if __name__ == "__main__":
    # 测试模型
    import sys
    sys.path.insert(0, "/home/ssddata/linshuijin/replicaLT/baseline")
    from config import get_default_config
    
    config = get_default_config()
    model = create_model(config)
    
    print(f"模型参数总数: {model.get_num_parameters():,}")
    print(f"可训练参数数: {model.get_num_trainable_parameters():,}")
    
    # 测试前向传播
    x = torch.randn(1, 1, 160, 192, 160)
    print(f"\n输入形状: {x.shape}")
    
    with torch.no_grad():
        y = model(x)
    
    print(f"输出形状: {y.shape}")
    print(f"输出范围: [{y.min():.4f}, {y.max():.4f}]")
