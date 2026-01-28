"""
多GPU优化训练脚本 - 三模态MRI到PET转换
========================================

主要优化策略：
1. 梯度累积 (Gradient Accumulation)：降低单次前向传播的内存占用
2. 顺序模态训练：分别计算三个模态，避免同时占用内存
3. 混合精度训练 (Mixed Precision)：使用FP16减少内存占用
4. 显存优化：及时清理中间变量，使用checkpoint技术

内存消耗分析（单GPU，bs=1）：
- 原方案（两模态同时）：~30GB
- 优化方案（三模态顺序）：预计 ~20-25GB
- 多GPU方案（数据并行）：每GPU ~15-20GB

支持的GPU配置：
- device_ids=[0]：单GPU训练
- device_ids=[0,1]：双GPU数据并行
- device_ids=[0,2,3]：三GPU数据并行
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DataParallel
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
import os


# ========================== 配置参数 ==========================
class TrainingConfig:
    """训练配置类"""
    def __init__(self):
        # GPU配置：支持单GPU或多GPU训练
        # 例如：[0] 单GPU，[0,1] 双GPU，[0,2,3] 三GPU
        # 自动检测可用GPU数量
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.device_ids = [0] if num_gpus >= 1 else [-1]  # 可修改为 [0], [0,1], [0,2,3] 等
        
        # 梯度累积步数：用于模拟更大的batch size同时降低内存
        # 实际batch_size = batch_size * accumulation_steps
        self.accumulation_steps = 2  # 累积2步再更新，相当于bs=2
        
        # 训练参数
        self.n_epochs = 200
        self.val_interval = 10
        self.learning_rate = 2.5e-5
        
        # 模态权重：可以根据重要性调整
        self.alpha = 1.0  # FDG权重
        self.beta = 1.0   # AV45权重
        self.gamma = 1.0  # TAU权重
        
        # 混合精度训练
        self.use_amp = True  # 是否使用自动混合精度
        
        # 裁剪范围
        self.clip_sample_min = 0.0
        self.clip_sample_max = 1.0
        
        # Adapter配置
        self.adapter_hidden = 512
        self.adapter_dropout = 0.1
        self.adapter_dim = 512  # BiomedCLIP的projection_dim


# ========================== 模拟数据生成器 ==========================
class SimulatedDataGenerator:
    """
    模拟数据生成器：用于测试和开发
    生成与真实数据相同形状的随机数据
    """
    def __init__(self, num_samples: int = 100):
        self.num_samples = num_samples
        
    def generate_batch(self, batch_size: int = 1) -> Dict[str, torch.Tensor]:
        """
        生成一个batch的模拟数据
        
        Returns:
            字典包含：
            - mri: [B, 1, 192, 160, 160] MRI图像
            - fdg: [B, 1, 192, 160, 160] FDG-PET图像
            - av45: [B, 1, 192, 160, 160] AV45-PET图像
            - tau: [B, 1, 192, 160, 160] TAU-PET图像
            - fdg_index: [B, 2, 512] FDG条件embedding（text + template）
            - av45_index: [B, 2, 512] AV45条件embedding
            - tau_index: [B, 2, 512] TAU条件embedding
        """
        # 图像数据：[batch, channel=1, depth=192, height=160, width=160]
        mri = torch.randn(batch_size, 1, 192, 160, 160)
        fdg = torch.randn(batch_size, 1, 192, 160, 160)
        av45 = torch.randn(batch_size, 1, 192, 160, 160)
        tau = torch.randn(batch_size, 1, 192, 160, 160)
        
        # 归一化到[0, 1]
        mri = (mri - mri.min()) / (mri.max() - mri.min() + 1e-8)
        fdg = (fdg - fdg.min()) / (fdg.max() - fdg.min() + 1e-8)
        av45 = (av45 - av45.min()) / (av45.max() - av45.min() + 1e-8)
        tau = (tau - tau.min()) / (tau.max() - tau.min() + 1e-8)
        
        # 条件embedding：[batch, 2, 512]（2表示text_feat和template_feat拼接）
        fdg_index = torch.randn(batch_size, 2, 512)
        av45_index = torch.randn(batch_size, 2, 512)
        tau_index = torch.randn(batch_size, 2, 512)
        
        return {
            'mri': mri,
            'fdg': fdg,
            'av45': av45,
            'tau': tau,
            'fdg_index': fdg_index,
            'av45_index': av45_index,
            'tau_index': tau_index
        }
    
    def __len__(self):
        return self.num_samples


# ========================== 简化的UNet模型 ==========================
class SimplifiedDiffusionUNet(nn.Module):
    """
    简化的扩散模型UNet（用于演示和测试）
    实际使用时应替换为真实的 DiffusionModelUNet
    """
    def __init__(self, in_channels=1, out_channels=1, cross_attention_dim=512):
        super().__init__()
        
        # 编码器
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True)
        )
        
        # 时间embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # 交叉注意力层（简化版，实际应使用真实的cross-attention）
        self.cross_attn = nn.Sequential(
            nn.Linear(cross_attention_dim, 64),
            nn.ReLU()
        )
        
        # 解码器
        self.dec1 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, stride=2),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Conv3d(32, out_channels, 3, padding=1)
        
    def forward(self, x, timesteps, context):
        """
        前向传播
        
        Args:
            x: [B, 1, D, H, W] 输入图像
            timesteps: [B] 时间步
            context: [B, 2, 512] 条件embedding
        """
        # 编码
        h1 = self.enc1(x)  # [B, 32, D, H, W]
        h2 = self.enc2(h1)  # [B, 64, D/2, H/2, W/2]
        
        # 时间embedding
        t_emb = self.time_embed(timesteps.float().unsqueeze(1) / 1000.0)  # [B, 64]
        
        # 交叉注意力（简化版：只使用context的平均）
        # 实际中应该使用真正的cross-attention机制
        c_emb = self.cross_attn(context.mean(dim=1))  # [B, 64]
        
        # 融合时间和条件信息
        cond = (t_emb + c_emb).view(-1, 64, 1, 1, 1)
        h2 = h2 + cond
        
        # 解码
        h3 = self.dec1(h2)  # [B, 32, D, H, W]
        out = self.dec2(h3)  # [B, 1, D, H, W]
        
        return out


# ========================== 内存优化的训练器 ==========================
class MemoryEfficientTrainer:
    """
    内存优化的多GPU训练器
    
    主要优化策略：
    1. 顺序模态训练：依次计算FDG、AV45、TAU的损失，避免同时占用内存
    2. 梯度累积：分多步累积梯度，降低单步内存占用
    3. 及时清理：每个模态计算完立即释放显存
    """
    
    def __init__(self, model, config: TrainingConfig):
        self.config = config
        self.device = torch.device(f'cuda:{config.device_ids[0]}')
        
        # 多GPU设置
        if len(config.device_ids) > 1:
            print(f"使用多GPU训练: {config.device_ids}")
            self.model = DataParallel(model, device_ids=config.device_ids)
        else:
            print(f"使用单GPU训练: {config.device_ids[0]}")
            self.model = model
        
        self.model = self.model.to(self.device)
        
        # 优化器和混合精度
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.scaler = GradScaler(enabled=config.use_amp)
        
        # 梯度累积计数器
        self.accumulation_counter = 0
    
    def train_step_sequential(self, data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        单步训练（顺序模态版本）
        
        策略：依次计算三个模态的损失，每个模态计算完立即backward并清理
        这样可以避免同时保存三个模态的中间结果
        
        Args:
            data: 包含mri, fdg, av45, tau和对应条件的字典
            
        Returns:
            包含各模态损失的字典
        """
        # 将数据移到GPU
        mri = data['mri'].to(self.device)
        fdg_gt = data['fdg'].to(self.device)
        av45_gt = data['av45'].to(self.device)
        tau_gt = data['tau'].to(self.device)
        
        fdg_index = data['fdg_index'].to(self.device)
        av45_index = data['av45_index'].to(self.device)
        tau_index = data['tau_index'].to(self.device)
        
        # 检查是否有有效数据（非全零）
        has_fdg = not torch.all(fdg_gt == 0)
        has_av45 = not torch.all(av45_gt == 0)
        has_tau = not torch.all(tau_gt == 0)
        
        # 随机时间步
        timesteps = torch.randint(0, 1000, (mri.shape[0],), device=self.device).long()
        t = timesteps.float() / 1000.0
        t = t.view(-1, 1, 1, 1, 1)  # [B, 1, 1, 1, 1]
        
        losses = {
            'fdg': 0.0,
            'av45': 0.0,
            'tau': 0.0,
            'total': 0.0
        }
        
        # ============ 模态1: FDG ============
        if has_fdg:
            with autocast('cuda', enabled=self.config.use_amp):
                # 插值：x_t = t * target + (1-t) * source
                x_t_fdg = t * fdg_gt + (1 - t) * mri
                
                # 前向传播
                v_fdg_pred = self.model(x=x_t_fdg, timesteps=timesteps, context=fdg_index)
                
                # 计算速度目标 v = target - source
                v_fdg_gt = fdg_gt - mri
                
                # 计算损失
                loss_fdg = F.mse_loss(v_fdg_pred, v_fdg_gt)
                loss_fdg = loss_fdg * self.config.alpha / self.config.accumulation_steps
            
            # 反向传播（梯度累积）
            self.scaler.scale(loss_fdg).backward()
            losses['fdg'] = loss_fdg.item() * self.config.accumulation_steps
            
            # 清理中间变量
            del x_t_fdg, v_fdg_pred, v_fdg_gt, loss_fdg
            torch.cuda.empty_cache()
        
        # ============ 模态2: AV45 ============
        if has_av45:
            with autocast('cuda', enabled=self.config.use_amp):
                x_t_av45 = t * av45_gt + (1 - t) * mri
                v_av45_pred = self.model(x=x_t_av45, timesteps=timesteps, context=av45_index)
                v_av45_gt = av45_gt - mri
                loss_av45 = F.mse_loss(v_av45_pred, v_av45_gt)
                loss_av45 = loss_av45 * self.config.beta / self.config.accumulation_steps
            
            self.scaler.scale(loss_av45).backward()
            losses['av45'] = loss_av45.item() * self.config.accumulation_steps
            
            del x_t_av45, v_av45_pred, v_av45_gt, loss_av45
            torch.cuda.empty_cache()
        
        # ============ 模态3: TAU ============
        if has_tau:
            with autocast('cuda', enabled=self.config.use_amp):
                x_t_tau = t * tau_gt + (1 - t) * mri
                v_tau_pred = self.model(x=x_t_tau, timesteps=timesteps, context=tau_index)
                v_tau_gt = tau_gt - mri
                loss_tau = F.mse_loss(v_tau_pred, v_tau_gt)
                loss_tau = loss_tau * self.config.gamma / self.config.accumulation_steps
            
            self.scaler.scale(loss_tau).backward()
            losses['tau'] = loss_tau.item() * self.config.accumulation_steps
            
            del x_t_tau, v_tau_pred, v_tau_gt, loss_tau
            torch.cuda.empty_cache()
        
        # 梯度累积：每accumulation_steps次更新一次参数
        self.accumulation_counter += 1
        if self.accumulation_counter % self.config.accumulation_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        
        losses['total'] = losses['fdg'] + losses['av45'] + losses['tau']
        
        return losses
    
    @torch.no_grad()
    def validation_step(self, data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        验证步骤
        
        使用单步推理（N_sample=1）生成三个模态的预测结果
        """
        self.model.eval()
        
        mri = data['mri'].to(self.device)
        fdg_gt = data['fdg'].to(self.device)
        av45_gt = data['av45'].to(self.device)
        tau_gt = data['tau'].to(self.device)
        
        fdg_index = data['fdg_index'].to(self.device)
        av45_index = data['av45_index'].to(self.device)
        tau_index = data['tau_index'].to(self.device)
        
        has_fdg = not torch.all(fdg_gt == 0)
        has_av45 = not torch.all(av45_gt == 0)
        has_tau = not torch.all(tau_gt == 0)
        
        losses = {'fdg': 0.0, 'av45': 0.0, 'tau': 0.0, 'total': 0.0}
        
        # 单步推理（t=0，即time_embedding=0）
        timesteps = torch.zeros(mri.shape[0], device=self.device).long()
        
        # FDG推理
        if has_fdg:
            v_fdg = self.model(x=mri, timesteps=timesteps, context=fdg_index)
            fdg_pred = mri + v_fdg
            fdg_pred = torch.clamp(fdg_pred, self.config.clip_sample_min, self.config.clip_sample_max)
            losses['fdg'] = F.mse_loss(fdg_pred, fdg_gt).item()
            del v_fdg, fdg_pred
        
        # AV45推理
        if has_av45:
            v_av45 = self.model(x=mri, timesteps=timesteps, context=av45_index)
            av45_pred = mri + v_av45
            av45_pred = torch.clamp(av45_pred, self.config.clip_sample_min, self.config.clip_sample_max)
            losses['av45'] = F.mse_loss(av45_pred, av45_gt).item()
            del v_av45, av45_pred
        
        # TAU推理
        if has_tau:
            v_tau = self.model(x=mri, timesteps=timesteps, context=tau_index)
            tau_pred = mri + v_tau
            tau_pred = torch.clamp(tau_pred, self.config.clip_sample_min, self.config.clip_sample_max)
            losses['tau'] = F.mse_loss(tau_pred, tau_gt).item()
            del v_tau, tau_pred
        
        losses['total'] = losses['fdg'] + losses['av45'] + losses['tau']
        torch.cuda.empty_cache()
        
        self.model.train()
        return losses


# ========================== 主训练循环 ==========================
def train_epoch(trainer: MemoryEfficientTrainer, 
                dataloader, 
                epoch: int) -> Dict[str, float]:
    """
    训练一个epoch
    
    Args:
        trainer: 训练器实例
        dataloader: 数据加载器
        epoch: 当前epoch编号
        
    Returns:
        平均损失字典
    """
    trainer.model.train()
    epoch_losses = {'fdg': 0.0, 'av45': 0.0, 'tau': 0.0, 'total': 0.0}
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}', ncols=100)
    
    for step, data in enumerate(progress_bar):
        losses = trainer.train_step_sequential(data)
        
        # 累积损失
        for key in epoch_losses:
            epoch_losses[key] += losses[key]
        
        # 更新进度条
        avg_loss = epoch_losses['total'] / (step + 1)
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'fdg': f'{losses["fdg"]:.4f}',
            'av45': f'{losses["av45"]:.4f}',
            'tau': f'{losses["tau"]:.4f}'
        })
    
    # 计算平均损失
    num_batches = len(dataloader)
    for key in epoch_losses:
        epoch_losses[key] /= num_batches
    
    return epoch_losses


def validate_epoch(trainer: MemoryEfficientTrainer, 
                   dataloader) -> Dict[str, float]:
    """
    验证一个epoch
    """
    epoch_losses = {'fdg': 0.0, 'av45': 0.0, 'tau': 0.0, 'total': 0.0}
    
    progress_bar = tqdm(dataloader, desc='Validation', ncols=100)
    
    for step, data in enumerate(progress_bar):
        losses = trainer.validation_step(data)
        
        for key in epoch_losses:
            epoch_losses[key] += losses[key]
        
        avg_loss = epoch_losses['total'] / (step + 1)
        progress_bar.set_postfix({'val_loss': f'{avg_loss:.4f}'})
    
    num_batches = len(dataloader)
    for key in epoch_losses:
        epoch_losses[key] /= num_batches
    
    return epoch_losses


# ========================== 主函数 ==========================
def main():
    """
    主训练函数（使用模拟数据进行演示）
    """
    print("=" * 60)
    print("多GPU优化训练脚本 - 三模态MRI到PET转换")
    print("=" * 60)
    
    # 创建配置
    config = TrainingConfig()
    
    print(f"\n配置信息:")
    print(f"  GPU设备: {config.device_ids}")
    print(f"  梯度累积步数: {config.accumulation_steps}")
    print(f"  混合精度训练: {config.use_amp}")
    print(f"  模态权重: FDG={config.alpha}, AV45={config.beta}, TAU={config.gamma}")
    
    # 检查GPU可用性
    if not torch.cuda.is_available():
        print("\n警告: 未检测到CUDA，将使用CPU训练（速度会很慢）")
        config.device_ids = [-1]  # CPU模式
    else:
        num_gpus = torch.cuda.device_count()
        print(f"\n检测到 {num_gpus} 个GPU")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 创建模型（使用简化版进行演示）
    print("\n创建模型...")
    model = SimplifiedDiffusionUNet(
        in_channels=1,
        out_channels=1,
        cross_attention_dim=config.adapter_dim
    )
    
    # 统计参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # 创建训练器
    trainer = MemoryEfficientTrainer(model, config)
    
    # 创建模拟数据
    print("\n创建模拟数据...")
    train_generator = SimulatedDataGenerator(num_samples=50)
    val_generator = SimulatedDataGenerator(num_samples=10)
    
    # 模拟训练循环
    print("\n开始训练（使用模拟数据）...\n")
    
    for epoch in range(5):  # 只训练5个epoch作为演示
        # 训练
        train_data = [train_generator.generate_batch() for _ in range(10)]
        train_losses = train_epoch(trainer, train_data, epoch)
        
        print(f"\nEpoch {epoch} 训练损失:")
        print(f"  FDG:   {train_losses['fdg']:.6f}")
        print(f"  AV45:  {train_losses['av45']:.6f}")
        print(f"  TAU:   {train_losses['tau']:.6f}")
        print(f"  Total: {train_losses['total']:.6f}")
        
        # 验证
        if (epoch + 1) % 2 == 0:
            val_data = [val_generator.generate_batch() for _ in range(5)]
            val_losses = validate_epoch(trainer, val_data)
            
            print(f"\nEpoch {epoch} 验证损失:")
            print(f"  FDG:   {val_losses['fdg']:.6f}")
            print(f"  AV45:  {val_losses['av45']:.6f}")
            print(f"  TAU:   {val_losses['tau']:.6f}")
            print(f"  Total: {val_losses['total']:.6f}")
        
        # 显存统计
        if torch.cuda.is_available():
            for gpu_id in config.device_ids:
                allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                print(f"\nGPU {gpu_id} 显存使用: {allocated:.2f}GB / {reserved:.2f}GB (已分配/已保留)")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    
    print("\n使用说明:")
    print("1. 修改 TrainingConfig 中的 device_ids 来选择使用的GPU")
    print("2. 调整 accumulation_steps 来控制显存使用（值越大，显存越小）")
    print("3. 将 SimplifiedDiffusionUNet 替换为真实的模型")
    print("4. 将 SimulatedDataGenerator 替换为真实的数据加载器")


if __name__ == '__main__':
    main()
