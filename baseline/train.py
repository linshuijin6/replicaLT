"""
baseline/train.py
=================
训练脚本

特性:
1. TensorBoard 日志记录
2. 混合精度训练
3. 梯度累积
4. 学习率调度（warmup + cosine）
5. 最佳模型保存
6. 早停
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import pandas as pd

# 添加 baseline 到 path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, get_default_config
from dataset import create_dataloaders
from model import create_model, ResidualUNet3D
from losses import create_loss, MetricsCalculator


# ============================================================================
# 日志设置
# ============================================================================

def setup_logging(output_dir: str) -> logging.Logger:
    """设置日志"""
    log_file = os.path.join(output_dir, "train.log")
    
    # 创建 logger
    logger = logging.getLogger("baseline")
    logger.setLevel(logging.INFO)
    
    # 文件 handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # 控制台 handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ============================================================================
# 学习率调度
# ============================================================================

def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr: float = 0.0,
):
    """带 warmup 的余弦学习率调度器"""
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr / optimizer.defaults["lr"], 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# 训练器
# ============================================================================

class Trainer:
    """训练器类"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # 日志
        self.logger = setup_logging(config.output_dir)
        self.logger.info(f"输出目录: {config.output_dir}")
        self.logger.info(f"设备: {self.device}")
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=config.output_dir)
        
        # 保存配置
        self._save_config()
        
        # 数据加载器
        self.logger.info("创建数据加载器...")
        self.train_loader, self.val_loader, self.test_loader, \
            self.train_df, self.val_df, self.test_df = create_dataloaders(config)
        
        self.logger.info(f"训练集: {len(self.train_df)} 样本")
        self.logger.info(f"验证集: {len(self.val_df)} 样本")
        self.logger.info(f"测试集: {len(self.test_df)} 样本")
        
        # 模型
        self.logger.info("创建模型...")
        self.model = create_model(config).to(self.device)
        self.logger.info(f"模型参数量: {self.model.get_num_parameters():,}")
        
        # 损失函数
        self.loss_fn = create_loss(config)
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        
        # 学习率调度器
        num_training_steps = len(self.train_loader) * config.train.epochs
        num_warmup_steps = len(self.train_loader) * config.train.warmup_epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            min_lr=config.train.min_lr,
        )
        
        # 混合精度
        self.scaler = GradScaler() if config.train.use_amp else None
        
        # 评估指标
        self.metrics_calc = MetricsCalculator()
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_mae = float("inf")
        self.best_val_ssim = 0.0
        self.patience_counter = 0
    
    def _save_config(self):
        """保存配置"""
        config_path = os.path.join(self.config.output_dir, "config.json")
        
        # 转换为可序列化的字典
        config_dict = {
            "data": vars(self.config.data),
            "model": vars(self.config.model),
            "loss": vars(self.config.loss),
            "train": vars(self.config.train),
            "output_dir": self.config.output_dir,
            "device": self.config.device,
        }
        
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        
        epoch_losses = []
        epoch_l1 = []
        epoch_ssim = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            mri = batch["mri"].to(self.device)
            tau = batch["tau"].to(self.device)
            weights = batch["weight"].to(self.device)
            
            # 前向传播
            if self.config.train.use_amp:
                with autocast():
                    pred = self.model(mri)
                    loss, loss_dict = self.loss_fn(pred, tau, weights)
                    loss = loss / self.config.train.accumulation_steps
                
                # 反向传播
                self.scaler.scale(loss).backward()
            else:
                pred = self.model(mri)
                loss, loss_dict = self.loss_fn(pred, tau, weights)
                loss = loss / self.config.train.accumulation_steps
                loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % self.config.train.accumulation_steps == 0:
                if self.config.train.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # 记录
            epoch_losses.append(loss_dict["loss"])
            epoch_l1.append(loss_dict["l1"])
            epoch_ssim.append(loss_dict["ssim"])
            
            # 更新进度条
            pbar.set_postfix({
                "loss": f"{loss_dict['loss']:.4f}",
                "L1": f"{loss_dict['l1']:.4f}",
                "SSIM": f"{loss_dict['ssim']:.4f}",
            })
            
            # TensorBoard
            if self.global_step % 10 == 0:
                self.writer.add_scalar("train/loss", loss_dict["loss"], self.global_step)
                self.writer.add_scalar("train/l1", loss_dict["l1"], self.global_step)
                self.writer.add_scalar("train/ssim", loss_dict["ssim"], self.global_step)
                self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.global_step)
        
        return {
            "loss": np.mean(epoch_losses),
            "l1": np.mean(epoch_l1),
            "ssim": np.mean(epoch_ssim),
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        
        all_mae = []
        all_psnr = []
        all_ssim = []
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            mri = batch["mri"].to(self.device)
            tau = batch["tau"].to(self.device)
            
            if self.config.train.use_amp:
                with autocast():
                    pred = self.model(mri)
            else:
                pred = self.model(mri)
            
            # 计算指标
            for i in range(pred.size(0)):
                metrics = self.metrics_calc.compute(pred[i:i+1], tau[i:i+1])
                all_mae.append(metrics["mae"])
                all_psnr.append(metrics["psnr"])
                all_ssim.append(metrics["ssim"])
        
        return {
            "mae": np.mean(all_mae),
            "psnr": np.mean(all_psnr),
            "ssim": np.mean(all_ssim),
            "mae_std": np.std(all_mae),
            "psnr_std": np.std(all_psnr),
            "ssim_std": np.std(all_ssim),
        }
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """保存 checkpoint"""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_mae": self.best_val_mae,
            "best_val_ssim": self.best_val_ssim,
        }
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        path = os.path.join(self.config.output_dir, "checkpoints", filename)
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = os.path.join(self.config.output_dir, "checkpoints", "best_model.pth")
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: str):
        """加载 checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_mae = checkpoint["best_val_mae"]
        self.best_val_ssim = checkpoint["best_val_ssim"]
        
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.logger.info(f"加载 checkpoint: {path}")
    
    def train(self):
        """完整训练流程"""
        self.logger.info("=" * 70)
        self.logger.info("开始训练")
        self.logger.info("=" * 70)
        
        for epoch in range(self.config.train.epochs):
            self.current_epoch = epoch + 1
            
            # 训练
            train_metrics = self.train_epoch()
            self.logger.info(
                f"Epoch {self.current_epoch}/{self.config.train.epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"L1: {train_metrics['l1']:.4f}, "
                f"SSIM: {train_metrics['ssim']:.4f}"
            )
            
            # 验证
            if self.current_epoch % self.config.train.val_freq == 0:
                val_metrics = self.validate()
                self.logger.info(
                    f"  Val MAE: {val_metrics['mae']:.4f} ± {val_metrics['mae_std']:.4f}, "
                    f"PSNR: {val_metrics['psnr']:.2f} ± {val_metrics['psnr_std']:.2f}, "
                    f"SSIM: {val_metrics['ssim']:.4f} ± {val_metrics['ssim_std']:.4f}"
                )
                
                # TensorBoard
                self.writer.add_scalar("val/mae", val_metrics["mae"], self.current_epoch)
                self.writer.add_scalar("val/psnr", val_metrics["psnr"], self.current_epoch)
                self.writer.add_scalar("val/ssim", val_metrics["ssim"], self.current_epoch)
                
                # 检查是否是最佳模型
                is_best = val_metrics["mae"] < self.best_val_mae
                if is_best:
                    self.best_val_mae = val_metrics["mae"]
                    self.best_val_ssim = val_metrics["ssim"]
                    self.patience_counter = 0
                    self.logger.info(f"  ★ 新的最佳模型! MAE: {self.best_val_mae:.4f}")
                else:
                    self.patience_counter += 1
                
                # 保存
                if self.config.train.save_best and is_best:
                    self.save_checkpoint("best_model.pth", is_best=True)
            
            # 定期保存
            if self.current_epoch % self.config.train.save_freq == 0:
                self.save_checkpoint(f"epoch_{self.current_epoch}.pth")
            
            # 早停
            if self.patience_counter >= self.config.train.early_stopping_patience:
                self.logger.info(f"早停触发，{self.config.train.early_stopping_patience} 个 epoch 无改善")
                break
        
        # 保存最终模型
        if self.config.train.save_last:
            self.save_checkpoint("last_model.pth")
        
        self.logger.info("=" * 70)
        self.logger.info(f"训练完成! 最佳 MAE: {self.best_val_mae:.4f}, SSIM: {self.best_val_ssim:.4f}")
        self.logger.info("=" * 70)
        
        # 关闭 writer
        self.writer.close()
        
        return self.best_val_mae, self.best_val_ssim


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MRI → TAU-PET Baseline Training")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=None, help="批大小")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的 checkpoint 路径")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    args = parser.parse_args()
    
    # 创建配置
    config = get_default_config()
    
    # 覆盖配置
    if args.epochs:
        config.train.epochs = args.epochs
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.lr:
        config.train.learning_rate = args.lr
    if args.output_dir:
        config.output_dir = args.output_dir
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "predictions"), exist_ok=True)
    
    # 创建训练器
    trainer = Trainer(config)
    
    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
