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
from report_error import email_on_error
# 添加 baseline 到 path
sys.path.insert(0, str(Path(__file__).parent))

from .config import Config, get_default_config
from .dataset import create_dataloaders
from .model import create_model, ResidualUNet3D
from .losses import create_loss, MetricsCalculator


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
        self.logger.info(f"目标 PET: {config.data.target_pet}")
        
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

        # 加载预训练 backbone 权重
        if config.condition.pretrained_backbone:
            self._load_pretrained_backbone(config.condition.pretrained_backbone)
        
        # 损失函数
        self.loss_fn = create_loss(config)
        
        # 优化器（条件模式下使用差异学习率）
        if config.condition.mode != "none":
            film_params = [p for n, p in self.model.named_parameters()
                           if "film_" in n or "condition_encoder" in n]
            backbone_params = [p for n, p in self.model.named_parameters()
                               if "film_" not in n and "condition_encoder" not in n]
            self.optimizer = optim.AdamW([
                {"params": backbone_params, "lr": config.train.learning_rate},
                {"params": film_params, "lr": config.train.learning_rate * 3},
            ], weight_decay=config.train.weight_decay)
        else:
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
        self.debug_print = False
        self.debug_batches = 1
    
    def _save_config(self):
        """保存配置"""
        config_path = os.path.join(self.config.output_dir, "config.json")
        
        # 转换为可序列化的字典
        config_dict = {
            "data": vars(self.config.data),
            "model": vars(self.config.model),
            "loss": vars(self.config.loss),
            "train": vars(self.config.train),
            "condition": vars(self.config.condition),
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
            weights = batch["weight"].to(self.device) if self.config.train.use_quality_weight else None
            
            condition = None
            if self.config.condition.mode != "none":
                condition = {
                    "clinical": batch["clinical"].to(self.device),
                    "plasma": batch["plasma"].to(self.device),
                    "clinical_mask": batch["clinical_mask"].to(self.device),
                    "plasma_mask": batch["plasma_mask"].to(self.device),
                    "sex": batch["sex"].to(self.device),
                    "source": batch["source"].to(self.device),
                }

            # 前向传播
            if self.config.train.use_amp:
                with autocast():
                    pred = self.model(mri, condition)
                    loss, loss_dict = self.loss_fn(pred, tau, weights)
                    film_reg = self.model.get_film_reg_loss()
                    loss_total = loss + film_reg
                    loss = loss_total / self.config.train.accumulation_steps
                
                # 反向传播
                self.scaler.scale(loss).backward()
            else:
                pred = self.model(mri, condition)
                loss, loss_dict = self.loss_fn(pred, tau, weights)
                film_reg = self.model.get_film_reg_loss()
                loss_total = loss + film_reg
                loss = loss_total / self.config.train.accumulation_steps
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
            loss_total_val = loss_total.item() if isinstance(loss_total, torch.Tensor) else float(loss_total)
            loss_dict["loss_total"] = loss_total_val
            loss_dict["film_reg"] = film_reg.item() if isinstance(film_reg, torch.Tensor) else float(film_reg)
            epoch_losses.append(loss_total_val)
            epoch_l1.append(loss_dict["l1"])
            epoch_ssim.append(loss_dict["ssim"])

            if self.debug_print and batch_idx < self.debug_batches:
                msg = {
                    "epoch": self.current_epoch,
                    "batch": batch_idx,
                    "mri_shape": tuple(mri.shape),
                    "tau_shape": tuple(tau.shape),
                    "loss_total": loss_dict["loss_total"],
                    "film_reg": loss_dict["film_reg"],
                }
                if condition is not None:
                    msg.update({
                        "clinical_mean": float(condition["clinical"].mean().item()),
                        "clinical_std": float(condition["clinical"].std().item()),
                        "plasma_mean": float(condition["plasma"].mean().item()),
                        "plasma_std": float(condition["plasma"].std().item()),
                        "clinical_mask_mean": float(condition["clinical_mask"].mean().item()),
                        "plasma_mask_mean": float(condition["plasma_mask"].mean().item()),
                        "sex_ids": condition["sex"].detach().cpu().unique().tolist(),
                        "source_ids": condition["source"].detach().cpu().unique().tolist(),
                    })
                self.logger.info(f"[DEBUG] {msg}")
            
            # 更新进度条
            pbar.set_postfix({
                "loss": f"{loss_dict['loss_total']:.4f}",
                "L1": f"{loss_dict['l1']:.4f}",
                "SSIM": f"{loss_dict['ssim']:.4f}",
            })
            
            # TensorBoard
            if self.global_step % 10 == 0:
                self.writer.add_scalar("train/loss", loss_dict["loss_total"], self.global_step)
                self.writer.add_scalar("train/l1", loss_dict["l1"], self.global_step)
                self.writer.add_scalar("train/ssim", loss_dict["ssim"], self.global_step)
                self.writer.add_scalar("train/film_reg", loss_dict["film_reg"], self.global_step)
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
            condition = None
            if self.config.condition.mode != "none":
                condition = {
                    "clinical": batch["clinical"].to(self.device),
                    "plasma": batch["plasma"].to(self.device),
                    "clinical_mask": batch["clinical_mask"].to(self.device),
                    "plasma_mask": batch["plasma_mask"].to(self.device),
                    "sex": batch["sex"].to(self.device),
                    "source": batch["source"].to(self.device),
                }
            
            if self.config.train.use_amp:
                with autocast():
                    pred = self.model(mri, condition)
            else:
                pred = self.model(mri, condition)
            
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

            if self.config.condition.mode != "none":
                self._apply_freeze_schedule()
            
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

    def _apply_freeze_schedule(self):
        freeze_epochs = self.config.condition.freeze_backbone_epochs
        if self.current_epoch <= freeze_epochs:
            self._set_backbone_trainable(False)
        else:
            if not all(p.requires_grad for n, p in self.model.named_parameters() if "condition_encoder" not in n and "film_" not in n):
                self._set_backbone_trainable(True)
                for group in self.optimizer.param_groups:
                    group["lr"] = self.config.train.learning_rate * self.config.condition.joint_lr_factor

    def _set_backbone_trainable(self, trainable: bool):
        for name, param in self.model.named_parameters():
            if "condition_encoder" in name or "film_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = trainable

    def _load_pretrained_backbone(self, ckpt_path: str):
        """从 baseline checkpoint 加载 backbone 权重（跳过 FiLM / condition_encoder）"""
        self.logger.info(f"加载预训练 backbone: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        src_state = ckpt.get("model_state_dict", ckpt)

        model_state = self.model.state_dict()
        loaded, skipped = [], []
        for key, val in src_state.items():
            if "condition_encoder" in key or "film_" in key:
                skipped.append(key)
                continue
            if key in model_state and model_state[key].shape == val.shape:
                model_state[key] = val
                loaded.append(key)
            else:
                skipped.append(key)

        self.model.load_state_dict(model_state)
        self.logger.info(f"  已加载 {len(loaded)} 个参数, 跳过 {len(skipped)} 个")


# ============================================================================
# 主函数
# ============================================================================
@email_on_error()
def main():
    parser = argparse.ArgumentParser(description="MRI → PET Baseline Training")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=None, help="批大小")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的 checkpoint 路径")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--target_pet", type=str, choices=["tau", "fdg", "av45"], default=None, help="目标 PET: tau/fdg/av45")
    parser.add_argument("--condition_mode", type=str, default=None, help="条件模式: none/clinical/plasma/both")
    parser.add_argument("--cuda_visible_devices", type=str, default=None, help="指定可见 GPU，如 0,1,2,3")
    parser.add_argument("--pretrained_backbone", type=str, default=None, help="预训练 backbone 权重路径（baseline best_model.pth）")
    parser.add_argument("--early_stopping", type=int, default=None, help="早停 patience")
    parser.add_argument("--accumulation_steps", type=int, default=None, help="梯度累积步数")
    parser.add_argument("--val_freq", type=int, default=None, help="验证频率（每 N 个 epoch）")
    parser.add_argument("--debug_print", action="store_true", help="打印关键变量")
    parser.add_argument("--debug_batches", type=int, default=1, help="每个 epoch 打印的 batch 数")
    parser.add_argument("--split_subjects_json", type=str, default=None, help="固定划分 JSON（train_subjects/val_subjects/test_subjects）")
    parser.add_argument("--split_train_json", type=str, default=None, help="核心方法 train_data_with_description.json")
    parser.add_argument("--split_val_json", type=str, default=None, help="核心方法 val_data_with_description.json")
    parser.add_argument("--split_test_json", type=str, default=None, help="可选 test_data_with_description.json")
    parser.add_argument("--split_fallback_test_from_val", action="store_true", help="未提供 test 划分时使用 val 作为 test")
    parser.add_argument("--cache_dir", type=str, default=None, help="SSD 缓存目录，加速数据加载")
    args = parser.parse_args()
    
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

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
    if args.target_pet:
        config.data.target_pet = args.target_pet
    if args.condition_mode:
        config.condition.mode = args.condition_mode
    if args.pretrained_backbone:
        config.condition.pretrained_backbone = args.pretrained_backbone
    if args.cuda_visible_devices:
        config.cuda_visible_devices = args.cuda_visible_devices
    if args.split_subjects_json:
        config.data.split_subjects_json = args.split_subjects_json
    if args.split_train_json:
        config.data.split_train_json = args.split_train_json
    if args.split_val_json:
        config.data.split_val_json = args.split_val_json
    if args.split_test_json:
        config.data.split_test_json = args.split_test_json
    if args.split_fallback_test_from_val:
        config.data.split_fallback_test_from_val = True
    if args.early_stopping is not None:
        config.train.early_stopping_patience = args.early_stopping
    if args.accumulation_steps is not None:
        config.train.accumulation_steps = args.accumulation_steps
    if args.val_freq is not None:
        config.train.val_freq = args.val_freq
    if args.cache_dir:
        config.data.cache_dir = args.cache_dir
    
    # 创建训练器
    trainer = Trainer(config)
    trainer.debug_print = args.debug_print
    trainer.debug_batches = args.debug_batches
    
    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
