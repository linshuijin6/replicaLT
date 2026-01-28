"""
adapter_v2/train.py
===================
TAU 单示踪剂 CoCoOp 训练脚本

特性：
- Subject-based batch sampling
- Three-way contrastive loss
- TensorBoard logging
- 断点续训 & 保存 best checkpoint
- Recall@K 验证指标
"""

import os
import sys
import yaml
import random
import argparse
from datetime import datetime
from pathlib import Path
from report_error import email_on_error
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 添加 clip_mri2pet 到 path（用于 BiomedCLIP 等组件）
REPO_ROOT = Path(__file__).parent.parent.parent / "CLIP-MRI2PET"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import TAUPlasmaDataset, SubjectBatchSampler, collate_fn
from models import CoCoOpTAUModel
from losses import compute_total_loss, compute_recall_at_k
from precompute_cache import get_cache_stats, generate_missing_caches


# ============================================================================
# 工具函数
# ============================================================================

def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """加载 YAML 配置"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    save_path: str,
):
    """保存 checkpoint"""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
    }
    torch.save(state, save_path)
    print(f"Checkpoint saved: {save_path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    load_path: str,
):
    """加载 checkpoint"""
    if not os.path.exists(load_path):
        print(f"No checkpoint found at {load_path}")
        return 0, 0.0
    
    state = torch.load(load_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    epoch = state.get("epoch", 0)
    best_metric = state.get("best_metric", 0.0)
    print(f"Loaded checkpoint from epoch {epoch}, best_metric={best_metric:.4f}")
    return epoch, best_metric


# ============================================================================
# 训练 & 验证
# ============================================================================

def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: dict,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter = None,
):
    """
    训练一个 epoch
    
    返回：{'total': float, 'img_class': float, ...}
    """
    model.train()
    
    # loss 权重直接在 config["training"] 下
    train_cfg = config["training"]
    
    total_losses = {
        "total": 0.0,
        "img_class": 0.0,
        "img_plasma": 0.0,
        "class_plasma": 0.0,
        "reg": 0.0,
    }
    n_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)
    for batch_idx, batch in enumerate(pbar):
        # batch keys: subjects, patch_emb, img_emb, label_idx, plasma_vals, plasma_mask
        
        patch_emb = batch["patch_emb"].to(device)   # (B, N, 768) - patch tokens
        label_idx = batch["label_idx"].to(device)   # (B,)
        plasma_vals = batch["plasma_vals"].to(device)  # (B, K)
        plasma_mask = batch["plasma_mask"].to(device)  # (B, K)
        
        optimizer.zero_grad()
        
        # =====================================================================
        # Forward
        # =====================================================================
        outputs = model(
            tau_tokens=patch_emb,
            diagnosis_id=label_idx,
            plasma_values=plasma_vals,
            plasma_mask=plasma_mask,
        )
        # outputs keys: img_emb, class_emb, plasma_emb, plasma_weights, logit_scale
        
        # =====================================================================
        # Loss
        # =====================================================================
        loss_dict = compute_total_loss(
            img_emb=outputs["img_emb"],
            class_emb=outputs["class_emb"],
            plasma_emb=outputs["plasma_emb"],
            logit_scale=outputs["logit_scale"],
            plasma_mask=plasma_mask,
            lambda_img_class=train_cfg["lambda_img_class"],
            lambda_img_plasma=train_cfg["lambda_img_plasma"],
            lambda_class_plasma=train_cfg["lambda_class_plasma"],
            lambda_reg=train_cfg["lambda_reg"],
            reg_type=train_cfg.get("reg_type", "high_sim_penalty"),
            reg_cos_max=train_cfg.get("reg_cos_max", 0.8),
        )
        
        loss = loss_dict["total"]
        loss.backward()
        
        # Gradient clipping
        max_grad_norm = config["training"].get("max_grad_norm", 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        # Accumulate losses
        for k in total_losses:
            total_losses[k] += loss_dict[k].item()
        n_batches += 1
        
        # 更新进度条
        pbar.set_postfix({
            "loss": loss.item(),
            "L_ic": loss_dict["img_class"].item(),
            "L_ip": loss_dict["img_plasma"].item(),
        })
        
        # TensorBoard step logging
        if writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar("train/loss_total", loss.item(), global_step)
            writer.add_scalar("train/logit_scale", outputs["logit_scale"].item(), global_step)
    
    # 平均
    for k in total_losses:
        total_losses[k] /= max(n_batches, 1)
    
    return total_losses


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    config: dict,
    device: torch.device,
) -> dict:
    """
    验证：计算 Recall@K
    
    返回：{'recall@1': float, 'recall@5': float, 'recall@10': float, 'loss': float}
    """
    model.eval()
    train_cfg = config["training"]
    k_list = config.get("eval", {}).get("recall_k", [1, 5, 10])
    
    all_img_emb = []
    all_class_emb = []
    total_loss = 0.0
    n_batches = 0
    
    for batch in tqdm(dataloader, desc="Validate", leave=False):
        patch_emb = batch["patch_emb"].to(device)
        label_idx = batch["label_idx"].to(device)
        plasma_vals = batch["plasma_vals"].to(device)
        plasma_mask = batch["plasma_mask"].to(device)
        
        outputs = model(
            tau_tokens=patch_emb,
            diagnosis_id=label_idx,
            plasma_values=plasma_vals,
            plasma_mask=plasma_mask,
        )
        
        loss_dict = compute_total_loss(
            img_emb=outputs["img_emb"],
            class_emb=outputs["class_emb"],
            plasma_emb=outputs["plasma_emb"],
            logit_scale=outputs["logit_scale"],
            plasma_mask=plasma_mask,
            lambda_img_class=train_cfg["lambda_img_class"],
            lambda_img_plasma=train_cfg["lambda_img_plasma"],
            lambda_class_plasma=train_cfg["lambda_class_plasma"],
            lambda_reg=train_cfg["lambda_reg"],
        )
        
        total_loss += loss_dict["total"].item()
        n_batches += 1
        
        all_img_emb.append(outputs["img_emb"].cpu())
        all_class_emb.append(outputs["class_emb"].cpu())
    
    # 拼接
    all_img_emb = torch.cat(all_img_emb, dim=0)   # (N, D)
    all_class_emb = torch.cat(all_class_emb, dim=0)  # (N, D)
    
    # Recall@K (img → class)
    recall_dict = compute_recall_at_k(
        queries=all_img_emb,
        gallery=all_class_emb,
        k_list=k_list,
    )
    
    recall_dict["loss"] = total_loss / max(n_batches, 1)
    return recall_dict


# ============================================================================
# 主函数
# ============================================================================
@email_on_error()
def main():
    parser = argparse.ArgumentParser(description="TAU CoCoOp Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的 checkpoint 路径")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（覆盖配置）")
    parser.add_argument("--epochs", type=int, default=None, help="训练 epochs（覆盖配置）")
    args = parser.parse_args()
    
    # =========================================================================
    # 加载配置
    # =========================================================================
    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    config = load_config(config_path)
    
    # 覆盖参数
    seed = args.seed if args.seed is not None else config["training"].get("seed", 42)
    epochs = args.epochs if args.epochs is not None else config["training"]["epochs"]
    
    set_seed(seed)
    
    # =========================================================================
    # 设备（按 config 限定 GPU）
    # =========================================================================
    gpu_cfg = config["training"].get("gpu", None)
    if gpu_cfg is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_cfg)
        print(f"CUDA_VISIBLE_DEVICES set to: {gpu_cfg}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # =========================================================================
    # 数据集
    # =========================================================================
    csv_path = script_dir / config["data"]["csv_path"]
    cache_dir = Path(config["data"]["cache_dir"])
    adni_root = config.get("data", {}).get("adni_root", "/mnt/nfsdata/nfsdata/ADNI/ADNI0103/Coregistration")
    diagnosis_csv = config.get("data", {}).get("diagnosis_csv", None)
    diagnosis_code_map = config.get("classes", {}).get("diagnosis_code_map", None)
    class_names = config["classes"]["names"]  # ["CN", "MCI", "AD"]
    
    # =========================================================================
    # 缓存验证与补充生成
    # =========================================================================
    print("\n" + "="*60)
    print("缓存状态检查")
    print("="*60)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_stats = get_cache_stats(
        csv_path=str(csv_path),
        cache_dir=str(cache_dir),
    )
    print(f"总样本数: {cache_stats['total']}")
    print(f"已缓存: {cache_stats['cached']}")
    print(f"缺失: {cache_stats['missing']}")
    
    # 记录缺失缓存的样本（用于后续过滤）
    missing_cache_set = set()
    
    if cache_stats['missing'] > 0:
        print(f"\n检测到 {cache_stats['missing']} 个缺失缓存，开始补充生成...")
        
        generated, missing_nifti = generate_missing_caches(
            csv_path=str(csv_path),
            cache_dir=str(cache_dir),
            adni_root=adni_root,
            device="cuda" if torch.cuda.is_available() else "cpu",
            gpu=gpu_cfg,
        )
        
        print(f"补充生成完成: {generated} 个")
        if missing_nifti:
            print(f"警告: {len(missing_nifti)} 个样本缺少 NIfTI 文件，将从训练中排除")
        
        # 再次检查
        cache_stats = get_cache_stats(
            csv_path=str(csv_path),
            cache_dir=str(cache_dir),
        )
        print(f"\n缓存更新后: 已缓存 {cache_stats['cached']}, 仍缺失 {cache_stats['missing']}")
        
        if cache_stats['missing'] > 0:
            print(f"警告: 仍有 {cache_stats['missing']} 个缺失，这些样本将不会参与训练")
            # 输出缺失缓存列表
            missing_list = cache_stats.get('missing_list', [])
            print(f"\n缺失缓存列表 (共 {len(missing_list)} 个):")
            for ptid, tau_id in missing_list[:20]:  # 最多显示前 20 个
                print(f"  - {ptid}_{tau_id}.vision.pt")
            if len(missing_list) > 20:
                print(f"  ... 还有 {len(missing_list) - 20} 个未显示")
            # 记录到 set 用于过滤
            missing_cache_set = {(ptid, tau_id) for ptid, tau_id in missing_list}
    else:
        print("✅ 所有缓存均已存在")
    
    print()
    
    # 加载全部数据（过滤缺失缓存的样本）
    full_dataset = TAUPlasmaDataset(
        csv_path=str(csv_path),
        cache_dir=str(cache_dir),
        plasma_keys=config["plasma"]["keys"],
        class_names=class_names,
        diagnosis_csv=diagnosis_csv,
        diagnosis_code_map=diagnosis_code_map,
        skip_cache_set=missing_cache_set,
    )
    print(f"Total samples: {len(full_dataset)}")
    if missing_cache_set:
        print(f"(已排除 {len(missing_cache_set)} 个缺失缓存的样本)")
    
    # 划分训练/验证
    val_ratio = config["data"].get("val_ratio", 0.15)
    
    from dataset import split_by_subject
    train_indices, val_indices = split_by_subject(full_dataset, val_ratio=val_ratio, seed=seed)

    # 重新构建 train/val dataset（共享 plasma_stats，确保归一化一致）
    train_dataset = TAUPlasmaDataset(
        csv_path=str(csv_path),
        cache_dir=str(cache_dir),
        plasma_keys=config["plasma"]["keys"],
        class_names=class_names,
        plasma_stats=full_dataset.plasma_stats,
        diagnosis_csv=diagnosis_csv,
        diagnosis_code_map=diagnosis_code_map,
        subset_indices=train_indices,
        skip_cache_set=missing_cache_set,
    )
    val_dataset = TAUPlasmaDataset(
        csv_path=str(csv_path),
        cache_dir=str(cache_dir),
        plasma_keys=config["plasma"]["keys"],
        class_names=class_names,
        plasma_stats=full_dataset.plasma_stats,
        diagnosis_csv=diagnosis_csv,
        diagnosis_code_map=diagnosis_code_map,
        subset_indices=val_indices,
        skip_cache_set=missing_cache_set,
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # DataLoader
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 4)
    
    # 训练集使用 SubjectBatchSampler
    train_sampler = SubjectBatchSampler(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # 验证集顺序遍历
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # =========================================================================
    # 模型
    # =========================================================================
    # 从 config 读取类别和提示信息
    prompt_template = config["classes"]["prompt_template"]
    plasma_prompts = config["plasma"]["prompts"]
    
    model_cfg = config["model"]
    model = CoCoOpTAUModel(
        biomedclip_path=model_cfg["biomedclip_path"],
        class_names=class_names,
        class_prompt_template=prompt_template,
        plasma_prompts=plasma_prompts,
        ctx_len=model_cfg.get("ctx_len", 4),
        proj_dim=model_cfg.get("proj_dim", 512),
        ctx_hidden_dim=model_cfg.get("ctx_hidden_dim", 1024),
        share_ctx_base=model_cfg.get("share_ctx_base", False),
        plasma_temperature=config["plasma"].get("temperature", 1.0),
    )
    model = model.to(device)
    
    # 统计可训练参数
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,}")
    
    # =========================================================================
    # 优化器
    # =========================================================================
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )
    
    # =========================================================================
    # 断点恢复
    # =========================================================================
    start_epoch = 0
    best_recall = 0.0
    
    if args.resume:
        start_epoch, best_recall = load_checkpoint(model, optimizer, args.resume)
        start_epoch += 1  # 从下一个 epoch 开始
    
    # =========================================================================
    # TensorBoard
    # =========================================================================
    log_cfg = config["log"]
    log_dir = script_dir / log_cfg.get("dir", "./runs")
    run_name = f"tau_cocoop_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=str(log_dir / run_name))
    
    # 保存配置
    writer.add_text("config", yaml.dump(config, default_flow_style=False))
    
    # =========================================================================
    # 训练循环
    # =========================================================================
    ckpt_dir = log_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(start_epoch, epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*60}")
        
        # 训练
        train_losses = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            config=config,
            device=device,
            epoch=epoch,
            writer=writer,
        )
        print(f"Train - total: {train_losses['total']:.4f}, "
              f"img_class: {train_losses['img_class']:.4f}, "
              f"img_plasma: {train_losses['img_plasma']:.4f}")
        
        # 验证
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            config=config,
            device=device,
        )
        print(f"Val   - loss: {val_metrics['loss']:.4f}, "
              f"R@1: {val_metrics['recall@1']:.4f}, "
              f"R@5: {val_metrics['recall@5']:.4f}, "
              f"R@10: {val_metrics['recall@10']:.4f}")
        
        # TensorBoard epoch logging
        writer.add_scalar("train_epoch/loss", train_losses["total"], epoch)
        writer.add_scalar("train_epoch/loss_img_class", train_losses["img_class"], epoch)
        writer.add_scalar("train_epoch/loss_img_plasma", train_losses["img_plasma"], epoch)
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/recall@1", val_metrics["recall@1"], epoch)
        writer.add_scalar("val/recall@5", val_metrics["recall@5"], epoch)
        writer.add_scalar("val/recall@10", val_metrics["recall@10"], epoch)
        
        # 保存 checkpoint
        current_recall = val_metrics["recall@1"]
        save_every = config["log"].get("save_every", 5)
        
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, current_recall,
                str(ckpt_dir / f"epoch_{epoch+1:03d}.pt")
            )
        
        # 保存 best model
        if current_recall > best_recall:
            best_recall = current_recall
            save_checkpoint(
                model, optimizer, epoch, best_recall,
                str(ckpt_dir / "best.pt")
            )
            print(f"[Best] New best recall@1: {best_recall:.4f}")
    
    writer.close()
    print(f"\nTraining finished. Best recall@1: {best_recall:.4f}")


if __name__ == "__main__":
    main()
