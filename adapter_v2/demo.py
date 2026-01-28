"""
adapter_v2/demo.py
==================
调试脚本：验证数据流和模型输出

功能：
1. 加载单个 batch，打印所有张量形状
2. 运行模型前向，打印中间变量
3. 计算损失，打印各分量
"""

import os
import sys
import yaml
from pathlib import Path

import torch

# 添加 clip_mri2pet 到 path
REPO_ROOT = Path(__file__).parent.parent.parent / "CLIP-MRI2PET"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import TAUPlasmaDataset, SubjectBatchSampler, collate_fn
from models import CoCoOpTAUModel
from losses import compute_total_loss


def main():
    script_dir = Path(__file__).parent
    
    # =========================================================================
    # 加载配置
    # =========================================================================
    config_path = script_dir / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    print("=" * 70)
    print("DEMO: 调试数据流和模型输出")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] 使用设备: {device}")
    
    # =========================================================================
    # 1. 加载数据集
    # =========================================================================
    print("\n" + "-" * 50)
    print("[STEP 1] 加载数据集")
    print("-" * 50)
    
    csv_path = script_dir / config["data"]["csv_path"]
    cache_dir = Path(config["data"]["cache_dir"])
    
    dataset = TAUPlasmaDataset(
        csv_path=str(csv_path),
        cache_dir=str(cache_dir),
        plasma_keys=config["plasma"]["keys"],
    )
    print(f"数据集大小: {len(dataset)}")
    
    # =========================================================================
    # 2. 获取单个样本
    # =========================================================================
    print("\n" + "-" * 50)
    print("[STEP 2] 单个样本测试")
    print("-" * 50)
    
    sample = dataset[0]
    print(f"sample keys: {list(sample.keys())}")
    print(f"  subject_id: {sample['subject_id']}")
    print(f"  tau_cls shape: {sample['tau_cls'].shape}")      # (512,)
    print(f"  tau_tokens shape: {sample['tau_tokens'].shape}")  # (N, 768)
    print(f"  diagnosis_id: {sample['diagnosis_id']}")
    print(f"  plasma_values: {sample['plasma_values']}")          # (K,)
    print(f"  plasma_mask: {sample['plasma_mask']}")          # (K,)
    
    # =========================================================================
    # 3. Batch 测试
    # =========================================================================
    print("\n" + "-" * 50)
    print("[STEP 3] Batch 数据测试")
    print("-" * 50)
    
    from torch.utils.data import DataLoader
    
    batch_size = min(4, len(dataset))
    sampler = SubjectBatchSampler(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)
    
    batch = next(iter(loader))
    
    print(f"batch keys: {list(batch.keys())}")
    print(f"  img_emb: {batch['img_emb'].shape}")        # (B, 512)
    print(f"  patch_emb: {batch['patch_emb'].shape}")    # (B, N, 768)
    print(f"  label_idx: {batch['label_idx'].shape}")    # (B,)
    print(f"  plasma_vals: {batch['plasma_vals'].shape}") # (B, K)
    print(f"  plasma_mask: {batch['plasma_mask'].shape}") # (B, K)
    print(f"  subjects: {batch['subjects']}")
    
    # =========================================================================
    # 4. 模型初始化
    # =========================================================================
    print("\n" + "-" * 50)
    print("[STEP 4] 模型初始化")
    print("-" * 50)
    
    # 从 config 读取
    class_names = config["classes"]["names"]  # ["CN", "MCI", "AD"]
    prompt_template = config["classes"]["prompt_template"]
    plasma_prompts = config["plasma"]["prompts"]
    
    # 打印提示信息
    print(f"Class names ({len(class_names)}): {class_names}")
    print(f"Class prompt template: {prompt_template}")
    print(f"Plasma prompts ({len(plasma_prompts)}):")
    for i, p in enumerate(plasma_prompts):
        print(f"  [{i}] {p}")
    
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
    model.eval()
    
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {n_trainable:,} / 总参数: {n_total:,}")
    
    # 打印模型结构（可训练部分）
    print("\n可训练模块:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {tuple(param.shape)}")
    
    # =========================================================================
    # 5. 前向传播
    # =========================================================================
    print("\n" + "-" * 50)
    print("[STEP 5] 前向传播")
    print("-" * 50)
    
    patch_emb = batch["patch_emb"].to(device)    # (B, N, 768)
    label_idx = batch["label_idx"].to(device)    # (B,)
    plasma_vals = batch["plasma_vals"].to(device)  # (B, K)
    plasma_mask = batch["plasma_mask"].to(device)  # (B, K)
    
    with torch.no_grad():
        outputs = model(
            tau_tokens=patch_emb,
            diagnosis_id=label_idx,
            plasma_values=plasma_vals,
            plasma_mask=plasma_mask,
        )
    
    print("模型输出:")
    print(f"  img_emb shape: {outputs['img_emb'].shape}")      # (B, 512)
    print(f"  class_emb shape: {outputs['class_emb'].shape}")  # (B, 512)
    print(f"  plasma_emb shape: {outputs['plasma_emb'].shape}") # (B, 512)
    print(f"  plasma_weights shape: {outputs['plasma_weights'].shape}")  # (B, K)
    print(f"  logit_scale: {outputs['logit_scale'].item():.4f}")
    
    print("\n采样输出值:")
    print(f"  img_emb[0, :5]: {outputs['img_emb'][0, :5].tolist()}")
    print(f"  class_emb[0, :5]: {outputs['class_emb'][0, :5].tolist()}")
    print(f"  plasma_emb[0, :5]: {outputs['plasma_emb'][0, :5].tolist()}")
    print(f"  plasma_weights[0]: {outputs['plasma_weights'][0].tolist()}")
    
    # 检查 L2 norm
    print("\nL2 范数检验（应接近 1.0）:")
    print(f"  img_emb L2 norm: {outputs['img_emb'][0].norm().item():.6f}")
    print(f"  class_emb L2 norm: {outputs['class_emb'][0].norm().item():.6f}")
    print(f"  plasma_emb L2 norm: {outputs['plasma_emb'][0].norm().item():.6f}")
    
    # =========================================================================
    # 6. 损失计算
    # =========================================================================
    print("\n" + "-" * 50)
    print("[STEP 6] 损失计算")
    print("-" * 50)
    
    train_cfg = config["training"]
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
    
    print("损失分解:")
    print(f"  L_total: {loss_dict['total'].item():.6f}")
    print(f"  L_img_class: {loss_dict['img_class'].item():.6f}")
    print(f"  L_img_plasma: {loss_dict['img_plasma'].item():.6f}")
    print(f"  L_class_plasma: {loss_dict['class_plasma'].item():.6f}")
    print(f"  L_reg: {loss_dict['reg'].item():.6f}")
    
    # =========================================================================
    # 7. 相似度矩阵
    # =========================================================================
    print("\n" + "-" * 50)
    print("[STEP 7] 相似度矩阵")
    print("-" * 50)
    
    with torch.no_grad():
        scale = outputs["logit_scale"]
        sim_img_class = scale * torch.matmul(outputs["img_emb"], outputs["class_emb"].t())
        sim_img_plasma = scale * torch.matmul(outputs["img_emb"], outputs["plasma_emb"].t())
    
    print("img ↔ class 相似度矩阵:")
    for i in range(min(4, sim_img_class.shape[0])):
        row = [f"{v:.2f}" for v in sim_img_class[i].tolist()[:4]]
        print(f"  row {i}: {row}")
    
    print("\nimg ↔ plasma 相似度矩阵:")
    for i in range(min(4, sim_img_plasma.shape[0])):
        row = [f"{v:.2f}" for v in sim_img_plasma[i].tolist()[:4]]
        print(f"  row {i}: {row}")
    
    print("\n" + "=" * 70)
    print("DEMO 完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
