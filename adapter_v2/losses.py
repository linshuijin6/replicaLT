"""
adapter_v2/losses.py
====================
TAU 单示踪剂 CoCoOp 的损失函数

损失组成：
1. L_img_class: Image ↔ Class 对比损失
2. L_img_plasma: Image ↔ Plasma 对比损失
3. L_class_plasma: Class ↔ Plasma 对比损失（权重小）
4. L_reg: Class/Plasma 正则（防止塌缩）
"""

from typing import Dict, Optional
import torch
import torch.nn.functional as F


# ============================================================================
# 对比损失（InfoNCE / CLIP style）
# ============================================================================

def contrastive_loss(
    a: torch.Tensor,
    b: torch.Tensor,
    logit_scale: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    双向对比损失（CLIP-style InfoNCE）
    
    Args:
        a: (B, D) - 第一组 embedding（已 L2 归一化）
        b: (B, D) - 第二组 embedding（已 L2 归一化）
        logit_scale: () - 可学习的 logit scale（exp 后的值）
        mask: (B,) - 可选，True 表示该样本参与计算
        
    Returns:
        loss: () - 标量损失
    """
    B = a.shape[0]
    device = a.device
    
    # 计算相似度矩阵
    # logits: (B, B) - a @ b^T * scale
    logits = logit_scale * torch.matmul(a, b.t())
    
    # 目标：对角线（正样本）
    targets = torch.arange(B, device=device)
    
    if mask is not None:
        # 只计算有效样本
        valid_indices = torch.where(mask)[0]
        if len(valid_indices) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # 子矩阵
        logits_sub = logits[valid_indices][:, valid_indices]
        targets_sub = torch.arange(len(valid_indices), device=device)
        
        loss_a2b = F.cross_entropy(logits_sub, targets_sub)
        loss_b2a = F.cross_entropy(logits_sub.t(), targets_sub)
    else:
        loss_a2b = F.cross_entropy(logits, targets)
        loss_b2a = F.cross_entropy(logits.t(), targets)
    
    return 0.5 * (loss_a2b + loss_b2a)


def class_prototype_loss(
    img_emb: torch.Tensor,
    class_emb_all: torch.Tensor,
    label_idx: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """
    类别原型级图文对齐损失（每样本对 C 个类别原型做 softmax）。

    Args:
        img_emb: (B, D) - 图像 embedding（已 L2 归一化）
        class_emb_all: (B, C, D) - 每样本的 C 个类别原型 embedding（已 L2 归一化）
        label_idx: (B,) - 类别标签，范围 [0, C-1]，无效标签可为 -1
        logit_scale: () - 可学习的 logit scale（exp 后）

    Returns:
        loss: () - 标量损失
    """
    if class_emb_all.dim() != 3:
        raise ValueError(
            f"class_emb_all 维度错误: expect (B, C, D), got {tuple(class_emb_all.shape)}"
        )

    B, C, D = class_emb_all.shape
    if img_emb.shape[0] != B or img_emb.shape[1] != D:
        raise ValueError(
            "img_emb 与 class_emb_all 形状不匹配: "
            f"img_emb={tuple(img_emb.shape)}, class_emb_all={tuple(class_emb_all.shape)}"
        )
    if label_idx.shape[0] != B:
        raise ValueError(
            f"label_idx 长度不匹配: label_idx={tuple(label_idx.shape)}, B={B}"
        )

    valid_mask = (label_idx >= 0) & (label_idx < C)
    if not valid_mask.any():
        return torch.tensor(0.0, device=img_emb.device, requires_grad=True)

    img_valid = img_emb[valid_mask]
    class_valid = class_emb_all[valid_mask]
    target_valid = label_idx[valid_mask].long()

    # logits: (B_valid, C)
    logits = logit_scale * torch.einsum("bd,bcd->bc", img_valid, class_valid)
    return F.cross_entropy(logits, target_valid)


# ============================================================================
# 正则损失
# ============================================================================

def high_sim_penalty(
    a: torch.Tensor,
    b: torch.Tensor,
    cos_max: float = 0.8,
) -> torch.Tensor:
    """
    高相似度惩罚：防止 class 和 plasma embedding 过于相似
    
    当 cos(a, b) > cos_max 时施加二次惩罚
    
    Args:
        a: (B, D) - 第一组 embedding（已 L2 归一化）
        b: (B, D) - 第二组 embedding（已 L2 归一化）
        cos_max: 最大允许余弦相似度
        
    Returns:
        loss: () - 标量损失
    """
    # 逐样本计算余弦相似度
    # cos_sim: (B,)
    cos_sim = (a * b).sum(dim=-1)
    
    # 超过 cos_max 的部分施加惩罚
    # penalty: (B,)
    penalty = F.relu(cos_sim - cos_max).pow(2)
    
    return penalty.mean()


def weak_orth_loss(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    弱正交损失：鼓励 a 和 b 正交
    
    直接最小化 cos^2(a, b)
    
    Args:
        a: (B, D) - 第一组 embedding（已 L2 归一化）
        b: (B, D) - 第二组 embedding（已 L2 归一化）
        
    Returns:
        loss: () - 标量损失
    """
    cos_sim = (a * b).sum(dim=-1)
    return cos_sim.pow(2).mean()


# ============================================================================
# 综合损失
# ============================================================================

def compute_total_loss(
    img_emb: torch.Tensor,
    class_emb: torch.Tensor,
    plasma_emb: torch.Tensor,
    logit_scale: torch.Tensor,
    plasma_mask: torch.Tensor,
    class_emb_all: Optional[torch.Tensor] = None,
    label_idx: Optional[torch.Tensor] = None,
    lambda_img_class: float = 1.0,
    lambda_img_plasma: float = 0.5,
    lambda_class_plasma: float = 0.1,
    lambda_reg: float = 0.01,
    reg_type: str = "high_sim_penalty",
    reg_cos_max: float = 0.8,
) -> Dict[str, torch.Tensor]:
    """
    计算总损失
    
    Args:
        img_emb: (B, D) - 图像 embedding（已 L2 归一化）
        class_emb: (B, D) - 类别 embedding（已 L2 归一化）
        plasma_emb: (B, D) - plasma embedding（已 L2 归一化）
        logit_scale: () - 可学习的 logit scale
        plasma_mask: (B, 5) - plasma 有效 mask
        class_emb_all: (B, C, D) - 每样本类别原型 embedding（可选）
        label_idx: (B,) - 类别标签（可选）
        lambda_*: 各损失权重
        reg_type: 正则类型 "high_sim_penalty" | "weak_orth"
        reg_cos_max: 用于 high_sim_penalty 的阈值
        
    Returns:
        Dict:
            total: 总损失
            img_class: L_img_class
            img_plasma: L_img_plasma
            class_plasma: L_class_plasma
            reg: L_reg
    """
    device = img_emb.device
    
    # 判断 plasma 是否有效（至少有一个有效值）
    # plasma_valid: (B,)
    plasma_valid = plasma_mask.any(dim=-1)
    
    # =========================================================================
    # L_img_class: Image ↔ Class
    # =========================================================================
    if class_emb_all is not None and label_idx is not None:
        L_img_class = class_prototype_loss(
            img_emb=img_emb,
            class_emb_all=class_emb_all,
            label_idx=label_idx,
            logit_scale=logit_scale,
        )
    else:
        # 兼容旧调用：回退到 batch 实例级 CLIP-style
        L_img_class = contrastive_loss(img_emb, class_emb, logit_scale)
    
    # =========================================================================
    # L_img_plasma: Image ↔ Plasma（只计算 plasma 有效的样本）
    # =========================================================================
    if plasma_valid.any():
        L_img_plasma = contrastive_loss(
            img_emb, plasma_emb, logit_scale, mask=plasma_valid
        )
    else:
        L_img_plasma = torch.tensor(0.0, device=device, requires_grad=True)
    
    # =========================================================================
    # L_class_plasma: Class ↔ Plasma（只计算 plasma 有效的样本）
    # =========================================================================
    if plasma_valid.any():
        L_class_plasma = contrastive_loss(
            class_emb, plasma_emb, logit_scale, mask=plasma_valid
        )
    else:
        L_class_plasma = torch.tensor(0.0, device=device, requires_grad=True)
    
    # =========================================================================
    # L_reg: 正则损失
    # =========================================================================
    if reg_type == "weak_orth":
        L_reg = weak_orth_loss(class_emb, plasma_emb)
    else:  # high_sim_penalty
        L_reg = high_sim_penalty(class_emb, plasma_emb, cos_max=reg_cos_max)
    
    # =========================================================================
    # 总损失
    # =========================================================================
    total = (
        lambda_img_class * L_img_class
        + lambda_img_plasma * L_img_plasma
        + lambda_class_plasma * L_class_plasma
        + lambda_reg * L_reg
    )
    
    return {
        "total": total,
        "img_class": L_img_class,
        "img_plasma": L_img_plasma,
        "class_plasma": L_class_plasma,
        "reg": L_reg,
    }


# ============================================================================
# 评测指标
# ============================================================================

@torch.no_grad()
def compute_retrieval_accuracy(
    a: torch.Tensor,
    b: torch.Tensor,
    logit_scale: torch.Tensor,
) -> Dict[str, float]:
    """
    计算检索准确率（batch 内）
    
    Args:
        a: (B, D) - query embedding
        b: (B, D) - gallery embedding
        logit_scale: () - logit scale
        
    Returns:
        Dict:
            acc_a2b: a → b 检索准确率
            acc_b2a: b → a 检索准确率
    """
    B = a.shape[0]
    device = a.device
    
    logits = logit_scale * torch.matmul(a, b.t())
    targets = torch.arange(B, device=device)
    
    pred_a2b = logits.argmax(dim=1)
    pred_b2a = logits.argmax(dim=0)
    
    acc_a2b = (pred_a2b == targets).float().mean().item()
    acc_b2a = (pred_b2a == targets).float().mean().item()
    
    return {"acc_a2b": acc_a2b, "acc_b2a": acc_b2a}


@torch.no_grad()
def compute_recall_at_k(
    queries: torch.Tensor,
    gallery: torch.Tensor,
    k_list: list = [1, 5, 10],
) -> Dict[str, float]:
    """
    计算 Recall@K（全量 gallery）
    
    Args:
        queries: (Q, D) - query embeddings
        gallery: (G, D) - gallery embeddings
        k_list: K 值列表
        
    Returns:
        Dict: {"recall@1": float, "recall@5": float, ...}
    """
    Q = queries.shape[0]
    device = queries.device
    
    # 假设 queries 和 gallery 一一对应（Q == G）
    # 正样本位置为对角线
    sims = torch.matmul(queries, gallery.t())  # (Q, G)
    
    results = {}
    for k in k_list:
        # 对每个 query 取 top-k 相似的 gallery 索引
        _, topk_indices = sims.topk(k, dim=1)  # (Q, k)
        # 正样本索引
        targets = torch.arange(Q, device=device).unsqueeze(1)  # (Q, 1)
        # 检查 target 是否在 top-k 中
        hits = (topk_indices == targets).any(dim=1).float()
        recall = hits.mean().item()
        results[f"recall@{k}"] = recall
    
    return results
