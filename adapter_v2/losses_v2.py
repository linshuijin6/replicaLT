"""
adapter_v2/losses_v2.py
=======================
TAU 单示踪剂 CoCoOp V2 的损失函数

损失组成：
1. L_img_class: Image ↔ Class 对比损失
2. L_img_plasma: Image ↔ Plasma 对比损失（核心）
3. L_class_plasma: Class ↔ Plasma 对比损失
4. L_reg: Class/Plasma 正则（防止塌缩）
5. L_redundancy: Context Net 冗余约束（防止 T_ctx 复刻 image embedding）

新增验证指标：
- Image → Plasma Retrieval
- Plasma → Image Retrieval
"""

from typing import Dict, Optional, List
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
    
    logits = logit_scale * torch.matmul(a, b.t())
    targets = torch.arange(B, device=device)
    
    if mask is not None:
        valid_indices = torch.where(mask)[0]
        if len(valid_indices) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        logits_sub = logits[valid_indices][:, valid_indices]
        targets_sub = torch.arange(len(valid_indices), device=device)
        
        loss_a2b = F.cross_entropy(logits_sub, targets_sub)
        loss_b2a = F.cross_entropy(logits_sub.t(), targets_sub)
    else:
        loss_a2b = F.cross_entropy(logits, targets)
        loss_b2a = F.cross_entropy(logits.t(), targets)
    
    return 0.5 * (loss_a2b + loss_b2a)


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
    """
    cos_sim = (a * b).sum(dim=-1)
    penalty = F.relu(cos_sim - cos_max).pow(2)
    return penalty.mean()


def weak_orth_loss(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    弱正交损失：鼓励 a 和 b 正交
    """
    cos_sim = (a * b).sum(dim=-1)
    return cos_sim.pow(2).mean()


# ============================================================================
# 冗余约束损失
# ============================================================================

def redundancy_loss(
    ctx_img_sim: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """
    冗余约束损失：防止 Context Net 输出（T_ctx）与 image embedding 过于相似
    
    目标：discouraging T_ctx 复刻 image embedding，确保 T_ctx 只提供有限调整
    
    Args:
        ctx_img_sim: (B,) - T_ctx 与 img_emb 的余弦相似度
        margin: 最大允许相似度
        
    Returns:
        loss: () - 标量损失
    """
    # 惩罚超过 margin 的相似度
    penalty = F.relu(ctx_img_sim.abs() - margin).pow(2)
    return penalty.mean()


def ctx_norm_regularization(
    T_ctx: torch.Tensor,
    max_norm: float = 1.0,
) -> torch.Tensor:
    """
    Context 范数正则：限制 T_ctx 的 L2 范数
    
    确保 T_ctx 只能提供有限的 adjustment
    
    Args:
        T_ctx: (B, ctx_len, ctx_dim) - Context Net 输出
        max_norm: 最大允许范数
        
    Returns:
        loss: () - 标量损失
    """
    # 计算每个样本的 T_ctx L2 范数
    norms = T_ctx.view(T_ctx.shape[0], -1).norm(dim=-1)  # (B,)
    # 惩罚超过 max_norm 的部分
    penalty = F.relu(norms - max_norm).pow(2)
    return penalty.mean()


# ============================================================================
# 综合损失
# ============================================================================

def compute_total_loss_v2(
    img_emb: torch.Tensor,
    class_emb: torch.Tensor,
    plasma_emb: torch.Tensor,
    logit_scale: torch.Tensor,
    plasma_mask: torch.Tensor,
    ctx_img_sim: torch.Tensor,
    T_ctx: Optional[torch.Tensor] = None,
    lambda_img_class: float = 1.0,
    lambda_img_plasma: float = 1.0,  # 核心损失，权重提升
    lambda_class_plasma: float = 0.1,
    lambda_reg: float = 0.01,
    lambda_redundancy: float = 0.1,  # 冗余约束
    lambda_ctx_norm: float = 0.01,  # Context 范数正则
    reg_type: str = "high_sim_penalty",
    reg_cos_max: float = 0.8,
    redundancy_margin: float = 0.3,
    ctx_max_norm: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    计算 V2 总损失
    
    Args:
        img_emb: (B, D) - 图像 embedding
        class_emb: (B, D) - 类别 embedding
        plasma_emb: (B, D) - plasma embedding
        logit_scale: () - logit scale
        plasma_mask: (B, 5) - plasma 有效 mask
        ctx_img_sim: (B,) - T_ctx 与 img_emb 的相似度
        T_ctx: (B, ctx_len, ctx_dim) - Context Net 输出（用于范数正则）
        lambda_*: 各损失权重
        
    Returns:
        Dict with all loss components
    """
    device = img_emb.device
    
    # plasma 有效 mask（至少有一个有效值）
    plasma_valid = plasma_mask.any(dim=-1)
    
    # =========================================================================
    # L_img_class: Image ↔ Class
    # =========================================================================
    L_img_class = contrastive_loss(img_emb, class_emb, logit_scale)
    
    # =========================================================================
    # L_img_plasma: Image ↔ Plasma（核心损失）
    # =========================================================================
    if plasma_valid.any():
        L_img_plasma = contrastive_loss(
            img_emb, plasma_emb, logit_scale, mask=plasma_valid
        )
    else:
        L_img_plasma = torch.tensor(0.0, device=device, requires_grad=True)
    
    # =========================================================================
    # L_class_plasma: Class ↔ Plasma
    # =========================================================================
    if plasma_valid.any():
        L_class_plasma = contrastive_loss(
            class_emb, plasma_emb, logit_scale, mask=plasma_valid
        )
    else:
        L_class_plasma = torch.tensor(0.0, device=device, requires_grad=True)
    
    # =========================================================================
    # L_reg: Class/Plasma 正则
    # =========================================================================
    if reg_type == "weak_orth":
        L_reg = weak_orth_loss(class_emb, plasma_emb)
    else:
        L_reg = high_sim_penalty(class_emb, plasma_emb, cos_max=reg_cos_max)
    
    # =========================================================================
    # L_redundancy: Context Net 冗余约束
    # =========================================================================
    L_redundancy = redundancy_loss(ctx_img_sim, margin=redundancy_margin)
    
    # =========================================================================
    # L_ctx_norm: Context 范数正则
    # =========================================================================
    if T_ctx is not None:
        L_ctx_norm = ctx_norm_regularization(T_ctx, max_norm=ctx_max_norm)
    else:
        L_ctx_norm = torch.tensor(0.0, device=device, requires_grad=True)
    
    # =========================================================================
    # 总损失
    # =========================================================================
    total = (
        lambda_img_class * L_img_class
        + lambda_img_plasma * L_img_plasma
        + lambda_class_plasma * L_class_plasma
        + lambda_reg * L_reg
        + lambda_redundancy * L_redundancy
        + lambda_ctx_norm * L_ctx_norm
    )
    
    return {
        "total": total,
        "img_class": L_img_class,
        "img_plasma": L_img_plasma,
        "class_plasma": L_class_plasma,
        "reg": L_reg,
        "redundancy": L_redundancy,
        "ctx_norm": L_ctx_norm,
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
    k_list: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """
    计算 Recall@K（全量 gallery）
    
    假设 queries 和 gallery 一一对应（Q == G）
    正样本位置为对角线
    """
    Q = queries.shape[0]
    device = queries.device
    
    sims = torch.matmul(queries, gallery.t())  # (Q, G)
    
    results = {}
    for k in k_list:
        _, topk_indices = sims.topk(k, dim=1)
        targets = torch.arange(Q, device=device).unsqueeze(1)
        hits = (topk_indices == targets).any(dim=1).float()
        recall = hits.mean().item()
        results[f"recall@{k}"] = recall
    
    return results


@torch.no_grad()
def compute_img_plasma_retrieval(
    img_embs: torch.Tensor,
    plasma_embs: torch.Tensor,
    plasma_valid_mask: torch.Tensor,
    k_list: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """
    Image → Plasma Retrieval / Plasma → Image Retrieval
    
    核心验证指标：
    在整个 validation set 上用 image embedding 检索所有 plasma embeddings
    
    Args:
        img_embs: (N, D) - 所有样本的 image embedding
        plasma_embs: (N, D) - 所有样本的 plasma embedding
        plasma_valid_mask: (N,) - plasma 有效 mask（True=有效）
        k_list: Recall@K 的 K 值列表
        
    Returns:
        Dict:
            img2plasma_recall@1: float
            img2plasma_recall@5: float
            img2plasma_recall@10: float
            plasma2img_recall@1: float
            plasma2img_recall@5: float
            plasma2img_recall@10: float
    """
    device = img_embs.device
    
    # 只保留 plasma 有效的样本
    valid_indices = torch.where(plasma_valid_mask)[0]
    if len(valid_indices) == 0:
        # 无有效样本
        results = {}
        for k in k_list:
            results[f"img2plasma_recall@{k}"] = 0.0
            results[f"plasma2img_recall@{k}"] = 0.0
        return results
    
    img_valid = img_embs[valid_indices]  # (M, D)
    plasma_valid = plasma_embs[valid_indices]  # (M, D)
    M = img_valid.shape[0]
    
    # 相似度矩阵
    sims = torch.matmul(img_valid, plasma_valid.t())  # (M, M)
    
    results = {}
    
    # Image → Plasma
    for k in k_list:
        k_actual = min(k, M)
        _, topk_indices = sims.topk(k_actual, dim=1)  # (M, k)
        targets = torch.arange(M, device=device).unsqueeze(1)  # (M, 1)
        hits = (topk_indices == targets).any(dim=1).float()
        results[f"img2plasma_recall@{k}"] = hits.mean().item()
    
    # Plasma → Image
    sims_t = sims.t()  # (M, M)
    for k in k_list:
        k_actual = min(k, M)
        _, topk_indices = sims_t.topk(k_actual, dim=1)
        targets = torch.arange(M, device=device).unsqueeze(1)
        hits = (topk_indices == targets).any(dim=1).float()
        results[f"plasma2img_recall@{k}"] = hits.mean().item()
    
    return results
