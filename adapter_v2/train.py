"""
adapter_v2/train.py
===================
TAU 单示踪剂 CoCoOp 训练脚本

特性：
- Subject-based batch sampling
- Three-way contrastive loss
- TensorBoard logging
- 断点续训 & 保存 best checkpoint
- 类别原型级 Recall@K 验证指标
"""

import os
import sys
import json
import yaml
import random
import argparse
from datetime import datetime
from pathlib import Path
from report_error import email_on_error
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    from sklearn.metrics import balanced_accuracy_score as sk_balanced_accuracy_score
    from sklearn.metrics import f1_score as sk_f1_score
    from sklearn.linear_model import LogisticRegression
except ImportError:
    sk_balanced_accuracy_score = None
    sk_f1_score = None
    LogisticRegression = None

# 添加 clip_mri2pet 到 path（用于 BiomedCLIP 等组件）
REPO_ROOT = Path(__file__).parent.parent.parent / "CLIP-MRI2PET"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import TAUPlasmaDataset, SubjectBatchSampler, collate_fn, split_by_subject
from models import CoCoOpTAUModel
from losses import compute_total_loss
from precompute_cache import get_cache_stats, generate_missing_caches


DEFAULT_PLASMA_KEYS = ["AB42_AB40_F", "pT217_F", "pT217_AB42_F", "NfL_Q", "GFAP_Q"]


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


def _normalize_plasma_key(key: str) -> str:
    """将 plasma key 归一化用于大小写兼容匹配。"""
    return str(key).strip().lower()


def resolve_plasma_config(config: dict) -> tuple[list, list]:
    """
    解析并校验 plasma 相关配置。

    支持：
    - available_keys / selected_keys / key_aliases / prompts_by_key（推荐）
    - keys + prompts（向后兼容）

    Returns
    -------
    selected_keys, plasma_prompts : (list[str], list[str])
        规范化后的 key 列表与按 key 对齐的 prompt 列表
    """
    plasma_cfg = config.get("plasma", {})

    available_keys = plasma_cfg.get("available_keys", plasma_cfg.get("keys", DEFAULT_PLASMA_KEYS))
    if isinstance(available_keys, str):
        available_keys = [available_keys]
    if not isinstance(available_keys, list) or len(available_keys) == 0:
        raise ValueError("plasma.available_keys/keys 不能为空")

    selected_keys = plasma_cfg.get("selected_keys", plasma_cfg.get("keys", available_keys))
    if isinstance(selected_keys, str):
        selected_keys = [selected_keys]
    if not isinstance(selected_keys, list) or len(selected_keys) == 0:
        raise ValueError("plasma.selected_keys/keys 不能为空")

    normalized_available = {}
    for key in available_keys:
        norm_key = _normalize_plasma_key(key)
        if norm_key not in normalized_available:
            normalized_available[norm_key] = key

    alias_cfg = plasma_cfg.get("key_aliases", {}) or {}
    alias_map = {}
    for alias, target in alias_cfg.items():
        alias_map[_normalize_plasma_key(alias)] = _normalize_plasma_key(target)

    resolved_keys = []
    seen = set()
    for key in selected_keys:
        norm_key = _normalize_plasma_key(key)
        norm_key = alias_map.get(norm_key, norm_key)

        if norm_key not in normalized_available:
            choices = ", ".join(available_keys)
            raise ValueError(
                f"未知 plasma key: {key}. 可选项为: [{choices}]"
            )

        canonical_key = normalized_available[norm_key]
        if canonical_key not in seen:
            resolved_keys.append(canonical_key)
            seen.add(canonical_key)

    prompts_by_key = plasma_cfg.get("prompts_by_key", None)
    if prompts_by_key is not None:
        normalized_prompts = {
            _normalize_plasma_key(k): v for k, v in prompts_by_key.items()
        }
        plasma_prompts = []
        missing_prompt_keys = []
        for key in resolved_keys:
            norm_key = _normalize_plasma_key(key)
            if norm_key not in normalized_prompts:
                missing_prompt_keys.append(key)
            else:
                plasma_prompts.append(normalized_prompts[norm_key])
        if missing_prompt_keys:
            raise ValueError(
                "plasma.prompts_by_key 缺少以下 key 的提示词: "
                + ", ".join(missing_prompt_keys)
            )
    else:
        legacy_prompts = plasma_cfg.get("prompts", None)
        if legacy_prompts is None:
            raise ValueError(
                "缺少 plasma.prompts_by_key（或旧版 plasma.prompts）。"
            )
        if len(legacy_prompts) != len(resolved_keys):
            raise ValueError(
                f"plasma.prompts 长度({len(legacy_prompts)})与 selected_keys 长度({len(resolved_keys)})不一致"
            )
        plasma_prompts = legacy_prompts

    if len(resolved_keys) != len(plasma_prompts):
        raise ValueError("plasma keys 与 prompts 数量不一致")

    return resolved_keys, plasma_prompts


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


def _compute_bal_acc_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> tuple[float, float]:
    """计算 Balanced Accuracy 与 Macro-F1（优先 sklearn，缺失时 fallback）。"""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size == 0:
        return float("nan"), float("nan")

    labels = list(range(num_classes))

    if sk_balanced_accuracy_score is not None and sk_f1_score is not None:
        bal_acc = float(sk_balanced_accuracy_score(y_true, y_pred))
        macro_f1 = float(sk_f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        return bal_acc, macro_f1

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    recalls = []
    f1s = []
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp

        denom_recall = tp + fn
        recall = float(tp / denom_recall) if denom_recall > 0 else 0.0
        recalls.append(recall)

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        if precision + recall > 0:
            f1s.append(2.0 * precision * recall / (precision + recall))
        else:
            f1s.append(0.0)

    return float(np.mean(recalls)), float(np.mean(f1s))


def _fit_linear_probe_on_val(
    img_emb: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    seed: int,
    config: dict,
) -> tuple[float, float]:
    """在线线性探针：仅在 val 特征上拟合并评估，不更新主模型。"""
    valid_mask = (labels >= 0) & (labels < num_classes)
    if valid_mask.sum().item() == 0:
        return float("nan"), float("nan")

    x = img_emb[valid_mask].detach().cpu().numpy().astype(np.float32)
    y = labels[valid_mask].detach().cpu().numpy().astype(np.int64)

    if np.unique(y).size < 2:
        return float("nan"), float("nan")

    if LogisticRegression is not None:
        clf = LogisticRegression(
            random_state=seed,
            max_iter=int(config.get("eval", {}).get("probe_max_iter", 1000)),
            multi_class="auto",
            class_weight="balanced",
        )
        clf.fit(x, y)
        pred = clf.predict(x)
        return _compute_bal_acc_macro_f1(y, pred, num_classes)

    fallback_cfg = config.get("eval", {})
    fallback_epochs = int(fallback_cfg.get("probe_fallback_epochs", 200))
    fallback_lr = float(fallback_cfg.get("probe_fallback_lr", 1e-2))
    fallback_weight_decay = float(fallback_cfg.get("probe_fallback_weight_decay", 1e-4))

    probe = nn.Linear(x.shape[1], num_classes)
    optimizer = optim.AdamW(
        probe.parameters(),
        lr=fallback_lr,
        weight_decay=fallback_weight_decay,
    )

    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y)

    with torch.enable_grad():
        probe.train()
        for _ in range(fallback_epochs):
            optimizer.zero_grad()
            logits = probe(x_t)
            loss = F.cross_entropy(logits, y_t)
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        pred = probe(x_t).argmax(dim=1).cpu().numpy()

    return _compute_bal_acc_macro_f1(y, pred, num_classes)


def _nan_mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=0))


def _sample_derangement(n: int, generator: torch.Generator, max_tries: int = 64) -> torch.Tensor | None:
    if n < 2:
        return None
    base = torch.arange(n, dtype=torch.long)
    for _ in range(max_tries):
        perm = torch.randperm(n, generator=generator)
        if torch.all(perm != base):
            return perm
    return None


def _build_within_class_perm(labels: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    n = labels.shape[0]
    perm = torch.full((n,), -1, dtype=torch.long)
    valid_mask = torch.zeros(n, dtype=torch.bool)

    for cls in torch.unique(labels):
        cls_idx = torch.where(labels == cls)[0]
        if cls_idx.numel() < 2:
            continue
        local_perm = _sample_derangement(int(cls_idx.numel()), generator)
        if local_perm is None:
            continue
        mapped = cls_idx[local_perm]
        perm[cls_idx] = mapped
        valid_mask[cls_idx] = True

    return perm, valid_mask


def _build_cross_class_perm(labels: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    n = labels.shape[0]
    if n < 2:
        return torch.full((n,), -1, dtype=torch.long), torch.zeros(n, dtype=torch.bool)

    deranged = _sample_derangement(n, generator)
    if deranged is None:
        return torch.full((n,), -1, dtype=torch.long), torch.zeros(n, dtype=torch.bool)

    valid_mask = labels != labels[deranged]
    perm = torch.full((n,), -1, dtype=torch.long)
    perm[valid_mask] = deranged[valid_mask]
    return perm, valid_mask


def _pairwise_diag_cosine_mean(img_emb: torch.Tensor, plasma_emb: torch.Tensor) -> float:
    if img_emb.shape[0] == 0:
        return float("nan")
    img_n = F.normalize(img_emb, dim=1)
    plasma_n = F.normalize(plasma_emb, dim=1)
    return float((img_n * plasma_n).sum(dim=1).mean().item())


def _mean_margin(img_emb: torch.Tensor, plasma_emb: torch.Tensor) -> float:
    n = img_emb.shape[0]
    if n < 2:
        return float("nan")
    img_n = F.normalize(img_emb, dim=1)
    plasma_n = F.normalize(plasma_emb, dim=1)
    sim = img_n @ plasma_n.t()
    diag = sim.diag()
    offdiag_mean = (sim.sum(dim=1) - diag) / (n - 1)
    margin = diag - offdiag_mean
    return float(margin.mean().item())


def _plasma_shuffle_drops(
    img_emb: torch.Tensor,
    plasma_emb: torch.Tensor,
    labels: torch.Tensor,
    mode: str,
    repeats: int,
    generator: torch.Generator,
) -> tuple[float, float, float, float, int]:
    score_drops = []
    margin_drops = []
    valid_repeat_count = 0

    for _ in range(repeats):
        if mode == "within":
            perm, valid_mask = _build_within_class_perm(labels, generator)
        elif mode == "cross":
            perm, valid_mask = _build_cross_class_perm(labels, generator)
        else:
            raise ValueError(f"Unknown shuffle mode: {mode}")

        valid_idx = torch.where(valid_mask)[0]
        if valid_idx.numel() == 0:
            score_drops.append(float("nan"))
            margin_drops.append(float("nan"))
            continue

        paired_idx = perm[valid_idx]
        base_img = img_emb[valid_idx]
        base_plasma = plasma_emb[valid_idx]
        shuf_plasma = plasma_emb[paired_idx]

        base_score = _pairwise_diag_cosine_mean(base_img, base_plasma)
        shuf_score = _pairwise_diag_cosine_mean(base_img, shuf_plasma)
        score_drops.append(base_score - shuf_score)

        base_margin = _mean_margin(base_img, base_plasma)
        shuf_margin = _mean_margin(base_img, shuf_plasma)
        margin_drops.append(base_margin - shuf_margin)
        valid_repeat_count += 1

    score_mean, score_std = _nan_mean_std(score_drops)
    margin_mean, margin_std = _nan_mean_std(margin_drops)
    return score_mean, score_std, margin_mean, margin_std, valid_repeat_count


def _is_significant_drop(mean_value: float, std_value: float, effect_thr: float, stability_thr: float) -> bool:
    if not np.isfinite(mean_value):
        return False
    if mean_value < effect_thr:
        return False
    if not np.isfinite(std_value):
        return False
    return (mean_value / (std_value + 1e-8)) >= stability_thr


def _build_plasma_shuffle_diagnosis(
    within_score_mean: float,
    within_score_std: float,
    cross_score_mean: float,
    cross_score_std: float,
    within_margin_mean: float,
    within_margin_std: float,
    cross_margin_mean: float,
    cross_margin_std: float,
    within_valid_repeats: int,
    cross_valid_repeats: int,
    repeats: int,
    effect_thr: float,
    stability_thr: float,
    min_valid_repeats: int,
) -> str:
    def _pick_effect(score_mean, score_std, margin_mean, margin_std):
        if np.isfinite(margin_mean):
            return margin_mean, margin_std, "margin"
        return score_mean, score_std, "score"

    within_mean, within_std, within_metric = _pick_effect(
        within_score_mean, within_score_std, within_margin_mean, within_margin_std
    )
    cross_mean, cross_std, cross_metric = _pick_effect(
        cross_score_mean, cross_score_std, cross_margin_mean, cross_margin_std
    )

    if within_valid_repeats < min_valid_repeats or cross_valid_repeats < min_valid_repeats:
        return (
            f"insufficient_data: within_valid={within_valid_repeats}/{repeats}, "
            f"cross_valid={cross_valid_repeats}/{repeats}, "
            f"within({within_metric})={within_mean:.4f}±{within_std:.4f}, "
            f"cross({cross_metric})={cross_mean:.4f}±{cross_std:.4f}"
        )

    within_sig = _is_significant_drop(within_mean, within_std, effect_thr, stability_thr)
    cross_sig = _is_significant_drop(cross_mean, cross_std, effect_thr, stability_thr)

    if cross_sig and not within_sig:
        conclusion = "plasma主要复述class语义"
    elif within_sig:
        conclusion = "plasma学到类内对齐信息（理想情况）"
    else:
        conclusion = "plasma分支未被有效利用或对齐失败"

    return (
        f"{conclusion}; within({within_metric})={within_mean:.4f}±{within_std:.4f}, "
        f"cross({cross_metric})={cross_mean:.4f}±{cross_std:.4f}, "
        f"valid_repeats={within_valid_repeats}/{repeats},{cross_valid_repeats}/{repeats}"
    )


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
    expected_plasma_dim: int | None = None,
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

        if expected_plasma_dim is not None:
            if plasma_vals.shape[-1] != expected_plasma_dim:
                raise ValueError(
                    f"plasma_vals 维度不匹配: got {plasma_vals.shape[-1]}, expected {expected_plasma_dim}"
                )
            if plasma_mask.shape[-1] != expected_plasma_dim:
                raise ValueError(
                    f"plasma_mask 维度不匹配: got {plasma_mask.shape[-1]}, expected {expected_plasma_dim}"
                )
        
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
        # outputs keys: img_emb, class_emb, class_emb_all, plasma_emb, plasma_z_used, logit_scale
        
        # =====================================================================
        # Loss
        # =====================================================================
        loss_dict = compute_total_loss(
            img_emb=outputs["img_emb"],
            class_emb=outputs["class_emb"],
            class_emb_all=outputs["class_emb_all"],
            label_idx=label_idx,
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
    expected_plasma_dim: int | None = None,
) -> dict:
    """
    验证：
    1) image 线性探针分类（Balanced Acc / Macro-F1）
    2) context 注入式 class_text 分类（Balanced Acc / Macro-F1）
    3) plasma shuffle 反事实对齐分析（score/margin drop）
    """
    model.eval()
    train_cfg = config["training"]
    eval_cfg = config.get("eval", {})
    repeats = int(eval_cfg.get("plasma_shuffle_repeats", 20))
    effect_thr = float(eval_cfg.get("plasma_drop_effect_threshold", 0.02))
    stability_thr = float(eval_cfg.get("plasma_drop_stability_threshold", 1.0))
    min_valid_repeats = int(eval_cfg.get("plasma_min_valid_repeats", 3))
    rng_seed = int(config["training"].get("seed", 42))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(rng_seed)
    
    all_logits = []
    all_targets = []
    all_img_emb = []
    all_plasma_emb = []
    
    for batch in tqdm(dataloader, desc="Validate", leave=False):
        patch_emb = batch["patch_emb"].to(device)
        label_idx = batch["label_idx"].to(device)
        plasma_vals = batch["plasma_vals"].to(device)
        plasma_mask = batch["plasma_mask"].to(device)

        if expected_plasma_dim is not None:
            if plasma_vals.shape[-1] != expected_plasma_dim:
                raise ValueError(
                    f"[val] plasma_vals 维度不匹配: got {plasma_vals.shape[-1]}, expected {expected_plasma_dim}"
                )
            if plasma_mask.shape[-1] != expected_plasma_dim:
                raise ValueError(
                    f"[val] plasma_mask 维度不匹配: got {plasma_mask.shape[-1]}, expected {expected_plasma_dim}"
                )
        
        outputs = model(
            tau_tokens=patch_emb,
            diagnosis_id=label_idx,
            plasma_values=plasma_vals,
            plasma_mask=plasma_mask,
        )

        # 类别原型 logits: (B, C=3)
        class_logits = outputs["logit_scale"] * torch.einsum(
            "bd,bcd->bc", outputs["img_emb"], outputs["class_emb_all"]
        )
        all_logits.append(class_logits.cpu())
        all_targets.append(label_idx.cpu())
        all_img_emb.append(outputs["img_emb"].cpu())
        all_plasma_emb.append(outputs["plasma_emb"].cpu())

    if len(all_logits) == 0:
        return {
            "probe_bal_acc": float("nan"),
            "probe_macro_f1": float("nan"),
            "inject_bal_acc": float("nan"),
            "inject_macro_f1": float("nan"),
            "plasma_score_drop_within_mean": float("nan"),
            "plasma_score_drop_within_std": float("nan"),
            "plasma_score_drop_cross_mean": float("nan"),
            "plasma_score_drop_cross_std": float("nan"),
            "plasma_margin_drop_within_mean": float("nan"),
            "plasma_margin_drop_within_std": float("nan"),
            "plasma_margin_drop_cross_mean": float("nan"),
            "plasma_margin_drop_cross_std": float("nan"),
            "plasma_shuffle_diagnosis": "insufficient_data: empty validation outputs",
        }

    all_logits = torch.cat(all_logits, dim=0)   # (N, C)
    all_targets = torch.cat(all_targets, dim=0)  # (N,)
    all_img_emb = torch.cat(all_img_emb, dim=0)
    all_plasma_emb = torch.cat(all_plasma_emb, dim=0)

    num_classes = all_logits.shape[1]
    valid_target_mask = (all_targets >= 0) & (all_targets < num_classes)
    valid_targets = all_targets[valid_target_mask]
    valid_logits = all_logits[valid_target_mask]
    valid_img_emb = all_img_emb[valid_target_mask]
    valid_plasma_emb = all_plasma_emb[valid_target_mask]

    if valid_targets.numel() == 0:
        return {
            "probe_bal_acc": float("nan"),
            "probe_macro_f1": float("nan"),
            "inject_bal_acc": float("nan"),
            "inject_macro_f1": float("nan"),
            "plasma_score_drop_within_mean": float("nan"),
            "plasma_score_drop_within_std": float("nan"),
            "plasma_score_drop_cross_mean": float("nan"),
            "plasma_score_drop_cross_std": float("nan"),
            "plasma_margin_drop_within_mean": float("nan"),
            "plasma_margin_drop_within_std": float("nan"),
            "plasma_margin_drop_cross_mean": float("nan"),
            "plasma_margin_drop_cross_std": float("nan"),
            "plasma_shuffle_diagnosis": "insufficient_data: no valid diagnosis label in validation set",
        }

    y_true = valid_targets.numpy().astype(np.int64)

    probe_bal_acc, probe_macro_f1 = _fit_linear_probe_on_val(
        img_emb=all_img_emb,
        labels=all_targets,
        num_classes=num_classes,
        seed=rng_seed,
        config=config,
    )

    inject_pred = valid_logits.argmax(dim=1).numpy().astype(np.int64)
    inject_bal_acc, inject_macro_f1 = _compute_bal_acc_macro_f1(
        y_true=y_true,
        y_pred=inject_pred,
        num_classes=num_classes,
    )

    within_score_mean, within_score_std, within_margin_mean, within_margin_std, within_valid_repeats = _plasma_shuffle_drops(
        img_emb=valid_img_emb,
        plasma_emb=valid_plasma_emb,
        labels=valid_targets,
        mode="within",
        repeats=repeats,
        generator=generator,
    )
    cross_score_mean, cross_score_std, cross_margin_mean, cross_margin_std, cross_valid_repeats = _plasma_shuffle_drops(
        img_emb=valid_img_emb,
        plasma_emb=valid_plasma_emb,
        labels=valid_targets,
        mode="cross",
        repeats=repeats,
        generator=generator,
    )

    diagnosis = _build_plasma_shuffle_diagnosis(
        within_score_mean=within_score_mean,
        within_score_std=within_score_std,
        cross_score_mean=cross_score_mean,
        cross_score_std=cross_score_std,
        within_margin_mean=within_margin_mean,
        within_margin_std=within_margin_std,
        cross_margin_mean=cross_margin_mean,
        cross_margin_std=cross_margin_std,
        within_valid_repeats=within_valid_repeats,
        cross_valid_repeats=cross_valid_repeats,
        repeats=repeats,
        effect_thr=effect_thr,
        stability_thr=stability_thr,
        min_valid_repeats=min_valid_repeats,
    )

    return {
        "probe_bal_acc": probe_bal_acc,
        "probe_macro_f1": probe_macro_f1,
        "inject_bal_acc": inject_bal_acc,
        "inject_macro_f1": inject_macro_f1,
        "plasma_score_drop_within_mean": within_score_mean,
        "plasma_score_drop_within_std": within_score_std,
        "plasma_score_drop_cross_mean": cross_score_mean,
        "plasma_score_drop_cross_std": cross_score_std,
        "plasma_margin_drop_within_mean": within_margin_mean,
        "plasma_margin_drop_within_std": within_margin_std,
        "plasma_margin_drop_cross_mean": cross_margin_mean,
        "plasma_margin_drop_cross_std": cross_margin_std,
        "plasma_shuffle_diagnosis": diagnosis,
    }


# ============================================================================
# 固定 Split 工具函数
# ============================================================================
def apply_fixed_split(
    full_dataset: TAUPlasmaDataset,
    train_indices: list,
    val_indices: list,
    val_split_json: str,
    seed: int,
    val_ratio: float,
) -> tuple:
    """
    应用或生成固定的 train/val 划分 JSON。

    与 run_probe_and_retrieval.py 中 build_data_splits 使用完全相同的 JSON 格式，
    确保两个脚本可互相读取同一个 fixed_split.json。

    Parameters
    ----------
    full_dataset : TAUPlasmaDataset
        完整数据集（用于获取 subject_id）
    train_indices, val_indices : list
        由 split_by_subject 生成的初始随机划分
    val_split_json : str
        固定划分 JSON 路径
    seed : int
        随机种子（仅写入 JSON 元信息）
    val_ratio : float
        验证比例（仅写入 JSON 元信息）

    Returns
    -------
    train_indices, val_indices : (list, list)
        最终使用的划分索引
    """
    split_path = Path(val_split_json)

    # 构建 subject -> indices 映射
    subject_to_indices: dict = {}
    for idx, sample in enumerate(full_dataset.samples):
        sid = sample["subject_id"]
        subject_to_indices.setdefault(sid, []).append(idx)

    if split_path.exists():
        with open(split_path, "r") as f:
            payload = json.load(f)
        val_subjects = set(payload.get("val_subjects", []))
        train_subjects = set(payload.get("train_subjects", []))

        if not val_subjects and not train_subjects:
            raise ValueError(
                f"无效 split JSON（缺少 train_subjects/val_subjects）: {split_path}"
            )
        if not val_subjects:
            val_subjects = set(subject_to_indices.keys()) - train_subjects
        if not train_subjects:
            train_subjects = set(subject_to_indices.keys()) - val_subjects

        unknown = (train_subjects | val_subjects) - set(subject_to_indices.keys())
        if unknown:
            print(f"[Split] WARN: split JSON 包含当前数据集中不存在的 subject: "
                  f"{sorted(list(unknown))[:5]}...")

        train_indices, val_indices = [], []
        for sid, indices in subject_to_indices.items():
            if sid in val_subjects:
                val_indices.extend(indices)
            elif sid in train_subjects:
                train_indices.extend(indices)

        if len(train_indices) == 0 or len(val_indices) == 0:
            raise ValueError(
                f"split JSON 应用后出现空集合: train={len(train_indices)}, val={len(val_indices)}"
            )
        print(f"[Split] Loaded fixed split from: {split_path}")
        print(f"[Split] Subjects train={len(train_subjects)} val={len(val_subjects)}")
    else:
        # 保存当前随机划分，供后续评估复用
        _train_subjects = sorted({full_dataset.samples[i]["subject_id"] for i in train_indices})
        _val_subjects = sorted({full_dataset.samples[i]["subject_id"] for i in val_indices})
        split_path.parent.mkdir(parents=True, exist_ok=True)
        with open(split_path, "w") as f:
            json.dump(
                {
                    "seed": seed,
                    "val_ratio": val_ratio,
                    "train_subjects": _train_subjects,
                    "val_subjects": _val_subjects,
                },
                f,
                indent=2,
            )
        print(f"[Split] Saved generated split to: {split_path}")

    return train_indices, val_indices


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
    parser.add_argument("--val_split_json", type=str, default="fixed_split.json",
                        help="固定 train/val 划分 JSON（包含 train_subjects/val_subjects）；"
                             "不存在时自动生成并保存，与 run_probe_and_retrieval.py 共享格式")
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

    # 解析 plasma 配置：支持 selected_keys + prompts_by_key，兼容大小写
    selected_plasma_keys, plasma_prompts = resolve_plasma_config(config)
    config.setdefault("plasma", {})["keys"] = selected_plasma_keys
    print(f"[Plasma] Selected keys ({len(selected_plasma_keys)}): {selected_plasma_keys}")

    print("[Plasma] weighting strategy=concat(text_feature, z_score) + shared projection + masked mean")
    
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
        plasma_keys=selected_plasma_keys,
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
    
    train_indices, val_indices = split_by_subject(full_dataset, val_ratio=val_ratio, seed=seed)

    # ── 固定 split 支持（与 run_probe_and_retrieval.py 共享同一 JSON）──────
    if args.val_split_json is not None:
        train_indices, val_indices = apply_fixed_split(
            full_dataset=full_dataset,
            train_indices=train_indices,
            val_indices=val_indices,
            val_split_json=args.val_split_json,
            seed=seed,
            val_ratio=val_ratio,
        )

    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")

    # 重新构建 train/val dataset（共享 plasma_stats，确保归一化一致）
    train_dataset = TAUPlasmaDataset(
        csv_path=str(csv_path),
        cache_dir=str(cache_dir),
        plasma_keys=selected_plasma_keys,
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
        plasma_keys=selected_plasma_keys,
        class_names=class_names,
        plasma_stats=full_dataset.plasma_stats,
        diagnosis_csv=diagnosis_csv,
        diagnosis_code_map=diagnosis_code_map,
        subset_indices=val_indices,
        skip_cache_set=missing_cache_set,
    )
    print(f"Train dataset: {len(train_dataset)}, Val dataset: {len(val_dataset)}")
    
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

    if len(plasma_prompts) != len(selected_plasma_keys):
        raise ValueError(
            f"plasma prompts 数量({len(plasma_prompts)})与 selected_keys 数量({len(selected_plasma_keys)})不一致"
        )
    
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
    best_metric = 0.0
    
    if args.resume:
        start_epoch, best_metric = load_checkpoint(model, optimizer, args.resume)
        start_epoch += 1  # 从下一个 epoch 开始
    
    # =========================================================================
    # TensorBoard
    # =========================================================================
    log_cfg = config["log"]
    log_dir = script_dir / log_cfg.get("dir", "./runs")
    run_name = f"{datetime.now().strftime('%m.%d')}_{os.getpid()}"
    run_dir = log_dir / run_name
    writer = SummaryWriter(log_dir=str(run_dir))
    
    # 保存配置
    writer.add_text("config", yaml.dump(config, default_flow_style=False))
    
    # =========================================================================
    # 训练循环
    # =========================================================================
    ckpt_dir = run_dir / "ckpt"
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
            expected_plasma_dim=len(selected_plasma_keys),
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
            expected_plasma_dim=len(selected_plasma_keys),
        )

        print(
            "Val   - "
            f"probe(BAcc/F1): {val_metrics['probe_bal_acc']:.4f}/{val_metrics['probe_macro_f1']:.4f}, "
            f"inject(BAcc/F1): {val_metrics['inject_bal_acc']:.4f}/{val_metrics['inject_macro_f1']:.4f}, "
            f"within_margin_drop: {val_metrics['plasma_margin_drop_within_mean']:.4f}±{val_metrics['plasma_margin_drop_within_std']:.4f}, "
            f"cross_margin_drop: {val_metrics['plasma_margin_drop_cross_mean']:.4f}±{val_metrics['plasma_margin_drop_cross_std']:.4f}"
        )
        print(f"Val   - plasma diagnosis: {val_metrics['plasma_shuffle_diagnosis']}")
        
        # TensorBoard epoch logging
        writer.add_scalar("train_epoch/loss", train_losses["total"], epoch)
        writer.add_scalar("train_epoch/loss_img_class", train_losses["img_class"], epoch)
        writer.add_scalar("train_epoch/loss_img_plasma", train_losses["img_plasma"], epoch)
        writer.add_scalar("val/probe_bal_acc", val_metrics["probe_bal_acc"], epoch)
        writer.add_scalar("val/probe_macro_f1", val_metrics["probe_macro_f1"], epoch)
        writer.add_scalar("val/inject_bal_acc", val_metrics["inject_bal_acc"], epoch)
        writer.add_scalar("val/inject_macro_f1", val_metrics["inject_macro_f1"], epoch)
        writer.add_scalar("val/plasma_score_drop_within_mean", val_metrics["plasma_score_drop_within_mean"], epoch)
        writer.add_scalar("val/plasma_score_drop_cross_mean", val_metrics["plasma_score_drop_cross_mean"], epoch)
        writer.add_scalar("val/plasma_margin_drop_within_mean", val_metrics["plasma_margin_drop_within_mean"], epoch)
        writer.add_scalar("val/plasma_margin_drop_cross_mean", val_metrics["plasma_margin_drop_cross_mean"], epoch)
        
        # 保存 checkpoint
        current_metric = val_metrics.get("inject_macro_f1", 0.0)
        save_every = config["log"].get("save_every", 5)
        
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, current_metric,
                str(ckpt_dir / f"epoch_{epoch+1:03d}.pt")
            )
        
        # 保存 best model
        if current_metric > best_metric:
            best_metric = current_metric
            save_checkpoint(
                model, optimizer, epoch, best_metric,
                str(ckpt_dir / "best.pt")
            )
            print(f"[Best] New best inject_macro_f1: {best_metric:.4f}")
    
    writer.close()
    print(f"\nTraining finished. Best inject_macro_f1: {best_metric:.4f}")


if __name__ == "__main__":
    main()
