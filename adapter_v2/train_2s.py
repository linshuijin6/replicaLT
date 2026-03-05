"""
adapter_v2/train_2s.py
======================
TAU 两阶段训练脚本（最小侵入）

阶段定义：
- Stage-1：仅训练 img↔class（+可选 reg），关闭 plasma 对比损失
- Stage-2：从 Stage-1 best 初始化，引入单一 plasma key（默认 pT217_F），
           在同 class 内做 K=3 分桶的 image->prototype CE

说明：
- 尽量复用 train.py 中的数据、验证与配置工具函数。
- 不改动 train.py 原训练逻辑。
"""

import os
import math
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from report_error import email_on_error

from dataset import TAUPlasmaDataset, SubjectBatchSampler, collate_fn, split_by_subject
from models import CoCoOpTAUModel
from losses import compute_total_loss

from train import (
    set_seed,
    load_config,
    resolve_plasma_config,
    validate,
    apply_fixed_split,
    get_cache_stats,
    generate_missing_caches,
)


def _normalize_key(key: str) -> str:
    return str(key).strip().lower()


def _resolve_stage2_single_key_and_prompt(config: dict, plasma_feature_name: str) -> tuple[str, str]:
    """
    解析 Stage-2 的单 key 与 prompt。

    约束：
    - Stage-2 必须单一 key（默认 pT217_F）
    - 若配置中缺该 key，抛清晰错误

    """
    plasma_cfg = config.get("plasma", {})
    target_norm = _normalize_key(plasma_feature_name)

    available_keys = plasma_cfg.get("available_keys", plasma_cfg.get("keys", []))
    if isinstance(available_keys, str):
        available_keys = [available_keys]

    key_aliases = plasma_cfg.get("key_aliases", {}) or {}
    alias_map = {_normalize_key(k): _normalize_key(v) for k, v in key_aliases.items()}
    target_norm = alias_map.get(target_norm, target_norm)

    canonical_key = None
    for key in available_keys:
        if _normalize_key(key) == target_norm:
            canonical_key = key
            break

    if canonical_key is None:
        raise ValueError(
            f"Stage-2 指定 plasma key '{plasma_feature_name}' 不在 config.plasma.available_keys/keys 中。"
            f"请检查配置字段映射后重试。"
        )

    prompts_by_key = plasma_cfg.get("prompts_by_key", None)
    if prompts_by_key is not None:
        prompt_map = {_normalize_key(k): v for k, v in prompts_by_key.items()}
        if target_norm not in prompt_map:
            raise ValueError(
                f"config.plasma.prompts_by_key 缺少 key '{canonical_key}' 的 prompt。"
            )
        return canonical_key, prompt_map[target_norm]

    # 兼容旧版 prompts 列表
    legacy_prompts = plasma_cfg.get("prompts", None)
    if legacy_prompts is None:
        raise ValueError(
            "缺少 config.plasma.prompts_by_key（或旧版 prompts），无法构建 Stage-2 单 key prompt。"
        )

    if len(available_keys) != len(legacy_prompts):
        raise ValueError(
            "旧版 plasma.prompts 长度与 available_keys/keys 长度不一致，无法定位 Stage-2 单 key prompt。"
        )

    key_to_prompt = {_normalize_key(k): p for k, p in zip(available_keys, legacy_prompts)}
    if target_norm not in key_to_prompt:
        raise ValueError(
            f"无法在旧版 prompts 中定位 key '{canonical_key}' 的 prompt。"
        )
    return canonical_key, key_to_prompt[target_norm]


def save_checkpoint_2s(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    stage: str,
    stage_epoch: int,
    best_metric: float,
    save_path: str,
    extra_state: dict | None = None,
):
    state = {
        "stage": stage,
        "stage_epoch": stage_epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra_state:
        state.update(extra_state)
    torch.save(state, save_path)
    print(f"Checkpoint saved: {save_path}")


def load_checkpoint_2s(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    load_path: str,
) -> tuple[str, int, float, dict]:
    if not os.path.exists(load_path):
        print(f"No checkpoint found at {load_path}")
        return "stage1", 0, 0.0, {}

    state = torch.load(load_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    if "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    stage = state.get("stage", "stage1")
    stage_epoch = int(state.get("stage_epoch", 0))
    best_metric = float(state.get("best_metric", 0.0))

    print(
        f"Loaded checkpoint: stage={stage}, stage_epoch={stage_epoch}, best_metric={best_metric:.4f}"
    )
    extra_state = {
        "stage2_memory_bank": state.get("stage2_memory_bank", None),
        "stage2_bank_momentum": state.get("stage2_bank_momentum", None),
    }
    return stage, stage_epoch, best_metric, extra_state


def _build_common_dataset(
    config: dict,
    csv_path: Path,
    cache_dir: Path,
    class_names: list[str],
    selected_keys: list[str],
    missing_cache_set: set,
    diagnosis_csv: str | None,
    diagnosis_code_map: dict | None,
) -> TAUPlasmaDataset:
    return TAUPlasmaDataset(
        csv_path=str(csv_path),
        cache_dir=str(cache_dir),
        plasma_keys=selected_keys,
        class_names=class_names,
        diagnosis_csv=diagnosis_csv,
        diagnosis_code_map=diagnosis_code_map,
        skip_cache_set=missing_cache_set,
    )


def _build_split_datasets(
    config: dict,
    full_dataset: TAUPlasmaDataset,
    train_indices: list[int],
    val_indices: list[int],
    csv_path: Path,
    cache_dir: Path,
    class_names: list[str],
    selected_keys: list[str],
    missing_cache_set: set,
    diagnosis_csv: str | None,
    diagnosis_code_map: dict | None,
) -> tuple[TAUPlasmaDataset, TAUPlasmaDataset]:
    train_dataset = TAUPlasmaDataset(
        csv_path=str(csv_path),
        cache_dir=str(cache_dir),
        plasma_keys=selected_keys,
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
        plasma_keys=selected_keys,
        class_names=class_names,
        plasma_stats=full_dataset.plasma_stats,
        diagnosis_csv=diagnosis_csv,
        diagnosis_code_map=diagnosis_code_map,
        subset_indices=val_indices,
        skip_cache_set=missing_cache_set,
    )
    return train_dataset, val_dataset


def _build_dataloaders(
    train_dataset: TAUPlasmaDataset,
    val_dataset: TAUPlasmaDataset,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
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

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader


def build_optimizer_for_stage2(
    model: torch.nn.Module,
    base_lr: float,
    weight_decay: float,
    image_lr_mult: float = 0.1,
    freeze_logit_scale: bool = True,
) -> torch.optim.Optimizer:
    """
    Stage-2 参数策略：
    - 冻结 class 核心：proj_class / base_ctx_class / (可选)logit_scale
    - 训练 plasma 分支：proj_plasma / base_ctx_plasma
    - image 侧可训练：token_pool / proj_img / context_net，lr 乘 image_lr_mult
    """
    for p in model.parameters():
        p.requires_grad = False

    # class 核心冻结
    for p in model.proj_class.parameters():
        p.requires_grad = False
    if isinstance(model.base_ctx_class, torch.nn.Parameter):
        model.base_ctx_class.requires_grad = False
    if freeze_logit_scale and hasattr(model, "logit_scale"):
        model.logit_scale.requires_grad = False

    # plasma 分支训练
    for p in model.proj_plasma.parameters():
        p.requires_grad = True

    if isinstance(model.base_ctx_plasma, torch.nn.Parameter):
        model.base_ctx_plasma.requires_grad = True

    # image 侧训练
    for module in [model.token_pool, model.proj_img, model.context_net]:
        for p in module.parameters():
            p.requires_grad = True

    # share_ctx_base=True 时，base_ctx_class/base_ctx_plasma 可能是同一参数
    if isinstance(model.base_ctx_plasma, torch.nn.Parameter) and isinstance(model.base_ctx_class, torch.nn.Parameter):
        if model.base_ctx_plasma.data_ptr() == model.base_ctx_class.data_ptr():
            model.base_ctx_class.requires_grad = True
            print("[Stage-2] WARN: base_ctx_class 与 base_ctx_plasma 共享参数，已保持可训练以支持 plasma 分支。")

    plasma_params = []
    image_params = []

    for _, p in model.proj_plasma.named_parameters():
        if p.requires_grad:
            plasma_params.append(p)

    if isinstance(model.base_ctx_plasma, torch.nn.Parameter) and model.base_ctx_plasma.requires_grad:
        plasma_params.append(model.base_ctx_plasma)

    for module in [model.token_pool, model.proj_img, model.context_net]:
        for _, p in module.named_parameters():
            if p.requires_grad:
                image_params.append(p)

    param_groups = []
    if plasma_params:
        param_groups.append({
            "params": plasma_params,
            "lr": base_lr,
            "weight_decay": weight_decay,
        })
    if image_params:
        param_groups.append({
            "params": image_params,
            "lr": base_lr * image_lr_mult,
            "weight_decay": weight_decay,
        })

    if not param_groups:
        raise ValueError("Stage-2 没有可训练参数，请检查冻结策略。")

    return optim.AdamW(param_groups)


def build_optimizer_for_stage1(
    model: torch.nn.Module,
    base_lr: float,
    weight_decay: float,
    freeze_logit_scale: bool = False,
) -> torch.optim.Optimizer:
    for p in model.parameters():
        p.requires_grad = True

    for p in model.proj_plasma.parameters():
        p.requires_grad = False

    if isinstance(model.base_ctx_plasma, torch.nn.Parameter):
        model.base_ctx_plasma.requires_grad = False

    if freeze_logit_scale and hasattr(model, "logit_scale"):
        model.logit_scale.requires_grad = False

    if isinstance(model.base_ctx_plasma, torch.nn.Parameter) and isinstance(model.base_ctx_class, torch.nn.Parameter):
        if model.base_ctx_plasma.data_ptr() == model.base_ctx_class.data_ptr():
            model.base_ctx_plasma.requires_grad = True
            print("[Stage-1] WARN: base_ctx_plasma 与 base_ctx_class 共享参数，保持该参数可训练以避免误冻结 class ctx。")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("Stage-1 没有可训练参数，请检查冻结策略。")

    return optim.AdamW(trainable_params, lr=base_lr, weight_decay=weight_decay)


def _compute_stage2_w_plasma(global_step: int, total_steps: int, warmup_frac: float, plasma_weight_max: float) -> float:
    if total_steps <= 0:
        return 0.0
    warmup_frac = max(0.0, float(warmup_frac))
    plasma_weight_max = float(plasma_weight_max)
    if warmup_frac <= 0.0:
        return plasma_weight_max

    warmup_steps = max(1, int(math.ceil(total_steps * warmup_frac)))
    progress = min(1.0, float(global_step) / float(warmup_steps))
    return plasma_weight_max * progress


def _sample_to_class_idx(sample: dict, class_to_idx: dict[str, int]) -> int:
    diagnosis = sample.get("diagnosis", None)
    if diagnosis is None:
        return -1
    return int(class_to_idx.get(diagnosis, -1))


def compute_classwise_plasma_thresholds(
    train_dataset: TAUPlasmaDataset,
    class_names: list[str],
    plasma_key: str,
) -> dict[int, dict[str, float]]:
    """
    仅使用训练集 samples 中原始 plasma_values/plasma_mask 与 diagnosis 计算阈值。
    返回每个 class 的 33%/66% 分位点。
    """
    if plasma_key not in train_dataset.plasma_keys:
        raise ValueError(
            f"plasma_key={plasma_key} 不在 train_dataset.plasma_keys={train_dataset.plasma_keys}"
        )

    key_idx = train_dataset.plasma_keys.index(plasma_key)
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    values_by_class: dict[int, list[float]] = {i: [] for i in range(len(class_names))}

    for sample in train_dataset.samples:
        cls = _sample_to_class_idx(sample, class_to_idx)
        if cls < 0:
            continue

        mask = bool(sample.get("plasma_mask", [False])[key_idx])
        if not mask:
            continue

        raw_v = float(sample.get("plasma_values", [0.0])[key_idx])
        values_by_class[cls].append(raw_v)

    thresholds = {}
    for cls_idx, vals in values_by_class.items():
        if len(vals) < 3:
            thresholds[cls_idx] = {
                "q33": float("nan"),
                "q66": float("nan"),
                "n": float(len(vals)),
            }
            continue

        arr = np.asarray(vals, dtype=np.float64)
        q33, q66 = np.quantile(arr, [1.0 / 3.0, 2.0 / 3.0])
        thresholds[cls_idx] = {
            "q33": float(q33),
            "q66": float(q66),
            "n": float(len(vals)),
        }

    return thresholds


def build_subject_bin_map_from_train_dataset(
    train_dataset: TAUPlasmaDataset,
    class_names: list[str],
    plasma_key: str,
    thresholds: dict[int, dict[str, float]],
) -> dict[str, int]:
    """
    根据训练集原始 plasma 值映射 subject -> bin_id in {0,1,2}。

    TODO(精确映射): 目前 batch 仅暴露 subject_id，若同一 subject 有多条记录且 bin 不同，
    这里使用众数 bin；若需逐样本精确映射，建议在 dataset/collate 中额外返回 sample 索引。
    """
    if plasma_key not in train_dataset.plasma_keys:
        raise ValueError(
            f"plasma_key={plasma_key} 不在 train_dataset.plasma_keys={train_dataset.plasma_keys}"
        )

    key_idx = train_dataset.plasma_keys.index(plasma_key)
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    subject_bins: dict[str, list[int]] = {}
    for sample in train_dataset.samples:
        sid = str(sample.get("subject_id"))
        cls = _sample_to_class_idx(sample, class_to_idx)
        if cls < 0:
            continue

        mask = bool(sample.get("plasma_mask", [False])[key_idx])
        if not mask:
            continue

        thr = thresholds.get(cls, {"q33": float("nan"), "q66": float("nan")})
        q33 = float(thr.get("q33", float("nan")))
        q66 = float(thr.get("q66", float("nan")))
        if not (np.isfinite(q33) and np.isfinite(q66)):
            continue

        v = float(sample.get("plasma_values", [0.0])[key_idx])
        if v <= q33:
            bin_id = 0
        elif v <= q66:
            bin_id = 1
        else:
            bin_id = 2

        subject_bins.setdefault(sid, []).append(bin_id)

    subject_bin_map: dict[str, int] = {}
    for sid, bins in subject_bins.items():
        if len(bins) == 0:
            continue
        binc = np.bincount(np.asarray(bins, dtype=np.int64), minlength=3)
        subject_bin_map[sid] = int(np.argmax(binc))

    return subject_bin_map


def compute_stage2_plasma_bin_loss(
    img_emb: torch.Tensor,
    plasma_emb: torch.Tensor,
    label_idx: torch.Tensor,
    batch_bin_idx: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Stage-2 plasma loss：
    - 在 batch 内按 class+bin 构建 plasma prototype（plasma_emb 均值）
    - 若某 class 的任一 bin 无样本，则该 class 样本不参与
    - 对有效样本做 image -> 3-bin plasma prototype 的 3-way softmax CE，目标为自身 bin
    """
    device = img_emb.device
    n = img_emb.shape[0]
    if n == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    valid_global = (label_idx >= 0) & (batch_bin_idx >= 0) & (batch_bin_idx <= 2)
    if not valid_global.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    total_loss = []
    classes = torch.unique(label_idx[valid_global])
    for cls in classes.tolist():
        cls_mask = valid_global & (label_idx == int(cls))
        cls_indices = torch.where(cls_mask)[0]
        if cls_indices.numel() == 0:
            continue

        prototypes = []
        has_all_bins = True
        for b in range(3):
            b_mask = cls_mask & (batch_bin_idx == b)
            b_idx = torch.where(b_mask)[0]
            if b_idx.numel() == 0:
                has_all_bins = False
                break
            proto = plasma_emb[b_idx].mean(dim=0, keepdim=True)
            proto = F.normalize(proto, dim=-1)
            prototypes.append(proto)

        if not has_all_bins:
            continue

        proto_mat = torch.cat(prototypes, dim=0)
        cls_img = img_emb[cls_indices]
        cls_target = batch_bin_idx[cls_indices].long()
        logits = logit_scale * torch.matmul(cls_img, proto_mat.t())
        loss_cls = F.cross_entropy(logits, cls_target)
        total_loss.append(loss_cls)

    if len(total_loss) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return torch.stack(total_loss).mean()


def init_stage2_memory_bank(
    num_classes: int,
    num_bins: int,
    emb_dim: int,
    device: torch.device,
) -> dict[str, torch.Tensor | int]:
    return {
        "prototypes": torch.zeros(num_classes, num_bins, emb_dim, device=device, dtype=torch.float32),
        "valid": torch.zeros(num_classes, num_bins, device=device, dtype=torch.bool),
        "count": torch.zeros(num_classes, num_bins, device=device, dtype=torch.long),
        "num_bins": int(num_bins),
    }


def _memory_bank_to_state(memory_bank: dict | None) -> dict | None:
    if memory_bank is None:
        return None
    return {
        "prototypes": memory_bank["prototypes"].detach().cpu(),
        "valid": memory_bank["valid"].detach().cpu(),
        "count": memory_bank["count"].detach().cpu(),
        "num_bins": int(memory_bank.get("num_bins", 3)),
    }


def _memory_bank_from_state(state: dict | None, device: torch.device) -> dict | None:
    if not state:
        return None
    if "prototypes" not in state or "valid" not in state or "count" not in state:
        return None
    return {
        "prototypes": state["prototypes"].to(device=device, dtype=torch.float32),
        "valid": state["valid"].to(device=device, dtype=torch.bool),
        "count": state["count"].to(device=device, dtype=torch.long),
        "num_bins": int(state.get("num_bins", 3)),
    }


@torch.no_grad()
def update_stage2_memory_bank_ema(
    plasma_emb: torch.Tensor,
    label_idx: torch.Tensor,
    batch_bin_idx: torch.Tensor,
    memory_bank: dict,
    momentum: float,
) -> None:
    prototypes = memory_bank["prototypes"]
    valid = memory_bank["valid"]
    count = memory_bank["count"]
    num_bins = int(memory_bank.get("num_bins", 3))

    valid_global = (label_idx >= 0) & (batch_bin_idx >= 0) & (batch_bin_idx < num_bins)
    if not valid_global.any():
        return

    classes = torch.unique(label_idx[valid_global])
    for cls in classes.tolist():
        cls = int(cls)
        cls_mask = valid_global & (label_idx == cls)
        for b in range(num_bins):
            b_mask = cls_mask & (batch_bin_idx == b)
            b_idx = torch.where(b_mask)[0]
            if b_idx.numel() == 0:
                continue

            proto = plasma_emb[b_idx].detach().mean(dim=0, keepdim=True).to(torch.float32)
            proto = F.normalize(proto, dim=-1).squeeze(0)

            if not bool(valid[cls, b]):
                prototypes[cls, b] = proto
                valid[cls, b] = True
            else:
                mixed = momentum * prototypes[cls, b] + (1.0 - momentum) * proto
                prototypes[cls, b] = F.normalize(mixed.unsqueeze(0), dim=-1).squeeze(0)

            count[cls, b] += int(b_idx.numel())


def compute_stage2_plasma_bank_loss(
    img_emb: torch.Tensor,
    label_idx: torch.Tensor,
    batch_bin_idx: torch.Tensor,
    logit_scale: torch.Tensor,
    memory_bank: dict,
    min_bins_for_loss: int = 2,
) -> tuple[torch.Tensor, dict[str, int]]:
    device = img_emb.device
    n = img_emb.shape[0]
    stats = {
        "valid_classes": 0,
        "skipped_classes": 0,
        "effective_samples": 0,
    }

    if n == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), stats

    num_bins = int(memory_bank.get("num_bins", 3))
    bank_prototypes = memory_bank["prototypes"]
    bank_valid = memory_bank["valid"]

    valid_global = (label_idx >= 0) & (batch_bin_idx >= 0) & (batch_bin_idx < num_bins)
    if not valid_global.any():
        return torch.tensor(0.0, device=device, requires_grad=True), stats

    total_loss = []
    classes = torch.unique(label_idx[valid_global])
    for cls in classes.tolist():
        cls = int(cls)
        cls_mask = valid_global & (label_idx == cls)

        valid_bins = torch.where(bank_valid[cls])[0]
        if valid_bins.numel() < int(min_bins_for_loss):
            stats["skipped_classes"] += 1
            continue

        cls_indices = torch.where(cls_mask)[0]
        if cls_indices.numel() == 0:
            stats["skipped_classes"] += 1
            continue

        if not torch.isin(batch_bin_idx[cls_indices], valid_bins).any():
            stats["skipped_classes"] += 1
            continue

        use_indices = cls_indices[torch.isin(batch_bin_idx[cls_indices], valid_bins)]
        if use_indices.numel() == 0:
            stats["skipped_classes"] += 1
            continue

        bin_to_local = torch.full((num_bins,), -1, device=device, dtype=torch.long)
        bin_to_local[valid_bins] = torch.arange(valid_bins.numel(), device=device, dtype=torch.long)
        target_global = batch_bin_idx[use_indices].long()
        target_local = bin_to_local[target_global]

        if (target_local < 0).any():
            stats["skipped_classes"] += 1
            continue

        proto_mat = bank_prototypes[cls, valid_bins]  # (k, d)
        cls_img = img_emb[use_indices]
        logits = logit_scale * torch.matmul(cls_img, proto_mat.t())
        loss_cls = F.cross_entropy(logits, target_local)
        total_loss.append(loss_cls)

        stats["valid_classes"] += 1
        stats["effective_samples"] += int(use_indices.numel())

    if len(total_loss) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), stats

    return torch.stack(total_loss).mean(), stats


def train_one_epoch_stage1(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: dict,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter | None,
    expected_plasma_dim: int,
) -> dict:
    model.train()
    train_cfg = config["training"]

    total_losses = {
        "total": 0.0,
        "img_class": 0.0,
        "img_plasma": 0.0,
        "class_plasma": 0.0,
        "reg": 0.0,
        "valid_label_ratio": 0.0,
        "class_entropy_norm": 0.0,
        "grad_norm": 0.0,
        "grad_clipped_ratio": 0.0,
        "loss_ema": 0.0,
        "img_class_ema": 0.0,
        "reg_ema": 0.0,
    }
    n_batches = 0
    ema_beta = float(train_cfg.get("stage1_log_ema_beta", 0.9))
    ema_beta = min(max(ema_beta, 0.0), 0.9999)
    ema_loss = None
    ema_img_class = None
    ema_reg = None
    n_classes = len(config.get("classes", {}).get("names", []))

    pbar = tqdm(dataloader, desc=f"Stage-1 Epoch {epoch}", leave=True)
    for batch_idx, batch in enumerate(pbar):
        patch_emb = batch["patch_emb"].to(device)
        label_idx = batch["label_idx"].to(device)
        plasma_vals = batch["plasma_vals"].to(device)
        plasma_mask = batch["plasma_mask"].to(device)

        if plasma_vals.shape[-1] != expected_plasma_dim or plasma_mask.shape[-1] != expected_plasma_dim:
            raise ValueError(
                f"Stage-1 plasma 维度不匹配: vals={plasma_vals.shape[-1]}, mask={plasma_mask.shape[-1]}, expected={expected_plasma_dim}"
            )

        optimizer.zero_grad()

        outputs = model(
            tau_tokens=patch_emb,
            diagnosis_id=label_idx,
            plasma_values=plasma_vals,
            plasma_mask=plasma_mask,
        )

        reg_w = float(train_cfg.get("lambda_reg", 0.0))
        loss_dict = compute_total_loss(
            img_emb=outputs["img_emb"],
            class_emb=outputs["class_emb"],
            class_emb_all=outputs["class_emb_all"],
            label_idx=label_idx,
            plasma_emb=outputs["plasma_emb"],
            logit_scale=outputs["logit_scale"],
            plasma_mask=plasma_mask,
            lambda_img_class=float(train_cfg.get("lambda_img_class", 1.0)),
            lambda_img_plasma=0.0,
            lambda_class_plasma=0.0,
            lambda_reg=reg_w,
            reg_type=train_cfg.get("reg_type", "high_sim_penalty"),
            reg_cos_max=float(train_cfg.get("reg_cos_max", 0.8)),
        )

        loss = loss_dict["total"]
        loss.backward()

        max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        grad_norm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        grad_norm = float(grad_norm_t.item() if torch.is_tensor(grad_norm_t) else grad_norm_t)
        grad_clipped = 1.0 if grad_norm > max_grad_norm else 0.0
        optimizer.step()

        valid_mask = (label_idx >= 0)
        valid_label_ratio = float(valid_mask.float().mean().item())

        class_entropy_norm = 0.0
        if valid_mask.any() and n_classes > 1:
            cls_counts = torch.bincount(label_idx[valid_mask].long(), minlength=n_classes).to(torch.float32)
            probs = cls_counts / cls_counts.sum().clamp_min(1.0)
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
            class_entropy_norm = float((entropy / math.log(float(n_classes))).item())

        cur_loss = float(loss.item())
        cur_img_class = float(loss_dict["img_class"].item())
        cur_reg = float(loss_dict["reg"].item())

        if ema_loss is None:
            ema_loss = cur_loss
            ema_img_class = cur_img_class
            ema_reg = cur_reg
        else:
            ema_loss = ema_beta * ema_loss + (1.0 - ema_beta) * cur_loss
            ema_img_class = ema_beta * ema_img_class + (1.0 - ema_beta) * cur_img_class
            ema_reg = ema_beta * ema_reg + (1.0 - ema_beta) * cur_reg

        for k in total_losses:
            if k in loss_dict:
                total_losses[k] += float(loss_dict[k].item())
        total_losses["valid_label_ratio"] += valid_label_ratio
        total_losses["class_entropy_norm"] += class_entropy_norm
        total_losses["grad_norm"] += grad_norm
        total_losses["grad_clipped_ratio"] += grad_clipped
        total_losses["loss_ema"] += float(ema_loss)
        total_losses["img_class_ema"] += float(ema_img_class)
        total_losses["reg_ema"] += float(ema_reg)
        n_batches += 1

        pbar.set_postfix({
            "loss": cur_loss,
            "loss_ema": float(ema_loss),
            "L_ic": cur_img_class,
            "L_ic_ema": float(ema_img_class),
            "L_reg": cur_reg,
            "vld": f"{valid_label_ratio:.2f}",
            "clip": int(grad_clipped),
        })

        if writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar("stage1/train/loss_total", cur_loss, global_step)
            writer.add_scalar("stage1/train/loss_total_ema", float(ema_loss), global_step)
            writer.add_scalar("stage1/train/loss_img_class", cur_img_class, global_step)
            writer.add_scalar("stage1/train/loss_img_class_ema", float(ema_img_class), global_step)
            writer.add_scalar("stage1/train/loss_reg", cur_reg, global_step)
            writer.add_scalar("stage1/train/loss_reg_ema", float(ema_reg), global_step)
            writer.add_scalar("stage1/train/valid_label_ratio", valid_label_ratio, global_step)
            writer.add_scalar("stage1/train/class_entropy_norm", class_entropy_norm, global_step)
            writer.add_scalar("stage1/train/grad_norm", grad_norm, global_step)
            writer.add_scalar("stage1/train/grad_clipped", grad_clipped, global_step)
            writer.add_scalar("stage1/train/logit_scale", float(outputs["logit_scale"].item()), global_step)

    for k in total_losses:
        total_losses[k] /= max(n_batches, 1)
    return total_losses


def train_one_epoch_stage2(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: dict,
    device: torch.device,
    stage2_epoch: int,
    stage2_epochs: int,
    warmup_frac: float,
    plasma_weight_max: float,
    subject_bin_map: dict[str, int],
    memory_bank: dict,
    bank_momentum: float,
    min_bins_for_loss: int,
    writer: SummaryWriter | None,
    expected_plasma_dim: int,
) -> dict:
    model.train()
    train_cfg = config["training"]
    w_plasma = 0.0
    total_steps = int(stage2_epochs * len(dataloader))

    total_losses = {
        "total": 0.0,
        "img_class": 0.0,
        "img_plasma_bin": 0.0,
        "valid_classes": 0.0,
        "effective_samples": 0.0,
        "skipped_classes": 0.0,
    }
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Stage-2 Epoch {stage2_epoch + 1}/{stage2_epochs}", leave=True)
    for batch_idx, batch in enumerate(pbar):
        patch_emb = batch["patch_emb"].to(device)
        label_idx = batch["label_idx"].to(device)
        plasma_vals = batch["plasma_vals"].to(device)
        plasma_mask = batch["plasma_mask"].to(device)
        subjects = batch["subjects"]

        if plasma_vals.shape[-1] != expected_plasma_dim or plasma_mask.shape[-1] != expected_plasma_dim:
            raise ValueError(
                f"Stage-2 plasma 维度不匹配: vals={plasma_vals.shape[-1]}, mask={plasma_mask.shape[-1]}, expected={expected_plasma_dim}"
            )

        bin_ids = [subject_bin_map.get(str(sid), -1) for sid in subjects]
        batch_bin_idx = torch.as_tensor(bin_ids, device=device, dtype=torch.long)

        optimizer.zero_grad()

        outputs = model(
            tau_tokens=patch_emb,
            diagnosis_id=label_idx,
            plasma_values=plasma_vals,
            plasma_mask=plasma_mask,
        )

        global_step = int(stage2_epoch * len(dataloader) + batch_idx + 1)
        w_plasma = _compute_stage2_w_plasma(
            global_step=global_step,
            total_steps=total_steps,
            warmup_frac=warmup_frac,
            plasma_weight_max=plasma_weight_max,
        )

        base_loss = compute_total_loss(
            img_emb=outputs["img_emb"],
            class_emb=outputs["class_emb"],
            class_emb_all=outputs["class_emb_all"],
            label_idx=label_idx,
            plasma_emb=outputs["plasma_emb"],
            logit_scale=outputs["logit_scale"],
            plasma_mask=plasma_mask,
            lambda_img_class=float(train_cfg.get("lambda_img_class", 1.0)),
            lambda_img_plasma=0.0,
            lambda_class_plasma=0.0,
            lambda_reg=0.0,
            reg_type=train_cfg.get("reg_type", "high_sim_penalty"),
            reg_cos_max=float(train_cfg.get("reg_cos_max", 0.8)),
        )

        update_stage2_memory_bank_ema(
            plasma_emb=outputs["plasma_emb"],
            label_idx=label_idx,
            batch_bin_idx=batch_bin_idx,
            memory_bank=memory_bank,
            momentum=float(bank_momentum),
        )

        loss_bin, loss_stats = compute_stage2_plasma_bank_loss(
            img_emb=outputs["img_emb"],
            label_idx=label_idx,
            batch_bin_idx=batch_bin_idx,
            logit_scale=outputs["logit_scale"],
            memory_bank=memory_bank,
            min_bins_for_loss=int(min_bins_for_loss),
        )

        loss = base_loss["img_class"] + float(w_plasma) * loss_bin
        loss.backward()

        max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_losses["total"] += float(loss.item())
        total_losses["img_class"] += float(base_loss["img_class"].item())
        total_losses["img_plasma_bin"] += float(loss_bin.item())
        total_losses["valid_classes"] += float(loss_stats["valid_classes"])
        total_losses["effective_samples"] += float(loss_stats["effective_samples"])
        total_losses["skipped_classes"] += float(loss_stats["skipped_classes"])
        n_batches += 1

        pbar.set_postfix({
            "loss": float(loss.item()),
            "L_ic": float(base_loss["img_class"].item()),
            "L_ip_bin": float(loss_bin.item()),
            "w_plasma": float(w_plasma),
            "valid_cls": int(loss_stats["valid_classes"]),
        })

        if writer is not None:
            writer.add_scalar("stage2/train/loss_total", float(loss.item()), global_step)
            writer.add_scalar("stage2/train/loss_img_class", float(base_loss["img_class"].item()), global_step)
            writer.add_scalar("stage2/train/loss_img_plasma_bin", float(loss_bin.item()), global_step)
            writer.add_scalar("stage2/train/w_plasma", float(w_plasma), global_step)
            writer.add_scalar("stage2/train/valid_classes", float(loss_stats["valid_classes"]), global_step)
            writer.add_scalar("stage2/train/effective_samples", float(loss_stats["effective_samples"]), global_step)
            writer.add_scalar("stage2/train/skipped_classes", float(loss_stats["skipped_classes"]), global_step)

    for k in total_losses:
        total_losses[k] /= max(n_batches, 1)
    total_losses["w_plasma"] = float(w_plasma)
    valid_sum = max(float(total_losses["valid_classes"]) + float(total_losses["skipped_classes"]), 1.0)
    total_losses["valid_class_ratio"] = float(total_losses["valid_classes"]) / valid_sum
    return total_losses


def _build_model(config: dict, class_names: list[str], plasma_prompts: list[str], device: torch.device) -> CoCoOpTAUModel:
    model_cfg = config["model"]
    prompt_template = config["classes"]["prompt_template"]
    model = CoCoOpTAUModel(
        biomedclip_path=model_cfg["biomedclip_path"],
        class_names=class_names,
        class_prompt_template=prompt_template,
        plasma_prompts=plasma_prompts,
        ctx_len=model_cfg.get("ctx_len", 4),
        proj_dim=model_cfg.get("proj_dim", 512),
        ctx_hidden_dim=model_cfg.get("ctx_hidden_dim", 1024),
        share_ctx_base=model_cfg.get("share_ctx_base", False),
        plasma_temperature=config.get("plasma", {}).get("temperature", 1.0),
    )
    return model.to(device)


def _prepare_cache_and_missing_set(config: dict, csv_path: Path, cache_dir: Path) -> set:
    adni_root = config.get("data", {}).get("adni_root", "/mnt/nfsdata/nfsdata/ADNI/ADNI0103/Coregistration")
    gpu_cfg = config["training"].get("gpu", None)

    print("\n" + "=" * 60)
    print("缓存状态检查")
    print("=" * 60)

    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_stats = get_cache_stats(csv_path=str(csv_path), cache_dir=str(cache_dir))
    print(f"总样本数: {cache_stats['total']}")
    print(f"已缓存: {cache_stats['cached']}")
    print(f"缺失: {cache_stats['missing']}")

    missing_cache_set = set()

    if cache_stats["missing"] > 0:
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

        cache_stats = get_cache_stats(csv_path=str(csv_path), cache_dir=str(cache_dir))
        print(f"\n缓存更新后: 已缓存 {cache_stats['cached']}, 仍缺失 {cache_stats['missing']}")

        if cache_stats["missing"] > 0:
            print(f"警告: 仍有 {cache_stats['missing']} 个缺失，这些样本将不会参与训练")
            missing_list = cache_stats.get("missing_list", [])
            print(f"\n缺失缓存列表 (共 {len(missing_list)} 个):")
            for ptid, tau_id in missing_list[:20]:
                print(f"  - {ptid}_{tau_id}.vision.pt")
            if len(missing_list) > 20:
                print(f"  ... 还有 {len(missing_list) - 20} 个未显示")
            missing_cache_set = {(ptid, tau_id) for ptid, tau_id in missing_list}
    else:
        print("✅ 所有缓存均已存在")

    print()
    return missing_cache_set


def _format_thr(v: float) -> str:
    if np.isfinite(v):
        return f"{v:.4f}"
    return "N/A"


@email_on_error()
def main():
    parser = argparse.ArgumentParser(description="TAU CoCoOp Two-Stage Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的 checkpoint 路径")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（覆盖配置）")
    parser.add_argument("--epochs", type=int, default=None, help="训练 epochs（覆盖配置）")
    parser.add_argument(
        "--val_split_json",
        type=str,
        default="fixed_split.json",
        help="固定 train/val 划分 JSON（与 train.py 共享格式）",
    )

    parser.add_argument("--stage1_epochs", type=int, default=50, help="Stage-1 epoch 数；不传则等于 --epochs")
    parser.add_argument("--stage2_epochs", type=int, default=50, help="Stage-2 epoch 数；<=0 则跳过 Stage-2")
    parser.add_argument("--stage1_patience", type=int, default=15, help="Stage-1 早停 patience；<=0 关闭早停")
    parser.add_argument("--stage1_freeze_logit_scale", action="store_true", help="Stage-1 冻结 logit_scale（用于降低步级抖动）")
    parser.add_argument("--plasma_warmup_frac", type=float, default=0.3, help="Stage-2 w_plasma warmup 比例")
    parser.add_argument("--plasma_weight_max", type=float, default=1.0, help="Stage-2 plasma loss 最大权重")
    parser.add_argument("--plasma_feature_name", type=str, default="pT217_F", help="Stage-2 单 key plasma 名称")

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    config = load_config(str(config_path))

    seed = args.seed if args.seed is not None else int(config["training"].get("seed", 42))
    epochs_cfg = int(config["training"]["epochs"])
    epochs = args.epochs if args.epochs is not None else epochs_cfg

    stage1_epochs = args.stage1_epochs if args.stage1_epochs is not None else epochs
    stage2_epochs = 0 if args.stage2_epochs is None else int(args.stage2_epochs)
    stage1_patience = int(args.stage1_patience)
    stage1_freeze_logit_scale = bool(args.stage1_freeze_logit_scale)
    if not stage1_freeze_logit_scale:
        stage1_freeze_logit_scale = bool(config.get("training", {}).get("stage1_freeze_logit_scale", False))
    if stage2_epochs < 0:
        stage2_epochs = 0

    set_seed(seed)

    gpu_cfg = config["training"].get("gpu", None)
    if gpu_cfg is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_cfg)
        print(f"CUDA_VISIBLE_DEVICES set to: {gpu_cfg}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    csv_path = script_dir / config["data"]["csv_path"]
    cache_dir = Path(config["data"]["cache_dir"])
    diagnosis_csv = config.get("data", {}).get("diagnosis_csv", None)
    diagnosis_code_map = config.get("classes", {}).get("diagnosis_code_map", None)
    class_names = config["classes"]["names"]

    # Stage-1 沿用 config selected_keys
    selected_plasma_keys_stage1, plasma_prompts_stage1 = resolve_plasma_config(config)
    config.setdefault("plasma", {})["keys"] = selected_plasma_keys_stage1
    print(f"[Stage-1][Plasma] Selected keys ({len(selected_plasma_keys_stage1)}): {selected_plasma_keys_stage1}")

    # Stage-2 强制单 key
    stage2_key, stage2_prompt = _resolve_stage2_single_key_and_prompt(config, args.plasma_feature_name)
    print(f"[Stage-2][Plasma] single key: {stage2_key}")

    # 缓存检查仅做一次
    missing_cache_set = _prepare_cache_and_missing_set(config, csv_path, cache_dir)

    # 先用 Stage-1 keys 构建 full_dataset 与固定 split
    full_dataset_stage1 = _build_common_dataset(
        config=config,
        csv_path=csv_path,
        cache_dir=cache_dir,
        class_names=class_names,
        selected_keys=selected_plasma_keys_stage1,
        missing_cache_set=missing_cache_set,
        diagnosis_csv=diagnosis_csv,
        diagnosis_code_map=diagnosis_code_map,
    )
    print(f"Total samples: {len(full_dataset_stage1)}")
    if missing_cache_set:
        print(f"(已排除 {len(missing_cache_set)} 个缺失缓存的样本)")

    val_ratio = float(config["data"].get("val_ratio", 0.15))
    train_indices, val_indices = split_by_subject(full_dataset_stage1, val_ratio=val_ratio, seed=seed)

    if args.val_split_json is not None:
        train_indices, val_indices = apply_fixed_split(
            full_dataset=full_dataset_stage1,
            train_indices=train_indices,
            val_indices=val_indices,
            val_split_json=args.val_split_json,
            seed=seed,
            val_ratio=val_ratio,
        )

    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")

    # Stage-1 dataset/loader
    train_dataset_s1, val_dataset_s1 = _build_split_datasets(
        config=config,
        full_dataset=full_dataset_stage1,
        train_indices=train_indices,
        val_indices=val_indices,
        csv_path=csv_path,
        cache_dir=cache_dir,
        class_names=class_names,
        selected_keys=selected_plasma_keys_stage1,
        missing_cache_set=missing_cache_set,
        diagnosis_csv=diagnosis_csv,
        diagnosis_code_map=diagnosis_code_map,
    )

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"].get("num_workers", 4))
    train_loader_s1, val_loader_s1 = _build_dataloaders(
        train_dataset=train_dataset_s1,
        val_dataset=val_dataset_s1,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # 统一 run dir: MM.DD_pid
    log_cfg = config["log"]
    log_dir = script_dir / log_cfg.get("dir", "./runs")
    run_name = f"{datetime.now().strftime('%m.%d')}_{os.getpid()}"
    run_dir = log_dir / run_name
    ckpt_dir = run_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(run_dir))
    writer.add_text("config", yaml.dump(config, default_flow_style=False))
    writer.add_text(
        "two_stage_args",
        json.dumps(
            {
                "stage1_epochs": stage1_epochs,
                "stage2_epochs": stage2_epochs,
                "stage1_patience": stage1_patience,
                "stage1_freeze_logit_scale": stage1_freeze_logit_scale,
                "plasma_warmup_frac": args.plasma_warmup_frac,
                "plasma_weight_max": args.plasma_weight_max,
                "plasma_feature_name": args.plasma_feature_name,
            },
            ensure_ascii=False,
            indent=2,
        ),
    )

    # ----------------------------
    # Stage-1
    # ----------------------------
    model = _build_model(config, class_names, plasma_prompts_stage1, device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[Stage-1] Trainable params: {n_trainable:,} / {n_total:,}")

    optimizer_s1 = build_optimizer_for_stage1(
        model=model,
        base_lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
        freeze_logit_scale=stage1_freeze_logit_scale,
    )
    print(f"[Stage-1] freeze_logit_scale={stage1_freeze_logit_scale}")

    start_s1_epoch = 0
    best_metric_s1 = 0.0
    epochs_without_improve = 0

    if args.resume:
        resume_stage, resume_stage_epoch, resume_best_metric, _ = load_checkpoint_2s(model, optimizer_s1, args.resume)
        if resume_stage == "stage1":
            start_s1_epoch = resume_stage_epoch + 1
            best_metric_s1 = resume_best_metric
            epochs_without_improve = 0
            print(f"[Resume] Stage-1 从 epoch={start_s1_epoch} 继续")
        else:
            print("[Resume] 加载到非 Stage-1 checkpoint，Stage-1 将从头训练（权重已加载）。")

    save_every = int(config.get("log", {}).get("save_every", 5))

    for epoch in range(start_s1_epoch, stage1_epochs):
        print(f"\n{'=' * 70}")
        print(f"Stage-1 | Epoch {epoch + 1}/{stage1_epochs}")
        print(f"{'=' * 70}")

        train_losses = train_one_epoch_stage1(
            model=model,
            dataloader=train_loader_s1,
            optimizer=optimizer_s1,
            config=config,
            device=device,
            epoch=epoch,
            writer=writer,
            expected_plasma_dim=len(selected_plasma_keys_stage1),
        )

        val_metrics = validate(
            model=model,
            dataloader=val_loader_s1,
            config=config,
            device=device,
            expected_plasma_dim=len(selected_plasma_keys_stage1),
        )

        print(
            "Train - "
            f"total: {train_losses['total']:.4f}, "
            f"img_class: {train_losses['img_class']:.4f}, "
            f"reg: {train_losses['reg']:.4f}"
        )

        print(
            "Val   - "
            f"probe(BAcc/F1): {val_metrics['probe_bal_acc']:.4f}/{val_metrics['probe_macro_f1']:.4f}, "
            f"inject(BAcc/F1): {val_metrics['inject_bal_acc']:.4f}/{val_metrics['inject_macro_f1']:.4f}, "
            "plasma: N/A (Stage-1 disabled)"
        )

        writer.add_scalar("stage1/train_epoch/loss_total", train_losses["total"], epoch)
        writer.add_scalar("stage1/train_epoch/loss_img_class", train_losses["img_class"], epoch)
        writer.add_scalar("stage1/train_epoch/loss_reg", train_losses["reg"], epoch)
        writer.add_scalar("stage1/val/probe_bal_acc", val_metrics["probe_bal_acc"], epoch)
        writer.add_scalar("stage1/val/probe_macro_f1", val_metrics["probe_macro_f1"], epoch)
        writer.add_scalar("stage1/val/inject_bal_acc", val_metrics["inject_bal_acc"], epoch)
        writer.add_scalar("stage1/val/inject_macro_f1", val_metrics["inject_macro_f1"], epoch)

        # Stage-1 plasma 指标记为 NaN
        writer.add_scalar("stage1/val/plasma_score_drop_within_mean", float("nan"), epoch)
        writer.add_scalar("stage1/val/plasma_score_drop_cross_mean", float("nan"), epoch)
        writer.add_scalar("stage1/val/plasma_margin_drop_within_mean", float("nan"), epoch)
        writer.add_scalar("stage1/val/plasma_margin_drop_cross_mean", float("nan"), epoch)

        current_metric = float(val_metrics.get("inject_macro_f1", 0.0))

        if (epoch + 1) % save_every == 0:
            save_checkpoint_2s(
                model=model,
                optimizer=optimizer_s1,
                stage="stage1",
                stage_epoch=epoch,
                best_metric=current_metric,
                save_path=str(ckpt_dir / f"epoch_{epoch + 1:03d}_stage1.pt"),
            )

        if current_metric > best_metric_s1:
            best_metric_s1 = current_metric
            epochs_without_improve = 0
            save_checkpoint_2s(
                model=model,
                optimizer=optimizer_s1,
                stage="stage1",
                stage_epoch=epoch,
                best_metric=best_metric_s1,
                save_path=str(ckpt_dir / "best_stage1.pt"),
            )
            print(f"[Stage-1][Best] inject_macro_f1 = {best_metric_s1:.4f}")
        else:
            epochs_without_improve += 1
            if stage1_patience > 0:
                print(f"[Stage-1][EarlyStop] no improve: {epochs_without_improve}/{stage1_patience}")

        if stage1_patience > 0 and epochs_without_improve >= stage1_patience:
            print(
                f"[Stage-1][EarlyStop] 连续 {stage1_patience} 个 epoch 未提升，"
                "提前结束 Stage-1，进入 Stage-2。"
            )
            break

    best_stage1_path = ckpt_dir / "best_stage1.pt"
    if not best_stage1_path.exists():
        save_checkpoint_2s(
            model=model,
            optimizer=optimizer_s1,
            stage="stage1",
            stage_epoch=max(stage1_epochs - 1, 0),
            best_metric=best_metric_s1,
            save_path=str(best_stage1_path),
        )

    # ----------------------------
    # Stage-2（可选）
    # ----------------------------
    if stage2_epochs <= 0:
        writer.close()
        print("\nStage-2 skipped (stage2_epochs <= 0).")
        print(f"Training finished. Stage-1 best inject_macro_f1: {best_metric_s1:.4f}")
        return

    print("\n" + "=" * 70)
    print("进入 Stage-2")
    print("=" * 70)

    # Stage-2 单 key 数据与模型
    selected_plasma_keys_s2 = [stage2_key]
    plasma_prompts_s2 = [stage2_prompt]

    full_dataset_stage2 = _build_common_dataset(
        config=config,
        csv_path=csv_path,
        cache_dir=cache_dir,
        class_names=class_names,
        selected_keys=selected_plasma_keys_s2,
        missing_cache_set=missing_cache_set,
        diagnosis_csv=diagnosis_csv,
        diagnosis_code_map=diagnosis_code_map,
    )

    train_dataset_s2, val_dataset_s2 = _build_split_datasets(
        config=config,
        full_dataset=full_dataset_stage2,
        train_indices=train_indices,
        val_indices=val_indices,
        csv_path=csv_path,
        cache_dir=cache_dir,
        class_names=class_names,
        selected_keys=selected_plasma_keys_s2,
        missing_cache_set=missing_cache_set,
        diagnosis_csv=diagnosis_csv,
        diagnosis_code_map=diagnosis_code_map,
    )
    train_loader_s2, val_loader_s2 = _build_dataloaders(
        train_dataset=train_dataset_s2,
        val_dataset=val_dataset_s2,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    thresholds = compute_classwise_plasma_thresholds(
        train_dataset=train_dataset_s2,
        class_names=class_names,
        plasma_key=stage2_key,
    )
    subject_bin_map = build_subject_bin_map_from_train_dataset(
        train_dataset=train_dataset_s2,
        class_names=class_names,
        plasma_key=stage2_key,
        thresholds=thresholds,
    )

    print("[Stage-2] classwise thresholds summary (q33 / q66):")
    for cls_idx, cls_name in enumerate(class_names):
        thr = thresholds.get(cls_idx, {"q33": float("nan"), "q66": float("nan"), "n": 0.0})
        print(
            f"  - {cls_name}: q33={_format_thr(float(thr['q33']))}, "
            f"q66={_format_thr(float(thr['q66']))}, n={int(thr['n'])}"
        )

    model_s2 = _build_model(config, class_names, plasma_prompts_s2, device)

    # 用 Stage-1 best 初始化
    state_s1 = torch.load(str(best_stage1_path), map_location="cpu")
    model_s2.load_state_dict(state_s1["model_state_dict"], strict=False)
    print(f"[Stage-2] Initialized from: {best_stage1_path}")

    lr = float(config["training"]["lr"])
    weight_decay = float(config["training"].get("weight_decay", 0.01))
    image_lr_mult = float(config.get("training", {}).get("image_lr_mult", 0.1))

    optimizer_s2 = build_optimizer_for_stage2(
        model=model_s2,
        base_lr=lr,
        weight_decay=weight_decay,
        image_lr_mult=image_lr_mult,
        freeze_logit_scale=True,
    )

    bank_momentum = 0.95
    min_bins_for_loss = 2
    bank_dim = int(config["model"].get("proj_dim", 512))
    memory_bank = init_stage2_memory_bank(
        num_classes=len(class_names),
        num_bins=3,
        emb_dim=bank_dim,
        device=device,
    )

    start_stage2_epoch = 0
    best_metric_s2 = 0.0
    if args.resume:
        resume_stage, resume_stage_epoch, resume_best_metric, resume_extra = load_checkpoint_2s(
            model_s2,
            optimizer_s2,
            args.resume,
        )
        if resume_stage == "stage2":
            start_stage2_epoch = resume_stage_epoch + 1
            best_metric_s2 = resume_best_metric
            bank_state = _memory_bank_from_state(resume_extra.get("stage2_memory_bank", None), device=device)
            if bank_state is not None:
                memory_bank = bank_state
            momentum_state = resume_extra.get("stage2_bank_momentum", None)
            if momentum_state is not None:
                bank_momentum = float(momentum_state)
            print(f"[Resume] Stage-2 从 epoch={start_stage2_epoch} 继续, bank_momentum={bank_momentum:.3f}")

    for s2_epoch in range(start_stage2_epoch, stage2_epochs):
        w_plasma = _compute_stage2_w_plasma(
            global_step=int(s2_epoch * len(train_loader_s2) + 1),
            total_steps=int(stage2_epochs * len(train_loader_s2)),
            warmup_frac=float(args.plasma_warmup_frac),
            plasma_weight_max=float(args.plasma_weight_max),
        )

        print(f"\n{'=' * 70}")
        print(f"Stage-2 | Epoch {s2_epoch + 1}/{stage2_epochs} | w_plasma={w_plasma:.4f}")
        print(f"{'=' * 70}")

        train_losses = train_one_epoch_stage2(
            model=model_s2,
            dataloader=train_loader_s2,
            optimizer=optimizer_s2,
            config=config,
            device=device,
            stage2_epoch=s2_epoch,
            stage2_epochs=stage2_epochs,
            warmup_frac=float(args.plasma_warmup_frac),
            plasma_weight_max=float(args.plasma_weight_max),
            subject_bin_map=subject_bin_map,
            memory_bank=memory_bank,
            bank_momentum=bank_momentum,
            min_bins_for_loss=min_bins_for_loss,
            writer=writer,
            expected_plasma_dim=1,
        )

        val_metrics = validate(
            model=model_s2,
            dataloader=val_loader_s2,
            config=config,
            device=device,
            expected_plasma_dim=1,
        )

        print(
            "Train - "
            f"total: {train_losses['total']:.4f}, "
            f"img_class: {train_losses['img_class']:.4f}, "
            f"img_plasma_bin: {train_losses['img_plasma_bin']:.4f}, "
            f"w_plasma: {train_losses['w_plasma']:.4f}, "
            f"valid_class_ratio: {train_losses['valid_class_ratio']:.3f}"
        )
        print(
            "Val   - "
            f"probe(BAcc/F1): {val_metrics['probe_bal_acc']:.4f}/{val_metrics['probe_macro_f1']:.4f}, "
            f"inject(BAcc/F1): {val_metrics['inject_bal_acc']:.4f}/{val_metrics['inject_macro_f1']:.4f}, "
            f"within_margin_drop: {val_metrics['plasma_margin_drop_within_mean']:.4f}±{val_metrics['plasma_margin_drop_within_std']:.4f}, "
            f"cross_margin_drop: {val_metrics['plasma_margin_drop_cross_mean']:.4f}±{val_metrics['plasma_margin_drop_cross_std']:.4f}"
        )
        print(f"Val   - plasma diagnosis: {val_metrics['plasma_shuffle_diagnosis']}")

        writer.add_scalar("stage2/train_epoch/loss_total", train_losses["total"], s2_epoch)
        writer.add_scalar("stage2/train_epoch/loss_img_class", train_losses["img_class"], s2_epoch)
        writer.add_scalar("stage2/train_epoch/loss_img_plasma_bin", train_losses["img_plasma_bin"], s2_epoch)
        writer.add_scalar("stage2/train_epoch/w_plasma", train_losses["w_plasma"], s2_epoch)
        writer.add_scalar("stage2/train_epoch/valid_classes", train_losses["valid_classes"], s2_epoch)
        writer.add_scalar("stage2/train_epoch/effective_samples", train_losses["effective_samples"], s2_epoch)
        writer.add_scalar("stage2/train_epoch/skipped_classes", train_losses["skipped_classes"], s2_epoch)
        writer.add_scalar("stage2/train_epoch/valid_class_ratio", train_losses["valid_class_ratio"], s2_epoch)
        writer.add_scalar("stage2/val/probe_bal_acc", val_metrics["probe_bal_acc"], s2_epoch)
        writer.add_scalar("stage2/val/probe_macro_f1", val_metrics["probe_macro_f1"], s2_epoch)
        writer.add_scalar("stage2/val/inject_bal_acc", val_metrics["inject_bal_acc"], s2_epoch)
        writer.add_scalar("stage2/val/inject_macro_f1", val_metrics["inject_macro_f1"], s2_epoch)
        writer.add_scalar("stage2/val/plasma_score_drop_within_mean", val_metrics["plasma_score_drop_within_mean"], s2_epoch)
        writer.add_scalar("stage2/val/plasma_score_drop_cross_mean", val_metrics["plasma_score_drop_cross_mean"], s2_epoch)
        writer.add_scalar("stage2/val/plasma_margin_drop_within_mean", val_metrics["plasma_margin_drop_within_mean"], s2_epoch)
        writer.add_scalar("stage2/val/plasma_margin_drop_cross_mean", val_metrics["plasma_margin_drop_cross_mean"], s2_epoch)

        current_metric = float(val_metrics.get("inject_macro_f1", 0.0))
        if (s2_epoch + 1) % save_every == 0:
            save_checkpoint_2s(
                model=model_s2,
                optimizer=optimizer_s2,
                stage="stage2",
                stage_epoch=s2_epoch,
                best_metric=current_metric,
                save_path=str(ckpt_dir / f"epoch_{s2_epoch + 1:03d}_stage2.pt"),
                extra_state={
                    "stage2_memory_bank": _memory_bank_to_state(memory_bank),
                    "stage2_bank_momentum": float(bank_momentum),
                },
            )

        if current_metric > best_metric_s2:
            best_metric_s2 = current_metric
            save_checkpoint_2s(
                model=model_s2,
                optimizer=optimizer_s2,
                stage="stage2",
                stage_epoch=s2_epoch,
                best_metric=best_metric_s2,
                save_path=str(ckpt_dir / "best_stage2.pt"),
                extra_state={
                    "stage2_memory_bank": _memory_bank_to_state(memory_bank),
                    "stage2_bank_momentum": float(bank_momentum),
                },
            )
            print(f"[Stage-2][Best] inject_macro_f1 = {best_metric_s2:.4f}")

    if not (ckpt_dir / "best_stage2.pt").exists():
        save_checkpoint_2s(
            model=model_s2,
            optimizer=optimizer_s2,
            stage="stage2",
            stage_epoch=max(stage2_epochs - 1, 0),
            best_metric=best_metric_s2,
            save_path=str(ckpt_dir / "best_stage2.pt"),
            extra_state={
                "stage2_memory_bank": _memory_bank_to_state(memory_bank),
                "stage2_bank_momentum": float(bank_momentum),
            },
        )

    writer.close()
    print("\nTraining finished.")
    print(f"Best Stage-1 inject_macro_f1: {best_metric_s1:.4f}")
    print(f"Best Stage-2 inject_macro_f1: {best_metric_s2:.4f}")


if __name__ == "__main__":
    main()
