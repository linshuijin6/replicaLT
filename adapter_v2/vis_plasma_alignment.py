"""
adapter_v2/vis_plasma_alignment.py
==================================
Plasma 对齐可视化脚本

目标：证明 plasma 分支在同一类别内部被编码/利用

流程：
1. 加载训练好的 checkpoint（包含 context net、projection）
2. Forward 得到 plasma_emb（512 维，经过 context 调制和 projection）
3. 对 plasma_emb 进行 UMAP 降维
4. 生成 1×3 并排子图：
   - 子图1: 真实 pT217_F 着色
   - 子图2: within-class shuffle 后的 pT217_F 着色（反事实对照）
   - 子图3: 从 plasma_emb 线性回归预测的 pT217_F 着色 + Spearman ρ
5. 为 CN/MCI/AD 三个类别分别生成 train 和 val 图（共 6 组）

输出：
- vis_plasma_output/{class_name}/{split}/1x3.png
- vis_plasma_output/{class_name}/{split}/meta.json
"""

import os
import sys
import json
import yaml
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# 检查 UMAP 依赖
try:
    from umap import UMAP
except ImportError:
    raise ImportError(
        "UMAP not found. Please install: pip install umap-learn"
    )

# 科学计算库
from scipy.stats import spearmanr, rankdata, skew
from sklearn.linear_model import LinearRegression

# 绘图库
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# 添加 adapter_v2 到 path
script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# 导入项目组件
from dataset import TAUPlasmaDataset, collate_fn
from models import CoCoOpTAUModel


# ============================================================================
# 工具函数
# ============================================================================

def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """加载 YAML 配置"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def _normalize_plasma_key(key: str) -> str:
    """归一化 plasma key（大小写兼容）"""
    return str(key).strip().lower()


def resolve_plasma_config(config: dict) -> Tuple[List[str], List[str]]:
    """
    解析 plasma 配置，返回 (selected_keys, plasma_prompts)
    """
    plasma_cfg = config.get("plasma", {})
    
    available_keys = plasma_cfg.get("available_keys", [])
    if isinstance(available_keys, str):
        available_keys = [available_keys]
    if not isinstance(available_keys, list) or len(available_keys) == 0:
        raise ValueError("plasma.available_keys 不能为空")
    
    selected_keys = plasma_cfg.get("selected_keys", available_keys)
    if isinstance(selected_keys, str):
        selected_keys = [selected_keys]
    if not isinstance(selected_keys, list) or len(selected_keys) == 0:
        raise ValueError("plasma.selected_keys 不能为空")
    
    # 归一化映射
    normalized_available = {}
    for key in available_keys:
        norm_key = _normalize_plasma_key(key)
        if norm_key not in normalized_available:
            normalized_available[norm_key] = key
    
    # 别名映射
    alias_cfg = plasma_cfg.get("key_aliases", {}) or {}
    alias_map = {}
    for alias, target in alias_cfg.items():
        alias_map[_normalize_plasma_key(alias)] = _normalize_plasma_key(target)
    
    # 解析选中的 keys
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
    
    # 解析 prompts
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
    
    return resolved_keys, plasma_prompts


def load_fixed_split(
    full_dataset: TAUPlasmaDataset,
    val_split_json: str,
    seed: int,
    val_ratio: float,
) -> Tuple[List[int], List[int]]:
    """
    加载或生成固定的 train/val 划分
    
    Returns:
        train_indices, val_indices
    """
    split_path = Path(val_split_json)
    
    # 构建 subject -> indices 映射
    subject_to_indices = {}
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
        print(f"[Split] Loaded from: {split_path}")
        print(f"[Split] Subjects train={len(train_subjects)} val={len(val_subjects)}")
    else:
        # 生成随机划分
        from dataset import split_by_subject
        train_indices, val_indices = split_by_subject(
            full_dataset, val_ratio=val_ratio, seed=seed
        )
        
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
        print(f"[Split] Saved to: {split_path}")
    
    return train_indices, val_indices


def load_model_and_checkpoint(
    config: dict,
    ckpt_path: str,
    device: torch.device,
) -> CoCoOpTAUModel:
    """
    加载模型和 checkpoint
    """
    # 解析 plasma 配置
    selected_plasma_keys, plasma_prompts = resolve_plasma_config(config)
    
    # 构建模型
    model_cfg = config["model"]
    class_names = config["classes"]["names"]
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
        plasma_temperature=config["plasma"].get("temperature", 1.0),
    )
    
    # 加载 checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    state = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        print(f"[Checkpoint] Loaded from epoch {state.get('epoch', '?')}")
    else:
        model.load_state_dict(state)
    
    model = model.to(device)
    model.eval()
    
    return model


def extract_plasma_embeddings(
    model: CoCoOpTAUModel,
    dataloader: DataLoader,
    device: torch.device,
    plasma_key_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    提取 plasma_emb、labels、plasma_vals、plasma_mask
    
    Returns:
        plasma_emb: (N, 512)
        labels: (N,)
        plasma_vals: (N,) - pT217_F 原始值
        plasma_mask: (N,) - 是否有效
        subject_ids: List[str]
    """
    all_plasma_emb = []
    all_labels = []
    all_plasma_vals = []
    all_plasma_mask = []
    all_subject_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extract embeddings", leave=False):
            patch_emb = batch["patch_emb"].to(device)
            label_idx = batch["label_idx"].to(device)
            plasma_vals = batch["plasma_vals"].to(device)
            plasma_mask = batch["plasma_mask"].to(device)
            
            # Forward
            outputs = model(
                tau_tokens=patch_emb,
                diagnosis_id=label_idx,
                plasma_values=plasma_vals,
                plasma_mask=plasma_mask,
            )
            
            # 提取 plasma_emb
            plasma_emb = outputs["plasma_emb"]  # (B, 512)
            
            # 提取目标 plasma key 的值与 mask
            target_plasma_val = plasma_vals[:, plasma_key_idx]  # (B,)
            target_plasma_mask = plasma_mask[:, plasma_key_idx]  # (B,)
            
            all_plasma_emb.append(plasma_emb.cpu().numpy())
            all_labels.append(label_idx.cpu().numpy())
            all_plasma_vals.append(target_plasma_val.cpu().numpy())
            all_plasma_mask.append(target_plasma_mask.cpu().numpy())
            all_subject_ids.extend(batch["subjects"])
    
    plasma_emb = np.concatenate(all_plasma_emb, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    plasma_vals = np.concatenate(all_plasma_vals, axis=0)
    plasma_mask = np.concatenate(all_plasma_mask, axis=0)
    
    return plasma_emb, labels, plasma_vals, plasma_mask, all_subject_ids


def within_class_shuffle(
    plasma_vals: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, bool]:
    """
    对 plasma_vals 做类内打乱（derangement：每个元素不与自身配对）
    
    Returns:
        shuffled_vals: (N,)
        success: bool - 是否成功生成 derangement
    """
    n = len(plasma_vals)
    if n < 2:
        return plasma_vals.copy(), False
    
    # 尝试生成 derangement
    max_tries = 100
    for _ in range(max_tries):
        perm = rng.permutation(n)
        if np.all(perm != np.arange(n)):
            return plasma_vals[perm], True
    
    # 失败时返回简单置换
    perm = rng.permutation(n)
    return plasma_vals[perm], False


def fit_linear_regressor(
    plasma_emb: np.ndarray,
    plasma_vals: np.ndarray,
    plasma_mask: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    从 plasma_emb 线性回归预测 plasma_vals
    
    Returns:
        plasma_hat: (N,) - 预测值
        spearman_rho: float - Spearman 相关系数
    """
    # 过滤有效样本
    valid_mask = plasma_mask.astype(bool) & np.isfinite(plasma_vals)
    
    if valid_mask.sum() < 2:
        return np.full_like(plasma_vals, np.nan), np.nan
    
    x = plasma_emb[valid_mask]
    y = plasma_vals[valid_mask]
    
    # 拟合
    reg = LinearRegression()
    reg.fit(x, y)
    
    # 预测（所有样本）
    plasma_hat = reg.predict(plasma_emb)
    
    # 计算 Spearman ρ（仅有效样本）
    y_hat_valid = plasma_hat[valid_mask]
    rho, _ = spearmanr(y, y_hat_valid)
    
    return plasma_hat, float(rho)


def _to_percentile_rank(values: np.ndarray) -> np.ndarray:
    """
    将数值转换为百分位排名 [0, 1]，用于着色。
    与 Spearman ρ（排名相关）语义一致。
    """
    ranks = rankdata(values, method="average")
    return (ranks - 1) / max(len(ranks) - 1, 1)


def plot_1x3_comparison(
    umap_coords: np.ndarray,
    plasma_real: np.ndarray,
    plasma_shuffled: np.ndarray,
    plasma_hat: np.ndarray,
    rho: float,
    class_name: str,
    split: str,
    output_path: str,
    dpi: int = 150,
    color_norm: str = "rank",
):
    """
    绘制 2×3 子图：
    - 第一行 1×3 UMAP 散点图（真实 / shuffle / 预测），按 color_norm 着色
    - 第二行 1×3 直方图，展示对应 pT217_F 值分布（揭示偏斜度）
    
    Args:
        color_norm: 颜色归一化方式
            - 'rank': 百分位排名着色（默认，与 Spearman ρ 一致）
            - 'linear': 经典 min-max 线性映射（原始行为）
            - 'percentile_clip': 用 2-98 百分位截断后线性映射
    """
    cmap = plt.cm.viridis
    
    # ---- 确定颜色数据和归一化方式 ----
    data_arrays = [plasma_real, plasma_shuffled, plasma_hat]
    color_label = "pT217_F"
    
    if color_norm == "rank":
        # 每个子图独立做排名映射
        color_arrays = [_to_percentile_rank(arr) for arr in data_arrays]
        norm = Normalize(vmin=0.0, vmax=1.0)
        color_label = "pT217_F (percentile rank)"
    elif color_norm == "percentile_clip":
        # 基于真实值的 2-98 百分位做截断
        all_vals = np.concatenate(data_arrays)
        p2 = np.nanpercentile(all_vals, 2)
        p98 = np.nanpercentile(all_vals, 98)
        if abs(p98 - p2) < 1e-8:
            p2, p98 = all_vals.min(), all_vals.max()
        norm = Normalize(vmin=p2, vmax=p98)
        color_arrays = data_arrays
        color_label = f"pT217_F (clipped [{p2:.3f}, {p98:.3f}])"
    else:  # linear
        vmin = np.nanmin([arr.min() for arr in data_arrays])
        vmax = np.nanmax([arr.max() for arr in data_arrays])
        norm = Normalize(vmin=vmin, vmax=vmax)
        color_arrays = data_arrays
    
    # ---- 创建 2×3 布局 ----
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.35, wspace=0.3)
    
    titles = [
        f"{class_name} {split} - Real pT217_F",
        f"{class_name} {split} - Shuffled pT217_F",
        f"{class_name} {split} - Predicted pT217_F (ρ={rho:.3f})",
    ]
    hist_labels = ["Real", "Shuffled", "Predicted"]
    hist_colors = ["#2196F3", "#FF9800", "#4CAF50"]
    
    scatter_axes = []
    for col_idx in range(3):
        ax = fig.add_subplot(gs[0, col_idx])
        scatter_axes.append(ax)
        sc = ax.scatter(
            umap_coords[:, 0], umap_coords[:, 1],
            c=color_arrays[col_idx], cmap=cmap, norm=norm,
            s=20, alpha=0.7, edgecolors='none',
        )
        ax.set_title(titles[col_idx], fontsize=11)
        ax.set_xlabel("UMAP 1")
        if col_idx == 0:
            ax.set_ylabel("UMAP 2")
        ax.grid(alpha=0.3)
    
    # 共享 colorbar
    fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=scatter_axes, label=color_label,
        shrink=0.85, pad=0.02,
    )
    
    # ---- 第二行：直方图 ----
    for col_idx in range(3):
        ax_hist = fig.add_subplot(gs[1, col_idx])
        vals = data_arrays[col_idx]  # 始终用原始值画直方图
        ax_hist.hist(
            vals, bins=min(50, max(10, len(vals) // 5)),
            color=hist_colors[col_idx], alpha=0.8, edgecolor='white', linewidth=0.5,
        )
        ax_hist.set_xlabel("pT217_F value")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title(f"{hist_labels[col_idx]} Distribution", fontsize=10)
        # 标注统计量
        med = np.median(vals)
        sk = float(skew(vals))
        ax_hist.axvline(med, color='red', linestyle='--', linewidth=1.2, label=f'median={med:.3f}')
        ax_hist.text(
            0.97, 0.95, f'skew={sk:.2f}\nn={len(vals)}',
            transform=ax_hist.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
        )
        ax_hist.legend(fontsize=7, loc='upper left')
        ax_hist.grid(alpha=0.2)
    
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"[Plot] Saved: {output_path}")


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Plasma Alignment Visualization")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--ckpt", type=str, required=True, help="训练好的 checkpoint 路径")
    parser.add_argument("--csv", type=str, default=None, help="数据 CSV（覆盖配置）")
    parser.add_argument("--cache_dir", type=str, default=None, help="缓存目录（覆盖配置）")
    parser.add_argument("--val_split_json", type=str, default="fixed_split.json",
                        help="固定 split JSON 路径")
    parser.add_argument("--plasma_key", type=str, default="pT217_F",
                        help="目标 plasma key（需在 selected_keys 中）")
    parser.add_argument("--output_dir", type=str, default="vis_plasma_output",
                        help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--batch_size", type=int, default=32, help="DataLoader batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--umap_n_neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--umap_min_dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--dpi", type=int, default=150, help="图像分辨率")
    parser.add_argument("--color_norm", type=str, default="rank",
                        choices=["rank", "linear", "percentile_clip"],
                        help="颜色归一化方式: rank(百分位排名,默认), linear(线性), percentile_clip(百分位截断)")
    parser.add_argument("--device", type=str, default=None, help="设备（如 cuda:0）")
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置
    config_path = script_dir / args.config
    config = load_config(config_path)
    
    # 设备
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 数据路径
    csv_path = args.csv or config["data"]["csv_path"]
    cache_dir = args.cache_dir or config["data"]["cache_dir"]
    
    # 解析 plasma 配置
    selected_plasma_keys, _ = resolve_plasma_config(config)
    
    # 查找目标 plasma key 的索引
    try:
        plasma_key_idx = selected_plasma_keys.index(args.plasma_key)
    except ValueError:
        raise ValueError(
            f"plasma_key '{args.plasma_key}' 不在 selected_keys 中: {selected_plasma_keys}"
        )
    
    print(f"\n{'='*60}")
    print(f"Plasma Alignment Visualization")
    print(f"{'='*60}")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Plasma key: {args.plasma_key} (index={plasma_key_idx})")
    print(f"Output dir: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # 加载模型
    print("Loading model...")
    model = load_model_and_checkpoint(config, args.ckpt, device)
    
    # 加载数据集
    print("Loading dataset...")
    class_names = config["classes"]["names"]
    diagnosis_csv = config.get("data", {}).get("diagnosis_csv", None)
    diagnosis_code_map = config.get("classes", {}).get("diagnosis_code_map", None)
    
    full_dataset = TAUPlasmaDataset(
        csv_path=csv_path,
        cache_dir=cache_dir,
        plasma_keys=selected_plasma_keys,
        class_names=class_names,
        diagnosis_csv=diagnosis_csv,
        diagnosis_code_map=diagnosis_code_map,
    )
    print(f"Total samples: {len(full_dataset)}")
    
    # 划分 train/val
    val_ratio = config["data"].get("val_ratio", 0.1)
    train_indices, val_indices = load_fixed_split(
        full_dataset=full_dataset,
        val_split_json=args.val_split_json,
        seed=args.seed,
        val_ratio=val_ratio,
    )
    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    
    # 输出目录
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # 为每个类别生成可视化
    for class_name in class_names:
        class_id = class_names.index(class_name)
        print(f"\n{'='*60}")
        print(f"Processing class: {class_name} (id={class_id})")
        print(f"{'='*60}")
        
        # 为 train 和 val 分别处理
        for split_name, split_indices in [("train", train_indices), ("val", val_indices)]:
            print(f"\n[{split_name.upper()}] Processing...")
            
            # 筛选当前类别的样本
            class_split_indices = [
                idx for idx in split_indices
                if full_dataset.samples[idx]["diagnosis"] == class_name
            ]
            
            if len(class_split_indices) == 0:
                print(f"[WARN] No samples for {class_name} in {split_name} split, skipping.")
                continue
            
            print(f"  Samples: {len(class_split_indices)}")
            
            # 构建子数据集
            subset = Subset(full_dataset, class_split_indices)
            dataloader = DataLoader(
                subset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_fn,
            )
            
            # 提取 plasma_emb
            plasma_emb, labels, plasma_vals, plasma_mask, subject_ids = extract_plasma_embeddings(
                model=model,
                dataloader=dataloader,
                device=device,
                plasma_key_idx=plasma_key_idx,
            )
            
            # 统计有效样本
            valid_mask = plasma_mask.astype(bool) & np.isfinite(plasma_vals)
            n_valid = valid_mask.sum()
            print(f"  Valid plasma samples: {n_valid}/{len(plasma_vals)}")
            
            if n_valid < 5:
                print(f"[WARN] Too few valid samples ({n_valid}), skipping.")
                continue
            
            # UMAP 降维（只对有效样本）
            print(f"  Running UMAP (n_neighbors={args.umap_n_neighbors}, min_dist={args.umap_min_dist})...")
            umap_model = UMAP(
                n_neighbors=args.umap_n_neighbors,
                min_dist=args.umap_min_dist,
                random_state=args.seed,
                n_components=2,
            )
            umap_coords = umap_model.fit_transform(plasma_emb[valid_mask])
            
            # Within-class shuffle
            rng = np.random.default_rng(args.seed)
            plasma_vals_valid = plasma_vals[valid_mask]
            plasma_shuffled_valid, shuffle_success = within_class_shuffle(
                plasma_vals_valid, rng
            )
            
            # 线性回归预测
            plasma_hat_valid, spearman_rho = fit_linear_regressor(
                plasma_emb[valid_mask],
                plasma_vals_valid,
                np.ones_like(plasma_vals_valid, dtype=bool),
            )
            
            print(f"  Spearman ρ: {spearman_rho:.4f}")
            print(f"  Shuffle success: {shuffle_success}")
            
            # 绘图
            class_output_dir = output_root / class_name / split_name
            class_output_dir.mkdir(parents=True, exist_ok=True)
            
            plot_path = class_output_dir / "1x3.png"
            plot_1x3_comparison(
                umap_coords=umap_coords,
                plasma_real=plasma_vals_valid,
                plasma_shuffled=plasma_shuffled_valid,
                plasma_hat=plasma_hat_valid,
                rho=spearman_rho,
                class_name=class_name,
                split=split_name,
                output_path=str(plot_path),
                dpi=args.dpi,
                color_norm=args.color_norm,
            )
            
            # 计算 plasma 值分布统计
            plasma_val_stats = {
                "min": float(np.min(plasma_vals_valid)),
                "max": float(np.max(plasma_vals_valid)),
                "median": float(np.median(plasma_vals_valid)),
                "mean": float(np.mean(plasma_vals_valid)),
                "std": float(np.std(plasma_vals_valid)),
                "p5": float(np.percentile(plasma_vals_valid, 5)),
                "p25": float(np.percentile(plasma_vals_valid, 25)),
                "p75": float(np.percentile(plasma_vals_valid, 75)),
                "p95": float(np.percentile(plasma_vals_valid, 95)),
                "skewness": float(skew(plasma_vals_valid)),
            }
            
            # K=1 softmax 退化检查
            n_plasma_keys = len(selected_plasma_keys)
            k1_warning = None
            if n_plasma_keys == 1:
                k1_warning = (
                    "K=1: softmax over single key always outputs weight=1.0. "
                    "Plasma value magnitude has NO effect on plasma_emb. "
                    "The high Spearman rho likely reflects image-to-biomarker correlation "
                    "(via ContextNet) rather than direct plasma value encoding. "
                    "Consider using multiple plasma keys or a different aggregation."
                )
                if split_name == "train" and class_id == 0:  # 只打印一次
                    print(f"\n  [WARNING] {k1_warning}")
            
            # 保存元信息
            meta = {
                "class_name": class_name,
                "class_id": class_id,
                "split": split_name,
                "n_samples": len(class_split_indices),
                "n_valid_plasma": int(n_valid),
                "plasma_key": args.plasma_key,
                "plasma_key_idx": plasma_key_idx,
                "n_plasma_keys": n_plasma_keys,
                "spearman_rho": float(spearman_rho) if np.isfinite(spearman_rho) else None,
                "shuffle_success": shuffle_success,
                "color_norm": args.color_norm,
                "plasma_val_stats": plasma_val_stats,
                "umap_params": {
                    "n_neighbors": args.umap_n_neighbors,
                    "min_dist": args.umap_min_dist,
                    "random_state": args.seed,
                },
                "seed": args.seed,
                "checkpoint": args.ckpt,
            }
            if k1_warning:
                meta["warning_k1_degeneracy"] = k1_warning
            meta_path = class_output_dir / "meta.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            
            print(f"  Saved: {plot_path}")
            print(f"  Saved: {meta_path}")
    
    print(f"\n{'='*60}")
    print(f"Visualization completed!")
    print(f"Output directory: {output_root}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
