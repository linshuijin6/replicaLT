"""
adapter_v2/run_probe_and_retrieval.py
========================================
主入口：对三种 image_emb 方案进行分类探针 + CoCoOp 条件文本分类评估，
最终输出可对比的结果表（控制台 + CSV/JSON）。

用法示例：
  cd adapter_v2
  python run_probe_and_retrieval.py \\
      --ckpt /path/to/best_model.pt \\
      --csv  /path/to/pairs_180d_dx_plasma_90d_matched.csv \\
      --cache_dir /path/to/cache \\
      --num_classes 3 \\
      --batch_size 32 \\
      --probe_lr 1e-3 \\
      --probe_epochs 100 \\
      --seed 42 \\
      --device cuda:0 \\
      --output_dir ./probe_results

结果表字段（CSV 输出，统一 4 位小数）：
  A 线性分类探针：
    A_balanced_acc
    A_macro_f1
    B Class 路条件文本分类（context_net 注入 ctx）：
        B_balanced_acc
        B_macro_f1
    C Plasma 路条件文本分类（pT217_F 四桶）：
        C_balanced_acc
        C_macro_f1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── 确保 adapter_v2 在 sys.path ──────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ── 确保 CLIP-MRI2PET 在 sys.path ─────────────────────────────────────────────
_CLIP_ROOT = _HERE.parent.parent / "CLIP-MRI2PET"
for _p in [str(_CLIP_ROOT), str(_CLIP_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── 本地模块 ──────────────────────────────────────────────────────────────────
from dataset import (
    TAUPlasmaDataset,
    SubjectBatchSampler,
    collate_fn,
    split_by_subject,
)
from embedding_extractor import EmbeddingExtractor
from linear_probe import LinearProbeTrainer
from models import CoCoOpTAUModel


# ============================================================================
# 工具函数
# ============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def _fmt_cell(v) -> str:
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "nan"
        return f"{float(v):.4f}"
    return str(v)


def print_table(rows: List[Dict], title: str = "Results") -> None:
    """在控制台打印对齐的结果表（浮点统一 4 位小数）"""
    if not rows:
        return
    keys = list(rows[0].keys())
    formatted_rows = [{k: _fmt_cell(r.get(k, "")) for k in keys} for r in rows]
    widths = {k: max(len(k), max(len(fr.get(k, "")) for fr in formatted_rows)) for k in keys}
    sep = "+-" + "-+-".join("-" * widths[k] for k in keys) + "-+"
    header = "| " + " | ".join(k.ljust(widths[k]) for k in keys) + " |"

    print(f"\n{'='*len(sep)}")
    print(f"  {title}")
    print(sep)
    print(header)
    print(sep)
    for fr in formatted_rows:
        line = "| " + " | ".join(fr.get(k, "").ljust(widths[k]) for k in keys) + " |"
        print(line)
    print(sep)


def print_grouped_results(rows: List[Dict], title: str) -> None:
    if not rows:
        return
    id_cols = [k for k in ["seed", "n_seeds", "emb_method"] if k in rows[0]]

    row_keys = set(rows[0].keys())
    is_summary = any(k.endswith("_mean") for k in row_keys)

    base_groups = [
        (
            "A 线性分类探针",
            ["A_balanced_acc", "A_macro_f1"],
        ),
        (
            "B Class 路条件文本分类",
            ["B_balanced_acc", "B_macro_f1"],
        ),
        (
            "C Plasma 路条件文本分类（pT217_F四桶）",
            ["C_balanced_acc", "C_macro_f1"],
        ),
    ]

    def _fmt_mean_std(mean_v, std_v) -> str:
        try:
            m = float(mean_v)
        except Exception:
            m = float("nan")
        try:
            s = float(std_v)
        except Exception:
            s = float("nan")

        if np.isfinite(m) and np.isfinite(s):
            return f"{m:.4f}±{s:.4f}"
        if np.isfinite(m):
            return f"{m:.4f}"
        return "nan"

    for g_title, metrics in base_groups:
        if is_summary:
            part_rows = []
            for r in rows:
                out = {c: r.get(c, "") for c in id_cols}
                for m in metrics:
                    mean_k = f"{m}_mean"
                    std_k = f"{m}_std"
                    if mean_k in row_keys or std_k in row_keys:
                        out[m] = _fmt_mean_std(r.get(mean_k, float("nan")), r.get(std_k, float("nan")))
                part_rows.append(out)
        else:
            cols = id_cols + [m for m in metrics if m in row_keys]
            part_rows = [{c: r.get(c, "") for c in cols} for r in rows]

        print_table(part_rows, title=f"{title} | {g_title}")


# ============================================================================
# 数据加载
# ============================================================================

def build_data_splits(
    csv_path: str | Path,
    cache_dir: str | Path,
    batch_size: int,
    val_ratio: float,
    seed: int,
    class_names: List[str],
    plasma_keys: List[str],
    num_workers: int,
    diagnosis_code_map: Optional[Dict] = None,
    val_split_json: Optional[str] = None,
):
    """
    构建 train/val DataLoader，返回额外元信息。

    Returns
    -------
    train_loader, val_loader, plasma_stats
    """
    skip_cache_set: set = set()

    # 加载完整数据集（用于计算 plasma stats 和 subject-level split）
    full_ds = TAUPlasmaDataset(
        csv_path=csv_path,
        cache_dir=cache_dir,
        plasma_keys=plasma_keys,
        class_names=class_names,
        diagnosis_code_map=diagnosis_code_map,
        skip_cache_set=skip_cache_set,
    )
    plasma_stats = full_ds.plasma_stats

    # Subject-level split（支持固定 split JSON）
    train_idx, val_idx = split_by_subject(full_ds, val_ratio=val_ratio, seed=seed)
    if val_split_json is not None:
        split_path = Path(val_split_json)
        subject_to_indices: Dict[str, List[int]] = {}
        for idx, sample in enumerate(full_ds.samples):
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
                all_subjects = set(subject_to_indices.keys())
                val_subjects = all_subjects - train_subjects
            if not train_subjects:
                all_subjects = set(subject_to_indices.keys())
                train_subjects = all_subjects - val_subjects

            unknown = (train_subjects | val_subjects) - set(subject_to_indices.keys())
            if unknown:
                print(f"[Split] WARN: split JSON 包含当前数据集中不存在的 subject: {sorted(list(unknown))[:5]}...")

            train_idx, val_idx = [], []
            for sid, indices in subject_to_indices.items():
                if sid in val_subjects:
                    val_idx.extend(indices)
                elif sid in train_subjects:
                    train_idx.extend(indices)

            if len(train_idx) == 0 or len(val_idx) == 0:
                raise ValueError(
                    f"split JSON 应用后出现空集合: train={len(train_idx)}, val={len(val_idx)}"
                )
            print(f"[Split] Loaded fixed split from: {split_path}")
            print(f"[Split] Subjects train={len(train_subjects)} val={len(val_subjects)}")
        else:
            train_subjects = sorted({full_ds.samples[i]["subject_id"] for i in train_idx})
            val_subjects = sorted({full_ds.samples[i]["subject_id"] for i in val_idx})
            split_path.parent.mkdir(parents=True, exist_ok=True)
            with open(split_path, "w") as f:
                json.dump(
                    {
                        "seed": seed,
                        "val_ratio": val_ratio,
                        "train_subjects": train_subjects,
                        "val_subjects": val_subjects,
                    },
                    f,
                    indent=2,
                )
            print(f"[Split] Saved generated split to: {split_path}")

    print(f"[Data] Train samples={len(train_idx)}, Val samples={len(val_idx)}")

    # 重建子集数据集
    train_ds = TAUPlasmaDataset(
        csv_path=csv_path,
        cache_dir=cache_dir,
        plasma_keys=plasma_keys,
        class_names=class_names,
        plasma_stats=plasma_stats,
        subset_indices=train_idx,
        diagnosis_code_map=diagnosis_code_map,
        skip_cache_set=skip_cache_set,
    )
    val_ds = TAUPlasmaDataset(
        csv_path=csv_path,
        cache_dir=cache_dir,
        plasma_keys=plasma_keys,
        class_names=class_names,
        plasma_stats=plasma_stats,
        subset_indices=val_idx,
        diagnosis_code_map=diagnosis_code_map,
        skip_cache_set=skip_cache_set,
    )

    train_sampler = SubjectBatchSampler(
        train_ds, batch_size, shuffle=True, drop_last=False, seed=seed
    )
    val_sampler = SubjectBatchSampler(
        val_ds, batch_size, shuffle=False, drop_last=False, seed=seed
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, plasma_stats


def _to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _to_float4(x) -> float:
    v = _to_float(x)
    if np.isfinite(v):
        return float(np.round(v, 4))
    return float("nan")


def aggregate_multi_seed(
    rows: List[Dict],
    metric_keys: List[str],
) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = {}
    for row in rows:
        grouped.setdefault(row["emb_method"], []).append(row)

    agg_rows: List[Dict] = []
    for emb_method, em_rows in grouped.items():
        out = {
            "emb_method": emb_method,
            "n_seeds": len(em_rows),
        }
        for k in metric_keys:
            vals = np.array([_to_float(r.get(k, float("nan"))) for r in em_rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                out[f"{k}_mean"] = float("nan")
                out[f"{k}_std"] = float("nan")
            else:
                out[f"{k}_mean"] = float(np.round(vals.mean(), 4))
                out[f"{k}_std"] = float(np.round(vals.std(ddof=0), 4))
        agg_rows.append(out)

    order = {"mean_pool": 0, "attn_pool": 1, "attn_pool_proj": 2}
    agg_rows.sort(key=lambda x: (order.get(x["emb_method"], 999), x["emb_method"]))
    return agg_rows


def _compute_pt217_quantile_thresholds(
    train_plasma_vals: torch.Tensor,
    train_plasma_mask: torch.Tensor,
    pt217_idx: int = 1,
) -> Tuple[float, float, float]:
    if train_plasma_vals.ndim != 2 or train_plasma_mask.ndim != 2:
        raise ValueError("train_plasma_vals/train_plasma_mask 必须是二维张量")
    if pt217_idx < 0 or pt217_idx >= train_plasma_vals.shape[1]:
        raise ValueError(f"pt217_idx 越界: {pt217_idx}")

    values = train_plasma_vals[:, pt217_idx]
    mask = train_plasma_mask[:, pt217_idx].bool()
    valid_values = values[mask]
    if valid_values.numel() < 8:
        raise ValueError(f"pT217_F 有效训练样本过少: {valid_values.numel()}")

    q1, q2, q3 = torch.quantile(valid_values.float(), torch.tensor([0.25, 0.5, 0.75]))
    return float(q1.item()), float(q2.item()), float(q3.item())


def _pt217_bucket_labels(
    plasma_vals: torch.Tensor,
    plasma_mask: torch.Tensor,
    thresholds: Tuple[float, float, float],
    pt217_idx: int = 1,
) -> torch.Tensor:
    labels = torch.full((plasma_vals.shape[0],), -1, dtype=torch.long)
    vals = plasma_vals[:, pt217_idx]
    valid = plasma_mask[:, pt217_idx].bool()
    if valid.any():
        q1, q2, q3 = thresholds
        b = torch.bucketize(vals[valid].float(), boundaries=torch.tensor([q1, q2, q3], dtype=torch.float32))
        labels[valid] = b.long()
    return labels


class ContextConditionedClassifier:
    """
    复用 CoCoOp checkpoint 的 context_net + 文本分支，
    对给定 image_emb 计算两路 logits：
      - class 路（3 类）
      - plasma pT217 四桶路（4 类）
    """

    def __init__(
        self,
        ckpt_path: str | Path,
        class_names: List[str],
        device: torch.device,
    ):
        self.device = device
        self.model = CoCoOpTAUModel(
            biomedclip_path="",
            class_names=class_names,
        )
        state = torch.load(str(ckpt_path), map_location="cpu")
        if "model_state_dict" in state:
            sd = state["model_state_dict"]
        elif "state_dict" in state:
            sd = state["state_dict"]
        else:
            sd = state
        self.model.load_state_dict(sd, strict=False)
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.pt217_bucket_prompts = [
            "plasma pT217 level is in the lowest quartile",
            "plasma pT217 level is in the second quartile",
            "plasma pT217 level is in the third quartile",
            "plasma pT217 level is in the highest quartile",
        ]

    @torch.no_grad()
    def logits_class(self, image_embs: torch.Tensor) -> torch.Tensor:
        return self._forward_logits(image_embs=image_embs, texts=self.model.class_prompts, route="class")

    @torch.no_grad()
    def logits_plasma_bucket(self, image_embs: torch.Tensor) -> torch.Tensor:
        return self._forward_logits(
            image_embs=image_embs,
            texts=self.pt217_bucket_prompts,
            route="plasma",
        )

    @torch.no_grad()
    def _forward_logits(self, image_embs: torch.Tensor, texts: List[str], route: str) -> torch.Tensor:
        img = image_embs.to(self.device).float()
        B, D = img.shape
        if D != self.model.D_pool:
            raise ValueError(f"image_emb 维度必须为 {self.model.D_pool}，当前 {D}")

        dtype = self.model.base_ctx_class.dtype
        g = img.to(dtype)
        img_norm = F.normalize(img, dim=-1)

        t_ctx_flat = self.model.context_net(g)
        t_ctx = t_ctx_flat.view(B, self.model.ctx_len, self.model.ctx_dim)

        if route == "class":
            base_ctx = self.model.base_ctx_class
            proj = self.model.proj_class
        elif route == "plasma":
            base_ctx = self.model.base_ctx_plasma
            proj = self.model.proj_plasma
        else:
            raise ValueError(f"未知 route: {route}")

        ctx = base_ctx.unsqueeze(0) + t_ctx
        prompts, token_ids = self.model._build_prompts_with_context(texts=texts, context=ctx, device=self.device)
        text_features = self.model.text_encoder(prompts, token_ids)
        text_embs = proj(text_features)
        text_embs = F.normalize(text_embs, dim=-1)

        scale = self.model.logit_scale.exp().clamp(max=100.0).float()
        logits = scale * torch.einsum("bd,bcd->bc", img_norm, text_embs.float())
        return logits.cpu()


def run_one_seed(args: argparse.Namespace, seed: int, output_dir: Path) -> Tuple[List[Dict], Dict]:
    set_seed(seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Config] Device: {device}  Seed: {seed}")
    print(f"[Config] Checkpoint: {args.ckpt}")

    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = args.class_names.split(",")
    plasma_keys = ["AB42_AB40_F", "pT217_F", "pT217_AB42_F", "NfL_Q", "GFAP_Q"]
    diagnosis_code_map = {1: "CN", 2: "MCI", 3: "AD"}

    print("\n" + "=" * 60)
    print("Step 1/3  构建数据集")
    print("=" * 60)
    train_loader, val_loader, plasma_stats = build_data_splits(
        csv_path=args.csv,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=seed,
        class_names=class_names,
        plasma_keys=plasma_keys,
        num_workers=args.num_workers,
        diagnosis_code_map=diagnosis_code_map,
        val_split_json=args.val_split_json,
    )

    print("\n" + "=" * 60)
    print("Step 2/3  提取 embedding（来自预缓存 .vision.pt）")
    print("=" * 60)

    extractor = EmbeddingExtractor(
        ckpt_path=args.ckpt,
        device=device,
    )

    print("[EmbeddingExtractor] 提取训练集 embedding…")
    train_embs = extractor.extract_from_dataloader(train_loader, device=device)

    print("[EmbeddingExtractor] 提取验证集 embedding…")
    val_embs = extractor.extract_from_dataloader(val_loader, device=device)

    for split_name, emb_dict in [("train", train_embs), ("val", val_embs)]:
        labels = emb_dict["labels"]
        valid = labels[labels >= 0]
        for cid, cname in enumerate(class_names):
            n = (valid == cid).sum().item()
            print(f"  [{split_name}] {cname}: {n} samples")

    print("\n" + "=" * 60)
    print("Step 3/3  Linear Probe + Context-Conditioned Classifiers")
    print("=" * 60)

    ctx_classifier = ContextConditionedClassifier(
        ckpt_path=args.ckpt,
        class_names=class_names,
        device=device,
    )

    val_labels = val_embs["labels"].long()

    if "plasma_vals" not in train_embs or "plasma_mask" not in train_embs:
        raise ValueError("train_embs 缺少 plasma_vals/plasma_mask，无法计算 pT217 四桶阈值")
    if "plasma_vals" not in val_embs or "plasma_mask" not in val_embs:
        raise ValueError("val_embs 缺少 plasma_vals/plasma_mask，无法计算 pT217 四桶标签")

    pt217_thresholds = _compute_pt217_quantile_thresholds(
        train_plasma_vals=train_embs["plasma_vals"],
        train_plasma_mask=train_embs["plasma_mask"],
        pt217_idx=1,
    )
    val_pt217_labels = _pt217_bucket_labels(
        plasma_vals=val_embs["plasma_vals"],
        plasma_mask=val_embs["plasma_mask"],
        thresholds=pt217_thresholds,
        pt217_idx=1,
    )
    print(
        f"[pT217 Quartiles] train thresholds: "
        f"Q1={pt217_thresholds[0]:.4f}, Q2={pt217_thresholds[1]:.4f}, Q3={pt217_thresholds[2]:.4f}"
    )

    EMB_SCHEMES = [
        ("mean_pool", "mean_pool_512", 768, "mean_pool"),
        ("attn_pool", "attn_pool", 512, "attn_pool"),
        ("attn_pool_proj", "attn_pool_proj", 512, "attn_pool_proj"),
    ]

    results_rows: List[Dict] = []

    train_labels_all = train_embs["labels"]
    class_weights = None
    if args.use_class_weight:
        class_weights = LinearProbeTrainer.compute_class_weights(
            train_labels_all, num_classes=args.num_classes
        )
        print(f"[ClassWeight] {class_weights.tolist()}")

    for cls_key, ret_key, cls_dim, scheme_name in EMB_SCHEMES:
        print(f"\n{'─'*60}")
        print(f">>> Scheme: {scheme_name}")
        print(f"    cls_emb={cls_key}({cls_dim}-dim)  ret_emb={ret_key}(512-dim)")
        print(f"{'─'*60}")

        set_seed(seed)

        trainer = LinearProbeTrainer(
            emb_dim=cls_dim,
            num_classes=args.num_classes,
            lr=args.probe_lr,
            epochs=args.probe_epochs,
            batch_size=args.probe_batch_size,
            device=device,
            class_weights=class_weights,
            patience=args.patience,
            save_dir=output_dir / "checkpoints",
            emb_key=scheme_name,
            seed=seed,
        )

        history = trainer.fit(
            train_embeddings=train_embs[cls_key],
            train_labels=train_embs["labels"],
            val_embeddings=val_embs[cls_key],
            val_labels=val_embs["labels"],
        )

        cls_metrics = trainer.evaluate(
            embeddings=val_embs[cls_key],
            labels=val_embs["labels"],
        )
        print(
            f"  [A-CLS] bacc={cls_metrics['balanced_accuracy']:.4f} "
            f"f1={cls_metrics['macro_f1']:.4f}"
        )

        ret_image_embs = val_embs[ret_key]
        b_metrics: Dict[str, float] = {"balanced_accuracy": float("nan"), "macro_f1": float("nan")}
        c_metrics: Dict[str, float] = {"balanced_accuracy": float("nan"), "macro_f1": float("nan")}

        if ret_image_embs.shape[-1] == 512:
            valid_b = val_labels >= 0
            if valid_b.any():
                logits_b = ctx_classifier.logits_class(ret_image_embs[valid_b])
                b_eval = LinearProbeTrainer.compute_metrics(logits=logits_b, labels=val_labels[valid_b].cpu())
                b_metrics["balanced_accuracy"] = b_eval["balanced_accuracy"]
                b_metrics["macro_f1"] = b_eval["macro_f1"]
                print(
                    f"  [B-Class] bacc={b_metrics['balanced_accuracy']:.4f} "
                    f"f1={b_metrics['macro_f1']:.4f}"
                )
            else:
                print(f"  [{scheme_name}] WARN: 无有效诊断标签样本，跳过 B 路")

            valid_c = val_pt217_labels >= 0
            if valid_c.any():
                logits_c = ctx_classifier.logits_plasma_bucket(ret_image_embs[valid_c])
                c_eval = LinearProbeTrainer.compute_metrics(logits=logits_c, labels=val_pt217_labels[valid_c].cpu())
                c_metrics["balanced_accuracy"] = c_eval["balanced_accuracy"]
                c_metrics["macro_f1"] = c_eval["macro_f1"]
                print(
                    f"  [C-pT217Q4] bacc={c_metrics['balanced_accuracy']:.4f} "
                    f"f1={c_metrics['macro_f1']:.4f}"
                )
            else:
                print(f"  [{scheme_name}] WARN: 无有效 pT217 标签样本，跳过 C 路")
        else:
            print(f"  [{scheme_name}] SKIP B/C: emb dim={ret_image_embs.shape[-1]} ≠ 512")

        row = {
            "seed": seed,
            "emb_method": scheme_name,
            # A 线性分类探针
            "A_balanced_acc": _to_float4(cls_metrics["balanced_accuracy"]),
            "A_macro_f1": _to_float4(cls_metrics["macro_f1"]),
            # B Class 路条件文本分类
            "B_balanced_acc": _to_float4(b_metrics["balanced_accuracy"]),
            "B_macro_f1": _to_float4(b_metrics["macro_f1"]),
            # C Plasma 路条件文本分类（pT217_F 四桶）
            "C_balanced_acc": _to_float4(c_metrics["balanced_accuracy"]),
            "C_macro_f1": _to_float4(c_metrics["macro_f1"]),
        }
        results_rows.append(row)

        history_path = output_dir / f"history_{scheme_name}.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    print_grouped_results(results_rows, title=f"Comparison Table (seed={seed})")
    return results_rows, {
        "seed": seed,
        "plasma_stats": plasma_stats,
        "pt217_quartile_thresholds": {
            "q1": _to_float4(pt217_thresholds[0]),
            "q2": _to_float4(pt217_thresholds[1]),
            "q3": _to_float4(pt217_thresholds[2]),
        },
    }


# ============================================================================
# 主流程
# ============================================================================

def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.seeds is not None and len(args.seeds) > 0:
        seeds = args.seeds
    else:
        seeds = [args.seed]

    all_rows: List[Dict] = []
    per_seed_meta: List[Dict] = []

    for i, seed in enumerate(seeds, start=1):
        if len(seeds) > 1:
            print("\n" + "#" * 80)
            print(f"Run {i}/{len(seeds)} | seed={seed}")
            print("#" * 80)
            run_output_dir = output_dir / f"seed_{seed}"
        else:
            run_output_dir = output_dir

        seed_rows, meta = run_one_seed(args=args, seed=seed, output_dir=run_output_dir)
        all_rows.extend(seed_rows)
        per_seed_meta.append(meta)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_csv_path = output_dir / f"results_{timestamp}.csv"
    all_json_path = output_dir / f"results_{timestamp}.json"

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(all_csv_path, index=False, float_format="%.4f")
    print(f"\n[Output] CSV → {all_csv_path}")

    metric_keys = [
        "A_balanced_acc",
        "A_macro_f1",
        "B_balanced_acc",
        "B_macro_f1",
        "C_balanced_acc",
        "C_macro_f1",
    ]

    if len(seeds) > 1:
        agg_rows = aggregate_multi_seed(all_rows, metric_keys=metric_keys)
        print_grouped_results(agg_rows, title=f"Multi-seed Summary ({len(seeds)} seeds)")
        agg_csv_path = output_dir / f"results_summary_{timestamp}.csv"
        pd.DataFrame(agg_rows).to_csv(agg_csv_path, index=False, float_format="%.4f")
        pd.DataFrame(agg_rows).to_csv(output_dir / "results_summary_latest.csv", index=False, float_format="%.4f")
        print(f"[Output] Summary CSV → {agg_csv_path}")
    else:
        print_grouped_results(all_rows, title="Comparison Table")
        agg_rows = []

    with open(all_json_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "config": vars(args),
                "seeds": seeds,
                "per_seed_meta": per_seed_meta,
                "results": all_rows,
                "summary": agg_rows,
            },
            f,
            indent=2,
        )
    print(f"[Output] JSON → {all_json_path}")

    df_all.to_csv(output_dir / "results_latest.csv", index=False, float_format="%.4f")
    print(f"[Output] Latest CSV → {output_dir / 'results_latest.csv'}")


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Linear Probe + CoCoOp Context-Conditioned Classification 对比实验"
    )

    # ── 必需参数（已设默认值，可直接运行）──────────────────────────────────
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/home/ssddata/linshuijin/replicaLT/adapter_v2/runs/checkpoints/best.pt",
        help="CoCoOpTAUModel 训练好的 checkpoint 路径（含 token_pool / proj_img / proj_plasma）",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="/mnt/nfsdata/nfsdata/lsj.14/replicaLT/adapter_finetune/gen_csv/pairs_180d_dx_plasma_90d_matched.csv",
        help="数据集 CSV 路径（pairs_180d_dx_plasma_90d_matched.csv 或类似文件）",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/mnt/nfsdata/nfsdata/lsj.14/ADNI_CSF_Adapter_cache",
        help="预缓存 .vision.pt 文件所在目录",
    )

    # ── 数据参数 ──────────────────────────────────────────────────────────
    parser.add_argument("--val_ratio",  type=float, default=0.1,  help="验证集比例")
    parser.add_argument("--val_split_json", type=str, default='fixed_split.json',
                        help="可选：固定 train/val 划分 JSON（包含 train_subjects/val_subjects）；不存在时自动生成并保存")
    parser.add_argument("--class_names", type=str, default="CN,MCI,AD",
                        help="类别名称，逗号分隔")
    parser.add_argument("--num_classes", type=int, default=3, help="类别数")
    parser.add_argument("--batch_size",  type=int, default=32,   help="DataLoader batch size")
    parser.add_argument("--num_workers", type=int, default=0,   help="DataLoader workers")

    # ── 线性探针超参 ──────────────────────────────────────────────────────
    parser.add_argument("--probe_lr",         type=float, default=1e-4, help="线性探针学习率")
    parser.add_argument("--probe_epochs",     type=int,   default=100,  help="最大训练 epoch")
    parser.add_argument("--probe_batch_size", type=int,   default=64,   help="LinearProbe mini-batch 大小")
    parser.add_argument("--patience",         type=int,   default=20,   help="Early stopping patience")
    parser.add_argument("--use_class_weight", action="store_true",
                        help="使用逆频率 class weight 处理类别不均衡")

    # ── 通用参数 ──────────────────────────────────────────────────────────
    parser.add_argument("--seed",       type=int,   default=42,        help="随机种子")
    parser.add_argument("--seeds",      type=int, nargs="+", default=[0,1,2,3,4],
                        help="多 seed 评估（如 --seeds 0 1 2 3 4）。若提供则覆盖 --seed")
    parser.add_argument("--device",     type=str,   default="cuda:4",  help="计算设备")
    parser.add_argument("--output_dir", type=str,   default="./probe_results",
                        help="输出目录（保存 CSV/JSON/checkpoint）")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
