"""
adapter_v2/linear_probe.py
===========================
LinearProbeTrainer：只训练线性分类头，冻结所有 encoder 组件。

输入：预提取的 embedding dict（来自 EmbeddingExtractor.extract_from_dataloader()）
输出：Accuracy、Balanced Accuracy、Macro-F1（二分类时加 AUROC）

设计要点：
  - 三种方案使用同一 subject-level split、同一超参，保证公平对比
  - 支持 class_weight 的 CrossEntropyLoss
  - 按 val macro-f1（或 balanced_acc）选最优 checkpoint
  - 每 epoch 打印 loss / acc / f1，训练过程透明可观察
  - TensorBoardX 可选（若不可用则跳过）
"""

from __future__ import annotations

import os
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        roc_auc_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ============================================================================
# 主类
# ============================================================================

class LinearProbeTrainer:
    """
    在预提取 embedding 上训练线性分类头。

    Parameters
    ----------
    emb_dim     : 输入 embedding 维度
    num_classes : 诊断类别数（默认 3: CN/MCI/AD）
    lr          : 学习率
    epochs      : 最大训练 epoch
    batch_size  : mini-batch 大小（针对 embedding tensor）
    device      : 计算设备
    class_weights : (num_classes,) 可选权重；None 表示均匀
    patience    : Early stopping patience（按 val macro-f1）；0 = 关闭
    save_dir    : 保存最优 checkpoint 的目录；None = 不保存
    emb_key     : embedding 类型标识，用于日志和文件名
    seed        : 随机种子
    """

    def __init__(
        self,
        emb_dim: int,
        num_classes: int = 3,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 64,
        device: torch.device | str = "cpu",
        class_weights: Optional[torch.Tensor] = None,
        patience: int = 20,
        save_dir: Optional[str | Path] = None,
        emb_key: str = "mean_pool",
        seed: int = 42,
    ):
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device) if isinstance(device, str) else device
        self.patience = patience
        self.save_dir = Path(save_dir) if save_dir else None
        self.emb_key = emb_key
        self.seed = seed

        # 线性分类头
        torch.manual_seed(seed)
        self.head = nn.Linear(emb_dim, num_classes).to(self.device)

        # 损失函数
        if class_weights is not None:
            w = class_weights.float().to(self.device)
        else:
            w = None
        self.criterion = nn.CrossEntropyLoss(weight=w)

        # 优化器（在 fit() 时初始化，方便多次复用）
        self.optimizer: Optional[torch.optim.Optimizer] = None

        # 最优模型权重（按 val macro-f1）
        self._best_head_state: Optional[dict] = None
        self._best_val_metric: float = -1.0

    # -------------------------------------------------------------------------
    def _make_loader(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        shuffle: bool,
    ) -> DataLoader:
        """构建 TensorDataset → DataLoader"""
        ds = TensorDataset(embeddings.float(), labels.long())
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    # -------------------------------------------------------------------------
    def fit(
        self,
        train_embeddings: torch.Tensor,   # (N_train, D)
        train_labels: torch.Tensor,       # (N_train,)
        val_embeddings: torch.Tensor,     # (N_val, D)
        val_labels: torch.Tensor,         # (N_val,)
    ) -> Dict[str, List[float]]:
        """
        训练线性分类头。

        Parameters
        ----------
        train_embeddings, train_labels : 训练集 embedding 与标签
        val_embeddings, val_labels     : 验证集 embedding 与标签

        Returns
        -------
        history: {"train_loss": [...], "val_macro_f1": [...], ...}
        """
        # ── 过滤 label=-1 的无效样本 ────────────────────────────────────
        train_mask = train_labels >= 0
        val_mask   = val_labels >= 0
        train_embeddings = train_embeddings[train_mask]
        train_labels     = train_labels[train_mask]
        val_embeddings   = val_embeddings[val_mask]
        val_labels       = val_labels[val_mask]

        # ── 重置 head ────────────────────────────────────────────────────
        torch.manual_seed(self.seed)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self._best_head_state = None
        self._best_val_metric = -1.0

        # ── 优化器 ───────────────────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            self.head.parameters(), lr=self.lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=self.lr * 0.1
        )

        train_loader = self._make_loader(train_embeddings, train_labels, shuffle=True)
        val_loader   = self._make_loader(val_embeddings, val_labels, shuffle=False)

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_balanced_acc": [],
            "val_macro_f1": [],
        }

        no_improve = 0

        for epoch in range(1, self.epochs + 1):
            # ── Train ──────────────────────────────────────────────────
            self.head.train()
            epoch_loss = 0.0
            all_preds: List[int] = []
            all_labels_: List[int] = []

            for emb_batch, lbl_batch in train_loader:
                emb_batch = emb_batch.to(self.device)
                lbl_batch = lbl_batch.to(self.device)

                self.optimizer.zero_grad()
                logits = self.head(emb_batch)
                loss = self.criterion(logits, lbl_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.head.parameters(), 5.0)
                self.optimizer.step()

                epoch_loss += loss.item() * len(lbl_batch)
                preds = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels_.extend(lbl_batch.cpu().tolist())

            scheduler.step()

            avg_train_loss = epoch_loss / max(len(train_embeddings), 1)
            train_acc = accuracy_score(all_labels_, all_preds) if SKLEARN_AVAILABLE else \
                        (sum(p == l for p, l in zip(all_preds, all_labels_)) / max(len(all_labels_), 1))

            # ── Validate ───────────────────────────────────────────────
            val_metrics = self._evaluate_loader(val_loader)

            history["train_loss"].append(avg_train_loss)
            history["train_acc"].append(float(train_acc))
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_balanced_acc"].append(val_metrics["balanced_accuracy"])
            history["val_macro_f1"].append(val_metrics["macro_f1"])

            # ── Early stopping ─────────────────────────────────────────
            cur_metric = val_metrics["macro_f1"]
            improved   = cur_metric > self._best_val_metric + 1e-5
            if improved:
                self._best_val_metric = cur_metric
                self._best_head_state = copy.deepcopy(self.head.state_dict())
                no_improve = 0
                best_marker = " ← best"
            else:
                no_improve += 1
                best_marker = ""

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"  [{self.emb_key}] Epoch {epoch:3d}/{self.epochs} | "
                    f"loss={avg_train_loss:.4f} acc={train_acc:.4f} | "
                    f"val_f1={cur_metric:.4f} val_bacc={val_metrics['balanced_accuracy']:.4f}"
                    f"{best_marker}"
                )

            if self.patience > 0 and no_improve >= self.patience:
                print(f"  [{self.emb_key}] Early stop @ epoch {epoch} (patience={self.patience})")
                break

        # ── 恢复最优权重 ─────────────────────────────────────────────────
        if self._best_head_state is not None:
            self.head.load_state_dict(self._best_head_state)

        # ── 保存 checkpoint ───────────────────────────────────────────────
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            save_path = self.save_dir / f"linear_probe_{self.emb_key}.pt"
            torch.save(
                {
                    "head_state_dict": self.head.state_dict(),
                    "emb_dim": self.emb_dim,
                    "num_classes": self.num_classes,
                    "best_val_macro_f1": self._best_val_metric,
                },
                save_path,
            )
            print(f"  [{self.emb_key}] Saved best probe → {save_path}")

        return history

    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _evaluate_loader(
        self, loader: DataLoader
    ) -> Dict[str, float]:
        """在 loader 上评估分类指标"""
        self.head.eval()

        all_logits: List[torch.Tensor] = []
        all_labels_: List[torch.Tensor] = []
        total_loss = 0.0

        for emb_batch, lbl_batch in loader:
            emb_batch = emb_batch.to(self.device)
            lbl_batch = lbl_batch.to(self.device)
            logits    = self.head(emb_batch)
            loss      = self.criterion(logits, lbl_batch)
            total_loss += loss.item() * len(lbl_batch)
            all_logits.append(logits.cpu())
            all_labels_.append(lbl_batch.cpu())

        all_logits  = torch.cat(all_logits, dim=0)    # (N, C)
        all_labels_ = torch.cat(all_labels_, dim=0)   # (N,)
        avg_loss    = total_loss / max(len(all_labels_), 1)

        return self.compute_metrics(all_logits, all_labels_, avg_loss)

    # -------------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(
        self,
        embeddings: torch.Tensor,   # (N, D)
        labels: torch.Tensor,       # (N,)
    ) -> Dict[str, float]:
        """
        在给定 embedding 上评估最终分类指标（使用最优权重）。

        Returns
        -------
        {
            "accuracy"          : float,
            "balanced_accuracy" : float,
            "macro_f1"          : float,
            "auroc"             : float | None,  (仅二分类)
            "loss"              : float,
        }
        """
        mask = labels >= 0
        embeddings = embeddings[mask]
        labels     = labels[mask]

        loader  = self._make_loader(embeddings, labels, shuffle=False)
        metrics = self._evaluate_loader(loader)
        return metrics

    # -------------------------------------------------------------------------
    @staticmethod
    def compute_metrics(
        logits: torch.Tensor,    # (N, C)
        labels: torch.Tensor,    # (N,)
        loss: float = 0.0,
    ) -> Dict[str, float]:
        """计算分类指标（sklearn）"""
        probs  = F.softmax(logits, dim=-1).numpy()   # (N, C)
        preds  = logits.argmax(dim=-1).numpy()        # (N,)
        labels = labels.numpy()                       # (N,)

        if SKLEARN_AVAILABLE:
            acc  = float(accuracy_score(labels, preds))
            bacc = float(balanced_accuracy_score(labels, preds))
            f1   = float(f1_score(labels, preds, average="macro", zero_division=0))

            # AUROC：仅二分类 or OvR 多分类
            auroc: Optional[float] = None
            try:
                n_cls = probs.shape[1]
                if n_cls == 2:
                    auroc = float(roc_auc_score(labels, probs[:, 1]))
                else:
                    auroc = float(
                        roc_auc_score(labels, probs, multi_class="ovr", average="macro")
                    )
            except Exception:
                auroc = None
        else:
            # 无 sklearn 的简单实现
            correct = int((preds == labels).sum())
            acc  = correct / max(len(labels), 1)
            bacc = acc  # fallback
            f1   = acc
            auroc = None

        result = {
            "accuracy":          acc,
            "balanced_accuracy": bacc,
            "macro_f1":          f1,
            "loss":              loss,
        }
        if auroc is not None:
            result["auroc"] = auroc
        return result

    # -------------------------------------------------------------------------
    @staticmethod
    def compute_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        计算各类别的逆频率权重（用于不平衡数据集）。

        Parameters
        ----------
        labels      : (N,) 标签张量（-1 视为无效，忽略）
        num_classes : 类别数

        Returns
        -------
        weights : (num_classes,) FloatTensor
        """
        valid = labels[labels >= 0]
        counts = torch.zeros(num_classes, dtype=torch.float32)
        for c in range(num_classes):
            counts[c] = (valid == c).sum().float()

        # 避免零除
        counts = counts.clamp(min=1.0)
        total  = counts.sum()
        # 权重 = total / (num_classes * count_c)
        weights = total / (num_classes * counts)
        return weights
