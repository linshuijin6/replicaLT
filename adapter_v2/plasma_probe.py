from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


class PlasmaProbeLogger:
    """
    逐 step 记录 plasma 的 raw/minmax/softmax 数据，并导出：
    1) sample 级明细 CSV
    2) step 级统计 CSV
    3) NPZ（便于后处理）
    4) 可选 matplotlib 图
    """

    def __init__(
        self,
        output_dir: str | Path,
        plasma_keys: Sequence[str] | None = None,
        enable_plots: bool = False,
        temperature: float = 1.0,
        eps: float = 1e-12,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.plots_dir = self.output_dir / "plots"
        self.enable_plots = bool(enable_plots)
        if self.enable_plots:
            self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.temperature = float(temperature)
        self.eps = float(eps)

        self._plasma_keys: list[str] | None = list(plasma_keys) if plasma_keys is not None else None
        self._num_features: int | None = len(self._plasma_keys) if self._plasma_keys is not None else None

        self._steps: list[int] = []
        self._raw_list: list[np.ndarray] = []
        self._minmax_list: list[np.ndarray] = []
        self._softmax_list: list[np.ndarray] = []
        self._mask_list: list[np.ndarray] = []
        self._sample_ids_list: list[np.ndarray] = []

    @property
    def plasma_keys(self) -> list[str]:
        if self._plasma_keys is None:
            if self._num_features is None:
                return []
            return [f"plasma_{i}" for i in range(self._num_features)]
        return list(self._plasma_keys)

    @property
    def num_steps(self) -> int:
        return len(self._steps)

    def log_step(
        self,
        step: int,
        raw: Any,
        mask: Any | None = None,
        minmax: Any | None = None,
        softmax: Any | None = None,
        sample_ids: Sequence[str] | None = None,
    ) -> None:
        """
        记录一个训练 step 的 plasma 数据。

        参数
        ----
        step:
            全局 step。
        raw:
            原始 plasma 值，形状 (B, K)。
        mask:
            有效位掩码，形状 (B, K)。True=有效。缺省时默认全 True。
        minmax:
            可选，外部给定的 min-max 归一化值，形状 (B, K)。
            若不提供，则按每个样本（行）在有效位上自动计算。
        softmax:
            可选，外部给定 softmax 权重，形状 (B, K)。
            若不提供，则按 raw（结合 mask）自动计算。
        sample_ids:
            可选，长度 B。
        """
        step_i = int(step)

        raw_np = self._to_2d_float(raw, name="raw")
        batch_size, num_features = raw_np.shape

        self._ensure_feature_shape(num_features)

        if mask is None:
            mask_np = np.ones_like(raw_np, dtype=bool)
        else:
            mask_np = self._to_2d_bool(mask, name="mask", expected_shape=raw_np.shape)

        if minmax is None:
            minmax_np = self._masked_row_minmax(raw_np, mask_np)
        else:
            minmax_np = self._to_2d_float(minmax, name="minmax", expected_shape=raw_np.shape)

        if softmax is None:
            softmax_np = self._masked_row_softmax(raw_np, mask_np, temperature=self.temperature)
        else:
            softmax_np = self._to_2d_float(softmax, name="softmax", expected_shape=raw_np.shape)

        ids_np = self._normalize_sample_ids(sample_ids, batch_size=batch_size, step=step_i)

        self._steps.append(step_i)
        self._raw_list.append(raw_np)
        self._mask_list.append(mask_np)
        self._minmax_list.append(minmax_np)
        self._softmax_list.append(softmax_np)
        self._sample_ids_list.append(ids_np)

    def export_sample_csv(self, file_path: str | Path | None = None) -> Path:
        """导出 sample 级明细 CSV。"""
        self._require_data()
        path = Path(file_path) if file_path is not None else self.output_dir / "plasma_probe_samples.csv"
        path.parent.mkdir(parents=True, exist_ok=True)

        header = [
            "step",
            "sample_index_in_step",
            "sample_id",
            "feature_index",
            "feature_name",
            "valid",
            "raw",
            "minmax",
            "softmax",
        ]

        keys = self.plasma_keys
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for step, raw_np, minmax_np, softmax_np, mask_np, ids_np in zip(
                self._steps,
                self._raw_list,
                self._minmax_list,
                self._softmax_list,
                self._mask_list,
                self._sample_ids_list,
            ):
                bsz, k = raw_np.shape
                for bi in range(bsz):
                    sample_id = ids_np[bi]
                    for ki in range(k):
                        writer.writerow(
                            [
                                step,
                                bi,
                                sample_id,
                                ki,
                                keys[ki],
                                int(mask_np[bi, ki]),
                                float(raw_np[bi, ki]),
                                float(minmax_np[bi, ki]),
                                float(softmax_np[bi, ki]),
                            ]
                        )
        return path

    def export_step_csv(self, file_path: str | Path | None = None) -> Path:
        """导出 step 级统计 CSV（每 step × 每 feature 一行）。"""
        self._require_data()
        path = Path(file_path) if file_path is not None else self.output_dir / "plasma_probe_steps.csv"
        path.parent.mkdir(parents=True, exist_ok=True)

        header = [
            "step",
            "batch_size",
            "feature_index",
            "feature_name",
            "valid_count",
            "valid_ratio",
            "raw_mean",
            "raw_std",
            "raw_min",
            "raw_max",
            "minmax_mean",
            "minmax_std",
            "softmax_mean",
            "softmax_std",
            "softmax_entropy_mean",
            "softmax_entropy_std",
        ]

        keys = self.plasma_keys
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for step, raw_np, minmax_np, softmax_np, mask_np in zip(
                self._steps,
                self._raw_list,
                self._minmax_list,
                self._softmax_list,
                self._mask_list,
            ):
                bsz, k = raw_np.shape
                entropy = self._row_entropy(softmax_np)

                for ki in range(k):
                    valid = mask_np[:, ki]
                    valid_count = int(valid.sum())
                    valid_ratio = float(valid_count / max(bsz, 1))

                    raw_vals = raw_np[:, ki][valid]
                    mm_vals = minmax_np[:, ki][valid]
                    sf_vals = softmax_np[:, ki][valid]
                    ent_vals = entropy[valid]

                    writer.writerow(
                        [
                            step,
                            bsz,
                            ki,
                            keys[ki],
                            valid_count,
                            valid_ratio,
                            self._safe_mean(raw_vals),
                            self._safe_std(raw_vals),
                            self._safe_min(raw_vals),
                            self._safe_max(raw_vals),
                            self._safe_mean(mm_vals),
                            self._safe_std(mm_vals),
                            self._safe_mean(sf_vals),
                            self._safe_std(sf_vals),
                            self._safe_mean(ent_vals),
                            self._safe_std(ent_vals),
                        ]
                    )
        return path

    def export_npz(self, file_path: str | Path | None = None, compressed: bool = True) -> Path:
        """导出 NPZ，包含所有 step 的拼接矩阵与索引。"""
        self._require_data()
        path = Path(file_path) if file_path is not None else self.output_dir / "plasma_probe_data.npz"
        path.parent.mkdir(parents=True, exist_ok=True)

        steps_arr = np.asarray(self._steps, dtype=np.int64)
        batch_sizes = np.asarray([arr.shape[0] for arr in self._raw_list], dtype=np.int64)
        step_offsets = np.zeros(len(batch_sizes) + 1, dtype=np.int64)
        step_offsets[1:] = np.cumsum(batch_sizes)

        raw_concat = np.concatenate(self._raw_list, axis=0)
        minmax_concat = np.concatenate(self._minmax_list, axis=0)
        softmax_concat = np.concatenate(self._softmax_list, axis=0)
        mask_concat = np.concatenate(self._mask_list, axis=0)
        sample_ids_concat = np.concatenate(self._sample_ids_list, axis=0).astype(str)

        payload = {
            "steps": steps_arr,
            "batch_sizes": batch_sizes,
            "step_offsets": step_offsets,
            "plasma_keys": np.asarray(self.plasma_keys, dtype=str),
            "raw": raw_concat,
            "minmax": minmax_concat,
            "softmax": softmax_concat,
            "mask": mask_concat,
            "sample_ids": sample_ids_concat,
        }

        if compressed:
            np.savez_compressed(path, **payload)
        else:
            np.savez(path, **payload)
        return path

    def export_plots(self, prefix: str = "plasma_probe") -> list[Path]:
        """
        导出可选 matplotlib 图。
        若 matplotlib 不可用，返回空列表。
        """
        self._require_data()
        if not self.enable_plots:
            return []

        try:
            import matplotlib.pyplot as plt
        except Exception:
            return []

        self.plots_dir.mkdir(parents=True, exist_ok=True)

        steps = np.asarray(self._steps, dtype=np.int64)
        keys = self.plasma_keys
        k = len(keys)

        raw_means = np.full((len(steps), k), np.nan, dtype=np.float64)
        mm_means = np.full((len(steps), k), np.nan, dtype=np.float64)
        sf_means = np.full((len(steps), k), np.nan, dtype=np.float64)

        for si, (raw_np, mm_np, sf_np, mask_np) in enumerate(
            zip(self._raw_list, self._minmax_list, self._softmax_list, self._mask_list)
        ):
            for ki in range(k):
                valid = mask_np[:, ki]
                if np.any(valid):
                    raw_means[si, ki] = float(np.mean(raw_np[:, ki][valid]))
                    mm_means[si, ki] = float(np.mean(mm_np[:, ki][valid]))
                    sf_means[si, ki] = float(np.mean(sf_np[:, ki][valid]))

        out_paths: list[Path] = []

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        for ki, key in enumerate(keys):
            axes[0].plot(steps, raw_means[:, ki], label=key)
            axes[1].plot(steps, mm_means[:, ki], label=key)
            axes[2].plot(steps, sf_means[:, ki], label=key)

        axes[0].set_title("Raw plasma mean per step")
        axes[1].set_title("MinMax plasma mean per step")
        axes[2].set_title("Softmax plasma mean per step")
        axes[2].set_xlabel("step")
        axes[0].set_ylabel("raw")
        axes[1].set_ylabel("minmax")
        axes[2].set_ylabel("softmax")
        for ax in axes:
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        trend_path = self.plots_dir / f"{prefix}_feature_means.png"
        fig.savefig(trend_path, dpi=180)
        plt.close(fig)
        out_paths.append(trend_path)

        # softmax 熵趋势（step 级）
        ent_means = []
        for sf_np in self._softmax_list:
            ent = self._row_entropy(sf_np)
            ent_means.append(self._safe_mean(ent))
        ent_means = np.asarray(ent_means, dtype=np.float64)

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(steps, ent_means, marker="o")
        ax2.set_title("Softmax entropy mean per step")
        ax2.set_xlabel("step")
        ax2.set_ylabel("entropy")
        ax2.grid(True, linestyle="--", alpha=0.35)
        fig2.tight_layout()
        ent_path = self.plots_dir / f"{prefix}_softmax_entropy.png"
        fig2.savefig(ent_path, dpi=180)
        plt.close(fig2)
        out_paths.append(ent_path)

        return out_paths

    def export_all(
        self,
        sample_csv_path: str | Path | None = None,
        step_csv_path: str | Path | None = None,
        npz_path: str | Path | None = None,
        with_plots: bool | None = None,
    ) -> dict[str, Path | list[Path]]:
        """一次性导出 sample CSV / step CSV / npz / (可选)plots。"""
        sample_p = self.export_sample_csv(sample_csv_path)
        step_p = self.export_step_csv(step_csv_path)
        npz_p = self.export_npz(npz_path)

        do_plots = self.enable_plots if with_plots is None else bool(with_plots)
        plot_paths: list[Path] = self.export_plots() if do_plots else []

        return {
            "sample_csv": sample_p,
            "step_csv": step_p,
            "npz": npz_p,
            "plots": plot_paths,
        }

    def reset(self) -> None:
        """清空内存中的已记录数据。"""
        self._steps.clear()
        self._raw_list.clear()
        self._minmax_list.clear()
        self._softmax_list.clear()
        self._mask_list.clear()
        self._sample_ids_list.clear()

    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        if x is None:
            raise ValueError("input is None")
        obj = x
        if hasattr(obj, "detach"):
            obj = obj.detach()
        if hasattr(obj, "cpu"):
            obj = obj.cpu()
        if hasattr(obj, "numpy"):
            return np.asarray(obj.numpy())
        return np.asarray(obj)

    def _to_2d_float(
        self,
        x: Any,
        name: str,
        expected_shape: tuple[int, int] | None = None,
    ) -> np.ndarray:
        arr = self._to_numpy(x)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be 2D (B, K), got shape={arr.shape}")
        arr = arr.astype(np.float64, copy=False)
        if expected_shape is not None and tuple(arr.shape) != tuple(expected_shape):
            raise ValueError(f"{name} shape mismatch: got {arr.shape}, expected {expected_shape}")
        return arr

    def _to_2d_bool(
        self,
        x: Any,
        name: str,
        expected_shape: tuple[int, int] | None = None,
    ) -> np.ndarray:
        arr = self._to_numpy(x)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be 2D (B, K), got shape={arr.shape}")
        arr = arr.astype(bool, copy=False)
        if expected_shape is not None and tuple(arr.shape) != tuple(expected_shape):
            raise ValueError(f"{name} shape mismatch: got {arr.shape}, expected {expected_shape}")
        return arr

    def _ensure_feature_shape(self, num_features: int) -> None:
        if self._num_features is None:
            self._num_features = int(num_features)
            if self._plasma_keys is None:
                self._plasma_keys = [f"plasma_{i}" for i in range(self._num_features)]
            elif len(self._plasma_keys) != self._num_features:
                raise ValueError(
                    f"plasma_keys length mismatch: {len(self._plasma_keys)} vs num_features={self._num_features}"
                )
            return

        if int(num_features) != self._num_features:
            raise ValueError(
                f"num_features mismatch across steps: got {num_features}, expected {self._num_features}"
            )

    @staticmethod
    def _masked_row_minmax(raw: np.ndarray, mask: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        x = raw.astype(np.float64, copy=False)
        m = mask.astype(bool, copy=False)

        x_work = np.where(m, x, np.nan)
        row_min = np.nanmin(x_work, axis=1, keepdims=True)
        row_max = np.nanmax(x_work, axis=1, keepdims=True)

        all_invalid = ~m.any(axis=1, keepdims=True)
        row_min = np.where(all_invalid, 0.0, row_min)
        row_max = np.where(all_invalid, 1.0, row_max)

        denom = np.maximum(row_max - row_min, eps)
        out = (x - row_min) / denom
        out = np.where(m, out, 0.0)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    @staticmethod
    def _masked_row_softmax(raw: np.ndarray, mask: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        x = raw.astype(np.float64, copy=False)
        m = mask.astype(bool, copy=False)

        t = max(float(temperature), 1e-8)
        logits = x / t
        logits = np.where(m, logits, -np.inf)

        row_max = np.max(logits, axis=1, keepdims=True)
        row_max = np.where(np.isfinite(row_max), row_max, 0.0)

        exp_logits = np.exp(logits - row_max)
        exp_logits = np.where(m, exp_logits, 0.0)
        denom = exp_logits.sum(axis=1, keepdims=True)

        out = np.divide(exp_logits, denom, out=np.zeros_like(exp_logits), where=denom > 0)

        all_invalid = ~m.any(axis=1, keepdims=True)
        k = max(int(x.shape[1]), 1)
        uniform = np.full_like(out, 1.0 / float(k))
        out = np.where(all_invalid, uniform, out)
        out = np.nan_to_num(out, nan=(1.0 / float(k)), posinf=0.0, neginf=0.0)
        return out

    def _normalize_sample_ids(
        self,
        sample_ids: Sequence[str] | None,
        batch_size: int,
        step: int,
    ) -> np.ndarray:
        if sample_ids is None:
            return np.asarray([f"step{step}_idx{i}" for i in range(batch_size)], dtype=str)

        if isinstance(sample_ids, np.ndarray):
            ids = sample_ids.astype(str)
        elif isinstance(sample_ids, (list, tuple)):
            ids = np.asarray([str(v) for v in sample_ids], dtype=str)
        else:
            # 支持可迭代对象
            if isinstance(sample_ids, Iterable):
                ids = np.asarray([str(v) for v in sample_ids], dtype=str)
            else:
                raise TypeError("sample_ids must be a sequence/iterable of strings")

        if ids.shape[0] != batch_size:
            raise ValueError(f"sample_ids length mismatch: {ids.shape[0]} vs batch_size={batch_size}")
        return ids

    @staticmethod
    def _safe_mean(x: np.ndarray) -> float:
        if x.size == 0:
            return float("nan")
        return float(np.mean(x))

    @staticmethod
    def _safe_std(x: np.ndarray) -> float:
        if x.size == 0:
            return float("nan")
        return float(np.std(x))

    @staticmethod
    def _safe_min(x: np.ndarray) -> float:
        if x.size == 0:
            return float("nan")
        return float(np.min(x))

    @staticmethod
    def _safe_max(x: np.ndarray) -> float:
        if x.size == 0:
            return float("nan")
        return float(np.max(x))

    def _row_entropy(self, probs: np.ndarray) -> np.ndarray:
        p = np.clip(probs.astype(np.float64, copy=False), self.eps, 1.0)
        return -np.sum(p * np.log(p), axis=1)

    def _require_data(self) -> None:
        if not self._steps:
            raise RuntimeError("no plasma probe data logged yet")
