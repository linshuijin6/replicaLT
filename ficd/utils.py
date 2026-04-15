from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from PIL import Image


def make_run_dir(run_root: str | Path, resume: str | None = None) -> Path:
    if resume:
        run_dir = Path(resume).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    run_root = Path(run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    run_name = f"{datetime.now():%y%m%d}.{os.getpid()}"
    run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("ficd")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def write_json(path: str | Path, payload: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def append_csv_row(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def find_checkpoint(run_dir: str | Path, checkpoint: str | None = None) -> Path:
    run_dir = Path(run_dir)
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = (run_dir / checkpoint_path).resolve()
        return checkpoint_path

    best_model = run_dir / "best_model.pt"
    if best_model.exists():
        return best_model

    ckpts = sorted(run_dir.glob("ckpt_epoch*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found under {run_dir}")
    return ckpts[-1]


def get_3d_slices(volume: torch.Tensor | np.ndarray, normalize: bool = True) -> dict[str, np.ndarray]:
    if isinstance(volume, torch.Tensor):
        vol = volume.detach().cpu().numpy()
    else:
        vol = np.asarray(volume)

    while vol.ndim > 3:
        vol = vol[0]

    h, w, d = vol.shape
    slices = {
        "axial": vol[:, :, d // 2],
        "coronal": vol[:, w // 2, :],
        "sagittal": vol[h // 2, :, :],
    }
    if normalize:
        for key, value in slices.items():
            v_min = float(value.min())
            v_max = float(value.max())
            if v_max - v_min > 1e-8:
                slices[key] = (value - v_min) / (v_max - v_min)
            else:
                slices[key] = np.zeros_like(value)
    return slices


def _figure_to_array(fig: plt.Figure) -> np.ndarray:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.1, dpi=150)
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer)
    array = np.array(image)
    if array.ndim == 3:
        array = array[:, :, :3].transpose(2, 0, 1)
    return array


def log_3d_volume_to_tensorboard(writer, tag_prefix: str, volume, global_step: int, cmap: str = "gray") -> None:
    for view_name, slice_2d in get_3d_slices(volume).items():
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(slice_2d, cmap=cmap, vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(view_name)
        writer.add_image(f"{tag_prefix}/{view_name}", _figure_to_array(fig), global_step)


def create_comparison_figure(volumes_dict: dict[str, Any]) -> plt.Figure:
    n_vols = len(volumes_dict)
    fig, axes = plt.subplots(3, n_vols, figsize=(3 * n_vols, 9))
    if n_vols == 1:
        axes = axes.reshape(3, 1)
    view_names = ["axial", "coronal", "sagittal"]

    for col_idx, (name, volume) in enumerate(volumes_dict.items()):
        if volume is None:
            for row_idx in range(3):
                axes[row_idx, col_idx].axis("off")
            continue
        slices = get_3d_slices(volume)
        cmap = "jet" if "Diff" in name else "gray"
        for row_idx, view_name in enumerate(view_names):
            ax = axes[row_idx, col_idx]
            ax.imshow(slices[view_name], cmap=cmap, vmin=0, vmax=1)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(name, fontsize=10)
    plt.tight_layout()
    return fig


def log_comparison_figure(writer, tag: str, volumes_dict: dict[str, Any], global_step: int) -> None:
    fig = create_comparison_figure(volumes_dict)
    writer.add_image(tag, _figure_to_array(fig), global_step)


def save_comparison_figure(path: str | Path, volumes_dict: dict[str, Any]) -> None:
    fig = create_comparison_figure(volumes_dict)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1, dpi=150)
    plt.close(fig)


def save_prediction_nifti(path: str | Path, tensor: torch.Tensor, affine: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    volume = tensor.detach().cpu().numpy()
    while volume.ndim > 3:
        volume = volume[0]
    nib.save(nib.Nifti1Image(volume.astype(np.float32), affine), str(path))
