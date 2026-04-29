from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "data": {
        "data_root": "/mnt/nfsdata/nfsdata/lsj.14/ADNI_CSF",
        "train_json": "train_data_with_description.json",
        "val_json": "val_data_with_description.json",
        "metadata_mode": "strict",
        "tracer": "tau",
        "training_split_ratio": 0.9,
        "crop": [11, 10, 20, 17, 0, 21],
        "target_shape": [160, 180, 160],
        "num_workers": 0,
        "pin_memory": False,
    },
    "model": {
        "spatial_dims": 3,
        "in_channels": 2,
        "out_channels": 1,
        "num_channels": [16, 32, 64],
        "attention_levels": [False, False, True],
        "num_head_channels": [0, 0, 64],
        "num_res_blocks": 2,
        "norm_num_groups": 8,
        "use_flash_attention": True,
        "with_conditioning": False,
    },
    "train": {
        "batch_size_train": 2,
        "batch_size_val": 1,
        "epochs": 50,
        "lr": 5e-5,
        "num_train_timesteps": 1000,
        "num_inference_steps": 1000,
        "save_every": 1,
        "val_every": 10,
        "image_log_interval": 10,
    },
    "logging": {
        "run_root": "runs/ficd_strict_tau",
        "tensorboard": True,
    },
}


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _resolve_path(repo_root: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((repo_root / path).resolve())


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    repo_root = config_path.parents[1]

    with config_path.open("r", encoding="utf-8") as handle:
        file_config = yaml.safe_load(handle) or {}

    config = _deep_update(copy.deepcopy(DEFAULT_CONFIG), file_config)
    config["config_path"] = str(config_path)
    config["repo_root"] = str(repo_root)

    data_cfg = config["data"]
    data_cfg["train_json"] = _resolve_path(repo_root, data_cfg["train_json"])
    data_cfg["val_json"] = _resolve_path(repo_root, data_cfg["val_json"])

    logging_cfg = config["logging"]
    logging_cfg["run_root"] = _resolve_path(repo_root, logging_cfg["run_root"])

    metadata_mode = str(data_cfg["metadata_mode"]).strip().lower()
    if metadata_mode not in {"strict", "aligned"}:
        raise ValueError(f"Unsupported data.metadata_mode={metadata_mode}")
    data_cfg["metadata_mode"] = metadata_mode

    tracer = str(data_cfg["tracer"]).strip().lower()
    if tracer != "tau":
        raise ValueError("This FICD baseline currently supports tracer=tau only.")
    data_cfg["tracer"] = tracer

    return config
