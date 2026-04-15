from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torchio as tio


def _load_json_list(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _is_zero_placeholder(path: str | None) -> bool:
    if not path:
        return False
    normalized = str(path).lower().replace("\\", "/")
    return "/zero/" in normalized or normalized.endswith("_zero.nii.gz")


def _normalize_item(item: dict[str, Any], source_split: str) -> dict[str, Any]:
    return {
        "subject_id": item["name"],
        "diagnosis": item.get("diagnosis"),
        "mri_path": item.get("mri"),
        "pet_path": item.get("tau"),
        "examdate": item.get("examdate"),
        "description": item.get("description"),
        "source_split": source_split,
    }


def _filter_items(
    items: list[dict[str, Any]],
    require_cn: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stats = {
        "total_in": len(items),
        "kept": 0,
        "missing_mri": 0,
        "missing_pet": 0,
        "zero_placeholder_pet": 0,
        "non_cn": 0,
    }
    kept: list[dict[str, Any]] = []

    for item in items:
        if not item.get("mri_path"):
            stats["missing_mri"] += 1
            continue
        if not item.get("pet_path"):
            stats["missing_pet"] += 1
            continue
        if _is_zero_placeholder(item["pet_path"]):
            stats["zero_placeholder_pet"] += 1
            continue
        if require_cn and str(item.get("diagnosis", "")).upper() != "CN":
            stats["non_cn"] += 1
            continue
        kept.append(item)

    stats["kept"] = len(kept)
    return kept, stats


def load_split_samples(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    data_cfg = config["data"]
    train_raw = [_normalize_item(item, "train_json") for item in _load_json_list(data_cfg["train_json"])]
    val_raw = [_normalize_item(item, "val_json") for item in _load_json_list(data_cfg["val_json"])]

    if data_cfg["metadata_mode"] == "strict":
        merged = train_raw + val_raw
        filtered, filter_stats = _filter_items(merged, require_cn=True)
        if not filtered:
            raise RuntimeError("No valid CN MRI/TAU samples remain after strict filtering.")

        rng = random.Random(int(config["seed"]))
        shuffled = filtered.copy()
        rng.shuffle(shuffled)

        split_ratio = float(data_cfg["training_split_ratio"])
        train_count = int(len(shuffled) * split_ratio)
        if len(shuffled) > 1:
            train_count = max(1, min(train_count, len(shuffled) - 1))
        else:
            train_count = len(shuffled)

        train_samples = shuffled[:train_count]
        val_samples = shuffled[train_count:]
        if not val_samples and train_samples:
            val_samples = [train_samples.pop()]

        split_stats = {
            "mode": "strict",
            "filter_stats": filter_stats,
            "train_count": len(train_samples),
            "val_count": len(val_samples),
        }
        return train_samples, val_samples, split_stats

    train_samples, train_stats = _filter_items(train_raw, require_cn=False)
    val_samples, val_stats = _filter_items(val_raw, require_cn=False)
    split_stats = {
        "mode": "aligned",
        "train_filter_stats": train_stats,
        "val_filter_stats": val_stats,
        "train_count": len(train_samples),
        "val_count": len(val_samples),
    }
    return train_samples, val_samples, split_stats


def build_transform(config: dict[str, Any]) -> tio.Compose:
    data_cfg = config["data"]
    return tio.Compose(
        [
            tio.RescaleIntensity(out_min_max=(-1, 1)),
            tio.Crop(tuple(int(x) for x in data_cfg["crop"])),
            tio.Resize(tuple(int(x) for x in data_cfg["target_shape"])),
        ]
    )


def build_dataset(samples: list[dict[str, Any]], transform: tio.Compose) -> tio.SubjectsDataset:
    subjects = []
    for sample in samples:
        subjects.append(
            tio.Subject(
                mri=tio.ScalarImage(sample["mri_path"]),
                pet=tio.ScalarImage(sample["pet_path"]),
                subject_id=sample["subject_id"],
                diagnosis=sample.get("diagnosis"),
                mri_path=sample["mri_path"],
                pet_path=sample["pet_path"],
                source_split=sample["source_split"],
            )
        )
    return tio.SubjectsDataset(subjects, transform=transform)
