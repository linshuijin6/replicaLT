#!/usr/bin/env python3
"""Check common subject-examdate pairs across all methods."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/home/data/linshuijin/replicaLT")

METHOD_JSON_PATHS = {
    "pasta": ROOT / "pasta/data/val_tabular.json",
    "ficd": ROOT / "val_data_with_description.json",
    "plasma": ROOT / "val_data_with_description.json",
    "legacy": ROOT / "val_data_with_description.json",
}

COMMON_KEYS_JSON = ROOT / "analysis/_common_subjects.json"


def load_json_list(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list")
    return data


def has_real_tau(item: dict) -> bool:
    tau_path = str(item.get("tau", "") or "")
    return bool(tau_path) and "/zero/" not in tau_path


def make_key(item: dict) -> tuple[str, str]:
    subject_id = str(item.get("name", "") or "")
    examdate = str(item.get("examdate", "") or "")
    return subject_id, examdate


def collect_keys(data: list[dict]) -> set[tuple[str, str]]:
    return {
        make_key(item)
        for item in data
        if item.get("name") and item.get("examdate") and has_real_tau(item)
    }


def main() -> None:
    method_keys: dict[str, set[tuple[str, str]]] = {}

    for method, path in METHOD_JSON_PATHS.items():
        data = load_json_list(path)
        keys = collect_keys(data)
        method_keys[method] = keys
        unique_subjects = len({subject_id for subject_id, _ in keys})
        print(
            f"{method}: total={len(data)}, with real TAU unique pairs={len(keys)}, "
            f"unique subjects={unique_subjects}"
        )

    common_keys = set.intersection(*method_keys.values())
    common_items = [
        {"name": subject_id, "examdate": examdate}
        for subject_id, examdate in sorted(common_keys)
    ]

    print(f"\nCommon subject-examdate pairs across all methods: {len(common_items)}")
    print("Pairs:", common_items)

    for method, keys in method_keys.items():
        missing = sorted(common_keys - keys)
        extra = sorted(keys - common_keys)
        print(f"\n{method} missing common pairs: {len(missing)}")
        print(f"{method} extra pairs beyond common set: {len(extra)}")

    COMMON_KEYS_JSON.write_text(
        json.dumps(common_items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved {len(common_items)} common pairs to {COMMON_KEYS_JSON}")


if __name__ == "__main__":
    main()
