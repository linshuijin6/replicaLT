#!/usr/bin/env python3
"""Extract test_data_with_description.json from val_data_with_description.json."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
COMMON_SUBJECTS_PATH = ROOT / "analysis" / "_common_subjects.json"
VAL_DATA_PATH = ROOT / "val_data_with_description.json"
TEST_DATA_PATH = ROOT / "test_data_with_description.json"


def main() -> None:
    subject_ids = set(json.loads(COMMON_SUBJECTS_PATH.read_text(encoding="utf-8")))
    val_data = json.loads(VAL_DATA_PATH.read_text(encoding="utf-8"))

    test_data = [item for item in val_data if item.get("name") in subject_ids]

    missing_subjects = sorted(subject_ids - {item.get("name") for item in test_data})
    if missing_subjects:
        raise ValueError(f"Subjects not found in val_data_with_description.json: {missing_subjects}")

    TEST_DATA_PATH.write_text(
        json.dumps(test_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Loaded {len(subject_ids)} subject IDs from {COMMON_SUBJECTS_PATH.name}")
    print(f"Extracted {len(test_data)} records from {VAL_DATA_PATH.name}")
    print(f"Wrote {TEST_DATA_PATH}")


if __name__ == "__main__":
    main()
