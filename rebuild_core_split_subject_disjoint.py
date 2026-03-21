#!/usr/bin/env python3
"""
Rebuild root train/val JSON into strict subject-disjoint splits using fixed_split.json.

Rules:
1) Subject assignment is anchored by fixed_split.json (train_subjects / val_subjects).
2) All samples of one subject go to a single split.
3) Subjects outside both lists are dropped and recorded in audit.
4) Duplicate samples (same name + examdate) are kept once (first occurrence).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def load_json_list(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list")
    return data


def subject_from_item(item: Dict) -> str:
    sid = item.get("name") or item.get("ptid") or item.get("Subject ID")
    if sid is None:
        return ""
    sid = str(sid)
    if "__" in sid:
        sid = sid.split("__", 1)[0]
    return sid


def item_key(item: Dict) -> Tuple[str, str]:
    return str(item.get("name", "")), str(item.get("examdate", ""))


def summarize_tracer(items: List[Dict]) -> Dict[str, int]:
    def valid(path: str) -> bool:
        return bool(path) and ("zero" not in str(path).lower())

    return {
        "fdg_non_zero": sum(1 for x in items if valid(x.get("fdg", ""))),
        "av45_non_zero": sum(1 for x in items if valid(x.get("av45", ""))),
        "tau_non_zero": sum(1 for x in items if valid(x.get("tau", ""))),
    }


def rebuild(
    fixed_split_path: Path,
    train_path: Path,
    val_path: Path,
) -> Tuple[List[Dict], List[Dict], Dict]:
    fixed = json.loads(fixed_split_path.read_text(encoding="utf-8"))
    fixed_train = set(map(str, fixed.get("train_subjects", [])))
    fixed_val = set(map(str, fixed.get("val_subjects", [])))

    overlap_fixed = fixed_train & fixed_val
    if overlap_fixed:
        raise ValueError(f"fixed_split has overlap between train/val subjects: {len(overlap_fixed)}")

    old_train = load_json_list(train_path)
    old_val = load_json_list(val_path)

    merged = old_train + old_val
    source_counter = Counter(["train"] * len(old_train) + ["val"] * len(old_val))

    new_train: List[Dict] = []
    new_val: List[Dict] = []

    dropped_outside = 0
    dropped_missing_subject = 0
    duplicates_removed = 0

    moved_train_to_val = 0
    moved_val_to_train = 0

    seen_keys = set()
    per_subject_count = defaultdict(int)

    for idx, item in enumerate(merged):
        key = item_key(item)
        if key in seen_keys:
            duplicates_removed += 1
            continue
        seen_keys.add(key)

        sid = subject_from_item(item)
        if not sid:
            dropped_missing_subject += 1
            continue

        per_subject_count[sid] += 1

        src = "train" if idx < len(old_train) else "val"

        if sid in fixed_train:
            if src == "val":
                moved_val_to_train += 1
            new_train.append(item)
        elif sid in fixed_val:
            if src == "train":
                moved_train_to_val += 1
            new_val.append(item)
        else:
            dropped_outside += 1

    train_subjects = {subject_from_item(x) for x in new_train}
    val_subjects = {subject_from_item(x) for x in new_val}

    audit = {
        "input": {
            "train_records": len(old_train),
            "val_records": len(old_val),
            "source_counts": dict(source_counter),
        },
        "fixed_split": {
            "train_subjects": len(fixed_train),
            "val_subjects": len(fixed_val),
        },
        "output": {
            "train_records": len(new_train),
            "val_records": len(new_val),
            "train_subjects": len(train_subjects),
            "val_subjects": len(val_subjects),
            "subject_overlap": len(train_subjects & val_subjects),
            "tracer": {
                "train": summarize_tracer(new_train),
                "val": summarize_tracer(new_val),
            },
        },
        "movement": {
            "moved_train_to_val": moved_train_to_val,
            "moved_val_to_train": moved_val_to_train,
        },
        "dropped": {
            "outside_fixed_split": dropped_outside,
            "missing_subject": dropped_missing_subject,
            "duplicate_samples_removed": duplicates_removed,
        },
        "checks": {
            "train_subjects_subset_fixed_train": train_subjects <= fixed_train,
            "val_subjects_subset_fixed_val": val_subjects <= fixed_val,
            "subject_disjoint": len(train_subjects & val_subjects) == 0,
        },
    }

    return new_train, new_val, audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild core train/val JSON into subject-disjoint splits")
    parser.add_argument("--fixed-split", default="fixed_split.json")
    parser.add_argument("--in-train", default="train_data_with_description.json")
    parser.add_argument("--in-val", default="val_data_with_description.json")
    parser.add_argument("--out-train", default="train_data_with_description.json")
    parser.add_argument("--out-val", default="val_data_with_description.json")
    parser.add_argument("--audit", default="split_rebuild_audit.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    fixed_split_path = Path(args.fixed_split)
    in_train_path = Path(args.in_train)
    in_val_path = Path(args.in_val)
    out_train_path = Path(args.out_train)
    out_val_path = Path(args.out_val)
    audit_path = Path(args.audit)

    new_train, new_val, audit = rebuild(fixed_split_path, in_train_path, in_val_path)

    print("=== Rebuild Summary ===")
    print(f"Output train records: {audit['output']['train_records']}")
    print(f"Output val records:   {audit['output']['val_records']}")
    print(f"Output train subjects: {audit['output']['train_subjects']}")
    print(f"Output val subjects:   {audit['output']['val_subjects']}")
    print(f"Subject overlap:       {audit['output']['subject_overlap']}")
    print(f"Moved train->val:      {audit['movement']['moved_train_to_val']}")
    print(f"Moved val->train:      {audit['movement']['moved_val_to_train']}")
    print(f"Dropped outside fixed: {audit['dropped']['outside_fixed_split']}")
    print(f"Dropped missing subj:  {audit['dropped']['missing_subject']}")
    print(f"Duplicates removed:    {audit['dropped']['duplicate_samples_removed']}")

    if args.dry_run:
        print("Dry-run enabled: no files written.")
        return

    out_train_path.write_text(json.dumps(new_train, ensure_ascii=False, indent=2), encoding="utf-8")
    out_val_path.write_text(json.dumps(new_val, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote train JSON: {out_train_path}")
    print(f"Wrote val JSON:   {out_val_path}")
    print(f"Wrote audit JSON: {audit_path}")


if __name__ == "__main__":
    main()
