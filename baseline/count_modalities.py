"""
baseline/count_modalities.py
============================
一次性统计脚本：统计 train/val JSON 中各模态（tau / fdg / av45）的有效与缺失样本数。

用法:
    cd /mnt/nfsdata/nfsdata/lsj.14/replicaLT
    python -m baseline.count_modalities
    # 或指定自定义 JSON:
    python -m baseline.count_modalities \
        --train_json path/to/train.json \
        --val_json   path/to/val.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from .dataset import _is_missing_modal_path


MODALITIES = ["mri", "tau", "fdg", "av45"]

DEFAULT_BASE = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN = str(DEFAULT_BASE / "train_data_with_description.json")
DEFAULT_VAL   = str(DEFAULT_BASE / "val_data_with_description.json")


def count_json(json_path: str, label: str) -> dict:
    """统计单个 JSON 文件中各模态的有效/缺失数量。"""
    with open(json_path, "r") as f:
        data = json.load(f)

    total = len(data)
    stats = {}
    for mod in MODALITIES:
        valid = 0
        missing = 0
        absent = 0  # JSON 中根本没有该字段
        for item in data:
            path_val = item.get(mod)
            if path_val is None:
                absent += 1
            elif _is_missing_modal_path(path_val):
                missing += 1
            else:
                valid += 1
        stats[mod] = {"valid": valid, "missing_zero": missing, "field_absent": absent}

    # 打印
    print(f"\n{'='*60}")
    print(f"  {label}: {json_path}")
    print(f"  总记录数: {total}")
    print(f"{'='*60}")
    print(f"  {'模态':<8} {'有效':>8} {'_zero缺失':>10} {'字段缺失':>10} {'有效率':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    for mod in MODALITIES:
        s = stats[mod]
        rate = f"{s['valid']/total*100:.1f}%" if total > 0 else "N/A"
        print(f"  {mod:<8} {s['valid']:>8} {s['missing_zero']:>10} {s['field_absent']:>10} {rate:>8}")
    print()

    return stats


def main():
    parser = argparse.ArgumentParser(description="统计各模态有效/缺失数量")
    parser.add_argument("--train_json", type=str, default=DEFAULT_TRAIN, help="train JSON 路径")
    parser.add_argument("--val_json",   type=str, default=DEFAULT_VAL,   help="val JSON 路径")
    args = parser.parse_args()

    all_stats = {}
    for label, path in [("TRAIN", args.train_json), ("VAL", args.val_json)]:
        if os.path.exists(path):
            all_stats[label] = count_json(path, label)
        else:
            print(f"[WARN] {label} JSON 不存在: {path}")

    # 汇总 (train + val)
    if len(all_stats) == 2:
        print(f"{'='*60}")
        print(f"  合计 (TRAIN + VAL)")
        print(f"{'='*60}")
        print(f"  {'模态':<8} {'有效':>8} {'_zero缺失':>10} {'字段缺失':>10}")
        print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
        for mod in MODALITIES:
            v = sum(all_stats[s][mod]["valid"] for s in all_stats)
            m = sum(all_stats[s][mod]["missing_zero"] for s in all_stats)
            a = sum(all_stats[s][mod]["field_absent"] for s in all_stats)
            print(f"  {mod:<8} {v:>8} {m:>10} {a:>10}")
        print()


if __name__ == "__main__":
    main()
