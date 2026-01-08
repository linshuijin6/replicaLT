#!/usr/bin/env python3
import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from tensorboard.backend.event_processing import event_accumulator


@dataclass
class ScalarStats:
    tag: str
    n: int
    last_step: int
    last_val: float
    min_step: int
    min_val: float
    max_step: int
    max_val: float


def load_event_accumulators(logdir: str) -> List[event_accumulator.EventAccumulator]:
    event_files = sorted(glob.glob(os.path.join(logdir, "events.out.tfevents.*")))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under: {logdir}")

    accumulators: List[event_accumulator.EventAccumulator] = []
    for path in event_files:
        ea = event_accumulator.EventAccumulator(
            path,
            size_guidance={
                event_accumulator.SCALARS: 0,
                event_accumulator.HISTOGRAMS: 0,
                event_accumulator.IMAGES: 0,
                event_accumulator.COMPRESSED_HISTOGRAMS: 0,
                event_accumulator.TENSORS: 0,
            },
        )
        ea.Reload()
        accumulators.append(ea)
    return accumulators


def merge_scalars(eas: List[event_accumulator.EventAccumulator]) -> Dict[str, List[Tuple[int, float]]]:
    merged: Dict[str, List[Tuple[int, float]]] = {}
    for ea in eas:
        for tag in ea.Tags().get("scalars", []):
            points = [(e.step, float(e.value)) for e in ea.Scalars(tag)]
            merged.setdefault(tag, []).extend(points)

    # sort and de-dup by step (keep last occurrence)
    for tag, pts in merged.items():
        pts.sort(key=lambda x: x[0])
        dedup: Dict[int, float] = {}
        for step, val in pts:
            dedup[step] = val
        merged[tag] = sorted(dedup.items(), key=lambda x: x[0])
    return merged


def compute_stats(tag: str, pts: List[Tuple[int, float]]) -> ScalarStats:
    steps = [s for s, _ in pts]
    vals = [v for _, v in pts]
    last_step, last_val = steps[-1], vals[-1]
    min_val = min(vals)
    max_val = max(vals)
    min_idx = vals.index(min_val)
    max_idx = vals.index(max_val)
    return ScalarStats(
        tag=tag,
        n=len(vals),
        last_step=last_step,
        last_val=last_val,
        min_step=steps[min_idx],
        min_val=min_val,
        max_step=steps[max_idx],
        max_val=max_val,
    )


def find_best_tag_pair(tags: List[str]) -> Optional[Tuple[str, str]]:
    # heuristic: try to find a train/val loss pair
    lower = {t.lower(): t for t in tags}
    candidates = [
        ("train/loss", "val/loss"),
        ("train_loss", "val_loss"),
        ("loss/train", "loss/val"),
        ("loss", "val/loss"),
    ]
    for a, b in candidates:
        if a in lower and b in lower:
            return lower[a], lower[b]
    return None


def simple_convergence_comment(loss_pts: List[Tuple[int, float]]) -> str:
    # compare first 10% vs last 10%
    n = len(loss_pts)
    if n < 20:
        return "点数较少，无法稳定判断收敛趋势。"
    k = max(2, n // 10)
    head = [v for _, v in loss_pts[:k]]
    tail = [v for _, v in loss_pts[-k:]]
    head_mean = sum(head) / len(head)
    tail_mean = sum(tail) / len(tail)
    if tail_mean < head_mean * 0.7:
        return "loss 明显下降，整体在收敛。"
    if tail_mean < head_mean * 0.9:
        return "loss 有下降，收敛但幅度一般。"
    if tail_mean <= head_mean * 1.05:
        return "loss 变化不大，可能较早进入平台期。"
    return "loss 后期高于前期，可能训练不稳定或学习率过高。"


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze TensorBoard event scalars in a run directory")
    ap.add_argument("logdir", help="TensorBoard run directory containing events.out.tfevents.*")
    args = ap.parse_args()

    eas = load_event_accumulators(args.logdir)
    merged = merge_scalars(eas)

    print(f"logdir: {args.logdir}")
    print(f"event_files: {len(eas)}")
    print(f"scalar_tags: {len(merged)}")

    if not merged:
        print("No scalar tags found.")
        return 0

    stats = [compute_stats(tag, pts) for tag, pts in merged.items()]
    stats.sort(key=lambda s: s.tag)

    print("\nTags:")
    for s in stats:
        print(f" - {s.tag}")

    print("\nScalar summary (last/min/max):")
    for s in stats:
        print(
            f"[{s.tag}] n={s.n} "
            f"last@{s.last_step}={s.last_val:.6g} "
            f"min@{s.min_step}={s.min_val:.6g} "
            f"max@{s.max_step}={s.max_val:.6g}"
        )

    pair = find_best_tag_pair([s.tag for s in stats])
    if pair:
        train_tag, val_tag = pair
        print("\nHeuristic analysis:")
        print(f" - detected train/val pair: {train_tag} vs {val_tag}")
        print(f" - train: {simple_convergence_comment(merged[train_tag])}")
        print(f" - val:   {simple_convergence_comment(merged[val_tag])}")

        # overfitting hint: train goes down while val goes up in last 10%
        def mean_last_k(pts, frac=0.1):
            n = len(pts)
            k = max(2, int(n * frac))
            tail = [v for _, v in pts[-k:]]
            return sum(tail) / len(tail)

        def mean_first_k(pts, frac=0.1):
            n = len(pts)
            k = max(2, int(n * frac))
            head = [v for _, v in pts[:k]]
            return sum(head) / len(head)

        tr_first, tr_last = mean_first_k(merged[train_tag]), mean_last_k(merged[train_tag])
        va_first, va_last = mean_first_k(merged[val_tag]), mean_last_k(merged[val_tag])
        if tr_last < tr_first and va_last > va_first * 1.05:
            print(" - overfitting hint: train loss 下降但 val loss 上升（后期）。")
        elif tr_last < tr_first and va_last < va_first:
            print(" - generalization hint: train/val loss 同步下降，泛化趋势较好。")
        else:
            print(" - unclear: train/val 变化不一致，建议结合 LR/指标曲线进一步判断。")
    else:
        print("\nHeuristic analysis:")
        print(" - 未识别到标准 train/val loss tag（例如 train/loss, val/loss）。")
        print(" - 你可以把常用的 tag 名告诉我，我再按你的命名规则做更精确的分析。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
