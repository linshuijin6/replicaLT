import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ADAPTER_V2_DIR = SCRIPT_DIR.parent
if str(ADAPTER_V2_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTER_V2_DIR))

from dataset import compute_plasma_stats, normalize_diagnosis

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
except Exception:
    LogisticRegression = None
    roc_auc_score = None
    train_test_split = None


DEFAULT_KEYS = ["AB42_AB40_F", "pT217_F", "pT217_AB42_F", "NfL_Q", "GFAP_Q"]


def _normalize_source(value: str) -> str:
    text = str(value).strip().upper()
    if text in {"", "NAN", "NONE"}:
        return "UPENN"
    if text in {"CN2", "C2N"}:
        return "C2N"
    return text


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _extract_raw_values(df: pd.DataFrame, key: str, na_value: float = -4.0) -> tuple[np.ndarray, np.ndarray]:
    if key not in df.columns:
        n = len(df)
        return np.full(n, np.nan, dtype=np.float64), np.zeros(n, dtype=bool)

    values = _safe_numeric(df[key]).to_numpy(dtype=np.float64)
    source = df["plasma_source"].astype(str).to_numpy()

    if key == "pT217_AB42_F":
        c2n_mask = source == "C2N"
        values[c2n_mask] = values[c2n_mask] / 100.0

    valid = np.isfinite(values) & (values >= 0.0) & (values != na_value)
    values[~valid] = np.nan
    return values, valid


def _normalize_values_by_source(
    raw_values: np.ndarray,
    valid_mask: np.ndarray,
    sources: np.ndarray,
    stats_by_source: Dict[str, Dict[str, Dict[str, float]]],
    key: str,
) -> np.ndarray:
    out = np.full(raw_values.shape[0], np.nan, dtype=np.float64)
    for source in np.unique(sources):
        idx = np.where(sources == source)[0]
        if idx.size == 0:
            continue
        source_stats = stats_by_source.get(source, {}).get(key, {"min": 0.0, "max": 1.0})
        min_v = float(source_stats.get("min", 0.0))
        max_v = float(source_stats.get("max", 1.0))
        denom = max(max_v - min_v, 1e-8)

        src_valid = idx[valid_mask[idx]]
        if src_valid.size == 0:
            continue
        norm = (raw_values[src_valid] - min_v) / denom
        out[src_valid] = np.clip(norm, 0.0, 1.0)
    return out


def _nan_stats(values: np.ndarray) -> Dict[str, float]:
    arr = values[np.isfinite(values)]
    if arr.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "median": float(np.median(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _wasserstein_1d(x: np.ndarray, y: np.ndarray, n_grid: int = 2048) -> float:
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    q = np.linspace(0.0, 1.0, n_grid, endpoint=True)
    xq = np.quantile(x, q)
    yq = np.quantile(y, q)
    return float(np.mean(np.abs(xq - yq)))


def _kl_hist(x: np.ndarray, y: np.ndarray, bins: int = 50) -> float:
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    vmin = float(min(np.min(x), np.min(y)))
    vmax = float(max(np.max(x), np.max(y)))
    if abs(vmax - vmin) < 1e-12:
        return 0.0
    px, edges = np.histogram(x, bins=bins, range=(vmin, vmax), density=False)
    py, _ = np.histogram(y, bins=edges, density=False)

    px = px.astype(np.float64) + 1e-8
    py = py.astype(np.float64) + 1e-8
    px = px / px.sum()
    py = py / py.sum()
    return float(np.sum(px * np.log(px / py)))


def _overlap_coeff(x: np.ndarray, y: np.ndarray, bins: int = 50) -> float:
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    vmin = float(min(np.min(x), np.min(y)))
    vmax = float(max(np.max(x), np.max(y)))
    if abs(vmax - vmin) < 1e-12:
        return 1.0
    px, edges = np.histogram(x, bins=bins, range=(vmin, vmax), density=True)
    py, _ = np.histogram(y, bins=edges, density=True)
    widths = np.diff(edges)
    ovl = np.sum(np.minimum(px, py) * widths)
    return float(max(0.0, min(1.0, ovl)))


def _plot_source_diagnosis(count_df: pd.DataFrame, save_path: Path) -> None:
    pivot = count_df.pivot(index="diagnosis", columns="plasma_source", values="count").fillna(0)
    pivot = pivot.reindex(index=["CN", "MCI", "AD"], fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    bottoms = np.zeros(pivot.shape[0], dtype=np.float64)
    for source in pivot.columns:
        vals = pivot[source].to_numpy(dtype=np.float64)
        ax.bar(pivot.index, vals, bottom=bottoms, label=source)
        bottoms += vals
    ax.set_title("Source × Diagnosis sample counts")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _plot_valid_rate(valid_df: pd.DataFrame, save_path: Path) -> None:
    keys = sorted(valid_df["key"].unique().tolist())
    sources = sorted(valid_df["plasma_source"].unique().tolist())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(keys), dtype=np.float64)
    width = 0.35 if len(sources) <= 2 else 0.8 / max(len(sources), 1)

    for i, source in enumerate(sources):
        sub = valid_df[valid_df["plasma_source"] == source].set_index("key")
        vals = [float(sub.loc[k, "valid_rate"]) if k in sub.index else 0.0 for k in keys]
        shift = (i - (len(sources) - 1) / 2.0) * width
        ax.bar(x + shift, vals, width=width, label=source)

    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Valid rate")
    ax.set_title("Plasma valid rate by source")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _plot_key_dist(
    key: str,
    source_data: Dict[str, np.ndarray],
    title: str,
    x_label: str,
    save_path: Path,
    bins: int = 50,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for source, vals in source_data.items():
        arr = vals[np.isfinite(vals)]
        if arr.size == 0:
            continue
        ax.hist(arr, bins=bins, alpha=0.45, density=True, label=f"{source} (n={arr.size})")
    ax.set_title(f"{title}: {key}")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_csv_path(config: dict, csv_path_arg: str | None, repo_root: Path) -> Path:
    if csv_path_arg is not None:
        return Path(csv_path_arg).expanduser().resolve()

    cfg_csv = config.get("data", {}).get("csv_path", None)
    if cfg_csv:
        cfg_path = Path(cfg_csv).expanduser()
        if cfg_path.exists():
            return cfg_path.resolve()

    fallback = repo_root / "adapter_finetune" / "data_csv" / "pairs_withPlasma.csv"
    if fallback.exists():
        return fallback.resolve()

    raise FileNotFoundError("未找到 CSV：请通过 --csv_path 指定可用文件")


def _build_source_classifier_metrics(df: pd.DataFrame, keys: List[str]) -> Dict[str, float]:
    metrics = {
        "sklearn_available": float(LogisticRegression is not None),
        "auc_values_only": np.nan,
        "auc_values_and_mask": np.nan,
        "n_samples": float(len(df)),
    }
    if LogisticRegression is None or roc_auc_score is None or train_test_split is None:
        return metrics

    src = (df["plasma_source"] == "C2N").astype(np.int64).to_numpy()
    if np.unique(src).size < 2:
        return metrics

    value_cols = [f"norm_{k}" for k in keys]
    mask_cols = [f"mask_{k}" for k in keys]

    x_values = df[value_cols].fillna(0.0).to_numpy(dtype=np.float64)
    x_value_mask = np.concatenate(
        [
            x_values,
            df[mask_cols].fillna(0).to_numpy(dtype=np.float64),
        ],
        axis=1,
    )

    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.3, random_state=42, stratify=src)

    def _fit_auc(features: np.ndarray) -> float:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        clf.fit(features[tr], src[tr])
        prob = clf.predict_proba(features[te])[:, 1]
        return float(roc_auc_score(src[te], prob))

    metrics["auc_values_only"] = _fit_auc(x_values)
    metrics["auc_values_and_mask"] = _fit_auc(x_value_mask)
    return metrics


def _decision(w1: float, auc: float) -> str:
    if np.isfinite(w1) and np.isfinite(auc):
        if w1 > 0.3 or auc > 0.75:
            return "建议显式区分source（source token/embedding或分支参数）"
        if w1 >= 0.1 or auc > 0.6:
            return "建议先做source条件化或损失分组实验"
        return "混用风险可控，优先保持当前方案"
    return "指标不足，需补充样本或检查数据"


def main():
    parser = argparse.ArgumentParser(description="Analyze UPENN/C2N plasma distribution and mixing risks")
    parser.add_argument("--config", type=str, default="adapter_v2/config.yaml")
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="adapter_v2/analysis/plasma_source_report")
    parser.add_argument("--keys", type=str, nargs="*", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config).expanduser().resolve()
    config = _load_config(config_path)
    csv_path = _resolve_csv_path(config, args.csv_path, repo_root)

    out_dir = Path(args.out_dir).expanduser().resolve()
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    id_columns = ["PTID", "id_mri", "id_fdg", "id_av45", "id_av1451", "plasma_source"]
    dtype_spec = {col: str for col in id_columns}
    df = pd.read_csv(csv_path, dtype=dtype_spec)

    if "id_av1451" in df.columns:
        df = df[(df["id_av1451"].notna()) & (df["id_av1451"] != "nan")].copy()

    df["plasma_source"] = df.get("plasma_source", "UPENN").fillna("UPENN").map(_normalize_source)

    diag_raw = None
    for col in ["diagnosis", "DX", "research_group", "DIAGNOSIS"]:
        if col in df.columns:
            diag_raw = df[col]
            break
    if diag_raw is None:
        df["diagnosis_norm"] = "UNKNOWN"
    else:
        df["diagnosis_norm"] = diag_raw.map(lambda x: normalize_diagnosis(x) or "UNKNOWN")

    if args.keys:
        keys = args.keys
    else:
        plasma_cfg = config.get("plasma", {})
        keys = plasma_cfg.get("available_keys", DEFAULT_KEYS)

    stats_by_source = compute_plasma_stats(
        df=df,
        plasma_keys=keys,
        by_source=True,
        source_col="plasma_source",
        na_value=-4.0,
    )

    source_diag = (
        df.groupby(["plasma_source", "diagnosis_norm"], dropna=False)
        .size()
        .reset_index(name="count")
        .rename(columns={"diagnosis_norm": "diagnosis"})
    )
    source_diag.to_csv(out_dir / "source_diagnosis_counts.csv", index=False)
    _plot_source_diagnosis(source_diag, fig_dir / "source_diagnosis_counts.png")

    rows = []
    dist_rows = []
    for key in keys:
        raw_vals, valid_mask = _extract_raw_values(df, key=key, na_value=-4.0)
        norm_vals = _normalize_values_by_source(
            raw_values=raw_vals,
            valid_mask=valid_mask,
            sources=df["plasma_source"].to_numpy(),
            stats_by_source=stats_by_source,
            key=key,
        )

        df[f"raw_{key}"] = raw_vals
        df[f"norm_{key}"] = norm_vals
        df[f"mask_{key}"] = valid_mask.astype(np.int64)

        source_data_raw = {}
        source_data_norm = {}
        for source in sorted(df["plasma_source"].unique().tolist()):
            src_idx = df["plasma_source"] == source
            src_raw = raw_vals[src_idx.to_numpy()]
            src_norm = norm_vals[src_idx.to_numpy()]
            source_data_raw[source] = src_raw
            source_data_norm[source] = src_norm

            n_total = int(src_idx.sum())
            n_valid = int(np.isfinite(src_raw).sum())
            stat_raw = _nan_stats(src_raw)
            stat_norm = _nan_stats(src_norm)
            rows.append(
                {
                    "key": key,
                    "plasma_source": source,
                    "n_total": n_total,
                    "n_valid": n_valid,
                    "valid_rate": float(n_valid / max(n_total, 1)),
                    **{f"raw_{k}": v for k, v in stat_raw.items()},
                    **{f"norm_{k}": v for k, v in stat_norm.items()},
                }
            )

        _plot_key_dist(
            key=key,
            source_data=source_data_raw,
            title="Raw plasma distribution by source",
            x_label="Raw value",
            save_path=fig_dir / f"raw_dist_{key}.png",
        )
        _plot_key_dist(
            key=key,
            source_data=source_data_norm,
            title="Normalized plasma distribution by source",
            x_label="Normalized value",
            save_path=fig_dir / f"norm_dist_{key}.png",
        )

        srcs = sorted(source_data_raw.keys())
        if len(srcs) >= 2:
            a, b = srcs[0], srcs[1]
            dist_rows.append(
                {
                    "key": key,
                    "scope": "all",
                    "raw_w1": _wasserstein_1d(source_data_raw[a], source_data_raw[b]),
                    "norm_w1": _wasserstein_1d(source_data_norm[a], source_data_norm[b]),
                    "raw_kl": _kl_hist(source_data_raw[a], source_data_raw[b]),
                    "norm_kl": _kl_hist(source_data_norm[a], source_data_norm[b]),
                    "raw_overlap": _overlap_coeff(source_data_raw[a], source_data_raw[b]),
                    "norm_overlap": _overlap_coeff(source_data_norm[a], source_data_norm[b]),
                    "source_a": a,
                    "source_b": b,
                }
            )

            for diagnosis in ["CN", "MCI", "AD"]:
                d_idx = df["diagnosis_norm"] == diagnosis
                a_vals_raw = raw_vals[(df["plasma_source"] == a).to_numpy() & d_idx.to_numpy()]
                b_vals_raw = raw_vals[(df["plasma_source"] == b).to_numpy() & d_idx.to_numpy()]
                a_vals_norm = norm_vals[(df["plasma_source"] == a).to_numpy() & d_idx.to_numpy()]
                b_vals_norm = norm_vals[(df["plasma_source"] == b).to_numpy() & d_idx.to_numpy()]
                dist_rows.append(
                    {
                        "key": key,
                        "scope": diagnosis,
                        "raw_w1": _wasserstein_1d(a_vals_raw, b_vals_raw),
                        "norm_w1": _wasserstein_1d(a_vals_norm, b_vals_norm),
                        "raw_kl": _kl_hist(a_vals_raw, b_vals_raw),
                        "norm_kl": _kl_hist(a_vals_norm, b_vals_norm),
                        "raw_overlap": _overlap_coeff(a_vals_raw, b_vals_raw),
                        "norm_overlap": _overlap_coeff(a_vals_norm, b_vals_norm),
                        "source_a": a,
                        "source_b": b,
                    }
                )

    valid_df = pd.DataFrame(rows)
    valid_df.to_csv(out_dir / "plasma_key_stats_by_source.csv", index=False)
    _plot_valid_rate(valid_df[["key", "plasma_source", "valid_rate"]], fig_dir / "valid_rate_by_source.png")

    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(out_dir / "distance_metrics_by_key.csv", index=False)

    source_clf = _build_source_classifier_metrics(df, keys=keys)

    all_scope = dist_df[dist_df["scope"] == "all"] if not dist_df.empty else pd.DataFrame()
    if not all_scope.empty:
        ref_row = all_scope.sort_values(by="norm_w1", ascending=False).iloc[0]
        worst_key = str(ref_row["key"])
        worst_norm_w1 = float(ref_row["norm_w1"])
    else:
        worst_key = "N/A"
        worst_norm_w1 = float("nan")

    summary = {
        "csv_path": str(csv_path),
        "n_rows": int(len(df)),
        "sources": sorted(df["plasma_source"].unique().tolist()),
        "keys": keys,
        "selected_keys_from_config": config.get("plasma", {}).get("selected_keys", []),
        "worst_norm_w1_key": worst_key,
        "worst_norm_w1": worst_norm_w1,
        "source_classifier": source_clf,
        "decision": _decision(w1=worst_norm_w1, auc=float(source_clf.get("auc_values_only", np.nan))),
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("Plasma source analysis done")
    print(f"CSV: {csv_path}")
    print(f"Output: {out_dir}")
    print(f"Rows: {len(df)} | Sources: {summary['sources']}")
    print(f"Worst norm-W1 key: {worst_key} ({worst_norm_w1:.4f})")
    print(f"AUC(values_only): {source_clf.get('auc_values_only', np.nan)}")
    print(f"Decision: {summary['decision']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
