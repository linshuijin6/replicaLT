"""
extract_tabular.py

从 ADNIMERGE2 R 包中提取 tabular 临床数据，并根据被试编号 (PTID) + 扫描日期
最近邻匹配原则，为 replicaLT 的 train/val JSON 中每条记录补充以下字段：
    AGE, PTGENDER, PTEDUCAT, MMSE, ADAS13, APOE4

匹配逻辑：
  - PTID 精确匹配
  - 扫描日期 (examdate) vs 临床评估日期 (VISDATE) 最近邻匹配
  - 若某指标无记录则填 NaN

APOE4 推导规则（来自 APOERES.GENOTYPE）：
  "<num>/<num>" 中含有 "4" 的等位基因数量 → APOE4 = 0 / 1 / 2

AGE 计算：从 PTDEMOG.PTDOBYY 和 examdate 计算年龄（整数年）。
"""

import json
import os
import warnings
import pyreadr
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────── paths ────────────────────────────
BASE_DIR = Path(__file__).parent
ADNIMERGE_DATA = Path("/mnt/nfsdata/nfsdata/lsj.14/PASTA/replicaLT_comparison/ADNIMERGE2/data")
REPLICA_DIR = Path("/mnt/nfsdata/nfsdata/lsj.14/replicaLT")

JSON_FILES = {
    "train": REPLICA_DIR / "train_data_with_description.json",
    "val":   REPLICA_DIR / "val_data_with_description.json",
}
OUTPUT_DIR = BASE_DIR / "t"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── helpers ──────────────────────────

def read_rda(name: str) -> pd.DataFrame:
    """读取 ADNIMERGE2/data/<name>.rda，返回 DataFrame。"""
    path = ADNIMERGE_DATA / f"{name}.rda"
    r = pyreadr.read_r(str(path))
    return list(r.values())[0]


def to_date(s) -> pd.Timestamp | None:
    """将各种日期格式转为 pd.Timestamp，失败返回 None。"""
    try:
        return pd.Timestamp(str(s))
    except Exception:
        return None


def closest_row(df: pd.DataFrame, ptid: str, exam_date: pd.Timestamp,
                date_col: str = "VISDATE") -> pd.Series | None:
    """
    在 df 中找到 PTID==ptid 且 date_col 与 exam_date 最近的行。
    返回该行的 pd.Series，若无有效匹配则返回 None。
    """
    sub = df[df["PTID"] == ptid].copy()
    if sub.empty:
        return None
    sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
    sub = sub.dropna(subset=[date_col])
    if sub.empty:
        return None
    idx = (sub[date_col] - exam_date).abs().idxmin()
    return sub.loc[idx]


def genotype_to_apoe4(genotype: str) -> int:
    """
    将 ApoE 基因型字符串转为 APOE4 携带数量 (0/1/2)。
    例: "3/4" -> 1, "4/4" -> 2, "3/3" -> 0
    """
    if not isinstance(genotype, str):
        return np.nan
    alleles = genotype.replace(" ", "").split("/")
    return alleles.count("4")


# ─────────────────────────── load rda tables ──────────────────
print("Loading ADNIMERGE2 tables ...")

ptdemog = read_rda("PTDEMOG")   # AGE(via PTDOBYY), PTGENDER, PTEDUCAT
mmse_df = read_rda("MMSE")      # MMSCORE, VISDATE
adas_df = read_rda("ADAS")      # TOTAL13, VISDATE
apoe_df = read_rda("APOERES")   # GENOTYPE (→ APOE4)

# normalise PTID (strip whitespace)
for df in [ptdemog, mmse_df, adas_df, apoe_df]:
    df["PTID"] = df["PTID"].astype(str).str.strip()

print(f"  PTDEMOG: {len(ptdemog)} rows, {ptdemog['PTID'].nunique()} subjects")
print(f"  MMSE:    {len(mmse_df)} rows, {mmse_df['PTID'].nunique()} subjects")
print(f"  ADAS:    {len(adas_df)} rows, {adas_df['PTID'].nunique()} subjects")
print(f"  APOERES: {len(apoe_df)} rows, {apoe_df['PTID'].nunique()} subjects")

# ─── APOE4: one entry per subject (baseline / sc visit preferred) ──
# 使用 VISCODE == 'sc' 或最早记录，每人只保留一条
apoe_df_sorted = apoe_df.sort_values("APTESTDT")
apoe_unique = apoe_df_sorted.drop_duplicates(subset="PTID", keep="first")
apoe_map = dict(zip(apoe_unique["PTID"], apoe_unique["GENOTYPE"]))

# ─── demographics: also mostly one-per-subject (baseline) ──────
ptdemog_sorted = ptdemog.sort_values("VISDATE")
ptdemog_unique = ptdemog_sorted.drop_duplicates(subset="PTID", keep="first")
demog_map = ptdemog_unique.set_index("PTID")[["PTGENDER", "PTEDUCAT", "PTDOBYY"]].to_dict("index")

# ─────────────────────────── main extraction ──────────────────

def extract_record(name: str, examdate: str) -> dict:
    """
    为一条 JSON 扫描记录提取 tabular 字段。
    name:      被试编号字符串，如 "082_S_6287"
    examdate:  扫描日期字符串，如 "2018-05-15"
    """
    ptid = name.strip()
    exam_ts = to_date(examdate)

    result = {
        "AGE":      np.nan,
        "PTGENDER": np.nan,
        "PTEDUCAT": np.nan,
        "MMSE":     np.nan,
        "ADAS13":   np.nan,
        "APOE4":    np.nan,
    }

    # ── demographics ──────────────────────────────────────────
    if ptid in demog_map:
        info = demog_map[ptid]
        result["PTGENDER"] = info.get("PTGENDER", np.nan)
        result["PTEDUCAT"] = info.get("PTEDUCAT", np.nan)
        # compute age from birth year and exam date
        try:
            birth_year = int(info["PTDOBYY"])
            age = exam_ts.year - birth_year
            result["AGE"] = age
        except Exception:
            pass

    # ── MMSE: closest visit ────────────────────────────────────
    row = closest_row(mmse_df, ptid, exam_ts, "VISDATE")
    if row is not None:
        result["MMSE"] = row.get("MMSCORE", np.nan)

    # ── ADAS-Cog-13: closest visit ────────────────────────────
    row = closest_row(adas_df, ptid, exam_ts, "VISDATE")
    if row is not None:
        result["ADAS13"] = row.get("TOTAL13", np.nan)

    # ── ApoE4 ─────────────────────────────────────────────────
    if ptid in apoe_map:
        result["APOE4"] = genotype_to_apoe4(apoe_map[ptid])

    return result


# ─────────────────────────── process JSONs ────────────────────

stats = {
    "total":    0,
    "has_age":  0,
    "has_mmse": 0,
    "has_adas": 0,
    "has_apoe": 0,
}

for split, json_path in JSON_FILES.items():
    out_path = OUTPUT_DIR / f"{split}_tabular.json"
    print(f"\nProcessing {split}: {json_path} ...")

    with open(json_path, "r") as f:
        records = json.load(f)

    enriched = []
    for rec in records:
        tab = extract_record(rec["name"], rec["examdate"])
        rec_out = dict(rec)    # preserve all existing fields
        rec_out.update(tab)    # add / overwrite tabular fields
        enriched.append(rec_out)

        stats["total"] += 1
        if not pd.isna(tab["AGE"]):    stats["has_age"]  += 1
        if not pd.isna(tab["MMSE"]):   stats["has_mmse"] += 1
        if not pd.isna(tab["ADAS13"]): stats["has_adas"] += 1
        if not pd.isna(tab.get("APOE4", np.nan)): stats["has_apoe"] += 1

    with open(out_path, "w") as f:
        json.dump(enriched, f, indent=2, default=str)
    print(f"  → Saved {len(enriched)} records to {out_path}")

# ─────────────────────────── summary ──────────────────────────
print("\n=== Match statistics (across train+val) ===")
total = stats["total"]
for key in ["has_age", "has_mmse", "has_adas", "has_apoe"]:
    label = key.replace("has_", "").upper()
    n = stats[key]
    print(f"  {label:<10}: {n:4d} / {total} ({100*n/total:.1f}%)")
print("\nDone!")
