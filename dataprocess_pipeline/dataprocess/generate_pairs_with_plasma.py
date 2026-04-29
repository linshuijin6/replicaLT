#!/usr/bin/env python3
import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# 输入文件默认路径
BASE = Path("/mnt/nfsdata/nfsdata/lsj.14/replicaLT/adapter_finetune")
PAIRS_CSV = BASE / "data_csv/pairs0106_filtered.csv"
PLASMA_UPENN_CSV = BASE / "data_csv/PLASMA_UPENN_ADNI3&4.0103.csv"
C2N_CSV = BASE / "C2N_PRECIVITYAD2_PLASMA_29Dec2025.csv"
OUT_CSV = BASE / "data_csv/pairs_withPlasma.csv"

UPENN_COLS = {
    "ptau217": "pT217_F",
    "ab42": "AB42_F",
    "ab40": "AB40_F",
    "ab4240": "AB42_AB40_F",
    "ptau217_ab42": "pT217_AB42_F",
    "nfl": "NfL_Q",
    "gfap": "GFAP_Q",
}

C2N_COLS = {
    "ptau217": "pT217_C2N",
    "ab4240": "AB42_AB40_C2N",
    # C2N 提供的是 pT217/npT217 比值，与 pT217/AB42 不同；按需求缺失则留空
    "ptau217_np": "pT217_npT217_C2N",
}

DATE_COLS = {
    "pairs": "EXAMDATE",
    "upenn": "EXAMDATE",
    "c2n": "EXAMDATE",
}


def parse_date(s: str) -> Optional[datetime]:
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    # 多种可能格式
    fmts = [
        "%Y/%m/%d",
        "%Y-%m-%d",
        "%Y/%-m/%-d",
        "%Y-%-m-%-d",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def is_missing(val) -> bool:
    if pd.isna(val):
        return True
    try:
        s = str(val).strip()
    except Exception:
        return True
    if s == "":
        return True
    # 处理缺失哨兵值 -4 / -4.0
    if s in {"-4", "-4.0", "-4.00"}:
        return True
    try:
        f = float(s)
        if f == -4.0:
            return True
    except Exception:
        pass
    return False


def choose_best_row(rows: List[Dict], target_date: Optional[datetime], date_col: str) -> Optional[Dict]:
    if not rows:
        return None
    if target_date is None:
        # 没有目标日期则返回首行（通常是唯一或已按来源最新）
        return rows[0]
    # 选择日期差绝对值最小的记录
    best = None
    best_diff = None
    for r in rows:
        d = parse_date(r.get(date_col, ""))
        if d is None:
            continue
        diff = abs((d - target_date).days)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best = r
    return best or rows[0]


def load_table(path: Path) -> pd.DataFrame:
    # 支持逗号分隔及引号包围的CSV
    return pd.read_csv(path)


def build_index(df: pd.DataFrame, key_col: str = "PTID") -> Dict[str, List[Dict]]:
    idx: Dict[str, List[Dict]] = {}
    for _, row in df.iterrows():
        ptid = str(row.get(key_col, "")).strip()
        if not ptid:
            continue
        idx.setdefault(ptid, []).append(row.to_dict())
    return idx


def generate_pairs_with_plasma(pairs_csv: Path, upenn_csv: Path, c2n_csv: Path, out_csv: Path) -> Tuple[int, Dict[str, int]]:
    pairs_df = load_table(pairs_csv)
    upenn_df = load_table(upenn_csv)
    c2n_df = load_table(c2n_csv)

    upenn_idx = build_index(upenn_df, "PTID")
    c2n_idx = build_index(c2n_df, "PTID")

    output_rows = []
    stats = {
        "rows_total": 0,
        "rows_upenn_match": 0,
        "rows_c2n_match": 0,
        "ptau217_filled": 0,
        "ab4240_filled": 0,
        "ptau217_ab42_filled": 0,
        "nfl_filled": 0,
        "gfap_filled": 0,
    }

    for _, row in pairs_df.iterrows():
        stats["rows_total"] += 1
        ptid = str(row.get("PTID", "")).strip()
        exam_date_str = row.get(DATE_COLS["pairs"], "")
        exam_date = parse_date(exam_date_str)

        # 默认空
        filled = {
            "PTID": ptid,
            "EXAMDATE": exam_date_str,
            "id_mri": row.get("id_mri", ""),
            "id_fdg": row.get("id_fdg", ""),
            "id_av45": row.get("id_av45", ""),
            "id_av1451": row.get("id_av1451", ""),
            # 目标指标
            "AB42_AB40": "",
            "pT217": "",
            "pT217_AB42": "",
            "NfL": "",
            "GFAP": "",
            # 来源标记
            "plasma_source": "",
        }

        # 先查 UPENN
        up_rows = upenn_idx.get(ptid, [])
        best_up = choose_best_row(up_rows, exam_date, DATE_COLS["upenn"]) if up_rows else None
        used_source = None

        def assign_from_upenn(r: Dict):
            nonlocal used_source
            if r is None:
                return
            used_source = "UPENN"
            stats["rows_upenn_match"] += 1
            # AB42/AB40
            v = r.get(UPENN_COLS["ab4240"])  # 已有比值列
            if not is_missing(v):
                filled["AB42_AB40"] = v
                stats["ab4240_filled"] += 1
            # p-tau217
            v = r.get(UPENN_COLS["ptau217"])
            if not is_missing(v):
                filled["pT217"] = v
                stats["ptau217_filled"] += 1
            # p-tau217/Aβ42
            v = r.get(UPENN_COLS["ptau217_ab42"])  # 直接提供
            if not is_missing(v):
                filled["pT217_AB42"] = v
                stats["ptau217_ab42_filled"] += 1
            # NfL
            v = r.get(UPENN_COLS["nfl"]) 
            if not is_missing(v):
                filled["NfL"] = v
                stats["nfl_filled"] += 1
            # GFAP
            v = r.get(UPENN_COLS["gfap"]) 
            if not is_missing(v):
                filled["GFAP"] = v
                stats["gfap_filled"] += 1

        def assign_from_c2n(r: Dict):
            nonlocal used_source
            if r is None:
                return
            filled_any = False
            # 仅作为缺失的回填：AB42/AB40 与 p-tau217
            if is_missing(filled["AB42_AB40"]):
                v = r.get(C2N_COLS["ab4240"])  # C2N 比值列
                if not is_missing(v):
                    filled["AB42_AB40"] = v
                    stats["ab4240_filled"] += 1
                    filled_any = True
            if is_missing(filled["pT217"]):
                v = r.get(C2N_COLS["ptau217"])  # C2N pT217
                if not is_missing(v):
                    filled["pT217"] = v
                    stats["ptau217_filled"] += 1
                    filled_any = True
            # pT217/AB42 在 C2N 中无对应，保持空
            # NfL/GFAP C2N 未提供，保持空
            if filled_any:
                used_source = "C2N"
                stats["rows_c2n_match"] += 1

        assign_from_upenn(best_up)

        # 如 UPENN 未匹配或部分缺失，则尝试 C2N
        if is_missing(filled["AB42_AB40"]) or is_missing(filled["pT217"]) or is_missing(filled["pT217_AB42"]) or is_missing(filled["NfL"]) or is_missing(filled["GFAP"]):
            c_rows = c2n_idx.get(ptid, [])
            best_c2n = choose_best_row(c_rows, exam_date, DATE_COLS["c2n"]) if c_rows else None
            assign_from_c2n(best_c2n)

        filled["plasma_source"] = used_source or ""
        output_rows.append(filled)

    # 写出
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "PTID",
        "EXAMDATE",
        "id_mri",
        "id_fdg",
        "id_av45",
        "id_av1451",
        "AB42_AB40",
        "pT217",
        "pT217_AB42",
        "NfL",
        "GFAP",
        "plasma_source",
    ]
    pd.DataFrame(output_rows)[cols].to_csv(out_csv, index=False)
    return stats["rows_total"], stats


def main():
    parser = argparse.ArgumentParser(description="生成 pairs_withPlasma.csv，优先 UPENN，缺失回填 C2N；无法填充留空")
    parser.add_argument("--pairs", default=str(PAIRS_CSV), help="pairs0106_filtered.csv 路径")
    parser.add_argument("--upenn", default=str(PLASMA_UPENN_CSV), help="PLASMA_UPENN CSV 路径")
    parser.add_argument("--c2n", default=str(C2N_CSV), help="C2N CSV 路径")
    parser.add_argument("--out", default=str(OUT_CSV), help="输出 pairs_withPlasma.csv 路径")
    args = parser.parse_args()

    pairs = Path(args.pairs)
    upenn = Path(args.upenn)
    c2n = Path(args.c2n)
    outp = Path(args.out)

    print("[INFO] 输入路径:")
    print(" - pairs:", pairs)
    print(" - upenn:", upenn)
    print(" - c2n:", c2n)
    print("[INFO] 输出路径:", outp)

    total, stats = generate_pairs_with_plasma(pairs, upenn, c2n, outp)

    print("[DONE] 生成完成")
    print(" - 总行数:", total)
    print(" - 匹配UPENN行数:", stats["rows_upenn_match"])
    print(" - 匹配C2N行数:", stats["rows_c2n_match"])
    print(" - AB42/AB40 填充数:", stats["ab4240_filled"])
    print(" - p-tau217 填充数:", stats["ptau217_filled"])
    print(" - p-tau217/AB42 填充数:", stats["ptau217_ab42_filled"])
    print(" - NfL 填充数:", stats["nfl_filled"])
    print(" - GFAP 填充数:", stats["gfap_filled"])


if __name__ == "__main__":
    main()
