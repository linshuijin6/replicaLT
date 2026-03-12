"""
precompute_plasma_emb.py
========================
离线预计算 plasma_emb，供 plasma_train.py 使用。

信号流（每个样本）:
  1. 加载 MRI vision cache → mri_tokens (196, 768)
  2. 从 plasma CSV 匹配 plasma_values + plasma_mask
  3. [冻结] CoCoOpTAUModel 前向:
     mri_tokens → mri_token_pool → g_mri (512,)
     g_mri → context_net → T_ctx (4, 768)
     base_ctx_plasma + T_ctx → BiomedCLIP TextEncoder → plasma_features (K, 512)
     plasma_weights × plasma_features → proj_plasma → plasma_emb (512,)
  4. 保存: {ptid}_plasma_emb.pt → {"plasma_emb": (512,), "plasma_mask": (K,)}

用法:
    python precompute_plasma_emb.py \
        --cocoop-ckpt  adapter_v2/runs/.../ckpt_best.pt \
        --config       adapter_v2/config.yaml \
        --output-dir   /path/to/plasma_emb_cache \
        --paired-json  train_data_with_description.json val_data_with_description.json \
        --gpu 5
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import yaml
import numpy as np
import pandas as pd
import torch

# adapter_v2 在同级目录
SCRIPT_DIR = Path(__file__).resolve().parent
ADAPTER_V2_DIR = SCRIPT_DIR / "adapter_v2"
if str(ADAPTER_V2_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTER_V2_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def load_cocoop_model(ckpt_path: str, config: dict, device: torch.device):
    """从 checkpoint 加载 CoCoOpTAUModel（冻结所有参数）"""
    from adapter_v2.models import CoCoOpTAUModel
    from adapter_v2.train import resolve_plasma_config

    selected_keys, plasma_prompts, inter_norm, intra_norm, intra_temp, _ = resolve_plasma_config(config)

    model_cfg = config["model"]
    class_names = config["classes"]["names"]
    prompt_template = config["classes"]["prompt_template"]

    model = CoCoOpTAUModel(
        biomedclip_path=model_cfg["biomedclip_path"],
        class_names=class_names,
        class_prompt_template=prompt_template,
        plasma_prompts=plasma_prompts,
        ctx_len=model_cfg.get("ctx_len", 4),
        proj_dim=model_cfg.get("proj_dim", 512),
        ctx_hidden_dim=model_cfg.get("ctx_hidden_dim", 1024),
        share_ctx_base=model_cfg.get("share_ctx_base", False),
        plasma_temperature=config["plasma"].get("temperature", 1.0),
        intra_norm=intra_norm,
        intra_temperature=intra_temp,
    )

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print(f"✅ CoCoOpTAUModel loaded from {ckpt_path}")
    return model, selected_keys, inter_norm


def load_plasma_table(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    if "examdate" in df.columns:
        df["examdate"] = pd.to_datetime(df["examdate"], errors="coerce")
    return df


def pick_plasma_row(plasma_table: pd.DataFrame, ptid: str) -> Dict[str, float]:
    """为 ptid 匹配最近的 plasma 记录"""
    df = plasma_table
    hit = df[df["ptid"] == ptid]
    if hit.empty:
        return {}
    row = hit.iloc[0].to_dict()
    for k in ["ab42_ab40_f", "pt217_ab42_f", "nfl_q", "gfap_q", "ab42_f", "ab40_f", "pt217_f"]:
        if k in row:
            try:
                row[k] = float(row[k])
            except Exception:
                pass
    return {k.upper(): v for k, v in row.items()}


def normalize_plasma_values(
    raw_row: Dict[str, float],
    selected_keys: List[str],
    plasma_stats: dict,
    inter_norm: str = "minmax",
    na_value: float = -4.0,
) -> tuple:
    """
    归一化 plasma 值，返回 (values_tensor, mask_tensor)

    Args:
        raw_row: 大写 key 的 plasma 字典
        selected_keys: 选定的 plasma key 列表
        plasma_stats: 通过 compute_plasma_stats 获得的统计量
        inter_norm: "minmax" 或 "zscore"
        na_value: 缺失标记值

    Returns:
        (FloatTensor(K,), BoolTensor(K,))
    """
    vals = []
    mask = []
    # plasma_stats 可能是 by_source 格式或 flat 格式
    # 这里简化：取第一个可用 source 的统计
    if isinstance(next(iter(plasma_stats.values()), None), dict) and not all(
        k in ("min", "max", "mean", "std") for k in next(iter(plasma_stats.values())).keys()
    ):
        # by_source 格式 — 取第一个 source
        source_key = next(iter(plasma_stats.keys()))
        stats = plasma_stats[source_key]
    else:
        stats = plasma_stats

    for key in selected_keys:
        raw = raw_row.get(key, None)
        if raw is None:
            vals.append(0.0)
            mask.append(False)
            continue
        try:
            fval = float(raw)
        except (ValueError, TypeError):
            vals.append(0.0)
            mask.append(False)
            continue
        if fval < 0 or fval == na_value or not np.isfinite(fval):
            vals.append(0.0)
            mask.append(False)
            continue

        key_stats = stats.get(key, {"min": 0.0, "max": 1.0, "mean": 0.0, "std": 1.0})
        if inter_norm == "zscore":
            normed = (fval - key_stats["mean"]) / key_stats["std"]
        else:
            denom = key_stats["max"] - key_stats["min"]
            if abs(denom) < 1e-8:
                denom = 1.0
            normed = (fval - key_stats["min"]) / denom
            normed = max(0.0, min(1.0, normed))
        vals.append(normed)
        mask.append(True)

    return torch.tensor(vals, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)


def find_mri_cache(ptid: str, mri_cache_dir: Path) -> Optional[Path]:
    """在 mri_cache_dir 中查找 {ptid}_*.mri_vision.pt"""
    pattern = f"{ptid}_*.mri_vision.pt"
    matches = list(mri_cache_dir.glob(pattern))
    if matches:
        return matches[0]
    # fallback: ptid 可能带下划线，直接 glob
    for f in mri_cache_dir.iterdir():
        if f.name.startswith(ptid) and f.name.endswith(".mri_vision.pt"):
            return f
    return None


def main():
    parser = argparse.ArgumentParser(description="预计算 plasma_emb 用于扩散模型训练")
    parser.add_argument("--cocoop-ckpt", type=str, required=True, help="CoCoOpTAUModel checkpoint 路径")
    parser.add_argument("--config", type=str, default="adapter_v2/config.yaml", help="adapter_v2 config.yaml")
    parser.add_argument("--output-dir", type=str, required=True, help="plasma_emb 输出目录")
    parser.add_argument("--paired-json", nargs="+", required=True, help="主 train.py 使用的 JSON 文件")
    parser.add_argument("--plasma-csv", type=str,
                        default="adapter_finetune/ADNI_csv/UPENN_PLASMA_FUJIREBIO_QUANTERIX_21Dec2025.csv")
    parser.add_argument("--mri-cache-dir", type=str, default=None,
                        help="MRI vision 缓存目录 (默认使用 config 中的 mri_cache_dir)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU 卡号")
    parser.add_argument("--force", action="store_true", help="强制重新计算已存在的缓存")
    args = parser.parse_args()

    # 设置 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 加载模型
    model, selected_keys, inter_norm = load_cocoop_model(args.cocoop_ckpt, config, device)

    # MRI cache 目录
    mri_cache_dir = Path(args.mri_cache_dir) if args.mri_cache_dir else Path(config["data"]["mri_cache_dir"])
    print(f"📂 MRI cache 目录: {mri_cache_dir}")

    # 加载 plasma CSV 并计算统计量
    from adapter_v2.dataset import compute_plasma_stats
    plasma_csv = pd.read_csv(args.plasma_csv)
    plasma_csv.columns = [c.strip() for c in plasma_csv.columns]
    plasma_table = load_plasma_table(args.plasma_csv)
    plasma_stats = compute_plasma_stats(
        plasma_csv, selected_keys, by_source=True,
        source_col="plasma_source", na_value=-4.0, norm_type=inter_norm,
    )
    print(f"📊 Plasma stats computed for keys: {selected_keys}")

    # 收集所有需要处理的 ptid
    all_ptids = set()
    for json_path in args.paired_json:
        with open(json_path, "r") as f:
            data = json.load(f)
        for item in data:
            all_ptids.add(item["name"])
    print(f"🔢 总样本数: {len(all_ptids)}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 预计算
    success, skip, fail = 0, 0, 0
    from tqdm import tqdm
    for ptid in tqdm(sorted(all_ptids), desc="预计算 plasma_emb"):
        out_path = output_dir / f"{ptid}_plasma_emb.pt"
        if out_path.exists() and not args.force:
            skip += 1
            continue

        # 查找 MRI cache
        mri_cache_path = find_mri_cache(ptid, mri_cache_dir)
        if mri_cache_path is None:
            print(f"⚠️  {ptid}: MRI cache 未找到，使用零向量 mri_tokens")
            mri_tokens = torch.zeros(196, 768)
        else:
            cache_payload = torch.load(str(mri_cache_path), map_location="cpu")
            # mri_vision.pt 通常包含 region_token (196, 768)
            mri_tokens = cache_payload.get("region_token", cache_payload.get("patch_tokens", torch.zeros(196, 768)))

        # 获取 plasma 值
        plasma_row = pick_plasma_row(plasma_table, ptid)
        plasma_values, plasma_mask = normalize_plasma_values(
            plasma_row, selected_keys, plasma_stats, inter_norm
        )

        # 前向推理（只需要 plasma 分支的输出）
        with torch.no_grad():
            mri_tokens_batch = mri_tokens.unsqueeze(0).to(device)  # (1, N, 768)
            plasma_values_batch = plasma_values.unsqueeze(0).to(device)  # (1, K)
            plasma_mask_batch = plasma_mask.unsqueeze(0).to(device)  # (1, K)

            # 使用一个虚拟的 tau_tokens（不使用 img_emb 输出）
            dummy_tau = torch.zeros_like(mri_tokens_batch)
            dummy_diag = torch.zeros(1, dtype=torch.long, device=device)

            outputs = model(
                tau_tokens=dummy_tau,
                mri_tokens=mri_tokens_batch,
                diagnosis_id=dummy_diag,
                plasma_values=plasma_values_batch,
                plasma_mask=plasma_mask_batch,
            )
            plasma_emb = outputs["plasma_emb"].squeeze(0).cpu()  # (512,)

        # 保存
        torch.save({
            "plasma_emb": plasma_emb,
            "plasma_mask": plasma_mask.cpu(),
        }, str(out_path))
        success += 1

    print(f"\n✅ 完成: success={success}, skip={skip}, fail={fail}")
    print(f"📁 输出目录: {output_dir}")


if __name__ == "__main__":
    main()
