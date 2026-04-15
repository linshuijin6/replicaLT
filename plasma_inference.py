"""
plasma_inference.py
===================
plasma_train.py 训练好的 checkpoint 推理评估脚本。

功能：
  - 加载 plasma_train.py 产出的 checkpoint（自动适配单/多 GPU 权重）
  - 在验证集上通过 rectified-flow Euler 积分生成 TAU PET
  - 逐样本计算 SSIM / PSNR / MAE / MSE
  - 输出：NIfTI 文件、三视面对比图、逐样本 CSV、汇总 JSON

用法：
  python plasma_inference.py --ckpt runs/plasma_xx/best_model.pt [--gpu 0] [--n_steps 1]
  python plasma_inference.py --ckpt runs/plasma_xx/best_model.pt --no_figures --max_samples 10
"""

import os
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Plasma MRI→PET inference & evaluation")
    p.add_argument("--ckpt", required=True, help="Checkpoint .pt / .pth path")
    p.add_argument("--gpu", type=int, default=0, help="Single GPU id for inference")
    p.add_argument("--n_steps", type=int, default=1,
                   help="Euler 积分步数 (1=单步，与训练 val 一致；>1 可能改善质量)")
    p.add_argument("--output_dir", default=None,
                   help="输出目录 (默认: checkpoint 同级 inference_results/)")
    p.add_argument("--plasma_emb_dir", default="/mnt/linshuijin/ADNI_plasma_cache",
                   help="预计算 plasma_emb 目录")
    p.add_argument("--val_json", default="./val_data_with_description.json",
                   help="验证集 JSON 路径")
    p.add_argument("--save_nifti", action="store_true", default=True,
                   help="保存 NIfTI 文件")
    p.add_argument("--no_nifti", dest="save_nifti", action="store_false")
    p.add_argument("--save_figures", action="store_true", default=True,
                   help="保存三视面对比图")
    p.add_argument("--no_figures", dest="save_figures", action="store_false")
    p.add_argument("--max_samples", type=int, default=None,
                   help="限制样本数（快速测试用）")
    p.add_argument("--legacy", action="store_true", default=False,
                   help="Legacy 对比模式：使用 train.py 产出的 checkpoint，"
                        "Token 0 改为 BiomedCLIP 编码的 old_descr 文本（替代 plasma_emb）")
    return p.parse_args()


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import json
import time
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F
from tqdm import tqdm
from monai.data import PersistentDataset, DataLoader
import monai.transforms as mt
from generative.networks.nets import DiffusionModelUNet
from generative.metrics import SSIMMetric
from monai.metrics import PSNRMetric


# ================================================================
# 工具类与函数（与 plasma_train.py 完全一致）
# ================================================================

def mat_load(filepath):
    if filepath is None:
        return None
    return nib.load(filepath).get_fdata()


def get_subject_id(filename):
    parts = filename.split("_")
    return f"{parts[0]}_{parts[1]}_{parts[2]}"


class FillMissingPET(mt.MapTransform):
    def __init__(self, keys, ref_key="mri"):
        super().__init__(keys)
        self.ref_key = ref_key

    def __call__(self, data):
        d = dict(data)
        ref = d.get(self.ref_key)
        if ref is None:
            return d
        for key in self.keys:
            if key == self.ref_key:
                continue
            if key not in d or d[key] is None:
                d[key] = np.zeros_like(ref)
        return d


class ReduceTo3D(mt.MapTransform):
    def __init__(self, keys, reduce="mean"):
        super().__init__(keys)
        self.reduce = reduce

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            v = d.get(k)
            if isinstance(v, np.ndarray) and v.ndim == 4:
                if self.reduce == "mean":
                    d[k] = v.mean(axis=-1)
                elif self.reduce == "max":
                    d[k] = v.max(axis=-1)
                elif self.reduce == "mid":
                    d[k] = v[..., v.shape[-1] // 2]
                else:
                    d[k] = v.mean(axis=-1)
            elif hasattr(v, "ndim") and v.ndim == 4:
                if self.reduce == "mean":
                    d[k] = v.mean(dim=-1)
                elif self.reduce == "max":
                    d[k] = torch.max(v, dim=-1).values
                elif self.reduce == "mid":
                    d[k] = v[..., v.shape[-1] // 2]
                else:
                    d[k] = v.mean(dim=-1)
        return d


# ================================================================
# 可视化
# ================================================================

def get_mid_slices(volume):
    """提取三视面中间切片，归一化到 0-1"""
    if isinstance(volume, torch.Tensor):
        vol = volume.detach().cpu().numpy()
    else:
        vol = np.array(volume)
    while vol.ndim > 3:
        vol = vol[0]
    h, w, d = vol.shape
    slices = {
        "axial": vol[:, :, d // 2],
        "coronal": vol[:, w // 2, :],
        "sagittal": vol[h // 2, :, :],
    }
    for key in slices:
        s = slices[key]
        s_min, s_max = s.min(), s.max()
        slices[key] = (s - s_min) / (s_max - s_min + 1e-8)
    return slices


def save_comparison_figure(mri, gt, pred, save_path, title=""):
    """生成 MRI | GT | Pred | |Diff| 三视面对比图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mri_s = get_mid_slices(mri)
    gt_s = get_mid_slices(gt)
    pred_s = get_mid_slices(pred)
    diff = torch.abs(pred.float() - gt.float()) if isinstance(pred, torch.Tensor) else np.abs(pred - gt)
    diff_s = get_mid_slices(diff)

    views = ["axial", "coronal", "sagittal"]
    cols = [
        ("MRI", mri_s, "gray"),
        ("GT TAU", gt_s, "gray"),
        ("Pred TAU", pred_s, "gray"),
        ("|Diff|", diff_s, "jet"),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for row, view in enumerate(views):
        for col, (name, slices_dict, cmap) in enumerate(cols):
            ax = axes[row, col]
            ax.imshow(slices_dict[view], cmap=cmap, vmin=0, vmax=1)
            ax.axis("off")
            if row == 0:
                ax.set_title(name, fontsize=12)
        axes[row, 0].set_ylabel(view, fontsize=11, rotation=90, labelpad=10)
    if title:
        fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ================================================================
# 模型加载
# ================================================================

def load_checkpoint_to_model(ckpt_path, device):
    """加载 checkpoint 到单 GPU DiffusionModelUNet（自动处理分布式权重前缀）"""
    model = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=(32, 64, 64, 128),
        attention_levels=(False, False, False, True),
        num_res_blocks=1,
        num_head_channels=(0, 0, 0, 128),
        with_conditioning=True,
        cross_attention_dim=512,
        use_flash_attention=True,
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt

    # 尝试直接加载
    try:
        model.load_state_dict(state_dict)
        print("✅ 权重直接加载成功")
    except RuntimeError:
        # 尝试去除分布式前缀
        cleaned = OrderedDict()
        for k, v in state_dict.items():
            new_k = k
            for prefix in ("module.", "model."):
                if new_k.startswith(prefix):
                    new_k = new_k[len(prefix):]
            cleaned[new_k] = v
        model.load_state_dict(cleaned)
        print("✅ 权重去除分布式前缀后加载成功")

    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", None)
    global_step = ckpt.get("global_step", "?")
    print(f"   epoch={epoch}, global_step={global_step}, val_loss={val_loss}")

    model.to(device)
    model.eval()
    return model, ckpt


# ================================================================
# 主流程
# ================================================================

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"📍 Device: {device}")

    # ---- 输出目录 ----
    ckpt_path = Path(args.ckpt).resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ckpt_path.parent / "inference_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 输出目录: {output_dir}")

    n_steps = args.n_steps
    plasma_emb_dir = args.plasma_emb_dir

    # ================================================================
    # 1. 加载模型
    # ================================================================
    print(f"\n🔧 加载 checkpoint: {ckpt_path}")
    model, ckpt_data = load_checkpoint_to_model(str(ckpt_path), device)

    # ================================================================
    # 2. BiomedCLIP 模态文本编码 (Token 1)
    # ================================================================
    from transformers import AutoProcessor, AutoModel

    local_model_path = "./BiomedCLIP"
    processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
    bio_model = AutoModel.from_pretrained(local_model_path, trust_remote_code=True)
    bio_model.to(device).eval()

    tau_text = (
        "TAU PET is a molecular neuroimaging technique that visualizes the spatial distribution of "
        "aggregated tau protein, which reflects the presence of neurofibrillary tangles associated "
        "with neurodegeneration. Tau PET highlights region-specific tau accumulation, particularly "
        "in medial temporal, parietal, and association cortices, providing a topographical map of "
        "tau pathology that correlates with disease stage, cognitive decline, and neuronal dysfunction."
    )
    inputs = processor(
        text=[tau_text], return_tensors="pt", padding=True, truncation=True, max_length=256
    ).to(device)
    with torch.no_grad():
        tau_feature_optimized = bio_model.get_text_features(**inputs)  # (1, 512)
    tau_feature_cpu = tau_feature_optimized.cpu()

    # ================================================================
    # 3. 加载验证集数据（筛选有 TAU GT 的样本）
    # ================================================================
    with open(args.val_json) as f:
        val_data_raw = json.load(f)

    # Legacy 模式：预构建 name→old_descr 映射（后续编码 Token 0 时使用）
    _legacy_raw_map = None
    if args.legacy:
        print("\n📎 Legacy 模式: 将使用 BiomedCLIP 编码 old_descr 作为 Token 0")
        _legacy_raw_map = {
            item.get("name", ""): item.get("old_descr", "NA") or "NA"
            for item in val_data_raw
        }
    else:
        del bio_model, processor
        torch.cuda.empty_cache()
        print("✅ BiomedCLIP 模态文本编码完成，已释放显存")

    # 从文件系统补全可能缺失的 tau 路径
    base_dir = "/mnt/nfsdata/nfsdata/lsj.14/ADNI_CSF"
    tau_dir = os.path.join(base_dir, "PET_MNI", "TAU")
    tau_dict = {}
    if os.path.isdir(tau_dir):
        for f in sorted(os.listdir(tau_dir)):
            if f.lower().endswith(".nii.gz") and "pet2mni" not in f.lower() and "full" not in f.lower():
                tau_dict[get_subject_id(f)] = os.path.join(tau_dir, f)

    val_data_filtered = []
    for item in val_data_raw:
        tau_path = item.get("tau") or tau_dict.get(item["name"])
        if tau_path and os.path.exists(tau_path):
            val_data_filtered.append({**item, "tau": tau_path})

    if args.max_samples:
        val_data_filtered = val_data_filtered[: args.max_samples]
    print(f"\n📊 验证样本: {len(val_data_filtered)} (有 TAU GT) / 全部 {len(val_data_raw)}")

    if len(val_data_filtered) == 0:
        print("❌ 没有可用样本，请检查数据路径")
        return

    # ================================================================
    # 4. 加载 Token 0（plasma_emb 或 legacy desc_text）
    # ================================================================
    if args.legacy:
        # Legacy: 编码每个样本的 old_descr 文本为 Token 0
        print("\n📎 Legacy: BiomedCLIP 编码 old_descr → desc_text_features")
        ordered_descrs = [_legacy_raw_map.get(item["name"], "NA") for item in val_data_filtered]
        text_inputs = processor(
            text=ordered_descrs, padding=True, truncation=True,
            return_tensors="pt", max_length=256,
        ).to(device)
        with torch.no_grad():
            val_plasma_embs = bio_model.get_text_features(**text_inputs).cpu()  # (N, 512)
        print(f"   desc_text_features shape: {val_plasma_embs.shape}")
        norms = val_plasma_embs.norm(dim=-1)
        print(f"   L2 norm: mean={norms.mean():.3f}, min={norms.min():.3f}, max={norms.max():.3f}")
        # 编码完毕，释放 BiomedCLIP
        del bio_model, processor
        torch.cuda.empty_cache()
        print("✅ BiomedCLIP 已释放")
    else:
        print(f"\n🧬 加载 plasma_emb: {plasma_emb_dir}")
        val_plasma_embs = []
        missing = 0
        for item in val_data_filtered:
            ptid = item["name"]
            emb_path = os.path.join(plasma_emb_dir, f"{ptid}_plasma_emb.pt")
            if os.path.exists(emb_path):
                payload = torch.load(emb_path, map_location="cpu")
                val_plasma_embs.append(payload["plasma_emb"])
            else:
                val_plasma_embs.append(torch.zeros(512))
                missing += 1
        val_plasma_embs = torch.stack(val_plasma_embs)  # (N, 512)
        print(f"   shape: {val_plasma_embs.shape}, 缺失: {missing}/{len(val_data_filtered)}")

    # ================================================================
    # 5. 构建 DataLoader（transforms 与训练 val 完全一致）
    # ================================================================
    def tau_index_transform(x):
        """context = [Token0, TAU_modality_text] → (2, 512)
        Token0: plasma_emb (默认) / desc_text_features (--legacy)"""
        return torch.cat([val_plasma_embs[x].unsqueeze(0), tau_feature_cpu], dim=0)

    pet_keys = ["mri", "tau"]
    val_data = [
        {
            "name": item["name"],
            "mri": item["mri"],
            "tau": item["tau"],
            "tau_index": idx,
        }
        for idx, item in enumerate(val_data_filtered)
    ]

    transforms = mt.Compose([
        mt.Lambdad(keys=pet_keys, func=mat_load),
        ReduceTo3D(keys=pet_keys, reduce="mean"),
        FillMissingPET(keys=pet_keys, ref_key="mri"),
        mt.CropForegroundd(keys=pet_keys, source_key="mri"),
        mt.EnsureChannelFirstd(keys=pet_keys, channel_dim="no_channel"),
        mt.HistogramNormalized(keys=["mri"]),
        mt.ResizeWithPadOrCropd(keys=pet_keys, spatial_size=[160, 192, 160]),
        mt.Spacingd(keys=pet_keys, pixdim=(1.0, 1.0, 1.0)),
        mt.NormalizeIntensityd(keys=pet_keys),
        mt.ScaleIntensityd(keys=pet_keys),
        mt.Lambdad(keys=["tau_index"], func=tau_index_transform),
    ])

    cache_dir = str(output_dir / "cache")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"\n📦 创建 PersistentDataset (cache: {cache_dir})")
    val_ds = PersistentDataset(data=val_data, transform=transforms, cache_dir=cache_dir)

    # 预热缓存
    print("   检查缓存...")
    for i in tqdm(range(len(val_ds)), desc="缓存预热", ncols=70, leave=False):
        try:
            _ = val_ds[i]
        except Exception as e:
            print(f"   ⚠️ 样本 {i} ({val_data[i]['name']}) 加载失败: {e}")

    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    # ================================================================
    # 6. 推理
    # ================================================================
    ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=7)
    psnr_metric = PSNRMetric(max_val=1.0)

    nifti_dir = output_dir / "nifti"
    figure_dir = output_dir / "figures"
    if args.save_nifti:
        nifti_dir.mkdir(exist_ok=True)
    if args.save_figures:
        figure_dir.mkdir(exist_ok=True)

    results = []
    clip_min, clip_max = 0, 1

    print(f"\n🚀 开始推理 (n_steps={n_steps})...")
    t_start = time.time()

    with torch.no_grad():
        for step, batch in enumerate(tqdm(val_loader, desc="Inference", ncols=100)):
            mri = batch["mri"].to(device)
            tau_gt = batch["tau"].to(device)
            tau_index = batch["tau_index"].to(device)
            name = batch["name"]
            if isinstance(name, (list, tuple)):
                name = name[0]

            # 跳过无 GT 的样本（全零填充）
            if torch.all(tau_gt == 0):
                tqdm.write(f"  {name} | 无 TAU GT, 跳过")
                continue

            # Rectified-flow Euler 积分
            x_t = mri.clone()
            for i in range(n_steps):
                t_val = i / n_steps
                time_emb = torch.Tensor([int(t_val * 1000)]).to(device)
                v_pred = model(x=x_t, timesteps=time_emb, context=tau_index)
                x_t = x_t + v_pred / n_steps
                x_t = torch.clamp(x_t, min=clip_min, max=clip_max)

            tau_pred = x_t

            # 逐样本指标
            ssim_val = ssim_metric(tau_gt.cpu(), tau_pred.cpu()).mean().item()
            psnr_val = psnr_metric(tau_gt.cpu(), tau_pred.cpu()).mean().item()
            mae_val = F.l1_loss(tau_pred.float(), tau_gt.float()).item()
            mse_val = F.mse_loss(tau_pred.float(), tau_gt.float()).item()

            results.append({
                "name": name,
                "ssim": ssim_val,
                "psnr": psnr_val,
                "mae": mae_val,
                "mse": mse_val,
            })

            tqdm.write(
                f"  {name} | SSIM={ssim_val:.4f}  PSNR={psnr_val:.2f}  MAE={mae_val:.4f}"
            )

            # 保存 NIfTI
            if args.save_nifti:
                affine = np.eye(4)
                pred_np = tau_pred.squeeze().cpu().numpy().astype(np.float32)
                nib.save(
                    nib.Nifti1Image(pred_np, affine),
                    str(nifti_dir / f"{name}_tau_pred.nii.gz"),
                )
                gt_np = tau_gt.squeeze().cpu().numpy().astype(np.float32)
                nib.save(
                    nib.Nifti1Image(gt_np, affine),
                    str(nifti_dir / f"{name}_tau_gt.nii.gz"),
                )
                mri_np = mri.squeeze().cpu().numpy().astype(np.float32)
                nib.save(
                    nib.Nifti1Image(mri_np, affine),
                    str(nifti_dir / f"{name}_mri.nii.gz"),
                )

            # 保存三视面对比图
            if args.save_figures:
                save_comparison_figure(
                    mri.cpu(),
                    tau_gt.cpu(),
                    tau_pred.cpu(),
                    str(figure_dir / f"{name}_comparison.png"),
                    title=f"{name}  |  SSIM={ssim_val:.4f}  PSNR={psnr_val:.2f}  MAE={mae_val:.4f}",
                )

    elapsed = time.time() - t_start
    print(f"\n⏱️  推理完成: {elapsed:.1f}s, {len(results)} 个有效样本")

    if len(results) == 0:
        print("❌ 无有效结果")
        return

    # ================================================================
    # 7. 保存结果
    # ================================================================
    df = pd.DataFrame(results)
    csv_path = output_dir / "metrics.csv"
    df.to_csv(csv_path, index=False)

    summary = {
        "checkpoint": str(ckpt_path),
        "mode": "legacy" if args.legacy else "plasma",
        "epoch": ckpt_data.get("epoch", "?"),
        "n_steps": n_steps,
        "n_samples": len(results),
        "elapsed_sec": round(elapsed, 1),
        "ssim_mean": round(float(df["ssim"].mean()), 4),
        "ssim_std": round(float(df["ssim"].std()), 4),
        "psnr_mean": round(float(df["psnr"].mean()), 2),
        "psnr_std": round(float(df["psnr"].std()), 2),
        "mae_mean": round(float(df["mae"].mean()), 4),
        "mae_std": round(float(df["mae"].std()), 4),
        "mse_mean": round(float(df["mse"].mean()), 4),
        "mse_std": round(float(df["mse"].std()), 4),
    }
    json_path = output_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"📊 汇总指标 ({len(results)} 样本):")
    print(f"   SSIM  = {summary['ssim_mean']:.4f} ± {summary['ssim_std']:.4f}")
    print(f"   PSNR  = {summary['psnr_mean']:.2f} ± {summary['psnr_std']:.2f}")
    print(f"   MAE   = {summary['mae_mean']:.4f} ± {summary['mae_std']:.4f}")
    print(f"   MSE   = {summary['mse_mean']:.4f} ± {summary['mse_std']:.4f}")
    print(f"{'='*50}")

    print(f"\n✅ 结果已保存:")
    print(f"   CSV:     {csv_path}")
    print(f"   JSON:    {json_path}")
    if args.save_nifti:
        print(f"   NIfTI:   {nifti_dir}/")
    if args.save_figures:
        print(f"   Figures: {figure_dir}/")


if __name__ == "__main__":
    main()
