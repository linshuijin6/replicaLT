"""
baseline/evaluate.py
=====================
评估脚本

特性:
1. 加载训练好的模型
2. 在测试集上评估
3. 分层指标（整体、按 pet_mfr、按 quality_class、按 diagnosis）
4. 保存评估结果到 CSV
5. 可选保存预测结果
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm
import numpy as np
import pandas as pd
import nibabel as nib

# 添加 baseline 到 path
sys.path.insert(0, str(Path(__file__).parent))

from .config import Config, get_default_config
from .dataset import create_dataloaders, center_crop_3d
from .model import create_model
from .losses import MetricsCalculator, ssim_3d


# ============================================================================
# 评估器
# ============================================================================

class Evaluator:
    """评估器类"""
    
    def __init__(
        self, 
        config: Config, 
        checkpoint_path: str,
        save_predictions: bool = False,
        use_amp: bool = True,
        roi_mask_path: Optional[str] = None,
    ):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.save_predictions = save_predictions
        self.use_amp = use_amp
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # 数据加载器
        print("创建数据加载器...")
        _, _, self.test_loader, _, _, self.test_df = create_dataloaders(config)
        print(f"测试集: {len(self.test_df)} 样本")
        
        # 模型
        print("创建模型...")
        self.model = create_model(config).to(self.device)
        
        # 加载权重
        self._load_checkpoint()
        
        # 评估指标
        self.metrics_calc = MetricsCalculator()

        # ROI mask（可选，接口预留）
        self.roi_mask = None
        if roi_mask_path:
            self.roi_mask = self._load_roi_mask(roi_mask_path)
        
        # 结果目录
        self.results_dir = os.path.dirname(checkpoint_path)
        if not os.path.exists(self.results_dir):
            self.results_dir = config.output_dir
    
    def _load_checkpoint(self):
        """加载模型权重"""
        print(f"加载 checkpoint: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"  Best MAE: {checkpoint.get('best_val_mae', 'N/A')}")
        else:
            # 直接是模型权重
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
    
    @torch.no_grad()
    def evaluate(self) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
        """
        评估测试集
        
        Returns:
            results_df: 每个样本的详细结果
            stratified_metrics: 分层统计指标
        """
        print("\n开始评估...")
        
        results = []
        
        for batch in tqdm(self.test_loader, desc="Evaluating"):
            mri = batch["mri"].to(self.device)
            tau = batch["tau"].to(self.device)
            meta = batch["meta"]
            condition = None
            if self.config.condition.mode != "none" and "clinical" in batch:
                condition = {
                    "clinical": batch["clinical"].to(self.device),
                    "plasma": batch["plasma"].to(self.device),
                    "clinical_mask": batch["clinical_mask"].to(self.device),
                    "plasma_mask": batch["plasma_mask"].to(self.device),
                    "sex": batch["sex"].to(self.device),
                    "source": batch["source"].to(self.device),
                }
            
            # 预测
            if self.use_amp:
                with autocast():
                    pred = self.model(mri, condition)
            else:
                pred = self.model(mri, condition)
            
            # 对每个样本计算指标
            for i in range(pred.size(0)):
                metrics = self.metrics_calc.compute(pred[i:i+1], tau[i:i+1])
                topk_metrics = self._compute_topk_metrics(pred[i:i+1], tau[i:i+1])
                roi_metrics = self._compute_roi_metrics(pred[i:i+1], tau[i:i+1])
                
                result = {
                    "ptid": meta["ptid"][i],
                    "mri_id": meta["mri_id"][i],
                    "tau_id": meta["tau_id"][i],
                    "pet_mfr": meta["pet_mfr"][i],
                    "quality_class": meta["quality_class"][i],
                    "diagnosis": meta["diagnosis"][i],
                    "weight": batch["weight"][i].item(),
                    "mae": metrics["mae"],
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "top5_mae": topk_metrics.get("top5_mae"),
                    "top5_ssim": topk_metrics.get("top5_ssim"),
                    "top10_mae": topk_metrics.get("top10_mae"),
                    "top10_ssim": topk_metrics.get("top10_ssim"),
                    "roi_mae": roi_metrics.get("roi_mae"),
                    "roi_ssim": roi_metrics.get("roi_ssim"),
                }
                if "pt217_f" in meta:
                    val = meta["pt217_f"][i]
                    result["pt217_f"] = val.item() if isinstance(val, torch.Tensor) else float(val)
                results.append(result)
                
                # 保存预测结果
                if self.save_predictions:
                    self._save_prediction(
                        pred[i, 0].cpu().numpy(),
                        tau[i, 0].cpu().numpy(),
                        mri[i, 0].cpu().numpy(),
                        result,
                    )
        
        # 创建 DataFrame
        results_df = pd.DataFrame(results)
        
        # 计算分层指标
        stratified_metrics = self._compute_stratified_metrics(results_df)
        
        return results_df, stratified_metrics
    
    def _save_prediction(
        self,
        pred: np.ndarray,
        tau: np.ndarray,
        mri: np.ndarray,
        meta: Dict,
    ):
        """保存预测结果为 NIfTI"""
        predictions_dir = os.path.join(self.results_dir, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
        
        # 文件名格式: ptid_mfr_quality_diagnosis_pred.nii.gz
        base_name = f"{meta['ptid']}_{meta['pet_mfr']}_{meta['quality_class']}_{meta['diagnosis']}"
        safe_base_name = self._sanitize_filename(base_name)
        
        # 创建 NIfTI（简单的 affine）
        affine = np.eye(4)
        
        # nibabel 不支持 float16，统一转为 float32
        pred = pred.astype(np.float32, copy=False)
        tau = tau.astype(np.float32, copy=False)
        mri = mri.astype(np.float32, copy=False)

        # 保存预测
        pred_nii = nib.Nifti1Image(pred, affine)
        nib.save(pred_nii, os.path.join(predictions_dir, f"{safe_base_name}_pred.nii.gz"))
        
        # 保存 ground truth
        tau_nii = nib.Nifti1Image(tau, affine)
        nib.save(tau_nii, os.path.join(predictions_dir, f"{safe_base_name}_gt.nii.gz"))
        
        # 保存差异图
        diff = np.abs(pred - tau).astype(np.float32, copy=False)
        diff_nii = nib.Nifti1Image(diff, affine)
        nib.save(diff_nii, os.path.join(predictions_dir, f"{safe_base_name}_diff.nii.gz"))

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """将路径不安全字符替换为下划线，避免创建意外子目录。"""
        return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
    
    def _compute_stratified_metrics(
        self, 
        df: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """计算分层指标"""
        metrics_cols = ["mae", "psnr", "ssim", "top5_mae", "top5_ssim", "top10_mae", "top10_ssim", "roi_mae", "roi_ssim"]
        
        def compute_stats(subset: pd.DataFrame) -> Dict[str, float]:
            stats = {"n": len(subset)}
            for col in metrics_cols:
                stats[f"{col}_mean"] = subset[col].mean()
                stats[f"{col}_std"] = subset[col].std()
                stats[f"{col}_median"] = subset[col].median()
                stats[f"{col}_min"] = subset[col].min()
                stats[f"{col}_max"] = subset[col].max()
            return stats
        
        stratified = {}
        
        # 整体
        stratified["overall"] = compute_stats(df)
        
        # 按 pet_mfr 分层
        for mfr in df["pet_mfr"].unique():
            if pd.notna(mfr) and mfr != "":
                subset = df[df["pet_mfr"] == mfr]
                if len(subset) > 0:
                    stratified[f"pet_mfr_{mfr}"] = compute_stats(subset)
        
        # 按 quality_class 分层
        for qc in df["quality_class"].unique():
            if pd.notna(qc) and qc != "":
                subset = df[df["quality_class"] == qc]
                if len(subset) > 0:
                    stratified[f"quality_{qc}"] = compute_stats(subset)
        
        # 按 diagnosis 分层
        for dx in df["diagnosis"].unique():
            if pd.notna(dx) and dx != "":
                subset = df[df["diagnosis"] == dx]
                if len(subset) > 0:
                    stratified[f"dx_{dx}"] = compute_stats(subset)

        # 按 plasma 高低分组（pT217）
        if "pt217_f" in df.columns:
            pt217_vals = pd.to_numeric(df["pt217_f"], errors="coerce")
            valid = pt217_vals.dropna()
            if len(valid) > 0:
                threshold = valid.median()
                high = df[pt217_vals >= threshold]
                low = df[pt217_vals < threshold]
                if len(high) > 0:
                    stratified["plasma_pt217_high"] = compute_stats(high)
                if len(low) > 0:
                    stratified["plasma_pt217_low"] = compute_stats(low)
        
        return stratified

    def _compute_topk_metrics(self, pred: torch.Tensor, tau: torch.Tensor) -> Dict[str, float]:
        pred = pred.float()
        tau = tau.float()
        results = {}
        for k in (5, 10):
            tau_flat = tau.view(-1)
            if tau_flat.numel() == 0:
                results[f"top{k}_mae"] = float("nan")
                results[f"top{k}_ssim"] = float("nan")
                continue
            threshold = torch.quantile(tau_flat, 1.0 - k / 100.0)
            mask = (tau >= threshold).float()
            if mask.sum() == 0:
                results[f"top{k}_mae"] = float("nan")
                results[f"top{k}_ssim"] = float("nan")
                continue
            masked_mae = torch.abs(pred - tau)[mask.bool()].mean().item()
            masked_pred = pred * mask
            masked_tau = tau * mask
            masked_ssim = ssim_3d(masked_pred, masked_tau, reduction="mean").item()
            results[f"top{k}_mae"] = masked_mae
            results[f"top{k}_ssim"] = masked_ssim
        return results

    def _load_roi_mask(self, path: str) -> np.ndarray:
        img = nib.load(path)
        img = nib.as_closest_canonical(img)
        data = img.get_fdata(dtype=np.float32)
        data = np.transpose(data, (2, 1, 0))
        data = center_crop_3d(data, self.config.data.target_shape)
        return data

    def _compute_roi_metrics(self, pred: torch.Tensor, tau: torch.Tensor) -> Dict[str, float]:
        if self.roi_mask is None:
            return {}
        mask = torch.from_numpy(self.roi_mask).to(pred.device)
        if mask.max() <= 0:
            return {}
        mask = (mask > 0).float().unsqueeze(0).unsqueeze(0)
        pred = pred.float()
        tau = tau.float()
        masked_pred = pred * mask
        masked_tau = tau * mask
        masked_mae = torch.abs(masked_pred - masked_tau)[mask.bool()].mean().item()
        masked_ssim = ssim_3d(masked_pred, masked_tau, reduction="mean").item()
        return {"roi_mae": masked_mae, "roi_ssim": masked_ssim}
    
    def save_results(
        self, 
        results_df: pd.DataFrame, 
        stratified_metrics: Dict[str, Dict[str, float]],
    ):
        """保存结果"""
        # 保存详细结果
        results_path = os.path.join(self.results_dir, "metrics_test.csv")
        results_df.to_csv(results_path, index=False)
        print(f"详细结果已保存: {results_path}")
        
        # 保存分层指标
        stratified_path = os.path.join(self.results_dir, "metrics_stratified.json")
        with open(stratified_path, "w") as f:
            json.dump(stratified_metrics, f, indent=2)
        print(f"分层指标已保存: {stratified_path}")
        
        # 生成摘要报告
        self._generate_report(results_df, stratified_metrics)
    
    def _generate_report(
        self, 
        df: pd.DataFrame, 
        stratified: Dict[str, Dict[str, float]],
    ):
        """生成评估报告"""
        report_path = os.path.join(self.results_dir, "evaluation_report.txt")
        
        with open(report_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("MRI → TAU-PET Baseline 评估报告\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"测试样本数: {len(df)}\n\n")
            
            # 整体指标
            overall = stratified["overall"]
            f.write("【整体指标】\n")
            f.write(f"  MAE:  {overall['mae_mean']:.4f} ± {overall['mae_std']:.4f}\n")
            f.write(f"  PSNR: {overall['psnr_mean']:.2f} ± {overall['psnr_std']:.2f}\n")
            f.write(f"  SSIM: {overall['ssim_mean']:.4f} ± {overall['ssim_std']:.4f}\n")
            f.write("\n")
            
            # 按 PET 厂商
            f.write("【按 PET 厂商分层】\n")
            for key, val in stratified.items():
                if key.startswith("pet_mfr_"):
                    mfr = key.replace("pet_mfr_", "")
                    f.write(f"  {mfr} (n={val['n']}):\n")
                    f.write(f"    MAE:  {val['mae_mean']:.4f} ± {val['mae_std']:.4f}\n")
                    f.write(f"    PSNR: {val['psnr_mean']:.2f} ± {val['psnr_std']:.2f}\n")
                    f.write(f"    SSIM: {val['ssim_mean']:.4f} ± {val['ssim_std']:.4f}\n")
            f.write("\n")
            
            # 按质量分类
            f.write("【按质量分类分层】\n")
            for key, val in stratified.items():
                if key.startswith("quality_"):
                    qc = key.replace("quality_", "")
                    f.write(f"  {qc} (n={val['n']}):\n")
                    f.write(f"    MAE:  {val['mae_mean']:.4f} ± {val['mae_std']:.4f}\n")
                    f.write(f"    PSNR: {val['psnr_mean']:.2f} ± {val['psnr_std']:.2f}\n")
                    f.write(f"    SSIM: {val['ssim_mean']:.4f} ± {val['ssim_std']:.4f}\n")
            f.write("\n")
            
            # 按诊断
            f.write("【按诊断分层】\n")
            for key, val in stratified.items():
                if key.startswith("dx_"):
                    dx = key.replace("dx_", "")
                    f.write(f"  {dx} (n={val['n']}):\n")
                    f.write(f"    MAE:  {val['mae_mean']:.4f} ± {val['mae_std']:.4f}\n")
                    f.write(f"    PSNR: {val['psnr_mean']:.2f} ± {val['psnr_std']:.2f}\n")
                    f.write(f"    SSIM: {val['ssim_mean']:.4f} ± {val['ssim_std']:.4f}\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        print(f"评估报告已保存: {report_path}")
        
        # 打印到控制台
        with open(report_path, "r") as f:
            print(f.read())


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MRI → TAU-PET Baseline Evaluation")
    parser.add_argument("checkpoint", type=str, help="模型 checkpoint 路径")
    parser.add_argument("--save_predictions", action="store_true", help="保存预测结果")
    parser.add_argument("--no_amp", action="store_true", help="禁用混合精度")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录（默认与 checkpoint 同目录）")
    parser.add_argument("--roi_mask", type=str, default=None, help="ROI mask NIfTI 路径（可选）")
    parser.add_argument("--condition_mode", type=str, default=None, help="条件模式: none/clinical/plasma/both")
    args = parser.parse_args()
    
    # 创建配置
    config = get_default_config()
    
    # 覆盖输出目录
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.condition_mode:
        config.condition.mode = args.condition_mode
    
    # 创建评估器
    evaluator = Evaluator(
        config=config,
        checkpoint_path=args.checkpoint,
        save_predictions=args.save_predictions,
        use_amp=not args.no_amp,
        roi_mask_path=args.roi_mask,
    )
    
    # 运行评估
    results_df, stratified_metrics = evaluator.evaluate()
    
    # 保存结果
    evaluator.save_results(results_df, stratified_metrics)


if __name__ == "__main__":
    main()
