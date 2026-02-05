"""
baseline/visualize.py
======================
可视化模块

特性:
1. 3 切片轴向视图（中间 + 前后各 1/4 处）
2. MRI / GT / Pred 对比
3. 差异热力图
4. 文件名: ptid_mfr_quality_diagnosis.png
5. 可选保存完整 3D NIfTI
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import random

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import nibabel as nib

# 添加 baseline 到 path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, get_default_config
from dataset import create_dataloaders
from model import create_model
from losses import MetricsCalculator


# ============================================================================
# 可视化工具函数
# ============================================================================

def create_custom_colormap() -> LinearSegmentedColormap:
    """创建自定义颜色映射（用于差异图）"""
    colors = [
        (0.0, 'white'),
        (0.2, 'lightblue'),
        (0.4, 'green'),
        (0.6, 'yellow'),
        (0.8, 'orange'),
        (1.0, 'red'),
    ]
    cmap = LinearSegmentedColormap.from_list(
        'custom_diff',
        [(c[0], c[1]) for c in colors]
    )
    return cmap


def get_slice_indices(volume_depth: int, n_slices: int = 3) -> List[int]:
    """
    获取切片索引
    
    Args:
        volume_depth: 体积的深度
        n_slices: 切片数量（默认 3：中间 + 前后各 1/4 处）
    
    Returns:
        切片索引列表
    """
    if n_slices == 3:
        return [
            volume_depth // 4,      # 前 1/4 处
            volume_depth // 2,      # 中间
            3 * volume_depth // 4,  # 后 3/4 处
        ]
    else:
        step = volume_depth // (n_slices + 1)
        return [step * (i + 1) for i in range(n_slices)]


def plot_comparison(
    mri: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    meta: Dict,
    metrics: Dict[str, float],
    save_path: str,
    n_slices: int = 3,
):
    """
    绘制 MRI / GT / Pred 对比图
    
    Args:
        mri: MRI 体积 [D, H, W]
        gt: Ground truth TAU 体积 [D, H, W]
        pred: 预测的 TAU 体积 [D, H, W]
        meta: 元数据（ptid, mfr, quality, diagnosis）
        metrics: 评估指标（mae, psnr, ssim）
        save_path: 保存路径
        n_slices: 切片数量
    """
    # 获取切片索引（轴向切片，沿第一个维度 D）
    slice_indices = get_slice_indices(mri.shape[0], n_slices)
    
    # 创建图形
    fig = plt.figure(figsize=(16, 4 * n_slices + 1))
    gs = gridspec.GridSpec(n_slices + 1, 4, height_ratios=[1] * n_slices + [0.1])
    
    # 自定义差异颜色映射
    diff_cmap = create_custom_colormap()
    
    # 计算差异
    diff = np.abs(pred - gt)
    
    # 确定显示范围
    vmin, vmax = 0, 1
    diff_max = np.max(diff)
    
    for row, slice_idx in enumerate(slice_indices):
        # 提取切片
        mri_slice = mri[slice_idx, :, :]
        gt_slice = gt[slice_idx, :, :]
        pred_slice = pred[slice_idx, :, :]
        diff_slice = diff[slice_idx, :, :]
        
        # MRI
        ax1 = fig.add_subplot(gs[row, 0])
        im1 = ax1.imshow(mri_slice.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        ax1.set_title(f'MRI (slice {slice_idx})', fontsize=10)
        ax1.axis('off')
        
        # Ground Truth
        ax2 = fig.add_subplot(gs[row, 1])
        im2 = ax2.imshow(gt_slice.T, cmap='hot', vmin=vmin, vmax=vmax, origin='lower')
        ax2.set_title(f'Ground Truth TAU', fontsize=10)
        ax2.axis('off')
        
        # Prediction
        ax3 = fig.add_subplot(gs[row, 2])
        im3 = ax3.imshow(pred_slice.T, cmap='hot', vmin=vmin, vmax=vmax, origin='lower')
        ax3.set_title(f'Predicted TAU', fontsize=10)
        ax3.axis('off')
        
        # Difference
        ax4 = fig.add_subplot(gs[row, 3])
        im4 = ax4.imshow(diff_slice.T, cmap=diff_cmap, vmin=0, vmax=max(0.1, diff_max), origin='lower')
        ax4.set_title(f'|Pred - GT|', fontsize=10)
        ax4.axis('off')
    
    # 添加颜色条
    cax1 = fig.add_subplot(gs[n_slices, 0])
    cax2 = fig.add_subplot(gs[n_slices, 1:3])
    cax3 = fig.add_subplot(gs[n_slices, 3])
    
    plt.colorbar(im1, cax=cax1, orientation='horizontal', label='MRI Intensity')
    plt.colorbar(im2, cax=cax2, orientation='horizontal', label='TAU Intensity')
    plt.colorbar(im4, cax=cax3, orientation='horizontal', label='Absolute Difference')
    
    # 添加标题
    title = (
        f"PTID: {meta['ptid']} | "
        f"MFR: {meta['pet_mfr']} | "
        f"Quality: {meta['quality_class']} | "
        f"Dx: {meta['diagnosis']}\n"
        f"MAE: {metrics['mae']:.4f} | "
        f"PSNR: {metrics['psnr']:.2f} dB | "
        f"SSIM: {metrics['ssim']:.4f}"
    )
    fig.suptitle(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_nifti(
    volume: np.ndarray,
    save_path: str,
    affine: Optional[np.ndarray] = None,
):
    """保存 NIfTI 文件"""
    if affine is None:
        affine = np.eye(4)
    
    nii = nib.Nifti1Image(volume, affine)
    nib.save(nii, save_path)


# ============================================================================
# 可视化器
# ============================================================================

class Visualizer:
    """可视化器类"""
    
    def __init__(
        self,
        config: Config,
        checkpoint_path: str,
        output_dir: Optional[str] = None,
        use_amp: bool = True,
    ):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.use_amp = use_amp
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # 输出目录
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = os.path.join(
                os.path.dirname(checkpoint_path), 
                "visualizations"
            )
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "nifti"), exist_ok=True)
        
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
    
    def _load_checkpoint(self):
        """加载模型权重"""
        print(f"加载 checkpoint: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
    
    @torch.no_grad()
    def visualize_samples(
        self,
        n_samples: Optional[int] = None,
        save_nifti_files: bool = False,
        random_select: bool = False,
    ):
        """
        可视化测试集样本
        
        Args:
            n_samples: 可视化的样本数量（None 表示全部）
            save_nifti_files: 是否保存 NIfTI 文件
            random_select: 是否随机选择样本
        """
        print(f"\n开始可视化（保存到: {self.output_dir}）...")
        
        # 确定要处理的样本
        all_indices = list(range(len(self.test_loader.dataset)))
        
        if n_samples is not None and n_samples < len(all_indices):
            if random_select:
                sample_indices = random.sample(all_indices, n_samples)
            else:
                sample_indices = all_indices[:n_samples]
        else:
            sample_indices = all_indices
        
        sample_set = set(sample_indices)
        
        visualized = 0
        for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Visualizing")):
            mri = batch["mri"].to(self.device)
            tau = batch["tau"].to(self.device)
            meta = batch["meta"]
            
            # 预测
            if self.use_amp:
                with autocast():
                    pred = self.model(mri)
            else:
                pred = self.model(mri)
            
            # 对每个样本
            for i in range(pred.size(0)):
                # 全局索引
                global_idx = batch_idx * self.test_loader.batch_size + i
                
                if global_idx not in sample_set:
                    continue
                
                # 获取数据
                mri_np = mri[i, 0].cpu().numpy()
                tau_np = tau[i, 0].cpu().numpy()
                pred_np = pred[i, 0].cpu().numpy()
                
                # 元数据
                sample_meta = {
                    "ptid": meta["ptid"][i],
                    "mri_id": meta["mri_id"][i],
                    "tau_id": meta["tau_id"][i],
                    "pet_mfr": meta["pet_mfr"][i],
                    "quality_class": meta["quality_class"][i],
                    "diagnosis": meta["diagnosis"][i],
                }
                
                # 计算指标
                metrics = self.metrics_calc.compute(pred[i:i+1], tau[i:i+1])
                
                # 文件名
                base_name = (
                    f"{sample_meta['ptid']}_"
                    f"{sample_meta['pet_mfr']}_"
                    f"{sample_meta['quality_class']}_"
                    f"{sample_meta['diagnosis']}"
                ).replace(" ", "_").replace("/", "_")
                
                # 保存可视化
                vis_path = os.path.join(self.output_dir, f"{base_name}.png")
                plot_comparison(
                    mri=mri_np,
                    gt=tau_np,
                    pred=pred_np,
                    meta=sample_meta,
                    metrics=metrics,
                    save_path=vis_path,
                )
                
                # 保存 NIfTI
                if save_nifti_files:
                    nifti_dir = os.path.join(self.output_dir, "nifti")
                    save_nifti(pred_np, os.path.join(nifti_dir, f"{base_name}_pred.nii.gz"))
                    save_nifti(tau_np, os.path.join(nifti_dir, f"{base_name}_gt.nii.gz"))
                    save_nifti(np.abs(pred_np - tau_np), os.path.join(nifti_dir, f"{base_name}_diff.nii.gz"))
                
                visualized += 1
                
                if n_samples is not None and visualized >= n_samples:
                    break
            
            if n_samples is not None and visualized >= n_samples:
                break
        
        print(f"可视化完成! 共保存 {visualized} 个样本")
    
    @torch.no_grad()
    def visualize_stratified(
        self,
        n_per_stratum: int = 2,
        save_nifti_files: bool = False,
    ):
        """
        分层可视化（每个分层采样若干样本）
        
        Args:
            n_per_stratum: 每个分层的样本数
            save_nifti_files: 是否保存 NIfTI 文件
        """
        print(f"\n分层可视化（每层 {n_per_stratum} 个样本）...")
        
        # 按分层收集样本索引
        strata_indices = {}
        
        for idx in range(len(self.test_df)):
            row = self.test_df.iloc[idx]
            
            # 按 pet_mfr
            mfr = row.get("pet_mfr", "Unknown")
            key = f"mfr_{mfr}"
            if key not in strata_indices:
                strata_indices[key] = []
            strata_indices[key].append(idx)
            
            # 按 quality_class
            qc = row.get("quality_class", "Unknown")
            key = f"quality_{qc}"
            if key not in strata_indices:
                strata_indices[key] = []
            strata_indices[key].append(idx)
            
            # 按 diagnosis
            dx = row.get("diagnosis", "Unknown")
            key = f"dx_{dx}"
            if key not in strata_indices:
                strata_indices[key] = []
            strata_indices[key].append(idx)
        
        # 每个分层随机选择样本
        selected_indices = set()
        for stratum, indices in strata_indices.items():
            n_select = min(n_per_stratum, len(indices))
            selected = random.sample(indices, n_select)
            selected_indices.update(selected)
        
        print(f"选中 {len(selected_indices)} 个样本进行可视化")
        
        # 可视化选中的样本
        visualized = 0
        for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Stratified Viz")):
            mri = batch["mri"].to(self.device)
            tau = batch["tau"].to(self.device)
            meta = batch["meta"]
            
            # 预测
            if self.use_amp:
                with autocast():
                    pred = self.model(mri)
            else:
                pred = self.model(mri)
            
            for i in range(pred.size(0)):
                global_idx = batch_idx * self.test_loader.batch_size + i
                
                if global_idx not in selected_indices:
                    continue
                
                # 获取数据
                mri_np = mri[i, 0].cpu().numpy()
                tau_np = tau[i, 0].cpu().numpy()
                pred_np = pred[i, 0].cpu().numpy()
                
                sample_meta = {
                    "ptid": meta["ptid"][i],
                    "mri_id": meta["mri_id"][i],
                    "tau_id": meta["tau_id"][i],
                    "pet_mfr": meta["pet_mfr"][i],
                    "quality_class": meta["quality_class"][i],
                    "diagnosis": meta["diagnosis"][i],
                }
                
                metrics = self.metrics_calc.compute(pred[i:i+1], tau[i:i+1])
                
                base_name = (
                    f"{sample_meta['ptid']}_"
                    f"{sample_meta['pet_mfr']}_"
                    f"{sample_meta['quality_class']}_"
                    f"{sample_meta['diagnosis']}"
                ).replace(" ", "_").replace("/", "_")
                
                vis_path = os.path.join(self.output_dir, f"{base_name}.png")
                plot_comparison(
                    mri=mri_np,
                    gt=tau_np,
                    pred=pred_np,
                    meta=sample_meta,
                    metrics=metrics,
                    save_path=vis_path,
                )
                
                if save_nifti_files:
                    nifti_dir = os.path.join(self.output_dir, "nifti")
                    save_nifti(pred_np, os.path.join(nifti_dir, f"{base_name}_pred.nii.gz"))
                    save_nifti(tau_np, os.path.join(nifti_dir, f"{base_name}_gt.nii.gz"))
                
                visualized += 1
        
        print(f"分层可视化完成! 共保存 {visualized} 个样本")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MRI → TAU-PET Baseline Visualization")
    parser.add_argument("checkpoint", type=str, help="模型 checkpoint 路径")
    parser.add_argument("--n_samples", type=int, default=None, help="可视化样本数量（默认全部）")
    parser.add_argument("--stratified", action="store_true", help="分层可视化")
    parser.add_argument("--n_per_stratum", type=int, default=2, help="每层样本数（分层模式）")
    parser.add_argument("--save_nifti", action="store_true", help="保存 NIfTI 文件")
    parser.add_argument("--random", action="store_true", help="随机选择样本")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--no_amp", action="store_true", help="禁用混合精度")
    args = parser.parse_args()
    
    # 创建配置
    config = get_default_config()
    
    # 创建可视化器
    visualizer = Visualizer(
        config=config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        use_amp=not args.no_amp,
    )
    
    # 运行可视化
    if args.stratified:
        visualizer.visualize_stratified(
            n_per_stratum=args.n_per_stratum,
            save_nifti_files=args.save_nifti,
        )
    else:
        visualizer.visualize_samples(
            n_samples=args.n_samples,
            save_nifti_files=args.save_nifti,
            random_select=args.random,
        )


if __name__ == "__main__":
    main()
