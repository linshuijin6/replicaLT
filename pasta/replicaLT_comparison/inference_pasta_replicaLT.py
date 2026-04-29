"""
inference_pasta_replicaLT.py
============================
PASTA 推理脚本（replicaLT 对比实验专用）。

输入 MRI（test.h5），通过训练好的 PASTA 扩散模型（EMA 权重）
使用 DDIM-100 采样生成合成 PET，重建 3D 体积并保存为 NIfTI (.nii.gz)。
同时计算 L1 / MSE / PSNR / SSIM 评估指标。

用法：
  cd /home/ssddata/linshuijin/PASTA
  python replicaLT_comparison/inference_pasta_replicaLT.py \
      --test_data data/test.h5 \
      --ckpt replicaLT_comparison/results/2026-04-12_331111/best_val_model.pt \
      --output_dir replicaLT_comparison/results/2026-04-12_331111/inference_output
"""

import argparse
import os
import sys
from glob import glob

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ema_pytorch import EMA

# ---------- 项目路径 ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PASTA_ROOT)

from src.diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType
from src.diffusion.respace import SpacedDiffusion, space_timesteps
from src.model.unet import UNetModel
from src.datasets.dataset import SlicedScanMRI2PETDataset
from src.utils.utils import load_config_from_yaml, set_seed_everywhere
from src.utils.data_utils import reconstruct_scan_from_2_5D_slices
from src.evals.ssim import ssim as compute_ssim

# ---------- 常量映射 ----------
OBJECTIVE = {
    'PREVIOUS_X': ModelMeanType.PREVIOUS_X,
    'START_X': ModelMeanType.START_X,
    'EPSILON': ModelMeanType.EPSILON,
    'VELOCITY': ModelMeanType.VELOCITY,
}
MODEL_VAR_TYPE = {
    'LEARNED': ModelVarType.LEARNED,
    'FIXED_SMALL': ModelVarType.FIXED_SMALL,
    'FIXED_LARGE': ModelVarType.FIXED_LARGE,
    'LEARNED_RANGE': ModelVarType.LEARNED_RANGE,
}
LOSS_TYPE = {'l1': LossType.MAE, 'l2': LossType.MSE}
LABEL_MAP = {0: 'CN', 1: 'AD', 2: 'MCI'}


def parse_args():
    parser = argparse.ArgumentParser(description='PASTA MRI→PET 推理')
    parser.add_argument('--test_data', type=str,
                        default=os.path.join(PASTA_ROOT, 'data', 'valid.h5'),
                        help='测试数据 h5 文件路径')
    parser.add_argument('--ckpt', type=str,
                        default=os.path.join(SCRIPT_DIR, 'results', '2026-04-12_331111', 'best_val_model.pt'),
                        help='模型 checkpoint 路径')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(SCRIPT_DIR, 'results', '2026-04-12_331111', 'inference_output'),
                        help='合成 PET 输出目录')
    parser.add_argument('--config', type=str,
                        default=os.path.join(SCRIPT_DIR, 'pasta_replicaLT.yaml'),
                        help='配置文件路径')
    parser.add_argument('--device', type=str, default='cuda',
                        help='推理设备 (cuda / cpu)')
    parser.add_argument('--gpu_id', type=int, default=1,
                        help='指定使用的 GPU 编号；当 device=cuda 时生效')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='每轮推理的切片批大小')
    parser.add_argument('--amp', type=str, default='fp16', choices=['none', 'fp16'],
                        help='推理自动混合精度模式')
    parser.add_argument('--max_batches', type=int, default=None,
                        help='仅推理前若干个 dataloader batch，用于 smoke test')
    parser.add_argument('--seed', type=int, default=666,
                        help='随机种子')
    return parser.parse_args()


def build_model(args_cfg, device, use_fp16=False):
    """按照训练配置构建 UNet + Encoder + SpacedDiffusion。"""
    model = UNetModel(
        image_size=args_cfg.image_size,
        in_channels=args_cfg.model_in_channels,
        model_channels=args_cfg.unet_dim,
        out_channels=args_cfg.out_channels_model,
        num_res_blocks=args_cfg.num_res_blocks,
        attention_resolutions=args_cfg.attention_resolutions,
        num_heads=args_cfg.num_heads,
        channel_mult=args_cfg.unet_dim_mults,
        resblock_updown=args_cfg.resblock_updown,
        dims=args_cfg.dims,
        dropout=args_cfg.dropout,
        use_fp16=False,
        use_scale_shift_norm=True,
        use_condition=True,
        use_time_condition=args_cfg.use_time_condition,
        cond_emb_channels=args_cfg.cond_emb_channels,
        tab_cond_dim=args_cfg.tab_cond_dim,
        use_tabular_cond=args_cfg.use_tabular_cond_model,
        with_attention=args_cfg.with_attention,
        cond_apply_method=args_cfg.cond_apply_method,
    )

    encoder = UNetModel(
        image_size=args_cfg.image_size,
        in_channels=args_cfg.encoder_in_channels,
        model_channels=args_cfg.unet_dim,
        out_channels=args_cfg.out_channels_encoder,
        num_res_blocks=args_cfg.num_res_blocks,
        attention_resolutions=args_cfg.attention_resolutions,
        num_heads=args_cfg.num_heads,
        channel_mult=args_cfg.unet_dim_mults,
        resblock_updown=args_cfg.resblock_updown,
        dims=args_cfg.dims,
        dropout=args_cfg.dropout,
        use_fp16=False,
        use_scale_shift_norm=True,
        use_condition=True,
        use_time_condition=args_cfg.use_time_condition,
        tab_cond_dim=args_cfg.tab_cond_dim,
        use_tabular_cond=args_cfg.use_tabular_cond_encoder,
        with_attention=args_cfg.with_attention,
        cond_apply_method=args_cfg.cond_apply_method,
    )

    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(args_cfg.timesteps, args_cfg.timestep_respacing),
        model=model,
        encoder=encoder,
        beta_schedule=args_cfg.beta_schedule,
        timesteps=args_cfg.timesteps,
        model_mean_type=OBJECTIVE[args_cfg.objective],
        model_var_type=MODEL_VAR_TYPE[args_cfg.model_var_type],
        loss_type=LOSS_TYPE[args_cfg.loss_type],
        gen_type=args_cfg.gen_type,
        use_fp16=use_fp16,
        condition=args_cfg.condition,
        reconstructed_loss=args_cfg.reconstructed_loss,
        recon_weight=args_cfg.recon_weight,
        rescale_intensity=args_cfg.rescale_intensity,
    )

    diffusion.to(device)
    return diffusion


def load_checkpoint(diffusion, ckpt_path, ema_decay, device):
    """加载 checkpoint 并构建 EMA 模型，返回 (diffusion, ema)。"""
    data = torch.load(ckpt_path, map_location=device)

    msg = diffusion.load_state_dict(data['model'], strict=False)
    print('====== 加载模型权重 ========')
    print("missing keys:", msg.missing_keys)
    print("unexpected keys:", msg.unexpected_keys)

    ema = EMA(diffusion, beta=ema_decay, update_every=10)
    ema.to(device)
    msg_ema = ema.load_state_dict(data['ema'], strict=False)
    print('====== 加载 EMA 权重 ========')
    print("ema missing keys:", msg_ema.missing_keys)
    print("ema unexpected keys:", msg_ema.unexpected_keys)

    diffusion.eval()
    ema.ema_model.eval()
    print(f'成功加载 checkpoint: {ckpt_path}')
    return diffusion, ema


def compute_psnr(img1, img2, v_max=1.0):
    """计算 PSNR（Peak Signal-to-Noise Ratio）。"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(v_max ** 2 / mse)


def resolve_device(args):
    if not torch.cuda.is_available() or args.device == 'cpu':
        return torch.device('cpu')
    if args.device.startswith('cuda:'):
        return torch.device(args.device)
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if visible_devices:
        mapped_devices = [dev.strip() for dev in visible_devices.split(',') if dev.strip()]
        if str(args.gpu_id) in mapped_devices:
            return torch.device(f'cuda:{mapped_devices.index(str(args.gpu_id))}')
        if len(mapped_devices) == 1:
            return torch.device('cuda:0')
    return torch.device(f'cuda:{args.gpu_id}')


def configure_amp(diffusion, ema, args, device):
    use_fp16 = args.amp == 'fp16' and device.type == 'cuda'
    diffusion.use_fp16 = use_fp16
    ema.ema_model.use_fp16 = use_fp16
    return use_fp16


def normalize_scores_file(scores_path):
    if not os.path.exists(scores_path):
        return 0

    unique_scores = {}
    ordered_uids = []
    trailer_lines = []

    with open(scores_path, 'r') as f:
        for line in f:
            if 'L1=' in line and 'SSIM=' in line:
                uid = line.split(' (', 1)[0]
                if uid not in unique_scores:
                    ordered_uids.append(uid)
                unique_scores[uid] = line.rstrip('\n')
            else:
                trailer_lines.append(line.rstrip('\n'))

    with open(scores_path, 'w') as f:
        for uid in ordered_uids:
            f.write(unique_scores[uid] + '\n')
        for line in trailer_lines:
            if line:
                f.write(line + '\n')
            else:
                f.write('\n')

    return len(unique_scores)


def verify_output_dir(output_dir, expected_subjects):
    syn_files = sorted(glob(os.path.join(output_dir, '*_syn_pet.nii.gz')))
    gt_files = sorted(glob(os.path.join(output_dir, '*_GT_pet.nii.gz')))
    scores_path = os.path.join(output_dir, 'eval_scores.txt')
    score_lines = normalize_scores_file(scores_path)

    missing_syn = expected_subjects - len(syn_files)
    missing_gt = expected_subjects - len(gt_files)
    missing_scores = expected_subjects - score_lines

    if missing_syn or missing_gt or missing_scores:
        raise RuntimeError(
            '输出完整性校验失败: '
            f'syn={len(syn_files)}/{expected_subjects}, '
            f'gt={len(gt_files)}/{expected_subjects}, '
            f'scores={score_lines}/{expected_subjects}, '
            f'missing_syn={max(missing_syn, 0)}, '
            f'missing_gt={max(missing_gt, 0)}, '
            f'missing_scores={max(missing_scores, 0)}'
        )

    print(
        f'输出完整性校验通过: syn={len(syn_files)}, '
        f'gt={len(gt_files)}, scores={score_lines}'
    )


@torch.inference_mode()
def run_inference(args):
    device = resolve_device(args)
    os.makedirs(args.output_dir, exist_ok=True)
    scores_path = os.path.join(args.output_dir, 'eval_scores.txt')
    if os.path.exists(scores_path):
        os.remove(scores_path)

    # ---- 1. 加载配置 ----
    cfg = load_config_from_yaml(args.config)

    # ---- 2. 构建模型 & 加载权重 ----
    diffusion = build_model(cfg, device, use_fp16=args.amp == 'fp16' and device.type == 'cuda')
    diffusion, ema = load_checkpoint(diffusion, args.ckpt, cfg.ema_decay, device)
    amp_enabled = configure_amp(diffusion, ema, args, device)
    print(f'使用设备: {device}')
    print(f'AMP 模式: {"fp16" if amp_enabled else "none"}')

    # ---- 3. 构建测试数据集 ----
    ds_test = SlicedScanMRI2PETDataset(
        resolution=cfg.eval_resolution,
        data_path=args.test_data,
        output_dim=cfg.input_slice_channel,
        direction=cfg.image_direction,
        random_flip=None,
        dx_labels=cfg.dx_labels,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
        drop_last=False,
    )

    # ---- 4. 推理 & 评估 ----
    input_slice_channel = cfg.input_slice_channel
    image_direction = cfg.image_direction
    tabular_cond = cfg.tabular_cond
    axis_map = {'coronal': -2, 'sagittal': -3, 'axial': -1}

    affine = np.array([
        [1.5, 0,   0,   0],
        [0,   1.5, 0,   0],
        [0,   0,   1.5, 0],
        [0,   0,   0,   1],
    ])

    all_l1, all_mse, all_psnr, all_ssim = [], [], [], []

    ssim_fn = compute_ssim  # 2D SSIM from src/evals/ssim.py

    print(f'\n=== 开始推理，共 {len(ds_test)} 个受试者 ===\n')

    for batch_idx, test_data in enumerate(tqdm(dl_test, desc='Inference'), start=1):
        test_data_mri = test_data[0]   # list of slice tensors
        test_data_pet = test_data[1]   # list of GT slice tensors
        dx_label = test_data[2]
        mri_uid = test_data[-1]

        if tabular_cond:
            tabular_data = test_data[3].to(device)
        else:
            tabular_data = None

        slices_per_sample_list = []
        GT_images_list = []

        # 逐切片采样
        for slice_num in range(len(test_data_mri)):
            mri_slice = test_data_mri[slice_num].to(device)
            pet_slice = test_data_pet[slice_num].to(device)
            cond_slice = mri_slice

            sample_pet = ema.ema_model.sample(
                shape=pet_slice.shape,
                cond=cond_slice,
                tab_cond=tabular_data,
                progress=False,
            )
            sample_pet = sample_pet.detach().cpu()
            pet_slice_cpu = pet_slice.detach().cpu()

            slices_per_sample_list.append(sample_pet)

            if input_slice_channel > 1:
                GT_images_list.append(
                    pet_slice_cpu[:, input_slice_channel // 2, ...]
                )
            else:
                GT_images_list.append(pet_slice_cpu)

            del mri_slice, pet_slice, cond_slice, sample_pet, pet_slice_cpu

        # 2.5D → 3D 重建
        if input_slice_channel > 1:
            reconstructed_slices = reconstruct_scan_from_2_5D_slices(slices_per_sample_list)
        else:
            reconstructed_slices = [s.squeeze(1) for s in slices_per_sample_list]

        whole_pet_sample = np.stack(
            [s.cpu().numpy() for s in reconstructed_slices],
            axis=axis_map[image_direction],
        )
        whole_GT_pet = np.stack(
            [s.cpu().numpy() for s in GT_images_list],
            axis=axis_map[image_direction],
        )

        assert whole_pet_sample.shape == whole_GT_pet.shape, \
            f'Shape mismatch: {whole_pet_sample.shape} vs {whole_GT_pet.shape}'

        # 逐受试者保存 & 评估
        for b in range(whole_pet_sample.shape[0]):
            img_label = LABEL_MAP[int(dx_label[b].detach().cpu().numpy())]
            uid = str(mri_uid[b])

            # 保存合成 PET
            syn_pet = whole_pet_sample[b].squeeze()
            gt_pet = whole_GT_pet[b].squeeze()

            pet_nii = nib.Nifti1Image(syn_pet, affine=affine)
            out_fname = os.path.join(args.output_dir, f'{uid}_{img_label}_syn_pet.nii.gz')
            pet_nii.to_filename(out_fname)

            # 同时保存 GT PET（便于对比）
            gt_nii = nib.Nifti1Image(gt_pet, affine=affine)
            gt_fname = os.path.join(args.output_dir, f'{uid}_{img_label}_GT_pet.nii.gz')
            gt_nii.to_filename(gt_fname)

            # 评估指标
            l1_val = np.abs(syn_pet - gt_pet).mean()
            mse_val = ((syn_pet - gt_pet) ** 2).mean()
            psnr_val = compute_psnr(syn_pet, gt_pet, v_max=1.0)

            # SSIM: 逐 axial 切片计算后取均值
            syn_t = torch.from_numpy(syn_pet).float().unsqueeze(0).unsqueeze(0)
            gt_t = torch.from_numpy(gt_pet).float().unsqueeze(0).unsqueeze(0)
            n_slices = syn_pet.shape[-1]
            ssim_vals = []
            for si in range(n_slices):
                s_syn = syn_t[..., si]  # (1, 1, H, W)
                s_gt = gt_t[..., si]
                if s_syn.sum() > 0 and s_gt.sum() > 0:
                    ssim_vals.append(ssim_fn(s_syn, s_gt, window_size=11).item())
            ssim_val = np.mean(ssim_vals) if ssim_vals else 0.0

            all_l1.append(l1_val)
            all_mse.append(mse_val)
            all_psnr.append(psnr_val)
            all_ssim.append(ssim_val)

            score_line = (
                f'{uid} ({img_label}): '
                f'L1={l1_val:.6f}  MSE={mse_val:.6f}  '
                f'PSNR={psnr_val:.4f}  SSIM={ssim_val:.6f}'
            )
            print(score_line)
            with open(scores_path, 'a') as f:
                f.write(score_line + '\n')

        if args.max_batches is not None and batch_idx >= args.max_batches:
            print(f'达到 max_batches={args.max_batches}，提前结束推理。')
            break

    # ---- 5. 汇总统计 ----
    summary = (
        f'\n===== 推理完成 =====\n'
        f'受试者数: {len(all_l1)}\n'
        f'平均 L1:   {np.mean(all_l1):.6f} ± {np.std(all_l1):.6f}\n'
        f'平均 MSE:  {np.mean(all_mse):.6f} ± {np.std(all_mse):.6f}\n'
        f'平均 PSNR: {np.mean(all_psnr):.4f} ± {np.std(all_psnr):.4f}\n'
        f'平均 SSIM: {np.mean(all_ssim):.6f} ± {np.std(all_ssim):.6f}\n'
        f'输出目录: {args.output_dir}\n'
    )
    print(summary)
    with open(scores_path, 'a') as f:
        f.write('\n' + summary)

    verify_output_dir(args.output_dir, len(all_l1))


if __name__ == '__main__':
    cli_args = parse_args()
    set_seed_everywhere(cli_args.seed)
    run_inference(cli_args)
