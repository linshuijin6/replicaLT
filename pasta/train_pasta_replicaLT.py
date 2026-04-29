"""
train_pasta_replicaLT.py
========================
PASTA 训练入口（replicaLT 对比实验专用）。

与原始 train_mri2pet.py 相同逻辑，仅修改 config 路径指向
replicaLT_comparison/pasta_replicaLT.yaml。

用法：
  cd /mnt/nfsdata/nfsdata/lsj.14/PASTA
  conda run -n xiaochou python replicaLT_comparison/train_pasta_replicaLT.py
"""

import argparse
import os
import sys
from report_error import email_on_error
# 将 PASTA 根目录加入 path
PASTA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PASTA_ROOT)

from src.diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType
from src.diffusion.respace import *
from src.trainer.trainer import Trainer
from src.model.unet import UNetModel
from src.utils.utils import *

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

# ★ 使用 replicaLT 对比配置
config_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'pasta_replicaLT.yaml'
)

@email_on_error()
def main():
    args = load_config_from_yaml(config_path)

    if 'RESULTS_DIR' in os.environ:
        old_dir = os.environ['RESULTS_DIR']
        # Replace shell PID in folder name with Python PID (matches nvidia-smi)
        base = old_dir.rsplit('_', 1)[0]
        new_dir = f"{base}_{os.getpid()}"
        if os.path.exists(old_dir) and old_dir != new_dir:
            os.rename(old_dir, new_dir)
        args.results_folder = new_dir

    args_dict = args.__dict__

    if not os.path.exists(args_dict['results_folder']):
        os.makedirs(args_dict['results_folder'])

    list_of_dict = [f'{key} : {args_dict[key]}' for key in args_dict]

    if not args_dict['eval_mode']:
        with open(os.path.join(args_dict['results_folder'], '_hyperparameters.yaml'), 'w') as data:
            [data.write(f'{st}\n') for st in list_of_dict]

    model = UNetModel(
        image_size=args.image_size,
        in_channels=args.model_in_channels,
        model_channels=args.unet_dim,
        out_channels=args.out_channels_model,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=args.attention_resolutions,
        num_heads=args.num_heads,
        channel_mult=args.unet_dim_mults,
        resblock_updown=args.resblock_updown,
        dims=args.dims,
        dropout=args.dropout,
        use_fp16=False,
        use_scale_shift_norm=True,
        use_condition=True,
        use_time_condition=args.use_time_condition,
        cond_emb_channels=args.cond_emb_channels,
        tab_cond_dim=args.tab_cond_dim,
        use_tabular_cond=args.use_tabular_cond_model,
        with_attention=args.with_attention,
        cond_apply_method=args.cond_apply_method,
    )

    encoder = UNetModel(
        image_size=args.image_size,
        in_channels=args.encoder_in_channels,
        model_channels=args.unet_dim,
        out_channels=args.out_channels_encoder,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=args.attention_resolutions,
        num_heads=args.num_heads,
        channel_mult=args.unet_dim_mults,
        resblock_updown=args.resblock_updown,
        dims=args.dims,
        dropout=args.dropout,
        use_fp16=False,
        use_scale_shift_norm=True,
        use_condition=True,
        use_time_condition=args.use_time_condition,
        tab_cond_dim=args.tab_cond_dim,
        use_tabular_cond=args.use_tabular_cond_encoder,
        with_attention=args.with_attention,
        cond_apply_method=args.cond_apply_method,
    )

    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(args.timesteps, args.timestep_respacing),
        model=model,
        encoder=encoder,
        beta_schedule=args.beta_schedule,
        timesteps=args.timesteps,
        model_mean_type=OBJECTIVE[args.objective],
        model_var_type=MODEL_VAR_TYPE[args.model_var_type],
        loss_type=LOSS_TYPE[args.loss_type],
        gen_type=args.gen_type,
        use_fp16=False,
        condition=args.condition,
        reconstructed_loss=args.reconstructed_loss,
        recon_weight=args.recon_weight,
        rescale_intensity=args.rescale_intensity,
    )

    trainer = Trainer(
        diffusion,
        folder=args.data_dir,
        input_slice_channel=args.input_slice_channel,
        train_batch_size=args.train_batch_size,
        train_lr=args.train_lr,
        train_num_steps=args.train_num_steps,
        save_and_sample_every=args.save_and_sample_every,
        num_samples=args.num_samples,
        gradient_accumulate_every=args.gradient_accumulate_every,
        ema_decay=args.ema_decay,
        amp=args.amp,
        fp16=args.fp16,
        calculate_fid=args.calculate_fid,
        dataset=args.dataset,
        image_direction=args.image_direction,
        num_slices=args.num_slices,
        tabular_cond=args.tabular_cond,
        results_folder=args.results_folder,
        resume=args.resume,
        pretrain=args.pretrain,
        test_batch_size=args.test_batch_size,
        eval_mode=args.eval_mode,
        eval_dataset=args.eval_dataset,
        eval_resolution=args.eval_resolution,
        model_cycling=args.model_cycling,
        ROI_mask=args.ROI_mask,
        dx_labels=args.dx_labels,
    )

    if trainer.eval_mode:
        if args.synthesis:
            synth_folder = os.path.join(trainer.results_folder, 'syn_pet')
            eval_model = os.path.join(trainer.results_folder, 'model.pt')
            trainer.evaluate(eval_model, synth_folder, synthesis=True,
                             synthesis_folder=synth_folder, get_ROI_loss=False)
        else:
            eval_folder = os.path.join(trainer.results_folder, 'eval')
            eval_model = os.path.join(trainer.results_folder, 'model.pt')
            trainer.evaluate(eval_model, eval_folder)
    else:
        trainer.train()


if __name__ == "__main__":
    set_seed_everywhere(666)
    main()
