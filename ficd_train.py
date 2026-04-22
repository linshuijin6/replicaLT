from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# Parse --gpu early, before any CUDA imports
def _set_gpu_early():
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu" and i + 1 < len(sys.argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
            break
_set_gpu_early()

from report_error import email_on_error
import numpy as np
import torch
import torch.nn.functional as F
from monai.metrics import PSNRMetric
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ficd.config import load_config
from ficd.data import build_dataset, build_transform, load_split_samples
from ficd.utils import (
    append_csv_row,
    find_checkpoint,
    log_3d_volume_to_tensorboard,
    log_comparison_figure,
    make_run_dir,
    save_comparison_figure,
    save_prediction_nifti,
    setup_logger,
    write_json,
)
from generative.inferers import DiffusionInferer
from generative.metrics import SSIMMetric
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FICD baseline training/evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--resume", default=None, help="Existing run directory to resume into.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path or name under run dir.")
    parser.add_argument("--eval-only", action="store_true", help="Run validation/inference only.")
    parser.add_argument("--seed", type=int, default=None, help="Optional runtime seed override.")
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index to use.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_unit_range(tensor: torch.Tensor) -> torch.Tensor:
    return torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0)


def build_model(config: dict, device: torch.device):
    model_cfg = config["model"]
    kwargs = dict(
        spatial_dims=model_cfg["spatial_dims"],
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        num_channels=model_cfg["num_channels"],
        attention_levels=model_cfg["attention_levels"],
        num_head_channels=model_cfg["num_head_channels"],
        num_res_blocks=model_cfg["num_res_blocks"],
        norm_num_groups=model_cfg["norm_num_groups"],
        use_flash_attention=model_cfg["use_flash_attention"],
        with_conditioning=model_cfg.get("with_conditioning", False),
    )
    if kwargs["with_conditioning"] and "cross_attention_dim" in model_cfg:
        kwargs["cross_attention_dim"] = model_cfg["cross_attention_dim"]
    model = DiffusionModelUNet(**kwargs)
    model.to(device)
    return model


def get_subject_id(batch_subject_ids, index: int) -> str:
    if isinstance(batch_subject_ids, list):
        return str(batch_subject_ids[index])
    return str(batch_subject_ids[index])


def get_affine(batch, index: int) -> np.ndarray:
    affine = batch["pet"]["affine"][index]
    if isinstance(affine, torch.Tensor):
        return affine.detach().cpu().numpy()
    return np.asarray(affine)


def build_volumes_for_logging(mri: torch.Tensor, gt: torch.Tensor, pred: torch.Tensor) -> dict[str, torch.Tensor]:
    gt_unit = to_unit_range(gt)
    pred_unit = to_unit_range(pred)
    diff_unit = torch.abs(pred_unit - gt_unit)
    return {
        "MRI": to_unit_range(mri),
        "PET_GT": gt_unit,
        "PET_Pred": pred_unit,
        "PET_Diff": diff_unit,
    }


@torch.no_grad()
def run_validation(
    *,
    model,
    inferer,
    loader: DataLoader,
    scheduler,
    device: torch.device,
    run_dir: Path,
    writer,
    global_step: int,
    epoch: int,
    num_inference_steps: int,
    save_predictions: bool,
    figure_suffix: str,
    logger,
) -> dict[str, float]:
    model.eval()
    scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)
    psnr_metric = PSNRMetric(max_val=1.0)
    ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=7)

    losses: list[float] = []
    psnrs: list[float] = []
    ssims: list[float] = []
    subject_metric_rows: list[dict[str, float | str]] = []
    visualization_payload = None

    predictions_dir = run_dir / "predictions" / figure_suffix
    if save_predictions:
        predictions_dir.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(loader, total=len(loader), ncols=100, desc=f"Validate {figure_suffix}", leave=False):
        mri = batch["mri"]["data"].float().to(device)
        gt = batch["pet"]["data"].float().to(device)
        input_noise = torch.randn_like(gt)
        with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            pred = inferer.sample(
                input_noise=input_noise,
                diffusion_model=model,
                scheduler=scheduler,
                conditioning=mri,
            )

        pred_unit = to_unit_range(pred)
        gt_unit = to_unit_range(gt)
        loss = F.l1_loss(pred_unit.float(), gt_unit.float())
        psnr_value = float(psnr_metric(pred_unit.cpu(), gt_unit.cpu()).mean().item())
        ssim_value = float(ssim_metric(gt_unit.cpu(), pred_unit.cpu()).mean().item())
        losses.append(float(loss.item()))
        psnrs.append(psnr_value)
        ssims.append(ssim_value)

        subject_ids = batch["subject_id"]
        for index in range(pred.shape[0]):
            subject_id = get_subject_id(subject_ids, index)
            pred_single = pred_unit[index : index + 1]
            gt_single = gt_unit[index : index + 1]
            subject_metric_rows.append(
                {
                    "subject_id": subject_id,
                    "l1_unit": float(F.l1_loss(pred_single, gt_single).item()),
                    "psnr": float(psnr_metric(pred_single.cpu(), gt_single.cpu()).mean().item()),
                    "ssim": float(ssim_metric(gt_single.cpu(), pred_single.cpu()).mean().item()),
                }
            )
            if save_predictions:
                save_prediction_nifti(
                    predictions_dir / f"{subject_id}.nii.gz",
                    pred_single,
                    get_affine(batch, index),
                )

        if visualization_payload is None:
            visualization_payload = {
                "subject_id": get_subject_id(subject_ids, 0),
                "volumes": build_volumes_for_logging(mri[0:1], gt[0:1], pred[0:1]),
            }

    metrics = {
        "val_loss": float(np.mean(losses)) if losses else 0.0,
        "val_psnr": float(np.mean(psnrs)) if psnrs else 0.0,
        "val_ssim": float(np.mean(ssims)) if ssims else 0.0,
    }

    if save_predictions:
        write_json(predictions_dir / "subject_metrics.json", subject_metric_rows)

    if writer is not None:
        writer.add_scalar("val/loss", metrics["val_loss"], epoch)
        writer.add_scalar("val/PSNR", metrics["val_psnr"], epoch)
        writer.add_scalar("val/SSIM", metrics["val_ssim"], epoch)
        if visualization_payload is not None:
            volumes = visualization_payload["volumes"]
            log_comparison_figure(writer, f"val/comparison_{figure_suffix}", volumes, global_step)
            log_3d_volume_to_tensorboard(writer, f"val/{figure_suffix}/MRI", volumes["MRI"], global_step)
            log_3d_volume_to_tensorboard(writer, f"val/{figure_suffix}/PET_GT", volumes["PET_GT"], global_step)
            log_3d_volume_to_tensorboard(writer, f"val/{figure_suffix}/PET_Pred", volumes["PET_Pred"], global_step)
            save_comparison_figure(
                run_dir / "figures" / f"{figure_suffix}_{visualization_payload['subject_id']}.png",
                volumes,
            )

    logger.info(
        "Validation %s | loss=%.6f | PSNR=%.4f | SSIM=%.4f",
        figure_suffix,
        metrics["val_loss"],
        metrics["val_psnr"],
        metrics["val_ssim"],
    )
    return metrics

@email_on_error()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    set_seed(int(config["seed"]))

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    run_dir = make_run_dir(config["logging"]["run_root"], resume=args.resume)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir)
    if config["logging"].get("tensorboard", True):
        writer = SummaryWriter(log_dir=str(run_dir))
    else:
        writer = None

    hparams = json.loads(json.dumps(config))
    hparams["run_dir"] = str(run_dir)
    hparams["resume"] = args.resume
    hparams["eval_only"] = args.eval_only
    hparams["checkpoint"] = args.checkpoint
    write_json(run_dir / "hparams.json", hparams)

    train_samples, val_samples, filter_stats = load_split_samples(config)
    write_json(run_dir / "samples_train.json", train_samples)
    write_json(run_dir / "samples_val.json", val_samples)
    write_json(run_dir / "filter_stats.json", filter_stats)

    transform = build_transform(config)
    train_dataset = build_dataset(train_samples, transform)
    val_dataset = build_dataset(val_samples, transform)
    data_cfg = config["data"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["train"]["batch_size_train"]),
        shuffle=True,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=bool(data_cfg["pin_memory"]),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["train"]["batch_size_val"]),
        shuffle=False,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=bool(data_cfg["pin_memory"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info("Run directory: %s", run_dir)
    logger.info("Train samples: %d | Val samples: %d", len(train_samples), len(val_samples))

    model = build_model(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["train"]["lr"]))

    scheduler = DDPMScheduler(
        num_train_timesteps=int(config["train"]["num_train_timesteps"]),
        schedule="scaled_linear_beta",
        beta_start=0.0005,
        beta_end=0.0195,
        clip_sample_min=-1.0,
        clip_sample_max=1.0,
    )
    inferer = DiffusionInferer(scheduler)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    if args.resume or args.eval_only or args.checkpoint:
        checkpoint_path = find_checkpoint(run_dir, args.checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint and not args.eval_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0))
        global_step = int(checkpoint.get("global_step", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        logger.info("Loaded checkpoint: %s", checkpoint_path)

    if args.eval_only:
        metrics = run_validation(
            model=model,
            inferer=inferer,
            loader=val_loader,
            scheduler=scheduler,
            device=device,
            run_dir=run_dir,
            writer=writer,
            global_step=global_step,
            epoch=start_epoch,
            num_inference_steps=int(config["train"]["num_inference_steps"]),
            save_predictions=True,
            figure_suffix="eval",
            logger=logger,
        )
        append_csv_row(
            run_dir / "metrics.csv",
            {
                "epoch": start_epoch,
                "split": "eval",
                "loss": metrics["val_loss"],
                "psnr": metrics["val_psnr"],
                "ssim": metrics["val_ssim"],
            },
        )
        if writer is not None:
            writer.flush()
            writer.close()
        return

    train_cfg = config["train"]
    psnr_metric = PSNRMetric(max_val=1.0)
    ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=7)
    total_start = time.time()

    for epoch_idx in range(start_epoch, int(train_cfg["epochs"])):
        epoch = epoch_idx + 1
        model.train()
        epoch_loss = 0.0
        epoch_noise_loss = 0.0
        epoch_x0_pred_loss = 0.0
        epoch_psnr = 0.0
        epoch_ssim = 0.0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=140)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            condition = batch["mri"]["data"].float().to(device)
            images = batch["pet"]["data"].float().to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                noise = torch.randn_like(images)
                intermediates_pred = torch.randn_like(images).to(device)
                x0_pred = torch.randn_like(images).to(device)
                timesteps = torch.randint(
                    0,
                    scheduler.num_train_timesteps,
                    (images.shape[0],),
                    device=images.device,
                ).long()
                noise_pred = inferer(
                    inputs=images,
                    diffusion_model=model,
                    noise=noise,
                    timesteps=timesteps,
                    condition=condition
                )
                noised_image = scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)
                for n in range(len(noise_pred)):
                    intermediates_pred[n,], x0_pred[n] = scheduler.step(
                        torch.unsqueeze(noise_pred[n, :, :, :, :], 0),
                        timesteps[n],
                        torch.unsqueeze(noised_image[n, :, :, :, :], 0),
                    )
                noise_loss = F.mse_loss(noise_pred.float(), noise.float())
                x0_pred_loss = F.l1_loss(x0_pred.float(), images.float())
                loss = noise_loss + x0_pred_loss
                pred_metric = to_unit_range(x0_pred.detach())
                gt_metric = to_unit_range(images.detach())
                psnr_value = float(psnr_metric(pred_metric.cpu(), gt_metric.cpu()).mean().item())
                ssim_value = float(ssim_metric(gt_metric.cpu(), pred_metric.cpu()).mean().item())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item())
            epoch_noise_loss += float(noise_loss.item())
            epoch_x0_pred_loss += float(x0_pred_loss.item())
            epoch_psnr += psnr_value
            epoch_ssim += ssim_value
            global_step += 1

            if writer is not None:
                writer.add_scalar("train/loss", epoch_loss / (step + 1), global_step)
                writer.add_scalar("train/step_loss", float(loss.item()), global_step)
                writer.add_scalar("optim/lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("train/epoch", epoch, global_step)

            progress_bar.set_postfix(
                {
                    "noise_loss": f"{epoch_noise_loss / (step + 1):.4f}",
                    "x0_loss": f"{epoch_x0_pred_loss / (step + 1):.4f}",
                    "loss": f"{epoch_loss / (step + 1):.4f}",
                    "PSNR": f"{epoch_psnr / (step + 1):.4f}",
                    "SSIM": f"{epoch_ssim / (step + 1):.4f}",
                }
            )

        epoch_avg_loss = epoch_loss / max(len(train_loader), 1)
        if writer is not None:
            writer.add_scalar("train/epoch_loss", epoch_avg_loss, epoch)
            if torch.cuda.is_available():
                writer.add_scalar(
                    "sys/gpu_mem_allocated_gb",
                    torch.cuda.max_memory_allocated(device) / (1024 ** 3),
                    epoch,
                )
                torch.cuda.reset_peak_memory_stats(device)

        logger.info(
            "Epoch %d | noise_loss=%.6f | x0_pred_loss=%.6f | total_loss=%.6f",
            epoch,
            epoch_noise_loss / max(len(train_loader), 1),
            epoch_x0_pred_loss / max(len(train_loader), 1),
            epoch_avg_loss,
        )

        if epoch % int(train_cfg["save_every"]) == 0:
            checkpoint_payload = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_val_loss": best_val_loss,
            }
            torch.save(checkpoint_payload, run_dir / f"ckpt_epoch{epoch}.pt")

        if epoch % int(train_cfg["val_every"]) == 0:
            save_predictions = epoch % int(train_cfg["image_log_interval"]) == 0
            metrics = run_validation(
                model=model,
                inferer=inferer,
                loader=val_loader,
                scheduler=scheduler,
                device=device,
                run_dir=run_dir,
                writer=writer,
                global_step=global_step,
                epoch=epoch,
                num_inference_steps=int(train_cfg["num_inference_steps"]),
                save_predictions=save_predictions,
                figure_suffix=f"epoch_{epoch}",
                logger=logger,
            )
            current_val_loss = metrics["val_loss"]
            if writer is not None:
                writer.add_scalar("val/best_loss", min(best_val_loss, current_val_loss), epoch)
            append_csv_row(
                run_dir / "metrics.csv",
                {
                    "epoch": epoch,
                    "split": "val",
                    "loss": current_val_loss,
                    "psnr": metrics["val_psnr"],
                    "ssim": metrics["val_ssim"],
                },
            )

            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_val_loss": best_val_loss,
                    },
                    run_dir / "best_model.pt",
                )
                logger.info("Saved best_model.pt with val_loss=%.6f", best_val_loss)

    logger.info("Training completed in %.2f seconds", time.time() - total_start)

    best_checkpoint = find_checkpoint(run_dir)
    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    metrics = run_validation(
        model=model,
        inferer=inferer,
        loader=val_loader,
        scheduler=scheduler,
        device=device,
        run_dir=run_dir,
        writer=writer,
        global_step=global_step,
        epoch=int(checkpoint.get("epoch", 0)),
        num_inference_steps=int(train_cfg["num_inference_steps"]),
        save_predictions=True,
        figure_suffix="best_model",
        logger=logger,
    )
    append_csv_row(
        run_dir / "metrics.csv",
        {
            "epoch": int(checkpoint.get("epoch", 0)),
            "split": "best_model",
            "loss": metrics["val_loss"],
            "psnr": metrics["val_psnr"],
            "ssim": metrics["val_ssim"],
        },
    )

    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
