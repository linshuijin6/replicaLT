import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import argparse
import sys
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from report_error import email_on_error
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, AutoProcessor

# Ensure the package root is importable when running as a script
THIS_DIR = Path(__file__).resolve().parent
PARENT = THIS_DIR.parent
for path in (THIS_DIR, PARENT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from dataprocess_pipeline.dataset import MODALITIES, PETAdapterDataset, collate_batch


class Adapter(nn.Module):
    def __init__(self, dim: int, hidden: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModalityClassifier(nn.Module):
    def __init__(self, dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def bounded_text_loss(text_stack: torch.Tensor, cos_min: float, cos_max: float) -> torch.Tensor:
    """用余弦相似度做夹逼：避免文本特征塌缩或过分分离。

    text_stack: (batch, num_modalities, dim)
    先对特征做 L2 归一化，再计算两两余弦相似度；相似度低于 cos_min 或高于 cos_max 都会被二次惩罚。
    推荐设置 0 < cos_min < cos_max < 1，例如 0.2/0.8。
    """
    # 归一化后计算余弦相似度
    normed = F.normalize(text_stack, dim=-1)
    sims = torch.einsum("bmd,bnd->bmn", normed, normed)
    upper = torch.triu_indices(text_stack.size(1), text_stack.size(1), offset=1)
    pairwise = sims[:, upper[0], upper[1]]
    loss = F.relu(cos_min - pairwise).pow(2) + F.relu(pairwise - cos_max).pow(2)
    return loss.mean()


def contrastive_loss(image_feats: torch.Tensor, text_feats: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    logits = logit_scale * image_feats @ text_feats.t()
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_i + loss_t)


def positive_pair_alignment_loss(image_feats: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
    """不使用跨被试负样本时的“正样本对齐”损失。

    对每个 (image,text) 正配对最大化余弦相似度（最小化 1-cos）。
    这会降低训练难度，但缺少跨样本负样本，通常判别性会弱一些。
    """
    imgs = F.normalize(image_feats, dim=-1)
    txts = F.normalize(text_feats, dim=-1)
    cos = (imgs * txts).sum(dim=-1)
    return (1.0 - cos).mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adapter-based finetuning for BiomedCLIP")
    parser.add_argument("--model-path", type=str, default="/mnt/nfsdata/nfsdata/lsj.14/replicaLT/BiomedCLIP", help="Path to BiomedCLIP weights")
    parser.add_argument("--root", type=str, default="./synthetic_pet", help="Root directory containing modality folders")
    parser.add_argument("--link-csv", type=str, default="/mnt/nfsdata/nfsdata/lsj.14/replicaLT/adapter_finetune/data_csv/pairs_withPlasma.csv", help="CSV linking modality IDs")
    parser.add_argument("--plasma-csv", type=str, default="/mnt/nfsdata/nfsdata/lsj.14/replicaLT/adapter_finetune/UPENN_PLASMA_FUJIREBIO_QUANTERIX_21Dec2025.csv", help="CSV with plasma biomarkers")
    parser.add_argument("--mri-csv", type=str, default="/mnt/nfsdata/nfsdata/lsj.14/replicaLT/adapter_finetune/MRI_PET_IDs.csv", help="CSV with MRI Study Date for examdate inference")
    parser.add_argument("--yaml", type=str, default=None, help="YAML with common texts and thresholds")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda1", type=float, default=0.1, help="Weight for bounded text loss")
    parser.add_argument("--lambda2", type=float, default=0.2, help="Weight for modality classifier")
    parser.add_argument("--delta-min", type=float, default=0.2)
    parser.add_argument("--delta-max", type=float, default=0.8)
    parser.add_argument("--max-steps", type=int, default=0, help="Optional cap on total update steps (0 = no cap)")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio split by subject_id (0=disable)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subject split")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic random data")
    parser.add_argument("--synthetic-count", type=int, default=8, help="Synthetic subject count")
    parser.add_argument("--cache-root", type=str, default="/mnt/nfsdata/nfsdata/ADNI/cached_npy")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--logdir", type=str, default="runs", help="TensorBoard log dir (event files)")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name (subfolder under logdir)")
    parser.add_argument("--log-every", type=int, default=1, help="Log every N steps")
    parser.add_argument("--save-every-epochs", type=int, default=100, help="Save checkpoint every N epochs")
    parser.add_argument(
        "--no-cross-subject-negatives",
        action="store_true",
        help=(
            "Disable cross-subject batch contrastive alignment (no negatives). "
            "Use positive-pair cosine alignment only (easier but usually less robust)."
        ),
    )
    return parser.parse_args()

@email_on_error()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    use_cross_subject_negatives = not args.no_cross_subject_negatives

    # 生成 run 名：优先用户指定；否则使用 日期_进程号，如 12.19_3716967
    if args.run_name:
        run_name = args.run_name
    else:
        now = datetime.now()
        run_name = f"{now:%m.%d}_{os.getpid()}"
    run_dir = Path(args.logdir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))
    print(f"Run directory: {run_dir}")
    print(f"Cross-subject negatives: {use_cross_subject_negatives}")

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    model.to(device)
    model.eval()
    # max_len = getattr(model.text_model.config, "max_position_embeddings", None) or 77
    for p in model.parameters():
        p.requires_grad = False

    feat_dim = model.config.projection_dim

    text_adapters: Dict[str, Adapter] = {m: Adapter(feat_dim) for m in MODALITIES}
    modality_classifier = ModalityClassifier(feat_dim, num_classes=len(MODALITIES))

    logit_scale_param = nn.Parameter(torch.tensor(float(model.logit_scale.item()), device=device))

    params = [logit_scale_param]
    params += list(modality_classifier.parameters())
    for adp in text_adapters.values():
        params += list(adp.parameters())

    optimizer = torch.optim.AdamW(params, lr=args.lr)

    dataset = PETAdapterDataset(
        root_dir=Path(args.root),
        link_csv=Path(args.link_csv) if args.link_csv else None,
        plasma_csv=Path(args.plasma_csv) if args.plasma_csv else None,
        mri_csv=Path(args.mri_csv) if args.mri_csv else None,
        yaml_config=Path(args.yaml) if args.yaml else None,
        processor=processor,
        synthetic=args.synthetic,
        synthetic_count=args.synthetic_count,
        cache_root=Path(args.cache_root),
    )

    def split_by_subject(samples: List[Dict], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
        subjects = [s.get("ptid", f"IDX_{idx}") for idx, s in enumerate(samples)]
        uniq_subjects = sorted(set(subjects))
        if val_ratio <= 0.0 or len(uniq_subjects) < 2:
            return list(range(len(samples))), []
        rng = random.Random(seed)
        rng.shuffle(uniq_subjects)
        val_count = max(1, int(len(uniq_subjects) * val_ratio))
        val_set = set(uniq_subjects[:val_count])
        train_idx = [i for i, sid in enumerate(subjects) if sid not in val_set]
        val_idx = [i for i, sid in enumerate(subjects) if sid in val_set]
        if not train_idx:
            # fallback: keep at least one sample for training
            train_idx, val_idx = val_idx[:1], val_idx[1:]
        return train_idx, val_idx

    train_indices, val_indices = split_by_subject(dataset.samples, args.val_ratio, args.seed)
    if val_indices:
        train_subset = torch.utils.data.Subset(dataset, train_indices)
        val_subset = torch.utils.data.Subset(dataset, val_indices)
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
        train_subjects = len({dataset.samples[i]["ptid"] for i in train_indices})
        val_subjects = len({dataset.samples[i]["ptid"] for i in val_indices})
        print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}, Subjects (train/val): {train_subjects}/{val_subjects}")
    else:
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
        val_loader = None

    for module in list(text_adapters.values()) + [modality_classifier]:
        module.to(device)

    # ==================== 计算全局模态频率权重 ====================
    # 统计训练集中各模态的样本数
    global_mod_counts = {m: 0 for m in MODALITIES}
    for idx in train_indices:
        sample = dataset.samples[idx]
        available = sample.get("available", [])
        # 如果 sample 没有 available 字段，从 ids 推断
        if not available:
            ids = sample.get("ids", {})
            available = [m for m in MODALITIES if ids.get(m)]
        for m in available:
            global_mod_counts[m] += 1
    
    # 计算逆频率权重: w_m ∝ 1 / freq_m，然后归一化使得 sum(w) = len(MODALITIES)
    total_samples_with_mod = sum(global_mod_counts.values())
    mod_weights = {}
    for m in MODALITIES:
        if global_mod_counts[m] > 0:
            # 逆频率权重
            mod_weights[m] = total_samples_with_mod / (len(MODALITIES) * global_mod_counts[m])
        else:
            mod_weights[m] = 1.0  # fallback
    
    print(f"\n📊 全局模态频率统计:")
    for m in MODALITIES:
        print(f"   {m}: count={global_mod_counts[m]}, weight={mod_weights[m]:.4f}")

    # 保存超参配置
    hparam_path = run_dir / "hparams.json"
    with hparam_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    global_step = 0

    def save_checkpoint(tag: str, epoch: int, global_step: int) -> None:
        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "text_adapters": {m: adp.state_dict() for m, adp in text_adapters.items()},
            "modality_classifier": modality_classifier.state_dict(),
            "logit_scale": logit_scale_param.detach().cpu(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        }
        out_path = run_dir / f"ckpt_{tag}.pt"
        torch.save(ckpt, out_path)
        print(f"Checkpoint saved to {out_path}")

    def forward_batch(batch: Dict, grad_enabled: bool, capture_texts: bool = False) -> Dict:
        with torch.set_grad_enabled(grad_enabled):
            batch_size = len(batch["ptid"])
            mod_offsets = {m: 0 for m in MODALITIES}

            logit_scale = logit_scale_param.exp().clamp(max=100)

            img_by_mod: Dict[str, list] = {m: [] for m in MODALITIES}
            txt_by_mod: Dict[str, list] = {m: [] for m in MODALITIES}
            txt_per_sample: list = [dict() for _ in range(batch_size)]

            first_texts = None
            first_available = None

            for sample_idx in range(batch_size):
                available = batch["available"][sample_idx]
                for modality in available:
                    offset = mod_offsets[modality]
                    mod_offsets[modality] += 1

                    img_feat = batch["pixel_values"][modality][offset].to(device)
                    img_feat = img_feat.view(-1)
                    img_by_mod[modality].append(img_feat)

                    text_inputs = processor(
                        text=batch["texts"][modality][offset],
                        padding=True,
                        return_tensors="pt",
                    ).to(device)
                    with torch.no_grad():
                        base_txt_feat = model.get_text_features(**text_inputs).squeeze(0)
                    txt_feat = text_adapters[modality](base_txt_feat)

                    txt_by_mod[modality].append(txt_feat)
                    txt_per_sample[sample_idx][modality] = txt_feat

                if capture_texts and sample_idx == 0:
                    first_texts = {m: batch["texts"][m][0] for m in available}
                    first_available = list(available)

            clip_losses = {}
            clip_loss_sum_weighted = torch.tensor(0.0, device=device)
            clip_loss_sum_unweighted = torch.tensor(0.0, device=device)  # 用于日志对比
            clip_weight_sum = 0.0
            clip_mod_count = 0
            mod_counts = {}
            clip_acc_i2t_correct = 0
            clip_acc_t2i_correct = 0
            clip_acc_total = 0
            for modality in MODALITIES:
                n_m = len(img_by_mod[modality])
                mod_counts[modality] = n_m
                if n_m == 0:
                    continue
                imgs = torch.stack(img_by_mod[modality], dim=0)
                txts = torch.stack(txt_by_mod[modality], dim=0)
                if use_cross_subject_negatives:
                    imgs_n = F.normalize(imgs, dim=-1)
                    txts_n = F.normalize(txts, dim=-1)
                    loss_m = contrastive_loss(imgs_n, txts_n, logit_scale)
                else:
                    loss_m = positive_pair_alignment_loss(imgs, txts)
                clip_losses[modality] = loss_m
                
                # 使用全局逆频率权重
                w_m = mod_weights[modality]
                clip_loss_sum_weighted = clip_loss_sum_weighted + w_m * loss_m
                clip_loss_sum_unweighted = clip_loss_sum_unweighted + loss_m
                clip_weight_sum += w_m
                clip_mod_count += 1

                if use_cross_subject_negatives:
                    sims = (logit_scale * imgs_n @ txts_n.t()).detach()
                    targets = torch.arange(n_m, device=device)
                    clip_acc_i2t_correct += (sims.argmax(dim=1) == targets).sum().item()
                    clip_acc_t2i_correct += (sims.argmax(dim=0) == targets).sum().item()
                    clip_acc_total += n_m
            
            # 加权平均 loss_clip（用于训练）
            loss_clip = clip_loss_sum_weighted / max(clip_weight_sum, 1e-8)
            # 未加权平均（用于日志对比）
            loss_clip_unweighted = clip_loss_sum_unweighted / max(clip_mod_count, 1)

            if use_cross_subject_negatives and clip_acc_total > 0:
                clip_acc_i2t = clip_acc_i2t_correct / clip_acc_total
                clip_acc_t2i = clip_acc_t2i_correct / clip_acc_total
            else:
                clip_acc_i2t = 0.0
                clip_acc_t2i = 0.0

            img_all = []
            target_all = []
            for modality in MODALITIES:
                n_m = len(img_by_mod[modality])
                if n_m == 0:
                    continue
                img_all.append(torch.stack(img_by_mod[modality], dim=0))
                mod_idx = MODALITIES.index(modality)
                target_all.append(torch.full((n_m,), mod_idx, device=device, dtype=torch.long))
            if img_all:
                img_all_t = F.normalize(torch.cat(img_all, dim=0), dim=-1)
                targets_t = torch.cat(target_all, dim=0)
                mod_logits = modality_classifier(img_all_t)
                loss_mod = F.cross_entropy(mod_logits, targets_t)
                acc_mod = (mod_logits.argmax(dim=-1) == targets_t).float().mean()
            else:
                loss_mod = torch.tensor(0.0, device=device)
                acc_mod = torch.tensor(0.0, device=device)

            loss_bound_sum = torch.tensor(0.0, device=device)
            for sample_idx in range(batch_size):
                feats = txt_per_sample[sample_idx]
                available = [m for m in MODALITIES if m in feats]
                if len(available) >= 2:
                    text_stack = torch.stack([feats[m] for m in available], dim=0).unsqueeze(0)
                    loss_bound_sum = loss_bound_sum + bounded_text_loss(text_stack, args.delta_min, args.delta_max)
            loss_bound = loss_bound_sum / max(batch_size, 1)

            total_loss = loss_clip + args.lambda1 * loss_bound + args.lambda2 * loss_mod

        return {
            "total_loss": total_loss,
            "loss_clip": loss_clip,
            "loss_clip_unweighted": loss_clip_unweighted,  # 未加权版本用于对比
            "loss_bound": loss_bound,
            "loss_mod": loss_mod,
            "acc_mod": acc_mod,
            "clip_acc_i2t": clip_acc_i2t,
            "clip_acc_t2i": clip_acc_t2i,
            "clip_acc_i2t_correct": clip_acc_i2t_correct,  # 用于 micro-average
            "clip_acc_t2i_correct": clip_acc_t2i_correct,
            "clip_acc_total": clip_acc_total,
            "mod_counts": mod_counts,
            "clip_losses": clip_losses,
            "first_texts": first_texts,
            "first_available": first_available,
            "logit_scale": logit_scale,
        }

    for epoch in range(1, args.epochs + 1):
        # 切换到训练模式（Adapter 含 Dropout）
        for adp in text_adapters.values():
            adp.train()
        modality_classifier.train()

        for batch in train_loader:
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
            optimizer.zero_grad(set_to_none=True)

            result = forward_batch(batch, grad_enabled=True, capture_texts=(global_step == 0))

            result["total_loss"].backward()
            optimizer.step()

            global_step += 1

            if result["first_texts"]:
                print("示例文本(仅当前批次第一个样本的可用模态):")
                for m in result["first_available"]:
                    print(f"{m}:\n{result['first_texts'][m]}\n---")

            if args.log_every > 0 and (global_step % args.log_every == 0):
                lr = optimizer.param_groups[0].get("lr", args.lr)
                writer.add_scalar("loss/total", float(result["total_loss"].item()), global_step)
                writer.add_scalar("loss/clip", float(result["loss_clip"].item()), global_step)
                writer.add_scalar("loss/clip_unweighted", float(result["loss_clip_unweighted"].item()), global_step)
                writer.add_scalar("loss/bound", float(result["loss_bound"].item()), global_step)
                writer.add_scalar("loss/mod", float(result["loss_mod"].item()), global_step)
                writer.add_scalar("metrics/mod_acc", float(result["acc_mod"].item()), global_step)
                writer.add_scalar("metrics/clip_acc_i2t", float(result["clip_acc_i2t"]), global_step)
                writer.add_scalar("metrics/clip_acc_t2i", float(result["clip_acc_t2i"]), global_step)
                writer.add_scalar("optim/lr", float(lr), global_step)
                writer.add_scalar("model/logit_scale", float(result["logit_scale"].mean().item()), global_step)
                for modality in MODALITIES:
                    writer.add_scalar(f"data/count_{modality}", float(result["mod_counts"].get(modality, 0)), global_step)
                for modality, loss_m in result["clip_losses"].items():
                    writer.add_scalar(f"loss/clip_{modality}", float(loss_m.item()), global_step)
                if global_step == 1 and result["first_texts"]:
                    for modality, text in result["first_texts"].items():
                        writer.add_text(f"text/sample0_{modality}", text, global_step)

            print(
                f"epoch={epoch} step={global_step} mode={'cross' if use_cross_subject_negatives else 'pos_only'} "
                f"loss={result['total_loss'].item():.4f} clip={result['loss_clip'].item():.4f} "
                f"bound={result['loss_bound'].item():.4f} mod={result['loss_mod'].item():.4f} acc={result['acc_mod'].item():.3f} "
                f"clip_i2t={result['clip_acc_i2t']:.3f} clip_t2i={result['clip_acc_t2i']:.3f} counts={result['mod_counts']}"
            )

        if val_loader is not None:
            # 切换到评估模式（禁用 Dropout）
            for adp in text_adapters.values():
                adp.eval()
            modality_classifier.eval()

            val_totals = {
                "total_loss": 0.0,
                "loss_clip": 0.0,
                "loss_clip_unweighted": 0.0,
                "loss_bound": 0.0,
                "loss_mod": 0.0,
                "acc_mod": 0.0,
            }
            # 用于 micro-average 的累计
            val_clip_i2t_correct = 0
            val_clip_t2i_correct = 0
            val_clip_total = 0
            val_mod_counts = {m: 0 for m in MODALITIES}
            val_steps = 0
            with torch.no_grad():
                for batch in val_loader:
                    out = forward_batch(batch, grad_enabled=False)
                    val_steps += 1
                    for key in val_totals.keys():
                        value = out[key]
                        if torch.is_tensor(value):
                            value = value.item()
                        val_totals[key] += float(value)
                    # 累计用于 micro-average
                    val_clip_i2t_correct += out["clip_acc_i2t_correct"]
                    val_clip_t2i_correct += out["clip_acc_t2i_correct"]
                    val_clip_total += out["clip_acc_total"]
                    # 累计各模态数量
                    for m in MODALITIES:
                        val_mod_counts[m] += out["mod_counts"].get(m, 0)
            if val_steps > 0:
                val_avg = {k: v / val_steps for k, v in val_totals.items()}
                # micro-average clip acc（按样本数加权）
                val_clip_acc_i2t = val_clip_i2t_correct / max(val_clip_total, 1)
                val_clip_acc_t2i = val_clip_t2i_correct / max(val_clip_total, 1)
                writer.add_scalar("val/loss_total", val_avg["total_loss"], epoch)
                writer.add_scalar("val/loss_clip", val_avg["loss_clip"], epoch)
                writer.add_scalar("val/loss_clip_unweighted", val_avg["loss_clip_unweighted"], epoch)
                writer.add_scalar("val/loss_bound", val_avg["loss_bound"], epoch)
                writer.add_scalar("val/loss_mod", val_avg["loss_mod"], epoch)
                writer.add_scalar("val/acc_mod", val_avg["acc_mod"], epoch)
                writer.add_scalar("val/clip_acc_i2t", val_clip_acc_i2t, epoch)
                writer.add_scalar("val/clip_acc_t2i", val_clip_acc_t2i, epoch)
                writer.add_scalar("val/model_logit_scale", float(logit_scale_param.exp().clamp(max=100).item()), epoch)
                # 记录验证集各模态数量
                for m in MODALITIES:
                    writer.add_scalar(f"val/count_{m}", float(val_mod_counts[m]), epoch)
                print(
                    f"[val] epoch={epoch} loss={val_avg['total_loss']:.4f} clip={val_avg['loss_clip']:.4f} "
                    f"bound={val_avg['loss_bound']:.4f} mod={val_avg['loss_mod']:.4f} acc={val_avg['acc_mod']:.3f} "
                    f"clip_i2t={val_clip_acc_i2t:.3f} clip_t2i={val_clip_acc_t2i:.3f} counts={val_mod_counts}"
                )

        if args.save_every_epochs > 0 and (epoch % args.save_every_epochs == 0):
            save_checkpoint(f"epoch{epoch}", epoch, global_step)

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    writer.flush()
    writer.close()
    save_checkpoint("last", epoch if 'epoch' in locals() else 0, global_step)
    print("Training loop finished (smoke test).")


if __name__ == "__main__":
    main()
