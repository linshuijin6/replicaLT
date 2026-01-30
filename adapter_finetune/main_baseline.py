"""
基线训练脚本：Adapter 微调 BiomedCLIP（不使用 plasma 引导）

设计目标：
- 冻结 BiomedCLIP 主干，仅训练文本侧 Adapter、ModalityClassifier、logit_scale
- 使用跨被试对比损失（InfoNCE）+ bounded_text_loss + modality_classifier
- 不引入 plasma 相关分支或损失
- 支持 AV45/TAU/FDG 三种示踪剂
- 评估指标：I2T / T2I Recall@1（Top-1 正确率）
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, AutoProcessor

# 确保可导入上级模块
THIS_DIR = Path(__file__).resolve().parent
PARENT = THIS_DIR.parent
for path in (THIS_DIR, PARENT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

# 仅使用三种 PET 模态
MODALITIES: Tuple[str, ...] = ("FDG", "AV45", "TAU")

# 模态 ID 列映射
MODALITY_ID_COL = {
    "MRI": "id_mri",
    "FDG": "id_fdg",
    "AV45": "id_av45",
    "TAU": "id_av1451",
}


# ============ 模型组件 ============

class Adapter(nn.Module):
    """文本特征适配器：轻量级 MLP 微调文本嵌入"""
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
    """模态分类器：辅助监督图像特征区分模态"""
    def __init__(self, dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============ 损失函数 ============

def contrastive_loss(image_feats: torch.Tensor, text_feats: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """跨被试对比损失（InfoNCE 风格）"""
    logits = logit_scale * image_feats @ text_feats.t()
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_i + loss_t)


def positive_pair_alignment_loss(image_feats: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
    """正样本余弦对齐损失（无负样本）"""
    imgs = F.normalize(image_feats, dim=-1)
    txts = F.normalize(text_feats, dim=-1)
    cos = (imgs * txts).sum(dim=-1)
    return (1.0 - cos).mean()


def bounded_text_loss(text_stack: torch.Tensor, cos_min: float, cos_max: float) -> torch.Tensor:
    """文本相似度约束：避免同一被试多模态文本特征塌缩或过度分离
    
    text_stack: (batch, num_modalities, dim)
    相似度低于 cos_min 或高于 cos_max 都会被二次惩罚
    """
    normed = F.normalize(text_stack, dim=-1)
    sims = torch.einsum("bmd,bnd->bmn", normed, normed)
    upper = torch.triu_indices(text_stack.size(1), text_stack.size(1), offset=1)
    pairwise = sims[:, upper[0], upper[1]]
    loss = F.relu(cos_min - pairwise).pow(2) + F.relu(pairwise - cos_max).pow(2)
    return loss.mean()


# ============ 基线数据集（不使用 plasma）============

def load_cached_feature(path: Path) -> torch.Tensor:
    """加载预缓存的 BiomedCLIP ImageEncoder 输出"""
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Cached feature 格式错误: {path}")
    cls = payload.get("cls_token")
    if cls is None:
        raise ValueError(f"Cached feature 缺少 cls_token: {path}")
    cls = torch.as_tensor(cls, dtype=torch.float32)
    if cls.ndim == 1:
        cls = cls.unsqueeze(0)
    return cls


def get_modality_text_baseline(modality: str) -> str:
    """基线文本模板：仅包含模态描述，不含 plasma 信息"""
    templates = {
        "FDG": (
            "This is an FDG PET brain image. "
            "FDG PET is a functional brain imaging technique that visualizes the dynamic changes in glucose metabolism, "
            "directly linked to neuronal energy demands and synaptic activity. "
            "Areas with decreased metabolic activity exhibit reduced signal intensity. "
            "High-intensity metabolic hotspots in gray matter are key markers of neuronal activity."
        ),
        "AV45": (
            "This is an AV45 PET brain image. "
            "AV45 PET is a molecular imaging technique that highlights the static distribution of amyloid-beta plaques, "
            "a critical pathological marker of Alzheimer's disease. "
            "This imaging modality provides a spatial map of amyloid deposition in cortical regions and "
            "can distinguish amyloid-positive areas from amyloid-negative white matter regions. "
            "The primary focus is on identifying amyloid deposition patterns to assess disease progression and pathological burden."
        ),
        "TAU": (
            "This is a TAU PET brain image. "
            "TAU PET is a molecular neuroimaging technique that visualizes the spatial distribution of aggregated tau protein, "
            "which reflects the presence of neurofibrillary tangles associated with neurodegeneration. "
            "Tau PET highlights region-specific tau accumulation, particularly in medial temporal, parietal, and association cortices, "
            "providing a topographical map of tau pathology that correlates with disease stage, cognitive decline, and neuronal dysfunction."
        ),
    }
    return templates.get(modality, f"This is a {modality} PET brain image.")


class BaselinePETDataset(Dataset):
    """基线数据集：不使用 plasma 信息，仅使用模态描述文本"""
    
    def __init__(
        self,
        link_csv: Path,
        cache_root: Path,
        processor,
        synthetic: bool = False,
        synthetic_count: int = 8,
    ) -> None:
        super().__init__()
        self.link_csv = Path(link_csv)
        self.cache_root = Path(cache_root)
        self.processor = processor
        self.synthetic = synthetic
        self.rng = random.Random(42)
        
        if synthetic:
            self.samples = self._build_synthetic_samples(synthetic_count)
        else:
            self.samples = self._build_real_samples()
        
        print(f"[BaselinePETDataset] 加载 {len(self.samples)} 个样本")

    def _build_synthetic_samples(self, count: int) -> List[Dict]:
        samples = []
        for idx in range(count):
            ptid = f"SYN_{idx:03d}"
            ids = {m: f"SYN_{idx:03d}_{m}" for m in MODALITIES}
            samples.append({"ptid": ptid, "ids": ids})
        return samples

    def _build_real_samples(self) -> List[Dict]:
        import pandas as pd
        link_df = pd.read_csv(self.link_csv)
        link_df.columns = [c.lower() for c in link_df.columns]
        
        samples: List[Dict] = []
        for _, row in link_df.iterrows():
            ptid = str(row.get("subject_id", row.iloc[0]))
            ids: Dict[str, Optional[str]] = {}
            for modality, col in MODALITY_ID_COL.items():
                ids[modality] = None
                if col in row and isinstance(row[col], str) and row[col].strip() != "":
                    ids[modality] = row[col].strip()
            
            # 至少有一种 PET 模态
            has_pet = any(ids[m] for m in MODALITIES)
            if not has_pet:
                continue
            samples.append({"ptid": ptid, "ids": ids})
        return samples

    def _load_feature(self, modality: str, sample: Dict) -> torch.Tensor:
        """加载预缓存的图像特征"""
        if self.synthetic:
            return torch.randn(1, 512, dtype=torch.float32)
        
        modality_id = sample["ids"].get(modality)
        if modality_id is None:
            raise FileNotFoundError(f"{modality} id missing for sample {sample['ptid']}")
        
        suffix_map = {"FDG": "fdg", "AV45": "av45", "TAU": "av1451"}
        suffix = suffix_map.get(modality, modality.lower())
        filename = f"{sample['ptid']}_{suffix}_{modality_id}.vision.pt"
        feat_path = self.cache_root / filename
        
        if not feat_path.exists():
            raise FileNotFoundError(f"Cached feature not found: {feat_path}")
        
        return load_cached_feature(feat_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        texts: Dict[str, str] = {}
        pixel_values: Dict[str, torch.Tensor] = {}
        available: List[str] = []
        
        for modality in MODALITIES:
            if sample["ids"].get(modality):
                # 基线：仅使用模态描述文本，不含 plasma
                texts[modality] = get_modality_text_baseline(modality)
                feature = self._load_feature(modality, sample)
                pixel_values[modality] = feature.squeeze(0)
                available.append(modality)
        
        return {
            "ptid": sample["ptid"],
            "texts": texts,
            "pixel_values": pixel_values,
            "available": available,
        }


def collate_batch(batch: List[Dict]) -> Dict:
    """批次整理函数"""
    ptids = [item["ptid"] for item in batch]
    texts: Dict[str, List[str]] = {m: [] for m in MODALITIES}
    pixel_values: Dict[str, List[torch.Tensor]] = {m: [] for m in MODALITIES}
    avail_list: List[List[str]] = []

    for item in batch:
        avail_list.append(item["available"])
        for modality in MODALITIES:
            if modality in item["texts"]:
                texts[modality].append(item["texts"][modality])
                pixel_values[modality].append(item["pixel_values"][modality])

    stacked_pixels: Dict[str, torch.Tensor] = {}
    for m, tensors in pixel_values.items():
        if tensors:
            stacked_pixels[m] = torch.stack(tensors, dim=0)

    return {"ptid": ptids, "texts": texts, "pixel_values": stacked_pixels, "available": avail_list}


# ============ 训练逻辑 ============

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline Adapter-based finetuning for BiomedCLIP (No Plasma)")
    parser.add_argument("--model-path", type=str, default="/mnt/nfsdata/nfsdata/lsj.14/replicaLT/BiomedCLIP")
    parser.add_argument("--link-csv", type=str, default="/mnt/nfsdata/nfsdata/lsj.14/replicaLT/adapter_finetune/data_csv/pairs_withPlasma.csv")
    parser.add_argument("--cache-root", type=str, default="/mnt/nfsdata/nfsdata/ADNI/cached_npy")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda1", type=float, default=0.1, help="Weight for bounded text loss")
    parser.add_argument("--lambda2", type=float, default=0.2, help="Weight for modality classifier")
    parser.add_argument("--delta-min", type=float, default=0.2)
    parser.add_argument("--delta-max", type=float, default=0.8)
    parser.add_argument("--max-steps", type=int, default=0, help="Optional cap on total update steps (0 = no cap)")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-count", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--logdir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every-epochs", type=int, default=100)
    parser.add_argument("--no-cross-subject-negatives", action="store_true",
                        help="Disable cross-subject contrastive loss, use positive-pair alignment only")
    return parser.parse_args()


def split_by_subject(samples: List[Dict], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    """按被试划分训练集/验证集，防止数据泄漏"""
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
        train_idx, val_idx = val_idx[:1], val_idx[1:]
    return train_idx, val_idx


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    use_cross_subject_negatives = not args.no_cross_subject_negatives

    # 生成 run 名
    if args.run_name:
        run_name = args.run_name
    else:
        now = datetime.now()
        run_name = f"baseline_{now:%m.%d}_{os.getpid()}"
    run_dir = Path(args.logdir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))
    
    print(f"=" * 60)
    print(f"基线训练（不使用 plasma 引导）")
    print(f"Run directory: {run_dir}")
    print(f"Cross-subject negatives: {use_cross_subject_negatives}")
    print(f"=" * 60)

    # 加载模型
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    feat_dim = model.config.projection_dim

    # 初始化可训练组件
    text_adapters: Dict[str, Adapter] = {m: Adapter(feat_dim) for m in MODALITIES}
    modality_classifier = ModalityClassifier(feat_dim, num_classes=len(MODALITIES))
    logit_scale_param = nn.Parameter(torch.tensor(float(model.logit_scale.item()), device=device))

    params = [logit_scale_param]
    params += list(modality_classifier.parameters())
    for adp in text_adapters.values():
        params += list(adp.parameters())

    optimizer = torch.optim.AdamW(params, lr=args.lr)

    # 加载数据集
    dataset = BaselinePETDataset(
        link_csv=Path(args.link_csv),
        cache_root=Path(args.cache_root),
        processor=processor,
        synthetic=args.synthetic,
        synthetic_count=args.synthetic_count,
    )

    # 划分训练/验证集
    train_indices, val_indices = split_by_subject(dataset.samples, args.val_ratio, args.seed)
    if val_indices:
        train_subset = torch.utils.data.Subset(dataset, train_indices)
        val_subset = torch.utils.data.Subset(dataset, val_indices)
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
        train_subjects = len({dataset.samples[i]["ptid"] for i in train_indices})
        val_subjects = len({dataset.samples[i]["ptid"] for i in val_indices})
        print(f"训练样本: {len(train_indices)}, 验证样本: {len(val_indices)}, 被试 (train/val): {train_subjects}/{val_subjects}")
    else:
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
        val_loader = None

    for module in list(text_adapters.values()) + [modality_classifier]:
        module.to(device)

    # 计算全局模态频率权重（逆频率加权）
    global_mod_counts = {m: 0 for m in MODALITIES}
    for idx in train_indices:
        sample = dataset.samples[idx]
        for m in MODALITIES:
            if sample["ids"].get(m):
                global_mod_counts[m] += 1

    total_samples_with_mod = sum(global_mod_counts.values())
    mod_weights = {}
    for m in MODALITIES:
        if global_mod_counts[m] > 0:
            mod_weights[m] = total_samples_with_mod / (len(MODALITIES) * global_mod_counts[m])
        else:
            mod_weights[m] = 1.0

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
        """前向计算一个批次"""
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

                    # 图像特征
                    img_feat = batch["pixel_values"][modality][offset].to(device)
                    img_feat = img_feat.view(-1)
                    img_by_mod[modality].append(img_feat)

                    # 文本特征（通过 Adapter）
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

            # CLIP 对比损失（按模态分别计算，再加权平均）
            clip_losses = {}
            clip_loss_sum_weighted = torch.tensor(0.0, device=device)
            clip_loss_sum_unweighted = torch.tensor(0.0, device=device)
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

                w_m = mod_weights[modality]
                clip_loss_sum_weighted = clip_loss_sum_weighted + w_m * loss_m
                clip_loss_sum_unweighted = clip_loss_sum_unweighted + loss_m
                clip_weight_sum += w_m
                clip_mod_count += 1

                # 计算 Recall@1
                if use_cross_subject_negatives:
                    sims = (logit_scale * imgs_n @ txts_n.t()).detach()
                    targets = torch.arange(n_m, device=device)
                    clip_acc_i2t_correct += (sims.argmax(dim=1) == targets).sum().item()
                    clip_acc_t2i_correct += (sims.argmax(dim=0) == targets).sum().item()
                    clip_acc_total += n_m

            loss_clip = clip_loss_sum_weighted / max(clip_weight_sum, 1e-8)
            loss_clip_unweighted = clip_loss_sum_unweighted / max(clip_mod_count, 1)

            if use_cross_subject_negatives and clip_acc_total > 0:
                clip_acc_i2t = clip_acc_i2t_correct / clip_acc_total
                clip_acc_t2i = clip_acc_t2i_correct / clip_acc_total
            else:
                clip_acc_i2t = 0.0
                clip_acc_t2i = 0.0

            # 模态分类损失
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

            # bounded_text_loss：约束同一被试多模态文本特征相似度
            loss_bound_sum = torch.tensor(0.0, device=device)
            for sample_idx in range(batch_size):
                feats = txt_per_sample[sample_idx]
                available = [m for m in MODALITIES if m in feats]
                if len(available) >= 2:
                    text_stack = torch.stack([feats[m] for m in available], dim=0).unsqueeze(0)
                    loss_bound_sum = loss_bound_sum + bounded_text_loss(text_stack, args.delta_min, args.delta_max)
            loss_bound = loss_bound_sum / max(batch_size, 1)

            # 总损失
            total_loss = loss_clip + args.lambda1 * loss_bound + args.lambda2 * loss_mod

        return {
            "total_loss": total_loss,
            "loss_clip": loss_clip,
            "loss_clip_unweighted": loss_clip_unweighted,
            "loss_bound": loss_bound,
            "loss_mod": loss_mod,
            "acc_mod": acc_mod,
            "clip_acc_i2t": clip_acc_i2t,
            "clip_acc_t2i": clip_acc_t2i,
            "clip_acc_i2t_correct": clip_acc_i2t_correct,
            "clip_acc_t2i_correct": clip_acc_t2i_correct,
            "clip_acc_total": clip_acc_total,
            "mod_counts": mod_counts,
            "clip_losses": clip_losses,
            "first_texts": first_texts,
            "first_available": first_available,
            "logit_scale": logit_scale,
        }

    # ============ 训练循环 ============
    for epoch in range(1, args.epochs + 1):
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

            # 首次打印示例文本
            if result["first_texts"]:
                print("示例文本（仅当前批次第一个样本的可用模态）:")
                for m in result["first_available"]:
                    print(f"{m}:\n{result['first_texts'][m][:200]}...\n---")

            # 日志记录
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

            print(
                f"epoch={epoch} step={global_step} mode={'cross' if use_cross_subject_negatives else 'pos_only'} "
                f"loss={result['total_loss'].item():.4f} clip={result['loss_clip'].item():.4f} "
                f"bound={result['loss_bound'].item():.4f} mod={result['loss_mod'].item():.4f} acc={result['acc_mod'].item():.3f} "
                f"clip_i2t={result['clip_acc_i2t']:.3f} clip_t2i={result['clip_acc_t2i']:.3f} counts={result['mod_counts']}"
            )

        # 验证
        if val_loader is not None:
            for adp in text_adapters.values():
                adp.eval()
            modality_classifier.eval()

            val_totals = {
                "total_loss": 0.0, "loss_clip": 0.0, "loss_clip_unweighted": 0.0,
                "loss_bound": 0.0, "loss_mod": 0.0, "acc_mod": 0.0,
            }
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
                    val_clip_i2t_correct += out["clip_acc_i2t_correct"]
                    val_clip_t2i_correct += out["clip_acc_t2i_correct"]
                    val_clip_total += out["clip_acc_total"]
                    for m in MODALITIES:
                        val_mod_counts[m] += out["mod_counts"].get(m, 0)

            if val_steps > 0:
                val_avg = {k: v / val_steps for k, v in val_totals.items()}
                val_clip_acc_i2t = val_clip_i2t_correct / max(val_clip_total, 1)
                val_clip_acc_t2i = val_clip_t2i_correct / max(val_clip_total, 1)

                writer.add_scalar("val/loss_total", val_avg["total_loss"], epoch)
                writer.add_scalar("val/loss_clip", val_avg["loss_clip"], epoch)
                writer.add_scalar("val/loss_bound", val_avg["loss_bound"], epoch)
                writer.add_scalar("val/loss_mod", val_avg["loss_mod"], epoch)
                writer.add_scalar("val/acc_mod", val_avg["acc_mod"], epoch)
                writer.add_scalar("val/clip_acc_i2t", val_clip_acc_i2t, epoch)
                writer.add_scalar("val/clip_acc_t2i", val_clip_acc_t2i, epoch)
                for m in MODALITIES:
                    writer.add_scalar(f"val/count_{m}", float(val_mod_counts[m]), epoch)

                print(
                    f"[val] epoch={epoch} loss={val_avg['total_loss']:.4f} clip={val_avg['loss_clip']:.4f} "
                    f"bound={val_avg['loss_bound']:.4f} mod={val_avg['loss_mod']:.4f} acc={val_avg['acc_mod']:.3f} "
                    f"clip_i2t={val_clip_acc_i2t:.3f} clip_t2i={val_clip_acc_t2i:.3f} counts={val_mod_counts}"
                )

        # 保存检查点
        if args.save_every_epochs > 0 and (epoch % args.save_every_epochs == 0):
            save_checkpoint(f"epoch{epoch}", epoch, global_step)

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    writer.flush()
    writer.close()
    save_checkpoint("last", epoch if 'epoch' in locals() else 0, global_step)
    print("基线训练完成。")


if __name__ == "__main__":
    main()
