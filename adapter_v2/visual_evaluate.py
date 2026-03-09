import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from tqdm import tqdm

try:
    import seaborn as sns
except Exception:
    sns = None


CLASS_COLORS = {
    0: "#1f77b4",
    1: "#ff7f0e",
    2: "#2ca02c",
}


def _build_val_loader(config: dict, split_json: Path):
    from torch.utils.data import DataLoader

    from dataset import TAUPlasmaDataset, collate_fn, split_by_subject
    from train import apply_fixed_split, resolve_plasma_config

    script_dir = Path(__file__).parent
    seed = int(config["training"].get("seed", 42))
    class_names = config["classes"]["names"]
    selected_plasma_keys, plasma_prompts = resolve_plasma_config(config)

    csv_path = script_dir / config["data"]["csv_path"]
    cache_dir = Path(config["data"]["cache_dir"])
    diagnosis_csv = config.get("data", {}).get("diagnosis_csv", None)
    diagnosis_code_map = config.get("classes", {}).get("diagnosis_code_map", None)

    full_dataset = TAUPlasmaDataset(
        csv_path=str(csv_path),
        cache_dir=str(cache_dir),
        plasma_keys=selected_plasma_keys,
        class_names=class_names,
        diagnosis_csv=diagnosis_csv,
        diagnosis_code_map=diagnosis_code_map,
        skip_cache_set=set(),
    )

    val_ratio = float(config["data"].get("val_ratio", 0.1))
    train_indices, val_indices = split_by_subject(full_dataset, val_ratio=val_ratio, seed=seed)
    train_indices, val_indices = apply_fixed_split(
        full_dataset=full_dataset,
        train_indices=train_indices,
        val_indices=val_indices,
        val_split_json=str(split_json),
        seed=seed,
        val_ratio=val_ratio,
    )

    val_dataset = TAUPlasmaDataset(
        csv_path=str(csv_path),
        cache_dir=str(cache_dir),
        plasma_keys=selected_plasma_keys,
        class_names=class_names,
        plasma_stats=full_dataset.plasma_stats,
        diagnosis_csv=diagnosis_csv,
        diagnosis_code_map=diagnosis_code_map,
        subset_indices=val_indices,
        skip_cache_set=set(),
    )

    batch_size = int(config["training"].get("batch_size", 16))
    num_workers = int(config["training"].get("num_workers", 4))
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return val_loader, selected_plasma_keys, plasma_prompts, class_names


def _build_model(config: dict, plasma_prompts: list[str], class_names: list[str], device: torch.device):
    from models import CoCoOpTAUModel

    model_cfg = config["model"]
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
    )
    return model.to(device)


def _collect_embeddings(model, dataloader, device, expected_plasma_dim: int | None = None):
    model.eval()

    all_img_emb = []
    all_plasma_emb = []
    all_labels = []
    all_subjects = []
    all_plasma_vals = []
    all_plasma_mask = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collect", leave=False):
            patch_emb = batch["patch_emb"].to(device)
            label_idx = batch["label_idx"].to(device)
            plasma_vals = batch["plasma_vals"].to(device)
            plasma_mask = batch["plasma_mask"].to(device)
            subjects = batch["subjects"]

            if expected_plasma_dim is not None:
                if plasma_vals.shape[-1] != expected_plasma_dim or plasma_mask.shape[-1] != expected_plasma_dim:
                    raise ValueError(
                        f"plasma dim mismatch: vals={plasma_vals.shape[-1]}, mask={plasma_mask.shape[-1]}, expected={expected_plasma_dim}"
                    )

            outputs = model(
                tau_tokens=patch_emb,
                diagnosis_id=label_idx,
                plasma_values=plasma_vals,
                plasma_mask=plasma_mask,
            )

            all_img_emb.append(outputs["img_emb"].detach().cpu())
            all_plasma_emb.append(outputs["plasma_emb"].detach().cpu())
            all_labels.append(label_idx.detach().cpu())
            all_subjects.extend([str(s) for s in subjects])
            all_plasma_vals.append(plasma_vals.detach().cpu())
            all_plasma_mask.append(plasma_mask.detach().cpu())

    img_emb = torch.cat(all_img_emb, dim=0)
    plasma_emb = torch.cat(all_plasma_emb, dim=0)
    labels = torch.cat(all_labels, dim=0)
    plasma_vals = torch.cat(all_plasma_vals, dim=0)
    plasma_mask = torch.cat(all_plasma_mask, dim=0)

    valid_mask = labels >= 0
    valid_idx = torch.where(valid_mask)[0]

    return {
        "img_emb": img_emb[valid_mask],
        "plasma_emb": plasma_emb[valid_mask],
        "labels": labels[valid_mask],
        "subjects": [all_subjects[i] for i in valid_idx.tolist()],
        "plasma_vals": plasma_vals[valid_mask],
        "plasma_mask": plasma_mask[valid_mask],
    }


def _plot_similarity_heatmap(img_emb, plasma_emb, labels, class_names, save_path: Path):
    img_n = F.normalize(img_emb, dim=1)
    plasma_n = F.normalize(plasma_emb, dim=1)
    sim = (img_n @ plasma_n.t()).numpy()

    order = np.argsort(labels.numpy(), kind="stable")
    sim_sorted = sim[np.ix_(order, order)]
    labels_sorted = labels.numpy()[order]

    plt.figure(figsize=(8, 7))
    if sns is not None:
        sns.heatmap(sim_sorted, cmap="RdBu_r", vmin=-1.0, vmax=1.0, cbar=True, xticklabels=False, yticklabels=False)
    else:
        plt.imshow(sim_sorted, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
        plt.colorbar(fraction=0.046, pad=0.04)

    class_boundaries = []
    for cls_id in range(len(class_names)):
        cls_count = int(np.sum(labels_sorted == cls_id))
        class_boundaries.append(cls_count)

    acc = 0
    centers = []
    for c in class_boundaries:
        if c > 0:
            centers.append(acc + c / 2.0)
            acc += c
        else:
            centers.append(acc)

    acc = 0
    for c in class_boundaries[:-1]:
        acc += c
        plt.axhline(acc, color="white", lw=1.2)
        plt.axvline(acc, color="white", lw=1.2)

    if len(sim_sorted) > 0:
        plt.xticks(centers, class_names, rotation=0)
        plt.yticks(centers, class_names, rotation=0)

    plt.title("Similarity Heatmap: sim(img_i, plasma_j)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _ecdf(values: np.ndarray):
    if values.size == 0:
        return np.array([]), np.array([])
    x = np.sort(values)
    y = np.arange(1, values.size + 1, dtype=np.float64) / float(values.size)
    return x, y


def _plot_intraclass_rank_curve(img_emb, plasma_emb, labels, class_names, save_path: Path):
    img_n = F.normalize(img_emb, dim=1)
    plasma_n = F.normalize(plasma_emb, dim=1)

    labels_np = labels.numpy().astype(np.int64)
    all_percentiles = []
    per_class = {i: [] for i in range(len(class_names))}

    for i in range(img_n.shape[0]):
        cls = labels_np[i]
        same_cls_idx = np.where(labels_np == cls)[0]
        if same_cls_idx.size <= 1:
            continue

        sims = torch.mv(plasma_n[same_cls_idx], img_n[i]).numpy()
        order_desc = np.argsort(-sims)
        self_local_idx = int(np.where(same_cls_idx == i)[0][0])
        rank_pos = int(np.where(order_desc == self_local_idx)[0][0])
        percentile = rank_pos / float(same_cls_idx.size - 1)

        all_percentiles.append(percentile)
        per_class[cls].append(percentile)

    plt.figure(figsize=(8, 6))
    x_all, y_all = _ecdf(np.asarray(all_percentiles, dtype=np.float64))
    if x_all.size > 0:
        plt.plot(x_all, y_all, lw=2.2, color="black", label=f"All (n={x_all.size})")

    for cls_id, cls_name in enumerate(class_names):
        x, y = _ecdf(np.asarray(per_class[cls_id], dtype=np.float64))
        if x.size == 0:
            continue
        plt.plot(x, y, lw=1.8, color=CLASS_COLORS.get(cls_id, None), label=f"{cls_name} (n={x.size})")

    plt.xlabel("Correct-pair rank percentile within class (0=best)")
    plt.ylabel("CDF")
    plt.title("Intra-class Rank Distribution")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.2)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _plot_tsne_joint(img_emb, plasma_emb, labels, class_names, save_path: Path, perplexity: float):
    n = img_emb.shape[0]
    if n == 0:
        return

    x = torch.cat([img_emb, plasma_emb], dim=0).numpy().astype(np.float32)
    max_valid_perp = max(2.0, float(x.shape[0] - 1))
    used_perplexity = min(float(perplexity), max_valid_perp)

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=used_perplexity,
        init="pca",
        learning_rate="auto",
    )
    z = tsne.fit_transform(x)

    z_img = z[:n]
    z_plasma = z[n:]
    labels_np = labels.numpy().astype(np.int64)

    plt.figure(figsize=(8.5, 7.5))
    for cls_id, cls_name in enumerate(class_names):
        idx = np.where(labels_np == cls_id)[0]
        if idx.size == 0:
            continue

        color = CLASS_COLORS.get(cls_id, None)
        plt.scatter(
            z_img[idx, 0],
            z_img[idx, 1],
            s=24,
            marker="o",
            c=[color],
            alpha=0.85,
            label=f"img-{cls_name}",
        )
        plt.scatter(
            z_plasma[idx, 0],
            z_plasma[idx, 1],
            s=30,
            marker="*",
            c=[color],
            alpha=0.85,
            label=f"plasma-{cls_name}",
        )

    for i in range(n):
        plt.plot(
            [z_img[i, 0], z_plasma[i, 0]],
            [z_img[i, 1], z_plasma[i, 1]],
            color="gray",
            lw=0.35,
            alpha=0.35,
        )

    handles, labels_legend = plt.gca().get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels_legend):
        if l not in uniq:
            uniq[l] = h
    plt.legend(uniq.values(), uniq.keys(), frameon=False, fontsize=8, ncol=2)
    plt.title("Joint t-SNE Projection (img vs plasma)")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or y.size < 3:
        return float("nan")
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _plot_scatter_plasma_sim(img_emb, plasma_emb, labels, plasma_vals, plasma_mask, plasma_keys, class_names, save_path: Path):
    img_n = F.normalize(img_emb, dim=1)
    plasma_n = F.normalize(plasma_emb, dim=1)
    pair_sim = (img_n * plasma_n).sum(dim=1).numpy()

    labels_np = labels.numpy().astype(np.int64)
    vals_np = plasma_vals.numpy()
    mask_np = plasma_mask.numpy().astype(bool)

    k = len(plasma_keys)
    fig, axes = plt.subplots(1, k, figsize=(4.6 * k, 4.0), squeeze=False)

    for col in range(k):
        ax = axes[0, col]
        valid = mask_np[:, col]
        x = vals_np[valid, col]
        y = pair_sim[valid]
        cls_v = labels_np[valid]

        for cls_id, cls_name in enumerate(class_names):
            idx = cls_v == cls_id
            if np.sum(idx) == 0:
                continue
            ax.scatter(
                x[idx],
                y[idx],
                s=18,
                alpha=0.78,
                c=CLASS_COLORS.get(cls_id, None),
                label=cls_name,
            )

        r = _safe_pearsonr(x, y)
        r_text = "nan" if not np.isfinite(r) else f"{r:.3f}"
        ax.set_title(f"{plasma_keys[col]}\nr={r_text}, n={x.size}", fontsize=10)
        ax.set_xlabel("plasma (normalized)")
        ax.set_ylabel("sim(img_i, plasma_i)")
        ax.grid(alpha=0.2)

    handles, labels_legend = axes[0, 0].get_legend_handles_labels()
    if len(handles) > 0:
        uniq = {}
        for h, l in zip(handles, labels_legend):
            if l not in uniq:
                uniq[l] = h
        fig.legend(uniq.values(), uniq.keys(), loc="upper center", ncol=len(uniq), frameon=False)

    fig.suptitle("Plasma value vs Pair Similarity")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def _default_output_dir(ckpt_path: Path) -> Path:
    script_dir = Path(__file__).parent
    run_name = ckpt_path.parent.parent.name if ckpt_path.parent.name == "ckpt" else ckpt_path.stem
    return script_dir / "vis_alignment" / run_name


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize image-plasma individual alignment from train.py checkpoints")
    parser.add_argument("--checkpoint", type=str, default='/home/ssddata/linshuijin/replicaLT/adapter_v2/runs/03.09_122698/ckpt/epoch_050.pt', help="checkpoint path, e.g. runs/<run>/ckpt/best.pt")
    parser.add_argument("--config", type=str, default="config.yaml", help="adapter_v2 config path")
    parser.add_argument("--split_json", type=str, default="fixed_split.json", help="fixed split json shared with train.py")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--output_dir", type=str, default=None, help="output directory for PNG figures")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    return parser.parse_args()


def main():
    from train import load_config

    args = parse_args()

    script_dir = Path(__file__).parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = script_dir / config_path

    split_json = Path(args.split_json)
    if not split_json.is_absolute():
        split_json = script_dir / split_json

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = (Path.cwd() / ckpt_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    output_dir = Path(args.output_dir) if args.output_dir is not None else _default_output_dir(ckpt_path)
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(str(config_path))

    requested_device = str(args.device).lower()
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    val_loader, plasma_keys, plasma_prompts, class_names = _build_val_loader(config=config, split_json=split_json)

    model = _build_model(config=config, plasma_prompts=plasma_prompts, class_names=class_names, device=device)
    state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state["model_state_dict"], strict=True)

    data = _collect_embeddings(
        model=model,
        dataloader=val_loader,
        device=device,
        expected_plasma_dim=len(plasma_keys),
    )

    if data["img_emb"].shape[0] == 0:
        raise RuntimeError("No valid validation samples after filtering invalid diagnosis labels.")

    heatmap_path = output_dir / "heatmap.png"
    rank_path = output_dir / "rank_curve.png"
    tsne_path = output_dir / "tsne_joint.png"
    scatter_path = output_dir / "scatter_plasma_sim.png"

    _plot_similarity_heatmap(
        img_emb=data["img_emb"],
        plasma_emb=data["plasma_emb"],
        labels=data["labels"],
        class_names=class_names,
        save_path=heatmap_path,
    )
    _plot_intraclass_rank_curve(
        img_emb=data["img_emb"],
        plasma_emb=data["plasma_emb"],
        labels=data["labels"],
        class_names=class_names,
        save_path=rank_path,
    )
    _plot_tsne_joint(
        img_emb=data["img_emb"],
        plasma_emb=data["plasma_emb"],
        labels=data["labels"],
        class_names=class_names,
        save_path=tsne_path,
        perplexity=float(args.perplexity),
    )
    _plot_scatter_plasma_sim(
        img_emb=data["img_emb"],
        plasma_emb=data["plasma_emb"],
        labels=data["labels"],
        plasma_vals=data["plasma_vals"],
        plasma_mask=data["plasma_mask"],
        plasma_keys=plasma_keys,
        class_names=class_names,
        save_path=scatter_path,
    )

    print(f"[Done] output dir: {output_dir}")
    print(f"[Done] {heatmap_path}")
    print(f"[Done] {rank_path}")
    print(f"[Done] {tsne_path}")
    print(f"[Done] {scatter_path}")


if __name__ == "__main__":
    main()
