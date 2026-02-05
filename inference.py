#%%
import os

from monai.metrics import PSNRMetric
from monai.transforms import Lambda

device_id = 5
# 定义裁剪的最小值和最大值
os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
import json

from monai.data import PersistentDataset
import nibabel as nib

def mat_load(filepath):
    """
    使用 nibabel 加载 NIfTI 文件，并转换为 NumPy 数组。
    """
    return nib.load(filepath).get_fdata()


def main():
    from generative.networks.nets import DiffusionModelUNet
    import monai.transforms as mt
    from transformers import AutoProcessor, AutoModel
    from generative.metrics import SSIMMetric

    import os

    import numpy as np
    import torch
    from monai.data import DataLoader
    from tqdm import tqdm
    from torch.amp import GradScaler  # 从 torch.amp 导入 GradScaler
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    cache_dir = '/mnt/nfsdata/nfsdata/linshuijin/cache'
    checkpoint_path = '/home/ssddata/linshuijin/replicaLT/runs/02.03_614278/ckpt_epoch180.pt'
    local_model_path = "/home/ssddata/linshuijin/replicaLT/BiomedCLIP"
    result_dir = '/mnt/nfsdata/nfsdata/linshuijin/ADNILT/results_0205ME'
    processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
    bio_model = AutoModel.from_pretrained(local_model_path, trust_remote_code=True)
    bio_model.to(device)
    bio_model.eval()
    tau_text_optimized = (
        "TAU PET is a molecular neuroimaging technique that visualizes the spatial distribution of "
            "aggregated tau protein, which reflects the presence of neurofibrillary tangles associated "
            "with neurodegeneration. Tau PET highlights region-specific tau accumulation, particularly "
            "in medial temporal, parietal, and association cortices, providing a topographical map of "
            "tau pathology that correlates with disease stage, cognitive decline, and neuronal dysfunction.")


    av45_text_optimized = (
        "AV45 PET is a molecular imaging technique that highlights the static distribution of amyloid-beta plaques, a critical pathological marker of Alzheimer's disease. This imaging modality provides a spatial map of amyloid deposition in cortical regions (e.g., the temporal, parietal, and frontal lobes) and can distinguish amyloid-positive areas from amyloid-negative white matter regions. The primary focus is on identifying amyloid deposition patterns to assess disease progression and pathological burden.")

    texts_optimized = [tau_text_optimized, av45_text_optimized]
    inputs_optimized = processor(text=texts_optimized, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        text_features_optimized = bio_model.get_text_features(**inputs_optimized)

    tau_feature_optimized = text_features_optimized[0]
    tau_feature_optimized = tau_feature_optimized.unsqueeze(0)
    av45_feature_optimized = text_features_optimized[1]
    av45_feature_optimized = av45_feature_optimized.unsqueeze(0)

    model = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=1,  # 1,
        out_channels=1,
        num_channels=(32, 64, 64, 128),
        attention_levels=(False, False, False, True),
        num_res_blocks=1,
        num_head_channels=(0, 0, 0, 128),
        with_conditioning=True,
        cross_attention_dim=512,
        use_flash_attention=True,
    )

    val_json_path = "./val_data_with_description.json"
    with open(val_json_path, "r") as f:
        val_data = json.load(f)
    val_data = [
        {
            "name": item["name"],
            "mri": item["mri"],
            "av45": item["av45"],
            "tau": item["tau"],
            "description": item.get("old_descr") or "",  # 确保 description 存在且不为 None
            "tau_index": idx,  # 使用本地索引，transform 中会加上 len(train_data) 偏移
            "av45_index": idx,
        }
        for idx, item in enumerate(val_data)
    ]
    modal_information = [data["description"] for data in val_data if data["description"] is not None]
    text_inputs = processor(
        modal_information,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=256 # 根据模型输入限制选择合适的 max_length
    ).to(device)
    with torch.no_grad():
        desc_text_features = bio_model.get_text_features(text_inputs['input_ids'])  # 使用模型方法提取嵌入
        print(f"Shape of features: {desc_text_features.shape}, Dtype: {desc_text_features.dtype}")

    def tau_index_transform(x):
        return torch.cat([desc_text_features[x].unsqueeze(0), tau_feature_optimized], dim=0)

    def av45_index_transform(x):
        return torch.cat([desc_text_features[x].unsqueeze(0), av45_feature_optimized], dim=0)

    val_transforms = mt.Compose([
        mt.Lambdad(keys=["mri", "av45", "tau"], func=mat_load),
        mt.EnsureChannelFirstd(keys=["mri", "av45", "tau"], channel_dim='no_channel'),
        mt.Orientationd(keys=["mri", "av45", "tau"], axcodes="LPI"),
        mt.CropForegroundd(keys=["mri", "av45", "tau"], source_key="mri"),
        mt.HistogramNormalized(keys=["mri"]),
        mt.ResizeWithPadOrCropd(keys=["mri", "av45", "tau"], spatial_size=[160, 192, 160]),
        mt.Spacingd(keys=["mri", "av45", "tau"], pixdim=(1.0, 1.0, 1.0)),
        mt.NormalizeIntensityd(keys=["mri", "av45", "tau"]),
        mt.ScaleIntensityd(keys=["mri", "av45", "tau"]),
        mt.Lambdad(keys=["tau_index"], func=tau_index_transform),
        mt.Lambdad(keys=["av45_index"], func=av45_index_transform),
    ])
    optimizer= torch.optim.Adam(params=model.parameters(), lr=2.5e-5)
    val_ds = PersistentDataset(data=val_data, transform=val_transforms, cache_dir=os.path.join(cache_dir, "val"))
    # ⭐ 遍历所有样本以触发缓存生成
    # for i in range(len(val_ds)):
    #     _ = val_ds[i]
    # print("✅ 全部样本缓存生成完成！")
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)



    scaler = GradScaler()
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 初始化模型和优化器
    model.load_state_dict(checkpoint['model'])
    model.eval()  # 设置为推理模式
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 定义评估指标
    ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=7)
    psnr_metric = PSNRMetric(max_val=1.0)

    # 推理过程
    clip_sample_min = 0  # 设置合适的最小值
    clip_sample_max = 1  # 设置合适的最大值

    # 创建保存目录
    output_dirs = {
        "mri": os.path.join(result_dir, "data_augmentation_mri"),
        "tau": os.path.join(result_dir, "data_augmentation_tau"),
        "av45": os.path.join(result_dir, "data_augmentation_av45"),
        "generated_tau": os.path.join(result_dir, "data_augmentation_tau_generated"),
        "generated_av45": os.path.join(result_dir, "data_augmentation_av45_generated")
    }
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # 数据增强变换
    orientation_transform = mt.Orientationd(keys=["mri", "av45", "tau"], axcodes="LPI")
    ssim_tau = []
    psnr_tau = []
    ssim_av45 = []
    psnr_av45 = []
    avg_ssim_tau = 0.0
    avg_psnr_tau = 0.0
    avg_ssim_av45 = 0.0
    avg_psnr_av45 = 0.0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Inference",ncols=160)
        for step, data_val in enumerate(pbar):
            # 数据增强处理
            data_val = data_val#data_val = orientation_transform(data_val)

            images = data_val["mri"].to(device)
            seg_tau = data_val["tau"].to(device)  # Ground truth tau
            seg_av45 = data_val["av45"].to(device)  # Ground truth AV45
            tau_index = data_val["tau_index"].to(device)
            av45_index = data_val["av45_index"].to(device)
            names = data_val["name"]
            # 检查 tau 和 AV45 数据是否为二值化（只有 0 和 1）
            has_tau = not torch.all((seg_tau == 0) | (seg_tau == 1))  # 如果不是二值化数据，则参与计算
            has_av45 = not torch.all((seg_av45 == 0) | (seg_av45 == 1))  # 如果不是二值化数据，则参与计算

            # 固定 time_embedding 和 t
            time_embedding = torch.tensor([0], device=device, dtype=torch.long)
            t = time_embedding.float() / 1000  # t = 0

            # tau 输出（仅当非二值化时计算）
            if has_tau:
                v_tau_output = model(
                    x=images,
                    timesteps=time_embedding,
                    context=tau_index
                )
                x_tau_t = images + (v_tau_output)  # t = 0，因此 x_tau_t = images
                x_tau_t = torch.clamp(x_tau_t, min=clip_sample_min, max=clip_sample_max)

                # 检查形状并保存生成的 tau 图像
                if x_tau_t.shape == (1, 1, 160, 192, 160):
                    x_tau_t = x_tau_t.squeeze()  # 去掉前两个维度 (1, 1)，得到 [160, 192, 160]
                    x_tau_np = x_tau_t.cpu().numpy()  # 转换为 NumPy 数组
                    output_path = os.path.join(output_dirs["generated_tau"], f'generated_tau_{names[0]}.nii.gz')  # 保存路径
                    nib.save(nib.Nifti1Image(x_tau_np, affine=np.eye(4)), output_path)  # 保存为 NIfTI 文件

            # AV45 输出（仅当非二值化时计算）
            if has_av45:
                v_av45_output = model(
                    x=images,
                    timesteps=time_embedding,
                    context=av45_index
                )
                x_av45_t = images + (v_av45_output)  # t = 0，因此 x_av45_t = images
                x_av45_t = torch.clamp(x_av45_t, min=clip_sample_min, max=clip_sample_max)

                # 检查形状并保存生成的 AV45 图像
                if x_av45_t.shape == (1, 1, 160, 192, 160):
                    x_av45_t = x_av45_t.squeeze()  # 去掉前两个维度 (1, 1)，得到 [160, 192, 160]
                    x_av45_np = x_av45_t.cpu().numpy()  # 转换为 NumPy 数组
                    output_path = os.path.join(output_dirs["generated_av45"], f'generated_av45_{names[0]}.nii.gz')  # 保存路径
                    nib.save(nib.Nifti1Image(x_av45_np, affine=np.eye(4)), output_path)  # 保存为 NIfTI 文件
                    # print(f"Saved generated AV45 image to {output_path}")

            # 保存真实的 MRI、tau 和 AV45 图像
            if images.shape == (1, 1, 160, 192, 160):
                images = images.squeeze()  # 去掉前两个维度 (1, 1)，得到 [160, 192, 160]
                images_np = images.cpu().numpy()  # 转换为 NumPy 数组
                output_path = os.path.join(output_dirs["mri"], f'real_mri_{names[0]}.nii.gz')  # 保存路径
                nib.save(nib.Nifti1Image(images_np, affine=np.eye(4)), output_path)  # 保存为 NIfTI 文件
                # print(f"Saved real MRI image to {output_path}")

            if seg_tau.shape == (1, 1, 160, 192, 160) and has_tau:
                seg_tau = seg_tau.squeeze()  # 去掉前两个维度 (1, 1)，得到 [160, 192, 160]
                seg_tau_np = seg_tau.cpu().numpy()  # 转换为 NumPy 数组
                output_path = os.path.join(output_dirs["tau"], f'real_tau_{names[0]}.nii.gz')  # 保存路径
                nib.save(nib.Nifti1Image(seg_tau_np, affine=np.eye(4)), output_path)  # 保存为 NIfTI 文件
                # print(f"Saved real tau image to {output_path}")

            if seg_av45.shape == (1, 1, 160, 192, 160) and has_av45:
                seg_av45 = seg_av45.squeeze()  # 去掉前两个维度 (1, 1)，得到 [160, 192, 160]
                seg_av45_np = seg_av45.cpu().numpy()  # 转换为 NumPy 数组
                output_path = os.path.join(output_dirs["av45"], f'real_av45_{names[0]}.nii.gz')  # 保存路径
                nib.save(nib.Nifti1Image(seg_av45_np, affine=np.eye(4)), output_path)  # 保存为 NIfTI 文件
                # print(f"Saved real AV45 image to {output_path}")

            # 计算 SSIM 和 PSNR（仅当非二值化时计算）
            if has_tau and x_tau_t is not None:
                ssim_tau_value = ssim_metric(seg_tau.unsqueeze(0).unsqueeze(0), x_tau_t.unsqueeze(0).unsqueeze(0))
                psnr_tau_value = psnr_metric(seg_tau.unsqueeze(0).unsqueeze(0), x_tau_t.unsqueeze(0).unsqueeze(0))
                tqdm.write(f"[tau] SSIM: {ssim_tau_value.mean().item():.4f}, PSNR: {psnr_tau_value.mean().item():.4f}")
                ssim_tau.append(ssim_tau_value.mean().item())
                psnr_tau.append(psnr_tau_value.mean().item())

            if has_av45 and x_av45_t is not None:
                ssim_av45_value = ssim_metric(seg_av45.unsqueeze(0).unsqueeze(0), x_av45_t.unsqueeze(0).unsqueeze(0))
                psnr_av45_value = psnr_metric(seg_av45.unsqueeze(0).unsqueeze(0), x_av45_t.unsqueeze(0).unsqueeze(0))
                tqdm.write(f"[AV45] SSIM: {ssim_av45_value.mean().item():.4f}, PSNR: {psnr_av45_value.mean().item():.4f}")
                ssim_av45.append(ssim_av45_value.mean().item())
                psnr_av45.append(psnr_av45_value.mean().item())

            if ssim_tau:
                avg_ssim_tau = sum(ssim_tau) / len(ssim_tau)
                avg_psnr_tau = sum(psnr_tau) / len(psnr_tau)
            if ssim_av45:
                avg_ssim_av45 = sum(ssim_av45) / len(ssim_av45)
                avg_psnr_av45 = sum(psnr_av45) / len(psnr_av45)

            info_str = (
                f"len={len(ssim_tau)}, ssim_tau={avg_ssim_tau:.4f}, psnr_tau={avg_psnr_tau:.4f}, "
                f"len={len(ssim_av45)}, ssim_av45={avg_ssim_av45:.4f}, psnr_av45={avg_psnr_av45:.4f}"
            )
            pbar.set_postfix({
                'ssim_tau': f"{avg_ssim_tau:.4f}",
                'psnr_tau': f"{avg_psnr_tau:.2f}",
                'ssim_av45': f"{avg_ssim_av45:.4f}",
                'psnr_av45': f"{avg_psnr_av45:.2f}",
            })
            pbar.write(info_str)


if __name__ == "__main__":
    main()