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
    cache_dir = '/mnt/nfsdata/nfsdata/linshuijin/ADNILT/cache_cpu'
    checkpoint_path = './checkpoint/first_part_95.pth'
    local_model_path = "/mnt/nfsdata/nfsdata/linshuijin/replicaLT/BiomedCLIP"
    result_dir = '/mnt/nfsdata/nfsdata/linshuijin/ADNILT/results'
    processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
    bio_model = AutoModel.from_pretrained(local_model_path, trust_remote_code=True)
    bio_model.to(device)
    bio_model.eval()
    fdg_text_optimized = (
        "FDG PET is a functional brain imaging technique that visualizes the dynamic changes in glucose metabolism, directly linked to neuronal energy demands and synaptic activity. It serves as a tool to assess functional connectivity and energy utilization across brain regions. Areas with decreased metabolic activity, such as those affected by neurodegenerative diseases, should exhibit reduced signal intensity. High-intensity metabolic hotspots in gray matter (e.g., the cerebral cortex and basal ganglia) are key markers of neuronal activity. "

    )

    av45_text_optimized = (
        "AV45 PET is a molecular imaging technique that highlights the static distribution of amyloid-beta plaques, a critical pathological marker of Alzheimer's disease. This imaging modality provides a spatial map of amyloid deposition in cortical regions (e.g., the temporal, parietal, and frontal lobes) and can distinguish amyloid-positive areas from amyloid-negative white matter regions. The primary focus is on identifying amyloid deposition patterns to assess disease progression and pathological burden.")

    texts_optimized = [fdg_text_optimized, av45_text_optimized]
    inputs_optimized = processor(text=texts_optimized, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        text_features_optimized = bio_model.get_text_features(**inputs_optimized)

    fdg_feature_optimized = text_features_optimized[0]
    fdg_feature_optimized = fdg_feature_optimized.unsqueeze(0)
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
            "fdg": item["fdg"],
            "description": item.get("description", {}),  # 确保 description 存在
            "fdg_index": item.get("fdg_index"),  # 保留 fdg_index
            "av45_index": item.get("av45_index"),  # 保留 av45_index
        }
        for item in val_data
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

    def fdg_index_transform(x):
        return torch.cat([desc_text_features[x].unsqueeze(0), fdg_feature_optimized], dim=0)

    def av45_index_transform(x):
        return torch.cat([desc_text_features[x].unsqueeze(0), av45_feature_optimized], dim=0)

    val_transforms = mt.Compose([
        mt.Lambdad(keys=["mri", "av45", "fdg"], func=mat_load),
        mt.EnsureChannelFirstd(keys=["mri", "av45", "fdg"], channel_dim='no_channel'),
        mt.Orientationd(keys=["mri", "av45", "fdg"], axcodes="LPI"),
        mt.CropForegroundd(keys=["mri", "av45", "fdg"], source_key="mri"),
        mt.HistogramNormalized(keys=["mri"]),
        mt.ResizeWithPadOrCropd(keys=["mri", "av45", "fdg"], spatial_size=[160, 192, 160]),
        mt.Spacingd(keys=["mri", "av45", "fdg"], pixdim=(1.0, 1.0, 1.0)),
        mt.NormalizeIntensityd(keys=["mri", "av45", "fdg"]),
        mt.ScaleIntensityd(keys=["mri", "av45", "fdg"]),
        mt.Lambdad(keys=["fdg_index"], func=fdg_index_transform),
        mt.Lambdad(keys=["av45_index"], func=av45_index_transform),
        Lambda(lambda x: x.cpu() if isinstance(x, torch.Tensor) else x)
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
    checkpoint = torch.load(checkpoint_path)

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
        "fdg": os.path.join(result_dir, "data_augmentation_fdg"),
        "av45": os.path.join(result_dir, "data_augmentation_av45"),
        "generated_fdg": os.path.join(result_dir, "data_augmentation_fdg_generated"),
        "generated_av45": os.path.join(result_dir, "data_augmentation_av45_generated")
    }
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # 数据增强变换
    orientation_transform = mt.Orientationd(keys=["mri", "av45", "fdg"], axcodes="LPI")
    ssim_fdg = []
    psnr_fdg = []
    ssim_av45 = []
    psnr_av45 = []
    avg_ssim_fdg = 0.0
    avg_psnr_fdg = 0.0
    avg_ssim_av45 = 0.0
    avg_psnr_av45 = 0.0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Inference",ncols=160)
        for step, data_val in enumerate(pbar):
            # 数据增强处理
            data_val = data_val#data_val = orientation_transform(data_val)

            images = data_val["mri"].to(device)
            seg_fdg = data_val["fdg"].to(device)  # Ground truth FDG
            seg_av45 = data_val["av45"].to(device)  # Ground truth AV45
            fdg_index = data_val["fdg_index"].to(device)
            av45_index = data_val["av45_index"].to(device)
            names = data_val["name"]
            # 检查 FDG 和 AV45 数据是否为二值化（只有 0 和 1）
            has_fdg = not torch.all((seg_fdg == 0) | (seg_fdg == 1))  # 如果不是二值化数据，则参与计算
            has_av45 = not torch.all((seg_av45 == 0) | (seg_av45 == 1))  # 如果不是二值化数据，则参与计算

            # 固定 time_embedding 和 t
            time_embedding = torch.tensor([0], device=device, dtype=torch.long)
            t = time_embedding.float() / 1000  # t = 0

            # FDG 输出（仅当非二值化时计算）
            if has_fdg:
                v_fdg_output = model(
                    x=images,
                    timesteps=time_embedding,
                    context=fdg_index
                )
                x_fdg_t = images + (v_fdg_output)  # t = 0，因此 x_fdg_t = images
                x_fdg_t = torch.clamp(x_fdg_t, min=clip_sample_min, max=clip_sample_max)

                # 检查形状并保存生成的 FDG 图像
                if x_fdg_t.shape == (1, 1, 160, 192, 160):
                    x_fdg_t = x_fdg_t.squeeze()  # 去掉前两个维度 (1, 1)，得到 [160, 192, 160]
                    x_fdg_np = x_fdg_t.cpu().numpy()  # 转换为 NumPy 数组
                    output_path = os.path.join(output_dirs["generated_fdg"], f'generated_fdg_{names[0]}.nii.gz')  # 保存路径
                    nib.save(nib.Nifti1Image(x_fdg_np, affine=np.eye(4)), output_path)  # 保存为 NIfTI 文件

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

            # 保存真实的 MRI、FDG 和 AV45 图像
            if images.shape == (1, 1, 160, 192, 160):
                images = images.squeeze()  # 去掉前两个维度 (1, 1)，得到 [160, 192, 160]
                images_np = images.cpu().numpy()  # 转换为 NumPy 数组
                output_path = os.path.join(output_dirs["mri"], f'real_mri_{names[0]}.nii.gz')  # 保存路径
                nib.save(nib.Nifti1Image(images_np, affine=np.eye(4)), output_path)  # 保存为 NIfTI 文件
                # print(f"Saved real MRI image to {output_path}")

            if seg_fdg.shape == (1, 1, 160, 192, 160) and has_fdg:
                seg_fdg = seg_fdg.squeeze()  # 去掉前两个维度 (1, 1)，得到 [160, 192, 160]
                seg_fdg_np = seg_fdg.cpu().numpy()  # 转换为 NumPy 数组
                output_path = os.path.join(output_dirs["fdg"], f'real_fdg_{names[0]}.nii.gz')  # 保存路径
                nib.save(nib.Nifti1Image(seg_fdg_np, affine=np.eye(4)), output_path)  # 保存为 NIfTI 文件
                # print(f"Saved real FDG image to {output_path}")

            if seg_av45.shape == (1, 1, 160, 192, 160) and has_av45:
                seg_av45 = seg_av45.squeeze()  # 去掉前两个维度 (1, 1)，得到 [160, 192, 160]
                seg_av45_np = seg_av45.cpu().numpy()  # 转换为 NumPy 数组
                output_path = os.path.join(output_dirs["av45"], f'real_av45_{names[0]}.nii.gz')  # 保存路径
                nib.save(nib.Nifti1Image(seg_av45_np, affine=np.eye(4)), output_path)  # 保存为 NIfTI 文件
                # print(f"Saved real AV45 image to {output_path}")

            # 计算 SSIM 和 PSNR（仅当非二值化时计算）
            if has_fdg and x_fdg_t is not None:
                ssim_fdg_value = ssim_metric(seg_fdg.unsqueeze(0).unsqueeze(0), x_fdg_t.unsqueeze(0).unsqueeze(0))
                psnr_fdg_value = psnr_metric(seg_fdg.unsqueeze(0).unsqueeze(0), x_fdg_t.unsqueeze(0).unsqueeze(0))
                tqdm.write(f"[FDG] SSIM: {ssim_fdg_value.mean().item():.4f}, PSNR: {psnr_fdg_value.mean().item():.4f}")
                ssim_fdg.append(ssim_fdg_value.mean().item())
                psnr_fdg.append(psnr_fdg_value.mean().item())

            if has_av45 and x_av45_t is not None:
                ssim_av45_value = ssim_metric(seg_av45.unsqueeze(0).unsqueeze(0), x_av45_t.unsqueeze(0).unsqueeze(0))
                psnr_av45_value = psnr_metric(seg_av45.unsqueeze(0).unsqueeze(0), x_av45_t.unsqueeze(0).unsqueeze(0))
                tqdm.write(f"[AV45] SSIM: {ssim_av45_value.mean().item():.4f}, PSNR: {psnr_av45_value.mean().item():.4f}")
                ssim_av45.append(ssim_av45_value.mean().item())
                psnr_av45.append(psnr_av45_value.mean().item())

            if ssim_fdg:
                avg_ssim_fdg = sum(ssim_fdg) / len(ssim_fdg)
                avg_psnr_fdg = sum(psnr_fdg) / len(psnr_fdg)
            if ssim_av45:
                avg_ssim_av45 = sum(ssim_av45) / len(ssim_av45)
                avg_psnr_av45 = sum(psnr_av45) / len(psnr_av45)

            info_str = (
                f"len={len(ssim_fdg)}, ssim_fdg={avg_ssim_fdg:.4f}, psnr_fdg={avg_psnr_fdg:.4f}, "
                f"len={len(ssim_av45)}, ssim_av45={avg_ssim_av45:.4f}, psnr_av45={avg_psnr_av45:.4f}"
            )
            pbar.set_postfix({
                'ssim_fdg': f"{avg_ssim_fdg:.4f}",
                'psnr_fdg': f"{avg_psnr_fdg:.2f}",
                'ssim_av45': f"{avg_ssim_av45:.4f}",
                'psnr_av45': f"{avg_psnr_av45:.2f}",
            })
            pbar.write(info_str)


if __name__ == "__main__":
    main()