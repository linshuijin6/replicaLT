#%%
import pandas as pd
from report_error import email_on_error
from sklearn.model_selection import train_test_split
import nibabel as nib
from generative.networks.nets.diffusion_model_unet import DistributedDiffusionModelUNet


def get_subject_id(filename):
    """从文件名中提取统一的 subject ID（前三个部分，如 '002_S_0295'）"""
    parts = filename.split('_')
    return f"{parts[0]}_{parts[1]}_{parts[2]}"


# 自定义加载函数
def mat_load(filepath):
    """
    使用 nibabel 加载 NIfTI 文件，并转换为 NumPy 数组。
    """
    return nib.load(filepath).get_fdata()


@email_on_error()
def main():
    import torch
    import logging
    logging.basicConfig(level=logging.INFO)
    from liutuo_utils import compare_3d_jet, compare_3d, donkey_noise_like
    from monai.data import PersistentDataset
    from transformers import AutoProcessor, AutoModel
    import os
    import json
    from transformers import AutoTokenizer, AutoModel
    from tqdm import tqdm
    from torch.amp import GradScaler, autocast  # 从 torch.amp 导入 GradScaler
    import torch.nn.functional as F

    # 0.训练参数设置
    base_dir = "/mnt/nfsdata/nfsdata/linshuijin/ADNILT/ADNI1234_mri_fdg_av45_original"
    cache_dir = '/mnt/nfsdata/nfsdata/linshuijin/ADNILT/tmp_cache'
    device_id = [2,3]
    # 定义裁剪的最小值和最大值
    clip_sample_min = 0  # 设置合适的最小值
    clip_sample_max = 1   # 设置合适的最大值

    # 定义模态权重
    alpha = 1.0  # FDG 损失权重
    beta = 1.0   # AV45 损失权重
    # 设置 PyTorch 的默认 CUDA 设备
    torch.cuda.set_device(device_id[0]) if len(device_id) == 1 else None
    size_of_dataset = None  # 设置为 None 以使用完整数据集，或设置为所需的样本数量
    n_epochs = 200
    val_interval =10
    checkpoint_dir = './checkpoint'

    # 是否继续上次训练
    last_epoch = 50

    # 确认当前默认 CUDA 设备
    current_device = torch.cuda.current_device()
    print(f"Switched to CUDA device: {current_device}")

    # 1. 加载 BiomedCLIP 模型和处理器
    local_model_path = "/mnt/nfsdata/nfsdata/linshuijin/replicaLT/BiomedCLIP"
    processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
    bio_model = AutoModel.from_pretrained(local_model_path, trust_remote_code=True)

    device = torch.device(f"cuda:{device_id[0]}" if torch.cuda.is_available() else "cpu")
    bio_model.to(device)
    bio_model.eval()
    # 加载优化后的描述
    fdg_text_optimized = (
        "FDG PET is a functional brain imaging technique that visualizes the dynamic changes "
        "in glucose metabolism, directly linked to neuronal energy demands and synaptic activity. "
        "It serves as a tool to assess functional connectivity and energy utilization across brain "
        "regions. Areas with decreased metabolic activity, such as those affected by neurodegenerative "
        "diseases, should exhibit reduced signal intensity. High-intensity metabolic hotspots in gray matter "
        "(e.g., the cerebral cortex and basal ganglia) are key markers of neuronal activity. "
    )
    av45_text_optimized = (
        "AV45 PET is a molecular imaging technique that highlights the static distribution of "
        "amyloid-beta plaques, a critical pathological marker of Alzheimer's disease. "
        "This imaging modality provides a spatial map of amyloid deposition in cortical regions "
        "(e.g., the temporal, parietal, and frontal lobes) and can distinguish amyloid-positive areas "
        "from amyloid-negative white matter regions. The primary focus is on identifying amyloid "
        "deposition patterns to assess disease progression and pathological burden."
    )
    # 提取特征向量
    texts_optimized = [fdg_text_optimized, av45_text_optimized]
    inputs_optimized = processor(text=texts_optimized, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features_optimized = bio_model.get_text_features(**inputs_optimized)

    # 计算相似度
    fdg_feature_optimized = text_features_optimized[0]
    fdg_feature_optimized = fdg_feature_optimized.unsqueeze(0)
    av45_feature_optimized = text_features_optimized[1]
    av45_feature_optimized = av45_feature_optimized.unsqueeze(0)

    # cosine_similarity_optimized = F.cosine_similarity(fdg_feature_optimized, av45_feature_optimized).item()

    # print(f"Cosine similarity between optimized FDG PET and AV45 PET features: {cosine_similarity_optimized:.4f}")

    # 2. 预处理数据，对患者信息生成Text Embeddings via BiomedCLIP
    mri_dir = os.path.join(base_dir, "mri_strip_registered")
    av45_dir = os.path.join(base_dir, "PET1_AV45_strip_registered")
    fdg_dir = os.path.join(base_dir, "PET2_FDG_strip_registered")
    csv_path = os.path.join(base_dir, 'filtered_subjects_with_description.csv')

    # JSON 文件保存路径（保持不变）
    train_json_path = "./train_data_with_description.json"
    val_json_path = "./val_data_with_description.json"

    # 加载 CSV 文件（保持不变）
    csv_data = pd.read_csv(csv_path)
    csv_dict = csv_data.set_index("Subject ID")["Description"].to_dict()

    # 获取文件列表（保持不变）
    mri_files = sorted(os.listdir(mri_dir))
    av45_files = sorted(os.listdir(av45_dir))
    fdg_files = sorted(os.listdir(fdg_dir))



    # 使用统一的 get_subject_id 处理所有文件（保持不变）
    mri_dict = {get_subject_id(f): os.path.join(mri_dir, f) for f in mri_files}
    av45_dict = {get_subject_id(f): os.path.join(av45_dir, f) for f in av45_files}
    fdg_dict = {get_subject_id(f): os.path.join(fdg_dir, f) for f in fdg_files}

    # 匹配文件并加入描述信息和 Subject ID
    paired_data = []
    for patient_id, mri_file in mri_dict.items():
        if patient_id in av45_dict and patient_id in fdg_dict:
            # 检查描述信息是否存在
            description = csv_dict.get(patient_id, None)  # 从 csv_dict 中获取 Description 信息
            # 构建数据条目
            paired_data.append({
                "name": patient_id,  # 添加 name 字段
                "mri": os.path.join(mri_dir, mri_file),
                "av45": os.path.join(av45_dir, av45_dict[patient_id]),
                "fdg": os.path.join(fdg_dir, fdg_dict[patient_id]),
                "description": description  # 加入 Description 信息
            })
    if size_of_dataset:
        paired_data = paired_data[:size_of_dataset]  # 根据需要调整数据集大小
    print(f"Total matched pairs with description: {len(paired_data)}")
# 在 paired_data 中新增键 fdg_index 和 av45_index
    for idx, data in enumerate(paired_data):
        data["fdg_index"] = idx  # 将样本在 paired_data 中的索引作为 fdg_index
        data["av45_index"] = idx

    modal_information = [data["description"] for data in paired_data if data["description"] is not None]

    # 对描述信息进行 Tokenizer 编码
    text_inputs = processor(
        modal_information,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=256 # 根据模型输入限制选择合适的 max_length
    ).to(device)

    # 转化为 BiomedCLIP 嵌入
    with torch.no_grad():
        desc_text_features = bio_model.get_text_features(text_inputs['input_ids'])  # 使用模型方法提取嵌入
        print(f"Shape of features: {desc_text_features.shape}, Dtype: {desc_text_features.dtype}")

    # 划分训练集和验证集
    train_data, val_data = train_test_split(paired_data, test_size=int(len(paired_data)*0.1), random_state=42)

    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")

    if not os.path.exists(train_json_path):
        # 保存到 JSON 文件
        with open(train_json_path, "w") as f:
            json.dump(train_data, f, indent=4)
        with open(val_json_path, "w") as f:
            json.dump(val_data, f, indent=4)
        print(f"Saved train data to: {train_json_path}")
        print(f"Saved validation data to: {val_json_path}")
    else:
        print(f"JSON files already exist. Skipping save step.")

    with open(train_json_path, "r") as f:
        train_data = json.load(f)
    print(f"Loaded {len(train_data)} training samples.")
    print("First training sample:", train_data[0])

    # 验证验证集
    with open(val_json_path, "r") as f:
        val_data = json.load(f)
    print(f"Loaded {len(val_data)} validation samples.")
    print("First validation sample:", val_data[0])

    from monai.data import CacheDataset, DataLoader
    import json

    import monai.transforms as mt

    # 定义 transform 函数
    def fdg_index_transform(x):
        return torch.cat([desc_text_features[x].unsqueeze(0), fdg_feature_optimized], dim=0)

    def av45_index_transform(x):
        return torch.cat([desc_text_features[x].unsqueeze(0), av45_feature_optimized], dim=0)


    # 3. 加载处理好的数据
    train_json_path = "./train_data_with_description.json"
    val_json_path = "./val_data_with_description.json"

    # 加载 JSON 文件
    with open(train_json_path, "r") as f:
        train_data = json.load(f)

    with open(val_json_path, "r") as f:
        val_data = json.load(f)

    # 转换数据格式
    train_data = [
        {
            "name": item["name"],
            "mri": item["mri"],
            "av45": item["av45"],
            "fdg": item["fdg"],
            "description": item.get("description", {}),  # 确保 description 存在
            "fdg_index": item.get("fdg_index"),  # 保留 fdg_index
            "av45_index": item.get("av45_index"),  # 保留 av45_index
        }
        for item in train_data
    ]

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

    # 构建数据增强 pipeline
    # 定义训练集数据增强流程
    train_transforms = mt.Compose([
        mt.Lambdad(keys=["mri", "av45", "fdg"], func=mat_load),  # 加载 NIfTI 文件
        mt.EnsureChannelFirstd(keys=["mri", "av45", "fdg"],channel_dim='no_channel'),
        mt.Orientationd(keys=["mri", "av45", "fdg"], axcodes="LPI"),
        mt.CropForegroundd(keys=["mri", "av45", "fdg"], source_key="mri"),
        mt.HistogramNormalized(keys=["mri"]),
        mt.ResizeWithPadOrCropd(keys=["mri", "av45", "fdg"], spatial_size=[160, 192, 160]),
        mt.Spacingd(keys=["mri", "av45", "fdg"], pixdim=(1.0, 1.0, 1.0)),
        mt.NormalizeIntensityd(keys=["mri", "av45", "fdg"]),
        mt.ScaleIntensityd(keys=["mri", "av45", "fdg"]),
        mt.Lambdad(keys=["fdg_index"], func=fdg_index_transform),  # 添加 fdg_index 转换
        mt.Lambdad(keys=["av45_index"], func=av45_index_transform),  # 添加 av45_index 转换
    ])

    # 定义验证集数据增强流程（通常与训练集一致，但不含随机性增强）
    val_transforms = mt.Compose([
        mt.Lambdad(keys=["mri", "av45", "fdg"], func=mat_load),
        mt.EnsureChannelFirstd(keys=["mri", "av45", "fdg"],channel_dim='no_channel'),
        mt.Orientationd(keys=["mri", "av45", "fdg"], axcodes="LPI"),
        mt.CropForegroundd(keys=["mri", "av45", "fdg"], source_key="mri"),
        mt.HistogramNormalized(keys=["mri"]),
        mt.ResizeWithPadOrCropd(keys=["mri", "av45", "fdg"], spatial_size=[160, 192, 160]),
        mt.Spacingd(keys=["mri", "av45", "fdg"], pixdim=(1.0, 1.0, 1.0)),
        mt.NormalizeIntensityd(keys=["mri", "av45", "fdg"]),
        mt.ScaleIntensityd(keys=["mri", "av45", "fdg"]),
        mt.Lambdad(keys=["fdg_index"], func=fdg_index_transform),
        mt.Lambdad(keys=["av45_index"], func=av45_index_transform),
    ])

    # TODO: 保存为 HDF5 格式以加快加载速度
    # 保存 train 和 val
    # save_to_hdf5(train_data, train_transforms, os.path.join(h5_dir, "train.h5"))
    # save_to_hdf5(val_data, val_transforms, os.path.join(h5_dir, "val.h5"))

    # 创建 DataLoader
    # train_ds_h5 = HDF5Dataset(os.path.join(h5_dir, "train.h5"))
    # train_loader_h5 = DataLoader(train_ds_h5, batch_size=1, shuffle=True, num_workers=0)
    #
    # val_ds = HDF5Dataset(os.path.join(h5_dir, "val.h5"))
    # val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # 构建 CacheDataset
    # train_ds = CacheDataset(data=train_data, transform=train_transforms, num_workers=0)
    train_ds = PersistentDataset(data=train_data, transform=train_transforms, cache_dir=os.path.join(cache_dir, "train"))
    # # ⭐ 遍历所有样本以触发缓存生成
    # for i in range(len(train_ds)):
    #     _ = train_ds[i]
    # print("✅ 全部样本缓存生成完成！")
    # # benchmark_dataloader(train_ds, batch_size=1, num_workers_list=[0,2,4,8])
    #
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)    #shuffle=True

    # val_ds = CacheDataset(data=val_data, transform=val_transforms, num_workers=0, )
    val_ds = PersistentDataset(data=val_data, transform=val_transforms, cache_dir=os.path.join(cache_dir, "val"))
    # ⭐ 遍历所有样本以触发缓存生成
    # for i in range(len(val_ds)):
    #     _ = val_ds[i]
    # print("✅ 全部样本缓存生成完成！")
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from generative.networks.nets import DiffusionModelUNet
    from generative.networks.schedulers import DDPMScheduler,DDIMScheduler
    from generative.inferers import DiffusionInferer

    # 4. 加载网络
    if len(device_id) == 1:
        model= DiffusionModelUNet(
            spatial_dims=3,
            in_channels=1,#1,
            out_channels=1,
            num_channels=(32,64,64,128),
            attention_levels=(False,False,False,True),
            num_res_blocks=1,
            num_head_channels=(0,0,0, 128),
            with_conditioning=True,
            cross_attention_dim=512,
            use_flash_attention=True,
        )
        model.to(device)
    elif len(device_id) ==2:
        model= DistributedDiffusionModelUNet(
            spatial_dims=3,
            in_channels=1,#1,
            out_channels=1,
            num_channels=(32,64,64,128),
            attention_levels=(False,False,False,True),
            num_res_blocks=1,
            num_head_channels=(0,0,0, 128),
            with_conditioning=True,
            cross_attention_dim=512,
            use_flash_attention=True,
            device_ids=device_id
        )
    else:
        raise ValueError("Currently only support 1 or 2 GPU(s) for training.")
    optimizer= torch.optim.Adam(params=model.parameters(), lr=2.5e-5)
    start_epoch = 0

    # 加载上次训练的检查点（如果有的话）
    if last_epoch:
        # Load the checkpoint
        # checkpoint_dir = '/home/ssddata/liutuo/checkpoint/mri2pet _two trace_add clip_flow_noHistogramNormalized'
        checkpoint = torch.load(f'{checkpoint_dir}/first_part_{last_epoch}.pth')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch after the last saved one
        val_interval = 5  # Keep validation interval unchanged

    scheduler = DDPMScheduler(prediction_type="v_prediction", num_train_timesteps=1000,schedule="scaled_linear_beta",
                              beta_start=0.0005, beta_end=0.0195)
    scheduler.set_timesteps(num_inference_steps=1000)
    epoch_loss_list = []
    val_epoch_loss_list = []
    scaler = GradScaler()

    # 5. 训练网络
    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, data in progress_bar:
            images = data["mri"].to(device)
            seg_fdg = data["fdg"].to(device)  # this is the ground truth segmentation
            seg_av45 = data["av45"].to(device)
            fdg_index = data["fdg_index"].to(device)
            av45_index = data["av45_index"].to(device)

            optimizer.zero_grad(set_to_none=True)
            timesteps = torch.randint(0, 1000, (len(images),)).to(device)  # pick a random time step t

            with autocast(device_type='cuda', enabled=True):
                # Create time_embedding
                time_embedding = torch.randint(
                    0, 1000, (images.shape[0],), device=images.device
                ).long()

                if epoch >= 140:
                    time_embedding = torch.tensor([0], device=images.device, dtype=torch.long)

                # Create time
                t = time_embedding.float() / 1000

                # 检查 FDG 数据是否为二值化（只有 0 和 1）
                has_fdg = not torch.all(seg_fdg == 0)  # 如果不是二值化数据，则参与计算
                has_av45 = not torch.all(seg_av45 == 0)  # 如果不是二值化数据，则参与计算

                # 默认损失为 0
                loss_fdg = torch.tensor(0.0, device=device)
                loss_av45 = torch.tensor(0.0, device=device)

                # 计算 FDG 损失
                if has_fdg:  # 如果不是二值化数据，计算 FDG 损失
                    x_t_fdg = t * seg_fdg + (1 - t) * images
                    v_fdg_prediction = model(x=x_t_fdg, timesteps=time_embedding, context=fdg_index)
                    v_fdg = seg_fdg - images
                    loss_fdg = F.mse_loss(v_fdg.float(), v_fdg_prediction.float())

                # 计算 AV45 损失
                if has_av45:  # 如果不是二值化数据，计算 AV45 损失
                    x_t_av45 = t * seg_av45 + (1 - t) * images
                    v_av45_prediction = model(x=x_t_av45, timesteps=time_embedding, context=av45_index)
                    v_av45 = seg_av45 - images
                    loss_av45 = F.mse_loss(v_av45.float(), v_av45_prediction.float())

                # 加权总损失
                loss = alpha * loss_fdg + beta * loss_av45

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            progress_bar.set_postfix({
                "loss": epoch_loss / (step + 1),
                # "FDG": "✓" if has_fdg else "✗",
                # "AV45": "✓" if has_av45 else "✗"
            })

            torch.cuda.empty_cache()

        epoch_loss_list.append(epoch_loss / (step + 1))
        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_epoch_loss = 0

            for step, data_val in enumerate(val_loader):
                # 验证阶段的数据加载
                images = data_val["mri"].to(device)
                seg_fdg = data_val["fdg"].to(device)  # this is the ground truth segmentation
                seg_av45 = data_val["av45"].to(device)
                fdg_index = data_val["fdg_index"].to(device)
                av45_index = data_val["av45_index"].to(device)

                # 检查 FDG 和 AV45 数据是否为二值化（只有 0 和 1）
                has_fdg = not torch.all(seg_fdg == 0)  # 如果不是二值化数据，则参与计算
                has_av45 = not torch.all(seg_av45 == 0)  # 如果不是二值化数据，则参与计算

                x_t = images
                N_sample = 1
                N_sample_tensor = torch.tensor(N_sample, dtype=torch.float32)

                progress_bar = [(i / N_sample) for i in range(N_sample)]
                for t in progress_bar:  # go through the noising process
                    with autocast(device_type='cuda', enabled=False):
                        with torch.no_grad():
                            time_embedding = int(t * 1000)

                            # FDG 输出（仅当非二值化时计算）
                            if has_fdg:
                                v_fdg_output = model(x=x_t, timesteps=torch.Tensor((time_embedding,)).to(x_t.device), context=fdg_index)
                                x_fdg_t = x_t + (v_fdg_output / N_sample_tensor)
                                x_fdg_t = torch.clamp(x_fdg_t, min=clip_sample_min, max=clip_sample_max)
                            else:
                                x_fdg_t = None  # 跳过 FDG 计算

                            # AV45 输出（仅当非二值化时计算）
                            if has_av45:
                                v_av45_output = model(x=x_t, timesteps=torch.Tensor((time_embedding,)).to(x_t.device), context=av45_index)
                                x_av45_t = x_t + (v_av45_output / N_sample_tensor)
                                x_av45_t = torch.clamp(x_av45_t, min=clip_sample_min, max=clip_sample_max)
                            else:
                                x_av45_t = None  # 跳过 AV45 计算

                # 默认损失为 0
                val_fdg_loss = torch.tensor(0.0, device=device)
                val_av45_loss = torch.tensor(0.0, device=device)

                # 计算 FDG 损失（仅当非二值化时计算）
                if has_fdg and x_fdg_t is not None:
                    val_fdg_loss = F.mse_loss(x_fdg_t.float(), seg_fdg.float())

                # 计算 AV45 损失（仅当非二值化时计算）
                if has_av45 and x_av45_t is not None:
                    val_av45_loss = F.mse_loss(x_av45_t.float(), seg_av45.float())

                # 加权总损失
                val_loss = alpha * val_fdg_loss + beta * val_av45_loss

                val_epoch_loss += val_loss.item()
                torch.cuda.empty_cache()

            print("Epoch", epoch + 1, "Validation loss", val_epoch_loss / (step + 1))
            val_epoch_loss_list.append(val_epoch_loss / (step + 1))

            # Saving model parameters
            print('epoch:', epoch + 1)
            print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint, f'{checkpoint_dir}/first_part_{epoch + 1}.pth')
            print('Saved all parameters!\n')

            # 可视化和评估指标
            current_fdg_img = x_fdg_t.to('cpu') if x_fdg_t is not None else None
            current_av45_img = x_av45_t.to('cpu') if x_av45_t is not None else None
            labels_fdg = seg_fdg.to('cpu')
            labels_av45 = seg_av45.to('cpu')

            # 如果存在非二值化的 FDG 或 AV45 数据，则进行可视化和评估
            if current_fdg_img is not None or current_av45_img is not None:
                compare_3d([images, labels_fdg, current_fdg_img, labels_av45, current_av45_img])
                compare_3d_jet([current_fdg_img - labels_fdg, current_av45_img - labels_av45])

            from generative.metrics import SSIMMetric
            from monai.metrics import MAEMetric, MSEMetric, PSNRMetric

            ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=7)
            psnr_metric = PSNRMetric(1.0)

            # 计算 SSIM 和 PSNR（仅当非二值化时计算）
            if has_fdg and current_fdg_img is not None:
                ssim_fdg_value = ssim_metric(labels_fdg, current_fdg_img)
                psnr_fdg_value = psnr_metric(labels_fdg, current_fdg_img)
                print(f"FDG SSIM: {ssim_fdg_value.mean().item()}")
                print(f"FDG PSNR: {psnr_fdg_value.mean().item()}")

            if has_av45 and current_av45_img is not None:
                ssim_av45_value = ssim_metric(labels_av45, current_av45_img)
                psnr_av45_value = psnr_metric(labels_av45, current_av45_img)
                print(f"AV45 SSIM: {ssim_av45_value.mean().item()}")
                print(f"AV45 PSNR: {psnr_av45_value.mean().item()}")


if __name__ == "__main__":
    main()
