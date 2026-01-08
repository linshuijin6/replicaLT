#%%
from report_error import email_on_error


@email_on_error()
def main():
    from torch.distributed.pipeline.sync import Pipe
    from generative.networks.nets.diffusion_model_unet import DistributedDiffusionModelUNet
    import torch
    import os
    import logging
    logging.basicConfig(level=logging.INFO)
    import monai.transforms as mt
    import monai.data as md
    import scipy.io as sio
    import numpy as np
    from liutuo_utils import compare_3d_jet, compare_3d, donkey_noise_like
    import matplotlib.pyplot as plt
    import pandas as pd
    from dataset import save_to_hdf5, HDF5Dataset, benchmark_dataloader
    from monai.data import PersistentDataset
    cache_dir = '/mnt/nfsdata/nfsdata/linshuijin/ADNILT/tmp_cache'
    device_ids = [1,2]  # 指定要使用的 GPU 设备 ID 列表
    device_id = device_ids[0]
    devices_cuda = [torch.device(f'cuda:{device_ids[0]}'), torch.device(f'cuda:{device_ids[1]}')]

    # 设置 PyTorch 的默认 CUDA 设备
    torch.cuda.set_device(device_id)
    size_of_dataset = None  # 设置为 None 以使用完整数据集，或设置为所需的样本数量
    n_epochs = 200
    val_interval =10
    checkpoint_dir = './checkpoint'



    # 确认当前默认 CUDA 设备
    current_device = torch.cuda.current_device()
    print(f"Switched to CUDA device: {current_device}")
    #%%
    import torch
    from transformers import AutoProcessor, AutoModel
    import torch.nn.functional as F

    # 1. 加载 BiomedCLIP 模型和处理器
    local_model_path = "./BiomedCLIP"
    processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(local_model_path, trust_remote_code=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    # 加载优化后的描述
    fdg_text_optimized = (
        "FDG PET is a functional brain imaging technique that visualizes the dynamic changes in glucose metabolism, directly linked to neuronal energy demands and synaptic activity. It serves as a tool to assess functional connectivity and energy utilization across brain regions. Areas with decreased metabolic activity, such as those affected by neurodegenerative diseases, should exhibit reduced signal intensity. High-intensity metabolic hotspots in gray matter (e.g., the cerebral cortex and basal ganglia) are key markers of neuronal activity. "

    )

    av45_text_optimized = (
        "AV45 PET is a molecular imaging technique that highlights the static distribution of amyloid-beta plaques, a critical pathological marker of Alzheimer's disease. This imaging modality provides a spatial map of amyloid deposition in cortical regions (e.g., the temporal, parietal, and frontal lobes) and can distinguish amyloid-positive areas from amyloid-negative white matter regions. The primary focus is on identifying amyloid deposition patterns to assess disease progression and pathological burden."

    )

    # 提取特征向量
    texts_optimized = [fdg_text_optimized, av45_text_optimized]
    inputs_optimized = processor(text=texts_optimized, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        text_features_optimized = model.get_text_features(**inputs_optimized)

    # 计算相似度
    fdg_feature_optimized = text_features_optimized[0]
    fdg_feature_optimized = fdg_feature_optimized.unsqueeze(0)
    av45_feature_optimized = text_features_optimized[1]
    av45_feature_optimized = av45_feature_optimized.unsqueeze(0)

    # cosine_similarity_optimized = F.cosine_similarity(fdg_feature_optimized, av45_feature_optimized).item()

    # print(f"Cosine similarity between optimized FDG PET and AV45 PET features: {cosine_similarity_optimized:.4f}")

    #%%
    import os
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import json

    # 数据目录（保持不变）
    base_dir = "/mnt/nfsdata/nfsdata/linshuijin/ADNILT/ADNI1234_mri_fdg_av45_original"
    mri_dir = "/mnt/nfsdata/nfsdata/linshuijin/ADNILT/ADNI1234_mri_fdg_av45_original/mri_strip_registered"
    av45_dir = "/mnt/nfsdata/nfsdata/linshuijin/ADNILT/ADNI1234_mri_fdg_av45_original/PET1_AV45_strip_registered"
    fdg_dir = "/mnt/nfsdata/nfsdata/linshuijin/ADNILT/ADNI1234_mri_fdg_av45_original/PET2_FDG_strip_registered"
    csv_path = '/mnt/nfsdata/nfsdata/linshuijin/replicaLT/filtered_subjects_with_description.csv'

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

    def get_subject_id(filename):
        """从文件名中提取统一的 subject ID（前三个部分，如 '002_S_0295'）"""
        parts = filename.split('_')
        return f"{parts[0]}_{parts[1]}_{parts[2]}"

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

    import torch
    from transformers import AutoTokenizer, AutoModel

    # 示例：假设已经加载 BiomedCLIP 模型和处理器
    # 这里替换为您实际使用的 BiomedCLIP 模型和 tokenizer
    local_model_path = "/mnt/nfsdata/nfsdata/linshuijin/replicaLT/BiomedCLIP"
    processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(local_model_path, trust_remote_code=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()


    # 假设 paired_data 已经加载
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
        desc_text_features = model.get_text_features(text_inputs['input_ids'])  # 使用模型方法提取嵌入
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
    #%%
    # 验证训练集
    with open(train_json_path, "r") as f:
        train_data = json.load(f)
    print(f"Loaded {len(train_data)} training samples.")
    print("First training sample:", train_data[0])

    # 验证验证集
    with open(val_json_path, "r") as f:
        val_data = json.load(f)
    print(f"Loaded {len(val_data)} validation samples.")
    print("First validation sample:", val_data[0])


    #%%
    from monai.transforms import Compose, LoadImaged, Lambdad, EnsureChannelFirstd, Orientationd, CropForegroundd, \
        HistogramNormalized, ResizeWithPadOrCropd, Spacingd, NormalizeIntensityd, ScaleIntensityd
    from monai.data import CacheDataset, DataLoader
    import nibabel as nib
    import numpy as np
    import json

    import monai.transforms as mt

    # 定义 transform 函数
    def fdg_index_transform(x):
        return torch.cat([desc_text_features[x].unsqueeze(0), fdg_feature_optimized], dim=0)

    def av45_index_transform(x):
        return torch.cat([desc_text_features[x].unsqueeze(0), av45_feature_optimized], dim=0)


    # 自定义加载函数
    def mat_load(filepath):
        """
        使用 nibabel 加载 NIfTI 文件，并转换为 NumPy 数组。
        """
        return nib.load(filepath).get_fdata()

    # 加载 JSON 文件
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

    # 测试 DataLoader
    for batch_data in train_loader:
        print("MRI shape:", batch_data["mri"].shape)
        print("AV45 shape:", batch_data["av45"].shape)
        print("FDG shape:", batch_data["fdg"].shape)
        print("description:", train_data[0]["description"])  # 从原始数据访问描述信息
        break
    #%%
    for batch in val_loader:
        image_mri =batch["mri"].to(device)
        seg_fdg = batch["fdg"].to(device) # this is the ground truth segmentation
        seg_av45 = batch["av45"].to(device)
        fdg_index=batch["fdg_index"].to(device)
        av45_index=batch["av45_index"].to(device)
        names = batch["name"]  # Extract subject information


        for idx, name in enumerate(names):
             print(f"Subject name: {name}")
        compare_3d([image_mri, seg_fdg, seg_av45])
        break  # Uncomment to compare only the first batch , label_1,label_2

    #%%
    # for batch in train_loader:
    #     image_mri = batch["mri"].to(device)
    #     seg_fdg = batch["fdg"].to(device)  # this is the ground truth segmentation
    #     seg_av45 = batch["av45"].to(device)
    #     fdg_index = batch["fdg_index"].to(device)
    #     av45_index = batch["av45_index"].to(device)
    #     names = batch["name"]  # Extract subject information
    #
    #     # 遍历当前 batch 中的所有 subject
    #     for idx, name in enumerate(names):
    #         # 检查是否为目标 subject
    #         if name == "016_S_4952":
    #             print(f"Found target subject: {name}")
    #
    #             # 提取当前 subject 的数据
    #             target_image_mri = image_mri[idx].unsqueeze(0)  # 增加 batch 维度
    #             target_seg_fdg = seg_fdg[idx].unsqueeze(0)      # 增加 batch 维度
    #             target_seg_av45 = seg_av45[idx].unsqueeze(0)    # 增加 batch 维度
    #
    #             # 打印相关信息
    #             print(f"Subject name: {name}")
    #             print(f"Image MRI shape: {target_image_mri.shape}")
    #             print(f"Segmentation FDG shape: {target_seg_fdg.shape}")
    #             print(f"Segmentation AV45 shape: {target_seg_av45.shape}")
    #
    #             # 提取 AV45 的像素值
    #             av45_data = target_seg_av45.squeeze().cpu().numpy()  # 移除 batch 和 channel 维度，并转换为 NumPy 数组
    #
    #             # 计算 AV45 的像素值最大值和最小值
    #             av45_max = np.max(av45_data)
    #             av45_min = np.min(av45_data)
    #             print(f"AV45 pixel max value: {av45_max}")
    #             print(f"AV45 pixel min value: {av45_min}")
    #
    #             # 提取唯一值
    #             unique_values = np.unique(av45_data)
    #             print(f"Unique pixel values in AV45: {unique_values}")
    #
    #             # 检查是否只有 0 和 1
    #             if np.array_equal(unique_values, [0]):
    #                 print("The AV45 data contains only 0 .")
    #             else:
    #                 print("The AV45 data contains values other than 0 and 1.")
    #
    #             # 可视化或进一步处理
    #             compare_3d([target_image_mri, target_seg_fdg, target_seg_av45])
    #
    #             # 如果只需要处理目标 subject，可以在此处退出循环
    #             break
    #
    #     # 如果已经找到目标 subject，可以退出外层循环
    #     if name == "016_S_4952":
    #         break
    #%%
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from torch.amp import GradScaler, autocast  # 从 torch.amp 导入 GradScaler

    from generative.networks.nets import DiffusionModelUNet
    from generative.networks.schedulers import DDPMScheduler,DDIMScheduler
    from generative.inferers import DiffusionInferer
    # model= DistributedDiffusionModelUNet(
    #     spatial_dims=3,
    #     in_channels=1,#1,
    #     out_channels=1,
    #     num_channels=(32,64,64,128),
    #     attention_levels=(False,False,False,True),
    #     num_res_blocks=1,
    #     num_head_channels=(0,0,0, 128),
    #     with_conditioning=True,
    #     cross_attention_dim=512,
    #     use_flash_attention=True,
    #     device_ids=[id for id in device_ids]
    # )
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
    model = Pipe(model, devices=devices_cuda, chunks=1)
    # model.to(device)
    scheduler = DDPMScheduler(prediction_type="v_prediction", num_train_timesteps=1000,schedule="scaled_linear_beta",
                              beta_start=0.0005, beta_end=0.0195)
    scheduler.set_timesteps(num_inference_steps=1000)
    optimizer= torch.optim.Adam(params=model.parameters(), lr=2.5e-5)
    inferer = DiffusionInferer(scheduler)
    #%%
    import os
    import tempfile
    import time
    from liutuo_utils import donkey_noise_like
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from monai import transforms
    from monai.apps import DecathlonDataset
    from monai.config import print_config
    from monai.data import DataLoader
    from monai.utils import set_determinism
    from torch.cuda.amp import GradScaler, autocast
    from tqdm import tqdm
    from torch.amp import GradScaler, autocast  # 从 torch.amp 导入 GradScaler






    epoch_loss_list = []
    val_epoch_loss_list = []
    scaler = GradScaler()
    total_start = time.time()


    # 定义裁剪的最小值和最大值
    # 定义裁剪的最小值和最大值
    clip_sample_min = 0  # 设置合适的最小值
    clip_sample_max = 1   # 设置合适的最大值

    # 定义模态权重
    alpha = 1.0  # FDG 损失权重
    beta = 1.0   # AV45 损失权重

    for epoch in range(n_epochs):
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
    #%%
    import os
    import tempfile
    import time
    from liutuo_utils import donkey_noise_like
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from monai import transforms
    from monai.apps import DecathlonDataset
    from monai.config import print_config
    from monai.data import DataLoader
    from monai.utils import set_determinism
    from torch.cuda.amp import GradScaler, autocast
    from tqdm import tqdm
    from torch.amp import GradScaler, autocast  # 从 torch.amp 导入 GradScaler


    scaler = GradScaler()

    # Load the checkpoint
    checkpoint_dir = '/home/ssddata/liutuo/checkpoint/mri2pet _two trace_add clip_flow_noHistogramNormalized'
    checkpoint = torch.load(f'{checkpoint_dir}/first_part_154.pth')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch after the last saved one

    # Set new training parameters
    new_epochs =140  # Number of additional epochs to train
    total_epochs = start_epoch + new_epochs
    val_interval =5 # Keep validation interval unchanged
    epoch_loss_list = []  # Initialize epoch loss list
    val_epoch_loss_list = []  # Initialize validation epoch loss list

    # Training loop
    clip_sample_min = 0  # 设置合适的最小值
    clip_sample_max = 1   # 设置合适的最大值

    # 定义模态权重
    alpha = 1.0  # FDG 损失权重
    beta = 1.0   # AV45 损失权重

    for epoch in range(start_epoch, total_epochs):
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

                if epoch >= 120:
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
    #%%
    import os
    import torch
    from monai.metrics import PSNRMetric
    from generative.metrics import SSIMMetric
    from tqdm import tqdm
    import numpy as np
    import nibabel as nib  # 用于保存 NIfTI 格式的图像
    import monai.transforms as mt

    # 加载检查点
    checkpoint_dir = '/home/ssddata/liutuo/checkpoint/mri2pet _two trace_add clip_flow'
    checkpoint_path = '/home/ssddata/liutuo/checkpoint/mri2pet _two trace_add clip_flow_noHistogramNormalized/first_part_165.pth'
    checkpoint = torch.load(checkpoint_path)

    # 初始化模型和优化器
    model.load_state_dict(checkpoint['model'])
    model.eval()  # 设置为推理模式
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 定义评估指标
    ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=7)
    psnr_metric = PSNRMetric(max_val=1.0)

    # 推理过程
    clip_sample_min = 0  # 设置合适的最小值
    clip_sample_max = 1  # 设置合适的最大值

    # 创建保存目录
    output_dirs = {
        "mri": "/home/ssddata/liutuo/liutuo_data/data_augmentation_mri",
        "fdg": "/home/ssddata/liutuo/liutuo_data/data_augmentation_fdg",
        "av45": "/home/ssddata/liutuo/liutuo_data/data_augmentation_av45",
        "generated_fdg": "/home/ssddata/liutuo/liutuo_data/data_augmentation_fdg_generated",
        "generated_av45": "/home/ssddata/liutuo/liutuo_data/data_augmentation_av45_generated"
    }
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # 数据增强变换
    orientation_transform = mt.Orientationd(keys=["mri", "av45", "fdg"], axcodes="LPI")

    with torch.no_grad():
        for step, data_val in enumerate(tqdm(train_loader, desc="Inference")):
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
                    print(f"Saved generated FDG image to {output_path}")

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
                    print(f"Saved generated AV45 image to {output_path}")

            # 保存真实的 MRI、FDG 和 AV45 图像
            if images.shape == (1, 1, 160, 192, 160):
                images = images.squeeze()  # 去掉前两个维度 (1, 1)，得到 [160, 192, 160]
                images_np = images.cpu().numpy()  # 转换为 NumPy 数组
                output_path = os.path.join(output_dirs["mri"], f'real_mri_{names[0]}.nii.gz')  # 保存路径
                nib.save(nib.Nifti1Image(images_np, affine=np.eye(4)), output_path)  # 保存为 NIfTI 文件
                print(f"Saved real MRI image to {output_path}")

            if seg_fdg.shape == (1, 1, 160, 192, 160):
                seg_fdg = seg_fdg.squeeze()  # 去掉前两个维度 (1, 1)，得到 [160, 192, 160]
                seg_fdg_np = seg_fdg.cpu().numpy()  # 转换为 NumPy 数组
                output_path = os.path.join(output_dirs["fdg"], f'real_fdg_{names[0]}.nii.gz')  # 保存路径
                nib.save(nib.Nifti1Image(seg_fdg_np, affine=np.eye(4)), output_path)  # 保存为 NIfTI 文件
                print(f"Saved real FDG image to {output_path}")

            if seg_av45.shape == (1, 1, 160, 192, 160):
                seg_av45 = seg_av45.squeeze()  # 去掉前两个维度 (1, 1)，得到 [160, 192, 160]
                seg_av45_np = seg_av45.cpu().numpy()  # 转换为 NumPy 数组
                output_path = os.path.join(output_dirs["av45"], f'real_av45_{names[0]}.nii.gz')  # 保存路径
                nib.save(nib.Nifti1Image(seg_av45_np, affine=np.eye(4)), output_path)  # 保存为 NIfTI 文件
                print(f"Saved real AV45 image to {output_path}")

            # 计算 SSIM 和 PSNR（仅当非二值化时计算）
            if has_fdg and x_fdg_t is not None:
                ssim_fdg_value = ssim_metric(seg_fdg.unsqueeze(0).unsqueeze(0), x_fdg_t.unsqueeze(0).unsqueeze(0))
                psnr_fdg_value = psnr_metric(seg_fdg.unsqueeze(0).unsqueeze(0), x_fdg_t.unsqueeze(0).unsqueeze(0))
                print(f"[FDG] SSIM: {ssim_fdg_value.mean().item():.4f}, PSNR: {psnr_fdg_value.mean().item():.4f}")

            if has_av45 and x_av45_t is not None:
                ssim_av45_value = ssim_metric(seg_av45.unsqueeze(0).unsqueeze(0), x_av45_t.unsqueeze(0).unsqueeze(0))
                psnr_av45_value = psnr_metric(seg_av45.unsqueeze(0).unsqueeze(0), x_av45_t.unsqueeze(0).unsqueeze(0))
                print(f"[AV45] SSIM: {ssim_av45_value.mean().item():.4f}, PSNR: {psnr_av45_value.mean().item():.4f}")
    #%%
    import os
    import numpy as np
    import nibabel as nib

    # 定义文件夹路径
    folder_path = "/home/ssddata/liutuo/liutuo_data/New Folder"

    # 翻转函数（同时沿 x, y, z 轴翻转）
    def flip_nifti_along_xyz(file_path, output_dir):
        # 加载 NIfTI 文件
        img = nib.load(file_path)
        data = img.get_fdata()  # 获取图像数据（NumPy 数组）
        affine = img.affine  # 获取仿射矩阵

        # 同时沿 x 轴、y 轴和 z 轴翻转
        flipped_xyz = np.flip(data, axis=(0, 1, 2))  # 沿 (x, y, z) 轴同时翻转
        flipped_img_xyz = nib.Nifti1Image(flipped_xyz, affine)

        # 构造输出文件名
        output_path_xyz = os.path.join(output_dir, f"flipped_xyz_{os.path.basename(file_path)}")
        nib.save(flipped_img_xyz, output_path_xyz)

        print(f"Processed: {file_path}")
        print(f"Saved flipped image to: {output_path_xyz}")

    # 创建输出目录
    output_dir = os.path.join(folder_path, "flipped_images")
    os.makedirs(output_dir, exist_ok=True)

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".nii.gz"):  # 只处理 .nii.gz 文件
            file_path = os.path.join(folder_path, filename)
            flip_nifti_along_xyz(file_path, output_dir)
    #%%
    import os
    import numpy as np
    import nibabel as nib

    # 定义输入文件夹路径
    input_folder = "/home/ssddata/liutuo/liutuo_data/data_augmentation_av45"

    # 定义输出文件夹路径
    output_folder = "/home/ssddata/liutuo/liutuo_data/data_augmentation_av45_flip"
    os.makedirs(output_folder, exist_ok=True)  # 创建输出目录（如果不存在）

    # 翻转函数（同时沿 x, y, z 轴翻转）
    def flip_nifti_along_xyz(file_path, output_dir):
        # 加载 NIfTI 文件
        img = nib.load(file_path)
        data = img.get_fdata()  # 获取图像数据（NumPy 数组）
        affine = img.affine  # 获取仿射矩阵

        # 同时沿 x 轴、y 轴和 z 轴翻转
        flipped_xyz = np.flip(data, axis=(0, 1, 2))  # 沿 (x, y, z) 轴同时翻转
        flipped_img_xyz = nib.Nifti1Image(flipped_xyz, affine)

        # 构造输出文件名
        output_path_xyz = os.path.join(output_dir, f"{os.path.basename(file_path)}")
        nib.save(flipped_img_xyz, output_path_xyz)

        print(f"Processed: {file_path}")
        print(f"Saved flipped image to: {output_path_xyz}")

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".nii.gz"):  # 只处理 .nii.gz 文件
            file_path = os.path.join(input_folder, filename)
            flip_nifti_along_xyz(file_path, output_folder)

if __name__ == "__main__":
    main()
