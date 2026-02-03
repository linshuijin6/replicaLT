# created 14, saved 
""" abstract of code """
""" main code """
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 避免多进程时的 tokenizer 警告

def main():
    from liutuo_utils import compare_3d_jet
    from generative.metrics import SSIMMetric
    from monai.metrics import MAEMetric, MSEMetric, PSNRMetric
    from generative.networks.nets import DiffusionModelUNet
    from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
    from generative.inferers import DiffusionInferer
    from sklearn.model_selection import train_test_split
    from liutuo_utils import compare_3d
    import os
    import torch
    from monai.transforms import Compose, LoadImaged, Lambdad, EnsureChannelFirstd, Orientationd, CropForegroundd, \
        HistogramNormalized, ResizeWithPadOrCropd, Spacingd, NormalizeIntensityd, ScaleIntensityd
    from monai.data import CacheDataset, DataLoader
    import nibabel as nib
    import numpy as np
    import json
    from transformers import AutoProcessor, AutoModel
    import monai.transforms as mt
    import torch.nn.functional as F
    import pandas as pd
    from tqdm import tqdm
    from torch.amp import GradScaler, autocast  # 从 torch.amp 导入 GradScaler
    import time

    size_of_dataset = None  # 设置为 None 以使用完整数据集，或设置为所需的样本数量
    n_epochs = 200
    val_interval = 10
    epoch_loss_list = []
    val_epoch_loss_list = []
    scaler = GradScaler()
    total_start = time.time()
    checkpoint_dir = '/home/ssddata/liutuo/checkpoint/mri2pet _two trace_add clip_flow_noHistogramNormalized'

    # 定义裁剪的最小值和最大值
    # 定义裁剪的最小值和最大值
    clip_sample_min = 0  # 设置合适的最小值
    clip_sample_max = 1  # 设置合适的最大值

    # 定义模态权重
    alpha = 1.0  # FDG 损失权重
    beta = 1.0  # AV45 损失权重

    # 1. 加载 BiomedCLIP 模型和处理器
    local_model_path = "./BiomedCLIP"
    processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(local_model_path, trust_remote_code=True)

    gpu_id = '5'

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    # 加载优化后的描述
    # fdg_text_optimized = (
    #     "FDG PET is a functional brain imaging technique that visualizes the dynamic changes in glucose metabolism, directly linked to neuronal energy demands and synaptic activity. It serves as a tool to assess functional connectivity and energy utilization across brain regions. Areas with decreased metabolic activity, such as those affected by neurodegenerative diseases, should exhibit reduced signal intensity. High-intensity metabolic hotspots in gray matter (e.g., the cerebral cortex and basal ganglia) are key markers of neuronal activity. "

    # )

    av45_text_optimized = (
        "AV45 PET is a molecular imaging technique that highlights the static distribution of amyloid-beta plaques, a critical pathological marker of Alzheimer's disease. This imaging modality provides a spatial map of amyloid deposition in cortical regions (e.g., the temporal, parietal, and frontal lobes) and can distinguish amyloid-positive areas from amyloid-negative white matter regions. The primary focus is on identifying amyloid deposition patterns to assess disease progression and pathological burden."

    )
    tau_text_optimized = (
        "TAU PET is a molecular neuroimaging technique that visualizes the spatial distribution of "
            "aggregated tau protein, which reflects the presence of neurofibrillary tangles associated "
            "with neurodegeneration. Tau PET highlights region-specific tau accumulation, particularly "
            "in medial temporal, parietal, and association cortices, providing a topographical map of "
            "tau pathology that correlates with disease stage, cognitive decline, and neuronal dysfunction.")

    # 提取特征向量
    texts_optimized = [tau_text_optimized, av45_text_optimized]
    inputs_optimized = processor(text=texts_optimized, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        text_features_optimized = model.get_text_features(**inputs_optimized)

    # 计算相似度
    fdg_feature_optimized = text_features_optimized[0]
    fdg_feature_optimized = fdg_feature_optimized.unsqueeze(0)
    av45_feature_optimized = text_features_optimized[1]
    av45_feature_optimized = av45_feature_optimized.unsqueeze(0)
    #
    # cosine_similarity_optimized = F.cosine_similarity(fdg_feature_optimized, av45_feature_optimized).item()
    #
    # print(f"Cosine similarity between optimized FDG PET and AV45 PET features: {cosine_similarity_optimized:.4f}")
    # 这里替换为您实际使用的 BiomedCLIP 模型和 tokenizer

    # 数据目录（保持不变）
    base_dir = "/mnt/nfsdata/nfsdata/linshuijin/ADNILT/ADNI1234_mri_fdg_av45_original"
    mri_dir = "/mnt/nfsdata/nfsdata/linshuijin/ADNILT/ADNI1234_mri_fdg_av45_original/mri_strip_registered"
    av45_dir = "/mnt/nfsdata/nfsdata/linshuijin/ADNILT/ADNI1234_mri_fdg_av45_original/PET1_AV45_strip_registered"
    fdg_dir = "/mnt/nfsdata/nfsdata/linshuijin/ADNILT/ADNI1234_mri_fdg_av45_original/PET2_FDG_strip_registered"
    csv_path = '/mnt/nfsdata/nfsdata/linshuijin/replicaLT/filtered_subjects_with_description.csv'

    # JSON 文件保存路径（保持不变）
    train_json_path = "./train_data_with_description.json"
    val_json_path = "./val_data_with_description.json"

    # # 加载 CSV 文件（保持不变）
    # csv_data = pd.read_csv(csv_path)
    # csv_dict = csv_data.set_index("Subject ID")["Description"].to_dict()

    # # 获取文件列表（保持不变）
    # mri_files = sorted(os.listdir(mri_dir))
    # av45_files = sorted(os.listdir(av45_dir))
    # fdg_files = sorted(os.listdir(fdg_dir))

    # def get_subject_id(filename):
    #     """从文件名中提取统一的 subject ID（前三个部分，如 '002_S_0295'）"""
    #     parts = filename.split('_')
    #     return f"{parts[0]}_{parts[1]}_{parts[2]}"

    # # 使用统一的 get_subject_id 处理所有文件（保持不变）
    # mri_dict = {get_subject_id(f): os.path.join(mri_dir, f) for f in mri_files}
    # av45_dict = {get_subject_id(f): os.path.join(av45_dir, f) for f in av45_files}
    # fdg_dict = {get_subject_id(f): os.path.join(fdg_dir, f) for f in fdg_files}

    # # 匹配文件并加入描述信息和 Subject ID
    # paired_data = []
    # for patient_id, mri_file in mri_dict.items():
    #     if patient_id in av45_dict and patient_id in fdg_dict:
    #         # 检查描述信息是否存在
    #         description = csv_dict.get(patient_id, None)  # 从 csv_dict 中获取 Description 信息
    #         # 构建数据条目
    #         paired_data.append({
    #             "name": patient_id,  # 添加 name 字段
    #             "mri": os.path.join(mri_dir, mri_file),
    #             "av45": os.path.join(av45_dir, av45_dict[patient_id]),
    #             "fdg": os.path.join(fdg_dir, fdg_dict[patient_id]),
    #             "description": description  # 加入 Description 信息
    #         })

    # if size_of_dataset:
    #     paired_data = paired_data[:size_of_dataset]  # 根据需要调整数据集大小

    # print(f"Total matched pairs with description: {len(paired_data)}")
    # # 在 paired_data 中新增键 fdg_index 和 av45_index
    # for idx, data in enumerate(paired_data):
    #     data["fdg_index"] = idx  # 将样本在 paired_data 中的索引作为 fdg_index
    #     data["av45_index"] = idx
    with open(train_json_path, "r") as f:
        train_data = json.load(f)
    print(f"Loaded {len(train_data)} training samples.")
    print("First training sample:", train_data[0])

    # 处理 null 字段的函数：生成全零 NIfTI 文件并返回路径
    def get_zero_nifti_path(name, modality, reference_path=None):
        """
        生成全零 NIfTI 文件路径，如果文件不存在则创建。
        使用参考文件的 affine 和形状来确保兼容性。
        
        Args:
            name: 受试者名称
            modality: 模态类型 ('av45', 'fdg', 'tau')
            reference_path: 参考 NIfTI 文件路径（用于获取正确的 affine 和形状）
        
        Returns:
            零数据文件的路径
        """
        modality_upper = modality.upper()
        # 使用单独的路径存放零数据文件，避免权限问题
        base_dir = f"/mnt/nfsdata/nfsdata/linshuijin/zero/{modality_upper}"
        zero_filename = f"{name}_{modality}_zero.nii.gz"
        zero_filepath = os.path.join(base_dir, zero_filename)
        
        # 如果文件不存在，创建全零数据
        if not os.path.exists(zero_filepath):
            os.makedirs(base_dir, exist_ok=True)
            
            # 使用参考文件获取正确的 affine 和形状
            if reference_path and os.path.exists(reference_path):
                ref_img = nib.load(reference_path)
                reference_shape = ref_img.shape[:3]  # 只取前3维
                affine = ref_img.affine
            else:
                # 默认形状和 affine
                reference_shape = (160, 192, 160)
                affine = np.eye(4)
            
            zero_data = np.zeros(reference_shape, dtype=np.float32)
            zero_img = nib.Nifti1Image(zero_data, affine=affine)
            nib.save(zero_img, zero_filepath)
            print(f"Created zero NIfTI file: {zero_filepath}")
        
        return zero_filepath

    def fill_null_fields(data_list, modalities=['av45', 'fdg', 'tau']):
        """
        填充数据列表中的 null 字段，用全零 NIfTI 文件路径替换。
        使用 MRI 文件作为参考来生成零数据。
        
        Args:
            data_list: 数据列表
            modalities: 需要检查的模态列表
        
        Returns:
            处理后的数据列表
        """
        for item in data_list:
            name = item.get("name", "unknown")
            reference_path = item.get("mri")  # 使用 MRI 作为参考
            for modality in modalities:
                if modality in item and item[modality] is None:
                    item[modality] = get_zero_nifti_path(name, modality, reference_path)
        return data_list

    # 填充训练集中的 null 字段
    train_data = fill_null_fields(train_data, modalities=['av45', 'fdg', 'tau'])

    # 验证验证集
    with open(val_json_path, "r") as f:
        val_data = json.load(f)
    print(f"Loaded {len(val_data)} validation samples.")
    print("First validation sample:", val_data[0])

    # 填充验证集中的 null 字段
    val_data = fill_null_fields(val_data, modalities=['av45', 'fdg', 'tau'])

        # 转换数据格式
    train_data = [
        {
            "name": item["name"],
            "mri": item["mri"],
            "av45": item["av45"],
            "fdg": item["fdg"],
            "description": item.get("description") or "",  # 确保 description 存在且不为 None
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
            "description": item.get("description") or "",  # 确保 description 存在且不为 None
            "fdg_index": item.get("fdg_index"),  # 保留 fdg_index
            "av45_index": item.get("av45_index"),  # 保留 av45_index
        }
        for item in val_data
    ]
    # import torch
    # from transformers import AutoTokenizer, AutoModel

    # # 示例：假设已经加载 BiomedCLIP 模型和处理器
    # # 这里替换为您实际使用的 BiomedCLIP 模型和 tokenizer
    # local_model_path = "/mnt/nfsdata/nfsdata/linshuijin/replicaLT/BiomedCLIP"
    # processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
    # model = AutoModel.from_pretrained(local_model_path, trust_remote_code=True)
    # model.to(device)
    # model.eval()

    # 假设 paired_data 已经加载
    paired_data = train_data + val_data
    modal_information = [data["description"] for data in paired_data if data["description"] is not None]

    # 对描述信息进行 Tokenizer 编码
    text_inputs = processor(
        modal_information,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=256  # 根据模型输入限制选择合适的 max_length
    ).to(device)

    # 转化为 BiomedCLIP 嵌入
    with torch.no_grad():
        desc_text_features = model.get_text_features(text_inputs['input_ids'])  # 使用模型方法提取嵌入
        print(f"Shape of features: {desc_text_features.shape}, Dtype: {desc_text_features.dtype}")
    
    # 将特征转移到 CPU，确保缓存可以跨 GPU 使用
    desc_text_features_cpu = desc_text_features.cpu()
    fdg_feature_optimized_cpu = fdg_feature_optimized.cpu()
    av45_feature_optimized_cpu = av45_feature_optimized.cpu()

    # # 划分训练集和验证集
    # train_data, val_data = train_test_split(paired_data, test_size=int(len(paired_data) * 0.1), random_state=42)

    # print(f"Training set size: {len(train_data)}")
    # print(f"Validation set size: {len(val_data)}")

    # 保存到 JSON 文件
    # with open(train_json_path, "w") as f:
    #     json.dump(train_data, f, indent=4)

    # with open(val_json_path, "w") as f:
    #     json.dump(val_data, f, indent=4)

    # print(f"Saved train data to: {train_json_path}")
    # print(f"Saved validation data to: {val_json_path}")


    # 定义 transform 函数（使用 CPU 张量，确保跨 GPU 兼容）
    def fdg_index_transform_t(x):
        return torch.cat([desc_text_features_cpu[x].unsqueeze(0), fdg_feature_optimized_cpu], dim=0)

    def av45_index_transform_t(x):
        return torch.cat([desc_text_features_cpu[x].unsqueeze(0), av45_feature_optimized_cpu], dim=0)

    def fdg_index_transform_v(x):
        return torch.cat([desc_text_features_cpu[x+len(train_data)].unsqueeze(0), fdg_feature_optimized_cpu], dim=0)

    def av45_index_transform_v(x):
        return torch.cat([desc_text_features_cpu[x+len(train_data)].unsqueeze(0), av45_feature_optimized_cpu], dim=0)
    # 自定义加载函数
    def mat_load(filepath):
        """
        使用 nibabel 加载 NIfTI 文件，并转换为 NumPy 数组。
        如果数据是 4D，取第一个时间点转换为 3D。
        """
        data = nib.load(filepath).get_fdata()
        # 处理 4D 数据：取第一个时间点（或对所有时间点取平均）
        if len(data.shape) == 4:
            data = data[:, :, :, 0]  # 取第一个时间点
        return data

    # # 加载 JSON 文件
    # train_json_path = "./train_data_with_description.json"
    # val_json_path = "./val_data_with_description.json"

    # # 加载 JSON 文件
    # with open(train_json_path, "r") as f:
    #     train_data = json.load(f)

    # with open(val_json_path, "r") as f:
    #     val_data = json.load(f)



    # 构建数据增强 pipeline
    # 定义训练集数据增强流程
    train_transforms = mt.Compose([
        mt.Lambdad(keys=["mri", "av45", "fdg"], func=mat_load),  # 加载 NIfTI 文件
        mt.EnsureChannelFirstd(keys=["mri", "av45", "fdg"], channel_dim='no_channel'),
        mt.Orientationd(keys=["mri", "av45", "fdg"], axcodes="LPI"),
        mt.CropForegroundd(keys=["mri", "av45", "fdg"], source_key="mri"),
        mt.HistogramNormalized(keys=["mri"]),
        mt.ResizeWithPadOrCropd(keys=["mri", "av45", "fdg"], spatial_size=[160, 192, 160]),
        mt.Spacingd(keys=["mri", "av45", "fdg"], pixdim=(1.0, 1.0, 1.0)),
        mt.NormalizeIntensityd(keys=["mri", "av45", "fdg"]),
        mt.ScaleIntensityd(keys=["mri", "av45", "fdg"]),
        mt.Lambdad(keys=["fdg_index"], func=fdg_index_transform_t),  # 添加 fdg_index 转换
        mt.Lambdad(keys=["av45_index"], func=av45_index_transform_t),  # 添加 av45_index 转换
    ])

    # 定义验证集数据增强流程（通常与训练集一致，但不含随机性增强）
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
        mt.Lambdad(keys=["fdg_index"], func=fdg_index_transform_v),
        mt.Lambdad(keys=["av45_index"], func=av45_index_transform_v),
    ])

    # 构建 CacheDataset（使用 PersistentDataset 实现持久化缓存）
    from monai.data import PersistentDataset
    
    # 定义缓存目录
    cache_dir = "/mnt/nfsdata/nfsdata/linshuijin/cache"
    train_cache_dir = os.path.join(cache_dir, "train")
    val_cache_dir = os.path.join(cache_dir, "val")
    os.makedirs(train_cache_dir, exist_ok=True)
    os.makedirs(val_cache_dir, exist_ok=True)
    
    print(f"Using persistent cache directory: {cache_dir}")
    print("First run will process and cache data. Subsequent runs will load from cache.")
    
    # 使用 num_workers 加速缓存生成
    # 注意：num_workers > 0 时使用多进程，首次运行时可加速缓存生成
    # 缓存文件使用 CPU 张量存储，可在任意 GPU 上加载使用
    num_workers_for_cache = 4  # 根据 CPU 核心数调整，建议 4-8
    
    train_ds = PersistentDataset(data=train_data, transform=train_transforms, cache_dir=train_cache_dir)
    train_loader = DataLoader(
        train_ds, 
        batch_size=1, 
        shuffle=True, 
        num_workers=num_workers_for_cache,
        pin_memory=True,  # 加速 CPU 到 GPU 的数据传输
        persistent_workers=True if num_workers_for_cache > 0 else False  # 保持 worker 进程存活
    )

    val_ds = PersistentDataset(data=val_data, transform=val_transforms, cache_dir=val_cache_dir)
    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False, 
        num_workers=num_workers_for_cache,
        pin_memory=True,
        persistent_workers=True if num_workers_for_cache > 0 else False
    )

    # 测试 DataLoader
    # for batch_data in train_loader:
    #     print("MRI shape:", batch_data["mri"].shape)
    #     print("AV45 shape:", batch_data["av45"].shape)
    #     print("FDG shape:", batch_data["fdg"].shape)
    #     print("description:", train_data[0]["description"])  # 从原始数据访问描述信息
    #     break

    # for batch in val_loader:
    #     image_mri = batch["mri"].to(device)
    #     seg_fdg = batch["fdg"].to(device)  # this is the ground truth segmentation
    #     seg_av45 = batch["av45"].to(device)
    #     fdg_index = batch["fdg_index"].to(device)
    #     av45_index = batch["av45_index"].to(device)
    #     names = batch["name"]  # Extract subject information

    #     for idx, name in enumerate(names):
    #         print(f"Subject name: {name}")
    #     # compare_3d([image_mri, seg_fdg, seg_av45])
    #     break  # Uncomment to compare only the first batch , label_1,label_2

    # 定义模型、优化器、调度器等
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

    model.to(device)
    scheduler = DDPMScheduler(prediction_type="v_prediction", num_train_timesteps=1000, schedule="scaled_linear_beta",
                              beta_start=0.0005, beta_end=0.0195)
    scheduler.set_timesteps(num_inference_steps=1000)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=2.5e-5)
    inferer = DiffusionInferer(scheduler)

    # 训练循环
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

                # 如果两个模态都没有有效数据，跳过这个样本
                if not has_fdg and not has_av45:
                    continue

                # 默认损失为 0
                loss_fdg = torch.tensor(0.0, device=device, requires_grad=False)
                loss_av45 = torch.tensor(0.0, device=device, requires_grad=False)

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
                                v_fdg_output = model(x=x_t, timesteps=torch.Tensor((time_embedding,)).to(x_t.device),
                                                     context=fdg_index)
                                x_fdg_t = x_t + (v_fdg_output / N_sample_tensor)
                                x_fdg_t = torch.clamp(x_fdg_t, min=clip_sample_min, max=clip_sample_max)
                            else:
                                x_fdg_t = None  # 跳过 FDG 计算

                            # AV45 输出（仅当非二值化时计算）
                            if has_av45:
                                v_av45_output = model(x=x_t, timesteps=torch.Tensor((time_embedding,)).to(x_t.device),
                                                      context=av45_index)
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


    return True


if __name__ == '__main__':
    main()
