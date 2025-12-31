# created 14, saved 
""" abstract of code """
"""
This script processes MRI, AV45, and FDG imaging data, matches them with subject descriptions from a CSV file, 
and prepares the data for use with the BiomedCLIP model. It includes steps for data pairing, embedding generation, 
and splitting into training and validation datasets.

Key functionalities:
1. Load and process imaging data and descriptions.
2. Match data based on subject IDs.
3. Generate embeddings using the BiomedCLIP model.
4. Split data into training and validation sets.
5. Save the processed data to JSON files.
"""

""" main code """
import os
import pandas as pd
from transformers import AutoProcessor, AutoModel
import json
from sklearn.model_selection import train_test_split

def main():
    # 数据目录（保持不变）
    base_dir = "/home/ssddata/liutuo/liutuo_data/ADNI1234_mri_fdg_av45_original"
    mri_dir = "/home/ssddata/liutuo/liutuo_data/ADNI1234_mri_fdg_av45_original/mri_strip_registered"
    av45_dir = "/home/ssddata/liutuo/liutuo_data/ADNI1234_mri_fdg_av45_original/PET1_AV45_strip_registered"
    fdg_dir = "/home/ssddata/liutuo/liutuo_data/ADNI1234_mri_fdg_av45_original/PET2_FDG_strip_registered"
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
        max_length=256  # 根据模型输入限制选择合适的 max_length
    ).to(device)

    # 转化为 BiomedCLIP 嵌入
    with torch.no_grad():
        desc_text_features = model.get_text_features(text_inputs['input_ids'])  # 使用模型方法提取嵌入
        print(f"Shape of features: {desc_text_features.shape}, Dtype: {desc_text_features.dtype}")

    # 划分训练集和验证集
    train_data, val_data = train_test_split(paired_data, test_size=int(len(paired_data) * 0.1), random_state=42)

    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")

    # 保存到 JSON 文件
    with open(train_json_path, "w") as f:
        json.dump(train_data, f, indent=4)

    with open(val_json_path, "w") as f:
        json.dump(val_data, f, indent=4)

    print(f"Saved train data to: {train_json_path}")
    print(f"Saved validation data to: {val_json_path}")

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

    return True


if __name__ == '__main__':
    main()
