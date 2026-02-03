# created 14, saved 
""" abstract of code """
"""
This script processes MRI, FDG, AV45, and TAU (AV1451) imaging data from pairs_withPlasma.csv,
matches them with subject descriptions from pairs_with_demog.csv, 
and prepares the data for use with the BiomedCLIP model.

Key functionalities:
1. Load pairs_withPlasma.csv to get image IDs for each modality.
2. Locate nii.gz files based on image IDs from Coregistration directory.
3. For missing modalities, create zero-filled nii.gz files matching MRI dimensions.
4. Match with pairs_with_demog.csv to get demographic descriptions.
5. Generate embeddings using the BiomedCLIP model.
6. Split data into training and validation sets.
7. Save the processed data to JSON files.
"""

""" main code """
import os
import glob
import pandas as pd
import numpy as np
import nibabel as nib
from transformers import AutoProcessor, AutoModel
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def find_nii_file(base_dir, subdir, subject_id, image_id):
    """
    根据subject_id和image_id查找nii.gz文件
    文件名格式: <subject_id>*<image_id>.nii.gz
    """
    if pd.isna(image_id) or image_id == '':
        return None
    
    search_pattern = os.path.join(base_dir, subdir, f"{subject_id}*{image_id}.nii.gz")
    matches = glob.glob(search_pattern)
    
    if matches:
        return matches[0]
    
    # 如果没找到，尝试更宽松的搜索模式
    search_pattern = os.path.join(base_dir, subdir, f"*{image_id}.nii.gz")
    matches = glob.glob(search_pattern)
    
    if matches:
        return matches[0]
    
    return None


def create_zero_nii(reference_nii_path, output_path, overwrite=False):
    """
    根据参考的nii.gz文件创建一个相同格式但全0的nii.gz文件
    
    Args:
        reference_nii_path: 参考nii.gz文件路径
        output_path: 输出文件路径
        overwrite: 是否覆盖已存在的文件
    """
    if os.path.exists(output_path) and not overwrite:
        return output_path
    
    try:
        # 读取参考文件
        ref_img = nib.load(reference_nii_path)
        ref_data = ref_img.get_fdata()
        
        # 创建全0数组
        zero_data = np.zeros_like(ref_data)
        
        # 创建新的nii图像
        zero_img = nib.Nifti1Image(zero_data.astype(np.float32), ref_img.affine, ref_img.header)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存
        nib.save(zero_img, output_path)
        return output_path
    except Exception as e:
        print(f"Error creating zero nii file: {e}")
        return None


def normalize_examdate(date_str):
    """标准化日期格式用于匹配"""
    if pd.isna(date_str) or date_str == '':
        return None
    # 处理不同的日期格式
    date_str = str(date_str).strip()
    # 将 '/' 替换为 '-' 进行统一
    date_str = date_str.replace('/', '-')
    
    # 解析日期并重新格式化为 YYYY-MM-DD (补零格式)
    try:
        # 尝试解析 YYYY-M-D 或 YYYY-MM-DD 格式
        parts = date_str.split('-')
        if len(parts) == 3:
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            # 返回统一格式 YYYY-MM-DD
            return f"{year:04d}-{month:02d}-{day:02d}"
    except (ValueError, IndexError):
        pass
    
    return date_str


def main():
    # ==================== 配置路径 ====================
    # 配对CSV路径
    pairs_csv_path = '/home/ssddata/linshuijin/replicaLT/adapter_finetune/gen_csv/pairs_180d_dx_plasma_90d_matched.csv'
    demog_csv_path = '/home/ssddata/linshuijin/replicaLT/adapter_finetune/gen_csv/pairs_180d_dx_with_demog.csv'
    
    # 影像文件根目录
    coregistration_base = '/mnt/nfsdata/nfsdata/lsj.14/ADNI_CSF'
    
    # 各模态子目录
    mri_subdir = 'MRI'
    fdg_subdir = 'PET_MNI/FDG'
    av45_subdir = 'PET_MNI/AV45'
    tau_subdir = 'PET_MNI/TAU'
    
    # 零填充文件保存目录
    zero_files_dir = '/mnt/nfsdata/nfsdata/linshuijin/zero'
    
    # JSON 文件保存路径
    train_json_path = "./train_data_with_description.json"
    val_json_path = "./val_data_with_description.json"

    # ==================== 加载CSV文件 ====================
    print("Loading CSV files...")
    pairs_df = pd.read_csv(pairs_csv_path)
    demog_df = pd.read_csv(demog_csv_path)
    
    print(f"Pairs CSV: {len(pairs_df)} rows")
    print(f"Demog CSV: {len(demog_df)} rows")
    
    # 创建demog字典，key为(subject_id, examdate)
    # 先标准化demog中的日期格式
    demog_df['examdate_norm'] = demog_df['examdate'].apply(normalize_examdate)
    demog_dict = {}
    for _, row in demog_df.iterrows():
        key = (row['subject_id'], row['examdate_norm'])
        demog_dict[key] = {
            'description': row['description'],
            'diagnosis': row.get('diagnosis', None),
            'old_descr': row.get('old_descr', None)
        }
    
    # 定义缺失值预设值
    MISSING_DIAGNOSIS = "UNKNOWN"
    MISSING_PLASMA = -999.0  # 用于标识缺失的plasma值
    
    # ==================== 处理每一行配对数据 ====================
    paired_data = []
    missing_mri = 0
    missing_descriptions = 0
    zero_files_created = {'fdg': 0, 'av45': 0, 'tau': 0}
    
    print("\nProcessing pairs...")
    for idx, row in tqdm(pairs_df.iterrows(), total=len(pairs_df)):
        subject_id = row['PTID']
        examdate = row['EXAMDATE']
        examdate_norm = normalize_examdate(examdate)
        
        id_mri = row.get('id_mri', None)
        id_fdg = row.get('id_fdg', None)
        id_av45 = row.get('id_av45', None)
        id_av1451 = row.get('id_av1451', None)
        
        # 1. 首先查找MRI文件（必须存在）
        mri_path = find_nii_file(coregistration_base, mri_subdir, subject_id, id_mri)
        if mri_path is None:
            missing_mri += 1
            continue
        
        # 2. 获取description和diagnosis
        demog_info = demog_dict.get((subject_id, examdate_norm), None)
        if demog_info is None:
            # 尝试用原始格式匹配
            demog_info = demog_dict.get((subject_id, examdate), None)
        
        if demog_info is not None:
            description = demog_info['description']
            diagnosis = demog_info['diagnosis'] if demog_info['diagnosis'] else MISSING_DIAGNOSIS
            old_descr = demog_info['old_descr']
        else:
            missing_descriptions += 1
            description = None  # 允许description为空
            diagnosis = MISSING_DIAGNOSIS
            old_descr = None
        
        # 3. 获取plasma相关字段（从pairs_df当前行直接获取）
        def get_plasma_value(value):
            """获取plasma值，缺失则返回预设值"""
            if pd.isna(value) or value == '':
                return MISSING_PLASMA
            return float(value)
        
        plasma_data = {
            'AB42_AB40': get_plasma_value(row.get('AB42_AB40', None)),
            'pT217': get_plasma_value(row.get('pT217', None)),
            'pT217_AB42': get_plasma_value(row.get('pT217_AB42', None)),
            'NfL': get_plasma_value(row.get('NfL', None)),
            'GFAP': get_plasma_value(row.get('GFAP', None))
        }
        
        # 4. 查找其他模态文件
        fdg_path = find_nii_file(coregistration_base, fdg_subdir, subject_id, id_fdg)
        av45_path = find_nii_file(coregistration_base, av45_subdir, subject_id, id_av45)
        tau_path = find_nii_file(coregistration_base, tau_subdir, subject_id, id_av1451)
        
        # 5. 对于缺失的模态，创建零填充文件
        # 文件命名格式: {subject_id}_{modality}_zero.nii.gz
        
        if fdg_path is None:
            zero_fdg_path = os.path.join(zero_files_dir, 'FDG', f"{subject_id}_fdg_zero.nii.gz")
            fdg_path = create_zero_nii(mri_path, zero_fdg_path, overwrite=True)
            if fdg_path:
                zero_files_created['fdg'] += 1
        
        if av45_path is None:
            zero_av45_path = os.path.join(zero_files_dir, 'AV45', f"{subject_id}_av45_zero.nii.gz")
            av45_path = create_zero_nii(mri_path, zero_av45_path, overwrite=True)
            if av45_path:
                zero_files_created['av45'] += 1
        
        if tau_path is None:
            zero_tau_path = os.path.join(zero_files_dir, 'TAU', f"{subject_id}_tau_zero.nii.gz")
            tau_path = create_zero_nii(mri_path, zero_tau_path, overwrite=True)
            if tau_path:
                zero_files_created['tau'] += 1
        
        # 6. 构建数据条目
        data_entry = {
            "name": subject_id,
            "examdate": examdate,
            "mri": mri_path,
            "fdg": fdg_path,
            "av45": av45_path,
            "tau": tau_path,
            "description": description,
            "old_descr": old_descr,
            "diagnosis": diagnosis,
            "plasma": plasma_data
        }
        
        paired_data.append(data_entry)
    
    print(f"\n=== Processing Summary ===")
    print(f"Total pairs processed: {len(paired_data)}")
    print(f"Missing MRI (skipped): {missing_mri}")
    print(f"Missing descriptions: {missing_descriptions}")
    print(f"Zero files created - FDG: {zero_files_created['fdg']}, AV45: {zero_files_created['av45']}, TAU: {zero_files_created['tau']}")

    # ==================== 添加索引 ====================
    for idx, data in enumerate(paired_data):
        data["fdg_index"] = idx
        data["av45_index"] = idx
        data["tau_index"] = idx

    # ==================== 生成BiomedCLIP嵌入 ====================
    # import torch
    # from transformers import AutoTokenizer, AutoModel

    # local_model_path = "/mnt/nfsdata/nfsdata/lsj.14/replicaLT/BiomedCLIP"
    
    # print("\nLoading BiomedCLIP model...")
    # processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
    # model = AutoModel.from_pretrained(local_model_path, trust_remote_code=True)

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model.to(device)
    # model.eval()

    # 获取有description的样本
    # modal_information = [data["description"] for data in paired_data if data["description"] is not None]
    
    # if modal_information:
    #     print(f"\nEncoding {len(modal_information)} descriptions...")
    #     # 对描述信息进行 Tokenizer 编码
    #     text_inputs = processor(
    #         modal_information,
    #         padding=True,
    #         truncation=True,
    #         return_tensors="pt",
    #         max_length=256
    #     ).to(device)

    #     # 转化为 BiomedCLIP 嵌入
    #     with torch.no_grad():
    #         desc_text_features = model.get_text_features(text_inputs['input_ids'])
    #         print(f"Shape of features: {desc_text_features.shape}, Dtype: {desc_text_features.dtype}")

    # ==================== 划分训练集和验证集 ====================
    val_size = max(1, int(len(paired_data) * 0.1))
    train_data, val_data = train_test_split(paired_data, test_size=val_size, random_state=42)

    print(f"\nTraining set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")

    # ==================== 保存到JSON文件 ====================
    with open(train_json_path, "w") as f:
        json.dump(train_data, f, indent=4)

    with open(val_json_path, "w") as f:
        json.dump(val_data, f, indent=4)

    print(f"\nSaved train data to: {train_json_path}")
    print(f"Saved validation data to: {val_json_path}")

    # ==================== 验证保存结果 ====================
    with open(train_json_path, "r") as f:
        train_data = json.load(f)
    print(f"\nLoaded {len(train_data)} training samples.")
    if train_data:
        print("First training sample:", json.dumps(train_data[0], indent=2))

    with open(val_json_path, "r") as f:
        val_data = json.load(f)
    print(f"Loaded {len(val_data)} validation samples.")
    if val_data:
        print("First validation sample:", json.dumps(val_data[0], indent=2))
    return True


if __name__ == '__main__':
    main()
