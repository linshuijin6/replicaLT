"""
convert_nifti_to_h5.py
======================
将 replicaLT 项目的 NIfTI 数据（MRI + TAU PET）转换为 PASTA 需要的 HDF5 格式。

数据来源：
  - extract_tabular.py 生成的 train_tabular.json / val_tabular.json
    （含真实临床字段: AGE, PTGENDER, PTEDUCAT, MMSE, ADAS13, APOE4）
  - MRI: /mnt/nfsdata/nfsdata/lsj.14/ADNI_CSF/MRI/
  - TAU: /mnt/nfsdata/nfsdata/lsj.14/ADNI_CSF/PET_MNI/TAU/

Tabular 字段说明：
  columns = ['age', 'gender', 'education', 'MMSE', 'ADAS13', 'ApoE4']
  对应 JSON 字段: AGE, PTGENDER(Male=1/Female=0), PTEDUCAT, MMSE, ADAS13, APOE4
  NaN 值保留，由 PASTA 的 process_tabular_data 追加 missing mask 处理。

输出格式（PASTA HDF5）：
  /<image_id>/MRI/T1/data    → (113, 137, 113) float32
  /<image_id>/PET/FDG/data   → (113, 137, 113) float32  (实际为 TAU PET)
  /<image_id>/tabular        → (6,) float32 真实临床特征（NaN 保留）
  /<image_id>.attrs['DX']    → 诊断标签字符串
  /<image_id>.attrs['RID']   → 0 (占位)
  /<image_id>.attrs['VISCODE'] → '' (占位)
  /stats/tabular/columns    → ['age','gender','education','MMSE','ADAS13','ApoE4']
  /stats/tabular/mean       → (6,) 训练集均值（仅在 train.h5 计算，val/test 复用）
  /stats/tabular/stddev     → (6,) 训练集标准差

关于重采样与公平性：
  PASTA 的设计范式是 2.5D（逐切片处理），其标准输入尺寸为 (113, 137, 113)
  @ 1.5mm 等距体素。这是 PASTA 论文中使用的配置，在此分辨率下运行 PASTA
  是给予其最优条件，符合公平对比原则。replicaLT 则在更高的 (160,192,160) @ 1.0mm
  3D 体积下训练，两者各自使用自身设计的最优分辨率。

用法：
  # 在 xiaochou 环境中运行
  cd /mnt/nfsdata/nfsdata/lsj.14/PASTA
  conda run -n xiaochou python replicaLT_comparison/convert_nifti_to_h5.py
"""

import os
import json
import sys
import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm
from report_error import email_on_error
# ============ 配置 ============
REPLICA_LT_DIR = "/mnt/nfsdata/nfsdata/lsj.14/replicaLT"

# ★ 使用 extract_tabular.py 生成的、含真实临床字段的 JSON
TABULAR_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRAIN_JSON = os.path.join(TABULAR_DATA_DIR, "train_tabular.json")
VAL_JSON   = os.path.join(TABULAR_DATA_DIR, "val_tabular.json")

OUTPUT_DIR = "/mnt/linshuijin/data"
TARGET_SHAPE = (113, 137, 113)  # PASTA 标准输入
TARGET_SPACING = (1.5, 1.5, 1.5)  # PASTA 标准体素

# 诊断标签映射（从 replicaLT 的 description 或 diagnosis 字段提取）
# PASTA 支持: CN, Dementia, MCI
DX_MAP = {
    "CN": "CN",
    "Cognitively normal": "CN",
    "NL": "CN",
    "AD": "Dementia",
    "Dementia": "Dementia",
    "MCI": "MCI",
    "EMCI": "MCI",
    "LMCI": "MCI",
    "SMC": "CN",  # Subjective Memory Concern → CN
}


def resample_to_target(data, original_affine, target_spacing=TARGET_SPACING,
                       target_shape=TARGET_SHAPE):
    """
    将 3D 体积重采样到目标体素间距和形状。
    使用 scipy 的 zoom 函数进行三线性插值。
    """
    from scipy.ndimage import zoom

    # 获取原始体素间距
    original_spacing = np.abs(np.diag(original_affine)[:3])

    # 计算缩放因子
    zoom_factors = original_spacing / np.array(target_spacing)

    # 重采样
    resampled = zoom(data, zoom_factors, order=1)  # 三线性插值

    # CropOrPad 到目标形状
    resampled = crop_or_pad(resampled, target_shape)

    return resampled


def crop_or_pad(data, target_shape):
    """
    中心裁剪或零填充到目标形状。
    """
    result = np.zeros(target_shape, dtype=data.dtype)
    src_shape = data.shape

    # 计算每个维度的起始和结束索引
    starts_src = []
    ends_src = []
    starts_dst = []
    ends_dst = []

    for s, t in zip(src_shape, target_shape):
        if s >= t:
            # 需要裁剪
            start_s = (s - t) // 2
            starts_src.append(start_s)
            ends_src.append(start_s + t)
            starts_dst.append(0)
            ends_dst.append(t)
        else:
            # 需要填充
            start_d = (t - s) // 2
            starts_src.append(0)
            ends_src.append(s)
            starts_dst.append(start_d)
            ends_dst.append(start_d + s)

    result[starts_dst[0]:ends_dst[0],
           starts_dst[1]:ends_dst[1],
           starts_dst[2]:ends_dst[2]] = data[starts_src[0]:ends_src[0],
                                              starts_src[1]:ends_src[1],
                                              starts_src[2]:ends_src[2]]
    return result


def extract_diagnosis(item):
    """
    从 replicaLT 数据条目中提取诊断标签。
    优先使用 'diagnosis' 字段，否则尝试从 'description' 解析。
    """
    dx = str(item.get("diagnosis", ""))
    if dx and dx in DX_MAP:
        return DX_MAP[dx]

    # 如果 diagnosis 字段存在但不在映射表中，尝试直接使用
    if dx:
        for key, val in DX_MAP.items():
            if key.lower() in dx.lower():
                return val

    # 默认返回 MCI（大部分 ADNI 数据为 MCI）
    return "MCI"


def make_unique_key(name, examdate, seen_keys):
    """
    生成唯一的 HDF5 组名。
    同一被试存在多次扫描时，使用 '{name}__{examdate}' 区分。
    若同日仍有重复（极少见），追加 _v2, _v3 ... 后缀。
    """
    base = f"{name}__{examdate}" if examdate else name
    key = base
    suffix = 2
    while key in seen_keys:
        key = f"{base}_v{suffix}"
        suffix += 1
    seen_keys.add(key)
    return key


# ─────────────────────────── tabular helpers ──────────────────

GENDER_MAP = {"Male": 1.0, "Female": 0.0, "M": 1.0, "F": 0.0}


def extract_tabular_vector(item) -> np.ndarray:
    """
    从一条 JSON record 中提取 6 维 tabular 向量：
      [AGE, PTGENDER(1/0), PTEDUCAT, MMSE, ADAS13, APOE4]
    NaN 值保留，由 PASTA 的 process_tabular_data 生成 missing mask。
    """
    def safe_float(v):
        try:
            f = float(v)
            return np.nan if np.isnan(f) else f
        except (TypeError, ValueError):
            return np.nan

    age     = safe_float(item.get("AGE",      np.nan))
    gender  = GENDER_MAP.get(str(item.get("PTGENDER", "")), np.nan)
    edu     = safe_float(item.get("PTEDUCAT", np.nan))
    mmse    = safe_float(item.get("MMSE",     np.nan))
    adas13  = safe_float(item.get("ADAS13",   np.nan))
    apoe4   = safe_float(item.get("APOE4",    np.nan))

    return np.array([age, gender, edu, mmse, adas13, apoe4], dtype=np.float32)


def compute_tabular_stats(data_list) -> tuple:
    """
    计算训练集各列的均值和标准差（忽略 NaN）。
    返回 (mean_6, std_6) 两个 float32 数组。
    """
    vectors = [extract_tabular_vector(item) for item in data_list
               if item.get("tau")]  # 只对最终进入 h5 的样本计算
    if not vectors:
        return np.zeros(6, dtype=np.float32), np.ones(6, dtype=np.float32)
    mat = np.stack(vectors, axis=0)  # (N, 6)
    mean = np.nanmean(mat, axis=0).astype(np.float32)
    std  = np.nanstd(mat,  axis=0).astype(np.float32)
    std  = np.where(std < 1e-6, 1.0, std)  # 防止除零
    return mean, std


def convert_split(data_list, output_path, split_name,
                  tabular_mean=None, tabular_std=None):
    """
    将一个数据划分（train/val/test）转换为 HDF5 文件。
    tabular_mean / tabular_std: 若为 None，从当前 split 自行计算（仅 train 使用）。
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 仅保留有 TAU PET 的样本
    filtered = [item for item in data_list if item.get("tau")]
    print(f"\n{'='*60}")
    print(f"[{split_name}] 总样本: {len(data_list)}, 有 TAU: {len(filtered)}")
    print(f"输出文件: {output_path}")
    print(f"目标形状: {TARGET_SHAPE}")
    print(f"{'='*60}")

    skipped = 0
    written = 0

    # 若未传入统计量则自行计算（train split）
    if tabular_mean is None or tabular_std is None:
        print(f"  [{split_name}] 计算 tabular 统计量（mean / stddev）...")
        tabular_mean, tabular_std = compute_tabular_stats(data_list)
    print(f"  [{split_name}] tabular mean  = {tabular_mean}")
    print(f"  [{split_name}] tabular stddev = {tabular_std}")

    with h5py.File(output_path, 'w') as h5f:
        # 写入统计信息组（使用真实训练集统计量）
        stats_grp = h5f.create_group("stats")
        tab_grp = stats_grp.create_group("tabular")
        columns = ['age', 'gender', 'education', 'MMSE', 'ADAS13', 'ApoE4']
        tab_grp.create_dataset("columns", data=np.array(columns, dtype='S'))
        tab_grp.create_dataset("mean",   data=tabular_mean)
        tab_grp.create_dataset("stddev", data=tabular_std)

        seen_keys = set()
        for item in tqdm(filtered, desc=f"Converting {split_name}"):
            name = item["name"]
            examdate = item.get("examdate", "").replace("-", "")
            h5_key = make_unique_key(name, examdate, seen_keys)
            mri_path = item["mri"]
            tau_path = item["tau"]

            # 检查文件是否存在
            if not os.path.exists(mri_path):
                print(f"  ⚠️  MRI 文件不存在: {mri_path}")
                skipped += 1
                continue
            if not os.path.exists(tau_path):
                print(f"  ⚠️  TAU 文件不存在: {tau_path}")
                skipped += 1
                continue

            try:
                # 加载 NIfTI
                mri_nib = nib.load(mri_path)
                tau_nib = nib.load(tau_path)

                mri_data = mri_nib.get_fdata().astype(np.float32)
                tau_data = tau_nib.get_fdata().astype(np.float32)

                # 处理 4D 数据（取均值或中间帧）
                if mri_data.ndim == 4:
                    mri_data = mri_data.mean(axis=-1)
                if tau_data.ndim == 4:
                    tau_data = tau_data.mean(axis=-1)

                # 重采样到 PASTA 标准尺寸
                mri_resampled = resample_to_target(mri_data, mri_nib.affine)
                tau_resampled = resample_to_target(tau_data, tau_nib.affine)

                # 确保无 NaN
                mri_resampled = np.nan_to_num(mri_resampled, copy=False)
                tau_resampled = np.nan_to_num(tau_resampled, copy=False)

                # 提取诊断标签
                dx = extract_diagnosis(item)

                # 创建 HDF5 组（以唯一 key 命名，避免同一被试多次扫描冲突）
                grp = h5f.create_group(h5_key)

                # MRI 数据
                mri_grp = grp.create_group("MRI")
                t1_grp = mri_grp.create_group("T1")
                t1_grp.create_dataset("data", data=mri_resampled, dtype='float32',
                                      compression='gzip', compression_opts=4)

                # TAU PET 数据（写入 PET/FDG 路径以兼容 PASTA 代码）
                pet_grp = grp.create_group("PET")
                fdg_grp = pet_grp.create_group("FDG")
                fdg_grp.create_dataset("data", data=tau_resampled, dtype='float32',
                                       compression='gzip', compression_opts=4)

                # Tabular 真实临床数据（NaN 保留，由 PASTA 的 process_tabular_data 处理 missing mask）
                tab_vec = extract_tabular_vector(item)
                grp.create_dataset("tabular", data=tab_vec)

                # 属性
                grp.attrs['DX'] = dx
                grp.attrs['RID'] = 0
                grp.attrs['VISCODE'] = ''

                written += 1

            except Exception as e:
                print(f"  ❌ 处理失败 {h5_key}: {e}")
                skipped += 1
                continue

    print(f"\n[{split_name}] 完成: 成功 {written}, 跳过 {skipped}")
    return written

@email_on_error()
def main():
    # 加载 replicaLT 数据划分
    print("📂 加载 replicaLT 数据划分...")
    with open(TRAIN_JSON, 'r') as f:
        train_data = json.load(f)
    with open(VAL_JSON, 'r') as f:
        val_data = json.load(f)

    print(f"  训练集: {len(train_data)} 样本")
    print(f"  验证集: {len(val_data)} 样本")

    # 将验证集一半作为 valid，一半作为 test（PASTA 需要 train/valid/test 三份）
    # 或者直接用 val 作为 valid，再用 val 作为 test（对比实验中常用做法）
    mid = len(val_data) // 2
    valid_data = val_data[:mid]
    test_data = val_data[mid:]
    print(f"  -> valid: {len(valid_data)}, test: {len(test_data)}")

    # 转换
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ★ 先计算训练集统计量，val/test 复用，确保归一化范围一致
    print("\n计算训练集 tabular 统计量...")
    train_mean, train_std = compute_tabular_stats(
        [item for item in train_data if item.get("tau")]
    )

    n_train = convert_split(train_data,
                            os.path.join(OUTPUT_DIR, "train.h5"), "train",
                            tabular_mean=train_mean, tabular_std=train_std)
    n_valid = convert_split(valid_data,
                            os.path.join(OUTPUT_DIR, "valid.h5"), "valid",
                            tabular_mean=train_mean, tabular_std=train_std)
    n_test = convert_split(test_data,
                           os.path.join(OUTPUT_DIR, "test.h5"), "test",
                           tabular_mean=train_mean, tabular_std=train_std)

    print(f"\n{'='*60}")
    print(f"✅ 全部转换完成!")
    print(f"  train: {n_train} 样本 → {os.path.join(OUTPUT_DIR, 'train.h5')}")
    print(f"  valid: {n_valid} 样本 → {os.path.join(OUTPUT_DIR, 'valid.h5')}")
    print(f"  test:  {n_test} 样本 → {os.path.join(OUTPUT_DIR, 'test.h5')}")
    print(f"{'='*60}")

    # 验证 HDF5 结构
    print("\n🔍 验证 HDF5 结构（train.h5 前 3 个样本）:")
    with h5py.File(os.path.join(OUTPUT_DIR, "train.h5"), 'r') as f:
        count = 0
        for name in f.keys():
            if name == "stats":
                print(f"  stats: columns={list(f['stats/tabular/columns'][:])}")
                continue
            grp = f[name]
            mri_shape = grp['MRI/T1/data'].shape
            pet_shape = grp['PET/FDG/data'].shape
            dx = grp.attrs['DX']
            print(f"  {name}: MRI={mri_shape}, PET={pet_shape}, DX={dx}")
            count += 1
            if count >= 3:
                break


if __name__ == "__main__":
    main()
