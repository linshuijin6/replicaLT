import os
import json
import time
import shutil
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

TABULAR_DATA_DIR = "/mnt/nfsdata/nfsdata/lsj.14/PASTA/replicaLT_comparison/data"
TRAIN_JSON = os.path.join(TABULAR_DATA_DIR, "train_tabular.json")

LOCAL_DIR = "/mnt/ssd/linshuijin/test_nifti_perf"

TARGET_SHAPE = (113, 137, 113)
TARGET_SPACING = (1.5, 1.5, 1.5)

def crop_or_pad(data, target_shape):
    result = np.zeros(target_shape, dtype=data.dtype)
    src_shape = data.shape

    starts_src = []
    ends_src = []
    starts_dst = []
    ends_dst = []

    for s, t in zip(src_shape, target_shape):
        if s >= t:
            start_s = (s - t) // 2
            starts_src.append(start_s)
            ends_src.append(start_s + t)
            starts_dst.append(0)
            ends_dst.append(t)
        else:
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

def resample_to_target(data, original_affine, target_spacing=TARGET_SPACING,
                       target_shape=TARGET_SHAPE):
    original_spacing = np.abs(np.diag(original_affine)[:3])
    zoom_factors = original_spacing / np.array(target_spacing)
    resampled = zoom(data, zoom_factors, order=1)
    resampled = crop_or_pad(resampled, target_shape)
    return resampled

def test_speed(sample_idx, item, base_dir=""):
    print(f"\n样本 {sample_idx}: {item['name']}")
    
    mri_path = item["mri"]
    tau_path = item["tau"]

    if base_dir:
        mri_path = os.path.join(base_dir, os.path.basename(mri_path))
        tau_path = os.path.join(base_dir, os.path.basename(tau_path))

    # Test I/O time
    t0 = time.time()
    mri_nib = nib.load(mri_path)
    tau_nib = nib.load(tau_path)
    
    mri_data = mri_nib.get_fdata().astype(np.float32)
    tau_data = tau_nib.get_fdata().astype(np.float32)

    if mri_data.ndim == 4:
        mri_data = mri_data.mean(axis=-1)
    if tau_data.ndim == 4:
        tau_data = tau_data.mean(axis=-1)
    t1 = time.time()
    io_time = t1 - t0

    # Test processing time
    mri_resampled = resample_to_target(mri_data, mri_nib.affine)
    tau_resampled = resample_to_target(tau_data, tau_nib.affine)
    
    mri_resampled = np.nan_to_num(mri_resampled, copy=False)
    tau_resampled = np.nan_to_num(tau_resampled, copy=False)
    t2 = time.time()
    process_time = t2 - t1
    
    return io_time, process_time

def main():
    print("加载 JSON...")
    with open(TRAIN_JSON, 'r') as f:
        train_data = json.load(f)
        
    filtered = [item for item in train_data if item.get("tau")]
    num_samples = min(5, len(filtered)) # Test 5 samples
    samples = filtered[:num_samples]
    
    print(f"准备测试 {num_samples} 个样本...\n")
    
    # === 1. Test NFS ===
    print("=== 测试 1: 从 NFS (原始路径) 读取 ===")
    nfs_io_times = []
    nfs_proc_times = []
    for i, item in enumerate(samples):
        if not os.path.exists(item["mri"]) or not os.path.exists(item["tau"]):
            continue
        io_t, proc_t = test_speed(i+1, item)
        nfs_io_times.append(io_t)
        nfs_proc_times.append(proc_t)
        print(f"  I/O 耗时: {io_t:.3f} s")
        print(f"  处理 耗时: {proc_t:.3f} s")
        
    avg_nfs_io = np.mean(nfs_io_times)
    avg_nfs_proc = np.mean(nfs_proc_times)
    
    # === 2. Copy to Local ===
    os.makedirs(LOCAL_DIR, exist_ok=True)
    print("\n复制文件到本地 SSD 进行测试...")
    for item in samples:
        if not os.path.exists(item["mri"]) or not os.path.exists(item["tau"]):
            continue
        shutil.copy2(item["mri"], LOCAL_DIR)
        shutil.copy2(item["tau"], LOCAL_DIR)
        
    # === 3. Test Local ===
    print("\n=== 测试 2: 从本地 SSD 读取 ===")
    ssd_io_times = []
    ssd_proc_times = []
    for i, item in enumerate(samples):
        io_t, proc_t = test_speed(i+1, item, base_dir=LOCAL_DIR)
        ssd_io_times.append(io_t)
        ssd_proc_times.append(proc_t)
        print(f"  I/O 耗时: {io_t:.3f} s")
        print(f"  处理 耗时: {proc_t:.3f} s")
        
    avg_ssd_io = np.mean(ssd_io_times)
    avg_ssd_proc = np.mean(ssd_proc_times)
    
    print("\n=== 结论 ===")
    print(f"NFS 平均 I/O 耗时: {avg_nfs_io:.3f} s")
    print(f"SSD 平均 I/O 耗时: {avg_ssd_io:.3f} s")
    print(f"处理 (重采样等) 平均耗时: {(avg_nfs_proc + avg_ssd_proc) / 2:.3f} s")
    
    # Clean up
    shutil.rmtree(LOCAL_DIR)
    
if __name__ == "__main__":
    main()
