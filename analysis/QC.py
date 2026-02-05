import os, glob
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import gaussian_filter, laplace, sobel
from tqdm import tqdm

# ========== 配置路径 ==========
TAU_DIR = '/mnt/nfsdata/nfsdata/lsj.14/ADNI_CSF/PET_MNI/TAU'
MRI_DIR = '/mnt/nfsdata/nfsdata/lsj.14/ADNI_CSF/MRI'
CSV_PATH = '/home/ssddata/linshuijin/replicaLT/adapter_finetune/gen_csv/pairs_180d_dx_plasma_90d_matched_with_demog.csv'
OUTPUT_CSV = '/home/ssddata/linshuijin/replicaLT/analysis/tau_qc_results.csv'

def assign_quality_by_quantile(df):
    # 以数据分布自适应阈值：你可以调整这些分位数
    q = {}
    for col in ["hf_ratio", "cv_low", "lap_abs_mean"]:
        x = df[col].dropna()
        if len(x) == 0:
            continue
        q[col] = {
            "p60": x.quantile(0.60),
            "p90": x.quantile(0.90),
        }

    # leak_outside 用更严格一些：p95 以上大概率有问题
    x = df["leak_outside"].dropna()
    q["leak_outside"] = {"p95": x.quantile(0.95)} if len(x) else {"p95": 0.01}

    quality = []
    weight = []

    for _, r in df.iterrows():
        # hard QC：明显泄漏
        if pd.notna(r["leak_outside"]) and r["leak_outside"] > q["leak_outside"]["p95"]:
            quality.append("Noisy")
            weight.append(0.3)
            continue

        # soft QC：噪声桶（越大越噪）
        noisy_flags = 0
        clean_flags = 0

        for col in ["hf_ratio", "cv_low", "lap_abs_mean"]:
            if col not in q or pd.isna(r[col]):
                continue
            if r[col] >= q[col]["p90"]:
                noisy_flags += 1
            if r[col] <= q[col]["p60"]:
                clean_flags += 1

        if noisy_flags >= 1:
            quality.append("Noisy")
            weight.append(0.4)
        elif clean_flags >= 2:
            quality.append("Clean")
            weight.append(1.0)
        else:
            quality.append("Medium")
            weight.append(0.7)

    return quality, weight



def load_nii(path):
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return data, img.header.get_zooms()[:3]

def brain_mask_from_mri(mri, eps=0.02):
    # 你已去颅骨，通常背景为 0
    return mri > eps

def roughness_metrics(pet, mask):
    pet = pet * mask

    pet_s05 = gaussian_filter(pet, sigma=0.5)
    lap = laplace(pet_s05)
    lap_abs_mean = float(np.mean(np.abs(lap[mask])))

    gx = sobel(pet_s05, axis=0)
    gy = sobel(pet_s05, axis=1)
    gz = sobel(pet_s05, axis=2)
    grad = np.sqrt(gx**2 + gy**2 + gz**2)
    grad_mean = float(np.mean(grad[mask]))

    # 高频能量占比（平滑 sigma=2.0）
    pet_s2 = gaussian_filter(pet, sigma=2.0)
    hf = pet - pet_s2
    hf_ratio = float(np.mean(hf[mask]**2) / (np.mean(pet[mask]**2) + 1e-12))

    # 低强度区域 CV（用脑内强度的第20百分位作为阈值）
    x = pet[mask]
    thr = np.percentile(x, 20)
    # 避免把接近0的背景/极低值纳入，导致CV虚高
    low = mask & (pet <= thr) & (pet > 0.02)
    if np.sum(low) < 1000:
        cv_low = np.nan
    else:
        cv_low = float(pet[low].std() / (pet[low].mean() + 1e-8))


    return lap_abs_mean, grad_mean, hf_ratio, cv_low

def leakage_metrics(pet, mri_mask, pet_eps=1e-6):
    # 先用阈值去掉插值残留，再限定在MRI脑mask内统计更稳健
    pet_mask = pet > pet_eps
    # 只计算“PET显著非零”中落在MRI脑外的比例
    outside = float(np.sum(pet_mask & (~mri_mask)) / (np.sum(pet_mask) + 1e-8))
    return outside


def edge_alignment_metric(mri, pet, mask):
    # 边缘对齐代理：MRI 边缘处 PET 梯度是否"跟着变"
    mri_s = gaussian_filter(mri * mask, sigma=0.5)
    pet_s = gaussian_filter(pet * mask, sigma=0.5)

    m_g = np.sqrt(sobel(mri_s,0)**2 + sobel(mri_s,1)**2 + sobel(mri_s,2)**2)
    p_g = np.sqrt(sobel(pet_s,0)**2 + sobel(pet_s,1)**2 + sobel(pet_s,2)**2)

    thr = np.percentile(m_g[mask], 80)
    edge = mask & (m_g >= thr)
    if np.sum(edge) < 1000:
        return np.nan
    corr = np.corrcoef(m_g[edge].ravel(), p_g[edge].ravel())[0,1]
    return float(corr)


def build_pairs_from_csv(csv_path, tau_dir, mri_dir):
    """
    从CSV读取数据，构建MRI/TAU配对列表
    CSV中相关列: PTID, ID_MRI, ID_AV1451
    TAU文件格式: {PTID}__{ID_MRI}__{ID_AV1451}.nii.gz
    MRI文件格式: {PTID}__{ID_MRI}.nii.gz
    """
    df = pd.read_csv(csv_path)
    pairs = []
    missing_tau = []
    missing_mri = []
    
    for idx, row in df.iterrows():
        ptid = str(row['PTID'])
        id_mri = row['ID_MRI']
        id_tau = row['ID_AV1451']
        
        # 跳过无TAU ID的行
        if pd.isna(id_tau) or id_tau == '':
            continue
        
        # 构建文件路径
        # TAU文件: {PTID}__I{ID_MRI}__I{ID_AV1451}.nii.gz
        tau_filename = f"{ptid}__I{int(id_mri)}__I{int(id_tau)}.nii.gz"
        tau_path = os.path.join(tau_dir, tau_filename)
        
        # MRI文件: {PTID}__I{ID_MRI}.nii.gz
        mri_filename = f"{ptid}__I{int(id_mri)}.nii.gz"
        mri_path = os.path.join(mri_dir, mri_filename)
        
        # 检查文件是否存在
        if not os.path.exists(tau_path):
            missing_tau.append(tau_filename)
            continue
        if not os.path.exists(mri_path):
            missing_mri.append(mri_filename)
            continue
        
        pairs.append({
            'ptid': ptid,
            'id_mri': id_mri,
            'id_tau': id_tau,
            'mri_path': mri_path,
            'tau_path': tau_path,
            'diagnosis': row.get('diagnosis', ''),
            'examdate': row.get('EXAMDATE', ''),
        })
    
    print(f"Total rows in CSV: {len(df)}")
    print(f"Rows with TAU ID: {len(df[df['ID_AV1451'].notna()])}")
    print(f"Valid pairs found: {len(pairs)}")
    print(f"Missing TAU files: {len(missing_tau)}")
    print(f"Missing MRI files: {len(missing_mri)}")
    
    if missing_tau[:5]:
        print(f"  Sample missing TAU: {missing_tau[:5]}")
    if missing_mri[:5]:
        print(f"  Sample missing MRI: {missing_mri[:5]}")
    
    return pairs


def compute_qc_score(row_dict):
    """
    根据QC指标计算综合质量评分
    返回 'Clean', 'Medium', 'Noisy' 分类和对应的训练权重
    """
    # 这里可以根据实际数据分布调整阈值
    score = 0
    
    # leak_outside: 越小越好
    if row_dict['leak_outside'] < 0.01:
        score += 2
    elif row_dict['leak_outside'] < 0.05:
        score += 1
    
    # hf_ratio: 高频能量占比，适中为好
    if 0.01 < row_dict['hf_ratio'] < 0.1:
        score += 2
    elif row_dict['hf_ratio'] < 0.2:
        score += 1
    
    # edge_corr: 边缘对齐相关性，越高越好
    # if not np.isnan(row_dict['edge_corr']):
    #     if row_dict['edge_corr'] > 0.5:
    #         score += 2
    #     elif row_dict['edge_corr'] > 0.3:
    #         score += 1
    
    # cv_low: 低强度区域变异系数，适中为好
    if row_dict['cv_low'] < 0.5:
        score += 2
    elif row_dict['cv_low'] < 1.0:
        score += 1
    
    # 分类
    if score >= 6:
        return 'Clean', 1.0
    elif score >= 3:
        return 'Medium', 0.7
    else:
        return 'Noisy', 0.3


def detect_outliers_mad(df, columns, threshold=3.0):
    """
    使用 median ± 3*MAD 检测离群值
    """
    outlier_mask = pd.Series(False, index=df.index)
    
    for col in columns:
        if col not in df.columns:
            continue
        data = df[col].dropna()
        if len(data) == 0:
            continue
        median = data.median()
        mad = np.median(np.abs(data - median))
        if mad == 0:
            mad = 1e-8
        lower = median - threshold * 1.4826 * mad
        upper = median + threshold * 1.4826 * mad
        outlier_mask |= (df[col] < lower) | (df[col] > upper)
    
    return outlier_mask


def main():
    print("=" * 60)
    print("TAU PET Quality Control Analysis")
    print("=" * 60)
    
    # 构建配对列表
    print("\n[1/3] Building MRI/TAU pairs from CSV...")
    pairs = build_pairs_from_csv(CSV_PATH, TAU_DIR, MRI_DIR)
    
    if len(pairs) == 0:
        print("No valid pairs found. Exiting.")
        return
    
    # 计算QC指标
    print(f"\n[2/3] Computing QC metrics for {len(pairs)} pairs...")
    rows = []
    for pair in tqdm(pairs, desc="Processing"):
        try:
            mri, mz = load_nii(pair['mri_path'])
            pet, pz = load_nii(pair['tau_path'])
            if mri.shape != (182, 218, 182) or pet.shape != (182, 218, 182):
                raise ValueError(f"Unexpected shape: mri {mri.shape}, pet {pet.shape}")

            
            mask = brain_mask_from_mri(mri, eps=0.0)
            
            outside = leakage_metrics(pet, mask)
            lap_m, grad_m, hf_ratio, cv_low = roughness_metrics(pet, mask)
            edge_corr = edge_alignment_metric(mri, pet, mask)
            
            row = {
                "ptid": pair['ptid'],
                "id_mri": pair['id_mri'],
                "id_tau": pair['id_tau'],
                "diagnosis": pair['diagnosis'],
                "examdate": pair['examdate'],
                "tau_file": os.path.basename(pair['tau_path']),
                "shape": str(mri.shape),
                "mri_zooms": str(mz),
                "pet_zooms": str(pz),
                "pet_min": float(pet[mask].min()) if mask.sum() > 0 else np.nan,
                "pet_max": float(pet[mask].max()) if mask.sum() > 0 else np.nan,
                "pet_mean": float(pet[mask].mean()) if mask.sum() > 0 else np.nan,
                "pet_std": float(pet[mask].std()) if mask.sum() > 0 else np.nan,
                "leak_outside": outside,
                "lap_abs_mean": lap_m,
                "grad_mean": grad_m,
                "hf_ratio": hf_ratio,
                "cv_low": cv_low,
                "edge_corr": edge_corr,
            }
            rows.append(row)
        except Exception as e:
            print(f"Error processing {pair['tau_path']}: {e}")
            continue
    
    # 创建DataFrame
    df = pd.DataFrame(rows)
    
    # 检测离群值
    qc_cols = ['leak_outside', 'lap_abs_mean', 'grad_mean', 'hf_ratio', 'cv_low', 'edge_corr']
    df['is_outlier'] = detect_outliers_mad(df, qc_cols)
    
    # 计算质量分类和训练权重
    quality, weight = assign_quality_by_quantile(df)
    df["quality_class"] = quality
    df["train_weight"] = weight

    
    # 保存结果
    print(f"\n[3/3] Saving results to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False)
    
    # 输出统计摘要
    print("\n" + "=" * 60)
    print("QC Summary Statistics")
    print("=" * 60)
    print(f"\nTotal samples analyzed: {len(df)}")
    print(f"\nQuality classification:")
    print(df['quality_class'].value_counts())
    print(f"\nOutliers detected: {df['is_outlier'].sum()}")
    
    print("\nQC Metrics Summary:")
    for col in qc_cols:
        if col in df.columns:
            print(f"  {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}, "
                  f"min={df[col].min():.4f}, max={df[col].max():.4f}")
    
    print(f"\nResults saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
