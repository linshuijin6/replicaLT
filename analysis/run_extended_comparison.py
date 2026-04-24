#!/usr/bin/env python3
"""
Extended MRI -> TAU PET comparison pipeline.

Runs the planned Phase A-F analyses and generates a consolidated markdown report.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from comparison_common import EXTENDED_OUT_DIR, load_subject_metadata
from ext_clinical_fidelity import run_clinical_fidelity
from ext_correlation import run_correlation
from ext_deployability import run_deployability
from ext_diagnosis import run_diagnosis
from ext_robustness import run_robustness
from ext_staging import run_staging


def _fmt_num(x, digits=3):
    return "N/A" if pd.isna(x) else f"{x:.{digits}f}"


def _fmt_ci(lo, hi, digits=3):
    if pd.isna(lo) or pd.isna(hi):
        return "N/A"
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"


def _render_report(deploy_df, clinical_df, staging_df, diagnosis_df, robustness_outputs, corr_df):
    meta = load_subject_metadata()
    diag_counts = meta["diagnosis"].value_counts().to_dict()
    lines = []
    lines.append("# MRI → TAU PET: Extended Comparison Report")
    lines.append("")
    lines.append(
        f"*Generated: {datetime.now():%Y-%m-%d %H:%M} | Subjects: {len(meta)} "
        f"(CN={diag_counts.get('CN', 0)}, MCI={diag_counts.get('MCI', 0)}, AD={diag_counts.get('AD', 0)}) | "
        "Space: 160×192×160 (1mm MNI)*"
    )
    lines.append("")
    lines.append("> **Note**: Intensities are normalized to [0,1]; staging thresholds fitted on training set real PET.")
    lines.append("> Bootstrap CI: 2000 resamples, 95% level. Paired tests remain available in the base comparison outputs.")
    lines.append("")

    lines.append("## 1. Deployability")
    lines.append("")
    lines.append("| Method | Params (M) | Size (MB) | Train Time (h) | Infer total (s) | Infer/subj (s) | Steps | Res |")
    lines.append("|--------|-----------:|----------:|---------------:|----------------:|---------------:|------:|-----|")
    for _, row in deploy_df.iterrows():
        lines.append(
            f"| {row['method']} | {_fmt_num(row['params'] / 1e6 if pd.notna(row['params']) else float('nan'), 2)} | "
            f"{_fmt_num(row['model_size_MB'], 1)} | {_fmt_num(row['train_time_hours'], 1)} | "
            f"{_fmt_num(row['inference_time_total_sec'], 0)} | {_fmt_num(row['inference_time_per_subject_sec'], 1)} | "
            f"{_fmt_num(row['diffusion_steps'], 0)} | {row['output_resolution']} |"
        )
    lines.append("")

    lines.append("## 2. Clinical Fidelity (Regional Uptake Consistency)")
    lines.append("")
    roi_order = ["Global_cortical", "Temporal_metaROI", "Braak_I_II", "Braak_III_IV", "Braak_V_VI", "Entorhinal_proxy", "Hippocampus", "Amygdala"]
    for roi in roi_order:
        sub = clinical_df[clinical_df['roi'] == roi]
        if sub.empty:
            continue
        lines.append(f"### 2.{roi_order.index(roi)+1} {roi}")
        lines.append("")
        lines.append("| Method | MAE | 95% CI | Pearson r | Spearman ρ | Bias | LoA |")
        lines.append("|--------|----:|--------|----------:|-----------:|-----:|-----|")
        for _, row in sub.iterrows():
            lines.append(
                f"| {row['method']} | {_fmt_num(row['MAE'], 4)} | {_fmt_ci(row['MAE_lo'], row['MAE_hi'], 4)} | "
                f"{_fmt_num(row['pearson_r'], 3)} | {_fmt_num(row['spearman_r'], 3)} | {_fmt_num(row['bias'], 4)} | "
                f"[{_fmt_num(row['loa_lo'], 4)}, {_fmt_num(row['loa_hi'], 4)}] |"
            )
        lines.append("")

    lines.append("## 3. Tau Staging / Positivity")
    lines.append("")
    lines.append("| Method | Tau Pos. Acc. | 95% CI | Stage Agree. | 95% CI | Weighted κ |")
    lines.append("|--------|--------------:|--------|-------------:|--------|-----------:|")
    for _, row in staging_df.iterrows():
        lines.append(
            f"| {row['method']} | {_fmt_num(row['tau_positivity_accuracy'], 3)} | {_fmt_ci(row['tau_pos_acc_lo'], row['tau_pos_acc_hi'], 3)} | "
            f"{_fmt_num(row['stage_agreement'], 3)} | {_fmt_ci(row['stage_agree_lo'], row['stage_agree_hi'], 3)} | {_fmt_num(row['weighted_kappa'], 3)} |"
        )
    lines.append("")
    lines.append("> Confusion matrices: `staging_confusion_matrices.png`")
    lines.append("")

    lines.append("## 4. Downstream Diagnosis Classification")
    lines.append("")
    lines.append("> LOOCV with multinomial logistic regression. Balanced accuracy is the primary metric because of class imbalance.")
    lines.append("")
    lines.append("| Feature Source | N | Balanced Acc | 95% CI | Macro-F1 | AUC (OvR) |")
    lines.append("|----------------|--:|-------------:|--------|---------:|----------:|")
    for _, row in diagnosis_df.iterrows():
        lines.append(
            f"| {row['feature_source']} | {int(row['n_subjects']) if pd.notna(row['n_subjects']) else 0} | {_fmt_num(row.get('balanced_accuracy'))} | "
            f"{_fmt_ci(row.get('bal_acc_lo'), row.get('bal_acc_hi'))} | {_fmt_num(row.get('macro_f1'))} | {_fmt_num(row.get('auc_ovr'))} |"
        )
    lines.append("")
    lines.append(f"> Class counts in the 43-subject cohort: CN={diag_counts.get('CN', 0)}, MCI={diag_counts.get('MCI', 0)}, AD={diag_counts.get('AD', 0)}.")
    lines.append("")

    by_diag, by_vendor, kw = robustness_outputs
    lines.append("## 5. Robustness")
    lines.append("")
    lines.append("### 5.1 Stratified by Diagnosis")
    lines.append("")
    lines.append("| Method | Stratum | N | SSIM mean±std | PSNR mean±std | MAE mean±std |")
    lines.append("|--------|---------|--:|---------------|---------------|--------------|")
    for _, row in by_diag.iterrows():
        lines.append(
            f"| {row['method']} | {row['stratum']} | {int(row['n'])} | {row['ssim_mean']:.3f}±{row['ssim_std']:.3f} | "
            f"{row['psnr_mean']:.3f}±{row['psnr_std']:.3f} | {row['mae_mean']:.3f}±{row['mae_std']:.3f} |"
        )
    lines.append("")
    lines.append("### 5.2 Stratified by Scanner Vendor")
    lines.append("")
    lines.append("| Method | Vendor | N | SSIM mean±std | PSNR mean±std | MAE mean±std |")
    lines.append("|--------|--------|--:|---------------|---------------|--------------|")
    for _, row in by_vendor.iterrows():
        lines.append(
            f"| {row['method']} | {row['vendor']} | {int(row['n'])} | {row['ssim_mean']:.3f}±{row['ssim_std']:.3f} | "
            f"{row['psnr_mean']:.3f}±{row['psnr_std']:.3f} | {row['mae_mean']:.3f}±{row['mae_std']:.3f} |"
        )
    lines.append("")

    lines.append("## 6. Cognitive / Biomarker Correlation")
    lines.append("")
    lines.append("> Temporal meta-ROI uptake vs MMSE/ADAS13 (Spearman ρ, bootstrap 95% CI)")
    lines.append("")
    lines.append("| Feature Source | Clinical Var | Spearman ρ | 95% CI | p-value | N |")
    lines.append("|----------------|-------------|-----------:|--------|--------:|--:|")
    for _, row in corr_df[corr_df['clinical_var'].isin(['MMSE', 'ADAS13'])].iterrows():
        lines.append(
            f"| {row['feature_source']} | {row['clinical_var']} | {_fmt_num(row['spearman_rho'])} | {_fmt_ci(row['rho_lo'], row['rho_hi'])} | "
            f"{_fmt_num(row['p_value'])} | {int(row['n'])} |"
        )
    lines.append("")
    lines.append("### 6.1 APOE4 Stratification")
    lines.append("")
    lines.append("| Feature Source | Carrier mean | Non-carrier mean | Mann-Whitney p |")
    lines.append("|----------------|-------------:|-----------------:|---------------:|")
    for _, row in corr_df[corr_df['clinical_var'] == 'APOE4_stratification'].iterrows():
        lines.append(
            f"| {row['feature_source']} | {_fmt_num(row.get('carrier_mean'), 4)} | {_fmt_num(row.get('noncarrier_mean'), 4)} | {_fmt_num(row.get('p_value'), 4)} |"
        )
    lines.append("")
    lines.append("> Scatter plots: `correlation_scatter_plots.png`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*End of report. Run `python analysis/run_extended_comparison.py` to regenerate.*")
    return "\n".join(lines)


def main():
    EXTENDED_OUT_DIR.mkdir(parents=True, exist_ok=True)
    deploy_df = run_deployability()
    clinical_df = run_clinical_fidelity()
    staging_df = run_staging()
    diagnosis_df = run_diagnosis()
    robustness_outputs = run_robustness()
    corr_df = run_correlation()

    report = _render_report(deploy_df, clinical_df, staging_df, diagnosis_df, robustness_outputs, corr_df)
    report_path = EXTENDED_OUT_DIR / "extended_comparison_report.md"
    report_path.write_text(report)
    print(f"Saved markdown report to {report_path}")


if __name__ == "__main__":
    main()
