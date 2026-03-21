"""
Detailed PET dose analysis — Modules 1–7 (skip Module 3 → now included).
Reads: analysis/pet_dose_analysis_report.csv  (1360 rows, from analyze_pet_dose.py)
       analysis/Images_PET__ADNI3_4___PLASMA_with_T1_My_Table_15Mar2026.csv  (dose CSV with VSWEIGHT)
Outputs:
  analysis/figures/dose_distribution.png   — Module 1
  analysis/figures/dose_by_site.png        — Module 2
  analysis/figures/dose_temporal.png        — Module 3
  analysis/dose_low_samples.csv            — Module 4
  analysis/dose_unmatched.csv              — Module 5
  analysis/dose_summary_report.txt         — summary
  analysis/pairs_dose_qc_passed.csv        — Module 7
"""
import os, sys, csv, re, collections
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

WORK = "/mnt/nfsdata/nfsdata/lsj.14/replicaLT"
REPORT_CSV = os.path.join(WORK, "analysis", "pet_dose_analysis_report.csv")
DOSE_CSV   = os.path.join(WORK, "analysis",
             "Images_PET__ADNI3_4___PLASMA_with_T1_My_Table_15Mar2026.csv")
FIG_DIR    = os.path.join(WORK, "analysis", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

LOW_THRESH = {"TAU": 9.0, "AV45_FP": 7.5, "FDG": 4.5}
DIAG_MAP   = {"1": "CU", "2": "MCI", "3": "AD"}

# ── Load data ─────────────────────────────────────────────────────────────────
print("[0] Loading data ...")
df = pd.read_csv(REPORT_CSV, dtype=str)
df['DIAGNOSIS_label'] = df['DIAGNOSIS'].map(DIAG_MAP).fillna('Other')
for col in ['dose_tau_mci', 'dose_av45fp_mci', 'dose_fdg_mci',
            'dose_av45fbn_mci', 'dose_match_day_diff']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df['EXAMDATE_dt'] = pd.to_datetime(df['EXAMDATE'], errors='coerce')
df['exam_year'] = df['EXAMDATE_dt'].dt.year
df['site'] = df['PTID'].str[:3]

# Load dose CSV for VSWEIGHT (Module 6)
dose_raw = pd.read_csv(DOSE_CSV, dtype=str)
# Handle duplicate SCANQLTY column (pandas auto-renames to SCANQLTY.1)
dose_raw.rename(columns=lambda c: c.strip('"').strip(), inplace=True)

print(f"  Report: {len(df)} rows;  Dose CSV: {len(dose_raw)} rows")

report_lines = []  # collect summary text
def rprint(msg):
    print(msg)
    report_lines.append(msg)

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1: Dose distribution visualisation
# ═══════════════════════════════════════════════════════════════════════════════
rprint("\n" + "="*70)
rprint("MODULE 1: Dose Distribution Visualisation")
rprint("="*70)

fig = plt.figure(figsize=(20, 18))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.30)

tracer_info = [
    ("TAU",     "dose_tau_mci",    "id_av1451", "#2196F3"),
    ("AV45_FP", "dose_av45fp_mci", "id_av45",   "#FF9800"),
    ("FDG",     "dose_fdg_mci",    "id_fdg",    "#4CAF50"),
]

for col_idx, (tracer, dose_col, id_col, color) in enumerate(tracer_info):
    mask = df[id_col].notna() & (df[id_col] != '')
    vals = df.loc[mask, dose_col].dropna()
    thresh = LOW_THRESH[tracer]
    n_low = (vals < thresh).sum()

    rprint(f"\n  {tracer}: N={len(vals)}, mean={vals.mean():.2f}±{vals.std():.2f}, "
           f"median={vals.median():.2f}, range={vals.min():.2f}–{vals.max():.2f} mCi")
    rprint(f"    Low-dose (<{thresh}): {n_low}/{len(vals)} ({100*n_low/len(vals):.1f}%)")

    # Row 1: Histograms
    ax1 = fig.add_subplot(gs[0, col_idx])
    ax1.hist(vals, bins=30, color=color, alpha=0.75, edgecolor='white')
    ax1.axvline(thresh, color='red', ls='--', lw=2, label=f'Threshold={thresh}')
    ax1.set_title(f'{tracer} Dose Distribution (N={len(vals)})', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Dose (mCi)')
    ax1.set_ylabel('Count')
    ax1.legend(fontsize=9)

    # Row 2: Box plots by diagnosis
    ax2 = fig.add_subplot(gs[1, col_idx])
    diag_data = []
    diag_labels = []
    for dcode in ['1', '2', '3']:
        dname = DIAG_MAP[dcode]
        dvals = df.loc[mask & (df['DIAGNOSIS'] == dcode), dose_col].dropna()
        if len(dvals) > 0:
            diag_data.append(dvals.values)
            n_low_d = (dvals < thresh).sum()
            diag_labels.append(f'{dname}\n(N={len(dvals)}, low={n_low_d})')
    if diag_data:
        bp = ax2.boxplot(diag_data, labels=diag_labels, patch_artist=True,
                         widths=0.6, showmeans=True,
                         meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
        colors_box = ['#81D4FA', '#FFE082', '#EF9A9A']
        for patch, c in zip(bp['boxes'], colors_box[:len(bp['boxes'])]):
            patch.set_facecolor(c)
        ax2.axhline(thresh, color='red', ls='--', lw=1.5, alpha=0.7)
    ax2.set_title(f'{tracer} by Diagnosis', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Dose (mCi)')

    # Row 3: Yearly trend
    ax3 = fig.add_subplot(gs[2, col_idx])
    year_grp = df.loc[mask & df[dose_col].notna()].groupby('exam_year')[dose_col]
    years = sorted(year_grp.groups.keys())
    means = [year_grp.get_group(y).mean() for y in years]
    stds  = [year_grp.get_group(y).std()  for y in years]
    ns    = [len(year_grp.get_group(y))   for y in years]
    n_lows = [(year_grp.get_group(y) < thresh).sum() for y in years]
    ax3.errorbar(years, means, yerr=stds, fmt='o-', color=color, capsize=4, lw=2)
    ax3.axhline(thresh, color='red', ls='--', lw=1.5, alpha=0.7)
    for y, m, n, nl in zip(years, means, ns, n_lows):
        label = f'n={n}'
        if nl > 0:
            label += f'\nlow={nl}'
        ax3.annotate(label, (y, m), textcoords='offset points',
                     xytext=(0, 12), ha='center', fontsize=7)
    ax3.set_title(f'{tracer} Yearly Trend', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Dose (mCi)')
    ax3.set_xticks(years)

    # Print yearly stats
    rprint(f"  {tracer} yearly low-dose counts:")
    for y, nl, n in zip(years, n_lows, ns):
        if nl > 0:
            rprint(f"    {y}: {nl}/{n} ({100*nl/n:.1f}%)")

fig.suptitle('PET Injection Dose Analysis', fontsize=16, fontweight='bold', y=0.98)
fig.savefig(os.path.join(FIG_DIR, 'dose_distribution.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
rprint(f"\n  -> Saved {FIG_DIR}/dose_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2: Site-level analysis
# ═══════════════════════════════════════════════════════════════════════════════
rprint("\n" + "="*70)
rprint("MODULE 2: Site-Level Analysis")
rprint("="*70)

fig2, axes2 = plt.subplots(1, 3, figsize=(22, 8))

for ax, (tracer, dose_col, id_col, color) in zip(axes2, tracer_info):
    mask = df[id_col].notna() & (df[id_col] != '') & df[dose_col].notna()
    sub = df.loc[mask, ['site', dose_col]].copy()
    thresh = LOW_THRESH[tracer]

    site_stats = sub.groupby('site').agg(
        n=(dose_col, 'count'),
        mean=(dose_col, 'mean'),
        std=(dose_col, 'std'),
        n_low=(dose_col, lambda x: (x < thresh).sum())
    ).reset_index()
    site_stats['low_rate'] = site_stats['n_low'] / site_stats['n']
    site_stats = site_stats.sort_values('low_rate', ascending=False)

    # Flag sites with high low-dose rate (>20% AND n_low >= 2)
    flagged = site_stats[(site_stats['low_rate'] > 0.20) & (site_stats['n_low'] >= 2)]

    rprint(f"\n  {tracer}: {len(site_stats)} sites, {len(flagged)} flagged (low-dose rate > 20%)")
    if len(flagged):
        for _, r in flagged.iterrows():
            rprint(f"    Site {r['site']}: {int(r['n_low'])}/{int(r['n'])} "
                   f"({100*r['low_rate']:.0f}%), mean={r['mean']:.2f} mCi")

    # Plot: top 30 sites by sample count, coloured by low-dose rate
    top = site_stats.nlargest(30, 'n')
    bars = ax.bar(range(len(top)), top['mean'], yerr=top['std'],
                  color=[('red' if lr > 0.2 and nl >= 2 else color)
                         for lr, nl in zip(top['low_rate'], top['n_low'])],
                  alpha=0.7, capsize=2, edgecolor='white')
    ax.axhline(thresh, color='red', ls='--', lw=1.5)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top['site'], rotation=90, fontsize=7)
    ax.set_title(f'{tracer} Mean Dose by Site (top 30)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dose (mCi)')
    ax.set_xlabel('Site')
    # Annotate sample counts
    for i, (_, r) in enumerate(top.iterrows()):
        ax.annotate(f'n={int(r["n"])}', (i, r['mean'] + r['std'] + 0.1),
                    ha='center', fontsize=5, rotation=90)

fig2.suptitle('Dose by Acquisition Site (red = high low-dose rate)', fontsize=14, fontweight='bold')
fig2.tight_layout(rect=[0, 0, 1, 0.95])
fig2.savefig(os.path.join(FIG_DIR, 'dose_by_site.png'), dpi=150, bbox_inches='tight')
plt.close(fig2)
rprint(f"\n  -> Saved {FIG_DIR}/dose_by_site.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3: Temporal trend (detailed per-year breakdown)
# ═══════════════════════════════════════════════════════════════════════════════
rprint("\n" + "="*70)
rprint("MODULE 3: Temporal Trend (detailed)")
rprint("="*70)

fig3, axes3 = plt.subplots(1, 3, figsize=(20, 6))
for ax, (tracer, dose_col, id_col, color) in zip(axes3, tracer_info):
    mask = df[id_col].notna() & (df[id_col] != '') & df[dose_col].notna()
    sub = df.loc[mask].copy()
    thresh = LOW_THRESH[tracer]

    # Scatter plot: each sample as a dot, with yearly mean line overlay
    ax.scatter(sub['EXAMDATE_dt'], sub[dose_col], s=8, alpha=0.3, color=color)
    year_grp = sub.groupby('exam_year')[dose_col]
    years = sorted(year_grp.groups.keys())
    means = [year_grp.get_group(y).mean() for y in years]
    # Plot yearly mean as connected line at mid-year
    mid_dates = [datetime(int(y), 7, 1) for y in years]
    ax.plot(mid_dates, means, 'k-o', lw=2, markersize=5, zorder=5)
    ax.axhline(thresh, color='red', ls='--', lw=1.5)
    ax.set_title(f'{tracer} Dose Over Time', fontsize=12, fontweight='bold')
    ax.set_xlabel('Exam Date')
    ax.set_ylabel('Dose (mCi)')
    ax.tick_params(axis='x', rotation=30)

fig3.suptitle('Dose Temporal Scatter + Yearly Mean', fontsize=14, fontweight='bold')
fig3.tight_layout(rect=[0, 0, 1, 0.93])
fig3.savefig(os.path.join(FIG_DIR, 'dose_temporal.png'), dpi=150, bbox_inches='tight')
plt.close(fig3)
rprint(f"\n  -> Saved {FIG_DIR}/dose_temporal.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 4: Low-dose sample precise list
# ═══════════════════════════════════════════════════════════════════════════════
rprint("\n" + "="*70)
rprint("MODULE 4: Low-Dose Sample List")
rprint("="*70)

low_rows = []
for tracer, dose_col, id_col, _ in tracer_info:
    mask = (df[id_col].notna() & (df[id_col] != '') &
            df[dose_col].notna() & (df[dose_col] < LOW_THRESH[tracer]))
    sub = df.loc[mask].copy()
    for _, r in sub.iterrows():
        low_rows.append({
            'tracer':       tracer,
            'PTID':         r['PTID'],
            'EXAMDATE':     r['EXAMDATE'],
            'image_id':     r[id_col],
            'dose_mci':     r[dose_col],
            'threshold':    LOW_THRESH[tracer],
            'DIAGNOSIS':    r['DIAGNOSIS_label'],
            'site':         r['site'],
            'pT217_F':      r.get('pT217_F', ''),
            'AB42_AB40_F':  r.get('AB42_AB40_F', ''),
            'NfL_Q':        r.get('NfL_Q', ''),
            'GFAP_Q':       r.get('GFAP_Q', ''),
            'plasma_date_diff': r.get('plasma_date_diff', ''),
        })

low_df = pd.DataFrame(low_rows)
LOW_OUT = os.path.join(WORK, "analysis", "dose_low_samples.csv")
low_df.to_csv(LOW_OUT, index=False)
rprint(f"\n  Total low-dose samples: {len(low_df)}")
if len(low_df):
    rprint(f"  By tracer: {low_df.groupby('tracer').size().to_dict()}")
    rprint(f"  By diagnosis: {low_df.groupby('DIAGNOSIS').size().to_dict()}")
    rprint(f"  -> Saved {LOW_OUT}")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 5: Unmatched sample analysis
# ═══════════════════════════════════════════════════════════════════════════════
rprint("\n" + "="*70)
rprint("MODULE 5: Unmatched Sample Analysis")
rprint("="*70)

unmatched_rows = []
for tracer, dose_col, id_col, _ in tracer_info:
    # Unmatched = has image ID but dose is NaN
    mask = (df[id_col].notna() & (df[id_col] != '') & df[dose_col].isna())
    sub = df.loc[mask].copy()
    for _, r in sub.iterrows():
        unmatched_rows.append({
            'tracer':           tracer,
            'PTID':             r['PTID'],
            'EXAMDATE':         r['EXAMDATE'],
            'image_id':         r[id_col],
            'DIAGNOSIS':        r['DIAGNOSIS_label'],
            'site':             r['site'],
            'dose_match_visit': r.get('dose_match_visit', ''),
            'dose_match_day_diff': r.get('dose_match_day_diff', ''),
        })

un_df = pd.DataFrame(unmatched_rows)
UN_OUT = os.path.join(WORK, "analysis", "dose_unmatched.csv")
un_df.to_csv(UN_OUT, index=False)

rprint(f"\n  Total unmatched (has image, no dose): {len(un_df)}")
if len(un_df):
    rprint(f"  By tracer: {un_df.groupby('tracer').size().to_dict()}")
    rprint(f"  By diagnosis: {un_df.groupby('DIAGNOSIS').size().to_dict()}")
    # Compare diagnosis distribution: matched vs unmatched
    for tracer, dose_col, id_col, _ in tracer_info:
        has_img = df[id_col].notna() & (df[id_col] != '')
        matched   = df.loc[has_img & df[dose_col].notna(), 'DIAGNOSIS_label'].value_counts(normalize=True)
        unmatched = df.loc[has_img & df[dose_col].isna(),  'DIAGNOSIS_label'].value_counts(normalize=True)
        if len(unmatched):
            rprint(f"\n  {tracer} diagnosis distribution (matched vs unmatched):")
            for dx in ['CU', 'MCI', 'AD']:
                m_pct = matched.get(dx, 0) * 100
                u_pct = unmatched.get(dx, 0) * 100
                rprint(f"    {dx}: matched={m_pct:.1f}% vs unmatched={u_pct:.1f}%")
    # Year distribution of unmatched
    un_df['_year'] = pd.to_datetime(un_df['EXAMDATE'], errors='coerce').dt.year
    rprint(f"\n  Unmatched by year:")
    rprint(f"  {un_df.groupby(['tracer','_year']).size().to_string()}")
rprint(f"\n  -> Saved {UN_OUT}")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 6: Dose vs Body Weight
# ═══════════════════════════════════════════════════════════════════════════════
rprint("\n" + "="*70)
rprint("MODULE 6: Dose vs Body Weight")
rprint("="*70)

# Build weight lookup from dose CSV: subject_id → latest VSWEIGHT
dose_raw['VSWEIGHT_num'] = pd.to_numeric(dose_raw.get('VSWEIGHT', pd.Series(dtype=str)),
                                          errors='coerce')
weight_lookup = (dose_raw.dropna(subset=['VSWEIGHT_num'])
                 .groupby('subject_id')['VSWEIGHT_num']
                 .last()  # latest record
                 .to_dict())
rprint(f"\n  Subjects with weight data: {len(weight_lookup)}")

df['weight_kg'] = df['PTID'].map(weight_lookup)
df['has_weight'] = df['weight_kg'].notna()

fig6, axes6 = plt.subplots(1, 3, figsize=(20, 6))
for ax, (tracer, dose_col, id_col, color) in zip(axes6, tracer_info):
    mask = (df[id_col].notna() & (df[id_col] != '') &
            df[dose_col].notna() & df['has_weight'])
    sub = df.loc[mask].copy()
    sub['dose_per_kg'] = sub[dose_col] / sub['weight_kg']
    n = len(sub)
    if n == 0:
        ax.set_title(f'{tracer}: No weight data')
        continue

    rprint(f"\n  {tracer}: N with weight={n}")
    rprint(f"    Weight: mean={sub['weight_kg'].mean():.1f}±{sub['weight_kg'].std():.1f} kg")
    rprint(f"    Dose/kg: mean={sub['dose_per_kg'].mean():.4f}±{sub['dose_per_kg'].std():.4f} mCi/kg")

    # Correlation
    corr = sub[[dose_col, 'weight_kg']].corr().iloc[0, 1]
    rprint(f"    Corr(dose, weight): r={corr:.3f}")

    ax.scatter(sub['weight_kg'], sub[dose_col], s=10, alpha=0.4, color=color)
    # Fit line
    z = np.polyfit(sub['weight_kg'], sub[dose_col], 1)
    xline = np.linspace(sub['weight_kg'].min(), sub['weight_kg'].max(), 100)
    ax.plot(xline, np.polyval(z, xline), 'k--', lw=1.5)
    ax.set_title(f'{tracer} (N={n}, r={corr:.3f})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Body Weight (kg)')
    ax.set_ylabel('Dose (mCi)')
    ax.axhline(LOW_THRESH[tracer], color='red', ls='--', lw=1, alpha=0.7)

fig6.suptitle('Dose vs Body Weight', fontsize=14, fontweight='bold')
fig6.tight_layout(rect=[0, 0, 1, 0.93])
fig6.savefig(os.path.join(FIG_DIR, 'dose_vs_weight.png'), dpi=150, bbox_inches='tight')
plt.close(fig6)
rprint(f"\n  -> Saved {FIG_DIR}/dose_vs_weight.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 7: Training Data Impact & QC-passed Subset
# ═══════════════════════════════════════════════════════════════════════════════
rprint("\n" + "="*70)
rprint("MODULE 7: Training Data Impact Assessment")
rprint("="*70)

# Flag rows where ANY tracer has low dose
df['_low_tau'] = (df['dose_tau_mci'] < LOW_THRESH['TAU']) & df['id_av1451'].notna() & (df['id_av1451'] != '')
df['_low_av45'] = (df['dose_av45fp_mci'] < LOW_THRESH['AV45_FP']) & df['id_av45'].notna() & (df['id_av45'] != '')
df['_low_fdg'] = (df['dose_fdg_mci'] < LOW_THRESH['FDG']) & df['id_fdg'].notna() & (df['id_fdg'] != '')
# Fill NaN flags to False (unmatched dose → not flagged as low-dose, keep them)
df['_low_tau'] = df['_low_tau'].fillna(False)
df['_low_av45'] = df['_low_av45'].fillna(False)
df['_low_fdg'] = df['_low_fdg'].fillna(False)
df['any_low_dose'] = df['_low_tau'] | df['_low_av45'] | df['_low_fdg']

n_total = len(df)
n_flagged = df['any_low_dose'].sum()
n_passed = n_total - n_flagged

rprint(f"\n  Total rows: {n_total}")
rprint(f"  Low-dose flagged: {n_flagged} ({100*n_flagged/n_total:.1f}%)")
rprint(f"  QC-passed: {n_passed} ({100*n_passed/n_total:.1f}%)")

# Diagnosis distribution comparison
rprint(f"\n  Diagnosis distribution (original → QC-passed):")
for dx in ['CU', 'MCI', 'AD']:
    n_orig = (df['DIAGNOSIS_label'] == dx).sum()
    n_qc   = ((df['DIAGNOSIS_label'] == dx) & ~df['any_low_dose']).sum()
    rprint(f"    {dx}: {n_orig} ({100*n_orig/n_total:.1f}%) → "
           f"{n_qc} ({100*n_qc/n_passed:.1f}%)  [removed {n_orig - n_qc}]")

# Per-tracer image count comparison
rprint(f"\n  Per-tracer image counts (original → QC-passed):")
for tracer, dose_col, id_col, _ in tracer_info:
    has_img = df[id_col].notna() & (df[id_col] != '')
    n_orig = has_img.sum()
    n_qc = (has_img & ~df['any_low_dose']).sum()
    rprint(f"    {tracer}: {n_orig} → {n_qc} (removed {n_orig - n_qc})")

# Write QC-passed CSV (original report columns, no internal flags)
qc_df = df.loc[~df['any_low_dose']].copy()
# Output columns = same as original report
out_cols = ['PTID', 'EXAMDATE', 'DIAGNOSIS', 'id_mri', 'id_fdg', 'id_av45', 'id_av1451',
            'pT217_F', 'AB42_F', 'AB40_F', 'AB42_AB40_F', 'pT217_AB42_F',
            'NfL_Q', 'GFAP_Q', 'plasma_source', 'plasma_date', 'plasma_date_diff',
            'dose_match_visit', 'dose_match_est_date', 'dose_match_day_diff',
            'dose_tau_mci', 'low_dose_tau',
            'dose_av45fbn_mci', 'low_dose_av45fbn',
            'dose_av45fp_mci', 'low_dose_av45fp',
            'dose_fdg_mci', 'low_dose_fdg']
QC_OUT = os.path.join(WORK, "analysis", "pairs_dose_qc_passed.csv")
qc_df[out_cols].to_csv(QC_OUT, index=False)
rprint(f"\n  -> Saved {QC_OUT} ({len(qc_df)} rows)")


# ── Write summary report ─────────────────────────────────────────────────────
SUMMARY_OUT = os.path.join(WORK, "analysis", "dose_summary_report.txt")
with open(SUMMARY_OUT, 'w') as f:
    f.write('\n'.join(report_lines))
rprint(f"\n  -> Saved {SUMMARY_OUT}")
print("\n✓ All modules complete.")
