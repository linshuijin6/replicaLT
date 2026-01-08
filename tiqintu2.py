import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import json

# 设置路径
output_dir = "/home/ssddata/liutuo/liutuo_data/shuijin/statistics_of_regions_av45_adjust"

# 设置统计脑区编号
id_list = [1025, 2025, 1009, 1015, 1030, 1034, 2009, 2015, 2030, 2034]


# 导入脑区编号与名称的键值对
json_file = 'brain_labels.json'
search_list = [f'region_{i}.0.csv' for i in id_list]  # 假设有200个区域

with open(json_file, "r", encoding="utf-8") as f:
    brain_dict = json.load(f)
def get_significance_label(p_val):
    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    else:
        return "ns"

def analyze_cn_vs_mci():
    csv_files = [f for f in os.listdir(output_dir) if f.endswith(".csv")]
    if not csv_files:
        print("No CSV files found.")
        return

    # 可选：只处理特定区域，比如从第86个开始（按你原逻辑）
    # csv_files = sorted(csv_files)[86:]
    csv_files = search_list

    print(f"Analyzing {len(csv_files)} regions: {csv_files}")

    # 定义要比较的组
    GROUPS_OF_INTEREST = ['AD', 'CN']
    # 如果你的MCI分散在EMCI/LMCI，取消注释下面这行
    # GROUP_MAP = {'EMCI': 'MCI', 'LMCI': 'MCI', 'MCI': 'MCI', 'CN': 'CN'}

    all_real_data = []
    all_syn_data = []
    real_sig_labels = {}
    syn_sig_labels = {}

    for csv_file in csv_files:
        csv_path = os.path.join(output_dir, csv_file)
        if not os.path.exists(csv_path):
            continue

        region = os.path.splitext(csv_file)[0].replace("region_", "")
        region = brain_dict.get(region.split('.')[0], None)  # 使用 JSON 映射名称
        df = pd.read_csv(csv_path)

        # 确保有 Research Group 列
        if 'Research Group' not in df.columns:
            print(f"Warning: {csv_file} missing 'Research Group', skip.")
            continue

        # 可选：合并 EMCI/LMCI 到 MCI
        # df['Research Group'] = df['Research Group'].map(GROUP_MAP).dropna()

        # 只保留 CN 和 MCI
        df_filtered = df[df['Research Group'].isin(GROUPS_OF_INTEREST)].copy()
        if len(df_filtered) < 2:
            print(f"Skip {region}: not enough CN/MCI samples.")
            continue

        # 确保两组都存在
        groups_present = df_filtered['Research Group'].unique()
        if not {f'{GROUPS_OF_INTEREST[0]}', f'{GROUPS_OF_INTEREST[1]}'}.issubset(groups_present):
            print(f"Skip {region}: missing CN or MCI.")
            continue

        # 提取 real 和 syn 数据（长格式）
        real_long = df_filtered[['Research Group', 'real_av45_mean']].rename(
            columns={'real_av45_mean': 'Value'}
        )
        real_long['Region'] = region
        real_long['Metric'] = 'Real AV45'
        all_real_data.append(real_long)

        syn_long = df_filtered[['Research Group', 'syn_av45_mean']].rename(
            columns={'syn_av45_mean': 'Value'}
        )
        syn_long['Region'] = region
        syn_long['Metric'] = 'Syn AV45'
        all_syn_data.append(syn_long)

        # === t 检验: CN vs MCI for real ===
        cn_real = df_filtered[df_filtered['Research Group'] == 'CN']['real_av45_mean'].dropna()
        mci_real = df_filtered[df_filtered['Research Group'] == 'MCI']['real_av45_mean'].dropna()
        if len(cn_real) > 0 and len(mci_real) > 0:
            _, p_real = ttest_ind(cn_real, mci_real, equal_var=False)
            real_sig_labels[region] = get_significance_label(p_real)
        else:
            real_sig_labels[region] = "ns"

        # === t 检验: CN vs MCI for syn ===
        cn_syn = df_filtered[df_filtered['Research Group'] == 'CN']['syn_av45_mean'].dropna()
        mci_syn = df_filtered[df_filtered['Research Group'] == 'MCI']['syn_av45_mean'].dropna()
        if len(cn_syn) > 0 and len(mci_syn) > 0:
            _, p_syn = ttest_ind(cn_syn, mci_syn, equal_var=False)
            syn_sig_labels[region] = get_significance_label(p_syn)
        else:
            syn_sig_labels[region] = "ns"

    if not all_real_data:
        print("No valid data for plotting.")
        return

    # 合并数据
    df_real = pd.concat(all_real_data, ignore_index=True)
    df_syn = pd.concat(all_syn_data, ignore_index=True)

    # 绘图函数
    def plot_metric(df, sig_labels, title, filename):
        plt.figure(figsize=(18, 10))
        sns.set_theme(style="whitegrid")
        palette = {"CN": "#2E86AB", "MCI": "#A23B72"}  # 蓝 vs 紫

        ax = sns.violinplot(
            data=df,
            x="Region",
            y="Value",
            hue="Research Group",
            inner="box",
            split=False,  # 不 split，因为不是配对数据
            palette=palette,
            linewidth=1.2
        )

        # 标注显著性（在每组区域上方）
        regions = df["Region"].unique()
        for i, region in enumerate(regions):
            label = sig_labels.get(region, "ns")
            color = "black" if label != "ns" else "gray"
            # 计算该区域最大值
            max_val = df[df["Region"] == region]["Value"].max()
            y_pos = max_val + (df["Value"].max() - df["Value"].min()) * 0.05  # 动态偏移
            plt.text(i, y_pos, label, ha="center", va="bottom",
                     fontsize=14, color=color, fontweight="bold")

        plt.title(title, fontsize=18, pad=20)
        plt.xlabel("Brain Region", fontsize=14)
        plt.ylabel("Mean AV45 Uptake", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title="Group", title_fontsize=12, fontsize=12, loc="upper right")

        plt.tight_layout()
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {plot_path}")
        plt.show()

    # 绘制两个图
    plot_metric(
        df_real, real_sig_labels,
        f"{GROUPS_OF_INTEREST[0]} vs {GROUPS_OF_INTEREST[1]}: Real AV45 Uptake Across Brain Regions",
        "cn_vs_mci_real_av45_violin.png"
    )

    plot_metric(
        df_syn, syn_sig_labels,
        f"{GROUPS_OF_INTEREST[0]} vs f{GROUPS_OF_INTEREST[1]}: Synthetic AV45 Uptake Across Brain Regions",
        "cn_vs_mci_syn_av45_violin.png"
    )

if __name__ == "__main__":
    analyze_cn_vs_mci()