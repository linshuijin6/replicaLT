import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import json
from scipy.stats import gaussian_kde

# === 基础设定 ===
# model_pet = 'fdg'  # or 'av45'
model_pet = 'av45'  # or 'fdg'

size_title = 28
size_label = 26
size_small = 20
output_dir = f"/home/ssddata/liutuo/liutuo_data/shuijin/statistics_of_regions_{model_pet}_adjust"
json_file = "brain_labels.json"

# for av45
# id_list = [1025, 2025, 1009, 1015, 1030, 1034, 2009, 2015, 2030, 2034]  # related av45, all
# id_list = [1025, 1009, 1015, 1030, 1034] # only lh
# id_list = [2025, 2009, 2015, 2030, 2034]  # the others
id_list = [ 28, 60, 53, 7, 17]  # no related 

# for fdg
# id_list = [1010, 2010, 1031, 17, 53] # related fdg
# id_list = [2031] # the others

# id_list = [10, 49, 1019, 1011, 1013]  # no related 
search_list = [f"region_{i}.0.csv" for i in id_list]

# 加载脑区名称
with open(json_file, "r", encoding="utf-8") as f:
    brain_dict = json.load(f)

def get_significance_label(p):
    if p < 0.001: return "****"
    elif p < 0.01: return "***"
    elif p < 0.05: return "**"
    else: return "ns"

# === 半小提琴图函数 ===
def half_violin(ax, data_left, data_right, x_pos, color_left, color_right, hatch_right=None, bw_method=0.4):
    """
    绘制半小提琴图：左为Real，右为Syn
    """
    if len(data_left) > 1:
        kde_left = gaussian_kde(data_left, bw_method=bw_method)
        y = np.linspace(min(data_left.min(), data_right.min()),
                        max(data_left.max(), data_right.max()), 200)
        v_left = kde_left(y)
        v_left = v_left / v_left.max() * 0.25
        ax.fill_betweenx(y, x_pos - v_left, x_pos, facecolor=color_left, alpha=0.7, linewidth=0)
        ax.plot(x_pos - v_left, y, color=color_left, lw=1.2)

    if len(data_right) > 1:
        kde_right = gaussian_kde(data_right, bw_method=bw_method)
        y = np.linspace(min(data_left.min(), data_right.min()),
                        max(data_left.max(), data_right.max()), 200)
        v_right = kde_right(y)
        v_right = v_right / v_right.max() * 0.25
        ax.fill_betweenx(y, x_pos, x_pos + v_right, facecolor='none',
                         edgecolor=color_right, alpha=0.5, linewidth=1.2, hatch=hatch_right)
        ax.plot(x_pos + v_right, y, color=color_right, lw=1.2)

# === 新增函数：绘制显著性中括号 ===
def add_bracket(ax, x1, x2, y, h, text, linestyle='solid'):
    """
    绘制横向中括号式显著性标记
    """
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color='black', lw=1.4, linestyle=linestyle)
    ax.text((x1+x2)/2, y+h+0.02, text, ha='center', va='bottom', fontsize=size_label)

# === 主分析函数 ===
def analyze_cn_vs_ad():
    GROUPS = ["CN", "AD"]
    # csv_files = [f for f in os.listdir(output_dir) if f in search_list]
    csv_files = search_list
    all_data = []
    real_pvals, syn_pvals = {}, {}

    for csv_file in csv_files:
        csv_path = os.path.join(output_dir, csv_file)
        df = pd.read_csv(csv_path)
        if "Research Group" not in df.columns:
            continue

        region = os.path.splitext(csv_file)[0].replace("region_", "")
        region_name = brain_dict.get(region.split(".")[0], region)

        df_filtered = df[df["Research Group"].isin(GROUPS)]
        if len(df_filtered) == 0:
            continue

        cn_real = df_filtered[df_filtered["Research Group"] == "CN"][f"real_{model_pet}_mean"].dropna()
        ad_real = df_filtered[df_filtered["Research Group"] == "AD"][f"real_{model_pet}_mean"].dropna()
        cn_syn  = df_filtered[df_filtered["Research Group"] == "CN"][f"syn_{model_pet}_mean"].dropna()
        ad_syn  = df_filtered[df_filtered["Research Group"] == "AD"][f"syn_{model_pet}_mean"].dropna()

        if len(cn_real) and len(ad_real):
            _, p_real = ttest_ind(cn_real, ad_real, equal_var=False)
            real_pvals[region_name] = p_real
        if len(cn_syn) and len(ad_syn):
            _, p_syn = ttest_ind(cn_syn, ad_syn, equal_var=False)
            syn_pvals[region_name] = p_syn

        all_data.append((region_name, cn_real, ad_real, cn_syn, ad_syn))

    # === 绘图 ===
    fig, ax = plt.subplots(figsize=(20, 16))
    sns.set(style="whitegrid", context="talk")

    colors = {"CN": "#3CB371", "AD": "#DC143C"}  # CN绿、AD红
    x_ticks = []

    for i, (region, cn_real, ad_real, cn_syn, ad_syn) in enumerate(all_data):
        x_pos = i

        # 左右分离更大：防止重叠
        half_violin(ax, cn_real, cn_syn, x_pos - 0.25, colors["CN"], colors["CN"], hatch_right="//")
        half_violin(ax, ad_real, ad_syn, x_pos + 0.25, colors["AD"], colors["AD"], hatch_right="//")

        # 显著性标注（中括号方式）
        y_max = max(max(cn_real.max(), ad_real.max()), max(cn_syn.max(), ad_syn.max()))
        y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03

        real_label = get_significance_label(real_pvals.get(region, 1))
        syn_label  = get_significance_label(syn_pvals.get(region, 1))

        # 实填显著（实线）
        add_bracket(ax, x_pos - 0.35, x_pos + 0.35, y_max + y_offset, 0.05, real_label, linestyle='solid')
        # 虚填显著（虚线，略高）
        add_bracket(ax, x_pos - 0.35, x_pos + 0.35, y_max + y_offset + 0.12, 0.05, syn_label, linestyle='dashed')

        x_ticks.append(region)

    ax.set_xticks(range(len(x_ticks)))
    ax.set_xticklabels(x_ticks, rotation=45, ha="center", fontsize=size_label)
    ax.set_ylabel(f"Mean {model_pet.upper()} Uptake", fontsize=size_label)
    ax.set_yticks([0,0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(ax.get_yticks(), fontsize=size_label)

    ax.set_xlabel("Brain Region", fontsize=size_label)
    ax.set_title("CN vs AD: Left = Real, Right = Syn (Slash = Synthetic Data)", fontsize=size_title, pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 图例
    legend_handles = [
        plt.Line2D([], [], color=colors["CN"], lw=8, label="CN (Healthy)"),
        plt.Line2D([], [], color=colors["AD"], lw=8, label="AD (Patient)"),
        plt.Line2D([], [], color="k", lw=0, label="Slash = Synthetic Data"),
        plt.Line2D([], [], color="k", lw=1.4, linestyle='solid', label="Solid Bracket: Real Significance"),
        plt.Line2D([], [], color="k", lw=1.4, linestyle='dashed', label="Dashed Bracket: Syn Significance"),

    ]
    ax.legend(handles=legend_handles, fontsize=size_small, loc="best")

    plt.tight_layout()
    out_path = os.path.join('.', "CN_vs_AD_half_violin_real_syn_bracket.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)
    print(f"✅ 图像已保存：{out_path}")
    plt.show()

if __name__ == "__main__":
    analyze_cn_vs_ad()
