import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import json


n=0
# extended data gen
n = 6
n_per_plot = 15
size_plot = (30,12)


# 定义路径
size_title = 28
size_label = 26
size_small = 20

# pet_name = 'AV45'  # 或 'AV45'
pet_name = 'FDG'  # 或 'AV45'

output_dir = f"./statistics_of_regions_{pet_name.lower()}_adjust"
plot_path = f"{n}_fig4_{pet_name.lower()}.png"

json_file = 'brain_labels.json'
id_list = [1025, 2025, 1032, 2032, 1033]
search_list = [f'region_{i}.0.csv' for i in id_list]  # 假设有200个区域

with open(json_file, "r", encoding="utf-8") as f:
    brain_dict = json.load(f)

# id_name = {'1010':'ctx-lh-isthmuscingulate', '2010':'ctx-rh-isthmuscingulate', '1031':'ctx-lh-supramarginal', '2031':'ctx-rh-supramarginal', '17':'Left-Hippocampus', '53':'Right-Hippocampus', '11':'Left-Caudate', '50':'Right-Caudate', '13':'Left-Putamen', '52':'Right-Putamen', '12':'Left-Pallidum', '51':'Right-Pallidum', '10':'Left-Thalamus-Proper', '49':'Right-Thalamus-Proper'}

# 加载 CSV 文件
# 加载 CSV
def load_csv(file_path):
    return pd.read_csv(file_path)

def analyze_and_plot():
    # 获取所有 CSV
    csv_files = [f for f in os.listdir(output_dir) if f.endswith(".csv") and f not in search_list]
    if not csv_files:
        print("No CSV files found.")
        return

    # 从第 86 个开始（你可以根据需要调整）
    if n > 0:
        csv_files = sorted(csv_files)[n_per_plot*(n-1):n_per_plot*n]  # 推荐加 sorted 保证顺序
    else:
        csv_files = search_list
    print(f"Analyzing {len(csv_files)} regions: {csv_files}")

    all_data = []
    significance_labels = {}

    for csv_file in csv_files:
        csv_path = os.path.join(output_dir, csv_file)
        if not os.path.exists(csv_path):
            print(f"Skip missing file: {csv_path}")
            continue

        # 提取区域名：如 "region_prefrontal.csv" -> "prefrontal"
        region = os.path.splitext(csv_file)[0].replace("region_", "")
        region = brain_dict.get(region.split('.')[0], None)  # 使用 JSON 映射名称

        # 读取数据
        df = load_csv(csv_path)
        real_FDG = df[f"real_{pet_name.lower()}_mean"].values
        syn_FDG = df[f"syn_{pet_name.lower()}_mean"].values

        # t 检验（Welch's t-test）
        t_stat, p_val = ttest_ind(real_FDG, syn_FDG, equal_var=False)

        # 显著性标签
        if p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        else:
            sig = "ns"
        significance_labels[region] = sig

        # 打印结果
        print(f"{region}: t={t_stat:.3f}, p={p_val:.4f} → {sig}")

        # 转为长格式
        data_long = pd.DataFrame({
            f"Real {pet_name}": real_FDG,
            f"Syn {pet_name}": syn_FDG
        }).melt(var_name="Type", value_name="Mean Value")
        data_long["Region"] = region
        all_data.append(data_long)

    if not all_data:
        print("No valid data to plot.")
        return

    # 合并数据
    combined_df = pd.concat(all_data, ignore_index=True)

    # 绘图设置
    sns.set_theme(style="white")
    palette = {f"Real {pet_name}": "#3498DB", f"Syn {pet_name}": "#E74C3C"}
    plt.figure(figsize=size_plot)

    # 小提琴图 + 箱线图
    ax = sns.violinplot(
        data=combined_df,
        x="Region",
        y="Mean Value",
        hue="Type",
        inner=None,
        split=True,
        palette=palette,
        linewidth=1.2,
        saturation=0.9
    )

    # 动态标注显著性（每个区域独立高度）
    regions = combined_df["Region"].unique()
    for i, region in enumerate(regions):
        label = significance_labels[region]
        color = "black" if label != "ns" else "gray"

        # 获取该区域最大值，向上偏移 0.05（根据你的数据尺度调整）
        region_max = combined_df[combined_df["Region"] == region]["Mean Value"].max()
        y_pos = region_max + 0.05

        plt.text(i, y_pos, label, ha="center", va="bottom",
                 fontsize=size_label, color=color, fontweight="bold")

    # 图形美化
    plt.title(f"Comparison of Real vs. Synthetic {pet_name} Uptake Across Brain Regions", 
              fontsize=size_title, pad=20)
    plt.xlabel("Brain Region", fontsize=size_label)
    plt.ylabel(f"Mean {pet_name} Uptake", fontsize=size_label)
    plt.yticks(np.arange(0, 1.01, 0.2), fontsize=size_label)
    plt.xticks(rotation=45, ha="right", fontsize=size_label)
    
    plt.legend(fontsize=size_small, 
               loc="lower left", frameon=True)

    # 保存
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", transparent=True)

    print(f"\nPlot saved to: {plot_path}")
    plt.show()

# 运行
if __name__ == "__main__":
    analyze_and_plot()