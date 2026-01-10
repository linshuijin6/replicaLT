"""
内存使用对比可视化脚本
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据（单位：GB）
methods = ['原方案\n(同时三模态)', '顺序模态', '顺序+\n混合精度', '顺序+\n梯度累积(2)', '双GPU\n数据并行']
memory_usage = [30, 15, 10, 8, 4]  # 每个方案的显存占用
colors = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#3498db']

# 创建图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 图1: 柱状图
bars = ax1.bar(range(len(methods)), memory_usage, color=colors, alpha=0.8, edgecolor='black')
ax1.set_ylabel('显存占用 (GB)', fontsize=12, fontweight='bold')
ax1.set_title('不同优化方案的显存占用对比', fontsize=14, fontweight='bold')
ax1.set_xticks(range(len(methods)))
ax1.set_xticklabels(methods, fontsize=10)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# 在柱子上标注数值
for bar, mem in zip(bars, memory_usage):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{mem}GB',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# 添加基准线
ax1.axhline(y=24, color='red', linestyle='--', linewidth=2, alpha=0.5, label='24GB显卡上限')
ax1.legend(fontsize=10)

# 图2: 节省比例
savings = [0, 50, 67, 73, 87]  # 相对于原方案的节省百分比
bars2 = ax2.barh(range(len(methods)), savings, color=colors, alpha=0.8, edgecolor='black')
ax2.set_xlabel('显存节省比例 (%)', fontsize=12, fontweight='bold')
ax2.set_title('相对于原方案的显存节省', fontsize=14, fontweight='bold')
ax2.set_yticks(range(len(methods)))
ax2.set_yticklabels(methods, fontsize=10)
ax2.grid(axis='x', alpha=0.3, linestyle='--')

# 标注节省比例
for bar, saving in zip(bars2, savings):
    width = bar.get_width()
    if width > 0:
        ax2.text(width + 2, bar.get_y() + bar.get_height()/2.,
                 f'{saving}%',
                 ha='left', va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('memory_comparison.png', dpi=300, bbox_inches='tight')
print("✓ 已保存内存对比图: memory_comparison.png")

# 创建第二个图：训练速度对比
fig2, ax = plt.subplots(figsize=(10, 6))

methods2 = ['原方案', '顺序模态', '顺序+\n混合精度', '顺序+\n梯度累积(2)', '双GPU']
speeds = [1.0, 0.6, 0.7, 0.5, 1.5]  # 相对速度（原方案=1.0）
memories = [30, 15, 10, 8, 4]

scatter = ax.scatter(memories, speeds, s=[500]*len(methods2), 
                    c=colors, alpha=0.6, edgecolors='black', linewidth=2)

# 标注每个点
for i, method in enumerate(methods2):
    ax.annotate(method, (memories[i], speeds[i]), 
               textcoords="offset points", xytext=(0,10), 
               ha='center', fontsize=10, fontweight='bold')

ax.set_xlabel('显存占用 (GB)', fontsize=12, fontweight='bold')
ax.set_ylabel('相对训练速度', fontsize=12, fontweight='bold')
ax.set_title('显存占用 vs 训练速度权衡', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')

# 添加理想区域标注
ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.3, label='基准速度')
ax.axvline(x=24, color='red', linestyle='--', alpha=0.3, label='24GB上限')
ax.fill_between([0, 15], 0.8, 2.0, alpha=0.1, color='green', label='理想区域')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('speed_vs_memory.png', dpi=300, bbox_inches='tight')
print("✓ 已保存速度对比图: speed_vs_memory.png")

# 创建第三个图：不同GPU配置的对比
fig3, ax = plt.subplots(figsize=(12, 6))

gpu_configs = [
    '单GPU\n24GB',
    '单GPU\n16GB',
    '双GPU\n24GB×2',
    '双GPU\n16GB×2',
    '四GPU\n24GB×4'
]

config_memory = [10, 8, 6, 4, 3]  # 单卡显存占用
config_speed = [0.7, 0.5, 1.5, 1.3, 2.5]  # 相对训练速度
config_colors = ['#3498db', '#9b59b6', '#2ecc71', '#f39c12', '#e74c3c']

bars = ax.bar(range(len(gpu_configs)), config_speed, color=config_colors, 
              alpha=0.8, edgecolor='black', linewidth=2)

ax.set_ylabel('相对训练速度', fontsize=12, fontweight='bold')
ax.set_title('不同GPU配置的性能对比', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(gpu_configs)))
ax.set_xticklabels(gpu_configs, fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='单GPU基准速度')
ax.legend(fontsize=10)

# 在柱子上标注显存和速度
for i, (bar, mem, speed) in enumerate(zip(bars, config_memory, config_speed)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{speed}x\n{mem}GB/卡',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('gpu_config_comparison.png', dpi=300, bbox_inches='tight')
print("✓ 已保存GPU配置对比图: gpu_config_comparison.png")

print("\n所有可视化图表已生成！")
print("1. memory_comparison.png - 内存占用对比")
print("2. speed_vs_memory.png - 速度与内存权衡")
print("3. gpu_config_comparison.png - GPU配置对比")
