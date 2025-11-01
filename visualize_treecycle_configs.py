#!/usr/bin/env python3
"""
TreeCycle 配置可视化对比

生成一个展示不同配置规模的对比图
"""

import matplotlib.pyplot as plt
import numpy as np

# 配置数据
configs = {
    '小规模-1\n(d=4,bf=15)': {'nodes': 54_241, 'edges': 1_138_782, 'memory': 0.02, 'time': 10},
    '小规模-2\n(d=5,bf=10)': {'nodes': 111_111, 'edges': 2_333_108, 'memory': 0.04, 'time': 20},
    '当前配置\n(d=5,bf=15)': {'nodes': 813_616, 'edges': 17_085_657, 'memory': 0.29, 'time': 60},
    '大规模-1\n(d=6,bf=15)': {'nodes': 12_204_241, 'edges': 256_288_770, 'memory': 4.34, 'time': 120},
    '大规模-2\n(d=6,bf=20)': {'nodes': 67_368_421, 'edges': 1_414_736_476, 'memory': 23.98, 'time': 240},
    '超大规模\n(d=6,bf=25)': {'nodes': 254_313_151, 'edges': 5_340_575_710, 'memory': 90.54, 'time': 480},
}

config_names = list(configs.keys())
nodes = [configs[c]['nodes'] for c in config_names]
edges = [configs[c]['edges'] for c in config_names]
memory = [configs[c]['memory'] for c in config_names]
time_mins = [configs[c]['time'] for c in config_names]

# 颜色
colors = ['#4CAF50', '#8BC34A', '#FFD700', '#FF9800', '#FF5722', '#9C27B0']
highlight_color = '#FFD700'  # 当前配置高亮

# 创建 2x2 子图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('TreeCycle 图配置对比', fontsize=16, fontweight='bold')

# 1. 节点数对比（对数刻度）
ax1 = axes[0, 0]
bars1 = ax1.bar(range(len(config_names)), nodes, color=colors)
bars1[2].set_color(highlight_color)  # 高亮当前配置
bars1[2].set_edgecolor('red')
bars1[2].set_linewidth(2)
ax1.set_yscale('log')
ax1.set_ylabel('节点数 (log scale)', fontsize=12)
ax1.set_title('节点数对比', fontsize=13, fontweight='bold')
ax1.set_xticks(range(len(config_names)))
ax1.set_xticklabels(config_names, fontsize=9, rotation=0)
ax1.grid(axis='y', alpha=0.3)

# 添加数值标签
for i, (bar, val) in enumerate(zip(bars1, nodes)):
    if val < 1_000_000:
        label = f"{val/1000:.0f}K"
    else:
        label = f"{val/1_000_000:.1f}M"
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), label,
             ha='center', va='bottom', fontsize=8, fontweight='bold')

# 2. 边数对比（对数刻度）
ax2 = axes[0, 1]
bars2 = ax2.bar(range(len(config_names)), edges, color=colors)
bars2[2].set_color(highlight_color)
bars2[2].set_edgecolor('red')
bars2[2].set_linewidth(2)
ax2.set_yscale('log')
ax2.set_ylabel('边数 (log scale)', fontsize=12)
ax2.set_title('边数对比', fontsize=13, fontweight='bold')
ax2.set_xticks(range(len(config_names)))
ax2.set_xticklabels(config_names, fontsize=9, rotation=0)
ax2.grid(axis='y', alpha=0.3)

# 添加数值标签
for i, (bar, val) in enumerate(zip(bars2, edges)):
    if val < 1_000_000:
        label = f"{val/1000:.0f}K"
    elif val < 1_000_000_000:
        label = f"{val/1_000_000:.0f}M"
    else:
        label = f"{val/1_000_000_000:.1f}B"
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), label,
             ha='center', va='bottom', fontsize=8, fontweight='bold')

# 3. 内存需求对比
ax3 = axes[1, 0]
bars3 = ax3.bar(range(len(config_names)), memory, color=colors)
bars3[2].set_color(highlight_color)
bars3[2].set_edgecolor('red')
bars3[2].set_linewidth(2)
ax3.set_ylabel('估算内存 (GB)', fontsize=12)
ax3.set_title('内存需求对比', fontsize=13, fontweight='bold')
ax3.set_xticks(range(len(config_names)))
ax3.set_xticklabels(config_names, fontsize=9, rotation=0)
ax3.grid(axis='y', alpha=0.3)

# 添加数值标签
for bar, val in zip(bars3, memory):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val:.2f}",
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# 4. 生成时间对比
ax4 = axes[1, 1]
bars4 = ax4.bar(range(len(config_names)), [t/60 for t in time_mins], color=colors)  # 转换为小时
bars4[2].set_color(highlight_color)
bars4[2].set_edgecolor('red')
bars4[2].set_linewidth(2)
ax4.set_ylabel('估算时间 (小时)', fontsize=12)
ax4.set_title('生成时间对比', fontsize=13, fontweight='bold')
ax4.set_xticks(range(len(config_names)))
ax4.set_xticklabels(config_names, fontsize=9, rotation=0)
ax4.grid(axis='y', alpha=0.3)

# 添加数值标签
for bar, val in zip(bars4, time_mins):
    if val < 60:
        label = f"{val}min"
    else:
        label = f"{val/60:.1f}h"
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(), label,
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# 添加图例说明
fig.text(0.5, 0.02, 
         '⭐ 黄色高亮: 当前配置 (d=5, bf=15, cp=0.2) | '
         '🟢 绿色: 推荐测试配置 | '
         '🟠 橙/红色: 大规模配置',
         ha='center', fontsize=10, style='italic',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout(rect=[0, 0.05, 1, 0.96])
plt.savefig('treecycle_config_comparison.png', dpi=150, bbox_inches='tight')
print("✓ 图表已保存: treecycle_config_comparison.png")

# 创建 Scalability 曲线
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('TreeCycle Scalability 曲线', fontsize=16, fontweight='bold')

# 节点数 vs 边数
ax1 = axes2[0]
ax1.scatter(nodes, edges, c=colors, s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
ax1.scatter([nodes[2]], [edges[2]], c=[highlight_color], s=300, 
            marker='*', edgecolors='red', linewidth=2, label='当前配置')

# 添加标签
for i, name in enumerate(config_names):
    # 简化标签
    short_name = name.split('\n')[0]
    ax1.annotate(short_name, (nodes[i], edges[i]), 
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('节点数', fontsize=12)
ax1.set_ylabel('边数', fontsize=12)
ax1.set_title('节点数 vs 边数（对数刻度）', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 节点数 vs 内存
ax2 = axes2[1]
ax2.scatter(nodes, memory, c=colors, s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
ax2.scatter([nodes[2]], [memory[2]], c=[highlight_color], s=300, 
            marker='*', edgecolors='red', linewidth=2, label='当前配置')

# 添加标签
for i, name in enumerate(config_names):
    short_name = name.split('\n')[0]
    ax2.annotate(short_name, (nodes[i], memory[i]), 
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

ax2.set_xscale('log')
ax2.set_xlabel('节点数', fontsize=12)
ax2.set_ylabel('估算内存 (GB)', fontsize=12)
ax2.set_title('节点数 vs 内存需求', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('treecycle_scalability_curves.png', dpi=150, bbox_inches='tight')
print("✓ 图表已保存: treecycle_scalability_curves.png")

print("\n" + "="*70)
print("可视化完成！")
print("="*70)
print("生成的文件:")
print("  1. treecycle_config_comparison.png - 配置对比（4个子图）")
print("  2. treecycle_scalability_curves.png - Scalability 曲线（2个子图）")
print("="*70)
