import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import rcParams

# 设置全局字体为 Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 8
rcParams['axes.labelsize'] = 9
rcParams['axes.titlesize'] = 9
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['legend.fontsize'] = 6

# 定义一致的颜色方案和图案（降低饱和度）
COLORS = {
    'ApxIChase': '#5B9BD5',      # 柔和蓝色
    'HeuIChase': '#ED7D31',      # 柔和橙色
    'GNNExplainer': '#70AD47',   # 柔和绿色
    'PGExplainer': '#E15759',    # 柔和红色
    'Exhaustive': '#A682B3',     # 柔和紫色
}

# 柱状图填充图案
HATCHES = {
    'ApxIChase': '///',
    'HeuIChase': '\\\\\\',
    'GNNExplainer': 'xxx',
    'PGExplainer': '...',
    'Exhaustive': '+++',
}

# 折线图标记
MARKERS = {
    'ApxIChase': '^',      # 三角形
    'HeuIChase': 's',      # 方框
    'GNNExplainer': 'o',   # 圆圈
    'PGExplainer': 'x',    # 叉叉
    'Exhaustive': 'D',     # 钻石
}

# 方法名称缩写映射（用于图例）
LEGEND_LABELS = {
    'ApxIChase': 'ApxC',
    'HeuIChase': 'HeuC',
    'GNNExplainer': 'GEX',
    'PGExplainer': 'PGX',
    'Exhaustive': 'Exh',
}

# 图表尺寸
FIG_WIDTH = 3.5  # inches (约 8.9 cm)
FIG_HEIGHT = 2.6  # inches (约 6.6 cm)

figure_1 = {
    "Dataset": ["MUTAG", "ATLAS", "Cora", "BAShape"],
    "ApxIChase": [3.5, 52.22, 157.36, 270.34],
    "HeuIChase": [0.91, 3.36, 3.09, 22.82],
    "GNNExplainer": [1.71, 3.46, 2.06, 9.34],
    "PGExplainer": [1.62, 31.17, 31.61, None],  # None 表示缺失值
    "Exhaustive": [124.87, 531.36, 1662.93, None],
}
df_figure_1 = pd.DataFrame(figure_1)
#对df_figure_1画柱状图，支持对数坐标
#overall_efficiency, total run time(in seconds)

figure_2 = {
    "GNN_Type": ["GCN_2", "GAT_2", "Sage_2"],
    "ApxIChase": [157.36, 185.42, 132.18],
    "HeuIChase": [3.09, 3.51, 2.68],
    "GNNExplainer": [2.06, 2.43, 1.79],
    "PGExplainer": [31.61, 37.94, 27.12],
    "Exhaustive": [1662.93, 1898.55, 1437.26],
}
df_figure_2 = pd.DataFrame(figure_2)
#对df_figure_2画柱状图，支持对数坐标
#Total Run time(in seconds) for different GNN architectures

figure_3 = {
    "L": [1, 2, 3],
    "ApxIChase": [13.5, 157.36, 319.95],
    "HeuIChase": [0.16, 3.09, 13.27],
    "GNNExplainer": [1.5, 2.06, 3.81],
    "PGExplainer": [21.91, 31.61, 45.33],
    "Exhaustive": [142.6, 1662.93, 3567.14],
}
df_figure_3 = pd.DataFrame(figure_3)
#对df_figure_3画折线图，支持对数坐标
#Total Run time(in seconds) for different hop numbers (L)

figure4 ={
    "k": [1, 2, 4, 6, 8],
    "ApxIChase": [156.74, 154.08, 155.68, 157.36, 156.45],
    "HeuIChase": [3.14, 3.12, 3.03, 3.09, 3.16],
    "GNNExplainer": [2.06, 2.06, 2.06, 2.06, 2.06],
    "PGExplainer": [31.61, 31.61, 31.61, 31.61, 31.61],
    "Exhaustive": [1666.96, 1678.33, 1666.43, 1662.93, 1634.8],
}
df_figure_4 = pd.DataFrame(figure4)
#对df_figure_4画折线图，支持对数坐标
#Total Run time(in seconds) for different constraint sizes (k)

df_figure_5 = {
    "Dataset": ["MUTAG", "ATLAS", "Cora", "BAShape"],
    "ApxIChase": [0.98, 0.90, 0.82, 0.99],
    "HeuIChase": [0.97, 0.88, 0.88, 0.95],
    "GNNExplainer": [0.85, 0.76, 0.65, 0.78],
    "PGExplainer": [0.91, 0.79, 0.64, None],   # None = missing
    "Exhaustive": [0.96, 0.85, 0.90, None],
}

df_figure_5 = pd.DataFrame(df_figure_5)
#对df_figure_5画柱状图
#overall fidelity score( 1 - fidelity- ),higher the better

df_figure_6 = {
    "Dataset": ["MUTAG", "ATLAS", "Cora", "BAShape"],
    "ApxIChase": [0.533, 0.57, 0.88, 0.57],
    "HeuIChase": [0.54, 0.42, 0.184, 0.2425],
    "GNNExplainer": [0.635, 0.67, 0.9, 0.58],
    "PGExplainer": [0.465, 0.58, 0.9, None],
    "Exhaustive": [0.69, 0.71, 0.872, None],
}
df_figure_6 = pd.DataFrame(df_figure_6)
#对df_figure_6画柱状图
#overall conciseness score(higher the better)

df_figure_7 = {
    "B": [1, 2, 4, 6, 8],
    "ApxIChase": [0.2, 0.4, 0.8, 0.8, 0.8],
    "HeuIChase": [0.2, 0.4, 0.5, 0.6, 0.6],
    "GNNExplainer": [0, 0, 0, 0, 0],
    "PGExplainer": [0, 0, 0, 0, 0],
    "Exhaustive": [1, 1, 1, 1.0, 1.0],
}
df_figure_7 = pd.DataFrame(df_figure_7)
#对df_figure_7画折线图
#Coverage score for different repair budget (B)

df_figure_8 = {
    "k": [1, 2, 4, 6, 8],
    "ApxIChase": [0.8, 0.8, 0.8, 0.8, 0.8],
    "HeuIChase": [0.6, 0.6, 0.6, 0.6, 0.6],
    "GNNExplainer": [0, 0, 0, 0, 0],
    "PGExplainer": [0, 0, 0, 0, 0],
    "Exhaustive": [1, 1, 1, 1.0, 1.0],
}
df_figure_8 = pd.DataFrame(df_figure_8)
#对df_figure_8画折线图
#Coverage score varying window sizes (k)

df_figure_9 = {
    "k": [1, 2, 4, 6, 8],
    "ApxIChase": [0.23, 0.58, 0.63, 0.88, 0.88],
    "HeuIChase": [0.184, 0.184, 0.184, 0.184, 0.184],
    "GNNExplainer": [0.9, 0.9, 0.9, 0.9, 0.9],
    "PGExplainer": [0.9, 0.9, 0.9, 0.9, 0.9],
    "Exhaustive": [0.21, 0.55, 0.6, 0.87, 0.87],
}
df_figure_9 = pd.DataFrame(df_figure_9)
#对df_figure_9画折线图
#Conciseness score varying window sizes (k)


def plot_bar_chart(df, x_col, y_cols, ylabel, xlabel, filename, use_log=True, legend_loc='upper left', legend_ncol=1, ylim_top=None, ylim_bottom=None):
    """绘制柱状图"""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    x = np.arange(len(df[x_col]))
    width = 0.13  # 柱子宽度
    n_bars = len(y_cols)
    
    for i, col in enumerate(y_cols):
        offset = (i - n_bars/2 + 0.5) * width
        values = df[col].values
        
        # 使用缩写作为图例标签
        label = LEGEND_LABELS.get(col, col)
        
        bars = ax.bar(x + offset, values, width, 
                     label=label,
                     color=COLORS.get(col, '#333333'),
                     hatch=HATCHES.get(col, ''),
                     edgecolor='black',
                     linewidth=0.5)
    
    # 设置坐标轴
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(df[x_col], rotation=0, fontsize=9)
    ax.tick_params(axis='y', labelsize=9)
    
    # 网格
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    
    # 对数坐标
    if use_log:
        ax.set_yscale('log')
    
    # 设置 y 轴范围（如果指定）
    if ylim_bottom is not None or ylim_top is not None:
        current_ylim = ax.get_ylim()
        new_bottom = ylim_bottom if ylim_bottom is not None else current_ylim[0]
        new_top = ylim_top if ylim_top is not None else current_ylim[1]
        ax.set_ylim(bottom=new_bottom, top=new_top)
    
    # 隐藏超出范围的刻度标签
    if ylim_top is not None:
        yticks = ax.get_yticks()
        # 过滤掉大于等于 ylim_top 的刻度
        visible_ticks = [tick for tick in yticks if tick < ylim_top]
        if ylim_bottom is not None:
            visible_ticks = [tick for tick in visible_ticks if tick >= ylim_bottom]
        ax.set_yticks(visible_ticks)
    
    # 图例（使用自定义位置）
    legend = ax.legend(loc=legend_loc,
                      frameon=False,
                      fontsize=6.5, handlelength=1.2, handletextpad=0.3,
                      labelspacing=0.3, ncol=legend_ncol, borderpad=0.3)
    
    # 背景
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # 不使用 tight_layout，因为已手动调整
    
    # 只保存 PNG 格式
    png_filename = filename.replace('.pdf', '.png')
    plt.savefig(png_filename, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    print(f"Saved: {png_filename}")
    plt.close()


def plot_line_chart(df, x_col, y_cols, ylabel, xlabel, filename, use_log=True, legend_loc='upper left', legend_ncol=1, ylim_top=None, ylim_bottom=None, xticks=None):
    """绘制折线图"""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    for col in y_cols:
        values = df[col].values
        color = COLORS.get(col, '#333333')
        # 使用缩写作为图例标签
        label = LEGEND_LABELS.get(col, col)
        
        ax.plot(df[x_col], values,
               marker=MARKERS.get(col, 'o'),
               color=color,
               label=label,
               linewidth=1.3,  # 减小线条粗细
               markersize=5.5,  # 减小标记尺寸
               markerfacecolor='none',  # 标记填充透明
               markeredgecolor=color,   # 标记边缘颜色与线条一致
               markeredgewidth=1.3)  # 稍微减细边缘
    
    # 设置坐标轴
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(axis='both', labelsize=9)
    
    # 设置 x 轴刻度（如果指定）
    if xticks is not None:
        ax.set_xticks(xticks)
    
    # 网格
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    
    # 对数坐标
    if use_log:
        ax.set_yscale('log')
    
    # 设置 y 轴范围（如果指定）
    if ylim_bottom is not None or ylim_top is not None:
        current_ylim = ax.get_ylim()
        new_bottom = ylim_bottom if ylim_bottom is not None else current_ylim[0]
        new_top = ylim_top if ylim_top is not None else current_ylim[1]
        ax.set_ylim(bottom=new_bottom, top=new_top)
    
    # 隐藏超出范围的刻度标签
    if ylim_top is not None:
        yticks = ax.get_yticks()
        # 过滤掉大于等于 ylim_top 的刻度
        visible_ticks = [tick for tick in yticks if tick < ylim_top]
        if ylim_bottom is not None:
            visible_ticks = [tick for tick in visible_ticks if tick >= ylim_bottom]
        ax.set_yticks(visible_ticks)
    
    # 图例（使用自定义位置）
    legend = ax.legend(loc=legend_loc,
                      frameon=False,
                      fontsize=6.5, handlelength=1.2, handletextpad=0.3,
                      labelspacing=0.3, ncol=legend_ncol, borderpad=0.3)
    
    # 背景
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # 不使用 tight_layout，因为已手动调整
    
    # 只保存 PNG 格式
    png_filename = filename.replace('.pdf', '.png')
    plt.savefig(png_filename, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    print(f"Saved: {png_filename}")
    plt.close()


# ========== 生成所有图表 ==========

# Figure 1: Overall efficiency (bar chart, log scale)
# Legend in upper left
plot_bar_chart(
    df_figure_1, 
    'Dataset',
    ['ApxIChase', 'HeuIChase', 'GNNExplainer', 'PGExplainer', 'Exhaustive'],
    'Run time (sec)',
    'Dataset',
    'figure_runtime_datasets.pdf',
    use_log=True,
    legend_loc='upper left',
    legend_ncol=1
)

# Figure 2: Different GNN architectures (bar chart, log scale)
# Legend in upper left
plot_bar_chart(
    df_figure_2,
    'GNN_Type',
    ['ApxIChase', 'HeuIChase', 'GNNExplainer', 'PGExplainer', 'Exhaustive'],
    'Run time (sec)',
    'GNN Type',
    'figure_runtime_gnn.pdf',
    use_log=True,
    legend_loc='upper left',
    legend_ncol=1
)

# Figure 3: Different hop numbers L (line chart, log scale)
# Legend in lower right, 2 rows (3 top, 2 bottom)
plot_line_chart(
    df_figure_3,
    'L',
    ['ApxIChase', 'HeuIChase', 'GNNExplainer', 'PGExplainer', 'Exhaustive'],
    'Run time (sec)',
    '|L|',
    'figure_runtime_hops.pdf',
    use_log=True,
    legend_loc='lower right',
    legend_ncol=2,
    xticks=[1, 2, 3]
)

# Figure 4: Different constraint sizes k (line chart, log scale)
# Legend between HeuC and PGX lines, 2 rows (3 top, 2 bottom)
plot_line_chart(
    df_figure_4,
    'k',
    ['ApxIChase', 'HeuIChase', 'GNNExplainer', 'PGExplainer', 'Exhaustive'],
    'Run time (sec)',
    'k',
    'figure_runtime_window.pdf',
    use_log=True,
    legend_loc='center left',
    legend_ncol=3,
    xticks=[1, 2, 4, 6, 8]
)

# Figure 5: Fidelity score (bar chart, no log)
# Legend horizontal on top, extend y-axis to 1.2 (without showing tick)
plot_bar_chart(
    df_figure_5,
    'Dataset',
    ['ApxIChase', 'HeuIChase', 'GNNExplainer', 'PGExplainer', 'Exhaustive'],
    'Fidelity (1 - Fidelity-)',
    'Dataset',
    'figure_fidelity.pdf',
    use_log=False,
    legend_loc='upper center',
    legend_ncol=5,
    ylim_top=1.2
)

# Figure 6: Conciseness score (bar chart, no log)
# Legend in upper left
plot_bar_chart(
    df_figure_6,
    'Dataset',
    ['ApxIChase', 'HeuIChase', 'GNNExplainer', 'PGExplainer', 'Exhaustive'],
    'Conciseness',
    'Dataset',
    'figure_conciseness.pdf',
    use_log=False,
    legend_loc='upper left',
    legend_ncol=1
)

# Figure 7: Coverage vs Budget B (line chart, no log)
# Legend horizontal on top, y-axis from -0.05 to 1.2 (only show ticks from 0 to 1.0)
plot_line_chart(
    df_figure_7,
    'B',
    ['ApxIChase', 'HeuIChase', 'GNNExplainer', 'PGExplainer', 'Exhaustive'],
    'Coverage',
    'B',
    'figure_coverage_budget.pdf',
    use_log=False,
    legend_loc='upper center',
    legend_ncol=5,
    ylim_top=1.2,
    ylim_bottom=-0.05,
    xticks=[1, 2, 4, 6, 8]
)

# Figure 8: Coverage vs window size k (line chart, no log)
# Legend horizontal on top, y-axis from -0.05 to 1.2 (only show ticks from 0 to 1.0)
plot_line_chart(
    df_figure_8,
    'k',
    ['ApxIChase', 'HeuIChase', 'GNNExplainer', 'PGExplainer', 'Exhaustive'],
    'Coverage',
    'k',
    'figure_coverage_window.pdf',
    use_log=False,
    legend_loc='upper center',
    legend_ncol=5,
    ylim_top=1.2,
    ylim_bottom=-0.05,
    xticks=[1, 2, 4, 6, 8]
)

# Figure 9: Conciseness vs window size k (line chart, no log)
# Legend horizontal on top, y-axis from -0.05 to 1.2 (only show ticks from 0 to 1.0)
plot_line_chart(
    df_figure_9,
    'k',
    ['ApxIChase', 'HeuIChase', 'GNNExplainer', 'PGExplainer', 'Exhaustive'],
    'Conciseness',
    'k',
    'figure_conciseness_window.pdf',
    use_log=False,
    legend_loc='upper center',
    legend_ncol=5,
    ylim_top=1.2,
    ylim_bottom=-0.05,
    xticks=[1, 2, 4, 6, 8]
)

print("\n✓ All figures generated successfully!")