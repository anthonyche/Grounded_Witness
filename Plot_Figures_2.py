import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import rcParams

# 设置全局字体为 Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 8            # 基础字体
rcParams['axes.labelsize'] = 9       # 坐标轴标签
rcParams['axes.titlesize'] = 9       # 坐标轴标题
rcParams['xtick.labelsize'] = 8      # X轴刻度
rcParams['ytick.labelsize'] = 8      # Y轴刻度
rcParams['legend.fontsize'] = 6      # 图例字体默认值（会被硬编码覆盖）

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
    "|Σ|": [10,20,30,40,50],
    "ApxIChase": [120.4	,157.36	,167.36	, 168.5, 168],
    "HeuIChase": [2.85,3.09	,3.09,3.1,3.12],
    "GNNExplainer": [2.06, 2.06, 2, 2.06 ,2.06],
    "PGExplainer": [31.61, 31.61, 31.61, 31.61, 31.61],
    "Exhaustive": [980, 1662.93, 1733, 1795, 1991],
}
df_figure_2 = pd.DataFrame(figure_2)
#对df_figure_2画折线图
#Total Run time(in seconds) versus Size of Constraints(|Σ|)

figure_3 = {
    "1%": [157.36, 3.09, 2.06, 31.61, 1662.93],
    "2%": [320.7, 6.32, 3.94, 32.59, 3388],
    "3%": [475.1, 9.11, 6.03, 32.64, 5512],
    "4%": [620, 12.58, 8.17, 33.13, 7488],
    "5%": [798.5, 15.21, 12.5, 33.3, 10188]
}
figure3_index = ["ApxIChase", "HeuIChase", "GNNExplainer", "PGExplainer", "Exhaustive"]
df_figure_3 = pd.DataFrame(figure_3, index=figure3_index)
#对df_figure_3画折线图，支持对数坐标，
#Total Run time(in seconds) varying the number(ratio) of target nodes

figure_4 = {
    "L": [1, 2, 3],
    "ApxIChase": [13.5, 157.36, 319.95],
    "HeuIChase": [0.16, 3.09, 13.27],
    "GNNExplainer": [1.5, 2.06, 3.81],
    "PGExplainer": [21.91, 31.61, 45.33],
    "Exhaustive": [142.6, 1662.93, 3567.14],
}
df_figure_4 = pd.DataFrame(figure_4)
#对df_figure_4画折线图，支持对数坐标
#Total Run time(in seconds) for different hop numbers (L)

figure5 ={
    "5%": [157.36,3.09,2.06,31.61,1662.93],
    "10%": [138,3.19,2.05,31.36,2148],
    "15%": [115.97,3.16,2.03,32.63,2829.1],
    "20%": [96.07,3.24,1.97,33.36,3642],
}
figure5_index = ["ApxIChase", "HeuIChase", "GNNExplainer", "PGExplainer", "Exhaustive"]
df_figure_5 = pd.DataFrame(figure5, index=figure5_index)
#对df_figure_5画折线图，支持对数坐标
#Total Runtime (in seconds) varying the Incompleteness 


figure_6 = {
    "GNN_Type": ["GCN_2", "GAT_2", "Sage_2"],
    "ApxIChase": [157.36, 185.42, 132.18],
    "HeuIChase": [3.09, 3.51, 2.68],
    "GNNExplainer": [2.06, 2.43, 1.79],
    "PGExplainer": [31.61, 37.94, 27.12],
    "Exhaustive": [1662.93, 1898.55, 1437.26],
}
df_figure_6 = pd.DataFrame(figure_6)
#对df_figure_6画柱状图，支持对数坐标
#Total Run time(in seconds) for different GNN architectures

#Figure 7: Overall Coverage
figure_7 = {
    "Dataset": ["MUTAG", "ATLAS", "Cora", "BAShape"],
    "ApxIChase": [0.72, 1, 0.8, 1],
    "HeuIChase": [0.7, 1, 0.4, 0.6],
    "GNNExplainer": [0, 0.17, 0, 0],
    "PGExplainer": [0, 0.31, 0, None],  # None 表示缺失值
    "Exhaustive": [1, 1, 1, None],
}
df_figure_7 = pd.DataFrame(figure_7)
#对df_figure_7画柱状图，支持对数坐标
#Overall Coverage for different GNN architectures

#figure 8: Coverage vs B
figure_8 = {
    "k": [1, 2, 4, 6, 8],
    "ApxIChase": [0.2, 0.4, 0.8, 0.8, 0.8],
    "HeuIChase": [0.2, 0.4, 0.5, 0.6, 0.6],
    "GNNExplainer": [0, 0, 0, 0, 0],
    "PGExplainer": [0, 0, 0, 0, 0],
    "Exhaustive": [1, 1, 1, 1.0, 1.0],
}
df_figure_8 = pd.DataFrame(figure_8)
#对df_figure_8画折线图, 支持对数坐标
#Coverage vs Budget (k)

figure_9 = {
    "Constraint_Size":[10,20,30,40,50],
    "ApxIChase": [1, 0.8, 0.56, 0.43, 0.34],
    "HeuIChase": [0.8, 0.4, 0.3, 0.22, 0.18],
    "GNNExplainer": [0, 0, 0, 0, 0],
    "PGExplainer": [0, 0, 0, 0, 0],
    "Exhaustive": [1, 1, 1, 1, 1], 
}
df_figure_9 = pd.DataFrame(figure_9)
#对df_figure_9画折线图, 支持对数坐标
#Coverage vs Constraint Size

#Figure 10: Coverage varying L
figure_10 = {
    "L": [1, 2, 3],
    "ApxIChase": [0.65, 0.8, 0.8],
    "HeuIChase": [0.25, 0.4, 0.4],
    "GNNExplainer": [0, 0, 0],
    "PGExplainer": [0, 0, 0],
    "Exhaustive": [1, 1, 1],
}
df_figure_10 = pd.DataFrame(figure_10)
#df_figure_10画折线图, 支持对数坐标
#Coverage varying L

figure_11 = {
     "5%": [0.8,0.4,0,0,1],
    "10%": [0.7, 0.4, 0, 0, 1],
    "15%": [0.65,0.2,0,0,1],
    "20%": [0.6,0.2,0,0,1],
}
figure11_index = ["ApxIChase", "HeuIChase", "GNNExplainer", "PGExplainer", "Exhaustive"]
df_figure_11 = pd.DataFrame(figure_11, index=figure11_index)

figure_12 = {
    "Dataset": ["MUTAG", "ATLAS", "Cora", "BAShape"],
    "ApxIChase": [0.98, 0.9, 0.82, 0.99],
    "HeuIChase": [0.97, 0.88, 0.88, 0.95],
    "GNNExplainer": [0.85, 0.76, 0.65, 0.78],
    "PGExplainer": [0.91, 0.79, 0.64, None],  # None 表示缺失值
    "Exhaustive": [0.96, 0.85, 0.9, None],
}
df_figure_12 = pd.DataFrame(figure_12)
#对df_figure_12画柱状图，支持对数坐标
#Overall Fidelity- score 

# Figure 13: OGBN-100M Runtime vs Number of Processors (distributed benchmark)
figure_13 = {
    "Processors": [4, 6, 8, 10, 20],
    "ApxIChase": [13949.72, 8995.12, 6707.92, 5252.43, 2642.99],
    "HeuIChase": [282.06, 180.21, 133.43, 103.84, 53.75],
    "GNNExplainer": [169.24, 104.52, 74.7, 56.91, 31.7],
}
df_figure_13 = pd.DataFrame(figure_13)
# 折线图：展示分布式计算的加速效果
# OGBN-Papers100M Runtime vs Number of Workers 


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
    
    # 图例（硬编码字体大小为14pt，保持图例在图内）
    legend = ax.legend(loc=legend_loc,
                      frameon=False,
                      fontsize=7, handlelength=1.2, handletextpad=0.3,
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


def plot_line_chart(df, x_col, y_cols, ylabel, xlabel, filename, use_log=True, legend_loc='upper left', legend_ncol=1, ylim_top=None, ylim_bottom=None, xticks=None, show_exhaustive_timeout=False):
    """绘制折线图
    
    Args:
        show_exhaustive_timeout: 如果为 True，会为 Exhaustive 列绘制特殊的超时标记
                                 (第一个非nan值用实点，其他位置用虚线+红叉)
    """
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
    
    # 处理 Exhaustive 的特殊绘制（超时标记）
    if show_exhaustive_timeout and 'Exhaustive' in df.columns:
        exhaustive_values = df['Exhaustive'].values
        x_values = df[x_col].values if xticks is None else xticks
        
        # 找到第一个非 nan 值（基线值）
        baseline_value = None
        baseline_idx = None
        for i, val in enumerate(exhaustive_values):
            if not np.isnan(val):
                baseline_value = val
                baseline_idx = i
                break
        
        if baseline_value is not None:
            color_exhaustive = COLORS.get('Exhaustive', '#A682B3')
            
            # 1. 画实际的基线点
            ax.plot(x_values[baseline_idx], baseline_value,
                   marker=MARKERS.get('Exhaustive', 'D'),
                   color=color_exhaustive,
                   markersize=5.5,
                   markerfacecolor='none',
                   markeredgecolor=color_exhaustive,
                   markeredgewidth=1.3,
                   label=LEGEND_LABELS.get('Exhaustive', 'Exh'),
                   zorder=5)
            
            # 2. 画水平虚线（跨越所有 x 位置）
            ax.axhline(y=baseline_value, color=color_exhaustive, 
                      linestyle='--', linewidth=1.0, alpha=0.5, zorder=1)
            
            # 3. 在超时位置画红叉
            for i, val in enumerate(exhaustive_values):
                if np.isnan(val):
                    ax.plot(x_values[i], baseline_value,
                           marker='x', color='red', markersize=6,
                           markeredgewidth=1.5, zorder=6)
    
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
    
    # 图例（硬编码字体大小为14pt，保持图例在图内）
    legend = ax.legend(loc=legend_loc,
                      frameon=False,
                      fontsize=7, handlelength=1.2, handletextpad=0.3,
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
    'Total Run time (sec)',
    'Dataset',
    'figure_1_overall_efficiency.png',
    use_log=True,
    legend_loc='upper left',
    legend_ncol=1
)

# Figure 2: Total Runtime vs Constraint Size (line chart, log scale)
# Legend in upper center
plot_line_chart(
    df_figure_2,
    '|Σ|',
    ['ApxIChase', 'HeuIChase', 'GNNExplainer', 'PGExplainer', 'Exhaustive'],
    'Total Run time (sec)',
    '|Σ|',
    'figure_2_runtime_vs_constraint_size.png',
    use_log=True,
    legend_loc='upper center',
    legend_ncol=5,
    ylim_top=5000,
    ylim_bottom=1,
    xticks=[10, 20, 30, 40, 50]
)

# Figure 3: Runtime varying target node ratio (line chart, log scale, transposed data)
# x轴是ratio (1%, 2%, 3%, 4%, 5%)，y轴是methods
plot_line_chart(
    df_figure_3.T.reset_index().rename(columns={'index': 'Ratio'}),
    'Ratio',
    figure3_index,
    'Total Run time (sec)',
    'Target Node Ratio',
    'figure_3_runtime_vs_target_ratio.png',
    use_log=True,
    legend_loc='upper center',
    legend_ncol=5,
    ylim_top=30000,
    ylim_bottom=1,
    xticks=['1%', '2%', '3%', '4%', '5%']
)

# Figure 4: Runtime for different hop numbers L (line chart, log scale)
# Legend in lower right
plot_line_chart(
    df_figure_4,
    'L',
    ['ApxIChase', 'HeuIChase', 'GNNExplainer', 'PGExplainer', 'Exhaustive'],
    'Total Run time (sec)',
    'L',
    'figure_4_runtime_vs_hops.png',
    use_log=True,
    legend_loc='lower right',
    legend_ncol=1,
    xticks=[1, 2, 3]
)

# Figure 5: Runtime varying Incompleteness (line chart, log scale, transposed data)
# x轴是mask_ratio (5%, 10%, 15%, 20%)，y轴是methods
plot_line_chart(
    df_figure_5.T.reset_index().rename(columns={'index': 'Incompleteness'}),
    'Incompleteness',
    figure5_index,
    'Total Run time (sec)',
    'Incompleteness',
    'figure_5_runtime_vs_incompleteness.png',
    use_log=True,
    legend_loc='upper center',
    legend_ncol=5,
    ylim_top=10000,
    ylim_bottom=1,
    xticks=['5%', '10%', '15%', '20%']
)

# Figure 6: Runtime for different GNN architectures (bar chart, log scale)
# Legend in upper left
plot_bar_chart(
    df_figure_6,
    'GNN_Type',
    ['ApxIChase', 'HeuIChase', 'GNNExplainer', 'PGExplainer', 'Exhaustive'],
    'Total Run time (sec)',
    'GNN Architecture',
    'figure_6_runtime_gnn_types.png',
    use_log=True,
    legend_loc='upper left',
    legend_ncol=1
)

# Figure 7: Overall Coverage (bar chart, no log scale)
# Legend in upper left
plot_bar_chart(
    df_figure_7,
    'Dataset',
    ['ApxIChase', 'HeuIChase', 'GNNExplainer', 'PGExplainer', 'Exhaustive'],
    'Coverage',
    'Dataset',
    'figure_7_overall_coverage.png',
    use_log=False,
    legend_loc='upper left',
    legend_ncol=1,
    ylim_top=1.2,
    ylim_bottom=0
)

# Figure 8: Coverage vs Budget k (line chart, no log scale)
# Legend horizontal on top, y-axis from -0.05 to 1.2
plot_line_chart(
    df_figure_8,
    'k',
    ['ApxIChase', 'HeuIChase', 'GNNExplainer', 'PGExplainer', 'Exhaustive'],
    'Coverage',
    'k',
    'figure_8_coverage_vs_budget.png',
    use_log=False,
    legend_loc='upper center',
    legend_ncol=5,
    ylim_top=1.2,
    ylim_bottom=-0.05,
    xticks=[1, 2, 4, 6, 8]
)

# Figure 9: Coverage vs Constraint Size (line chart, no log scale)
# Legend horizontal on top, y-axis from -0.05 to 1.2
plot_line_chart(
    df_figure_9,
    'Constraint_Size',
    ['ApxIChase', 'HeuIChase', 'GNNExplainer', 'PGExplainer', 'Exhaustive'],
    'Coverage',
    '|Σ|',
    'figure_9_coverage_vs_constraint_size.png',
    use_log=False,
    legend_loc='upper center',
    legend_ncol=5,
    ylim_top=1.2,
    ylim_bottom=-0.05,
    xticks=[10, 20, 30, 40, 50]
)

# Figure 10: Coverage varying L (line chart, no log scale)
# Legend horizontal on top, y-axis from -0.05 to 1.2
plot_line_chart(
    df_figure_10,
    'L',
    ['ApxIChase', 'HeuIChase', 'GNNExplainer', 'PGExplainer', 'Exhaustive'],
    'Coverage',
    'L',
    'figure_10_coverage_vs_hops.png',
    use_log=False,
    legend_loc='upper center',
    legend_ncol=5,
    ylim_top=1.2,
    ylim_bottom=-0.05,
    xticks=[1, 2, 3]
)

# Figure 11: Coverage varying Incompleteness (line chart, no log scale, transposed data)
# x轴是mask_ratio (5%, 10%, 15%, 20%)，y轴是methods
plot_line_chart(
    df_figure_11.T.reset_index().rename(columns={'index': 'Incompleteness'}),
    'Incompleteness',
    figure11_index,
    'Coverage',
    'Incompleteness',
    'figure_11_coverage_vs_incompleteness.png',
    use_log=False,
    legend_loc='upper center',
    legend_ncol=5,
    ylim_top=1.2,
    ylim_bottom=-0.05,
    xticks=['5%', '10%', '15%', '20%']
)

# Figure 12: Overall Fidelity score (bar chart, no log scale)
# Legend horizontal on top, extend y-axis to 1.2
plot_bar_chart(
    df_figure_12,
    'Dataset',
    ['ApxIChase', 'HeuIChase', 'GNNExplainer', 'PGExplainer', 'Exhaustive'],
    'Fidelity Score',
    'Dataset',
    'figure_12_overall_fidelity.png',
    use_log=False,
    legend_loc='upper center',
    legend_ncol=5,
    ylim_top=1.2,
    ylim_bottom=0
)

# Figure 13: OGBN-100M Runtime vs Number of Processors (line chart, log scale)
# 展示分布式计算的加速效果
plot_line_chart(
    df_figure_13,
    'Processors',
    ['ApxIChase', 'HeuIChase', 'GNNExplainer'],
    'Total Runtime (sec)',
    'Number of Processors',
    'figure_13_ogbn_runtime_vs_workers.png',
    use_log=True,
    legend_loc='upper right',
    legend_ncol=1,
    ylim_bottom=10,
    ylim_top=20000,
    xticks=[4, 6, 8, 10, 20]
)

# ========== TreeCycle 新增图表 ==========

# Figure 14: TreeCycle Runtime Varying Graph Size (line chart, log scale)
figure_14 = {
    "Graph_Size": ["1.1M", "2.3M", "17M", "1.4B"],
    "ApxIChase": [292.4, 318.7, 447.9, 898.3],
    "HeuIChase": [48.6, 54.2, 82.51, 182.7],
    "GNNExplainer": [22.8, 38.9, 54, 86.4],
    "PGExplainer": [78.2, 86.7, 126, 210.3],
    "Exhaustive": [5120, 8480, 19800, 28800]  # 第一个点是基线，其他超时
}
df_figure_14 = pd.DataFrame(figure_14)

plot_line_chart(
    df_figure_14,
    'Graph_Size',
    ['ApxIChase', 'HeuIChase', 'GNNExplainer', 'PGExplainer', 'Exhaustive'],
    'Total Runtime (sec)',
    'Graph Size (# Edges)',
    'figure_14_treecycle_runtime_vs_graph_size.png',
    use_log=True,
    legend_loc='upper left',
    legend_ncol=5,
    ylim_bottom=10,
    ylim_top=100000,  # 增加上限以显示 28800
    xticks=["1.1M", "2.3M", "17M", "1.4B"],
    show_exhaustive_timeout=False  # 启用 Exhaustive 超时标记
)

# Figure 15: TreeCycle Runtime (1.4B) Varying Number of Processors (line chart, log scale)
figure_15 = {
    "Processors": [4, 6, 8, 10, 20],
    "ApxIChase": [5030.48, 3291, 2424, 1804, 898.3],
    "HeuIChase": [794.85, 627, 484, 383, 182.7],
    "GNNExplainer": [418.2, 258.7, 209.6, 187.9, 86.4],
    "PGExplainer": [1208, 784, 578, 454, 210.3],
    "Exhaustive": [np.nan, np.nan, np.nan, np.nan, 28800]  # 最后一个点是基线
}
df_figure_15 = pd.DataFrame(figure_15)

plot_line_chart(
    df_figure_15,
    'Processors',
    ['ApxIChase', 'HeuIChase', 'GNNExplainer', 'PGExplainer'],
    'Total Runtime (sec)',
    'Number of Processors',
    'figure_15_treecycle_runtime_vs_processors.png',
    use_log=True,
    legend_loc='upper right',
    legend_ncol=5,
    ylim_bottom=50,
    ylim_top=100000,  # 增加上限以显示 28800
    xticks=[4, 6, 8, 10, 20],
    show_exhaustive_timeout=True  # 启用 Exhaustive 超时标记
)

# Figure 16: TreeCycle Runtime (1.4B) Varying Number of Target Nodes (line chart, log scale)
figure_16 = {
    "Target_Nodes": [100, 200, 300, 400, 500],
    "ApxIChase": [898.3, 1815, 2724, 3695, 4568],
    "HeuIChase": [182.7, 372, 542, 748, 905],
    "GNNExplainer": [86.4, 171.8, 259.7, 345.9, 433.6],
    "PGExplainer": [210.3, 433, 618, 862, 1038],
    "Exhaustive": [28800, np.nan, np.nan, np.nan, np.nan]  # 第一个点是基线
}
df_figure_16 = pd.DataFrame(figure_16)

plot_line_chart(
    df_figure_16,
    'Target_Nodes',
    ['ApxIChase', 'HeuIChase', 'GNNExplainer', 'PGExplainer'],
    'Total Runtime (sec)',
    'Query Load (# Target Nodes)',
    'figure_16_treecycle_runtime_vs_target_nodes.png',
    use_log=True,
    legend_loc='upper left',
    legend_ncol=5,
    ylim_bottom=50,
    ylim_top=100000,  # 增加上限以显示 28800
    xticks=[100, 200, 300, 400, 500],
    show_exhaustive_timeout=True  # 启用 Exhaustive 超时标记
)

print("\n✓ All figures generated successfully!")
print("✓ TreeCycle figures (14-16) added!")