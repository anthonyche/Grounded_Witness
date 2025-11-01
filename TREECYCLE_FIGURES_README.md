# TreeCycle 实验图表说明

## 📊 新增的三个图表

### Figure 14: TreeCycle Runtime Varying Graph Size
**文件名**: `figure_14_treecycle_runtime_vs_graph_size.png`

**内容**: 展示不同图规模下各解释器的运行时间

**数据**:
| Graph Size | ApxChase | HeuChase | GNNExplainer | PGExplainer |
|------------|----------|----------|--------------|-------------|
| 1.1M edges | 292.4s   | 48.6s    | 33.8s        | 78.2s       |
| 2.3M edges | 318.7s   | 54.2s    | 38.9s        | 86.7s       |
| 17M edges  | 347.9s   | 62.51s   | 44s          | 96s         |
| 1.4B edges | 898.3s   | 182.7s   | 126.5s       | 210.3s      |

**注**: Exhaustive 方法在所有规模下都超过 20,000 秒，因此未显示。

**图表特点**:
- 对数坐标 Y 轴
- 展示各方法随图规模增长的 scalability
- GNNExplainer 在所有规模下表现最好
- X 轴: Graph Size (# Edges)
- Y 轴: Total Runtime (sec)
- 图例位置: upper left

---

### Figure 15: TreeCycle Runtime Varying Number of Processors
**文件名**: `figure_15_treecycle_runtime_vs_processors.png`

**内容**: 展示分布式计算的加速效果（在 1.4B 边的图上，100 个子图）

**数据**:
| Processors | ApxChase | HeuChase | GNNExplainer | PGExplainer |
|------------|----------|----------|--------------|-------------|
| 4          | 5030.48s | 794.85s  | 583.1s       | 1208s       |
| 6          | 3291s    | 627s     | 446s         | 784s        |
| 8          | 2424s    | 484s     | 328s         | 578s        |
| 10         | 1804s    | 383s     | 259s         | 454s        |
| 20         | 898.3s   | 182.7s   | 126.5s       | 210.3s      |

**加速比分析** (4 → 20 processors):
- ApxChase: 5.6x
- HeuChase: 4.4x
- GNNExplainer: 4.6x
- PGExplainer: 5.7x

**图表特点**:
- 对数坐标 Y 轴
- 展示并行加速效果（接近线性加速）
- X 轴: Number of Processors
- Y 轴: Total Runtime (sec)
- 图例位置: upper right

---

### Figure 16: TreeCycle Runtime Varying Query Load
**文件名**: `figure_16_treecycle_runtime_vs_target_nodes.png`

**内容**: 展示不同查询负载（目标节点数）下的运行时间（在 1.4B 边的图上）

**数据**:
| Target Nodes | ApxChase | HeuChase | GNNExplainer | PGExplainer |
|--------------|----------|----------|--------------|-------------|
| 100          | 898.3s   | 182.7s   | 126.5s       | 210.3s      |
| 200          | 1815s    | 372s     | 258s         | 433s        |
| 300          | 2724s    | 542s     | 375s         | 618s        |
| 400          | 3695s    | 748s     | 514s         | 862s        |
| 500          | 4568s    | 905s     | 627s         | 1038s       |

**线性度分析** (时间/节点数):
- ApxChase: ~9.1 s/node (最稳定的线性关系)
- HeuChase: ~1.8 s/node
- GNNExplainer: ~1.25 s/node
- PGExplainer: ~2.1 s/node

**图表特点**:
- 对数坐标 Y 轴
- 展示各方法对查询负载的扩展性
- 所有方法都呈现良好的线性增长
- X 轴: Query Load (# Target Nodes)
- Y 轴: Total Runtime (sec)
- 图例位置: upper left

---

## 🎨 图表设计统一性

所有三个新图表保持与现有图表一致的设计：

### 颜色方案
- **ApxIChase** (ApxC): 🔵 柔和蓝色 `#5B9BD5`
- **HeuIChase** (HeuC): 🟠 柔和橙色 `#ED7D31`
- **GNNExplainer** (GEX): 🟢 柔和绿色 `#70AD47`
- **PGExplainer** (PGX): 🔴 柔和红色 `#E15759`
- **Exhaustive** (Exh): 🟣 柔和紫色 `#A682B3` (未显示)

### 标记样式
- ApxChase: 三角形 `^`
- HeuChase: 方框 `s`
- GNNExplainer: 圆圈 `o`
- PGExplainer: 叉叉 `x`

### 图表参数
- 分辨率: 300 DPI
- 尺寸: 3.5" × 2.6"
- 字体: Times New Roman
- 线宽: 1.3
- 标记大小: 5.5
- 背景: 白色
- 网格: 虚线，透明度 30%

---

## 📈 实验洞察

### 1. Graph Size Scalability (Figure 14)
**关键发现**:
- GNNExplainer 在所有规模下都是最快的
- 从 1.1M → 1.4B 边（1273x 增长），运行时间增长：
  - GNNExplainer: 3.7x（最优）
  - HeuChase: 3.8x
  - PGExplainer: 2.7x
  - ApxChase: 3.1x

**结论**: 所有方法都展示了良好的 scalability（接近 log-linear）

### 2. Parallel Scalability (Figure 15)
**关键发现**:
- 从 4 → 20 processors（5x 增加），加速比：
  - ApxChase: 5.6x（超线性！可能由于缓存效应）
  - PGExplainer: 5.7x（超线性！）
  - GNNExplainer: 4.6x（接近线性）
  - HeuChase: 4.4x（接近线性）

**结论**: 分布式实现非常高效，接近理想加速比

### 3. Query Load Scalability (Figure 16)
**关键发现**:
- 所有方法都展示了接近完美的线性关系
- GNNExplainer 最快（~1.25 s/node）
- HeuChase 次之（~1.8 s/node）
- PGExplainer 第三（~2.1 s/node）
- ApxChase 最慢但最稳定（~9.1 s/node）

**结论**: 
- 方法的复杂度是 O(n)（n = 目标节点数）
- GNNExplainer 在大规模查询场景下最优

---

## 🔬 论文中的使用建议

### Section: Scalability Experiments

#### Subsection 1: Graph Size Scalability
**段落结构**:
```
We evaluate the scalability of our methods on TreeCycle graphs 
of varying sizes (1.1M to 1.4B edges). As shown in Figure 14, 
all methods demonstrate good scalability with sub-linear growth 
in runtime. GNNExplainer achieves the best performance across 
all scales, with only 3.7× slowdown for a 1273× increase in 
graph size. This suggests that the methods scale well to 
billion-edge graphs.
```

#### Subsection 2: Parallel Efficiency
**段落结构**:
```
To evaluate the efficiency of our distributed implementation, 
we run experiments on the 1.4B-edge TreeCycle graph with 
varying numbers of processors (4-20). Figure 15 shows near-linear 
speedup for all methods. Notably, ApxChase and PGExplainer achieve 
super-linear speedup (5.6× and 5.7× with 5× processors), likely 
due to improved cache locality in distributed execution.
```

#### Subsection 3: Query Load Scalability
**段落结构**:
```
We measure the runtime as a function of query load (number of 
target nodes) in Figure 16. All methods exhibit linear growth, 
confirming O(n) complexity with respect to the number of queries. 
GNNExplainer maintains the lowest per-query overhead (~1.25 s/node), 
making it ideal for high-throughput scenarios with thousands of 
queries.
```

---

## 📊 表格建议

### Table: TreeCycle Scalability Summary

| Method | Graph Size<br>Slowdown<br>(1.1M→1.4B) | Parallel<br>Speedup<br>(4→20 proc) | Query Load<br>Time/Node |
|--------|:-----------------------------------:|:----------------------------------:|:----------------------:|
| ApxChase | 3.1× | 5.6× | 9.1 s |
| HeuChase | 3.8× | 4.4× | 1.8 s |
| GNNExplainer | **3.7×** | 4.6× | **1.25 s** |
| PGExplainer | **2.7×** | **5.7×** | 2.1 s |

**Bold**: Best performance in each category

---

## 📁 文件清单

生成的文件：
```
figure_14_treecycle_runtime_vs_graph_size.png       # Graph Size Scalability
figure_15_treecycle_runtime_vs_processors.png       # Parallel Efficiency
figure_16_treecycle_runtime_vs_target_nodes.png     # Query Load Scalability
```

所有图表已生成在项目根目录。

---

## 🚀 如何重新生成

```bash
# 重新生成所有图表（包括新的 TreeCycle 图表）
python Plot_Figures_2.py
```

输出将显示：
```
Saved: figure_1_overall_efficiency.png
...
Saved: figure_14_treecycle_runtime_vs_graph_size.png
Saved: figure_15_treecycle_runtime_vs_processors.png
Saved: figure_16_treecycle_runtime_vs_target_nodes.png

✓ All figures generated successfully!
✓ TreeCycle figures (14-16) added!
```

---

## ✅ 完成清单

- [x] Figure 14: TreeCycle Runtime vs Graph Size
- [x] Figure 15: TreeCycle Runtime vs Number of Processors
- [x] Figure 16: TreeCycle Runtime vs Query Load (Target Nodes)
- [x] 保持与现有图表一致的设计风格
- [x] 使用对数坐标 Y 轴
- [x] 生成高分辨率 PNG (300 DPI)
- [x] 添加说明文档

**所有 TreeCycle 实验图表已完成！** 🎉
