# 解释质量评估指标

本项目实现了两个关键的解释质量指标：**Fidelity-** 和 **Conciseness**。

## 1. Fidelity- (忠实度)

### 定义
```
Fidelity- = Pr(M(G)) - Pr(M(G_s))
```

其中：
- `G` 是原始图
- `G_s` 是解释子图（witness或explanation）
- `M` 是训练好的GNN模型
- `Pr` 是模型对目标类别的预测概率

对于MUTAG图分类任务，我们使用原始图的**predicted label**作为目标类别。

### 解释
- **正值**：解释子图的预测概率 < 原图预测概率
  - 表示移除的边包含了对预测重要的结构
  - 解释捕获了关键信息
  
- **负值**：解释子图的预测概率 > 原图预测概率
  - 表示移除的边包含噪声或干扰信息
  - 解释可能过滤了噪声，使预测更清晰
  
- **绝对值越大**：解释的影响越显著

### 计算位置
- `src/utils.py`: `compute_fidelity_minus()` 函数
- `src/Run_Experiment.py`: 在每个方法的`_run_one_graph_*`函数中调用

---

## 2. Conciseness (简洁性)

### 定义
```
Conciseness = 1 - (|E_explanation| / |E_original|)
```

其中：
- `|E_explanation|` 是解释子图的边数
- `|E_original|` 是原始图的边数

### 解释
- **值越高**：解释越简洁（使用更少的边）
- **范围**: [0, 1]
  - 1.0 = 最简洁（没有使用任何边）
  - 0.0 = 使用了所有原图的边
  
- **目标**：在保持高fidelity的同时，最大化conciseness

### 示例
- 原图有26条边，解释子图有12条边
- Conciseness = 1 - 12/26 = 0.538
- 意味着解释使用了约46%的原图边

### 计算位置
- `src/Run_Experiment.py`: 在每个方法的witness/explanation处理循环中计算

---

## 使用方法

### 1. 运行实验并收集指标

```bash
# 运行单个图
python src/Run_Experiment.py --input 0

# 运行完整测试集
python src/Run_Experiment.py --run_all
```

输出示例：
```
===== Fidelity- Statistics =====
Overall Average Fidelity-: -0.146754
Fidelity- per graph: ['-0.0802', '-0.0660', ...]
Min Fidelity-: -0.293687
Max Fidelity-: -0.056055
Std Fidelity-: 0.077813

===== Conciseness Statistics =====
Overall Average Conciseness: 0.538462
Conciseness per graph: ['0.5385', '0.5385', ...]
Min Conciseness: 0.538462
Max Conciseness: 0.538462
Std Conciseness: 0.000000
```

### 2. 分析和对比多个方法

```bash
python analyze_fidelity.py
```

输出包括：
- 每个方法的详细统计
- 按Fidelity-排名
- 按Conciseness排名

---

## 指标存储

所有指标保存在 `results/{method_name}/metrics_graph_{idx}.json`：

```json
{
  "graph_dataset_index": 39,
  "num_witnesses": 1,
  "witnesses": [
    {
      "index": 0,
      "num_edges": 12,
      "fidelity_minus": -0.0802,
      "conciseness": 0.5385
    }
  ],
  "avg_fidelity_minus": -0.0802,
  "avg_conciseness": 0.5385,
  "original_num_edges": 26
}
```

---

## 支持的方法

所有方法都支持这两个指标：

1. **ApxChase** - 流式近似chase算法
2. **ExhaustChase** - 穷举式enforcement + ApxChase
3. **HeuChase** - 启发式候选选择
4. **GNNExplainer** - 基于梯度的解释器（使用top-k边）
5. **PGExplainer** - 参数化解释器（使用top-k边）

注意：GNNExplainer和PGExplainer的conciseness由配置文件中的`topk`参数控制。

---

## 配置参数

在 `config.yaml` 中设置：

```yaml
# GNNExplainer的top-k边数
gnnexplainer_topk: 10

# PGExplainer的top-k边数  
pgexplainer_topk: 10
```

---

## 理想的解释

一个好的解释应该：
- **高Fidelity-**（绝对值）：对预测有显著影响
- **高Conciseness**：使用尽可能少的边
- **平衡权衡**：在简洁性和忠实度之间找到最佳平衡点

---

## 参考文献

这些指标被广泛用于评估GNN解释器的质量：

1. Ying et al. "GNNExplainer: Generating Explanations for Graph Neural Networks" (NeurIPS 2019)
2. Luo et al. "Parameterized Explainer for Graph Neural Network" (NeurIPS 2020)
3. Yuan et al. "Explainability in Graph Neural Networks: A Taxonomic Survey" (TPAMI 2022)
