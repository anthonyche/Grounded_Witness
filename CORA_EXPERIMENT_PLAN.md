# Cora Impact of Factor 实验方案

## ⚠️ 关键发现

### Constraint Coverage 问题
- **Cora 只能 cover 1个约束**（通常是 `cora_citation_triangle`）
- **原因**: Cora的约束依赖节点标签(类别)匹配，2-hop子图内很难同时满足多个不同类别的约束
- **结论**: **放弃 "Constraint Size" 实验**，因为：
  - 复制约束没有意义（实际工作量不变，只是coverage指标下降）
  - 增加约束数量时，如果实际能match的不变，时间也不会变化
  - 只有真正能match且需要repair的constraint才会影响时间

### Edge Masking 机制
- ✅ **确认**: Mask 是在 **L-hop subgraph** 上做的（不是全图）
- ✅ **已改为比例**: `mask_ratio` (0.0-1.0)，例如 0.15 = 删除子图15%的边
- 计算方式: `删除边数 = int(subgraph_edges * mask_ratio)`

---

## 📋 实验方案（调整后）

### Overall 实验（所有数据集）

已有数据：
- ✅ Runtime on different datasets (Figure 1)
- ✅ Fidelity- (Figure 5)  
- ✅ Conciseness (Figure 6)

### Cora 上的 Impact of Factor 实验

#### ✅ 已有实验
1. Runtime varying L (Figure 3)
2. Runtime varying k (window size) (Figure 4)
3. Conciseness varying k (Figure 9)

#### 🆕 需要新增的实验

---

### **实验1: Runtime varying Incompleteness (mask_ratio)** ⭐⭐⭐
**优先级: 最高**

```yaml
# config.yaml 设置
L: 2              # 固定
k: 4              # 固定
Budget: 8         # 固定
mask_ratio: [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]  # 依次运行
```

**数据结构**:
```python
df_cora_runtime_incompleteness = {
    "mask_ratio": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25],
    "ApxIChase": [],    # 需要运行实验
    "HeuIChase": [],
    "GNNExplainer": [],
    "PGExplainer": [],
    "Exhaustive": [],
}
```

**预期结果**:
- Mask越多，子图越小，时间可能略微下降
- ApxChase/HeuChase 通过 backchase 能部分修复，应该比 baseline 更鲁棒
- 展示算法在不完整图上的鲁棒性

**运行命令**:
```bash
# 分别运行不同 mask_ratio
for ratio in 0.0 0.05 0.10 0.15 0.20 0.25; do
    # 修改 config.yaml 中的 mask_ratio
    sed -i '' "s/^mask_ratio:.*/mask_ratio: $ratio/" config.yaml
    
    # 运行实验
    python -m src.Run_Experiment_Node --config config.yaml --run_all
done
```

---

### **实验2: Coverage varying Incompleteness** ⭐⭐⭐
**优先级: 最高**

```yaml
# config.yaml 设置
L: 2
k: 4
Budget: 8
mask_ratio: [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
```

**数据结构**:
```python
df_cora_coverage_incompleteness = {
    "mask_ratio": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25],
    "ApxIChase": [],    # Average coverage ratio
    "HeuIChase": [],
    "GNNExplainer": [],
    "PGExplainer": [],
    "Exhaustive": [],
}
```

**预期结果**:
- Mask越多，coverage可能下降（constraint match变少）
- 但 ApxChase/Exhaustive 通过 backchase 能部分恢复 coverage
- 展示 backchase 的修复能力

---

### **实验3: Runtime varying L (hop number)** ⭐⭐
**优先级: 中**（可能已有部分数据）

```yaml
L: [1, 2, 3]
k: 4
Budget: 8
mask_ratio: 0.15
```

**数据结构**:
```python
df_cora_runtime_L = {
    "L": [1, 2, 3],
    "ApxIChase": [],
    "HeuIChase": [],
    "GNNExplainer": [],
    "PGExplainer": [],
    "Exhaustive": [],
}
```

**预期结果**:
- L越大，子图越大，时间增加
- L越大，能match的constraint越多，coverage增加

---

### **实验4: Coverage varying L** ⭐⭐
**优先级: 中**

```yaml
L: [1, 2, 3]
k: 4
Budget: 8
mask_ratio: 0.15
```

**数据结构**:
```python
df_cora_coverage_L = {
    "L": [1, 2, 3],
    "ApxIChase": [],
    "HeuIChase": [],
    "GNNExplainer": [0, 0, 0],  # GNN方法不用constraint
    "PGExplainer": [0, 0, 0],
    "Exhaustive": [],
}
```

**预期结果**:
- L=1: coverage ~ 0.2 (很小的子图，难以match constraint)
- L=2: coverage ~ 0.4-0.6 (能match 1-2个constraint)
- L=3: coverage ~ 0.6-0.8 (更大子图，可能match更多)

---

### **实验5: Runtime/Coverage varying Budget** ⭐
**优先级: 低**（如果时间允许）

```yaml
L: 2
k: 4
Budget: [2, 4, 6, 8, 10]
mask_ratio: 0.15
```

**数据结构**:
```python
df_cora_runtime_budget = {
    "Budget": [2, 4, 6, 8, 10],
    "ApxIChase": [],
    "HeuIChase": [],
    "GNNExplainer": [],  # 不受Budget影响
    "PGExplainer": [],
    "Exhaustive": [],
}

df_cora_coverage_budget = {
    "Budget": [2, 4, 6, 8, 10],
    "ApxIChase": [],
    "HeuIChase": [],
    "Exhaustive": [],
}
```

**预期结果**:
- Budget越大，允许的repair cost越大，coverage增加
- 但时间也可能增加（需要尝试更多repair路径）

---

## ❌ 放弃的实验

### ~~Constraint Size 实验~~ 
**原因**: 
- Cora 实际只能 cover 1个约束
- 复制约束不会改变实际工作量
- 只会让 coverage 指标人为下降（分母增大）
- 不能体现真实的"增加约束数量"效果

### ~~Number of Target Nodes 实验~~
**原因**:
- 这是 scalability 实验，不是 impact of factor
- 可以放到 overall scalability analysis 中

---

## 🎯 实验执行优先级

1. **立即执行** (展示核心贡献):
   - Runtime varying Incompleteness
   - Coverage varying Incompleteness

2. **尽快执行** (补充完整性):
   - Runtime varying L (可能已有)
   - Coverage varying L

3. **时间允许** (额外分析):
   - Runtime/Coverage varying Budget

---

## 📊 预期图表

### Figure X: Runtime vs Incompleteness (mask_ratio)
- X轴: mask_ratio (0%, 5%, 10%, 15%, 20%, 25%)
- Y轴: Average runtime (seconds)
- 5条线: ApxIChase, HeuIChase, GNNExplainer, PGExplainer, Exhaustive

**预期**: ApxIChase/HeuIChase 曲线较平稳（鲁棒），baseline 可能波动较大

### Figure Y: Coverage vs Incompleteness  
- X轴: mask_ratio
- Y轴: Average coverage ratio (0-1)
- 5条线（GNN方法可能接近0）

**预期**: ApxIChase/Exhaustive 曲线下降较慢（backchase修复能力强）

### Figure Z: Coverage vs L (hop number)
- X轴: L (1, 2, 3)
- Y轴: Average coverage ratio
- 3条线: ApxIChase, HeuIChase, Exhaustive

**预期**: L越大，coverage越高（更大子图，更容易match constraint）

---

## 🔧 代码修改总结

### ✅ 已完成
1. `Edge_masking.py`: 添加 `mask_ratio` 参数支持
2. `config.yaml`: 添加 `mask_ratio: 0.15` 配置
3. `Run_Experiment_Node.py`: 传入 `mask_ratio` 参数
4. `Run_Experiment.py`: 传入 `mask_ratio` 参数（图分类任务）

### 使用方式
```yaml
# config.yaml
mask_ratio: 0.15  # 删除L-hop子图15%的边

# 或者使用旧的绝对数量方式
max_masks: 3      # 删除固定3条边（不推荐用于节点分类）
```

### 注意事项
- `mask_ratio` 优先级高于 `max_masks`
- 如果同时指定，`mask_ratio` 会覆盖 `max_masks`
- 对于节点分类（Cora），**强烈建议使用 `mask_ratio`**
- 对于图分类（MUTAG），两种方式都可以

---

## 💡 论文写作建议

### 强调的点
1. **Incompleteness 实验很有意义**：
   - 现实中的图数据往往是不完整的
   - 展示算法的鲁棒性（robustness）
   - Backchase 能修复部分缺失信息

2. **放弃 Constraint Size 实验是合理的**：
   - 诚实地说明："Cora只能cover 1个约束，增加约束数量没有实际意义"
   - 改为分析："为什么只能cover 1个？" → 因为2-hop子图太小
   - 引出：L (hop number) 实验更有意义

3. **Mask比例 vs 绝对数量**：
   - 说明为什么用比例更合理（子图大小不一）
   - 这是对节点分类任务的改进

### 可能的 Limitation
- Cora的约束设计可能需要优化（更适合小子图的约束）
- 可以讨论：如何设计更容易在L-hop子图上match的约束

---

## 📝 Next Steps

1. ✅ 代码修改已完成
2. ⏳ 运行 Incompleteness 实验（mask_ratio: 0.0 → 0.25）
3. ⏳ 运行 L 实验（L: 1, 2, 3）
4. ⏳ 收集数据，更新 `Plot_Figures.py`
5. ⏳ 生成新的图表
6. ⏳ 更新论文
