# ✅ 代码修改完成总结

## 修改内容

###  1. `src/Edge_masking.py` ✅
**新增功能**: 支持 `mask_ratio` 参数（按比例删除边）

**主要变更**:
- `mask_edges_by_constraints()` 添加 `mask_ratio` 参数
- `mask_edges_for_node_classification()` 添加 `mask_ratio` 参数
- 计算逻辑: `删除边数 = int(subgraph_edges * mask_ratio)`
- 特殊处理: `mask_ratio=0.0` 时不删除任何边
- 向后兼容: 仍支持 `max_masks` 参数，但 `mask_ratio` 优先级更高

**代码示例**:
```python
# 删除L-hop子图15%的边
masked_subgraph, dropped_edges = mask_edges_for_node_classification(
    data,
    target_node,
    constraints,
    num_hops=2,
    mask_ratio=0.15,  # 删除15%的边
    seed=42
)
```

---

### 2. `config.yaml` ✅
**新增配置项**: `mask_ratio: 0.15`

**变更**:
```yaml
# 旧配置 (不推荐用于节点分类)
max_masks : 3    # 删除固定3条边

# 新配置 (推荐)
mask_ratio: 0.15  # 删除L-hop子图15%的边
```

---

### 3. `src/Run_Experiment_Node.py` ✅
**更新调用**: 传入 `mask_ratio` 参数

**变更**:
```python
# 旧代码
masked_subgraph, dropped_edges, node_subset = mask_edges_for_node_classification(
    data, target_node, constraints,
    num_hops=config.get("L", 2),
    max_masks=config.get("max_masks", 1),
    seed=config.get("random_seed"),
)

# 新代码
masked_subgraph, dropped_edges, node_subset = mask_edges_for_node_classification(
    data, target_node, constraints,
    num_hops=config.get("L", 2),
    max_masks=config.get("max_masks", 1),
    mask_ratio=config.get("mask_ratio", None),  # 新增
    seed=config.get("random_seed"),
)
```

---

### 4. `src/Run_Experiment.py` ✅
**更新调用**: 所有4处 `mask_edges_by_constraints()` 调用都已更新

---

## 测试结果

### `test_mask_ratio.py` ✅
所有测试通过！

**测试覆盖**:
1. ✅ 基础 `mask_ratio` 功能
   - `mask_ratio=0.0`: 不删除任何边
   - `mask_ratio=0.1, 0.2, 0.3`: 按比例删除
   
2. ✅ 节点分类 L-hop subgraph mask
   - 不同L (1, 2)
   - 不同 `mask_ratio` (0.0, 0.2)
   - 正确提取子图并mask
   
3. ✅ 向后兼容性
   - `max_masks` 参数仍然有效
   - `mask_ratio` 覆盖 `max_masks`

---

## 使用方式

### 方案1: 使用 `mask_ratio` (推荐用于节点分类)
```yaml
# config.yaml
mask_ratio: 0.15  # 删除L-hop子图15%的边
```

**优点**:
- 对不同大小的子图公平
- 更容易理解和调参

**实验参数**:
```python
mask_ratios = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]  # 0% - 25%
```

---

### 方案2: 使用 `max_masks` (保留用于图分类)
```yaml
# config.yaml
max_masks: 3  # 删除固定3条边
```

**适用场景**:
- 图分类任务（MUTAG）
- 图大小相对一致

---

## 实验方案调整

### ❌ 放弃的实验
- **Constraint Size (k)**: Cora只能cover 1个约束，增加k没有实际意义

### ✅ 保留的实验
1. **Runtime varying Incompleteness** (mask_ratio) ⭐⭐⭐
2. **Coverage varying Incompleteness** (mask_ratio) ⭐⭐⭐
3. **Runtime varying L** (hop number) ⭐⭐
4. **Coverage varying L** ⭐⭐
5. **Runtime/Coverage varying Budget** ⭐

详见: `CORA_EXPERIMENT_PLAN.md`

---

## 关键发现

### 1. Mask 是在 L-hop subgraph 上做的 ✅
- **节点分类 (Cora)**: 先提取L-hop子图 → 再mask子图的边
- **图分类 (MUTAG)**: 直接mask整个图的边

### 2. 为什么用比例而不是绝对数量？
- Cora的L-hop子图大小差异很大
  - Hub节点: 50-100条边
  - 普通节点: 10-20条边
- 使用绝对数量（如删除3条边）:
  - 对小图影响大（30%）
  - 对大图影响小（3%）
- 使用比例（如删除15%）:
  - 对所有节点影响一致
  - 更公平的比较

### 3. Constraint Coverage 问题
- Cora实际只能cover 1个约束
- 原因: 2-hop子图太小，无法同时满足多个跨类别的约束
- 解决方案: 
  - 放弃"Constraint Size"实验
  - 改为测试L (hop number)的影响

---

## Next Steps

1. ✅ 代码修改完成
2. ⏳ 运行 Incompleteness 实验
   ```bash
   for ratio in 0.0 0.05 0.10 0.15 0.20 0.25; do
       # 修改 config.yaml 中的 mask_ratio
       python -m src.Run_Experiment_Node --config config.yaml --run_all
   done
   ```
3. ⏳ 运行 L 实验 (L=1, 2, 3)
4. ⏳ 收集数据并更新 `Plot_Figures.py`
5. ⏳ 生成新图表

---

## 文件清单

### 修改的文件
- ✅ `src/Edge_masking.py` (重写，修复bug)
- ✅ `config.yaml` (添加 `mask_ratio: 0.15`)
- ✅ `src/Run_Experiment_Node.py` (传入 `mask_ratio`)
- ✅ `src/Run_Experiment.py` (传入 `mask_ratio`)

### 新增的文件
- ✅ `test_mask_ratio.py` (测试脚本)
- ✅ `CORA_EXPERIMENT_PLAN.md` (实验方案文档)
- ✅ `IMPLEMENTATION_SUMMARY.md` (本文件)

### 备份文件
- `src/Edge_masking_old.py` (原始版本，可删除)

---

## 验证清单

- [x] `mask_ratio=0.0` 时不删除任何边
- [x] `mask_ratio>0` 时按比例删除
- [x] L-hop子图提取正确
- [x] 连通性保持正常工作
- [x] 向后兼容 `max_masks` 参数
- [x] `mask_ratio` 优先级高于 `max_masks`
- [x] 所有测试通过

---

## 注意事项

1. **连通性约束**: 
   - 如果图结构特殊（如环状），可能无法删除期望数量的边
   - 实际删除数量 ≤ 期望数量
   - 这是正常的，保证图连通性

2. **配置优先级**:
   - 如果同时指定 `mask_ratio` 和 `max_masks`
   - `mask_ratio` 会覆盖 `max_masks`
   - 建议只使用一个

3. **日志输出**:
   - 启用 `mask_ratio` 时会打印日志：
     ```
     [mask_edges_by_constraints] Using mask_ratio=0.15: 
         total_edges=20, will_mask=3
     ```
   - 方便调试和验证

---

## 参考

- 实验方案: `CORA_EXPERIMENT_PLAN.md`
- 测试脚本: `test_mask_ratio.py`
- 配置文件: `config.yaml`
