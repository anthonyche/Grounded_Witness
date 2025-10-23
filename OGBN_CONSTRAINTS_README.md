# OGBN-Papers100M 约束说明

## 概述

为 OGBN-Papers100M 数据集添加了 5 个结构约束（TGDs），用于测试 HeuChase 和 ApxChase 在有约束条件下的运行时性能。

## 数据集特点

- **节点**: 111,059,956 篇论文
- **边**: 1,615,685,872 条引用关系
- **标签**: 172 个 arXiv 学科类别
- **任务**: 节点分类（论文主题预测）

## 约束设计原则

### 1. 基于引用模式
- **Co-citation（共同引用）**: 如果两篇论文 A 和 B 都引用了论文 C，那么 A 和 B 可能有直接引用关系
- **Cross-field（跨领域）**: 相关领域的论文经常互相引用（如 AI ↔ CV ↔ NLP）
- **Hub papers（枢纽论文）**: 理论/系统类论文经常被应用论文引用

### 2. 学科分类
为了确保在 2-hop 子图中能匹配到约束，我们使用了宽泛的学科范围：

```python
CS_AI_ML_LABELS = list(range(0, 30))      # AI, ML, Neural Networks
CS_CV_LABELS = list(range(30, 50))        # Computer Vision
CS_NLP_LABELS = list(range(50, 70))       # NLP, CL, IR
CS_THEORY_LABELS = list(range(70, 100))   # Theory, Algorithms
CS_SYSTEMS_LABELS = list(range(100, 130)) # Systems, Networks
CS_SECURITY_LABELS = list(range(130, 150))# Security, Cryptography
CS_OTHER_LABELS = list(range(150, 172))   # Other areas
```

## 定义的约束

### 1. 同领域共同引用 (ogbn_cocitation_same_field)

**模式**: Triangle completion - 从 co-citation 推断直接引用

```
HEAD (观察到):          BODY (推断):
  A → C                    A → B
  B → C                    
  (A, B 都引用 C)         (A 和 B 直接引用)
```

**适用**: AI/ML/CV/NLP 领域内的论文

**意义**: 如果两篇论文都引用了同一篇论文，它们很可能研究相关主题，应该互相引用。

---

### 2. AI/ML ↔ CV 跨领域桥接 (ogbn_ai_cv_bridge)

**模式**: Cross-field citation via bridge paper

```
HEAD (观察到):               BODY (推断):
  A (AI/ML) → C (bridge)       A → B
  B (CV) → C (bridge)          
  (两者都引用桥接论文)         (跨领域直接引用)
```

**适用**: AI/ML 和 CV 之间

**意义**: 深度学习在视觉领域的应用使得 AI/ML 和 CV 密切相关（如 CNN, ResNet, Vision Transformers）。

---

### 3. AI/ML ↔ NLP 跨领域桥接 (ogbn_ai_nlp_bridge)

**模式**: Cross-field citation via bridge paper

```
HEAD (观察到):               BODY (推断):
  A (AI/ML) → C (bridge)       A → B
  B (NLP) → C (bridge)         
```

**适用**: AI/ML 和 NLP 之间

**意义**: Transformer、BERT、GPT 等模型使得 AI/ML 和 NLP 高度融合。

---

### 4. 理论论文作为枢纽 (ogbn_theory_hub)

**模式**: Theory papers as hubs connecting applied papers

```
HEAD (观察到):               BODY (推断):
  A (applied) → T (theory)     A → B
  B (applied) → T (theory)     
  (应用论文引用理论)           (应用论文互引)
```

**适用**: 理论论文连接应用研究

**意义**: 引用相同理论基础的应用论文往往研究相关问题。

---

### 5. 系统论文作为枢纽 (ogbn_systems_hub)

**模式**: Systems papers as hubs connecting applied papers

```
HEAD (观察到):                  BODY (推断):
  A (applied) → S (systems)      A → B
  B (applied) → S (systems)      
  (应用论文引用系统)              (应用论文互引)
```

**适用**: 系统论文连接应用研究

**意义**: 使用相同系统/工具的应用研究往往解决相关问题（如都用 TensorFlow, PyTorch）。

## 约束验证流程

### 在 HeuChase/ApxChase 中的使用

1. **HEAD Matching**: 
   - 在候选子图上运行 `find_head_matches(subgraph, tgd)`
   - 找到所有满足 HEAD 模式的节点绑定

2. **BODY Verification (Backchase)**:
   - 对每个 HEAD 匹配，计算需要添加多少条边来满足 BODY
   - 如果 repair cost ≤ B (budget)，则约束被 grounded

3. **Coverage Tracking**:
   - 维护 `Sigma_star` 集合记录已 grounded 的约束
   - 窗口 `W_k` 中的 witnesses 按 coverage 和其他指标排序

### 性能指标

```python
result = {
    'num_witnesses': len(S_k),           # 生成的 witnesses 数量
    'coverage': len(Sigma_star),         # grounded 的约束数量
    'runtime': elapsed_time,             # 解释时间
    'grounded_constraints': [...],       # 具体 grounded 的约束名称
}
```

## 配置参数

### HeuChase
```python
{
    'Sigma': CONSTRAINTS_OGBN_PAPERS,  # 5 个约束
    'L': 2,                             # 2-hop subgraph
    'k': 10,                            # window size
    'B': 5,                             # backchase budget
    'm': 6,                             # Edmonds candidates
    'noise_std': 1e-3,                  # diversity noise
}
```

### ApxChase
```python
{
    'Sigma': CONSTRAINTS_OGBN_PAPERS,
    'L': 2,
    'k': 10,
    'B': 5,                             # backchase budget
}
```

## 预期效果

### Runtime 测试

通过在不同 worker 数量下测试，我们可以：

1. **测量约束的影响**:
   - 有约束 vs 无约束的运行时间对比
   - 约束匹配和 backchase 的开销

2. **评估可扩展性**:
   - 约束系统在分布式环境下的性能
   - 负载均衡对有约束任务的影响

3. **Coverage 分析**:
   - 2-hop 子图中能 ground 多少约束
   - 不同约束的匹配频率

## 使用方法

### 快速测试（5 nodes, 2 workers）
```bash
python test_distributed_quick.py
```

### 完整基准测试（100 nodes, 2/4/6/8/10 workers）
```bash
sbatch run_ogbn_distributed_bench.slurm
```

### 检查约束加载
```python
from constraints import get_constraints

# Load OGBN-Papers100M constraints
constraints = get_constraints('OGBN-PAPERS100M')
print(f"Loaded {len(constraints)} constraints:")
for tgd in constraints:
    print(f"  - {tgd['name']}")
    print(f"    HEAD: {len(tgd['head']['edges'])} edges")
    print(f"    BODY: {len(tgd['body']['edges'])} edges")
```

## 输出示例

### 约束加载输出
```
Loading constraints for OGBN-Papers100M...
  Loaded 5 constraints:
    1. ogbn_cocitation_same_field
    2. ogbn_ai_cv_bridge
    3. ogbn_ai_nlp_bridge
    4. ogbn_theory_hub
    5. ogbn_systems_hub
```

### Explainer 运行输出
```
Worker 0: Processing 50 tasks...
  [ApxChase] Candidate #1: add edge (42,137); current |E(G_s)|=15
  [ApxChase] Gamma(G)=3 (new=2); names(new)=['ogbn_cocitation_same_field', 'ogbn_ai_cv_bridge']
  [ApxChase] Scores: conc=0.1234, rpr=0.0567, delta=0.8901
  [ApxChase] Heap push (|W_k| -> 1).
  ...
  [ApxChase] stats: candidates=150, verified=89, admitted=12, final |W_k|=10, |Γ|=4
Worker 0: Completed 50 tasks in 123.45s
```

### 结果统计
```json
{
  "explainer": "heuchase",
  "num_workers": 4,
  "num_nodes": 100,
  "total_time": 567.89,
  "task_runtime_mean": 5.43,
  "coverage_mean": 3.2,
  "num_witnesses_mean": 8.5
}
```

## 与其他数据集的对比

| Dataset | Nodes | Labels | Constraints | Pattern Type |
|---------|-------|--------|-------------|--------------|
| MUTAG | 188 graphs | 7 atom types | 8 | Chemical structure |
| Cora | 2,708 | 7 topics | 5 | Citation triangle |
| BAShape | 2,100 | 4 roles | 3 | House motif |
| **OGBN-Papers** | **111M** | **172 subjects** | **5** | **Co-citation** |

## 技术细节

### 约束存储位置
- **文件**: `src/constraints.py`
- **变量**: `CONSTRAINTS_OGBN_PAPERS`
- **注册**: `_REGISTRY['OGBN-PAPERS100M']`

### 依赖模块
- `src/matcher.py`: HEAD matching, backchase repair
- `src/heuchase.py`: HeuChase algorithm
- `src/apxchase.py`: ApxChase algorithm

### 注意事项

1. **标签范围是估计值**: 实际的 172 个 arXiv 类别映射可能不同，建议根据实际数据调整
2. **约束宽泛性**: 为了在 2-hop 子图中能匹配，约束使用了较宽的标签范围
3. **Budget 设置**: `B=5` 允许最多添加 5 条边来修复约束，可根据需求调整
4. **Runtime vs Coverage**: 约束越多，匹配开销越大，但 coverage 更高

## 参考

- OGBN-Papers100M: https://ogb.stanford.edu/docs/nodeprop/#ogbn-papers100M
- ArXiv categories: https://arxiv.org/category_taxonomy
- TGD (Tuple-Generating Dependencies): Database theory constraint formalism
