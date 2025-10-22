import json
import torch

# 读取结果
with open('results/MUTAG/apxchase_mutag/metrics_graph_61.json', 'r') as f:
    data = json.load(f)

# 加载 masked graph
masked_graph = torch.load('results/MUTAG/apxchase_mutag/masked_graph_61.pt')

print("="*70)
print("Graph 61 - Masking Impact Analysis")
print("="*70)

print(f"\n1. Original Graph (Clean):")
if hasattr(masked_graph, '_clean'):
    clean_graph = masked_graph._clean
    print(f"   Nodes: {clean_graph.num_nodes}")
    print(f"   Edges: {clean_graph.edge_index.size(1)}")
else:
    print("   No clean graph reference stored")

print(f"\n2. Masked Graph (Input to explainer):")
print(f"   Nodes: {masked_graph.num_nodes}")
print(f"   Edges: {masked_graph.edge_index.size(1)}")
print(f"   Dropped edges: {data['num_dropped_edges']} (undirected pairs)")
print(f"   Dropped edge list: {data['dropped_edges'][:5]}..." if len(data['dropped_edges']) > 5 else f"   Dropped edge list: {data['dropped_edges']}")

print(f"\n3. Model Prediction on Masked Graph:")
print(f"   Predicted label: {data['predicted_label']}")
print(f"   Class 0 prob: {data['prediction_confidence'][0]:.4f}")
print(f"   Class 1 prob: {data['prediction_confidence'][1]:.4f}")

print(f"\n4. Witnesses (Explanations):")
for i, w in enumerate(data['witnesses'][:3]):
    print(f"\n   Witness {i}:")
    print(f"     Nodes: {w['num_nodes']}, Edges: {w['num_edges']}")
    print(f"     Fidelity-: {w['fidelity_minus']:.6f}")
    
    # 计算子图的预测概率
    prob_masked = data['prediction_confidence'][1]
    prob_witness = prob_masked - w['fidelity_minus']
    
    print(f"     Pr(M(masked)) = {prob_masked:.4f}")
    print(f"     Pr(M(witness)) = {prob_witness:.4f}")
    print(f"     Improvement: {(prob_witness - prob_masked):.4f} ({(prob_witness - prob_masked)/prob_masked * 100:+.2f}%)")

print("\n" + "="*70)
print("分析:")
print("="*70)
print("✓ 模型在干净图上训练")
print("✓ Masked graph 移除了一些边（模拟噪声/缺失）")
print("✓ Witness 可能通过以下方式改善预测:")
print("  1. 去除了 masked graph 中的噪声边")
print("  2. 通过 repair edges 恢复了被错误 mask 的关键边")
print("  3. 聚焦于最相关的子结构")
print("\n这解释了为什么 Fidelity- 是负值：")
print("  Witness 比有噪声的 masked graph 更接近模型训练时的干净数据！")
