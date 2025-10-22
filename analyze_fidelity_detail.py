import json

with open('results/MUTAG/apxchase_mutag/metrics_graph_61.json', 'r') as f:
    data = json.load(f)

print("="*60)
print("Graph 61 Analysis")
print("="*60)
print(f"\nOriginal Graph:")
print(f"  Predicted label: {data['predicted_label']}")
print(f"  Probabilities: {data['prediction_confidence']}")
print(f"  Class 0: {data['prediction_confidence'][0]:.4f}")
print(f"  Class 1: {data['prediction_confidence'][1]:.4f}")

print(f"\nWitnesses (Explanation Subgraphs):")
print("-"*60)

for i, w in enumerate(data['witnesses'][:3]):
    print(f"\nWitness {i}:")
    print(f"  Nodes: {w['num_nodes']}, Edges: {w['num_edges']}")
    print(f"  Fidelity-: {w['fidelity_minus']:.6f}")
    
    # 根据 Fidelity- = Pr(G) - Pr(Gs) 推算 Pr(Gs)
    prob_original = data['prediction_confidence'][1]  # Class 1
    prob_subgraph = prob_original - w['fidelity_minus']
    
    print(f"  Pr(M(G)) = {prob_original:.6f}")
    print(f"  Pr(M(Gs)) = {prob_subgraph:.6f} (推算)")
    print(f"  Difference: {w['fidelity_minus']:.6f}")
    
    if w['fidelity_minus'] < 0:
        print(f"  ⚠️  子图预测概率更高 (+{abs(w['fidelity_minus']):.2%})")
    else:
        print(f"  ✓  子图预测概率下降 (-{w['fidelity_minus']:.2%})")

print("\n" + "="*60)
print("解释:")
print("="*60)
print("Fidelity- = Pr(M(G)) - Pr(M(Gs))")
print("  - 正值: 子图预测概率下降（信息丢失）")
print("  - 负值: 子图预测概率上升（去噪/聚焦）")
print("  - 接近0: 子图保留了关键信息")
