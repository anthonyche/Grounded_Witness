import json

# 读取 ApxChase 结果
with open('results/MUTAG/apxchase_mutag/metrics_graph_61.json', 'r') as f:
    data = json.load(f)

print("Graph 61 - ApxChase")
print(f"Predicted label: {data['predicted_label']}")
print(f"Prediction confidence (probs): {data['prediction_confidence']}")
print(f"\nWitnesses:")
for i, w in enumerate(data['witnesses'][:3]):
    print(f"  Witness {i}: {w['num_nodes']} nodes, {w['num_edges']} edges")
    print(f"    Fidelity-: {w['fidelity_minus']:.4f}")
    print(f"    Conciseness: {w['conciseness']:.4f}")
    print()

# 读取 GNNExplainer 结果
with open('results/MUTAG/gnnexplainer_mutag/metrics_graph_61.json', 'r') as f:
    data_gnn = json.load(f)
    
print(f"\nGNNExplainer:")
print(f"  Fidelity-: {data_gnn['avg_fidelity_minus']:.4f}")
print(f"  Conciseness: {data_gnn['avg_conciseness']:.4f}")

# 读取 PGExplainer 结果  
with open('results/MUTAG/pgexplainer_mutag/metrics_graph_61.json', 'r') as f:
    data_pg = json.load(f)
    
print(f"\nPGExplainer:")
print(f"  Fidelity-: {data_pg['avg_fidelity_minus']:.4f}")
print(f"  Conciseness: {data_pg['avg_conciseness']:.4f}")
