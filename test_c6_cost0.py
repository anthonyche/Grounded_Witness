import json
import sys
sys.path.append('src')
from matcher import find_head_matches, backchase_repair_cost, _data_to_nx
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from constraints import TGD_C6_CLOSURE_1

# Load witness edges
with open('results/MUTAG/heuchase_mutag/case_graph_61_expl_edges_all.json', 'r') as f:
    data_list = json.load(f)

# Load original graph for node types
dataset = TUDataset(root='datasets/TUDataset', name='MUTAG')
orig_graph = dataset[61]

# Check ALL witnesses
for w_idx in range(len(data_list)):
    witness_edges = data_list[w_idx]['edges']
    
    # Create witness data
    num_nodes = 22
    x = torch.zeros((num_nodes, orig_graph.x.size(1)), dtype=torch.float)
    x[:11] = orig_graph.x
    x[11:, 0] = 1.0  # Assume new nodes are carbons

    edge_index = torch.tensor([[u for u,v in witness_edges] + [v for u,v in witness_edges],
                               [v for u,v in witness_edges] + [u for u,v in witness_edges]], dtype=torch.long)
    witness_data = Data(x=x, edge_index=edge_index)

    # Find all HEAD matches
    matches = find_head_matches(witness_data, TGD_C6_CLOSURE_1)
    print(f'\nWitness {w_idx+1}: {len(witness_edges)} edges, {len(matches)} HEAD matches')

    # Test each match
    witness_nodes = set(range(22))
    cost_0_matches = []

    for i, binding in enumerate(matches):
        result = backchase_repair_cost(witness_data, TGD_C6_CLOSURE_1, binding, B=10, witness_nodes=witness_nodes)
        within_budget, cost, repairs = result
        
        if cost == 0:
            cost_0_matches.append(binding)

    if cost_0_matches:
        print(f'  ✓ Found {len(cost_0_matches)} matches with cost=0!')
        print(f'  First cost=0 match: {cost_0_matches[0]}')
    else:
        print(f'  ✗ No matches with cost=0')
