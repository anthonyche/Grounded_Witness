#!/usr/bin/env python
"""Quick test for Yelp node classification - minimal configuration"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Yelp
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures, ToUndirected
from src.constraints import get_constraints
from src.apxchase import ApxChase


class SimpleGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
    
    def forward(self, data):
        if isinstance(data, Data):
            x, edge_index = data.x, data.edge_index
        else:
            x, edge_index = data
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=-1)


def main():
    print("="*70)
    print("QUICK YELP TEST - Minimal Config")
    print("="*70)
    
    device = torch.device('cpu')
    
    # Load data
    print("\n1. Loading Yelp dataset...")
    dataset = Yelp(root='./datasets', transform=NormalizeFeatures())
    data = dataset[0]
    data.edge_index = ToUndirected()(data).edge_index
    print(f"   Nodes: {data.num_nodes:,}, Edges: {data.edge_index.size(1):,}")
    
    # Load model
    print("\n2. Loading trained model...")
    input_dim = data.x.size(1)
    output_dim = data.y.size(1)
    hidden_dim = 64  # Match trained model
    
    model = SimpleGCN(input_dim, hidden_dim, output_dim).to(device)
    checkpoint = torch.load('models/Yelp_gcn_model.pth', map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"   ✓ Model loaded successfully")
    
    # Generate predictions
    print("\n3. Generating reference predictions...")
    with torch.no_grad():
        out = model(data.to(device))
        data.y_ref = out.argmax(dim=-1).cpu()
    unique_preds = torch.unique(data.y_ref).numel()
    print(f"   ✓ Predictions generated ({unique_preds} unique classes)")
    
    # Load constraints
    print("\n4. Loading Yelp constraints...")
    constraints = get_constraints('YELP')
    print(f"   ✓ Loaded {len(constraints)} constraints")
    for tgd in constraints:
        print(f"      - {tgd['name']}")
    
    # Configure explainer - VERY MINIMAL
    target_node = 2
    config = {
        'L': 1,        # Only 1-hop (tiny subgraph)
        'k': 2,        # Keep only 2 candidates
        'Budget': 2,   # Max 2 edge additions
    }
    
    print(f"\n5. Running ApxChase...")
    print(f"   Target node: {target_node}")
    print(f"   Config: L={config['L']}, k={config['k']}, B={config['Budget']}")
    
    explainer = ApxChase(
        model=model,
        Sigma=constraints,
        L=config['L'],
        k=config['k'],
        B=config['Budget'],
        alpha=1.0,
        beta=0.0,
        gamma=1.0,
        debug=True
    )
    
    try:
        Sigma_star, S_k = explainer.explain_node(data, target_node)
        
        print(f"\n{'='*70}")
        print("✓ SUCCESS!")
        print(f"{'='*70}")
        print(f"Grounded constraints: {len(Sigma_star)}")
        print(f"Witness candidates: {len(S_k)}")
        
        if Sigma_star:
            print(f"\nGrounded constraints:")
            for name in sorted(list(Sigma_star)):
                print(f"  - {name}")
        
        if S_k:
            best = S_k[0]
            print(f"\nBest witness: {best.num_nodes} nodes, {best.edge_index.size(1)} edges")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
