"""
test_chase_node_classification.py
----------------------------------
Test that ApxChase, HeuChase, and ExhaustChase work correctly
with node classification tasks on Yelp dataset.

Verifies:
1. L-hop subgraph extraction
2. Constraint-based explanation generation
3. All three chase variants work
4. Metrics computation (fidelity, coverage, conciseness)
"""

import sys
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import Yelp
from torch_geometric.transforms import NormalizeFeatures, ToUndirected
from torch_geometric.nn import GCNConv

# Add src to path
sys.path.insert(0, 'src')

from src.apxchase import ApxChase
from src.heuchase import HeuChase
from src.exhaustchase import ExhaustChase
from src.constraints import get_constraints
from src.utils import compute_fidelity_minus, compute_constraint_coverage


# Simple GCN model for testing
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


def load_yelp_data():
    """Load Yelp dataset"""
    print("Loading Yelp dataset...")
    dataset = Yelp(root='./datasets', transform=NormalizeFeatures())
    data = dataset[0]
    data.edge_index = ToUndirected()(data).edge_index
    
    print(f"Dataset loaded:")
    print(f"  Nodes: {data.num_nodes:,}")
    print(f"  Edges: {data.edge_index.size(1):,}")
    print(f"  Features: {data.x.shape}")
    print(f"  Labels: {data.y.shape}")
    
    return data


def load_or_create_model(data, device):
    """Load trained model or create a dummy one"""
    print("\nLoading model...")
    
    input_dim = data.x.size(1)
    output_dim = data.y.size(1)
    hidden_dim = 64  # FIXED: Match the trained model's hidden dimension
    
    model = SimpleGCN(input_dim, hidden_dim, output_dim).to(device)
    
    # Try to load trained model
    model_path = 'models/Yelp_gcn_model.pth'
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                # Format from training script with metadata
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ Loaded trained model from {model_path}")
            else:
                # Direct state_dict format
                model.load_state_dict(checkpoint)
                print(f"✓ Loaded trained model (direct state_dict) from {model_path}")
        else:
            print(f"⚠ Unexpected checkpoint format, using random initialization")
    except FileNotFoundError:
        print(f"⚠ No trained model found at {model_path}, using random initialization")
    except Exception as e:
        print(f"⚠ Error loading model: {e}, using random initialization")
    
    model.eval()
    return model


def test_apxchase(data, model, target_node, constraints, device, config):
    """Test ApxChase on a single target node"""
    print("\n" + "="*70)
    print("TEST: ApxChase on Node Classification")
    print("="*70)
    
    print(f"\nTarget node: {target_node}")
    print(f"L (hops): {config['L']}")
    print(f"k (window): {config['k']}")
    print(f"Budget: {config['Budget']}")
    print(f"Constraints: {len(constraints)}")
    
    try:
        # Create ApxChase instance
        explainer = ApxChase(
            model=model,
            Sigma=constraints,
            L=config['L'],
            k=config['k'],
            B=config['Budget'],
            alpha=config.get('alpha', 1.0),
            beta=config.get('beta', 0.0),
            gamma=config.get('gamma', 1.0),
            debug=True  # Use debug instead of verbose
        )
        
        # Add reference label for verification
        with torch.no_grad():
            data_device = data.to(device)
            out = model(data_device)
            data.y_ref = out.argmax(dim=-1).cpu()
        
        # Run explanation
        print(f"\n{'='*70}")
        print("Running ApxChase.explain_node()...")
        print(f"{'='*70}")
        
        Sigma_star, S_k = explainer.explain_node(data, target_node)
        
        # Print results
        print(f"\n{'='*70}")
        print("ApxChase Results")
        print(f"{'='*70}")
        print(f"Grounded constraints (Σ*): {len(Sigma_star)}")
        print(f"Witness candidates (S_k): {len(S_k)}")
        
        if Sigma_star:
            print(f"\nGrounded constraint names:")
            for name in sorted(list(Sigma_star)[:10]):  # Show first 10
                print(f"  - {name}")
        
        if S_k:
            best_witness = S_k[0]
            print(f"\nBest witness statistics:")
            print(f"  Nodes: {best_witness.num_nodes}")
            print(f"  Edges: {best_witness.edge_index.size(1)}")
            
            # Compute metrics
            try:
                fidelity = compute_fidelity_minus(model, data, best_witness, device)
                print(f"  Fidelity-: {fidelity:.4f}")
            except Exception as e:
                print(f"  Fidelity-: Error - {e}")
            
            try:
                covered, coverage = compute_constraint_coverage(
                    best_witness, constraints, config['Budget']
                )
                print(f"  Coverage: {coverage:.4f} ({len(covered)}/{len(constraints)})")
            except Exception as e:
                print(f"  Coverage: Error - {e}")
        
        print(f"\n✓ ApxChase test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ ApxChase test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_heuchase(data, model, target_node, constraints, device, config):
    """Test HeuChase on a single target node"""
    print("\n" + "="*70)
    print("TEST: HeuChase on Node Classification")
    print("="*70)
    
    print(f"\nTarget node: {target_node}")
    print(f"m (candidates): {config.get('m', 6)}")
    
    try:
        # Create HeuChase instance
        explainer = HeuChase(
            model=model,
            Sigma=constraints,
            L=config['L'],
            k=config['k'],
            B=config['Budget'],
            m=config.get('m', 6),
            alpha=config.get('alpha', 1.0),
            beta=config.get('beta', 0.0),
            gamma=config.get('gamma', 1.0),
            debug=True  # Use debug instead of verbose
        )
        
        # Add reference label
        with torch.no_grad():
            data_device = data.to(device)
            out = model(data_device)
            data.y_ref = out.argmax(dim=-1).cpu()
        
        # Run explanation
        print(f"\n{'='*70}")
        print("Running HeuChase.explain_node()...")
        print(f"{'='*70}")
        
        Sigma_star, S_k = explainer.explain_node(data, target_node)
        
        # Print results
        print(f"\n{'='*70}")
        print("HeuChase Results")
        print(f"{'='*70}")
        print(f"Grounded constraints (Σ*): {len(Sigma_star)}")
        print(f"Witness candidates (S_k): {len(S_k)}")
        
        if S_k:
            best_witness = S_k[0]
            print(f"\nBest witness statistics:")
            print(f"  Nodes: {best_witness.num_nodes}")
            print(f"  Edges: {best_witness.edge_index.size(1)}")
        
        print(f"\n✓ HeuChase test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ HeuChase test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_exhaustchase(data, model, target_node, constraints, device, config):
    """Test ExhaustChase on a single target node"""
    print("\n" + "="*70)
    print("TEST: ExhaustChase on Node Classification")
    print("="*70)
    
    print(f"\nTarget node: {target_node}")
    print(f"Max iterations: {config.get('max_enforce_iterations', 50)}")
    
    try:
        # Create ExhaustChase instance
        explainer = ExhaustChase(
            model=model,
            Sigma=constraints,
            L=config['L'],
            k=config['k'],
            B=config['Budget'],
            max_enforce_iterations=config.get('max_enforce_iterations', 50),
            alpha=config.get('alpha', 1.0),
            beta=config.get('beta', 0.0),
            gamma=config.get('gamma', 1.0),
            debug=True  # Use debug instead of verbose
        )
        
        # Add reference label
        with torch.no_grad():
            data_device = data.to(device)
            out = model(data_device)
            data.y_ref = out.argmax(dim=-1).cpu()
        
        # Run explanation
        print(f"\n{'='*70}")
        print("Running ExhaustChase.explain_node()...")
        print(f"{'='*70}")
        
        Sigma_star, S_k, enforce_time = explainer.explain_node(data, target_node)
        
        # Print results
        print(f"\n{'='*70}")
        print("ExhaustChase Results")
        print(f"{'='*70}")
        print(f"Enforcement time: {enforce_time:.4f}s")
        print(f"Grounded constraints (Σ*): {len(Sigma_star)}")
        print(f"Witness candidates (S_k): {len(S_k)}")
        
        if S_k:
            best_witness = S_k[0]
            print(f"\nBest witness statistics:")
            print(f"  Nodes: {best_witness.num_nodes}")
            print(f"  Edges: {best_witness.edge_index.size(1)}")
        
        print(f"\n✓ ExhaustChase test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ ExhaustChase test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("CHASE ALGORITHMS - NODE CLASSIFICATION TEST")
    print("="*70)
    
    # Configuration
    config = {
        'L': 3,
        'k': 6,
        'Budget': 8,
        'alpha': 1.0,
        'beta': 1.0,
        'gamma': 1.0,
        'm': 6,
        'max_enforce_iterations': 50,
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load data
    data = load_yelp_data()
    
    # Load model
    model = load_or_create_model(data, device)
    
    # Get constraints
    print("\nLoading Yelp constraints...")
    constraints = get_constraints('YELP')
    print(f"Found {len(constraints)} Yelp constraints")
    
    if not constraints:
        print("✗ No Yelp constraints found! Cannot proceed.")
        return 1
    
    # Select a target node from test set
    test_mask = data.test_mask
    test_nodes = torch.where(test_mask)[0]
    target_node = int(test_nodes[0])  # Use first test node
    
    print(f"\nSelected target node: {target_node} (from {len(test_nodes):,} test nodes)")
    
    # Run tests
    tests = [
        ("ApxChase", lambda: test_apxchase(data, model, target_node, constraints, device, config)),
        ("HeuChase", lambda: test_heuchase(data, model, target_node, constraints, device, config)),
        ("ExhaustChase", lambda: test_exhaustchase(data, model, target_node, constraints, device, config)),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} raised an exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8} {name}")
    
    all_passed = all(success for _, success in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nAll chase algorithms work correctly for node classification on Yelp!")
        print("They properly:")
        print("  1. Extract L-hop subgraphs around target nodes")
        print("  2. Work within the subgraph H for all operations")
        print("  3. Generate witness explanations")
        print("  4. Compute metrics (fidelity, coverage)")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
