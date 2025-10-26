"""
Quick test for TreeCycle benchmark with timeout mechanism
"""
import torch
import signal
import time

# Test timeout handler
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out")

def slow_function(sleep_time):
    """Simulates a slow operation"""
    print(f"Starting slow function (will sleep for {sleep_time}s)...")
    time.sleep(sleep_time)
    print("Slow function completed!")
    return "success"

def test_timeout():
    """Test timeout mechanism"""
    print("="*60)
    print("Testing Timeout Mechanism")
    print("="*60)
    
    # Test 1: Function completes before timeout
    print("\nTest 1: Function completes within timeout (5s limit, 2s task)")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)
    try:
        result = slow_function(2)
        print(f"✓ Result: {result}")
    except TimeoutException:
        print("✗ Unexpected timeout!")
    finally:
        signal.alarm(0)
    
    # Test 2: Function times out
    print("\nTest 2: Function exceeds timeout (2s limit, 5s task)")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(2)
    try:
        result = slow_function(5)
        print(f"✗ Should have timed out! Result: {result}")
    except TimeoutException as e:
        print(f"✓ Timeout caught correctly: {e}")
    finally:
        signal.alarm(0)
    
    print("\n" + "="*60)
    print("Timeout mechanism working correctly!")
    print("="*60)

def test_explainer_imports():
    """Test that all explainers can be imported"""
    print("\n" + "="*60)
    print("Testing Explainer Imports")
    print("="*60)
    
    import sys
    sys.path.append('src')
    
    try:
        from heuchase import HeuChase
        print("✓ HeuChase imported")
    except Exception as e:
        print(f"✗ HeuChase import failed: {e}")
    
    try:
        from apxchase import ApxChase
        print("✓ ApxChase imported")
    except Exception as e:
        print(f"✗ ApxChase import failed: {e}")
    
    try:
        from exhaustchase import ExhaustChase
        print("✓ ExhaustChase imported")
    except Exception as e:
        print(f"✗ ExhaustChase import failed: {e}")
    
    try:
        from baselines import PGExplainerBaseline, run_gnn_explainer_node
        print("✓ PGExplainer and GNNExplainer imported")
    except Exception as e:
        print(f"✗ Baseline imports failed: {e}")
    
    try:
        from constraints import get_constraints
        constraints = get_constraints('TREECYCLE')
        print(f"✓ TreeCycle constraints loaded ({len(constraints)} constraints)")
    except Exception as e:
        print(f"✗ Constraints loading failed: {e}")

def test_data_loading():
    """Test that TreeCycle data can be loaded"""
    print("\n" + "="*60)
    print("Testing Data Loading")
    print("="*60)
    
    try:
        data = torch.load('datasets/TreeCycle/treecycle_d5_bf15_n813616.pt')
        print(f"✓ TreeCycle data loaded: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
    
    try:
        from torch_geometric.nn import GCNConv
        import torch.nn as nn
        
        class GCN(nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels):
                super().__init__()
                self.conv1 = GCNConv(in_channels, hidden_channels)
                self.conv2 = GCNConv(hidden_channels, out_channels)
            
            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index)
                x = torch.relu(x)
                x = self.conv2(x, edge_index)
                return x
        
        model = GCN(in_channels=data.x.size(1), hidden_channels=32, 
                   out_channels=len(torch.unique(data.y)))
        model.load_state_dict(torch.load('models/TreeCycle_gcn_d5_bf15_n813616.pth'))
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")

if __name__ == '__main__':
    test_timeout()
    test_explainer_imports()
    test_data_loading()
    
    print("\n" + "="*60)
    print("All Pre-flight Checks Completed!")
    print("Ready to run benchmark_treecycle_distributed.py")
    print("="*60)
