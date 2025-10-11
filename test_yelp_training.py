#!/usr/bin/env python3
"""
Quick test script to verify Train_Yelp_HPC.py works correctly
Tests with 1 epoch to ensure everything runs without errors
"""

import sys
import os

# Add src to path
sys.path.insert(0, 'src')

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv, GATConv, SAGEConv
        from torch_geometric.datasets import Yelp
        from torch_geometric.transforms import NormalizeFeatures, ToUndirected
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_dataset_loading():
    """Test if Yelp dataset can be loaded"""
    print("\nTesting dataset loading...")
    try:
        from utils import dataset_func, set_seed
        
        config = {
            'data_name': 'Yelp',
            'data_root': './datasets',
            'random_seed': 42,
            'num_target_nodes': 10,
        }
        
        set_seed(42)
        data_resource = dataset_func(config)
        data = data_resource['data']
        
        print(f"✓ Dataset loaded successfully")
        print(f"  Nodes: {data.num_nodes:,}")
        print(f"  Edges: {data.edge_index.size(1):,}")
        print(f"  Features: {data.x.size(1)}")
        print(f"  Multi-label: {data_resource.get('multi_label', False)}")
        
        return True, data_resource
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_model_creation():
    """Test if models can be created"""
    print("\nTesting model creation...")
    try:
        # Import from Train_Yelp_HPC
        sys.path.insert(0, 'src')
        from Train_Yelp_HPC import GCN, GCN_1, GCN_2, GAT, GraphSAGE, get_model
        
        models = ['gcn1', 'gcn2', 'gcn', 'gat', 'sage']
        
        for model_name in models:
            model = get_model(model_name, input_dim=300, hidden_dim=64, output_dim=100)
            num_params = sum(p.numel() for p in model.parameters())
            print(f"✓ {model_name}: {num_params:,} parameters")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_one_epoch(data_resource):
    """Test if training can run for one epoch"""
    print("\nTesting training (1 epoch)...")
    try:
        import torch
        from Train_Yelp_HPC import get_model, train_node_classification
        
        config = {
            'model_name': 'gcn1',
            'hidden_dim': 64,
            'num_epochs': 1,
            'learning_rate': 0.01,
            'weight_decay': 5e-4,
            'patience': 50,
        }
        
        device = torch.device('cpu')
        data = data_resource['data']
        is_multi_label = data_resource.get('multi_label', False)
        
        input_dim = data.x.size(1)
        output_dim = data.y.size(1) if len(data.y.shape) > 1 else int(data.y.max().item()) + 1
        
        model = get_model('gcn1', input_dim=input_dim, hidden_dim=64, output_dim=output_dim)
        
        print("Running 1 epoch...")
        best_state, best_val, best_test = train_node_classification(
            model, data, config, device, is_multi_label
        )
        
        print(f"✓ Training successful")
        print(f"  Val metric: {best_val:.4f}")
        print(f"  Test acc: {best_test:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*70)
    print("TESTING Train_Yelp_HPC.py")
    print("="*70)
    
    # Test 1: Imports
    if not test_imports():
        print("\n✗ Tests failed: Cannot import required modules")
        print("Please install: pip install torch torch-geometric pyyaml")
        return False
    
    # Test 2: Dataset loading
    success, data_resource = test_dataset_loading()
    if not success:
        print("\n✗ Tests failed: Cannot load Yelp dataset")
        return False
    
    # Test 3: Model creation
    if not test_model_creation():
        print("\n✗ Tests failed: Cannot create models")
        return False
    
    # Test 4: Training
    if not test_training_one_epoch(data_resource):
        print("\n✗ Tests failed: Cannot run training")
        return False
    
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED!")
    print("="*70)
    print("\nThe training script is ready to run on HPC.")
    print("Submit with: sbatch train_yelp.slurm")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
