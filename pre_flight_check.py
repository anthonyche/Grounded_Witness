#!/usr/bin/env python3
"""
Pre-flight check for OGBN-Papers100M distributed benchmark
Run this before submitting expensive HPC jobs to catch common issues
"""

import sys
import os

def check_imports():
    """Check all required imports"""
    print("=" * 70)
    print("1. Checking Python imports...")
    print("=" * 70)
    
    errors = []
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
    except ImportError as e:
        errors.append(f"✗ PyTorch not found: {e}")
    
    try:
        import torch_geometric
        print(f"✓ PyG: {torch_geometric.__version__}")
    except ImportError as e:
        errors.append(f"✗ PyG not found: {e}")
    
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
        print(f"✓ OGB: OK")
    except ImportError as e:
        errors.append(f"✗ OGB not found: {e}")
    
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError as e:
        errors.append(f"✗ NumPy not found: {e}")
    
    try:
        import yaml
        print(f"✓ PyYAML: OK")
    except ImportError as e:
        errors.append(f"✗ PyYAML not found: {e}")
    
    try:
        import psutil
        print(f"✓ psutil: {psutil.__version__}")
    except ImportError as e:
        errors.append(f"✗ psutil not found: {e}")
    
    return errors


def check_files():
    """Check required files exist"""
    print("\n" + "=" * 70)
    print("2. Checking required files...")
    print("=" * 70)
    
    errors = []
    required_files = [
        'config.yaml',
        'src/benchmark_ogbn_distributed.py',
        'src/heuchase.py',
        'src/apxchase.py',
        'src/baselines.py',
        'src/constraints.py',
        'src/Train_OGBN_HPC_MiniBatch.py',
        'models/OGBN_Papers100M_epoch_20.pth',
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / 1024**2
            print(f"✓ {file_path} ({size_mb:.2f} MB)")
        else:
            errors.append(f"✗ Missing: {file_path}")
            print(f"✗ Missing: {file_path}")
    
    return errors


def check_model():
    """Check model can be loaded"""
    print("\n" + "=" * 70)
    print("3. Checking model loading...")
    print("=" * 70)
    
    errors = []
    
    try:
        import torch
        sys.path.append('src')
        from Train_OGBN_HPC_MiniBatch import GCN_2_OGBN
        
        model_path = 'models/OGBN_Papers100M_epoch_20.pth'
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print(f"✓ Checkpoint loaded")
        print(f"  Keys: {list(checkpoint.keys())}")
        print(f"  Hidden dim: {checkpoint.get('hidden_dim', 'N/A')}")
        
        model = GCN_2_OGBN(
            input_dim=128,
            hidden_dim=checkpoint['hidden_dim'],
            output_dim=172,
            dropout=0.5
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"✓ Model instantiated and loaded successfully")
        
    except Exception as e:
        errors.append(f"✗ Model loading failed: {e}")
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
    
    return errors


def check_constraints():
    """Check constraints can be loaded"""
    print("\n" + "=" * 70)
    print("4. Checking constraints...")
    print("=" * 70)
    
    errors = []
    
    try:
        sys.path.append('src')
        from constraints import get_constraints
        
        constraints = get_constraints('OGBN-PAPERS100M')
        print(f"✓ Loaded {len(constraints)} constraints:")
        for i, tgd in enumerate(constraints, 1):
            print(f"  {i}. {tgd['name']}")
        
        if len(constraints) == 0:
            errors.append("✗ No constraints found!")
        
    except Exception as e:
        errors.append(f"✗ Constraint loading failed: {e}")
        print(f"✗ Constraint loading failed: {e}")
        import traceback
        traceback.print_exc()
    
    return errors


def check_explainers():
    """Check explainers can be imported"""
    print("\n" + "=" * 70)
    print("5. Checking explainer imports...")
    print("=" * 70)
    
    errors = []
    
    try:
        sys.path.append('src')
        from heuchase import HeuChase
        print(f"✓ HeuChase imported")
    except Exception as e:
        errors.append(f"✗ HeuChase import failed: {e}")
        print(f"✗ HeuChase import failed: {e}")
    
    try:
        from apxchase import ApxChase
        print(f"✓ ApxChase imported")
    except Exception as e:
        errors.append(f"✗ ApxChase import failed: {e}")
        print(f"✗ ApxChase import failed: {e}")
    
    try:
        from baselines import run_gnn_explainer_node
        print(f"✓ GNNExplainer (baselines) imported")
    except Exception as e:
        errors.append(f"✗ GNNExplainer import failed: {e}")
        print(f"✗ GNNExplainer import failed: {e}")
    
    return errors


def check_cache_consistency():
    """Check cache filename generation is consistent"""
    print("\n" + "=" * 70)
    print("6. Checking cache filename consistency...")
    print("=" * 70)
    
    errors = []
    
    try:
        import hashlib
        
        # Test node IDs
        node_ids = [1, 2, 3, 4, 5]
        num_hops = 2
        
        # Method 1: load_tasks_from_cache_only
        node_ids_tuple = tuple(sorted(node_ids))
        node_str = str(node_ids_tuple)
        hash_val1 = hashlib.md5(node_str.encode()).hexdigest()[:8]
        cache_file1 = f"cache/subgraphs/subgraphs_hops{num_hops}_nodes{len(node_ids)}_hash{hash_val1}.pt"
        
        # Method 2: Coordinator.create_tasks (after fix)
        node_ids_tuple = tuple(sorted(node_ids))
        node_str = str(node_ids_tuple)
        hash_val2 = hashlib.md5(node_str.encode()).hexdigest()[:8]
        cache_file2 = f"cache/subgraphs/subgraphs_hops{num_hops}_nodes{len(node_ids)}_hash{hash_val2}.pt"
        
        if cache_file1 == cache_file2:
            print(f"✓ Cache filename generation is consistent")
            print(f"  Test filename: {cache_file1}")
        else:
            errors.append(f"✗ Cache filename mismatch!")
            print(f"✗ Method 1: {cache_file1}")
            print(f"✗ Method 2: {cache_file2}")
        
    except Exception as e:
        errors.append(f"✗ Cache check failed: {e}")
        print(f"✗ Cache check failed: {e}")
    
    return errors


def check_config():
    """Check config.yaml is valid"""
    print("\n" + "=" * 70)
    print("7. Checking config.yaml...")
    print("=" * 70)
    
    errors = []
    
    try:
        import yaml
        
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✓ config.yaml loaded successfully")
        
        # Check critical fields
        critical_fields = {
            'L': int,
            'k': int,
            'Budget': int,
            'device': str,
            'num_target_nodes': int,
        }
        
        for field, expected_type in critical_fields.items():
            if field in config:
                if isinstance(config[field], expected_type):
                    print(f"✓ {field}: {config[field]}")
                else:
                    errors.append(f"✗ {field} has wrong type (expected {expected_type})")
                    print(f"✗ {field}: {config[field]} (wrong type)")
            else:
                errors.append(f"✗ Missing field: {field}")
                print(f"✗ Missing field: {field}")
        
    except Exception as e:
        errors.append(f"✗ Config loading failed: {e}")
        print(f"✗ Config loading failed: {e}")
    
    return errors


def check_memory():
    """Check available memory"""
    print("\n" + "=" * 70)
    print("8. Checking system memory...")
    print("=" * 70)
    
    errors = []
    
    try:
        import psutil
        
        mem = psutil.virtual_memory()
        mem_total_gb = mem.total / 1024**3
        mem_available_gb = mem.available / 1024**3
        
        print(f"✓ Total memory: {mem_total_gb:.2f} GB")
        print(f"✓ Available memory: {mem_available_gb:.2f} GB")
        
        # OGBN-Papers100M needs ~78GB
        if mem_available_gb < 80:
            print(f"⚠️  Warning: May need more memory (OGBN-Papers100M uses ~78GB)")
            print(f"⚠️  Consider using --use-cache-only after generating cache")
        else:
            print(f"✓ Sufficient memory for full dataset loading")
        
    except Exception as e:
        print(f"⚠️  Could not check memory: {e}")
    
    return errors


def main():
    """Run all checks"""
    print("\n" + "=" * 70)
    print("OGBN-Papers100M Distributed Benchmark - Pre-flight Check")
    print("=" * 70)
    
    all_errors = []
    
    # Run all checks
    all_errors.extend(check_imports())
    all_errors.extend(check_files())
    all_errors.extend(check_model())
    all_errors.extend(check_constraints())
    all_errors.extend(check_explainers())
    all_errors.extend(check_cache_consistency())
    all_errors.extend(check_config())
    all_errors.extend(check_memory())
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if all_errors:
        print(f"❌ Found {len(all_errors)} error(s):")
        for error in all_errors:
            print(f"  {error}")
        print("\n⚠️  Please fix these issues before running the benchmark!")
        sys.exit(1)
    else:
        print("✅ All checks passed! Ready to run benchmark.")
        print("\nTo run:")
        print("  sbatch run_ogbn_distributed_bench.slurm")
        print("\nOr generate cache first:")
        print("  sbatch generate_cache.slurm")
        sys.exit(0)


if __name__ == '__main__':
    main()
