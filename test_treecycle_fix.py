#!/usr/bin/env python3
"""
Quick test of fixed TreeCycle distributed benchmark
Test with 5 nodes, 2 workers, HeuChase only
"""

import subprocess
import sys

def main():
    print("="*70)
    print("Testing FIXED TreeCycle Distributed Benchmark")
    print("="*70)
    print("\nKey fixes applied:")
    print("  ✓ Removed top-level explainer imports")
    print("  ✓ Added dynamic imports inside worker_process (like OGBN)")
    print("  ✓ Removed signal/alarm timeout mechanism")
    print("  ✓ Removed unused timeout_seconds parameter")
    print("\nTest configuration:")
    print("  - 5 target nodes (quick test)")
    print("  - 2 workers")
    print("  - HeuChase only")
    print("  - Pure CPU")
    print("\nStarting test...\n")
    
    # Run benchmark with minimal settings
    cmd = [
        sys.executable,
        "-c",
        """
import torch
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

# Quick inline test
import sys
sys.path.append('.')
exec(open('benchmark_treecycle_distributed_v2.py').read().replace(
    "NUM_TARGETS = 100",
    "NUM_TARGETS = 5"
).replace(
    "NUM_WORKERS = 20",
    "NUM_WORKERS = 2"
).replace(
    "EXPLAINERS = ['heuchase', 'apxchase', 'exhaustchase', 'gnnexplainer', 'pgexplainer']",
    "EXPLAINERS = ['heuchase']"
))
"""
    ]
    
    print(f"Running quick test...\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True, timeout=300)
        print("\n" + "="*70)
        print("✓ Test completed successfully!")
        print("="*70)
        print("\n🎉 The fix worked! Workers can now run without hanging!")
        return 0
    except subprocess.TimeoutExpired:
        print("\n" + "="*70)
        print("✗ Test timeout (5 minutes)")
        print("="*70)
        return 1
    except subprocess.CalledProcessError as e:
        print("\n" + "="*70)
        print("✗ Test failed!")
        print("="*70)
        print(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n\n✗ Test interrupted by user")
        return 130

if __name__ == "__main__":
    sys.exit(main())
