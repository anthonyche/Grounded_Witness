#!/usr/bin/env python3
"""
Test TreeCycle distributed benchmark with simplified CPU-only version
Quick test with 10 nodes, 2 workers, HeuChase only
"""

import subprocess
import sys

def main():
    print("="*70)
    print("Testing TreeCycle Distributed Benchmark (CPU-only, simplified)")
    print("="*70)
    print("\nTest configuration:")
    print("  - 10 target nodes")
    print("  - 2 workers")
    print("  - HeuChase only")
    print("  - Pure CPU (like OGBN)")
    print("  - No custom verify functions")
    print("  - No GPU complications")
    print("\nStarting test...\n")
    
    # Run benchmark with minimal settings
    cmd = [
        sys.executable,
        "benchmark_treecycle_distributed_v2.py",
        "--num_targets", "10",
        "--num_workers", "2",
        "--explainer", "heuchase",
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print("\n" + "="*70)
        print("✓ Test completed successfully!")
        print("="*70)
        return 0
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
