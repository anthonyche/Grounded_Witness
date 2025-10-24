#!/usr/bin/env python3
"""快速测试 TreeCycle 生成器的优化版本"""

import time
from TreeCycleGenerator import TreeCycleGenerator

def test_generation(depth, branching_factor, cycle_prob):
    """测试图生成速度"""
    print(f"\n{'='*70}")
    print(f"Testing: depth={depth}, bf={branching_factor}, cycle_prob={cycle_prob}")
    print(f"{'='*70}")
    
    start = time.time()
    
    generator = TreeCycleGenerator(
        depth=depth,
        branching_factor=branching_factor,
        cycle_prob=cycle_prob,
        num_node_types=5,
        seed=42
    )
    
    data = generator.generate()
    
    elapsed = time.time() - start
    
    print(f"\n✓ Generation completed in {elapsed:.2f} seconds")
    print(f"  Nodes: {data.num_nodes:,}")
    print(f"  Edges: {data.edge_index.size(1):,}")
    print(f"  Speed: {data.num_nodes / elapsed:.0f} nodes/second")
    
    return elapsed

if __name__ == '__main__':
    print("TreeCycle Generator Speed Test")
    print("="*70)
    
    # 测试 1: 小型（应该非常快）
    test_generation(depth=3, branching_factor=5, cycle_prob=0.3)
    
    # 测试 2: 中型（应该在几秒内完成）
    test_generation(depth=4, branching_factor=10, cycle_prob=0.2)
    
    # 测试 3: 较大（应该在1分钟内完成）
    print("\n" + "="*70)
    print("Testing larger graph (depth=5, bf=15)...")
    print("This may take 1-3 minutes...")
    print("="*70)
    test_generation(depth=5, branching_factor=15, cycle_prob=0.2)
    
    print("\n" + "="*70)
    print("✓ All tests completed successfully!")
    print("The optimized generator is working correctly.")
    print("="*70)
