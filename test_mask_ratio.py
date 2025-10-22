"""
测试 mask_ratio 功能是否正常工作
"""
import torch
from torch_geometric.data import Data
from src.Edge_masking import mask_edges_by_constraints, mask_edges_for_node_classification
from src.constraints import get_constraints

def test_mask_ratio_basic():
    """测试基础的 mask_ratio 功能"""
    print("=" * 60)
    print("测试 1: 基础 mask_ratio 功能")
    print("=" * 60)
    
    # 创建一个简单的图: 10个节点，20条有向边（10条无向边）
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 0],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8, 0, 9]
    ], dtype=torch.long)
    
    x = torch.randn(10, 16)
    y = torch.randint(0, 3, (10,))
    
    data = Data(x=x, edge_index=edge_index, y=y)
    
    print(f"原始图: {data.num_nodes} 节点, {data.edge_index.size(1)} 条有向边 ({data.edge_index.size(1)//2} 条无向边)")
    
    # 测试不同的 mask_ratio
    constraints = get_constraints("Cora")
    
    for ratio in [0.0, 0.1, 0.2, 0.3]:
        masked_data, dropped = mask_edges_by_constraints(
            data,
            constraints,
            mask_ratio=ratio,
            seed=42
        )
        
        expected_drop = max(1, int((data.edge_index.size(1) // 2) * ratio))
        actual_drop = len(dropped)
        
        print(f"\nmask_ratio={ratio:.1f}: 预期删除 {expected_drop} 条无向边, 实际删除 {actual_drop} 条")
        print(f"  原图: {data.edge_index.size(1)//2} 条无向边 → 新图: {masked_data.edge_index.size(1)//2} 条无向边")
        
        if ratio == 0.0:
            assert actual_drop == 0, "ratio=0.0 应该不删除任何边"
        else:
            # 由于连通性约束，实际删除数量可能小于期望
            assert actual_drop <= expected_drop, f"实际删除数量应该 <= 期望: {actual_drop} > {expected_drop}"
            assert actual_drop >= min(1, expected_drop), f"至少应该删除1条边（如果期望>0）"
    
    print("\n✅ 测试通过：mask_ratio 功能正常")


def test_node_classification_mask():
    """测试节点分类中的 mask_ratio"""
    print("\n" + "=" * 60)
    print("测试 2: 节点分类 L-hop subgraph mask")
    print("=" * 60)
    
    # 创建一个小图
    num_nodes = 20
    # 环状结构
    edges = []
    for i in range(num_nodes):
        j = (i + 1) % num_nodes
        edges.append([i, j])
        edges.append([j, i])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    x = torch.randn(num_nodes, 16)
    y = torch.randint(0, 3, (num_nodes,))
    
    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)
    
    print(f"全图: {num_nodes} 节点, {edge_index.size(1)} 条有向边")
    
    # 测试 L-hop subgraph 提取和 mask
    target_node = 0
    constraints = get_constraints("Cora")
    
    for L in [1, 2]:
        for ratio in [0.0, 0.2]:
            print(f"\nL={L}, mask_ratio={ratio:.1f}:")
            
            masked_subgraph, dropped, node_subset = mask_edges_for_node_classification(
                data,
                target_node,
                constraints,
                num_hops=L,
                mask_ratio=ratio,
                seed=42
            )
            
            subgraph_undirected_edges = masked_subgraph.edge_index.size(1) // 2
            print(f"  L-hop子图: {len(node_subset)} 节点")
            print(f"  删除前估计: ~{(L*2+1)*2//2} 条无向边")  # 粗略估计
            print(f"  删除: {len(dropped)} 条无向边")
            print(f"  删除后: {subgraph_undirected_edges} 条无向边")
            print(f"  Target节点子图ID: {masked_subgraph.target_node_subgraph_id}")
            
            if ratio == 0.0:
                assert len(dropped) == 0
    
    print("\n✅ 测试通过：节点分类 mask 功能正常")


def test_backward_compatibility():
    """测试向后兼容性：max_masks 参数仍然有效"""
    print("\n" + "=" * 60)
    print("测试 3: 向后兼容性 (max_masks)")
    print("=" * 60)
    
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 0],
        [1, 0, 2, 1, 3, 2, 0, 3]
    ], dtype=torch.long)
    
    x = torch.randn(4, 16)
    y = torch.randint(0, 3, (4,))
    
    data = Data(x=x, edge_index=edge_index, y=y)
    constraints = get_constraints("Cora")
    
    print(f"原始图: {data.edge_index.size(1)//2} 条无向边")
    
    # 使用 max_masks (旧方式)
    masked_data1, dropped1 = mask_edges_by_constraints(
        data,
        constraints,
        max_masks=2,  # 删除2条边
        seed=42
    )
    
    print(f"使用 max_masks=2: 删除了 {len(dropped1)} 条边")
    
    # 使用 mask_ratio (新方式)
    masked_data2, dropped2 = mask_edges_by_constraints(
        data,
        constraints,
        mask_ratio=0.5,  # 删除50%的边 (4条无向边 * 0.5 = 2条)
        seed=42
    )
    
    print(f"使用 mask_ratio=0.5: 删除了 {len(dropped2)} 条边")
    
    # mask_ratio 应该覆盖 max_masks
    masked_data3, dropped3 = mask_edges_by_constraints(
        data,
        constraints,
        max_masks=1,
        mask_ratio=0.5,  # 这个应该生效
        seed=42
    )
    
    print(f"同时指定(mask_ratio应覆盖): 删除了 {len(dropped3)} 条边")
    assert len(dropped3) == len(dropped2), "mask_ratio 应该覆盖 max_masks"
    
    print("\n✅ 测试通过：向后兼容性正常")


if __name__ == "__main__":
    try:
        test_mask_ratio_basic()
        test_node_classification_mask()
        test_backward_compatibility()
        
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！mask_ratio 功能正常工作")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
