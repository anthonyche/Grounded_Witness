"""
测试 OGBN-Papers100M 约束加载和验证
"""

import sys
sys.path.append('src')

from constraints import get_constraints, validate_tgd

print("=" * 70)
print("OGBN-Papers100M 约束加载测试")
print("=" * 70)

# Test 1: Load constraints
print("\n[1/3] 加载约束...")
try:
    constraints = get_constraints('OGBN-PAPERS100M')
    print(f"  ✓ 成功加载 {len(constraints)} 个约束")
except Exception as e:
    print(f"  ✗ 加载失败: {e}")
    exit(1)

# Test 2: Validate each constraint
print("\n[2/3] 验证约束结构...")
for i, tgd in enumerate(constraints, 1):
    try:
        validate_tgd(tgd)
        name = tgd['name']
        head_edges = len(tgd['head']['edges'])
        body_edges = len(tgd['body']['edges'])
        head_nodes = len(tgd['head']['nodes'])
        body_nodes = len(tgd['body']['nodes'])
        
        print(f"  {i}. {name}")
        print(f"     HEAD: {head_nodes} nodes, {head_edges} edges")
        print(f"     BODY: {body_nodes} nodes, {body_edges} edges")
        
        # Check node label ranges
        for part in ['head', 'body']:
            for var, spec in tgd[part]['nodes'].items():
                if 'in' in spec:
                    labels = spec['in']
                    if labels:
                        print(f"       {part}.{var}: labels {min(labels)}-{max(labels)} ({len(labels)} total)")
        
        print(f"     ✓ 验证通过")
        
    except Exception as e:
        print(f"  ✗ 约束 {i} 验证失败: {e}")
        exit(1)

# Test 3: Check constraint patterns
print("\n[3/3] 约束模式分析...")
pattern_types = {
    'triangle': 0,  # 2 edges HEAD → 1 edge BODY
    'bridge': 0,    # 2 edges HEAD → 1 edge BODY (cross-type)
    'hub': 0,       # 2 edges HEAD → 1 edge BODY (via hub)
}

for tgd in constraints:
    head_edges = len(tgd['head']['edges'])
    body_edges = len(tgd['body']['edges'])
    name = tgd['name']
    
    if 'cocitation' in name or 'triangle' in name:
        pattern_types['triangle'] += 1
    elif 'bridge' in name:
        pattern_types['bridge'] += 1
    elif 'hub' in name:
        pattern_types['hub'] += 1

print(f"  模式统计:")
for ptype, count in pattern_types.items():
    print(f"    {ptype}: {count} 个约束")

print("\n" + "=" * 70)
print("✓ 所有测试通过!")
print("=" * 70)

print("\n约束使用示例:")
print("```python")
print("from constraints import get_constraints")
print("from heuchase import HeuChase")
print("")
print("# Load constraints")
print("constraints = get_constraints('OGBN-PAPERS100M')")
print("")
print("# Initialize explainer with constraints")
print("explainer = HeuChase(")
print("    model=model,")
print("    Sigma=constraints,  # Use OGBN constraints")
print("    L=2,")
print("    k=10,")
print("    B=5,")
print("    m=6,")
print(")")
print("")
print("# Run explanation")
print("Sigma_star, S_k = explainer._run(H=subgraph, root=target_node)")
print("print(f'Grounded {len(Sigma_star)} constraints, found {len(S_k)} witnesses')")
print("```")
