"""
解释为什么无向图有环就没有 arborescence
"""
import networkx as nx

print("="*60)
print("1. 无向图 + 环 → 可以有 Spanning Tree")
print("="*60)

# 创建无向图（带环）
G_undirected = nx.Graph()
G_undirected.add_edges_from([
    (0, 1), (1, 2), (2, 3), (3, 0),  # 环: 0-1-2-3-0
    (0, 4), (1, 5)                    # 额外的边
])

print(f"无向图: {G_undirected.number_of_nodes()} nodes, {G_undirected.number_of_edges()} edges")
print(f"有环: {len(nx.cycle_basis(G_undirected))} cycles found")

# 可以找到 Spanning Tree
mst = nx.minimum_spanning_tree(G_undirected)
print(f"✅ Minimum Spanning Tree: {mst.number_of_nodes()} nodes, {mst.number_of_edges()} edges")
print(f"   是树: {nx.is_tree(mst)}")

print("\n" + "="*60)
print("2. 有向图（从无向图转换） → 没有 Arborescence")
print("="*60)

# 把无向图转成有向图（每条边变双向）
G_directed = nx.DiGraph()
for u, v in G_undirected.edges():
    G_directed.add_edge(u, v, weight=1.0)
    G_directed.add_edge(v, u, weight=1.0)  # 双向！

print(f"有向图: {G_directed.number_of_nodes()} nodes, {G_directed.number_of_edges()} edges")
print(f"是 DAG: {nx.is_directed_acyclic_graph(G_directed)}")
print(f"强连通: {nx.is_strongly_connected(G_directed)}")

# 尝试找 arborescence
try:
    arb = nx.maximum_spanning_arborescence(G_directed, attr='weight')
    print(f"✅ Arborescence found: {arb.number_of_nodes()} nodes, {arb.number_of_edges()} edges")
except Exception as e:
    print(f"❌ 找不到 Arborescence!")
    print(f"   错误: {e}")

print("\n" + "="*60)
print("3. 原因分析")
print("="*60)

print("""
关键点：
1. **Arborescence 只适用于有向图**
   - nx.maximum_spanning_arborescence() 输入必须是 DiGraph
   
2. **无向图转有向图会产生"人工环"**
   - 无向边 u-v 转成: u→v + v→u
   - 这形成了 2-cycle: u→v→u
   
3. **Arborescence 要求 DAG（无环）**
   - 任何双向边都违反 DAG 性质
   - 所以找不到 arborescence

图示：
   无向图           有向图（双向边）      问题
   0 - 1    →     0 ←→ 1           这是环！
                  ↑ ⤨ ↑ ⤨
                  3 ←→ 2           到处都是环！

结论：
- ✅ 无向图可以有 Spanning Tree（用 MST）
- ❌ 无向图转成的有向图不能有 Arborescence（因为双向边 = 环）
""")

print("\n" + "="*60)
print("4. HeuChase 的问题")
print("="*60)

print("""
TreeCycleGenerator 生成的图:
  edge_list = []
  edge_list.append([u, v])  # u → v
  edge_list.append([v, u])  # v → u  <-- 这就是双向！
  
heuchase.py 读取这个图:
  G = nx.DiGraph()  # 当成有向图！
  G.add_edge(u, v)  # u → v
  G.add_edge(v, u)  # v → u  <-- 形成环！
  
  Ar = nx.maximum_spanning_arborescence(G)  # 失败！
  
解决方案:
1. 用 nx.Graph() 代替 nx.DiGraph() → 可以用 MST
2. 或者用 ApxChase（不依赖 Edmonds）← 我们的选择 ✅
""")

print("\n" + "="*60)
print("5. 正确的理解")
print("="*60)

print("""
问题不在于"无向图有环就没有 arborescence"

而是:
1. Arborescence 本身就是**有向图的概念**
2. 无向图应该用 **Spanning Tree**，不是 Arborescence
3. heuchase.py 错误地把无向图（双向边）当成有向图处理
4. 导致 Edmonds 算法失败（它期望 DAG）

类比：
- 无向图找树 = 用 MST（最小生成树）✅
- 有向图找树 = 用 Arborescence（最大树形图）✅
- 无向图找 Arborescence = 类型错误！❌
  （就像给数组排序但传入的是字典）
""")
