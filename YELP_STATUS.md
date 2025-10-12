# Yelp Node Classification - Quick Reference

## ✅ Current Status

**All chase algorithms work for node classification on Yelp!**

| Component | Status | Notes |
|-----------|--------|-------|
| ApxChase | ✅ Ready | Uses `explain_node(data, v_t)` |
| HeuChase | ✅ Ready | Uses `explain_node(data, v_t)` |
| ExhaustChase | ✅ Ready | Uses `explain_node(data, v_t)` |
| Yelp Constraints | ✅ Created | 7 TGD constraints in `constraints.py` |
| L-hop Subgraph | ✅ Implemented | Uses `k_hop_subgraph()` in `_prepare_subgraph()` |
| Metrics | ✅ Compatible | Fidelity-, Coverage, Conciseness |
| Models | ✅ Trained | 5 models available (GCN×3, GAT, SAGE) |

## 🔍 Key Verification

### 1. L-hop Subgraph Extraction

**All three algorithms extract L-hop subgraph H:**

```python
# In apxchase.py, heuchase.py, exhaustchase.py lines ~434-445
def _prepare_subgraph(self, data: Data, v_t: int) -> Data:
    """Extract L-hop subgraph around v_t (node task)."""
    node_idx, ei, _, _ = k_hop_subgraph(v_t, self.L, data.edge_index, relabel_nodes=True)
    x = data.x[node_idx] if getattr(data, 'x', None) is not None else None
    out = Data(x=x, edge_index=ei)
    out._nodes_in_full = node_idx.clone()
    out.num_nodes = int(node_idx.numel())
    ...
    return out
```

✅ **Confirmed**: Subgraph H is extracted with L hops around target node

### 2. Operations Within H

**All operations happen within subgraph H:**

```python
# explain_node() workflow
def explain_node(self, data: Data, v_t: int):
    H = self._prepare_subgraph(data, v_t)  # ← Extract L-hop H
    H.task = 'node'
    H.root = int(v_t)
    ...
    return self._run(H, root=v_t)  # ← Run chase within H
```

**Within _run(H, root):**
- ✅ Candidate generation: Adds edges from nodes in H
- ✅ Verification: Checks predictions on H
- ✅ Constraint matching: Finds TGD matches in H
- ✅ Window maintenance: Tracks candidates derived from H
- ✅ Repair cost: Computed within H's edge space

### 3. Yelp Constraints

**7 constraints defined in `src/constraints.py`:**

```python
CONSTRAINTS_YELP = [
    TGD_YELP_TRIANGLE_CLOSURE,      # A-B-C → A-C
    TGD_YELP_TRIANGLE_COMPLETE,     # A-C, B-C → A-B  
    TGD_YELP_SQUARE_CLOSURE,        # A-B-C-D → A-D
    TGD_YELP_STAR_TO_CLIQUE,        # H-A, H-B → A-B
    TGD_YELP_DIAMOND_CLOSURE,       # Diamond → B-C
    TGD_YELP_PATH_EXTENSION,        # A-B-C → C-D
    TGD_YELP_COMMON_NEIGHBOR,       # A-C, B-C → A-D, B-D
]
```

✅ **Format**: Same dict structure as MUTAG (compatible with matcher.py)
✅ **Node types**: Use `{"in": list(range(100))}` (matches any node)

## 🧪 Testing

### Test 1: Constraints and Subgraphs

```bash
python test_yelp_constraints.py
```

**Tests:**
- ✅ Yelp constraints validation
- ✅ L-hop subgraph extraction
- ✅ Matcher functions (find_head_matches, backchase_repair_cost, Gamma)
- ✅ Format compatibility with MUTAG

### Test 2: Chase Algorithms

```bash
python test_chase_node_classification.py
```

**Tests:**
- ✅ ApxChase.explain_node() on Yelp
- ✅ HeuChase.explain_node() on Yelp
- ✅ ExhaustChase.explain_node() on Yelp
- ✅ Metrics computation (fidelity, coverage)
- ✅ Subgraph H is used throughout

### Expected Output

```
======================================================================
TEST: ApxChase on Node Classification
======================================================================

Running ApxChase.explain_node()...
Start explain_node: v_t=123456, |V(H)|=245, |E(H)|=1834, L=3, k=6, B=8, |Sigma|=7

ApxChase Results
----------------------------------------------------------------------
Grounded constraints (Σ*): 5
Witness candidates (S_k): 6

Best witness statistics:
  Nodes: 78
  Edges: 234
  Fidelity-: 0.8234
  Coverage: 0.7143 (5/7)

✓ ApxChase test PASSED
```

## 📊 Usage Example

### Single Node Explanation

```python
from apxchase import ApxChase
from constraints import get_constraints
from torch_geometric.datasets import Yelp

# Load data
dataset = Yelp(root='./datasets')
data = dataset[0]

# Load model
model = ...  # Your trained GNN model

# Get constraints
constraints = get_constraints('YELP')

# Create explainer
explainer = ApxChase(
    model=model,
    L=3,           # 3-hop neighborhood
    k=6,           # window size
    B=8,           # repair budget
    Sigma=constraints,
    verbose=True
)

# Explain a target node
target_node = 12345
Sigma_star, S_k = explainer.explain_node(data, target_node)

# Best witness
best_witness = S_k[0]
print(f"Witness nodes: {best_witness.num_nodes}")
print(f"Witness edges: {best_witness.edge_index.size(1)}")
```

### Batch Explanation

```python
# Explain multiple target nodes
target_nodes = [0, 1, 2, 3, 4]
results = []

for target_node in target_nodes:
    Sigma_star, S_k = explainer.explain_node(data, target_node)
    
    # Compute metrics
    witness = S_k[0]
    fidelity = compute_fidelity_minus(model, data, witness, device)
    covered, coverage = compute_constraint_coverage(witness, constraints, Budget=8)
    
    results.append({
        'target_node': target_node,
        'grounded_constraints': len(Sigma_star),
        'witness_nodes': witness.num_nodes,
        'witness_edges': witness.edge_index.size(1),
        'fidelity_minus': fidelity,
        'coverage': coverage,
    })
```

## 📁 Key Files

| File | Purpose |
|------|---------|
| `src/constraints.py` | Yelp TGD constraints (CONSTRAINTS_YELP) |
| `src/apxchase.py` | ApxChase with explain_node() |
| `src/heuchase.py` | HeuChase with explain_node() |
| `src/exhaustchase.py` | ExhaustChase with explain_node() |
| `src/matcher.py` | Pattern matching (works on subgraph H) |
| `src/node_explainer_utils.py` | Helper functions for node explanations |
| `test_yelp_constraints.py` | Test constraints & subgraphs |
| `test_chase_node_classification.py` | Test chase algorithms |
| `NODE_CLASSIFICATION_SUPPORT.md` | Full documentation |

## 🔧 Configuration

In `config.yaml`:

```yaml
exp_name: apxchase_yelp

L: 3              # L-hop neighborhood size
k: 6              # Window size for candidate tracking
Budget: 8         # Maximum repair cost allowed

model_name: "gcn"
hidden_dim: 32

data_name: "Yelp"
input_dim: 300
output_dim: 100

target_nodes: [0,1,2,3,4,5,6,7,8,9,  # Nodes to explain
               10,11,12,...,99]
```

## ✨ Key Takeaways

1. **L-hop subgraph extraction**: ✅ All algorithms use `_prepare_subgraph(data, v_t)`
2. **Operations within H**: ✅ All candidate generation, verification, matching happens in H
3. **Yelp constraints**: ✅ 7 TGD constraints defined and validated
4. **Metrics**: ✅ Fidelity-, Coverage, Conciseness all work
5. **Testing**: ✅ Comprehensive tests verify functionality

## 🚀 Next Steps

**To run full Yelp experiments:**

```bash
# 1. Verify models exist
ls models/Yelp_*.pth

# 2. Run quick test
python test_chase_node_classification.py

# 3. Run full experiment (for all target nodes)
python src/Run_Experiment.py --exp_name apxchase_yelp --output results/yelp_apxchase
```

**Results will include:**
- Per-node explanations (witnesses)
- Grounded constraints for each node
- Metrics: Fidelity-, Coverage, Conciseness
- Timing statistics

---

## 📝 Summary Checklist

- ✅ ApxChase supports node classification via `explain_node(data, v_t)`
- ✅ HeuChase supports node classification via `explain_node(data, v_t)`
- ✅ ExhaustChase supports node classification via `explain_node(data, v_t)`
- ✅ L-hop subgraph H is extracted for each target node
- ✅ All operations (candidate gen, verification, matching) happen within H
- ✅ Yelp constraints defined (7 TGDs) in `constraints.py`
- ✅ Constraints compatible with matcher.py interface
- ✅ Tests verify end-to-end functionality
- ✅ Documentation complete

**Status: READY FOR YELP EXPERIMENTS** 🎉
