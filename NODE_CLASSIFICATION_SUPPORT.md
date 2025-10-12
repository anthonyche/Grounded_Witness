# Node Classification Support Documentation

## Overview

All explanation methods (ApxChase, HeuChase, ExhaustChase, GNNExplainer, PGExplainer) now support **node classification tasks** in addition to graph classification.

## Key Architectural Features

### 1. L-hop Subgraph Extraction

For each target node `v_t`, we extract an L-hop neighborhood subgraph `H`:

```python
# In apxchase.py, heuchase.py, exhaustchase.py
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

**Key Points:**
- Uses PyTorch Geometric's `k_hop_subgraph` utility
- Extracts all nodes within L hops of target node
- Relabels nodes to 0-based consecutive IDs
- Preserves mapping back to original graph node IDs

### 2. All Operations Within Subgraph H

Once subgraph H is extracted, ALL explanation operations occur within H:

1. **Candidate Generation**: Add edges only from nodes in H
2. **Verification**: Check predictions on subgraph H
3. **Constraint Matching**: Find TGD matches within H
4. **Window Maintenance**: Track candidates derived from H

```python
# Example from apxchase.py
def explain_node(self, data: Data, v_t: int) -> Tuple[Set, List[Data]]:
    H = self._prepare_subgraph(data, v_t)  # Extract L-hop subgraph
    H.task = 'node'
    H.root = int(v_t)
    ...
    return self._run(H, root=v_t)  # All operations within H
```

### 3. Unified API

Both node and graph-level tasks use the same interface:

```python
# Node classification (Yelp, BAShape, etc.)
explainer = ApxChase(model, L, k, B, Sigma, ...)
Sigma_star, S_k = explainer.explain_node(data, target_node_id)

# Graph classification (MUTAG, etc.)
explainer = ApxChase(model, L, k, B, Sigma, ...)
Sigma_star, S_k = explainer.explain_graph(data)
```

## Yelp Dataset Support

### 1. TGD Constraints (`src/constraints.py`)

Created 7 structural TGD constraints for Yelp based on common network motifs:

| Constraint | Description | Pattern |
|------------|-------------|---------|
| `yelp_triangle_closure` | Triangle formation | A-B-C path → A-C edge |
| `yelp_triangle_complete` | Triangle completion | A-C, B-C → A-B edge |
| `yelp_square_closure` | 4-cycle formation | A-B-C-D path → A-D edge |
| `yelp_star_to_clique` | Hub connectivity | H-A, H-B → A-B edge |
| `yelp_diamond_closure` | Diamond pattern | A-B-D, A-C-D → B-C edge |
| `yelp_path_extension` | Path elongation | A-B-C → C-D edge |
| `yelp_common_neighbor` | Shared neighbors | A-C, B-C → A-D, B-D |

**Format Compatibility:**
- Uses same dict structure as MUTAG constraints
- Compatible with `matcher.py` functions
- Works with `find_head_matches()` and `backchase_repair_cost()`

**Node Type Specification:**
```python
# Yelp nodes don't have discrete types like MUTAG
# Use "any" type that matches all nodes
def _any_node_type(num_classes=100):
    return {"in": list(range(num_classes))}

# Example constraint
TGD_YELP_TRIANGLE_CLOSURE = {
    "name": "yelp_triangle_closure",
    "head": {
        "nodes": {
            "A": _any_node_type(),
            "B": _any_node_type(),
            "C": _any_node_type(),
        },
        "edges": [("A", "B"), ("B", "C")],
        ...
    },
    ...
}
```

### 2. Data Loading (`src/utils.py`)

Yelp dataset loading already implemented:

```python
if data_name == "Yelp":
    dataset = Yelp(root=data_root, transform=NormalizeFeatures())
    data = dataset[0]
    data = ToUndirected()(data)
    config['multi_label'] = len(data.y.shape) > 1
    # Sample target nodes from test set
    test_idx = torch.where(data.test_mask)[0]
    target_nodes = test_idx[:num_target_nodes].tolist()
```

**Dataset Statistics:**
- Nodes: 716,847
- Edges: 13,954,819
- Features: 300-dimensional
- Labels: 100 classes (multi-label)
- Graph: Undirected, no self-loops

### 3. Model Support

All 5 trained models support node classification:

```python
# Model inference for node classification
data = data.to(device)
out = model(data.x, data.edge_index)  # Shape: [num_nodes, num_classes]
pred = out.argmax(dim=-1)  # Node-level predictions
target_pred = pred[target_node]  # Prediction for specific node
```

**Available Models:**
- `Yelp_gcn1_model.pth` (1-layer GCN)
- `Yelp_gcn2_model.pth` (2-layer GCN)
- `Yelp_gcn_model.pth` (3-layer GCN)
- `Yelp_gat_model.pth` (3-layer GAT)
- `Yelp_sage_model.pth` (3-layer GraphSAGE)

## Implementation Details

### Chase Algorithm Workflow

For each target node, the workflow is:

```
1. Extract L-hop subgraph H around target node v_t
   └─ H contains all nodes within L hops
   └─ H.num_nodes << full_graph.num_nodes

2. Set reference prediction
   └─ Get model's prediction on full graph for v_t
   └─ Store as H.y_ref for verification

3. Run chase algorithm WITHIN H
   └─ Generate edge candidates from nodes in H
   └─ Add edges to H (not full graph)
   └─ Verify predictions on modified H
   └─ Match constraints within H
   └─ Maintain window of candidates

4. Return witnesses derived from H
   └─ Witness nodes ⊆ H nodes
   └─ Witness edges ⊆ potential edges in H
```

### Verification Strategy

```python
def _default_verify_witness(model, v_t: Optional[int], Gs: Data) -> bool:
    # For node classification (v_t is not None)
    if hasattr(Gs, 'task') and Gs.task == 'node' and v_t is not None:
        out = model(Gs)
        y_hat = out.argmax(dim=-1)
        y_ref = Gs.y_ref
        return (y_ref[int(v_t)] == y_hat[int(v_t)])
    # For graph classification (v_t is None)
    else:
        out = model(Gs)
        y_hat = out.argmax(dim=-1)
        y_ref = Gs.y_ref
        return (y_ref[0] == y_hat[0])
```

### Constraint Matching in Subgraphs

The matcher operates on the subgraph H:

```python
# Find head matches within H
matches = find_head_matches(H, tgd)

# For each match, compute repair cost within H
for binding in matches:
    feasible, cost, repairs = backchase_repair_cost(H, tgd, binding, Budget)
    if feasible:
        # This constraint can be grounded in H
        grounded.add(tgd['name'])
```

**Key Properties:**
- VF2 subgraph isomorphism works on H's topology
- Node labels come from H.x (features)
- Repair edges are within H's node set
- No operations touch the full graph

## Utility Functions (`src/node_explainer_utils.py`)

Provides helper functions for node-level explanations:

### Core Functions

```python
# Extract L-hop subgraph
subgraph, subset, target_new_id = extract_l_hop_subgraph(
    data, target_node, num_hops=3, relabel_nodes=True
)

# Batch extraction for multiple nodes
subgraphs = batch_extract_l_hop_subgraphs(
    data, target_nodes=[0,1,2,3,4], num_hops=3
)

# Validate subgraph is suitable for explanation
is_valid = validate_subgraph_for_explanation(subgraph)

# Package results
result = create_explanation_result(
    target_node, witness_subgraph, original_subset, metrics
)
```

### Mapping Between Subgraph and Full Graph

```python
# Subgraph node IDs → Original node IDs
original_ids = subgraph.original_node_ids

# Edge list in original IDs
edges_full = edges_to_full_graph_ids(edge_list, subset)

# Get labels for subgraph nodes
labels = get_subgraph_node_labels(data, subset)
```

## Testing

### Test Files

1. **`test_yelp_constraints.py`**
   - Validates Yelp TGD constraints
   - Tests subgraph extraction
   - Tests matcher functions on Yelp data
   - Verifies format compatibility with MUTAG

2. **`test_chase_node_classification.py`**
   - Tests all 3 chase algorithms on Yelp
   - Verifies L-hop subgraph extraction
   - Confirms operations stay within H
   - Computes metrics (fidelity, coverage, conciseness)

### Running Tests

```bash
# Test constraints and subgraph extraction
python test_yelp_constraints.py

# Test chase algorithms on node classification
python test_chase_node_classification.py
```

### Expected Output

```
======================================================================
TEST: ApxChase on Node Classification
======================================================================

Target node: 123456
L (hops): 3
k (window): 6
Budget: 8
Constraints: 7

======================================================================
Running ApxChase.explain_node()...
======================================================================
Start explain_node: v_t=123456, |V(H)|=245, |E(H)|=1834, L=3, k=6, B=8, |Sigma|=7
...
Grounded constraints (Σ*): 5
Witness candidates (S_k): 6

Best witness statistics:
  Nodes: 78
  Edges: 234
  Fidelity-: 0.8234
  Coverage: 0.7143 (5/7)

✓ ApxChase test PASSED
```

## Configuration for Yelp

In `config.yaml`:

```yaml
exp_name: apxchase_yelp

# Hyperparameters
L: 3              # hop neighborhood size
k: 6              # window size
Budget: 8         # repair budget

# Model
model_name: "gcn"
hidden_dim: 32

# Data
data_name: "Yelp"
input_dim: 300
output_dim: 100
target_nodes: [0,1,2,3,...,99]  # Target nodes to explain
```

## Metrics for Node Classification

All metrics work for node-level explanations:

### 1. Fidelity- (Prediction Preservation)

```python
fidelity_minus = compute_fidelity_minus(
    model, original_graph, explanation_subgraph, device
)
# Measures: P(y|G) - P(y|G_s)
# Lower is better (explanation preserves prediction)
```

### 2. Coverage (Constraint Satisfaction)

```python
covered_constraints, coverage_ratio = compute_constraint_coverage(
    subgraph, constraints, Budget
)
# Measures: |covered| / |total|
# Higher is better (more constraints satisfied)
```

### 3. Conciseness (Compactness)

```python
conciseness = 1.0 / witness.edge_index.size(1)
# Smaller explanation is better
```

## Integration with Existing Code

### Run_Experiment.py Integration

The main experiment runner supports both tasks:

```python
# Node classification mode
if config['data_name'] in ['Yelp', 'BAShape']:
    target_nodes = config['target_nodes']
    
    for target_node in target_nodes:
        # Extract L-hop subgraph
        # Run explanation
        # Compute metrics
        # Save results
        
# Graph classification mode
elif config['data_name'] in ['MUTAG']:
    for graph_idx in test_indices:
        # Run explanation on whole graph
        # Compute metrics
        # Save results
```

### Baseline Methods

GNNExplainer and PGExplainer also support node classification:

```python
# GNNExplainer for node
explainer = GNNExplainer(model, epochs=100)
node_feat_mask, edge_mask = explainer.explain_node(
    target_node, data.x, data.edge_index
)

# PGExplainer for node
explainer = PGExplainer(model, ...)
explainer.train(data)  # Train explainer
explanation = explainer.explain_node(target_node, data)
```

## Summary

✅ **All chase algorithms support node classification**
  - ApxChase: Streaming candidate generation within L-hop H
  - HeuChase: Heuristic-guided search within L-hop H
  - ExhaustChase: Exhaustive enforcement within L-hop H

✅ **L-hop subgraph extraction implemented**
  - Uses PyG's `k_hop_subgraph` utility
  - Preserves mapping to original graph
  - All operations confined to subgraph H

✅ **Yelp constraints defined**
  - 7 structural TGD constraints
  - Based on common network motifs
  - Compatible with existing matcher

✅ **Testing infrastructure complete**
  - Constraint validation tests
  - Subgraph extraction tests
  - End-to-end chase algorithm tests

✅ **Metrics compatible**
  - Fidelity-, Coverage, Conciseness
  - All work for node-level explanations

## Next Steps

To run full Yelp experiments:

```bash
# 1. Ensure trained models are available
ls models/Yelp_*.pth

# 2. Run experiments
python src/Run_Experiment.py --exp_name apxchase_yelp

# 3. Results will be saved to
results/yelp_apxchase/
```

The system is now fully ready for node classification experiments on Yelp! 🎉
