# Grounding Graph Explanations with Data Constraints

This repository contains the implementation of our paper on grounding GNN explanations using data constraints and chase-based repair algorithms.

## 📋 Overview

This project implements several methods for explaining GNN predictions with data constraints:
- **ApxIChase**
- **HeuIChase**
- **Exhaustive**: Exhaustive naivechase 
- **Baselines**: GNNExplainer and PGExplainer

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/anthonyche/Grounded_Witness.git
cd Grounded_Witness

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10
- PyTorch 2.3
- PyTorch Geometric 2.6
- See `requirements.txt` for full list

### Basic Usage

```bash
# Run experiments with default config
python src/Run_Experiment_Node.py --config config.yaml

# Run with custom settings
python src/Run_Experiment_Node.py --config config.yaml --input <graph_index> --output <save_dir>
```

## Supported Datasets

| Dataset | Nodes | Edges | Classes | Type |
|---------|-------|-------|---------|------|
| **MUTAG** | 188 graphs | - | 2 | Graph Classification |
| **Cora** | 2,708 | 5,429 | 7 | Node Classification |
| **BAShape** | 2,020,000 | 12,055,704 | 4 | Synthetic |
| **Yelp** | 716,847 | 13,954,819 | 100 | Node Classification |
| **OGBN-Papers100M** | 111M | 1.6B | 172 | Large-scale |



##  Configuration Guide

Edit `config.yaml` to customize experiments:

### Key Parameters

```yaml
# Experiment name (determines which method to run)
exp_name: heuchase_mutag  # Options: heuchase_*, apxchase_*, gnnexplainer_*, pgexplainer_*

# Data Constraint Parameters
L: 2              # Number of GNN hops (neighborhood size)
k: 6              # Window size for candidate generation
Budget: 8         # Repair budget (max edge modifications)
mask_ratio: 0.05  # Edge masking ratio (0.0-1.0)
preserve_connectivity: true  # Keep graph connected during masking

# Objective Weights
alpha: 0.2    # Conciseness weight
beta: 0.2     # Repair penalty weight  
gamma: 0.6    # Coverage weight

# Method-specific
heuchase_m: 20           # Number of candidates for HeuIChase
heuchase_noise_std: 0.2  # Noise for diversity in sampling
```

### Dataset Selection

Uncomment the desired dataset in `config.yaml`:

```yaml
# For MUTAG (graph classification)
data_name: "MUTAG"
model_name: "gcn_graph_3"

# For Cora (node classification)
# data_name: "Cora"
# model_name: "gcn2"
# target_nodes: [61, 1879, 570, ...]  # Nodes to explain

# For BAShape (synthetic)
# data_name: "BAShape"
# num_target_nodes: 100  # Auto-sample target nodes

# For OGBN-Papers100M (large-scale)
# See config_ogbn_papers100m.yaml
```

### Model Configuration

```yaml
model_name: "gcn2"     # GCN with 2 layers
# Options: gcn2, gat_graph_3, sage_graph_3, gcn_yelp_3, etc.

hidden_dim: 16         # Hidden layer dimension
# Use 16 for BAShape, 32 for others
```

## 🔬 Running Experiments

### 1. Train GNN Models

```bash
# Train on small datasets (MUTAG, Cora)
python src/model.py --dataset MUTAG

# Train on BAShape
python train_BAShape.py

# Train on TreeCycle (synthetic)
python train_Treecycle.py
```

Models are saved to `models/`.

### 2. Run Explanation Methods

```bash
# ApxIChase
python src/Run_Experiment_Node.py --config config.yaml
# (set exp_name: apxchase_mutag)

# HeuIChase  
python src/Run_Experiment_Node.py --config config.yaml
# (set exp_name: heuchase_mutag)

# GNNExplainer baseline
python src/Run_Experiment_Node.py --config config.yaml
# (set exp_name: gnnexplainer_mutag)

# PGExplainer baseline
python src/Run_Experiment_Node.py --config config.yaml
# (set exp_name: pgexplainer_mutag)
```

Results are saved to `results/<dataset>/<method>/`.

### 3. Large-Scale Experiments (OGBN-Papers100M)

```bash
# Distributed benchmark
python src/benchmark_ogbn_distributed.py

# TreeCycle distributed benchmark
python benchmark_treecycle_distributed_v2.py
```

See `config_ogbn_papers100m.yaml` for large-scale configurations.


### Scalability Experiments

```bash
# Figure 13: OGBN-Papers100M runtime vs workers
python src/benchmark_ogbn_distributed.py

# Figures 14-16: TreeCycle scalability
python benchmark_treecycle_distributed_v2.py
```

## 📁 Project Structure

```
GroundingGEXP/
├── config.yaml                      # Main configuration file
├── config_ogbn_papers100m.yaml      # Large-scale config
├── requirements.txt                 # Dependencies
├── src/
│   ├── Run_Experiment_Node.py       # Main experiment runner
│   ├── apxchase.py                  # ApxIChase implementation
│   ├── heuchase.py                  # HeuIChase implementation
│   ├── exhaustchase.py              # Exhaustive chase
│   ├── baselines.py                 # GNNExplainer/PGExplainer
│   ├── constraints.py               # Constraint definitions
│   ├── model.py                     # GNN models
│   └── utils.py                     # Utility functions
├── datasets/                        # Auto-downloaded datasets
├── models/                          # Trained GNN models
├── results/                         # Experiment results
├── Plot_Figures_2.py               # Generate paper figures
└── README.md                        # This file
```

## 🎯 Key Configuration Examples

### Example 1: Quick Test on MUTAG

```yaml
exp_name: heuchase_mutag
data_name: "MUTAG"
model_name: "gcn_graph_3"
L: 2
k: 6
Budget: 8
heuchase_m: 20
```

### Example 2: Coverage Experiment

```yaml
# Vary budget to test coverage
Budget: 1    # Low budget
# Budget: 2
# Budget: 4
# Budget: 6
# Budget: 8  # High budget
```

### Example 3: Constraint Size Experiment

```yaml
# Vary number of constraints
L: 1         # 1-hop neighborhood
# L: 2       # 2-hop neighborhood
# L: 3       # 3-hop neighborhood
```

### Example 4: Target Node Selection

```yaml
# Option 1: Auto-sample
num_target_nodes: 100

# Option 2: Manual selection
target_nodes: [61, 1879, 570, 2039, 2668]
```



**Happy Experimenting! 🚀**
