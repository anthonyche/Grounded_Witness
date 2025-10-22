import copy
import sys
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, global_mean_pool
from torch.nn import Sequential as Seq, Linear, ReLU, BatchNorm1d
from utils import *

from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)


class GCN_1(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN_1, self).__init__()
        self.conv1 = GCNConv(input_dim, output_dim)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)

        return F.log_softmax(x, dim=1)


class GCN_2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN_2, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GAT, self).__init__()
        heads = 8

        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.6)
        self.conv3 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=True, dropout=0.6)

    def forward(self, x, edge_index):
        
        x, edge_index = x, edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=-1)
    

class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphSAGE, self).__init__()
        
        # Define the layers
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # First layer with ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Second layer with ReLU activation
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Third layer (output layer)
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=-1)


def get_model(model_name, input_dim, hidden_dim, output_dim):
    """Create model based on name"""
    if model_name == 'gcn':
        return GCN(input_dim, hidden_dim, output_dim)
    elif model_name == 'gcn1':
        return GCN_1(input_dim, hidden_dim, output_dim)
    elif model_name == 'gcn2':
        return GCN_2(input_dim, hidden_dim, output_dim)
    elif model_name == 'gat':
        return GAT(input_dim, hidden_dim, output_dim)
    elif model_name == 'sage':
        return GraphSAGE(input_dim, hidden_dim, output_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_node_classification(model, data, config, device, is_multi_label=False):
    """Train a GNN model for node classification on Yelp"""
    model = model.to(device)
    data = data.to(device)
    
    model_name = config['model_name']
    num_epochs = config.get('num_epochs', 200)
    learning_rate = config.get('learning_rate', 0.01)
    weight_decay = config.get('weight_decay', 5e-4)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    best_val_metric = 0.0
    best_state = None
    best_test_acc = 0.0
    patience = config.get('patience', 50)
    patience_counter = 0
    
    print(f"\n{'='*70}")
    print(f"Training {model_name} on Yelp (Multi-label: {is_multi_label})")
    print(f"{'='*70}")
    print(f"Epochs: {num_epochs}, LR: {learning_rate}, Weight Decay: {weight_decay}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        if is_multi_label:
            # Multi-label: convert log_softmax output to probabilities
            out = torch.exp(out)
            loss = F.binary_cross_entropy(out[data.train_mask], data.y[data.train_mask].float())
        else:
            # Single-label: use NLL loss with log_softmax output
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        optimizer.step()
        
        # Evaluation
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                
                if is_multi_label:
                    # Multi-label classification
                    out = torch.exp(out)
                    pred = (out > 0.5).float()
                    
                    # Exact match accuracy
                    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).all(dim=1).float().mean().item()
                    val_acc = (pred[data.val_mask] == data.y[data.val_mask]).all(dim=1).float().mean().item()
                    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).all(dim=1).float().mean().item()
                    
                    # Also compute hamming accuracy (average per-label accuracy)
                    train_hamming = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
                    val_hamming = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
                    test_hamming = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
                    
                    print(f'Epoch {epoch:03d} | Loss: {loss.item():.4f} | '
                          f'Train: {train_acc:.4f}/{train_hamming:.4f} | '
                          f'Val: {val_acc:.4f}/{val_hamming:.4f} | '
                          f'Test: {test_acc:.4f}/{test_hamming:.4f}')
                    
                    # Use hamming accuracy for early stopping
                    current_metric = val_hamming
                else:
                    # Single-label classification
                    pred = out.argmax(dim=-1)
                    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
                    val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
                    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
                    
                    print(f'Epoch {epoch:03d} | Loss: {loss.item():.4f} | '
                          f'Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}')
                    
                    current_metric = val_acc
                
                # Save best model
                if current_metric > best_val_metric:
                    best_val_metric = current_metric
                    best_test_acc = test_acc
                    best_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break
    
    print(f"\n{'='*70}")
    print(f"Training completed!")
    print(f"Best Val Metric: {best_val_metric:.4f} | Corresponding Test Acc: {best_test_acc:.4f}")
    print(f"{'='*70}\n")
    
    return best_state, best_val_metric, best_test_acc


def main():
    """Main training function"""
    import os
    import time
    
    # Configuration
    config = {
        'data_name': 'Yelp',
        'data_root': './datasets',
        'random_seed': 42,
        'hidden_dim': 32,
        'num_epochs': 200,
        'learning_rate': 0.01,
        'weight_decay': 5e-4,
        'num_target_nodes': 50,
        'patience': 50,
    }
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"YELP DATASET - GNN MODEL TRAINING")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*70}\n")
    
    # Set random seed
    set_seed(config['random_seed'])
    
    # Load Yelp dataset
    print("Loading Yelp dataset...")
    start_time = time.time()
    data_resource = dataset_func(config)
    data = data_resource['data']
    is_multi_label = data_resource.get('multi_label', False)
    
    input_dim = config['input_dim']
    output_dim = config['output_dim']
    
    load_time = time.time() - start_time
    print(f"Dataset loaded in {load_time:.2f} seconds")
    print(f"\nDataset Statistics:")
    print(f"  Nodes: {data.num_nodes:,}")
    print(f"  Edges: {data.edge_index.size(1):,}")
    print(f"  Features: {input_dim}")
    print(f"  Classes: {output_dim}")
    print(f"  Multi-label: {is_multi_label}")
    print(f"  Train nodes: {data.train_mask.sum().item():,}")
    print(f"  Val nodes: {data.val_mask.sum().item():,}")
    print(f"  Test nodes: {data.test_mask.sum().item():,}")
    
    # Define models to train
    models_to_train = [
        {'model_name': 'gcn1', 'layers': 1, 'description': 'GCN 1-layer'},
        {'model_name': 'gcn2', 'layers': 2, 'description': 'GCN 2-layer'},
        {'model_name': 'gcn', 'layers': 3, 'description': 'GCN 3-layer'},
        {'model_name': 'gat', 'layers': 3, 'description': 'GAT 3-layer'},
        {'model_name': 'sage', 'layers': 3, 'description': 'GraphSAGE 3-layer'},
    ]
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Train each model
    results = []
    total_start = time.time()
    
    for i, model_config in enumerate(models_to_train, 1):
        print(f"\n{'#'*70}")
        print(f"# MODEL {i}/{len(models_to_train)}: {model_config['description']}")
        print(f"{'#'*70}")
        
        # Update config
        config['model_name'] = model_config['model_name']
        
        # Create model
        model = get_model(
            model_config['model_name'],
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            output_dim=output_dim
        )
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")
        
        # Train
        train_start = time.time()
        best_state, best_val_metric, best_test_acc = train_node_classification(
            model, data, config, device, is_multi_label
        )
        train_time = time.time() - train_start
        
        # Save model
        model_path = f'models/Yelp_{model_config["model_name"]}_model.pth'
        torch.save(best_state, model_path)
        print(f"Model saved to: {model_path}")
        print(f"Training time: {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
        
        results.append({
            'model': model_config['model_name'],
            'description': model_config['description'],
            'layers': model_config['layers'],
            'params': num_params,
            'val_metric': best_val_metric,
            'test_acc': best_test_acc,
            'train_time': train_time,
            'path': model_path
        })
    
    total_time = time.time() - total_start
    
    # Print final summary
    print(f"\n\n{'='*70}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"{'='*70}\n")
    
    print(f"{'Model':<20} {'Layers':<8} {'Params':<12} {'Val Metric':<12} {'Test Acc':<12} {'Time (min)':<12}")
    print(f"{'-'*70}")
    for r in results:
        print(f"{r['description']:<20} {r['layers']:<8} {r['params']:<12,} "
              f"{r['val_metric']:<12.4f} {r['test_acc']:<12.4f} {r['train_time']/60:<12.2f}")
    
    print(f"{'='*70}\n")
    print("All models trained successfully!")
    print("Models saved in: models/")
    
    # Save results to file
    import json
    results_file = 'models/Yelp_training_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")


if __name__ == '__main__':
    main()
