"""
Train GCN model on ogbn-papers100M dataset for scalability testing.

Dataset: ogbn-papers100M
- 111M papers (nodes)
- Citation graph (edges)
- 1.5M arXiv papers with labels (172 classes)
- Task: Predict arXiv subject areas
- Split: Time-based (train: <=2017, val: 2018, test: >=2019)

Model: 2-layer GCN
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import time
import json
import os
import gc
import argparse
import yaml

print("=" * 70)
print("OGBN-PAPERS100M - GCN TRAINING (2-layer)")
print("=" * 70)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("=" * 70)
print()

# ============================================================================
# Model Definition
# ============================================================================

class GCN_2_OGBN(torch.nn.Module):
    """2-layer GCN for ogbn-papers100M node classification"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# ============================================================================
# Helper Functions
# ============================================================================

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def train_epoch(model, data, train_idx, optimizer, device):
    """Train for one epoch"""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data.x, data.edge_index)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

@torch.no_grad()
def evaluate(model, data, split_idx, evaluator, device):
    """Evaluate model on train/val/test splits"""
    model.eval()
    
    out = model(data.x, data.edge_index)
    y_pred = out.argmax(dim=-1, keepdim=True)
    
    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    
    return train_acc, valid_acc, test_acc

# ============================================================================
# Main Training Function
# ============================================================================

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train GCN on ogbn-papers100M')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_root', type=str, default='./datasets',
                        help='Root directory for dataset')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay')
    parser.add_argument('--log_steps', type=int, default=1,
                        help='Log every n epochs')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory to save model')
    
    args = parser.parse_args()
    
    # Load config if exists (optional, command-line args take precedence)
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f) or {}
        print(f"Loaded config from {args.config}")
    
    # Get parameters (command-line args override config)
    epochs = args.epochs
    lr = args.lr
    hidden_dim = args.hidden_dim
    dropout = args.dropout
    weight_decay = args.weight_decay
    log_steps = args.log_steps
    
    print(f"Training configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Dropout: {dropout}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Device: {device}")
    print("=" * 70)
    print()
    
    # Load dataset
    print("Loading ogbn-papers100M dataset...")
    print("WARNING: This is a large dataset (~60GB). Loading may take several minutes.")
    start_time = time.time()
    
    dataset = PygNodePropPredDataset(name='ogbn-papers100M', root=args.data_root)
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    
    load_time = time.time() - start_time
    print(f"Dataset loaded in {load_time/60:.2f} minutes")
    print()
    
    # Print dataset statistics
    print("Dataset Statistics:")
    print(f"  Nodes: {data.num_nodes:,}")
    print(f"  Edges: {data.edge_index.size(1):,}")
    print(f"  Features: {data.x.size(1)}")
    print(f"  Classes: {dataset.num_classes}")
    print(f"  Train nodes: {split_idx['train'].size(0):,}")
    print(f"  Val nodes: {split_idx['valid'].size(0):,}")
    print(f"  Test nodes: {split_idx['test'].size(0):,}")
    print()
    
    # Move data to device
    print("Moving data to device...")
    try:
        data = data.to(device)
        train_idx = split_idx['train'].to(device)
        print(f"Data successfully moved to {device}")
    except RuntimeError as e:
        print(f"Warning: Could not move data to GPU: {e}")
        print("This dataset is very large. Consider using CPU or mini-batch training.")
        raise e
    
    if torch.cuda.is_available():
        print(f"GPU Memory after loading data:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print()
    
    # Create model
    input_dim = data.x.size(1)
    output_dim = dataset.num_classes
    
    model = GCN_2_OGBN(input_dim, hidden_dim, output_dim, dropout=dropout).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: 2-layer GCN")
    print(f"Parameters: {num_params:,}")
    print()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Evaluator
    evaluator = Evaluator(name='ogbn-papers100M')
    
    # Training loop
    print("=" * 70)
    print("Starting training...")
    print("=" * 70)
    print()
    
    best_val_acc = 0
    best_epoch = 0
    best_model_state = None
    
    training_start = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Train
        loss = train_epoch(model, data, train_idx, optimizer, device)
        
        epoch_time = time.time() - epoch_start
        
        # Evaluate
        if epoch % log_steps == 0 or epoch == 1:
            train_acc, valid_acc, test_acc = evaluate(model, data, split_idx, evaluator, device)
            
            print(f"Epoch {epoch:03d} | "
                  f"Loss: {loss:.4f} | "
                  f"Train: {train_acc:.4f} | "
                  f"Val: {valid_acc:.4f} | "
                  f"Test: {test_acc:.4f} | "
                  f"Time: {epoch_time:.2f}s")
            
            # Save best model
            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                best_epoch = epoch
                best_model_state = model.state_dict().copy()
                
                # Save checkpoint
                os.makedirs(args.save_dir, exist_ok=True)
                model_path = os.path.join(args.save_dir, 'OGBN_Papers100M_gcn2_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_acc': train_acc,
                    'valid_acc': valid_acc,
                    'test_acc': test_acc,
                    'loss': loss,
                    'hidden_dim': hidden_dim,
                    'input_dim': input_dim,
                    'output_dim': output_dim,
                }, model_path)
                print(f"  → Saved best model (val_acc: {valid_acc:.4f})")
        
        # Clear cache periodically
        if epoch % 10 == 0:
            clear_gpu_memory()
    
    training_time = time.time() - training_start
    
    # Final evaluation with best model
    print()
    print("=" * 70)
    print("Training completed!")
    print("=" * 70)
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        train_acc, valid_acc, test_acc = evaluate(model, data, split_idx, evaluator, device)
        
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation accuracy: {valid_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Training time: {training_time/60:.2f} minutes")
        print(f"Model saved to: {model_path}")
        print("=" * 70)
        
        # Save training results
        results = {
            'dataset': 'ogbn-papers100M',
            'model': 'GCN-2',
            'hidden_dim': hidden_dim,
            'num_params': num_params,
            'best_epoch': best_epoch,
            'train_acc': float(train_acc),
            'valid_acc': float(valid_acc),
            'test_acc': float(test_acc),
            'training_time': training_time,
            'epochs': epochs,
            'lr': lr,
            'dropout': dropout,
            'weight_decay': weight_decay,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        results_file = os.path.join(args.save_dir, 'OGBN_Papers100M_training_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {results_file}")
    
    print()
    print("Training script completed!")

if __name__ == '__main__':
    main()
