"""
Train GCN model on ogbn-papers100M dataset using mini-batch training.

This version uses NeighborLoader to handle the large graph that doesn't fit in GPU memory.

Dataset: ogbn-papers100M
- 111M papers (nodes)
- Citation graph (edges)
- 1.5M arXiv papers with labels (172 classes)
- Task: Predict arXiv subject areas
- Split: Time-based (train: <=2017, val: 2018, test: >=2019)

Model: 2-layer GCN with mini-batch training
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import NeighborLoader
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import time
import json
import os
import gc
import argparse
import yaml
from tqdm import tqdm

print("=" * 70)
print("OGBN-PAPERS100M - GCN TRAINING (2-layer, Mini-batch)")
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

def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch using mini-batches"""
    model.train()
    total_loss = 0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass on batch
        out = model(batch.x, batch.edge_index)[:batch.batch_size]
        loss = F.nll_loss(out, batch.y[:batch.batch_size].squeeze(1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.batch_size
        total_samples += batch.batch_size
        
        pbar.set_postfix({'loss': loss.item()})
        
        # Clear cache periodically
        del batch, out, loss
        if total_samples % 10000 == 0:
            clear_gpu_memory()
    
    return total_loss / total_samples

@torch.no_grad()
def evaluate(model, loader, evaluator, device, desc='Evaluating'):
    """Evaluate model using mini-batches"""
    model.eval()
    
    y_true_list = []
    y_pred_list = []
    
    pbar = tqdm(loader, desc=desc)
    for batch in pbar:
        batch = batch.to(device)
        
        # Forward pass on batch
        out = model(batch.x, batch.edge_index)[:batch.batch_size]
        y_pred = out.argmax(dim=-1, keepdim=True)
        
        y_true_list.append(batch.y[:batch.batch_size].cpu())
        y_pred_list.append(y_pred.cpu())
        
        # Clear cache
        del batch, out, y_pred
        clear_gpu_memory()
    
    y_true = torch.cat(y_true_list, dim=0)
    y_pred = torch.cat(y_pred_list, dim=0)
    
    acc = evaluator.eval({
        'y_true': y_true,
        'y_pred': y_pred,
    })['acc']
    
    return acc

# ============================================================================
# Main Training Function
# ============================================================================

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train GCN on ogbn-papers100M with mini-batches')
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
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for training')
    parser.add_argument('--num_neighbors', type=int, nargs='+', default=[15, 10],
                        help='Number of neighbors to sample per layer')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--log_steps', type=int, default=1,
                        help='Log every n epochs')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory to save model')
    
    args = parser.parse_args()
    
    # Get parameters
    epochs = args.epochs
    lr = args.lr
    hidden_dim = args.hidden_dim
    dropout = args.dropout
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    num_neighbors = args.num_neighbors
    num_workers = args.num_workers
    log_steps = args.log_steps
    
    print(f"Training configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Dropout: {dropout}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Batch size: {batch_size}")
    print(f"  Neighbor sampling: {num_neighbors}")
    print(f"  Num workers: {num_workers}")
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
    
    # Create mini-batch loaders (data stays on CPU)
    print("Creating mini-batch data loaders...")
    print(f"  Using NeighborLoader with sampling: {num_neighbors}")
    print(f"  Batch size: {batch_size}")
    
    train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=split_idx['train'],
        num_workers=num_workers,
        persistent_workers=False,  # Don't cache data in workers (saves RAM)
        shuffle=True,
    )
    
    val_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=split_idx['valid'],
        num_workers=num_workers,
        persistent_workers=False,  # Don't cache data in workers (saves RAM)
        shuffle=False,
    )
    
    test_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=split_idx['test'],
        num_workers=num_workers,
        persistent_workers=False,  # Don't cache data in workers (saves RAM)
        shuffle=False,
    )
    
    print("Data loaders created successfully!")
    print()
    
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
    print(f"Model: 2-layer GCN (mini-batch)")
    print(f"Parameters: {num_params:,}")
    print()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Evaluator
    evaluator = Evaluator(name='ogbn-papers100M')
    
    # Training loop
    print("=" * 70)
    print("Starting mini-batch training...")
    print("=" * 70)
    print()
    
    best_val_acc = 0
    best_epoch = 0
    best_model_state = None
    
    training_start = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        epoch_time = time.time() - epoch_start
        
        # Evaluate
        if epoch % log_steps == 0 or epoch == 1:
            print(f"\nEpoch {epoch:03d} evaluation:")
            train_acc = evaluate(model, train_loader, evaluator, device, desc='Train eval')
            valid_acc = evaluate(model, val_loader, evaluator, device, desc='Val eval')
            test_acc = evaluate(model, test_loader, evaluator, device, desc='Test eval')
            
            print(f"Epoch {epoch:03d} | "
                  f"Loss: {train_loss:.4f} | "
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
                model_path = os.path.join(args.save_dir, 'OGBN_Papers100M_gcn2_minibatch_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_acc': train_acc,
                    'valid_acc': valid_acc,
                    'test_acc': test_acc,
                    'train_loss': train_loss,
                    'hidden_dim': hidden_dim,
                    'input_dim': input_dim,
                    'output_dim': output_dim,
                    'batch_size': batch_size,
                    'num_neighbors': num_neighbors,
                }, model_path)
                print(f"  → Saved best model (val_acc: {valid_acc:.4f})")
        else:
            print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | Time: {epoch_time:.2f}s")
        
        # Clear cache periodically
        if epoch % 5 == 0:
            clear_gpu_memory()
    
    training_time = time.time() - training_start
    
    # Final evaluation with best model
    print()
    print("=" * 70)
    print("Training completed!")
    print("=" * 70)
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("\nFinal evaluation with best model:")
        train_acc = evaluate(model, train_loader, evaluator, device, desc='Final train')
        valid_acc = evaluate(model, val_loader, evaluator, device, desc='Final val')
        test_acc = evaluate(model, test_loader, evaluator, device, desc='Final test')
        
        print(f"\nBest epoch: {best_epoch}")
        print(f"Best validation accuracy: {valid_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Training time: {training_time/60:.2f} minutes")
        print(f"Model saved to: {model_path}")
        print("=" * 70)
        
        # Save training results
        results = {
            'dataset': 'ogbn-papers100M',
            'model': 'GCN-2-MiniBatch',
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
            'batch_size': batch_size,
            'num_neighbors': num_neighbors,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        results_file = os.path.join(args.save_dir, 'OGBN_Papers100M_minibatch_training_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {results_file}")
    
    print()
    print("Training script completed!")

if __name__ == '__main__':
    main()
