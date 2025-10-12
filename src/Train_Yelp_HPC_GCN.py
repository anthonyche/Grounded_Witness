"""
Train GCN models (1, 2, 3 layers) on Yelp dataset with memory optimization.

Memory optimization strategies:
1. Small hidden dimension (32)
2. Aggressive dropout
3. Explicit memory clearing between operations
4. Lower learning rate for stability
5. Efficient multi-label training
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Yelp
from torch_geometric.transforms import NormalizeFeatures, ToUndirected
from torch_geometric.nn import GCNConv
import time
import json
import os
import gc

print("=" * 70)
print("YELP DATASET - GCN TRAINING (1/2/3 layers)")
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
# Model Definitions (Memory-Optimized)
# ============================================================================

class GCN_1(torch.nn.Module):
    """1-layer GCN for node classification"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(input_dim, output_dim)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x

class GCN_2(torch.nn.Module):
    """2-layer GCN for node classification"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GCN_3(torch.nn.Module):
    """3-layer GCN for node classification"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x

# ============================================================================
# Helper Functions
# ============================================================================

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def compute_multilabel_metrics(pred, target, mask):
    """Compute metrics for multi-label classification"""
    pred_binary = (torch.sigmoid(pred[mask]) > 0.5).float()
    target_binary = target[mask].float()
    
    # Exact match accuracy
    exact_match = (pred_binary == target_binary).all(dim=1).float().mean().item()
    
    # Hamming accuracy (per-label accuracy)
    hamming = (pred_binary == target_binary).float().mean().item()
    
    return exact_match, hamming

def train_node_classification(model, data, config, device, model_name):
    """Train a GNN model for node classification on Yelp"""
    
    print("=" * 70)
    print(f"Training {model_name} on Yelp (Multi-label: True)")
    print("=" * 70)
    
    epochs = config.get('epochs', 200)
    lr = config.get('lr', 0.005)
    weight_decay = config.get('weight_decay', 5e-4)
    patience = config.get('patience', 50)
    
    print(f"Epochs: {epochs}, LR: {lr}, Weight Decay: {weight_decay}")
    print(f"Device: {device}")
    print(f"Patience: {patience}")
    print("=" * 70)
    print()
    
    # Move model to device
    model = model.to(device)
    
    # Move data to device (with error handling)
    try:
        data = data.to(device)
        print("Data successfully moved to GPU")
    except RuntimeError as e:
        print(f"Warning: Could not move data to GPU: {e}")
        print("Falling back to CPU training...")
        device = torch.device('cpu')
        model = model.to(device)
        data = data.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()  # For multi-label
    
    best_val_metric = 0
    best_epoch = 0
    patience_counter = 0
    
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    
    start_time = time.time()
    
    try:
        for epoch in range(1, epochs + 1):
            # Training
            model.train()
            optimizer.zero_grad()
            
            out = model(data.x, data.edge_index)
            loss = criterion(out[train_mask], data.y[train_mask].float())
            
            loss.backward()
            optimizer.step()
            
            # Clear intermediate tensors
            del out, loss
            if epoch % 10 == 0:
                clear_gpu_memory()
            
            # Evaluation
            if epoch % 10 == 0 or epoch == 1:
                model.eval()
                with torch.no_grad():
                    out = model(data.x, data.edge_index)
                    
                    train_exact, train_hamming = compute_multilabel_metrics(out, data.y, train_mask)
                    val_exact, val_hamming = compute_multilabel_metrics(out, data.y, val_mask)
                    test_exact, test_hamming = compute_multilabel_metrics(out, data.y, test_mask)
                    
                    # Calculate current loss for logging
                    current_loss = criterion(out[train_mask], data.y[train_mask].float()).item()
                    
                    del out
                
                print(f"Epoch {epoch:03d} | Loss: {current_loss:.4f} | "
                      f"Train: {train_exact:.4f}/{train_hamming:.4f} | "
                      f"Val: {val_exact:.4f}/{val_hamming:.4f} | "
                      f"Test: {test_exact:.4f}/{test_hamming:.4f}")
                
                # Save best model based on validation hamming accuracy
                if val_hamming > best_val_metric:
                    best_val_metric = val_hamming
                    best_epoch = epoch
                    patience_counter = 0
                    
                    # Save model
                    model_path = f"models/Yelp_{model_name}_model.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_metric': val_hamming,
                        'test_exact': test_exact,
                        'test_hamming': test_hamming,
                    }, model_path)
                    print(f"  → Saved best model (val_hamming: {val_hamming:.4f})")
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} checks)")
                    break
        
        training_time = time.time() - start_time
        
        # Load best model for final evaluation
        checkpoint = torch.load(f"models/Yelp_{model_name}_model.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            test_exact, test_hamming = compute_multilabel_metrics(out, data.y, test_mask)
            del out
        
        print()
        print("=" * 70)
        print(f"Training completed in {training_time/60:.2f} minutes")
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation hamming: {best_val_metric:.4f}")
        print(f"Final test exact match: {test_exact:.4f}")
        print(f"Final test hamming: {test_hamming:.4f}")
        print("=" * 70)
        print()
        
        return {
            'model_name': model_name,
            'training_time': training_time,
            'best_epoch': best_epoch,
            'best_val_metric': best_val_metric,
            'test_exact': test_exact,
            'test_hamming': test_hamming,
            'success': True,
            'error': None
        }
    
    except RuntimeError as e:
        if "out of memory" in str(e):
            print()
            print("=" * 70)
            print(f"ERROR: GPU Out of Memory for {model_name}")
            print(f"Error: {str(e)}")
            print("=" * 70)
            print()
            
            return {
                'model_name': model_name,
                'training_time': 0,
                'best_epoch': 0,
                'best_val_metric': 0,
                'test_exact': 0,
                'test_hamming': 0,
                'success': False,
                'error': 'OOM'
            }
        else:
            raise e
    except Exception as e:
        print()
        print("=" * 70)
        print(f"ERROR: Unexpected error for {model_name}")
        print(f"Error: {str(e)}")
        print("=" * 70)
        print()
        
        return {
            'model_name': model_name,
            'training_time': 0,
            'best_epoch': 0,
            'best_val_metric': 0,
            'test_exact': 0,
            'test_hamming': 0,
            'success': False,
            'error': str(e)
        }

# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    # Configuration with memory-friendly settings
    config = {
        'data_root': './datasets',
        'epochs': 200,
        'lr': 0.005,
        'weight_decay': 5e-4,
        'hidden_dim': 32,  # Small hidden dim for memory efficiency
        'patience': 50,
    }
    
    # Load dataset
    print("Loading Yelp dataset...")
    start_time = time.time()
    
    dataset = Yelp(root=config['data_root'], transform=NormalizeFeatures())
    data = dataset[0]
    
    # Make undirected (keep original, no self-loops)
    data.edge_index = ToUndirected()(data).edge_index
    
    load_time = time.time() - start_time
    print(f"Dataset loaded in {load_time:.2f} seconds")
    print()
    
    # Print dataset statistics
    print("Dataset Statistics:")
    print(f"  Nodes: {data.num_nodes:,}")
    print(f"  Edges: {data.edge_index.size(1):,}")
    print(f"  Features: {data.x.size(1)}")
    print(f"  Classes: {data.y.size(1)}")
    print(f"  Multi-label: True")
    print(f"  Train nodes: {data.train_mask.sum().item():,}")
    print(f"  Val nodes: {data.val_mask.sum().item():,}")
    print(f"  Test nodes: {data.test_mask.sum().item():,}")
    print()
    
    input_dim = data.x.size(1)
    output_dim = data.y.size(1)
    hidden_dim = config['hidden_dim']
    
    # Models to train: GCN with 1, 2, and 3 layers
    models_to_train = [
        ('gcn1', 'GCN 1-layer', GCN_1, {'dropout': 0.5}),
        ('gcn2', 'GCN 2-layer', GCN_2, {'dropout': 0.5}),
        ('gcn3', 'GCN 3-layer', GCN_3, {'dropout': 0.5}),
    ]
    
    results = []
    successful_models = []
    failed_models = []
    
    total_start_time = time.time()
    
    for i, (model_name, description, model_class, model_kwargs) in enumerate(models_to_train, 1):
        print()
        print("#" * 70)
        print(f"# MODEL {i}/{len(models_to_train)}: {description}")
        print("#" * 70)
        print()
        
        # Clear GPU memory before training each model
        clear_gpu_memory()
        
        if torch.cuda.is_available():
            print(f"GPU Memory before training:")
            print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
            print()
        
        try:
            # Create model
            model = model_class(input_dim, hidden_dim, output_dim, **model_kwargs)
            
            num_params = sum(p.numel() for p in model.parameters())
            print(f"Model parameters: {num_params:,}")
            print()
            
            # Train model
            result = train_node_classification(model, data, config, device, model_name)
            results.append(result)
            
            if result['success']:
                successful_models.append(model_name)
                print(f"✓ {model_name.upper()} training completed successfully!")
            else:
                failed_models.append((model_name, result['error']))
                print(f"✗ {model_name.upper()} training failed: {result['error']}")
            
            # Clean up
            del model
            clear_gpu_memory()
            
            if torch.cuda.is_available():
                print(f"\nGPU Memory after cleanup:")
                print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
                print()
            
        except Exception as e:
            print(f"✗ Unexpected error training {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            results.append({
                'model_name': model_name,
                'success': False,
                'error': str(e)
            })
            failed_models.append((model_name, str(e)))
            clear_gpu_memory()
    
    total_time = time.time() - total_start_time
    
    # Print summary
    print()
    print("=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Successful models: {len(successful_models)}/{len(models_to_train)}")
    print("=" * 70)
    print()
    
    # Print detailed results table
    if any(r['success'] for r in results):
        print(f"{'Model':<15} {'Layers':<10} {'Params':<12} {'Val Metric':<12} {'Test Hamming':<12} {'Time (min)':<12}")
        print("-" * 90)
        
        for result in results:
            if result['success']:
                # Get model info
                model_info = next((m for m in models_to_train if m[0] == result['model_name']), None)
                layers = model_info[1] if model_info else 'N/A'
                
                # Recreate model to count params
                try:
                    temp_model = model_info[2](input_dim, hidden_dim, output_dim, **model_info[3])
                    params = sum(p.numel() for p in temp_model.parameters())
                    del temp_model
                except:
                    params = 0
                
                print(f"{result['model_name'].upper():<15} {layers:<10} {params:<12,} "
                      f"{result['best_val_metric']:<12.4f} {result['test_hamming']:<12.4f} "
                      f"{result['training_time']/60:<12.2f}")
        
        print("=" * 90)
        print()
    
    if failed_models:
        print("Failed Models:")
        for model_name, error in failed_models:
            print(f"  ✗ {model_name.upper()}: {error}")
        print()
    
    # Save results
    results_file = 'models/Yelp_GCN_training_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_time': total_time,
            'config': config,
            'models': results,
            'successful_models': successful_models,
            'failed_models': [(name, err) for name, err in failed_models],
        }, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    print()
    
    if len(successful_models) > 0:
        print(f"✓ Successfully trained {len(successful_models)} model(s)!")
        print("Models saved in: models/")
        for model_name in successful_models:
            print(f"  - models/Yelp_{model_name}_model.pth")
    
    if len(failed_models) > 0:
        print()
        print(f"✗ {len(failed_models)} model(s) failed.")
        print("Possible solutions:")
        print("  1. Further reduce hidden_dim (e.g., to 16)")
        print("  2. Use fewer layers")
        print("  3. Request GPU with more memory")
        print("  4. Use CPU training (slower but no memory limit)")
    
    print()
    print("=" * 70)
    print("Training script completed!")
    print("=" * 70)

if __name__ == '__main__':
    main()
