import copy
import sys
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, global_mean_pool
from torch.nn import Sequential as Seq, Linear, ReLU, BatchNorm1d
from utils import *

from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

print(f"[model.py] Running from: {__file__}")

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


class GCN_4(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN_4, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index)

        return F.log_softmax(x, dim=1)


# ============================================================================
# Yelp-specific Models (Multi-label Node Classification)
# These models return raw logits (no log_softmax) for BCEWithLogitsLoss
# ============================================================================

class GCN_Yelp_1(torch.nn.Module):
    """1-layer GCN for Yelp multi-label node classification"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(input_dim, output_dim)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x  # Raw logits for BCEWithLogitsLoss

class GCN_Yelp_2(torch.nn.Module):
    """2-layer GCN for Yelp multi-label node classification"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # Raw logits for BCEWithLogitsLoss

class GCN_Yelp_3(torch.nn.Module):
    """3-layer GCN for Yelp multi-label node classification"""
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
        return x  # Raw logits for BCEWithLogitsLoss

class GAT_Yelp(torch.nn.Module):
    """Memory-optimized GAT for Yelp multi-label node classification"""
    def __init__(self, input_dim, hidden_dim, output_dim, heads=2, dropout=0.6):
        super().__init__()
        # Reduced heads from 8 to 2 to save memory
        self.conv1 = GATConv(input_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hidden_dim, output_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x  # Raw logits for BCEWithLogitsLoss

class SAGE_Yelp(torch.nn.Module):
    """Memory-optimized GraphSAGE for Yelp multi-label node classification"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, output_dim)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x  # Raw logits for BCEWithLogitsLoss


class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GAT, self).__init__()
        heads = 8

        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.6)
        self.conv3 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        
        x, edge_index = x, edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=-1)


class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GIN, self).__init__()

        self.conv1 = GINConv(Seq(Linear(input_dim, hidden_dim),
                                 ReLU(),
                                 Linear(hidden_dim, hidden_dim),
                                 ReLU(),
                                 BatchNorm1d(hidden_dim)), train_eps=True)
        
        self.conv2 = GINConv(Seq(Linear(hidden_dim, hidden_dim),
                                 ReLU(),
                                 Linear(hidden_dim, hidden_dim),
                                 ReLU(),
                                 BatchNorm1d(hidden_dim)), train_eps=True)
        
        self.conv3 = GINConv(Seq(Linear(hidden_dim, hidden_dim),
                                 ReLU(),
                                 Linear(hidden_dim, hidden_dim),
                                 ReLU(),
                                 BatchNorm1d(hidden_dim)), train_eps=True)
        
        self.lin = Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x, edge_index = x, edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        return self.lin(x)




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
        x = F.dropout(x, p=0.5, training=self.training)

        # Second layer with ReLU activation
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Third layer (output layer)
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=-1)  # For NLL loss (single-label classification)

# three layer GCN graph classifier
class GCNGraphClassifier_3(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.lin = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

# two layer GCN graph classifier
class GCNGraphClassifier_2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

#single layer GCN graph classifier
class GCNGraphClassifier_1(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.lin = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x


# three layer GIN graph classifier
class GINGraphClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GINConv(Seq(Linear(input_dim, hidden_dim),
                                 ReLU(),
                                 Linear(hidden_dim, hidden_dim),
                                 ReLU(),
                                 BatchNorm1d(hidden_dim)), train_eps=True)
        self.conv2 = GINConv(Seq(Linear(hidden_dim, hidden_dim),
                                 ReLU(),
                                 Linear(hidden_dim, hidden_dim),
                                 ReLU(),
                                 BatchNorm1d(hidden_dim)), train_eps=True)
        self.conv3 = GINConv(Seq(Linear(hidden_dim, hidden_dim),
                                 ReLU(),
                                 Linear(hidden_dim, hidden_dim),
                                 ReLU(),
                                 BatchNorm1d(hidden_dim)), train_eps=True)
        self.lin = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)


# three layer GraphSAGE graph classifier
# three layer GraphSAGE graph classifier
class GraphSAGEGraphClassifier_3(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.lin = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

# three layer GAT graph classifier
class GATGraphClassifier_3(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        heads = 8
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.6)
        # Use heads=1 and concat=False to keep hidden_dim features before pooling
        self.conv3 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=0.6)
        self.lin = Linear(hidden_dim, output_dim)

        self.dropout_p = 0.5

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.lin(x)
        return x

def get_model(config):
    model_name = config['model_name']
    input_dim = config['input_dim']
    hidden_dim = config['hidden_dim']
    output_dim = config['output_dim']

    if model_name == 'gcn':
        model = GCN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        return model
    elif model_name == 'gat':
        model = GAT(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        return model 
    elif model_name == 'gin':
        model = GIN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        return model 
    elif model_name == 'sage':
        model = GraphSAGE(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        return model 
    elif model_name == 'gcn1':
        model = GCN_1(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        return model 
    elif model_name == 'gcn2':
        model = GCN_2(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        return model 
    elif model_name == 'gcn4':
        model = GCN_4(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        return model 
    elif model_name == 'gcn_graph_3':
        model = GCNGraphClassifier_3(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        return model
    elif model_name == 'gcn_graph_2':
        model = GCNGraphClassifier_2(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        return model
    elif model_name == 'gcn_graph_1':
        model = GCNGraphClassifier_1(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        return model
    elif model_name == 'sage_graph_3':
        model = GraphSAGEGraphClassifier_3(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        return model
    elif model_name == 'gat_graph_3':
        model = GATGraphClassifier_3(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        return model
    # Yelp-specific models (multi-label node classification)
    elif model_name == 'gcn_yelp_1':
        dropout = config.get('dropout', 0.5)
        model = GCN_Yelp_1(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout)
        return model
    elif model_name == 'gcn_yelp_2':
        dropout = config.get('dropout', 0.5)
        model = GCN_Yelp_2(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout)
        return model
    elif model_name == 'gcn_yelp_3':
        dropout = config.get('dropout', 0.5)
        model = GCN_Yelp_3(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout)
        return model
    elif model_name == 'gat_yelp':
        heads = config.get('heads', 2)
        dropout = config.get('dropout', 0.6)
        model = GAT_Yelp(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, heads=heads, dropout=dropout)
        return model
    elif model_name == 'sage_yelp':
        dropout = config.get('dropout', 0.5)
        model = SAGE_Yelp(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout)
        return model
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def train_graph_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = F.cross_entropy(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs
    return total_loss / max(total_graphs, 1)


def evaluate_graph(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_graphs = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = F.cross_entropy(logits, batch.y, reduction='sum')
            preds = logits.argmax(dim=-1)
            total_loss += loss.item()
            correct += (preds == batch.y).sum().item()
            total_graphs += batch.num_graphs
    avg_loss = total_loss / max(total_graphs, 1)
    accuracy = correct / max(total_graphs, 1)
    return avg_loss, accuracy



def main(config_file, output_dir):
    # Load configuration
    config = load_config(config_file)
    data_name = config['data_name']
    model_name = config['model_name']
    random_seed = config['random_seed']
    set_seed(random_seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    dataset_resource = dataset_func(config)
    print(f"[model.py] dataset_resource type: {type(dataset_resource)}")
    if isinstance(dataset_resource, InMemoryDataset):
        print("[model.py] Got an InMemoryDataset -> will NOT call .to(device) on the dataset itself.")

    # Graph-level datasets: either a dict of loaders or a raw InMemoryDataset (e.g., TUDataset like MUTAG)
    if (isinstance(dataset_resource, dict) and 'train_loader' in dataset_resource) or isinstance(dataset_resource, InMemoryDataset):
        if isinstance(dataset_resource, dict):
            train_loader = dataset_resource['train_loader']
            val_loader = dataset_resource['val_loader']
            test_loader = dataset_resource['test_loader']
        else:
            # Build loaders from the raw dataset
            full_ds = dataset_resource
            n = len(full_ds)
            # 70/15/15 split with fixed seed
            gen = torch.Generator().manual_seed(random_seed)
            n_train = int(0.70 * n)
            n_val   = int(0.15 * n)
            n_test  = n - n_train - n_val
            train_ds, val_ds, test_ds = torch.utils.data.random_split(full_ds, [n_train, n_val, n_test], generator=gen)
            batch_size = int(config.get('batch_size', 128))
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
            test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

        model = get_model(config).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 0.01),
            weight_decay=config.get('weight_decay', 5e-4),
        )
        num_epochs = config.get('num_epochs', 200)

        best_state = None
        best_val_acc = -float('inf')
        best_metrics = {}

        for epoch in range(1, num_epochs + 1):
            train_loss = train_graph_epoch(model, train_loader, optimizer, device)
            val_loss, val_acc = evaluate_graph(model, val_loader, device)
            test_loss, test_acc = evaluate_graph(model, test_loader, device)

            print(f'Epoch {epoch:03d} | '
                  f'Train Loss: {train_loss:.4f} | '
                  f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | '
                  f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                best_metrics = {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'test_loss': test_loss,
                    'test_acc': test_acc,
                }

        if best_state is None:
            best_state = copy.deepcopy(model.state_dict())

        model_path = f'models/{data_name}_{model_name}_model.pth'
        torch.save(best_state, model_path)

        print('Training complete.')
        if best_metrics:
            print(f'Best validation accuracy: {best_metrics["val_acc"]:.4f}')
            print(f'Corresponding test accuracy: {best_metrics["test_acc"]:.4f}')
        print(f'Model saved to: {model_path}')
        return

    # Node-level datasets (Planetoid, Yelp, etc.)
    # Check if dataset_resource is a dict (Yelp returns dict) or Data object (Planetoid returns Data)
    if isinstance(dataset_resource, dict):
        data = dataset_resource['data']
        is_multi_label = dataset_resource.get('multi_label', False)
    else:
        data = dataset_resource
        is_multi_label = False
    
    # Keep data on CPU; batches/tensors are moved to device inside the loops.

    # If this is a node-level task (single giant graph with masks), but the
    # config accidentally specifies a *graph-classification* model (e.g.,
    # `gcn_graph_3`), map it to its node-level counterpart so the forward
    # signature matches (x, edge_index).
    graph2node = {
        'gcn_graph_1': 'gcn',
        'gcn_graph_2': 'gcn',
        'gcn_graph_3': 'gcn',
        'gat_graph_3': 'gat',
        'sage_graph_3': 'sage',
    }
    orig_name = config.get('model_name')
    if orig_name in graph2node:
        remapped = graph2node[orig_name]
        print(f"[model.py] Detected node-level task; remapping model '{orig_name}' -> '{remapped}'")
        # Use a shallow copy so we don't mutate the original config reference
        config = dict(config)
        config['model_name'] = remapped
    model = get_model(config).to(device)
    best_loss = float('inf')
    best_model_state = None
    best_val_metric = 0.0

    num_epochs = config.get('num_epochs', 200)
    learning_rate = config.get('learning_rate', 0.01)
    weight_decay = config.get('weight_decay', 5e-4)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print(f"\nTraining {model_name} on {data_name}")
    print(f"Multi-label: {is_multi_label}, Epochs: {num_epochs}")
    print("-" * 60)
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        if is_multi_label:
            # Multi-label classification (e.g., Yelp)
            # Remove log_softmax by taking exp (since model outputs log_softmax)
            out = torch.exp(out)  # Convert log probabilities back to probabilities
            loss = F.binary_cross_entropy(out[data.train_mask], data.y[data.train_mask].float())
        elif model_name == 'gin':
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        else:
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        optimizer.step()
        
        # Evaluation
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                
                if is_multi_label:
                    out = torch.exp(out)
                    # For multi-label, use threshold 0.5
                    pred = (out > 0.5).float()
                    
                    # Calculate accuracy (exact match ratio)
                    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).all(dim=1).float().mean().item()
                    val_acc = (pred[data.val_mask] == data.y[data.val_mask]).all(dim=1).float().mean().item()
                    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).all(dim=1).float().mean().item()
                else:
                    pred = out.argmax(dim=-1)
                    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
                    val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
                    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
                
                print(f'Epoch {epoch:03d} | Loss: {loss.item():.4f} | '
                      f'Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}')
                
                if val_acc > best_val_metric:
                    best_val_metric = val_acc
                    best_model_state = copy.deepcopy(model.state_dict())
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            if best_model_state is None:
                best_model_state = copy.deepcopy(model.state_dict())

    if best_model_state is None:
        best_model_state = copy.deepcopy(model.state_dict())

    model_path = 'models/{}_{}_model.pth'.format(data_name, model_name)
    torch.save(best_model_state, model_path)

    print('\n' + '='*60)
    print('Training complete!')
    print(f'Seed: {config["random_seed"]}')
    print(f'Dataset: {config["data_name"]}')
    print(f'Model: {config["model_name"]}')
    print(f'Best validation metric: {best_val_metric:.4f}')
    print(f'Model saved to: {model_path}')
    print('='*60)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python src/model.py <config_file> <output_dir>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    output_dir = sys.argv[2]
    main(config_file, output_dir)
