import random
import os
import numpy as np
import yaml
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, TUDataset
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.utils import remove_self_loops
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def dataset_func(config):
    
    data_dir = "./datasets"
    data_name = config['data_name']
    data_size = config.get('data_size')
    num_class = config.get('output_dim')
    num_test = config.get('num_test', 0)
    random_seed = config['random_seed']
    os.makedirs(data_dir, exist_ok=True)
    set_seed(random_seed)

    if data_name == "MUTAG":
        dataset_root = os.path.join(data_dir, "TUDataset")
        transform = T.NormalizeFeatures()
        dataset = TUDataset(
            root=dataset_root,
            name="MUTAG",
            use_node_attr=True,
            transform=transform
        ).shuffle()

        # Update config with dataset-derived dims
        config['input_dim'] = dataset.num_node_features
        config['output_dim'] = dataset.num_classes
        config['data_size'] = len(dataset)

        # Train/Val/Test split via ratios in config (defaults: 0.8/0.1/0.1)
        train_ratio = float(config.get('train_ratio', 0.8))
        val_ratio = float(config.get('val_ratio', 0.1))
        test_ratio = float(config.get('test_ratio', 0.1))
        total = len(dataset)

        # Convert ratios to lengths and fix rounding
        train_len = int(round(total * train_ratio))
        val_len = int(round(total * val_ratio))
        # Ensure the three parts sum to total
        if train_len + val_len > total:
            val_len = max(0, total - train_len)
        test_len = total - train_len - val_len

        # Reproducible split
        g = torch.Generator()
        g.manual_seed(random_seed)
        train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len], generator=g)

        # DataLoaders (test_loader uses batch_size=1 for per-graph inference)
        batch_size = int(config.get('batch_size', 32))
        loaders = {
            "dataset": dataset,
            "train_loader": DataLoader(train_set, batch_size=batch_size, shuffle=True),
            "val_loader": DataLoader(val_set, batch_size=batch_size, shuffle=False),
            "test_loader": DataLoader(test_set, batch_size=1, shuffle=False),
        }
        return loaders

    if data_name == "BAHouse":

        data = torch.load('./datasets/BAHouse/BAHouse.pt')
        num_nodes = data.x.size(0)

        # Create new masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        # Example: 60% train, 20% val, 20% test
        num_train = int(0.6 * num_nodes)
        num_val = num_nodes - num_train - num_test

        # Set the masks
        train_mask[:num_train] = 1
        val_mask[num_train:num_train + num_val] = 1
        test_mask[num_train + num_val:] = 1

        # Assign the new masks to the data object
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        print(data)
        return data
    

    if data_size is None or num_class is None:
        raise ValueError("Planetoid datasets require 'data_size' and 'output_dim' in the config.")

    num_train_per_class = (data_size - num_test)//num_class
    data = Planetoid(root=data_dir, name=data_name, split='random', num_train_per_class=num_train_per_class, num_val=0, num_test=num_test)[0]
    return data


def get_save_path(dataset, apx_name):
    # Get the directory of the current script
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Define base directory for results relative to the script's directory
    base_results_directory = os.path.join(current_directory, "results")
    os.makedirs(base_results_directory, exist_ok=True)

    dataset_path = os.path.join(base_results_directory, dataset)
    os.makedirs(dataset_path, exist_ok=True)

    method_path = os.path.join(dataset_path, apx_name)
    os.makedirs(method_path, exist_ok=True)

    return method_path

# Load all batches later
def load_precomputed(save_dir='precomputed/'):
    precomputed_data = {}
    for fname in sorted(os.listdir(save_dir)):
        if fname.endswith('.pt'):
            batch_data = torch.load(os.path.join(save_dir, fname))
            precomputed_data.update(batch_data)
    return precomputed_data
