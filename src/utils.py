import random
import os
import numpy as np
import yaml
import torch
import math
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, TUDataset, Yelp, DBLP
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.utils import remove_self_loops
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torch_geometric.transforms import ToUndirected, AddSelfLoops, NormalizeFeatures
from torch_geometric.data import Data

try:
    from constraints import DBLP_NODE_TYPE, DBLP_REL_TYPE
except ImportError:
    from .constraints import DBLP_NODE_TYPE, DBLP_REL_TYPE


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


def _project_feature_block(x: torch.Tensor | None, out_dim: int, seed: int, seed_offset: int, num_nodes: int) -> torch.Tensor:
    if out_dim <= 0:
        return torch.zeros((num_nodes, 0), dtype=torch.float)
    if x is None or not isinstance(x, torch.Tensor) or x.numel() == 0:
        return torch.zeros((num_nodes, out_dim), dtype=torch.float)
    x = x.detach().cpu().to(torch.float)
    gen = torch.Generator(device='cpu')
    gen.manual_seed(int(seed + seed_offset))
    scale = 1.0 / math.sqrt(max(1, x.size(1)))
    proj = torch.randn((x.size(1), out_dim), generator=gen, dtype=torch.float) * scale
    return x @ proj


def _build_dblp_homogeneous_view(config: dict, data_dir: str, random_seed: int) -> dict:
    dataset_root = os.path.join(data_dir, "DBLP")
    dataset = DBLP(root=dataset_root)
    hetero = dataset[0]

    node_types = ["author", "paper", "term", "conference"]
    relation_specs = [
        ("author", "paper", DBLP_REL_TYPE["author_paper"]),
        ("paper", "term", DBLP_REL_TYPE["paper_term"]),
        ("paper", "conference", DBLP_REL_TYPE["paper_conference"]),
    ]
    projection_dim = int(config.get("dblp_projection_dim", 128))
    num_type_dims = len(node_types)

    offsets = {}
    cursor = 0
    for node_type in node_types:
        count = int(hetero[node_type].num_nodes)
        offsets[node_type] = cursor
        cursor += count

    feature_parts = []
    label_parts = []
    train_parts = []
    val_parts = []
    test_parts = []
    type_parts = []

    for type_idx, node_type in enumerate(node_types):
        store = hetero[node_type]
        num_nodes = int(store.num_nodes)
        projected = _project_feature_block(
            getattr(store, "x", None),
            projection_dim,
            random_seed,
            type_idx * 9973,
            num_nodes,
        )
        one_hot = torch.zeros((num_nodes, num_type_dims), dtype=torch.float)
        one_hot[:, type_idx] = 1.0
        feature_parts.append(torch.cat([projected, one_hot], dim=1))

        type_parts.append(torch.full((num_nodes,), int(DBLP_NODE_TYPE[node_type]), dtype=torch.long))
        if node_type == "author":
            label_parts.append(store.y.detach().cpu().to(torch.long))
            train_parts.append(store.train_mask.detach().cpu().to(torch.bool))
            val_parts.append(store.val_mask.detach().cpu().to(torch.bool))
            test_parts.append(store.test_mask.detach().cpu().to(torch.bool))
        else:
            label_parts.append(torch.full((num_nodes,), -1, dtype=torch.long))
            train_parts.append(torch.zeros(num_nodes, dtype=torch.bool))
            val_parts.append(torch.zeros(num_nodes, dtype=torch.bool))
            test_parts.append(torch.zeros(num_nodes, dtype=torch.bool))

    edge_blocks = []
    rel_blocks = []
    for src_type, dst_type, rel_id in relation_specs:
        edge_index = hetero[(src_type, "to", dst_type)].edge_index.detach().cpu().to(torch.long)
        src = edge_index[0] + int(offsets[src_type])
        dst = edge_index[1] + int(offsets[dst_type])
        forward = torch.stack([src, dst], dim=0)
        reverse = torch.stack([dst, src], dim=0)
        edge_blocks.extend([forward, reverse])
        rel_blocks.extend([
            torch.full((forward.size(1),), int(rel_id), dtype=torch.long),
            torch.full((reverse.size(1),), int(rel_id), dtype=torch.long),
        ])

    data = Data(
        x=torch.cat(feature_parts, dim=0),
        edge_index=torch.cat(edge_blocks, dim=1),
        y=torch.cat(label_parts, dim=0),
    )
    data.num_nodes = int(data.x.size(0))
    data.train_mask = torch.cat(train_parts, dim=0)
    data.val_mask = torch.cat(val_parts, dim=0)
    data.test_mask = torch.cat(test_parts, dim=0)
    data.y_type = torch.cat(type_parts, dim=0)
    data.native_node_type = data.y_type.clone()
    data.node_labels = data.y_type.clone()
    data.edge_rel_type = torch.cat(rel_blocks, dim=0)
    data.node_type_names = node_types
    data.target_node_type = "author"
    data.target_node_type_id = int(DBLP_NODE_TYPE["author"])
    data.original_num_nodes_by_type = {node_type: int(hetero[node_type].num_nodes) for node_type in node_types}
    data.node_offsets = {node_type: int(offset) for node_type, offset in offsets.items()}
    data.num_classes = int(hetero["author"].y.max().item()) + 1

    author_test_indices = torch.where(data.test_mask)[0].detach().cpu().tolist()
    target_nodes = config.get("target_nodes", None)
    if target_nodes is None:
        num_samples = int(config.get("num_target_nodes", config.get("max_targets", min(64, len(author_test_indices)))))
        if len(author_test_indices) > num_samples:
            gen = torch.Generator()
            gen.manual_seed(random_seed)
            perm = torch.randperm(len(author_test_indices), generator=gen)
            target_nodes = [int(author_test_indices[idx]) for idx in perm[:num_samples].tolist()]
        else:
            target_nodes = [int(v) for v in author_test_indices]
    else:
        target_nodes = [int(v) for v in target_nodes]

    config["input_dim"] = int(data.x.size(1))
    config["output_dim"] = int(data.num_classes)
    config["num_nodes"] = int(data.num_nodes)
    config["num_test"] = int(data.test_mask.sum().item())
    config["data_size"] = int(hetero["author"].num_nodes)
    config["constraint_type_source"] = str(config.get("constraint_type_source", "native_type"))

    return {
        "dataset": dataset,
        "hetero_data": hetero,
        "data": data,
        "input_dim": int(data.x.size(1)),
        "output_dim": int(data.num_classes),
        "num_nodes": int(data.num_nodes),
        "target_population_size": int(hetero["author"].num_nodes),
        "multi_label": False,
        "splits": {
            "train_mask": data.train_mask,
            "val_mask": data.val_mask,
            "test_mask": data.test_mask,
        },
        "target_nodes": target_nodes,
    }


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
            "splits": {
                "train_indices": list(getattr(train_set, "indices", [])),
                "val_indices": list(getattr(val_set, "indices", [])),
                "test_indices": list(getattr(test_set, "indices", [])),
            },
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
    
    if data_name == "BAShape":
        # Load BAShape dataset for node classification
        # BAShape is a BA graph with house motifs attached
        # Labels: 0=BA base (99%), 1=house top (0.4%), 2=house middle (0.4%), 3=house bottom (0.2%)
        data_path = os.path.join(data_dir, 'BAShape', 'BAShape.pt')
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"BAShape dataset not found at {data_path}. Please generate it first.")
        
        print(f"[dataset_func] Loading BAShape dataset from {data_path}...")
        data = torch.load(data_path)
        
        print(f"[dataset_func] Original BAShape data: nodes={data.num_nodes}, edges={data.edge_index.size(1)}")
        
        # Ensure undirected graph (BAShape should already be undirected)
        if not hasattr(data, 'is_undirected') or not data.is_undirected():
            data = ToUndirected()(data)
            print(f"[dataset_func] After ToUndirected: edges={data.edge_index.size(1)}")
        
        # Update config with dataset-derived dimensions
        config['input_dim'] = data.x.size(1) if data.x is not None else 1
        config['output_dim'] = int(data.y.max().item()) + 1  # 4 labels: 0,1,2,3
        config['num_nodes'] = data.num_nodes
        config['multi_label'] = False
        
        print(f"[dataset_func] BAShape dimensions: input_dim={config['input_dim']}, output_dim={config['output_dim']}, num_nodes={config['num_nodes']}")
        
        # Create train/val/test masks if not present
        if not hasattr(data, 'train_mask') or data.train_mask is None:
            print(f"[dataset_func] Creating train/val/test splits...")
            num_nodes = data.num_nodes
            
            # Use label-stratified sampling to ensure each split has house nodes
            # House nodes (labels 1,2,3) are rare, so we need to be careful
            house_indices = (data.y > 0).nonzero(as_tuple=True)[0]  # labels 1,2,3
            ba_indices = (data.y == 0).nonzero(as_tuple=True)[0]     # label 0
            
            print(f"[dataset_func] House nodes: {len(house_indices)} ({100*len(house_indices)/num_nodes:.2f}%)")
            print(f"[dataset_func] BA base nodes: {len(ba_indices)} ({100*len(ba_indices)/num_nodes:.2f}%)")
            
            # Split house nodes: 60% train, 20% val, 20% test
            gen = torch.Generator()
            gen.manual_seed(random_seed)
            house_perm = torch.randperm(len(house_indices), generator=gen)
            
            house_train_size = int(0.6 * len(house_indices))
            house_val_size = int(0.2 * len(house_indices))
            
            house_train = house_indices[house_perm[:house_train_size]]
            house_val = house_indices[house_perm[house_train_size:house_train_size+house_val_size]]
            house_test = house_indices[house_perm[house_train_size+house_val_size:]]
            
            # Split BA nodes: 60% train, 20% val, 20% test
            ba_perm = torch.randperm(len(ba_indices), generator=gen)
            
            ba_train_size = int(0.6 * len(ba_indices))
            ba_val_size = int(0.2 * len(ba_indices))
            
            ba_train = ba_indices[ba_perm[:ba_train_size]]
            ba_val = ba_indices[ba_perm[ba_train_size:ba_train_size+ba_val_size]]
            ba_test = ba_indices[ba_perm[ba_train_size+ba_val_size:]]
            
            # Combine splits
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            train_mask[torch.cat([house_train, ba_train])] = True
            val_mask[torch.cat([house_val, ba_val])] = True
            test_mask[torch.cat([house_test, ba_test])] = True
            
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask
            
            print(f"[dataset_func] Splits: train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")
        
        # Get target nodes for explanation (sample from test set, prioritize house nodes)
        target_nodes = config.get('target_nodes', None)
        if target_nodes is None:
            # Sample house nodes from test set for explanation
            num_samples = config.get('num_target_nodes', 50)
            test_house_indices = torch.logical_and(data.test_mask, data.y > 0).nonzero(as_tuple=True)[0]
            
            if len(test_house_indices) > num_samples:
                gen = torch.Generator()
                gen.manual_seed(random_seed)
                perm = torch.randperm(len(test_house_indices), generator=gen)
                target_nodes = test_house_indices[perm[:num_samples]].tolist()
            else:
                target_nodes = test_house_indices.tolist()
            
            print(f"[dataset_func] Sampled {len(target_nodes)} house nodes from test set as targets")
        else:
            print(f"[dataset_func] Using {len(target_nodes)} pre-defined target nodes")
        
        # Return data resource dictionary (consistent with Yelp format)
        data_resource = {
            "data": data,
            "input_dim": config['input_dim'],
            "output_dim": config['output_dim'],
            "num_nodes": config['num_nodes'],
            "multi_label": False,
            "splits": {
                "train_mask": data.train_mask,
                "val_mask": data.val_mask,
                "test_mask": data.test_mask
            },
            "target_nodes": target_nodes
        }
        
        return data_resource
    
    if data_name == "Yelp":
        # Load Yelp dataset for node classification
        data_root = config.get("data_root", data_dir)
        os.makedirs(data_root, exist_ok=True)
        
        print(f"[dataset_func] Loading Yelp dataset from {data_root}...")
        dataset = Yelp(root=data_root, transform=NormalizeFeatures())
        data = dataset[0]
        
        print(f"[dataset_func] Original Yelp data: nodes={data.num_nodes}, edges={data.edge_index.size(1)}")
        
        # Ensure undirected graph
        data = ToUndirected()(data)
        print(f"[dataset_func] After ToUndirected: edges={data.edge_index.size(1)}")
        
        # Keep original graph structure without adding self-loops
        
        # Update config with dataset-derived dimensions
        config['input_dim'] = data.x.size(1)
        # Yelp is multi-label classification: y is [num_nodes, num_labels]
        if len(data.y.shape) > 1:
            config['output_dim'] = data.y.size(1)  # number of labels
            config['multi_label'] = True
        else:
            config['output_dim'] = int(data.y.max().item()) + 1
            config['multi_label'] = False
        config['num_nodes'] = data.num_nodes
        
        print(f"[dataset_func] Yelp dimensions: input_dim={config['input_dim']}, output_dim={config['output_dim']}, num_nodes={config['num_nodes']}, multi_label={config.get('multi_label', False)}")
        
        # Add KMeans clustering to create pseudo node types for constraint matching
        # Cache the clustering results to avoid re-computation
        cluster_cache_path = os.path.join(data_root, 'yelp_kmeans_clusters.pt')
        n_clusters = config.get('n_node_types', 16)  # Default 16 types
        
        if os.path.exists(cluster_cache_path):
            print(f"[dataset_func] Loading cached KMeans clusters from {cluster_cache_path}...")
            cluster_labels = torch.load(cluster_cache_path)
            data.y_type = cluster_labels
            print(f"[dataset_func] ✅ Loaded {n_clusters} cached node types")
        else:
            print(f"[dataset_func] Running KMeans clustering on node features (first time)...")
            from sklearn.cluster import KMeans
            emb = data.x.cpu().numpy()  # 300-dim features
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
            cluster_labels = kmeans.fit_predict(emb)
            data.y_type = torch.tensor(cluster_labels, dtype=torch.long)
            # Save for future runs
            torch.save(data.y_type, cluster_cache_path)
            print(f"[dataset_func] ✅ Created and cached {n_clusters} node types")
        
        print(f"[dataset_func] Type distribution: {torch.bincount(data.y_type).tolist()}")
        
        # Get target nodes for explanation
        target_nodes = config.get('target_nodes', None)
        if target_nodes is None:
            # Sample from test set
            num_samples = config.get('num_target_nodes', 50)
            test_indices = data.test_mask.nonzero(as_tuple=True)[0]
            if len(test_indices) > num_samples:
                # Random sampling with seed
                gen = torch.Generator()
                gen.manual_seed(random_seed)
                perm = torch.randperm(len(test_indices), generator=gen)
                target_nodes = test_indices[perm[:num_samples]].tolist()
            else:
                target_nodes = test_indices.tolist()
            print(f"[dataset_func] Sampled {len(target_nodes)} target nodes from test set")
        else:
            print(f"[dataset_func] Using {len(target_nodes)} provided target nodes")
        
        # Prepare data resource dict
        data_resource = {
            "dataset": dataset,
            "data": data,
            "input_dim": config['input_dim'],
            "output_dim": config['output_dim'],
            "num_nodes": config['num_nodes'],
            "multi_label": config.get('multi_label', False),
            "splits": {
                "train_mask": data.train_mask,
                "val_mask": data.val_mask,
                "test_mask": data.test_mask
            },
            "target_nodes": target_nodes
        }
        
        return data_resource

    if data_name == "DBLP":
        print(f"[dataset_func] Loading DBLP dataset from {os.path.join(data_dir, 'DBLP')}...")
        data_resource = _build_dblp_homogeneous_view(config, data_dir, random_seed)
        data = data_resource["data"]
        print(
            "[dataset_func] DBLP homogeneous view: "
            f"nodes={data.num_nodes}, edges={data.edge_index.size(1)}, "
            f"input_dim={config['input_dim']}, output_dim={config['output_dim']}, "
            f"author_test={int(data.test_mask.sum().item())}"
        )
        return data_resource
    

    if data_size is None or num_class is None:
        raise ValueError("Planetoid datasets require 'data_size' and 'output_dim' in the config.")

    # Calculate train/val/test split: 60% train, 20% val, 20% test (standard for Planetoid)
    num_val = int(data_size * 0.2)  # 20% for validation
    num_train_per_class = (data_size - num_val - num_test) // num_class
    data = Planetoid(root=data_dir, name=data_name, split='random', num_train_per_class=num_train_per_class, num_val=num_val, num_test=num_test)[0]
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


def compute_fidelity_minus(model, original_graph, explanation_subgraph, device, is_node=False, target_node_id=None):
    """
    计算Fidelity- (Fidelity Minus) 指标
    
    Fidelity- = Pr(M(G)) - Pr(M(G_s))
    其中:
    - G 是原始图 (original_graph)
    - G_s 是解释子图 (explanation_subgraph)
    - M 是GNN模型
    - Pr 是对目标类别的预测概率
    
    对于图分类任务：使用原始图的predicted label作为目标类别
    对于节点分类任务：使用目标节点的predicted label作为目标类别
    
    Args:
        model: 训练好的GNN模型
        original_graph: 原始图 (torch_geometric.data.Data)
        explanation_subgraph: 解释子图/witness (torch_geometric.data.Data)
        device: torch.device
        is_node: bool, 是否为节点分类任务 (default: False)
        target_node_id: int, 目标节点在子图中的ID (仅用于节点分类)
    
    Returns:
        float: raw fidelity^- 值 = Pr(M(G)) - Pr(M(G_s))。
        按论文定义这是误差型指标，越小越好；若用于用户可视化，通常画 1 - fidelity^-。
    """
    model.eval()
    
    with torch.no_grad():
        # 1. 获取原始图的预测
        original_graph = original_graph.to(device)
        
        if is_node:
            # 节点分类：模型接受 (x, edge_index)
            logits_original = model(original_graph.x, original_graph.edge_index)
            probs_original = torch.softmax(logits_original, dim=-1)
            
            # 获取目标节点的ID
            if target_node_id is None:
                target_node_id = getattr(original_graph, 'target_node_subgraph_id', 
                                        getattr(original_graph, '_target_node_subgraph_id', 0))
            
            predicted_label = logits_original[target_node_id].argmax(dim=-1).item()
            prob_original = probs_original[target_node_id, predicted_label].item()
        else:
            # 图分类：模型接受 Data 对象
            logits_original = model(original_graph)
            probs_original = torch.softmax(logits_original, dim=-1)
            
            # 对于图分类，predicted label
            if probs_original.dim() > 1:
                probs_original = probs_original.squeeze(0)
            predicted_label = logits_original.argmax(dim=-1).item()
            prob_original = probs_original[predicted_label].item()
        
        # 2. 获取解释子图的预测
        explanation_subgraph = explanation_subgraph.to(device)
        
        if is_node:
            # 节点分类
            logits_subgraph = model(explanation_subgraph.x, explanation_subgraph.edge_index)
            probs_subgraph = torch.softmax(logits_subgraph, dim=-1)
            
            # 获取目标节点在子图中的ID
            target_id_in_subgraph = getattr(explanation_subgraph, 'target_node_subgraph_id',
                                           getattr(explanation_subgraph, '_target_node_subgraph_id', 0))
            prob_subgraph = probs_subgraph[target_id_in_subgraph, predicted_label].item()
        else:
            # 图分类
            # 确保子图有batch属性
            if not hasattr(explanation_subgraph, 'batch') or explanation_subgraph.batch is None:
                explanation_subgraph.batch = torch.zeros(
                    explanation_subgraph.num_nodes, 
                    dtype=torch.long, 
                    device=device
                )
            
            logits_subgraph = model(explanation_subgraph)
            probs_subgraph = torch.softmax(logits_subgraph, dim=-1)
            
            if probs_subgraph.dim() > 1:
                probs_subgraph = probs_subgraph.squeeze(0)
            prob_subgraph = probs_subgraph[predicted_label].item()
        
        # 3. 计算 Fidelity- = Pr(M(G)) - Pr(M(G_s))
        fidelity_minus = prob_original - prob_subgraph
        
    return fidelity_minus


def compute_direct_constraint_coverage(subgraph, constraints, workload_graph=None, return_stats: bool = False):
    """
    计算 baseline witness 自身直接满足的约束覆盖率。

    baseline 不构造 grounded provenance graph，也不执行 G_s -> G_g 的
    backchase 扩展；只有当单个 witness 自身已经严格满足约束时，才记为覆盖。
    因此这里把 witness 自身同时作为 witness 图和 observed 图来评估，
    不允许借用 witness 外部边，也不允许假设性新增 ΔE。
    为了与主实验输出对齐，这里同时计算：
    - global coverage = |Covered(Q)| / |Σ|
    - normalized coverage = |Covered(Q)| / |Active(Q)|
    其中 Active(Q) 来自当前 witness 图上的 consequent 匹配与 antecedent
    节点齐全性判断。
    """
    try:
        from matcher import find_pattern_matches, backchase_repair_cost
        from grounding_semantics import evaluate_grounding, constraint_activation_summary
    except ImportError:
        try:
            from .matcher import find_pattern_matches, backchase_repair_cost
            from .grounding_semantics import evaluate_grounding, constraint_activation_summary
        except ImportError:
            empty = {
                "covered_constraint_names": [],
                "hit_constraint_names": [],
                "active_constraint_names": [],
                "covered_constraint_count": 0,
                "hit_constraint_count": 0,
                "active_constraint_count": 0,
                "coverage_ratio_global": 0.0,
                "coverage_ratio_normalized": 0.0,
            }
            return empty if return_stats else ([], 0.0)

    workload = subgraph

    if not hasattr(subgraph, "_nodes_in_full") or getattr(subgraph, "_nodes_in_full") is None:
        subgraph._nodes_in_full = torch.arange(subgraph.num_nodes)
    if not hasattr(subgraph, "_nodes_in_observed") or getattr(subgraph, "_nodes_in_observed") is None:
        subgraph._nodes_in_observed = torch.arange(subgraph.num_nodes)
    if not hasattr(workload, "_nodes_in_full") or getattr(workload, "_nodes_in_full") is None:
        workload._nodes_in_full = torch.arange(workload.num_nodes)
    if not hasattr(workload, "_nodes_in_observed") or getattr(workload, "_nodes_in_observed") is None:
        workload._nodes_in_observed = torch.arange(workload.num_nodes)

    grounded = evaluate_grounding(
        subgraph,
        constraints,
        0,
        observed_graph=workload,
        find_pattern_matches_fn=find_pattern_matches,
        backchase_repair_cost_fn=backchase_repair_cost,
    )
    covered_names = sorted(grounded)
    activation = constraint_activation_summary(workload, constraints, workload, find_pattern_matches)
    hit_names = sorted(activation["hit_names"])
    active_names = sorted(activation["active_names"])
    total_constraints = len(constraints)
    coverage_ratio_global = len(covered_names) / total_constraints if total_constraints > 0 else 0.0
    coverage_ratio_normalized = len(covered_names) / len(active_names) if active_names else 0.0

    stats = {
        "covered_constraint_names": covered_names,
        "hit_constraint_names": hit_names,
        "active_constraint_names": active_names,
        "covered_constraint_count": len(covered_names),
        "hit_constraint_count": len(hit_names),
        "active_constraint_count": len(active_names),
        "coverage_ratio_global": float(coverage_ratio_global),
        "coverage_ratio_normalized": float(coverage_ratio_normalized),
    }
    if return_stats:
        return stats
    return covered_names, float(coverage_ratio_normalized)
