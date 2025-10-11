import random
import os
import numpy as np
import yaml
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, TUDataset, Yelp
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.utils import remove_self_loops
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torch_geometric.transforms import ToUndirected, AddSelfLoops, NormalizeFeatures


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


def compute_fidelity_minus(model, original_graph, explanation_subgraph, device):
    """
    计算Fidelity- (Fidelity Minus) 指标
    
    Fidelity- = Pr(M(G)) - Pr(M(G_s))
    其中:
    - G 是原始图 (original_graph)
    - G_s 是解释子图 (explanation_subgraph)
    - M 是GNN模型
    - Pr 是对目标类别的预测概率
    
    对于MUTAG图分类任务，我们使用原始图的predicted label作为目标类别
    
    Args:
        model: 训练好的GNN模型
        original_graph: 原始图 (torch_geometric.data.Data)
        explanation_subgraph: 解释子图/witness (torch_geometric.data.Data)
        device: torch.device
    
    Returns:
        float: fidelity- 值，越大表示解释越重要（移除后预测概率下降越多）
    """
    model.eval()
    
    with torch.no_grad():
        # 1. 获取原始图的预测
        original_graph = original_graph.to(device)
        logits_original = model(original_graph)
        probs_original = torch.softmax(logits_original, dim=-1)
        
        # 对于图分类，predicted label
        if probs_original.dim() > 1:
            probs_original = probs_original.squeeze(0)
        predicted_label = logits_original.argmax(dim=-1).item()
        prob_original = probs_original[predicted_label].item()
        
        # 2. 获取解释子图的预测
        explanation_subgraph = explanation_subgraph.to(device)
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


def compute_constraint_coverage(subgraph, constraints, Budget=8):
    """
    计算解释子图覆盖了多少个约束
    
    使用与ApxChase相同的matching逻辑:
    1. 对每个constraint，在subgraph上找head matches
    2. 检查repair cost是否 <= Budget
    3. 如果满足，则该constraint被覆盖
    
    Args:
        subgraph: 解释子图 (torch_geometric.data.Data)
        constraints: 约束列表
        Budget: repair cost的预算阈值
    
    Returns:
        tuple: (covered_constraint_names, coverage_ratio)
            - covered_constraint_names: list of str, 被覆盖的约束名称
            - coverage_ratio: float, 覆盖比例 = |covered| / |total|
    """
    try:
        from matcher import find_head_matches, backchase_repair_cost
    except ImportError:
        # 如果matcher模块不可用，返回空结果
        return [], 0.0
    
    covered_names = []
    
    for constraint in constraints:
        constraint_name = constraint.get("name", str(constraint))
        
        try:
            # 1. 找到head matches
            head_matches = find_head_matches(subgraph, constraint)
            
            if not head_matches:
                continue
            
            # 2. 对每个match检查repair cost
            for match in head_matches:
                cost = backchase_repair_cost(subgraph, constraint, match, Budget)
                
                # 如果cost有效且在预算内，则该constraint被覆盖
                if cost is not None and cost <= Budget:
                    covered_names.append(constraint_name)
                    break  # 一个constraint只需要一个有效match即可
                    
        except Exception as e:
            # 如果matching出错，跳过该constraint
            continue
    
    # 去重并排序
    covered_names = sorted(set(covered_names))
    coverage_ratio = len(covered_names) / len(constraints) if len(constraints) > 0 else 0.0
    
    return covered_names, coverage_ratio
