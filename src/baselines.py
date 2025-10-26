# baselines.py
# Lightweight wrappers for GNNExplainer and PGExplainer
# to be called from Run_Experiment.py in our setting (graph classification).
# We assume `model` is a trained PyG model that accepts a `Data` (graph) and returns logits.

from __future__ import annotations

from typing import Optional, Dict, Any, List, Tuple, Union

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# New-style Explainer API (PyG >= 2.3)
from torch_geometric.explain import Explainer, ModelConfig
from torch_geometric.explain.config import MaskType, ModelMode, ModelTaskLevel
from torch_geometric.explain.algorithm import GNNExplainer, PGExplainer


class _DataModelWrapper(torch.nn.Module):
    """
    Wraps a model with signature model(Data) -> logits into a module with
    signature forward(x, edge_index, batch=None, **kwargs) so that it can be
    used by PyG's Explainer/GNNExplainer/PGExplainer.
    """
    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x: Tensor, edge_index: Tensor, batch: Optional[Tensor] = None, **kwargs) -> Tensor:
        data = Data(x=x, edge_index=edge_index)
        if batch is not None:
            data.batch = batch
        return self.base_model(data)


def _move_data_to_device(data: Data, device: torch.device) -> Data:
    # PyG Data implements .to; cloning not strictly necessary for inference.
    return data.to(device)


# -----------------------------
# GNNExplainer baseline (graph)
# -----------------------------
def run_gnn_explainer_graph(
    model: torch.nn.Module,
    graph: Data,
    *,
    epochs: int = 100,
    lr: float = 0.01,
    feat_mask_type: str = "feature",  # 'feature' | 'individual_feature' | 'scalar'
    allow_edge_mask: bool = True,
    perturbations_per_epoch: int = 5,  # kept for compatibility; unused in new API
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Runs **new-style** PyG Explainer API with the GNNExplainer algorithm on a single graph.
    Returns dict with 'edge_mask', 'node_feat_mask', 'pred', 'prob'.

    feat_mask_type mapping -> MaskType:
      - 'feature' -> MaskType.common_attributes (one vector shared across nodes)
      - 'individual_feature' -> MaskType.attributes (per-node, per-feature)
      - 'scalar' -> MaskType.object (one scalar per node)
    """
    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(dev).eval()
    H = _move_data_to_device(graph, dev)

    # Map legacy string to MaskType:
    if feat_mask_type == "feature":
        node_mask_type = MaskType.common_attributes
    elif feat_mask_type == "individual_feature":
        node_mask_type = MaskType.attributes
    elif feat_mask_type == "scalar":
        node_mask_type = MaskType.object
    else:
        raise ValueError(f"Invalid feat_mask_type: {feat_mask_type}")

    edge_mask_type = MaskType.object if allow_edge_mask else None

    wrapped_model = _DataModelWrapper(model)
    # Build the new-style Explainer wrapper with the GNNExplainer backend:
    algorithm = GNNExplainer(epochs=epochs, lr=lr)

    explainer = Explainer(
        model=wrapped_model,
        algorithm=algorithm,
        explanation_type='model',
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
        model_config=ModelConfig(
            mode=ModelMode.multiclass_classification,  # graph classification logits
            task_level=ModelTaskLevel.graph,
            return_type='raw',  # model returns raw logits
        ),
    )

    # Run explanation:
    explanation = explainer(x=H.x, edge_index=H.edge_index)

    # Fetch masks (may be None depending on config):
    node_mask = explanation.get('node_mask') if hasattr(explanation, 'get') else getattr(explanation, 'node_mask', None)
    edge_mask = explanation.get('edge_mask') if hasattr(explanation, 'get') else getattr(explanation, 'edge_mask', None)

    with torch.no_grad():
        out = model(H)
        prob = torch.softmax(out, dim=-1).squeeze(0)
        pred = int(prob.argmax().item())

    return {
        "edge_mask": None if edge_mask is None else edge_mask.detach().cpu(),
        "node_feat_mask": None if node_mask is None else node_mask.detach().cpu(),
        "pred": pred,
        "prob": prob.detach().cpu(),
    }


# -----------------------------
# GNNExplainer baseline (node)
# -----------------------------
def run_gnn_explainer_node(
    model: torch.nn.Module,
    data: Data,
    target_node: int,
    *,
    epochs: int = 100,
    lr: float = 0.01,
    feat_mask_type: str = "feature",
    allow_edge_mask: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Run GNNExplainer for node classification tasks.
    Similar to run_gnn_explainer_graph but for single node explanation.
    """
    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(dev).eval()
    H = _move_data_to_device(data, dev)
    
    # Map feat_mask_type to MaskType
    if feat_mask_type == "feature":
        node_mask_type = MaskType.common_attributes
    elif feat_mask_type == "individual_feature":
        node_mask_type = MaskType.attributes
    elif feat_mask_type == "scalar":
        node_mask_type = MaskType.object
    else:
        raise ValueError(f"Invalid feat_mask_type: {feat_mask_type}")
    
    edge_mask_type = MaskType.object if allow_edge_mask else None
    
    # Wrapper for node classification model
    class NodeModelWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.model = base_model
        
        def forward(self, x, edge_index, **kwargs):
            return self.model(x, edge_index)
    
    wrapped_model = NodeModelWrapper(model)
    algorithm = GNNExplainer(epochs=epochs, lr=lr)
    
    explainer = Explainer(
        model=wrapped_model,
        algorithm=algorithm,
        explanation_type='model',
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
        model_config=ModelConfig(
            mode=ModelMode.multiclass_classification,
            task_level=ModelTaskLevel.node,  # Node-level task
            return_type='raw',
        ),
    )
    
    # Run explanation for target node
    explanation = explainer(
        x=H.x,
        edge_index=H.edge_index,
        index=target_node,  # Explain this specific node
    )
    
    # Extract masks
    node_mask = explanation.get('node_mask') if hasattr(explanation, 'get') else getattr(explanation, 'node_mask', None)
    edge_mask = explanation.get('edge_mask') if hasattr(explanation, 'get') else getattr(explanation, 'edge_mask', None)
    
    # Get prediction
    with torch.no_grad():
        out = model(H.x, H.edge_index)
        prob = torch.softmax(out[target_node], dim=-1)
        pred = int(prob.argmax().item())
    
    return {
        "edge_mask": None if edge_mask is None else edge_mask.detach().cpu(),
        "node_feat_mask": None if node_mask is None else node_mask.detach().cpu(),
        "pred": pred,
        "prob": prob.detach().cpu(),
        "target_node": target_node,
    }


# ---------------------------
# PGExplainer baseline (graph) — NEW API
# ---------------------------
class PGExplainerBaseline:
    """
    Wrapper around the **new** torch_geometric.explain.algorithm.PGExplainer
    used via the generic Explainer. This version supports training the
    explainer on a loader, then explaining single graphs.

    Notes:
    - Unlike GNNExplainer, PGExplainer is *parametric* and needs training
      across samples. Use `fit(loader)` before calling `explain_graph`.
    - For convenience, `quick_fit=True` does a tiny warm-up on the given graph.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        epochs: int = 100,
        lr: float = 0.003,
        temp: Union[float, Tuple[float, float]] = (5.0, 1.0),
        allow_edge_mask: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = model.to(self.device).eval()
        self.epochs = epochs
        self.lr = lr
        # Normalize temperature to the new API expectation: a (t0, t1) tuple.
        _temp = temp
        if isinstance(_temp, (int, float)):
            _temp = (float(_temp), float(_temp))
        elif isinstance(_temp, tuple):
            if len(_temp) != 2:
                raise ValueError("temp must be either a float or a (t0, t1) tuple.")
            _temp = (float(_temp[0]), float(_temp[1]))
        else:
            raise ValueError("temp must be either a float or a (t0, t1) tuple.")
        self.temp = _temp
        self.allow_edge_mask = allow_edge_mask

        # Wrap base model to (x, edge_index[, batch]) signature:
        self._wrapped = _DataModelWrapper(self.model)

        # Build new-style algorithm + Explainer
        self._algorithm = PGExplainer(epochs=self.epochs, lr=self.lr, temp=self.temp)
        self._explainer = Explainer(
            model=self._wrapped,
            algorithm=self._algorithm,
            explanation_type='phenomenon',
            node_mask_type=None,  # typical PGExplainer focuses on edges for graph tasks
            edge_mask_type=MaskType.object if self.allow_edge_mask else None,
            model_config=ModelConfig(
                mode=ModelMode.multiclass_classification,
                task_level=ModelTaskLevel.graph,
                return_type='raw',
            ),
        )
        self._is_trained = False

    def fit(self, loader: DataLoader) -> None:
        """
        Train PGExplainer on a loader of graphs.
        Properly handles PGExplainer's API which requires epoch, x, edge_index as inputs.
        """
        import time
        
        print(f"[PGExplainer] 开始训练 - epochs: {self.epochs}, 学习率: {self.lr}")
        
        # Normalize to device: materialize dataset to self.device
        dataset = getattr(loader, 'dataset', None)
        if dataset is None:
            raise ValueError("PGExplainerBaseline.fit expects a DataLoader with an accessible dataset.")

        device_dataset: List[Data] = []
        for data in dataset:
            d = data.clone() if hasattr(data, 'clone') else Data.from_dict(data.to_dict())
            device_dataset.append(_move_data_to_device(d, self.device))

        batch_size = getattr(loader, 'batch_size', None) or 1
        dev_loader = DataLoader(device_dataset, batch_size=batch_size, shuffle=True)
        
        print(f"[PGExplainer] 数据集大小: {len(device_dataset)} 图, 批次大小: {batch_size}")
        
        # 添加性能测量
        start_time = time.time()
        
        # Loop through epochs and batches to train the explainer
        for epoch in range(self.epochs):
            epoch_start = time.time()
            batch_count = 0
            
            for batch in dev_loader:
                batch_count += 1
                x = batch.x
                edge_index = batch.edge_index
                batch_index = getattr(batch, 'batch', None)
                
                # Get target from model prediction (or use batch.y if available)
                with torch.no_grad():
                    out = self._wrapped(x=x, edge_index=edge_index, batch=batch_index)
                    target = getattr(batch, 'y', out.argmax(dim=-1))
                
                # Call train with explicit epoch parameter
                self._algorithm.train(
                    epoch=epoch,
                    model=self._wrapped,
                    x=x,
                    edge_index=edge_index,
                    target=target,
                    batch=batch_index
                )
            
            epoch_end = time.time()
            # 每隔几个epoch打印一次，避免输出太多
            if epoch % max(1, self.epochs // 10) == 0 or epoch == self.epochs - 1:
                print(f"[PGExplainer] Epoch {epoch+1}/{self.epochs} 完成, 用时: {epoch_end - epoch_start:.4f}秒, 处理 {batch_count} 批次")
        
        end_time = time.time()
        print(f"[PGExplainer] 训练完成! 总用时: {end_time - start_time:.4f}秒")
                
        self._is_trained = True
        return

    def explain_graph(
        self,
        graph: Data,
        *,
        quick_fit: bool = False,
        copies: int = 20,
    ) -> Dict[str, Any]:
        """
        Explain a single masked graph. If `quick_fit=True` or the explainer
        has not been trained yet, performs a proper training pass before
        generating explanations (no zero-epoch fallbacks).
        """
        import time
        
        H = _move_data_to_device(graph, self.device)
        self.model.eval()

        # Ensure the PGExplainer is trained
        if quick_fit or not getattr(self, '_is_trained', False):
            print(f"[PGExplainer] {'快速拟合' if quick_fit else '首次训练'} - 复制图 {copies} 次用于训练")
            train_start = time.time()
            tmp_list = [H.clone() for _ in range(max(1, copies))]
            tmp_loader = DataLoader(tmp_list, batch_size=1, shuffle=True)
            self.fit(tmp_loader)
            train_end = time.time()
            print(f"[PGExplainer] 训练/拟合阶段完成, 用时: {train_end - train_start:.4f}秒")
        else:
            print(f"[PGExplainer] 使用已训练模型生成解释")

        with torch.no_grad():
            out = self.model(H)
            prob = torch.softmax(out, dim=-1).squeeze(0)
            pred = int(prob.argmax().item())
            
        print(f"[PGExplainer] 生成图解释, 预测类别: {pred}, 置信度: {prob[pred]:.4f}")
        explain_start = time.time()
        # Run new-style explanation (now that the algorithm is trained)
        explanation = self._explainer(x=H.x, edge_index=H.edge_index, target=int(pred))
        explain_end = time.time()
        
        edge_mask = explanation.get('edge_mask') if hasattr(explanation, 'get') else getattr(explanation, 'edge_mask', None)
        node_mask = explanation.get('node_mask') if hasattr(explanation, 'get') else getattr(explanation, 'node_mask', None)
        
        # 报告解释结果的基本信息
        print(f"[PGExplainer] 解释生成完成, 用时: {explain_end - explain_start:.4f}秒")
        if edge_mask is not None:
            print(f"[PGExplainer] 边掩码形状: {edge_mask.shape}, 平均值: {edge_mask.mean().item():.4f}")
        if node_mask is not None:
            print(f"[PGExplainer] 节点掩码形状: {node_mask.shape}, 平均值: {node_mask.mean().item():.4f}")

        return {
            "edge_mask": None if edge_mask is None else edge_mask.detach().cpu(),
            "node_feat_mask": None if node_mask is None else node_mask.detach().cpu(),
            "pred": pred,
            "prob": prob.detach().cpu(),
        }


# -----------------------------
# PGExplainer baseline (node)
# -----------------------------
# Global cache for trained PGExplainer instances
_pg_explainer_cache = {}

class PGExplainerNodeCache:
    """Cache for trained PGExplainer to avoid retraining for each node."""
    
    def __init__(self, model, full_data, device, epochs=30, lr=0.003):
        # Don't call model.to(device) - model is already on correct device
        # Calling .to() again might create issues
        self.model = model.eval()
        self.full_data = _move_data_to_device(full_data, device)
        self.device = device
        self.explainer = None
        self.wrapped_model = None
        self._train(epochs, lr)
    
    def _train(self, epochs, lr):
        """Train PGExplainer once on the full graph."""
        # Wrapper for node classification model
        class NodeModelWrapper(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.model = base_model
            
            def forward(self, x, edge_index, **kwargs):
                return self.model(x, edge_index)
        
        self.wrapped_model = NodeModelWrapper(self.model)
        algorithm = PGExplainer(epochs=epochs, lr=lr)
        
        self.explainer = Explainer(
            model=self.wrapped_model,
            algorithm=algorithm,
            explanation_type='phenomenon',
            node_mask_type=None,
            edge_mask_type=MaskType.object,
            model_config=ModelConfig(
                mode=ModelMode.multiclass_classification,
                task_level=ModelTaskLevel.node,
                return_type='raw',
            ),
        )
        
        # Train on multiple nodes from the full graph
        num_train_nodes = min(100, self.full_data.x.size(0) // 2)
        # Create train_indices on the same device as the data
        train_indices = torch.randperm(self.full_data.x.size(0), device=self.device)[:num_train_nodes]
        
        print(f"[PGExplainer] Training once on {self.full_data.x.size(0)} nodes, {self.full_data.edge_index.size(1)} edges")
        print(f"[PGExplainer] Training with {num_train_nodes} sample nodes")
        print(f"[PGExplainer] Device check: x={self.full_data.x.device}, edge_index={self.full_data.edge_index.device}, y={self.full_data.y.device}, model={next(self.model.parameters()).device}")
        
        # Force CUDA context if using GPU (fix for PyG internal tensor creation)
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
            print(f"[PGExplainer] Set CUDA device context to {self.device}")
        
        for epoch in range(1, epochs + 1):
            for idx in train_indices:
                idx_int = int(idx.item())
                # Ensure we're in the right CUDA context
                if self.device.type == 'cuda':
                    with torch.cuda.device(self.device):
                        loss = algorithm.train(
                            epoch,
                            model=self.wrapped_model,
                            x=self.full_data.x,
                            edge_index=self.full_data.edge_index,
                            index=idx_int,
                            target=self.full_data.y,
                        )
                else:
                    loss = algorithm.train(
                        epoch,
                        model=self.wrapped_model,
                        x=self.full_data.x,
                        edge_index=self.full_data.edge_index,
                        index=idx_int,
                        target=self.full_data.y,
                    )
        
        print(f"[PGExplainer] Training completed after {epochs} epochs")
    
    def explain(self, subgraph_data, target_node):
        """Explain a specific node using the trained explainer."""
        H = _move_data_to_device(subgraph_data, self.device)
        
        # Force CUDA context if using GPU
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        
        # Get target label
        with torch.no_grad():
            out = self.model(H.x, H.edge_index)
            target_label = out[target_node].argmax()
        
        # Generate explanation with CUDA context
        if self.device.type == 'cuda':
            with torch.cuda.device(self.device):
                explanation = self.explainer(
                    x=H.x,
                    edge_index=H.edge_index,
                    index=target_node,
                    target=target_label,
                )
        else:
            explanation = self.explainer(
                x=H.x,
                edge_index=H.edge_index,
                index=target_node,
                target=target_label,
            )
        
        return explanation, out, target_label


def run_pgexplainer_node(
    model: torch.nn.Module,
    data: Data,
    target_node: int,
    *,
    epochs: int = 30,
    lr: float = 0.003,
    feat_mask_type: str = "feature",
    allow_edge_mask: bool = True,
    device: Optional[torch.device] = None,
    full_data: Optional[Data] = None,  # Full graph for training
    use_cache: bool = True,  # Whether to use cached trained explainer
) -> Dict[str, Any]:
    """
    Run PGExplainer for node classification tasks.
    Uses a cached trained explainer to avoid retraining for each node.
    
    Args:
        model: GNN model
        data: Subgraph data around target node
        target_node: Node index in the subgraph (after relabeling)
        full_data: Full graph data for training (if None, uses data)
        epochs: Training epochs (only used if training new explainer)
        lr: Learning rate
        device: Device to use
        use_cache: If True, reuse trained explainer across multiple nodes
        
    Returns:
        Dictionary with edge_mask, pred, prob, target_node
    """
    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    
    # Determine which data to use for training
    training_data = full_data if full_data is not None else data
    if not hasattr(training_data, 'num_nodes') or training_data.num_nodes is None:
        training_data.num_nodes = training_data.x.size(0)
    
    # Create cache key based on model and data
    cache_key = id(model)
    
    # Get or create cached explainer
    if use_cache and cache_key in _pg_explainer_cache:
        print(f"[PGExplainer] Using cached trained explainer for node {target_node}")
        pg_cache = _pg_explainer_cache[cache_key]
    else:
        if use_cache:
            print(f"[PGExplainer] Training new explainer (will be cached)")
        else:
            print(f"[PGExplainer] Training new explainer (cache disabled)")
        
        pg_cache = PGExplainerNodeCache(
            model=model,
            full_data=training_data,
            device=dev,
            epochs=epochs,
            lr=lr
        )
        
        if use_cache:
            _pg_explainer_cache[cache_key] = pg_cache
    
    # Use the trained explainer to explain this specific node
    explanation, out, target_label = pg_cache.explain(data, target_node)
    
    # Extract masks
    node_mask = explanation.get('node_mask') if hasattr(explanation, 'get') else getattr(explanation, 'node_mask', None)
    edge_mask = explanation.get('edge_mask') if hasattr(explanation, 'get') else getattr(explanation, 'edge_mask', None)
    
    # Get prediction (already computed in explain)
    prob = torch.softmax(out[target_node], dim=-1)
    pred = int(prob.argmax().item())
    
    return {
        "edge_mask": None if edge_mask is None else edge_mask.detach().cpu(),
        "node_feat_mask": None if node_mask is None else node_mask.detach().cpu(),
        "pred": pred,
        "prob": prob.detach().cpu(),
        "target_node": target_node,
    }


__all__ = [
    "run_gnn_explainer_graph",
    "run_gnn_explainer_node",
    "run_pgexplainer_node",
    "PGExplainerBaseline",
]
