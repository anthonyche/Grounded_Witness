import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import os

# 加载数据
graph_path = 'datasets/TreeCycle/treecycle_d5_bf15_n813616.pt'
data = torch.load(graph_path)

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# 两层GCN模型
torch.manual_seed(42)
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GCN(
    in_channels=data.x.size(1),
    hidden_channels=32,
    out_channels=len(torch.unique(data.y))
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 训练参数
epochs = 30
for epoch in range(1, epochs+1):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out, data.y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0 or epoch == 1:
        pred = out.argmax(dim=1)
        acc = (pred == data.y).float().mean().item()
        print(f'Epoch: {epoch:3d}, Loss: {loss.item():.4f}, Acc: {acc:.4f}')

# 保存模型
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/TreeCycle_gcn_d5_bf15_n813616.pth')
print('Model saved to models/TreeCycle_gcn_d5_bf15_n813616.pth')
