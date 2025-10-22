import torch
from src.utils import dataset_func, load_config
from src.model import get_model

config = load_config('config.yaml')
device = torch.device('cpu')

# Load data
print("Loading BAShape data...")
data_resource = dataset_func(config)
data = data_resource['data'].to(device)
print(f"Data loaded: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")

# Load model
print("\nLoading model...")
model = get_model(config).to(device)
model_path = 'models/BAShape_gcn_model.pth'
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()
print("Model loaded")

# Test on full graph
print("\nTesting on full graph...")
with torch.no_grad():
    logits = model(data.x, data.edge_index)
    preds = logits.argmax(dim=-1)
    
# Compute accuracy per class
print('\n=== Model performance on full graph ===')
for label in range(4):
    mask = data.y == label
    if mask.sum() > 0:
        correct = (preds[mask] == label).sum().item()
        total = mask.sum().item()
        acc = 100 * correct / total
        print(f'Label {label}: {acc:.2f}% ({correct}/{total})')

# Overall accuracy
overall_acc = 100 * (preds == data.y).sum().item() / data.num_nodes
print(f'Overall: {overall_acc:.2f}%')

# Test on house nodes specifically
house_mask = data.y > 0
if house_mask.sum() > 0:
    house_correct = (preds[house_mask] == data.y[house_mask]).sum().item()
    house_total = house_mask.sum().item()
    house_acc = 100 * house_correct / house_total
    print(f'\nHouse nodes (labels 1,2,3): {house_acc:.2f}% ({house_correct}/{house_total})')
