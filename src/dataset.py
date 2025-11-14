import warnings

warnings.filterwarnings("ignore")

from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader

dataset = PygGraphPropPredDataset(name="ogbg-moltox21")
print(dataset)

split_idx = dataset.get_idx_split()
train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)

# Load first batch
batch = next(iter(train_loader))
print(batch)
print("Node features:")
print(batch.x)  # Node features
