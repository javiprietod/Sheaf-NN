# Dataset Texas
import torch
from torch_geometric.datasets import WebKB
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_split_masks(data, num_classes = 5):
        """Merge all splits into one and create a new stratified 70/0/30 split."""
        labels = data.y
        num_nodes = data.num_nodes
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        for class_label in range(num_classes):
            class_indices = (labels == class_label).nonzero(as_tuple=True)[0]
            num_in_class = class_indices.size(0)
            num_train = int(0.7 * num_in_class)
            num_val = int(0.1 * num_in_class)
            # Shuffle indices
            perm = torch.randperm(num_in_class)
            shuffled_indices = class_indices[perm]
            train_indices = shuffled_indices[:num_train]
            val_indices = shuffled_indices[num_train:num_train+num_val]
            test_indices = shuffled_indices[num_train+num_val:]
            train_mask[train_indices] = True
            val_mask[val_indices] = True
            test_mask[test_indices] = True
        return train_mask, val_mask, test_mask

def load_texas_data():
    dataset = WebKB(
        root="../dataset/WebKB",
        name="Texas",
        transform=NormalizeFeatures(),
    )
    data: Data = dataset[0]

    data.x = data.x.float()
    data.y = data.y.long()
    data.edge_index = data.edge_index.long()

    num_nodes = data.num_nodes
    num_classes = dataset.num_classes

    data = data.to(device)
    torch.manual_seed(42)

    train_mask, val_mask, test_mask = calculate_split_masks(data)

    print(f"Texas | num_nodes={num_nodes} | num_classes={num_classes}")
    print(
        f"Train={int(train_mask.sum())}, "
        f"val={int(val_mask.sum())}, test={int(test_mask.sum())}"
    )

    # Pesos de clase (para evitar colapso en la clase 3)
    train_labels = data.y[train_mask]
    class_counts = torch.bincount(train_labels, minlength=num_classes).float()
    eps = 1e-6
    class_weights = class_counts.sum() / (class_counts + eps)
    class_weights = class_weights / class_weights.mean()
    class_weights = class_weights.to(device)
    print("class_counts:", class_counts.tolist())
    print("class_weights:", class_weights.tolist())

    return data, train_mask, val_mask, test_mask, num_classes, class_weights
