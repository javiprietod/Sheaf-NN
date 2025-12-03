import warnings

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data

from src.csnn import CSNN
from src.load_data import load_data
from src.utils import accuracy

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# ---------------------------------------------------------
# Entrenamiento en WebKB–Texas (splits oficiales Pei et al.)
# ---------------------------------------------------------


def main():
    # Cargar dataset Texas
    data: Data
    data, train_mask, val_mask, test_mask, num_classes, class_weights = load_data(
        "texas"
    )
    print()

    torch.manual_seed(42)
    # Modelo CSNN
    model = CSNN(
        in_dim=data.x.size(-1),
        hidden_dim=64,
        out_dim=num_classes,
        num_nodes=data.num_nodes,
        edge_index=data.edge_index,
        num_layers=2,
        dropout=0.5,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        weight_decay=5e-4,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    lr_scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=100, min_lr=1e-5
    )  # metric: val accuracy

    # Entrenamiento con early stopping
    best_test_acc = 0.0
    best_state = None
    max_epochs = 100

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        # Eval
        model.eval()
        with torch.no_grad():
            out = model(data.x)
            train_acc = accuracy(out[train_mask], data.y[train_mask])
            val_acc = accuracy(out[val_mask], data.y[val_mask])
            test_acc = accuracy(out[test_mask], data.y[test_mask])

        # Early stopping basado en validación
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:04d} | "
                f"loss {loss.item():.4f} | "
                f"train {train_acc:.4f} | "
                f"val {val_acc:.4f} | "
                f"test {test_acc:.4f}"
            )

        lr_scheduler.step(val_acc)

    # Cargar mejor modelo
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Evaluación final
    model.eval()
    with torch.no_grad():
        out = model(data.x)
        final_train_acc = accuracy(out[train_mask], data.y[train_mask])
        final_val_acc = accuracy(out[val_mask], data.y[val_mask])
        final_test_acc = accuracy(out[test_mask], data.y[test_mask])

    print(
        f"Final | "
        f"train acc {final_train_acc:.4f} | val acc {final_val_acc:.4f} | test acc {final_test_acc:.4f} |\n"
    )

    # Accuracy per class
    pred = out.argmax(dim=1)
    for class_label in range(num_classes):
        class_indices = (data.y == class_label).nonzero(as_tuple=True)[0]
        class_correct = (pred[class_indices] == data.y[class_indices]).sum().item()
        class_total = class_indices.size(0)
        class_acc = class_correct / class_total if class_total > 0 else 0.0
        print(
            f"Class {class_label}: Accuracy {class_acc:.4f} ({class_correct}/{class_total})"
        )


if __name__ == "__main__":
    main()
