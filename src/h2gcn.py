import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import WebKB
from torch_geometric.utils import to_undirected


######################################
# Utilidades
######################################

def accuracy(logits, y):
    preds = logits.argmax(dim=-1)
    return (preds == y).float().mean().item()


def build_row_norm_adj(edge_index, num_nodes, device):
    """
    Construye una matriz de adyacencia dispersa normalizada por filas
    a partir de edge_index (sin autoslóops).
    A_hat = D^{-1} A, donde D_ii = grado de i.
    """
    row, col = edge_index  # [E], [E]

    # Grados de cada nodo: deg[i] = número de aristas que salen de i
    # torch.bincount funciona en CPU y GPU
    deg = torch.bincount(row, minlength=num_nodes).float().to(device)

    # Inversa del grado (evitar división por cero)
    deg_inv = 1.0 / deg.clamp(min=1.0)

    # Peso de cada arista = 1 / deg[u]
    values = deg_inv[row]

    adj = torch.sparse_coo_tensor(
        indices=torch.stack([row, col]),
        values=values,
        size=(num_nodes, num_nodes),
        device=device,
    )

    return adj


######################################
# Modelos
######################################

class MLP(nn.Module):
    """
    Baseline fuerte sin usar el grafo; sólo usa las features.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index=None):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class H2GCNLike(nn.Module):
    """
    Implementación simplificada de H2GCN:
      - Proyecta features a un espacio oculto.
      - Agrega vecinos de 1-hop y 2-hop por separado.
      - Concatena [ego, 1-hop, 2-hop] y clasifica.

    No reproduce todos los detalles del paper, pero
    captura las ideas clave que funcionan bien en grafos heterófilos.
    """
    def __init__(self, in_dim, hidden_dim, out_dim,
                 edge_index, num_nodes, dropout=0.5, use_second_order=True,
                 device="cpu"):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)
        self.use_second_order = use_second_order

        # Convertir el grafo a no dirigido (recomendado para Texas)
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)

        # Matriz de adyacencia normalizada por filas (1-hop)
        self.A1 = build_row_norm_adj(edge_index, num_nodes, device)

        # Matriz de 2-hop: A2 = A1 * A1 (también normalizada aproximadamente)
        if use_second_order:
            # Como Texas es muy pequeño (183 nodos), se puede usar denso sin problemas
            A1_dense = self.A1.to_dense()
            A2_dense = A1_dense @ A1_dense

            # Quitar contribuciones de 0-hop y 1-hop (opcional, para enfatizar 2-hop “nuevo”)
            mask_1hop = (A1_dense > 0).float()
            eye = torch.eye(num_nodes, device=device)
            A2_dense = A2_dense * (1.0 - mask_1hop) * (1.0 - eye)

            # Renormalizar filas
            row_sum = A2_dense.sum(dim=-1, keepdim=True).clamp(min=1.0)
            A2_dense = A2_dense / row_sum

            self.A2 = A2_dense  # denso
        else:
            self.A2 = None

        # Proyección de entrada
        self.lin_in = nn.Linear(in_dim, hidden_dim, bias=True)

        # Clasificador desde la concatenación [x0, x1, x2]
        mult = 3 if use_second_order else 2
        self.lin_out = nn.Linear(hidden_dim * mult, out_dim, bias=True)

    def forward(self, x, edge_index=None):
        # x: [N, F]
        x0 = F.relu(self.lin_in(x))   # ego-embedding
        x0 = self.dropout(x0)

        # 1-hop
        x1 = torch.matmul(self.A1, x0)
        x1 = self.dropout(x1)

        if self.use_second_order:
            # 2-hop (usando A2 densa precomputada)
            x2 = torch.matmul(self.A2, x0)
            x2 = self.dropout(x2)
            z = torch.cat([x0, x1, x2], dim=-1)
        else:
            z = torch.cat([x0, x1], dim=-1)

        out = self.lin_out(z)
        return out


######################################
# Entrenamiento genérico
######################################

def train_model(model, data, train_mask, val_mask, test_mask,
                num_classes, lr=0.01, weight_decay=5e-4,
                max_epochs=2000, patience=200, class_weighting=True):
    device = next(model.parameters()).device

    # Pérdida con pesos de clase para combatir el desbalance
    train_labels = data.y[train_mask]
    class_counts = torch.bincount(train_labels, minlength=num_classes).float()

    if class_weighting:
        eps = 1e-6
        class_weights = class_counts.sum() / (class_counts + eps)
        class_weights = class_weights / class_weights.mean()
        class_weights = class_weights.to(device)
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    x = data.x
    y = data.y
    edge_index = data.edge_index

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        # Eval
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
            train_acc = accuracy(out[train_mask], y[train_mask])
            val_acc   = accuracy(out[val_mask],   y[val_mask])
            test_acc  = accuracy(out[test_mask],  y[test_mask])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 50 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:04d} | "
                f"loss {loss.item():.4f} | "
                f"train {train_acc:.4f} | val {val_acc:.4f} | test {test_acc:.4f}"
            )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Cargar mejor modelo (según validación)
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        final_test_acc = accuracy(out[test_mask], y[test_mask])

    print(f"Final test accuracy (best val): {final_test_acc:.4f}")
    return final_test_acc


######################################
# Script principal (Texas)
######################################

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar dataset Texas
    dataset = WebKB(root="../dataset/WebKB", name="Texas")
    data = dataset[0]
    data.x = data.x.float()
    data.y = data.y.long()
    data.edge_index = data.edge_index.long()

    data = data.to(device)
    num_nodes = data.num_nodes
    num_classes = dataset.num_classes

    # Split oficial (como en tu código)
    def calculate_split_masks(data):
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
            # Shuffle indices
            perm = torch.randperm(num_in_class)
            shuffled_indices = class_indices[perm]
            train_indices = shuffled_indices[:num_train]
            test_indices = shuffled_indices[num_train:]
            train_mask[train_indices] = True
            test_mask[test_indices] = True
        return train_mask, val_mask, test_mask

    train_mask, val_mask, test_mask = calculate_split_masks(data)

    print(f"Dataset Texas | nodes {num_nodes} | classes {num_classes}")
    print(f"train {int(train_mask.sum())}, "
          f"val {int(val_mask.sum())}, test {int(test_mask.sum())}")

    # 1) Baseline MLP
    print("\n=== Entrenando MLP (baseline) ===")
    mlp = MLP(
        in_dim=data.x.size(-1),
        hidden_dim=64,
        out_dim=num_classes,
        dropout=0.5,
    ).to(device)

    train_model(
        mlp, data,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_classes=num_classes,
        lr=0.01,
        weight_decay=5e-4,
        max_epochs=2000,
        patience=200,
        class_weighting=True,
    )

    # 2) Modelo tipo H2GCN
    print("\n=== Entrenando H2GCNLike ===")
    h2gcn_like = H2GCNLike(
        in_dim=data.x.size(-1),
        hidden_dim=64,
        out_dim=num_classes,
        edge_index=data.edge_index,
        num_nodes=num_nodes,
        dropout=0.5,
        use_second_order=True,
        device=device,
    ).to(device)

    train_model(
        h2gcn_like, data,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_classes=num_classes,
        lr=0.01,
        weight_decay=5e-4,
        max_epochs=2000,
        patience=200,
        class_weighting=True,
    )
