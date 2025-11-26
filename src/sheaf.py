import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# Capa CSNN: sheaf dirigido + comportamiento cooperativo
# ---------------------------------------------------------

class CSNNLayer(nn.Module):
    """
    Capa de Cooperative Sheaf Neural Network (CSNN), inspirada en Ribeiro et al. (2025).

    - Grafo dirigido: edge_index[0] -> edge_index[1]
    - Para cada nodo i se aprenden:
        * v_src[i]  -> genera mapa ortogonal Q_src[i] (Householder)
        * v_tgt[i]  -> genera mapa ortogonal Q_tgt[i]
        * alpha_src[i] >= 0  (escala de S_i)
        * alpha_tgt[i] >= 0  (escala de T_i)
      y se definen mapas conformes:
        S_i = s_i * Q_src[i],   T_i = t_i * Q_tgt[i]

    - Se construyen dos Laplacianos de haz:
        L_out(x)      : difusión usando S (out-degree)
        L_in_T(x)     : difusión usando T (transpose del in-degree, aproximado)

    - Actualización de la capa:
        x' = x - eps_out * W_out(L_out(x)) - eps_in * W_in(L_in_T(x))
        x' = phi( W_feat(x') )
    """

    def __init__(self, dim, num_nodes, edge_index, eps_init=0.1):
        super().__init__()
        self.dim = dim
        self.num_nodes = num_nodes

        # Estructura del grafo dirigido
        self.register_buffer("edge_index", edge_index)  # [2, E]
        src, dst = edge_index

        # Grados out e in para normalización
        deg_out = torch.bincount(src, minlength=num_nodes).float()
        deg_in = torch.bincount(dst, minlength=num_nodes).float()
        self.register_buffer("deg_out", deg_out)
        self.register_buffer("deg_in", deg_in)

        # Parámetros de Householder + escalas conformes
        self.v_src = nn.Parameter(torch.randn(num_nodes, dim))
        self.v_tgt = nn.Parameter(torch.randn(num_nodes, dim))
        self.alpha_src = nn.Parameter(torch.zeros(num_nodes))  # softplus -> positivo
        self.alpha_tgt = nn.Parameter(torch.zeros(num_nodes))

        # Transformaciones lineales para L_out y L_in^T
        self.lin_out = nn.Linear(dim, dim, bias=False)
        self.lin_in = nn.Linear(dim, dim, bias=False)
        self.lin_feat = nn.Linear(dim, dim, bias=True)

        # Pasos de tiempo aprendidos
        self.eps_out = nn.Parameter(torch.tensor(eps_init))
        self.eps_in = nn.Parameter(torch.tensor(eps_init))

    def _conformal_maps(self, x):
        """
        Construye S_i y T_i a partir de Householder + escalas.
        Devuelve:
            S: [n, d, d]
            T: [n, d, d]
        """
        device = x.device
        n, d = self.num_nodes, self.dim

        I = torch.eye(d, device=device).unsqueeze(0).expand(n, -1, -1)  # [n, d, d]

        # --- Source maps S_i ---
        v = self.v_src
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-6)           # [n, d]
        vvT = torch.einsum("ni,nj->nij", v, v)                  # [n, d, d]
        Q_src = I - 2.0 * vvT                                   # [n, d, d] (Householder)
        s = F.softplus(self.alpha_src).view(n, 1, 1)            # [n,1,1] escala positiva
        S = s * Q_src                                           # [n, d, d]

        # --- Target maps T_i ---
        v = self.v_tgt
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-6)
        vvT = torch.einsum("ni,nj->nij", v, v)
        Q_tgt = I - 2.0 * vvT
        t = F.softplus(self.alpha_tgt).view(n, 1, 1)
        T = t * Q_tgt

        return S, T

    def sheaf_L_out(self, x, S):
        """
        L_out(x) usando mapas S (out-degree sheaf Laplacian, normalizado).
        Acumula contribuciones en el nodo source de cada arista.
        """
        src, dst = self.edge_index
        x_src = x[src]              # [E, d]
        x_dst = x[dst]              # [E, d]
        S_src = S[src]              # [E, d, d]
        S_dst = S[dst]              # [E, d, d]

        # Transporte: S_i^T S_j x_j
        Sj_xj = torch.einsum("eoi,ei->eo", S_dst, x_dst)              # [E, d]
        Sj_xj_to_i = torch.einsum("eio,eo->ei", S_src.transpose(1, 2), Sj_xj)
        contrib = x_src - Sj_xj_to_i                                  # [E, d]

        out = x.new_zeros(x.size(0), x.size(1))
        out.index_add_(0, src, contrib)

        # Normalización por grado out (approx. Laplaciano normalizado)
        deg_out = self.deg_out.clamp_min(1.0).view(-1, 1)
        out = out / deg_out
        return out

    def sheaf_L_in_T(self, x, T):
        """
        L_in^T(x) aproximado usando mapas T.
        Acumula contribuciones en el nodo destino de cada arista.
        """
        src, dst = self.edge_index
        x_src = x[src]              # [E, d]
        x_dst = x[dst]              # [E, d]
        T_src = T[src]              # [E, d, d]
        T_dst = T[dst]              # [E, d, d]

        # Transporte "inverso": T_j^T T_i x_i
        Ti_xi = torch.einsum("eoi,ei->eo", T_src, x_src)              # [E, d]
        Ti_xi_to_j = torch.einsum("eio,eo->ei", T_dst.transpose(1, 2), Ti_xi)
        contrib = x_dst - Ti_xi_to_j                                  # [E, d]

        out = x.new_zeros(x.size(0), x.size(1))
        out.index_add_(0, dst, contrib)

        # Normalización por grado in
        deg_in = self.deg_in.clamp_min(1.0).view(-1, 1)
        out = out / deg_in
        return out

    def forward(self, x):
        """
        x : [num_nodes, dim]
        """
        S, T = self._conformal_maps(x)           # [n, d, d] cada uno

        L_out_x = self.sheaf_L_out(x, S)
        L_inT_x = self.sheaf_L_in_T(x, T)

        # Discretización de ecuación de calor + mezclas
        update_out = self.lin_out(L_out_x)
        update_in = self.lin_in(L_inT_x)

        x_new = x - self.eps_out * update_out - self.eps_in * update_in
        x_new = self.lin_feat(x_new)
        x_new = F.relu(x_new)
        return x_new


# ---------------------------------------------------------
# Red CSNN completa
# ---------------------------------------------------------

class CSNN(nn.Module):
    """
    CSNN para clasificación de nodos:
      - Proyección inicial a hidden_dim.
      - L capas CSNNLayer.
      - Clasificador linear final.
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_nodes,
        edge_index,
        num_layers=4,
        dropout=0.5,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList(
            [
                CSNNLayer(
                    dim=hidden_dim,
                    num_nodes=num_nodes,
                    edge_index=edge_index,
                )
                for _ in range(num_layers)
            ]
        )

        self.readout = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_proj(x)
        x = F.relu(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)

        out = self.readout(x)
        return out
