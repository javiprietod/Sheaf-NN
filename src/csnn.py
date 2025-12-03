import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# Capa CSNN: sheaf dirigido + comportamiento cooperativo
# ---------------------------------------------------------


class CSNNLayer(nn.Module):
    """
    Cooperative Sheaf Neural Network (CSNN) layer, inspired by Ribeiro et al. (2025).

    - Directed graph: edge_index[0] -> edge_index[1]
    - For each node i we learn:
        * v_src[i]  -> produces orthogonal map Q_src[i] (Householder)
        * v_tgt[i]  -> produces orthogonal map Q_tgt[i]
        * alpha_src[i] >= 0  (scale for S_i)
        * alpha_tgt[i] >= 0  (scale for T_i)
      and define conformal maps:
        S_i = s_i * Q_src[i],   T_i = t_i * Q_tgt[i]

    - Two sheaf Laplacians are constructed:
        L_out(x)  : diffusion using S (out-degree)
        L_in_T(x) : diffusion using T (approximate transpose of in-degree)

    - Layer update rule:
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
        Build S_i and T_i from Householder reflections and scales.
        Returns:
            S: [n, d, d]
            T: [n, d, d]
        """
        device = x.device
        n, d = self.num_nodes, self.dim

        I_tensor = (
            torch.eye(d, device=device).unsqueeze(0).expand(n, -1, -1)
        )  # [n, d, d]

        # --- Source maps S_i ---
        v = self.v_src
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-6)  # [n, d]
        vvT = torch.einsum("ni,nj->nij", v, v)  # [n, d, d]
        Q_src = I_tensor - 2.0 * vvT  # [n, d, d] (Householder)
        s = F.softplus(self.alpha_src).view(n, 1, 1)  # [n,1,1] escala positiva
        S = s * Q_src  # [n, d, d]

        # --- Target maps T_i ---
        v = self.v_tgt
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-6)
        vvT = torch.einsum("ni,nj->nij", v, v)
        Q_tgt = I_tensor - 2.0 * vvT
        t = F.softplus(self.alpha_tgt).view(n, 1, 1)
        T = t * Q_tgt

        return S, T

    def sheaf_L_out(self, x, S):
        """
        L_out(x) using S maps (out-degree sheaf Laplacian, normalized).
        Accumulates contributions at the source node of each edge.
        """
        src, dst = self.edge_index
        x_src = x[src]  # [E, d]
        x_dst = x[dst]  # [E, d]
        S_src = S[src]  # [E, d, d]
        S_dst = S[dst]  # [E, d, d]

        # Transporte: S_i^T S_j x_j
        Sj_xj = torch.einsum("eoi,ei->eo", S_dst, x_dst)  # [E, d]
        Sj_xj_to_i = torch.einsum("eio,eo->ei", S_src.transpose(1, 2), Sj_xj)
        contrib = x_src - Sj_xj_to_i  # [E, d]

        out = x.new_zeros(x.size(0), x.size(1))
        out.index_add_(0, src, contrib)

        # Normalización por grado out (approx. Laplaciano normalizado)
        deg_out = self.deg_out.clamp_min(1.0).view(-1, 1)
        out = out / deg_out
        return out

    def sheaf_L_in_T(self, x, T):
        """
        Approximate L_in^T(x) using T maps.
        Accumulates contributions at the destination node of each edge.
        """
        src, dst = self.edge_index
        x_src = x[src]  # [E, d]
        x_dst = x[dst]  # [E, d]
        T_src = T[src]  # [E, d, d]
        T_dst = T[dst]  # [E, d, d]

        # Transporte "inverso": T_j^T T_i x_i
        Ti_xi = torch.einsum("eoi,ei->eo", T_src, x_src)  # [E, d]
        Ti_xi_to_j = torch.einsum("eio,eo->ei", T_dst.transpose(1, 2), Ti_xi)
        contrib = x_dst - Ti_xi_to_j  # [E, d]

        out = x.new_zeros(x.size(0), x.size(1))
        out.index_add_(0, dst, contrib)

        # Normalización por grado in
        deg_in = self.deg_in.clamp_min(1.0).view(-1, 1)
        out = out / deg_in
        return out

    def forward(self, x):
        """
        x: [num_nodes, dim]
        """
        S, T = self._conformal_maps(x)  # [n, d, d] cada uno

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
    CSNN for node classification:
        - initial projection to `hidden_dim`.
        - L `CSNNLayer` layers.
        - final linear classifier.
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
