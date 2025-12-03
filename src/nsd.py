"""
Neural Sheaf Diffusion (NSD) module.
"""

import torch
import torch.nn as nn


def build_sheaf_laplacian_scalar(n_nodes, edges, F_ve, F_ue):
    """
    n_nodes: number of nodes
    edges: [m, 2] tensor with (v, u)
    F_ve, F_ue: [m] tensors with the maps F_{v->e}, F_{u->e}

    Returns the sheaf Laplacian Delta_F (n*d x n*d). For scalar maps (d=1)
    this is an [n, n] matrix.
    """
    m = edges.shape[0]
    L = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)

    # Para cada arista e = (v, u):
    for e_idx in range(m):
        v = edges[e_idx, 0].item()
        u = edges[e_idx, 1].item()
        fv = F_ve[e_idx]
        fu = F_ue[e_idx]

        # Según el paper:
        # L_F(x)_v = sum_{v,u~e} F_v^T(F_v x_v - F_u x_u)
        L[v, v] += torch.dot(fv, fv)
        L[u, u] += torch.dot(fu, fu)
        L[v, u] -= torch.dot(fv, fu)
        L[u, v] -= torch.dot(fu, fv)

    # Normalización por D^{-1/2}
    d = torch.diag(L)
    # Evitar división por cero
    d_clamped = torch.clamp(d, min=1e-6)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(d_clamped))
    Delta = D_inv_sqrt @ L @ D_inv_sqrt

    return Delta


class PhiGeneral(nn.Module):
    """
    General Phi: produces distinct F_{v->e} and F_{u->e} (asymmetric model).
    """

    def __init__(self, in_dim, hidden_dim=32, d=1):
        super().__init__()
        self.mlp_v = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, d)
        )
        self.mlp_u = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, d)
        )

    def forward(self, x_v, x_u):
        """
        x_v, x_u: [m, f]
        returns: F_ve, F_ue: [m]
        """
        inp_v = torch.cat([x_v, x_u], dim=-1)
        F_ve = self.mlp_v(inp_v).squeeze(-1)
        inp_u = torch.cat([x_u, x_v], dim=-1)
        F_ue = self.mlp_u(inp_u).squeeze(-1)

        return F_ve, F_ue


class PhiSymmetric(nn.Module):
    """
    Symmetric Phi: F_{v->e} = F_{u->e} (weighted graph Laplacian-like model).
    """

    def __init__(self, in_dim, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_v, x_u):
        inp = torch.cat([x_v, x_u], dim=-1)
        F_e = self.mlp(inp).squeeze(-1)
        return F_e, F_e  # F_v = F_u


class SheafDiffusionModel(nn.Module):
    def __init__(self, in_features, hidden_phi=32, symmetric=False, T=20, n_classes=2):
        super().__init__()
        self.T = T
        in_dim_phi = 2 * in_features
        if symmetric:
            self.phi = PhiSymmetric(in_dim_phi, hidden_dim=hidden_phi)
        else:
            self.phi = PhiGeneral(in_dim_phi, hidden_dim=hidden_phi, d=2)

        # Clasificador lineal sobre X^T
        self.classifier = nn.Linear(in_features, n_classes)

    def forward(self, X0, edges):
        """
        X0: [n, f]
        edges: [m, 2]
        """
        n_nodes = X0.shape[0]
        X = X0

        # Construimos F a partir de X0
        v_idx = edges[:, 0]
        u_idx = edges[:, 1]
        x_v = X0[v_idx]
        x_u = X0[u_idx]
        F_ve, F_ue = self.phi(x_v, x_u)

        # Laplaciano de sheaf
        Delta = build_sheaf_laplacian_scalar(n_nodes, edges, F_ve, F_ue)

        # X^{t+1} = X^t - ∆_F X^t
        for _ in range(self.T):
            X = X - Delta @ X

        # Clasificador lineal
        logits = self.classifier(X)
        return logits

    def get_diffusion_trajectory(self, X0, edges):
        self.eval()
        with torch.no_grad():
            n_nodes = X0.shape[0]
            X = X0

            v_idx = edges[:, 0]
            u_idx = edges[:, 1]
            x_v = X0[v_idx]
            x_u = X0[u_idx]
            F_ve, F_ue = self.phi(x_v, x_u)
            Delta = build_sheaf_laplacian_scalar(n_nodes, edges, F_ve, F_ue)

            trajectory = [X.detach().cpu().numpy()]

            for _ in range(self.T):
                X = X - Delta @ X
                trajectory.append(X.detach().cpu().numpy())

            return trajectory
