import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MixHopConv


class MixHopNet(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5, powers=(0, 1, 2)):
        super().__init__()
        self.powers = powers

        # First MixHop layer
        self.conv1 = MixHopConv(in_dim, hidden_dim, powers=powers)
        # conv1 output dimension is hidden_dim * len(powers)
        conv1_out_dim = hidden_dim * len(powers)

        # Second MixHop layer (optional, still relatively shallow)
        self.conv2 = MixHopConv(conv1_out_dim, hidden_dim, powers=powers)
        conv2_out_dim = hidden_dim * len(powers)

        # Final linear classifier
        self.lin = torch.nn.Linear(conv2_out_dim, out_dim)

        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin(x)
        return x
