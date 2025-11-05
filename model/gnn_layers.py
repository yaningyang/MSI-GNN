import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=4, dropout=0.3):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels // heads, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = self.act(x)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        return x
