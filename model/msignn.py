import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_variant import TransformerVariant
from .gnn_layers import GATEncoder

class MSI_GNN(nn.Module):
    def __init__(self, in_dim, config):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = config.hidden_dim
        self.transformer = TransformerVariant(in_dim, d_model=config.hidden_dim,
                                              n_heads=config.transformer_heads,
                                              n_layers=config.transformer_layers,
                                              dropout=config.dropout,
                                              feature_mha=config.use_feature_mha)
        self.gat = GATEncoder(in_channels=config.hidden_dim, hidden_channels=config.gnn_hidden,
                              heads=config.gat_heads, dropout=config.dropout)
        self.classifier = nn.Sequential(
            nn.Linear(config.gnn_hidden, config.gnn_hidden//2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.gnn_hidden//2, 1)
        )
        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index):
        """
        x: (N, d_in)
        edge_index: (2, E)
        returns logits (N,) and optionally attention weights from transformer
        """
        h, feat_attn = self.transformer(x)  # h: (N, hidden_dim)
        h = F.relu(h)
        h = self.gat(h, edge_index)
        logits = self.classifier(h).squeeze(-1)
        return logits, feat_attn
