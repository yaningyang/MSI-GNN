import torch
import torch.nn as nn

class TransformerVariant(nn.Module):
    """
    Transformer-variant module that can operate in two modes:
    - feature_mha=True: treat each node's d features as tokens (seq_len=d), apply MHA across features.
      Input: x (N, d) -> output (N, d_model)
    - feature_mha=False: simple feedforward residual blocks across feature vector.
    """
    def __init__(self, in_dim, d_model=128, n_heads=8, n_layers=2, dropout=0.3, feature_mha=True):
        super().__init__()
        self.feature_mha = feature_mha
        self.in_dim = in_dim
        self.d_model = d_model
        self.n_heads = n_heads

        if feature_mha:
            # project scalar feature -> embedding per feature token
            self.token_proj = nn.Linear(1, d_model)   # maps scalar to embedding
            self.pos_emb = nn.Parameter(torch.randn(in_dim, d_model) * 0.02)
            self.mha_layers = nn.ModuleList([nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=False) for _ in range(n_layers)])
            self.ffns = nn.ModuleList([nn.Sequential(
                nn.Linear(d_model, d_model*4),
                nn.ReLU(),
                nn.Linear(d_model*4, d_model),
                nn.Dropout(dropout)
            ) for _ in range(n_layers)])
            self.norms1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
            self.norms2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
            # final pooling projection back to node embedding
            self.pool_proj = nn.Linear(d_model, d_model)
        else:
            # feedforward residual encoder on node vector
            layers = []
            for _ in range(n_layers):
                layers.append(nn.Sequential(
                    nn.LayerNorm(in_dim),
                    nn.Linear(in_dim, d_model),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, in_dim),
                ))
            self.ff_blocks = nn.ModuleList(layers)
            self.out_proj = nn.Linear(in_dim, d_model)

    def forward(self, x):
        """
        x: torch.tensor (N, d_in)
        returns: h (N, d_model)
        """
        N, d = x.shape
        if self.feature_mha:
            # build token sequence: for each node we create seq_len=d tokens; to vectorize across batch:
            # tokens shape for MHA: (L, batch, E) where L=d, batch=N, E=d_model
            # Step1: take x -> (N, d, 1) then project each scalar -> embedding
            x_feats = x.unsqueeze(-1)  # (N, d, 1)
            # project scalars to token embeddings with shared linear
            # reshape to (N*d, 1) -> project -> (N*d, d_model) -> reshape (N, d, d_model)
            tokens = self.token_proj(x_feats.view(-1,1))  # (N*d, d_model)
            tokens = tokens.view(N, d, self.d_model)  # (N, L, E)
            # add positional embeddings (feature-position)
            tokens = tokens + self.pos_emb.unsqueeze(0)  # (1, d, E) broadcast to (N,d,E)
            # transpose to (L, batch, E)
            tokens = tokens.permute(1, 0, 2).contiguous()
            attn_weights_all = None
            for i, (mha, ffn, n1, n2) in enumerate(zip(self.mha_layers, self.ffns, self.norms1, self.norms2)):
                attn_out, attn_weights = mha(tokens, tokens, tokens, need_weights=True)
                tokens = n1(tokens.permute(1,0,2) + attn_out.permute(1,0,2)).permute(1,0,2)
                # feedforward
                ff = ffn(tokens.permute(1,0,2)).permute(1,0,2)
                tokens = n2(tokens + ff)
                attn_weights_all = attn_weights  # last layer's attn weights (L, L) aggregated across heads by MHA returns (batch_heads?)
            # pool tokens to node embedding: tokens (L, N, E) -> (N, L, E) -> mean over L
            tokens = tokens.permute(1,0,2)  # (N, L, E)
            pooled = tokens.mean(dim=1)  # (N, E)
            h = self.pool_proj(pooled)
            # optionally return attn weights for attribution
            return h, attn_weights_all
        else:
            h = x
            for block in self.ff_blocks:
                h = h + block(h)
            h = self.out_proj(h)
            return h, None
