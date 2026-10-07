import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad_(False)

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class PatchEmbedding(nn.Module):
    """Embed normalized patches: (B*C, L, 1) -> (B*C, N, D)."""

    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.projector = nn.Conv1d(
            in_channels=1, out_channels=d_model, kernel_size=patch_len, stride=stride
        )
        self.pos_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_raw_patches=False):

        x_values = x.squeeze(-1)
        raw_patches = x_values.unfold(
            dimension=1, size=self.patch_len, step=self.stride
        )
        weight = self.projector.weight.squeeze(1)
        x = F.linear(raw_patches, weight, self.projector.bias)
        x = x + self.pos_embedding(x)
        x = self.dropout(x)

        if return_raw_patches:
            return x, raw_patches
        return x


class CrossVariableContextExchange(nn.Module):
    def __init__(
        self, d_model, n_vars, bottleneck=4, num_heads=4, dropout=0.1, max_vars=512
    ):
        super().__init__()
        self.d_model = d_model
        self.bottleneck = bottleneck
        self.max_vars = max_vars

        # Variable embeddings are sliced to the actual channel count.
        self.var_embedding = nn.Parameter(torch.randn(1, max_vars, 1, d_model) * 0.02)

        # All queries attend to the joint variable-patch sequence.
        self.routing_vectors = nn.Parameter(
            torch.randn(1, max_vars * bottleneck, d_model) * 0.02
        )

        self.mha_agg = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_agg = nn.LayerNorm(d_model)

        self.mha_dist = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_out = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_embed, B, N_vars):
        N_patches = x_embed.shape[1]
        d = self.d_model

        X_global = x_embed.reshape(B, N_vars, N_patches, d)

        X_global = X_global + self.var_embedding[:, :N_vars]

        # Joint variable-patch tokens: (B, C*N, D)
        X_flat = X_global.reshape(B, N_vars * N_patches, d)

        # Aggregate context into the query bank.
        c = self.bottleneck
        R = self.routing_vectors[:, : N_vars * c, :].expand(
            B, -1, -1
        )  # (B, N_vars*c, d)
        A, _ = self.mha_agg(R, X_flat, X_flat)
        A = self.norm_agg(A + R)

        # Redistribute context to variable-patch tokens.
        Z_enrich, _ = self.mha_dist(X_flat, A, A)
        Z_flat = self.norm_out(Z_enrich + X_flat)

        return Z_flat.reshape(B * N_vars, N_patches, d)
