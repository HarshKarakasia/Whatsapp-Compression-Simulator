"""4-layer, 8-head transformer over DCT tokens."""
import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, hidden), nn.GELU(), nn.Linear(hidden, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class FrequencyBranch(nn.Module):
    """forward(dct_tokens [B, n_tokens, dct_coeffs]) -> [B, n_tokens, embed_dim]."""

    def __init__(self, config: dict):
        super().__init__()
        embed_dim = config["embed_dim"]
        self.input_embed = nn.Linear(config["dct_coeffs"], embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, config["n_tokens"], embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, config["num_heads"], config["mlp_ratio"]) for _ in range(config["depth"])]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, dct_tokens: torch.Tensor) -> torch.Tensor:
        x = self.input_embed(dct_tokens) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
