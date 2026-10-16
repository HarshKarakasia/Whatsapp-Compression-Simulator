"""Cross-attention fusion: RGB tokens as queries, DCT tokens as keys/values,
mean-pooled to a single z per frame."""
import torch
import torch.nn as nn


class FusionModule(nn.Module):
    """forward(rgb_tokens [B, n_rgb, D], dct_tokens [B, n_dct, D]) -> z [B, D]."""

    def __init__(self, config: dict):
        super().__init__()
        embed_dim = config["embed_dim"]
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, config["num_heads"], batch_first=True)

    def forward(self, rgb_tokens: torch.Tensor, dct_tokens: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(rgb_tokens)
        kv = self.norm_kv(dct_tokens)
        fused, _ = self.cross_attn(q, kv, kv, need_weights=False)  # [B, n_rgb, D]
        return fused.mean(dim=1)  # pool over RGB query positions -> [B, D]
