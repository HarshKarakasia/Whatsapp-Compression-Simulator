"""K small expert heads (768->256->1), gate-weighted combination."""
import torch
import torch.nn as nn


class Experts(nn.Module):
    """forward(z [B, D], gate_weights [B, K]) -> logits [B, 1], gate-weighted
    sum over all K experts (non-top-k experts contribute zero via gate_weights)."""

    def __init__(self, embed_dim: int, num_experts: int, hidden_dim: int):
        super().__init__()
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)) for _ in range(num_experts)]
        )

    def forward(self, z: torch.Tensor, gate_weights: torch.Tensor) -> torch.Tensor:
        expert_outs = torch.cat([e(z) for e in self.experts], dim=-1)  # [B, K]
        combined = (expert_outs * gate_weights).sum(dim=-1, keepdim=True)  # [B, 1]
        return combined
