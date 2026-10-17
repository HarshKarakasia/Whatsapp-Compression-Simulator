"""Gating MLP 768->256->K with soft top-2 expert selection."""
import torch
import torch.nn as nn


class Router(nn.Module):
    """forward(z [B, D]) -> dict with:
    - gate_weights [B, K]: top-2 per sample, zeroed elsewhere, renormalized
      to sum to 1 over the selected experts -- what Experts actually uses.
    - probs [B, K]: raw softmax over all K experts, pre-top-k -- what
      L_bal's load-balancing math needs.
    - logits [B, K]: pre-softmax -- what L_route's cross-entropy needs.
    """

    def __init__(self, embed_dim: int, num_experts: int, hidden_dim: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, z: torch.Tensor) -> dict:
        logits = self.gate(z)  # [B, K]
        probs = torch.softmax(logits, dim=-1)

        top_vals, top_idx = probs.topk(self.top_k, dim=-1)  # [B, top_k] each
        top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True)  # renormalize

        gate_weights = torch.zeros_like(probs)
        gate_weights.scatter_(1, top_idx, top_vals)
        return {"gate_weights": gate_weights, "probs": probs, "logits": logits}
