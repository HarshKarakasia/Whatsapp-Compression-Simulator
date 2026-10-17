"""L_bal: Switch Transformer-style load balancing loss."""
import torch


def balance_loss(router_probs: torch.Tensor) -> torch.Tensor:
    """router_probs: [B,K] raw softmax (pre-top-k). Returns
    K * sum_k(f_k * P_k): f_k is the fraction of samples whose top-1
    choice is expert k, P_k is the mean routing probability mass on
    expert k across the batch. Minimized (-> 1.0) when routing is uniform."""
    k = router_probs.shape[-1]
    assignment = router_probs.argmax(dim=-1)  # [B]
    f = torch.stack([(assignment == i).float().mean() for i in range(k)])
    p = router_probs.mean(dim=0)  # [K]
    return k * (f * p).sum()
