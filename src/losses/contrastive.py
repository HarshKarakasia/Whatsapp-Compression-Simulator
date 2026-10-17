"""L_con: source-aware InfoNCE. A real clip and all its fakes/transforms
share a source_id (Phase 2's leakage-aware grouping) -- positives are same-
source pairs within the batch, negatives are everything else."""
import torch
import torch.nn.functional as F


def con_loss(z: torch.Tensor, source_ids: list, temperature: float = 0.07) -> torch.Tensor:
    """z: [B,D] embeddings (e.g. z_video). source_ids: length-B group ids.
    Returns 0 (not NaN) if no sample in the batch has a same-source
    partner -- a real batch should be built to avoid this, but a loss
    function shouldn't blow up if it happens."""
    b = z.shape[0]
    device = z.device
    z_norm = F.normalize(z, dim=-1)
    sim = z_norm @ z_norm.t() / temperature  # [B,B]

    self_mask = torch.eye(b, dtype=torch.bool, device=device)
    sim = sim.masked_fill(self_mask, float("-inf"))

    ids = list(source_ids)
    same_source = torch.tensor([[ids[i] == ids[j] for j in range(b)] for i in range(b)], device=device)
    pos_mask = same_source & ~self_mask  # [B,B]

    has_positive = pos_mask.any(dim=1)
    if not has_positive.any():
        return (z.sum() * 0.0).abs()  # stays in the autograd graph, always 0

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    # the self position is -inf in log_prob; masked_fill it to 0 before
    # multiplying by pos_mask, since -inf * 0 (False) is NaN, not 0
    log_prob = log_prob.masked_fill(self_mask, 0.0)
    pos_log_prob = (log_prob * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
    return -pos_log_prob[has_positive].mean()
