"""L_cls: binary cross-entropy between the predicted fake-probability logit
and the real/fake label."""
import torch
import torch.nn.functional as F


def cls_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """logits: [B,1] raw logits (pre-sigmoid). labels: [B] or [B,1], 0/1."""
    labels = labels.float().view_as(logits)
    return F.binary_cross_entropy_with_logits(logits, labels)
