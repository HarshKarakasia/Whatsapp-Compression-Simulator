"""L_route: cross-entropy between the router's raw logits and the
ffprobe-derived cluster target (configs/model.yaml's router_clusters)."""
import torch
import torch.nn.functional as F


def router_targets_to_indices(router_targets: list[str], cluster_map: dict[str, int]) -> torch.Tensor:
    default = cluster_map.get("default", 0)
    indices = [cluster_map.get(rt, default) for rt in router_targets]
    return torch.tensor(indices, dtype=torch.long)


def route_loss(router_logits: torch.Tensor, router_targets: list[str], cluster_map: dict[str, int]) -> torch.Tensor:
    """router_logits: [B,K] pre-softmax. router_targets: length-B list of
    router_target strings (e.g. "h264_G1", "clean_G0") from Phase 2's probe.py."""
    target_idx = router_targets_to_indices(router_targets, cluster_map).to(router_logits.device)
    return F.cross_entropy(router_logits, target_idx)
