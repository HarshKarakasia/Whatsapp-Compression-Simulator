"""Combined, weighted total loss (Phase 6). Weights come from
configs/model.yaml's loss_weights, never hardcoded here."""
import torch

from src.losses.balance import balance_loss
from src.losses.classification import cls_loss
from src.losses.contrastive import con_loss
from src.losses.routing import route_loss


def total_loss(
    model_out: dict,
    labels: torch.Tensor,
    source_ids: list,
    router_targets: list,
    cluster_map: dict,
    weights: dict,
    temperature: float = 0.07,
) -> dict:
    l_cls = cls_loss(model_out["logits"], labels)
    l_route = route_loss(model_out["router_logits"], router_targets, cluster_map)
    l_con = con_loss(model_out["z_video"], source_ids, temperature)
    l_bal = balance_loss(model_out["router_probs"])

    total = weights["cls"] * l_cls + weights["route"] * l_route + weights["con"] * l_con + weights["bal"] * l_bal

    return {"total": total, "cls": l_cls, "route": l_route, "con": l_con, "bal": l_bal}
