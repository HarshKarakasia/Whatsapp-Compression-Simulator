"""Comparison runner: MoE (top-2) vs single-head ablation vs no-L_con
ablation, sliced by transformation depth and seen/unseen chain.

Expects rows shaped {"router_target", "label", "pred_moe",
"pred_single_head", "pred_no_con"} -- one row per evaluated clip, with
all three variants' predictions co-located so slices stay paired.
"""
from src.eval.metrics import accuracy, auc, ece
from src.eval.slicing import slice_by_depth, slice_by_seen_unseen

VARIANTS = ["moe", "single_head", "no_con"]


def _metrics_for_rows(rows: list[dict], variant: str) -> dict:
    y_true = [r["label"] for r in rows]
    y_prob = [r[f"pred_{variant}"] for r in rows]
    return {"n": len(rows), "auc": auc(y_true, y_prob), "acc": accuracy(y_true, y_prob), "ece": ece(y_true, y_prob)}


def comparison_table_by_depth(rows: list[dict]) -> list[dict]:
    """One row per (variant, depth) plus an overall row per variant."""
    table = []
    depth_slices = slice_by_depth(rows)
    for variant in VARIANTS:
        table.append({"variant": variant, "slice": "overall", "key": "all", **_metrics_for_rows(rows, variant)})
        for depth, depth_rows in sorted(depth_slices.items()):
            table.append({"variant": variant, "slice": "depth", "key": f"G{depth}", **_metrics_for_rows(depth_rows, variant)})
    return table


def comparison_table_seen_unseen(rows: list[dict], seen_router_targets: set[str]) -> list[dict]:
    table = []
    su_slices = slice_by_seen_unseen(rows, seen_router_targets)
    for variant in VARIANTS:
        for key, su_rows in su_slices.items():
            if su_rows:
                table.append({"variant": variant, "slice": "seen_unseen", "key": key, **_metrics_for_rows(su_rows, variant)})
    return table
