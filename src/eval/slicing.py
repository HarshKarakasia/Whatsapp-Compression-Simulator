"""Slices evaluation rows by transformation depth (G0-G3) and seen-vs-
unseen chain, keyed off router_target (Phase 2's probe.py field).

"Seen" chains: the router_targets configs/model.yaml's router_clusters
maps explicitly (clean_G0 + the 2-codec x 2-depth reduced set this
project's scale actually trains on -- see Appendix A / Phase 6
DECISIONS.md). "Unseen": everything else (deeper hops, vp9/av1) -- the
BUILD.md spec references "the held-out-chain list from the spec" without
enumerating it in BUILD.md itself; this ties the seen/unseen split to the
same reduced-cluster design already used for L_route, rather than
inventing an unrelated list, and is documented here for that reason.
"""
import re

ROUTER_TARGET_RE = re.compile(r"^(clean_G0|([a-z0-9]+)_G(\d+))$")


def depth_of(router_target: str) -> int:
    if router_target == "clean_G0":
        return 0
    m = ROUTER_TARGET_RE.match(router_target)
    if not m:
        raise ValueError(f"malformed router_target: {router_target!r}")
    return int(m.group(3))


def slice_by_depth(rows: list[dict]) -> dict[int, list[dict]]:
    slices: dict[int, list[dict]] = {}
    for row in rows:
        d = depth_of(row["router_target"])
        slices.setdefault(d, []).append(row)
    return slices


def slice_by_seen_unseen(rows: list[dict], seen_router_targets: set[str]) -> dict[str, list[dict]]:
    slices = {"seen": [], "unseen": []}
    for row in rows:
        key = "seen" if row["router_target"] in seen_router_targets else "unseen"
        slices[key].append(row)
    return slices
