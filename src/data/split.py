"""Leakage-aware source_id-level train/val/test splitter (Phase 2).

Splits are assigned per source_id, not per row: every file derived from the
same source video (e.g. a real clip and all its fakes/transforms in FF++)
must land in exactly one of train/val/test.
"""
import random


class LeakageError(ValueError):
    """Raised when a source_id ends up in more than one split partition."""


def group_by_source(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["source_id"], []).append(row)
    return groups


def assert_no_leakage(splits: dict[str, list[dict]]) -> None:
    seen: dict[str, str] = {}
    for split_name, rows in splits.items():
        for row in rows:
            sid = row["source_id"]
            if sid in seen and seen[sid] != split_name:
                raise LeakageError(
                    f"source_id {sid!r} appears in both {seen[sid]!r} and {split_name!r}"
                )
            seen[sid] = split_name


def split_by_source(
    rows: list[dict],
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Group rows by source_id, assign each whole group to train/val/test
    (~ratios of source_ids, not rows), and verify the result leaks nothing."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")

    groups = group_by_source(rows)
    source_ids = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(source_ids)

    n = len(source_ids)
    n_train = round(n * ratios[0])
    n_val = round(n * ratios[1])

    train_ids = set(source_ids[:n_train])
    val_ids = set(source_ids[n_train : n_train + n_val])

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for sid, group_rows in groups.items():
        if sid in train_ids:
            splits["train"].extend(group_rows)
        elif sid in val_ids:
            splits["val"].extend(group_rows)
        else:
            splits["test"].extend(group_rows)

    assert_no_leakage(splits)
    return splits
